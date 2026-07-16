"""Tests for BatchPCA: batchwise-unfolded (multiway) PCA on batch data."""

import numpy as np
import pandas as pd
import pytest

from process_improve.batch._batch_pca import BatchPCA
from process_improve.batch.datasets import load_nylon
from process_improve.batch.preprocessing import resample_to_reference


@pytest.fixture
def aligned_nylon() -> dict:
    """Nylon batches resampled to a common length, ready for unfolding."""
    batches = load_nylon()
    tags = list(next(iter(batches.values())).columns)
    return resample_to_reference(batches, columns_to_align=tags, reference_batch=1)


def _synthetic_good_batches(n_batches: int = 12, n_timesteps: int = 40, *, seed_offset: int = 0) -> dict:
    """Build a set of similar, smoothly-varying good batches (three tags)."""
    t = np.linspace(0, 1, n_timesteps)
    out = {}
    for i in range(n_batches):
        rng = np.random.default_rng(i + seed_offset)
        out[f"g{i}"] = pd.DataFrame(
            {
                "A": np.sin(2 * np.pi * t) + 0.02 * rng.standard_normal(n_timesteps),
                "B": t**2 + 0.02 * rng.standard_normal(n_timesteps),
                "C": np.exp(t) + 0.02 * rng.standard_normal(n_timesteps),
            }
        )
    return out


def test_fit_on_real_nylon(aligned_nylon: dict) -> None:
    """Fitting on aligned nylon yields batch-level scores and reshapeable loadings."""
    model = BatchPCA(n_components=3).fit(aligned_nylon)
    assert model.scores_.shape == (57, 3)
    assert model.n_batches_ == 57
    assert model.n_tags_ == 10
    # loadings reshape to a (tag, time) grid
    grid = model.loadings_.iloc[:, 0].unstack(level="sequence")  # noqa: PD010
    assert grid.shape == (model.n_tags_, model.n_timesteps_)
    # cumulative R2 is monotone non-decreasing and within [0, 1]
    r2 = model.r2_cumulative_.to_numpy()
    assert np.all(np.diff(r2) >= -1e-9)
    assert 0.0 <= r2[0] <= r2[-1] <= 1.0


def test_transform_round_trip(aligned_nylon: dict) -> None:
    """Transforming the training batches reproduces the fitted scores."""
    model = BatchPCA(n_components=3).fit(aligned_nylon)
    scores = model.transform(aligned_nylon)
    np.testing.assert_allclose(scores.to_numpy(), model.scores_.to_numpy(), atol=1e-8)


def test_diagnose_matches_internal_pca(aligned_nylon: dict) -> None:
    """Diagnosing the training data matches the fitted diagnostics."""
    model = BatchPCA(n_components=3).fit(aligned_nylon)
    result = model.diagnose(aligned_nylon)
    np.testing.assert_allclose(result.scores.to_numpy(), model.scores_.to_numpy(), atol=1e-8)
    np.testing.assert_allclose(
        result.hotellings_t2.to_numpy(), model.hotellings_t2_.to_numpy(), atol=1e-8
    )


def test_fit_returns_self_and_limits(aligned_nylon: dict) -> None:
    """Fitting returns self; the limit helpers return positive finite numbers."""
    model = BatchPCA(n_components=3)
    assert model.fit(aligned_nylon) is model
    assert model.hotellings_t2_limit(0.95) > 0
    assert model.spe_limit(0.95) > 0


def test_injected_fault_is_flagged() -> None:
    """A batch with a large deviation trips the SPE and T2 limits."""
    good = _synthetic_good_batches()
    model = BatchPCA(n_components=2).fit(good)

    faulty = _synthetic_good_batches(n_batches=1, seed_offset=99)
    only = next(iter(faulty.values()))
    only["A"] = only["A"] + 1.5  # sustained upset on tag A
    result = model.diagnose({"bad": only})

    assert float(result.spe.iloc[-1]) > model.spe_limit(0.95)
    assert float(result.hotellings_t2.iloc[-1, -1]) > model.hotellings_t2_limit(0.95)


def test_initial_conditions_join() -> None:
    """The Z block is appended to the unfolded row, one row per batch."""
    good = _synthetic_good_batches()
    z = pd.DataFrame(
        {"charge": [10.0 + i for i in range(12)], "purity": [0.9 - 0.01 * i for i in range(12)]},
        index=list(good.keys()),
    )
    model = BatchPCA(n_components=2).fit(good, initial_conditions=z)
    # 3 tags x 40 samples + 2 initial conditions
    assert model.loadings_.shape[0] == 3 * 40 + 2
    assert model.n_initial_conditions_ == 2
    z_rows = [label for label in model.loadings_.index if label[1] == ""]
    assert {label[0] for label in z_rows} == {"charge", "purity"}


def test_initial_conditions_index_mismatch_raises() -> None:
    """A Z block whose index does not match the batch ids is rejected."""
    good = _synthetic_good_batches(n_batches=4)
    z = pd.DataFrame({"charge": [1.0, 2.0, 3.0]}, index=["g0", "g1", "g2"])  # missing g3
    with pytest.raises(ValueError, match="one row per batch"):
        BatchPCA(n_components=2).fit(good, initial_conditions=z)


def test_transform_wrong_length_raises(aligned_nylon: dict) -> None:
    """Projecting batches of a different length gives a clear error."""
    model = BatchPCA(n_components=2).fit(aligned_nylon)
    shorter = {k: v.iloc[:50].reset_index(drop=True) for k, v in aligned_nylon.items()}
    with pytest.raises(ValueError, match="training column layout"):
        model.transform(shorter)


def test_unscaled_keeps_centring_only() -> None:
    """scale=False leaves the scaling factors at 1.0 (centring still applied)."""
    good = _synthetic_good_batches()
    model = BatchPCA(n_components=2, scale=False).fit(good)
    np.testing.assert_allclose(model.scale_.to_numpy(), 1.0)
    # centring is non-trivial
    assert np.any(np.abs(model.center_.to_numpy()) > 1e-6)
