"""Tests for BatchPLS: batchwise-unfolded PLS from trajectories to final quality."""

import numpy as np
import pandas as pd
import pytest

from process_improve.batch._batch_pls import BatchPLS
from process_improve.batch.datasets import load_dryer
from process_improve.batch.preprocessing import resample_to_reference


@pytest.fixture
def aligned_dryer() -> dict:
    """Dryer batches resampled to a common length (ClockTime dropped)."""
    batches = load_dryer()
    tags = [c for c in next(iter(batches.values())).columns if c != "ClockTime"]
    trimmed = {k: v[tags] for k, v in batches.items()}
    return resample_to_reference(trimmed, columns_to_align=tags, reference_batch=1)


@pytest.fixture
def dryer_quality(aligned_dryer: dict) -> pd.DataFrame:
    """Return a synthetic quality block driven by the mean dryer temperature."""
    return pd.DataFrame(
        {"final": [float(b["DryerTemp"].mean()) for b in aligned_dryer.values()]},
        index=list(aligned_dryer.keys()),
    )


def test_fit_predict_on_dryer(aligned_dryer: dict, dryer_quality: pd.DataFrame) -> None:
    """The model fits and predicts the quality of the training batches."""
    model = BatchPLS(n_components=3).fit(aligned_dryer, dryer_quality)
    assert model.scores_.shape == (len(aligned_dryer), 3)
    assert isinstance(model.x_weights_.index, pd.MultiIndex)
    assert model.r2_cumulative_.iloc[-1] > 0.7
    result = model.predict(aligned_dryer)
    assert result.y_hat.shape == (len(aligned_dryer), 1)
    # Align predictions to the quality index by label before comparing (the
    # unfolded row order need not match the input dict's insertion order).
    y_hat = result.y_hat.reindex(dryer_quality.index).to_numpy().ravel()
    corr = np.corrcoef(y_hat, dryer_quality["final"].to_numpy())[0, 1]
    assert corr > 0.8


def test_weights_reshape_to_grid(aligned_dryer: dict, dryer_quality: pd.DataFrame) -> None:
    """The trajectory weights reshape to a (tag, time) grid."""
    model = BatchPLS(n_components=2).fit(aligned_dryer, dryer_quality)
    grid = model.x_weights_.iloc[:, 0].unstack(level="sequence")  # noqa: PD010
    assert grid.shape == (model.n_tags_, model.n_timesteps_)


def test_initial_conditions_join(aligned_dryer: dict, dryer_quality: pd.DataFrame) -> None:
    """The Z block is appended to each unfolded row."""
    z = pd.DataFrame({"charge": [float(i) for i in range(len(aligned_dryer))]}, index=list(aligned_dryer.keys()))
    model = BatchPLS(n_components=2).fit(aligned_dryer, dryer_quality, initial_conditions=z)
    assert model.n_initial_conditions_ == 1
    assert model.x_weights_.shape[0] == model.n_tags_ * model.n_timesteps_ + 1
    result = model.predict(aligned_dryer, initial_conditions=z)
    assert result.y_hat.shape[0] == len(aligned_dryer)


def test_synthetic_recovers_signal() -> None:
    """A batch whose quality is a known function of its trajectories is predicted well."""
    rng = np.random.default_rng(0)
    n_batches, n_timesteps = 30, 25
    t = np.linspace(0, 1, n_timesteps)
    batches = {}
    quality = []
    for i in range(n_batches):
        level = rng.uniform(-1, 1)
        # A carries a batch-specific level offset (so its deviation from the
        # mean trajectory encodes `level`); B is a shared ramp with noise.
        a = level + 0.5 * t + 0.02 * rng.standard_normal(n_timesteps)
        b = t + 0.02 * rng.standard_normal(n_timesteps)
        batches[f"b{i}"] = pd.DataFrame({"A": a, "B": b})
        quality.append(3.0 * level + 0.5)
    y = pd.DataFrame({"q": quality}, index=list(batches.keys()))
    model = BatchPLS(n_components=2).fit(batches, y)
    pred = model.predict(batches).y_hat.reindex(y.index)
    corr = np.corrcoef(pred.to_numpy().ravel(), y["q"].to_numpy())[0, 1]
    assert corr > 0.9


def test_y_must_be_dataframe(aligned_dryer: dict) -> None:
    """A non-DataFrame Y is rejected."""
    with pytest.raises(TypeError, match="DataFrame"):
        BatchPLS(n_components=2).fit(aligned_dryer, [1, 2, 3])


def test_y_index_must_match(aligned_dryer: dict) -> None:
    """A Y whose index does not match the batches is rejected."""
    y = pd.DataFrame({"q": [1.0, 2.0]}, index=["x", "y"])
    with pytest.raises(ValueError, match="one row per batch"):
        BatchPLS(n_components=2).fit(aligned_dryer, y)


def test_predict_wrong_length_raises(aligned_dryer: dict, dryer_quality: pd.DataFrame) -> None:
    """Predicting on batches of a different length is rejected."""
    model = BatchPLS(n_components=2).fit(aligned_dryer, dryer_quality)
    shorter = {k: v.iloc[:50].reset_index(drop=True) for k, v in aligned_dryer.items()}
    with pytest.raises(ValueError, match="training column layout"):
        model.predict(shorter)


def test_transform_returns_scores(aligned_dryer: dict, dryer_quality: pd.DataFrame) -> None:
    """Transform returns the batch-level scores."""
    model = BatchPLS(n_components=2).fit(aligned_dryer, dryer_quality)
    scores = model.transform(aligned_dryer)
    assert scores.shape == (len(aligned_dryer), 2)
