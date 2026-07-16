"""Tests for the batch sklearn transformer facades and f_rupture."""

import numpy as np
import pandas as pd
import pytest

from process_improve.batch._transformers import (
    BatchFeatureExtractor,
    BatchScaler,
    DTWAligner,
    ResampleAligner,
)
from process_improve.batch.datasets import load_nylon


@pytest.fixture
def nylon() -> dict:
    """Load the nylon batch dictionary."""
    return load_nylon()


@pytest.fixture
def nylon_tags(nylon: dict) -> list:
    """Return the nylon tag names."""
    return list(next(iter(nylon.values())).columns)


def test_batch_scaler_round_trip(nylon: dict, nylon_tags: list) -> None:
    """Apply then reverse scaling returns the original trajectories."""
    scaler = BatchScaler(nylon_tags).fit(nylon)
    scaled = scaler.transform(nylon)
    restored = scaler.inverse_transform(scaled)
    np.testing.assert_allclose(nylon[1][nylon_tags].to_numpy(), restored[1][nylon_tags].to_numpy(), atol=1e-6)


def test_batch_scaler_learns_ranges(nylon: dict, nylon_tags: list) -> None:
    """The fitted scaler exposes a per-tag range table."""
    scaler = BatchScaler(nylon_tags).fit(nylon)
    assert scaler.scale_df_.shape[0] == len(nylon_tags)


def test_resample_aligner_equal_lengths(nylon: dict, nylon_tags: list) -> None:
    """Resampling gives every batch the reference length."""
    aligner = ResampleAligner(nylon_tags, reference_batch=1).fit(nylon)
    aligned = aligner.transform(nylon)
    assert len({v.shape[0] for v in aligned.values()}) == 1
    assert aligner.reference_batch_ == 1
    assert next(iter(aligned.values())).shape[0] == nylon[1].shape[0]


def test_resample_aligner_auto_reference(nylon: dict, nylon_tags: list) -> None:
    """With no reference, one is chosen and recorded."""
    aligner = ResampleAligner(nylon_tags).fit(nylon)
    assert aligner.reference_batch_ in nylon


def test_dtw_aligner_train_and_new(nylon: dict, nylon_tags: list) -> None:
    """DTW aligns the training set to equal length and a new batch to the reference."""
    keys = list(nylon)
    train = {k: nylon[k] for k in keys[:6]}
    held_key = keys[7]
    aligner = DTWAligner(nylon_tags, settings={"show_progress": False}).fit(train)
    aligned = aligner.transform(train)
    assert len({v.shape[0] for v in aligned.values()}) == 1
    new_out = aligner.transform({held_key: nylon[held_key]})
    assert new_out[held_key].shape[0] == aligner._n_reference_rows


def test_feature_extractor_matrix_shape(nylon: dict, nylon_tags: list) -> None:
    """The feature matrix is (n_batches, n_tags * n_features), batch-indexed."""
    extractor = BatchFeatureExtractor(features=("mean", "std", "last"), tags=nylon_tags).fit(nylon)
    matrix = extractor.transform(nylon)
    assert matrix.shape == (57, len(nylon_tags) * 3)
    assert matrix.index.name == "batch_id"
    assert "Tag01_mean" in matrix.columns
    assert extractor.feature_names_out_ == list(matrix.columns)


def test_feature_extractor_feeds_pls(nylon: dict, nylon_tags: list) -> None:
    """The feature matrix is a valid X block for a PLS-to-quality model."""
    from process_improve.multivariate import PLS, MCUVScaler

    features = BatchFeatureExtractor(features=("mean", "std"), tags=nylon_tags).fit_transform(nylon)
    y = pd.Series({b: float(nylon[b]["Tag01"].mean()) for b in nylon}, name="q").to_frame()
    x_scaled = MCUVScaler().fit_transform(features)
    y_scaled = MCUVScaler().fit_transform(y)
    model = PLS(n_components=2).fit(x_scaled, y_scaled)
    assert model.r2_cumulative_.iloc[-1] > 0.5


def test_feature_extractor_unknown_feature(nylon: dict) -> None:
    """An unknown feature name is rejected."""
    with pytest.raises(ValueError, match="Unknown feature"):
        BatchFeatureExtractor(features=("mean", "banana")).fit(nylon)


def test_f_rupture_detects_step() -> None:
    """f_rupture locates a step change in each batch's trajectory."""
    pytest.importorskip("ruptures")
    from process_improve.batch.features import f_rupture

    def make(step_at: int) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        x = np.concatenate([np.zeros(step_at), np.full(30 - step_at, 5.0)]) + 0.01 * rng.standard_normal(30)
        return pd.DataFrame({"x": x})

    melted = pd.concat(
        [make(10).assign(batch_id=1), make(20).assign(batch_id=2)], ignore_index=True
    )
    out = f_rupture(melted, columns=["x"], batch_col="batch_id", penalty=5.0)
    assert list(out.columns) == ["x_rupture"]
    values = out["x_rupture"].to_numpy()
    assert values[0] == pytest.approx(10, abs=2)
    assert values[1] == pytest.approx(20, abs=2)


def test_f_rupture_requires_single_column() -> None:
    """f_rupture rejects a multi-column request."""
    pytest.importorskip("ruptures")
    from process_improve.batch.features import f_rupture

    df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0], "batch_id": [1, 1]})
    with pytest.raises(ValueError, match="single column"):
        f_rupture(df, columns=["x", "y"], batch_col="batch_id")
