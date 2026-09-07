"""``r2_per_variable_`` is the cumulative per-column R2 on every fit path (PCA: svd, nipals, tsr; PLS)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from process_improve.multivariate import PCA, PLS, MCUVScaler


def _direct_cumulative_r2(x: pd.DataFrame, scores: np.ndarray, loadings: np.ndarray) -> np.ndarray:
    """1 - SS(residual after a components) / SS(column), per column, ignoring the missing cells."""
    values = x.to_numpy(dtype=float)
    total = np.nansum(values**2, axis=0)
    out = np.empty((values.shape[1], scores.shape[1]))
    for a in range(scores.shape[1]):
        residual = values - scores[:, : a + 1] @ loadings[:, : a + 1].T
        out[:, a] = 1 - np.nansum(residual**2, axis=0) / total
    return out


@pytest.fixture
def data() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    latent = rng.normal(size=(60, 3))
    x = latent @ rng.normal(size=(3, 9)) + 0.3 * rng.normal(size=(60, 9))
    frame = pd.DataFrame(x, columns=[f"x{k}" for k in range(9)])
    return MCUVScaler().fit_transform(frame)


@pytest.mark.parametrize("algorithm", ["svd", "nipals", "tsr"])
def test_pca_paths_agree_with_the_direct_definition(data: pd.DataFrame, algorithm: str) -> None:
    x = data.copy()
    if algorithm != "svd":  # the missing-data paths, on data with gaps
        x.iloc[4, 1] = np.nan
        x.iloc[17, 6] = np.nan
    model = PCA(n_components=3, algorithm=algorithm).fit(x)
    expected = _direct_cumulative_r2(x, model.scores_.to_numpy(), model.loadings_.to_numpy())
    np.testing.assert_allclose(model.r2_per_variable_.to_numpy(), expected, atol=2e-3)
    diffs = np.diff(model.r2_per_variable_.to_numpy(), axis=1)
    assert (diffs > -2e-3).all()  # cumulative: each added component can only raise a column's R2
    assert model.r2_per_variable_.iloc[:, -1].mean() == pytest.approx(model.r2_cumulative_.iloc[-1], abs=2e-3)


def test_pls_matches_the_direct_definition(data: pd.DataFrame) -> None:
    rng = np.random.default_rng(11)
    y = pd.DataFrame(data.to_numpy() @ rng.normal(size=(9, 2)) + 0.1 * rng.normal(size=(60, 2)), columns=["y1", "y2"])
    model = PLS(n_components=3, scale=False).fit(data, MCUVScaler().fit_transform(y))
    expected = _direct_cumulative_r2(data, model.scores_.to_numpy(), model.x_loadings_.to_numpy())
    np.testing.assert_allclose(model.r2_per_variable_.to_numpy(), expected, atol=1e-8)
