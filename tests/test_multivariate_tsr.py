"""Tests for the Trimmed Score Regression (TSR) PCA algorithm.

PCA's ``algorithm="tsr"`` branch (multivariate.methods._fit_tsr,
lines 464-524) handles missing data via Folch-Fortuny / DOI 10.1002/cem.750
and was previously uncovered.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from process_improve.multivariate.methods import PCA, MCUVScaler


@pytest.fixture
def kamyr_with_missing() -> pd.DataFrame:
    """Load the Kamyr dataset, which contains missing values."""
    folder = pathlib.Path(__file__).parents[1] / "process_improve" / "datasets" / "multivariate"
    return pd.read_csv(folder / "kamyr.csv", index_col=None, header=None)


def test_pca_tsr_fits_with_missing_data(kamyr_with_missing: pd.DataFrame) -> None:
    """TSR algorithm should fit a PCA model on data with missing values."""
    x_mcuv = MCUVScaler().fit_transform(kamyr_with_missing)
    pca = PCA(n_components=2, algorithm="tsr")

    model = pca.fit(x_mcuv)
    assert model.algorithm_ == "tsr"
    assert model.has_missing_data_ is True

    # Loadings should still be (approximately) orthonormal.
    loadings = model.loadings_.values
    gram = loadings.T @ loadings
    assert np.allclose(gram, np.eye(gram.shape[0]), atol=1e-2)


def test_pca_tsr_records_iteration_metadata(kamyr_with_missing: pd.DataFrame) -> None:
    """The TSR fit should populate fitting_info_ with iteration count + timing."""
    x_mcuv = MCUVScaler().fit_transform(kamyr_with_missing)
    pca = PCA(n_components=2, algorithm="tsr").fit(x_mcuv)

    assert "iterations" in pca.fitting_info_
    assert "timing" in pca.fitting_info_
    assert pca.fitting_info_["iterations"] >= 1
    assert pca.fitting_info_["timing"] >= 0.0


def test_pca_tsr_explained_variance_increases_with_components(
    kamyr_with_missing: pd.DataFrame,
) -> None:
    """Cumulative R² across components should monotonically increase."""
    x_mcuv = MCUVScaler().fit_transform(kamyr_with_missing)
    pca = PCA(n_components=3, algorithm="tsr").fit(x_mcuv)

    r2_cum = list(pca.r2_cumulative_)
    assert all(r2_cum[i] <= r2_cum[i + 1] + 1e-9 for i in range(len(r2_cum) - 1))
    assert r2_cum[-1] <= 1.0 + 1e-6
