"""Contributions for rows with missing cells: defined at the observed cells, NaN at the missing ones."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from process_improve.multivariate import PCA, PLS, MCUVScaler
from process_improve.multivariate._diagnostics import score_contributions, spe_contributions, t2_contributions

INCOMPLETE_ROWS = ["r3", "r10", "r25"]


@pytest.fixture(scope="module")
def data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forty rows on two latent directions, scaled, with four cells missing in three rows; one response."""
    rng = np.random.default_rng(7)
    latent = rng.standard_normal((40, 2))
    X = pd.DataFrame(
        latent @ rng.standard_normal((2, 8)) + 0.1 * rng.standard_normal((40, 8)),
        columns=[f"x{k}" for k in range(8)],
        index=[f"r{i}" for i in range(40)],
    )
    X.loc["r3", ["x1", "x5"]] = np.nan
    X.loc["r10", "x0"] = np.nan
    X.loc["r25", "x7"] = np.nan
    y = pd.DataFrame(latent[:, [0]] + 0.05 * rng.standard_normal((40, 1)), columns=["y"], index=X.index)
    return MCUVScaler().fit_transform(X), MCUVScaler().fit_transform(y)


@pytest.fixture(scope="module")
def pca(data: tuple[pd.DataFrame, pd.DataFrame]) -> PCA:
    model = PCA(n_components=2).fit(data[0])
    assert model.has_missing_data_
    return model


class TestPCAWithMissingCells:
    def test_nan_only_at_the_missing_cells(self, pca: PCA, data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        X, _ = data
        for frame in (spe_contributions(pca, X), score_contributions(pca, X, component=2), t2_contributions(pca, X)):
            assert frame.isna().equals(X.isna())

    def test_training_rows_reproduce_the_stored_diagnostics(
        self, pca: PCA, data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        # A NIPALS fit stores the last-iteration scores, which differ from a projection on the converged
        # loadings by the convergence tolerance (the loadings are not exactly orthogonal); the incomplete
        # rows are held to the same tolerance as the complete ones, and exactly to the "scp" projection.
        X, _ = data
        residuals = spe_contributions(pca, X)
        np.testing.assert_allclose((residuals**2).sum(axis=1), pca.spe_.iloc[:, -1] ** 2, rtol=1e-3)
        for a in (1, 2):
            contributions = score_contributions(pca, X, component=a)
            np.testing.assert_allclose(contributions.sum(axis=1), pca.scores_.iloc[:, a - 1], rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(t2_contributions(pca, X).sum(axis=1), pca.hotellings_t2_.iloc[:, -1], rtol=1e-2)
        projected = pca.project(X, method="scp")
        np.testing.assert_allclose((residuals**2).sum(axis=1), projected.spe**2, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(
            score_contributions(pca, X, component=2).sum(axis=1), projected.scores.iloc[:, 1], rtol=1e-9, atol=1e-12
        )

    def test_complete_rows_take_the_complete_data_path(self, pca: PCA, data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        X, _ = data
        complete = X.dropna()
        for function, kwargs in (
            (spe_contributions, {}),
            (score_contributions, {"component": 1}),
            (t2_contributions, {}),
        ):
            with_nan = function(pca, X, **kwargs).loc[complete.index]
            without = function(pca, complete, **kwargs)
            np.testing.assert_array_equal(with_nan.to_numpy(), without.to_numpy())

    def test_other_estimators_agree_with_project(self, pca: PCA, data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        X, _ = data
        for method in ("tsr", "pmp"):
            projected = pca.project(X, method=method)
            residuals = spe_contributions(pca, X, method=method)
            np.testing.assert_allclose((residuals**2).sum(axis=1), projected.spe**2, rtol=1e-9, atol=1e-12)
            first = score_contributions(pca, X, component=1, method=method)
            np.testing.assert_allclose(first.sum(axis=1), projected.scores.iloc[:, 0], rtol=1e-9, atol=1e-12)
            t2 = t2_contributions(pca, X, method=method)
            np.testing.assert_allclose(t2.sum(axis=1), projected.hotellings_t2, rtol=1e-9, atol=1e-12)

    def test_an_all_nan_row_is_rejected(self, pca: PCA, data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        X = data[0].copy()
        X.loc["r0"] = np.nan
        with pytest.raises(ValueError, match="no observed features"):
            spe_contributions(pca, X)


class TestPLSWithMissingCells:
    def test_contributions_sum_to_the_projected_scores_and_spe(self, data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        X, y = data
        pls = PLS(n_components=2, scale=False).fit(X, y)
        for method in ("scp", "tsr"):
            projected = pls.project(X, method=method)
            residuals = spe_contributions(pls, X, method=method)
            assert residuals.isna().equals(X.isna())
            np.testing.assert_allclose((residuals**2).sum(axis=1), projected.spe**2, rtol=1e-9, atol=1e-12)
            second = score_contributions(pls, X, component=2, method=method)
            np.testing.assert_allclose(second.sum(axis=1), projected.scores.iloc[:, 1], rtol=1e-9, atol=1e-12)
            np.testing.assert_allclose(
                t2_contributions(pls, X, method=method).sum(axis=1), projected.hotellings_t2, rtol=1e-9, atol=1e-12
            )
