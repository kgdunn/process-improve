"""Contributions for rows with missing cells: defined at the observed cells, NaN at the missing ones."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from process_improve.multivariate import PCA, PLS, MCUVScaler
from process_improve.multivariate._diagnostics import (
    _complete_rows_apart,
    score_contributions,
    spe_contributions,
    t2_contributions,
)
from process_improve.multivariate.methods import AdaptivePCA

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

    def test_scalings_skip_the_missing_cells(self, pca: PCA, data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        """A missing cell's NaN neither stops ``"maximum"`` from scaling nor leaves its row out of ``"within"``.

        Miller, Swanson and Heckler (1994), pages 20 and 21: divide by the largest absolute
        contribution in the data set, or by the row's total absolute contribution. Both are
        taken over the observed cells, as the row sums are.
        """
        X, _ = data
        raw = score_contributions(pca, X, component=1).to_numpy()
        largest = score_contributions(pca, X, component=1, scaling="maximum")
        np.testing.assert_allclose(largest, raw / np.nanmax(np.abs(raw)), rtol=1e-14)
        within = score_contributions(pca, X, component=1, scaling="within")
        np.testing.assert_allclose(within, raw / np.nansum(np.abs(raw), axis=1, keepdims=True), rtol=1e-14)
        np.testing.assert_allclose(np.nansum(np.abs(within.to_numpy()), axis=1), np.ones(len(X)), rtol=1e-14)
        assert within.isna().equals(X.isna())

    def test_the_ridge_reaches_the_estimators(self, pca: PCA, data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        X, _ = data
        for method in ("tsr", "pmp"):
            projected = pca.project(X, method=method, ridge=0.5).scores.iloc[:, 0]
            first = score_contributions(pca, X, component=1, method=method, ridge=0.5).sum(axis=1)
            np.testing.assert_allclose(first, projected, rtol=1e-9, atol=1e-12)
            unridged = pca.project(X, method=method).scores.iloc[:, 0]
            assert np.abs(first - unridged).loc[INCOMPLETE_ROWS].min() > 1e-6

    def test_an_all_nan_row_is_rejected(self, pca: PCA, data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        X = data[0].copy()
        X.loc["r0"] = np.nan
        message = "Row 0 has no observed features (all-NaN); its contributions cannot be computed."
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            spe_contributions(pca, X)


def test_a_model_without_training_residuals_still_gets_contributions_for_missing_cells(
    data: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """An adaptive PCA keeps no residual block, which only the "tsr" estimator needs; "scp" goes ahead.

    The complete rows' residuals are ``x - x P P'`` (the loadings are orthonormal); the
    incomplete rows' are NaN exactly at their missing cells.
    """
    X, _ = data
    model = AdaptivePCA(n_components=2).fit(X.dropna())
    assert not hasattr(model, "_x_residuals")
    residuals = spe_contributions(model, X)
    assert residuals.isna().equals(X.isna())
    complete = X.dropna().to_numpy()
    loadings = model.loadings_.to_numpy()
    expected = complete - complete @ loadings @ loadings.T
    np.testing.assert_allclose(residuals.dropna().to_numpy(), expected, rtol=1e-9, atol=1e-12)


def test_complete_rows_go_to_the_product_apart_from_the_rest() -> None:
    """The complete rows, and the rows with a missing cell, are each multiplied as a set of their own."""
    X = np.array([[1.0, 2.0], [np.nan, 1.0], [3.0, 4.0]])
    out = _complete_rows_apart(X, lambda rows: np.full((int(rows.sum()), 2), float(rows.sum())))
    np.testing.assert_array_equal(out, [[2.0, 2.0], [1.0, 1.0], [2.0, 2.0]])


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
