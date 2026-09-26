"""Latent-variable methods for categorical and count data (#176).

Reference values come from two independent sources: Greenacre's published analysis
of his staff-by-smoking table, and the ``prince`` library (0.21), which reproduces
those published figures exactly. Beyond matching them, the tests pin the identities
that define correspondence analysis, since a method can match one table by accident
but not an identity.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2_contingency

from process_improve.multivariate._common import SpecificationWarning
from process_improve.multivariate.methods import CA


@pytest.fixture
def smoke() -> pd.DataFrame:
    """Greenacre's staff group (rows) by smoking category (columns) table."""
    return pd.DataFrame(
        [[4, 2, 3, 2], [4, 3, 7, 4], [25, 10, 12, 4], [18, 24, 33, 13], [10, 6, 7, 2]],
        index=["SM", "JM", "SE", "JE", "SC"],
        columns=["none", "light", "medium", "heavy"],
    )


def _align(found: pd.DataFrame, reference: np.ndarray) -> np.ndarray:
    """Flip each axis of ``found`` to agree in sign with ``reference``; SVD signs are arbitrary."""
    values = found.to_numpy()
    return values * np.sign(np.sum(values * reference, axis=0))


class TestAgainstPublishedValues:
    def test_principal_inertias(self, smoke: pd.DataFrame) -> None:
        ca = CA(n_components=3).fit(smoke)
        np.testing.assert_allclose(ca.eigenvalues_, [0.074759, 0.010017, 0.000414], atol=5e-7)
        assert ca.total_inertia_ == pytest.approx(0.08519, abs=5e-6)

    def test_principal_coordinates(self, smoke: pd.DataFrame) -> None:
        """Against prince 0.21, rounded there to five decimals."""
        rows = np.array(
            [
                [0.06577, 0.19374, 0.07098],
                [-0.25896, 0.24330, -0.03371],
                [0.38059, 0.01066, -0.00516],
                [-0.23295, -0.05774, 0.00331],
                [0.20109, -0.07891, -0.00808],
            ]
        )
        columns = np.array(
            [
                [0.39331, 0.03049, -0.00089],
                [-0.09946, -0.14106, 0.02200],
                [-0.19632, -0.00736, -0.02566],
                [-0.29378, 0.19777, 0.02621],
            ]
        )
        ca = CA(n_components=3).fit(smoke)
        np.testing.assert_allclose(_align(ca.row_coordinates_, rows), rows, atol=1e-5)
        # The columns share the rows' axes, so the rows' sign alignment must fix them too.
        signs = np.sign(np.sum(ca.row_coordinates_.to_numpy() * rows, axis=0))
        np.testing.assert_allclose(ca.column_coordinates_.to_numpy() * signs, columns, atol=1e-5)


class TestDefiningIdentities:
    def test_total_inertia_is_chi_squared_over_n(self, smoke: pd.DataFrame) -> None:
        chi2 = chi2_contingency(smoke.to_numpy(), correction=False)[0]
        assert CA().fit(smoke).total_inertia_ == pytest.approx(chi2 / smoke.to_numpy().sum(), rel=1e-12)

    def test_map_distance_is_the_chi_squared_distance(self, smoke: pd.DataFrame) -> None:
        """On the full-dimensional map, Euclidean distance between rows is their chi-squared distance.

        This is the property that makes the map worth reading, so it is checked
        directly from the profiles rather than trusted.
        """
        ca = CA(n_components=3).fit(smoke)
        counts = smoke.to_numpy(dtype=float)
        profiles = counts / counts.sum(axis=1, keepdims=True)
        column_mass = counts.sum(axis=0) / counts.sum()
        coords = ca.row_coordinates_.to_numpy()
        for i in range(len(smoke)):
            for j in range(i + 1, len(smoke)):
                chi2_distance = np.sqrt(np.sum((profiles[i] - profiles[j]) ** 2 / column_mass))
                assert np.linalg.norm(coords[i] - coords[j]) == pytest.approx(chi2_distance, rel=1e-10)

    def test_contributions_and_cos2_account_for_everything(self, smoke: pd.DataFrame) -> None:
        ca = CA(n_components=3).fit(smoke)
        np.testing.assert_allclose(ca.row_contributions_.sum(), 1.0)
        np.testing.assert_allclose(ca.column_contributions_.sum(), 1.0)
        # Over every axis, a category's cos2 accounts for all of its own inertia.
        np.testing.assert_allclose(ca.row_cos2_.sum(axis=1), 1.0)
        np.testing.assert_allclose(ca.column_cos2_.sum(axis=1), 1.0)

    def test_scaling_every_count_changes_nothing(self, smoke: pd.DataFrame) -> None:
        """CA reads profiles, not counts, so the same table in other units maps identically."""
        np.testing.assert_allclose(CA().fit(smoke).row_coordinates_, CA().fit(smoke * 7).row_coordinates_)

    def test_transposing_swaps_rows_and_columns(self, smoke: pd.DataFrame) -> None:
        ca, transposed = CA().fit(smoke), CA().fit(smoke.T)
        np.testing.assert_allclose(ca.eigenvalues_, transposed.eigenvalues_)
        np.testing.assert_allclose(np.abs(ca.row_coordinates_), np.abs(transposed.column_coordinates_))


class TestSupplementaryPoints:
    def test_transforming_the_fitted_table_gives_the_fitted_coordinates(self, smoke: pd.DataFrame) -> None:
        """The transition formula, checked where its answer is already known."""
        ca = CA(n_components=3).fit(smoke)
        np.testing.assert_allclose(ca.transform(smoke), ca.row_coordinates_)
        np.testing.assert_allclose(ca.transform_columns(smoke), ca.column_coordinates_)

    def test_a_supplementary_row_does_not_move_the_axes(self, smoke: pd.DataFrame) -> None:
        """Placed on the map, it takes no part in defining it."""
        ca = CA().fit(smoke)
        before = ca.row_coordinates_.copy()
        placed = ca.transform(pd.DataFrame([[40, 10, 5, 1]], columns=smoke.columns, index=["new"]))
        assert list(placed.index) == ["new"]
        pd.testing.assert_frame_equal(ca.row_coordinates_, before)

    def test_supplementary_shape_is_checked(self, smoke: pd.DataFrame) -> None:
        ca = CA().fit(smoke)
        with pytest.raises(ValueError, match="expected 4"):
            ca.transform(np.ones((2, 3)))
        with pytest.raises(ValueError, match="expected 5"):
            ca.transform_columns(np.ones((4, 2)))
        with pytest.raises(ValueError, match="sums to zero"):
            ca.transform(np.zeros((1, 4)))

    @pytest.mark.parametrize("bad", [-1.0, np.nan, np.inf])
    def test_supplementary_counts_must_be_finite_and_non_negative(self, smoke: pd.DataFrame, bad: float) -> None:
        """A negative or missing count has no profile, and would silently distort one if let through."""
        ca = CA().fit(smoke)
        counts = np.array([[40.0, 10.0, 5.0, bad]])
        with pytest.raises(ValueError, match="finite, non-negative"):
            ca.transform(counts)
        with pytest.raises(ValueError, match="finite, non-negative"):
            ca.transform_columns(np.array([[1.0], [2.0], [3.0], [4.0], [bad]]))


class TestDegenerateTables:
    def test_an_independent_table_is_explained_not_mapped(self) -> None:
        """Its inertia is floating-point noise; reporting shares of it would present noise as structure."""
        independent = pd.DataFrame(np.outer([10, 20, 30], [1, 2, 3, 4]).astype(float))
        with pytest.raises(ValueError, match="no association between its rows and columns"):
            CA().fit(independent)

    def test_a_rank_deficient_table_keeps_only_its_real_axes(self) -> None:
        """Two rows with one profile leave a single axis; a second would be noise over noise."""
        table = pd.DataFrame([[10, 20, 30], [20, 40, 60], [30, 10, 5]], dtype=float)
        with pytest.warns(SpecificationWarning, match="has 1 with any inertia"):
            ca = CA(n_components=2).fit(table)
        assert ca.n_components_ == 1
        np.testing.assert_allclose(ca.explained_inertia_, [1.0])

    @pytest.mark.parametrize(
        ("table", "message"),
        [
            ([[1.0, -2.0], [3.0, 4.0]], "negative"),
            ([[1.0, np.nan], [3.0, 4.0]], "finite"),
            ([[0.0, 0.0], [3.0, 4.0]], "sum to zero"),
            ([[1.0, 2.0, 3.0]], "at least two rows"),
        ],
    )
    def test_invalid_tables_are_refused(self, table: list, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            CA().fit(pd.DataFrame(table))

    def test_n_components_must_be_positive(self, smoke: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            CA(n_components=0).fit(smoke)


class TestApi:
    def test_refitting_gives_the_same_signs(self, smoke: pd.DataFrame) -> None:
        pd.testing.assert_frame_equal(CA().fit(smoke).row_coordinates_, CA().fit(smoke).row_coordinates_)

    def test_labels_survive(self, smoke: pd.DataFrame) -> None:
        ca = CA().fit(smoke)
        assert list(ca.row_coordinates_.index) == list(smoke.index)
        assert list(ca.column_coordinates_.index) == list(smoke.columns)

    def test_map_plot(self, smoke: pd.DataFrame) -> None:
        fig = CA().fit(smoke).map_plot()
        assert [trace.name for trace in fig.data] == ["rows", "columns"]
        with pytest.raises(ValueError, match="Axes run from 1 to 2"):
            CA().fit(smoke).map_plot(axis_y=3)

    def test_fit_transform(self, smoke: pd.DataFrame) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            np.testing.assert_allclose(CA().fit_transform(smoke), CA().fit(smoke).row_coordinates_)
