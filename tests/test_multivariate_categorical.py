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
from process_improve.multivariate.methods import CA, FAMD, MCA


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


# ---------------------------------------------------------------------------
# MCA (#177)
# ---------------------------------------------------------------------------


@pytest.fixture
def batches() -> pd.DataFrame:
    """Categorical batch attributes with real associations: grade A goes with s1 and days."""
    return pd.DataFrame(
        {
            "grade": list("AAAAABBBBCCCCCAB"),
            "supplier": [
                "s1",
                "s1",
                "s1",
                "s2",
                "s1",
                "s2",
                "s2",
                "s1",
                "s2",
                "s3",
                "s3",
                "s3",
                "s2",
                "s3",
                "s1",
                "s3",
            ],
            "shift": [
                *("day", "day", "day", "day", "night", "day", "night", "night", "day"),
                *("night", "night", "night", "night", "day", "day", "night"),
            ],
        }
    )


class TestMCAAgainstReferences:
    def test_eigenvalues_and_total_inertia(self, batches: pd.DataFrame) -> None:
        """Total inertia of an indicator matrix is (J - Q) / Q whatever the data: here (8 - 3) / 3."""
        mca = MCA(n_components=3).fit(batches)
        np.testing.assert_allclose(mca.eigenvalues_, [0.72518, 0.472534, 0.261829], atol=1e-6)
        assert mca.total_inertia_ == pytest.approx(5 / 3)

    def test_coordinates_against_prince(self, batches: pd.DataFrame) -> None:
        """The first three observations and the grade levels, against prince 0.21."""
        mca = MCA(n_components=3).fit(batches)
        rows = np.array([[-1.08248, -0.51280, -0.02134]] * 3)
        signs = np.sign(np.sum(mca.row_coordinates_.to_numpy()[:3] * rows, axis=0))
        np.testing.assert_allclose(mca.row_coordinates_.to_numpy()[:3] * signs, rows, atol=1e-5)
        grades = np.array([[-1.09064, -0.54237, -0.03352], [0.14226, 1.23813, 0.58788], [1.16651, -0.58728, -0.54766]])
        found = mca.column_coordinates_.loc[["grade=A", "grade=B", "grade=C"]].to_numpy()
        np.testing.assert_allclose(found * signs, grades, atol=1e-5)

    def test_only_j_minus_q_axes_carry_inertia(self, batches: pd.DataFrame) -> None:
        """An indicator matrix has J - Q real axes; CA's null-axis rule must find exactly those."""
        with pytest.warns(SpecificationWarning, match="has 5 with any inertia"):
            assert MCA(n_components=7).fit(batches).n_components_ == 5


class TestMCACorrections:
    def test_benzecri_against_prince(self, batches: pd.DataFrame) -> None:
        mca = MCA(n_components=3, correction="benzecri").fit(batches)
        np.testing.assert_allclose(100 * mca.corrected_explained_inertia_, [88.7944, 11.2056, 0.0], atol=1e-4)

    def test_greenacre_sums_over_every_axis(self, batches: pd.DataFrame) -> None:
        """The adjusted total is the Burt matrix's inertia, a sum over *all* axes.

        prince 0.21 sums only over the axes kept, which reports 87.85% here instead of
        79.76% and makes the answer depend on ``n_components``. Computed directly from
        Greenacre's formula instead, with every eigenvalue.
        """
        every = MCA(n_components=5).fit(batches).eigenvalues_
        n_vars, n_levels = 3, 8
        adjusted = np.where(every > 1 / n_vars, (n_vars / (n_vars - 1)) ** 2 * (every - 1 / n_vars) ** 2, 0.0)
        total = n_vars / (n_vars - 1) * (np.sum(every**2) - (n_levels - n_vars) / n_vars**2)
        mca = MCA(n_components=2, correction="greenacre").fit(batches)
        np.testing.assert_allclose(mca.corrected_explained_inertia_, adjusted[:2] / total)
        np.testing.assert_allclose(100 * mca.corrected_explained_inertia_, [79.7588, 10.0653], atol=1e-4)

    @pytest.mark.parametrize("correction", ["benzecri", "greenacre"])
    def test_a_correction_does_not_depend_on_how_many_axes_are_kept(
        self, batches: pd.DataFrame, correction: str
    ) -> None:
        one = MCA(n_components=1, correction=correction).fit(batches)
        three = MCA(n_components=3, correction=correction).fit(batches)
        np.testing.assert_allclose(one.corrected_explained_inertia_, three.corrected_explained_inertia_[:1])

    def test_unassociated_variables_give_zero_shares_not_nan(self) -> None:
        """Every eigenvalue at or below 1/Q corrects to zero; prince divides that by zero and reports NaN."""
        balanced = pd.DataFrame(
            {
                "grade": list("AABBCCABCABC"),
                "supplier": ["s1", "s2"] * 6,
                "shift": [
                    "day",
                    "day",
                    "night",
                    "night",
                    "day",
                    "night",
                    "night",
                    "day",
                    "day",
                    "night",
                    "day",
                    "night",
                ],
            }
        )
        mca = MCA(n_components=2, correction="benzecri").fit(balanced)
        assert np.all(np.isfinite(mca.corrected_explained_inertia_))

    def test_no_correction_keeps_the_raw_shares(self, batches: pd.DataFrame) -> None:
        mca = MCA(n_components=2).fit(batches)
        np.testing.assert_array_equal(mca.corrected_explained_inertia_, mca.explained_inertia_)

    def test_unknown_correction_refused(self, batches: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="correction must be one of"):
            MCA(correction="burt").fit(batches)


class TestMCASupplementary:
    def test_transforming_the_fitted_data_gives_the_fitted_coordinates(self, batches: pd.DataFrame) -> None:
        mca = MCA(n_components=3).fit(batches)
        np.testing.assert_allclose(mca.transform(batches), mca.row_coordinates_)

    def test_a_supplementary_outcome_lands_by_the_attributes_that_go_with_it(self, batches: pd.DataFrame) -> None:
        """The issue's use case: which attribute combinations go with good and bad batches.

        Here "good" is exactly the grade-A batches, so it must land on grade A, and it
        must do so without having shaped the axes.
        """
        mca = MCA(n_components=2).fit(batches)
        before = mca.row_coordinates_.copy()
        outcome = pd.DataFrame({"outcome": np.where(batches["grade"] == "A", "good", "bad")})
        placed = mca.transform_columns(outcome)
        np.testing.assert_allclose(placed.loc["outcome=good"], mca.column_coordinates_.loc["grade=A"], atol=1e-10)
        pd.testing.assert_frame_equal(mca.row_coordinates_, before)

    def test_an_unseen_level_is_refused(self, batches: pd.DataFrame) -> None:
        """Dropping it would leave the row's profile summing to less than one, misplacing it."""
        mca = MCA().fit(batches)
        new = batches.iloc[:1].copy()
        new["grade"] = "Z"
        with pytest.raises(ValueError, match=r"levels \['Z'\] that the model was not fitted on"):
            mca.transform(new)

    def test_the_variables_must_match(self, batches: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="Expected the variables"):
            MCA().fit(batches).transform(batches[["shift", "grade", "supplier"]])


class TestMCAInputs:
    def test_missing_values_are_refused_with_a_way_forward(self, batches: pd.DataFrame) -> None:
        gapped = batches.copy()
        gapped.iloc[2, 1] = None
        with pytest.raises(ValueError, match="fill the gaps with an explicit level"):
            MCA().fit(gapped)

    def test_one_variable_is_refused(self, batches: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="at least two categorical variables"):
            MCA().fit(batches[["grade"]])

    def test_levels_are_named_by_variable(self, batches: pd.DataFrame) -> None:
        assert "supplier=s2" in MCA().fit(batches).column_coordinates_.index

    def test_numeric_codes_are_treated_as_categories(self, batches: pd.DataFrame) -> None:
        coded = batches.assign(grade=batches["grade"].map({"A": 1, "B": 2, "C": 3}))
        assert pd.api.types.is_integer_dtype(coded["grade"])
        assert "grade=1" in MCA().fit(coded).column_coordinates_.index

    def test_map_plot_is_inherited(self, batches: pd.DataFrame) -> None:
        assert len(MCA().fit(batches).map_plot().data) == 2


# ---------------------------------------------------------------------------
# FAMD (#178)
# ---------------------------------------------------------------------------


@pytest.fixture
def batch_record(batches: pd.DataFrame) -> pd.DataFrame:
    """Numeric readings alongside the categorical attributes: temperature and pressure track grade."""
    return pd.DataFrame(
        {
            "temperature": [
                *(80.1, 81.3, 79.8, 82.0, 80.5, 75.2, 74.8, 76.1),
                *(75.5, 70.3, 69.8, 71.2, 70.9, 69.5, 80.8, 75.0),
            ],
            "pressure": [2.1, 2.3, 2.0, 2.4, 2.2, 1.8, 1.7, 1.9, 1.8, 1.4, 1.3, 1.5, 1.6, 1.2, 2.2, 1.9],
            "grade": batches["grade"],
            "supplier": batches["supplier"],
        }
    )


class TestFAMDAgainstReferences:
    def test_eigenvalues_against_prince(self, batch_record: pd.DataFrame) -> None:
        famd = FAMD(n_components=3).fit(batch_record)
        np.testing.assert_allclose(famd.eigenvalues_, [3.659437, 1.420221, 0.58708], atol=1e-6)

    def test_total_inertia_counts_each_variable_fairly(self, batch_record: pd.DataFrame) -> None:
        """One per numeric column, levels minus one per categorical column: 2 + 2 + 2.

        This is FAMD's whole point, and it holds only with population variance: the
        sample variance would put the total at 5.875 and tilt the balance.
        """
        assert FAMD().fit(batch_record).total_inertia_ == pytest.approx(6.0)

    def test_coordinates_against_prince(self, batch_record: pd.DataFrame) -> None:
        famd = FAMD(n_components=3).fit(batch_record)
        rows = np.array([[1.98877, 0.92202, 0.05145], [2.41886, 0.90349, 0.01531], [1.80937, 0.93273, 0.06818]])
        np.testing.assert_allclose(np.abs(famd.row_coordinates_.to_numpy()[:3]), rows, atol=1e-5)
        numeric = np.array([[0.99004, 0.02490, 0.01494], [0.96600, 0.05085, 0.04141]])
        np.testing.assert_allclose(
            np.abs(famd.column_coordinates_.loc[["temperature", "pressure"]]), numeric, atol=1e-5
        )


class TestFAMDReducesToPCAAndMCA:
    """The issue's defining claim, checked as two exact identities."""

    def test_all_numeric_is_pca_of_the_correlation_matrix(self, batch_record: pd.DataFrame) -> None:
        numeric = batch_record[["temperature", "pressure"]]
        expected = np.sort(np.linalg.eigvalsh(np.corrcoef(numeric.to_numpy().T)))[::-1]
        np.testing.assert_allclose(FAMD().fit(numeric).eigenvalues_, expected)

    def test_all_categorical_is_mca_scaled_by_q(self, batch_record: pd.DataFrame) -> None:
        """FAMD's matrix on categorical data is sqrt(Q) times CA's standardised residuals.

        So its eigenvalues are exactly Q times MCA's and its observations sit sqrt(Q)
        times further out. This cross-checks FAMD against MCA from an independent
        derivation, not just against a reference table.
        """
        categorical, n_vars = batch_record[["grade", "supplier"]], 2
        famd, mca = FAMD(n_components=3).fit(categorical), MCA(n_components=3).fit(categorical)
        np.testing.assert_allclose(famd.eigenvalues_, n_vars * mca.eigenvalues_)
        # Absolute tolerance: several observations sit exactly on an axis's zero, where
        # 0.0 against 2e-16 is an infinite relative error but no disagreement at all.
        np.testing.assert_allclose(
            np.abs(famd.row_coordinates_), np.sqrt(n_vars) * np.abs(mca.row_coordinates_), atol=1e-12
        )


class TestFAMDBehaviour:
    def test_numeric_coordinates_are_correlations(self, batch_record: pd.DataFrame) -> None:
        famd = FAMD().fit(batch_record)
        for column in ("temperature", "pressure"):
            correlation = np.corrcoef(batch_record[column], famd.row_coordinates_[1])[0, 1]
            assert famd.column_coordinates_.loc[column, 1] == pytest.approx(correlation)

    def test_contributions_account_for_everything(self, batch_record: pd.DataFrame) -> None:
        famd = FAMD(n_components=3).fit(batch_record)
        np.testing.assert_allclose(famd.row_contributions_.sum(), 1.0)
        np.testing.assert_allclose(famd.column_contributions_.sum(), 1.0)

    def test_transforming_the_fitted_data_gives_the_fitted_coordinates(self, batch_record: pd.DataFrame) -> None:
        famd = FAMD(n_components=3).fit(batch_record)
        np.testing.assert_allclose(famd.transform(batch_record), famd.row_coordinates_)

    def test_numeric_codes_can_be_declared_categorical(self, batch_record: pd.DataFrame) -> None:
        """A line coded 1, 2, 3 is a category, not a quantity; left numeric it would be scaled as one."""
        coded = batch_record.assign(line=[1, 2, 3] * 5 + [1])
        famd = FAMD(categorical=["line"]).fit(coded)
        assert "line" in famd.categorical_
        assert "line=2" in famd.column_coordinates_.index
        assert FAMD().fit(coded).numeric_ == ["temperature", "pressure", "line"]

    def test_booleans_are_categorical(self, batch_record: pd.DataFrame) -> None:
        famd = FAMD().fit(batch_record.assign(rework=[True, False] * 8))
        assert "rework" in famd.categorical_


class TestFAMDInputs:
    def test_missing_values_are_refused(self, batch_record: pd.DataFrame) -> None:
        gapped = batch_record.copy()
        gapped.iloc[3, 0] = np.nan
        with pytest.raises(ValueError, match="Impute the numeric ones"):
            FAMD().fit(gapped)

    def test_a_constant_numeric_column_is_refused(self, batch_record: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="constant"):
            FAMD().fit(batch_record.assign(setpoint=5.0))

    def test_n_components_must_be_positive(self, batch_record: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            FAMD(n_components=0).fit(batch_record)

    def test_a_table_with_no_variation_is_refused(self) -> None:
        """Single-level categorical columns centre to exactly zero: there is nothing to analyse."""
        with pytest.raises(ValueError, match="no variation"):
            FAMD().fit(pd.DataFrame({"grade": ["A"] * 6, "supplier": ["s1"] * 6}))

    def test_asking_for_more_axes_than_exist_warns_and_keeps_the_rest(self, batch_record: pd.DataFrame) -> None:
        """Two numeric columns and two three-level ones span 2 + 2 + 2 = 6 axes."""
        with pytest.warns(SpecificationWarning, match="keeping 6"):
            famd = FAMD(n_components=50).fit(batch_record)
        assert famd.n_components_ == 6

    def test_categorical_must_name_real_columns(self, batch_record: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="not in the data"):
            FAMD(categorical=["colour"]).fit(batch_record)

    def test_new_data_must_match(self, batch_record: pd.DataFrame) -> None:
        famd = FAMD().fit(batch_record)
        with pytest.raises(ValueError, match="Expected the columns"):
            famd.transform(batch_record[["pressure", "temperature", "grade", "supplier"]])
        unseen = batch_record.iloc[:1].assign(grade="Z")
        with pytest.raises(ValueError, match="not fitted on"):
            famd.transform(unseen)
        gapped = batch_record.copy()
        gapped.iloc[0, 0] = np.nan
        with pytest.raises(ValueError, match="must not have missing values"):
            famd.transform(gapped)

    def test_map_plot(self, batch_record: pd.DataFrame) -> None:
        fig = FAMD().fit(batch_record).map_plot()
        assert fig.layout.title.text == "FAMD map"
