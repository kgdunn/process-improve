"""Latent-variable methods for categorical, mixed and multi-block data (#176-#179).

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
from process_improve.multivariate.methods import CA, FAMD, MCA, MFA


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
        coded = batches.replace({"A": 1, "B": 2, "C": 3})
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

    def test_map_plot(self, batch_record: pd.DataFrame) -> None:
        fig = FAMD().fit(batch_record).map_plot()
        assert fig.layout.title.text == "FAMD map"


@pytest.fixture
def blocks() -> pd.DataFrame:
    """Twelve samples: four spectral readings and two lab results, mostly one driver.

    The ``prince`` (0.21) reference values below were computed from this table with the
    groups in :data:`GROUPS`.
    """
    return pd.DataFrame(
        {
            "spec1": [-0.572, -0.198, 0.267, -0.115, -0.699, 0.267, 1.427, 1.260, -0.742, -0.855, -0.823, 0.147],
            "spec2": [0.397, -0.104, 0.417, -0.172, -0.673, 0.428, 1.001, 0.884, -0.752, -1.103, -0.559, 0.148],
            "spec3": [-0.322, 0.093, -0.405, 0.343, 0.158, 0.093, -0.900, -0.713, 0.783, 1.171, 1.061, 0.547],
            "spec4": [0.846, 0.394, 0.783, -0.378, -0.537, 0.624, 0.789, 1.105, -0.532, -0.987, -1.097, -0.223],
            "lab1": [0.033, -0.849, 2.151, -0.038, -0.907, 0.594, 3.400, 2.554, -1.091, -3.633, -1.221, 0.424],
            "lab2": [1.004, -0.618, 1.822, -1.320, -0.662, 0.935, 0.049, 2.002, 0.189, -0.633, -0.378, -1.091],
        }
    )


GROUPS = {"spectra": ["spec1", "spec2", "spec3", "spec4"], "lab": ["lab1", "lab2"]}


def _driven_groups(seed: int, widths: tuple[int, ...], *, shared: bool) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Build groups of noisy copies of one driver each: the same driver for all when ``shared``."""
    rng = np.random.default_rng(seed)
    common = rng.normal(size=60)
    columns, groups = {}, {}
    for number, width in enumerate(widths):
        driver = common if shared else rng.normal(size=60)
        groups[f"g{number}"] = [f"g{number}_{i}" for i in range(width)]
        columns.update({name: driver + rng.normal(scale=0.3, size=60) for name in groups[f"g{number}"]})
    return pd.DataFrame(columns), groups


class TestMFAAgainstPrince:
    def test_eigenvalues_and_total_inertia(self, blocks: pd.DataFrame) -> None:
        mfa = MFA(GROUPS, n_components=3).fit(blocks)
        np.testing.assert_allclose(mfa.eigenvalues_, [1.888801, 0.369139, 0.077757], atol=1e-6)
        assert mfa.total_inertia_ == pytest.approx(2.40674, abs=1e-5)

    def test_row_and_partial_coordinates(self, blocks: pd.DataFrame) -> None:
        mfa = MFA(GROUPS, n_components=3).fit(blocks)
        rows = np.array([[0.62168, 0.64592, 0.54650], [0.43613, 0.26752, 0.51972], [1.61944, 0.64999, 0.18662]])
        np.testing.assert_allclose(np.abs(mfa.row_coordinates_.to_numpy()[:3]), rows, atol=1e-5)
        partial = mfa.partial_row_coordinates_
        np.testing.assert_allclose(np.abs(partial["spectra"].iloc[0]), [0.68311, 0.10375, 1.39513], atol=1e-5)
        np.testing.assert_allclose(np.abs(partial["lab"].iloc[0]), [0.56026, 1.18809, 0.30214], atol=1e-5)


class TestMFABalancesTheGroups:
    """What the group weighting is for, shown where plain PCA gets it wrong."""

    def test_a_wide_group_does_not_outvote_a_narrow_one(self) -> None:
        """Forty columns on one driver against two on an unrelated one.

        PCA of the concatenated table hands its first axis to the wide group, with 19
        times the inertia of the narrow group's axis; the narrow group makes up 0.1% of
        it. MFA caps each group's leading direction at an inertia of 1,
        so the two drivers come out as equals: an eigenvalue near 1 each, and one axis
        owned by each group.
        """
        data, groups = _driven_groups(seed=0, widths=(40, 2), shared=False)
        standardised = ((data - data.mean()) / data.std(ddof=0)).to_numpy()
        pca_eigenvalues = np.linalg.eigvalsh(standardised.T @ standardised / len(data))[::-1]
        assert pca_eigenvalues[0] / pca_eigenvalues[1] > 15

        mfa = MFA(groups).fit(data)
        np.testing.assert_allclose(mfa.eigenvalues_, [1.0, 1.0], atol=0.15)
        assert set(mfa.group_contributions_.idxmax()) == {"g0", "g1"}

    @pytest.mark.parametrize("n_groups", [2, 3])
    def test_first_eigenvalue_approaches_the_number_of_groups_when_they_agree(self, n_groups: int) -> None:
        """Each group adds at most 1 to an axis, so full agreement sums to the group count."""
        data, groups = _driven_groups(seed=1, widths=(5, 3, 2)[:n_groups], shared=True)
        eigenvalue = MFA(groups).fit(data).eigenvalues_[0]
        assert n_groups - 0.1 < eigenvalue <= n_groups

    def test_a_change_of_units_in_one_group_changes_nothing(self, blocks: pd.DataFrame) -> None:
        """Even unscaled, since the weight divides the group's own size back out.

        Reporting the lab results in thousandths would otherwise hand them the whole
        analysis: unscaled PCA's leading eigenvalue grows 700,000-fold.
        """
        in_thousandths = blocks.assign(**{column: 1000 * blocks[column] for column in GROUPS["lab"]})
        before = MFA(GROUPS, scale=False).fit(blocks)
        after = MFA(GROUPS, scale=False).fit(in_thousandths)
        np.testing.assert_allclose(after.eigenvalues_, before.eigenvalues_)
        np.testing.assert_allclose(after.row_coordinates_, before.row_coordinates_)

    def test_group_weights_are_one_over_each_groups_first_eigenvalue(self, blocks: pd.DataFrame) -> None:
        mfa = MFA(GROUPS).fit(blocks)
        for name, columns in GROUPS.items():
            leading = np.linalg.eigvalsh(np.corrcoef(blocks[columns].to_numpy().T))[-1]
            assert mfa.group_weights_[name] == pytest.approx(1 / leading)


class TestMFAIdentities:
    def test_each_observation_is_the_mean_of_its_partial_points(self, blocks: pd.DataFrame) -> None:
        mfa = MFA(GROUPS, n_components=3).fit(blocks)
        mean_partial = sum(mfa.partial_row_coordinates_.values()) / len(GROUPS)
        np.testing.assert_allclose(mean_partial, mfa.row_coordinates_)

    def test_group_coordinates_lie_in_the_unit_interval_and_sum_to_the_eigenvalues(self, blocks: pd.DataFrame) -> None:
        mfa = MFA(GROUPS, n_components=3).fit(blocks)
        assert ((mfa.group_coordinates_ >= 0) & (mfa.group_coordinates_ <= 1)).all().all()
        np.testing.assert_allclose(mfa.group_coordinates_.sum(), mfa.eigenvalues_)

    def test_column_coordinates_are_correlations(self, blocks: pd.DataFrame) -> None:
        mfa = MFA(GROUPS, n_components=3).fit(blocks)
        for column in blocks.columns:
            for axis in (1, 2, 3):
                correlation = np.corrcoef(blocks[column], mfa.row_coordinates_[axis])[0, 1]
                assert mfa.column_coordinates_.loc[column, axis] == pytest.approx(correlation)

    def test_contributions_account_for_everything(self, blocks: pd.DataFrame) -> None:
        mfa = MFA(GROUPS, n_components=3).fit(blocks)
        np.testing.assert_allclose(mfa.column_contributions_.sum(), 1.0)
        np.testing.assert_allclose(mfa.group_contributions_.sum(), 1.0)

    def test_first_eigenvalue_lies_between_one_and_the_number_of_groups(self, blocks: pd.DataFrame) -> None:
        assert 1 <= MFA(GROUPS).fit(blocks).eigenvalues_[0] <= len(GROUPS)


class TestMFANewObservations:
    def test_transforming_the_fitted_data_gives_the_fitted_coordinates(self, blocks: pd.DataFrame) -> None:
        mfa = MFA(GROUPS).fit(blocks)
        np.testing.assert_allclose(mfa.transform(blocks), mfa.row_coordinates_)

    def test_new_observations_use_the_fitted_centre_and_scale(self, blocks: pd.DataFrame) -> None:
        """Three samples on their own land where they did in the fit, not re-centred on themselves."""
        mfa = MFA(GROUPS).fit(blocks)
        np.testing.assert_allclose(mfa.transform(blocks.iloc[:3]), mfa.row_coordinates_.iloc[:3])

    def test_new_data_must_have_every_grouped_column(self, blocks: pd.DataFrame) -> None:
        mfa = MFA(GROUPS).fit(blocks)
        with pytest.raises(ValueError, match="lack the columns"):
            mfa.transform(blocks.drop(columns="lab2"))

    def test_new_data_must_be_complete(self, blocks: pd.DataFrame) -> None:
        mfa = MFA(GROUPS).fit(blocks)
        gapped = blocks.copy()
        gapped.iloc[0, 0] = np.nan
        with pytest.raises(ValueError, match="must not have missing values"):
            mfa.transform(gapped)


class TestMFAInputs:
    def test_columns_in_no_group_are_ignored(self, blocks: pd.DataFrame) -> None:
        with_extra = blocks.assign(operator=list("ABCABCABCABC"))
        np.testing.assert_allclose(MFA(GROUPS).fit(with_extra).eigenvalues_, MFA(GROUPS).fit(blocks).eigenvalues_)

    @pytest.mark.parametrize(
        ("groups", "message"),
        [
            ({"spectra": GROUPS["spectra"]}, "at least two groups"),
            ({"spectra": GROUPS["spectra"], "lab": []}, "has no columns"),
            ({"spectra": GROUPS["spectra"], "lab": ["lab1", "viscosity"]}, "not in the data"),
            ({"spectra": GROUPS["spectra"], "lab": ["lab1", "spec1"]}, "is in both group"),
        ],
    )
    def test_invalid_groups_are_refused(self, blocks: pd.DataFrame, groups: dict, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            MFA(groups).fit(blocks)

    def test_a_categorical_column_points_to_famd(self, blocks: pd.DataFrame) -> None:
        with_grade = blocks.assign(grade=list("ABABABABABAB"))
        with pytest.raises(ValueError, match="use FAMD"):
            MFA({**GROUPS, "context": ["grade"]}).fit(with_grade)

    def test_missing_values_are_refused(self, blocks: pd.DataFrame) -> None:
        gapped = blocks.copy()
        gapped.iloc[2, 4] = np.nan
        with pytest.raises(ValueError, match="complete data"):
            MFA(GROUPS).fit(gapped)

    def test_a_constant_column_is_refused(self, blocks: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="constant column"):
            MFA({**GROUPS, "lab": ["lab1", "lab2", "setpoint"]}).fit(blocks.assign(setpoint=5.0))

    def test_an_unscaled_group_with_no_variation_is_refused(self, blocks: pd.DataFrame) -> None:
        constant = blocks.assign(setpoint=5.0, target=2.0)
        with pytest.raises(ValueError, match="no variation"):
            MFA({**GROUPS, "fixed": ["setpoint", "target"]}, scale=False).fit(constant)

    def test_n_components_must_be_positive(self, blocks: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            MFA(GROUPS, n_components=0).fit(blocks)

    def test_asking_for_more_axes_than_exist_warns_and_keeps_the_rest(self, blocks: pd.DataFrame) -> None:
        with pytest.warns(SpecificationWarning, match="keeping 6"):
            mfa = MFA(GROUPS, n_components=10).fit(blocks)
        assert mfa.n_components_ == 6

    def test_map_plot(self, blocks: pd.DataFrame) -> None:
        fig = MFA(GROUPS).fit(blocks).map_plot()
        assert fig.layout.title.text == "MFA map"
        assert fig.layout.xaxis.title.text.startswith("Axis 1")
