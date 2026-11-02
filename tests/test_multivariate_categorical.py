"""Latent-variable methods for categorical, mixed and multi-block data (#176-#179).

Reference values come from two independent sources: Greenacre's published analysis
of his staff-by-smoking table, and the ``prince`` library (0.21), which reproduces
those published figures exactly. Beyond matching them, the tests pin the identities
that define correspondence analysis, since a method can match one table by accident
but not an identity.
"""

from __future__ import annotations

import itertools
import re
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from scipy.stats import chi2_contingency
from sklearn.exceptions import NotFittedError

from process_improve.multivariate._categorical import _flip_signs
from process_improve.multivariate._common import SpecificationWarning
from process_improve.multivariate.methods import CA, FAMD, MBPCA, MCA, MFA
from process_improve.visualization.themes import DEFAULT_THEME


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
# Checked by hand: each number against a calculation that does not use CA
# ---------------------------------------------------------------------------


def _ca_by_hand(table: pd.DataFrame, axes: int | None = None) -> dict[str, np.ndarray]:
    """Greenacre's formulas: the standardised residuals, their SVD, and the principal coordinates.

    ``axes`` is how many have inertia: ``min(I, J) - 1`` on a table of full rank, the default. Each
    axis is signed so that the largest-magnitude entry of its left singular vector is positive.
    """
    counts = table.to_numpy(dtype=float)
    proportions = counts / counts.sum()
    rows, columns = proportions.sum(axis=1), proportions.sum(axis=0)
    expected = np.outer(rows, columns)
    u, singular, vt = np.linalg.svd((proportions - expected) / np.sqrt(expected))
    axes = min(counts.shape) - 1 if axes is None else axes
    u, singular, v = u[:, :axes], singular[:axes], vt[:axes].T
    signs = np.sign(u[np.abs(u).argmax(axis=0), np.arange(axes)])
    u, v = u * signs, v * signs
    return {
        "rows": rows,
        "columns": columns,
        "eigenvalues": singular**2,
        "row_standard": u / np.sqrt(rows)[:, np.newaxis],
        "column_standard": v / np.sqrt(columns)[:, np.newaxis],
        "F": u / np.sqrt(rows)[:, np.newaxis] * singular,
        "G": v / np.sqrt(columns)[:, np.newaxis] * singular,
    }


def _axes_frame(values: np.ndarray, index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(values, index=index, columns=list(range(1, values.shape[1] + 1)))


class TestCAByHand:
    def test_every_attribute_of_a_fit_that_keeps_two_of_three_axes(self, smoke: pd.DataFrame) -> None:
        """Masses, inertias, coordinates, contributions and cos2, labels included.

        cos2 divides by a category's squared distance over *every* axis, here three, not the two kept.
        """
        ca = CA().fit(smoke)
        hand = _ca_by_hand(smoke)
        F, G, eigenvalues = hand["F"], hand["G"], hand["eigenvalues"]

        assert ca.n_components_ == 2
        np.testing.assert_allclose(ca.eigenvalues_, eigenvalues[:2], rtol=1e-12)
        assert ca.total_inertia_ == pytest.approx(eigenvalues.sum(), rel=1e-12)
        np.testing.assert_allclose(ca.explained_inertia_, eigenvalues[:2] / eigenvalues.sum(), rtol=1e-12)
        pd.testing.assert_series_equal(ca.row_masses_, pd.Series(hand["rows"], index=smoke.index, name="mass"))
        pd.testing.assert_series_equal(ca.column_masses_, pd.Series(hand["columns"], index=smoke.columns, name="mass"))
        expected = {
            "row_coordinates_": _axes_frame(F[:, :2], smoke.index),
            "column_coordinates_": _axes_frame(G[:, :2], smoke.columns),
            "row_contributions_": _axes_frame(hand["rows"][:, None] * F[:, :2] ** 2 / eigenvalues[:2], smoke.index),
            "column_contributions_": _axes_frame(
                hand["columns"][:, None] * G[:, :2] ** 2 / eigenvalues[:2], smoke.columns
            ),
            "row_cos2_": _axes_frame(F[:, :2] ** 2 / (F**2).sum(axis=1, keepdims=True), smoke.index),
            "column_cos2_": _axes_frame(G[:, :2] ** 2 / (G**2).sum(axis=1, keepdims=True), smoke.columns),
        }
        for name, frame in expected.items():
            pd.testing.assert_frame_equal(getattr(ca, name), frame, rtol=0, atol=1e-12, obj=name)

    def test_the_signs_follow_the_largest_entry_of_each_left_singular_vector(self) -> None:
        """Each axis is flipped, left and right together, when its largest-magnitude left entry is negative."""
        left = np.array([[0.6, 0.8], [-0.8, 0.6]])
        right = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        flipped_left, flipped_right = _flip_signs(left, right)
        np.testing.assert_array_equal(flipped_left, [[-0.6, 0.8], [0.8, 0.6]])
        np.testing.assert_array_equal(flipped_right, [[-1.0, 2.0], [-3.0, 4.0], [-5.0, 6.0]])

    def test_an_array_is_read_as_a_table_numbered_from_zero(self, smoke: pd.DataFrame) -> None:
        ca = CA(n_components=3).fit(smoke.to_numpy())
        hand = _ca_by_hand(smoke)
        pd.testing.assert_frame_equal(ca.row_coordinates_, _axes_frame(hand["F"], pd.RangeIndex(5)), rtol=0, atol=1e-12)
        pd.testing.assert_frame_equal(
            ca.column_coordinates_, _axes_frame(hand["G"], pd.RangeIndex(4)), rtol=0, atol=1e-12
        )

    def test_supplementary_rows_and_columns_by_the_transition_formula(self, smoke: pd.DataFrame) -> None:
        """A row's profile times the column standard coordinates; a column's profile times the row ones."""
        ca = CA().fit(smoke)
        hand = _ca_by_hand(smoke)
        rows = pd.DataFrame([[40, 10, 5, 1], [1, 2, 3, 30]], columns=smoke.columns, index=["new", "other"])
        profiles = rows.to_numpy() / rows.to_numpy().sum(axis=1, keepdims=True)
        pd.testing.assert_frame_equal(
            ca.transform(rows),
            _axes_frame(profiles @ hand["column_standard"][:, :2], rows.index),
            rtol=0,
            atol=1e-12,
        )
        columns = pd.DataFrame({"pass": [3, 9, 40, 70, 20], "fail": [8, 9, 11, 18, 5]}, index=smoke.index)
        profiles = columns.to_numpy() / columns.to_numpy().sum(axis=0)
        pd.testing.assert_frame_equal(
            ca.transform_columns(columns),
            _axes_frame(profiles.T @ hand["row_standard"][:, :2], columns.columns),
            rtol=0,
            atol=1e-12,
        )
        placed = ca.transform(rows.to_numpy())
        pd.testing.assert_index_equal(placed.index, pd.RangeIndex(2))
        np.testing.assert_allclose(placed.to_numpy(), ca.transform(rows).to_numpy(), rtol=0, atol=0)

    def test_a_supplementary_point_with_an_empty_cell_is_placed(self, smoke: pd.DataFrame) -> None:
        """A zero count is a profile like any other; only a row (or column) that sums to zero has none.

        The check summed along the other axis: one new row with no heavy smokers was refused because
        its "heavy" column summed to zero, and a row of zeros among others came back as NaN.
        """
        ca = CA().fit(smoke)
        hand = _ca_by_hand(smoke)
        row = pd.DataFrame([[40, 10, 5, 0]], columns=smoke.columns, index=["new"])
        pd.testing.assert_frame_equal(
            ca.transform(row),
            _axes_frame(row.to_numpy() / 55 @ hand["column_standard"][:, :2], row.index),
            rtol=0,
            atol=1e-12,
        )
        columns = pd.DataFrame({"pass": [3, 0, 40, 70, 20], "fail": [8, 0, 11, 18, 5]}, index=smoke.index)
        profiles = columns.to_numpy() / columns.to_numpy().sum(axis=0)
        pd.testing.assert_frame_equal(
            ca.transform_columns(columns),
            _axes_frame(profiles.T @ hand["row_standard"][:, :2], columns.columns),
            rtol=0,
            atol=1e-12,
        )

    @pytest.mark.parametrize("method", ["transform", "transform_columns", "map_plot"])
    def test_an_unfitted_model_is_refused(self, smoke: pd.DataFrame, method: str) -> None:
        arguments = () if method == "map_plot" else (smoke,)
        with pytest.raises(NotFittedError):
            getattr(CA(), method)(*arguments)


def _zero_sum_message(label: str, names: list) -> str:
    return re.escape(
        f"{label} {names} of X sum to zero. A category that never occurs has no profile to compare, and its mass "
        "of zero would divide the chi-squared distance by zero. Drop it before fitting."
    )


class TestCARefusals:
    @pytest.mark.parametrize(
        ("table", "message"),
        [
            (
                pd.DataFrame([[1.0, 2.0, 3.0]]),
                r"X must be a 2-D table with at least two rows and two columns; got \(1, 3\)\.",
            ),
            (pd.DataFrame([[1.0, np.inf], [3.0, 4.0]]), r"X must contain only finite counts \(no NaN or infinity\)\."),
            # A missing count in a nullable integer table is NaN as a float, not an object numpy cannot test.
            (
                pd.DataFrame([[1, pd.NA], [3, 4]], dtype="Int64"),
                r"X must contain only finite counts \(no NaN or infinity\)\.",
            ),
            (pd.DataFrame([[1.0, -2.0], [3.0, 4.0]]), r"X must not contain negative counts\."),
            (pd.DataFrame([[0, 0], [3, 4]], index=["a", "b"]), _zero_sum_message("Rows", ["a"])),
            (pd.DataFrame([[1, 0, 2], [3, 0, 4]], columns=["p", "q", "r"]), _zero_sum_message("Columns", ["q"])),
        ],
    )
    def test_invalid_tables_are_refused_in_full(self, table: pd.DataFrame, message: str) -> None:
        with pytest.raises(ValueError, match=f"^{message}$"):
            CA().fit(table)

    def test_n_components_below_one_is_refused_and_one_is_kept(self, smoke: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match=r"^n_components must be at least 1; got 0\.$"):
            CA(n_components=0).fit(smoke)
        assert CA(n_components=1).fit(smoke).row_coordinates_.shape == (5, 1)

    def test_no_association_is_refused_in_full(self) -> None:
        independent = pd.DataFrame(np.outer([10, 20, 30], [1, 2, 3, 4]).astype(float))
        message = (
            "The table shows no association between its rows and columns: every row has the same profile, so the "
            "total inertia is zero and there are no axes to find. That is a result, not a fault in the data, but "
            "there is no map to draw from it."
        )
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            CA().fit(independent)

    @pytest.mark.parametrize(("delta", "kept"), [(1.0, False), (10.0, True)])
    def test_an_axis_is_kept_only_above_an_inertia_of_1e_minus_12(self, delta: float, kept: bool) -> None:
        """A 2 x 2 table's one inertia is (ad - bc)^2 over the product of its margins.

        With a = b = c = 1e6 and d = 1e6 + delta that is 6.25e-14 delta^2: below 1e-12 for delta = 1,
        so the axis is noise; above it for delta = 10, so it is kept.
        """
        a = b = c = 1e6
        d = 1e6 + delta
        inertia = (a * d - b * c) ** 2 / ((a + b) * (c + d) * (a + c) * (b + d))
        table = pd.DataFrame([[a, b], [c, d]])
        if not kept:
            assert inertia < 1e-12
            with pytest.raises(ValueError, match="no association"):
                CA().fit(table)
            return
        assert inertia > 1e-12
        with pytest.warns(SpecificationWarning):
            ca = CA().fit(table)
        assert ca.total_inertia_ == pytest.approx(inertia, rel=1e-6)

    def test_too_many_axes_are_clamped_with_the_tables_shape_in_the_warning(self) -> None:
        """Two rows of one profile in a 3 x 4 table leave one axis."""
        table = pd.DataFrame([[10, 20, 30, 40], [20, 40, 60, 80], [30, 10, 5, 1]], dtype=float)
        message = "Asked for 2 axes, but this 3 x 4 table has 1 with any inertia; keeping 1."
        with pytest.warns(SpecificationWarning, match=f"^{re.escape(message)}$"):
            CA().fit(table)

    def test_asking_for_exactly_the_axes_there_are_does_not_warn(self, smoke: pd.DataFrame) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert CA(n_components=3).fit(smoke).n_components_ == 3

    def test_supplementary_points_are_refused_in_full(self, smoke: pd.DataFrame) -> None:
        ca = CA().fit(smoke)
        shared = "must match the fitted table along the shared dimension"
        cases = [
            (ca.transform, np.ones((2, 3)), rf"^Supplementary rows {shared}: expected 4, got shape \(2, 3\)\.$"),
            (
                ca.transform_columns,
                np.ones((4, 2)),
                rf"^Supplementary columns {shared}: expected 5, got shape \(4, 2\)\.$",
            ),
            (
                ca.transform,
                pd.DataFrame([[1, 2, 3, pd.NA]], dtype="Int64"),
                r"^Supplementary rows must be finite, non-negative counts\.$",
            ),
            (
                ca.transform_columns,
                np.array([[1.0], [2.0], [-3.0], [4.0], [5.0]]),
                r"^Supplementary columns must be finite, non-negative counts\.$",
            ),
            # One row of two sums to zero: each is checked, not their total.
            (
                ca.transform,
                np.array([[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0]]),
                r"^A supplementary row sums to zero, so it has no profile to place\.$",
            ),
            (
                ca.transform_columns,
                np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]]),
                r"^A supplementary column sums to zero, so it has no profile to place\.$",
            ),
        ]
        for method, counts, message in cases:
            with pytest.raises(ValueError, match=message):
                method(counts)


class TestMapPlot:
    def test_the_points_and_the_layout(self, smoke: pd.DataFrame) -> None:
        ca = CA(n_components=3).fit(smoke)
        fig = ca.map_plot()
        for trace, frame, name, symbol in zip(
            fig.data,
            (ca.row_coordinates_, ca.column_coordinates_),
            ("rows", "columns"),
            ("circle", "triangle-up"),
            strict=True,
        ):
            np.testing.assert_array_equal(trace.x, frame[1].to_numpy())
            np.testing.assert_array_equal(trace.y, frame[2].to_numpy())
            assert (trace.name, trace.mode, trace.textposition) == (name, "markers+text", "top center")
            assert list(trace.text) == [str(label) for label in frame.index]
            assert (trace.marker.color, trace.marker.symbol, trace.marker.size) == (None, symbol, 10)
            assert trace.hovertemplate == f"%{{text}}: (%{{x:.3f}}, %{{y:.3f}})<extra>{name}</extra>"
        share = 100 * _ca_by_hand(smoke)["eigenvalues"] / _ca_by_hand(smoke)["eigenvalues"].sum()
        layout = fig.layout
        assert layout.title.text == "Correspondence analysis map"
        assert layout.xaxis.title.text == f"Axis 1 ({share[0]:.1f}% of inertia)"
        assert layout.yaxis.title.text == f"Axis 2 ({share[1]:.1f}% of inertia)"
        assert (layout.xaxis.zeroline, layout.yaxis.zeroline) == (True, True)
        assert (layout.yaxis.scaleanchor, layout.yaxis.scaleratio) == ("x", 1)
        assert (layout.width, layout.height) == (600, 600)
        assert layout.template == go.Figure(layout={"template": DEFAULT_THEME}).layout.template

    def test_other_axes_and_settings(self, smoke: pd.DataFrame) -> None:
        ca = CA(n_components=3).fit(smoke)
        settings = {
            "title": "Staff",
            "row_color": "red",
            "column_color": "blue",
            "html_image_height": 400,
            "html_aspect_ratio_w_over_h": 1.5,
            "template": "plotly_white",
        }
        fig = ca.map_plot(3, 1, settings=settings)
        np.testing.assert_array_equal(fig.data[0].x, ca.row_coordinates_[3].to_numpy())
        np.testing.assert_array_equal(fig.data[1].y, ca.column_coordinates_[1].to_numpy())
        assert (fig.data[0].marker.color, fig.data[1].marker.color) == ("red", "blue")
        share = 100 * ca.explained_inertia_
        assert fig.layout.xaxis.title.text == f"Axis 3 ({share[2]:.1f}% of inertia)"
        assert fig.layout.yaxis.title.text == f"Axis 1 ({share[0]:.1f}% of inertia)"
        assert (fig.layout.title.text, fig.layout.width, fig.layout.height) == ("Staff", 600, 400)
        assert fig.layout.template == go.Figure(layout={"template": "plotly_white"}).layout.template
        with pytest.raises(ValueError, match=r"^Axes run from 1 to 3; got 0\.$"):
            ca.map_plot(0, 1)
        with pytest.raises(ValueError, match=r"^Axes run from 1 to 3; got 4\.$"):
            ca.map_plot(1, 4)


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


def _indicator_by_hand(frame: pd.DataFrame) -> pd.DataFrame:
    """One 0/1 column per level, sorted as text within each variable, named "variable=level"."""
    columns = {}
    for variable in frame.columns:
        text = frame[variable].astype(str)
        for level in sorted(text.unique()):
            columns[f"{variable}={level}"] = (text == level).to_numpy(dtype=float)
    return pd.DataFrame(columns, index=frame.index)


def _adjusted_by_hand(eigenvalues: np.ndarray, n_vars: int) -> np.ndarray:
    """Benzecri: (Q / (Q - 1))^2 (lambda - 1/Q)^2 above 1/Q, and 0 at or below it."""
    return np.array(
        [(n_vars / (n_vars - 1)) ** 2 * (e - 1 / n_vars) ** 2 if e > 1 / n_vars else 0.0 for e in eigenvalues]
    )


class TestMCAByHand:
    def test_mca_is_ca_of_the_indicator_matrix(self, batches: pd.DataFrame) -> None:
        """Eight levels of three variables: J - Q = 5 axes with inertia, 2 kept by default."""
        mca = MCA().fit(batches)
        indicator = _indicator_by_hand(batches)
        hand = _ca_by_hand(indicator, axes=5)
        assert mca.n_components_ == 2
        assert mca.variables_ == ["grade", "supplier", "shift"]
        np.testing.assert_allclose(mca.eigenvalues_, hand["eigenvalues"][:2], rtol=1e-12)
        assert mca.total_inertia_ == pytest.approx(5 / 3, rel=1e-12)
        pd.testing.assert_frame_equal(
            mca.row_coordinates_, _axes_frame(hand["F"][:, :2], batches.index), rtol=0, atol=1e-12
        )
        pd.testing.assert_frame_equal(
            mca.column_coordinates_, _axes_frame(hand["G"][:, :2], indicator.columns), rtol=0, atol=1e-12
        )
        # Without a correction the "corrected" values are the raw ones.
        np.testing.assert_array_equal(mca.corrected_eigenvalues_, mca.eigenvalues_)
        np.testing.assert_array_equal(mca.corrected_explained_inertia_, mca.explained_inertia_)

    @pytest.mark.parametrize("correction", ["benzecri", "greenacre"])
    def test_the_corrections_at_three_variables(self, batches: pd.DataFrame, correction: str) -> None:
        """Benzecri divides by the sum of every adjusted eigenvalue; Greenacre by the Burt matrix's
        adjusted inertia, Q / (Q - 1) (sum lambda^2 - (J - Q) / Q^2), over every axis.
        """
        every = _ca_by_hand(_indicator_by_hand(batches), axes=5)["eigenvalues"]
        adjusted = _adjusted_by_hand(every, 3)
        total = adjusted.sum() if correction == "benzecri" else 3 / 2 * (np.sum(every**2) - 5 / 9)
        mca = MCA(n_components=3, correction=correction).fit(batches)
        np.testing.assert_allclose(mca.corrected_eigenvalues_, adjusted[:3], rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(mca.corrected_explained_inertia_, adjusted[:3] / total, rtol=1e-12, atol=1e-15)

    @pytest.mark.parametrize(
        "levels",
        [
            {"grade": ["A", "B"], "supplier": ["s1", "s2"]},
            {"grade": ["A", "B", "C"], "supplier": ["s1", "s2"]},
            {"grade": ["A", "B"], "supplier": ["s1", "s2"], "shift": ["day", "night"]},
        ],
    )
    @pytest.mark.parametrize("correction", ["benzecri", "greenacre"])
    def test_variables_crossed_in_full_have_no_corrected_inertia(self, levels: dict, correction: str) -> None:
        """Every combination once: no association, so every eigenvalue is exactly 1/Q and every share is 0.

        Rounding left an eigenvalue 1e-16 above 1/Q, and Benzecri's shares were that noise over itself:
        50% on axis 1 of the 2 x 2 crossing, 100% on the 3 x 2 one.
        """
        crossed = pd.DataFrame(list(itertools.product(*levels.values())), columns=list(levels))
        mca = MCA(correction=correction).fit(crossed)
        np.testing.assert_allclose(mca.eigenvalues_, [1 / len(levels)] * 2, rtol=1e-12)
        np.testing.assert_array_equal(mca.corrected_explained_inertia_, np.zeros(2), strict=True)

    def test_arrays_are_read_as_frames_numbered_from_zero(self, batches: pd.DataFrame) -> None:
        mca = MCA(n_components=3).fit(batches.to_numpy())
        reference = MCA(n_components=3).fit(batches)
        assert mca.variables_ == ["0", "1", "2"]
        assert list(mca.column_coordinates_.index[:3]) == ["0=A", "0=B", "0=C"]
        np.testing.assert_allclose(mca.row_coordinates_, reference.row_coordinates_, rtol=0, atol=1e-12)
        np.testing.assert_allclose(mca.transform(batches.to_numpy()[:4]), reference.row_coordinates_[:4], atol=1e-12)
        outcome = np.where(batches["grade"] == "A", "good", "bad")[:, np.newaxis]
        placed = mca.transform_columns(outcome)
        assert list(placed.index) == ["0=bad", "0=good"]

    def test_a_few_observations_are_placed_from_their_levels(self, batches: pd.DataFrame) -> None:
        """Their indicator rows, against the fitted levels, by the transition formula.

        Three observations leave levels out, so the indicator has columns that sum to zero; those
        rows were refused as "summing to zero" before.
        """
        mca = MCA().fit(batches)
        hand = _ca_by_hand(_indicator_by_hand(batches), axes=5)
        few = batches.iloc[[0, 5, 9]]
        rows = _indicator_by_hand(batches).iloc[[0, 5, 9]].to_numpy()
        pd.testing.assert_frame_equal(
            mca.transform(few), _axes_frame(rows / 3 @ hand["column_standard"][:, :2], few.index), rtol=0, atol=1e-12
        )

    @pytest.mark.parametrize("method", ["transform", "transform_columns"])
    def test_an_unfitted_model_is_refused(self, batches: pd.DataFrame, method: str) -> None:
        with pytest.raises(NotFittedError):
            getattr(MCA(), method)(batches)

    def test_the_refusals_in_full(self, batches: pd.DataFrame) -> None:
        gapped = batches.copy()
        gapped.iloc[2, 1] = None
        missing = (
            "Columns ['supplier'] have missing values. MCA needs every observation to have a level in every "
            "variable; fill the gaps with an explicit level (for example 'unknown') if their absence is itself "
            "informative, or drop those rows."
        )
        one = (
            "MCA needs at least two categorical variables; with one, use CA on its cross-tabulation against "
            "another variable, or simply its frequency table."
        )
        for frame, message in ((gapped, missing), (batches[["grade"]], one)):
            with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
                MCA().fit(frame)
        with pytest.raises(
            ValueError, match=r"^correction must be one of \(None, 'benzecri', 'greenacre'\); got 'burt'\.$"
        ):
            MCA(correction="burt").fit(batches)
        mca = MCA().fit(batches)
        reordered = "Expected the variables ['grade', 'supplier', 'shift']; got ['shift', 'grade', 'supplier']."
        with pytest.raises(ValueError, match=f"^{re.escape(reordered)}$"):
            mca.transform(batches[["shift", "grade", "supplier"]])
        unseen = batches.iloc[:1].assign(grade="Z")
        with pytest.raises(ValueError, match=r"^Column 'grade' has levels \['Z'\] that the model was not fitted on\.$"):
            mca.transform(unseen)


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


def _famd_matrix_by_hand(fitted: pd.DataFrame, rows: pd.DataFrame, categorical: list[str]) -> tuple[np.ndarray, list]:
    """Pages' coding, with the statistics of ``fitted`` applied to ``rows``.

    A numeric column is centred and divided by its population sd; a level of frequency p
    becomes delta / sqrt(p) - sqrt(p), delta its 0/1 indicator.
    """
    blocks, names = [], []
    for column in [c for c in fitted.columns if c not in categorical]:
        values = fitted[column].to_numpy(dtype=float)
        blocks.append((rows[column].to_numpy(dtype=float) - values.mean()) / values.std())
        names.append(column)
    for column in categorical:
        text = fitted[column].astype(str)
        for level in sorted(text.unique()):
            p = (text == level).mean()
            blocks.append((rows[column].astype(str) == level).to_numpy(dtype=float) / np.sqrt(p) - np.sqrt(p))
            names.append(f"{column}={level}")
    return np.column_stack(blocks), names


def _famd_by_hand(frame: pd.DataFrame, categorical: list[str], axes: int) -> dict:
    """Decompose the coded matrix over sqrt(n) by SVD, each axis signed as CA's, and project the rows."""
    coded, names = _famd_matrix_by_hand(frame, frame, categorical)
    u, singular, vt = np.linalg.svd(coded / np.sqrt(len(frame)))
    u, singular, v = u[:, :axes], singular[:axes], vt[:axes].T
    signs = np.sign(u[np.abs(u).argmax(axis=0), np.arange(axes)])
    v = v * signs
    return {"coded": coded, "names": names, "eigenvalues": singular**2, "V": v, "F": coded @ v, "singular": singular}


@pytest.fixture
def labelled_record(batch_record: pd.DataFrame) -> pd.DataFrame:
    return batch_record.set_axis([f"batch{i}" for i in range(16)])


class TestFAMDByHand:
    def test_every_attribute_by_hand(self, labelled_record: pd.DataFrame) -> None:
        """Two numeric columns and two three-level ones: 2 + 2 + 2 = 6 axes, 2 kept by default."""
        famd = FAMD().fit(labelled_record)
        hand = _famd_by_hand(labelled_record, ["grade", "supplier"], axes=6)
        F, V, eigenvalues = hand["F"], hand["V"], hand["eigenvalues"]
        assert (famd.numeric_, famd.categorical_) == (["temperature", "pressure"], ["grade", "supplier"])
        assert famd.n_components_ == 2
        np.testing.assert_allclose(famd.eigenvalues_, eigenvalues[:2], rtol=1e-12)
        assert famd.total_inertia_ == pytest.approx(6.0, rel=1e-12)
        np.testing.assert_allclose(famd.explained_inertia_, eigenvalues[:2] / 6, rtol=1e-12)
        index = labelled_record.index
        expected = {
            "row_coordinates_": _axes_frame(F[:, :2], index),
            "column_coordinates_": _axes_frame(V[:, :2] * hand["singular"][:2], pd.Index(hand["names"])),
            "row_contributions_": _axes_frame(F[:, :2] ** 2 / 16 / eigenvalues[:2], index),
            "column_contributions_": _axes_frame(V[:, :2] ** 2, pd.Index(hand["names"])),
            "row_cos2_": _axes_frame(F[:, :2] ** 2 / (F**2).sum(axis=1, keepdims=True), index),
        }
        for name, frame in expected.items():
            pd.testing.assert_frame_equal(getattr(famd, name), frame, rtol=0, atol=1e-12, obj=name)

        new = labelled_record.iloc[[2, 7, 11]].assign(temperature=[90.0, 70.0, 75.5]).set_axis(["n1", "n2", "n3"])
        coded, _ = _famd_matrix_by_hand(labelled_record, new, ["grade", "supplier"])
        pd.testing.assert_frame_equal(famd.transform(new), _axes_frame(coded @ V[:, :2], new.index), rtol=0, atol=1e-12)

    def test_a_row_at_the_centroid_has_a_cos2_of_zero(self) -> None:
        """Its squared distance over every axis is 0: its share of each is 0, not 0 / 0."""
        frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [2.0, 9.0, 5.0, 1.0, 8.0]})
        famd = FAMD().fit(frame)
        F = _famd_by_hand(frame, [], axes=2)["F"]
        assert np.all(F[2] == 0.0)
        cos2 = np.zeros_like(F)
        moved = [0, 1, 3, 4]
        cos2[moved] = F[moved] ** 2 / (F[moved] ** 2).sum(axis=1, keepdims=True)
        pd.testing.assert_frame_equal(famd.row_cos2_, _axes_frame(cos2, frame.index), rtol=0, atol=1e-12)

    def test_nullable_columns_are_read_as_floats(self, batch_record: pd.DataFrame) -> None:
        """An Int64 and a Float64 column, as pandas reads a file with nullable dtypes, fit as float64 ones do."""
        nullable = batch_record.assign(
            temperature=batch_record["temperature"].astype("Float64"),
            pressure=(10 * batch_record["pressure"]).round().astype("Int64"),
        )
        reference = batch_record.assign(pressure=(10 * batch_record["pressure"]).round())
        famd, expected = FAMD(n_components=3).fit(nullable), FAMD(n_components=3).fit(reference)
        np.testing.assert_allclose(famd.eigenvalues_, expected.eigenvalues_, rtol=1e-12)
        np.testing.assert_allclose(famd.transform(nullable), expected.row_coordinates_, rtol=0, atol=1e-12)

    def test_an_array_is_read_as_a_frame_numbered_from_zero(self, batch_record: pd.DataFrame) -> None:
        numeric = batch_record[["temperature", "pressure"]]
        famd, reference = FAMD().fit(numeric.to_numpy()), FAMD().fit(numeric)
        assert famd.numeric_ == ["0", "1"]
        pd.testing.assert_index_equal(famd.row_coordinates_.index, pd.RangeIndex(16))
        np.testing.assert_allclose(famd.row_coordinates_, reference.row_coordinates_, rtol=0, atol=1e-12)
        placed = famd.transform(numeric.to_numpy()[:3])
        pd.testing.assert_index_equal(placed.index, pd.RangeIndex(3))
        np.testing.assert_allclose(placed, reference.row_coordinates_[:3], rtol=0, atol=1e-12)

    def test_a_column_in_small_units_is_not_constant(self, batch_record: pd.DataFrame) -> None:
        """Standardised, it is the same column: only equal values make one constant."""
        small = batch_record.assign(pressure=batch_record["pressure"] * 1e-12)
        np.testing.assert_allclose(
            FAMD(n_components=6).fit(small).eigenvalues_, FAMD(n_components=6).fit(batch_record).eigenvalues_
        )

    @pytest.mark.parametrize("rows", [12, 16])
    def test_a_constant_numeric_column_is_refused_whatever_its_sd_rounds_to(
        self, batch_record: pd.DataFrame, rows: int
    ) -> None:
        """0.1 twelve times has a population sd of 1.4e-17, not 0, and sixteen times one of exactly 0.

        The column was refused only at an sd of exactly 0: over twelve rows it became a column of
        -1, uncentred, and added an inertia of 1 to the total, an axis of its own.
        """
        assert bool(np.full(rows, 0.1).std() == 0.0) is (rows == 16)
        message = "Numeric columns ['flat'] are constant, so they carry no information. Drop them."
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            FAMD().fit(batch_record.iloc[:rows].assign(flat=0.1))

    def test_the_number_of_axes(self, batch_record: pd.DataFrame) -> None:
        assert FAMD(n_components=1).fit(batch_record).row_coordinates_.shape == (16, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert FAMD(n_components=6).fit(batch_record).n_components_ == 6
        message = "Asked for 50 axes, but this table has 6 with any inertia; keeping 6."
        with pytest.warns(SpecificationWarning, match=f"^{re.escape(message)}$"):
            FAMD(n_components=50).fit(batch_record)

    def test_the_refusals_in_full(self, batch_record: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match=r"^n_components must be at least 1; got 0\.$"):
            FAMD(n_components=0).fit(batch_record)
        gapped = batch_record.copy()
        gapped.iloc[3, 0] = np.nan
        missing = (
            "Columns ['temperature'] have missing values. Impute the numeric ones, give the categorical ones an "
            "explicit level, or drop those rows."
        )
        with pytest.raises(ValueError, match=f"^{re.escape(missing)}$"):
            FAMD().fit(gapped)
        with pytest.raises(ValueError, match=r"^categorical names columns that are not in the data: \['colour'\]\.$"):
            FAMD(categorical=["colour"]).fit(batch_record)
        with pytest.raises(
            ValueError, match=r"^The table has no variation to analyse: every observation is the same\.$"
        ):
            FAMD().fit(pd.DataFrame({"grade": ["A"] * 6, "supplier": ["s1"] * 6}))
        famd = FAMD().fit(batch_record)
        reordered = (
            "Expected the columns ['temperature', 'pressure', 'grade', 'supplier']; got "
            "['pressure', 'temperature', 'grade', 'supplier']."
        )
        with pytest.raises(ValueError, match=f"^{re.escape(reordered)}$"):
            famd.transform(batch_record[["pressure", "temperature", "grade", "supplier"]])
        with pytest.raises(ValueError, match=r"^New observations must not have missing values\.$"):
            famd.transform(gapped)
        with pytest.raises(NotFittedError):
            FAMD().transform(batch_record)

    def test_map_plot_at_its_defaults_and_with_settings(self, batch_record: pd.DataFrame) -> None:
        famd = FAMD(n_components=3).fit(batch_record)
        fig = famd.map_plot()
        np.testing.assert_array_equal(fig.data[0].x, famd.row_coordinates_[1].to_numpy())
        np.testing.assert_array_equal(fig.data[1].y, famd.column_coordinates_[2].to_numpy())
        assert fig.layout.title.text == "FAMD map"
        assert famd.map_plot(3, 2, settings={"title": "Mixed"}).layout.title.text == "Mixed"


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


class TestMFAAgainstMBPCA:
    def test_the_two_weightings_agree_when_blocks_share_their_first_eigenvalue_share(self) -> None:
        """MBPCA divides a block by its width, MFA by its first eigenvalue: equal when their ratio is.

        Both blocks here have eigenvalues 3 and 1 over four columns, so the leading
        direction holds 75% of each and the two weightings differ by one constant.
        MBPCA's NIPALS and MFA's SVD are independent implementations, so agreeing to
        rounding error checks each against the other.
        """
        rng = np.random.default_rng(1)

        def centred(values: np.ndarray) -> np.ndarray:
            return values - values.mean()

        def orthogonal_to(values: np.ndarray, other: np.ndarray) -> np.ndarray:
            values, other = centred(values), centred(other)
            return values - (values @ other) / (other @ other) * other

        first = centred(rng.normal(size=50))
        second = centred(0.7 * first + rng.normal(size=50))
        blocks = {
            "a": pd.DataFrame({"a0": first, "a1": first, "a2": first, "a3": orthogonal_to(rng.normal(size=50), first)}),
            "b": pd.DataFrame(
                {"b0": second, "b1": second, "b2": second, "b3": orthogonal_to(rng.normal(size=50), second)}
            ),
        }
        mbpca = MBPCA(n_components=2).fit(blocks).super_scores_.to_numpy()
        mfa = MFA(n_components=2).fit(blocks).row_coordinates_.to_numpy()
        for axis in range(2):
            assert abs(np.corrcoef(mbpca[:, axis], mfa[:, axis])[0, 1]) == pytest.approx(1.0, abs=1e-9)


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

    def test_a_dict_of_blocks_gives_the_same_fit_as_one_frame_with_groups(self, blocks: pd.DataFrame) -> None:
        """The MBPCA input form, so the two weightings can be compared on the same data."""
        as_blocks = {name: blocks[columns] for name, columns in GROUPS.items()}
        from_blocks, from_groups = MFA().fit(as_blocks), MFA(GROUPS).fit(blocks)
        assert from_blocks.groups_ == GROUPS
        np.testing.assert_allclose(from_blocks.eigenvalues_, from_groups.eigenvalues_)
        np.testing.assert_allclose(from_blocks.transform(as_blocks), from_groups.row_coordinates_)

    def test_blocks_and_groups_together_are_refused(self, blocks: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="not both"):
            MFA(GROUPS).fit({name: blocks[columns] for name, columns in GROUPS.items()})

    def test_one_frame_needs_groups(self, blocks: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="groups="):
            MFA().fit(blocks)
        with pytest.raises(TypeError, match="must be a dict"):
            MFA([GROUPS["spectra"], GROUPS["lab"]]).fit(blocks)

    def test_blocks_must_share_their_rows(self, blocks: pd.DataFrame) -> None:
        shuffled = blocks[GROUPS["lab"]].iloc[::-1]
        with pytest.raises(ValueError, match="same row index"):
            MFA().fit({"spectra": blocks[GROUPS["spectra"]], "lab": shuffled})

    def test_column_names_must_be_unique_across_blocks(self, blocks: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="unique across blocks"):
            MFA().fit({"a": blocks[["spec1", "spec2"]], "b": blocks[["spec2", "lab1"]]})

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


def _mfa_by_hand(frame: pd.DataFrame, groups: dict[str, list[str]], *, scale: bool = True) -> dict:
    """Escofier and Pages: each group centred (and scaled), weighted by one over its first eigenvalue.

    The first eigenvalue is the largest of the group's covariance (or correlation) matrix, by
    eigvalsh; the weighted groups side by side are decomposed by SVD over sqrt(n), and every
    axis with inertia is kept, signed as CA's.
    """
    n, weighted, weights = len(frame), {}, {}
    for name, columns in groups.items():
        values = frame[columns].to_numpy(dtype=float)
        centred = values - values.mean(axis=0)
        if scale:
            centred = centred / values.std(axis=0)
        weights[name] = 1 / np.linalg.eigvalsh(centred.T @ centred / n)[-1]
        weighted[name] = centred * np.sqrt(weights[name])
    combined = np.hstack(list(weighted.values()))
    u, singular, vt = np.linalg.svd(combined / np.sqrt(n))
    axes = int(np.sum(singular**2 > 1e-12))
    signs = np.sign(u[np.abs(u)[:, :axes].argmax(axis=0), np.arange(axes)])
    v, singular = vt[:axes].T * signs, singular[:axes]
    rows, start = {}, 0
    for name, block in weighted.items():
        rows[name] = slice(start, start + block.shape[1])
        start += block.shape[1]
    column_weights = np.concatenate([[weights[name]] * len(columns) for name, columns in groups.items()])
    return {
        "weights": weights,
        "eigenvalues": singular**2,
        "F": combined @ v,
        "partial": {name: len(groups) * block @ v[rows[name]] for name, block in weighted.items()},
        "groups": {name: (v[rows[name]] ** 2).sum(axis=0) * singular**2 for name in groups},
        "columns": v * singular / np.sqrt(column_weights)[:, np.newaxis],
        "contributions": v**2,
    }


@pytest.fixture
def three_groups() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Three groups of 3, 2 and 2 columns on one driver, with labelled rows."""
    data, groups = _driven_groups(seed=2, widths=(3, 2, 2), shared=True)
    return data.set_axis([f"s{i}" for i in range(len(data))]), groups


class TestMFAByHand:
    @pytest.mark.parametrize("scale", [True, False])
    def test_every_attribute_by_hand(self, three_groups: tuple, scale: bool) -> None:
        """Three groups, so the third group's columns start past the first two groups' columns."""
        data, groups = three_groups
        if not scale:
            data = data.assign(g1_0=1000 * data["g1_0"], g2_1=1e-6 * data["g2_1"])
        mfa = MFA(groups, scale=scale).fit(data)
        hand = _mfa_by_hand(data, groups, scale=scale)
        eigenvalues, used = hand["eigenvalues"], [c for columns in groups.values() for c in columns]
        assert mfa.groups_ == groups
        pd.testing.assert_series_equal(mfa.group_weights_, pd.Series(hand["weights"], name="weight"), rtol=1e-12)
        assert mfa.n_components_ == 2
        np.testing.assert_allclose(mfa.eigenvalues_, eigenvalues[:2], rtol=1e-12)
        assert mfa.total_inertia_ == pytest.approx(eigenvalues.sum(), rel=1e-12)
        np.testing.assert_allclose(mfa.explained_inertia_, eigenvalues[:2] / eigenvalues.sum(), rtol=1e-12)
        expected = {
            "row_coordinates_": _axes_frame(hand["F"][:, :2], data.index),
            "column_coordinates_": _axes_frame(hand["columns"][:, :2], pd.Index(used)),
            "column_contributions_": _axes_frame(hand["contributions"][:, :2], pd.Index(used)),
            "group_coordinates_": _axes_frame(
                np.array([hand["groups"][name][:2] for name in groups]), pd.Index(list(groups))
            ),
            "group_contributions_": _axes_frame(
                np.array([hand["groups"][name][:2] / eigenvalues[:2] for name in groups]), pd.Index(list(groups))
            ),
        }
        for name, frame in expected.items():
            pd.testing.assert_frame_equal(getattr(mfa, name), frame, rtol=0, atol=1e-10, obj=name)
        assert list(mfa.partial_row_coordinates_) == list(groups)
        for name, partial in hand["partial"].items():
            pd.testing.assert_frame_equal(
                mfa.partial_row_coordinates_[name], _axes_frame(partial[:, :2], data.index), rtol=0, atol=1e-10
            )
        new = data.iloc[[3, 8]].set_axis(["n1", "n2"])
        pd.testing.assert_frame_equal(
            mfa.transform(new), _axes_frame(hand["F"][[3, 8], :2], new.index), rtol=0, atol=1e-10
        )

    def test_an_array_is_read_as_a_frame_numbered_from_zero(self, blocks: pd.DataFrame) -> None:
        by_position = {"spectra": [0, 1, 2, 3], "lab": [4, 5]}
        mfa, reference = MFA(by_position).fit(blocks.to_numpy()), MFA(GROUPS).fit(blocks)
        pd.testing.assert_index_equal(mfa.row_coordinates_.index, pd.RangeIndex(12))
        np.testing.assert_allclose(mfa.row_coordinates_, reference.row_coordinates_, rtol=0, atol=1e-12)
        np.testing.assert_allclose(mfa.transform(blocks.to_numpy()), reference.row_coordinates_, rtol=0, atol=1e-12)

    def test_nullable_columns_are_read_as_floats(self, blocks: pd.DataFrame) -> None:
        """An Int64 and a Float64 column in one group, as pandas reads a file with nullable dtypes."""
        rounded = blocks.assign(lab1=(1000 * blocks["lab1"]).round())
        nullable = rounded.assign(lab1=rounded["lab1"].astype("Int64"), lab2=rounded["lab2"].astype("Float64"))
        mfa, reference = MFA(GROUPS).fit(nullable), MFA(GROUPS).fit(rounded)
        np.testing.assert_allclose(mfa.eigenvalues_, reference.eigenvalues_, rtol=1e-12)
        np.testing.assert_allclose(mfa.transform(nullable), reference.row_coordinates_, rtol=0, atol=1e-12)

    @pytest.mark.parametrize("scale", [True, False])
    def test_a_group_far_from_zero_fits_as_it_does_near_zero(self, blocks: pd.DataFrame, scale: bool) -> None:
        """Lab results 1e7 above zero vary as much as they do near it: the offset is centred away.

        A group was refused for "no variation" when its first eigenvalue was below 1e-12 of the
        mean square of its raw values, which an offset of 1e7 on values of sd 1 makes 1e14.
        """
        offset = blocks.assign(lab1=blocks["lab1"] + 1e7, lab2=blocks["lab2"] + 1e7)
        mfa, reference = MFA(GROUPS, scale=scale).fit(offset), MFA(GROUPS, scale=scale).fit(blocks)
        np.testing.assert_allclose(mfa.eigenvalues_, reference.eigenvalues_, rtol=1e-6)
        np.testing.assert_allclose(mfa.row_coordinates_, reference.row_coordinates_, rtol=0, atol=1e-6)

    @pytest.mark.parametrize("scale", [True, False])
    def test_a_group_in_small_units_fits_as_it_does_in_large_ones(self, blocks: pd.DataFrame, scale: bool) -> None:
        small = blocks.assign(lab1=blocks["lab1"] * 1e-12, lab2=blocks["lab2"] * 1e-12)
        mfa, reference = MFA(GROUPS, scale=scale).fit(small), MFA(GROUPS, scale=scale).fit(blocks)
        np.testing.assert_allclose(mfa.eigenvalues_, reference.eigenvalues_, rtol=1e-10)

    def test_a_constant_column_is_refused_whatever_its_sd_rounds_to(self, blocks: pd.DataFrame) -> None:
        """0.1 twelve times has a population sd of 1.4e-17, not 0.

        Divided by that, the column was one of -1s, uncentred, and the group took it in unrefused.
        """
        assert np.full(12, 0.1).std() > 0
        message = "Group 'lab' has a constant column; drop it before fitting."
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            MFA({**GROUPS, "lab": ["lab1", "lab2", "setpoint"]}).fit(blocks.assign(setpoint=0.1))

    def test_an_unscaled_group_may_hold_a_constant_column(self, blocks: pd.DataFrame) -> None:
        """Centred, it is a column of zeros, which changes nothing; a group of nothing else has no variation."""
        with_setpoint = blocks.assign(setpoint=0.1)
        mfa = MFA({**GROUPS, "lab": ["lab1", "lab2", "setpoint"]}, scale=False).fit(with_setpoint)
        reference = MFA(GROUPS, scale=False).fit(blocks)
        np.testing.assert_allclose(mfa.eigenvalues_, reference.eigenvalues_, rtol=1e-12)
        message = "Group 'fixed' has no variation, so there is nothing to weight it by."
        for scale, expected in (
            (False, message),
            (True, "Group 'fixed' has a constant column; drop it before fitting."),
        ):
            with pytest.raises(ValueError, match=f"^{re.escape(expected)}$"):
                MFA({**GROUPS, "fixed": ["setpoint"]}, scale=scale).fit(with_setpoint)

    def test_the_number_of_axes(self, blocks: pd.DataFrame) -> None:
        assert MFA(GROUPS, n_components=1).fit(blocks).row_coordinates_.shape == (12, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert MFA(GROUPS, n_components=6).fit(blocks).n_components_ == 6
        message = "Asked for 10 axes, but this table has 6 with any inertia; keeping 6."
        with pytest.warns(SpecificationWarning, match=f"^{re.escape(message)}$"):
            MFA(GROUPS, n_components=10).fit(blocks)

    def test_the_refusals_in_full(self, blocks: pd.DataFrame) -> None:
        as_blocks = {name: blocks[columns] for name, columns in GROUPS.items()}
        cases = [
            (MFA(GROUPS), as_blocks, "Pass either a dict of blocks or groups=, not both."),
            (MFA(), blocks, "Say which columns form each group with groups=, or pass a dict of blocks."),
            (MFA(), {"spectra": as_blocks["spectra"]}, "MFA needs at least two groups; with one, it is just a PCA."),
            (MFA({"spectra": GROUPS["spectra"], "lab": []}), blocks, "Group 'lab' has no columns."),
            (
                MFA({"spectra": GROUPS["spectra"], "lab": ["lab1", "viscosity"]}),
                blocks,
                "Group 'lab' names column 'viscosity', which is not in the data.",
            ),
            (
                MFA({"spectra": GROUPS["spectra"], "lab": ["lab1", "spec1"]}),
                blocks,
                "Column 'spec1' is in both group 'spectra' and group 'lab'.",
            ),
            (
                MFA({**GROUPS, "context": ["rework"]}),
                blocks.assign(rework=[True, False] * 6),
                (
                    "Column 'rework' in group 'context' is not numeric. MFA here takes numeric groups; for a table "
                    "mixing numeric and categorical columns use FAMD."
                ),
            ),
            (
                MFA(),
                {"spectra": as_blocks["spectra"], "lab": as_blocks["lab"].iloc[::-1]},
                "Every block must have the same row index, in the same order.",
            ),
            (
                MFA(),
                {"a": blocks[["spec1", "spec2"]], "b": blocks[["spec2", "lab1"]]},
                "Column names must be unique across blocks; ['spec2'] repeat.",
            ),
            (MFA(GROUPS, n_components=0), blocks, "n_components must be at least 1; got 0."),
        ]
        gapped = blocks.copy()
        gapped.iloc[2, 4] = np.nan
        cases.append(
            (MFA(GROUPS), gapped, "MFA needs complete data; impute or drop the rows with missing values first.")
        )
        for model, data, message in cases:
            with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
                model.fit(data)
        with pytest.raises(TypeError, match=r"^groups must be a dict of group name to column names; got list\.$"):
            MFA([GROUPS["spectra"], GROUPS["lab"]]).fit(blocks)

        mfa = MFA(GROUPS).fit(blocks)
        with pytest.raises(ValueError, match=r"^New observations lack the columns \['lab2'\]\.$"):
            mfa.transform(blocks.drop(columns="lab2"))
        with pytest.raises(ValueError, match=r"^New observations must not have missing values\.$"):
            mfa.transform(gapped)
        with pytest.raises(NotFittedError):
            MFA(GROUPS).transform(blocks)

    def test_map_plot_at_its_defaults_and_with_settings(self, blocks: pd.DataFrame) -> None:
        mfa = MFA(GROUPS, n_components=3).fit(blocks)
        fig = mfa.map_plot()
        np.testing.assert_array_equal(fig.data[0].x, mfa.row_coordinates_[1].to_numpy())
        np.testing.assert_array_equal(fig.data[1].y, mfa.column_coordinates_[2].to_numpy())
        assert mfa.map_plot(3, 2, settings={"title": "Blocks"}).layout.title.text == "Blocks"
