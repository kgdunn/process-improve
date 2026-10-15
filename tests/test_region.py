"""Tests for the region diagnostics of :class:`DesignRegion` and for renaming factors in constraints."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from scipy.spatial import ConvexHull, QhullError

from process_improve.experiments import Constraint, DesignRegion, Factor, rename_factors
from process_improve.experiments import _region_geometry as geometry
from process_improve.experiments.designs import _as_constraints
from process_improve.experiments.factor import as_constraints

# The heat limit of tests/test_designs_constrained.py: it cuts the corner (150, 60) off the box.
HEAT_FACTORS = [Factor(name="T", low=100, high=150), Factor(name="D", low=20, high=60)]
HEAT = "3*T + 5*D <= 600"


# ---------------------------------------------------------------------------
# Constraints as text
# ---------------------------------------------------------------------------


class TestConstraintsAsText:
    def test_a_bare_string_a_dict_and_a_constraint_give_the_same_region(self):
        points = np.array([[1.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [0.2, 0.9]])
        expected = DesignRegion(HEAT_FACTORS, [Constraint(expression=HEAT)]).feasible(points)
        for given in (HEAT, {"expression": HEAT}, {"expression": HEAT, "type": "linear"}):
            region = DesignRegion(HEAT_FACTORS, [given])
            assert region.constraints == [Constraint(expression=HEAT)]
            np.testing.assert_array_equal(region.feasible(points), expected)

    def test_forms_mix_freely_and_keep_their_order(self):
        region = DesignRegion(HEAT_FACTORS, ["T <= 140", Constraint(expression="D >= 25"), {"expression": HEAT}])
        assert [c.expression for c in region.constraints] == ["T <= 140", "D >= 25", HEAT]

    def test_a_tuple_of_constraints_is_accepted(self):
        assert len(DesignRegion(HEAT_FACTORS, (HEAT,)).constraints) == 1

    def test_anything_else_is_refused(self):
        with pytest.raises(TypeError, match="Constraint, a dict or an expression string"):
            DesignRegion(HEAT_FACTORS, [42])  # type: ignore[list-item]

    def test_the_designs_module_keeps_the_old_name(self):
        assert _as_constraints is as_constraints
        assert as_constraints(None) is None

    def test_a_text_region_round_trips_through_its_dict(self):
        region = DesignRegion(HEAT_FACTORS, [HEAT])
        again = DesignRegion.from_dict(region.to_dict())
        assert again.constraints == region.constraints


# ---------------------------------------------------------------------------
# Renaming factors inside a constraint
# ---------------------------------------------------------------------------


class TestRenameFactors:
    def test_a_rename_gives_the_canonical_text(self):
        assert rename_factors(HEAT, {"T": "Temp"}) == "3 * Temp + 5 * D <= 600"

    def test_a_swap_is_done_all_at_once(self):
        assert rename_factors(HEAT, {"T": "D", "D": "T"}) == "3 * D + 5 * T <= 600"

    def test_a_cycle_of_three_is_done_all_at_once(self):
        assert rename_factors("A + 2*B + 3*C <= 6", {"A": "B", "B": "C", "C": "A"}) == "B + 2 * C + 3 * A <= 6"

    def test_function_names_are_untouched(self):
        assert rename_factors("sqrt(log) + log(log) <= 2", {"log": "L", "sqrt": "S"}) == "sqrt(L) + log(L) <= 2"

    def test_a_factor_may_be_renamed_to_a_function_name(self):
        text = rename_factors("sqrt(A) <= 2", {"A": "sqrt"})
        assert text == "sqrt(sqrt) <= 2"
        region = DesignRegion([Factor(name="sqrt", low=1, high=9)], [text])
        np.testing.assert_array_equal(region.feasible(np.array([[-1.0], [1.0]])), [True, False])

    def test_the_inverse_rename_gives_back_the_canonical_text(self):
        for expression in (HEAT, "0.5 <= A/B <= 2", "abs(A - 1e-5) + exp(-B) < 3", "-A**2 >= -4"):
            forward = {"A": "Alpha", "B": "Beta", "T": "Temp", "D": "Dose"}
            inverse = {new: old for old, new in forward.items()}
            assert rename_factors(rename_factors(expression, forward), inverse) == rename_factors(expression, {})

    def test_an_empty_rename_gives_the_canonical_text(self):
        assert rename_factors("3*T+5*D<=600", {}) == "3 * T + 5 * D <= 600"
        assert rename_factors("0.5<=A/B<=2", {}) == "0.5 <= A / B <= 2"

    def test_names_the_expression_does_not_use_are_ignored(self):
        assert rename_factors("T <= 140", {"Z": "Zeta"}) == "T <= 140"

    def test_the_constraint_method_keeps_the_type(self):
        renamed = Constraint(expression="A*B <= 3", type="nonlinear").renamed({"A": "B", "B": "A"})
        assert renamed == Constraint(expression="B * A <= 3", type="nonlinear")

    @pytest.mark.parametrize(
        ("expression", "names", "match"),
        [
            ("T <= 3", {"T": "1x"}, "not a valid identifier"),
            ("T <= 3", {"T": "for"}, "not a valid identifier"),
            ("T <= 3", {"T": "Temp (C)"}, "not a valid identifier"),
            ("T + D <= 3", {"T": "D"}, "merge the factors"),
            ("T == 3", {}, "only <, <=, >, >="),
            ("T = 3", {}, "not a valid expression"),
            ("T + 3", {}, "must be an inequality"),
            ("T(3) <= 1", {}, "Unsupported syntax"),
            ("__import__('os') <= 1", {}, "Unsupported syntax"),
            ("T" * 501 + " <= 1", {}, "longer than 500"),
        ],
    )
    def test_bad_input_is_refused(self, expression, names, match):
        with pytest.raises(ValueError, match=match):
            rename_factors(expression, names)

    def test_a_rename_that_makes_the_text_too_long_is_refused(self):
        expression = " + ".join(["A"] * 120) + " <= 1"
        with pytest.raises(ValueError, match="longer than 500"):
            rename_factors(expression, {"A": "Alpha"})


# ---------------------------------------------------------------------------
# Geometry of the cut box
# ---------------------------------------------------------------------------


def _brute_vertices(lows, highs, coefficients, rhs) -> np.ndarray:
    """Every vertex of ``{low <= X <= high, coefficients @ X <= rhs}``: solve each choice of k boundaries."""
    k = len(lows)
    a = np.vstack([np.eye(k), -np.eye(k), np.reshape(coefficients, (-1, k))]).astype(float)
    b = np.concatenate([highs, -np.asarray(lows, dtype=float), np.ravel(rhs)]).astype(float)
    found = []
    for rows in itertools.combinations(range(len(a)), k):
        system = a[list(rows)]
        if abs(np.linalg.det(system)) < 1e-9:
            continue
        x = np.linalg.solve(system, b[list(rows)])
        if np.all(a @ x <= b + 1e-7 * (1 + np.abs(b))):
            found.append(x)
    return _distinct(np.reshape(found, (-1, k)))


def _distinct(points: np.ndarray, atol: float = 1e-7) -> np.ndarray:
    """Drop every point within ``atol`` (in each coordinate) of one kept before it."""
    kept: list[np.ndarray] = []
    for p in points:
        if not any(np.max(np.abs(p - q)) <= atol for q in kept):
            kept.append(p)
    return np.reshape(kept, (-1, points.shape[1]))


def _same_rows(got: np.ndarray, expected: np.ndarray, atol: float = 1e-7) -> bool:
    """Whether the two sets of points match one to one, within ``atol``, in any order."""
    got = np.asarray(got, dtype=float).reshape(-1, expected.shape[1])
    if len(got) != len(expected):
        return False
    if not len(got):
        return True
    gap = np.abs(got[:, None, :] - expected[None, :, :]).max(axis=2)
    return bool((gap.min(axis=1) <= atol).all() and (gap.min(axis=0) <= atol).all())


class TestGeometry:
    def test_round_significant(self):
        got = geometry.round_significant(np.array([250.00000000000009, 1.65e-14, -0.30000000000000004, 0.0, 1e22 / 3]))
        assert got.tolist() == [250.0, 1.65e-14, -0.3, 0.0, 3.33333333333e21]

    def test_box_vertices_match_every_choice_of_k_boundaries(self):
        rng = np.random.default_rng(3)
        for k, m in [(1, 1), (2, 1), (3, 3), (4, 4), (5, 3)]:
            for _ in range(10):
                a = rng.normal(size=(m, k))
                a /= np.linalg.norm(a, axis=1, keepdims=True)
                b = rng.uniform(-0.5, 1.2, m)
                expected = _brute_vertices(-np.ones(k), np.ones(k), a, b)
                assert _same_rows(geometry.box_vertices(a, b), expected)

    def test_box_vertices_of_a_zero_row(self):
        assert geometry.box_vertices(np.zeros((1, 2)), np.array([-1.0])).shape == (0, 2)
        assert len(geometry.box_vertices(np.zeros((0, 3)), np.zeros(0))) == 8

    def test_box_vertices_refuse_too_many(self, monkeypatch):
        monkeypatch.setattr(geometry, "MAX_VERTICES", 4)
        with pytest.raises(ValueError, match="more than 4 vertices"):  # a square with one corner cut has 5
            geometry.box_vertices(np.array([[1.0, 1.0]]) / math.sqrt(2), np.array([1.0]))
        with pytest.raises(ValueError, match="8 corners"):
            geometry.box_vertices(np.zeros((0, 3)), np.zeros(0))

    def test_convex_polygon(self):
        square = np.array([[1.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [0.0, 0.0], [0.5, -1.0]])
        assert geometry.convex_polygon(square).tolist() == [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]
        assert geometry.convex_polygon(np.empty((0, 2))).shape == (0, 2)
        assert geometry.convex_polygon(np.array([[0.5, 0.5], [0.5, 0.5]])).tolist() == [[0.5, 0.5]]
        line = np.array([[1.0, 1.0], [0.0, 0.0], [0.5, 0.5], [-1.0, -1.0]])
        assert geometry.convex_polygon(line).tolist() == [[-1.0, -1.0], [1.0, 1.0]]

    def test_convex_polygon_falls_back_to_a_segment(self, monkeypatch):
        def flat(*_args, **_kwargs):
            raise QhullError("QH6154 initial simplex is flat")

        monkeypatch.setattr(geometry, "ConvexHull", flat)
        points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        assert len(geometry.convex_polygon(points)) == 2

    def test_clip_square(self):
        half = geometry.clip_square([(np.array([1.0, 0.0]), 0.0)])
        assert geometry.convex_polygon(half).tolist() == [[-1.0, -1.0], [0.0, -1.0], [0.0, 1.0], [-1.0, 1.0]]
        assert geometry.clip_square([(np.array([1.0, 0.0]), -2.0)]).shape == (0, 2)
        assert geometry.clip_square([]).shape == (4, 2)

    def test_points_inside_lie_in_the_hull_and_are_fixed(self):
        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
        points = geometry.points_inside(vertices, 500)
        assert points.shape == (500, 3)
        hull = ConvexHull(vertices)
        assert np.all(points @ hull.equations[:, :3].T + hull.equations[:, 3] <= 1e-12)
        np.testing.assert_array_equal(points, geometry.points_inside(vertices, 500))
        np.testing.assert_array_equal(geometry.points_inside(vertices[:1], 3), np.zeros((3, 3)))
