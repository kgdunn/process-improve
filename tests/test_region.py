"""Tests for the region diagnostics of :class:`DesignRegion` and for renaming factors in constraints."""

from __future__ import annotations

import doctest
import importlib
import itertools
import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from scipy.spatial import ConvexHull, QhullError

from process_improve.experiments import Constraint, DesignRegion, Factor, rename_factors
from process_improve.experiments import _region_geometry as geometry
from process_improve.experiments.designs import _as_constraints
from process_improve.experiments.factor import as_constraints

# The heat limit of tests/test_designs_constrained.py: it cuts the corner (150, 60) off the box.
HEAT_FACTORS = [Factor(name="T", low=100, high=150), Factor(name="D", low=20, high=60)]
HEAT = "3*T + 5*D <= 600"
#: The three factors of the corner examples in the separating-limit spec.
TPF = [
    Factor(name="T", low=150, high=200, units="degC"),
    Factor(name="P", low=1, high=3, units="bar"),
    Factor(name="F", low=10, high=20, units="L/min"),
]
UNIT_SQUARE = [Factor(name="A", low=-1, high=1), Factor(name="B", low=-1, high=1)]
MIXTURE = [Factor(name=f"x{i}", type="mixture") for i in range(3)]

_settings = settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)


# ---------------------------------------------------------------------------
# Constraints as text
# ---------------------------------------------------------------------------


class TestConstraintsAsText:
    def test_a_bare_string_a_dict_and_a_constraint_give_the_same_region(self):
        points = np.array([[1.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [0.2, 0.9]])
        expected = DesignRegion(HEAT_FACTORS, [Constraint(expression=HEAT)]).feasible(points)
        for form in (HEAT, {"expression": HEAT}, {"expression": HEAT, "type": "linear"}):
            region = DesignRegion(HEAT_FACTORS, [form])
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

    @pytest.mark.parametrize("alone", ["T <= 140", {"expression": "T <= 140"}, Constraint(expression="T <= 140")])
    def test_a_single_limit_outside_a_list_is_refused_with_the_fix(self, alone):
        """Iterated, a string gives its characters and a dict its keys: the error would blame a limit named 'T'."""
        with pytest.raises(TypeError, match=r"wrap a single limit in a list: \["):
            DesignRegion(HEAT_FACTORS, alone)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="wrap a single limit in a list"):
            as_constraints(alone)  # type: ignore[arg-type]

    def test_the_designs_module_keeps_the_old_name(self):
        assert _as_constraints is as_constraints
        assert as_constraints(None) is None

    def test_a_text_region_round_trips_through_its_dict(self):
        region = DesignRegion(HEAT_FACTORS, [HEAT])
        again = DesignRegion.from_dict(region.to_dict())
        assert again.constraints == region.constraints

    def test_a_dict_takes_every_form_the_constructor_takes(self):
        factors = [f.model_dump(mode="json") for f in HEAT_FACTORS]
        written = DesignRegion.from_dict({"factors": factors, "constraints": [HEAT, {"expression": "T <= 140"}]})
        assert [c.expression for c in written.constraints] == [HEAT, "T <= 140"]
        for missing in ({"factors": factors}, {"factors": factors, "constraints": None}):
            assert DesignRegion.from_dict(missing).constraints == []
        with pytest.raises(TypeError, match="Constraint, a dict or an expression string"):
            DesignRegion.from_dict({"factors": factors, "constraints": [3]})


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
# Helpers: independent answers to compare the region's against
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


def _inside_polygon(points: np.ndarray, polygon: np.ndarray, tol: float) -> np.ndarray:
    """Whether each point lies in a convex counter-clockwise polygon (or on a segment, or at a point)."""
    if len(polygon) == 1:
        return np.linalg.norm(points - polygon[0], axis=1) <= tol
    if len(polygon) == 2:
        p, q = polygon
        t = np.clip((points - p) @ (q - p) / ((q - p) @ (q - p)), 0.0, 1.0)
        return np.linalg.norm(points - (p + t[:, None] * (q - p)), axis=1) <= tol
    edges = np.roll(polygon, -1, axis=0) - polygon
    rel = points[:, None, :] - polygon[None, :, :]
    cross = edges[None, :, 0] * rel[:, :, 1] - edges[None, :, 1] * rel[:, :, 0]
    lengths = np.linalg.norm(edges, axis=1)
    return np.all(cross >= -tol * lengths, axis=1)


def _is_hull_of(polygon: np.ndarray, points: np.ndarray, tol: float = 1e-6) -> bool:
    """Whether ``polygon`` is the convex hull of ``points``, with no corner that lies on an edge.

    Its corners must all be among the points, every point must lie inside it, and it must turn
    at every corner: together these pin the hull down without computing one.
    """
    if not len(points):
        return not len(polygon)
    corners_are_points = all(np.min(np.max(np.abs(points - c), axis=1)) <= tol for c in polygon)
    holds_points = _inside_polygon(points, polygon, tol).all()
    if len(polygon) < 3:
        return bool(corners_are_points and holds_points)
    before, after = polygon - np.roll(polygon, 1, axis=0), np.roll(polygon, -1, axis=0) - polygon
    turns = before[:, 0] * after[:, 1] - before[:, 1] * after[:, 0]
    return bool(corners_are_points and holds_points and np.all(turns > 0))


@st.composite
def linear_regions(draw, max_factors: int = 4, max_limits: int = 4):
    """Draw a box of 2 to ``max_factors`` factors with integer ranges, cut by integer linear limits.

    Integer coefficients and right-hand sides put many vertices where more than k boundaries
    meet, the cases a vertex enumeration most often gets wrong.
    """
    k = draw(st.integers(2, max_factors))
    lows = draw(st.lists(st.integers(-5, 5), min_size=k, max_size=k))
    spans = draw(st.lists(st.integers(1, 6), min_size=k, max_size=k))
    highs = [lo + s for lo, s in zip(lows, spans, strict=True)]
    coefficients, rhs = [], []
    for _ in range(draw(st.integers(1, max_limits))):
        c = draw(st.lists(st.integers(-3, 3), min_size=k, max_size=k).filter(any))
        smallest = sum(ci * (lo if ci > 0 else hi) for ci, lo, hi in zip(c, lows, highs, strict=True))
        largest = sum(ci * (hi if ci > 0 else lo) for ci, lo, hi in zip(c, lows, highs, strict=True))
        coefficients.append(c)
        rhs.append(draw(st.integers(smallest - 1, largest + 1)))
    names = [f"X{j}" for j in range(k)]
    factors = [Factor(name=n, low=lo, high=hi) for n, lo, hi in zip(names, lows, highs, strict=True)]
    expressions = [
        " + ".join(f"{ci}*{n}" for ci, n in zip(c, names, strict=True)) + f" <= {r}"
        for c, r in zip(coefficients, rhs, strict=True)
    ]
    return (
        factors,
        expressions,
        np.array(lows, float),
        np.array(highs, float),
        np.array(coefficients, float),
        np.array(rhs, float),
    )


# ---------------------------------------------------------------------------
# Share kept, emptiness, implied bounds
# ---------------------------------------------------------------------------


class TestVolumeFraction:
    def test_the_heat_limit_keeps_five_eighths_of_the_box(self):
        assert DesignRegion(HEAT_FACTORS, [HEAT]).volume_fraction() == pytest.approx(0.625, abs=0.005)

    def test_a_circle_keeps_a_quarter_of_pi(self):
        kept = DesignRegion(UNIT_SQUARE, ["A**2 + B**2 <= 1"]).volume_fraction()
        assert kept == pytest.approx(math.pi / 4, rel=0.005)

    def test_it_is_deterministic(self):
        first = DesignRegion(HEAT_FACTORS, [HEAT]).volume_fraction()
        region = DesignRegion(HEAT_FACTORS, [HEAT])
        region.shadow("T", "D")
        assert region.volume_fraction() == first
        assert region.volume_fraction() == first  # from the cache

    def test_no_limits_keep_everything_and_conflicting_ones_nothing(self):
        assert DesignRegion(HEAT_FACTORS).volume_fraction() == 1.0
        assert DesignRegion(HEAT_FACTORS, ["T <= 110", "T >= 140"]).volume_fraction() == 0.0

    def test_another_sample_size(self):
        kept = DesignRegion(HEAT_FACTORS, [HEAT]).volume_fraction(n_samples=1000)
        assert kept == pytest.approx(0.625, abs=0.01)

    def test_a_sample_size_below_one_is_refused(self):
        with pytest.raises(ValueError, match="at least 1"):
            DesignRegion(HEAT_FACTORS, [HEAT]).volume_fraction(n_samples=0)

    def test_categorical_factors_are_left_out(self):
        factors = [*HEAT_FACTORS, Factor(name="Cat", type="categorical", levels=["a", "b"])]
        assert DesignRegion(factors, [HEAT]).volume_fraction() == DesignRegion(HEAT_FACTORS, [HEAT]).volume_fraction()
        assert DesignRegion([factors[-1]]).volume_fraction() == 1.0

    def test_a_region_without_numeric_factors_holds_or_fails_as_a_whole(self):
        cat = [Factor(name="Cat", type="categorical", levels=["a", "b"])]
        assert DesignRegion(cat, ["1 <= 2"]).volume_fraction() == 1.0
        assert DesignRegion(cat, ["2 <= 1"]).volume_fraction() == 0.0
        assert DesignRegion(cat, ["2 <= 1"]).is_empty()
        assert DesignRegion(cat, ["2 <= 1"]).implied_bounds() is None
        assert DesignRegion(cat, ["1 <= 2"]).implied_bounds() == {}

    @_settings
    @given(linear_regions())
    def test_it_matches_the_hull_volume_of_the_vertices(self, case):
        factors, expressions, lows, highs, coefficients, rhs = case
        vertices = _brute_vertices(lows, highs, coefficients, rhs)
        try:
            exact = ConvexHull(vertices).volume / np.prod(highs - lows) if len(vertices) > len(lows) else 0.0
        except QhullError:  # a flat region: no volume
            exact = 0.0
        assert DesignRegion(factors, expressions).volume_fraction() == pytest.approx(exact, abs=0.005)


class TestIsEmpty:
    def test_conflicting_linear_limits_are_empty(self):
        assert DesignRegion(HEAT_FACTORS, ["T <= 110", "T >= 140"]).is_empty()
        assert not DesignRegion(HEAT_FACTORS, [HEAT]).is_empty()
        assert not DesignRegion(HEAT_FACTORS).is_empty()

    def test_conflicting_curved_limits_are_empty(self):
        assert DesignRegion(UNIT_SQUARE, ["A**2 + B**2 <= 0.01", "A >= 0.5"]).is_empty()
        assert DesignRegion(UNIT_SQUARE, ["A**2 + B**2 >= 3"]).is_empty()
        assert not DesignRegion(UNIT_SQUARE, ["A**2 + B**2 <= 0.01"]).is_empty()

    def test_a_thin_region_left_by_linear_limits_is_found_for_a_curved_one(self):
        """The box sample misses a sliver 0.002 wide; the points spread inside the linear polytope do not."""
        region = DesignRegion(UNIT_SQUARE, ["A + B <= 0.001", "A + B >= -0.001", "A*B <= 0"])
        assert not region.is_empty()

    def test_a_flat_region_is_not_empty(self):
        assert not DesignRegion(HEAT_FACTORS, ["T <= 100"]).is_empty()

    def test_a_limit_comparing_numbers_only(self):
        assert DesignRegion(HEAT_FACTORS, ["2 <= 1"]).is_empty()
        assert not DesignRegion(HEAT_FACTORS, ["1 <= 2"]).is_empty()


class TestImpliedBounds:
    def test_a_limit_tightens_both_factors(self):
        bounds = DesignRegion(TPF[:2], ["T + 25*P <= 215"]).implied_bounds()
        assert bounds == {"T": (150.0, 190.0), "P": (1.0, 2.6)}

    def test_the_heat_limit_tightens_nothing(self):
        assert DesignRegion(HEAT_FACTORS, [HEAT]).implied_bounds() == {"T": (100.0, 150.0), "D": (20.0, 60.0)}

    def test_conflicting_limits_have_no_bounds(self):
        assert DesignRegion(HEAT_FACTORS, ["T <= 110", "T >= 140"]).implied_bounds() is None

    def test_curved_limits_are_left_out(self):
        bounds = DesignRegion(UNIT_SQUARE, ["A <= 0.5", "A**2 + B**2 <= 0.25"]).implied_bounds()
        assert bounds == {"A": (-1.0, 0.5), "B": (-1.0, 1.0)}

    def test_linear_programs_agree_with_the_vertices(self, monkeypatch):
        expected = DesignRegion(TPF, ["T + 25*P <= 215", "F >= T/10"]).implied_bounds()
        monkeypatch.setattr(geometry, "MAX_VERTICES", 2)  # too few for any box: the linear programs take over
        region = DesignRegion(TPF, ["T + 25*P <= 215", "F >= T/10"])
        assert region.implied_bounds() == pytest.approx(expected)
        assert DesignRegion(TPF, ["T <= 160", "T >= 170"]).implied_bounds() is None
        assert DesignRegion(TPF, ["T <= 160", "T >= 170"]).is_empty()
        assert not region.is_empty()

    @_settings
    @given(linear_regions())
    def test_they_equal_the_extremes_of_the_vertices(self, case):
        factors, expressions, lows, highs, coefficients, rhs = case
        vertices = _brute_vertices(lows, highs, coefficients, rhs)
        bounds = DesignRegion(factors, expressions).implied_bounds()
        if not len(vertices):
            assert bounds is None
            return
        assert bounds is not None
        got = np.array(list(bounds.values()))
        np.testing.assert_allclose(got[:, 0], vertices.min(axis=0), atol=1e-7)
        np.testing.assert_allclose(got[:, 1], vertices.max(axis=0), atol=1e-7)


class TestCentroid:
    def test_the_heat_region(self):
        """The region is a 50 x 10 rectangle (centroid (125, 25)) plus a triangle of area 750 (centroid (116.7, 40))."""
        middle = DesignRegion(HEAT_FACTORS, [HEAT]).centroid()
        assert middle == pytest.approx({"T": 120.0, "D": 34.0}, abs=0.05)

    def test_without_limits_it_is_the_centre_of_the_box(self):
        assert DesignRegion(HEAT_FACTORS).centroid() == {"T": 125.0, "D": 40.0}

    def test_a_ring_gives_an_allowed_point_near_its_centre(self):
        region = DesignRegion(UNIT_SQUARE, ["A**2 + B**2 >= 0.5"])
        middle = region.centroid()
        assert middle is not None
        point = np.array([[middle["A"], middle["B"]]])
        assert region.feasible(point)[0]
        assert np.hypot(*point[0]) == pytest.approx(math.sqrt(0.5), abs=0.01)

    def test_a_sliver_too_thin_for_the_box_sample(self):
        region = DesignRegion(UNIT_SQUARE, ["A + B <= 0.001", "A + B >= -0.001", "A*B <= 0"])
        middle = region.centroid()
        assert middle is not None
        assert region.feasible(np.array([[middle["A"], middle["B"]]]))[0]

    def test_an_empty_region_has_none(self):
        assert DesignRegion(HEAT_FACTORS, ["T <= 110", "T >= 140"]).centroid() is None


# ---------------------------------------------------------------------------
# Redundant limits
# ---------------------------------------------------------------------------


class TestRedundantConstraints:
    def test_a_duplicate_limit_is_redundant(self):
        assert DesignRegion(HEAT_FACTORS, [HEAT, HEAT]).redundant_constraints() == [0, 1]

    def test_a_looser_limit_is_redundant(self):
        assert DesignRegion(HEAT_FACTORS, ["T <= 140", "T <= 130", HEAT]).redundant_constraints() == [0]

    def test_a_limit_outside_the_box_is_redundant(self):
        assert DesignRegion(HEAT_FACTORS, ["T <= 160", HEAT]).redundant_constraints() == [0]

    def test_no_limits_and_needed_limits(self):
        assert DesignRegion(HEAT_FACTORS).redundant_constraints() == []
        assert DesignRegion(HEAT_FACTORS, [HEAT, "T >= 110"]).redundant_constraints() == []

    def test_a_chain_counts_as_one_limit(self):
        """A chain is redundant only when neither of its comparisons removes anything."""
        assert DesignRegion(HEAT_FACTORS, ["110 <= T <= 140", "T <= 130"]).redundant_constraints() == []
        assert DesignRegion(HEAT_FACTORS, ["100 <= T <= 140", "T <= 130"]).redundant_constraints() == [0]

    def test_every_limit_of_an_empty_region_but_the_one_emptying_it(self):
        region = DesignRegion(HEAT_FACTORS, ["T <= 140", "T + D >= 300"])
        assert region.redundant_constraints() == [0]

    def test_curved_limits(self):
        assert DesignRegion(UNIT_SQUARE, ["A**2 + B**2 <= 4"]).redundant_constraints() == [0]
        assert DesignRegion(UNIT_SQUARE, ["A**2 + B**2 <= 1"]).redundant_constraints() == []
        region = DesignRegion(UNIT_SQUARE, ["A**2 + B**2 <= 0.5", "A**2 + B**2 <= 0.6"])
        assert region.redundant_constraints() == [1]

    def test_a_linear_limit_made_redundant_by_a_curved_one(self):
        """The linear program cannot see the circle; the sampled check can."""
        region = DesignRegion(UNIT_SQUARE, ["A <= 0.9", "A**2 + B**2 <= 0.5"])
        assert region.redundant_constraints() == [0]
        assert DesignRegion(UNIT_SQUARE, ["A <= 0.5", "A**2 + B**2 <= 0.5"]).redundant_constraints() == []

    def test_a_limit_comparing_numbers_only(self):
        assert DesignRegion(HEAT_FACTORS, ["1 <= 2", HEAT]).redundant_constraints() == [0]
        assert DesignRegion(HEAT_FACTORS, ["2 <= 1", HEAT]).redundant_constraints() == [1]

    def test_a_region_without_numeric_factors(self):
        cat = [Factor(name="Cat", type="categorical", levels=["a", "b"])]
        assert DesignRegion(cat, ["2 <= 1", "1 <= 2"]).redundant_constraints() == [1]
        assert DesignRegion(cat, ["2 <= 1", "3 <= 1"]).redundant_constraints() == [0, 1]


# ---------------------------------------------------------------------------
# Affine forms of the limits
# ---------------------------------------------------------------------------


class TestLinearForms:
    @pytest.mark.parametrize(
        ("expression", "form"),
        [
            (HEAT, ({"T": 3.0, "D": 5.0}, None, 600.0)),
            ("600 >= 3*T + 5*D", ({"T": 3.0, "D": 5.0}, None, 600.0)),
            ("3*T + 5*D < 600", ({"T": 3.0, "D": 5.0}, None, 600.0)),
            ("400 <= 3*T + 5*D <= 600", ({"T": 3.0, "D": 5.0}, 400.0, 600.0)),
            ("600 >= 3*T + 5*D >= 400", ({"T": 3.0, "D": 5.0}, 400.0, 600.0)),
            ("3*T >= 600 - 5*D", ({"T": 3.0, "D": 5.0}, 600.0, None)),
            ("T <= 2*D", ({"T": 1.0, "D": -2.0}, None, 0.0)),
            ("T >= 2*D", ({"T": 1.0, "D": -2.0}, 0.0, None)),
            ("T - T + D <= 50", ({"D": 1.0}, None, 50.0)),
            ("0.1*T + 0.2*D <= 0.3*100", ({"T": 0.1, "D": 0.2}, None, 30.0)),
            ("(T + D)/3 <= 250/3", ({"T": 0.333333333333, "D": 0.333333333333}, None, 83.3333333333)),
            ("T <= 140 <= 200", ({"T": 1.0}, None, 140.0)),
            ("T <= 140 >= T", ({"T": 1.0}, None, 140.0)),
            ("abs(T) <= 140", ({"T": 1.0}, None, 140.0)),  # affine over the box, where T > 0
        ],
    )
    def test_affine_limits(self, expression, form):
        assert DesignRegion(HEAT_FACTORS, [expression]).linear_forms() == [form]

    @pytest.mark.parametrize(
        "expression", ["T*D <= 6000", "T <= D <= 2", "T <= 140 <= 100", "1 <= 2", "2 <= 1", "abs(T - 125) <= 10"]
    )
    def test_limits_without_one_affine_form(self, expression):
        assert DesignRegion(HEAT_FACTORS, [expression]).linear_forms() == [None]

    def test_one_entry_per_limit_in_order(self):
        forms = DesignRegion(HEAT_FACTORS, [HEAT, "T*D <= 6000", "D >= 25"]).linear_forms()
        assert forms == [({"T": 3.0, "D": 5.0}, None, 600.0), None, ({"D": 1.0}, 25.0, None)]

    def test_numbers_carry_no_floating_point_residue(self):
        factors = [Factor(name="T", low=0.1, high=0.7), Factor(name="D", low=1e-3, high=3e-3)]
        (coefficients, low, high), *_ = DesignRegion(factors, ["0.1*T + 0.7*D - 0.3 <= 0.2 - 0.1"]).linear_forms()
        assert coefficients == {"T": 0.1, "D": 0.7}
        assert low is None
        assert high == 0.4

    def test_mixture_limits_have_forms_too(self):
        assert DesignRegion(MIXTURE, ["x0 + 2*x1 <= 0.8"]).linear_forms() == [({"x0": 1.0, "x1": 2.0}, None, 0.8)]


# ---------------------------------------------------------------------------
# Vertices
# ---------------------------------------------------------------------------


class TestVertices:
    def test_the_heat_limit_cuts_one_corner(self):
        vertices = DesignRegion(HEAT_FACTORS, [HEAT]).vertices()
        assert vertices.tolist() == [[100.0, 20.0], [100.0, 60.0], [150.0, 20.0], [150.0, 30.0]]

    def test_no_limits_give_the_corners(self):
        assert len(DesignRegion(TPF).vertices()) == 8

    def test_an_empty_region_has_none(self):
        assert DesignRegion(HEAT_FACTORS, ["T <= 110", "T >= 140"]).vertices().shape == (0, 2)

    def test_curved_limits_have_none(self):
        with pytest.raises(ValueError, match="curved constraints"):
            DesignRegion(UNIT_SQUARE, ["A**2 + B**2 <= 1", "A <= 0.5"]).vertices()

    def test_too_many_vertices_are_not_listed(self):
        factors = [Factor(name=f"X{j}", low=0, high=1) for j in range(17)]
        region = DesignRegion(factors, ["X0 + X1 <= 1.5"])
        with pytest.raises(ValueError, match="131072 corners"):
            region.vertices()
        assert region.implied_bounds()["X0"] == (0.0, 1.0)  # type: ignore[index]
        assert not region.is_empty()

    def test_eight_factors_cut_by_one_limit(self):
        factors = [Factor(name=f"X{j}", low=0, high=1 + j) for j in range(8)]
        expression = " + ".join(f"X{j}" for j in range(8)) + " <= 12"
        lows, highs = np.zeros(8), 1.0 + np.arange(8)
        expected = _brute_vertices(lows, highs, np.ones((1, 8)), [12.0])
        assert _same_rows(DesignRegion(factors, [expression]).vertices(), expected)

    def test_cuts_through_corners_and_coinciding_limits(self):
        """Degenerate vertices, where more boundaries meet than there are factors."""
        limits = ["T + P <= 152", "T - 25*P >= 125", "T + P <= 152", "F <= 0.4*T - 50"]
        region = DesignRegion(TPF, limits)
        coefficients = [[1, 1, 0], [-1, 25, 0], [1, 1, 0], [-0.4, 0, 1]]
        expected = _brute_vertices([150, 1, 10], [200, 3, 20], coefficients, [152, -125, 152, -50])
        assert _same_rows(region.vertices(), expected)

    @_settings
    @given(linear_regions())
    def test_they_match_every_choice_of_k_boundaries(self, case):
        factors, expressions, lows, highs, coefficients, rhs = case
        expected = _brute_vertices(lows, highs, coefficients, rhs)
        assert _same_rows(DesignRegion(factors, expressions).vertices(), expected)


# ---------------------------------------------------------------------------
# Pictures: shadows, slices, lines and slack
# ---------------------------------------------------------------------------


class TestShadow:
    def test_the_heat_limit(self):
        shadow = DesignRegion(HEAT_FACTORS, [HEAT]).shadow("T", "D")
        assert shadow == {"kind": "polygon", "points": [[100.0, 20.0], [150.0, 20.0], [150.0, 30.0], [100.0, 60.0]]}

    def test_the_axes_can_be_swapped(self):
        shadow = DesignRegion(HEAT_FACTORS, [HEAT]).shadow("D", "T")
        assert shadow["points"] == [[20.0, 100.0], [60.0, 100.0], [30.0, 150.0], [20.0, 150.0]]

    def test_a_shadow_is_wider_than_a_slice(self):
        """Some F lets every (T, P) run, but with F at 20 the corner (200, 3) cannot."""
        region = DesignRegion(TPF, ["T + 25*P + 5*F <= 350"])
        assert region.shadow("T", "P")["points"] == [[150.0, 1.0], [200.0, 1.0], [200.0, 3.0], [150.0, 3.0]]
        assert region.slice_polygon("T", "P", {"F": 20})["points"] == [
            [150.0, 1.0],
            [200.0, 1.0],
            [200.0, 2.0],
            [175.0, 3.0],
            [150.0, 3.0],
        ]

    def test_a_segment(self):
        assert DesignRegion(HEAT_FACTORS, ["T <= 100"]).shadow("T", "D")["points"] == [[100.0, 20.0], [100.0, 60.0]]
        diagonal = DesignRegion(HEAT_FACTORS, ["T + D <= 130", "T + D >= 130"]).shadow("T", "D")
        assert diagonal["points"] == [[100.0, 30.0], [110.0, 20.0]]

    def test_a_point(self):
        assert DesignRegion(HEAT_FACTORS, ["T <= 100", "D >= 60"]).shadow("T", "D")["points"] == [[100.0, 60.0]]

    def test_an_empty_region(self):
        assert DesignRegion(HEAT_FACTORS, ["T <= 110", "T >= 140"]).shadow("T", "D") == {
            "kind": "polygon",
            "points": [],
        }

    def test_a_curved_region_gives_a_mask(self):
        shadow = DesignRegion(UNIT_SQUARE, ["A**2 + B**2 <= 1"]).shadow("A", "B")
        assert shadow["kind"] == "mask"
        z = np.array(shadow["z"])
        assert z.shape == (40, 40)
        assert shadow["x"][0] == -0.975
        assert shadow["x"][-1] == 0.975
        centres = np.array(np.meshgrid(shadow["x"], shadow["y"]))
        radius = np.hypot(*centres)
        assert z[radius < 0.95].all()  # every cell well inside the circle holds a sampled point
        assert not z[radius > 1.05].any()

    def test_a_mask_is_drawn_in_actual_units(self):
        factors = [Factor(name="A", low=0, high=4), Factor(name="B", low=10, high=12)]
        shadow = DesignRegion(factors, ["(A - 2)**2 + (B - 11)**2 <= 1"]).shadow("A", "B")
        assert shadow["x"][:2] == [0.05, 0.15]
        assert shadow["y"][:2] == [10.025, 10.075]

    def test_the_linear_programs_fall_back_to_a_mask(self, monkeypatch):
        monkeypatch.setattr(geometry, "MAX_VERTICES", 2)
        assert DesignRegion(HEAT_FACTORS, [HEAT]).shadow("T", "D")["kind"] == "mask"

    @pytest.mark.parametrize(("x", "y", "match"), [("T", "T", "two different"), ("T", "Z", "not a numeric factor")])
    def test_bad_factor_names(self, x, y, match):
        with pytest.raises(ValueError, match=match):
            DesignRegion(HEAT_FACTORS, [HEAT]).shadow(x, y)

    @_settings
    @given(linear_regions())
    def test_it_is_the_hull_of_the_projected_vertices_and_holds_every_allowed_point(self, case):
        factors, expressions, lows, highs, coefficients, rhs = case
        region = DesignRegion(factors, expressions)
        points = np.array(region.shadow("X0", "X1")["points"]).reshape(-1, 2)
        vertices = _brute_vertices(lows, highs, coefficients, rhs)
        if not len(vertices):
            assert not len(points)
            return
        assert _is_hull_of(points, vertices[:, :2])
        # Every allowed point of a sample of the box falls inside the shadow.
        sample = lows + (highs - lows) * np.random.default_rng(0).random((4000, len(lows)))
        allowed = sample[np.all(sample @ coefficients.T <= rhs + 1e-12, axis=1)]
        assert _inside_polygon(allowed[:, :2], points, tol=1e-6).all()


class TestSlicePolygon:
    def test_the_heat_limit_does_not_depend_on_a_third_factor(self):
        factors = [*HEAT_FACTORS, Factor(name="F", low=0, high=1)]
        for f in (0.0, 0.5, 1.0):
            sliced = DesignRegion(factors, [HEAT]).slice_polygon("T", "D", {"F": f})
            assert sliced["points"] == [[100.0, 20.0], [150.0, 20.0], [150.0, 30.0], [100.0, 60.0]]

    def test_the_held_values_move_the_limit(self):
        region = DesignRegion(TPF, ["T + 25*P + 5*F <= 350"])
        assert region.slice_polygon("T", "P", {"F": 10})["points"] == [
            [150.0, 1.0],
            [200.0, 1.0],
            [200.0, 3.0],
            [150.0, 3.0],
        ]
        assert region.slice_polygon("T", "P")["points"] == [  # F at its centre, 15
            [150.0, 1.0],
            [200.0, 1.0],
            [200.0, 3.0],
            [150.0, 3.0],
        ]
        assert len(region.slice_polygon("T", "P", {"F": 20})["points"]) == 5

    def test_entries_for_the_two_axes_are_ignored(self):
        region = DesignRegion(TPF, ["T + 25*P + 5*F <= 350"])
        assert region.slice_polygon("T", "P", {"T": 1e6, "P": -5, "F": 20}) == region.slice_polygon("T", "P", {"F": 20})

    def test_nothing_can_be_run_in_some_slices(self):
        region = DesignRegion(TPF, ["T + 25*P + 5*F <= 250"])
        assert region.slice_polygon("T", "P", {"F": 10})["points"] == [[150.0, 1.0], [175.0, 1.0], [150.0, 2.0]]
        assert region.slice_polygon("T", "P", {"F": 20})["points"] == []  # 150 + 25 + 100 > 250
        assert region.slice_polygon("T", "P", {"F": 5}) == {"kind": "polygon", "points": []}  # outside F's range
        assert DesignRegion(TPF, ["F <= 12"]).slice_polygon("T", "P", {"F": 15}) == {"kind": "polygon", "points": []}

    def test_a_limit_on_the_held_factors_alone_keeps_or_empties_the_whole_rectangle(self):
        rectangle = [[150.0, 1.0], [200.0, 1.0], [200.0, 3.0], [150.0, 3.0]]
        assert DesignRegion(TPF, ["F <= 12"]).slice_polygon("T", "P", {"F": 11})["points"] == rectangle

    def test_a_slice_can_be_a_segment_or_a_point(self):
        region = DesignRegion(TPF, ["T + 25*P + 5*F <= 225"])
        assert region.slice_polygon("T", "P", {"F": 10})["points"] == [[150.0, 1.0]]
        region = DesignRegion(TPF, ["T + 5*F <= 200"])
        assert region.slice_polygon("T", "P", {"F": 10})["points"] == [[150.0, 1.0], [150.0, 3.0]]

    def test_a_curved_region_gives_a_mask_that_agrees_with_feasible(self):
        factors = [*UNIT_SQUARE, Factor(name="C", low=0, high=1)]
        region = DesignRegion(factors, ["A**2 + B**2 <= C"])
        sliced = region.slice_polygon("A", "B", {"C": 0.5})
        assert sliced["kind"] == "mask"
        z = np.array(sliced["z"])
        assert z.shape == (61, 61)
        a, b = np.meshgrid(sliced["x"], sliced["y"])
        coded = np.column_stack([a.ravel(), b.ravel(), np.zeros(a.size)])
        np.testing.assert_array_equal(z.ravel(), region.feasible(coded).astype(int))
        assert sliced["x"][0] == -1.0
        assert sliced["x"][-1] == 1.0

    @pytest.mark.parametrize(
        ("at", "match"),
        [({"Z": 1.0}, "not a numeric factor"), ({"F": float("nan")}, "finite"), ({"F": float("inf")}, "finite")],
    )
    def test_bad_held_values(self, at, match):
        with pytest.raises(ValueError, match=match):
            DesignRegion(TPF, ["T <= 190"]).slice_polygon("T", "P", at)

    @_settings
    @given(linear_regions(max_factors=4), st.integers(0, 8))
    def test_its_corners_meet_every_limit_and_it_is_the_clipped_rectangle(self, case, eighths):
        """Held values on a grid of eighths: exact, so the boundary cases are decided the same way by both."""
        factors, expressions, lows, highs, coefficients, rhs = case
        share = eighths / 8
        region = DesignRegion(factors, expressions)
        held = lows[2:] + share * (highs[2:] - lows[2:])
        at = {f"X{j + 2}": float(v) for j, v in enumerate(held)}
        points = np.array(region.slice_polygon("X0", "X1", at)["points"]).reshape(-1, 2)
        # The slice is {low <= (X0, X1) <= high, c[:2] @ p <= rhs - c[2:] @ held}: try every pair of boundaries.
        expected = _brute_vertices(lows[:2], highs[:2], coefficients[:, :2], rhs - coefficients[:, 2:] @ held)
        assert _is_hull_of(points, expected)
        full = np.column_stack([points, np.tile(held, (len(points), 1))])
        assert np.all(full @ coefficients.T <= rhs + 1e-7)
        assert np.all((points >= lows[:2] - 1e-9) & (points <= highs[:2] + 1e-9))


class TestSliceLines:
    def test_each_linear_limit_gives_its_boundary_in_the_rectangle(self):
        factors = [*HEAT_FACTORS, Factor(name="F", low=0, high=10)]
        region = DesignRegion(factors, [HEAT, "400 <= 3*T + 5*D + F <= 620", "T*D <= 9000", "F <= 20"])
        assert region.slice_lines("T", "D") == [
            {"limit": 0, "points": [[100.0, 60.0], [150.0, 30.0]]},
            {"limit": 1, "points": [[105.0, 60.0], [150.0, 33.0]]},
        ]
        assert region.slice_lines("T", "D", {"F": 10})[1] == {
            "limit": 1,
            "points": [[103.333333333, 60.0], [150.0, 32.0]],
        }

    def test_a_chain_can_give_two_lines(self):
        region = DesignRegion(HEAT_FACTORS, ["560 <= 3*T + 5*D <= 600"])
        assert [line["limit"] for line in region.slice_lines("T", "D")] == [0, 0]

    def test_a_boundary_touching_only_a_corner_gives_no_line(self):
        assert DesignRegion(HEAT_FACTORS, ["T + D <= 120"]).slice_lines("T", "D") == []
        assert DesignRegion(HEAT_FACTORS, ["T + D <= 125"]).slice_lines("T", "D") != []


class TestSliceSlack:
    def test_it_is_zero_or_below_exactly_where_a_point_can_be_run(self):
        factors = [*HEAT_FACTORS, Factor(name="F", low=0, high=1)]
        for limits in ([HEAT, "D >= 25 + 10*F"], [HEAT, "T*D <= 5000 + 1000*F"]):
            region = DesignRegion(factors, limits)
            t, d = np.meshgrid(np.linspace(100, 150, 51), np.linspace(20, 60, 41))
            slack = region.slice_slack("T", "D", t, d, {"F": 0.5})
            assert slack.shape == (41, 51)
            coded = np.column_stack([(t.ravel() - 125) / 25, (d.ravel() - 40) / 20, np.zeros(t.size)])
            np.testing.assert_array_equal(slack.ravel() <= 0, region.feasible(coded))

    def test_a_linear_limit_gives_its_distance_in_coded_units(self):
        region = DesignRegion(HEAT_FACTORS, [HEAT])
        slack = region.slice_slack("T", "D", np.array([100.0, 150.0, 150.0, 125.0]), np.array([20.0, 60.0, 30.0, 40.0]))
        # In coded units the limit is 0.6 x + 0.8 y <= 0.2; (150, 30) lies on it and (100, 20) on two box faces.
        np.testing.assert_allclose(slack, [0.0, 1.2, 0.0, -0.2])

    def test_it_is_linear_along_a_line_for_linear_limits(self):
        region = DesignRegion(HEAT_FACTORS, [HEAT])
        t = np.linspace(140, 150, 11)
        slack = region.slice_slack("T", "D", t, np.full(11, 59.0))
        assert np.allclose(np.diff(slack, 2), 0.0, atol=1e-12)

    def test_points_outside_the_ranges_cannot_be_run(self):
        slack = DesignRegion(HEAT_FACTORS).slice_slack("T", "D", np.array([90.0, 125.0]), np.array([40.0, 70.0]))
        np.testing.assert_allclose(slack, [0.4, 0.5])

    def test_a_row_and_a_column_broadcast_to_a_grid(self):
        region = DesignRegion(HEAT_FACTORS, [HEAT])
        grid = region.slice_slack("T", "D", np.linspace(100, 150, 5)[None, :], np.linspace(20, 60, 3)[:, None])
        assert grid.shape == (3, 5)


# ---------------------------------------------------------------------------
# Mixtures and the geometry helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "args"),
    [
        ("volume_fraction", ()),
        ("centroid", ()),
        ("is_empty", ()),
        ("implied_bounds", ()),
        ("redundant_constraints", ()),
        ("vertices", ()),
        ("shadow", ("x0", "x1")),
        ("slice_polygon", ("x0", "x1")),
        ("slice_lines", ("x0", "x1")),
        ("slice_slack", ("x0", "x1", np.zeros(2), np.zeros(2))),
    ],
)
def test_box_diagnostics_refuse_a_mixture(method, args):
    with pytest.raises(ValueError, match="mixture proportions"):
        getattr(DesignRegion(MIXTURE, ["x0 <= 0.5"]), method)(*args)


@pytest.mark.parametrize("module", ["_region_geometry", "region", "compare", "separating", "designs_constrained"])
def test_the_examples_in_the_docstrings_print_what_they_show(module):
    """No CI step runs doctests, so an example whose output drifted (an array repr, say) would go unnoticed."""
    result = doctest.testmod(importlib.import_module(f"process_improve.experiments.{module}"), report=False)
    assert result.attempted > 0
    assert result.failed == 0


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

    def test_convex_polygon_drops_a_point_just_off_an_edge(self):
        """(1/3, -1/3) lies on the edge from (-1, -1) to (1, 0); rounding 1/3 to 12 decimals moves it off by 5e-13.

        The hull alone then lists it as a fourth corner.
        """
        third = round(1 / 3, 12)
        points = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 0.0], [third, -third]])
        assert len(ConvexHull(points).vertices) == 4
        assert geometry.convex_polygon(points).tolist() == [[-1.0, -1.0], [1.0, -1.0], [1.0, 0.0]]

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


# ---------------------------------------------------------------------------
# Eight factors and eight limits, as the browser app sends them on every edit
# ---------------------------------------------------------------------------

EIGHT = [Factor(name=n, low=10 * (i + 1), high=10 * (i + 1) + 20 + 5 * i) for i, n in enumerate("ABCDEFGH")]
EIGHT_LIMITS = [
    "A + 2*B <= 60",
    "3*C - D >= 20",
    "A + B + C + D + E + F + G + H <= 500",
    "E + 0.5*F <= 140",
    "G - H <= -5",
    "2*A + C + 3*H >= 200",
    "B + D + F + H <= 260",
    "10 <= A - B + G <= 80",
]


class TestEightFactors:
    def test_every_limit_is_recognised_as_linear(self):
        forms = DesignRegion(EIGHT, EIGHT_LIMITS).linear_forms()
        assert all(form is not None for form in forms)
        assert forms[7] == ({"A": 1.0, "B": -1.0, "G": 1.0}, 10.0, 80.0)

    def test_the_diagnostics_agree_with_each_other(self):
        region = DesignRegion(EIGHT, EIGHT_LIMITS)
        vertices = region.vertices()
        coded = (vertices - region._centre) / region._half
        assert region.feasible(coded, tol=1e-7).all()
        bounds = region.implied_bounds()
        assert bounds is not None
        np.testing.assert_allclose(np.array(list(bounds.values())), np.column_stack([vertices.min(0), vertices.max(0)]))
        assert region.redundant_constraints() == [5]
        assert not region.is_empty()
        assert 0 < region.volume_fraction() < 0.01
        # A shadow holds every vertex's projection.
        for x, y in [("A", "B"), ("C", "H"), ("E", "F")]:
            ix, iy = region.names.index(x), region.names.index(y)
            polygon = np.array(region.shadow(x, y)["points"])
            assert _inside_polygon(vertices[:, [ix, iy]], polygon, tol=1e-6).all()

    def test_a_curved_limit_among_them(self):
        region = DesignRegion(EIGHT, [*EIGHT_LIMITS[:7], "A*B <= 700"])
        assert not region.is_empty()
        assert region.shadow("A", "B")["kind"] == "mask"
        assert sum(map(sum, region.shadow("A", "B")["z"])) > 0
        assert region.slice_polygon("A", "B")["kind"] == "mask"
