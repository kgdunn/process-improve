"""Tests for separating_constraint and constraint_through_cuts: a straight limit proposed from settings."""

from __future__ import annotations

import ast
import itertools
import logging
import pickle
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from process_improve import experiments
from process_improve.experiments import (
    Constraint,
    DesignRegion,
    Factor,
    NotSeparableError,
    constraint_through_cuts,
    separating_constraint,
)
from process_improve.experiments import separating as sep

#: The three factors of the spec's examples, and their eight corners (first factor slowest) and centre.
TPF = [
    Factor(name="T", low=150, high=200, units="degC"),
    Factor(name="P", low=1, high=3, units="bar"),
    Factor(name="F", low=10, high=20, units="L/min"),
]
CORNERS = [list(c) for c in itertools.product((150, 200), (1, 3), (10, 20))]
CENTRE = [175, 2, 15]
SQUARE = [Factor(name="A", low=-1, high=1), Factor(name="B", low=-1, high=1)]
#: Six factors whose half-ranges (25, 1, 5, 0.5, 22.5 and 1) make coefficients that are exactly halfway at a digit.
SIX = [
    Factor(name=name, low=low, high=high)
    for name, low, high in [("A", 150, 200), ("B", 1, 3), ("C", 10, 20), ("D", 0, 1), ("E", 5, 50), ("F", -1, 1)]
]

_settings = settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def _split(rule, centre: bool = True) -> tuple[list[list[float]], list[list[float]]]:
    """Return the corners ``rule`` marks as cannot, and the other corners (with the centre) as can."""
    cannot = [c for c in CORNERS if rule(*c)]
    can = [c for c in CORNERS if not rule(*c)] + ([CENTRE] if centre else [])
    return cannot, can


def _six_factor_split() -> tuple[list[list[float]], list[list[float]]]:
    """Return the corners of ``SIX`` with B to F all high as cannot, and the other corners (with the centre) as can."""
    corners = [list(c) for c in itertools.product(*[(f.low, f.high) for f in SIX])]
    cannot = [c for c in corners if all(v == f.high for v, f in zip(c[1:], SIX[1:], strict=True))]
    can = [c for c in corners if c not in cannot] + [[(f.low + f.high) / 2 for f in SIX]]
    return cannot, can


def _coded(points) -> np.ndarray:
    """Return settings of the three spec factors in coded units."""
    centre, half = np.array([175.0, 2.0, 15.0]), np.array([25.0, 1.0, 5.0])
    return (np.asarray(points, dtype=float) - centre) / half


def _direction(constraint: Constraint, factors: list[Factor]) -> np.ndarray:
    """Return the limit's coefficients in coded units, signed so the excluded side is where they are large."""
    coefficients, low, high = DesignRegion(factors, [constraint]).linear_forms()[0]
    a = np.array([coefficients.get(f.name, 0.0) for f in factors])
    assert (low is None) != (high is None)
    half = np.array([(f.high - f.low) / 2 for f in factors])
    return (a if high is not None else -a) * half


def _named(constraint: Constraint) -> set[str]:
    """Return the factor names the limit's expression uses."""
    return {node.id for node in ast.walk(ast.parse(constraint.expression, mode="eval")) if isinstance(node, ast.Name)}


def _move(constraint: Constraint, factors: list[Factor], w: np.ndarray) -> float:
    """Return how far the limit's coded, L1-normalised coefficients are from those of ``w``."""
    got = _direction(constraint, factors)
    return float(np.abs(got / np.abs(got).sum() - w / np.abs(w).sum()).sum())


# ---------------------------------------------------------------------------
# The pinned proposals of the spec
# ---------------------------------------------------------------------------


class TestPinnedProposals:
    @pytest.mark.parametrize(
        ("rule", "expected"),
        [
            (lambda t, p, f: (t, p, f) == (200, 3, 20), "T + 25 * P + 5 * F <= 350"),
            (lambda t, p, _f: t == 200 and p == 3, "T + 25 * P <= 250"),
            (lambda t, p, _f: t == 200 and p == 1, "T - 25 * P <= 150"),
            (lambda t, _p, _f: t == 200, "T <= 190"),
        ],
        ids=["one corner", "T and P high", "T high and P low", "every T-high corner"],
    )
    def test_a_marked_pattern_gives_the_pinned_limit(self, rule, expected):
        cannot, can = _split(rule)
        limit = separating_constraint(TPF, cannot, can)
        assert limit == Constraint(expression=expected, type="linear")
        region = DesignRegion(TPF, [limit])
        assert not region.feasible(_coded(cannot)).any()
        assert region.feasible(_coded(can)).all()

    def test_a_factor_the_pattern_ignores_has_an_exactly_zero_coefficient(self):
        cannot, can = _split(lambda t, p, _f: t == 200 and p == 3)
        limit = separating_constraint(TPF, cannot, can)
        assert DesignRegion(TPF, [limit]).linear_forms() == [({"T": 1.0, "P": 25.0}, None, 250.0)]
        w = sep._max_margin_plane(_coded(cannot), _coded(can))
        assert w[2] == 0.0

    def test_the_settings_that_can_be_run_pull_the_cut(self):
        # The centre (T = 175) can be run, so the cut sits halfway between it and T = 200: the plane of maximum
        # margin is T = 187.5, and 190 is the roundest number in the middle fifth of (175, 200). Without the
        # centre the plane is T = 175, and 180 the roundest number in (170, 180).
        assert separating_constraint(TPF, *_split(lambda t, _p, _f: t == 200)).expression == "T <= 190"
        assert separating_constraint(TPF, *_split(lambda t, _p, _f: t == 200, centre=False)).expression == "T <= 180"

    @pytest.mark.parametrize(
        ("rule", "expected"),
        [
            (lambda t, _p, _f: t == 150, "T >= 160"),
            (lambda t, p, _f: t == 150 and p == 3, "T - 25 * P >= 100"),
        ],
    )
    def test_the_first_term_is_positive_so_a_low_corner_gives_greater_or_equal(self, rule, expected):
        assert separating_constraint(TPF, *_split(rule)).expression == expected

    def test_a_single_factor(self):
        limit = separating_constraint(TPF[:1], [[200]], [[150]])
        assert limit.expression == "T <= 180"

    def test_a_cut_close_to_settings_on_both_sides_keeps_the_digits_it_needs(self):
        # Two digits (B coefficient 1.9) would move the coefficients by under 1%, but put (1, 0.536) and (-1, -0.536)
        # on the wrong sides; a third digit keeps every setting on its side.
        cannot = [[-1, -0.536], [1, 0.538]]
        can = [[-1, -0.538], [1, 0.536]]
        limit = separating_constraint(SQUARE, cannot, can)
        assert limit.expression == "A - 1.86 * B >= 0"
        region = DesignRegion(SQUARE, [limit])
        assert not region.feasible(np.array(cannot)).any()
        assert region.feasible(np.array(can)).all()

    def test_coefficients_are_decimals_when_the_ranges_call_for_them(self):
        factors = [Factor(name="A", low=0, high=1), Factor(name="B", low=0, high=3)]
        corners = [list(c) for c in itertools.product((0, 1), (0, 3))]
        limit = separating_constraint(factors, [[1, 3]], [c for c in corners if c != [1, 3]])
        # Coded, the cut is A + B <= 1; in actual units 3 * A + B, written with the smallest coefficient 1.
        assert limit.expression == "3 * A + B <= 4.5"


# ---------------------------------------------------------------------------
# Random threshold patterns on the corners and the centre
# ---------------------------------------------------------------------------


@st.composite
def threshold_patterns(draw, free: bool = False):
    """Draw factors and a pattern of corners (plus the centre) cut off by a random plane, in coded units.

    With ``free``, the plane ignores at least one factor, which then splits no group of settings.
    """
    k = draw(st.integers(2 if free else 1, 5))
    lows = draw(st.lists(st.integers(-100, 100), min_size=k, max_size=k))
    spans = draw(st.lists(st.sampled_from([0.5, 1, 2, 4, 10, 25, 50, 200]), min_size=k, max_size=k))
    weights = draw(st.lists(st.integers(-3, 3), min_size=k, max_size=k))
    if free:
        ignored = draw(st.lists(st.booleans(), min_size=k, max_size=k).filter(lambda m: any(m) and not all(m)))
        weights = [0 if skip else w for w, skip in zip(weights, ignored, strict=True)]
    assume(any(weights))
    theta = draw(st.one_of(st.sampled_from([-1.0, 0.0, 0.5, 1.0, 2.0]), st.floats(-2.5, 2.5)))
    factors = [Factor(name=f"X{j}", low=lows[j], high=lows[j] + spans[j]) for j in range(k)]
    coded = np.vstack([np.array(list(itertools.product((-1.0, 1.0), repeat=k))), np.zeros((1, k))])
    marked = coded @ np.array(weights, dtype=float) > theta
    assume(marked.any() and not marked.all())
    return factors, coded, marked, np.array(weights) != 0


def _actual(factors: list[Factor], coded: np.ndarray) -> np.ndarray:
    low, high = np.array([f.low for f in factors]), np.array([f.high for f in factors])
    return (low + high) / 2 + coded * (high - low) / 2


class TestRandomPatterns:
    @_settings
    @given(threshold_patterns())
    def test_the_proposal_rules_out_exactly_the_marked_settings(self, pattern):
        factors, coded, marked, _ = pattern
        actual = _actual(factors, coded)
        limit = separating_constraint(factors, actual[marked], actual[~marked])
        region = DesignRegion(factors, [limit])
        np.testing.assert_array_equal(region.feasible(coded), ~marked)
        slack = region.inequalities[0](coded)
        assert np.all(slack[marked] > 0)  # strictly: no setting lies on the limit
        assert np.all(slack[~marked] < 0)

    @_settings
    @given(threshold_patterns())
    def test_rounding_moves_the_coefficients_by_at_most_one_percent(self, pattern):
        factors, coded, marked, _ = pattern
        actual = _actual(factors, coded)
        limit = separating_constraint(factors, actual[marked], actual[~marked])
        w = sep._max_margin_plane(coded[marked], coded[~marked])
        assert _move(limit, factors, w) <= 0.01 + 1e-9

    @_settings
    @given(threshold_patterns(free=True))
    def test_a_factor_the_pattern_ignores_gets_exactly_zero(self, pattern):
        factors, coded, marked, used = pattern
        actual = _actual(factors, coded)
        w = sep._max_margin_plane(coded[marked], coded[~marked])
        # A factor with no weight splits no group: reflecting it maps both sets to themselves, so the unique plane
        # of maximum margin cannot depend on it.
        assert np.all(w[~used] == 0.0)
        limit = separating_constraint(factors, actual[marked], actual[~marked])
        assert not _named(limit) & {f.name for f, u in zip(factors, used, strict=True) if not u}

    def test_the_plane_found_is_the_one_of_maximum_margin(self):
        # Against an independent solve of the same problem, by SLSQP from a point that separates the sets.
        rng = np.random.default_rng(0)
        for _ in range(30):
            k = int(rng.integers(2, 6))
            coded = np.vstack([np.array(list(itertools.product((-1.0, 1.0), repeat=k))), np.zeros((1, k))])
            marked = coded @ rng.normal(size=k) > rng.uniform(-1, 2)
            if marked.all() or not marked.any():
                continue
            cannot, can = coded[marked], coded[~marked]
            w_lp, c_lp, t = sep._separation(cannot, can)
            reference = sep._slsqp_plane(cannot, can, (w_lp / t, c_lp / t))
            w = sep._max_margin_plane(cannot, can)
            np.testing.assert_allclose(w / np.linalg.norm(w), reference / np.linalg.norm(reference), atol=1e-6)


# ---------------------------------------------------------------------------
# Ties, symmetry and determinism
# ---------------------------------------------------------------------------


class TestTiesAndDeterminism:
    def test_the_linear_program_alone_has_many_optima_and_the_proposal_takes_the_symmetric_one(self):
        # T high with P and F both low or both high cannot be run; T high with P and F apart is left out. The
        # pattern is unchanged by reflecting P and F together, or by swapping them.
        cannot = [[200, 1, 10], [200, 3, 20]]
        can = [c for c in CORNERS if c[0] == 150] + [CENTRE]
        p, q = _coded(cannot), _coded(can)
        _, _, best = sep._separation(p, q)

        def margin(w, c):
            w = np.asarray(w, dtype=float)
            return min(np.min(p @ w - c), np.min(c - q @ w))

        tilted, level, other = margin([1, -0.5, 0.5], 0.5), margin([1, 0, 0], 0.5), margin([1, 0.5, -0.5], 0.5)
        assert tilted == level == other == pytest.approx(best)  # three different planes reach the optimum
        assert separating_constraint(TPF, cannot, can).expression == "T <= 190"

    def test_a_pattern_symmetric_in_two_factors_gives_equal_coded_coefficients(self):
        # Only (200, 3, 20) cannot be run, against the T-low corners and the centre: swapping P and F (in coded
        # units) leaves the pattern unchanged, so their coded coefficients agree.
        can = [c for c in CORNERS if c[0] == 150] + [CENTRE]
        limit = separating_constraint(TPF, [[200, 3, 20]], can)
        assert limit.expression == "T + 12.5 * P + 2.5 * F <= 260"
        direction = _direction(limit, TPF)
        assert direction[1] == direction[2]

    def test_equal_coefficients_exactly_halfway_between_two_round_alike(self):
        # B to F all high cannot be run, whatever A; the plane of maximum margin gives B to F the same coded
        # coefficient, which in actual units, the smallest made 1, is exactly 22.5 for B and for F. Rounded as the
        # solver returned them, a few units in the last place apart, B went to 23 and F to 22.
        cannot, can = _six_factor_split()
        limit = separating_constraint(SIX, cannot, can)
        assert limit.expression == "22.5 * B + 4.5 * C + 45 * D + E + 22.5 * F <= 250"
        direction = _direction(limit, SIX)
        assert direction[0] == 0.0
        np.testing.assert_array_equal(direction[1:], direction[1])

    def test_the_solvers_last_bit_noise_does_not_change_the_limit(self):
        # Native Python and Pyodide find planes a few units in the last place apart. Nudged by that much, each
        # coefficient its own way, the plane of the test above is still written the same.
        cannot, can = (np.array(points, dtype=float) for points in _six_factor_split())
        half = np.array([(f.high - f.low) / 2 for f in SIX])
        w = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        expected = sep._rounded(w / half, half, cannot, can)
        assert sep._expression([f.name for f in SIX], *expected).startswith("22.5 * B + ")
        eps = np.finfo(float).eps
        rng = np.random.default_rng(0)
        for _ in range(20):
            nudged = w * (1.0 + rng.integers(-4, 5, size=len(w)) * eps)
            assert sep._rounded(nudged / half, half, cannot, can) == expected

    def test_the_square_corner_gives_the_diagonal(self):
        cannot = [[1, 1]]
        can = [[-1, -1], [-1, 1], [1, -1], [0, 0]]
        assert separating_constraint(SQUARE, cannot, can).expression == "A + B <= 1"

    def test_the_same_question_gives_the_same_answer(self):
        cannot, can = _split(lambda t, p, f: t + 25 * p + 5 * f > 300)
        first = separating_constraint(TPF, cannot, can)
        assert separating_constraint(TPF, cannot, can) == first
        rng = np.random.default_rng(3)
        shuffled = separating_constraint(TPF, rng.permutation(cannot), rng.permutation(can))
        assert shuffled == first

    def test_every_form_of_the_settings_gives_the_same_answer(self):
        cannot, can = _split(lambda t, p, _f: t == 200 and p == 3)
        expected = separating_constraint(TPF, cannot, can)
        names = ["T", "P", "F"]
        frames = pd.DataFrame(cannot, columns=names)[["F", "T", "P"]], pd.DataFrame(can, columns=names)
        assert separating_constraint(TPF, *frames) == expected
        dicts = [dict(zip(names, c, strict=True)) for c in cannot], [dict(zip(names, c, strict=True)) for c in can]
        assert separating_constraint(TPF, *dicts) == expected
        assert separating_constraint(TPF, np.array(cannot), np.array(can)) == expected


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


class TestRefusals:
    def test_the_xor_pattern_has_no_setting_to_blame(self):
        with pytest.raises(NotSeparableError, match="interleaved") as info:
            separating_constraint(SQUARE, [[1, 1], [-1, -1]], [[1, -1], [-1, 1]])
        assert isinstance(info.value, ValueError)
        assert (info.value.inside_can, info.value.inside_cannot) == ([], [])

    def test_the_xor_pattern_with_the_centre_blames_the_centre(self):
        with pytest.raises(NotSeparableError, match=r"can\[2\] \(A = 0, B = 0\)") as info:
            separating_constraint(SQUARE, [[1, 1], [-1, -1]], [[1, -1], [-1, 1], [0, 0]])
        assert (info.value.inside_can, info.value.inside_cannot) == ([], [2])

    def test_every_corner_marked_blames_the_centre(self):
        with pytest.raises(NotSeparableError, match="lies between settings that cannot be run") as info:
            separating_constraint(TPF, CORNERS, [CENTRE])
        assert (info.value.inside_can, info.value.inside_cannot) == ([], [0])

    def test_a_marked_setting_between_settings_that_can_be_run(self):
        with pytest.raises(NotSeparableError, match="lies between settings that can be run") as info:
            separating_constraint(TPF, [CENTRE], CORNERS)
        assert (info.value.inside_can, info.value.inside_cannot) == ([0], [])

    def test_a_setting_that_can_be_run_between_marked_settings(self):
        cannot = [c for c in CORNERS if c[0] == 200]
        can = [c for c in CORNERS if c[0] == 150] + [CENTRE, [200, 2, 15]]  # the middle of the T-high face
        with pytest.raises(NotSeparableError) as info:
            separating_constraint(TPF, cannot, can)
        assert (info.value.inside_can, info.value.inside_cannot) == ([], [5])

    def test_a_setting_in_both_lists_is_blamed_in_both(self):
        with pytest.raises(NotSeparableError) as info:
            separating_constraint(TPF, [[200, 3, 20], [200, 3, 10]], [[150, 1, 10], [200, 3, 10]])
        assert (info.value.inside_can, info.value.inside_cannot) == ([1], [1])

    def test_nothing_marked_is_refused_but_is_not_called_inseparable(self):
        with pytest.raises(ValueError, match="nothing to rule out") as info:
            separating_constraint(TPF, [], CORNERS)
        assert not isinstance(info.value, NotSeparableError)

    def test_every_corner_marked_with_nothing_that_can_be_run_is_refused(self):
        with pytest.raises(ValueError, match="any cut would do") as info:
            separating_constraint(TPF, CORNERS, [])
        assert not isinstance(info.value, NotSeparableError)

    def test_the_error_keeps_its_indices_through_pickling(self):
        error = NotSeparableError("message", inside_can=[3, 1], inside_cannot=[2])
        again = pickle.loads(pickle.dumps(error))  # noqa: S301 - our own object
        assert (str(again), again.inside_can, again.inside_cannot) == ("message", [1, 3], [2])


class TestInputs:
    @pytest.mark.parametrize(
        ("cannot", "match"),
        [
            ([[210, 3, 20]], r"Setting 0 of 'cannot', T = 210, P = 3, F = 20, is outside"),
            ([[200, 3]], r"one setting per row.*shape \(1, 2\)"),
            ([200, 3, 20], r"one setting per row.*shape \(3,\)"),
            ([[200, float("nan"), 20]], "finite number"),
            ([{"T": 200, "P": 3}], r"no value for the factors \['F'\]"),
            (pd.DataFrame({"T": [200], "P": [3]}), r"no column for the factors \['F'\]"),
        ],
    )
    def test_settings_that_cannot_be_read_are_refused(self, cannot, match):
        with pytest.raises(ValueError, match=match):
            separating_constraint(TPF, cannot, [CENTRE])

    def test_a_setting_within_rounding_of_a_range_end_is_accepted(self):
        limit = separating_constraint(TPF, [[200 + 1e-12, 3, 20]], [CENTRE])
        assert DesignRegion(TPF, [limit]).feasible(np.zeros((1, 3)))[0]

    @pytest.mark.parametrize(
        ("factors", "match"),
        [
            ([], "at least one factor"),
            ([*TPF[:2], Factor(name="C", type="categorical", levels=["a", "b"])], "categorical"),
            ([Factor(name="x1", type="mixture"), Factor(name="x2", type="mixture")], "mixture"),
            ([Factor(name="Temp (C)", low=0, high=1)], "not a valid identifier"),
            ([Factor(name="lambda", low=0, high=1)], "not a valid identifier"),
            ([Factor(name="T", low=0, high=1), Factor(name="T", low=0, high=2)], "different names"),
        ],
    )
    def test_factors_a_limit_cannot_name_are_refused(self, factors, match):
        with pytest.raises(ValueError, match=match):
            separating_constraint(factors, [], [])

    def test_the_names_are_exported(self):
        for name in ("separating_constraint", "constraint_through_cuts", "NotSeparableError"):
            assert name in experiments.__all__
            assert getattr(experiments, name) is getattr(sep, name)


# ---------------------------------------------------------------------------
# Fallbacks of the plane of maximum margin
# ---------------------------------------------------------------------------


class TestFallbacks:
    @pytest.mark.parametrize(
        "rule",
        [
            lambda t, p, f: (t, p, f) == (200, 3, 20),
            lambda t, p, _f: t == 200 and p == 3,
            lambda t, p, _f: t == 200 and p == 1,
            lambda t, _p, _f: t == 200,
        ],
    )
    def test_slsqp_gives_the_same_limits_when_nnls_cannot_be_certified(self, monkeypatch, rule):
        cannot, can = _split(rule)
        expected = separating_constraint(TPF, cannot, can)
        monkeypatch.setattr(sep, "_least_distance", lambda _g, _h: None)
        assert separating_constraint(TPF, cannot, can) == expected

    def test_slsqp_finds_its_own_start_when_none_is_given(self, monkeypatch):
        cannot, can = _split(lambda t, p, _f: t == 200 and p == 3)
        p, q = _coded(cannot), _coded(can)
        expected = sep._max_margin_plane(p, q)
        monkeypatch.setattr(sep, "_least_distance", lambda _g, _h: None)
        np.testing.assert_allclose(sep._max_margin_plane(p, q), expected, atol=1e-7)

    def test_an_uncertified_nnls_answer_is_refused(self, monkeypatch):
        # A degenerate problem on which nnls stops short: its answer misses the inequalities, and is not kept.
        monkeypatch.setattr(sep, "nnls", lambda e, _f: (np.full(e.shape[1], 1e-3), 0.0))
        g = np.array([[1.0, 0.0], [0.0, 1.0]])
        assert sep._least_distance(g, np.ones(2)) is None
        monkeypatch.setattr(sep, "nnls", lambda e, _f: (np.ones(e.shape[1]), 0.0))
        assert sep._least_distance(g, np.ones(2)) is None  # 1 - h @ u <= 0: no common point

    def test_coefficients_too_small_to_matter_are_zeroed_only_when_the_cut_still_separates(self, monkeypatch):
        cannot, can = _split(lambda t, p, _f: t == 200 and p == 1)
        monkeypatch.setattr(sep, "_SNAP", 0.6)  # would zero every coefficient of the plane T - P = 1 (coded)
        assert separating_constraint(TPF, cannot, can).expression == "T - 25 * P <= 150"

    def test_the_linear_program_plane_is_the_last_resort(self, monkeypatch, caplog):
        cannot, can = _split(lambda t, p, f: (t, p, f) == (200, 3, 20))
        monkeypatch.setattr(sep, "_least_distance", lambda _g, _h: None)
        monkeypatch.setattr(sep, "minimize", lambda *_a, **_k: SimpleNamespace(x=np.zeros(4)))
        with caplog.at_level(logging.WARNING, logger=sep.__name__):
            limit = separating_constraint(TPF, cannot, can)
        assert "maximum margin was not found" in caplog.text
        region = DesignRegion(TPF, [limit])
        assert not region.feasible(_coded(cannot)).any()
        assert region.feasible(_coded(can)).all()


# ---------------------------------------------------------------------------
# The limit through cut points
# ---------------------------------------------------------------------------


class TestConstraintThroughCuts:
    def test_the_spec_example(self):
        limit = constraint_through_cuts(TPF[:2], {"T": 200, "P": 3}, {"T": 190, "P": 2.5})
        assert limit == Constraint(expression="T + 20 * P <= 250", type="linear")

    def test_a_factor_that_does_not_limit_this_drops_out(self):
        for cuts in ({"T": 190, "P": 2.5, "F": None}, {"T": 190, "P": 2.5}):
            assert constraint_through_cuts(TPF, [200, 3, 20], cuts).expression == "T + 20 * P <= 250"
        assert constraint_through_cuts(TPF, (200, 3, 20), {"T": 190, "P": None}).expression == "T <= 190"

    @pytest.mark.parametrize(
        ("corner", "cuts", "expected"),
        [
            ({"T": 200, "P": 3}, {"T": 187, "P": 2.5}, "T + 26 * P <= 265"),
            ({"T": 200, "P": 3}, {"T": 187.35, "P": 2.37}, "1260 * T + 25300 * P <= 311961"),
            ({"T": 150, "P": 1}, {"T": 175}, "T >= 175"),
            ({"T": 200, "P": 1}, {"T": 150, "P": 3}, "T - 25 * P <= 125"),
        ],
    )
    def test_the_plane_through_the_cuts_is_exact(self, corner, cuts, expected):
        assert constraint_through_cuts(TPF[:2], corner, cuts).expression == expected

    def test_a_low_corner_keeps_the_settings_above_the_cut(self):
        limit = constraint_through_cuts(TPF, {"T": 150, "P": 1, "F": 20}, {"T": 160, "P": 2, "F": 15})
        assert limit.expression == "T + 10 * P - 2 * F >= 130"

    @_settings
    @given(
        st.lists(st.booleans(), min_size=3, max_size=3),
        st.lists(st.integers(1, 99), min_size=3, max_size=3),
        st.lists(st.booleans(), min_size=3, max_size=3).filter(any),
    )
    def test_the_cut_points_lie_on_the_limit_and_the_corner_is_ruled_out(self, highs, percents, limited):
        corner = [f.high if high else f.low for f, high in zip(TPF, highs, strict=True)]
        cuts = {}
        for f, v, pct, used in zip(TPF, corner, percents, limited, strict=True):
            step = (f.high - f.low) * pct / 100
            cuts[f.name] = round(v - step if v == f.high else v + step, 10) if used else None  # as a person types it
        limit = constraint_through_cuts(TPF, corner, cuts)
        coefficients, low, high = DesignRegion(TPF, [limit]).linear_forms()[0]
        bound = high if high is not None else low

        def value(point: list[float]) -> float:
            return sum(coefficients.get(f.name, 0.0) * x for f, x in zip(TPF, point, strict=True))

        assert (value(corner) > bound) if high is not None else (value(corner) < bound)
        for j, f in enumerate(TPF):
            if cuts[f.name] is not None:
                point = list(corner)
                point[j] = cuts[f.name]
                assert value(point) == pytest.approx(bound, rel=1e-12, abs=1e-12)
        assert set(coefficients) == {f.name for f in TPF if cuts[f.name] is not None}

    def test_numbers_needing_more_than_twelve_digits_are_rounded_to_twelve(self):
        corner = {"T": 200, "P": 3}
        cuts = {"T": 187.123456789, "P": 2.987654321}
        limit = constraint_through_cuts(TPF[:2], corner, cuts)
        coefficients, low, high = DesignRegion(TPF[:2], [limit]).linear_forms()[0]
        assert low is None
        assert coefficients["T"] == 1.0
        for point in ([cuts["T"], 3], [200, cuts["P"]]):
            value = coefficients["T"] * point[0] + coefficients["P"] * point[1]
            assert value == pytest.approx(high, rel=1e-10)

    @pytest.mark.parametrize(
        ("corner", "cuts", "match"),
        [
            ({"T": 200, "P": 3}, {"T": 200}, "at or beyond the corner"),
            ({"T": 150, "P": 3}, {"T": 150}, "at or beyond the corner"),
            ({"T": 200, "P": 3}, {"T": 210}, "outside its range"),
            ({"T": 150, "P": 3}, {"T": 140}, "outside its range"),
            ({"T": 200, "P": 3}, {"T": float("nan")}, "outside its range"),
            ({"T": 200, "P": 3}, {"T": None, "P": None}, "at least one factor"),
            ({"T": 200, "P": 3}, {}, "at least one factor"),
            ({"T": 200, "P": 3}, {"Z": 1}, r"name \['Z'\]"),
            ({"T": 190, "P": 3}, {"T": 180}, "not a corner of the box"),
            ({"T": 200}, {"T": 180}, r"no value for the factors \['P'\]"),
            ([200], {"T": 180}, "must give 2 values"),
        ],
    )
    def test_cuts_that_do_not_describe_a_limit_are_refused(self, corner, cuts, match):
        with pytest.raises(ValueError, match=match):
            constraint_through_cuts(TPF[:2], corner, cuts)


# ---------------------------------------------------------------------------
# The helpers that write the limit
# ---------------------------------------------------------------------------


class TestWriting:
    @pytest.mark.parametrize(
        ("low", "high", "expected"),
        [
            (325, 375, 350),  # the middle fifth (345, 355) holds a multiple of 50
            (175, 200, 190),  # (185, 190): 190 is a multiple of 10, and no multiple of 20 or 50 fits
            (150, 200, 180),  # (170, 180): 180 is a multiple of 20, 170 only of 10
            (-2, 2, 0),  # zero is the roundest of all
            (-2, 8, 2),  # (2, 4): two multiples of 2, equally near the middle; the smaller in size wins
            (-8, 2, -2),
            (Fraction(1, 10), Fraction(3, 10), Fraction(1, 5)),
            (Fraction(1001, 1000), Fraction(1002, 1000), Fraction(10015, 10000)),
        ],
    )
    def test_the_roundest_number_in_the_middle_fifth(self, low, high, expected):
        assert sep._roundest(Fraction(low), Fraction(high)) == expected

    @_settings
    @given(st.fractions(-1000, 1000), st.fractions(Fraction(1, 10**6), 1000))
    def test_the_roundest_number_lies_in_the_middle_fifth(self, low, width):
        rhs = sep._roundest(low, low + width)
        middle = low + width / 2
        assert abs(rhs - middle) <= width / 10
        assert sep._terminates(rhs)

    @pytest.mark.parametrize(
        ("value", "text"),
        [(Fraction(25), "25"), (Fraction(1, 25), "0.04"), (Fraction(-5, 2), "-2.5"), (Fraction(1, 10**5), "0.00001")],
    )
    def test_decimals_are_written_in_positional_notation(self, value, text):
        assert sep._decimal_text(value) == text

    def test_digits_written_leave_out_a_coefficient_of_one(self):
        assert sep._digits_written([Fraction(1), Fraction(25), Fraction(5)]) == 3
        assert sep._digits_written([Fraction(1, 25), Fraction(1), Fraction(1, 5)]) == 5
        assert sep._digits_written([Fraction(-1), Fraction(0)]) == 0
