"""The optimum search under straight constraints: exact gradients, and only the constraints a free factor enters."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import optimize
from scipy.optimize import Bounds, LinearConstraint, milp

from process_improve.experiments import DesignRegion, Factor, optimize_responses
from process_improve.experiments.optimization import (
    _ranges_from_coding,
    _search_space,
    _SearchSpace,
    _slsqp_inequality,
)

NAMES = list("abcdef")
HIGH, MINIMUM = 0.5, 0.1
FIRST, SECOND = NAMES[:3], NAMES[3:]
RULES = [
    f"0.2 <= {' + '.join(FIRST)} <= 0.6",
    f"0.4 <= {' + '.join(SECOND)}",
    f"2 <= count_nonzero({', '.join(NAMES)}) <= 4",
]
GOAL = [{"response": "y", "goal": "minimize", "low": 0.0, "high": 60.0}]


def _blend_region() -> DesignRegion:
    """Six components in two groups: each left out or at 10% to 50%, two to four of them in a blend."""
    return DesignRegion([Factor(name=n, type="mixture", high=HIGH, min_when_used=MINIMUM) for n in NAMES], RULES)


def _linear_blending(coefficients: np.ndarray) -> dict:
    return {
        "response_name": "y",
        "factor_names": NAMES,
        "coefficients": [{"term": n, "coefficient": float(c)} for n, c in zip(NAMES, coefficients, strict=True)],
    }


def _exact_minimum(coefficients: np.ndarray) -> float:
    """Return the smallest response over the region, by mixed-integer linear programming.

    ``x`` is the blend and ``z`` says which components it uses.
    """
    k = len(NAMES)
    eye, none = np.eye(k), np.zeros((1, k))
    in_first = np.array([[1.0 if n in FIRST else 0.0 for n in NAMES]])
    constraints = [
        LinearConstraint(np.hstack([eye, -HIGH * eye]), -np.inf, 0),  # x <= high z
        LinearConstraint(np.hstack([eye, -MINIMUM * eye]), 0, np.inf),  # x >= minimum z
        LinearConstraint(np.hstack([np.ones((1, k)), none]), 1, 1),
        LinearConstraint(np.hstack([in_first, none]), 0.2, 0.6),
        LinearConstraint(np.hstack([1 - in_first, none]), 0.4, np.inf),
        LinearConstraint(np.hstack([none, np.ones((1, k))]), 2, 4),
    ]
    solved = milp(
        np.concatenate([coefficients, np.zeros(k)]),
        constraints=constraints,
        integrality=np.concatenate([np.zeros(k), np.ones(k)]),
        bounds=Bounds(0, np.concatenate([np.full(k, HIGH), np.ones(k)])),
    )
    return float(solved.fun)


class TestGradients:
    """A straight constraint carries its gradient; a curved one is left to finite differences."""

    def test_a_straight_constraint_has_its_exact_gradient(self) -> None:
        a = np.array([0.5, 0.0, -2.0])
        constraint = _slsqp_inequality(lambda x: x @ a - 1.0, (a, 1.0), [2, 0, 1], 4)
        x = np.array([0.3, 0.1, 0.4, 0.9])
        numeric = optimize.approx_fprime(x, constraint["fun"], 1e-7)
        np.testing.assert_allclose(constraint["jac"](x), numeric, atol=1e-6)
        np.testing.assert_array_equal(constraint["jac"](x), [0.0, 2.0, -0.5, 0.0])

    def test_the_rounding_of_a_fitted_form_is_not_a_slope(self) -> None:
        a = np.array([1e-17, 1.0])
        constraint = _slsqp_inequality(lambda x: x @ a, (a, 0.0), [0, 1], 2)
        np.testing.assert_array_equal(constraint["jac"](np.zeros(2)), [0.0, -1.0])

    def test_a_curved_constraint_has_no_gradient(self) -> None:
        constraint = _slsqp_inequality(lambda x: x[:, 0] * x[:, 1] - 1.0, None, [0, 1], 2)
        assert "jac" not in constraint
        assert constraint["fun"](np.array([2.0, 3.0])) == -5.0

    @pytest.mark.slow
    def test_every_limit_of_a_blend_region_is_straight(self) -> None:
        space = _search_space(NAMES, None, _blend_region(), np.random.default_rng(0), 4)
        assert all("jac" in c for c in space.constraints)
        assert sum(c["type"] == "eq" for c in space.constraints) == 1
        np.testing.assert_array_equal(space.constraints[-1]["jac"](space.starts[0]), np.ones(len(NAMES)))


class TestConstraintsOfASearch:
    """A search is given the constraints that a factor free to move enters."""

    def _space(self) -> _SearchSpace:
        on_first = {"type": "ineq", "fun": lambda x: x[0], "jac": lambda _x: np.array([1.0, 0.0, 0.0])}
        on_both = {"type": "ineq", "fun": lambda x: 1 - x[0] - x[1], "jac": lambda _x: np.array([-1.0, -1.0, 0.0])}
        curved = {"type": "ineq", "fun": lambda x: 1 - x[0] * x[2]}
        return _SearchSpace([(0.0, 1.0)] * 3, [on_first, on_both, curved], [], gaps=[(0, 0.0, 0.2), (1, 0.0, 0.2)])

    def test_a_constraint_on_factors_left_out_is_not_passed(self) -> None:
        space = self._space()
        kept = space.constraints_at(np.array([0.0, 0.5, 0.5]))  # the first factor is left out: it stays at 0
        assert kept == space.constraints[1:]

    def test_every_constraint_is_passed_when_its_factors_are_in_use(self) -> None:
        space = self._space()
        assert space.constraints_at(np.array([0.5, 0.5, 0.5])) == space.constraints

    def test_a_constraint_without_a_gradient_is_always_passed(self) -> None:
        space = self._space()
        assert space.constraints_at(np.zeros(3)) == space.constraints[2:]

    @pytest.mark.slow
    def test_a_search_that_leaves_components_out_still_moves(self) -> None:
        # A fitted gradient kept its rounding (1e-15 where the slope is 0), which gave the limits of the components
        # left out a slope on the ones in use. SLSQP then stopped at its start, on half of these searches.
        coefficients = np.array([30.0, 5.0, 40.0, 20.0, 10.0, 50.0])
        region = _blend_region()
        space = _search_space(NAMES, None, region, np.random.default_rng(0), 6)
        gradients = np.array([c["jac"](space.starts[0]) for c in space.constraints])
        assert not ((gradients != 0.0) & (np.abs(gradients) < 1e-9)).any()

        def response(x: np.ndarray) -> float:
            return float((x @ coefficients - 5.0) / 45.0)

        for start in space.starts:
            assert (start == 0.0).any()
            found = optimize.minimize(
                response, start, method="SLSQP", bounds=space.bounds_at(start), constraints=space.constraints_at(start)
            )
            assert found.fun < response(start) - 1e-3


class TestOptimum:
    """The optimum of a blend with groups, minimums and a limit on how many components are used."""

    @pytest.mark.slow
    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_it_is_the_exact_optimum(self, seed: int) -> None:
        coefficients = np.random.default_rng(seed).uniform(5, 50, len(NAMES))
        region = _blend_region()
        best = optimize_responses([_linear_blending(coefficients)], goals=GOAL, region=region)["desirability"]
        blend = np.array([best["optimal_coded"][n] for n in NAMES])
        assert best["within_region"]
        assert region.feasible(blend[None, :])[0]
        assert blend @ coefficients == pytest.approx(_exact_minimum(coefficients), abs=1e-4)

    def test_a_curved_limit_is_still_followed(self) -> None:
        amounts = [Factor(name=n, low=0, high=10) for n in "AB"]
        region = DesignRegion(amounts, ["(A - 5) * (B - 5) <= 4"])
        model = {
            "response_name": "y",
            "factor_names": ["A", "B"],
            "coefficients": [{"term": t, "coefficient": c} for t, c in {"Intercept": 10.0, "A": 3.0, "B": 2.0}.items()],
        }
        goal = [{"response": "y", "goal": "maximize", "low": 0.0, "high": 20.0}]
        best = optimize_responses([model], goals=goal, region=region)["desirability"]
        coded = best["optimal_coded"]
        assert best["within_region"]
        assert 25 * coded["A"] * coded["B"] == pytest.approx(4.0, abs=1e-4)  # on the curve, where y is largest


class TestRangesOfComponents:
    """A range given for a mixture component is not used: a proportion is the same on either scale."""

    @pytest.mark.slow
    def test_the_optimum_is_reported_as_the_blend(self) -> None:
        coefficients = np.array([30.0, 5.0, 40.0, 20.0, 10.0, 50.0])
        ranges = {n: {"low": 0.0, "high": HIGH} for n in NAMES}
        best = optimize_responses(
            [_linear_blending(coefficients)], goals=GOAL, region=_blend_region(), factor_ranges=ranges
        )["desirability"]
        assert best["optimal_actual"] == best["optimal_coded"]  # it was read as coded, and did not sum to 1
        assert sum(best["optimal_actual"].values()) == pytest.approx(1.0)

    def test_a_range_for_a_process_factor_is_kept(self) -> None:
        ranges = {"a": {"low": 0.0, "high": 0.5}, "T": {"low": 20.0, "high": 60.0}}
        kept = _ranges_from_coding([{"response_name": "y"}], ranges, components=["a"])
        assert kept == {"a": {"low": -1.0, "high": 1.0}, "T": ranges["T"]}
        assert _ranges_from_coding([{"response_name": "y"}], None, components=["a"]) is None
