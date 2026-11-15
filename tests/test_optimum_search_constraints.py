"""The optimum search under straight constraints: exact gradients, and only the constraints a free factor enters."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy import optimize
from scipy.optimize import Bounds, LinearConstraint, milp

from process_improve.experiments import DesignRegion, Factor, optimize_responses
from process_improve.experiments._desirability import individual_desirability, overall_desirability
from process_improve.experiments.optimization import (
    _of_different_sets,
    _ranges_from_coding,
    _search_space,
    _SearchSpace,
    _slsqp_inequality,
    _straight_form,
    _tangent,
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
        a = np.array([0.5, 0.0, -2.0])  # over the region's columns, which are the model's factors 2, 0 and 1
        form = _straight_form((a, 1.0), [2, 0, 1], 4)
        assert form is not None
        np.testing.assert_array_equal(form[0], [0.0, -2.0, 0.5, 0.0])
        constraint = _slsqp_inequality(lambda x: x @ a - 1.0, form, [2, 0, 1])
        x = np.array([0.3, 0.1, 0.4, 0.9])
        numeric = optimize.approx_fprime(x, constraint["fun"], 1e-7)
        np.testing.assert_allclose(constraint["jac"](x), numeric, atol=1e-6)
        np.testing.assert_array_equal(constraint["jac"](x), [0.0, 2.0, -0.5, 0.0])

    def test_the_rounding_of_a_fitted_form_is_not_a_slope(self) -> None:
        form = _straight_form((np.array([1e-17, 1.0]), 0.0), [0, 1], 2)
        assert form is not None
        np.testing.assert_array_equal(form[0], [0.0, 1.0])

    def test_a_curved_constraint_has_no_gradient(self) -> None:
        assert _straight_form(None, [0, 1], 2) is None
        constraint = _slsqp_inequality(lambda x: x[:, 0] * x[:, 1] - 1.0, None, [0, 1])
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


def _with_blending(name: str, linear: np.ndarray, pairs: np.ndarray) -> tuple[dict, object]:
    """Return a blending model with a term for every pair of components, and a function that predicts with it."""
    index = list(itertools.combinations(range(len(NAMES)), 2))
    terms = [{"term": n, "coefficient": float(c)} for n, c in zip(NAMES, linear, strict=True)]
    terms += [
        {"term": f"{NAMES[i]}:{NAMES[j]}", "coefficient": float(c)} for (i, j), c in zip(index, pairs, strict=True)
    ]
    quadratic = np.zeros((len(NAMES), len(NAMES)))
    for (i, j), c in zip(index, pairs, strict=True):
        quadratic[i, j] = c
    return {"response_name": name, "factor_names": NAMES, "coefficients": terms}, lambda x: float(
        x @ linear + x @ quadratic @ x
    )


def _best_of_every_set(desirability: object, region: DesignRegion, rng: np.random.Generator) -> float:
    """Return the largest desirability that searches inside every allowed set of components find."""
    k, best = len(NAMES), 0.0
    in_first = np.array([1.0 if n in FIRST else 0.0 for n in NAMES])
    for size in (2, 3, 4):
        for used in itertools.combinations(range(k), size):
            columns = list(used)
            constraints = [
                {"type": "eq", "fun": lambda z: z.sum() - 1.0},
                {"type": "ineq", "fun": lambda z, columns=columns: 0.6 - in_first[columns] @ z},
                {"type": "ineq", "fun": lambda z, columns=columns: in_first[columns] @ z - 0.2},
            ]

            def score(z: np.ndarray, columns: list[int] = columns) -> float:
                x = np.zeros(k)
                x[columns] = z
                return -desirability(x)  # type: ignore[operator]

            for _ in range(4):
                start = MINIMUM + (1 - MINIMUM * size) * rng.dirichlet(np.ones(size))
                res = optimize.minimize(
                    score, start, method="SLSQP", bounds=[(MINIMUM, HIGH)] * size, constraints=constraints
                )
                x = np.zeros(k)
                x[columns] = res.x
                if region.feasible(x[None, :], tol=1e-6)[0]:
                    best = max(best, -res.fun)
    return best


class TestCurvedModels:
    """Models with blending terms are searched for: from promising sets, their neighbours, the tangents' best set."""

    def test_the_tangent_of_a_model_is_its_value_and_slopes_at_the_point(self) -> None:
        terms = {"Intercept": 2.0, "a": 3.0, "b": -1.0, "a:b": 4.0, "I(a ** 2)": 0.5, "a:b:c": 2.0}
        coefficients = [{"term": term, "coefficient": value} for term, value in terms.items()]
        x = np.array([0.3, 0.2, 0.5, 0.0, 0.0, 0.0])

        def model(point: np.ndarray) -> float:
            a, b, c = point[:3]
            return 2.0 + 3.0 * a - b + 4.0 * a * b + 0.5 * a**2 + 2.0 * a * b * c

        slopes, intercept = _tangent(coefficients, NAMES, x)
        np.testing.assert_allclose(slopes, optimize.approx_fprime(x, model, 1e-7), atol=1e-6)
        assert slopes @ x + intercept == pytest.approx(model(x))

    def test_the_results_searched_from_use_different_sets(self) -> None:
        space = _SearchSpace([(0.0, 1.0)] * 2, [], [], gaps=[(0, 0.0, 0.2), (1, 0.0, 0.2)])
        results = [
            optimize.OptimizeResult(x=np.array(x), fun=fun)
            for x, fun in (([0.5, 0.0], -0.9), ([0.6, 0.0], -0.8), ([0.5, 0.5], -0.7), ([0.0, 0.4], -0.6))
        ]
        kept = _of_different_sets(results, space, 2)
        assert [res.fun for res in kept] == [-0.9, -0.7]  # the second uses the set of the first
        assert len(_of_different_sets(results, space, 5)) == 3

    @pytest.mark.slow
    def test_the_starts_are_the_most_promising_settings_of_different_sets(self) -> None:
        coefficients = np.array([30.0, 5.0, 40.0, 20.0, 10.0, 50.0])
        region = _blend_region()

        def response(x: np.ndarray) -> float:
            return float(x @ coefficients)

        space = _search_space(NAMES, None, region, np.random.default_rng(0), 8, score=response)
        sets = {tuple(start > 0) for start in space.starts}
        assert len(sets) == len(space.starts) == 8
        scores = [response(start) for start in space.starts]
        assert scores == sorted(scores)
        blends = region.support_points()
        assert scores[0] == pytest.approx(min(response(blend) for blend in blends))

    @pytest.mark.slow
    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_two_responses_with_blending_terms_reach_the_best_of_every_set(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        n_pairs = len(NAMES) * (len(NAMES) - 1) // 2
        fitted, predict = zip(
            *(_with_blending(f"y{i}", rng.uniform(5, 50, len(NAMES)), rng.normal(0, 25, n_pairs)) for i in range(2)),
            strict=True,
        )
        goals = [
            {"response": "y0", "goal": "minimize", "low": 10.0, "high": 45.0},
            {"response": "y1", "goal": "maximize", "low": 10.0, "high": 45.0},
        ]

        def desirability(x: np.ndarray) -> float:
            d_values = [individual_desirability(f(x), g) for f, g in zip(predict, goals, strict=True)]
            return overall_desirability(d_values, goals)

        region = _blend_region()
        best = optimize_responses(list(fitted), goals=goals, region=region)["desirability"]
        assert best["within_region"]
        assert not best["proven_optimal"] or best["composite_desirability"] == 1.0
        assert best["composite_desirability"] >= _best_of_every_set(desirability, region, rng) - 1e-4
