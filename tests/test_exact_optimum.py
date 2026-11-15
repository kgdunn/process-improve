"""The optimum of straight-line models over straight constraints is found exactly, and proven."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy import optimize

from process_improve.experiments import DesignRegion, Factor, optimize_responses
from process_improve.experiments import _exact_optimum as exact
from process_improve.experiments._desirability import individual_desirability, overall_desirability
from process_improve.experiments._exact_optimum import ExactOptimum, StraightRegion, exact_optimum


def _judge(models: list[tuple[np.ndarray, float]], goals: list[dict], importances: list[float] | None = None):
    """Return the composite desirability of a setting, as the library scores it."""

    def desirability(x: np.ndarray) -> float:
        d_values = [individual_desirability(float(a @ x + c), g) for (a, c), g in zip(models, goals, strict=True)]
        return overall_desirability(d_values, goals, importances)

    return desirability


def _found(region: StraightRegion, models: list, goals: list, importances: list[float] | None = None) -> ExactOptimum:
    found = exact_optimum(region, models, goals, importances, _judge(models, goals, importances))
    assert found is not None
    return found


UP = (np.array([1.0]), 0.0)  # y = x
DOWN = (np.array([-1.0]), 1.0)  # y = 1 - x
UNIT = StraightRegion([(0.0, 1.0)])


def _maximize(name: str = "y", **more: float) -> dict:
    return {"response": name, "goal": "maximize", "low": 0.0, "high": 1.0, **more}


class TestOneFactor:
    """Cases whose answer is known in closed form: the weighted geometric mean of ``x`` and ``1 - x``."""

    @pytest.mark.parametrize(("first", "second"), [(1.0, 1.0), (1.0, 3.0), (5.0, 2.0)])
    def test_two_responses_that_pull_apart(self, first: float, second: float) -> None:
        # D = (x^a (1 - x)^b)^(1 / (a + b)) is largest at x = a / (a + b).
        found = _found(UNIT, [UP, DOWN], [_maximize("up"), _maximize("down")], [first, second])
        share = first / (first + second)
        assert found.proven
        assert found.x[0] == pytest.approx(share, abs=1e-3)
        best = (share**first * (1 - share) ** second) ** (1 / (first + second))
        assert found.bound == pytest.approx(best, rel=1e-6)

    def test_the_weight_of_a_goal_moves_the_optimum_as_an_importance_does(self) -> None:
        # d = x^3 beside d = 1 - x: the product x^3 (1 - x) is largest at x = 3/4.
        found = _found(UNIT, [UP, DOWN], [_maximize("up", weight=3.0), _maximize("down")])
        assert found.proven
        assert found.x[0] == pytest.approx(0.75, abs=1e-3)

    def test_a_target_is_reached(self) -> None:
        goal = {
            "response": "y",
            "goal": "target",
            "low": 0.0,
            "target": 0.6,
            "high": 1.0,
            "weight": 2.0,
            "weight_high": 0.5,
        }
        found = _found(UNIT, [UP], [goal])
        assert found.proven
        assert found.x[0] == pytest.approx(0.6, abs=1e-6)
        assert found.bound == pytest.approx(1.0)

    def test_a_target_beside_a_goal_that_pulls_past_it(self) -> None:
        # d1 = x / 0.6 below the target and (1 - x) / 0.4 above it, and d2 = x: past the target their product falls.
        target = {"response": "t", "goal": "target", "low": 0.0, "target": 0.6, "high": 1.0}
        found = _found(UNIT, [UP, UP], [target, _maximize("up")])
        assert found.proven
        assert found.x[0] == pytest.approx(0.6, abs=1e-4)
        assert found.bound == pytest.approx(np.sqrt(0.6), rel=1e-6)

    def test_an_in_range_goal_stops_the_setting_just_inside_its_limit(self) -> None:
        held = {"response": "held", "goal": "in_range", "low": 0.0, "high": 0.4}
        models, goals = [UP, UP], [_maximize("up"), held]
        found = _found(UNIT, models, goals)
        assert found.proven
        assert 0.4 - 1e-6 < found.x[0] < 0.4  # on the limit it would score 0 by rounding as often as not
        assert _judge(models, goals)(found.x) == pytest.approx(0.4, abs=1e-6)
        assert found.bound == pytest.approx(0.4, abs=1e-9)  # the bound is of the whole range

    def test_no_setting_scores_above_zero(self) -> None:
        out_of_reach = {"response": "y", "goal": "maximize", "low": 2.0, "high": 3.0}
        assert exact_optimum(UNIT, [UP], [out_of_reach], None, _judge([UP], [out_of_reach])) is None


class TestFactorsLeftOut:
    """Factors that are left out or used at no less than a minimum, and limits on how many are used."""

    GAINS = (np.array([3.0, 2.0, 1.0]), 0.0)
    GOAL = ({"response": "y", "goal": "maximize", "low": 0.0, "high": 10.0},)

    def _region(self, fewest: int, most: int, rows: list | None = None) -> StraightRegion:
        gaps = [(j, 0.0, 0.2) for j in range(3)]
        return StraightRegion([(0.0, 1.0)] * 3, rows or [], gaps=gaps, counts=[([0, 1, 2], fewest, most)])

    def test_at_most_one_factor_is_the_one_that_gains_most(self) -> None:
        found = _found(self._region(0, 1), [self.GAINS], list(self.GOAL))
        assert found.proven
        np.testing.assert_allclose(found.x, [1.0, 0.0, 0.0], atol=1e-9)

    def test_a_factor_left_out_is_at_its_zero_exactly(self) -> None:
        found = _found(self._region(0, 2), [self.GAINS], list(self.GOAL))
        np.testing.assert_array_equal(found.x[2:], [0.0])
        np.testing.assert_allclose(found.x[:2], [1.0, 1.0], atol=1e-9)

    def test_a_factor_that_must_be_used_is_at_its_minimum_when_it_costs(self) -> None:
        costs = (np.array([-3.0, -2.0, -1.0]), 10.0)
        found = _found(self._region(1, 3), [costs], list(self.GOAL))
        assert found.proven
        np.testing.assert_allclose(found.x, [0.0, 0.0, 0.2], atol=1e-9)  # the cheapest one, at its minimum

    def test_a_limit_on_a_total_changes_the_set(self) -> None:
        # A + B <= 1.1 keeps A and B from both being at 1: the best is A at 1 with C, not A with B at 0.2 ... 0.1.
        row = (np.array([1.0, 1.0, 0.0]), 1.1)
        found = _found(self._region(0, 2), [self.GAINS], list(self.GOAL), None)
        limited = _found(self._region(0, 2, [row]), [self.GAINS], list(self.GOAL), None)
        assert found.bound > limited.bound
        np.testing.assert_allclose(limited.x, [1.0, 0.0, 1.0], atol=1e-9)  # 3 + 1 beats 3 * 0.9 + 2 * 0.2

    def test_a_search_box_that_ends_below_the_minimum_leaves_the_factor_out(self) -> None:
        region = StraightRegion([(0.0, 1.0), (0.0, 0.1)], gaps=[(0, 0.0, 0.2), (1, 0.0, 0.2)])
        found = _found(region, [(np.array([1.0, 5.0]), 0.0)], list(self.GOAL))
        np.testing.assert_allclose(found.x, [1.0, 0.0], atol=1e-9)

    def test_a_blend_sums_to_one(self) -> None:
        region = StraightRegion([(0.0, 0.6)] * 3, components=[0, 1, 2], gaps=[(j, 0.0, 0.3) for j in range(3)])
        found = _found(region, [self.GAINS], list(self.GOAL))
        assert found.proven
        np.testing.assert_allclose(found.x, [0.6, 0.4, 0.0], atol=1e-9)
        assert found.x.sum() == pytest.approx(1.0)


class TestTheBound:
    """The bound and the setting agree, or the result is not proven."""

    def test_a_programme_that_is_not_solved_to_the_end_is_not_proven(self, monkeypatch: pytest.MonkeyPatch) -> None:
        solve = exact._Programme.solve

        def stopped_early(self: exact._Programme, *args: object, **kwargs: object) -> tuple | None:
            solved = solve(self, *args, **kwargs)  # type: ignore[arg-type]
            return None if solved is None else (*solved[:3], False)

        monkeypatch.setattr(exact._Programme, "solve", stopped_early)
        found = _found(UNIT, [UP, DOWN], [_maximize("up"), _maximize("down")])
        assert not found.proven
        assert found.x[0] == pytest.approx(0.5, abs=0.05)

    def test_rounds_that_run_out_leave_the_best_setting_unproven(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(exact, "_ROUNDS", 1)
        monkeypatch.setattr(exact, "_AGREEMENT", 0.0)
        found = _found(UNIT, [UP, DOWN], [_maximize("up"), _maximize("down", weight=3.0)])
        assert not found.proven
        assert 0.0 < found.x[0] < 1.0

    def test_a_programme_the_solver_fails_on_is_solved_again_without_presolve(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from process_improve.experiments import designs_omars_ilp

        run, presolved = designs_omars_ilp._run_milp, []

        def failing_with_presolve(cost: np.ndarray, constraints: list, options: dict, **more: object):
            presolved.append(options.get("presolve", True))
            if options.get("presolve", True):
                return optimize.OptimizeResult(x=None, status=4, fun=None)
            return run(cost, constraints, options, **more)  # type: ignore[arg-type]

        monkeypatch.setattr(designs_omars_ilp, "_run_milp", failing_with_presolve)
        found = _found(UNIT, [UP], [_maximize()])
        assert found.proven
        assert found.x[0] == pytest.approx(1.0)
        assert presolved[:2] == [True, False]

    def test_a_programme_without_a_solution_is_not_solved_again(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from process_improve.experiments import designs_omars_ilp

        run, calls = designs_omars_ilp._run_milp, []

        def counted(cost: np.ndarray, constraints: list, options: dict, **more: object):
            calls.append(options)
            return run(cost, constraints, options, **more)  # type: ignore[arg-type]

        monkeypatch.setattr(designs_omars_ilp, "_run_milp", counted)
        out_of_reach = {"response": "y", "goal": "maximize", "low": 2.0, "high": 3.0}
        assert exact_optimum(UNIT, [UP], [out_of_reach], None, _judge([UP], [out_of_reach])) is None
        assert len(calls) == 1

    def test_a_set_whose_programme_is_not_solved_keeps_the_setting_it_began_with(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        solve = exact._Programme.solve

        def only_the_whole_programme(self: exact._Programme, used: np.ndarray | None = None, **more: object):
            return solve(self, used, **more) if used is None else None  # type: ignore[arg-type]

        monkeypatch.setattr(exact._Programme, "solve", only_the_whole_programme)
        found = _found(UNIT, [UP, DOWN], [_maximize("up"), _maximize("down")])
        assert 0.0 < found.x[0] < 1.0

    def test_time_that_runs_out_leaves_the_best_setting_unproven(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(exact, "_SECONDS", 0.0)
        monkeypatch.setattr(exact, "_AGREEMENT", 0.0)
        found = _found(UNIT, [UP, DOWN], [_maximize("up"), _maximize("down", weight=3.0)])
        assert not found.proven
        assert 0.0 < found.x[0] < 1.0

    def test_a_tangent_next_to_one_already_there_is_not_added(self) -> None:
        programme = exact._Programme(UNIT, [UP], [_maximize()], None)
        n_rows = len(programme.rows)
        assert not programme.add_tangents(np.array([1.0 - 1e-6]))
        assert programme.add_tangents(np.array([0.123]))
        assert len(programme.rows) == n_rows + 1


NAMES = list("abcdef")
RULES = ["a + b + c <= 0.6", f"2 <= count_nonzero({', '.join(NAMES)}) <= 4"]


def _blend_region() -> DesignRegion:
    return DesignRegion([Factor(name=n, type="mixture", high=0.5, min_when_used=0.1) for n in NAMES], RULES)


def _linear(name: str, coefficients: np.ndarray) -> dict:
    terms = [{"term": n, "coefficient": float(c)} for n, c in zip(NAMES, coefficients, strict=True)]
    return {"response_name": name, "factor_names": NAMES, "coefficients": terms}


def _best_of_every_set(desirability, region: DesignRegion, rng: np.random.Generator) -> float:
    """Return the largest desirability a search finds in each allowed set of components, the best of them all."""
    k, best = len(NAMES), 0.0
    in_first = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    for size in (2, 3, 4):
        for used in itertools.combinations(range(k), size):
            columns = list(used)
            constraints = [
                {"type": "eq", "fun": lambda z: z.sum() - 1.0},
                {"type": "ineq", "fun": lambda z, columns=columns: 0.6 - in_first[columns] @ z},
            ]

            def score(z: np.ndarray, columns: list[int] = columns) -> float:
                x = np.zeros(k)
                x[columns] = z
                return -desirability(x)

            for _ in range(3):
                start = 0.1 + (1 - 0.1 * size) * rng.dirichlet(np.ones(size))
                res = optimize.minimize(
                    score, start, method="SLSQP", bounds=[(0.1, 0.5)] * size, constraints=constraints
                )
                x = np.zeros(k)
                x[columns] = res.x
                if region.feasible(x[None, :], tol=1e-6)[0]:
                    best = max(best, -res.fun)
    return best


class TestOptimizeResponses:
    """``optimize_responses`` takes the exact route for straight models and says so in ``proven_optimal``."""

    @pytest.mark.slow
    @pytest.mark.parametrize("seed", [1, 2, 3, 4])
    def test_three_responses_over_a_blend_with_ingredients_left_out(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        coefficients = [rng.uniform(5, 50, len(NAMES)) for _ in range(3)]
        goals = [
            {"response": "y0", "goal": "minimize", "low": 10.0, "high": 45.0, "weight": 2.0},
            {"response": "y1", "goal": "maximize", "low": 10.0, "high": 45.0},
            {"response": "y2", "goal": "target", "low": 15.0, "target": 27.0, "high": 40.0},
        ]
        importances = [1.0, 2.0, 1.0]
        region = _blend_region()
        models = [_linear(f"y{i}", c) for i, c in enumerate(coefficients)]
        best = optimize_responses(models, goals=goals, region=region, response_importance=importances)["desirability"]
        assert best["proven_optimal"]
        assert best["within_region"]
        judge = _judge([(c, 0.0) for c in coefficients], goals, importances)
        searched = _best_of_every_set(judge, region, rng)
        assert best["composite_desirability"] >= searched - 1e-6  # no search of every set finds better
        assert best["composite_desirability"] == pytest.approx(searched, abs=2e-3)

    def test_amounts_under_a_limit_on_how_many_are_used(self) -> None:
        amounts = [Factor(name=n, low=0, high=10, min_when_used=2) for n in "ABCD"]
        gains = {"Intercept": 10.0, "A": 3.0, "B": 2.0, "C": 1.0, "D": 0.5}
        model = {
            "response_name": "y",
            "factor_names": list("ABCD"),
            "coefficients": [{"term": term, "coefficient": value} for term, value in gains.items()],
        }
        best = optimize_responses(
            [model],
            goals=[{"response": "y", "goal": "maximize", "low": 0.0, "high": 30.0}],
            region=DesignRegion(amounts, ["count_nonzero(A, B, C, D) <= 2"]),
        )["desirability"]
        assert best["proven_optimal"]
        assert best["optimal_coded"] == pytest.approx({"A": 1.0, "B": 1.0, "C": -1.0, "D": -1.0})

    def test_alternatives_are_searched_for_beside_the_proven_setting(self) -> None:
        amounts = [Factor(name=n, low=0, high=10, min_when_used=2) for n in "ABC"]
        gains = {"Intercept": 10.0, "A": 3.0, "B": 2.9, "C": 1.0}
        model = {
            "response_name": "y",
            "factor_names": list("ABC"),
            "coefficients": [{"term": term, "coefficient": value} for term, value in gains.items()],
        }
        best = optimize_responses(
            [model],
            goals=[{"response": "y", "goal": "maximize", "low": 0.0, "high": 30.0}],
            region=DesignRegion(amounts, ["count_nonzero(A, B, C) <= 1"]),
            n_alternatives=1,
        )["desirability"]
        assert best["proven_optimal"]
        assert best["optimal_coded"] == pytest.approx({"A": 1.0, "B": -1.0, "C": -1.0})
        assert [round(other["optimal_coded"]["B"], 6) for other in best["alternatives"]] == [
            1.0
        ]  # B alone, nearly as good

    def test_without_a_region_the_search_box_is_the_region(self) -> None:
        model = {
            "response_name": "y",
            "factor_names": ["A", "B"],
            "coefficients": [{"term": t, "coefficient": c} for t, c in {"Intercept": 5.0, "A": 2.0, "B": -1.0}.items()],
        }
        best = optimize_responses([model], goals=[{"response": "y", "goal": "maximize", "low": 0.0, "high": 20.0}])
        assert best["desirability"]["proven_optimal"]
        assert best["desirability"]["optimal_coded"] == pytest.approx({"A": 1.0, "B": -1.0})

    def test_a_factor_without_an_effect_stays_where_the_search_began(self) -> None:
        # Every B scores alike. The exact search would put B at an end of its range; the search leaves it in the
        # middle, where it began, and that is the setting reported, with the proof that nothing scores higher.
        model = {
            "response_name": "y",
            "factor_names": ["A", "B"],
            "coefficients": [{"term": t, "coefficient": c} for t, c in {"Intercept": 10.0, "A": 2.0, "B": 0.0}.items()],
        }
        best = optimize_responses([model], goals=[{"response": "y", "goal": "maximize", "low": 0.0, "high": 20.0}])
        assert best["desirability"]["proven_optimal"]
        assert best["desirability"]["optimal_coded"]["A"] == pytest.approx(1.0)
        assert -1.0 < best["desirability"]["optimal_coded"]["B"] < 1.0

    def test_a_model_with_an_interaction_is_searched_for_and_not_proven(self) -> None:
        amounts = [Factor(name=n, low=0, high=10, min_when_used=2) for n in "AB"]
        terms = {"Intercept": 10.0, "A": 2.0, "B": 3.0, "A:B": -6.0}
        model = {
            "response_name": "y",
            "factor_names": ["A", "B"],
            "coefficients": [{"term": term, "coefficient": value} for term, value in terms.items()],
        }
        best = optimize_responses(
            [model],
            goals=[{"response": "y", "goal": "maximize", "low": 0.0, "high": 30.0}],
            region=DesignRegion(amounts),
        )["desirability"]
        assert not best["proven_optimal"]
        assert best["optimal_coded"] == pytest.approx({"A": -1.0, "B": 1.0})  # 10 - 2 + 3 + 6: A is left out

    def test_a_desirability_of_one_is_proven_whatever_the_model(self) -> None:
        amounts = [Factor(name=n, low=0, high=10, min_when_used=2) for n in "AB"]
        terms = {"Intercept": 10.0, "A": 2.0, "B": 3.0, "A:B": -6.0}
        model = {
            "response_name": "y",
            "factor_names": ["A", "B"],
            "coefficients": [{"term": term, "coefficient": value} for term, value in terms.items()],
        }
        best = optimize_responses(
            [model],
            goals=[{"response": "y", "goal": "maximize", "low": 0.0, "high": 12.0}],
            region=DesignRegion(amounts),
        )["desirability"]
        assert best["composite_desirability"] == 1.0
        assert best["proven_optimal"]

    def test_a_counted_factor_without_a_minimum_leaves_nothing_to_prove(self) -> None:
        # B counts as used at any amount above 0. Were it to cost, "at least two" would be met by A and a trace of
        # B, a setting that is approached and never reached, so the exact search is not used with such a factor.
        amounts = [Factor(name="A", low=0, high=10, min_when_used=2), Factor(name="B", low=0, high=10)]
        model = {
            "response_name": "y",
            "factor_names": ["A", "B"],
            "coefficients": [{"term": t, "coefficient": c} for t, c in {"Intercept": 10.0, "A": 3.0, "B": 2.0}.items()],
        }
        best = optimize_responses(
            [model],
            goals=[{"response": "y", "goal": "maximize", "low": 0.0, "high": 30.0}],
            region=DesignRegion(amounts, ["count_nonzero(A, B) >= 2"]),
        )["desirability"]
        assert not best["proven_optimal"]
        assert best["within_region"]
        assert best["optimal_coded"] == pytest.approx({"A": 1.0, "B": 1.0})
