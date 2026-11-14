"""A factor that is 0 or at least its ``min_when_used``: an ingredient left out, or used at no less than a minimum."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from scipy import optimize
from scipy.optimize import Bounds, LinearConstraint, milp

from process_improve.experiments import (
    DesignRegion,
    Factor,
    evaluate_design,
    generate_design,
    optimize_responses,
)
from process_improve.experiments import _uniform_sampling as sampling
from process_improve.experiments._uniform_sampling import UniformSampler
from process_improve.experiments.designs_constrained import build_candidates, region_with_rules
from process_improve.experiments.factor import DesignResult, as_constraints
from process_improve.experiments.optimization import _best_set_of_factors, _search_space, _SearchSpace

A = Factor(name="A", low=0, high=50, units="ppm", min_when_used=5)
B = Factor(name="B", low=0, high=50, units="ppm", min_when_used=30)  # the minimum is above the middle of the range
C = Factor(name="C", low=10, high=20)
FACTORS = [A, B, C]


def _traces(result: DesignResult, factors: list[Factor]) -> int:
    """Count the settings between 0 and a factor's minimum, over every run of a design."""
    count = 0
    for f in factors:
        if f.min_when_used is not None:
            amount = result.design_actual[f.name].to_numpy(dtype=float)
            count += int(((amount > 1e-9) & (amount < f.min_when_used - 1e-9)).sum())
    return count


class TestFactor:
    def test_the_minimum_is_kept(self) -> None:
        assert A.min_when_used == 5
        assert C.min_when_used is None

    @pytest.mark.parametrize(
        ("spec", "message"),
        [
            ({"low": 1, "high": 50, "min_when_used": 5}, "needs low=0"),
            ({"low": 0, "high": 50, "min_when_used": 50}, "must lie between 0 and high"),
            ({"low": 0, "high": 50, "min_when_used": 0}, "must lie between 0 and high"),
            ({"low": 0, "high": 50, "min_when_used": -1}, "must lie between 0 and high"),
            ({"low": 0, "high": 50, "min_when_used": float("nan")}, "must lie between 0 and high"),
            ({"type": "mixture", "min_when_used": 0.1}, "is for a continuous factor"),
            ({"type": "categorical", "levels": ["a", "b"], "min_when_used": 1}, "is for a continuous factor"),
        ],
    )
    def test_a_minimum_that_cannot_be_used_is_refused(self, spec: dict, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            Factor(name="X", **spec)


class TestCandidates:
    def test_no_candidate_sets_a_factor_between_0_and_its_minimum(self) -> None:
        region = region_with_rules(FACTORS, as_constraints(["A + B <= 60"]) or [])
        coded, cats, _ = build_candidates(region, model_type="quadratic")
        amounts = region.actual(coded)
        assert not np.any((amounts["A"] > 1e-9) & (amounts["A"] < 5 - 1e-9))
        assert not np.any((amounts["B"] > 1e-9) & (amounts["B"] < 30 - 1e-9))
        assert np.all(region.slack(coded, cats) <= 1e-9)

    def test_a_grid_level_below_the_minimum_moves_up_to_it(self) -> None:
        region = region_with_rules(FACTORS, [])
        coded, _cats, _ = build_candidates(region, n_levels=3, model_type="quadratic")
        assert sorted(set(region.actual(coded)["A"].round(9))) == [0.0, 25.0, 50.0]
        assert sorted(set(region.actual(coded)["B"].round(9))) == [0.0, 30.0, 50.0]  # 25 is below B's minimum

    def test_the_slack_is_the_distance_to_the_nearer_end_of_the_gap(self) -> None:
        region = region_with_rules([A], [])
        coded = np.array([[-1.0], [-0.95], [-0.85], [-0.8], [0.0]])  # 0, 1.25, 3.75, 5 and 25 ppm
        np.testing.assert_allclose(region.slack(coded), [0.0, 0.05, 0.05, 0.0, -0.8], atol=1e-12)

    def test_a_factor_without_a_minimum_has_no_gap(self) -> None:
        region = region_with_rules([C], [])
        assert region.gaps() == []
        coded = np.array([[-0.9]])
        assert region.out_of_gaps(coded) is coded


class TestDesigns:
    @pytest.mark.parametrize(("model_type", "budget"), [("main_effects", 8), ("interactions", 10), ("quadratic", 14)])
    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_an_optimal_design_has_no_traces(self, model_type: str, budget: int, seed: int) -> None:
        result = generate_design(FACTORS, budget=budget, model_type=model_type, random_state=seed)
        assert result.design_type == "d_optimal"  # chosen because a factor has a minimum
        assert _traces(result, FACTORS) == 0
        assert result.metadata["region"]["factors"][0]["min_when_used"] == 5

    def test_a_quadratic_design_uses_the_minimum_as_a_level(self) -> None:
        result = generate_design(FACTORS, budget=14, model_type="quadratic", random_state=1)
        assert sorted(set(result.design_actual["B"].round(9))) == [0.0, 30.0, 50.0]

    def test_a_linear_limit_is_kept_too(self) -> None:
        result = generate_design(
            FACTORS, budget=14, model_type="quadratic", constraints=["A + B <= 60"], random_state=2
        )
        assert _traces(result, FACTORS) == 0
        assert (result.design_actual["A"] + result.design_actual["B"]).max() <= 60 + 1e-9

    def test_the_lattice_a_design_is_polished_on_skips_the_gap(self) -> None:
        factors = [Factor(name=f"X{i}", low=0, high=50, min_when_used=20) for i in range(6)]
        result = generate_design(factors, budget=32, model_type="quadratic", random_state=3)
        assert result.metadata["polish_levels"] == 5  # whose second level, 12.5, is below the minimum
        assert _traces(result, factors) == 0

    def test_candidates_between_0_and_the_minimum_are_dropped(self) -> None:
        candidates = pd.DataFrame(
            {
                "A": [0, 2, 5, 50, 0, 50, 25, 3],
                "B": [0, 0, 30, 50, 50, 0, 40, 10],
                "C": [10, 20, 10, 20, 20, 10, 15, 15],
            }
        )
        result = generate_design(
            FACTORS, design_type="d_optimal", budget=6, model_type="main_effects", candidates=candidates, random_state=1
        )
        assert result.metadata["n_candidates"] == 6
        assert _traces(result, FACTORS) == 0

    def test_fixed_runs_are_kept_as_given(self, caplog: pytest.LogCaptureFixture) -> None:
        fixed = pd.DataFrame({"A": [-0.9], "B": [-1.0], "C": [0.0]})  # A at 2.5 ppm, a trace
        with caplog.at_level(logging.WARNING):
            result = generate_design(FACTORS, budget=8, model_type="main_effects", fixed_runs=fixed, random_state=1)
        assert result.metadata["n_fixed_runs_outside_region"] == 1
        assert _traces(result, FACTORS) == 1

    @pytest.mark.parametrize("design_type", ["ccd", "box_behnken", "full_factorial"])
    def test_a_classical_design_with_a_trace_is_refused(self, design_type: str) -> None:
        with pytest.raises(ValueError, match=r"sets B \(min_when_used=30\) between 0 and its minimum"):
            generate_design(FACTORS, design_type=design_type)

    def test_a_classical_design_without_a_trace_is_built(self) -> None:
        result = generate_design(FACTORS, design_type="full_factorial", n_center_points=0)
        assert _traces(result, FACTORS) == 0
        assert len(result.design_actual) == 8

    @pytest.mark.slow
    def test_ingredients_in_buckets_are_left_out_or_used_at_their_minimum(self) -> None:
        names = [[f"B{b + 1}_{i + 1}" for i in range(3)] for b in range(5)]
        flat = [name for bucket in names for name in bucket]
        factors = [Factor(name=name, low=0, high=20, units="%", min_when_used=5) for name in flat]
        limits = [f"5 <= {' + '.join(bucket)} <= 20" for bucket in names] + [f"{' + '.join(flat)} <= 60"]
        result = generate_design(factors, budget=30, model_type="main_effects", constraints=limits, random_state=1)
        assert _traces(result, factors) == 0
        amounts = result.design_actual[flat].to_numpy(dtype=float)
        assert amounts.reshape(30, 5, 3).sum(axis=2).min() >= 5 - 1e-6  # every bucket is represented


class TestRegion:
    def test_a_trace_is_outside_the_region(self) -> None:
        region = DesignRegion(FACTORS, ["A + B <= 60"])
        coded = np.array(
            [[-1, -1, 0], [-0.9, -1, 0], [-0.8, -1, 0], [0, 0.1, 0], [0, 0.2, 0], [1, 1, 0]], dtype=float
        )  # A, B in ppm: (0, 0), (2.5, 0), (5, 0), (25, 27.5), (25, 30), (50, 50)
        np.testing.assert_array_equal(region.feasible(coded), [True, False, True, False, True, False])
        assert region.broken_limits(coded) == [[], ["A = 0 or A >= 5"], [], ["B = 0 or B >= 30"], [], ["A + B <= 60"]]

    def test_the_region_is_written_and_read_with_its_minimums(self) -> None:
        region = DesignRegion(FACTORS, ["A + B <= 60"])
        spec = region.to_dict()
        assert [f.get("min_when_used") for f in spec["factors"]] == [5.0, 30.0, None]
        assert "min_when_used" not in spec["factors"][2]  # a factor without one is written as before
        again = DesignRegion.from_dict(spec)
        coded = np.array([[-0.9, -1.0, 0.0], [-0.8, -1.0, 0.0]])
        np.testing.assert_array_equal(again.feasible(coded), [False, True])

    def test_a_design_is_judged_on_its_determinant(self) -> None:
        result = generate_design(FACTORS, budget=8, model_type="main_effects", random_state=1)
        assert evaluate_design(result, model="main_effects", metric="d_efficiency")["d_efficiency"] > 50


def _shares(points: np.ndarray) -> tuple[float, float, float]:
    """Return the share of points that leave A out, that leave B out, and that use both."""
    out_a, out_b = points[:, 0] <= -1 + 1e-9, points[:, 1] <= -1 + 1e-9
    return float(out_a.mean()), float(out_b.mean()), float((~out_a & ~out_b).mean())


class TestSampling:
    """A factor with a minimum is left out as often as it is used, before the constraints act."""

    def test_without_constraints_each_factor_is_left_out_half_the_time(self) -> None:
        region = DesignRegion(FACTORS)
        points = region.sample(20_000, np.random.default_rng(1))
        assert region.feasible(points).all()
        assert _shares(points) == pytest.approx((0.5, 0.5, 0.25), abs=0.02)
        used = (points[points[:, 0] > -1 + 1e-9, 0] + 1.0) * 25.0
        assert used.min() >= 5 - 1e-9
        assert used.mean() == pytest.approx(27.5, abs=0.5)  # uniform from 5 to 50

    def test_a_constraint_conditions_the_shares(self) -> None:
        # Of the four sets of factors in use, A + B <= 60 cuts only the one with both, to a third of it:
        # the sets weigh 1/4, 1/4, 1/4 and 1/12, so A is left out in (1/4 + 1/4) / (10/12) of the points.
        region = DesignRegion(FACTORS, ["A + B <= 60"])
        points = region.sample(20_000, np.random.default_rng(1))
        assert region.feasible(points).all()
        assert _shares(points) == pytest.approx((0.6, 0.6, 0.1), abs=0.02)

    def test_the_chains_keep_those_shares(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sampling, "_REJECTION_DRAWS", 10)  # a handful of points to start 1000 chains from
        region = DesignRegion(FACTORS, ["A + B <= 60"])
        points = region.sample(20_000, np.random.default_rng(1))
        assert region.feasible(points).all()
        assert _shares(points) == pytest.approx((0.6, 0.6, 0.1), abs=0.04)

    def test_the_chains_agree_with_rejection_under_a_curved_constraint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        region = DesignRegion(FACTORS, ["A * B <= 900"])
        by_rejection = _shares(region.sample(20_000, np.random.default_rng(1)))
        monkeypatch.setattr(sampling, "_REJECTION_DRAWS", 10)
        points = region.sample(20_000, np.random.default_rng(2))
        assert region.feasible(points).all()
        assert _shares(points) == pytest.approx(by_rejection, abs=0.04)

    def test_chains_started_from_seeds_reach_every_set(self) -> None:
        # No proposal is inside the region, so the chains start from the two seeds, which use A alone.
        region = DesignRegion([A, B], ["A + B <= 60"])
        sampler = UniformSampler(
            region.inequalities,
            (np.full(2, -1.0), np.full(2, 1.0)),
            lambda m: np.full((m, 2), 2.0),
            seeds=lambda: np.array([[-0.8, -1.0], [1.0, -1.0]]),
            tolerances=region._widen(),
            gaps=region.gaps,
        )
        points = sampler.draw(20_000, np.random.default_rng(4))
        assert region.feasible(points).all()
        assert _shares(points) == pytest.approx((0.6, 0.6, 0.1), abs=0.04)

    def test_a_thin_piece_is_sampled_from_the_boundary_points(self) -> None:
        # Only the set with both in use has A + B between 59.5 and 60: B alone reaches 50.
        region = DesignRegion([A, B], ["59.5 <= A + B <= 60"])
        points = region.sample(300, np.random.default_rng(5))
        assert region.feasible(points).all()
        assert _shares(points)[2] == 1.0

    def test_points_and_levels_are_drawn_together_under_rules(self) -> None:
        catalyst = Factor(name="catalyst", type="categorical", levels=["X", "Y"])
        region = DesignRegion([A, B, catalyst], ["not (catalyst == 'Y' and A > 20)"])
        frame = region.sample_frame(2000, np.random.default_rng(6))
        assert region.feasible(frame[["A", "B"]].to_numpy(dtype=float), labels=frame).all()
        assert (frame["A"] <= -1 + 1e-9).mean() == pytest.approx(0.5, abs=0.1)


class TestCriteria:
    def test_an_i_optimal_design_has_no_traces(self) -> None:
        result = generate_design(
            FACTORS,
            design_type="i_optimal",
            budget=14,
            model_type="quadratic",
            constraints=["A + B <= 60"],
            random_state=1,
        )
        assert _traces(result, FACTORS) == 0

    def test_a_design_is_judged_over_the_region_with_factors_left_out(self) -> None:
        result = generate_design(
            FACTORS, budget=14, model_type="quadratic", constraints=["A + B <= 60"], random_state=2
        )
        found = evaluate_design(result, model="quadratic", metric=["g_efficiency", "average_prediction_variance"])
        assert 0 < found["g_efficiency"] <= 100
        assert found["average_prediction_variance"] > 0


def _model(coefficients: dict[str, float], names: list[str]) -> dict:
    """Return a fitted model as ``optimize_responses`` takes it."""
    return {
        "response_name": "y",
        "factor_names": names,
        "coefficients": [{"term": term, "coefficient": value} for term, value in coefficients.items()],
    }


RANGES = {f.name: {"low": f.low, "high": f.high} for f in FACTORS}
GOALS = [{"response": "y", "goal": "maximize", "low": 0.0, "high": 30.0}]


class TestSearch:
    """The optimum leaves a factor out or uses it at no less than its minimum."""

    def test_the_optimum_leaves_out_the_factor_that_does_not_pay(self) -> None:
        # Coded: y = 10 + 2 A + 3 B + 0.5 C - 6 A B. B alone gives 17.5, A alone 15.5, both 9.5.
        model = _model({"Intercept": 10.0, "A": 2.0, "B": 3.0, "C": 0.5, "A:B": -6.0}, ["A", "B", "C"])
        for constraints in ([], ["A + B <= 60"]):
            found = optimize_responses(
                [model],
                goals=GOALS,
                method="desirability",
                region=DesignRegion(FACTORS, constraints),
                factor_ranges=RANGES,
            )["desirability"]
            assert found["optimal_actual"] == pytest.approx({"A": 0.0, "B": 50.0, "C": 20.0}, abs=1e-4)
            assert found["predicted_responses"]["y"] == pytest.approx(17.5, abs=1e-4)
            assert found["within_region"] is True

    def test_a_factor_in_use_stays_at_or_above_its_minimum(self) -> None:
        # y falls with A, yet rises by 4 when A is in use at all: A sits at its minimum of 5 ppm, coded -0.8.
        model = _model({"Intercept": 10.0, "A": -5.0, "B": -1.0, "C": 0.5, "I(A ** 2)": 9.0}, ["A", "B", "C"])
        found = optimize_responses(
            [model], goals=GOALS, method="desirability", region=DesignRegion(FACTORS), factor_ranges=RANGES
        )["desirability"]
        amount = found["optimal_actual"]["A"]
        assert amount == pytest.approx(0.0, abs=1e-6) or amount >= 5 - 1e-6
        assert found["optimal_actual"]["B"] == pytest.approx(0.0, abs=1e-6)

    def test_a_search_keeps_the_set_of_factors_its_start_uses(self) -> None:
        space = _SearchSpace([(-1.0, 1.0)] * 3, [], [], [(0, -0.8), (1, 0.2)])
        assert space.bounds_at(np.array([-1.0, 0.5, 0.0])) == [(-1.0, -1.0), (0.2, 1.0), (-1.0, 1.0)]
        assert space.bounds_at(np.array([0.3, -1.0, 0.0])) == [(-0.8, 1.0), (-1.0, -1.0), (-1.0, 1.0)]

    def test_a_search_box_that_ends_below_a_minimum_leaves_the_factor_out(self) -> None:
        space = _SearchSpace([(-1.0, 0.0), (-1.0, 1.0)], [], [], [(0, 0.2)])
        assert space.bounds_at(np.array([-0.5, 0.0])) == [(-1.0, -1.0), (-1.0, 1.0)]

    def test_the_search_box_does_not_reach_below_0(self) -> None:
        space = _search_space(["A", "B", "C"], (-1.5, 1.5), DesignRegion(FACTORS), np.random.default_rng(1), 6)
        assert space.bounds[:2] == [(-1.0, 1.5), (-1.0, 1.5)]
        assert space.bounds[2] == (-1.5, 1.5)
        assert space.gaps == [(0, pytest.approx(-0.8)), (1, pytest.approx(0.2))]
        assert all(start[0] <= -1 + 1e-9 or start[0] >= -0.8 - 1e-9 for start in space.starts)

    def test_one_factor_is_exchanged_for_another_under_a_limit(self) -> None:
        # At most one of A and B fits under A + B <= 40 once B is at its minimum of 30: from A alone, bringing B
        # in breaks the limit and leaving A out scores less, so only the exchange reaches B alone.
        region = DesignRegion([A, B], ["A + B <= 40", "A + B >= 5"])
        space = _SearchSpace(
            [(-1.0, 1.0)] * 2,
            [{"type": "ineq", "fun": lambda x, g=g: -float(g(x[None, :])[0])} for g in region.inequalities],
            [],
            region.gaps,
        )

        def objective(x: np.ndarray) -> float:
            return float(-(x[0] + 3.0 * x[1]))

        start = np.array([0.6, -1.0])  # A alone at its largest, 40 ppm
        best = optimize.minimize(
            objective, start, method="SLSQP", bounds=space.bounds_at(start), constraints=space.constraints
        )
        assert best.x == pytest.approx([0.6, -1.0], abs=1e-6)
        found = _best_set_of_factors(
            objective, space, lambda x: bool(region.feasible(x[None, :], tol=1e-6)[0]), best, None
        )
        assert found.x == pytest.approx([-1.0, 0.6], abs=1e-6)  # B alone at 40 ppm

    def test_a_pareto_front_has_no_traces(self) -> None:
        names = ["A", "B", "C"]
        models = [
            _model({"Intercept": 10.0, "A": 2.0, "B": 3.0, "C": 0.5}, names),
            {**_model({"Intercept": 5.0, "A": 1.0, "B": 2.0, "C": -0.5}, names), "response_name": "cost"},
        ]
        goals = [
            {"response": "y", "goal": "maximize", "low": 0.0, "high": 30.0},
            {"response": "cost", "goal": "minimize", "low": 0.0, "high": 10.0},
        ]
        region = DesignRegion(FACTORS)
        found = optimize_responses(models, goals=goals, method="pareto_front", region=region, n_pareto_points=5)
        front = found["pareto_front"]["front"]
        assert len(front) >= 2
        coded = np.array([[point["coded"][name] for name in names] for point in front])
        assert region.feasible(coded, tol=1e-6).all()

    @pytest.mark.slow
    def test_the_optimum_over_buckets_matches_a_mixed_integer_programme(self) -> None:
        names = [[f"B{b + 1}_{i + 1}" for i in range(3)] for b in range(5)]
        flat = [name for bucket in names for name in bucket]
        factors = [Factor(name=name, low=0, high=20, units="%", min_when_used=5) for name in flat]
        limits = [f"5 <= {' + '.join(bucket)} <= 20" for bucket in names] + [f"{' + '.join(flat)} <= 60"]
        region = DesignRegion(factors, limits)
        slope = np.random.default_rng(7).normal(size=15).round(2)  # of the coded amounts
        model = _model({"Intercept": 0.0} | dict(zip(flat, slope.tolist(), strict=True)), flat)
        goals = [{"response": "y", "goal": "maximize", "low": -40.0, "high": 40.0}]
        found = optimize_responses(
            [model],
            goals=goals,
            method="desirability",
            region=region,
            factor_ranges={name: {"low": 0, "high": 20} for name in flat},
        )["desirability"]
        amounts = np.array([found["optimal_actual"][name] for name in flat])
        assert not np.any((amounts > 1e-6) & (amounts < 5 - 1e-6))

        # The same optimum, exactly: amounts a from 5 z to 20 z with z binary, and y = slope @ (a / 10 - 1).
        cost = np.concatenate([-slope / 10.0, np.zeros(15)])
        bucket = np.kron(np.eye(5), np.ones(3))
        rows = np.vstack(
            [
                np.hstack([bucket, np.zeros((5, 15))]),
                np.hstack([np.ones((1, 15)), np.zeros((1, 15))]),
                np.hstack([np.eye(15), -20.0 * np.eye(15)]),
                np.hstack([-np.eye(15), 5.0 * np.eye(15)]),
            ]
        )
        lower = np.concatenate([np.full(5, 5.0), np.full(31, -np.inf)])
        upper = np.concatenate([np.full(5, 20.0), [60.0], np.zeros(30)])
        exact = milp(
            cost,
            constraints=LinearConstraint(rows, lower, upper),
            integrality=np.concatenate([np.zeros(15), np.ones(15)]),
            bounds=Bounds(np.zeros(30), np.concatenate([np.full(15, 20.0), np.ones(15)])),
        )
        assert found["predicted_responses"]["y"] == pytest.approx(-exact.fun - slope.sum(), abs=1e-3)
