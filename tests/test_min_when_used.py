"""A factor that is 0 or at least its ``min_when_used``: an ingredient left out, or used at no less than a minimum."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import DesignRegion, Factor, evaluate_design, generate_design
from process_improve.experiments.designs_constrained import build_candidates, region_with_rules
from process_improve.experiments.factor import DesignResult, as_constraints
from process_improve.experiments.optimization import _search_space

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

    def test_sampling_and_searching_are_refused(self) -> None:
        region = DesignRegion(FACTORS)
        rng = np.random.default_rng(1)
        with pytest.raises(ValueError, match="cannot be sampled"):
            region.sample(5, rng)
        with pytest.raises(ValueError, match="cannot be searched"):
            _search_space(["A", "B", "C"], None, region, rng, 3)

    def test_a_design_is_judged_on_its_determinant(self) -> None:
        result = generate_design(FACTORS, budget=8, model_type="main_effects", random_state=1)
        assert evaluate_design(result, model="main_effects", metric="d_efficiency")["d_efficiency"] > 50
