"""Regression tests for the bugs found reviewing the constrained-DOE work (#624, #632-#635).

Each test reproduces a reviewer's failure scenario and checks the corrected behaviour.
"""

from __future__ import annotations

import logging
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import Constraint, Factor, designs_optimal, evaluate_design, generate_design
from process_improve.experiments.optimization import optimize_responses
from process_improve.experiments.region import DesignRegion


def _continuous(k: int, low: float = 0, high: float = 1) -> list[Factor]:
    return [Factor(name=f"X{i + 1}", low=low, high=high) for i in range(k)]


@pytest.fixture
def no_pyoptex(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the built-in candidate exchange, as on a process-improve[all] install."""
    monkeypatch.setattr(designs_optimal, "_PYOPTEX_AVAILABLE", False)


class TestAutoSelectFallsBackWhenNoSupersaturatedDesignExists:
    @pytest.mark.parametrize(
        ("k", "budget"), [(3, 2), (3, 3), (4, 3), (5, 3), (5, 5), (6, 5), (7, 5), (8, 7), (10, 9), (10, 7), (12, 8)]
    )
    def test_previously_working_budgets_still_return_a_design(self, k: int, budget: int) -> None:
        """No supersaturated design exists for these (odd, too small, or aliased) budgets: fall back as before."""
        result = generate_design(_continuous(k), budget=budget)
        assert result.design_type in {"d_optimal", "plackett_burman"}

    @pytest.mark.parametrize(("k", "budget"), [(10, 6), (8, 6), (22, 12)])
    def test_supersaturated_is_chosen_when_an_unaliased_one_exists(self, k: int, budget: int) -> None:
        result = generate_design(_continuous(k), budget=budget)
        assert result.design_type == "supersaturated"
        assert result.metadata["n_fully_aliased_pairs"] == 0


@pytest.mark.usefixtures("no_pyoptex")
class TestFixedRunsThatCannotSupportTheModel:
    @pytest.mark.parametrize("design_type", ["d_optimal", "i_optimal", "a_optimal", "e_optimal"])
    def test_budget_rises_to_cover_the_missing_rank(self, design_type: str, caplog: pytest.LogCaptureFixture) -> None:
        """Three identical centre runs span 1 of 6 quadratic coefficients: 5 free runs are needed, not 3."""
        centre = pd.DataFrame({"X1": [0.0] * 3, "X2": [0.0] * 3})
        with caplog.at_level(logging.WARNING):
            result = generate_design(
                _continuous(2), design_type=design_type, model_type="quadratic", budget=6, fixed_runs=centre
            )
        assert result.n_runs == 8
        assert "span only 1 of the 6 model coefficients" in caplog.text
        x = result.design[["X1", "X2"]].to_numpy(dtype=float)
        model = np.column_stack([np.ones(len(x)), x, x[:, 0] * x[:, 1], x**2])
        assert np.linalg.matrix_rank(model) == 6


@pytest.mark.usefixtures("no_pyoptex")
def test_fixed_runs_stay_first_in_the_run_sheet() -> None:
    """Runs already performed are not shuffled into the new runs; the new runs are still randomised."""
    fixed = pd.DataFrame({"X1": [0.0, 1.0], "X2": [0.0, -1.0], "X3": [0.0, 1.0]})
    result = generate_design(_continuous(3, 0, 10), design_type="d_optimal", budget=10, fixed_runs=fixed)
    first_two = result.design[["X1", "X2", "X3"]].iloc[:2].to_numpy(dtype=float)
    np.testing.assert_array_equal(first_two, fixed.to_numpy())
    assert result.run_order[:2] == [1, 2]
    assert sorted(result.run_order[2:]) == list(range(3, 11))


class TestScheffeAnalysisTestsMeaningfulHypotheses:
    """A Scheffe model has no intercept, so 'beta_i = 0' and 'effect = 2 x coefficient' mean nothing."""

    @pytest.fixture
    def fitted(self) -> tuple[pd.DataFrame, pd.Series]:
        """Simplex-centroid design, replicated, with x1:x2 synergy and no other non-linear blending."""
        rng = np.random.default_rng(7)
        points = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5), (1 / 3, 1 / 3, 1 / 3)]
        x = pd.DataFrame(points * 2, columns=["x1", "x2", "x3"])
        y = 10 * x.x1 + 5 * x.x2 + 2 * x.x3 + 8 * x.x1 * x.x2 + rng.normal(0, 0.2, len(x))
        return x, pd.Series(y, name="y")

    def _analyse(self, fitted: tuple[pd.DataFrame, pd.Series], kind: str) -> dict:
        from process_improve.experiments.analysis import analyze_experiment

        x, y = fitted
        return analyze_experiment(x, y, model="scheffe_quadratic", analysis_type=kind)  # results merge flat

    def test_anova_tests_the_linear_block_jointly(self, fitted: tuple[pd.DataFrame, pd.Series]) -> None:
        rows = {r["source"]: r for r in self._analyse(fitted, "anova")["anova_table"]}
        assert set(rows) == {"Linear mixture", "x1:x2", "x1:x3", "x2:x3", "Residual"}
        assert rows["Linear mixture"]["df"] == 2
        assert rows["Linear mixture"]["p_value"] < 0.001
        model_ss = sum(r["sum_sq"] for name, r in rows.items() if name != "Residual")
        assert model_ss < ((fitted[1] - fitted[1].mean()) ** 2).sum()  # no SS beyond the corrected total

    def test_significance_lists_only_blending_terms(self, fitted: tuple[pd.DataFrame, pd.Series]) -> None:
        result = self._analyse(fitted, "significance")
        assert result["significant_terms"] == ["x1:x2"]
        assert set(result["not_significant_terms"]) == {"x1:x3", "x2:x3"}
        assert result["linear_blending_differs"] is True

    def test_effects_are_in_the_cox_direction(self, fitted: tuple[pd.DataFrame, pd.Series]) -> None:
        result = self._analyse(fitted, "effects")
        assert result["effect_direction"] == "cox"
        assert sum(result["effects"].values()) == pytest.approx(0.0, abs=1e-9)  # q/(q-1) times a centred vector
        assert max(result["effects"], key=result["effects"].get) == "x1"

    def test_lenth_is_refused_with_a_reason(self, fitted: tuple[pd.DataFrame, pd.Series]) -> None:
        result = self._analyse(fitted, "lenth_method")
        assert result["lenth_method"] is None
        assert "anova" in result["note"]


@pytest.mark.usefixtures("no_pyoptex")
class TestThinRegionsAreSampled:
    """Regions far below 1% of the box, which generate_design accepts, must also evaluate and sample."""

    def test_evaluate_a_thin_box_region(self) -> None:
        factors = [Factor(name=n, low=0, high=10) for n in "ABC"]  # 'A + B + C <= 3' is 0.45% of the box
        result = generate_design(
            factors, design_type="d_optimal", budget=10, constraints=[Constraint(expression="A + B + C <= 3")]
        )
        assert evaluate_design(result, metric="i_efficiency")["i_efficiency"] > 0

    def test_six_factor_region_of_65_parts_per_million(self) -> None:
        factors = [Factor(name=f"A{i}", low=0, high=1) for i in range(6)]
        constraint = Constraint(expression=" + ".join(f"A{i}" for i in range(6)) + " <= 0.6")
        result = generate_design(factors, design_type="i_optimal", model_type="main_effects", constraints=[constraint])
        metrics = evaluate_design(result, model="main_effects", metric=["i_efficiency", "g_efficiency"])
        assert metrics["g_efficiency"] > 0
        assert generate_design(factors, design_type="maximin", budget=12, constraints=[constraint]).n_runs == 12

    def test_thin_mixture_region(self) -> None:
        factors = [Factor(name=n, type="mixture", low=0, high=1) for n in "ABC"]  # 0.24% of the simplex
        cap = Constraint(expression="A + B <= 0.05")
        result = generate_design(factors, design_type="i_optimal", model_type="scheffe_linear", constraints=[cap])
        assert evaluate_design(result, model="scheffe_linear", metric="i_efficiency")["i_efficiency"] > 0

    @pytest.mark.parametrize("seed", [0, 1])
    def test_sample_is_uniform_where_the_answer_is_known(self, seed: int) -> None:
        """A + ... + A5 <= 0.6 on [0, 1]^6 is a simplex corner: every coordinate has mean 0.6 / 7."""
        region = DesignRegion(
            [Factor(name=f"A{i}", low=0, high=1) for i in range(6)],
            [Constraint(expression=" + ".join(f"A{i}" for i in range(6)) + " <= 0.6")],
        )
        points = region.sample(20_000, np.random.default_rng(seed))
        assert region.feasible(points).all()
        assert (0.5 + 0.5 * points).mean() == pytest.approx(0.6 / 7, abs=0.002)

    def test_non_convex_region(self) -> None:
        """Shrinkage on the chord handles a thin ring, which affine chords cannot describe."""
        region = DesignRegion(
            [Factor(name=n, low=-1, high=1) for n in "AB"],
            [Constraint(expression="A**2 + B**2 <= 1"), Constraint(expression="A**2 + B**2 >= 0.99")],
        )
        points = region.sample(20_000, np.random.default_rng(0))
        assert region.feasible(points).all()
        assert np.abs(points.mean(axis=0)).max() < 0.1  # spread round the ring, not stuck where it started

    def test_empty_region_is_reported(self) -> None:
        region = DesignRegion([Factor(name=n, low=0, high=1) for n in "AB"], [Constraint(expression="A + B >= 3")])
        with pytest.raises(ValueError, match="No point inside the region"):
            region.sample(100, np.random.default_rng(0))


class TestSequencesInThinRegionsStopEarly:
    def test_sobol_raises_instead_of_exhausting_memory(self) -> None:
        """'A + B >= 19.99' keeps 5 in 10 million of the box: the sequence stops at 2**22 points."""
        from process_improve.experiments.designs_space_filling import space_filling_design

        factors = [Factor(name=n, low=0, high=10) for n in "AB"]
        thin = [Constraint(expression="A + B >= 19.99")]
        with pytest.raises(ValueError, match="method='maximin'"):
            space_filling_design(factors, 200, "sobol", thin, random_state=0)
        points, _meta = space_filling_design(factors, 20, "maximin", thin, random_state=0)
        assert DesignRegion(factors, thin).feasible(points).all()


class TestBoxRegionHonoursWiderSearchBounds:
    """A box region's constraints apply on top of search_bounds; its own [-1, 1] cube is not re-imposed."""

    REGION = DesignRegion(
        [Factor(name="T", low=100, high=150), Factor(name="D", low=20, high=60)], [Constraint(expression="D <= 50")]
    )
    # y = 60 + 8T + 6D - 2T^2 - 3D^2 (coded): T peaks at 2, the edge of the search box; 'D <= 50' is coded D <= 0.5.
    MODEL: ClassVar[dict] = {
        "response_name": "y",
        "factor_names": ["T", "D"],
        "coefficients": [
            {"term": "Intercept", "coefficient": 60.0},
            {"term": "T", "coefficient": 8.0},
            {"term": "D", "coefficient": 6.0},
            {"term": "I(T ** 2)", "coefficient": -2.0},
            {"term": "I(D ** 2)", "coefficient": -3.0},
        ],
    }
    MAXIMISE: ClassVar[list] = [{"response": "y", "goal": "maximize", "low": 40, "high": 75}]

    def test_desirability_finds_an_optimum_outside_the_coded_cube(self) -> None:
        out = optimize_responses([self.MODEL], self.MAXIMISE, search_bounds=(-2, 2), region=self.REGION)
        optimum = out["desirability"]
        assert optimum["optimal_coded"]["T"] == pytest.approx(2.0, abs=1e-3)
        assert optimum["optimal_coded"]["D"] == pytest.approx(0.5, abs=1e-3)
        assert optimum["within_region"] is True

    def test_pareto_front_spans_the_search_box(self) -> None:
        cost = {**self.MODEL, "response_name": "cost", "coefficients": [{"term": "T", "coefficient": 1.0}]}
        goals = [*self.MAXIMISE, {"response": "cost", "goal": "minimize"}]
        out = optimize_responses(
            [self.MODEL, cost], goals, method="pareto_front", search_bounds=(-2, 2), region=self.REGION
        )
        t_values = [point["coded"]["T"] for point in out["pareto_front"]["front"]]
        assert min(t_values) < -1.5
        assert max(t_values) > 1.5


@pytest.mark.usefixtures("no_pyoptex")
class TestEOptimalExchangeDoesNotStopEarly:
    @pytest.mark.slow
    @pytest.mark.parametrize("seed", [42, 2, 3])
    def test_five_factors_in_eight_runs_reach_the_regular_fraction(self, seed: int) -> None:
        """The 2^(5-2) fraction has lambda_min = 8; seeds used to stop at 4.67, 4.0 and 5.65."""
        result = generate_design(
            _continuous(5, -1, 1), design_type="e_optimal", budget=8, model_type="main_effects", random_seed=seed
        )
        assert result.metadata["min_eigenvalue"] == pytest.approx(8.0)

    @pytest.mark.slow
    def test_a_repeated_smallest_eigenvalue_is_split_and_raised(self) -> None:
        """Four factors, interactions, 12 runs stalled at lambda_min = 4 with multiplicity 2-3."""
        result = generate_design(_continuous(4, -1, 1), design_type="e_optimal", budget=12, model_type="interactions")
        assert result.metadata["min_eigenvalue"] == pytest.approx(8.0)

    def test_phi_p_separates_designs_that_tie_on_lambda_min(self) -> None:
        from process_improve.experiments.designs_constrained import _phi_p

        repeated, split = np.array([4.0, 4.0, 9.0]), np.array([4.0, 6.0, 7.0])
        assert _phi_p(repeated) < _phi_p(split) < 4.0
