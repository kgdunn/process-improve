"""``evaluate_design`` and ``analyze_experiment`` follow the model a region leaves estimable."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import (
    DesignRegion,
    DesignResult,
    Factor,
    analyze_experiment,
    evaluate_design,
    generate_design,
)
from process_improve.experiments.evaluate import _settings_over

NAMES = list("ABCD")
AMOUNTS = [Factor(name=name, low=0, high=10, min_when_used=2) for name in NAMES]
ONE_OF_THREE = ["count_nonzero(A, B, C) == 1"]
RANK_DEFICIENT = "Design is rank-deficient for the specified model."


@pytest.fixture(scope="module")
def one_of_three() -> DesignResult:
    """Exactly one of A, B and C in every run, so no run uses two of them together."""
    return generate_design(AMOUNTS, budget=14, model_type="interactions", constraints=ONE_OF_THREE, random_state=1)


@pytest.fixture(scope="module")
def halves() -> DesignResult:
    """Two pairs of components that may each supply at most half of a blend supply exactly half each."""
    components = [Factor(name=name, type="mixture", high=0.5) for name in "abcd"]
    return generate_design(
        components,
        budget=10,
        model_type="scheffe_quadratic",
        constraints=["a + b <= 0.5", "c + d <= 0.5"],
        random_state=1,
    )


class TestEvaluateDesign:
    """A design is judged on the columns its region can estimate."""

    @pytest.mark.parametrize("model", [None, "interactions"])
    def test_interactions_of_factors_never_used_together_are_left_out(
        self, one_of_three: DesignResult, model: str | None
    ) -> None:
        result = evaluate_design(one_of_three, model=model, metric=["d_efficiency", "vif"])
        assert (
            "['A:B', 'A:C', 'B:C'] inestimable over every setting they allow" in result["notes"]["inestimable_columns"]
        )
        assert result["d_efficiency"] > 0
        assert set(result["vif"]) == {"A", "B", "C", "D", "A:D", "B:D", "C:D"}

    def test_the_metrics_are_those_of_the_model_without_the_lost_columns(self, one_of_three: DesignResult) -> None:
        reduced = evaluate_design(one_of_three, model="interactions", metric="d_efficiency")
        named = evaluate_design(one_of_three, model="A + B + C + D + A:D + B:D + C:D", metric="d_efficiency")
        assert reduced["d_efficiency"] == pytest.approx(named["d_efficiency"])
        assert "notes" not in named

    def test_a_group_of_components_held_to_a_total_costs_a_column(self, halves: DesignResult) -> None:
        linear = evaluate_design(halves, model="scheffe_linear", metric="d_efficiency")
        assert "['d'] inestimable" in linear["notes"]["inestimable_columns"]
        assert linear["d_efficiency"] > 0
        quadratic = evaluate_design(halves, model="scheffe_quadratic", metric="d_efficiency")
        assert halves.metadata["inestimable_columns"] == ["d", "a:d", "b:c", "b:d"]
        assert "['d', 'a:d', 'b:c', 'b:d'] inestimable" in quadratic["notes"]["inestimable_columns"]
        assert quadratic["d_efficiency"] > 0

    def test_a_design_too_small_for_the_model_is_still_rank_deficient(self) -> None:
        design = generate_design(AMOUNTS, budget=6, model_type="main_effects", constraints=ONE_OF_THREE, random_state=1)
        result = evaluate_design(design, model="interactions", metric="d_efficiency")
        assert result["d_efficiency"] is None
        assert result["notes"]["d_efficiency"] == RANK_DEFICIENT  # the region's own 8 columns need 8 runs

    def test_a_deficiency_the_region_does_not_explain_is_reported_as_before(self) -> None:
        names = ["x1", "x2", "x3"]
        factors = [Factor(name=name, low=0, high=10) for name in names]
        design = generate_design(
            factors, budget=5, model_type="main_effects", constraints=["x1 + x2 + x3 <= 20"], random_state=1
        )
        result = evaluate_design(design, model="interactions", metric="d_efficiency")
        assert result["d_efficiency"] is None
        assert list(result["notes"]) == ["d_efficiency"]

    def test_a_design_that_estimates_the_whole_model_is_judged_on_all_of_it(self) -> None:
        design = generate_design(AMOUNTS, budget=14, model_type="interactions", random_state=1)
        result = evaluate_design(design, model="interactions", metric="vif")
        assert len(result["vif"]) == 10
        assert "notes" not in result


class TestSettingsOverARegion:
    """The settings a column is judged on: a sample of the region and its candidate settings."""

    def test_a_box_gives_its_sample_and_its_support_points(self) -> None:
        region = DesignRegion(AMOUNTS, ONE_OF_THREE)
        settings = _settings_over(region, np.random.default_rng(0))
        assert list(settings.columns) == NAMES
        assert len(settings) > 4000
        assert region.feasible(settings.to_numpy()).all()

    def test_a_blend_with_a_component_left_out_gives_its_candidates_alone(self) -> None:
        components = [Factor(name=name, type="mixture", high=0.6, min_when_used=0.1) for name in "abc"]
        region = DesignRegion(components)
        with pytest.raises(ValueError, match="min_when_used"):
            region.sample(10, np.random.default_rng(0))
        settings = _settings_over(region, np.random.default_rng(0))
        assert 0 < len(settings) < 4000
        np.testing.assert_allclose(settings.sum(axis=1), 1.0)


class TestAnalyzeExperiment:
    """The default model leaves out the interactions the runs cannot estimate, and says why."""

    def test_two_factors_never_used_together(self, one_of_three: DesignResult) -> None:
        runs = one_of_three.design_actual[NAMES].copy()
        noise = np.random.default_rng(1).normal(0, 0.1, len(runs))
        runs["y"] = 5 + 0.8 * runs["A"] + 0.5 * runs["B"] + 0.3 * runs["C"] + 0.4 * runs["D"] + noise
        with pytest.warns(UserWarning, match="no run has A and B both above their lowest values; no run has A and C"):
            fit = analyze_experiment(runs, response_column="y", analysis_type=["coefficients"])
        assert [c["term"] for c in fit["coefficients"]] == ["Intercept", "A", "B", "C", "D", "A:D", "B:D", "C:D"]

    def test_a_factor_held_at_one_value_on_a_level_of_the_other(self) -> None:
        runs = pd.DataFrame({"A": [0, 0, 1, 0, 0, 1], "B": [0, 1, 1, 0, 1, 1], "y": [1.0, 2.1, 3.9, 1.1, 1.9, 4.1]})
        with pytest.warns(UserWarning, match="A:B has a factor that takes one value on a level of the other"):
            fit = analyze_experiment(runs, response_column="y", analysis_type=["coefficients"])
        assert [c["term"] for c in fit["coefficients"]] == ["Intercept", "A", "B"]
