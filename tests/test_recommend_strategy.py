"""Tests for Tool 8: recommend_strategy — multi-stage DOE strategy recommender."""

from __future__ import annotations

import json

import pytest

from process_improve.experiments.factor import (
    Factor,
    Response,
    ResponseGoal,
)
from process_improve.experiments.strategy.budget import (
    allocate_budget,
    estimate_confirmation_runs,
    estimate_rsm_runs,
    estimate_screening_runs,
)
from process_improve.experiments.strategy.domain_templates import (
    DOMAIN_TEMPLATES,
    get_domain_template,
)
from process_improve.experiments.strategy.engine import (
    _parse_prior_knowledge,
    recommend_strategy,
)
from process_improve.experiments.strategy.models import (
    DOEProblemSpec,
    DomainType,
    ExperimentalStage,
    ExperimentalStrategy,
    TransitionRule,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def two_factors():
    return [Factor(name="A", low=0, high=100), Factor(name="B", low=0, high=100)]


@pytest.fixture
def three_factors():
    return [Factor(name="A", low=0, high=100), Factor(name="B", low=0, high=100), Factor(name="C", low=0, high=100)]


@pytest.fixture
def seven_factors():
    return [Factor(name=chr(65 + i), low=0, high=100) for i in range(7)]


@pytest.fixture
def twelve_factors():
    return [Factor(name=f"X{i+1}", low=0, high=100) for i in range(12)]


@pytest.fixture
def basic_responses():
    return [Response(name="Yield", goal="maximize"), Response(name="Purity", goal="maximize")]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Test input validation and error handling."""

    def test_empty_factors_raises(self):
        with pytest.raises(ValueError, match="At least one factor"):
            recommend_strategy(factors=[])

    def test_invalid_domain_raises(self, two_factors):
        with pytest.raises(ValueError, match="Unknown domain"):
            recommend_strategy(factors=two_factors, domain="nonexistent_domain")

    def test_invalid_detail_level_raises(self, two_factors):
        with pytest.raises(ValueError, match="detail_level"):
            recommend_strategy(factors=two_factors, detail_level="expert")

    def test_zero_budget_raises(self, two_factors):
        with pytest.raises(ValueError, match="positive integer"):
            recommend_strategy(factors=two_factors, budget=0)

    def test_negative_budget_raises(self, two_factors):
        with pytest.raises(ValueError, match="positive integer"):
            recommend_strategy(factors=two_factors, budget=-10)

    def test_single_factor_accepted(self):
        result = recommend_strategy(factors=[Factor(name="A", low=0, high=100)])
        assert "stages" in result
        assert result["total_estimated_runs"] > 0


# ---------------------------------------------------------------------------
# Response model validation
# ---------------------------------------------------------------------------


class TestResponseModel:
    """Test the Response Pydantic model."""

    def test_basic_response(self):
        r = Response(name="Yield", goal="maximize")
        assert r.name == "Yield"
        assert r.goal == ResponseGoal.maximize

    def test_target_requires_value(self):
        with pytest.raises(ValueError, match="target"):
            Response(name="pH", goal="target")

    def test_target_with_value(self):
        r = Response(name="pH", goal="target", target=7.0)
        assert r.target == 7.0

    def test_importance_default(self):
        r = Response(name="Yield", goal="maximize")
        assert r.importance == 1.0


# ---------------------------------------------------------------------------
# Prior knowledge parsing
# ---------------------------------------------------------------------------


class TestPriorKnowledgeParsing:
    """Test prior knowledge text → confidence mapping."""

    def test_high_confidence_keywords(self):
        pk = _parse_prior_knowledge("Temperature is confirmed to be significant", ["Temperature"])
        assert pk.confidence >= 0.8

    def test_medium_confidence_keywords(self):
        pk = _parse_prior_knowledge("Literature suggests pH matters", ["pH"])
        assert 0.5 <= pk.confidence <= 0.8

    def test_low_confidence_keywords(self):
        pk = _parse_prior_knowledge("We suspect temperature is important", ["Temperature"])
        assert 0.2 <= pk.confidence <= 0.6

    def test_no_knowledge(self):
        pk = _parse_prior_knowledge(None, [])
        assert pk.confidence == 0.0

    def test_empty_string(self):
        pk = _parse_prior_knowledge("", [])
        assert pk.confidence == 0.0

    def test_no_prior_data(self):
        pk = _parse_prior_knowledge("No prior data available", [])
        assert pk.confidence < 0.3

    def test_factor_extraction(self):
        pk = _parse_prior_knowledge(
            "Temperature is known to be significant and pH is important",
            ["Temperature", "pH", "Pressure"],
        )
        assert "Temperature" in pk.known_significant_factors

    def test_unknown_text_moderate_confidence(self):
        pk = _parse_prior_knowledge("Some random text without keywords", [])
        assert 0.1 <= pk.confidence <= 0.5


# ---------------------------------------------------------------------------
# Budget allocation
# ---------------------------------------------------------------------------


class TestBudgetAllocation:
    """Test budget allocation across stages."""

    def test_standard_allocation(self):
        result = allocate_budget(40, 7, needs_screening=True, needs_rsm=True)
        assert result["screening"] > 0
        assert result["optimization"] > 0
        assert result["confirmation"] >= 3
        assert result["total"] <= 40

    def test_no_budget_ideal(self):
        result = allocate_budget(None, 7, needs_screening=True, needs_rsm=True)
        assert result["is_tight"] is False
        assert result["total"] > 0

    def test_confirmation_minimum(self):
        result = allocate_budget(10, 3, needs_screening=True, needs_rsm=True)
        assert result["confirmation"] >= 3

    def test_tight_budget_warning(self):
        result = allocate_budget(10, 7, needs_screening=True, needs_rsm=True)
        assert result["is_tight"] is True

    def test_screening_only(self):
        result = allocate_budget(20, 7, needs_screening=True, needs_rsm=False)
        assert result["optimization"] == 0
        assert result["screening"] > 0

    def test_rsm_only(self):
        result = allocate_budget(20, 3, needs_screening=False, needs_rsm=True)
        assert result["screening"] == 0
        assert result["optimization"] > 0


# ---------------------------------------------------------------------------
# Run estimation
# ---------------------------------------------------------------------------


class TestRunEstimation:
    """Test run count estimation functions."""

    def test_pb_12_runs_for_7_factors(self):
        runs = estimate_screening_runs(7, "plackett_burman")
        assert runs == 8  # next mult of 4 >= 8

    def test_pb_12_runs_for_11_factors(self):
        runs = estimate_screening_runs(11, "plackett_burman")
        assert runs == 12  # next mult of 4 >= 12

    def test_dsd_runs(self):
        runs = estimate_screening_runs(7, "definitive_screening")
        assert runs == 15  # 2*7 + 1

    def test_full_factorial_runs(self):
        runs = estimate_screening_runs(3, "full_factorial")
        assert runs == 8  # 2^3

    def test_bbd_runs_3_factors(self):
        runs = estimate_rsm_runs(3, "box_behnken", center_points=3)
        assert runs == 15  # 12 + 3

    def test_ccd_runs_3_factors(self):
        runs = estimate_rsm_runs(3, "ccd", center_points=3)
        assert runs == 17  # 8 + 6 + 3

    def test_confirmation_minimum(self):
        assert estimate_confirmation_runs(3) == 3
        assert estimate_confirmation_runs(5) == 5
        assert estimate_confirmation_runs(1) == 3  # Clamped to 3


# ---------------------------------------------------------------------------
# Domain templates
# ---------------------------------------------------------------------------


class TestDomainTemplates:
    """Test domain-specific strategy adjustments."""

    def test_all_domains_present(self):
        for domain in DomainType:
            assert domain.value in DOMAIN_TEMPLATES

    def test_pharma_prefers_dsd(self):
        template = get_domain_template("pharma_formulation")
        assert template["screening_preference"] == "definitive_screening"

    def test_fermentation_prefers_pb(self):
        template = get_domain_template("fermentation")
        assert template["screening_preference"] == "plackett_burman"

    def test_cell_culture_prefers_dsd(self):
        template = get_domain_template("cell_culture")
        assert template["screening_preference"] == "definitive_screening"

    def test_general_no_preference(self):
        template = get_domain_template("general")
        assert template["screening_preference"] is None

    def test_unknown_domain_falls_back(self):
        template = get_domain_template("made_up_domain")
        assert template == DOMAIN_TEMPLATES["general"]

    def test_templates_have_notes(self):
        for name, template in DOMAIN_TEMPLATES.items():
            assert "notes" in template, f"Template {name} missing 'notes'"
            assert "novice" in template["notes"] or "intermediate" in template["notes"]


# ---------------------------------------------------------------------------
# Screening strategy selection
# ---------------------------------------------------------------------------


class TestScreeningStrategy:
    """Test screening design selection for different factor counts."""

    def test_two_factors_no_screening(self, two_factors):
        result = recommend_strategy(factors=two_factors)
        stage_names = [s["stage_name"] for s in result["stages"]]
        assert "Screening" not in stage_names

    def test_three_factors_factorial(self, three_factors):
        result = recommend_strategy(factors=three_factors, budget=30)
        screening = [s for s in result["stages"] if s["stage_name"] == "Screening"]
        assert len(screening) == 1
        assert screening[0]["design_type"] in ("full_factorial", "fractional_factorial")

    def test_seven_factors_screening(self, seven_factors):
        result = recommend_strategy(factors=seven_factors, budget=40)
        screening = [s for s in result["stages"] if s["stage_name"] == "Screening"]
        assert len(screening) == 1
        assert screening[0]["design_type"] in ("plackett_burman", "definitive_screening", "fractional_factorial")

    def test_mixture_factors(self):
        factors = [Factor(name=f"x{i}", type="mixture", low=0, high=1) for i in range(4)]
        result = recommend_strategy(factors=factors)
        screening = [s for s in result["stages"] if s["stage_name"] == "Screening"]
        if screening:
            assert "mixture" in screening[0]["design_type"] or "simplex" in screening[0]["design_type"]

    def test_hard_to_change_split_plot(self, seven_factors):
        result = recommend_strategy(factors=seven_factors, hard_to_change_factors=["A", "B"])
        for stage in result["stages"]:
            if stage["stage_name"] in ("Screening", "Optimization"):
                assert stage["design_params"].get("split_plot") is True


# ---------------------------------------------------------------------------
# Multi-stage strategy assembly
# ---------------------------------------------------------------------------


class TestMultiStageStrategy:
    """Test complete multi-stage strategy assembly."""

    def test_classic_three_stage(self, seven_factors, basic_responses):
        result = recommend_strategy(factors=seven_factors, responses=basic_responses, budget=40)
        stage_names = [s["stage_name"] for s in result["stages"]]
        assert "Screening" in stage_names
        assert "Confirmation" in stage_names
        assert len(result["stages"]) >= 2

    def test_two_factor_no_screening(self, two_factors, basic_responses):
        result = recommend_strategy(factors=two_factors, responses=basic_responses, budget=20)
        stage_names = [s["stage_name"] for s in result["stages"]]
        assert "Screening" not in stage_names
        assert "Confirmation" in stage_names

    def test_skip_screening_high_confidence(self, seven_factors, basic_responses):
        result = recommend_strategy(
            factors=seven_factors,
            responses=basic_responses,
            prior_knowledge="Published and validated results confirm Temperature and pH are significant.",
        )
        stage_names = [s["stage_name"] for s in result["stages"]]
        assert "Screening" not in stage_names

    def test_confirmation_always_present(self, three_factors):
        result = recommend_strategy(factors=three_factors)
        stage_names = [s["stage_name"] for s in result["stages"]]
        assert "Confirmation" in stage_names

    def test_stages_numbered_sequentially(self, seven_factors):
        result = recommend_strategy(factors=seven_factors, budget=40)
        for i, stage in enumerate(result["stages"]):
            assert stage["stage_number"] == i + 1


# ---------------------------------------------------------------------------
# Transition rules
# ---------------------------------------------------------------------------


class TestTransitionRules:
    """Test transition rules between stages."""

    def test_screening_has_transition_rules(self, seven_factors):
        result = recommend_strategy(factors=seven_factors, budget=40)
        screening = [s for s in result["stages"] if s["stage_name"] == "Screening"]
        if screening:
            assert len(screening[0]["transition_rules"]) > 0

    def test_confirmation_has_transition_rules(self, three_factors):
        result = recommend_strategy(factors=three_factors)
        confirmation = [s for s in result["stages"] if s["stage_name"] == "Confirmation"]
        assert len(confirmation) == 1
        assert len(confirmation[0]["transition_rules"]) > 0


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------


class TestOutputStructure:
    """Test output dict has expected shape."""

    def test_output_keys(self, seven_factors):
        result = recommend_strategy(factors=seven_factors, budget=40)
        expected_keys = {
            "strategy_id", "stages", "total_estimated_runs",
            "budget_allocation", "assumptions", "risks",
            "alternative_strategies", "domain", "detail_level", "reasoning",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_stages_non_empty(self, two_factors):
        result = recommend_strategy(factors=two_factors)
        assert len(result["stages"]) >= 1

    def test_strategy_id_deterministic(self, seven_factors):
        r1 = recommend_strategy(factors=seven_factors, budget=40)
        r2 = recommend_strategy(factors=seven_factors, budget=40)
        assert r1["strategy_id"] == r2["strategy_id"]

    def test_json_serializable(self, seven_factors, basic_responses):
        result = recommend_strategy(factors=seven_factors, responses=basic_responses, budget=40)
        serialized = json.dumps(result)
        assert isinstance(serialized, str)

    def test_total_runs_matches_stages(self, seven_factors):
        result = recommend_strategy(factors=seven_factors, budget=60)
        total = sum(s["estimated_runs"] for s in result["stages"])
        assert result["total_estimated_runs"] == total

    def test_reasoning_non_empty(self, seven_factors):
        result = recommend_strategy(factors=seven_factors, budget=40)
        assert len(result["reasoning"]) >= 1

    def test_assumptions_non_empty(self, seven_factors):
        result = recommend_strategy(factors=seven_factors, budget=40)
        assert len(result["assumptions"]) >= 1


# ---------------------------------------------------------------------------
# Real-world scenarios from the question bank
# ---------------------------------------------------------------------------


class TestRealWorldScenarios:
    """Integration tests matching specific questions from the 162-question bank."""

    def test_q1_seven_factors_how_to_start(self):
        """Q1: I have 7 factors, how do I even start planning a DOE."""
        factors = [Factor(name=f"Factor_{i+1}", low=0, high=100) for i in range(7)]
        result = recommend_strategy(factors=factors, budget=40)
        assert len(result["stages"]) >= 2
        assert result["total_estimated_runs"] <= 40

    def test_q63_eight_factors_screening(self):
        """Q63: 8 factors — screening to narrow to 2-3 in 16 runs."""
        factors = [Factor(name=chr(65 + i), low=0, high=100) for i in range(8)]
        result = recommend_strategy(factors=factors, budget=40)
        screening = [s for s in result["stages"] if s["stage_name"] == "Screening"]
        assert len(screening) == 1

    def test_q64_chemical_engineer_maximize_yield(self):
        """Q64: T, P, catalyst% in ~20 runs — propose full strategy."""
        factors = [
            Factor(name="Temperature", low=150, high=200, units="degC"),
            Factor(name="Pressure", low=1, high=5, units="bar"),
            Factor(name="Catalyst", low=1, high=5, units="%"),
        ]
        responses = [Response(name="Yield", goal="maximize")]
        result = recommend_strategy(factors=factors, responses=responses, budget=20)
        assert result["total_estimated_runs"] <= 20

    def test_q65_expensive_experiments(self):
        """Q65: $5000/run, budget for 25 runs."""
        factors = [Factor(name=f"X{i+1}", low=0, high=100) for i in range(6)]
        responses = [Response(name="Output", goal="maximize")]
        result = recommend_strategy(factors=factors, responses=responses, budget=25)
        assert result["total_estimated_runs"] <= 25
        assert len(result["stages"]) >= 2

    def test_q104_fermentation_7_factors(self):
        """Q104: Optimize fermentation medium, 7 factors."""
        factors = [
            Factor(name="pH", low=5.0, high=8.0),
            Factor(name="Temperature", low=25, high=40, units="degC"),
            Factor(name="Glucose", low=5, high=30, units="g/L"),
            Factor(name="Yeast_extract", low=1, high=10, units="g/L"),
            Factor(name="Agitation", low=100, high=300, units="rpm"),
            Factor(name="Aeration", low=0.5, high=2.0, units="vvm"),
            Factor(name="Inoculum", low=1, high=10, units="%"),
        ]
        responses = [Response(name="Yield", goal="maximize")]
        result = recommend_strategy(factors=factors, responses=responses, budget=40, domain="fermentation")
        assert result["domain"] == "fermentation"
        screening = [s for s in result["stages"] if s["stage_name"] == "Screening"]
        assert len(screening) == 1

    def test_q117_brewing_parameters(self):
        """Q117: Screen and optimize brewing parameters."""
        factors = [
            Factor(name="pH", low=4.0, high=6.0),
            Factor(name="Brix", low=10, high=20),
            Factor(name="Time", low=24, high=72, units="h"),
            Factor(name="Inoculum", low=1, high=5, units="%"),
            Factor(name="Temperature", low=20, high=35, units="degC"),
        ]
        responses = [Response(name="Alcohol", goal="maximize")]
        result = recommend_strategy(factors=factors, responses=responses, budget=30)
        assert len(result["stages"]) >= 2

    def test_q131_ipsc_differentiation(self):
        """Q131: iPSC differentiation, 6 conditions, 21-day runs."""
        factors = [Factor(name=f"Condition_{i+1}", low=0, high=100) for i in range(6)]
        responses = [Response(name="Differentiation_efficiency", goal="maximize")]
        result = recommend_strategy(factors=factors, responses=responses, domain="cell_culture")
        assert result["domain"] == "cell_culture"

    def test_q134_stem_cell_minimal_runs(self):
        """Q134: Expensive/slow experiments, minimal runs."""
        factors = [Factor(name=f"Factor_{i+1}", low=0, high=100) for i in range(6)]
        responses = [Response(name="Viability", goal="maximize")]
        result = recommend_strategy(factors=factors, responses=responses, domain="cell_culture", budget=20)
        assert result["total_estimated_runs"] <= 20

    def test_q149_two_stage_pb_then_rsm(self):
        """Q149: PB screening then RSM for significant factors."""
        factors = [Factor(name=chr(65 + i), low=0, high=100) for i in range(8)]
        responses = [Response(name="Response", goal="maximize")]
        result = recommend_strategy(factors=factors, responses=responses, budget=40)
        stage_names = [s["stage_name"] for s in result["stages"]]
        assert "Screening" in stage_names
        assert "Confirmation" in stage_names


# ---------------------------------------------------------------------------
# Tool spec integration
# ---------------------------------------------------------------------------


class TestToolSpecIntegration:
    """Test tool registration and execution."""

    def test_tool_registered(self):
        from process_improve.tool_spec import get_tool_specs

        specs = get_tool_specs()
        names = [s["name"] for s in specs]
        assert "recommend_strategy" in names

    def test_execute_tool_call(self):
        from process_improve.tool_spec import execute_tool_call

        result = execute_tool_call(
            "recommend_strategy",
            {
                "factors": [
                    {"name": "A", "low": 0, "high": 100},
                    {"name": "B", "low": 0, "high": 100},
                    {"name": "C", "low": 0, "high": 100},
                ],
                "budget": 20,
            },
        )
        assert "error" not in result
        assert "stages" in result

    def test_error_handling(self):
        from process_improve.tool_spec import execute_tool_call

        result = execute_tool_call("recommend_strategy", {"factors": []})
        assert "error" in result


# ---------------------------------------------------------------------------
# Pydantic model tests
# ---------------------------------------------------------------------------


class TestModels:
    """Test Pydantic model construction and properties."""

    def test_doe_problem_spec_properties(self, seven_factors):
        spec = DOEProblemSpec(factors=seven_factors)
        assert spec.n_factors == 7
        assert spec.n_continuous == 7
        assert spec.n_categorical == 0
        assert spec.n_mixture == 0
        assert spec.has_mixture is False
        assert spec.has_hard_to_change is False

    def test_experimental_stage_construction(self):
        stage = ExperimentalStage(
            stage_number=1,
            stage_name="Screening",
            design_type="plackett_burman",
            estimated_runs=12,
        )
        assert stage.stage_number == 1
        assert stage.design_type == "plackett_burman"

    def test_transition_rule_construction(self):
        rule = TransitionRule(
            condition="2-5 significant factors",
            action="proceed_to_rsm",
            fallback="run_additional_screening",
        )
        assert rule.condition == "2-5 significant factors"

    def test_strategy_model_dump(self):
        strategy = ExperimentalStrategy(strategy_id="abc123", total_estimated_runs=40)
        d = strategy.model_dump()
        assert d["strategy_id"] == "abc123"
        assert d["total_estimated_runs"] == 40

    def test_domain_type_enum(self):
        assert DomainType("fermentation") == DomainType.fermentation
        with pytest.raises(ValueError, match="nonexistent"):
            DomainType("nonexistent")
