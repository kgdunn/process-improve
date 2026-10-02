"""Tests for Tool 8: recommend_strategy - multi-stage DOE strategy recommender."""

from __future__ import annotations

import json

import pytest

from process_improve.experiments.factor import (
    Constraint,
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
    _screening_design_params,
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
    return [Factor(name=f"X{i + 1}", low=0, high=100) for i in range(12)]


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

    @pytest.mark.parametrize(
        "text",
        [
            "Published studies confirmed Temperature and pH are significant, as expected.",
            "Temperature was confirmed significant in a validated study; the role of Salt is unknown.",
        ],
    )
    def test_strongest_evidence_sets_the_confidence(self, text):
        """A weak cue ('expected', 'unknown') used to cap the confidence although the text reports confirmed results."""
        pk = _parse_prior_knowledge(text, ["Temperature", "pH", "Salt"])
        assert pk.confidence == 0.9
        assert pk.has_supporting_data

    def test_significant_factor_regex_runs_in_linear_time(self):
        r"""SEC-29 (#278) regression guard.

        The previous ``_SIGNIFICANT_FACTOR_PATTERN`` used ``[\\w\\s]*?`` which is
        O(n^2) on whitespace-heavy input. Combined with the ``max_string``
        ceiling, a ~50KB whitespace payload burned significant CPU. The fix
        bounded the capture to ``\\w+(?:\\s\\w+){0,4}`` (linear time). This
        test sends a 50KB whitespace-heavy payload and asserts the parse
        completes well under a second.
        """
        import time

        # 50KB of mostly whitespace with one trigger phrase at the end.
        payload = (" " * 49_900) + "Temperature is significant"
        start = time.perf_counter()
        pk = _parse_prior_knowledge(payload, ["Temperature"])
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, (
            f"_SIGNIFICANT_FACTOR_PATTERN took {elapsed:.3f}s on a 50KB "
            "whitespace payload; the bounded regex should be linear in size."
        )
        # Sanity: still extracts the legitimate trigger.
        assert "Temperature" in pk.known_significant_factors


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
        # An odd count uses the conference matrix of the next even order: 2*7 + 3.
        assert estimate_screening_runs(7, "definitive_screening") == 17
        assert estimate_screening_runs(8, "dsd") == 17
        # No conference matrix of order 22 exists, so 21 and 22 factors use order 24.
        assert estimate_screening_runs(22, "dsd") == 49

    def test_pb_runs_match_generated_design(self):
        # 24 factors need 28 runs, which pyDOE3 cannot build; 89 factors step past 92 to 96.
        assert estimate_screening_runs(24, "plackett_burman") == 28
        assert estimate_screening_runs(89, "plackett_burman") == 96

    def test_full_factorial_runs(self):
        runs = estimate_screening_runs(3, "full_factorial")
        assert runs == 8  # 2^3

    def test_bbd_runs_3_factors(self):
        runs = estimate_rsm_runs(3, "box_behnken", n_center_points=3)
        assert runs == 15  # 12 + 3

    def test_ccd_runs_3_factors(self):
        runs = estimate_rsm_runs(3, "ccd", n_center_points=3)
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
                assert stage["design_type"] == "d_optimal"
                assert stage["design_params"]["hard_to_change"] == ["A", "B"]


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
            "strategy_id",
            "stages",
            "total_estimated_runs",
            "budget_allocation",
            "assumptions",
            "risks",
            "alternative_strategies",
            "domain",
            "detail_level",
            "reasoning",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_stages_non_empty(self, two_factors):
        result = recommend_strategy(factors=two_factors)
        assert len(result["stages"]) >= 1

    def test_strategy_id_deterministic(self, seven_factors):
        r1 = recommend_strategy(factors=seven_factors, budget=40)
        r2 = recommend_strategy(factors=seven_factors, budget=40)
        assert r1["strategy_id"] == r2["strategy_id"]

    def test_strategy_id_changes_with_every_input_that_changes_the_strategy(self, seven_factors, basic_responses):
        """The id hashed only names, budget, domain and hard-to-change factors, so different plans shared it."""
        base = {"factors": seven_factors, "responses": basic_responses}
        variants = [
            {},
            {"prior_knowledge": "Published: A and B are significant."},
            {"constraints": [Constraint(expression="A + B <= 150")]},
            {"detail_level": "novice"},
            {"factors": [*seven_factors[:-1], Factor(name="G", low=0, high=50)]},
            {"responses": [Response(name="Yield", goal="minimize"), Response(name="Purity", goal="maximize")]},
        ]
        ids = [recommend_strategy(**{**base, **variant})["strategy_id"] for variant in variants]
        assert len(set(ids)) == len(ids)

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
        factors = [Factor(name=f"Factor_{i + 1}", low=0, high=100) for i in range(7)]
        result = recommend_strategy(factors=factors, budget=40)
        assert len(result["stages"]) >= 2
        assert result["total_estimated_runs"] <= 40

    def test_q63_eight_factors_screening(self):
        """Q63: 8 factors - screening to narrow to 2-3 in 16 runs."""
        factors = [Factor(name=chr(65 + i), low=0, high=100) for i in range(8)]
        result = recommend_strategy(factors=factors, budget=40)
        screening = [s for s in result["stages"] if s["stage_name"] == "Screening"]
        assert len(screening) == 1

    def test_q64_chemical_engineer_maximize_yield(self):
        """Q64: T, P, catalyst% in ~20 runs - propose full strategy."""
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
        factors = [Factor(name=f"X{i + 1}", low=0, high=100) for i in range(6)]
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
        factors = [Factor(name=f"Condition_{i + 1}", low=0, high=100) for i in range(6)]
        responses = [Response(name="Differentiation_efficiency", goal="maximize")]
        result = recommend_strategy(factors=factors, responses=responses, domain="cell_culture")
        assert result["domain"] == "cell_culture"

    def test_q134_stem_cell_minimal_runs(self):
        """Q134: Expensive/slow experiments, minimal runs."""
        factors = [Factor(name=f"Factor_{i + 1}", low=0, high=100) for i in range(6)]
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
        """Empty factors list is rejected by pydantic min_length=1."""
        import pytest

        from process_improve.tool_safety import ToolInputInvalidError
        from process_improve.tool_spec import execute_tool_call

        with pytest.raises(ToolInputInvalidError):
            execute_tool_call("recommend_strategy", {"factors": []})


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


class TestScreeningDesignParams:
    """Each screening design type carries its own parameter dict."""

    def test_fractional_factorial_asks_for_resolution_iv_and_centre_points(self):
        assert _screening_design_params("fractional_factorial", 6) == {
            "resolution": 4,
            "n_center_points": 3,
        }

    def test_plackett_burman_has_no_centre_points(self):
        assert _screening_design_params("plackett_burman", 6) == {"n_center_points": 0}

    def test_definitive_screening_needs_no_parameters(self):
        """``fake_factor`` is not a generate_design argument; the DSD handles an even count itself (#638)."""
        assert _screening_design_params("dsd", 16) == {}

    def test_supersaturated_carries_its_run_count(self):
        assert _screening_design_params("supersaturated", 6) == {"budget": 6}

    def test_unknown_design_type_carries_no_parameters(self):
        assert _screening_design_params("d_optimal", 6) == {}


# ---------------------------------------------------------------------------
# Every recommended stage can be generated (#638)
# ---------------------------------------------------------------------------


def _continuous(k: int) -> list[Factor]:
    return [Factor(name=f"X{i}", low=0, high=10) for i in range(k)]


_SCENARIOS = {
    "3 factors": {"factors": _continuous(3)},
    "5 factors": {"factors": _continuous(5)},
    "8 factors, roomy budget": {"factors": _continuous(8), "budget": 80},
    "8 factors, very tight budget": {"factors": _continuous(8), "budget": 18},
    "10 factors in 6 runs": {"factors": _continuous(10), "budget": 6},
    "12 factors, curvature prior": {"factors": _continuous(12), "prior_knowledge": "we know the ranges well"},
    "bounded mixture": {
        "factors": [Factor(name=f"m{i}", type="mixture", low=0.1, high=0.8) for i in range(3)],
    },
    "constrained": {
        "factors": _continuous(3),
        "constraints": [Constraint(expression="X0 + X1 <= 15")],
    },
}


@pytest.mark.parametrize("domain", [d.value for d in DomainType])
@pytest.mark.parametrize("scenario", list(_SCENARIOS))
def test_every_stage_but_confirmation_is_a_generate_design_call(scenario: str, domain: str) -> None:
    """The agent flow is recommend_strategy, then generate_design with each stage's type and parameters."""
    from process_improve.experiments import generate_design

    inputs = _SCENARIOS[scenario]
    strategy = recommend_strategy(**inputs, domain=domain)
    for stage in strategy["stages"]:
        if stage["stage_name"] == "Confirmation":
            assert stage["design_type"] == "replicates_at_optimum"
            continue
        factors = [f for f in inputs["factors"] if f.name in stage["factors"]]
        params = {"constraints": inputs.get("constraints"), **stage["design_params"]}
        result = generate_design(factors, design_type=stage["design_type"], **params)
        assert result.n_runs > 0


def test_a_budget_below_k_plus_1_recommends_a_supersaturated_design() -> None:
    strategy = recommend_strategy(factors=_continuous(10), budget=6)
    screening = next(s for s in strategy["stages"] if s["stage_name"] == "Screening")
    assert screening["design_type"] == "supersaturated"
    assert screening["design_params"] == {"budget": 6}


def test_an_all_mixture_problem_optimises_with_a_mixture_design() -> None:
    """The optimisation stage was a CCD on the components, which gave coded +/-alpha rows, not proportions."""
    inputs = _SCENARIOS["bounded mixture"]
    strategy = recommend_strategy(**inputs)
    stage = next(s for s in strategy["stages"] if s["stage_name"] == "Optimization")
    assert stage["design_type"] == "mixture"
    assert stage["factors"] == [f.name for f in inputs["factors"]]


# ---------------------------------------------------------------------------
# The recommended stages build what the plan says (#138, #139, #141-#145, #149)
# ---------------------------------------------------------------------------


def _stage_factors(factors: list[Factor], stage: dict) -> list[Factor]:
    return [f for f in factors if f.name in stage["factors"]]


def _build(factors: list[Factor], stage: dict):
    from process_improve.experiments import generate_design

    return generate_design(_stage_factors(factors, stage), design_type=stage["design_type"], **stage["design_params"])


def _maximise() -> list[Response]:
    return [Response(name="y", goal="maximize")]


_OPTIMAL = {"d_optimal", "i_optimal", "a_optimal", "e_optimal"}


@pytest.mark.parametrize("domain", [d.value for d in DomainType])
@pytest.mark.parametrize("k", [2, 3, 4, 5, 6, 8, 12])
def test_estimated_runs_are_the_runs_the_stage_builds(k: int, domain: str) -> None:
    """A budget lowered estimated_runs without changing the design, which then needed 2-5 times more runs."""
    factors = _continuous(k)
    for budget in (None, 8, 15, 25, 40, 100):
        strategy = recommend_strategy(factors=factors, responses=_maximise(), budget=budget, domain=domain)
        for stage in strategy["stages"]:
            if stage["design_type"] == "replicates_at_optimum":
                assert stage["estimated_runs"] == stage["design_params"]["n_replicates"]
            elif stage["design_type"] in _OPTIMAL:
                # The budget sets an optimal design's size; building one per case would take minutes.
                assert stage["estimated_runs"] == stage["design_params"]["budget"]
            else:
                assert _build(factors, stage).n_runs == stage["estimated_runs"], (budget, stage)
        total = sum(s["estimated_runs"] for s in strategy["stages"])
        assert strategy["total_estimated_runs"] == total
        if budget is not None and total > budget:
            assert any(f"Budget of {budget} runs is below the smallest plan" in r for r in strategy["risks"])


@pytest.mark.parametrize(("k", "budget"), [(5, 25), (6, 20)])
def test_a_tight_budget_changes_the_design_not_just_its_run_count(k: int, budget: int) -> None:
    """Five factors in 25 runs: smaller designs, not the ideal ones with a lower estimate; six in 20: a single DSD."""
    factors = _continuous(k)
    strategy = recommend_strategy(factors=factors, responses=_maximise(), budget=budget)
    assert strategy["total_estimated_runs"] <= budget
    for stage in strategy["stages"]:
        if stage["design_type"] != "replicates_at_optimum":
            assert _build(factors, stage).n_runs == stage["estimated_runs"]
    assert any(f"Budget of {budget} runs" in r for r in strategy["risks"])


def test_a_budget_below_every_plan_says_so() -> None:
    """The reproducer: five factors in 15 runs used to report 15 runs for designs that build 39."""
    strategy = recommend_strategy(factors=_continuous(5), responses=_maximise(), budget=15)
    assert strategy["total_estimated_runs"] == 16  # a 13-run DSD and 3 confirmation runs
    assert any("Budget of 15 runs is below the smallest plan" in r for r in strategy["risks"])


def test_an_infeasible_budget_is_reported_not_hidden() -> None:
    strategy = recommend_strategy(factors=_continuous(3), responses=_maximise(), budget=8)
    assert strategy["total_estimated_runs"] > 8
    assert any("Budget of 8 runs is below the smallest plan" in r for r in strategy["risks"])


def test_the_constrained_optimisation_stage_builds_a_feasible_quadratic_design() -> None:
    factors = [Factor(name=n, low=0, high=1) for n in "ABC"]
    constraints = [Constraint(expression="A + B <= 1.5")]
    strategy = recommend_strategy(factors=factors, responses=_maximise(), constraints=constraints)
    stage = next(s for s in strategy["stages"] if s["stage_name"] == "Optimization")
    assert stage["design_type"] == "d_optimal"
    assert stage["design_params"]["model_type"] == "quadratic"
    result = _build(factors, stage)
    assert result.n_runs == stage["estimated_runs"]
    actual = result.design_actual
    assert (actual["A"] + actual["B"]).max() <= 1.5 + 1e-9
    assert all(actual[name].round(6).nunique() >= 3 for name in "ABC")  # three levels: a quadratic is estimable


def test_hard_to_change_factors_give_a_split_plot_generate_design_call() -> None:
    factors = [Factor(name=n, low=0, high=1) for n in "ABCD"]
    strategy = recommend_strategy(factors=factors, responses=_maximise(), hard_to_change_factors=["A"])
    split = [s for s in strategy["stages"] if "hard_to_change" in s["design_params"]]
    assert split
    for stage in split:
        assert stage["design_params"]["hard_to_change"] == ["A"]
        assert stage["design_type"] == "d_optimal"
        assert _build(factors, stage).n_runs == stage["estimated_runs"]


def test_an_unknown_hard_to_change_factor_is_refused() -> None:
    with pytest.raises(ValueError, match="Zebra"):
        recommend_strategy(factors=_continuous(4), hard_to_change_factors=["Zebra"])


@pytest.mark.parametrize("domain", ["food_science", "cell_culture"])
def test_two_factors_never_get_a_box_behnken_design(domain: str) -> None:
    """A Box-Behnken design does not exist for two factors."""
    factors = [Factor(name=n, low=0, high=1) for n in "AB"]
    stage = recommend_strategy(factors=factors, responses=_maximise(), domain=domain)["stages"][0]
    assert stage["design_type"] == "ccd"
    assert stage["design_params"]["alpha"] == "face_centered"
    assert _build(factors, stage).n_runs == stage["estimated_runs"]


def test_a_three_level_categorical_factor_gets_designs_that_build() -> None:
    factors = [Factor(name="A", type="categorical", levels=["x", "y", "z"]), *_continuous(3)]
    strategy = recommend_strategy(factors=factors, responses=_maximise())
    for stage in strategy["stages"]:
        if stage["design_type"] != "replicates_at_optimum":
            assert _build(factors, stage).n_runs == stage["estimated_runs"]


def test_mixture_components_with_process_factors_are_refused() -> None:
    factors = [Factor(name=n, type="mixture", low=0, high=1) for n in "ABC"] + [Factor(name="T", low=0, high=1)]
    with pytest.raises(ValueError, match="Mixture-process"):
        recommend_strategy(factors=factors, responses=_maximise())


def test_known_significant_factors_are_the_ones_optimised() -> None:
    names = ["Temperature", "pH", "Stirring", "Time", "Feed", "Salt"]
    factors = [Factor(name=n, low=0, high=1) for n in names]
    text = "Published literature confirms Temperature and pH are significant."
    strategy = recommend_strategy(factors=factors, responses=_maximise(), prior_knowledge=text)
    stage = strategy["stages"][0]
    assert stage["stage_name"] == "Optimization"
    assert stage["factors"] == ["Temperature", "pH"]
    assert _build(factors, stage).n_runs == stage["estimated_runs"]


def test_after_screening_the_optimisation_factors_are_marked_as_placeholders() -> None:
    strategy = recommend_strategy(factors=_continuous(6), responses=_maximise())
    stage = next(s for s in strategy["stages"] if s["stage_name"] == "Optimization")
    assert "screening finds" in stage["purpose"]


def test_one_factor_with_a_goal_gets_an_optimisation_stage() -> None:
    factors = [Factor(name="T", low=0, high=1)]
    strategy = recommend_strategy(factors=factors, responses=_maximise())
    stage = next(s for s in strategy["stages"] if s["stage_name"] == "Optimization")
    design = _build(factors, stage)
    assert design.n_runs == stage["estimated_runs"]
    assert design.design["T"].round(6).nunique() == 3


@pytest.mark.parametrize("q", [3, 4, 5])
@pytest.mark.parametrize("bounds", [(0.0, 1.0), (0.05, 0.8)])
def test_mixture_stages_build_their_estimated_runs(q: int, bounds: tuple[float, float]) -> None:
    factors = [Factor(name=f"M{i}", type="mixture", low=bounds[0], high=bounds[1]) for i in range(q)]
    strategy = recommend_strategy(factors=factors, responses=_maximise())
    for stage in strategy["stages"]:
        if stage["design_type"] == "mixture":
            assert _build(factors, stage).n_runs == stage["estimated_runs"]


def test_domain_considerations_and_extra_stages_reach_the_output() -> None:
    """The pharma note promised a design-space stage that never appeared, and the considerations were unused."""
    strategy = recommend_strategy(
        factors=_continuous(6), responses=_maximise(), domain="pharma_formulation", detail_level="novice"
    )
    template = get_domain_template("pharma_formulation")
    for consideration in template["special_considerations"]:
        assert consideration in strategy["risks"]
    assert any("design space" in line and "not scheduled" in line for line in strategy["reasoning"])
    assert not any("strategy includes a design space" in line for line in strategy["reasoning"])
