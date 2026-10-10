"""Tests for Tool 7: doe_knowledge - DOE knowledge graph and query engine."""

from __future__ import annotations

import pytest

from process_improve.experiments.knowledge import doe_knowledge
from process_improve.experiments.knowledge.engine import (
    _extract_detail,
    _keyword_search,
    load_knowledge_graph,
    query_design_properties,
    reload_knowledge_graph,
)
from process_improve.experiments.knowledge.models import (
    ConceptNode,
    DecisionRuleNode,
    DesignTypeNode,
    DiagnosticNode,
    KnowledgeGraph,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fresh_graph():
    """Ensure each test starts with a freshly loaded graph."""
    reload_knowledge_graph()
    yield
    reload_knowledge_graph()


# ---------------------------------------------------------------------------
# Loading and data integrity
# ---------------------------------------------------------------------------


class TestKnowledgeGraphLoading:
    """Verify that YAML data files load correctly into the graph."""

    def test_graph_loads_without_error(self):
        graph = load_knowledge_graph()
        assert isinstance(graph, KnowledgeGraph)

    def test_design_types_loaded(self):
        graph = load_knowledge_graph()
        assert len(graph.design_types) >= 5
        for dt in graph.design_types.values():
            assert isinstance(dt, DesignTypeNode)
            assert dt.id
            assert dt.display_name
            assert dt.category

    def test_decision_rules_loaded(self):
        graph = load_knowledge_graph()
        assert len(graph.decision_rules) >= 5
        for rule in graph.decision_rules:
            assert isinstance(rule, DecisionRuleNode)
            assert rule.id
            assert rule.conditions

    def test_diagnostics_loaded(self):
        graph = load_knowledge_graph()
        assert len(graph.diagnostics) >= 5
        for diag in graph.diagnostics.values():
            assert isinstance(diag, DiagnosticNode)
            assert diag.visual_pattern
            assert diag.indicates

    def test_concepts_loaded(self):
        graph = load_knowledge_graph()
        assert len(graph.concepts) >= 5
        for concept in graph.concepts.values():
            assert isinstance(concept, ConceptNode)
            assert concept.title
            assert concept.content

    def test_keyword_index_built(self):
        graph = load_knowledge_graph()
        assert len(graph.keyword_index) > 0
        # "resolution" should be indexed
        assert "resolution" in graph.keyword_index

    def test_topic_index_built(self):
        graph = load_knowledge_graph()
        assert "design_selection" in graph.topic_index
        assert "troubleshooting" in graph.topic_index
        assert "statistical_concepts" in graph.topic_index

    def test_singleton_caching(self):
        g1 = load_knowledge_graph()
        g2 = load_knowledge_graph()
        assert g1 is g2

    def test_reload_clears_cache(self):
        g1 = load_knowledge_graph()
        g2 = reload_knowledge_graph()
        assert g1 is not g2


# ---------------------------------------------------------------------------
# Design selection queries
# ---------------------------------------------------------------------------


class TestDesignSelection:
    """Test context-aware design selection rules."""

    def test_screening_many_factors_tight_budget(self):
        result = doe_knowledge(
            topic="design_selection",
            context={"n_factors": 7, "budget": 15, "goal": "screening"},
        )
        assert result["n_results"] >= 1
        rule = result["results"][0]
        assert rule["type"] == "decision_rule"
        assert rule["recommendation"]["primary"] == "plackett_burman"

    def test_screening_moderate_factors(self):
        result = doe_knowledge(
            topic="design_selection",
            context={"n_factors": 5, "goal": "screening"},
        )
        assert result["n_results"] >= 1
        rule = result["results"][0]
        assert rule["recommendation"]["primary"] == "fractional_factorial"

    def test_rsm_sequential(self):
        result = doe_knowledge(
            topic="design_selection",
            context={"n_factors": 3, "goal": "optimization", "sequential": True},
        )
        assert result["n_results"] >= 1
        rule = result["results"][0]
        assert rule["recommendation"]["primary"] == "ccd"

    def test_rsm_fresh_start(self):
        result = doe_knowledge(
            topic="design_selection",
            context={"n_factors": 4, "goal": "optimization", "sequential": False},
        )
        assert result["n_results"] >= 1
        rule = result["results"][0]
        assert rule["recommendation"]["primary"] == "box_behnken"

    def test_two_factor_study(self):
        result = doe_knowledge(
            topic="design_selection",
            context={"n_factors": 2, "goal": "optimization"},
        )
        assert result["n_results"] >= 1
        rule = result["results"][0]
        assert rule["recommendation"]["primary"] == "full_factorial"

    def test_hard_to_change_factors(self):
        result = doe_knowledge(
            topic="design_selection",
            context={"has_hard_to_change": True},
        )
        assert result["n_results"] >= 1
        rule = result["results"][0]
        assert rule["recommendation"]["primary"] == "d_optimal"
        assert "hard_to_change" in rule["recommendation"]["rationale"]

    def test_design_selection_fallback_to_keyword(self):
        """When no context matches, falls back to keyword search."""
        result = doe_knowledge(
            query="Box-Behnken design",
            topic="design_selection",
        )
        assert result["n_results"] >= 1

    def test_detail_levels_differ(self):
        ctx = {"n_factors": 7, "budget": 15, "goal": "screening"}
        novice = doe_knowledge(topic="design_selection", context=ctx, detail_level="novice")
        expert = doe_knowledge(topic="design_selection", context=ctx, detail_level="expert")
        # Both should return results
        assert novice["n_results"] >= 1
        assert expert["n_results"] >= 1
        # Explanations should differ
        assert novice["results"][0]["explanation"] != expert["results"][0]["explanation"]


# ---------------------------------------------------------------------------
# Troubleshooting / diagnostics
# ---------------------------------------------------------------------------


class TestTroubleshooting:
    """Test diagnostic pattern queries."""

    def test_funnel_residuals(self):
        result = doe_knowledge(query="funnel residuals", topic="troubleshooting")
        assert result["n_results"] >= 1
        diag = result["results"][0]
        assert diag["type"] == "diagnostic"
        assert "heteroscedasticity" in diag["indicates"][0]["problem"]

    def test_curved_residuals(self):
        result = doe_knowledge(query="curved residual pattern", topic="troubleshooting")
        assert result["n_results"] >= 1

    def test_all_diagnostics_no_query(self):
        """With no query, return all diagnostics."""
        result = doe_knowledge(topic="troubleshooting")
        assert result["n_results"] >= 5

    def test_lack_of_fit(self):
        result = doe_knowledge(query="lack of fit", topic="diagnostics")
        assert result["n_results"] >= 1


# ---------------------------------------------------------------------------
# Statistical concepts
# ---------------------------------------------------------------------------


class TestStatisticalConcepts:
    """Test concept definition queries."""

    def test_resolution_concept(self):
        result = doe_knowledge(query="resolution", topic="statistical_concepts")
        assert result["n_results"] >= 1
        concept = result["results"][0]
        assert concept["type"] == "concept"
        assert concept["title"] == "Design Resolution"

    def test_aliasing_concept(self):
        result = doe_knowledge(query="aliasing confounding", topic="statistical_concepts")
        assert result["n_results"] >= 1
        titles = [r["title"] for r in result["results"]]
        assert any("Aliasing" in t for t in titles)

    def test_all_concepts_no_query(self):
        result = doe_knowledge(topic="statistical_concepts")
        assert result["n_results"] >= 5

    def test_concept_detail_levels(self):
        novice = doe_knowledge(query="randomization", topic="statistical_concepts", detail_level="novice")
        expert = doe_knowledge(query="randomization", topic="statistical_concepts", detail_level="expert")
        assert novice["n_results"] >= 1
        assert expert["n_results"] >= 1
        assert novice["results"][0]["content"] != expert["results"][0]["content"]


# ---------------------------------------------------------------------------
# Design types
# ---------------------------------------------------------------------------


class TestDesignTypes:
    """Test design type queries."""

    def test_specific_design_type(self):
        result = doe_knowledge(query="Box-Behnken", topic="design_types")
        assert result["n_results"] >= 1
        dt = result["results"][0]
        assert dt["type"] == "design_type"
        assert "Box-Behnken" in dt["display_name"]

    def test_all_design_types_no_query(self):
        result = doe_knowledge(topic="design_types")
        assert result["n_results"] >= 5

    def test_design_type_has_properties(self):
        result = doe_knowledge(query="CCD", topic="design_types")
        assert result["n_results"] >= 1
        dt = result["results"][0]
        assert "properties" in dt
        assert "min_runs" in dt
        assert "advantages" in dt


# ---------------------------------------------------------------------------
# Generic (no topic) queries
# ---------------------------------------------------------------------------


class TestGenericQueries:
    """Test broad keyword search without topic filtering."""

    def test_generic_query_returns_results(self):
        result = doe_knowledge(query="factorial design")
        assert result["n_results"] >= 1
        assert result["topic"] == ""

    def test_generic_query_mixed_types(self):
        result = doe_knowledge(query="resolution aliasing")
        types = {r["type"] for r in result["results"]}
        # Should find both concepts and design types
        assert len(types) >= 1

    def test_generic_query_finds_a_diagnostic(self):
        """Without a topic, a residual-pattern word reaches the diagnostics too."""
        result = doe_knowledge(query="funnel")
        diagnostics = [r for r in result["results"] if r["type"] == "diagnostic"]
        assert diagnostics
        assert set(diagnostics[0]) == {"type", "id", "display_name", "visual_pattern"}

    def test_design_selection_keyword_fallback_returns_only_design_types(self):
        """Without context, design selection searches by keyword and keeps only the design types it hits."""
        top_hits = _keyword_search(load_knowledge_graph(), "aliasing")[:5]
        assert ("concept", "aliasing") in {(node_type, node_id) for node_type, node_id, _ in top_hits}
        result = doe_knowledge(query="aliasing", topic="design_selection")
        design_type_hits = [node_id for node_type, node_id, _ in top_hits if node_type == "design_type"]
        assert design_type_hits
        assert [r["id"] for r in result["results"]] == design_type_hits


class TestQueryHelpers:
    """The text and topic helpers behind every query."""

    @pytest.mark.parametrize(
        ("content", "expected"),
        [
            pytest.param({"novice": " simple ", "expert": "deep"}, "simple", id="intermediate-missing-falls-to-novice"),
            pytest.param({"expert": "deep"}, "deep", id="only-expert"),
            pytest.param({"summary": " other "}, "other", id="unknown-level-uses-the-first"),
            pytest.param({}, "", id="no-text"),
        ],
    )
    def test_extract_detail_falls_back(self, content: dict[str, str], expected: str) -> None:
        """A missing level falls back to intermediate, novice, expert, then any text, then the empty string."""
        assert _extract_detail(content, "intermediate") == expected

    def test_design_properties_lists_only_concepts(self) -> None:
        """An entry of another type filed under the design-properties topic is not returned as a concept."""
        concept = ConceptNode(id="resolution", title="Design Resolution", topic="design_properties", content={})
        graph = KnowledgeGraph(
            concepts={"resolution": concept},
            topic_index={"design_properties": ["fold_over_rule", "resolution"]},
        )
        results = query_design_properties(graph, "", "intermediate")
        assert [r["id"] for r in results] == ["resolution"]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Test that invalid inputs raise appropriate errors."""

    def test_invalid_detail_level(self):
        with pytest.raises(ValueError, match="detail_level"):
            doe_knowledge(query="test", detail_level="beginner")

    def test_invalid_topic(self):
        with pytest.raises(ValueError, match="topic"):
            doe_knowledge(query="test", topic="nonexistent_topic")

    def test_empty_query_and_no_topic(self):
        """Empty query with no topic should return empty results, not crash."""
        result = doe_knowledge()
        assert result["n_results"] == 0
        assert result["results"] == []


# ---------------------------------------------------------------------------
# Return structure
# ---------------------------------------------------------------------------


class TestReturnStructure:
    """Verify the result dict has the expected shape."""

    def test_result_keys(self):
        result = doe_knowledge(query="CCD", topic="design_types")
        assert "results" in result
        assert "query" in result
        assert "topic" in result
        assert "detail_level" in result
        assert "n_results" in result

    def test_n_results_matches_list_length(self):
        result = doe_knowledge(query="resolution", topic="statistical_concepts")
        assert result["n_results"] == len(result["results"])

    def test_detail_level_echoed(self):
        result = doe_knowledge(query="CCD", detail_level="expert")
        assert result["detail_level"] == "expert"


# ---------------------------------------------------------------------------
# Tool spec integration
# ---------------------------------------------------------------------------


class TestToolSpecIntegration:
    """Verify doe_knowledge is registered and callable via tool_spec."""

    def test_tool_registered(self):
        from process_improve.tool_spec import get_tool_specs

        specs = get_tool_specs()
        names = [s["name"] for s in specs]
        assert "doe_knowledge" in names

    def test_execute_tool_call(self):
        from process_improve.tool_spec import execute_tool_call

        result = execute_tool_call(
            "doe_knowledge",
            {
                "query": "resolution",
                "topic": "statistical_concepts",
            },
        )
        assert "results" in result
        assert result["n_results"] >= 1


# ---------------------------------------------------------------------------
# Topic coverage: every documented topic should be queryable
# ---------------------------------------------------------------------------


class TestAllTopicsQueryable:
    """Each valid topic routes to a handler and returns a well-formed result."""

    @pytest.mark.parametrize(
        "topic",
        [
            "design_properties",
            "optimization",
            "response_surface",
            "screening",
            "analysis_methods",
        ],
    )
    def test_topic_returns_structured_result(self, topic: str) -> None:
        result = doe_knowledge(query="design", topic=topic)
        assert result["topic"] == topic
        assert "results" in result
        assert result["n_results"] == len(result["results"])

    def test_design_properties_without_query(self) -> None:
        """design_properties with no query returns all design-property concepts."""
        result = doe_knowledge(topic="design_properties")
        assert isinstance(result["results"], list)

    def test_design_properties_with_query_narrows(self) -> None:
        result = doe_knowledge(query="resolution", topic="design_properties")
        assert isinstance(result["results"], list)

    @pytest.mark.parametrize("detail_level", ["novice", "intermediate", "expert"])
    def test_detail_levels_for_design_properties(self, detail_level: str) -> None:
        result = doe_knowledge(query="resolution", topic="design_properties", detail_level=detail_level)
        assert result["detail_level"] == detail_level

    def test_troubleshooting_keyword_search_paths(self) -> None:
        """A broad troubleshooting query exercises the multi-node-type result builder."""
        result = doe_knowledge(query="design model residual", topic="troubleshooting")
        assert isinstance(result["results"], list)


# ---------------------------------------------------------------------------
# Statements in the knowledge base that the library can check (#152-#155)
# ---------------------------------------------------------------------------


def _design_type(design_id: str):
    return load_knowledge_graph().design_types[design_id]


def test_rotatable_axial_distance_is_the_fourth_root_of_the_cube_size() -> None:
    """The default (intermediate) text gave alpha = k^0.25: 1.19 for k=2 where a rotatable CCD needs 1.414."""
    text = _design_type("ccd").description["intermediate"]
    assert "k^0.25" not in text
    assert "(2^k)^(1/4)" in text


def test_plackett_burman_partial_alias_coefficients() -> None:
    """The coefficients are 1/3 in the 12-run design and 0.2 or 0.6 in the 20-run one, not 1/(N-1)."""
    import itertools

    pbdesign = pytest.importorskip("pyDOE3").pbdesign
    coefficients = {}
    for k in (11, 19):
        x = pbdesign(k)
        n = len(x)
        coefficients[n] = {
            round(abs(x[:, i] @ (x[:, a] * x[:, b])) / n, 4)
            for i in range(k)
            for a, b in itertools.combinations([j for j in range(k) if j != i], 2)
        }
    assert coefficients == {12: {round(1 / 3, 4)}, 20: {0.2, 0.6}}
    text = _design_type("plackett_burman").description["expert"]
    assert "1/(N-1)" not in text
    assert "1/3" in text
    assert "0.6" in text


def test_min_run_examples_are_the_designs_the_library_builds() -> None:
    """Four factors cannot be placed in four runs at resolution III, and OMARS for k=3 has 9 runs, not 13."""
    from process_improve.experiments import generate_design
    from process_improve.experiments.factor import Factor

    def factors(k: int) -> list[Factor]:
        return [Factor(name=f"X{i}", low=0, high=1) for i in range(k)]

    for key, runs in _design_type("fractional_factorial").min_runs["examples"].items():
        k, _, roman = key.split("_")
        resolution = {"III": 3, "IV": 4, "V": 5}[roman]
        built = generate_design(factors(int(k)), design_type="fractional_factorial", resolution=resolution)
        assert built.n_runs - 3 == runs, key  # generate_design adds 3 centre points by default
    for design_id, design_type in (("omars", "omars"), ("dsd", "dsd")):
        for k, runs in _design_type(design_id).min_runs["examples"].items():
            assert generate_design(factors(int(k)), design_type=design_type).n_runs == runs, (design_id, k)


def test_foldover_raises_the_resolution_of_a_resolution_iii_fraction_only() -> None:
    """Folding over a resolution IV fraction reproduces its runs; the text claimed it always gains one."""
    text = _design_type("fractional_factorial").description["expert"]
    assert "raises resolution by one" not in text
    assert "resolution III" in text
    assert "resolution IV fraction does not" in text


# ---------------------------------------------------------------------------
# A topic restricts the results to that topic's entries (#151)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("topic", ["analysis_methods", "optimization", "screening", "response_surface"])
def test_a_topic_restricts_the_results(topic: str) -> None:
    graph = load_knowledge_graph()
    allowed = set(graph.topic_index[topic])
    result = doe_knowledge(query="desirability center points resolution effect screening design", topic=topic)
    assert result["results"]
    assert {r["id"] for r in result["results"]} <= allowed
    assert {r["id"] for r in doe_knowledge(topic=topic)["results"]} == allowed


@pytest.mark.parametrize("topic", ["worked_examples", "interpretation"])
def test_topics_without_data_are_not_offered(topic: str) -> None:
    with pytest.raises(ValueError, match="topic must be one of"):
        doe_knowledge(query="design", topic=topic)
