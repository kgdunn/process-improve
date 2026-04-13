"""YAML loader and in-memory query engine for the DOE knowledge graph.

Loads YAML data files at first access (lazy singleton), builds keyword
and topic indices, and provides filtered traversal for ``doe_knowledge()``.
"""

from __future__ import annotations

import operator as _op
import re
from pathlib import Path
from typing import Any

import yaml

from process_improve.experiments.knowledge.models import (
    ConceptNode,
    DecisionRuleNode,
    DesignTypeNode,
    DiagnosticNode,
    InterpretationGuide,
    KnowledgeGraph,
    WorkedExample,
)

_DATA_DIR = Path(__file__).parent / "data"
_GRAPH: KnowledgeGraph | None = None


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------


def _load_yaml(filename: str) -> list[dict[str, Any]]:
    """Load a YAML file from the data directory and return its contents."""
    path = _DATA_DIR / filename
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data if isinstance(data, list) else []


def _index_texts(
    index: dict[str, list[tuple[str, str]]],
    node_type: str,
    node_id: str,
    texts: list[str],
) -> None:
    """Add all tokens from *texts* to the keyword *index*."""
    for text in texts:
        for raw_token in text.split():
            clean_token = raw_token.lower().strip(".,;:!?()\"'")
            if len(clean_token) >= 3:
                index.setdefault(clean_token, []).append((node_type, node_id))


def _build_keyword_index(graph: KnowledgeGraph) -> dict[str, list[tuple[str, str]]]:
    """Build a keyword -> [(node_type, node_id), ...] lookup.

    Keywords are extracted from *id*, *display_name* / *title*, and
    *description* / *content* text (split on whitespace, lowered, length >= 3).
    """
    index: dict[str, list[tuple[str, str]]] = {}

    for dt in graph.design_types.values():
        _index_texts(index, "design_type", dt.id, [
            dt.id.replace("_", " "), dt.display_name, *dt.description.values(),
        ])

    for diag in graph.diagnostics.values():
        _index_texts(index, "diagnostic", diag.id, [
            diag.id.replace("_", " "), diag.display_name, diag.visual_pattern,
        ])

    for concept in graph.concepts.values():
        _index_texts(index, "concept", concept.id, [
            concept.id.replace("_", " "), concept.title, *concept.content.values(),
        ])

    for guide in graph.interpretation_guides.values():
        _index_texts(index, "interpretation", guide.id, [
            guide.id.replace("_", " "), guide.title,
        ])

    # De-duplicate entries within each keyword
    return {k: list(dict.fromkeys(v)) for k, v in index.items()}


def _build_topic_index(graph: KnowledgeGraph) -> dict[str, list[str]]:
    """Build a topic -> [node_id, ...] lookup."""
    index: dict[str, list[str]] = {}

    for dt in graph.design_types.values():
        index.setdefault(dt.category, []).append(dt.id)
        index.setdefault("design_types", []).append(dt.id)

    for rule in graph.decision_rules:
        index.setdefault(rule.topic, []).append(rule.id)
        index.setdefault("decision_rules", []).append(rule.id)

    for diag in graph.diagnostics.values():
        index.setdefault("troubleshooting", []).append(diag.id)
        index.setdefault("diagnostics", []).append(diag.id)

    for concept in graph.concepts.values():
        index.setdefault(concept.topic, []).append(concept.id)
        index.setdefault("statistical_concepts", []).append(concept.id)

    for guide in graph.interpretation_guides.values():
        index.setdefault(guide.topic, []).append(guide.id)
        index.setdefault("interpretation", []).append(guide.id)

    for example in graph.worked_examples.values():
        index.setdefault(example.topic, []).append(example.id)
        index.setdefault("worked_examples", []).append(example.id)

    return index


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------


def load_knowledge_graph() -> KnowledgeGraph:
    """Load all YAML data files and build the in-memory knowledge graph.

    The graph is cached as a module-level singleton.  Call
    ``reload_knowledge_graph()`` to force a fresh load.
    """
    global _GRAPH  # noqa: PLW0603
    if _GRAPH is not None:
        return _GRAPH

    graph = KnowledgeGraph()

    # Design types
    for entry in _load_yaml("design_types.yaml"):
        node = DesignTypeNode(
            id=entry["id"],
            display_name=entry["display_name"],
            category=entry["category"],
            description=entry.get("description", {}),
            suitable_for=entry.get("suitable_for", []),
            properties=entry.get("properties", {}),
            min_runs=entry.get("min_runs", {}),
            supports_models=entry.get("supports_models", []),
            can_augment_to=entry.get("can_augment_to", []),
            advantages=entry.get("advantages", []),
            disadvantages=entry.get("disadvantages", []),
            common_misconceptions=entry.get("common_misconceptions", []),
            references=entry.get("references", []),
        )
        graph.design_types[node.id] = node

    # Decision rules
    for entry in _load_yaml("decision_rules.yaml"):
        node = DecisionRuleNode(
            id=entry["id"],
            topic=entry["topic"],
            description=entry["description"],
            conditions=entry.get("conditions", []),
            recommend=entry.get("recommend", {}),
            explanation=entry.get("explanation", {}),
        )
        graph.decision_rules.append(node)

    # Diagnostics
    for entry in _load_yaml("diagnostics.yaml"):
        node = DiagnosticNode(
            id=entry["id"],
            display_name=entry["display_name"],
            visual_pattern=entry["visual_pattern"],
            indicates=entry.get("indicates", []),
            remedies=entry.get("remedies", []),
            severity=entry.get("severity", "moderate"),
        )
        graph.diagnostics[node.id] = node

    # Concepts
    for entry in _load_yaml("concepts.yaml"):
        node = ConceptNode(
            id=entry["id"],
            title=entry["title"],
            topic=entry["topic"],
            content=entry.get("content", {}),
            related_to=entry.get("related_to", []),
            related_questions=entry.get("related_questions", []),
        )
        graph.concepts[node.id] = node

    # Interpretation guides
    for entry in _load_yaml("interpretation_guides.yaml"):
        node = InterpretationGuide(
            id=entry["id"],
            topic=entry["topic"],
            title=entry["title"],
            content=entry.get("content", {}),
            related_questions=entry.get("related_questions", []),
        )
        graph.interpretation_guides[node.id] = node

    # Worked examples
    for entry in _load_yaml("worked_examples.yaml"):
        node = WorkedExample(
            id=entry["id"],
            title=entry["title"],
            demonstrates=entry.get("demonstrates", []),
            topic=entry["topic"],
            scenario=entry.get("scenario", ""),
            steps=entry.get("steps", []),
            related_questions=entry.get("related_questions", []),
            detail_level=entry.get("detail_level", "intermediate"),
        )
        graph.worked_examples[node.id] = node

    # Build indices
    graph.keyword_index = _build_keyword_index(graph)
    graph.topic_index = _build_topic_index(graph)

    _GRAPH = graph
    return graph


def reload_knowledge_graph() -> KnowledgeGraph:
    """Force a fresh load of the knowledge graph (useful for testing)."""
    global _GRAPH  # noqa: PLW0603
    _GRAPH = None
    return load_knowledge_graph()


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

_VALID_DETAIL_LEVELS = {"novice", "intermediate", "expert"}
_VALID_TOPICS = {
    "design_selection",
    "design_properties",
    "design_types",
    "analysis_methods",
    "interpretation",
    "troubleshooting",
    "diagnostics",
    "optimization",
    "statistical_concepts",
    "screening",
    "response_surface",
    "worked_examples",
}


def _extract_detail(content: dict[str, str], detail_level: str) -> str:
    """Return the text for the requested detail level, falling back gracefully."""
    if detail_level in content:
        return content[detail_level].strip()
    # Fallback: intermediate -> novice -> expert -> first available
    for fallback in ("intermediate", "novice", "expert"):
        if fallback in content:
            return content[fallback].strip()
    if content:
        return next(iter(content.values())).strip()
    return ""


_OPS: dict[str, Any] = {
    "==": _op.eq,
    "!=": _op.ne,
    ">=": _op.ge,
    "<=": _op.le,
    ">": _op.gt,
    "<": _op.lt,
    "in": lambda a, b: a in b,
}


def _match_condition(condition: dict[str, Any], context: dict[str, Any]) -> bool:
    """Check whether a single condition matches the given context."""
    key = condition.get("key", "")
    op = condition.get("op", "==")
    value = condition.get("value")

    if key not in context:
        return False

    fn = _OPS.get(op)
    return fn(context[key], value) if fn else False


def _match_all_conditions(conditions: list[dict[str, Any]], context: dict[str, Any]) -> bool:
    """Return True if all conditions match the context."""
    return all(_match_condition(c, context) for c in conditions)


def _tokenize_query(query: str) -> list[str]:
    """Split a query string into lowercase tokens of length >= 3."""
    tokens = re.findall(r"[a-zA-Z0-9_]+", query.lower())
    return [t for t in tokens if len(t) >= 3]


def _keyword_search(graph: KnowledgeGraph, query: str) -> list[tuple[str, str, int]]:
    """Return (node_type, node_id, match_count) ranked by relevance."""
    tokens = _tokenize_query(query)
    if not tokens:
        return []

    scores: dict[tuple[str, str], int] = {}
    for token in tokens:
        for node_type, node_id in graph.keyword_index.get(token, []):
            key = (node_type, node_id)
            scores[key] = scores.get(key, 0) + 1

    ranked = [(nt, nid, score) for (nt, nid), score in scores.items()]
    ranked.sort(key=lambda x: x[2], reverse=True)
    return ranked


# ---------------------------------------------------------------------------
# Topic-specific query handlers
# ---------------------------------------------------------------------------


def query_design_selection(
    graph: KnowledgeGraph,
    query: str,
    context: dict[str, Any] | None,
    detail_level: str,
) -> list[dict[str, Any]]:
    """Find matching decision rules based on context and/or keyword search."""
    results: list[dict[str, Any]] = []

    if context:
        results.extend(
            {
                "type": "decision_rule",
                "id": rule.id,
                "description": rule.description,
                "recommendation": rule.recommend,
                "explanation": _extract_detail(rule.explanation, detail_level),
            }
            for rule in graph.decision_rules
            if _match_all_conditions(rule.conditions, context)
        )

    # If no context matches or no context provided, fall back to keyword search
    if not results and query:
        keyword_hits = _keyword_search(graph, query)
        for node_type, node_id, _score in keyword_hits[:5]:
            if node_type == "design_type" and node_id in graph.design_types:
                dt = graph.design_types[node_id]
                results.append({
                    "type": "design_type",
                    "id": dt.id,
                    "display_name": dt.display_name,
                    "description": _extract_detail(dt.description, detail_level),
                    "suitable_for": dt.suitable_for,
                    "properties": dt.properties,
                    "min_runs": dt.min_runs,
                })

    return results


def query_design_properties(
    graph: KnowledgeGraph,
    query: str,
    detail_level: str,
) -> list[dict[str, Any]]:
    """Return concepts and design types matching design property queries."""
    results: list[dict[str, Any]] = []

    # Check concepts with topic == design_properties
    for concept_id in graph.topic_index.get("design_properties", []):
        concept = graph.concepts.get(concept_id)
        if concept:
            results.append({
                "type": "concept",
                "id": concept.id,
                "title": concept.title,
                "content": _extract_detail(concept.content, detail_level),
                "related_to": concept.related_to,
            })

    # Narrow down via keyword search if a query is provided
    if query:
        keyword_hits = _keyword_search(graph, query)
        hit_ids = {nid for _, nid, _ in keyword_hits[:10]}
        results = [r for r in results if r["id"] in hit_ids] or results

    return results


def query_troubleshooting(
    graph: KnowledgeGraph,
    query: str,
    detail_level: str,
) -> list[dict[str, Any]]:
    """Return diagnostic patterns matching the query."""
    results: list[dict[str, Any]] = []

    if query:
        keyword_hits = _keyword_search(graph, query)
        seen = set()
        for node_type, node_id, _score in keyword_hits:
            if node_type == "diagnostic" and node_id not in seen:
                seen.add(node_id)
                diag = graph.diagnostics[node_id]
                remedies = [
                    r for r in diag.remedies
                    if detail_level == "expert" or r.get("detail", "") != ""
                ]
                results.append({
                    "type": "diagnostic",
                    "id": diag.id,
                    "display_name": diag.display_name,
                    "visual_pattern": diag.visual_pattern,
                    "indicates": diag.indicates,
                    "remedies": remedies,
                    "severity": diag.severity,
                })
    else:
        # No query: return all diagnostics
        results.extend(
            {
                "type": "diagnostic",
                "id": diag.id,
                "display_name": diag.display_name,
                "visual_pattern": diag.visual_pattern,
                "indicates": diag.indicates,
                "remedies": diag.remedies,
                "severity": diag.severity,
            }
            for diag in graph.diagnostics.values()
        )

    return results


def query_statistical_concepts(
    graph: KnowledgeGraph,
    query: str,
    detail_level: str,
) -> list[dict[str, Any]]:
    """Return concept definitions matching the query."""
    results: list[dict[str, Any]] = []

    if query:
        keyword_hits = _keyword_search(graph, query)
        seen = set()
        for node_type, node_id, _score in keyword_hits:
            if node_type == "concept" and node_id not in seen:
                seen.add(node_id)
                concept = graph.concepts[node_id]
                results.append({
                    "type": "concept",
                    "id": concept.id,
                    "title": concept.title,
                    "content": _extract_detail(concept.content, detail_level),
                    "related_to": concept.related_to,
                })
    else:
        results.extend(
            {
                "type": "concept",
                "id": concept.id,
                "title": concept.title,
                "content": _extract_detail(concept.content, detail_level),
                "related_to": concept.related_to,
            }
            for concept in graph.concepts.values()
        )

    return results


def query_design_types(
    graph: KnowledgeGraph,
    query: str,
    detail_level: str,
) -> list[dict[str, Any]]:
    """Return design type entries matching the query."""
    results: list[dict[str, Any]] = []

    if query:
        keyword_hits = _keyword_search(graph, query)
        seen = set()
        for node_type, node_id, _score in keyword_hits:
            if node_type == "design_type" and node_id not in seen:
                seen.add(node_id)
                dt = graph.design_types[node_id]
                results.append(_format_design_type(dt, detail_level))
    else:
        results.extend(_format_design_type(dt, detail_level) for dt in graph.design_types.values())

    return results


def _format_design_type(dt: DesignTypeNode, detail_level: str) -> dict[str, Any]:
    """Format a DesignTypeNode into a result dict."""
    return {
        "type": "design_type",
        "id": dt.id,
        "display_name": dt.display_name,
        "category": dt.category,
        "description": _extract_detail(dt.description, detail_level),
        "suitable_for": dt.suitable_for,
        "properties": dt.properties,
        "min_runs": dt.min_runs,
        "supports_models": dt.supports_models,
        "advantages": dt.advantages,
        "disadvantages": dt.disadvantages,
        "common_misconceptions": dt.common_misconceptions,
        "references": dt.references,
    }


def query_generic(
    graph: KnowledgeGraph,
    query: str,
    detail_level: str,
) -> list[dict[str, Any]]:
    """Fallback: keyword search across all node types."""
    results: list[dict[str, Any]] = []
    keyword_hits = _keyword_search(graph, query)

    seen = set()
    for node_type, node_id, _score in keyword_hits[:10]:
        if (node_type, node_id) in seen:
            continue
        seen.add((node_type, node_id))

        if node_type == "design_type" and node_id in graph.design_types:
            dt = graph.design_types[node_id]
            results.append({
                "type": "design_type",
                "id": dt.id,
                "display_name": dt.display_name,
                "description": _extract_detail(dt.description, detail_level),
            })
        elif node_type == "concept" and node_id in graph.concepts:
            concept = graph.concepts[node_id]
            results.append({
                "type": "concept",
                "id": concept.id,
                "title": concept.title,
                "content": _extract_detail(concept.content, detail_level),
            })
        elif node_type == "diagnostic" and node_id in graph.diagnostics:
            diag = graph.diagnostics[node_id]
            results.append({
                "type": "diagnostic",
                "id": diag.id,
                "display_name": diag.display_name,
                "visual_pattern": diag.visual_pattern,
            })
        elif node_type == "interpretation" and node_id in graph.interpretation_guides:
            guide = graph.interpretation_guides[node_id]
            results.append({
                "type": "interpretation_guide",
                "id": guide.id,
                "title": guide.title,
                "content": _extract_detail(guide.content, detail_level),
            })

    return results
