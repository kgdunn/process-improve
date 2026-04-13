"""Public API for the DOE knowledge graph.

Provides :func:`doe_knowledge` — the single entry-point for retrieving
DOE concepts, design selection logic, interpretation guidance, diagnostic
patterns, and worked examples.
"""

from __future__ import annotations

from typing import Any

from process_improve.experiments.knowledge.engine import (
    _VALID_DETAIL_LEVELS,
    _VALID_TOPICS,
    load_knowledge_graph,
    query_design_properties,
    query_design_selection,
    query_design_types,
    query_generic,
    query_statistical_concepts,
    query_troubleshooting,
)


def doe_knowledge(
    query: str = "",
    topic: str = "",
    context: dict[str, Any] | None = None,
    detail_level: str = "intermediate",
) -> dict[str, Any]:
    """Retrieve DOE domain knowledge from the in-memory knowledge graph.

    Parameters
    ----------
    query : str, optional
        Natural-language query (e.g. ``"What is design resolution?"``).
    topic : str, optional
        Restrict the search to a specific topic.  Valid values:

        ``"design_selection"``, ``"design_properties"``, ``"design_types"``,
        ``"analysis_methods"``, ``"interpretation"``, ``"troubleshooting"``,
        ``"diagnostics"``, ``"optimization"``, ``"statistical_concepts"``,
        ``"screening"``, ``"response_surface"``, ``"worked_examples"``.
    context : dict, optional
        Experimental context for design-selection queries.  Recognised keys:

        * ``n_factors`` (int) — number of factors
        * ``budget`` (int) — maximum number of runs
        * ``goal`` (str) — ``"screening"`` | ``"optimization"``
        * ``sequential`` (bool) — whether the design will be augmented later
        * ``curvature_important`` (bool)
        * ``has_hard_to_change`` (bool)
    detail_level : str, default ``"intermediate"``
        One of ``"novice"``, ``"intermediate"``, ``"expert"``.  Controls the
        depth of the returned explanations.

    Returns
    -------
    dict
        A dictionary with keys:

        * ``"results"`` — list of matching knowledge entries
        * ``"query"`` — the original query string
        * ``"topic"`` — the topic used for filtering
        * ``"detail_level"`` — the detail level applied
        * ``"n_results"`` — number of results returned

    Examples
    --------
    >>> from process_improve.experiments.knowledge import doe_knowledge
    >>> result = doe_knowledge(
    ...     query="Which design for 7 screening factors?",
    ...     topic="design_selection",
    ...     context={"n_factors": 7, "budget": 15, "goal": "screening"},
    ... )
    >>> result["n_results"] >= 1
    True

    >>> result = doe_knowledge(query="resolution", topic="statistical_concepts")
    >>> result["results"][0]["title"]
    'Design Resolution'

    See Also
    --------
    process_improve.experiments.knowledge.engine.load_knowledge_graph :
        Low-level access to the full knowledge graph object.
    """
    # Validate inputs
    if detail_level not in _VALID_DETAIL_LEVELS:
        msg = f"detail_level must be one of {sorted(_VALID_DETAIL_LEVELS)}, got {detail_level!r}"
        raise ValueError(msg)

    if topic and topic not in _VALID_TOPICS:
        msg = f"topic must be one of {sorted(_VALID_TOPICS)} or empty, got {topic!r}"
        raise ValueError(msg)

    graph = load_knowledge_graph()

    # Route to topic-specific handler
    results: list[dict[str, Any]]
    if topic == "design_selection":
        results = query_design_selection(graph, query, context, detail_level)
    elif topic in ("troubleshooting", "diagnostics"):
        results = query_troubleshooting(graph, query, detail_level)
    elif topic == "design_properties":
        results = query_design_properties(graph, query, detail_level)
    elif topic == "statistical_concepts":
        results = query_statistical_concepts(graph, query, detail_level)
    elif topic == "design_types":
        results = query_design_types(graph, query, detail_level)
    elif topic:
        # Other valid topics: use generic keyword search filtered by topic
        results = query_generic(graph, query, detail_level)
    else:
        # No topic: broad keyword search
        results = query_generic(graph, query, detail_level)

    return {
        "results": results,
        "query": query,
        "topic": topic,
        "detail_level": detail_level,
        "n_results": len(results),
    }
