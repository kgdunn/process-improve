"""DOE knowledge graph - definitions, decision logic, interpretation guidance.

This subpackage encodes DOE domain knowledge as structured YAML files and
provides an in-memory query engine for retrieving it.

Quick start::

    from process_improve.experiments.knowledge import doe_knowledge

    # Design selection guidance
    result = doe_knowledge(
        query="Which design for 7 screening factors?",
        topic="design_selection",
        context={"n_factors": 7, "budget": 15, "goal": "screening"},
    )

    # Concept definitions
    result = doe_knowledge(
        query="What is design resolution?",
        topic="statistical_concepts",
    )

    # Troubleshooting
    result = doe_knowledge(
        query="funnel shaped residuals",
        topic="troubleshooting",
    )

The book's own graph of terms, with the line of the book behind every term and link, is
:class:`ConceptGraph`, queried in Cypher (it needs ``pip install relate-db``)::

    from process_improve.experiments.knowledge import ConceptGraph

    with ConceptGraph() as graph:
        graph.query("MATCH (:Term {id: 'curvature'})-[:REMEDIED_BY]->(fix) RETURN fix.name AS remedy")
"""

from process_improve.experiments.knowledge.api import doe_knowledge
from process_improve.experiments.knowledge.concept_graph import ConceptGraph

__all__ = ["ConceptGraph", "doe_knowledge"]
