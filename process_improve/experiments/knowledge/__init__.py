"""DOE knowledge graph — definitions, decision logic, interpretation guidance.

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
"""

from process_improve.experiments.knowledge.api import doe_knowledge

__all__ = ["doe_knowledge"]
