# (c) Kevin Dunn, 2010-2026. MIT License.
"""MCP tool wrapper: ``doe_knowledge`` (ENG-02)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from process_improve.experiments._tools import _TOOL_EXPECTED_EXCEPTIONS, _register, logger
from process_improve.tool_spec import clean, tool_spec


class DoeKnowledgeInput(BaseModel):
    """Input contract for ``doe_knowledge``."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        "",
        description=(
            "Natural-language query, e.g. 'What is design resolution?' or "
            "'funnel shaped residuals'."
        ),
    )
    topic: Literal[
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
        "",
    ] = Field(
        "",
        description="Restrict search to a topic. Leave empty for broad search.",
    )
    context: dict[str, Any] | None = Field(
        None,
        description=(
            "Experimental context for design-selection queries. "
            "Keys: n_factors (int), budget (int), goal ('screening'|'optimization'), "
            "sequential (bool), curvature_important (bool), has_hard_to_change (bool)."
        ),
    )
    detail_level: Literal["novice", "intermediate", "expert"] = Field(
        "intermediate",
        description="Depth of explanation: novice, intermediate, or expert.",
    )


@tool_spec(
    name="doe_knowledge",
    description=(
        "Retrieve DOE (Design of Experiments) domain knowledge: design-type descriptions, "
        "design-selection decision logic, statistical concept definitions, residual-diagnostic "
        "troubleshooting guides, interpretation guidance, and worked examples. "
        "Use this tool whenever the user asks a conceptual DOE question, needs help choosing "
        "a design, or wants to understand how to interpret DOE results."
    ),
    input_model=DoeKnowledgeInput,
    examples="""
    # "Which design should I use for 7 screening factors with a budget of 15 runs?"
        -> ``doe_knowledge(query="screening 7 factors 15 runs",
                topic="design_selection",
                context={"n_factors": 7, "budget": 15, "goal": "screening"})``

    # "What is design resolution?"
        -> ``doe_knowledge(query="What is design resolution?",
                topic="statistical_concepts")``
    """,
    category="experiments",
)
def doe_knowledge_tool(spec: DoeKnowledgeInput) -> dict[str, Any]:
    """Query the DOE knowledge graph."""
    try:
        from process_improve.experiments.knowledge import doe_knowledge  # noqa: PLC0415

        return clean(doe_knowledge(
            query=spec.query,
            topic=spec.topic,
            context=spec.context,
            detail_level=spec.detail_level,
        ))
    except _TOOL_EXPECTED_EXCEPTIONS as e:
        logger.exception("Tool doe_knowledge failed")
        return {"error": str(e)}


_register("doe_knowledge")
