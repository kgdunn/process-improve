# (c) Kevin Dunn, 2010-2026. MIT License.
"""MCP tool wrapper: ``recommend_strategy`` (ENG-02)."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from process_improve.experiments._tools import _TOOL_EXPECTED_EXCEPTIONS, _register, logger
from process_improve.tool_spec import clean, tool_spec


class RecommendStrategyInput(BaseModel):
    """Input contract for ``recommend_strategy``."""

    model_config = ConfigDict(extra="forbid")

    factors: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        description="All candidate experimental factors.",
    )
    responses: list[dict[str, Any]] | None = Field(
        None,
        description="Response variables with optimisation goals.",
    )
    budget: int | None = Field(
        None,
        ge=1,
        description="Total run budget across all stages. Omit for ideal allocation.",
    )
    constraints: list[dict[str, Any]] | None = Field(
        None,
        description="Factor-space constraints. Each entry: expression (str), optional type ('linear'|'nonlinear').",
    )
    hard_to_change_factors: list[str] | None = Field(
        None,
        description="Factor names that are expensive to reset (triggers split-plot).",
    )
    prior_knowledge: str | None = Field(
        None,
        description=(
            "Free-text description of prior knowledge, e.g. "
            "'Published literature confirms Temperature and pH are significant.' "
            "or 'No prior data - first time running this process.'"
        ),
    )
    existing_data: list[dict[str, Any]] | None = Field(
        None,
        description="Prior experimental data as list of dicts (optional).",
    )
    domain: Literal[
        "pharma_formulation",
        "fermentation",
        "food_science",
        "extraction",
        "analytical_method",
        "cell_culture",
        "bioprocess",
        "general",
    ] | None = Field(
        None,
        description="Application domain for domain-specific adjustments. Default: 'general'.",
    )
    detail_level: Literal["novice", "intermediate"] = Field(
        "intermediate",
        description="Depth of explanations in the output.",
    )


@tool_spec(
    name="recommend_strategy",
    description=(
        "Recommend a multi-stage experimental strategy given a DOE problem description. "
        "Given factors, responses, budget, constraints, domain, and prior knowledge, "
        "applies deterministic decision rules to recommend a staged experimental plan "
        "(screening then optimisation then confirmation). "
        "Returns a structured strategy with stage-by-stage design types, estimated run counts, "
        "transition rules, budget allocation, assumptions, risks, and alternative approaches. "
        "Use this when the user asks 'How should I plan my experiments?' or 'What design strategy "
        "should I use for N factors?'"
    ),
    input_model=RecommendStrategyInput,
    examples="""
    # "I have 7 factors - how do I plan my experiments?"
        -> ``recommend_strategy(factors=[{"name": "A", "low": 0, "high": 100}, ...7 factors...],
                budget=40, domain="general")``

    # "Optimize fermentation with 7 factors in ~40 runs"
        -> ``recommend_strategy(factors=[{"name": "pH", "low": 5, "high": 8}, ...],
                responses=[{"name": "Yield", "goal": "maximize"}],
                budget=40, domain="fermentation")``
    """,
    category="experiments",
)
def recommend_strategy_tool(spec: RecommendStrategyInput) -> dict[str, Any]:
    """Recommend a multi-stage experimental strategy."""
    try:
        from process_improve.experiments.factor import Constraint, Factor, Response  # noqa: PLC0415
        from process_improve.experiments.strategy import recommend_strategy  # noqa: PLC0415

        factor_objects = [Factor(**f) for f in spec.factors]
        response_objects = [Response(**r) for r in spec.responses] if spec.responses else None
        constraint_objects = (
            [Constraint(**c) for c in spec.constraints] if spec.constraints else None
        )

        df = pd.DataFrame(spec.existing_data) if spec.existing_data else None

        result = recommend_strategy(
            factors=factor_objects,
            responses=response_objects,
            budget=spec.budget,
            constraints=constraint_objects,
            hard_to_change_factors=spec.hard_to_change_factors,
            prior_knowledge=spec.prior_knowledge,
            existing_data=df,
            domain=spec.domain,
            detail_level=spec.detail_level,
        )
        return clean(result)
    except _TOOL_EXPECTED_EXCEPTIONS as e:
        logger.exception("Tool recommend_strategy failed")
        return {"error": str(e)}


_register("recommend_strategy")
