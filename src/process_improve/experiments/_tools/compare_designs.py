# (c) Kevin Dunn, 2010-2026. MIT License.
"""MCP tool wrapper: ``compare_designs``, designs scored for one fixed model on the measures of design quality."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from process_improve.experiments._tools import _TOOL_EXPECTED_EXCEPTIONS, _register, logger
from process_improve.tool_spec import clean, tool_spec


class CompareDesignsInput(BaseModel):
    """Input contract for ``compare_designs``."""

    model_config = ConfigDict(extra="forbid")

    designs: dict[str, list[dict[str, float]]] = Field(
        ...,
        min_length=1,
        description=(
            "The designs to compare, by label: each design's runs in coded units (-1 to +1), one dict per run "
            "with a value per factor, as generate_design returns them under 'design_coded'. Every design has the "
            "same factors."
        ),
    )
    model: Literal["main_quadratic", "main_effects", "interactions", "quadratic"] = Field(
        "main_quadratic",
        description=(
            "The model every design is scored on: 'main_quadratic' (intercept, main effects and squares), "
            "'main_effects', 'interactions' (main effects and two-factor interactions) or 'quadratic' (all of them)."
        ),
    )
    effect_size: float = Field(
        1.0,
        gt=0.0,
        description="The coefficient, in noise standard deviations, whose detection the power measures.",
    )
    significance_level: float = Field(0.05, gt=0.0, lt=1.0, description="The level of the tests behind the power.")
    region: dict[str, Any] | None = Field(
        None,
        description=(
            "The settings that can be run, as DesignRegion.to_dict writes it (factors and constraints), in the "
            "designs' coded units. Default: the whole cube."
        ),
    )
    include_correlations: bool = Field(
        False, description="Also return each design's correlation matrix of its effects (large for many factors)."
    )


@tool_spec(
    name="compare_designs",
    description=(
        "Score designs side by side for one fixed model, without simulation: whether each can fit the model, "
        "its degrees of freedom, the A, D, E, G and I criteria (coefficient and prediction variances), the "
        "largest correlation between second-order terms, and the power to detect a main effect, an "
        "interaction and a square of a given size. Each design is scored at its own size. Returns 'table' "
        "(the measures by design, as text), 'values' (the numbers), 'notes' on each design, and the measures' "
        "definitions. For how the designs do when the model is not known in advance, or terms outside it bias "
        "the effects, use simulate_design_performance."
    ),
    input_model=CompareDesignsInput,
    examples="""
    # "Is a Box-Behnken or a face-centred CCD better for a quadratic in 3 factors?"
        -> ``compare_designs(designs={"BBD": [...], "CCD": [...]}, model="quadratic")``
    """,
    category="experiments",
)
def compare_designs_tool(spec: CompareDesignsInput) -> dict[str, Any]:
    """Score the designs for one model."""
    try:
        from process_improve.experiments.compare import compare_designs  # noqa: PLC0415

        comparison = compare_designs(
            {label: pd.DataFrame(runs) for label, runs in spec.designs.items()},
            spec.model,
            effect_size=spec.effect_size,
            significance_level=spec.significance_level,
            region=spec.region,
        )
        output = comparison.to_dict()
        if not spec.include_correlations:
            output.pop("correlations", None)
        output["table"] = comparison.table.reset_index(names="Measure").to_dict(orient="records")
        return clean(output)
    except _TOOL_EXPECTED_EXCEPTIONS as e:
        logger.exception("Tool compare_designs failed")
        return {"error": str(e)}


_register("compare_designs")
