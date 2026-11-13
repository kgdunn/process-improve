# (c) Kevin Dunn, 2010-2026. MIT License.
"""MCP tool wrapper: ``interaction_components``, the two-degree-of-freedom components of three-level interactions."""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from process_improve.experiments._tools import _TOOL_EXPECTED_EXCEPTIONS, _register, logger
from process_improve.tool_spec import clean, tool_spec


class InteractionComponentsInput(BaseModel):
    """Input contract for ``interaction_components``."""

    model_config = ConfigDict(extra="forbid")

    design_matrix: list[dict[str, Any]] = Field(
        ...,
        min_length=3,
        description=(
            "One dict per run: the factors' settings and the response. Replicates and fractions are fine. "
            "Example: [{'A': 15, 'B': 125, 'y': -2}, {'A': 15, 'B': 125, 'y': -1}, ...]"
        ),
    )
    response_column: str = Field(..., description="Name of the response column in design_matrix.")
    # Bounded: k factors give (3^k - 1) / 2 rows, 364 for six.
    factor_names: list[str] = Field(
        ...,
        min_length=1,
        max_length=6,
        description=(
            "The three-level factors, at most six, in the order the components are written in. Each has exactly "
            "three settings, its levels 0, 1 and 2, lowest first. The components of an interaction depend on this "
            "order; their total does not."
        ),
    )


@tool_spec(
    name="interaction_components",
    description=(
        "Split each interaction of three-level factors into its orthogonal components of two degrees of freedom "
        "each, as Montgomery and Davies analyse a 3^k factorial: AB and AB^2 for two factors (Yates's J and I), "
        "ABC, ABC^2, AB^2C and AB^2C^2 for three (Z, Y, X and W). Each component groups the runs by a sum of the "
        "factors' levels modulo 3, and its sum of squares is that of the three group totals. Returns "
        "'components', one row per main effect and per component: 'effect', 'component' (Montgomery's name, "
        "AB2), 'yates' (I(AB) or W(ABC); empty beyond three factors), 'df' (2) and 'sum_of_squares'. Use it to "
        "see which component carries an interaction, or what a design that confounds a component with blocks "
        "or with another effect leaves; for the ANOVA itself use analyze_experiment."
    ),
    input_model=InteractionComponentsInput,
    examples="""
    # "Split the AB interaction of my replicated 3^2 into its components"
        -> ``interaction_components(design_matrix=[{"A": 15, "B": 125, "y": -2}, ...], response_column="y",
                factor_names=["A", "B"])``
    """,
    category="experiments",
)
def interaction_components_tool(spec: InteractionComponentsInput) -> dict[str, Any]:
    """Return the main effects and the components of interaction, with their sums of squares."""
    try:
        from process_improve.experiments.three_level_components import interaction_components  # noqa: PLC0415

        table = interaction_components(pd.DataFrame(spec.design_matrix), spec.response_column, spec.factor_names)
        return clean(
            {
                "factor_names": spec.factor_names,
                "response_name": spec.response_column,
                "components": table.to_dict(orient="records"),
            }
        )
    except _TOOL_EXPECTED_EXCEPTIONS as e:
        logger.exception("Tool interaction_components failed")
        return {"error": str(e)}


_register("interaction_components")
