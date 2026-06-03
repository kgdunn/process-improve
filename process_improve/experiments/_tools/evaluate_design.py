# (c) Kevin Dunn, 2010-2026. MIT License.
"""MCP tool wrapper: ``evaluate_design`` (ENG-02)."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from process_improve.experiments._tools import _TOOL_EXPECTED_EXCEPTIONS, _register, logger
from process_improve.tool_spec import clean, tool_spec


class EvaluateDesignInput(BaseModel):
    """Input contract for ``evaluate_design``."""

    model_config = ConfigDict(extra="forbid")

    design_matrix: list[dict[str, Any]] = Field(
        ...,
        min_length=2,
        description=(
            "List of dictionaries, one per experimental run. Each dict maps "
            "factor name to coded value. Example: [{'A': -1, 'B': -1}, ...]"
        ),
    )
    model: Literal["main_effects", "interactions", "quadratic"] | None = Field(
        None,
        description=(
            "Model type to evaluate against. 'main_effects' = main effects only, "
            "'interactions' = main effects + 2-factor interactions (default), "
            "'quadratic' = interactions + squared terms."
        ),
    )
    metric: str | list[str] = Field(
        "d_efficiency",
        description=(
            "One or more metric names to compute. Default: 'd_efficiency'. "
            "Options include d_efficiency, i_efficiency, g_efficiency, "
            "prediction_variance, vif, condition_number, power, "
            "degrees_of_freedom, alias_structure, confounding, resolution, "
            "defining_relation, clear_effects, minimum_aberration."
        ),
    )
    effect_size: float | None = Field(
        None,
        description="Expected effect size for power calculation.",
    )
    alpha: float = Field(
        0.05,
        description="Significance level (default 0.05).",
    )
    sigma: float | None = Field(
        None,
        description="Estimated noise standard deviation.",
    )


@tool_spec(
    name="evaluate_design",
    description=(
        "Evaluate the quality of an experimental design matrix by computing metrics such as "
        "D-efficiency, G-efficiency, I-efficiency, VIF, condition number, alias structure, "
        "confounding pattern, resolution, power, prediction variance, degrees of freedom, "
        "clear effects, and minimum aberration. "
        "The design_matrix should be a list of dictionaries with factor names as keys and "
        "coded values (-1/+1) as values. "
        "Use this after generating a design to check if it meets quality criteria, or to "
        "compare alternative designs."
    ),
    input_model=EvaluateDesignInput,
    examples="""
    # "What is the D-efficiency of my 2^3 factorial design?"
        -> ``evaluate_design(design_matrix=[{"A":-1,"B":-1,"C":-1}, ...],
                metric="d_efficiency", model="interactions")``

    # "Check VIF and condition number"
        -> ``evaluate_design(design_matrix=[...],
                metric=["vif", "condition_number"], model="interactions")``

    # "What is the power to detect an effect of size 2 with noise SD of 1?"
        -> ``evaluate_design(design_matrix=[...],
                metric="power", effect_size=2.0, sigma=1.0)``
    """,
    category="experiments",
)
def evaluate_design_tool(spec: EvaluateDesignInput) -> dict[str, Any]:
    """Evaluate design quality."""
    try:
        from process_improve.experiments.evaluate import evaluate_design  # noqa: PLC0415

        df = pd.DataFrame(spec.design_matrix)
        result = evaluate_design(
            df,
            model=spec.model,
            metric=spec.metric,
            effect_size=spec.effect_size,
            alpha=spec.alpha,
            sigma=spec.sigma,
        )
        return clean(result)
    except _TOOL_EXPECTED_EXCEPTIONS as e:
        logger.exception("Tool evaluate_design failed")
        return {"error": str(e)}


_register("evaluate_design")
