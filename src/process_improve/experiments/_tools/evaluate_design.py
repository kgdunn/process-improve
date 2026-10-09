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
    model: (
        Literal[
            "main_effects",
            "interactions",
            "quadratic",
            "scheffe_linear",
            "scheffe_quadratic",
            "scheffe_special_cubic",
        ]
        | None
    ) = Field(
        None,
        description=(
            "Model type to evaluate against. 'main_effects' = main effects only, "
            "'interactions' = main effects + 2-factor interactions (default), "
            "'quadratic' = interactions + squared terms; the 'scheffe_*' models are for "
            "mixture proportions (the default when a mixture region is given)."
        ),
    )
    metric: str | list[str] = Field(
        "d_efficiency",
        description=(
            "One or more metric names to compute. Default: 'd_efficiency'. "
            "Options: d_efficiency, average_prediction_variance, g_efficiency, a_optimality, "
            "e_optimality, fds, prediction_variance, vif, condition_number, "
            "correlation, power, degrees_of_freedom, alias_structure, alias_matrix, "
            "confounding, resolution, defining_relation, clear_effects, "
            "minimum_aberration, moment_aberration."
        ),
    )
    effect_size: float | None = Field(
        None,
        description=(
            "Anticipated model coefficient in coded units for the power calculation, i.e. half the "
            "high-minus-low effect of a two-level factor."
        ),
    )
    alpha: float = Field(
        0.05,
        description="Significance level (default 0.05).",
    )
    sigma: float | None = Field(
        None,
        description="Estimated noise standard deviation.",
    )
    region: dict[str, Any] | Literal["cuboidal", "spherical"] | None = Field(
        None,
        description=(
            "Region the prediction-variance metrics (average_prediction_variance, g_efficiency, fds) are taken "
            "over: pass metadata['region'] from generate_design for a constrained or mixture design, or "
            "'cuboidal' / 'spherical'. Default: the cube."
        ),
    )
    random_state: int = Field(
        42,
        ge=0,
        description="Seed for the region sampler (default: 42).",
    )
    categorical_coding: Literal["effect", "treatment"] = Field(
        "effect",
        description=(
            "How a categorical (label) factor is coded. 'effect' (default): sum-to-zero effect coding, the "
            "usual convention in DoE software and the coding generate_design's optimal designs use. "
            "'treatment': 0/1 dummy coding against the first level. A-, E-optimality, VIF, condition number, "
            "power and the alias matrix depend on the coding; prediction variance and D rankings do not."
        ),
    )


@tool_spec(
    name="evaluate_design",
    description=(
        "Evaluate the quality of an experimental design matrix by computing metrics such as "
        "D-efficiency, G-efficiency, the average prediction variance (I-criterion), VIF, condition "
        "number, alias structure, "
        "confounding pattern, resolution, power, prediction variance, degrees of freedom, "
        "clear effects, and minimum aberration. "
        "The design_matrix should be a list of dictionaries with factor names as keys and "
        "coded values (-1/+1) as values (proportions for a mixture). For a constrained or mixture "
        "design, pass the region from generate_design's metadata so the prediction variance is judged "
        "over the settings the design may use. "
        "Use this after generating a design to check if it meets quality criteria, or to "
        "compare alternative designs."
    ),
    input_model=EvaluateDesignInput,
    rng={"uses_rng": True, "seed_param": "random_state", "default_seed": 42},
    examples="""
    # "What is the D-efficiency of my 2^3 factorial design?"
        -> ``evaluate_design(design_matrix=[{"A":-1,"B":-1,"C":-1}, ...],
                metric="d_efficiency", model="interactions")``

    # "Check VIF and condition number"
        -> ``evaluate_design(design_matrix=[...],
                metric=["vif", "condition_number"], model="interactions")``

    # "How well does my constrained design predict inside its region?"
        -> ``evaluate_design(design_matrix=[...], model="quadratic",
                metric=["average_prediction_variance", "g_efficiency"],
                region=<metadata["region"] returned by generate_design>)``

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
        from process_improve.experiments.region import DesignRegion  # noqa: PLC0415

        df = pd.DataFrame(spec.design_matrix)
        region = DesignRegion.from_dict(spec.region) if isinstance(spec.region, dict) else spec.region
        result = evaluate_design(
            df,
            model=spec.model,
            metric=spec.metric,
            effect_size=spec.effect_size,
            alpha=spec.alpha,
            sigma=spec.sigma,
            region=region,
            random_state=spec.random_state,
            categorical_coding=spec.categorical_coding,
        )
        return clean(result)
    except _TOOL_EXPECTED_EXCEPTIONS as e:
        logger.exception("Tool evaluate_design failed")
        return {"error": str(e)}


_register("evaluate_design")
