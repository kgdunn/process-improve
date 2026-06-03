# (c) Kevin Dunn, 2010-2026. MIT License.
"""MCP tool wrapper: ``optimize_responses`` (ENG-02)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from process_improve.experiments._tools import _TOOL_EXPECTED_EXCEPTIONS, _register, logger
from process_improve.tool_spec import clean, tool_spec


class OptimizeResponsesInput(BaseModel):
    """Input contract for ``optimize_responses``."""

    model_config = ConfigDict(extra="forbid")

    fitted_models: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        description=(
            "One or more fitted models from analyze_experiment. Each entry must "
            "include 'coefficients' (list of {term, coefficient}), 'factor_names' "
            "(list of strings), and optionally 'response_name', 'mse_residual', 'r_squared'."
        ),
    )
    goals: list[dict[str, Any]] | None = Field(
        None,
        description=(
            "Per-response optimisation goals. Each entry: response (str), "
            "goal ('maximize'|'minimize'|'target'), low, high, optional target, "
            "weight, importance. Required for 'desirability' method."
        ),
    )
    method: Literal[
        "desirability",
        "steepest_ascent",
        "steepest_descent",
        "stationary_point",
        "canonical_analysis",
    ] = Field(
        "desirability",
        description="Optimisation method (default: 'desirability').",
    )
    factor_ranges: dict[str, dict[str, float]] | None = Field(
        None,
        description=(
            'Factor bounds in actual units, e.g. {"Temperature": {"low": 150, "high": 200}}. '
            "Used to convert coded settings to actual units in the output."
        ),
    )
    step_size: float = Field(
        0.5,
        description="Step size in coded units for steepest ascent/descent (default 0.5).",
    )
    n_steps: int = Field(
        10,
        ge=1,
        description="Number of steps for steepest ascent/descent (default 10).",
    )
    desirability_weights: list[float] | None = Field(
        None,
        description="Importance weights for composite desirability (overrides per-goal importance).",
    )


@tool_spec(
    name="optimize_responses",
    description=(
        "Find optimal factor settings for one or multiple responses from fitted experimental models. "
        "Supports several methods: 'desirability' (Derringer-Suich desirability functions for single or "
        "multi-response optimisation), 'steepest_ascent' / 'steepest_descent' (move along the gradient "
        "of a first-order model), 'stationary_point' (locate the optimum of a second-order model), "
        "'canonical_analysis' (eigenvalue decomposition to classify the response surface shape). "
        "Each fitted_model must include coefficients (as returned by analyze_experiment with "
        "analysis_type='coefficients'), factor_names, and response_name. "
        "For desirability, each goal specifies whether to maximize, minimize, or target a value."
    ),
    input_model=OptimizeResponsesInput,
    examples="""
    # "Find the stationary point of my quadratic model"
        -> ``optimize_responses(fitted_models=[{"response_name": "yield",
                "coefficients": [{"term": "Intercept", "coefficient": 40},
                    {"term": "A", "coefficient": 5.25}, {"term": "B", "coefficient": -2},
                    {"term": "I(A ** 2)", "coefficient": -3}, {"term": "I(B ** 2)", "coefficient": -1.5},
                    {"term": "A:B", "coefficient": 1.5}],
                "factor_names": ["A", "B"]}],
            method="stationary_point")``

    # "Optimize two responses using desirability"
        -> ``optimize_responses(fitted_models=[model1, model2],
                goals=[{"response": "yield", "goal": "maximize", "low": 30, "high": 50},
                       {"response": "cost", "goal": "minimize", "low": 10, "high": 40}],
                method="desirability")``

    # "Generate a steepest ascent path from a first-order model"
        -> ``optimize_responses(fitted_models=[model],
                method="steepest_ascent", step_size=0.5, n_steps=8,
                factor_ranges={"Temperature": {"low": 150, "high": 200}})``
    """,
    category="experiments",
)
def optimize_responses_tool(spec: OptimizeResponsesInput) -> dict[str, Any]:
    """Optimize experimental responses."""
    try:
        from process_improve.experiments.optimization import optimize_responses  # noqa: PLC0415

        result = optimize_responses(
            fitted_models=spec.fitted_models,
            goals=spec.goals,
            method=spec.method,
            factor_ranges=spec.factor_ranges,
            step_size=spec.step_size,
            n_steps=spec.n_steps,
            desirability_weights=spec.desirability_weights,
        )
        return clean(result)
    except _TOOL_EXPECTED_EXCEPTIONS as e:
        logger.exception("Tool optimize_responses failed")
        return {"error": str(e)}


_register("optimize_responses")
