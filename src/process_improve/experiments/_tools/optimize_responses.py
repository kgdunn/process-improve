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
            "Per-response optimisation goals. Each entry: response (str, matched against the model's "
            "response_name so the two lists need not be in the same order), "
            "goal ('maximize'|'minimize'|'target'), low, high, optional target, "
            "weight, weight_high, importance. Required for 'desirability' method. "
            "'weight' shapes that response's own desirability ramp; 'importance' sets how much the "
            "response counts relative to the others in the composite. They are different things."
        ),
    )
    method: Literal[
        "desirability",
        "steepest_ascent",
        "steepest_descent",
        "stationary_point",
        "canonical_analysis",
        "ridge_analysis",
        "pareto_front",
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
        description=(
            "Number of steps along a path: the steepest ascent/descent steps, or the radii reported "
            "by ridge analysis over and above the centre (default 10)."
        ),
    )
    response_importance: list[float] | None = Field(
        None,
        description=(
            "Relative importance per response in the composite desirability, aligned with fitted_models. "
            "Overrides the per-goal 'importance'. This is not the per-goal 'weight', which shapes an "
            "individual response's ramp."
        ),
    )
    significance_level: float = Field(
        0.05,
        gt=0.0,
        lt=1.0,
        description="Alpha for intervals reported at the optimum (default 0.05, giving 95% intervals).",
    )
    search_bounds: list[float] | dict[str, list[float]] | None = Field(
        None,
        description=(
            "Coded region to search, as [low, high] applied to every factor, or a mapping from factor "
            "name to its own [low, high]. Defaults to the factorial cube, [-1, 1]. That default suits a "
            "two-level design but understates a central composite design, whose axial runs sit at plus "
            "or minus alpha: pass [-1.41, 1.41] for a two-factor rotatable central composite design so "
            "the search covers the region the experiment actually explored."
        ),
    )
    desirability_weights: list[float] | None = Field(
        None,
        description="Deprecated alias for 'response_importance'. Use 'response_importance' instead.",
    )
    ridge_direction: Literal["maximize", "minimize"] = Field(
        "maximize",
        description="Which ridge method='ridge_analysis' traces (default 'maximize').",
    )
    n_pareto_points: int = Field(
        21,
        ge=2,
        description=(
            "Target number of weight vectors for method='pareto_front' (default 21). The front returned "
            "is usually smaller, since dominated and duplicate solutions are dropped."
        ),
    )


def _as_bounds(
    raw: list[float] | dict[str, list[float]] | None,
) -> tuple[float, float] | dict[str, tuple[float, float]] | None:
    """Convert the JSON-friendly list form into the tuples the library expects."""
    if raw is None:
        return None
    if isinstance(raw, dict):
        return {name: (float(pair[0]), float(pair[1])) for name, pair in raw.items()}
    return (float(raw[0]), float(raw[1]))


@tool_spec(
    name="optimize_responses",
    description=(
        "Find optimal factor settings for one or multiple responses from fitted experimental models. "
        "Supports several methods: 'desirability' (Derringer-Suich desirability functions for single or "
        "multi-response optimisation), 'steepest_ascent' / 'steepest_descent' (move along the gradient "
        "of a first-order model), 'stationary_point' (locate the optimum of a second-order model), "
        "'canonical_analysis' (eigenvalue decomposition to classify the response surface shape), "
        "'ridge_analysis' (trace the best point on spheres of increasing radius from the centre, which is "
        "what to use when the stationary point falls outside the region the experiment covered, or is a "
        "saddle), and 'pareto_front' (the set of non-dominated compromises across two or more responses, "
        "when the trade-off itself is the thing worth seeing rather than one desirability-weighted point). "
        "Each fitted_model must include coefficients (as returned by analyze_experiment with "
        "analysis_type='coefficients'), factor_names, and response_name. "
        "For desirability, each goal specifies whether to maximize, minimize, or target a value. "
        "The desirability result also carries a 'responses' list, pairing each model's coefficients "
        "with its specification limits, which can be passed straight to visualize_doe(plot_type='overlay') "
        "to see the region where every response is simultaneously within specification."
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

    # "Optimize two responses, counting yield twice as heavily as cost"
        -> ``optimize_responses(fitted_models=[model1, model2],
                goals=[{"response": "yield", "goal": "maximize", "low": 30, "high": 50},
                       {"response": "cost", "goal": "minimize", "low": 10, "high": 40}],
                method="desirability", response_importance=[2.0, 1.0])``

    # "Generate a steepest ascent path from a first-order model"
        -> ``optimize_responses(fitted_models=[model],
                method="steepest_ascent", step_size=0.5, n_steps=8,
                factor_ranges={"Temperature": {"low": 150, "high": 200}})``

    # "The optimum is outside my design space - how good can I get within the region I explored?"
        -> ``optimize_responses(fitted_models=[model], method="ridge_analysis",
                n_steps=10, search_bounds=[-1.41, 1.41])``

    # "Show me the trade-off between yield and cost, not a single weighted answer"
        -> ``optimize_responses(fitted_models=[model1, model2],
                goals=[{"response": "yield", "goal": "maximize"},
                       {"response": "cost", "goal": "minimize"}],
                method="pareto_front", n_pareto_points=21)``
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
            response_importance=spec.response_importance,
            significance_level=spec.significance_level,
            search_bounds=_as_bounds(spec.search_bounds),
            desirability_weights=spec.desirability_weights,
            ridge_direction=spec.ridge_direction,
            n_pareto_points=spec.n_pareto_points,
        )
        return clean(result)
    except _TOOL_EXPECTED_EXCEPTIONS as e:
        logger.exception("Tool optimize_responses failed")
        return {"error": str(e)}


_register("optimize_responses")
