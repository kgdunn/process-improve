# (c) Kevin Dunn, 2010-2026. MIT License.
"""MCP tool wrapper: ``analyze_experiment`` (ENG-02)."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from process_improve.experiments._tools import _TOOL_EXPECTED_EXCEPTIONS, _register, logger
from process_improve.tool_spec import clean, tool_spec


class AnalyzeExperimentInput(BaseModel):
    """Input contract for ``analyze_experiment``."""

    model_config = ConfigDict(extra="forbid")

    design_matrix: list[dict[str, Any]] = Field(
        ...,
        min_length=2,
        description=(
            "List of dicts, one per run. Must contain factor columns and "
            "optionally the response column. Example: "
            "[{'A': -1, 'B': -1, 'y': 28}, {'A': 1, 'B': -1, 'y': 36}, ...]"
        ),
    )
    response_column: str = Field(
        ...,
        description="Name of the response column in the design_matrix.",
    )
    model: str | None = Field(
        None,
        description=(
            "Model type ('main_effects', 'interactions' default, or 'quadratic') "
            "or an explicit Wilkinson formula string (e.g. 'y ~ A*B'). "
            "Formulas are validated by validate_formula_is_safe at the dispatch site."
        ),
    )
    analysis_type: str | list[str] = Field(
        "anova",
        description=(
            "One or more analysis types to run. Default: 'anova'. "
            "Options: anova, effects, coefficients, significance, "
            "residual_diagnostics, lack_of_fit, curvature_test, "
            "model_selection, box_cox, lenth_method, confidence_intervals, "
            "prediction, confirmation_test."
        ),
    )
    significance_level: float = Field(
        0.05,
        description="Significance level (default 0.05).",
    )
    transform: Literal["log", "sqrt", "inverse", "box_cox"] | None = Field(
        None,
        description="Optional response transform before fitting.",
    )
    new_points: list[dict[str, Any]] | None = Field(
        None,
        description="New factor settings for prediction or confirmation.",
    )
    observed_at_new: list[float] | None = Field(
        None,
        description="Observed values at new_points (for confirmation testing).",
    )


@tool_spec(
    name="analyze_experiment",
    description=(
        "Fit a model to experimental data and run statistical analyses. "
        "Supports ANOVA, effects, coefficients with p-values, significance testing, "
        "residual diagnostics (Shapiro-Wilk, Durbin-Watson, Breusch-Pagan, Cook's distance), "
        "lack-of-fit test, curvature test (center points vs factorial points), "
        "stepwise model selection (AIC/BIC), Box-Cox transformation, "
        "Lenth's method (PSE for unreplicated factorials), confidence intervals, "
        "prediction with prediction intervals, and confirmation run testing. "
        "Always returns a model summary with R-squared, adj-R-squared, pred-R-squared, and adequate precision. "
        "The design_matrix should contain factor columns with coded values (-1/+1). "
        "The response can be in a separate column or included in design_matrix."
    ),
    input_model=AnalyzeExperimentInput,
    examples="""
    # "Run ANOVA on my 2^2 factorial experiment"
        -> ``analyze_experiment(design_matrix=[{"A":-1,"B":-1,"y":28}, ...],
                response_column="y", analysis_type="anova")``

    # "Check residual diagnostics and lack of fit"
        -> ``analyze_experiment(design_matrix=[...], response_column="y",
                analysis_type=["residual_diagnostics", "lack_of_fit"])``

    # "Use Lenth's method on my unreplicated factorial"
        -> ``analyze_experiment(design_matrix=[...], response_column="y",
                analysis_type="lenth_method")``
    """,
    category="experiments",
)
def analyze_experiment_tool(spec: AnalyzeExperimentInput) -> dict[str, Any]:
    """Analyze experimental data."""
    try:
        from process_improve.experiments.analysis import analyze_experiment  # noqa: PLC0415

        df = pd.DataFrame(spec.design_matrix)
        np_df = pd.DataFrame(spec.new_points) if spec.new_points else None

        result = analyze_experiment(
            design_matrix=df,
            response_column=spec.response_column,
            model=spec.model,
            analysis_type=spec.analysis_type,
            significance_level=spec.significance_level,
            transform=spec.transform,
            new_points=np_df,
            observed_at_new=spec.observed_at_new,
        )
        return clean(result)
    except _TOOL_EXPECTED_EXCEPTIONS as e:
        logger.exception("Tool analyze_experiment failed")
        return {"error": str(e)}


_register("analyze_experiment")
