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
            "prediction, confirmation_test, split_plot. Use split_plot for a design "
            "with hard-to-change factors: the others treat every run as independent."
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
    coding: Literal["actual", "coded", "auto"] | dict[str, dict[str, float]] | None = Field(
        None,
        description=(
            "Scale of the coefficients and confidence_intervals. 'actual': the factors as given. 'coded': refitted "
            "with each factor from -1 at its lowest value to +1 at its highest. 'auto': 'coded', but 'actual' for a "
            "mixture model. Or ranges, e.g. {'T': {'low': 150, 'high': 200}}, that map to -1 and +1, such as a "
            "central composite design's cube levels rather than its axial runs. Omitted: 'actual' until 2.0 and "
            "'auto' after; when that change would affect the result, it carries a coding_note saying so. Use "
            "'auto' when the coefficients go to optimize_responses, which evaluates models at coded settings and "
            "refuses actual-unit coefficients; coded results also carry coefficients_actual, the equation in the "
            "factors' own units. With interactions, a coded main effect is tested at the centre of the design, "
            "an actual one where the other factors are zero."
        ),
    )
    new_points: list[dict[str, Any]] | None = Field(
        None,
        description="New factor settings for prediction or confirmation.",
    )
    observed_at_new: list[float] | None = Field(
        None,
        description="Observed values at new_points (for confirmation testing).",
    )
    whole_plot: str | None = Field(
        None,
        description=(
            "Column labelling each run's whole plot, for analysis_type='split_plot'. "
            "Default: the 'WholePlot' column that generate_design adds to a split-plot design."
        ),
    )


@tool_spec(
    name="analyze_experiment",
    description=(
        "Fit a model to experimental data and run statistical analyses. "
        "Supports ANOVA, effects, coefficients with p-values, significance testing, "
        "residual diagnostics (Shapiro-Wilk, Durbin-Watson, Breusch-Pagan, Cook's distance), "
        "lack-of-fit test, curvature test (center points vs factorial points), "
        "stepwise model selection with effect heredity (AICc by default, or AIC/BIC), Box-Cox transformation, "
        "Lenth's method (PSE for unreplicated factorials), confidence intervals, "
        "prediction with prediction intervals, confirmation run testing, and the REML analysis of "
        "split-plot designs (hard-to-change factors) with Satterthwaite degrees of freedom. "
        "Always returns a model summary with R-squared, adj-R-squared, pred-R-squared, and adequate precision. "
        "Factor columns may be coded (-1/+1) or in actual units; for actual units, set coding='auto' to get "
        "coefficients on the -1/+1 scale, as optimize_responses needs. "
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

    # "Fit my experiment in actual units, then find the best settings"
        -> ``analyze_experiment(design_matrix=[{"T":150,"P":1,"y":10}, ...],
                response_column="y", analysis_type="coefficients", coding="auto")``,
           then pass the result to ``optimize_responses``

    # "Analyse my split-plot experiment; temperature was hard to change"
        -> ``analyze_experiment(design_matrix=[...], response_column="y",
                analysis_type="split_plot")``
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
            coding=spec.coding,
            new_points=np_df,
            observed_at_new=spec.observed_at_new,
            whole_plot=spec.whole_plot,
        )
        return clean(result)
    except _TOOL_EXPECTED_EXCEPTIONS as e:
        logger.exception("Tool analyze_experiment failed")
        return {"error": str(e)}


_register("analyze_experiment")
