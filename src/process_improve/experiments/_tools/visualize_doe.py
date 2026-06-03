# (c) Kevin Dunn, 2010-2026. MIT License.
"""MCP tool wrapper: ``visualize_doe`` (ENG-02)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from process_improve.experiments._tools import _TOOL_EXPECTED_EXCEPTIONS, _register, logger
from process_improve.tool_spec import clean, tool_spec


class VisualizeDoeInput(BaseModel):
    """Input contract for ``visualize_doe``."""

    model_config = ConfigDict(extra="forbid")

    plot_type: Literal[
        "pareto", "half_normal", "daniel",
        "main_effects", "interaction", "perturbation",
        "residuals_vs_fitted", "normal_probability",
        "residuals_vs_order", "box_cox",
        "contour", "surface_3d", "prediction_variance",
        "cube_plot", "square_plot",
        "desirability_contour", "overlay",
        "ridge_trace", "steepest_ascent_path",
        "fds_plot", "power_curve",
    ] = Field(
        ...,
        description="Type of DOE plot to generate.",
    )
    analysis_results: dict[str, Any] | None = Field(
        None,
        description=(
            "Results from fit_linear_model or analyze_experiment. "
            "Should contain keys like 'coefficients', 'effects', "
            "'residual_diagnostics', 'lenth_method', 'model_summary'."
        ),
    )
    design_data: list[dict[str, Any]] | None = Field(
        None,
        description=(
            "Raw design matrix as a list of dicts (one per run). "
            "Factor columns should be coded -1/+1."
        ),
    )
    response_column: str | None = Field(
        None,
        description="Name of the response column in design_data.",
    )
    factors_to_plot: list[str] | None = Field(
        None,
        description=(
            "Which factors to plot (2 for contour/interaction, "
            "3 for cube_plot). If omitted, inferred from data."
        ),
    )
    hold_values: dict[str, float] | None = Field(
        None,
        description=(
            "Coded values for factors not being plotted "
            "(default 0 = centre). E.g. {'C': 0.5}."
        ),
    )
    highlight_significant: bool = Field(
        True,
        description="Highlight significant effects on Pareto/half-normal plots.",
    )
    confidence_level: float = Field(
        0.95,
        description="Confidence level for thresholds (default 0.95).",
    )
    backend: Literal["both", "plotly", "echarts"] = Field(
        "both",
        description="Which rendering backend(s) to include in output.",
    )


@tool_spec(
    name="visualize_doe",
    description=(
        "Generate DOE visualisations from analysis results or design data. "
        "Supports 21 plot types: significance plots (pareto, half_normal, daniel), "
        "factor-effect plots (main_effects, interaction, perturbation), "
        "diagnostic plots (residuals_vs_fitted, normal_probability, residuals_vs_order, box_cox), "
        "response-surface plots (contour, surface_3d, prediction_variance), "
        "cube plot (cube_plot), square plot (square_plot), "
        "optimisation plots (desirability_contour, overlay, ridge_trace, steepest_ascent_path), "
        "and design-quality plots (fds_plot, power_curve). "
        "Returns both Plotly and ECharts configurations for dual-backend rendering. "
        "Pass analysis_results from fit_linear_model or analyze_experiment, or raw design_data."
    ),
    input_model=VisualizeDoeInput,
    examples="""
    # "Show me a Pareto chart of my effects"
        -> ``visualize_doe(plot_type="pareto",
                analysis_results={"effects": {"A": 5.2, "B": -3.1, "A:B": 1.0}})``

    # "Draw a contour plot of Temperature vs Pressure"
        -> ``visualize_doe(plot_type="contour",
                analysis_results={"coefficients": [...]},
                factors_to_plot=["Temperature", "Pressure"])``
    """,
    category="experiments",
)
def visualize_doe_tool(spec: VisualizeDoeInput) -> dict[str, Any]:
    """Generate a DOE visualisation."""
    try:
        from process_improve.experiments.visualization import visualize_doe  # noqa: PLC0415

        result = visualize_doe(
            plot_type=spec.plot_type,
            analysis_results=spec.analysis_results,
            design_data=spec.design_data,
            response_column=spec.response_column,
            factors_to_plot=spec.factors_to_plot,
            hold_values=spec.hold_values,
            highlight_significant=spec.highlight_significant,
            confidence_level=spec.confidence_level,
            backend=spec.backend,
        )
        return clean(result)
    except _TOOL_EXPECTED_EXCEPTIONS as e:
        logger.exception("Tool visualize_doe failed")
        return {"error": str(e)}


_register("visualize_doe")
