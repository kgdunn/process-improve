"""Public API for DOE visualization (Tool 6).

:func:`visualize_doe` is the single entry point.  It dispatches to the
correct plot class, builds the ChartSpec IR, runs the requested backend
adapters, and returns a JSON-serialisable dict.
"""

from __future__ import annotations

from typing import Any


def visualize_doe(  # noqa: PLR0913
    *,
    plot_type: str,
    analysis_results: dict[str, Any] | None = None,
    design_data: list[dict[str, Any]] | None = None,
    response_column: str | None = None,
    factors_to_plot: list[str] | None = None,
    hold_values: dict[str, float] | None = None,
    highlight_significant: bool = True,
    confidence_level: float = 0.95,
    backend: str = "both",
) -> dict[str, Any]:
    """Generate a DOE visualisation.

    Parameters
    ----------
    plot_type : str
        One of the 20 supported DOE plot types (see
        :mod:`~process_improve.experiments.visualization.plots`).
    analysis_results : dict or None
        Output dict from :func:`analyze_experiment`.  Required for most
        plot types that need fitted model data.
    design_data : list[dict] or None
        Raw design matrix as a list of row-dicts.  Used when
        *analysis_results* is not provided (e.g. main-effects from raw
        data).
    response_column : str or None
        Name of the response column in *design_data*.
    factors_to_plot : list[str] or None
        Subset of factors to display (2 for contour/interaction).
    hold_values : dict or None
        Fixed values for factors not being plotted.
    highlight_significant : bool
        Auto-highlight significant effects (Pareto / half-normal).
    confidence_level : float
        Confidence level for reference lines (default 0.95).
    backend : str
        ``"both"``, ``"plotly"``, or ``"echarts"``.

    Returns
    -------
    dict[str, Any]
        Keys: ``plot_type``, ``title``, ``plotly`` (Plotly figure dict),
        ``echarts`` (ECharts option dict), ``data`` (raw computed data).
    """
    from process_improve.experiments.visualization.plots.registry import create_plot  # noqa: PLC0415

    plot = create_plot(
        plot_type=plot_type,
        analysis_results=analysis_results,
        design_data=design_data,
        response_column=response_column,
        factors_to_plot=factors_to_plot,
        hold_values=hold_values,
        highlight_significant=highlight_significant,
        confidence_level=confidence_level,
    )

    spec = plot.to_spec()

    result: dict[str, Any] = {
        "plot_type": plot_type,
        "title": spec.title,
        "data": spec.to_data_dict(),
    }

    if backend in ("both", "plotly"):
        result["plotly"] = plot.to_plotly()
    else:
        result["plotly"] = None

    if backend in ("both", "echarts"):
        result["echarts"] = plot.to_echarts()
    else:
        result["echarts"] = None

    return result
