"""Public API for DOE visualization (Tool 6).

:func:`visualize_doe` is the single entry point.  It dispatches to the
correct plot class, builds the ChartSpec IR, runs the requested backend
adapters, and returns a JSON-serialisable dict.

:func:`main_effects_plot` is a convenience wrapper that accepts either
a fitted :class:`~process_improve.experiments.models.Model` or a
DataFrame, and returns a Plotly figure directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import plotly.graph_objects as go

if TYPE_CHECKING:
    from process_improve.experiments.models import Model


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
    factor_labels: dict[str, str] | None = None,
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
    factor_labels : dict or None
        Mapping ``{factor_symbol: full_name}`` used for axis labels and
        bar/legend entries (e.g. ``{"A": "Temperature [°C]"}``).
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
        factor_labels=factor_labels,
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


def main_effects_plot(
    model_or_data: Model | pd.DataFrame,
    response_column: str | None = None,
    factors_to_plot: list[str] | None = None,
    *,
    factor_labels: dict[str, str] | None = None,
) -> go.Figure:
    """Generate a main-effects plot from a fitted model or a design DataFrame.

    For each factor, the mean response at every observed level is drawn
    as a line.  A flat line indicates no main effect; a steep line
    indicates a strong one.  The grand mean is shown as a horizontal
    dotted reference line.

    Parameters
    ----------
    model_or_data : Model or pandas.DataFrame
        Either a fitted ``Model`` returned by
        :func:`~process_improve.experiments.models.lm`, or a DataFrame
        containing one column per factor plus a response column.
    response_column : str, optional
        Name of the response column.  Required when *model_or_data* is
        a DataFrame; ignored (and inferred from the model formula) when
        a fitted ``Model`` is supplied.
    factors_to_plot : list of str, optional
        Subset of factors to display.  By default all main-effect
        factors are shown — for a ``Model`` these are the level-1
        factors from the formula; for a DataFrame they are all columns
        other than *response_column*.
    factor_labels : dict, optional
        Mapping ``{factor_symbol: full_name}`` used for legend entries
        (e.g. ``{"A": "Temperature [°C]"}``).

    Returns
    -------
    plotly.graph_objects.Figure
        Plotly figure ready for display or further customisation.

    Examples
    --------
    From a fitted model::

        from process_improve.experiments import c, gather, lm, main_effects_plot
        A = c(-1, +1, -1, +1)
        B = c(-1, -1, +1, +1)
        y = c(52, 74, 62, 80, name="y")
        expt = gather(A=A, B=B, y=y)
        model = lm("y ~ A + B", expt)
        fig = main_effects_plot(model)

    From a raw DataFrame::

        fig = main_effects_plot(df, response_column="yield")

    See Also
    --------
    visualize_doe : General DOE plotting entry point.
    """
    from process_improve.experiments.models import Model as _Model  # noqa: PLC0415

    if isinstance(model_or_data, _Model):
        df = pd.DataFrame(model_or_data.data).copy()
        response_column = model_or_data.get_response_name()
        if factors_to_plot is None:
            factors_to_plot = model_or_data.get_factor_names(level=1)
    elif isinstance(model_or_data, pd.DataFrame):
        if response_column is None:
            raise ValueError(
                "response_column must be provided when passing a DataFrame.",
            )
        df = pd.DataFrame(model_or_data).copy()
    else:
        raise TypeError(
            "model_or_data must be a fitted Model or a pandas DataFrame; "
            f"got {type(model_or_data).__name__}.",
        )

    if response_column not in df.columns:
        raise ValueError(
            f"response_column {response_column!r} not found in data columns: "
            f"{list(df.columns)}.",
        )

    design_data = df.to_dict(orient="records")

    result = visualize_doe(
        plot_type="main_effects",
        design_data=design_data,
        response_column=response_column,
        factors_to_plot=factors_to_plot,
        factor_labels=factor_labels,
        backend="plotly",
    )
    return go.Figure(result["plotly"])
