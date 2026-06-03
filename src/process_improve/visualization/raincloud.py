"""Raincloud plot.

A raincloud combines three complementary views of a distribution in one
figure: a one-sided violin (the density "cloud"), a boxplot (the five-number
summary), and the jittered raw observations (the "rain"). It is drawn with
Plotly so it inherits the package's registered base theme.
"""

from __future__ import annotations

import pandas as pd

try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover - exercised via env-without-plotly
    from process_improve._extras import _MissingExtra
    go = _MissingExtra("plotly", "plotting")  # type: ignore[assignment]


def raincloud(  # noqa: PLR0913
    data: pd.DataFrame | pd.Series,
    value: str | None = None,
    group: str | None = None,
    *,
    title: str = "",
    orientation: str = "h",
    template: str | None = None,
) -> go.Figure:
    """Draw a raincloud plot.

    Parameters
    ----------
    data : pandas.DataFrame or pandas.Series
        The data to plot. A Series is treated as a single, ungrouped sample.
    value : str or None
        Name of the numeric column in ``data`` to plot. Required when ``data``
        is a DataFrame.
    group : str or None
        Optional name of a categorical column; one raincloud is drawn per
        unique group value.
    title : str, optional
        Figure title.
    orientation : {"h", "v"}, optional
        ``"h"`` draws horizontal rainclouds (the default and usual
        orientation); ``"v"`` draws them vertically.
    template : str or None, optional
        Plotly template name. When ``None``, the package's registered default
        theme is used.

    Returns
    -------
    plotly.graph_objects.Figure
        A figure with one :class:`plotly.graph_objects.Violin` trace per group,
        each showing the density cloud, the box, and the jittered raw points.

    Examples
    --------
    >>> raincloud(df, value="yield", group="reactor")
    >>> raincloud(pd.Series([1.0, 2.0, 3.0]))
    """
    if orientation not in ("h", "v"):
        raise ValueError(f"orientation must be 'h' or 'v', got {orientation!r}.")

    if isinstance(data, pd.Series):
        column = value or (str(data.name) if data.name is not None else "value")
        frame = data.to_frame(name=column)
        value = column
    else:
        if value is None:
            raise ValueError("`value` (the numeric column name) is required when `data` is a DataFrame.")
        frame = data

    if group is None:
        groups: list[tuple[str, pd.Series]] = [("", frame[value])]
    else:
        groups = [(str(name), sub[value]) for name, sub in frame.groupby(group, sort=False)]

    fig = go.Figure()
    for name, series in groups:
        values = series.dropna()
        # A one-sided violin is the "cloud"; box_visible adds the box; and
        # points="all" with an offset draws the jittered raw "rain".
        violin_axis = {"x": values} if orientation == "h" else {"y": values}
        fig.add_trace(
            go.Violin(
                name=name,
                side="positive",
                box_visible=True,
                meanline_visible=True,
                points="all",
                jitter=0.3,
                pointpos=-0.9,
                **violin_axis,
            )
        )

    fig.update_layout(title=title, showlegend=group is not None)
    if template is not None:
        fig.update_layout(template=template)
    return fig
