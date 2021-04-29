# Built-in libraries
import math
import random
from typing import Any, Dict, Optional

# Plotting settings
import plotly.graph_objects as go
from plotly.offline import plot as plotoffline
import seaborn as sns

import numpy as np
import pandas as pd


def get_rgba_from_triplet(incolour, alpha=1, as_string=False):
    """
    Convert the input colour triplet (list) to a Plotly rgba(r,g,b,a) string if
    `as_string` is True. If `False` it will return the list of 3 integer RGB
    values.

    E.g.    [0.9677975592919913, 0.44127456009157356, 0.5358103155058701] -> 'rgba(246,112,136,1)'
    """
    assert len(incolour) == 3
    colours = [max(0, int(math.floor(c * 255))) for c in incolour]
    if as_string:
        return f"rgba({colours[0]},{colours[1]},{colours[2]},{float(alpha)})"
    else:
        return colours


def plot_to_HTML(filename: str, fig: dict):
    config = dict(
        {
            "scrollZoom": True,
            "displayModeBar": True,
            "editable": False,
            "displaylogo": False,
            "showLink": False,
        }
    )
    return plotoffline(
        figure_or_data=fig,
        config=config,
        filename=filename,
        include_plotlyjs="cdn",
        include_mathjax="cdn",
        auto_open=False,
    )


def plot__all_batches_per_tag(
    df_dict: dict,
    tag: str,
    tag_y2: str = None,
    time_column: str = None,
    extra_info="",
    batches_to_highlight={},
    x_axis_label: str = "Time [sequence order]",
    highlight_width: int = 5,
    html_image_height: int = 900,
    html_aspect_ratio_w_over_h: float = 16 / 9,
    y1_limits: tuple = (None, None),
    y2_limits: tuple = (None, None),
) -> go.Figure:
    """Plots a particular `tag` over all batches in the given dataframe `df`.

    Parameters
    ----------
    df_dict : dict
        Standard data format for batches.
    tag : str
        Which tag to plot? [on the y1 (left) axis]
    tag_y2 : str, optional
        Which tag to plot? [on the y2 (right) axis]
        Tag will be plotted with different scaling on the secondary axis, to allow time-series
        comparisons to be easier.
    time_column : str, optional
        Which tag on the x-axis. If not specified, creates sequential integers, starting from 0
        if left as the default, `None`.
    extra_info : str, optional
        Used in the plot title to add any extra details, by default ""
    batches_to_highlight : dict, optional
        keys: an rgba colour string; for example: "rgba(255,0,0,0.9)"
        values: a list of batch identifiers (must be valid keys in `df_dict`).
        The highlighted batches will be shown with a heavier line.
    x_axis_label : str, optional
        String label for the x-axis, by default "Time [sequence order]"
    highlight_width: int, optional
        The width of the highlighted lines; default = 5.
    html_image_height : int, optional
        HTML image output height, by default 900
    html_aspect_ratio_w_over_h : float, optional
        HTML image aspect ratio: 16/9 (therefore the default width will be 1600 px)
    y1_limits: tuple, optional
        Axis limits enforced on the y1 (left) axis. Default is (None, None) which means the data
        themselves are used to determine the limits. Specify BOTH limits. Plotly requires
        (at the moment https://github.com/plotly/plotly.js/issues/400) that you specify both.
        Order: (low limit, high limit)
    y2_limits: tuple, optional
        Axis limits enforced on the y2 (right) axis. Default is (None, None) which means the data
        themselves are used to determine the limits. Specify BOTH limits. Plotly requires
        (at the moment https://github.com/plotly/plotly.js/issues/400) that you specify both.


    Returns
    -------
    go.Figure
        Standard Plotly fig object (dictionary-like).
    """
    unique_items = list(df_dict.keys())
    n_colours = len(unique_items)
    random.seed(13)
    colours = list(sns.husl_palette(n_colours))
    random.shuffle(colours)
    colours = [get_rgba_from_triplet(c, as_string=True) for c in colours]
    colour_assignment = dict(zip(unique_items, colours))

    # traces = []
    # highlight_traces = []
    regular_style = dict()
    highlight_dict = {}
    for key, val in batches_to_highlight.items():
        highlight_dict.update({item: key for item in val if item in df_dict.keys()})

    fig = go.Figure()

    for batch_name, batch_df in df_dict.items():
        assert (
            tag in batch_df.columns
        ), f"Tag '{tag}' not found in the batch with id {batch_name}."
        if tag_y2:
            assert (
                tag_y2 in batch_df.columns
            ), f"Tag '{tag}' not found in the batch with id {batch_name}."
        if time_column in batch_df.columns:
            time_data = batch_df[time_column]
        else:
            time_data = list(range(batch_df.shape[0]))

        if batch_name in highlight_dict.keys():
            continue  # come to this later
        else:
            regular_style["color"] = colour_assignment[batch_name]
            fig.add_trace(
                go.Scatter(
                    x=time_data,
                    y=batch_df[tag],
                    name=batch_name,
                    line=regular_style,
                    mode="lines",
                    opacity=0.8,
                    yaxis="y1",
                )
            )
            if tag_y2:
                fig.add_trace(
                    go.Scatter(
                        x=time_data,
                        y=batch_df[tag_y2],
                        name=batch_name,
                        line=regular_style,
                        mode="lines",
                        opacity=0.8,
                        yaxis="y2",
                    )
                )

    # Add the highlighted batches last: therefore, sadly, we have to do another run-through.
    # Plotly does not yet support z-orders.
    for batch_name, batch_df in df_dict.items():
        if time_column in batch_df.columns:
            time_data = batch_df[time_column]
        else:
            time_data = list(range(batch_df.shape[0]))

        if batch_name in highlight_dict.keys():
            fig.add_trace(
                go.Scatter(
                    x=time_data,
                    y=batch_df[tag],
                    line=dict(width=highlight_width, color=highlight_dict[batch_name]),
                    name=batch_name,
                    mode="lines",
                    opacity=0.8,
                    yaxis="y1",
                )
            )
            if tag_y2:
                fig.add_trace(
                    go.Scatter(
                        x=time_data,
                        y=batch_df[tag_y2],
                        line=dict(
                            width=highlight_width, color=highlight_dict[batch_name]
                        ),
                        name=batch_name,
                        mode="lines",
                        opacity=0.8,
                        yaxis="y2",
                    )
                )

    yaxis1_dict = dict(
        title=tag, gridwidth=2, matches="y1", showticklabels=True, side="left"
    )
    if (y1_limits[0] is not None) or (y1_limits[1] is not None):
        yaxis1_dict["autorange"] = False
        yaxis1_dict["range"] = y1_limits

    yaxis2_dict: Dict[str, Any] = dict(
        title=tag_y2, gridwidth=2, matches="y2", showticklabels=True, side="right"
    )
    if (y2_limits[0] is not None) or (y2_limits[1] is not None):
        yaxis2_dict["autorange"] = False
        yaxis2_dict["range"] = y2_limits

    fig.update_layout(
        title=f"Plot of: '{tag}'"
        + (f" on left axis; with '{str(tag_y2)}' on right axis." if tag_y2 else ".")
        + (f" [{str(extra_info)}]" if extra_info else ""),
        hovermode="closest",
        showlegend=True,
        legend=dict(
            orientation="h",
            traceorder="normal",
            font=dict(family="sans-serif", size=12, color="#000"),
            bordercolor="#DDDDDD",
            borderwidth=1,
        ),
        autosize=False,
        xaxis=dict(title=x_axis_label, gridwidth=1),
        yaxis=yaxis1_dict,
        width=html_aspect_ratio_w_over_h * html_image_height,
        height=html_image_height,
    )
    if tag_y2:
        fig.update_layout(yaxis2=yaxis2_dict)
    return fig


def plot__tag_time(
    source: pd.Series,
    overlap: bool = False,
    filled: bool = False,
    showlegend: bool = True,
    tag_order: Optional[list] = None,
    x_axis_label: str = "Time, grouped per tag",
    y_axis_label: str = "",
    html_image_height: int = 900,
    html_aspect_ratio_w_over_h: float = 16 / 9,
):

    """Plots a vector of information, for every tag [index level 0] over every time [index level 1]

    Parameters
    ----------
    source : pd.Series
        `source` series must be a multi-level Pandas index. Level 0 gives the unique names for each
        tag, and level 1 gives the names for the 'time' or 'sequence' within a tag. The level 1
        must be numeric and monotonic.
    overlap : bool, optional
        Should all tags overlap [True], or be plotted side-by-side [default; False]
    filled : bool, optional
        Should the area below the line be filled. Only makes sense if `overlap` is False.
    tag_order : Optional[list], optional
        Indicate the order of the tags on the x-axis. Makes sense if `overlap` is False.
    colour_order : Optional[list], optional
        A list of unique, or cycling colours to use, per tag.
    x_axis_label : str, optional
        Label for the x-axis, by default "Time, grouped per tag"
    html_image_height : int, optional
        Height, in pixels. By default 900
    html_aspect_ratio_w_over_h : float, optional
        Determines the image width, as a ratio of the height: by default 16/9
    """
    assert isinstance(source, pd.Series), "`source` must be a Pandas series"
    assert len(source.index.levels) == 2, "`source` must have a multilevel index of 2"
    tag_group = source.index.levels[0]
    n_colours = len(tag_group)
    random.seed(13)
    colours = list(sns.husl_palette(n_colours))
    colours = [get_rgba_from_triplet(c, as_string=True) for c in colours]
    colour_assignment = dict(zip(tag_group, colours))

    time_group = source.index.levels[1].values
    deltas = np.diff(time_group)
    assert all(deltas > 0), "Level 2 of the index must be numeric, increasing"
    traces = []

    offset = np.nanmean(deltas)
    for tag, series in source.unstack(level=0).items():
        x_axis = time_group
        fill = None
        if not (overlap):  # ie: side-by-side
            if filled:
                fill = "tozeroy"
            offset += x_axis[-1]

        trace = go.Scatter(
            x=x_axis,
            y=series,
            name=tag,
            mode="lines",
            fill=fill,
            fillcolor=colour_assignment[tag],
            line=dict(color=colour_assignment[tag], width=2),
            legendgroup=tag,
            showlegend=showlegend,
        )
        traces.append(trace)

    layout = go.Layout(
        title=source.name or y_axis_label,
        hovermode="closest",
        showlegend=True,
        legend=dict(
            orientation="h",
            traceorder="normal",
            font=dict(family="sans-serif", size=12, color="#000"),
            bordercolor="#DDDDDD",
            borderwidth=1,
        ),
        autosize=False,
        xaxis=dict(title=x_axis_label, gridwidth=1),
        yaxis=dict(title=y_axis_label, gridwidth=2),
        width=html_aspect_ratio_w_over_h * html_image_height,
        height=html_image_height,
    )

    return dict(data=traces, layout=layout)
