# Built-in libraries
import math
import random
from typing import Optional

# Plotting settings
import plotly.graph_objects as go
from plotly.offline import plot as plotoffline
import seaborn as sns
import pandas as pd


def get_rgba_from_triplet(incolour, alpha=1, string=False):
    """
    Convert the input colour triplet (list) to a Plotly rgba(r,g,b,a) string if
    `string` is True. If `False` it will return the list of 3 integer RGB
    values.

    E.g.    [0.9677975592919913, 0.44127456009157356, 0.5358103155058701] -> 'rgba(247,113,137,1)'
    """
    assert len(incolour) == 3
    colours = [max(0, int(math.floor(c * 255))) for c in incolour]
    if string:
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
    time_column: str = None,
    extra_info="",
    batches_to_highlight=[],
    x_axis_label: str = "Time [sequence order]",
    html_image_height: int = 900,
    html_aspect_ratio_w_over_h: float = 16 / 9,
) -> dict:
    """Plots a particular `tag` over all batches in the given dataframe `df`.

    Parameters
    ----------
    df_dict : dict
        Standard data format for batches.
    tag : str
        Which tag to plot?
    time_column : str, optional
        Which tag on the x-axis. If not specified, creates sequential integers, starting from 0
        if left as the default, `None`.
    extra_info : str, optional
        Used in the plot title to add any extra details, by default ""
    batches_to_highlight : list, optional
        Batch identifiers (keys) that will be shown with a heavier line, by default []
    x_axis_label : str, optional
        String label for the x-axis, by default "Time [sequence order]"

    html_image_height : int, optional
        HTML image output height, by default 900
    html_aspect_ratio_w_over_h : float, optional
        HTML image aspect ratio: 16/9 (therefore the default width will be 1600 px)

    Returns
    -------
    dict
        Standard Plotly dict with two keys: `data` and `layout`.
    """
    unique_items = list(df_dict.keys())
    n_colours = len(unique_items)
    random.seed(13)
    colours = list(sns.husl_palette(n_colours))
    random.shuffle(colours)
    colours = [get_rgba_from_triplet(c, string=True) for c in colours]
    colour_assignment = dict(zip(unique_items, colours))

    traces = []
    highlight_traces = []
    highlight_style = dict(width=6, color="rgba(255,0,0,0.9)")
    regular_style = dict()
    for batch_name, batch_df in df_dict.items():
        assert tag in batch_df.columns, f"Tag '{tag}' not found in the batch with id {batch_name}."
        if time_column in batch_df.columns:
            time_data = batch_df[time_column]
        else:
            time_data = list(range(batch_df.shape[0]))

        if batch_name in batches_to_highlight:
            trace = go.Scatter(
                x=time_data,
                y=batch_df[tag],
                line=highlight_style,
                name=batch_name,
                mode="lines",
                opacity=0.8,
            )
            highlight_traces.append(trace)
        else:
            regular_style["color"] = colour_assignment[batch_name]
            trace = go.Scatter(
                x=time_data,
                y=batch_df[tag],
                name=batch_name,
                line=regular_style,
                mode="lines",
                opacity=0.8,
            )
            traces.append(trace)

    # Add the highlighted one last
    traces.extend(highlight_traces)

    layout = go.Layout(
        title=f"For all batches: '{tag}'." + (f" [{str(extra_info)}]" if extra_info else ""),
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
        yaxis=dict(title=tag, gridwidth=2),
        width=html_aspect_ratio_w_over_h * html_image_height,
        height=html_image_height,
    )
    return dict(data=traces, layout=layout)


def plot__tag_time(
    source: pd.Series,
    overlap: bool = False,
    filled: bool = False,
    tag_order: Optional[list] = None,
    x_axis_label: str = "Time, grouped per tag",
    html_image_height: int = 900,
    html_aspect_ratio_w_over_h: float = 16 / 9,
):

    """Plots a vector of information, for every tag [index level 0] over every time [index level 1]

    Parameters
    ----------
    source : pd.Series
        `source` series must be a multi-level Pandas index. Level 0 gives the unique names for each
        tag, and level 1 gives the names for the 'time' or 'sequence' within a tag.
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
    # TODO : Check if multi-level index
    # TODO: assign colours
    # TODO: extract names of tags; reorder if necessary
    # TODO:
