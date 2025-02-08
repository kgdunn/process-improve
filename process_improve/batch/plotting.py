# Built-in libraries
import json
import math
import random
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np

# Plotting settings
import plotly.graph_objects as go
import seaborn as sns
from plotly.offline import plot as plotoffline

from .data_input import check_valid_batch_dict


def get_rgba_from_triplet(incolour: list, alpha: float = 1, as_string: bool = False):
    """
    Convert the input colour triplet (list) to a Plotly rgba(r,g,b,a) string if
    `as_string` is True. If `False` it will return the list of 3 integer RGB
    values.

    E.g.    [0.9677975592919913, 0.44127456009157356, 0.5358103155058701] -> 'rgba(246,112,136,1)'
    """
    assert 3 <= len(incolour) <= 4, "`incolour` must be a list of 3 or 4 values; ignores 4th entry"
    colours = [max(0, int(math.floor(c * 255))) for c in list(incolour)[0:3]]
    if as_string:
        return f"rgba({colours[0]},{colours[1]},{colours[2]},{float(alpha)})"
    else:
        return colours


def plot_to_HTML(filename: str, fig: dict):
    config = dict(
        scrollZoom=True,
        displayModeBar=True,
        editable=False,
        displaylogo=False,
        showLink=False,
        resonsive=True,
    )
    return plotoffline(
        figure_or_data=fig,
        config=config,
        filename=filename,
        include_plotlyjs="cdn",
        include_mathjax="cdn",
        auto_open=False,
    )


def plot_all_batches_per_tag(
    df_dict: dict,
    tag: str,
    tag_y2: str = None,
    time_column: str = None,
    extra_info: str = "",
    batches_to_highlight: dict = {},
    x_axis_label: str = "Time [sequence order]",
    highlight_width: int = 5,
    html_image_height: int = 900,
    html_aspect_ratio_w_over_h: float = 16 / 9,
    y1_limits: tuple = (None, None),
    y2_limits: tuple = (None, None),
) -> go.Figure:
    """Plot a particular `tag` over all batches in the given dataframe `df`.

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
        keys: an string which can be json.loads(...) and turns into a Plotly line specifier.
        For example:
            batches_to_highlight = grouper= {'{"width": 2, "color": "rgba(255,0,0,0.5)"}': redlist}

            will plot batch identifiers (must be valid keys in `df_dict`) in the "redlist" list
            with that colour and linewidth.
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
    default_line_width = 2
    unique_items = list(df_dict.keys())
    n_colours = len(unique_items)
    random.seed(13)
    colours = list(sns.husl_palette(n_colours))
    random.shuffle(colours)
    colours = [get_rgba_from_triplet(c, as_string=True) for c in colours]
    line_styles = {k: dict(width=default_line_width, color=v) for k, v in zip(unique_items, colours)}
    for key, val in batches_to_highlight.items():
        line_styles.update({item: json.loads(key) for item in val if item in df_dict})

    highlight_list = []
    for val in batches_to_highlight.values():
        highlight_list.extend(val)

    highlight_list = list(set(highlight_list))

    fig = go.Figure()

    for batch_id, batch_df in df_dict.items():
        assert tag in batch_df.columns, f"Tag '{tag}' not found in the batch with id {batch_id}."
        if tag_y2:
            assert tag_y2 in batch_df.columns, f"Tag '{tag}' not found in the batch with id {batch_id}."
        time_data = batch_df[time_column] if time_column in batch_df.columns else list(range(batch_df.shape[0]))

        if batch_id in highlight_list:
            continue  # come to this later
        else:
            fig.add_trace(
                go.Scatter(
                    x=time_data,
                    y=batch_df[tag],
                    name=batch_id,
                    line=line_styles[batch_id],
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
                        name=batch_id,
                        line=line_styles[batch_id],
                        mode="lines",
                        opacity=0.8,
                        yaxis="y2",
                    )
                )

    # Add the highlighted batches last: therefore, sadly, we have to do another run-through.
    # Plotly does not yet support z-orders.
    for batch_id, batch_df in df_dict.items():
        time_data = batch_df[time_column] if time_column in batch_df.columns else list(range(batch_df.shape[0]))

        if batch_id in highlight_list:
            fig.add_trace(
                go.Scatter(
                    x=time_data,
                    y=batch_df[tag],
                    line=line_styles[batch_id],
                    name=batch_id,
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
                        line=line_styles[batch_id],
                        name=batch_id,
                        mode="lines",
                        opacity=0.8,
                        yaxis="y2",
                    )
                )

    yaxis1_dict = dict(title=tag, gridwidth=2, matches="y1", showticklabels=True, side="left")
    if (y1_limits[0] is not None) or (y1_limits[1] is not None):
        yaxis1_dict["autorange"] = False
        yaxis1_dict["range"] = y1_limits

    yaxis2_dict: dict[str, Any] = dict(title=tag_y2, gridwidth=2, matches="y2", showticklabels=True, side="right")
    if (y2_limits[0] is not None) or (y2_limits[1] is not None):
        yaxis2_dict["autorange"] = False
        yaxis2_dict["range"] = y2_limits

    fig.update_layout(
        title=f"Plot of: '{tag}'"
        + (f" on left axis; with '{tag_y2}' on right axis." if tag_y2 else ".")
        + (f" [{extra_info}]" if extra_info else ""),
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


def colours_per_batch_id(
    batch_ids: list,
    batches_to_highlight: dict,
    default_line_width: float,
    use_default_colour: bool = False,
    colour_map: Callable = partial(sns.color_palette, "hls"),
) -> dict[Any, dict]:
    """
    Return a colour to use for each trace in the plot. A dictionary: keys are batch ids, and
    the value is a colour and line width setting for Plotly.

    override_default_colour: bool
        If True, then the default colour is used (grey: 0.5, 0.5, 0.5)
    """
    random.seed(13)
    n_colours = len(batch_ids)
    colours = list(colour_map(n_colours)) if not (use_default_colour) else [(0.5, 0.5, 0.5)] * n_colours
    random.shuffle(colours)
    colours = [get_rgba_from_triplet(c, as_string=True) for c in colours]
    colour_assignment = {key: dict(width=default_line_width, color=val) for key, val in zip(list(batch_ids), colours)}
    for key, val in batches_to_highlight.items():
        colour_assignment.update({item: json.loads(key) for item in val if item in batch_ids})
    return colour_assignment


# flake8: noqa: C901
def plot_multitags(
    df_dict: dict,
    batch_list: list = None,
    tag_list: list = None,
    time_column: str = None,
    batches_to_highlight: dict = {},
    settings: dict = None,
    fig=None,
) -> go.Figure:
    """
    Plot all the tags for a batch; or a subset of tags, if specified in `tag_list`.

    Parameters
    ----------
    df_dict : dict
        Standard data format for batches.

    batch_list : list [default: None, will plot all batches in df_dict]
        Which batches to plot; if provided, must be a list of valid keys into df_dict.

    tag_list : list [default: None, will plot all tags in the dataframes]
        Which tags to plot; tags will also be plotted in this order, or in the order of the
        first dataframe if not specified.

    time_column : str, optional
        Which tag on the x-axis. If not specified, creates sequential integers, starting from 0
        if left as the default, `None`.

    batches_to_highlight : dict, optional
        keys: an string which can be json.loads(...) and turns into a Plotly line specifier.
        For example:
            batches_to_highlight = grouper= {'{"width": 2, "color": "rgba(255,0,0,0.5)"}': redlist}

            will plot batch identifiers (must be valid keys in `df_dict`) in the "redlist" list
            with that colour and linewidth.

    settings : dict
        Default settings are = {
            "nrows": 1 [int],
                Number of rows in the plot.

            "ncols": None
                None = use as many columns as required to plot the data; else, supply an integer.

            "x_axis_label": "Time, grouped per tag"  <-- still TODO: make this show up.
                What label is added to the x-axis?

            "title": ""
                Overall plot title

            "show_legend": True,
                Add a legend item for each tag

            "html_image_height": 900,
                in pixels

            "html_aspect_ratio_w_over_h": 16/9,
                sets the image width, as a ratio of the height

        }

    fig : go.Figure
        If supplied, uses the existing Plotly figure to draw in.

    """
    font_size = 12
    margin_dict = dict(l=10, r=10, b=5, t=80)  # Defaults: l=80, r=80, t=100, b=80
    hovertemplate = "Time: %{x}\ny: %{y}"

    # This will be clumsy, until we have Python 3.9. TODO: use pydantic instead
    # This will be clumsy, until we have Python 3.9. TODO: use pydantic instead
    default_settings: dict[str, Any] = dict(
        # Pydantic: int
        nrows=1,
        # Pydantic: int
        ncols=0,
        # Pydantic: str
        x_axis_label="Time, grouped per tag",
        # Pydantic: str
        title="",
        # Pydantic: bool
        show_legend=True,
        # Pydantic: >0
        html_image_height=900,
        # Pydantic: >0
        html_aspect_ratio_w_over_h=16 / 9,
        # Pydantic: >0
        default_line_width=2,
        # Pydantic: callable
        colour_map=sns.husl_palette,
        # Pydantic: bool
        animate=False,
        # Pydantic: list
        animate_batches_to_highlight=[],
        # Pydantic: bool
        animate_show_slider=True,
        # Pydantic: bool
        animate_show_pause=True,
        # Pydantic: str
        animate_slider_prefix="Index: ",
        # Pydantic: bool
        # fraction of figure height. Default should be OK, but depends if the
        # legend is show and length of batch names
        animate_slider_vertical_offset=-0.3,
        # Pydantic: > 0
        animate_line_width=4,  # the animated lines are drawn on top of the historical lines
        # Pydantic: optional or int
        animate_n_frames=None,  # takes max frames required to give every time step 1 frame.
        # Pydantic: int >= 0
        animate_framerate_milliseconds=0,
    )
    if settings:
        default_settings.update(settings)

    settings = default_settings

    if len(settings["animate_batches_to_highlight"]) == 0:
        settings["animate"] = False
    if settings["animate"]:
        # override for animations, because we want to see everything in frame zero
        settings["default_line_width"] = 0.5
        # Override these settings for animations, because we want to see everything in frame zero
        animation_colour_assignment = colours_per_batch_id(
            batch_ids=list(df_dict.keys()),
            batches_to_highlight=batches_to_highlight or dict(),
            default_line_width=settings["animate_line_width"],
            use_default_colour=False,
            colour_map=settings["colour_map"],
        )

    else:
        # Adjust the other animate settings in such a way that the regular functionality works
        settings["animate_show_slider"] = False
        settings["animate_show_pause"] = False
        settings["animate_line_width"] = 0
        settings["animate_n_frames"] = 0
        settings["animate_batches_to_highlight"] = []

    if fig is None:
        fig = go.Figure()

    batch1 = df_dict[next(iter(df_dict.keys()))]
    if tag_list is None:
        tag_list = list(batch1.columns)
    tag_list = list(tag_list)  # Force it; sometimes we get non-list inputs

    if batch_list is None:
        batch_list = list(df_dict.keys())
    batch_list = list(batch_list)
    if settings["animate"]:
        for batch_id in settings["animate_batches_to_highlight"]:
            batch_list.remove(batch_id)
        # Afterwards, add them back, at the end.
        batch_list.extend(settings["animate_batches_to_highlight"])

    if time_column in tag_list:
        tag_list.remove(time_column)

    # Check that the tag_list is present in all batches.
    assert check_valid_batch_dict({k: v[tag_list] for k, v in df_dict.items() if k in batch_list}, no_nan=False)

    if settings["ncols"] == 0:
        settings["ncols"] = int(np.ceil(len(tag_list) / int(settings["nrows"])))

    specs = [[{"type": "scatter"}] * int(settings["ncols"])] * int(settings["nrows"])

    fig.set_subplots(
        rows=settings["nrows"],
        cols=settings["ncols"],
        shared_xaxes="all",
        shared_yaxes=False,
        start_cell="top-left",
        vertical_spacing=0.2 / settings["nrows"],
        horizontal_spacing=0.2 / settings["ncols"],
        subplot_titles=tag_list,
        specs=specs,
    )

    colour_assignment = colours_per_batch_id(
        batch_ids=list(df_dict.keys()),
        batches_to_highlight=batches_to_highlight,
        default_line_width=settings["default_line_width"],
        # if animating, yes, use grey for all lines; unless `batches_to_highlight` was specified
        use_default_colour=settings["animate"] if settings["animate"] and (len(batches_to_highlight) == 0) else False,
        colour_map=settings["colour_map"],
    )

    # Initial plot (what is visible before animation starts)
    longest_time_length: int = 0
    for batch_id in batch_list:
        batch_df = df_dict[batch_id]

        # Time axis values
        time_data = batch_df[time_column] if time_column in batch_df.columns else list(range(batch_df.shape[0]))

        longest_time_length = max(longest_time_length, len(time_data))

        row = col = 1
        for tag in tag_list:
            showlegend = settings["show_legend"] if tag == tag_list[0] else False
            # This feels right, but leads to the animated batched taking the places of the
            # first few non-animated batches in the legend.
            # Ugh, even without this, they still overwrite them. Sadly.
            # if batch_id in settings["animate_batches_to_highlight"]:
            #    showlegend = False  # overridden. If required, we will add it during the animation

            trace = go.Scatter(
                x=time_data,
                y=batch_df[tag],
                name=batch_id,
                mode="lines",
                hovertemplate=hovertemplate,
                line=colour_assignment[batch_id],
                legendgroup=batch_id,
                # Only add batch_id to legend the first time it is plotted (the first subplot)
                showlegend=showlegend,
                xaxis=fig.get_subplot(row, col)[1]["anchor"],
                yaxis=fig.get_subplot(row, col)[0]["anchor"],
            )
            fig.add_trace(trace)

            col += 1
            if col > settings["ncols"]:
                row += 1
                col = 1

    # Create the slider; will be ignore later if not required
    # https://plotly.com/python/reference/layout/sliders/
    slider_baseline_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "font": {"size": font_size},
        "currentvalue": {
            "font": {"size": font_size},
            "prefix": settings["animate_slider_prefix"],
            "visible": True,
            "xanchor": "left",
        },
        "transition": {
            "duration": settings["animate_framerate_milliseconds"],
            "easing": "linear",
        },
        "pad": {"b": 0, "t": 0},
        "lenmode": "fraction",
        "len": 0.9,
        "x": 0.05,
        "y": settings["animate_slider_vertical_offset"],
        "name": "Slider",
        "steps": [],
    }
    # Create other animation settings. Again, these will be ignored if not needed
    frames: list = []
    slider_steps = []
    frame_settings = dict(
        frame={"duration": settings["animate_framerate_milliseconds"], "redraw": True},
        mode="immediate",
        transition={"duration": 0},
    )
    settings["animate_n_frames"] = (
        settings["animate_n_frames"] if settings["animate_n_frames"] >= 0 else longest_time_length
    )

    for index in np.linspace(0, longest_time_length, settings["animate_n_frames"]):
        # TO OPTIMIZE: add hover template only on the last iteration
        # TO OPTIMIZE: can you add only the incremental new piece of animation?

        index = int(np.floor(index))
        frame_name = f"{index}"  # this is the link with the slider and the animation in the play button
        one_frame = generate_one_frame(
            df_dict,
            tag_list,
            fig,
            up_to_index=index + 1,
            time_column=time_column,
            batch_ids_to_animate=settings["animate_batches_to_highlight"],
            animation_colour_assignment=animation_colour_assignment,
            show_legend=settings["show_legend"],
            hovertemplate=hovertemplate,
            max_columns=settings["ncols"],
        )

        frames.append(go.Frame(data=one_frame, name=frame_name))
        slider_dict = dict(
            args=[
                [frame_name],
                frame_settings,
            ],
            label=frame_name,
            method="animate",
        )
        slider_steps.append(slider_dict)

    # Buttons: for animations
    button_play = dict(
        label="Play",
        method="animate",
        args=[
            None,
            dict(
                frame=dict(duration=0, redraw=False),
                transition=dict(duration=30, easing="quadratic-in-out"),
                fromcurrent=True,
                mode="immediate",
            ),
        ],
    )
    button_pause = dict(
        label="Pause",
        method="animate",
        args=[
            # https://plotly.com/python/animations/
            # Note the None is in a list!
            [[None]],  # TODO: does not work at the moment.
            dict(
                frame=dict(duration=0, redraw=False),
                transition=dict(duration=0),
                mode="immediate",
            ),
        ],
    )

    # OK, pull things together to render the fig
    slider_baseline_dict["steps"] = slider_steps
    button_list: list[Any] = []
    if settings["animate"]:
        fig.update(frames=frames)
        button_list.append(button_play)
        if settings["animate_show_pause"]:
            button_list.append(button_pause)

    fig.update_layout(
        title=settings["title"],
        margin=margin_dict,
        hovermode="closest",
        showlegend=settings["show_legend"],
        legend=dict(
            orientation="h",
            traceorder="normal",
            font=dict(family="sans-serif", size=12, color="#000"),
            bordercolor="#DDDDDD",
            borderwidth=1,
        ),
        autosize=False,
        xaxis=dict(
            gridwidth=1,
            mirror=True,  # ticks are mirror at the top of the frame also
            showspikes=True,
            visible=True,
        ),
        yaxis=dict(
            gridwidth=2,
            type="linear",
            autorange=True,
            showspikes=True,
            visible=True,
            showline=True,  # show a separating line
            side="left",  # show on the RHS
        ),
        width=settings["html_aspect_ratio_w_over_h"] * settings["html_image_height"],
        height=settings["html_image_height"],
        sliders=[slider_baseline_dict] if settings["animate_show_slider"] else [],
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=0,
                x=1.05,
                xanchor="left",
                yanchor="bottom",
                buttons=button_list,
            )
        ],
    )

    return fig


def generate_one_frame(
    df_dict: dict,
    tag_list: list,
    fig,
    up_to_index,
    time_column,
    batch_ids_to_animate: list,
    animation_colour_assignment,
    show_legend=False,
    hovertemplate: str = "",
    max_columns=0,
) -> list[dict]:
    """
    Return a list of dictionaries.

    Each entry in the list is for each subplot; in the order of the subplots.
    Since each subplot is a tag, we need the `tag_list` as input.
    """
    output = []
    row = col = 1
    for tag in tag_list:
        for batch_id in batch_ids_to_animate:
            # These 4 lines are duplicated from the outside function
            if time_column in df_dict[batch_id].columns:
                time_data = df_dict[batch_id][time_column]
            else:
                time_data = list(range(df_dict[batch_id].shape[0]))

            output.append(
                go.Scatter(
                    x=time_data[0:up_to_index],
                    y=df_dict[batch_id][tag][0:up_to_index],
                    name=batch_id,
                    mode="lines",
                    hovertemplate=hovertemplate,
                    line=animation_colour_assignment[batch_id],
                    legendgroup=batch_id,
                    showlegend=show_legend if tag == tag_list[0] else False,
                    xaxis=fig.get_subplot(row, col)[1]["anchor"],
                    yaxis=fig.get_subplot(row, col)[0]["anchor"],
                )
            )

        # One level outdented: if the loop for the tags, not in the loop for
        # the `batch_ids_to_animate`!
        col += 1
        if col > max_columns:
            row += 1
            col = 1

    return output
