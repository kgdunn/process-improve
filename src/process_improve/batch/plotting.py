from __future__ import annotations

# Built-in libraries
import json
import math
import random
import warnings
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, cast

import numpy as np
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

# Plotting settings -- ENG-13 (#295): live in the optional ``[plotting]``
# extra. Module import succeeds without them; any actual plot call raises
# a clear "install the extra" ImportError via ``_MissingExtra``.
try:
    import plotly.graph_objects as go
    import seaborn as sns
    from plotly.offline import plot as plotoffline
except ImportError:  # pragma: no cover - exercised via env-without-plotly
    from process_improve._extras import _MissingExtra

    go = _MissingExtra("plotly", "plotting")  # type: ignore[assignment]
    sns = _MissingExtra("seaborn", "plotting")  # type: ignore[assignment]
    plotoffline = _MissingExtra("plotly", "plotting")  # type: ignore[assignment]

from process_improve._warnings import user_stacklevel

from .data_input import check_valid_batch_dict


class MultiTagPlotSettings(BaseModel):
    """Settings for :func:`plot_multitags`.

    All fields have sensible defaults; pass a plain dict of overrides to
    ``plot_multitags(..., settings=...)``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    nrows: int = 1
    ncols: int = 0
    x_axis_label: str = "Time, grouped per tag"
    title: str = ""
    show_legend: bool = True
    mode: str = "lines"
    html_image_height: int = 900
    html_aspect_ratio_w_over_h: float = 16 / 9
    default_line_width: float = 2
    # ``colour_map``, the former name, is still accepted; plot_multitags and the colour_map property warn of it.
    color_map: Callable = Field(default=sns.husl_palette, validation_alias=AliasChoices("color_map", "colour_map"))
    animate: bool = False
    animate_batches_to_highlight: list = Field(default_factory=list)
    animate_show_slider: bool = True
    animate_show_pause: bool = True
    animate_slider_prefix: str = "Index: "
    # Fraction of figure height; depends on whether the legend is shown and the
    # length of the batch names.
    animate_slider_vertical_offset: float = -0.3
    # The animated lines are drawn on top of the historical lines.
    animate_line_width: float = 4
    # If None, uses the max number of frames needed for one frame per time step.
    animate_n_frames: int | None = None
    animate_framerate_milliseconds: int = 0

    @property
    def colour_map(self) -> Callable:
        """The former name of ``color_map``: deprecated since 2.7.0 and removed in 3.0."""
        warnings.warn(
            "process_improve.batch.plotting.MultiTagPlotSettings.colour_map is deprecated since 2.7.0 and will be "
            "removed in 3.0; use color_map instead.",
            DeprecationWarning,
            stacklevel=user_stacklevel(),
        )
        return self.color_map


def get_rgba_from_triplet(
    incolor: Sequence[float] | None = None,
    alpha: float = 1,
    as_string: bool = False,
    *,
    incolour: Sequence[float] | None = None,
) -> str | list:
    """
    Convert the input color triplet (list) to a Plotly rgba(r,g,b,a) string if
    `as_string` is True. If `False` it will return the list of 3 integer RGB
    values.

    E.g.    [0.9677975592919913, 0.44127456009157356, 0.5358103155058701] -> 'rgba(246,112,136,1)'

    `incolour`, the former name of `incolor`, is deprecated since 2.7.0 and removed in 3.0.
    """
    if incolour is not None:
        warnings.warn(
            "process_improve.batch.plotting.get_rgba_from_triplet(incolour=...) is deprecated since 2.7.0 and will be "
            "removed in 3.0; use incolor=... instead.",
            DeprecationWarning,
            stacklevel=user_stacklevel(),
        )
        incolor = incolour
    if incolor is None:
        raise TypeError("get_rgba_from_triplet() missing 1 required argument: 'incolor'")
    if not 3 <= len(incolor) <= 4:
        raise ValueError(f"`incolor` must be a list of 3 or 4 values; got {len(incolor)} entries.")
    colors = [max(0, math.floor(c * 255)) for c in list(incolor)[0:3]]
    if as_string:
        return f"rgba({colors[0]},{colors[1]},{colors[2]},{float(alpha)})"
    else:
        return colors


def plot_to_HTML(filename: str, fig: dict) -> str:  # noqa: N802
    """Export a Plotly figure to an HTML file."""
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


def plot_all_batches_per_tag(  # noqa: PLR0912, PLR0913
    df_dict: dict,
    tag: str,
    tag_y2: str | None = None,
    time_column: str | None = None,
    extra_info: str = "",
    batches_to_highlight: dict | None = None,
    x_axis_label: str = "Time [sequence order]",
    highlight_width: int = 5,
    html_image_height: int = 900,
    html_aspect_ratio_w_over_h: float = 16 / 9,
    y1_limits: tuple = (None, None),
    y2_limits: tuple = (None, None),
    mode: str = "lines",
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
        Keys are JSON strings parseable by ``json.loads`` into a Plotly line specifier.
        For example::

            batches_to_highlight = {'{"width": 2, "color": "rgba(255,0,0,0.5)"}': redlist}

        will plot the batch identifiers in ``redlist`` with that color and linewidth.
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
    mode: str, optional
        Plotly trace draw mode, by default "lines". Use "lines+markers" to also
        show a marker at each data point, or "markers" for markers only.


    Returns
    -------
    go.Figure
        Standard Plotly fig object (dictionary-like).
    """
    if batches_to_highlight is None:
        batches_to_highlight = {}
    default_line_width = 2
    unique_items = list(df_dict.keys())
    n_colors = len(unique_items)
    random.seed(13)
    colors = list(sns.husl_palette(n_colors))
    random.shuffle(colors)
    colors = [get_rgba_from_triplet(c, as_string=True) for c in colors]
    line_styles = {k: dict(width=default_line_width, color=v) for k, v in zip(unique_items, colors, strict=False)}
    for key, val in batches_to_highlight.items():
        # SEC-32 (#281): each key must be a JSON-encoded plotly line-style
        # spec. Decode once outside the comprehension so a bad key raises
        # a clear ValueError at the API surface rather than a confusing
        # ``JSONDecodeError`` deep inside the comprehension.
        try:
            style_spec = json.loads(key)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"batches_to_highlight: each key must be a JSON-encoded "
                f'plotly line-style spec (e.g. \'{{"width":4,"color":"red"}}\'). '
                f"Got {key!r}."
            ) from exc
        line_styles.update({item: style_spec for item in val if item in df_dict})

    highlight_list = []
    for val in batches_to_highlight.values():
        highlight_list.extend(val)

    highlight_list = list(set(highlight_list))

    fig = go.Figure()

    for batch_id, batch_df in df_dict.items():
        if tag not in batch_df.columns:
            raise KeyError(f"Tag '{tag}' not found in the batch with id {batch_id}.")
        if tag_y2 and tag_y2 not in batch_df.columns:
            raise KeyError(f"Tag '{tag_y2}' not found in the batch with id {batch_id}.")
        time_data = batch_df[time_column] if time_column in batch_df.columns else list(range(batch_df.shape[0]))

        if batch_id in highlight_list:
            continue  # come to this later
        fig.add_trace(
            go.Scatter(
                x=time_data,
                y=batch_df[tag],
                name=batch_id,
                line=line_styles[batch_id],
                mode=mode,
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
                    mode=mode,
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
                    mode=mode,
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
                        mode=mode,
                        opacity=0.8,
                        yaxis="y2",
                    )
                )

    yaxis1_dict = dict(title=tag, matches="y1", showticklabels=True, side="left")
    if (y1_limits[0] is not None) or (y1_limits[1] is not None):
        yaxis1_dict["autorange"] = False
        yaxis1_dict["range"] = y1_limits

    yaxis2_dict: dict[str, Any] = dict(title=tag_y2, matches="y2", showticklabels=True, side="right")
    if (y2_limits[0] is not None) or (y2_limits[1] is not None):
        yaxis2_dict["autorange"] = False
        yaxis2_dict["range"] = y2_limits

    fig.update_layout(
        title=f"Plot of: '{tag}'"
        + (f" on left axis; with '{tag_y2}' on right axis." if tag_y2 else ".")
        + (f" [{extra_info}]" if extra_info else ""),
        hovermode="closest",
        showlegend=True,
        autosize=False,
        xaxis=dict(title=x_axis_label),
        yaxis=yaxis1_dict,
        width=html_aspect_ratio_w_over_h * html_image_height,
        height=html_image_height,
    )
    if tag_y2:
        fig.update_layout(yaxis2=yaxis2_dict)
    return fig


def colors_per_batch_id(
    batch_ids: list,
    batches_to_highlight: dict,
    default_line_width: float,
    use_default_color: bool = False,
    color_map: Callable | None = None,
) -> dict[Any, dict]:
    """
    Return a color to use for each trace in the plot. A dictionary: keys are batch ids, and
    the value is a color and line width setting for Plotly.

    use_default_color: bool
        If True, then the default color is used (gray: 0.5, 0.5, 0.5)
    """
    if color_map is None:
        color_map = partial(sns.color_palette, "hls")
    random.seed(13)
    n_colors = len(batch_ids)
    colors = list(color_map(n_colors)) if not (use_default_color) else [(0.5, 0.5, 0.5)] * n_colors
    random.shuffle(colors)
    colors = [get_rgba_from_triplet(c, as_string=True) for c in colors]
    color_assignment = {
        key: dict(width=default_line_width, color=val) for key, val in zip(list(batch_ids), colors, strict=False)
    }
    for key, val in batches_to_highlight.items():
        # SEC-32 (#281) -- decode outside the comprehension so a bad key
        # raises ValueError at the API surface rather than JSONDecodeError
        # inside the dict-merge.
        try:
            color_spec = json.loads(key)
        except json.JSONDecodeError as exc:
            raise ValueError(f"batches_to_highlight: each key must be a JSON-encoded color spec. Got {key!r}.") from exc
        color_assignment.update({item: color_spec for item in val if item in batch_ids})
    return color_assignment


def colours_per_batch_id(
    batch_ids: list,
    batches_to_highlight: dict,
    default_line_width: float,
    use_default_colour: bool = False,
    colour_map: Callable | None = None,
) -> dict[Any, dict]:
    """Return :func:`colors_per_batch_id`; this, its former name, is deprecated since 2.7.0 and removed in 3.0."""
    warnings.warn(
        "process_improve.batch.plotting.colours_per_batch_id is deprecated since 2.7.0 and will be removed in 3.0; "
        "use colors_per_batch_id(..., use_default_color=..., color_map=...) instead.",
        DeprecationWarning,
        stacklevel=user_stacklevel(),
    )
    return colors_per_batch_id(
        batch_ids, batches_to_highlight, default_line_width, use_default_color=use_default_colour, color_map=colour_map
    )


def _with_color_map(settings: dict) -> dict:
    """Return ``settings`` with ``colour_map``, the former name of ``color_map``, renamed, with the policy's warning."""
    if "colour_map" not in settings:
        return settings
    warnings.warn(
        "process_improve.batch.plotting.plot_multitags(settings={'colour_map':...}) is deprecated since 2.7.0 and "
        "will be removed in 3.0; use settings={'color_map':...} instead.",
        DeprecationWarning,
        stacklevel=user_stacklevel(),
    )
    renamed = {key: value for key, value in settings.items() if key != "colour_map"}
    renamed.setdefault("color_map", settings["colour_map"])
    return renamed


# flake8: noqa: C901
def plot_multitags(  # noqa: PLR0912, PLR0915
    df_dict: dict,
    batch_list: list | None = None,
    tag_list: list | None = None,
    time_column: str | None = None,
    batches_to_highlight: dict | None = None,
    settings: dict | None = None,
    fig: go.Figure | None = None,
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
        Keys are JSON strings parseable by ``json.loads`` into a Plotly line specifier.
        For example::

            batches_to_highlight = {'{"width": 2, "color": "rgba(255,0,0,0.5)"}': redlist}

        will plot the batch identifiers in ``redlist`` with that color and linewidth.

    settings : dict
        Default settings::

            {
                "nrows": 1,                             # int: number of subplot rows
                "ncols": None,                          # int or None: columns (None = auto)
                "x_axis_label": "Time, grouped per tag",# str: x-axis label
                "title": "",                            # str: overall plot title
                "show_legend": True,                    # bool: show legend
                "mode": "lines",                        # str: Plotly trace mode
                                                        #      e.g. "lines+markers"
                "html_image_height": 900,               # int: image height in pixels
                "html_aspect_ratio_w_over_h": 16/9,     # float: width as ratio of height
            }

    fig : go.Figure
        If supplied, uses the existing Plotly figure to draw in.

    """
    if batches_to_highlight is None:
        batches_to_highlight = {}
    font_size = 12
    hovertemplate = "Time: %{x}\ny: %{y}"

    config = MultiTagPlotSettings(**_with_color_map(settings or {}))

    if len(config.animate_batches_to_highlight) == 0:
        config.animate = False
    if config.animate:
        # override for animations, because we want to see everything in frame zero
        config.default_line_width = 0.5
        # Override these settings for animations, because we want to see everything in frame zero
        animation_color_assignment = colors_per_batch_id(
            batch_ids=list(df_dict.keys()),
            batches_to_highlight=batches_to_highlight or dict(),
            default_line_width=config.animate_line_width,
            use_default_color=False,
            color_map=config.color_map,
        )

    else:
        # Adjust the other animate settings in such a way that the regular functionality works
        config.animate_show_slider = False
        config.animate_show_pause = False
        config.animate_line_width = 0
        config.animate_n_frames = 0
        config.animate_batches_to_highlight = []

    if fig is None:
        fig = go.Figure()

    batch1 = df_dict[next(iter(df_dict.keys()))]
    if tag_list is None:
        tag_list = list(batch1.columns)
    tag_list = list(tag_list)  # Force it; sometimes we get non-list inputs

    if batch_list is None:
        batch_list = list(df_dict.keys())
    batch_list = list(batch_list)
    if config.animate:
        for batch_id in config.animate_batches_to_highlight:
            batch_list.remove(batch_id)
        # Afterwards, add them back, at the end.
        batch_list.extend(config.animate_batches_to_highlight)

    if time_column in tag_list:
        tag_list.remove(time_column)

    # Check that the tag_list is present in all batches; a failed check raises.
    check_valid_batch_dict({k: v[tag_list] for k, v in df_dict.items() if k in batch_list}, no_nan=False)

    if config.ncols == 0:
        config.ncols = int(np.ceil(len(tag_list) / int(config.nrows)))

    # Build a fresh dict per cell: `[[...] * ncols] * nrows` would alias one
    # row list across every row, so a placeholder like `[[None]]` does not work.
    specs: list[list[dict[str, str | bool | int | float] | None]] = [
        [{"type": "scatter"} for _ in range(int(config.ncols))] for _ in range(int(config.nrows))
    ]

    fig.set_subplots(
        rows=config.nrows,
        cols=config.ncols,
        shared_xaxes="all",
        shared_yaxes=False,
        start_cell="top-left",
        vertical_spacing=0.2 / config.nrows,
        horizontal_spacing=0.2 / config.ncols,
        subplot_titles=tag_list,
        specs=specs,
    )

    color_assignment = colors_per_batch_id(
        batch_ids=list(df_dict.keys()),
        batches_to_highlight=batches_to_highlight,
        default_line_width=config.default_line_width,
        # if animating, yes, use gray for all lines; unless `batches_to_highlight` was specified
        use_default_color=config.animate if config.animate and (len(batches_to_highlight) == 0) else False,
        color_map=config.color_map,
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
            showlegend = config.show_legend if tag == tag_list[0] else False
            # This feels right, but leads to the animated batched taking the places of the
            # first few non-animated batches in the legend.
            # Ugh, even without this, they still overwrite them. Sadly.
            # if batch_id in config.animate_batches_to_highlight:
            #    showlegend = False  # overridden. If required, we will add it during the animation

            trace = go.Scatter(
                x=time_data,
                y=batch_df[tag],
                name=batch_id,
                mode=config.mode,
                hovertemplate=hovertemplate,
                line=color_assignment[batch_id],
                legendgroup=batch_id,
                # Only add batch_id to legend the first time it is plotted (the first subplot)
                showlegend=showlegend,
                xaxis=cast("tuple[Any, ...]", fig.get_subplot(row, col))[1]["anchor"],
                yaxis=cast("tuple[Any, ...]", fig.get_subplot(row, col))[0]["anchor"],
            )
            fig.add_trace(trace)

            col += 1
            if col > config.ncols:
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
            "prefix": config.animate_slider_prefix,
            "visible": True,
            "xanchor": "left",
        },
        "transition": {
            "duration": config.animate_framerate_milliseconds,
            "easing": "linear",
        },
        "pad": {"b": 0, "t": 0},
        "lenmode": "fraction",
        "len": 0.9,
        "x": 0.05,
        "y": config.animate_slider_vertical_offset,
        "name": "Slider",
        "steps": [],
    }
    # Create other animation settings. Again, these will be ignored if not needed
    frames: list = []
    slider_steps = []
    frame_settings = dict(
        frame={"duration": config.animate_framerate_milliseconds, "redraw": True},
        mode="immediate",
        transition={"duration": 0},
    )
    n_frames = (
        config.animate_n_frames
        if config.animate_n_frames is not None and config.animate_n_frames >= 0
        else longest_time_length
    )
    config.animate_n_frames = n_frames

    for raw_index in np.linspace(0, longest_time_length, n_frames):
        # TO OPTIMIZE: add hover template only on the last iteration
        # TO OPTIMIZE: can you add only the incremental new piece of animation?

        index = int(np.floor(raw_index))
        frame_name = f"{index}"  # this is the link with the slider and the animation in the play button
        one_frame = generate_one_frame(
            df_dict,
            tag_list,
            fig,
            up_to_index=index + 1,
            time_column=time_column,
            batch_ids_to_animate=config.animate_batches_to_highlight,
            animation_color_assignment=animation_color_assignment,
            show_legend=config.show_legend,
            hovertemplate=hovertemplate,
            max_columns=config.ncols,
            mode=config.mode,
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
            # Note the None is in a list.
            [None],
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
    if config.animate:
        fig.update(frames=frames)
        button_list.append(button_play)
        if config.animate_show_pause:
            button_list.append(button_pause)

    fig.update_layout(
        title=config.title,
        hovermode="closest",
        showlegend=config.show_legend,
        autosize=False,
        width=config.html_aspect_ratio_w_over_h * config.html_image_height,
        height=config.html_image_height,
        sliders=[slider_baseline_dict] if config.animate_show_slider else [],
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


def generate_one_frame(  # noqa: PLR0913
    df_dict: dict,
    tag_list: list,
    fig: go.Figure,
    up_to_index: int,
    time_column: str | None,
    batch_ids_to_animate: list,
    animation_color_assignment: dict | None = None,
    show_legend: bool = False,
    hovertemplate: str = "",
    max_columns: int = 0,
    mode: str = "lines",
    *,
    animation_colour_assignment: dict | None = None,
) -> list[go.Scatter]:
    """
    Return a list of dictionaries.

    Each entry in the list is for each subplot; in the order of the subplots.
    Since each subplot is a tag, we need the `tag_list` as input.

    `animation_colour_assignment`, the former name of `animation_color_assignment`, is deprecated since 2.7.0 and
    removed in 3.0.
    """
    if animation_colour_assignment is not None:
        warnings.warn(
            "process_improve.batch.plotting.generate_one_frame(animation_colour_assignment=...) is deprecated since "
            "2.7.0 and will be removed in 3.0; use animation_color_assignment=... instead.",
            DeprecationWarning,
            stacklevel=user_stacklevel(),
        )
        animation_color_assignment = animation_colour_assignment
    if animation_color_assignment is None:
        raise TypeError("generate_one_frame() missing 1 required argument: 'animation_color_assignment'")
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
                    mode=mode,
                    hovertemplate=hovertemplate,
                    line=animation_color_assignment[batch_id],
                    legendgroup=batch_id,
                    showlegend=show_legend if tag == tag_list[0] else False,
                    xaxis=cast("tuple[Any, ...]", fig.get_subplot(row, col))[1]["anchor"],
                    yaxis=cast("tuple[Any, ...]", fig.get_subplot(row, col))[0]["anchor"],
                )
            )

        # One level outdented: if the loop for the tags, not in the loop for
        # the `batch_ids_to_animate`!
        col += 1
        if col > max_columns:
            row += 1
            col = 1

    return output
