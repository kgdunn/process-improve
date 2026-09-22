# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.

# Built-in libraries
from __future__ import annotations

import json
import typing
from collections.abc import Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, field_validator
from sklearn.base import BaseEstimator

from ._limits import hotellings_t2_limit, spe_calculation

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:  # pragma: no cover - exercised via env-without-plotly
    from process_improve._extras import _MissingExtra

    go = _MissingExtra("plotly", "plotting")  # type: ignore[assignment]
    make_subplots = _MissingExtra("plotly", "plotting")  # type: ignore[assignment]

from process_improve.visualization.themes import (
    DEFAULT_THEME,
    LIMIT_LINE_COLOR,
    REFERENCE_LINE_COLOR,
)


def _decode_highlight_style(key: str) -> dict:
    """Decode an ``items_to_highlight`` key into a Plotly marker-style dict.

    Each key must be a JSON-encoded Plotly marker/line-style spec. Decoding it
    here (rather than calling ``json.loads`` inline) means a malformed key
    raises a clear ``ValueError`` at the API surface instead of a confusing
    ``json.JSONDecodeError`` deep inside the trace-building loop. Mirrors the
    SEC-32 guard already applied in ``process_improve.batch.plotting``.
    """
    try:
        return json.loads(key)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"items_to_highlight: each key must be a JSON-encoded Plotly "
            f'style spec (e.g. \'{{"color": "red", "symbol": "cross"}}\'). '
            f"Got {key!r}."
        ) from exc


def _fitted_n_components(model: BaseEstimator) -> int:
    """Return the resolved component count of a fitted model (or its TPLS parent).

    Reads the fitted ``n_components_`` (#505: the constructor parameter is the
    user's request and can be ``None``); TPLS sub-model wrappers without one
    delegate to their ``_parent``.
    """
    for candidate in (model, getattr(model, "_parent", None)):
        if candidate is None:
            continue
        if hasattr(candidate, "n_components_"):
            return int(candidate.n_components_)
        if getattr(candidate, "n_components", None) is not None:
            return int(candidate.n_components)
    raise AttributeError("The model has no fitted n_components_; fit the model first.")


def _x_space_loadings(model: BaseEstimator) -> pd.DataFrame:
    """
    Return the X-space (P) loadings of any model the loading plot supports.

    PCA calls them ``loadings_``; PLS and the multi-block models call them
    ``x_loadings_``. TPLS has no single loadings matrix (its loadings are per block:
    ``p_loadings_z``, ``q_loadings_y``, and so on), so the super-level weights stand in,
    mirroring `score_plot`, which uses the super-level ``t_scores_super``.

    The plot functions can be handed either the estimator or its ``Plot`` accessor, so
    the parent is consulted when the accessor itself carries none of these names.

    Parameters
    ----------
    model : BaseEstimator
        A fitted estimator, or the ``Plot`` accessor wrapping one.

    Returns
    -------
    pd.DataFrame
        Loadings, with one column per component.

    Raises
    ------
    AttributeError
        If neither the model nor its parent exposes a usable loadings matrix.
    """
    for candidate in (model, getattr(model, "_parent", None)):
        if candidate is None:
            continue
        for attr in ("loadings_", "x_loadings_", "w_loadings_super"):
            if hasattr(candidate, attr):
                return getattr(candidate, attr)

    raise AttributeError(
        f"{type(model).__name__} exposes no X-space loadings to plot: expected one of "
        "`loadings_`, `x_loadings_` or `w_loadings_super`, on the model or its `_parent`. "
        "Has the model been fitted?"
    )


def plot_pre_checks(model: BaseEstimator, pc_horiz: int, pc_vert: int, pc_depth: int) -> bool:
    """Check the inputs for the plot functions are valid."""
    n_components = _fitted_n_components(model)
    if not 0 < pc_horiz <= n_components:
        raise ValueError(f"The model has {n_components} components. Ensure that 1 <= pc_horiz <= {n_components}.")
    if not 0 < pc_vert <= n_components:
        raise ValueError(f"The model has {n_components} components. Ensure that 1 <= pc_vert <= {n_components}.")
    if not -1 <= pc_depth <= n_components:
        raise ValueError(
            f"The model has {n_components} components. Ensure that pc_depth is -1 (no depth axis) "
            f"or 1 <= pc_depth <= {n_components}."
        )
    if len({pc_horiz, pc_vert, pc_depth}) != 3:
        raise ValueError("Specify distinct components for each axis.")

    return True


def _area_scale(sizes: pd.Series | None, index: pd.Index, size_max: float) -> dict:
    """Return the Plotly keys that make a marker's area, not its diameter, proportional to ``sizes``.

    One ``sizeref`` is computed for the whole series and shared by every trace, so that a
    highlighted point and a plain one of the same value are drawn the same size. Values that
    cannot be an area, or that do not cover the observations being plotted, raise instead.
    """
    if sizes is None:
        return {}
    values = pd.Series(sizes).reindex(index).astype(float)
    if values.isna().any():
        msg = f"`sizes` has no value for these observations: {list(values.index[values.isna()])[:5]}"
        raise ValueError(msg)
    if (values < 0).any():
        msg = "`sizes` cannot be negative: the marker area is proportional to it."
        raise ValueError(msg)
    largest = float(values.max())
    if largest <= 0:
        msg = "`sizes` must have at least one positive value to set the marker scale."
        raise ValueError(msg)
    return {"sizemode": "area", "sizeref": 2.0 * largest / size_max**2, "sizemin": 2}


def _sized_marker(styling: dict, index: list, sizes: pd.Series | None, marker_area: dict) -> dict:
    """Return the marker specification for one trace, with its area carrying ``sizes`` when given."""
    if sizes is None:
        return styling
    return {**styling, "size": pd.Series(sizes).reindex(index).astype(float).to_numpy(), **marker_area}


def _size_hover(index: list, sizes: pd.Series | None, size_name: str) -> dict:
    """Return hover text reporting the value the marker area stands for, so it can be read exactly."""
    if sizes is None:
        return {}
    values = pd.Series(sizes).reindex(index).astype(float)
    label = size_name or "size"
    return {
        "customdata": values.to_numpy(),
        "hovertemplate": "%{text}<br>" + label + ": %{customdata:.4g}<extra></extra>",
    }


def score_plot(  # noqa: C901, PLR0913
    model: BaseEstimator,
    pc_horiz: int = 1,
    pc_vert: int = 2,
    pc_depth: int = -1,
    items_to_highlight: dict[str, list] | None = None,
    settings: dict | None = None,
    fig: go.Figure | None = None,
    *,
    sizes: pd.Series | None = None,
    size_name: str = "",
) -> go.Figure:
    """Generate a 2D or 3D score plot for the given latent variable model.

    A 2D scatter on (``pc_horiz``, ``pc_vert``) is produced by default. Supplying
    ``pc_depth >= 1`` adds a third score axis and switches the underlying trace
    to ``Scatter3d``.

    Parameters
    ----------
    model : MVmodel object (PCA, or PLS)
        A latent variable model generated by this library.
    pc_horiz : int, optional
        Which component to plot on the horizontal axis, by default 1 (the first component)
    pc_vert : int, optional
        Which component to plot on the vertical axis, by default 2 (the second component)
    pc_depth : int, optional
        If pc_depth >= 1, then a 3D score plot is generated, with this component on the 3rd axis
    items_to_highlight : dict, optional
        Keys are JSON strings parseable by ``json.loads`` into a Plotly line specifier;
        values are lists of index names to highlight. For example::

            items_to_highlight = {'{"color": "red", "symbol": "cross"}': items_in_red}

        will highlight the items in ``items_in_red`` with the given colour and shape.

    sizes : pd.Series, optional
        One non-negative value per observation, indexed as the scores are. The marker
        **area** is made proportional to it, so that a marker of twice the area stands for
        twice the value, and the largest value is drawn ``settings["size_max"]`` pixels
        across. The plain and the highlighted traces share one scale, and a highlighted
        point keeps its own area rather than being enlarged, because two meanings on one
        channel cannot both be read. Give the reader that scale as well: an area cannot be
        read off a plot on its own.
    size_name : str, optional
        What ``sizes`` measures, for example ``"SPE"``; it names the value in the hover text.

    settings : dict
        Default settings::

            {
                "show_ellipse": True,          # bool: show the Hotelling's T2 ellipse
                "ellipse_conf_level": 0.95,    # float: ellipse confidence level (< 1.00)
                "title": "",                   # str: overall plot title. The
                                               # default is the empty string on
                                               # the 2D path (pc_depth <= 0) and
                                               # a "Score plot of component ..."
                                               # sentence on the 3D path
                                               # (pc_depth > 0).
                "show_labels": False,          # bool: add a label for each observation
                "show_legend": True,           # bool: show clickable legend
                "size_max": 26,                # float: diameter in pixels of the
                                               # largest marker, when `sizes` is given
                "html_image_height": 500,      # int: image height in pixels
                "html_aspect_ratio_w_over_h": 16/9,  # float: width as ratio of height
                "template": "pi_journal",        # str: registered Plotly theme name
            }

    Examples
    --------
    >>> pca = PCA(n_components=3).fit(X_scaled)
    >>> pca.score_plot()                          # PC1 vs PC2
    >>> pca.score_plot(pc_horiz=1, pc_vert=3)     # PC1 vs PC3
    >>> pca.score_plot(pc_horiz=1, pc_vert=2, pc_depth=3)  # 3D
    """
    plot_pre_checks(model, pc_horiz, pc_vert, pc_depth)
    data_to_plot = model.scores_ if hasattr(model, "scores_") else model._parent.t_scores_super
    ellipse_coordinates = (
        model.ellipse_coordinates if hasattr(model, "ellipse_coordinates") else model._parent.ellipse_coordinates
    )

    class Settings(BaseModel):
        """Validated display settings for the score plot."""

        show_ellipse: bool = True
        ellipse_conf_level: float = 0.95

        @field_validator("ellipse_conf_level")
        @classmethod
        def check_ellipse_conf_level(cls, val: float) -> float:
            """Check confidence value is in range."""
            if val >= 1:
                raise ValueError("0.0 < `ellipse_conf_level` < 1.0")
            if val <= 0:
                raise ValueError("0.0 < `ellipse_conf_level` < 1.0")
            return val

        title: str = (
            f"Score plot of component {pc_horiz} vs component {pc_vert} vs component {pc_depth}" if pc_depth > 0 else ""
        )
        show_labels: bool = False
        show_legend: bool = True
        size_max: float = 26.0
        html_image_height: float = 500.0
        html_aspect_ratio_w_over_h: float = 16 / 9.0
        template: str = DEFAULT_THEME

    setdict = Settings(**settings).model_dump() if settings else Settings().model_dump()
    marker_area = _area_scale(sizes, data_to_plot.index, setdict["size_max"])
    if fig is None:
        fig = go.Figure()

    name = "Scores [T]"
    fig.update_layout(xaxis_title_text=f"PC {pc_horiz}", yaxis_title_text=f"PC {pc_vert}")

    highlights: dict[str, list] = {}
    default_index = data_to_plot.index
    if items_to_highlight is not None:
        highlights = items_to_highlight.copy()
        for key, items in items_to_highlight.items():
            highlights[key] = list(set(items) & set(default_index))
            default_index = (set(default_index) ^ set(highlights[key])) & set(default_index)

    # Ensure it is back to a list
    default_index = list(default_index)

    # 3D plot
    if pc_depth >= 1:
        fig.add_trace(
            go.Scatter3d(
                x=data_to_plot.loc[default_index, pc_horiz],
                y=data_to_plot.loc[default_index, pc_vert],
                z=data_to_plot.loc[default_index, pc_depth],
                name=name,
                mode="markers+text" if setdict["show_labels"] else "markers",
                marker=_sized_marker({"symbol": "circle"}, default_index, sizes, marker_area),
                text=list(default_index),
                textposition="top center",
                **_size_hover(default_index, sizes, size_name),
            )
        )
        # Items to highlight, if any
        for key, index in highlights.items():
            styling = _decode_highlight_style(key)
            fig.add_trace(
                go.Scatter3d(
                    x=data_to_plot.loc[index, pc_horiz],
                    y=data_to_plot.loc[index, pc_vert],
                    z=data_to_plot.loc[index, pc_depth],
                    name=name,
                    mode="markers+text" if setdict["show_labels"] else "markers",
                    marker=_sized_marker(styling, index, sizes, marker_area),
                    text=list(index),
                    textposition="top center",
                    **_size_hover(index, sizes, size_name),
                )
            )
    else:
        # Regular 2D plot
        fig.add_trace(
            go.Scatter(
                x=data_to_plot.loc[default_index, pc_horiz],
                y=data_to_plot.loc[default_index, pc_vert],
                name=name,
                mode="markers+text" if setdict["show_labels"] else "markers",
                marker=_sized_marker({"symbol": "circle", "size": 7}, default_index, sizes, marker_area),
                text=default_index,
                textposition="top center",
                **_size_hover(default_index, sizes, size_name),
            )
        )
        # Items to highlight, if any
        for key, index in highlights.items():
            styling = _decode_highlight_style(key)
            fig.add_trace(
                go.Scatter(
                    x=data_to_plot.loc[index, pc_horiz],
                    y=data_to_plot.loc[index, pc_vert],
                    name=name,
                    mode="markers+text" if setdict["show_labels"] else "markers",
                    marker=_sized_marker(styling, index, sizes, marker_area),
                    text=list(index),
                    textposition="top center",
                    **_size_hover(index, sizes, size_name),
                )
            )
        if setdict["show_ellipse"]:
            ellipse = ellipse_coordinates(
                score_horiz=pc_horiz,
                score_vert=pc_vert,
                conf_level=setdict["ellipse_conf_level"],
            )
            fig.add_hline(y=0, line_color=REFERENCE_LINE_COLOR)
            fig.add_vline(x=0, line_color=REFERENCE_LINE_COLOR)
            fig.add_trace(
                go.Scatter(
                    x=ellipse[0],
                    y=ellipse[1],
                    name=f"Hotelling's T^2 [{setdict['ellipse_conf_level'] * 100:.4g}%]",
                    mode="lines",
                    line=dict(
                        color=LIMIT_LINE_COLOR,
                        width=2,
                    ),
                )
            )

    fig.update_layout(
        template=setdict["template"],
        title_text=setdict["title"],
        hovermode="closest",
        showlegend=setdict["show_legend"],
        autosize=False,
        width=setdict["html_aspect_ratio_w_over_h"] * setdict["html_image_height"],
        height=setdict["html_image_height"],
    )
    if pc_depth >= 1:
        fig.update_layout(
            scene=dict(
                xaxis=fig.to_dict()["layout"]["xaxis"],
                yaxis=fig.to_dict()["layout"]["xaxis"],
                zaxis=dict(
                    title_text=f"PC {pc_depth}",
                    mirror=True,
                    showspikes=True,
                    visible=True,
                    gridwidth=1,
                ),
            ),
        )
    return fig


def loading_plot(  # noqa: PLR0913
    model: BaseEstimator,
    loadings_type: str = "p",
    pc_horiz: int = 1,
    pc_vert: int = 2,
    settings: dict | None = None,
    fig: go.Figure | None = None,
) -> go.Figure:
    """Generate a 2-dimensional loadings for the given latent variable model.

    Parameters
    ----------
    model : MVmodel object (PCA, or PLS)
        A latent variable model generated by this library.

    loadings_type : str, optional
        A choice of the following:
            'p' : (default for PCA) : the P (projection) loadings: only option possible for PCA
            'w' : the W loadings: Suitable for PLS
            'w*' : (default for PLS) the W* (or R) loadings: Suitable for PLS
            'w*c' : the W* (from X-space) with C loadings from the Y-space: Suitable for PLS
            'c' : the C loadings from the Y-space: Suitable for PLS

        For PCA model any other choice besides 'p' will be ignored.

    pc_horiz : int, optional
        Which component to plot on the horizontal axis, by default 1 (the first component)
    pc_vert : int, optional
        Which component to plot on the vertical axis, by default 2 (the second component)
    settings : dict
        Default settings::

            {
                "title": "Loadings plot ...",  # str: overall plot title
                "show_labels": True,           # bool: add a label for each variable
                "html_image_height": 500,      # int: image height in pixels
                "html_aspect_ratio_w_over_h": 16/9,  # float: width as ratio of height
                "template": "pi_journal",        # str: registered Plotly theme name
            }

    Examples
    --------
    >>> pca.loading_plot()                                 # P loadings, PC1 vs PC2
    >>> pls.loading_plot(loadings_type="w*c")              # W* and C loadings
    >>> pls.loading_plot(loadings_type="w", pc_vert=3)     # W loadings, PC1 vs PC3
    """
    plot_pre_checks(model, pc_horiz, pc_vert, pc_depth=0)

    class Settings(BaseModel):
        """Validated display settings for the loadings plot."""

        title: str = f"Loadings plot [{loadings_type.upper()}] of component {pc_horiz} vs component {pc_vert}"
        show_labels: bool = True
        html_image_height: float = 500.0
        html_aspect_ratio_w_over_h: float = 16 / 9.0
        template: str = DEFAULT_THEME

    setdict = Settings(**settings).model_dump() if settings else Settings().model_dump()
    if fig is None:
        fig = go.Figure()

    # Resolve exactly one matrix, lazily. The previous version computed a "PCA default"
    # eagerly, before the branch that was meant to override it, so any model without
    # `loadings_` raised on that line whatever `loadings_type` asked for. That broke
    # `PLS.loading_plot()` for all five documented values (#568), and on the accessor
    # path the same expression resolved to `Plot.loadings`, the bound method, which then
    # reached `.loc` below (#564).
    extra = None
    requested = loadings_type.lower()
    if requested == "p":
        what = _x_space_loadings(model)
    elif requested == "w":
        what = model.x_weights_
    elif requested == "w*":
        what = model.direct_weights_
    elif requested == "w*c":
        loadings_type = loadings_type[0:-1]
        what = model.direct_weights_
        extra = model.y_loadings_
    elif requested == "c":
        what = model.y_loadings_
    else:
        raise ValueError(
            f"loadings_type={loadings_type!r} is not recognized; expected one of 'p', 'w', 'w*', 'w*c' or 'c'."
        )

    fig.add_trace(
        go.Scatter(
            x=what.loc[:, pc_horiz],
            y=what.loc[:, pc_vert],
            name="X-space loadings W*",
            mode="markers+text" if setdict["show_labels"] else "markers",
            marker=dict(
                symbol="circle",
                size=7,
            ),
            # Plotly's `text` wants a sequence of strings; the index may hold any dtype.
            text=[str(label) for label in what.index],
            textposition="top center",
        )
    )
    add_legend = False

    # Note, we have cut off the 'c' from loadings_type
    add_legend = False
    if loadings_type.lower() == "w*" and extra is not None:
        add_legend = True
        fig.add_trace(
            go.Scatter(
                x=extra.loc[:, pc_horiz],
                y=extra.loc[:, pc_vert],
                name="Y-space loadings C",
                mode="markers+text" if setdict["show_labels"] else "markers",
                marker=dict(
                    symbol="star",
                    size=8,
                ),
                text=extra.index,
                textposition="bottom center",
            )
        )

    fig.update_layout(xaxis_title_text=f"PC {pc_horiz}", yaxis_title_text=f"PC {pc_vert}")
    fig.add_hline(y=0, line_color=REFERENCE_LINE_COLOR)
    fig.add_vline(x=0, line_color=REFERENCE_LINE_COLOR)
    fig.update_layout(
        template=setdict["template"],
        title_text=setdict["title"],
        hovermode="closest",
        showlegend=add_legend,
        autosize=False,
        width=setdict["html_aspect_ratio_w_over_h"] * setdict["html_image_height"],
        height=setdict["html_image_height"],
    )
    return fig


def spe_plot(  # noqa: C901
    model: BaseEstimator,
    with_a: int = -1,
    items_to_highlight: dict[str, list] | None = None,
    settings: dict | None = None,
    fig: go.Figure | None = None,
) -> go.Figure:
    """Generate a squared-prediction error (SPE) plot for the given latent variable model using
    `with_a` number of latent variables. The default will use the total number of latent variables
    which have already been fitted.

    Parameters
    ----------
    model : MVmodel object (PCA, or PLS)
        A latent variable model generated by this library.
    with_a : int, optional
        Uses this many number of latent variables, and therefore shows the SPE after this number of
        model components. By default the total number of components fitted will be used.
    items_to_highlight : dict, optional
        Keys are JSON strings parseable by ``json.loads`` into a Plotly line specifier;
        values are lists of index names to highlight. For example::

            items_to_highlight = {'{"color": "red", "symbol": "cross"}': items_in_red}

        will highlight the items in ``items_in_red`` with the given colour and shape.

    settings : dict
        Default settings::

            {
                "show_limit": True,            # bool: show the SPE confidence limit line
                "conf_level": 0.95,            # float: confidence level for limit (< 1.00)
                "title": "SPE plot ...",        # str: overall plot title
                "default_marker": {...},        # dict: e.g. dict(symbol="circle", size=7)
                "show_labels": False,           # bool: add a label for each observation
                "show_legend": False,           # bool: show clickable legend
                "html_image_height": 500,       # int: image height in pixels
                "html_aspect_ratio_w_over_h": 16/9,  # float: width as ratio of height
                "template": "pi_journal",         # str: registered Plotly theme name
            }

    Examples
    --------
    >>> pca.spe_plot()
    >>> pca.spe_plot(settings={"conf_level": 0.99, "show_labels": True})
    """
    # TO CONSIDER: allow a setting `as_line`: which connects the points with line segments
    if with_a < 0:
        # Get the actual name of the last column in the model if negative indexing is used
        with_a = model.spe_.columns[with_a]
    elif with_a == 0:
        raise ValueError("`with_a` must be >= 1, or specified with negative indexing.")

    n_components = _fitted_n_components(model)
    if not with_a <= n_components:
        raise ValueError(f"`with_a` must be <= the number of components fitted ({n_components}); got {with_a}.")

    class Settings(BaseModel):
        """Validated display settings for the SPE plot."""

        show_limit: bool = True
        conf_level: float = 0.95

        @field_validator("conf_level")
        @classmethod
        def check_conf_level(cls, val: float) -> float:
            """Check confidence value is in range."""
            if val >= 1:
                raise ValueError("0.0 < `conf_level` < 1.0")
            if val <= 0:
                raise ValueError("0.0 < `conf_level` < 1.0")
            return val

        title: str = (
            "Squared prediction error plot after "
            f"fitting {with_a} component{'s' if with_a > 1 else ''}"
            f", with the {conf_level * 100}% confidence limit"
        )
        default_marker: dict = dict(symbol="circle", size=7)
        show_labels: bool = False
        show_legend: bool = False
        html_image_height: float = 500.0
        html_aspect_ratio_w_over_h: float = 16 / 9.0
        template: str = DEFAULT_THEME

    setdict = Settings(**settings).model_dump() if settings else Settings().model_dump()
    if fig is None:
        fig = go.Figure()

    name = f"SPE values after {with_a} component{'s' if with_a > 1 else ''}"
    highlights: dict[str, list] = {}
    default_index = model.spe_.index
    if items_to_highlight is not None:
        highlights = items_to_highlight.copy()
        for key, items in items_to_highlight.items():
            highlights[key] = list(set(items) & set(default_index))
            default_index = (set(default_index) ^ set(highlights[key])) & set(default_index)

    # Ensure it is back to a list
    default_index = list(default_index)
    fig.add_trace(
        go.Scatter(
            x=default_index,
            y=model.spe_.loc[default_index, with_a],
            name=name,
            mode="markers+text" if setdict["show_labels"] else "markers",
            marker=setdict["default_marker"],
            text=default_index,
            textposition="top center",
            showlegend=setdict["show_legend"],
        )
    )
    # Items to highlight, if any
    for key, index in highlights.items():
        styling = _decode_highlight_style(key)
        fig.add_trace(
            go.Scatter(
                x=index,
                y=model.spe_.loc[index, with_a],
                name=name,
                mode="markers+text" if setdict["show_labels"] else "markers",
                marker=styling,
                text=index,
                textposition="top center",
            )
        )

    # The limit must be computed from the SPE values at the SAME component
    # count as is being plotted. model.spe_limit() is hard-wired to the last
    # component, so for with_a < n_components it previously drew a limit that
    # is far too low and flagged nearly everything as an outlier.
    limit_SPE_conf_level = spe_calculation(model.spe_[with_a], conf_level=setdict["conf_level"])
    limit_name = f"{setdict['conf_level'] * 100:.3g}% limit"
    fig.add_hline(
        y=limit_SPE_conf_level,
        line_color=LIMIT_LINE_COLOR,
        annotation_text=limit_name,
        annotation_position="bottom right",
        name=limit_name,
    )
    fig.add_hline(y=0, line_color=REFERENCE_LINE_COLOR)
    fig.update_layout(
        template=setdict["template"],
        title_text=setdict["title"],
        hovermode="closest",
        showlegend=setdict["show_legend"],
        autosize=False,
        yaxis_title_text=name,
        width=setdict["html_aspect_ratio_w_over_h"] * setdict["html_image_height"],
        height=setdict["html_image_height"],
    )
    return fig


def t2_plot(  # noqa: C901
    model: BaseEstimator,
    with_a: int = -1,
    items_to_highlight: dict[str, list] | None = None,
    settings: dict | None = None,
    fig: go.Figure | None = None,
) -> go.Figure:
    """Generate a Hotelling's T2 (T^2) plot for the given latent variable model using
    `with_a` number of latent variables. The default will use the total number of latent variables
    which have already been fitted.

    Parameters
    ----------
    model : MVmodel object (PCA, or PLS)
        A latent variable model generated by this library.
    with_a : int, optional
        Uses this many number of latent variables, and therefore shows the Hotelling's T2 after
        this number of model components. By default the total number of components fitted will
        be used.
    items_to_highlight : dict, optional
        Keys are JSON strings parseable by ``json.loads`` into a Plotly line specifier;
        values are lists of index names to highlight. For example::

            items_to_highlight = {'{"color": "red", "symbol": "cross"}': items_in_red}

        will highlight the items in ``items_in_red`` with the given colour and shape.

    settings : dict
        Default settings. The default ``title`` interpolates the *class-level* ``conf_level``
        default (0.95), so a user-supplied ``conf_level`` correctly changes the limit line
        but the auto-generated title text still reads ``95.0%`` unless a ``title`` override
        is passed too::

            {
                "show_limit": True,            # bool: show the T2 confidence limit line
                "conf_level": 0.95,            # float: confidence level for limit (< 1.00)
                "title": "T2 plot ...",         # str: overall plot title
                "default_marker": {...},        # dict: e.g. dict(symbol="circle", size=7)
                "show_labels": False,           # bool: add a label for each observation
                "show_legend": False,           # bool: show clickable legend
                "html_image_height": 500,       # int: image height in pixels
                "html_aspect_ratio_w_over_h": 16/9,  # float: width as ratio of height
                "template": "pi_journal",         # str: registered Plotly theme name
            }

    Examples
    --------
    >>> pca.t2_plot()
    >>> pca.t2_plot(settings={"conf_level": 0.99, "show_labels": True})
    """
    # TO CONSIDER: allow a setting `as_line`: which connects the points with line segments
    if with_a < 0:
        with_a = model.hotellings_t2_.columns[with_a]
    elif with_a == 0:
        raise ValueError("`with_a` must be >= 1, or specified with negative indexing.")

    n_components = _fitted_n_components(model)
    if not with_a <= n_components:
        raise ValueError(f"`with_a` must be <= the number of components fitted ({n_components}); got {with_a}.")

    class Settings(BaseModel):
        """Validated display settings for the Hotelling's T2 plot."""

        show_limit: bool = True
        conf_level: float = 0.95

        @field_validator("conf_level")
        @classmethod
        def check_conf_level(cls, val: float) -> float:
            """Check confidence value is in range."""
            if val >= 1:
                raise ValueError("0.0 < `conf_level` < 1.0")
            if val <= 0:
                raise ValueError("0.0 < `conf_level` < 1.0")
            return val

        title: str = (
            f"Hotelling's T2 plot after fitting {with_a} component{'s' if with_a > 1 else ''}"
            f", with the {conf_level * 100}% confidence limit"
        )
        default_marker: dict = dict(symbol="circle", size=7)
        show_labels: bool = False
        show_legend: bool = False
        html_image_height: float = 500.0
        html_aspect_ratio_w_over_h: float = 16 / 9.0
        template: str = DEFAULT_THEME

    setdict = Settings(**settings).model_dump() if settings else Settings().model_dump()
    if fig is None:
        fig = go.Figure()

    name = f"T2 values after {with_a} component{'s' if with_a > 1 else ''}"
    highlights: dict[str, list] = {}
    default_index = model.hotellings_t2_.index
    if items_to_highlight is not None:
        highlights = items_to_highlight.copy()
        for key, items in items_to_highlight.items():
            highlights[key] = list(set(items) & set(default_index))
            default_index = (set(default_index) ^ set(highlights[key])) & set(default_index)

    # Ensure it is back to a list
    default_index = list(default_index)
    fig.add_trace(
        go.Scatter(
            x=default_index,
            y=model.hotellings_t2_.loc[default_index, with_a],
            name=name,
            mode="markers+text" if setdict["show_labels"] else "markers",
            marker=setdict["default_marker"],
            text=default_index,
            textposition="top center",
            showlegend=setdict["show_legend"],
        )
    )
    # Items to highlight, if any
    for key, index in highlights.items():
        styling = _decode_highlight_style(key)
        fig.add_trace(
            go.Scatter(
                x=index,
                y=model.hotellings_t2_.loc[index, with_a],
                name=name,
                mode="markers+text" if setdict["show_labels"] else "markers",
                marker=styling,
                text=index,
                textposition="top center",
            )
        )

    # The T2 limit must use the SAME component count as the plotted statistic.
    # model.hotellings_t2_limit() is hard-wired to the full model's A, so for
    # with_a < n_components it previously drew a limit that is too high,
    # hiding genuine outliers. (Component names are 1-based integers, so the
    # resolved ``with_a`` column label equals the component count.)
    limit_HT2_conf_level = hotellings_t2_limit(
        conf_level=setdict["conf_level"],
        n_components=int(with_a),
        n_rows=model.hotellings_t2_.shape[0],
    )
    limit_name = f"{setdict['conf_level'] * 100:.3g}% limit"
    fig.add_hline(
        y=limit_HT2_conf_level,
        line_color=LIMIT_LINE_COLOR,
        annotation_text=limit_name,
        annotation_position="bottom right",
        name=limit_name,
    )
    fig.add_hline(y=0, line_color=REFERENCE_LINE_COLOR)
    fig.update_layout(
        template=setdict["template"],
        title_text=setdict["title"],
        hovermode="closest",
        showlegend=setdict["show_legend"],
        autosize=False,
        yaxis_title_text=name,
        width=setdict["html_aspect_ratio_w_over_h"] * setdict["html_image_height"],
        height=setdict["html_image_height"],
    )
    return fig


def explained_variance_plot(
    model: BaseEstimator,
    settings: dict | None = None,
    fig: go.Figure | None = None,
) -> go.Figure:
    """Generate an explained-variance plot for a fitted latent variable model.

    Shows the variance explained by each component as bars, with the cumulative
    variance explained overlaid as a line. For PCA the variance refers to the
    X-block; for PLS it refers to the Y-block.

    Parameters
    ----------
    model : MVmodel object (PCA, or PLS)
        A fitted latent variable model generated by this library.
    settings : dict
        Default settings::

            {
                "as_percentage": True,         # bool: y-axis as a percentage, else a fraction
                "title": "Variance explained ...",   # str: overall plot title
                "bar_color": None,              # str|None: bar colour; None uses the theme
                "line_color": None,             # str|None: line colour; None uses the theme
                "show_legend": True,            # bool: show clickable legend
                "html_image_height": 500,       # int: image height in pixels
                "html_aspect_ratio_w_over_h": 16/9,  # float: width as ratio of height
                "template": "pi_journal",         # str: registered Plotly theme name
            }

    fig : go.Figure, optional
        An existing figure to draw onto. A new figure is created if omitted.

    Examples
    --------
    >>> pca.explained_variance_plot()
    >>> pls.explained_variance_plot(settings={"as_percentage": False})
    """
    if not hasattr(model, "r2_per_component_"):
        msg = "Model is not fitted. Call fit() before plotting the explained variance."
        raise ValueError(msg)

    # The model says which block it explains, rather than the plot inferring it from
    # the class name: PLSDA is a PLS whose r2_per_component_ is likewise the Y block
    # (its class indicators), so a name test would label its plot "X-variance" (#375).
    block_label = f"{getattr(model, '_variance_block', 'X')}-variance"

    class Settings(BaseModel):
        """Validated display settings for the explained-variance plot."""

        as_percentage: bool = True
        title: str = f"{block_label} explained per component"
        bar_color: str | None = None
        line_color: str | None = None
        show_legend: bool = True
        html_image_height: float = 500.0
        html_aspect_ratio_w_over_h: float = 16 / 9.0
        template: str = DEFAULT_THEME

    setdict = Settings(**settings).model_dump() if settings else Settings().model_dump()
    if fig is None:
        fig = go.Figure()

    scale = 100.0 if setdict["as_percentage"] else 1.0
    unit = "%" if setdict["as_percentage"] else "fraction"
    components = [str(component) for component in model.r2_per_component_.index]
    per_component = model.r2_per_component_.to_numpy(dtype=float) * scale
    cumulative = model.r2_cumulative_.to_numpy(dtype=float) * scale

    fig.add_trace(
        go.Bar(
            x=components,
            y=per_component,
            name="Per component",
            marker_color=setdict["bar_color"],
            showlegend=setdict["show_legend"],
        )
    )
    fig.add_trace(
        go.Scatter(
            x=components,
            y=cumulative,
            name="Cumulative",
            mode="lines+markers",
            line=dict(color=setdict["line_color"]),
            showlegend=setdict["show_legend"],
        )
    )
    fig.update_layout(
        template=setdict["template"],
        title_text=setdict["title"],
        hovermode="x",
        showlegend=setdict["show_legend"],
        autosize=False,
        xaxis=dict(title_text="Component", type="category"),
        yaxis=dict(title_text=f"Variance explained ({unit})", rangemode="tozero"),
        width=setdict["html_aspect_ratio_w_over_h"] * setdict["html_image_height"],
        height=setdict["html_image_height"],
    )
    return fig


def _per_component_r2(cumulative: pd.DataFrame, component: int) -> np.ndarray:
    """Per-component R2 (per variable) from a cumulative-R2-per-variable table."""
    columns: list[object] = list(cumulative.columns)
    position = columns.index(component)
    values = cumulative.iloc[:, position].to_numpy(dtype=float)
    if position == 0:
        return values
    return values - cumulative.iloc[:, position - 1].to_numpy(dtype=float)


def _correlation_loadings(cumulative_r2: pd.DataFrame, loadings: pd.DataFrame, component: int) -> np.ndarray:
    """Correlation of each variable with one component's scores.

    The squared correlation loading equals the fraction of a variable's
    variance explained by that component; the sign follows the loading.
    """
    explained = np.clip(_per_component_r2(cumulative_r2, component), 0.0, None)
    signs = np.sign(loadings[component].to_numpy(dtype=float))
    return signs * np.sqrt(explained)


def correlation_loadings_plot(  # noqa: C901, PLR0913
    model: BaseEstimator,
    pc_horiz: int = 1,
    pc_vert: int = 2,
    variance_ellipses: Sequence[float] = (0.5, 1.0),
    settings: dict | None = None,
    fig: go.Figure | None = None,
) -> go.Figure:
    """Generate a correlation loadings plot for a fitted latent variable model.

    Each variable is placed by its correlation with the scores of two
    components. A variable's squared distance from the origin is the fraction
    of its variance explained by those two components, so every variable lies
    inside the unit circle. Concentric ellipses mark variance-explained
    thresholds: a variable beyond the 50% ellipse has at least half of its
    variance captured by the two components shown.

    For PCA the X-variables are shown. For PLS both the X-variables and the
    Y-variables are overlaid against the X-scores, which reveals how process
    variables relate to quality variables.

    Parameters
    ----------
    model : MVmodel object (PCA, or PLS)
        A fitted latent variable model generated by this library.
    pc_horiz : int, default 1
        Component shown on the horizontal axis (1-based).
    pc_vert : int, default 2
        Component shown on the vertical axis (1-based).
    variance_ellipses : sequence of float, default (0.5, 1.0)
        Variance-explained thresholds, each a fraction in (0, 1], at which to
        draw a concentric ellipse. The conventional choice is the 50% and
        100% ellipses; any other thresholds (for example 0.75 and 0.95) are
        equally valid.
    settings : dict
        Default settings::

            {
                "title": "Correlation loadings ...",  # str: overall plot title
                "x_marker_color": None,         # str|None: X-variable marker colour; None uses the theme
                "y_marker_color": None,         # str|None: Y-variable marker colour (PLS); None uses the theme
                "ellipse_color": "grey",        # str: colour of the variance ellipses
                "show_labels": True,            # bool: label each variable
                "show_legend": True,            # bool: show clickable legend (PLS only)
                "html_image_height": 600,       # int: image height in pixels
                "html_aspect_ratio_w_over_h": 1.0,   # float: width as ratio of height
                "template": "pi_journal",         # str: registered Plotly theme name
            }

    fig : go.Figure, optional
        An existing figure to draw onto. A new figure is created if omitted.

    Examples
    --------
    >>> pca.correlation_loadings_plot()
    >>> pls.correlation_loadings_plot(pc_horiz=1, pc_vert=3)
    >>> pca.correlation_loadings_plot(variance_ellipses=(0.75, 0.95))
    """
    if not hasattr(model, "r2_per_variable_"):
        msg = "Model is not fitted. Call fit() before plotting the correlation loadings."
        raise ValueError(msg)

    available = list(model.r2_per_variable_.columns)
    for axis_name, component in (("pc_horiz", pc_horiz), ("pc_vert", pc_vert)):
        if component not in available:
            msg = f"{axis_name}={component} is not a fitted component; choose from {available}."
            raise ValueError(msg)
    if pc_horiz == pc_vert:
        msg = "pc_horiz and pc_vert must be different components."
        raise ValueError(msg)
    ellipse_levels = [float(level) for level in variance_ellipses]
    for level in ellipse_levels:
        if not 0 < level <= 1:
            msg = f"Each value in variance_ellipses must be a fraction in (0, 1]; got {level}."
            raise ValueError(msg)

    is_pls = hasattr(model, "r2y_per_variable_")
    x_loadings = model.x_loadings_ if is_pls else model.loadings_

    class Settings(BaseModel):
        """Validated display settings for the correlation-loadings plot."""

        title: str = f"Correlation loadings: components {pc_horiz} and {pc_vert}"
        x_marker_color: str | None = None
        y_marker_color: str | None = None
        ellipse_color: str = "grey"
        show_labels: bool = True
        show_legend: bool = True
        html_image_height: float = 600.0
        html_aspect_ratio_w_over_h: float = 1.0
        template: str = DEFAULT_THEME

    setdict = Settings(**settings).model_dump() if settings else Settings().model_dump()
    if fig is None:
        fig = go.Figure()

    # Variance ellipses, drawn first so the variable markers sit on top.
    for level in ellipse_levels:
        radius = float(np.sqrt(level))
        fig.add_shape(
            type="circle",
            xref="x",
            yref="y",
            x0=-radius,
            y0=-radius,
            x1=radius,
            y1=radius,
            line=dict(color=setdict["ellipse_color"], width=1, dash="dot"),
        )
        fig.add_annotation(
            x=0,
            y=radius,
            text=f"{level * 100:g}%",
            showarrow=False,
            yshift=9,
            font=dict(color=setdict["ellipse_color"], size=11),
        )
    fig.add_hline(y=0, line_color=REFERENCE_LINE_COLOR, line_width=1)
    fig.add_vline(x=0, line_color=REFERENCE_LINE_COLOR, line_width=1)

    mode = "markers+text" if setdict["show_labels"] else "markers"
    fig.add_trace(
        go.Scatter(
            x=_correlation_loadings(model.r2_per_variable_, x_loadings, pc_horiz),
            y=_correlation_loadings(model.r2_per_variable_, x_loadings, pc_vert),
            mode=mode,
            text=[str(name) for name in model.r2_per_variable_.index],
            textposition="top center",
            marker=dict(color=setdict["x_marker_color"], size=8, symbol="circle"),
            name="X-variables",
        )
    )
    if is_pls:
        fig.add_trace(
            go.Scatter(
                x=_correlation_loadings(model.r2y_per_variable_, model.y_loadings_, pc_horiz),
                y=_correlation_loadings(model.r2y_per_variable_, model.y_loadings_, pc_vert),
                mode=mode,
                text=[str(name) for name in model.r2y_per_variable_.index],
                textposition="top center",
                marker=dict(color=setdict["y_marker_color"], size=9, symbol="diamond"),
                name="Y-variables",
            )
        )

    def _axis_title(component: int) -> str:
        explained = float(model.r2_per_component_[component]) * 100.0
        return f"Component {component} ({explained:.1f}%)"

    axis_common: dict = dict(range=[-1.08, 1.08], zeroline=False)
    fig.update_layout(
        template=setdict["template"],
        title_text=setdict["title"],
        hovermode="closest",
        showlegend=setdict["show_legend"] and is_pls,
        autosize=False,
        xaxis=dict(title_text=_axis_title(pc_horiz), **axis_common),
        yaxis=dict(title_text=_axis_title(pc_vert), scaleanchor="x", scaleratio=1, **axis_common),
        width=setdict["html_aspect_ratio_w_over_h"] * setdict["html_image_height"],
        height=setdict["html_image_height"],
    )
    return fig


def predictions_vs_observed_plot(
    model: BaseEstimator,
    *,
    y_observed: pd.DataFrame | np.ndarray,
    variable: str | None = None,
    settings: dict | None = None,
    fig: go.Figure | None = None,
) -> go.Figure:
    """Generate an observed-vs-predicted (parity) plot for a fitted PLS model.

    Plots the calibration predictions against the observed Y values, with a
    ``y = x`` reference line and an RMSE annotation. Points lying close to the
    reference line indicate good predictions.

    Parameters
    ----------
    model : PLS object
        A fitted PLS model generated by this library.
    y_observed : array-like of shape (n_samples, n_targets)
        The observed Y values, on the same scale as the data used to fit the
        model (for example the scaled Y from :class:`MCUVScaler`).
    variable : str, optional
        Which Y-variable to plot. Defaults to the first Y-variable.
    settings : dict
        Default settings::

            {
                "title": "Observed vs predicted ...",  # str: overall plot title
                "marker_color": None,           # str|None: data-marker colour; None uses the theme
                "reference_color": "#9CA3AF",   # str: colour of the y = x line
                "html_image_height": 500,       # int: image height in pixels
                "html_aspect_ratio_w_over_h": 1.0,   # float: width as ratio of height
                "template": "pi_journal",         # str: registered Plotly theme name
            }

    fig : go.Figure, optional
        An existing figure to draw onto. A new figure is created if omitted.

    Examples
    --------
    >>> pls.predictions_vs_observed_plot(y_observed=Y_scaled)
    >>> pls.predictions_vs_observed_plot(y_observed=Y_scaled, variable="quality")
    """
    if not hasattr(model, "predictions_"):
        msg = "Model is not fitted. Call fit() before plotting predictions vs observed."
        raise ValueError(msg)

    y_observed = y_observed if isinstance(y_observed, pd.DataFrame) else pd.DataFrame(y_observed)
    if variable is None:
        variable = str(model.predictions_.columns[0])
    if variable not in model.predictions_.columns:
        msg = f"Unknown Y-variable '{variable}'. Known: {list(model.predictions_.columns)}."
        raise ValueError(msg)
    if variable not in y_observed.columns:
        msg = f"y_observed has no column '{variable}'. Its columns are {list(y_observed.columns)}."
        raise ValueError(msg)
    if y_observed.shape[0] != model.predictions_.shape[0]:
        msg = (
            f"y_observed must have {model.predictions_.shape[0]} rows (the number of training "
            f"observations), got {y_observed.shape[0]}."
        )
        raise ValueError(msg)

    observed = y_observed[variable].to_numpy(dtype=float)
    predicted = model.predictions_[variable].to_numpy(dtype=float)
    rmse = float(np.sqrt(np.mean((observed - predicted) ** 2)))
    lo = float(min(observed.min(), predicted.min()))
    hi = float(max(observed.max(), predicted.max()))
    pad = 0.05 * (hi - lo) if hi > lo else 1.0

    class Settings(BaseModel):
        """Validated display settings for the predictions-vs-observed plot."""

        title: str = f"Observed vs predicted for {variable}"
        marker_color: str | None = None
        reference_color: str = REFERENCE_LINE_COLOR
        html_image_height: float = 500.0
        html_aspect_ratio_w_over_h: float = 1.0
        template: str = DEFAULT_THEME

    setdict = Settings(**settings).model_dump() if settings else Settings().model_dump()
    if fig is None:
        fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=[lo - pad, hi + pad],
            y=[lo - pad, hi + pad],
            mode="lines",
            line=dict(color=setdict["reference_color"], dash="dash"),
            name="y = x",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=observed,
            y=predicted,
            mode="markers",
            marker=dict(color=setdict["marker_color"], size=7),
            name="Observations",
        )
    )
    fig.add_annotation(
        x=lo + 0.05 * (hi - lo),
        y=hi - 0.05 * (hi - lo),
        text=f"RMSE = {rmse:.4g}",
        showarrow=False,
    )
    axis_common: dict = dict(range=[lo - pad, hi + pad])
    fig.update_layout(
        template=setdict["template"],
        title_text=setdict["title"],
        hovermode="closest",
        showlegend=False,
        autosize=False,
        xaxis=dict(title_text=f"Observed: {variable}", **axis_common),
        yaxis=dict(title_text=f"Predicted: {variable}", scaleanchor="x", scaleratio=1, **axis_common),
        width=setdict["html_aspect_ratio_w_over_h"] * setdict["html_image_height"],
        height=setdict["html_image_height"],
    )
    return fig


def coefficient_plot(
    model: BaseEstimator,
    variable: str | None = None,
    settings: dict | None = None,
    fig: go.Figure | None = None,
) -> go.Figure:
    """Generate a bar plot of the PLS regression coefficients.

    Shows ``beta_coefficients_`` for one Y-variable: one bar per X-variable,
    mapping the (preprocessed) X onto the predicted Y. Tall bars mark the
    X-variables that most strongly drive the prediction.

    Parameters
    ----------
    model : PLS object
        A fitted PLS model generated by this library.
    variable : str, optional
        Which Y-variable's coefficients to plot. Defaults to the first one.
    settings : dict
        Default settings::

            {
                "title": "Regression coefficients ...",  # str: overall plot title
                "bar_color": None,              # str|None: bar colour; None uses the theme
                "html_image_height": 500,       # int: image height in pixels
                "html_aspect_ratio_w_over_h": 16/9,  # float: width as ratio of height
                "template": "pi_journal",         # str: registered Plotly theme name
            }

    fig : go.Figure, optional
        An existing figure to draw onto. A new figure is created if omitted.

    Examples
    --------
    >>> pls.coefficient_plot()
    >>> pls.coefficient_plot(variable="quality")
    """
    if not hasattr(model, "beta_coefficients_"):
        msg = "Model is not fitted. Call fit() before plotting the coefficients."
        raise ValueError(msg)

    if variable is None:
        variable = str(model.beta_coefficients_.columns[0])
    if variable not in model.beta_coefficients_.columns:
        msg = f"Unknown Y-variable '{variable}'. Known: {list(model.beta_coefficients_.columns)}."
        raise ValueError(msg)

    coefficients = model.beta_coefficients_[variable]
    features = [str(name) for name in coefficients.index]

    class Settings(BaseModel):
        """Validated display settings for the regression-coefficient plot."""

        title: str = f"Regression coefficients for {variable}"
        bar_color: str | None = None
        html_image_height: float = 500.0
        html_aspect_ratio_w_over_h: float = 16 / 9.0
        template: str = DEFAULT_THEME

    setdict = Settings(**settings).model_dump() if settings else Settings().model_dump()
    if fig is None:
        fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=features,
            y=coefficients.to_numpy(dtype=float),
            marker_color=setdict["bar_color"],
            name=f"beta: {variable}",
        )
    )
    fig.add_hline(y=0, line_color=REFERENCE_LINE_COLOR, line_width=1)
    fig.update_layout(
        template=setdict["template"],
        title_text=setdict["title"],
        hovermode="x",
        showlegend=False,
        autosize=False,
        xaxis=dict(title_text="X-variable", type="category"),
        yaxis=dict(title_text=f"Coefficient ({variable})"),
        width=setdict["html_aspect_ratio_w_over_h"] * setdict["html_image_height"],
        height=setdict["html_image_height"],
    )
    return fig


def confusion_matrix_plot(
    model: BaseEstimator,
    matrix: pd.DataFrame | None = None,
    settings: dict | None = None,
    fig: go.Figure | None = None,
) -> go.Figure:
    """Generate a confusion-matrix heat map for a fitted :class:`PLSDA` model.

    Rows are the true class, columns the predicted one, so the diagonal is what the model
    got right and every off-diagonal cell names a specific confusion: which class this one
    is mistaken for, which is the question a classification report cannot answer.

    Parameters
    ----------
    model : PLSDA object
        A fitted PLS-DA model generated by this library.
    matrix : pd.DataFrame, optional
        A confusion matrix to plot instead of the model's training-set one, indexed and
        labelled by class. Pass ``model.confusion(X_test, y_test).matrix`` to see the
        held-out picture, which is the one worth acting on: ``confusion_matrix_`` is
        fitted on the same rows it is scored on and will always look better.
    settings : dict
        Default settings::

            {
                "normalize": False,             # bool: show row fractions, not counts
                "title": "Confusion matrix",    # str: overall plot title
                "colorscale": "Blues",          # str: any Plotly colorscale name
                "show_values": True,            # bool: print the value in each cell
                "html_image_height": 500,       # int: image height in pixels
                "html_aspect_ratio_w_over_h": 1.0,  # float: width as ratio of height
                "template": "pi_journal",       # str: registered Plotly theme name
            }

    fig : go.Figure, optional
        An existing figure to draw onto. A new figure is created if omitted.

    Returns
    -------
    fig : go.Figure

    Raises
    ------
    ValueError
        If the model is not fitted and no ``matrix`` is supplied.

    Examples
    --------
    >>> model.confusion_matrix_plot()                                     # doctest: +SKIP
    >>> held_out = model.confusion(X_test, y_test).matrix                 # doctest: +SKIP
    >>> model.confusion_matrix_plot(held_out, {"normalize": True})        # doctest: +SKIP
    """
    if matrix is None:
        if not hasattr(model, "confusion_matrix_"):
            msg = "Model is not fitted. Call fit() before plotting the confusion matrix, or pass `matrix`."
            raise ValueError(msg)
        matrix = model.confusion_matrix_

    class Settings(BaseModel):
        """Validated display settings for the confusion-matrix plot."""

        normalize: bool = False
        title: str = "Confusion matrix"
        colorscale: str = "Blues"
        show_values: bool = True
        html_image_height: float = 500.0
        html_aspect_ratio_w_over_h: float = 1.0
        template: str = DEFAULT_THEME

    setdict = Settings(**settings).model_dump() if settings else Settings().model_dump()
    if fig is None:
        fig = go.Figure()

    counts = matrix.to_numpy(dtype=float)
    if setdict["normalize"]:
        # Row-normalise: each row then reads as "of the samples that really were this
        # class, what fraction went where". A row with no samples stays at zero rather
        # than becoming NaN, so the heat map has no holes in it.
        row_totals = counts.sum(axis=1, keepdims=True)
        values = np.divide(counts, row_totals, out=np.zeros_like(counts), where=row_totals > 0)
        text_format, colorbar_title = "{:.2f}", "Fraction of true class"
    else:
        values = counts
        text_format, colorbar_title = "{:.0f}", "Samples"

    # Built as a dict because plotly's `Heatmap` stubs type `x` / `y` as a scalar or a
    # float array and `text` as a flat sequence, while a categorical axis takes a list of
    # strings and an annotated heat map takes a list of rows. Both work at runtime; the
    # stubs are narrower than the widget.
    trace: dict[str, typing.Any] = {
        "z": values,
        "x": [str(label) for label in matrix.columns],
        "y": [str(label) for label in matrix.index],
        "colorscale": setdict["colorscale"],
        "colorbar": dict(title=colorbar_title),
        "hovertemplate": "True %{y}, predicted %{x}: %{z}<extra></extra>",
    }
    if setdict["show_values"]:
        trace["text"] = [[text_format.format(cell) for cell in row] for row in values]
        trace["texttemplate"] = "%{text}"
    fig.add_trace(go.Heatmap(**trace))
    fig.update_layout(
        template=setdict["template"],
        title_text=setdict["title"],
        xaxis=dict(title_text="Predicted class", type="category"),
        # Top-to-bottom row order, so the matrix reads the way it is printed.
        yaxis=dict(title_text="True class", type="category", autorange="reversed"),
        width=setdict["html_aspect_ratio_w_over_h"] * setdict["html_image_height"],
        height=setdict["html_image_height"],
    )
    return fig


def effect_summary_plot(
    model: BaseEstimator,
    settings: dict | None = None,
    fig: go.Figure | None = None,
) -> go.Figure:
    """Generate the per-term effect summary for a fitted :class:`ASCA` model.

    One bar per design term, showing the share of the total sum of squares it carries,
    with the residual alongside for scale. This is the plot to read first: it says which
    factor the variation actually belongs to, before any score plot is opened.

    Permutation p-values are annotated on the bars when
    :meth:`~process_improve.multivariate.ASCA.permutation_test` has been run, because a
    term's share and its significance answer different questions: a term can hold a large
    share simply by having many degrees of freedom.

    Parameters
    ----------
    model : ASCA object
        A fitted ASCA model generated by this library.
    settings : dict
        Default settings::

            {
                "include_residual": True,       # bool: draw the residual bar too
                "title": "Variation by design term",  # str: overall plot title
                "bar_color": None,              # str|None: bar colour; None uses the theme
                "html_image_height": 500,       # int: image height in pixels
                "html_aspect_ratio_w_over_h": 16/9,  # float: width as ratio of height
                "template": "pi_journal",       # str: registered Plotly theme name
            }

    fig : go.Figure, optional
        An existing figure to draw onto. A new figure is created if omitted.

    Returns
    -------
    fig : go.Figure

    Raises
    ------
    ValueError
        If the model is not fitted.

    Examples
    --------
    >>> model.effect_summary_plot()                                    # doctest: +SKIP
    >>> model.permutation_test(random_state=0)                          # doctest: +SKIP
    >>> model.effect_summary_plot()   # now annotated with p-values     # doctest: +SKIP
    """
    if not hasattr(model, "ssq_percent_"):
        msg = "Model is not fitted. Call fit() before plotting the effect summary."
        raise ValueError(msg)

    class Settings(BaseModel):
        """Validated display settings for the ASCA effect-summary plot."""

        include_residual: bool = True
        title: str = "Variation by design term"
        bar_color: str | None = None
        html_image_height: float = 500.0
        html_aspect_ratio_w_over_h: float = 16 / 9.0
        template: str = DEFAULT_THEME

    setdict = Settings(**settings).model_dump() if settings else Settings().model_dump()
    if fig is None:
        fig = go.Figure()

    terms = [*model.terms_, "residual"] if setdict["include_residual"] else list(model.terms_)
    shares = [float(model.ssq_percent_[name]) for name in terms]
    pvalues = getattr(model, "pvalues_", None)
    labels = [
        f"{share:.1f}%" if pvalues is None or name not in pvalues else f"{share:.1f}%<br>p={pvalues[name]:.3f}"
        for name, share in zip(terms, shares, strict=True)
    ]

    fig.add_trace(
        go.Bar(
            x=terms,
            y=shares,
            marker_color=setdict["bar_color"],
            text=labels,
            textposition="outside",
            hovertemplate="%{x}: %{y:.2f}% of the total sum of squares<extra></extra>",
            showlegend=False,
        )
    )
    fig.update_layout(
        template=setdict["template"],
        title_text=setdict["title"],
        xaxis=dict(title_text="Design term", type="category"),
        yaxis=dict(title_text="Share of total sum of squares (%)", rangemode="tozero"),
        width=setdict["html_aspect_ratio_w_over_h"] * setdict["html_image_height"],
        height=setdict["html_image_height"],
    )
    return fig


class Plot:
    """Create plots of estimators."""

    def __init__(self, parent: BaseEstimator) -> None:
        self._parent = parent

    def scores(self, pc_horiz: int = 1, pc_vert: int = 2, **kwargs) -> go.Figure:
        """Generate a score plot."""
        return score_plot(self, pc_horiz=pc_horiz, pc_vert=pc_vert, **kwargs)

    def loadings(self, pc_horiz: int = 1, pc_vert: int = 2, **kwargs) -> go.Figure:
        """Generate a loading plot."""
        return loading_plot(self, pc_horiz=pc_horiz, pc_vert=pc_vert, **kwargs)


#: Which panel of :func:`cv_criteria_plot` shows each selection rule's recommendation.
_CV_CRITERIA_PANELS: dict[str, tuple[int, int, str]] = {
    "q2_max": (1, 1, "Q2 max"),
    "q2_1se": (1, 1, "1-SE"),
    "van_der_voet": (1, 1, "van der Voet"),
    "score_correlation": (1, 2, "held-out r"),
    "covariance_permutation": (1, 2, "covariance"),
    "subspace_stability": (2, 1, "subspace"),
    "pv_spe_alarm": (2, 2, "SPE alarms"),
}


def cv_criteria_plot(result: typing.Any, settings: dict | None = None) -> go.Figure:  # noqa: ANN401
    """Plot the per-component table of :func:`compare_cv_criteria` as four small multiples.

    Each panel answers one question and marks the component count the matching
    selection rules recommend with a dashed vertical line:

    1. *Does the model predict Y?* In-sample :math:`R^2_Y` and cross-validated
       :math:`Q^2_Y` with a one-standard-error band (rules ``q2_max``, ``q2_1se``,
       ``van_der_voet``).
    2. *Does the inner relation hold on new rows?* Training and held-out correlation of
       the :math:`t_a` and :math:`u_a` scores, with the permutation-null threshold
       (rules ``score_correlation``, ``covariance_permutation``).
    3. *Do the weights stay put?* Jackknife-scaled per-component and subspace angles,
       with the stability threshold (rule ``subspace_stability``).
    4. *Are the monitoring limits right on new rows?* Out-of-sample SPE and
       :math:`T^2` alarm rates from Procrustes cross-validation, with the nominal rate
       and its binomial upper bound (rule ``pv_spe_alarm``).

    Parameters
    ----------
    result : sklearn.utils.Bunch
        The return value of :func:`compare_cv_criteria`.
    settings : dict, optional
        Default settings::

            {
                "title": "Validation criteria per component",  # str: overall title
                "html_image_height": 720,          # int: image height in pixels
                "html_aspect_ratio_w_over_h": 1.5, # float: width as ratio of height
                "template": "pi_journal",          # str: registered Plotly theme name
            }

    Returns
    -------
    go.Figure

    Examples
    --------
    >>> result = compare_cv_criteria(X, y, max_components=6, random_state=0)
    >>> cv_criteria_plot(result)
    """
    if not hasattr(result, "table") or not hasattr(result, "recommendations"):
        msg = "cv_criteria_plot expects the Bunch returned by compare_cv_criteria."
        raise ValueError(msg)

    class Settings(BaseModel):
        """Validated display settings for the validation-criteria plot."""

        title: str = "Validation criteria per component"
        html_image_height: float = 720.0
        html_aspect_ratio_w_over_h: float = 1.5
        template: str = DEFAULT_THEME

    setdict = Settings(**settings).model_dump() if settings else Settings().model_dump()
    table = result.table
    components = table.index.to_numpy()
    first, second = "#0072B2", "#D55E00"  # the first two slots of the pi_journal colorway

    fig = make_subplots(
        rows=2,
        cols=2,
        shared_xaxes=True,
        horizontal_spacing=0.09,
        vertical_spacing=0.14,
        subplot_titles=[
            "Does the model predict Y?",
            "Does the inner relation hold on new rows?",
            "Do the weights stay put?",
            "Are the monitoring limits right on new rows?",
        ],
    )

    panel_legend = {(1, 1): "legend", (1, 2): "legend2", (2, 1): "legend3", (2, 2): "legend4"}

    def _line(name: str, values: object, color: str, cell: tuple[int, int]) -> None:
        fig.add_trace(
            go.Scatter(
                x=components,
                y=np.asarray(values, dtype=float),
                name=name,
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=8, color=color),
                legend=panel_legend[cell],
                hovertemplate="%{y:.3g}",
            ),
            row=cell[0],
            col=cell[1],
        )

    def _reference(name: str, values: object, color: str, cell: tuple[int, int]) -> None:
        fig.add_trace(
            go.Scatter(
                x=components,
                y=np.broadcast_to(np.asarray(values, dtype=float), components.shape),
                name=name,
                mode="lines",
                line=dict(color=color, width=1.5, dash="dash", shape="hvh"),
                legend=panel_legend[cell],
                hovertemplate="%{y:.3g}",
            ),
            row=cell[0],
            col=cell[1],
        )

    # 1. Prediction, with a +/- 1 SE band around Q2.
    q2 = table["q2y"].to_numpy(dtype=float)
    q2_se = np.nan_to_num(table["q2y_se"].to_numpy(dtype=float))
    fig.add_trace(
        go.Scatter(
            x=np.r_[components, components[::-1]],
            y=np.r_[q2 + q2_se, (q2 - q2_se)[::-1]],
            fill="toself",
            fillcolor="rgba(213, 94, 0, 0.15)",
            line=dict(width=0),
            hoverinfo="skip",
            name="Q2 +/- 1 SE",
            legend="legend",
        ),
        row=1,
        col=1,
    )
    _line("R2Y (training)", table["r2y"], first, (1, 1))
    _line("Q2Y (cross-validated)", q2, second, (1, 1))

    # 2. Inner relation.
    _line("r(t, u) training", table["r_train"], first, (1, 2))
    _line("r(t, u) held-out", table["r_cv"], second, (1, 2))
    _reference("null threshold", table["r_cv_threshold"], REFERENCE_LINE_COLOR, (1, 2))

    # 3. Stability of the weights.
    _line("per component", table["angle_component_deg"], first, (2, 1))
    _line("subspace", table["angle_subspace_deg"], second, (2, 1))
    _reference("threshold", result.angle_threshold, REFERENCE_LINE_COLOR, (2, 1))

    # 4. Monitoring.
    _line("SPE", table["pv_spe_alarm_rate"], first, (2, 2))
    _line("T2", table["pv_t2_alarm_rate"], second, (2, 2))
    _reference("nominal", 1 - result.conf_level, REFERENCE_LINE_COLOR, (2, 2))
    _reference("binomial upper", result.alarm_rate_upper, LIMIT_LINE_COLOR, (2, 2))

    # Recommendations: one dashed line per distinct pick in a panel, labelled with the rules.
    picks: dict[tuple[int, int, int], list[str]] = {}
    for rule, n_components in result.recommendations["n_components"].items():
        if rule in _CV_CRITERIA_PANELS:
            row, col, label = _CV_CRITERIA_PANELS[rule]
            picks.setdefault((row, col, int(n_components)), []).append(label)
    for (row, col, n_components), labels in picks.items():
        text = ", ".join(labels)
        if n_components == 0:
            fig.add_annotation(
                text=f"{text}: none validated",
                xref="x domain" if (row, col) == (1, 1) else f"x{(row - 1) * 2 + col} domain",
                yref="y domain" if (row, col) == (1, 1) else f"y{(row - 1) * 2 + col} domain",
                x=0.02,
                y=0.02,
                xanchor="left",
                yanchor="bottom",
                showarrow=False,
                font=dict(size=10),
            )
            continue
        fig.add_vline(
            x=n_components,
            line=dict(color=REFERENCE_LINE_COLOR, width=1, dash="dot"),
            annotation_text=text,
            annotation_font_size=10,
            annotation_position="top",
            row=row,
            col=col,
        )

    legend_style = dict(font=dict(size=10), bgcolor="rgba(255,255,255,0.7)", xanchor="right", yanchor="top")
    fig.update_layout(
        template=setdict["template"],
        title_text=setdict["title"],
        hovermode="x unified",
        autosize=False,
        width=setdict["html_aspect_ratio_w_over_h"] * setdict["html_image_height"],
        height=setdict["html_image_height"],
        legend=dict(**legend_style, x=0.45, y=0.98),
        legend2=dict(**legend_style, x=1.0, y=0.98),
        legend3=dict(**legend_style, x=0.45, y=0.40),
        legend4=dict(**legend_style, x=1.0, y=0.40),
    )
    fig.update_xaxes(dtick=1, title_text="Number of components", row=2)
    fig.update_yaxes(title_text="Fraction of Y explained", row=1, col=1)
    fig.update_yaxes(title_text="Correlation", row=1, col=2)
    fig.update_yaxes(title_text="Angle (degrees)", range=[0, 90], row=2, col=1)
    fig.update_yaxes(title_text="Alarm rate", rangemode="tozero", row=2, col=2)
    return fig
