# (c) Kevin Dunn, 2010-2025. MIT License. Based on own private work over the years.

# Built-in libraries
from __future__ import annotations

import json

import plotly.graph_objects as go
from pydantic import BaseModel, field_validator
from sklearn.base import BaseEstimator


def plot_pre_checks(model: BaseEstimator, pc_horiz: int, pc_vert: int, pc_depth: int) -> bool:
    """Check the inputs for the plot functions are valid."""
    n_components = model.A if hasattr(model, "A") else model._parent.n_components
    assert (
        0 < pc_horiz <= n_components
    ), f"The model has {n_components} components. Ensure that 1 <= pc_horiz<={n_components}."
    assert (
        0 < pc_vert <= n_components
    ), f"The model has {n_components} components. Ensure that 1 <= pc_vert<={n_components}."
    assert (
        -1 <= pc_depth <= n_components
    ), f"The model has {n_components} components. Ensure that 1 <= pc_depth<={n_components}."
    assert len({pc_horiz, pc_vert, pc_depth}) == 3, "Specify distinct components for each axis"

    return True


def score_plot(  # noqa: C901, PLR0913
    model: BaseEstimator,
    pc_horiz: int = 1,
    pc_vert: int = 2,
    pc_depth: int = -1,
    items_to_highlight: dict[str, list] | None = None,
    settings: dict | None = None,
    fig: go.Figure | None = None,
) -> go.Figure:
    """Generate a 2-dimensional score plot for the given latent variable model.

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
        keys:   an string which can be json.loads(...) and turns into a Plotly line specifier.
        values: a list of identifiers for the items to highlight [index names]
        For example:
            items_to_highlight = {'{"color": "red", "symbol": "cross"}': items_in_red}

            will ensure the subset of the index listed in `items_in_red` in that colour and shape.

    settings : dict
        Default settings are = {
            "show_ellipse": True [bool],
                Should the Hotelling's T2 ellipse be added

            "ellipse_conf_level": 0.95 [float]
                If the ellipse is added, which confidence level is used. A number < 1.00.

            "title": f"Score plot of ... "
                Overall plot title

            "show_labels": False,
                Adds a label for each observation. Labels are always available in the hover.

            "show_legend": True,
                Shows a clickable legend (allows to turn the ellipse(s) on/off)

            "html_image_height": 500,
                in pixels

            "html_aspect_ratio_w_over_h": 16/9,
                sets the image width, as a ratio of the height

        }
    """
    plot_pre_checks(model, pc_horiz, pc_vert, pc_depth)
    margin_dict: dict = dict(l=10, r=10, b=5, t=80)  # Defaults: l=80, r=80, t=100, b=80
    data_to_plot = model.x_scores if hasattr(model, "x_scores") else model._parent.t_scores_super
    ellipse_coordinates = (
        model.ellipse_coordinates if hasattr(model, "ellipse_coordinates") else model._parent.ellipse_coordinates
    )

    class Settings(BaseModel):
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
        show_labels: bool = False  # TODO
        show_legend: bool = True
        html_image_height: float = 500.0
        html_aspect_ratio_w_over_h: float = 16 / 9.0

    setdict = Settings(**settings).model_dump() if settings else Settings().model_dump()
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
                marker=dict(
                    color="darkblue",
                    symbol="circle",
                ),
                text=list(default_index),
                textposition="top center",
            )
        )
        # Items to highlight, if any
        for key, index in highlights.items():
            styling = json.loads(key)
            fig.add_trace(
                go.Scatter3d(
                    x=data_to_plot.loc[index, pc_horiz],
                    y=data_to_plot.loc[index, pc_vert],
                    z=data_to_plot.loc[index, pc_depth],
                    name=name,
                    mode="markers+text" if setdict["show_labels"] else "markers",
                    marker=styling,
                    text=list(index),
                    textposition="top center",
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
                marker=dict(
                    color="darkblue",
                    symbol="circle",
                    size=7,
                ),
                text=default_index,
                textposition="top center",
            )
        )
        # Items to highlight, if any
        for key, index in highlights.items():
            styling = json.loads(key)
            fig.add_trace(
                go.Scatter(
                    x=data_to_plot.loc[index, pc_horiz],
                    y=data_to_plot.loc[index, pc_vert],
                    name=name,
                    mode="markers+text" if setdict["show_labels"] else "markers",
                    marker=styling,
                    text=list(index),
                    textposition="top center",
                )
            )
        if setdict["show_ellipse"]:
            ellipse = ellipse_coordinates(
                score_horiz=pc_horiz,
                score_vert=pc_vert,
                conf_level=setdict["ellipse_conf_level"],
            )
            fig.add_hline(y=0, line_color="black")
            fig.add_vline(x=0, line_color="black")
            fig.add_trace(
                go.Scatter(
                    x=ellipse[0],
                    y=ellipse[1],
                    name=f"Hotelling's T^2 [{setdict['ellipse_conf_level'] * 100:.4g}%]",
                    mode="lines",
                    line=dict(
                        color="red",
                        width=2,
                    ),
                )
            )

    fig.update_layout(
        title_text=setdict["title"],
        margin=margin_dict,
        hovermode="closest",
        showlegend=setdict["show_legend"],
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
            mirror=True,
            showspikes=True,
            visible=True,
        ),
        yaxis=dict(
            gridwidth=2,
            type="linear",
            autorange=True,
            showspikes=True,
            visible=True,
            showline=True,
            side="left",
        ),
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
        Default settings are = {
            "title": f"Loadings plot of component {pc_horiz} vs component {pc_vert}"
                Overall plot title

            "show_labels": True,
                Adds a label for each column. Labels are always available in the hover.

            "html_image_height": 500,
                in pixels

            "html_aspect_ratio_w_over_h": 16/9,
                sets the image width, as a ratio of the height

        }
    """
    plot_pre_checks(model, pc_horiz, pc_vert, pc_depth=0)
    margin_dict: dict = dict(l=10, r=10, b=5, t=80)  # Defaults: l=80, r=80, t=100, b=80

    class Settings(BaseModel):
        title: str = f"Loadings plot [{loadings_type.upper()}] of component {pc_horiz} vs component {pc_vert}"
        show_labels: bool = True
        html_image_height: float = 500.0
        html_aspect_ratio_w_over_h: float = 16 / 9.0

    setdict = Settings(**settings).model_dump() if settings else Settings().model_dump()
    if fig is None:
        fig = go.Figure()

    what = model.loadings  # PCA default
    if hasattr(model, "direct_weights"):
        what = model.direct_weights  # PLS default
    extra = None
    if loadings_type.lower() == "p":
        what = model.loadings
    if loadings_type.lower() == "w":
        what = model.x_weights
    elif loadings_type.lower() == "w*":
        what = model.direct_weights
    elif loadings_type.lower() == "w*c":
        loadings_type = loadings_type[0:-1]
        what = model.direct_weights
        extra = model.y_loadings
    elif loadings_type.lower() == "c":
        what = model.y_loadings

    fig.add_trace(
        go.Scatter(
            x=what.loc[:, pc_horiz],
            y=what.loc[:, pc_vert],
            name="X-space loadings W*",
            mode="markers+text" if setdict["show_labels"] else "markers",
            marker=dict(
                color="darkblue",
                symbol="circle",
                size=7,
            ),
            text=what.index,
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
                    color="purple",
                    symbol="star",
                    size=8,
                ),
                text=extra.index,
                textposition="bottom center",
            )
        )

    fig.update_layout(xaxis_title_text=f"PC {pc_horiz}", yaxis_title_text=f"PC {pc_vert}")
    fig.add_hline(y=0, line_color="black")
    fig.add_vline(x=0, line_color="black")
    fig.update_layout(
        title_text=setdict["title"],
        margin=margin_dict,
        hovermode="closest",
        showlegend=add_legend,
        autosize=False,
        xaxis=dict(
            gridwidth=1,
            mirror=True,
            showspikes=True,
            visible=True,
        ),
        yaxis=dict(
            gridwidth=2,
            type="linear",
            autorange=True,
            showspikes=True,
            visible=True,
            showline=True,
            side="left",
        ),
        width=setdict["html_aspect_ratio_w_over_h"] * setdict["html_image_height"],
        height=setdict["html_image_height"],
    )
    return fig


def spe_plot(
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
        keys:   an string which can be json.loads(...) and turns into a Plotly line specifier.
        values: a list of identifiers for the items to highlight [index names]
        For example:
            items_to_highlight = {'{"color": "red", "symbol": "cross"}': items_in_red}

            will ensure the subset of the index listed in `items_in_red` in that colour and shape.

    settings : dict
        Default settings are = {
            "show_limit": True [bool],
                Should the SPE limit be plotted.

            "conf_level": 0.95 [float]
                If the limit line is added, which confidence level is used. Number < 1.00.

            "title": f"Squared prediction error plot after fitting {with_a} components,
                       with the {conf_level*100}% confidence limit"
                Overall plot title

            "default_marker": optional, [dict]
                dict(color="darkblue", symbol="circle", size=7)

            "show_labels": False,
                Adds a label for each observation. Labels are always available in the hover.

            "show_legend": False,
                Shows a clickable legend (allows to turn the limit on/off)

            "html_image_height": 500,
                Image height, in pixels.

            "html_aspect_ratio_w_over_h": 16/9,
                Sets the image width, as a ratio of the height.

        }
    """
    # TO CONSIDER: allow a setting `as_line`: which connects the points with line segments
    margin_dict: dict = dict(l=10, r=10, b=5, t=80)  # Defaults: l=80, r=80, t=100, b=80

    if with_a < 0:
        # Get the actual name of the last column in the model if negative indexing is used
        with_a = model.squared_prediction_error.columns[with_a]
    elif with_a == 0:
        raise AssertionError("`with_a` must be >= 1, or specified with negative indexing")

    assert with_a <= model.A, "`with_a` must be <= the number of components fitted"

    class Settings(BaseModel):
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
        default_marker: dict = dict(color="darkblue", symbol="circle", size=7)
        show_labels: bool = False
        show_legend: bool = False
        html_image_height: float = 500.0
        html_aspect_ratio_w_over_h: float = 16 / 9.0

    setdict = Settings(**settings).model_dump() if settings else Settings().dict()
    if fig is None:
        fig = go.Figure()

    name = f"SPE values after {with_a} component{'s' if with_a > 1 else ''}"
    highlights: dict[str, list] = {}
    default_index = model.squared_prediction_error.index
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
            y=model.squared_prediction_error.loc[default_index, with_a],
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
        styling = json.loads(key)
        fig.add_trace(
            go.Scatter(
                x=index,
                y=model.squared_prediction_error.loc[index, with_a],
                name=name,
                mode="markers+text" if setdict["show_labels"] else "markers",
                marker=styling,
                text=index,
                textposition="top center",
            )
        )

    limit_SPE_conf_level = model.spe_limit(conf_level=setdict["conf_level"])
    name = f"{setdict['conf_level'] * 100:.3g}% limit"
    fig.add_hline(
        y=limit_SPE_conf_level,
        line_color="red",
        annotation_text=name,
        annotation_position="bottom right",
        name=name,
    )
    fig.add_hline(y=0, line_color="black")
    fig.update_layout(
        title_text=setdict["title"],
        margin=margin_dict,
        hovermode="closest",
        showlegend=setdict["show_legend"],
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
            mirror=True,
            showspikes=True,
            visible=True,
        ),
        yaxis=dict(
            title=name,
            gridwidth=2,
            type="linear",
            autorange=True,
            showspikes=True,
            visible=True,
            showline=True,  # show a separating line
            side="left",  # show on the RHS
        ),
        width=setdict["html_aspect_ratio_w_over_h"] * setdict["html_image_height"],
        height=setdict["html_image_height"],
    )
    return fig


def t2_plot(
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
        Uses this many number of latent variables, and therefore shows the SPE after this number of
        model components. By default the total number of components fitted will be used.
    items_to_highlight : dict, optional
        keys:   an string which can be json.loads(...) and turns into a Plotly line specifier.
        values: a list of identifiers for the items to highlight [index names]
        For example:
            items_to_highlight = {'{"color": "red", "symbol": "cross"}': items_in_red}

            will ensure the subset of the index listed in `items_in_red` in that colour and shape.

    settings : dict
        Default settings are = {
            "show_limit": True [bool],
                Should the T2 limit be plotted.

            "conf_level": 0.95 [float]
                If the limit line is added, which confidence level is used. Number < 1.00.

            "title": f"Hotelling's T2 plot after fitting {with_a} components,
                       with the {conf_level*100}% confidence limit""
                Overall plot title

            "default_marker": optional, [dict]
                dict(color="darkblue", symbol="circle", size=7)

            "show_labels": False,
                Adds a label for each observation. Labels are always available in the hover.

            "show_legend": False,
                Shows a clickable legend (allows to turn the limit on/off)

            "html_image_height": 500,
                Image height, in pixels.

            "html_aspect_ratio_w_over_h": 16/9,
                Sets the image width, as a ratio of the height.
        }
    """
    # TO CONSIDER: allow a setting `as_line`: which connects the points with line segments
    margin_dict: dict = dict(l=10, r=10, b=5, t=80)  # Defaults: l=80, r=80, t=100, b=80

    if with_a < 0:
        with_a = model.hotellings_t2.columns[with_a]

    # TODO: check `with_a`: what should it plot if `with_a` is zero, or > A?

    class Settings(BaseModel):
        show_limit: bool = True
        conf_level: float = 0.95  # TODO: check constraint < 1
        title: str = (
            f"Hotelling's T2 plot after fitting {with_a} component{'s' if with_a > 1 else ''}"
            f", with the {conf_level * 100}% confidence limit"
        )
        default_marker: dict = dict(color="darkblue", symbol="circle", size=7)
        show_labels: bool = False  # TODO
        show_legend: bool = False
        html_image_height: float = 500.0
        html_aspect_ratio_w_over_h: float = 16 / 9.0

    setdict = Settings(**settings).dict() if settings else Settings().dict()
    if fig is None:
        fig = go.Figure()

    name = f"T2 values after {with_a} component{'s' if with_a > 1 else ''}"
    highlights: dict[str, list] = {}
    default_index = model.hotellings_t2.index
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
            y=model.hotellings_t2.loc[default_index, with_a],
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
        styling = json.loads(key)
        fig.add_trace(
            go.Scatter(
                x=index,
                y=model.hotellings_t2.loc[index, with_a],
                name=name,
                mode="markers+text" if setdict["show_labels"] else "markers",
                marker=styling,
                text=index,
                textposition="top center",
            )
        )

    limit_HT2_conf_level = model.hotellings_t2_limit(conf_level=setdict["conf_level"])
    name = f"{setdict['conf_level'] * 100:.3g}% limit"
    fig.add_hline(
        y=limit_HT2_conf_level,
        line_color="red",
        annotation_text=name,
        annotation_position="bottom right",
        name=name,
    )
    fig.add_hline(y=0, line_color="black")
    fig.update_layout(
        title_text=setdict["title"],
        margin=margin_dict,
        hovermode="closest",
        showlegend=setdict["show_legend"],
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
            mirror=True,
            showspikes=True,
            visible=True,
        ),
        yaxis=dict(
            title_text=name,
            gridwidth=2,
            type="linear",
            autorange=True,
            showspikes=True,
            visible=True,
            showline=True,
            side="left",
        ),
        width=setdict["html_aspect_ratio_w_over_h"] * setdict["html_image_height"],
        height=setdict["html_image_height"],
    )
    return fig
