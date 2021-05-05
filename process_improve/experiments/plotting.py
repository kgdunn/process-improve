# (c) Kevin Dunn, 2019-2021. MIT License.
import webbrowser
import numpy as np

from matplotlib import cm
from bokeh.models import ColorBar, BasicTicker, LinearColorMapper, PrintfTickFormatter
from bokeh.plotting import figure, ColumnDataSource
from bokeh.plotting import show as show_plot
from bokeh.models import HoverTool, Range1d
from bokeh.colors import RGB

from .models import predict


def get_plot_title(main, model, prefix=""):
    """
    Constructs a sensible plot title from the ``model``.
    """
    if main is not None:
        main = prefix
        title = model.get_title()
        if title:
            main += f": {title}"

    return main


def pareto_plot(
    model,
    ylabel="Effect name",
    xlabel="Magnitude of effect",
    # show all factors and interactions
    up_to_level=None,
    main="Pareto plot",
    legendtitle="Sign of coefficients",
    negative=("Negative effects", "#0080FF"),
    positive=("Positive effects", "#FF8000"),
    show=True,
    plot_width=500,
    plot_height=None,
    # In the hover over the bars
    aliasing_up_to_level=2,
):

    # TODO: show full variable names + units on pareto plot for main factors
    #       an interactions

    """
    Plots the Pareto plot for a given model.

    Parameters
    ----------
    model: required; a model created by the package.
    ylabel: string; optional, default: "Effect name"
        The label on the y-axis of the Pareto plot.
    xlabel: string; optional, default: "Magnitude of effect"
        The label on the x-axis of the Pareto plot
    up_to_level: integer, default = None [all levels]
        Up to which level interactions should be displayed
    main: string; optional, default: "Pareto plot"
        The plot title.
    legendtitle: string; optional, default: "Sign of coefficients"
        The legend's title.
    negative: tuple; optional, default: ("Negative", "grey")
        The first entry is the legend text for negative coefficients, and the
        second entry is the colour of the negative coefficient bars.
    positive: tuple; optional, default: ("Positive", "black")
        The first entry is the legend text for positive coefficients, and the
        second entry is the colour of the positive coefficient bars.
    show: boolean; optional, default: True
        Whether or not to show the plot directly.
    aliasing_up_to_level: int; optional, default: 2
        Shows aliasing as hover entries on the bars, up to this level of
        interaction.

    Returns
    -------
    The plot handle. Can be further manipulated, e.g. for saving.

    Example
    -------

    model = linear(...)
    pareto_plot(model, main="Pareto plot for my experiment")

    p = pareto_plot(model, main="Pareto plot for my experiment", show=False)
    p.save('save_plot_to_figure.png')
    """
    # TODO:BokehDeprecationWarning: 'legend' keyword is deprecated, use explicit 'legend_label',
    #                               'legend_field', or 'legend_group' keywords instead
    # TODO: show error bars : see Bokeh annotations: Whiskers model
    # p.add_layout(
    # Whisker(source=e_source, base="base", upper="upper", lower="lower")
    # )
    # error_bars = model._OLS.conf_int()
    # http://holoviews.org/reference/elements/bokeh/ErrorBars.html
    # https://docs.bokeh.org/en/latest/docs/user_guide/annotations.html

    params = model.get_parameters()
    if up_to_level:
        assert isinstance(up_to_level, int), (
            "Specify an integer value for " "`up_to_level`."
        )
        keep = []
        for k in range(up_to_level):
            keep.extend(model.get_factor_names(level=k + 1))

        params = params.filter(keep)

    param_values = params.values
    beta_str = [f"+{i:0.4g}" if i > 0 else f"{i:0.4g}" for i in param_values]
    bar_colours = [negative[1] if p < 0 else positive[1] for p in param_values]
    bar_signs = ["Positive" if i > 0 else "Negative" for i in param_values]

    # Show the absolute parameter values, but we will colour code them
    params = params.abs()
    base_parameters = model.get_factor_names(level=1)
    full_names = []
    for param_name, _ in params.iteritems():
        if param_name in base_parameters:
            fname = model.data.pi_source.get(param_name, param_name)
            full_names.append(fname)
        else:
            full_names.append(f"Interaction between {param_name}")

    # Shuffle the collected information in the same way
    sort_order = params.argsort().values
    beta_str = [beta_str[i] for i in sort_order]
    bar_colours = [bar_colours[i] for i in sort_order]
    bar_signs = [bar_signs[i] for i in sort_order]
    full_names = [full_names[i] for i in sort_order]

    TOOLTIPS = [
        ("Short name", "@factor_names"),
        ("Full name", "@full_names"),
        ("Magnitude and sign", "@original_magnitude_with_sign"),
    ]

    alias_strings = model.get_aliases(
        aliasing_up_to_level, drop_intercept=True, websafe=True
    )

    if len(alias_strings) != 0:
        TOOLTIPS.append(
            ("Aliasing", "@alias_strings{safe}"),
        )
    else:
        alias_strings = [""] * len(params.values)
    alias_strings = [alias_strings[i] for i in sort_order]

    # And only right at the end you can sort the parameter values:
    params = params.sort_values(na_position="last")

    source = ColumnDataSource(
        data=dict(
            x=params.values,
            y=np.arange(1, len(params.index) + 1),
            factor_names=params.index.values,
            bar_colours=bar_colours,
            bar_signs=bar_signs,
            full_names=full_names,
            original_magnitude_with_sign=beta_str,
            alias_strings=alias_strings,
        )
    )

    p = figure(
        plot_width=plot_width,
        plot_height=plot_height or (500 + (len(params) - 8) * 20),
        tooltips=TOOLTIPS,
        title=get_plot_title(main, model, prefix="Pareto plot"),
    )
    p.hbar(
        y="y",
        right="x",
        height=0.5,
        left=0,
        fill_color="bar_colours",
        line_color="bar_colours",
        legend="bar_signs",
        source=source,
    )

    p.xaxis.axis_label_text_font_size = "14pt"
    p.xaxis.axis_label = xlabel
    p.xaxis.major_label_text_font_size = "14pt"
    p.xaxis.axis_label_text_font_style = "normal"
    p.xaxis.bounds = (0, params.max() * 1.05)

    p.yaxis.major_label_text_font_size = "14pt"
    p.yaxis.axis_label = ylabel
    p.yaxis.axis_label_text_font_size = "14pt"
    p.yaxis.axis_label_text_font_style = "normal"

    locations = source.data["y"].tolist()
    labels = source.data["factor_names"]
    p.yaxis.ticker = locations
    p.yaxis.major_label_overrides = dict(zip(locations, labels))

    p.legend.orientation = "vertical"
    p.legend.location = "bottom_right"

    if show:
        show_plot(p)
    else:
        return p


paretoPlot = pareto_plot


def contour_plot(
    model,
    xlabel=None,
    ylabel=None,
    main=None,
    xlim=(-3.2, 3.2),
    ylim=(-3.2, 3.2),
    colour_function="terrain",
    show=True,
    show_expt_data=True,
    figsize=(10, 10),
    dpi=100,
    other_factors=None,
):
    """
    Show a contour plot of the model.

    TODO:
    * two axes; for the real-world and coded units
    * Hover display of experimental data points
    * add a bit of jitter to the data if the numbers are exactly the same [option]


    NOTE: currently only works for variables with 2 factors. Check back for an update.

    """
    """
    valid.names <- colnames(model.frame(lsmodel))[dim(model.frame(lsmodel))[2]:2]
    if (!is.character(xlab)) {
        stop("The \"xlab\" input must be a character (string) name of a variable in the model.")
    }
    if (!is.character(ylab)) {
        stop("The \"ylab\" input must be a character (string) name of a variable in the model.")
    }
    if (!(xlab %in% valid.names)) {
        stop(paste("The variable \"", toString(xlab), "\" was not a variable name in the ",
            "linear model.\n Valid variable names are: ", toString(valid.names),
            sep = ""))
    }
    if (!(ylab %in% valid.names)) {
        stop(paste("The variable \"", toString(ylab), "\" was not a variable name in the ",
            "linear model.\n Valid variable names are: ", toString(valid.names),
            sep = ""))
    }
    """
    plt = contour_plot_bokeh(
        model,
        xlabel,
        ylabel,
        main,
        xlim,
        ylim,
        colour_function,
        show,
        show_expt_data,
        figsize,
        dpi,
        other_factors,
    )
    return plt

    # Matplotlib version
    if False:
        import matplotlib.pyplot as plt

        N = 25  # was a function input
        h_grid = np.linspace(xlim[0], xlim[1], num=N)
        v_grid = np.linspace(ylim[0], ylim[1], num=N)
        H, V = np.meshgrid(h_grid, v_grid)
        h_grid, v_grid = H.ravel(), V.ravel()

        pure_factors = model.get_factor_names(level=1)
        if xlabel is None:
            xlabel = pure_factors[0]
        else:
            xlabel = str(xlabel)

        if ylabel is None:
            ylabel = pure_factors[1]
        else:
            ylabel = str(ylabel)

        kwargs = {xlabel: h_grid, ylabel: v_grid}
        if other_factors is not None and isinstance(other_factors, dict):
            kwargs = kwargs.update(other_factors)

        # Look at which factors are included, and pop them out. The remaining
        # factors are specified at their zero level

        unspecified_factors = [i for i in pure_factors if i not in kwargs.keys()]
        for factor in unspecified_factors:
            kwargs[factor] = np.zeros_like(h_grid)

        assert sorted(kwargs.keys()) == sorted(
            pure_factors
        ), "Not all factors were specified."
        Z = predict(model, **kwargs)
        Z = Z.values.reshape(N, N)

        # Create a simple contour plot with labels using default colors.  The
        # inline argument to clabel will control whether the labels are draw
        # over the line segments of the contour, removing the lines beneath
        # the label
        _ = plt.figure(figsize=figsize, dpi=dpi, facecolor="white", edgecolor="white")
        levels = np.linspace(Z.min(), Z.max(), N)

        # Show the data from the experiment as dots on the plot
        if show_expt_data:
            plt.plot(
                model.data[xlabel],
                model.data[ylabel],
                "dimgrey",
                linestyle="",
                marker="o",
                ms=15,
                linewidth=2,
            )

        plt.title(get_plot_title(main, model, prefix="Contour plot"))
        plt.xlabel(xlabel, fontsize=12, fontweight="bold")
        plt.ylabel(ylabel, fontsize=12, fontweight="bold")

        # Set up the plot for the first time
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.grid(color="#DDDDDD")

        CS = plt.contour(H, V, Z, colors="black", levels=levels, linestyles="dotted")
        plt.clabel(CS, inline=True, fontsize=10, fmt="%1.0f")

        plt.imshow(
            Z,
            extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
            origin="lower",
            cmap=colour_function,  # 'RdGy',
            alpha=0.5,
        )
        plt.colorbar()

        if show:
            plt.show()


contourPlot = contour_plot


def interaction_plot():
    """
    Interaction plot
    """
    pass


def contour_plot_bokeh(
    model,
    xlabel=None,
    ylabel=None,
    main=None,
    xlim=(-3.2, 3.2),
    ylim=(-3.2, 3.2),
    colour_function="terrain",
    show=True,
    show_expt_data=True,
    figsize=(10, 10),
    dpi=50,
    other_factors=None,
):

    # TODO: show labels of contour plot

    # https://stackoverflow.com/questions/33533047/how-to-make-a-contour-plot-in-python-using-bokeh-or-other-libs

    dpi_max = dpi ** 3.5
    N = min(dpi, np.power(dpi_max, 0.5))

    h_grid = np.linspace(xlim[0], xlim[1], num=N)
    v_grid = np.linspace(ylim[0], ylim[1], num=N)
    H, V = np.meshgrid(h_grid, v_grid)
    h_grid, v_grid = H.ravel(), V.ravel()

    pure_factors = model.get_factor_names(level=1)
    if xlabel is None:
        xlabel = pure_factors[0]
    else:
        xlabel = str(xlabel)

    if ylabel is None:
        ylabel = pure_factors[1]
    else:
        ylabel = str(ylabel)

    kwargs = {xlabel: h_grid, ylabel: v_grid}
    if other_factors is not None and isinstance(other_factors, dict):
        kwargs = kwargs.update(other_factors)

    # Look at which factors are included, and pop them out. The remaining
    # factors are specified at their zero level

    unspecified_factors = [i for i in pure_factors if i not in kwargs.keys()]
    for factor in unspecified_factors:
        kwargs[factor] = np.zeros_like(h_grid)

    assert sorted(kwargs.keys()) == sorted(pure_factors), (
        "Not all factors " "were specified."
    )
    Z = predict(model, **kwargs)
    Z = Z.values.reshape(N, N)
    z_min, z_max = Z.min(), Z.max()
    levels = np.linspace(z_min, z_max, N)

    from matplotlib.pyplot import contour, clabel
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use("Agg")
    # Turn interactive plotting off
    plt.ioff()
    CS = contour(H, V, Z, levels=levels, linestyles="dotted")
    clabel(CS, inline=True, fontsize=10, fmt="%1.0f")
    # contour_labels = [(float(q._x), float(q._y), float(q._text))\
    #                                                 for q in CS.labelTexts]

    # Convert the Matplotlib colour mapper to Bokeh
    # https://stackoverflow.com/questions/49931311/using-matplotlibs-colormap-for-bokehs-color-bar
    mapper = getattr(cm, colour_function)
    colours = (255 * mapper(range(256))).astype("int")
    colour_palette = [RGB(*tuple(rgb)).to_hex() for rgb in colours]
    color_mapper = LinearColorMapper(palette=colour_palette, low=z_min, high=z_max)

    # Another alternative:
    # https://stackoverflow.com/questions/35315259/using-colormap-with-bokeh-scatter
    # colors = ["#%02x%02x%02x" % (int(r), int(g), int(b)) for \
    #        r, g, b, _ in 255*mpl.cm.viridis(mpl.colors.Normalize()(radii))]

    p = figure(
        x_range=xlim,
        y_range=ylim,
        # https://github.com/bokeh/bokeh/issues/2351
        tools="pan,wheel_zoom,box_zoom,box_select,lasso_select,reset,save",
    )
    # Create the image layer
    source = {"Xax": [h_grid], "Yax": [v_grid], "predictions": [Z]}
    h_image = p.image(
        source=source,
        image="predictions",
        x=xlim[0],
        y=ylim[0],
        dw=xlim[1] - xlim[0],
        dh=ylim[1] - ylim[0],
        color_mapper=color_mapper,
        global_alpha=0.5,  # with some transparency
        name="contour_image",
    )
    h1 = HoverTool(
        tooltips=[
            (xlabel, "@{Xax}{0.4g}"),
            (ylabel, "@{Yax}{0.4f}"),
            ("Predicted", "@{predictions}{0.4g}"),
        ],
        renderers=[h_image],
        formatters={"Predicted": "printf", xlabel: "printf", ylabel: "printf"},
    )

    color_bar = ColorBar(
        color_mapper=color_mapper,
        major_label_text_font_size="8pt",
        ticker=BasicTicker(max_interval=(z_max - z_min) / N * 2),
        formatter=PrintfTickFormatter(format="%.2f"),
        label_standoff=6,
        border_line_color=None,
        location=(0, 0),
    )

    p.add_layout(color_bar, "right")

    # Contour lines using Scipy:
    # scaler_y = (ylim[1] - ylim[0]) / (N - 1)
    # scaler_x = (xlim[1] - xlim[0]) / (N - 1)
    # for level in levels:
    # contours = measure.find_contours(Z, level)
    # for contour in contours:
    # x = contour[:, 1] * scaler_y + ylim[0]
    # y = contour[:, 0] * scaler_x + xlim[0]

    for _, cccontour in enumerate(CS.allsegs):
        if cccontour:
            x = cccontour[0][:, 0]
            y = cccontour[0][:, 1]
            p.line(x, y, line_dash="dashed", color="darkgrey", line_width=1)

    # TODO: bigger experimental markers
    # TODO: hover for the data point shows the factor settings for the data point

    if show_expt_data:

        source = ColumnDataSource(
            data=dict(
                x=model.data[xlabel],
                y=model.data[ylabel],
                output=model.data[model.get_response_name()].to_list(),
            )
        )
        h_expts = p.circle(
            x="x",
            y="y",
            color="black",
            source=source,
            # linestyle='',
            # marker='o',
            size=10,
            line_width=2,
            name="experimental_points",
        )
        # custom tooltip for the experimental points
        h2 = HoverTool(
            tooltips=[
                (xlabel, "$x{0.4g}"),
                (ylabel, "$y{0.4g}"),
                ("Actual value", "@{output}{0.4g}"),  # why not working???
            ],
            renderers=[h_expts],
            formatters={"Actual value": "printf", xlabel: "printf", ylabel: "printf"},
        )
        h2.point_policy = "snap_to_data"
        h2.line_policy = "none"

    # Axis labels:
    p.xaxis.axis_label_text_font_size = "14pt"
    p.xaxis.axis_label = xlabel
    p.xaxis.major_label_text_font_size = "14pt"
    p.xaxis.axis_label_text_font_style = "bold"
    p.xaxis.bounds = (xlim[0], xlim[1])

    p.yaxis.major_label_text_font_size = "14pt"
    p.yaxis.axis_label = ylabel
    p.yaxis.axis_label_text_font_size = "14pt"
    p.yaxis.axis_label_text_font_style = "bold"
    p.yaxis.bounds = (ylim[0], ylim[1])

    # Add the hover tooltips:
    p.add_tools(h1)
    p.add_tools(h2)

    if show:
        show_plot(p)
    return p

    # Show the data from the experiment as dots on the plot
    # TODO: add some jitter

    # plt.title(get_plot_title(main, model, prefix='Contour plot'))
    # plt.xlabel(xlabel, fontsize=12, fontweight="bold")
    # plt.ylabel(ylabel, fontsize=12, fontweight="bold")

    # Set up the plot for the first time
    # plt.xlim(xlim)
    # plt.ylim(ylim)
    # plt.grid(color='#DDDDDD')

    # CS=plt.contour(H, V, Z,
    # colors='black',
    # levels=levels,
    # linestyles='dotted')
    # plt.clabel(CS, inline=True, fontsize=10, fmt='%1.0f')

    # plt.imshow(Z, extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
    # origin='lower',
    # cmap=colour_function,  # 'RdGy',
    # alpha=0.5)
    # plt.colorbar()


def tradeoff_table(show_in_browser=True, show_pdf=False):
    """
    Shows the trade-off table in a web-browser. The PDF display is not
    supported yet.
    """
    # TOCONSIDER: show the image inline, esp for Jupyter notebooks.
    # https://github.com/bokeh/bokeh/issues/2426
    if show_in_browser:
        # Wrapping the image in HTML does not work in Jupyter notebooks.
        # fname = "trade-off-table.png"
        # fqp = f"file://{path.drive}/{'/'.join(path.parts[1:-1])}/media/{fname}"
        url = "https://yint.org/tradeoff"
        webbrowser.open_new_tab(url)


def plot_model(
    model,
    x_column,
    y_column=None,
    fig=None,
    x_slider=None,
    y_slider=None,
    show_expt_data=True,
    figsize=(10, 10),
    dpi=100,
    **kwargs,
):
    """
    Plots a `model` object with a given model input as `x_column` against
    the model's output: `y_column`. If `y_column` is not specified, then it
    found from the `model`.

    For model's with more than 1 inputs, the `y_column` is the variable to be
    plotted on the y-axis, and then the plot type is a contour plot.
    """
    pure_factors = model.get_factor_names(level=1)
    dpi_max = dpi ** 3.5  # should be reasonable for most modern computers
    per_axis_points = min(dpi, np.power(dpi_max, 1 / len(pure_factors)))

    # `oneD=True`: the x-variable is a model input, and the y-axis is a response
    oneD = False
    if y_column and y_column not in pure_factors:
        oneD = True
        y_column = model.get_response_name()

    param_names = [
        model.get_response_name(),
    ]
    param_names.extend(model.get_factor_names())

    # Not always a great test: y = I(1/d) does not pick up that "d" is the model
    # even though it is via the term encapsulated in I(...)
    # assert x_column in param_names, "x_column must exist in the model."
    assert y_column in param_names, "y_column must exist in the model."

    xrange = model.data[x_column].min(), model.data[x_column].max()
    xdelta = xrange[1] - xrange[0]
    xlim = kwargs.get("xlim", (xrange[0] - xdelta * 0.05, xrange[1] + xdelta * 0.05))
    h_grid = np.linspace(xlim[0], xlim[1], num=per_axis_points)
    plotdata = {x_column: h_grid}

    if not oneD:
        yrange = model.data[y_column].min(), model.data[y_column].max()
        ydelta = yrange[1] - yrange[0]
        ylim = kwargs.get(
            "ylim", (yrange[0] - ydelta * 0.05, yrange[1] + ydelta * 0.05)
        )

        v_grid = np.linspace(ylim[0], ylim[1], num=per_axis_points)
        H, V = np.meshgrid(h_grid, v_grid)
        h_grid, v_grid = H.ravel(), V.ravel()
        plotdata[x_column] = h_grid
        plotdata[y_column] = v_grid

    # TODO: handle the 2D case later

    # if other_factors is not None and isinstance(other_factors, dict):
    # plotdata = kwargs.update(other_factors)

    ## Look at which factors are included, and pop them out. The remaining
    ## factors are specified at their zero level

    # unspecified_factors = [i for i in pure_factors if i not in kwargs.keys()]
    # for factor in unspecified_factors:
    # plotdata[factor] = np.zeros_like(h_grid)

    # assert sorted(kwargs.keys()) == sorted(pure_factors), ("Not all factors "
    # "were specified.")

    Z = predict(model, **plotdata)

    if not oneD:
        assert False
    else:
        plotdata[y_column] = Z
        yrange = Z.min(), Z.max()
        ydelta = yrange[1] - yrange[0]
        ylim = kwargs.get(
            "ylim", (yrange[0] - ydelta * 0.05, yrange[1] + ydelta * 0.05)
        )

    if fig:
        p = fig
        prior_figure = True
    else:
        prior_figure = False
        p = figure(
            x_range=xlim,
            y_range=ylim,
            # https://github.com/bokeh/bokeh/issues/2351
            tools="pan,wheel_zoom,box_zoom, box_select,lasso_select,reset,save",
        )

    h_line = p.line(
        plotdata[x_column],
        plotdata[y_column],
        line_dash="solid",
        color=kwargs.get("color", "black"),
        line_width=kwargs.get("line_width", 2),
    )
    y_units = model.data.pi_units[y_column]

    tooltips = [(x_column, "$x")]
    if y_units:
        tooltips.append((f"Prediction of {y_column} [{y_units}]", "$y"))
    else:
        tooltips.append((f"Prediction of {y_column}", "$y"))

    tooltips.append(("Source", model.name or ""))

    # custom tooltip for the predicted prediction line
    h1 = HoverTool(tooltips=tooltips, renderers=[h_line])
    h1.line_policy = "nearest"

    if show_expt_data:
        source = ColumnDataSource(
            data=dict(
                x=model.data[x_column],
                y=model.data[y_column],
                output=model.data[model.get_response_name()].to_list(),
            )
        )
        h_expts = p.circle(
            x="x",
            y="y",
            color="black",
            source=source,
            size=10,
            line_width=2,
            name="Experimental_points",
        )
        h2 = HoverTool(
            tooltips=[
                (x_column, "$x"),
                (y_column, "$y"),
                ("Experimental value", "@output"),
            ],
            renderers=[h_expts],
        )

        h2.point_policy = "snap_to_data"
        h2.line_policy = "none"

    # Axis labels:
    p.xaxis.axis_label_text_font_size = "14pt"
    p.xaxis.axis_label = x_column
    p.xaxis.major_label_text_font_size = "14pt"
    p.xaxis.axis_label_text_font_style = "bold"

    p.yaxis.major_label_text_font_size = "14pt"
    p.yaxis.axis_label = y_column
    p.yaxis.axis_label_text_font_size = "14pt"
    p.yaxis.axis_label_text_font_style = "bold"
    if prior_figure:

        # p.xaxis.bounds =
        p.x_range = Range1d(
            min(xlim[0], p.x_range.start, min(model.data[x_column])),
            max(xlim[1], p.x_range.end, max(model.data[x_column])),
        )
        p.y_range = Range1d(
            min(ylim[0], p.y_range.start, min(model.data[y_column])),
            max(ylim[1], p.y_range.end, max(model.data[y_column])),
        )

    else:
        p.x_range = Range1d(xlim[0], xlim[1])
        p.y_range = Range1d(ylim[0], ylim[1])

    # Add the hover tooltips:
    p.add_tools(h1)
    p.add_tools(h2)

    show_plot(p)
    return p
