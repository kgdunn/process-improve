# (c) Kevin Dunn, 2019. MIT License.
from pathlib import Path
import webbrowser
import numpy as np
import pandas as pd

from bokeh.plotting import figure, ColumnDataSource
from bokeh.plotting import show as show_plot
from bokeh.models import HoverTool

try:
    from .models import predict
except ImportError:
    from models import predict


def get_plot_title(main, model, prefix=''):
    """
    Constructs a sensible plot title from the ``model``.
    """
    if main is not None:
        main = prefix
        title = model.get_title()
        if title:
            main += f': {title}'

    return main


def pareto_plot(model,
                ylabel="Effect name",
                xlabel="Magnitude of effect",

                # show all factors and interactions
                up_to_level=None,
                main="Pareto plot", legendtitle="Sign of coefficients",
                negative=("Negative", "grey"), positive=("Positive", "black"),
                show=True, plot_width=500, plot_height=None):
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

    Returns
    -------
    The plot handle. Can be further manipulated, e.g. for saving.

    Example
    -------

    model = linear()
    pareto_plot(model, main="Pareto plot for my experiment")

    p = pareto_plot(model, main="Pareto plot for my experiment", show=False)
    p.save('save_plot_to_figure.png')
    """
    # TODO: show error bars : see Bokeh annotations: Whiskers model
    # p.add_layout(
    # Whisker(source=e_source, base="base", upper="upper", lower="lower")
    # )
    # error_bars = model._OLS.conf_int()
    # http://holoviews.org/reference/elements/bokeh/ErrorBars.html
    # https://docs.bokeh.org/en/latest/docs/user_guide/annotations.html

    params = model.get_parameters()
    if up_to_level:
        assert isinstance(up_to_level, int), ("Specify an integer value for "
                                              "`up_to_level`.")
        keep = []
        for k in range(up_to_level):
            keep.extend(model.get_factor_names(level=k + 1))

        params = params.filter(keep)

    param_values = params.values
    beta_str = [f"+{i:0.4g}" if i > 0 else f'{i:0.4g}' for i in param_values]
    bar_colours = [negative[1] if p < 0 else positive[1] for p in param_values]
    bar_signs = ['Positive' if i > 0 else 'Negative' for i in param_values]

    params = params.abs()
    base_parameters = model.get_factor_names(level=1)
    full_names = []
    for param_name, param_value in params.iteritems():
        if param_name in base_parameters:
            fname = model.data.pi_source.get(param_name, param_name)
            full_names.append(fname)
        else:
            full_names.append(f'Interaction between {param_name}')

    # Shuffle the collected information in the same way
    beta_str = [beta_str[i] for i in params.argsort().values]
    bar_colours = [bar_colours[i] for i in params.argsort().values]
    bar_signs = [bar_signs[i] for i in params.argsort().values]
    full_names = [full_names[i] for i in params.argsort().values]
    params = params.sort_values(na_position='last')

    source = ColumnDataSource(data=dict(
        x=params.values,
        y=np.arange(1, len(params.index) + 1),
        factor_names=params.index.values,
        bar_colours=bar_colours,
        bar_signs=bar_signs,
        full_names=full_names,
        original_magnitude_with_sign=beta_str,
    ))
    TOOLTIPS = [
        ("Short name", "@factor_names"),
        ("Full name", "@full_names"),
        ("Magnitude and sign", "@original_magnitude_with_sign"),
    ]
    p = figure(plot_width=plot_width,
               plot_height=plot_height or (500 + (len(params) - 8) * 20),
               tooltips=TOOLTIPS,
               title=get_plot_title(main, model, prefix='Pareto plot'))
    p.hbar(y='y', right='x', height=0.5, left=0, fill_color='bar_colours',
           line_color='bar_colours', legend='bar_signs', source=source)

    p.xaxis.axis_label_text_font_size = '14pt'
    p.xaxis.axis_label = xlabel
    p.xaxis.major_label_text_font_size = '14pt'
    p.xaxis.axis_label_text_font_style = 'normal'
    p.xaxis.bounds = (0, params.max() * 1.05)

    p.yaxis.major_label_text_font_size = '14pt'
    p.yaxis.axis_label = ylabel
    p.yaxis.axis_label_text_font_size = '14pt'
    p.yaxis.axis_label_text_font_style = 'normal'

    locations = source.data['y'].tolist()
    labels = source.data['factor_names']
    p.yaxis.ticker = locations
    p.yaxis.major_label_overrides = dict(zip(locations, labels))

    p.legend.orientation = "vertical"
    p.legend.location = "bottom_right"

    if show:
        show_plot(p)
    else:
        return p

paretoPlot = pareto_plot


def contour_plot(model, xlabel=None, ylabel=None, main=None,
        N=25, xlim=(-3.2, 3.2), ylim=(-3.2, 3.2),
        colour_function="terrain", show=True, show_expt_data=True,
        figsize=(10, 10), dpi=100, other_factors=None):
    """
    Show a contour plot of the model.

    TODO:
    * two axes; for the real-world and coded units
    * Hover display of experimental data points
    * add a bit of jitter to the data if the numbers are exactly the same [option]


    NOTE: currently only works for variables with 2 factors. Check back next
          next week for an update.

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
        stop(paste("The variable \"", toString(xlab), "\" was not a variable name in the linear model.\n ",
            "Valid variable names are: ", toString(valid.names),
            sep = ""))
    }
    if (!(ylab %in% valid.names)) {
        stop(paste("The variable \"", toString(ylab), "\" was not a variable name in the linear model.\n ",
            "Valid variable names are: ", toString(valid.names),
            sep = ""))
    }
    """
    if True:

        plt = contour_plot_bokeh(model, xlabel, ylabel, main,
                                 N, xlim, ylim,
                                 colour_function, show,
                                 show_expt_data,
                                 figsize, dpi, other_factors)

    # Matplotlib version
    if False:
        import matplotlib.pyplot as plt
        h_grid = np.linspace(xlim[0], xlim[1], num=N)
        v_grid = np.linspace(ylim[0], ylim[1], num = N)
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

        assert sorted(kwargs.keys()) == sorted(pure_factors), ("Not all factors "
                                                               "were specified.")
        Z = predict(model, **kwargs)
        Z = Z.values.reshape(N, N)

        # Create a simple contour plot with labels using default colors.  The
        # inline argument to clabel will control whether the labels are draw
        # over the line segments of the contour, removing the lines beneath
        # the label
        _ = plt.figure(figsize=figsize, dpi=dpi, facecolor='white',
                         edgecolor='white')
        levels = np.linspace(Z.min(), Z.max(), N)

        # Show the data from the experiment as dots on the plot
        if show_expt_data:
            plt.plot(model.data[xlabel],
                     model.data[ylabel],
                     'dimgrey',
                     linestyle = '',
                     marker = 'o',
                     ms=15,
                     linewidth=2)

        plt.title(get_plot_title(main, model, prefix='Contour plot'))
        plt.xlabel(xlabel, fontsize=12, fontweight="bold")
        plt.ylabel(ylabel, fontsize=12, fontweight="bold")

        # Set up the plot for the first time
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.grid(color='#DDDDDD')

        CS = plt.contour(H, V, Z,
                         colors='black',
                         levels=levels,
                         linestyles='dotted')
        plt.clabel(CS, inline=True, fontsize=10, fmt='%1.0f')

        plt.imshow(Z, extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                   origin='lower',
                   cmap=colour_function,  # 'RdGy',
                   alpha=0.5)
        plt.colorbar()

        if show:
            plt.show()

    #return plt

contourPlot = contour_plot

def predict_plot():
    """Predictions via slides on a plot."""
    pass

def interaction_plot():
    """
    Interaction plot
    """
    pass

#SHOW variable names on pareto plot for main factors
#Can bokeh do nice contour plots?
def contour_plot_bokeh(model, xlabel=None, ylabel=None, main=None,
        N=25, xlim=(-3.2, 3.2), ylim=(-3.2, 3.2),
        colour_function="terrain", show=True, show_expt_data=True,
        figsize=(10, 10), dpi=100, other_factors=None):

    # TODO: show labels of contour plot

    # https://stackoverflow.com/questions/33533047/how-to-make-a-contour-plot-in-python-using-bokeh-or-other-libs
    from skimage import measure
    import numpy as np

    from bokeh.plotting import figure, show
    from bokeh.colors import RGB
    from matplotlib import cm

    from bokeh.models import (ColorBar,
                              BasicTicker,
                              LinearColorMapper,
                              PrintfTickFormatter)

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

    assert sorted(kwargs.keys()) == sorted(pure_factors), ("Not all factors "
                                                           "were specified.")
    Z = predict(model, **kwargs)
    Z = Z.values.reshape(N, N)
    z_min, z_max = Z.min(), Z.max()
    levels = np.linspace(z_min, z_max, N)

    from matplotlib.pyplot import contour, clabel
    CS = contour(H, V, Z, levels=levels, linestyles='dotted')
    clabel(CS, inline=True, fontsize=10, fmt='%1.0f')
    contour_labels = [(float(q._x), float(q._y), float(q._text))\
                                                     for q in CS.labelTexts]


    # Convert the Matplotlib colour mapper to Bokeh
    # https://stackoverflow.com/questions/49931311/using-matplotlibs-colormap-for-bokehs-color-bar
    mapper = getattr(cm, colour_function)
    colours = (255 * mapper(range(256))).astype('int')
    colour_palette = [RGB(*tuple(rgb)).to_hex() for rgb in colours]
    color_mapper = LinearColorMapper(palette=colour_palette,
                                     low=z_min,
                                     high=z_max)

    # Another alternative:
    # https://stackoverflow.com/questions/35315259/using-colormap-with-bokeh-scatter
    #colors = ["#%02x%02x%02x" % (int(r), int(g), int(b)) for r, g, b, _ in 255*mpl.cm.viridis(mpl.colors.Normalize()(radii))]


    p = figure(x_range=xlim,
               y_range=ylim,
                # https://github.com/bokeh/bokeh/issues/2351
               tools="pan,wheel_zoom,box_zoom, box_select,lasso_select,reset,save",
               )
    # Create the image layer
    h_image = p.image(image=[Z],
                      x=xlim[0],
                      y=ylim[0],
                      dw=xlim[1] - xlim[0],
                      dh=ylim[1] - ylim[0],
                      color_mapper=color_mapper,
                      global_alpha=0.5,           # with some transparency
                      name='contour_image',
                      )
    # hover_tool_names.append('contour_image')
    h1 = HoverTool(tooltips=[(xlabel, "$x"),
                             (ylabel, "$y"),
                             ("Predicted response", "@image")],
                   renderers=[h_image])  # custom tooltip for the predicted image

    color_bar = ColorBar(color_mapper=color_mapper,
                         major_label_text_font_size="8pt",
                         ticker=BasicTicker(max_interval=(z_max - z_min) / N * 2),
                         formatter=PrintfTickFormatter(format='%.2f'),
                         label_standoff=6,
                         border_line_color=None,
                         location=(0, 0))

    p.add_layout(color_bar, 'right')


    #Contour lines using Scipy:
    #scaler_y = (ylim[1] - ylim[0]) / (N - 1)
    #scaler_x = (xlim[1] - xlim[0]) / (N - 1)
    #for level in levels:
        #contours = measure.find_contours(Z, level)
        #for contour in contours:
            #x = contour[:, 1] * scaler_y + ylim[0]
            #y = contour[:, 0] * scaler_x + xlim[0]
            #


    for idx, cccontour in enumerate(CS.allsegs):
        if cccontour:
            x = cccontour[0][:,0]
            y = cccontour[0][:,1]
            p.line(x, y, line_dash = 'dashed', color='darkgrey', line_width=1)
            level = levels[idx]


    # TODO: bigger experimental markers
    # TODO: hover for the data point shows the factor settings for the data point

    if show_expt_data:

        source = ColumnDataSource(data=dict(
            x=model.data[xlabel],
            y=model.data[ylabel],
            output=model.data["y"].to_list(),  # <-- this needs to be generalized
            # bar_colours=bar_colours,
            # bar_signs=bar_signs,
            # full_names=full_names,
            # original_magnitude_with_sign=beta_str,
        ))
        h_expts = p.circle(x='x',
                           y='y',
                           color='black',
                           source=source,
                           # linestyle='',
                           # marker='o',
                           size=10,
                           line_width=2,
                           name='experimental_points',)
        h2 = HoverTool(tooltips=[(xlabel, "$x"),
                                 (ylabel, "$y"),
                                 #("Other factors", "NA"),
                                 ("Actual value", "@output")  # why not working???
                                 ],
                       renderers=[h_expts])  # custom tooltip for the predicted image

    # Axis labels:
    p.xaxis.axis_label_text_font_size = '14pt'
    p.xaxis.axis_label = xlabel
    p.xaxis.major_label_text_font_size = '14pt'
    p.xaxis.axis_label_text_font_style = 'bold'
    p.xaxis.bounds = (xlim[0], xlim[1])

    p.yaxis.major_label_text_font_size = '14pt'
    p.yaxis.axis_label = ylabel
    p.yaxis.axis_label_text_font_size = '14pt'
    p.yaxis.axis_label_text_font_style = 'bold'
    p.yaxis.bounds = (ylim[0], ylim[1])


    # Add the hover tooltips:
    p.add_tools(h1)
    p.add_tools(h2)

    if show:
        show_plot(p)





    # Show the data from the experiment as dots on the plot
    # TODO: add some jitter

    #plt.title(get_plot_title(main, model, prefix='Contour plot'))
    #plt.xlabel(xlabel, fontsize=12, fontweight="bold")
    #plt.ylabel(ylabel, fontsize=12, fontweight="bold")

    # Set up the plot for the first time
    # plt.xlim(xlim)
    # plt.ylim(ylim)
    # plt.grid(color='#DDDDDD')

    # CS=plt.contour(H, V, Z,
                     # colors='black',
                     # levels=levels,
                     # linestyles='dotted')
    #plt.clabel(CS, inline=True, fontsize=10, fmt='%1.0f')

    # plt.imshow(Z, extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
               # origin='lower',
               # cmap=colour_function,  # 'RdGy',
               # alpha=0.5)
    # plt.colorbar()



def tradeoff_table(show_in_browser= True,
                   show_pdf=False):
    """
    Shows the trade-off table in a web-browser. The PDF display is not
    supported yet.
    """
    # TOCONSIDER: show the image inline, esp for Jupyter notebooks.
    # https://github.com/bokeh/bokeh/issues/2426
    if show_in_browser:
        path = Path(__file__)
        # Wrapping the image in HTML does not work in Jupyter notebooks.
        fname = "trade-off-table.png"
        fqp = f"file://{path.drive}/{'/'.join(path.parts[1:-1])}/media/{fname}"
        url = 'https://yint.org/tradeoff'
        webbrowser.open_new_tab(url)
