# (c) Kevin Dunn, 2019. MIT License.
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

try:
    from .models import predict
except ImportError:
    from models import predict


def get_plot_title(main, model, prefix=''):
    """
    Constructs a sensible plot title from the ``model``.
    """
    if main is None:
        main = prefix
        if hasattr(model.data, '_pi_title'):
            main += ': ' + getattr(model.data, '_pi_title')

    return main


def get_param_names(model):

    params = model._OLS.params.copy()
    try:
        params.drop('Intercept', inplace=True)
    except KeyError:
        pass

    params.dropna(inplace=True)
    return params

def pareto_plot(model, ylabel="Effect name", xlabel="Magnitude of effect",
                main="Pareto plot", legendtitle="Sign of coefficients",
                negative=("Negative", "grey"), positive=("Positive", "black"),
                show=True):
    """
    Plots the Pareto plot for a given model.

    Parameters
    ----------
    model: required; a model created by the package.
    ylab: string; optional, default: "Effect name"
        The label on the y-axis of the Pareto plot.
    xlab: string; optional, default: "Magnitude of effect"
        The label on the x-axis of the Pareto plot
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
    # Inspired from: https://stackoverflow.com/questions/33737079/how-to-plot-horizontal-bar-chart-in-bokeh-python

    # TODO: show error bars
    #error_bars = model._OLS.conf_int()
    # http://holoviews.org/reference/elements/bokeh/ErrorBars.html


    TRY THIS: https://stackoverflow.com/questions/37173230/how-do-i-use-custom-labels-for-ticks-in-bokeh

        import pandas as pd
        from bokeh.charts import Bar, output_file, show
        from bokeh.models import FuncTickFormatter

        skills_list = ['cheese making', 'squanching', 'leaving harsh criticisms']
        pct_counts = [25, 40, 1]
        df = pd.DataFrame({'skill':skills_list, 'pct jobs with skill':pct_counts})
        p = Bar(df, 'index', values='pct jobs with skill', title="Top skills for ___ jobs", legend=False)
        label_dict = {}
        for i, s in enumerate(skills_list):
            label_dict[i] = s

        p.xaxis.formatter = FuncTickFormatter(code="""
            var labels = %s;
            return labels[tick];
        """ % label_dict)

        output_file("bar.html")
        show(p)


    params = get_param_names(model)

    param_values = params.values
    beta_str = [f"+{i:0.4g}" if i > 0 else f'{i:0.4g}' for i in param_values]
    bar_colours = [negative[1] if p < 0 else positive[1] for p in param_values]

    params = params.abs()
    # Shuffle the collected information in the same way
    beta_str = [beta_str[i] for i in params.argsort().values]
    bar_colours = [bar_colours[i] for i in params.argsort().values]
    params = params.sort_values(na_position='last')

    # -----
    from bokeh.plotting import figure, show, ColumnDataSource
    source = ColumnDataSource(data=dict(
        x=params.values,
        y=np.arange(1, len(params.index) + 1),
        factor_names=params.index.values,
        bar_colours=bar_colours,
        original_magnitude_with_sign=beta_str,

    ))
    TOOLTIPS = [
        ("Factor name", "@factor_names"),
        ("Magnitude and sign", "@original_magnitude_with_sign"),
    ]
    p = figure(plot_width=400, plot_height=400, tooltips=TOOLTIPS,
               title=get_plot_title(main, model, prefix='Pareto plot'))
    p.hbar(y='y', right='x', height=0.5, left=0, fill_color='bar_colours',
           line_color='bar_colours', source=source)

    p.xaxis.axis_label_text_font_size = '14pt'
    p.xaxis.axis_label = xlabel
    p.xaxis.major_label_text_font_size = '14pt'
    p.xaxis.axis_label_text_font_style = 'normal'

    p.yaxis.major_label_text_font_size = '14pt'
    p.yaxis.axis_label = ylabel
    p.yaxis.axis_label_text_font_size = '14pt'
    p.yaxis.axis_label_text_font_style = 'normal'

    p.xaxis.bounds = (0, params.max()*1.05)
    show(p)



paretoPlot = pareto_plot


def contour_plot(model, xlabel=None, ylabel=None, main=None,
        N=25, xlim=(-3.2, 3.2), ylim=(-3.2, 3.2),
        colour_function="terrain", show=True, show_expt_data=True,
        figsize=(10, 10), dpi=100):
    """
    Show a contour plot of the model.

    TODO:
    * more than 2 factors in model
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
    h_grid = np.linspace(xlim[0], xlim[1], num = N)
    v_grid = np.linspace(ylim[0], ylim[1], num = N)
    H, V = np.meshgrid(h_grid, v_grid)
    h_grid, v_grid = H.ravel(), V.ravel()

    params = get_param_names(model)
    if xlabel is None:
        xlabel = params.index[0]
    else:
        xlabel = str(xlabel)

    if ylabel is None:
        ylabel = params.index[1]
    else:
        ylabel = str(ylabel)


    kwargs = {xlabel: h_grid, ylabel: v_grid}
    Z = predict(model, **kwargs)
    Z = Z.values.reshape(N, N)

    # Create a simple contour plot with labels using default colors.  The
    # inline argument to clabel will control whether the labels are draw
    # over the line segments of the contour, removing the lines beneath
    # the label
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor='white',
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

    #from matplotlib.backends.backend_agg import FigureCanvasAgg

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

    return plt







contourPlot = contour_plot


def predict_plot():
    """Predictions via slides on a plot."""
    pass


def interaction_plot():
    """
    Interaction plot
    """
    pass
