# (c) Kevin Dunn, 2019. MIT License.

import pandas as pd
from bokeh.plotting import figure, show

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


    params = model._OLS.params.copy()
    try:
        params.drop('Intercept', inplace=True)
    except KeyError:
        pass

    params.dropna(inplace=True)
    param_values = params.values
    bar_colours = [negative[1] if p < 0 else positive[1] for p in param_values]
    params = params.abs()
    bar_colours = [bar_colours[i] for i in params.argsort().values]
    params = params.sort_values(na_position='last')


    data = {'Cities': {'Des_Moines': 80.0, 'Lubbock': -300.0, 'Minneapolis': 85.7,
                            'Orange_County': 80.0, 'Salt_Lake_City': 81.8, 'San_Diego': 80.0,
                            'San_Francisco': -400.0, 'Troy': -400.0, 'Wilmington': -300.0}}
    #df_data = pd.DataFrame(data).sort_values('Cities', ascending=False)
    df_data = pd.DataFrame(data)

    this_series = df_data.loc[:,'Cities']
    p = figure(width=800, height=600, y_range=this_series.index.tolist())

    p.grid.grid_line_alpha=1.0
    p.grid.grid_line_color = "white"

    p.xaxis.axis_label = 'xlabel'
    p.xaxis.axis_label_text_font_size = '14pt'
    p.xaxis.major_label_text_font_size = '14pt'
    #p.x_range = Range1d(0,50)
    #p.xaxis[0].ticker=FixedTicker(ticks=[i for i in xrange(0,5,1)])

    p.yaxis.major_label_text_font_size = '14pt'
    p.yaxis.axis_label = 'ylabel'

    p.yaxis.axis_label_text_font_size = '14pt'


#    j = 1
#    for k,v in this_series.iteritems():#
#
#        p.rect(x=v/2, y=j, width=abs(v), height=0.4,color=(76,114,176),
#          width_units="data", height_units="data")
#        j += 1

    show(p)





    #p = figure(width=800, height=600, y_range=params.index.tolist())
    #a = 2
    ##p.grid.grid_line_alpha=1.0
    ##p.grid.grid_line_color = "white"
    ##p.xaxis.axis_label = xlabel
    #show(p)

    #p.xaxis.axis_label_text_font_size = '14pt'
    ##p.xaxis.axis_label_text_font_style = 'normal'
    #p.xaxis.major_label_text_font_size = '14pt'

    #p.yaxis.major_label_text_font_size = '14pt'
    #p.yaxis.axis_label = ylabel
    #p.yaxis.axis_label_text_font_size = '14pt'
    ##p.yaxis.axis_label_text_font_style = 'normal'


    #for idx, key_val in enumerate(params.iteritems()):
        #key, val = key_val
        #print(key, val)
        #p.rect(x = val,
               #y = idx-0.5,
               #width = val,
               #height = 0.4,
               #color = bar_colours[idx],
               #width_units = "data",
               #height_units = "data")





paretoPlot = pareto_plot


def contour_plot():
    pass

contourPlot = contour_plot


def predict_plot():
    """Predictions via slides on a plot."""
    pass


def interaction_plot():
    """
    Interaction plot
    """
    pass