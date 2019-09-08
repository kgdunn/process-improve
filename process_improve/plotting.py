def pareto_plot(model, ylab="Effect name", xlab="Magnitude of effect",
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

    # TODO: which plotting library to use?
    pass



paretoPlot = pareto_plot
