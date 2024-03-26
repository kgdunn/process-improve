import pandas as pd


def raincloud(sequence: pd.Series) -> dict:
    """Creates a raincloud plot from a series of data. Raincloud plot = violin + boxplot + jitter.

    Parameters
    ----------
    sequence : pd.Series
        A Pandas data series for which a raincloud plot will be created

    Returns
    -------
    dict
        A dictionary, which can be sent directly to Plotly for visualization.
    """
    # TODO: split violin plot: https://plotly.com/python/violin/
    # TODO: rug/dotplot
    pass
