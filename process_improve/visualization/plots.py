import pandas as pd


def raincloud(sequence: pd.Series) -> dict | None:
    """Create a raincloud plot from a series of data. Raincloud plot = violin + boxplot + jitter.

    .. warning::
        This function is **not yet implemented**. The body is a stub and the
        call returns ``None``; it does not yet produce the described
        dictionary.

    Parameters
    ----------
    sequence : pd.Series
        A Pandas data series for which a raincloud plot will be created

    Returns
    -------
    dict | None
        When implemented, will return a dictionary that can be sent directly
        to Plotly for visualization. Currently returns ``None``.
    """
    # TODO: split violin plot: https://plotly.com/python/violin/
    # TODO: rug/dotplot
