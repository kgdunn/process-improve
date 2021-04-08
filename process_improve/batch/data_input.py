"""
Getting data into the required format for use with this library.

There are 3 useful ways to represent batch data.

`dict`: as a Python dictionary

    data = {
        "batch 1": data frame with varying number of rows, but same number of columns,
        "batch 2": etc
    }


`melt`: as a single Pandas data frame

    data = pd.DataFrame(...)

        * very large number of rows, for all batches stacked vertically on top of each other
        * some number of columns, one column per tag
        * one column, usually called "batch_id", indicates what the batch number is for that row
        * another column, usually called "time", indicates what the time is within that batch
        * typically sorted, but does not have to be

`wide`: as a single Pandas data frame, as for the "melted" version, but pivotted instead

    data = pd.DataFrame(...)

        * each row is a unique batch number
        * the multilevel column index has
            * level 0: the column name
            * level 1: the warped time

        This form only makes sense if the data are warped, so that there is the same number of
        unique elements in the level-1 column index.
"""

import pandas as pd


def dict_to_melted(indf: pd.DataFrame) -> dict:
    """Reverse of `melted_to_dict`"""
    pass


def dict_to_wide(indf: dict) -> pd.DataFrame:
    """
    Data must be warped already.
    """
    outdf = pd.DataFrame()
    return outdf


def melted_to_dict(in_df: pd.DataFrame, batch_id_col) -> dict:
    """
    Loads a "melted" data set, where one of the columns is the `batch_id_col`.
    The data are grouped along the unique values of `batch_id_col`, and each group is stored
    in a dictionary. The dictionary keys are the batch identifier, and the corresponding value
    is a Pandas dataframe of the batch data for that batch.
    """
    batches = {}

    for batch_id, batch in in_df.groupby(batch_id_col):
        batches[str(batch_id)] = batch

    return batches


def melted_to_wide(in_df: pd.DataFrame, batch_id_col) -> dict:
    """
    Data must be warped already.
    """
    pass


def wide_to_melted():
    pass


def wide_to_dict():
    pass


def melt_df_to_series(
    in_df: pd.DataFrame, exclude_columns=["batch_id"], name=None
) -> pd.Series:
    """Returns a Series with a multilevel-index, melted from the DataFrame"""
    out = in_df.drop(exclude_columns, axis=1).T.stack()
    out.name = name
    return out
