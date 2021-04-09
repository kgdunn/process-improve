"""
Getting data into the required format for use with this library.

There are 3 useful ways to represent batch data.

`dict`: as a Python dictionary

    data = {
        "batch 1": data frame with varying number of rows, but same number of columns.
        "batch 2": etc
    }

    The `keys` are unique identifiers for each batch, such as integers, or strings.

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

import numpy as np
import pandas as pd


def check_valid_batch_dict(in_dict: dict, no_nan=False) -> bool:
    """Check if the incoming dictionary of batch data is a valid dictionary of data.

    Checks:
    1. All batches in the dictionary have the same number of columns.
    2. All columns are numeric.
    3. If `no_nan` is True, also checks that there are no NaNs.

    Parameters
    ----------
    in_dict : dict
        A dictionary of batch data.

    no_nan : bool
        If True, will also check that no missing values are present.


    Returns
    -------
    bool
        True, if it passes the checks.
    """
    assert len(in_dict) > 1
    batch1 = in_dict[list(in_dict.keys())[0]]
    base_columns = set(batch1.columns)
    check = True
    for _, batch in in_dict.items():
        # Check 1
        check *= base_columns == set(batch.columns)
        # Check 2
        check *= batch.select_dtypes(include=[np.number]).shape[1] == batch.shape[1]
        # Check 3
        check *= batch.isna().values.sum() == 0

    return check


def dict_to_melted(in_df: pd.DataFrame) -> dict:
    """Reverse of `melted_to_dict`"""
    pass


def dict_to_wide(in_df: dict) -> pd.DataFrame:
    """
    Data must be warped already so that every batch has the same number of *rows*!
    """
    outdf = pd.DataFrame()
    # TODO: add a check on the rows

    # aligned_wide_df = in_df.pivot(index="batch_id", columns="sequence")
    # new_labels = [
    #     "-".join(item)
    #     for item in zip(
    #         aligned_wide_df.columns.get_level_values(0),
    #         [str(val).zfill(max_places) for val in aligned_wide_df.columns.get_level_values(1)],
    #     )
    # ]
    # aligned_wide_df.columns = new_labels

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
        batches[batch_id] = batch

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
