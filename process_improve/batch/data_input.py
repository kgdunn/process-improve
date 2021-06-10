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

`wide`: as a single Pandas data frame, as for the "melted" version, but pivoted instead, and
these `wide` dataframes *always* have a multilevel column index, to help distinguish the tags
from the time. It is a requirement of course that this representation is only for aligned data.

    data = pd.DataFrame(...)

        * each row is a unique batch number
        * the multilevel column index has
            * level 0: the column name
            * level 1: the aligned time

        This form only makes sense if the data are aligned, so that there is the same number of
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
    assert len(in_dict) > 1, "At least 1 batch is required in the dataframe dictionary."
    batch1 = in_dict[list(in_dict.keys())[0]]
    base_columns = set(batch1.columns)
    check = True
    for bid, batch in in_dict.items():
        # Check 1
        check = check & (base_columns == set(batch.columns))
        assert (
            check
        ), f"The column names must be the same in all batches. Differs in {bid}."

        # Check 2
        check *= batch.select_dtypes(include=[np.number]).shape[1] == batch.shape[1]
        assert check, f"All columns must be a numeric type. Differs in {bid}."

        # Check 3
        if no_nan:
            check *= batch.isna().values.sum() == 0
            assert check, f"No missing values allowed. Missing values found in {bid}."

    return bool(check)


def dict_to_melted(
    in_df: pd.DataFrame, insert_batch_id_column=True, insert_sequence_column=False
) -> pd.DataFrame:
    """Reverse of `melted_to_dict`"""
    out_df = pd.DataFrame()
    batch_id_col = "batch_id"

    num_rows = 0
    for idx, (batch_id, batch) in enumerate(in_df.items()):
        if idx == 0:
            num_rows = batch.shape[0]
            sequence = np.arange(0, num_rows)
        assert (
            num_rows == batch.shape[0]
        ), "All batches must have the same number of samples"

        if insert_batch_id_column and batch_id_col not in batch:
            batch.insert(0, batch_id_col, batch_id)

        if insert_sequence_column:
            batch.insert(0, "_sequence_", sequence)

        out_df = out_df.append(batch)

    return out_df


def dict_to_wide(in_df: dict, group_by_batch=False) -> pd.DataFrame:
    """
    Data must be aligned (warped) already so that every batch has the same number of *rows*!

    `group_by_batch`, if True, means that all the data from the first batch is on the left
    of the output dataframe, and the last batch is collected on the right.

    If `group_by_batch` is False, then data for the same tag are grouped together, side-by-side.
    """
    out_df = dict_to_melted(
        in_df=in_df, insert_batch_id_column=True, insert_sequence_column=True
    )
    aligned_wide_df = out_df.pivot(index="batch_id", columns="_sequence_")
    if group_by_batch:
        pass
        # TODO: use the hierarchical indexing and regroup the columns

    return aligned_wide_df


def melted_to_dict(in_df: pd.DataFrame, batch_id_col) -> dict:
    """
    Loads a "melted" data set, where one of the columns is the `batch_id_col`.
    The data are grouped along the unique values of `batch_id_col`, and each group is stored
    in a dictionary. The dictionary keys are the batch identifier, and the corresponding value
    is a Pandas dataframe of the batch data for that batch.
    """
    batches = {}
    assert (
        batch_id_col in in_df
    ), "The `batch_id_col` column does not exist in the incoming dataframe."
    for batch_id, batch in in_df.groupby(batch_id_col):
        batches[batch_id] = batch

    return batches


def melted_to_wide(in_df: pd.DataFrame, batch_id_col) -> dict:
    """
    Data must be aligned already.
    """
    assert batch_id_col in in_df
    return {}

    # max_places = int(np.ceil(np.log10(aligned_df["_sequence_"].max())))
    # aligned_wide_df = aligned_df.pivot(index="batch_id", columns="_sequence_")
    # new_labels = [
    #     "-".join(item)
    #     for item in zip(
    #         aligned_wide_df.columns.get_level_values(0),
    #         [str(val).zfill(max_places) for val in aligned_wide_df.columns.get_level_values(1)],
    #     )
    # ]
    # aligned_wide_df.columns = new_labels
    # TODO: add the column multilevel column index.
    # return dict_to_wide(melted_to_dict(in_df, batch_id_col))


def wide_to_melted(in_df: pd.DataFrame) -> pd.DataFrame:
    # dict_to_melted(dict_to_wide(in_df))
    return pd.DataFrame()


def wide_to_dict():
    pass


def melt_df_to_series(
    in_df: pd.DataFrame, exclude_columns=["batch_id"], name=None
) -> pd.Series:
    """Returns a Series with a multilevel-index, melted from the DataFrame"""
    out = in_df.drop(exclude_columns, axis=1).T.stack()
    out.name = name
    return out
