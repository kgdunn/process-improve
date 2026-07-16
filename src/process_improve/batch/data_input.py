"""
Getting data into the required format for use with this library.

There are 3 useful ways to represent batch data.

``dict``: as a Python dictionary. Example::

    data = {
        "batch 1": data frame with varying number of rows, but same number of columns,
        "batch 2": etc,
    }

The keys are unique identifiers for each batch, such as integers or strings.

``melt``: as a single Pandas data frame::

    data = pd.DataFrame(...)

Characteristics:

- very large number of rows, for all batches stacked vertically on top of each other
- some number of columns, one column per tag
- one column, usually called ``batch_id``, indicates what the batch number is for that row
- another column, usually called ``time``, indicates what the time is within that batch
- typically sorted, but does not have to be

``wide``: as a single Pandas data frame, as for the "melted" version, but pivoted instead.
These ``wide`` dataframes *always* have a multilevel column index to distinguish the tags
from the time. This representation is only valid for aligned data. Example::

    data = pd.DataFrame(...)

Characteristics:

- each row is a unique batch number
- the multilevel column index has level 0 = column name, level 1 = aligned time
- only makes sense if the data are aligned (same number of elements in each level-1 index)
"""
from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd


def check_valid_batch_dict(in_dict: dict, no_nan: bool = False) -> bool:
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
    if len(in_dict) < 1:
        raise ValueError("At least 1 batch is required in the dataframe dictionary.")
    batch1 = in_dict[next(iter(in_dict.keys()))]
    base_columns = set(batch1.columns)
    check = True
    for bid, batch in in_dict.items():
        # Check 1
        check = check & (base_columns == set(batch.columns))
        if not check:
            raise ValueError(
                f"The column names must be the same in all batches. Differs in {bid}. Base "
                f"columns = {base_columns}; this batch has: {set(batch.columns)}"
            )

        # Check 2
        check *= batch.select_dtypes(include=[np.number]).shape[1] == batch.shape[1]
        if not check:
            raise ValueError(f"All columns must be a numeric type. Differs in {bid}.")

        # Check 3
        if no_nan:
            check *= batch.isna().values.sum() == 0
            if not check:
                raise ValueError(f"No missing values allowed. Missing values found in {bid}.")

    return bool(check)


def dict_to_melted(
    in_df: dict,
    insert_batch_id_column: bool = True,
    insert_sequence_column: bool = False,
) -> pd.DataFrame:
    """Reverse of `melted_to_dict`."""
    batch_id_col = "batch_id"
    all_batches = []
    num_rows = 0
    for idx, (batch_id, batch) in enumerate(in_df.items()):
        if idx == 0:
            num_rows = batch.shape[0]
            sequence = np.arange(0, num_rows)
        if num_rows != batch.shape[0]:
            raise ValueError("All batches must have the same number of samples")

        subset = batch.copy()

        if insert_batch_id_column and batch_id_col not in batch:
            subset.insert(0, batch_id_col, batch_id)

        if insert_sequence_column:
            subset.insert(0, "_sequence_", sequence)

        all_batches.append(subset)

    return pd.concat(all_batches)


def dict_to_wide(in_df: dict, group_by_batch: bool = False) -> pd.DataFrame:
    """
    Convert aligned batch data from a dict to wide format.

    Each row of the output is one batch; the columns are a 2-level
    ``("tag", "sequence")`` index, so the data are only meaningful for aligned
    batches (every batch has the same number of samples).

    Parameters
    ----------
    in_df : dict
        Standard batch-data dictionary: keys are batch identifiers, values are
        per-batch dataframes with identical columns.
    group_by_batch : bool, optional
        Controls the ordering of the hierarchical column index.

        * ``False`` (default): columns are ordered ``(tag, sequence)``, so all
          time samples for a tag are grouped together, side-by-side.
        * ``True``: the levels are swapped to ``(sequence, tag)``, so all tags
          for a given time sample are grouped together.

    Returns
    -------
    pd.DataFrame
        Wide-format dataframe, one row per batch, with a 2-level column index.
    """
    out_df = dict_to_melted(in_df=in_df, insert_batch_id_column=True, insert_sequence_column=True)
    aligned_wide_df = out_df.pivot_table(index="batch_id", columns="_sequence_")
    aligned_wide_df.columns = aligned_wide_df.columns.set_names(["tag", "sequence"])
    if group_by_batch:
        aligned_wide_df = aligned_wide_df.swaplevel("tag", "sequence", axis=1).sort_index(axis=1)

    return aligned_wide_df


def melted_to_dict(in_df: pd.DataFrame, batch_id_col: str) -> dict:
    """
    Load a "melted" data set, where one of the columns is the `batch_id_col`.
    The data are grouped along the unique values of `batch_id_col`, and each group is stored
    in a dictionary. The dictionary keys are the batch identifier, and the corresponding value
    is a Pandas dataframe of the batch data for that batch.
    """
    if batch_id_col not in in_df:
        raise ValueError("The `batch_id_col` column does not exist in the incoming dataframe.")
    return {batch_id: batch for batch_id, batch in in_df.groupby(batch_id_col)}  # noqa: C416


def melted_to_wide(in_df: pd.DataFrame, batch_id_col: str, group_by_batch: bool = False) -> pd.DataFrame:
    """
    Convert aligned melted data to wide format.

    Parameters
    ----------
    in_df : pd.DataFrame
        Melted batch data: all batches stacked vertically, one column per tag,
        with a batch-identifier column. The batches must be aligned (the same
        number of rows per batch), because the wide format is only meaningful
        for aligned data.
    batch_id_col : str
        Name of the column holding the batch identifier.
    group_by_batch : bool, optional
        Passed through to :func:`dict_to_wide`; controls whether the 2-level
        column index is ordered ``(tag, sequence)`` (default) or
        ``(sequence, tag)``.

    Returns
    -------
    pd.DataFrame
        Wide-format dataframe: one row per batch, 2-level column index.
    """
    if batch_id_col not in in_df:
        raise ValueError(f"The `batch_id_col` column {batch_id_col!r} does not exist in the incoming dataframe.")
    return dict_to_wide(melted_to_dict(in_df, batch_id_col), group_by_batch=group_by_batch)


def wide_to_dict(in_df: pd.DataFrame) -> dict:
    """
    Convert wide-format batch data back to the standard dict format.

    Inverts :func:`dict_to_wide`: each row of the wide frame becomes one entry
    in the dictionary, with the 2-level ``(tag, sequence)`` column index
    pivoted back to a per-batch dataframe of one column per tag, indexed by
    sequence. Accepts either column-level ordering (``(tag, sequence)`` or the
    ``group_by_batch=True`` variant ``(sequence, tag)``).

    Parameters
    ----------
    in_df : pd.DataFrame
        Wide-format batch data: one row per batch, 2-level column index with
        levels named ``tag`` and ``sequence``.

    Returns
    -------
    dict
        Standard batch-data dictionary: keys are the wide frame's row index
        (batch identifiers), values are per-batch dataframes.
    """
    if in_df.columns.nlevels != 2 or set(in_df.columns.names) != {"tag", "sequence"}:
        raise ValueError(
            "The wide dataframe must have a 2-level column index with levels named "
            f"'tag' and 'sequence'; got levels {in_df.columns.names}."
        )
    out = {}
    for batch_id, row in in_df.iterrows():
        # Series.unstack is the direct inverse of the pivot in `dict_to_wide`;
        # pivot_table would aggregate and lose the (tag, sequence) fidelity.
        batch = row.unstack(level="tag")  # noqa: PD010
        batch.index.name = None
        batch.columns.name = None
        out[batch_id] = batch
    return out


def wide_to_melted(in_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert wide-format batch data to melted format.

    Inverts the melted-to-wide direction: the wide frame (one row per batch,
    2-level column index) is expanded back to a melted frame with all batches
    stacked vertically, one column per tag, and a ``batch_id`` column.

    Parameters
    ----------
    in_df : pd.DataFrame
        Wide-format batch data: one row per batch, 2-level column index with
        levels named ``tag`` and ``sequence``.

    Returns
    -------
    pd.DataFrame
        Melted batch data with a ``batch_id`` column.
    """
    return dict_to_melted(wide_to_dict(in_df), insert_batch_id_column=True)


def melt_df_to_series(in_df: pd.DataFrame, exclude_columns: list | None = None, name: str | None = None) -> pd.Series:
    """Return a Series with a multilevel-index, melted from the DataFrame."""
    if exclude_columns is None:
        exclude_columns = ["batch_id"]
    out = cast("pd.Series", in_df.drop(exclude_columns, axis=1).T.stack())  # noqa: PD013  # noqa: PD013
    out.name = name
    return out
