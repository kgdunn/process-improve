# (c) Kevin Dunn, 2010-2025. MIT License. Based on own private work over the years.
from __future__ import annotations

from collections.abc import KeysView
from typing import TypeAlias

import numpy as np
import pandas as pd

DataMatrix: TypeAlias = np.ndarray | pd.DataFrame

epsqrt = np.sqrt(np.finfo(float).eps)


class DataFrameDict(dict):
    def __init__(self, datadict: dict[str, dict[str, pd.DataFrame]]):
        """
        Initialize a DataFrameDict to handle partitionable and static dataframes.

        datadict: Dictionary with 3 keys, one for each block: Z, F and Y.
                  Each block is itself a dictionary of dataframes: dict[str, dict[str, pd.DataFrame]]
        """

        self.partitionable_blocks: list[str] = ["Z", "F", "Y"]
        self.datadict: dict[str, dict[str, pd.DataFrame]] = {}
        for block in self.partitionable_blocks:
            self.datadict[block] = datadict.get(block, {})
        first_group = next(iter(self.datadict["F"].keys()))
        self.n_samples = self.datadict["F"][first_group].shape[0]
        self.shape = (self.n_samples, len(self.datadict))

        # Some basic checks: each dataframe inside each block has the same number of rows
        for block in set(self.partitionable_blocks) & set(self.datadict.keys()):
            for group, df in self.datadict[block].items():
                if not isinstance(df, pd.DataFrame):
                    raise TypeError(f"Expected a DataFrame for block {block}, group '{group}'; got instead{type(df)}.")
                if df.shape[0] != self.n_samples:
                    raise ValueError(
                        f"DataFrames in block {block} must have the same number of rows ({self.n_samples}). "
                        f"Group {group} has {df.shape[0]} rows."
                    )

    def keys(self) -> KeysView[str]:
        """Return the keys of the DataFrameDict."""
        return self.datadict.keys()

    def __setitem__(self, key: str, value: pd.DataFrame | dict) -> None:
        """Set a DataFrame for a specific key in the DataFrameDict."""
        if key not in self.partitionable_blocks:
            raise KeyError(f"Key {key} is not a valid partitionable block. Valid keys are: {self.partitionable_blocks}")

        if not isinstance(value, pd.DataFrame):
            raise TypeError(f"Expected a DataFrame for key {key}, got {type(value)}.")
        if value.shape[0] != self.n_samples:
            raise ValueError(
                f"DataFrames in block {key} must have the same number of rows ({self.n_samples}). "
                f"Provided DataFrame has {value.shape[0]} rows."
            )
        self.datadict[key] = value

    def __getitem__(self, lookup: int | list[int] | str) -> DataFrameDict | dict[str, pd.DataFrame]:
        """Return a new DataFrameDict with partitioned data."""

        if isinstance(lookup, str):
            return self.datadict[lookup]  # returns the `dict[str, pd.DataFrame]` version of the function

        datadict: dict[str, dict[str, pd.DataFrame]] = {}
        for block in self.partitionable_blocks:
            datadict[block] = {}
            for group, df in self.datadict[block].items():
                match lookup:
                    case int() | np.integer():
                        datadict[block][group] = df.iloc[[lookup]]
                    case list():
                        datadict[block][group] = df.iloc[lookup]
                    case np.ndarray():
                        datadict[block][group] = df.iloc[lookup.tolist()]
                    case tuple():
                        if lookup[1] == Ellipsis:
                            datadict[block][group] = df.iloc[[int(item) for item in lookup[0]]]
                        else:
                            raise TypeError(f"Invalid tuple structure for lookup: {lookup}")
                    case _:
                        raise TypeError(
                            f"Lookup must be an int, list of ints, or a string. Got {lookup}; {type(lookup)}"
                        )

        return DataFrameDict(datadict)

    def __len__(self):
        """Return the number of samples in the DataFrameDict."""
        return self.n_samples

    def __repr__(self):
        """Return a string representation of the DataFrameDict."""
        groups_in_block_f = list(self.datadict["F"].keys())
        groups_in_block_z = list(self.datadict["Z"].keys())
        groups_in_block_y = list(self.datadict["Y"].keys())
        output = f"DataFrameDict with {len(self)} samples and {len(self.datadict)} blocks: {list(self.datadict.keys())}"
        output += f"\n  F groups: {groups_in_block_f}"
        output += f"\n  Z groups: {groups_in_block_z}"
        output += f"\n  Y groups: {groups_in_block_y}"
        return output
