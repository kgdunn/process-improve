"""
Getting data into the required format for use with this library.
"""

import pandas as pd


def load_melted_data_with_id(in_df: pd.DataFrame, batch_id_col) -> dict:
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
