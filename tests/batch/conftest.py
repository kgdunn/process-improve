import pathlib
import pytest
import pandas as pd

from batch_preprocessing.data_input import load_melted_data_with_id


@pytest.fixture
def nylon_data():
    nylon_raw = pd.read_csv(
        pathlib.Path(__file__).parents[0] / "fixtures" / "nylon.csv"
    )
    return load_melted_data_with_id(nylon_raw, batch_id_col="batch_id")


@pytest.fixture
def dryer_data():
    dryer_raw = pd.read_csv(
        pathlib.Path(__file__).parents[0] / "fixtures" / "dryer.csv"
    )
    return load_melted_data_with_id(dryer_raw, batch_id_col="batch_id")
