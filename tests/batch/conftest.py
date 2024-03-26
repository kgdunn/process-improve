import pathlib

import pandas as pd
import pytest

from process_improve.batch.data_input import melted_to_dict


@pytest.fixture()
def nylon_data():
    nylon_raw = pd.read_csv(pathlib.Path(__file__).parents[2] / "process_improve" / "datasets" / "batch" / "nylon.csv")
    return melted_to_dict(nylon_raw, batch_id_col="batch_id")


@pytest.fixture()
def dryer_data():
    dryer_raw = pd.read_csv(pathlib.Path(__file__).parents[2] / "process_improve" / "datasets" / "batch" / "dryer.csv")
    return melted_to_dict(dryer_raw, batch_id_col="batch_id")
