import pathlib

import pandas as pd
import pytest

from process_improve.batch.data_input import melted_to_dict, melted_to_wide


@pytest.fixture()
def nylon_raw_melteddata():
    nylon_raw = pd.read_csv(pathlib.Path(__file__).parents[2] / "process_improve" / "datasets" / "batch" / "nylon.csv")
    return nylon_raw


def test_melted_to_dict(nylon_raw_melteddata):
    out = melted_to_dict(nylon_raw_melteddata, batch_id_col="batch_id")
    assert len(out) == 57


def test_melted_to_wide(nylon_raw_melteddata):
    _ = melted_to_wide(nylon_raw_melteddata, batch_id_col="batch_id")
    # assert out.shape == pytest.approx([2, 3])


def test_wide_to_melted():
    pass


def test_wide_to_dict():
    pass


def test_melt_df_to_series():
    pass
