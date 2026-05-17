import pathlib

import pandas as pd
import pytest

from process_improve.batch.data_input import (
    dict_to_melted,
    dict_to_wide,
    melt_df_to_series,
    melted_to_dict,
    melted_to_wide,
    wide_to_melted,
)


@pytest.fixture
def aligned_batch_dict() -> dict:
    """Two batches with identical shape, suitable for dict/wide conversions."""
    return {
        "A": pd.DataFrame({"temp": [1.0, 2.0, 3.0], "press": [4.0, 5.0, 6.0]}),
        "B": pd.DataFrame({"temp": [7.0, 8.0, 9.0], "press": [10.0, 11.0, 12.0]}),
    }


@pytest.fixture
def nylon_raw_melteddata() -> pd.DataFrame:
    """Load raw nylon melted data."""
    return pd.read_csv(pathlib.Path(__file__).parents[2] / "process_improve" / "datasets" / "batch" / "nylon.csv")


def test_melted_to_dict(nylon_raw_melteddata: pd.DataFrame) -> None:
    """Test conversion from melted format to dictionary."""
    out = melted_to_dict(nylon_raw_melteddata, batch_id_col="batch_id")
    assert len(out) == 57


def test_melted_to_wide(nylon_raw_melteddata: pd.DataFrame) -> None:
    """Test conversion from melted format to wide format."""
    _ = melted_to_wide(nylon_raw_melteddata, batch_id_col="batch_id")
    # assert out.shape == pytest.approx([2, 3])


def test_wide_to_melted() -> None:
    """Test conversion from wide format to melted format."""
    out = wide_to_melted(pd.DataFrame({"x": [1, 2]}))
    assert isinstance(out, pd.DataFrame)
    assert out.empty


def test_wide_to_dict() -> None:
    """Test conversion from wide format to dictionary."""


def test_dict_to_melted_default_inserts_batch_id(aligned_batch_dict: dict) -> None:
    """With defaults, a batch_id column is added and no sequence column."""
    out = dict_to_melted(aligned_batch_dict)
    assert "batch_id" in out.columns
    assert "_sequence_" not in out.columns
    assert out.shape[0] == 6


def test_dict_to_melted_without_batch_id_column(aligned_batch_dict: dict) -> None:
    """When insert_batch_id_column is False, no batch_id column is added."""
    out = dict_to_melted(aligned_batch_dict, insert_batch_id_column=False)
    assert "batch_id" not in out.columns
    assert out.shape[0] == 6


def test_dict_to_wide_group_by_batch(aligned_batch_dict: dict) -> None:
    """dict_to_wide accepts the group_by_batch flag."""
    out = dict_to_wide(aligned_batch_dict, group_by_batch=True)
    assert out.shape[0] == 2


def test_melt_df_to_series() -> None:
    """Test melting a DataFrame to a Series."""
    df = pd.DataFrame({"batch_id": ["A", "A"], "temp": [1.0, 2.0], "press": [3.0, 4.0]})
    series = melt_df_to_series(df, name="obs")
    assert isinstance(series, pd.Series)
    assert series.name == "obs"
    assert len(series) == 4
