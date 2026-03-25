import pathlib

import pandas as pd
import pytest

from process_improve.batch.data_input import melted_to_dict, melted_to_wide


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


def test_wide_to_dict() -> None:
    """Test conversion from wide format to dictionary."""


def test_melt_df_to_series() -> None:
    """Test melting a DataFrame to a Series."""
