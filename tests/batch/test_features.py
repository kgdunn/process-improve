import pathlib
import pytest
from pytest import approx
import pandas as pd

import process_improve.batch.features as features

# General


@pytest.fixture(scope="module")
def batch_data():
    """Returns a small example of a batch data set."""
    folder = (
        pathlib.Path(__file__).parents[2] / "process_improve" / "datasets" / "batch"
    )
    return pd.read_csv(
        folder / "batch-fake-data.csv",
        index_col=1,
        header=0,
    )
    # return


def test_verify_file(batch_data):
    df = batch_data
    assert df.shape[0] == 501
    assert df.shape[1] == 5


def test_corner_cases(batch_data):
    """
    Certain corner cases: to ensure coverage in
    """
    df = batch_data
    data = df.set_index(pd.to_datetime(df["DateTime"])).drop("DateTime", axis=1)
    step1 = features.f_mean(data).reset_index()

    # Tests removal of internal columns.
    _, tags, *_ = features._prepare_data(step1)
    assert "__phase_grouper__" not in tags
    assert "__batch_grouper__" not in tags

    # Test calling a single tag name
    assert features.f_mean(data, tags="Temp1").loc[:, "Temp1_mean"].values[0] == approx(
        -19.482056, rel=1e-7
    )


def test_age_col_specification(batch_data):
    """
    Some features, like slopes, need to know a value from the x-axis. Often this is a column
    representing the time since start of the batch: age_col
    """
    df = batch_data
    df = df.drop("DateTime", axis=1)

    # Check that the index is currently the time-tag
    assert df.index.name == "UCI_minutes"

    # This test is for the case when the time_tag is NOT the index. So reset that:
    df = df.reset_index()
    slopes = features.f_slope(
        df, x_axis_tag="UCI_minutes", tags=["Temp1", "Temp2"], age_col="UCI_minutes"
    )
    assert slopes.shape == (1, 2)


def test_data_preprocessing(batch_data):
    """Simple tests regarding the mean, median, etc. Location-based features."""

    df = batch_data
    df = df.set_index(pd.to_datetime(df["DateTime"])).drop("DateTime", axis=1)
    data = df["Temp1"]
    assert features.f_mean(data).values[0] == approx([-19.482056], rel=1e-7)

    data = df["Temp2"]
    data.name = None
    assert features.f_mean(data).values[0] == approx([-47.649381], rel=1e-7)

    # TODO: Test the age column:


# Location
def test_location_features(batch_data):
    """Simple tests regarding the mean, median, etc. Location-based features."""

    assert features.f_mean(batch_data, tags=["Temp1", "Temp2", "Pressure1"]).values[
        0
    ] == approx([-19.482056, -47.649381, 0.444674], rel=1e-7)
    assert features.f_median(batch_data, tags=["Temp1", "Temp2", "Pressure1"]).values[
        0
    ] == approx([-28.19, -47.86, 0.3333059], rel=1e-7)


# Shape
def test_shape_features(batch_data):

    slopes = features.f_slope(
        batch_data,
        x_axis_tag="UCI_minutes",
        tags=["Temp1", "Temp2"],
        batch_col="Batch",
        age_col="UCI_minutes",
    )
    # Actual values checked against Datamore's robust linear regression fitting tool
    slopes.iloc[0]["Temp1_slope"] == approx(+0.009564041, rel=1e-7)
    slopes.iloc[1]["Temp1_slope"] == approx(+0.004404998, rel=1e-7)
    slopes.iloc[0]["Temp2_slope"] == approx(-0.000292716, rel=1e-7)
    slopes.iloc[1]["Temp2_slope"] == approx(+0.002852301, rel=1e-7)


# Cumulative features
def test_sum_features(batch_data):
    """Simple tests regarding the area
    Values were calculated manually in Excel."""

    assert features.f_sum(batch_data, tags=["Temp1", "Temp2", "Pressure1"]).values[
        0
    ] == approx([-9760.51, -23872.34, 222.781677], rel=1e-9)
    assert (
        features.f_sum(
            batch_data,
            tags=["Temp1", "Temp2", "Pressure1"],
            batch_col="Batch",
        ).values[0]
        == approx([-7304.88, -20801.57, 182.5183], rel=1e-6)
    )
    assert (
        features.f_area(
            batch_data,
            tags=["Temp1", "Temp2", "Pressure1"],
            batch_col="Batch",
            time_tag="UCI_minutes",
        ).values[0]
        == approx([-73095.6162, -207910.839, 1648.069279], rel=1e-7)
    )
