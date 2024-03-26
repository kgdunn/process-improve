import pathlib

import pandas as pd
import pytest

from process_improve.batch import features

# General


@pytest.fixture(scope="module")
def batch_data():
    """Return a small example of a batch data set."""
    folder = pathlib.Path(__file__).parents[2] / "process_improve" / "datasets" / "batch"
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
    assert features.f_mean(data, tags="Temp1").loc[:, "Temp1_mean"].values[0] == pytest.approx(-19.482056, rel=1e-7)


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
    slopes = features.f_slope(df, x_axis_tag="UCI_minutes", tags=["Temp1", "Temp2"], age_col="UCI_minutes")
    assert slopes.shape == (1, 2)


def test_data_preprocessing(batch_data):
    """Simple tests regarding the mean, median, etc. Location-based features."""

    df = batch_data
    df = df.set_index(pd.to_datetime(df["DateTime"])).drop("DateTime", axis=1)
    data = df["Temp1"]
    assert features.f_mean(data).values[0] == pytest.approx([-19.482056], rel=1e-7)

    data = df["Temp2"]
    data.name = None
    assert features.f_mean(data).values[0] == pytest.approx([-47.649381], rel=1e-7)

    # TODO: Test the age column:


# Location-based features
# ------------------------------------------
def test_location_features(batch_data):
    """Simple tests regarding the mean, median, etc. Location-based features."""

    assert features.f_mean(batch_data, tags=["Temp1", "Temp2", "Pressure1"], batch_col="Batch").loc[1].values[
        0
    ] == pytest.approx([-16.71597254, -47.60084668, 0.41766206], abs=1e-7)

    assert features.f_median(batch_data, tags=["Temp1", "Temp2", "Pressure1"], batch_col="Batch").loc[1].values[
        0
    ] == pytest.approx([-25.63, -47.78, 0.33330592], abs=1e-7)


# Scale-based features
# ------------------------------------------
def test_scale_features(batch_data):
    """Simple tests regarding the scale-based features."""

    assert features.f_std(batch_data, tags=["Temp1", "Temp2", "Pressure1"], batch_col="Batch").values[
        0
    ] == pytest.approx([23.19985051, 1.321310847, 1.691319825], rel=1e-7)
    assert features.f_iqr(batch_data, tags=["Temp1", "Temp2", "Pressure1"], batch_col="Batch").values[
        0
    ] == pytest.approx([27.54, 1.85, 0.06666118399999998], rel=1e-7)


# Shape-based features
# ------------------------------------------
def test_shape_features(batch_data):
    slopes = features.f_slope(
        batch_data,
        x_axis_tag="UCI_minutes",
        tags=["Temp1", "Temp2"],
        batch_col="Batch",
        age_col="UCI_minutes",
    )
    # Actual values checked against Datamore's robust linear regression fitting tool
    slopes.iloc[0]["Temp1_slope"] == pytest.approx(+0.009564041, rel=1e-7)
    slopes.iloc[1]["Temp1_slope"] == pytest.approx(+0.004404998, rel=1e-7)
    slopes.iloc[0]["Temp2_slope"] == pytest.approx(-0.000292716, rel=1e-7)
    slopes.iloc[1]["Temp2_slope"] == pytest.approx(+0.002852301, rel=1e-7)


# Cumulative features
def test_sum_features(batch_data):
    """Simple tests regarding the area
    Values were calculated manually in Excel."""

    assert features.f_sum(batch_data, tags=["Temp1", "Temp2", "Pressure1"]).values[0] == pytest.approx(
        [-9760.51, -23872.34, 222.781677], rel=1e-9
    )
    assert features.f_sum(
        batch_data,
        tags=["Temp1", "Temp2", "Pressure1"],
        batch_col="Batch",
    ).values[
        0
    ] == pytest.approx([-7304.88, -20801.57, 182.5183], rel=1e-6)
    assert features.f_area(
        batch_data,
        tags=["Temp1", "Temp2", "Pressure1"],
        batch_col="Batch",
        time_tag="UCI_minutes",
    ).values[0] == pytest.approx([-73095.6162, -207910.839, 1648.069279], rel=1e-7)


# Extreme features
# ------------------------------------------
def test_extreme_features(batch_data):
    """Simple tests regarding the extremum features."""

    assert features.f_min(batch_data, tags=["Temp1", "Temp2", "Pressure1"], batch_col="Batch").values[
        0
    ] == pytest.approx([-43.03, -49.92, 0.266644736], rel=1e-7)
    assert features.f_max(batch_data, tags=["Temp1", "Temp2", "Pressure1"], batch_col="Batch").values[
        0
    ] == pytest.approx([28.9, -42.14, 35.62373673], rel=1e-7)
    assert features.f_last(batch_data, tags=["Temp1", "Temp2", "Pressure1"], batch_col="Batch").values[
        0
    ] == pytest.approx([25.99, -47.45, 0.399967104], rel=1e-7)
    assert features.f_count(batch_data, tags=["Temp1", "Temp2", "Pressure1"], batch_col="Batch").values[
        0
    ] == pytest.approx([437, 437, 437], rel=1e-7)
