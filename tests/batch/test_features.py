import pathlib

import numpy as np
import pandas as pd
import pytest

from process_improve.batch import features
from process_improve.batch.features import cross

# General


@pytest.fixture(scope="module")
def batch_data() -> pd.DataFrame:
    """Return a small example of a batch data set."""
    folder = pathlib.Path(__file__).parents[2] / "src" / "process_improve" / "datasets" / "batch"
    return pd.read_csv(
        folder / "batch-fake-data.csv",
        index_col=1,
        header=0,
    )
    # return


def test_verify_file(batch_data: pd.DataFrame) -> None:
    """Verify the batch data file was loaded correctly."""
    df = batch_data
    assert df.shape[0] == 501
    assert df.shape[1] == 5


def test_corner_cases(batch_data: pd.DataFrame) -> None:
    """Certain corner cases: to ensure coverage."""
    df = batch_data
    data = df.set_index(pd.to_datetime(df["DateTime"])).drop("DateTime", axis=1)
    step1 = features.f_mean(data).reset_index()

    # Tests removal of internal columns.
    _, tags, *_ = features._prepare_data(step1)
    assert "__phase_grouper__" not in tags
    assert "__batch_grouper__" not in tags

    # Test calling a single tag name
    assert features.f_mean(data, tags="Temp1").loc[:, "Temp1_mean"].values[0] == pytest.approx(-19.482056, rel=1e-7)


def test_age_col_specification(batch_data: pd.DataFrame) -> None:
    """Some features, like slopes, need to know a value from the x-axis: age_col."""
    df = batch_data
    df = df.drop("DateTime", axis=1)

    # Check that the index is currently the time-tag
    assert df.index.name == "UCI_minutes"

    # This test is for the case when the time_tag is NOT the index. So reset that:
    df = df.reset_index()
    slopes = features.f_slope(df, x_axis_tag="UCI_minutes", tags=["Temp1", "Temp2"], age_col="UCI_minutes")
    assert slopes.shape == (1, 2)


def test_data_preprocessing(batch_data: pd.DataFrame) -> None:
    """Simple tests regarding the mean, median, etc. Location-based features."""

    df = batch_data
    df = df.set_index(pd.to_datetime(df["DateTime"])).drop("DateTime", axis=1)
    data = df["Temp1"]
    assert features.f_mean(data).values[0] == pytest.approx([-19.482056], rel=1e-7)

    data = df["Temp2"]
    data.name = None
    assert features.f_mean(data).values[0] == pytest.approx([-47.649381], rel=1e-7)


# Location-based features
# ------------------------------------------
def test_location_features(batch_data: pd.DataFrame) -> None:
    """Simple tests regarding the mean, median, etc. Location-based features."""

    assert features.f_mean(batch_data, tags=["Temp1", "Temp2", "Pressure1"], batch_col="Batch").loc[1].values[
        0
    ] == pytest.approx([-16.71597254, -47.60084668, 0.41766206], abs=1e-7)

    assert features.f_median(batch_data, tags=["Temp1", "Temp2", "Pressure1"], batch_col="Batch").loc[1].values[
        0
    ] == pytest.approx([-25.63, -47.78, 0.33330592], abs=1e-7)


# Scale-based features
# ------------------------------------------
def test_scale_features(batch_data: pd.DataFrame) -> None:
    """Simple tests regarding the scale-based features."""

    assert features.f_std(batch_data, tags=["Temp1", "Temp2", "Pressure1"], batch_col="Batch").values[
        0
    ] == pytest.approx([23.19985051, 1.321310847, 1.691319825], rel=1e-7)
    assert features.f_iqr(batch_data, tags=["Temp1", "Temp2", "Pressure1"], batch_col="Batch").values[
        0
    ] == pytest.approx([27.54, 1.85, 0.06666118399999998], rel=1e-7)


def test_f_robust_mad(batch_data: pd.DataFrame) -> None:
    """f_robust_mad returns a positive robust scale estimate per tag and batch."""
    tags = ["Temp1", "Temp2", "Pressure1"]
    mad = features.f_robust_mad(batch_data, tags=tags, batch_col="Batch")
    std = features.f_std(batch_data, tags=tags, batch_col="Batch")
    assert list(mad.columns) == ["Temp1_robust_mad", "Temp2_robust_mad", "Pressure1_robust_mad"]
    assert mad.shape == std.shape
    assert (mad.values > 0).all()


def test_f_agemin_agemax(batch_data: pd.DataFrame) -> None:
    """f_agemin / f_agemax report the index label where the min / max occurs."""
    agemin = features.f_agemin(batch_data, tags=["Temp1"], batch_col="Batch")
    agemax = features.f_agemax(batch_data, tags=["Temp1"], batch_col="Batch")
    assert list(agemin.columns) == ["Temp1_agemin"]
    assert list(agemax.columns) == ["Temp1_agemax"]

    # Cross-check the first batch against the raw data.
    first_batch = batch_data["Batch"].iloc[0]
    sub = batch_data[batch_data["Batch"] == first_batch]
    assert agemin.iloc[0, 0] == sub["Temp1"].idxmin()
    assert agemax.iloc[0, 0] == sub["Temp1"].idxmax()


# Shape-based features
# ------------------------------------------
def test_shape_features(batch_data: pd.DataFrame) -> None:
    """Test slope-based shape features."""
    slopes = features.f_slope(
        batch_data,
        x_axis_tag="UCI_minutes",
        tags=["Temp1", "Temp2"],
        batch_col="Batch",
        age_col="UCI_minutes",
    )
    # Actual values checked against Datamore's robust linear regression fitting tool
    assert slopes.iloc[0]["Temp1_slope"] == pytest.approx(+0.009564041, rel=1e-7)
    assert slopes.iloc[1]["Temp1_slope"] == pytest.approx(+0.004404998, rel=1e-7)
    assert slopes.iloc[0]["Temp2_slope"] == pytest.approx(-0.000292716, rel=1e-7)
    assert slopes.iloc[1]["Temp2_slope"] == pytest.approx(+0.002852301, rel=1e-7)


# Cumulative features
def test_sum_features(batch_data: pd.DataFrame) -> None:
    """Simple tests regarding the area.

    Values were calculated manually in Excel.
    """

    assert features.f_sum(batch_data, tags=["Temp1", "Temp2", "Pressure1"]).values[0] == pytest.approx(
        [-9760.51, -23872.34, 222.781677], rel=1e-9
    )
    assert features.f_sum(
        batch_data,
        tags=["Temp1", "Temp2", "Pressure1"],
        batch_col="Batch",
    ).values[0] == pytest.approx([-7304.88, -20801.57, 182.5183], rel=1e-6)

    assert features.f_area(
        batch_data,
        tags=["Temp1", "Temp2", "Pressure1"],
        batch_col="Batch",
        time_tag="UCI_minutes",
    ).values[0] == pytest.approx([-73095.6162, -207910.839, 1648.069279], rel=1e-7)


# Extreme features
# ------------------------------------------
def test_extreme_features(batch_data: pd.DataFrame) -> None:
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


# Cross (threshold crossing) helper and feature
# ------------------------------------------
def test_cross_rising() -> None:
    """cross() should find rising-edge threshold crossings."""
    series = pd.Series([-2, -1, 1, 2, 3], index=[0, 1, 2, 3, 4])
    result = cross(series, threshold=0, direction="rising")
    assert len(result) == 1
    assert result[0] == pytest.approx(1.5, abs=1e-6)


def test_cross_falling() -> None:
    """cross() should find falling-edge threshold crossings."""
    series = pd.Series([3, 2, 1, -1, -2], index=[0, 1, 2, 3, 4])
    result = cross(series, threshold=0, direction="falling")
    assert len(result) == 1
    assert result[0] == pytest.approx(2.5, abs=1e-6)


def test_cross_both_directions() -> None:
    """cross() with direction='cross' should find both rising and falling."""
    series = pd.Series([-1, 1, -1, 1], index=[0, 1, 2, 3])
    result = cross(series, threshold=0, direction="cross")
    assert len(result) == 3  # rising, falling, rising


def test_cross_only_index() -> None:
    """cross() with only_index=True should return integer indices."""
    series = pd.Series([-1, 1, 2], index=[10, 20, 30])
    result = cross(series, threshold=0, direction="rising", only_index=True)
    assert len(result) == 1
    assert result[0] == 0  # 0-based index


def test_cross_first_point_only() -> None:
    """cross() with first_point_only=True should return only the first crossing."""
    series = pd.Series([-1, 1, -1, 1], index=[0, 1, 2, 3])
    result = cross(series, threshold=0, direction="cross", first_point_only=True)
    # Should return a single value, not an array of 3
    assert np.isscalar(result) or len(result) == 1


def test_cross_with_nan() -> None:
    """cross() should handle NaN values by dropping them first."""
    series = pd.Series([-1, np.nan, 1, 2], index=[0, 1, 2, 3])
    result = cross(series, threshold=0, direction="rising")
    assert len(result) == 1


def test_f_crossing_with_batch_data(batch_data: pd.DataFrame) -> None:
    """f_crossing should find threshold crossings per batch."""
    df = batch_data
    result = features.f_crossing(
        df,
        tag="Temp1",
        time_tag="UCI_minutes",
        threshold=-25,
        direction="rising",
        batch_col="Batch",
    )
    assert result.shape[1] == 1  # one feature column
    assert "Temp1" in result.columns[0]


def test_f_elbow_returns_x_value(batch_data: pd.DataFrame) -> None:
    """f_elbow should return the x-axis value of the elbow per batch."""
    result = features.f_elbow(
        batch_data,
        x_axis_tag="UCI_minutes",
        tags=["Temp1"],
        batch_col="Batch",
    )
    assert result.shape[1] == 1
    assert "elbow" in str(result.columns[0])


def test_f_elbow_only_index(batch_data: pd.DataFrame) -> None:
    """f_elbow with only_index=True should return the elbow index."""
    result = features.f_elbow(
        batch_data,
        x_axis_tag="UCI_minutes",
        tags=["Temp1"],
        batch_col="Batch",
        only_index=True,
    )
    assert result.shape[1] == 1


def test_f_elbow_no_elbow_records_nan() -> None:
    """A straight line has no elbow; f_elbow should record np.nan, not a function."""
    n = 30
    straight_line = pd.DataFrame(
        {
            "Batch": ["B1"] * n,
            "time": np.arange(n, dtype=float),
            "signal": np.arange(n, dtype=float),
        }
    )
    result = features.f_elbow(
        straight_line,
        x_axis_tag="time",
        tags=["signal"],
        batch_col="Batch",
    )
    value = result.iloc[0, 0]
    assert isinstance(value, float)
    assert np.isnan(value)


class TestFRupture:
    """Change-point detection via `ruptures` (#198)."""

    @staticmethod
    def _stepped(steps: dict[str, int], n_samples: int = 100) -> pd.DataFrame:
        """One batch per entry, each with a single step at the given position."""
        rng = np.random.default_rng(0)
        frames = [
            pd.DataFrame(
                {
                    "batch_id": batch_id,
                    "Temperature": np.concatenate([rng.normal(0, 0.1, step), rng.normal(5, 0.1, n_samples - step)]),
                    "Flat": 7.0,
                }
            )
            for batch_id, step in steps.items()
        ]
        return pd.concat(frames, ignore_index=True)

    def test_it_finds_a_known_step_exactly(self) -> None:
        steps = {"A": 40, "B": 70, "C": 25}
        found = features.f_rupture(
            self._stepped(steps), tags=["Temperature"], batch_col="batch_id", settings={"jump": 1}
        ).droplevel(-1)

        assert list(found.columns) == ["Temperature_rupture"]
        for batch_id, step in steps.items():
            assert found.loc[batch_id, "Temperature_rupture"] == (step,)

    def test_a_constant_tag_has_no_change_points(self) -> None:
        found = features.f_rupture(self._stepped({"A": 40}), tags=["Flat"], batch_col="batch_id")

        assert found.iloc[0, 0] == ()

    def test_the_trailing_sentinel_is_removed(self) -> None:
        """`ruptures` ends its result with the signal length, which is not a change point."""
        found = features.f_rupture(
            self._stepped({"A": 40}), tags=["Temperature"], batch_col="batch_id", settings={"jump": 1}
        )

        assert 100 not in found.iloc[0, 0]

    def test_counts_give_a_numeric_feature(self) -> None:
        """The documented way to get a column that fits in a model matrix."""
        found = features.f_rupture(
            self._stepped({"A": 40, "B": 70}), tags=["Temperature", "Flat"], batch_col="batch_id"
        ).droplevel(-1)

        counts = found.map(len)
        assert counts.loc["A", "Temperature_rupture"] == 1
        assert counts.loc["A", "Flat_rupture"] == 0

    def test_a_larger_penalty_returns_no_more_change_points(self) -> None:
        data = self._stepped({"A": 40})
        found = [
            len(
                features.f_rupture(
                    data, tags=["Temperature"], batch_col="batch_id", settings={"penalty": penalty, "jump": 1}
                ).iloc[0, 0]
            )
            for penalty in (1.0, 10.0, 1000.0)
        ]

        assert found == sorted(found, reverse=True), f"expected monotone, got {found}"

    def test_the_default_penalty_adapts_to_the_signal_length(self) -> None:
        """`None` means log(n); a fixed 100.0 finds nothing under the bounded rbf cost."""
        data = self._stepped({"A": 40}, n_samples=100)

        adaptive = features.f_rupture(data, tags=["Temperature"], batch_col="batch_id", settings={"jump": 1})
        fixed = features.f_rupture(
            data, tags=["Temperature"], batch_col="batch_id", settings={"penalty": 100.0, "jump": 1}
        )

        assert adaptive.iloc[0, 0] == (40,)
        assert fixed.iloc[0, 0] == ()

    def test_a_signal_too_short_to_split_returns_empty(self) -> None:
        tiny = pd.DataFrame({"batch_id": ["A", "A", "A"], "T": [1.0, 2.0, 3.0]})

        assert features.f_rupture(tiny, tags=["T"], batch_col="batch_id").iloc[0, 0] == ()

    def test_missing_data_returns_empty_rather_than_raising(self) -> None:
        signal = np.concatenate([np.zeros(50), [np.nan], np.ones(49) * 5])
        data = pd.DataFrame({"batch_id": "A", "T": signal})

        assert features.f_rupture(data, tags=["T"], batch_col="batch_id").iloc[0, 0] == ()

    def test_an_unrecognized_setting_is_rejected(self) -> None:
        """`pen` is the ruptures spelling; this function takes `penalty`."""
        with pytest.raises(ValueError, match=r"unrecognized settings \['pen'\]"):
            features.f_rupture(
                self._stepped({"A": 40}), tags=["Temperature"], batch_col="batch_id", settings={"pen": 1}
            )

    @pytest.mark.dataset
    def test_it_runs_on_the_dryer_data(self) -> None:
        folder = pathlib.Path(__file__).parents[2] / "src" / "process_improve" / "datasets" / "batch"
        dryer = pd.read_csv(folder / "dryer.csv")
        subset = dryer[dryer["batch_id"].isin(dryer["batch_id"].unique()[:3])]

        found = features.f_rupture(subset, tags=["DryerTemp"], batch_col="batch_id").droplevel(-1)

        assert len(found) == 3
        # Every batch of this dryer run has structure to find, and the positions are
        # inside the batch rather than at its ends.
        for breaks in found["DryerTemp_rupture"]:
            assert len(breaks) > 0
            assert all(0 < position < len(subset) for position in breaks)
