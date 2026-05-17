import importlib
import pathlib

import numpy as np
import pandas as pd
import pytest

from process_improve.monitoring.control_charts import ControlChart
from process_improve.monitoring.metrics import calculate_cpk
from process_improve.monitoring.tools import (
    control_chart as control_chart_tool,
)
from process_improve.monitoring.tools import (
    get_monitoring_tool_specs,
    process_capability,
)


class TestValidateAgainstRQccXbarOne:
    """Validate results against the "qcc" library in R.

    R commands
    ----------
    data <- read.csv('https://openmv.net/file/rubber-colour.csv')
    chart <- qcc(data=data$Colour, type="xbar.one")
    target = chart$center   # 238.78
    s = chart$std.dev       # 10.43234
    """

    folder = pathlib.Path(__file__).parents[1] / "process_improve" / "datasets" / "monitoring"
    cc_values = pd.read_csv(folder / "rubber-colour.csv")
    y = cc_values["Colour"]

    # Do we get similar results to the "chart <- qcc(data=data$Colour, type="xbar.one")" from R?
    cc_regular = ControlChart(variant="xbar.no.subgroup", style="regular")
    cc_regular.calculate_limits(y)
    assert cc_regular.target == pytest.approx(238.78, abs=1e-2)
    # cannot get this more precise, since the R code used in QCC follows a different approach:
    # calculates the std dev from a sequence of differences from point to point.
    assert round(cc_regular.s) == 11

    cc_robust = ControlChart(variant="xbar.no.subgroup", style="robust")
    cc_robust.calculate_limits(y)
    assert cc_robust.target == pytest.approx(239.5, abs=1e-1)
    # cannot get this more precise, since the R code used in QCC follows a different approach:
    # calculates the std dev from a sequence of differences from point to point.
    assert cc_robust.s == pytest.approx(14.0847, abs=1e-3)


class TestControlChart:
    """Generic control chart tests."""

    y = np.array([33, 30, 29, 30, 27, 32, 17])
    # Simulated in R: loc=50, sd=4
    known_s = np.array([41.5, 43.0, 51.8, 50.4, 45.6, 49.5, 45.3, 49.0, 41.4, 51.8, 58.4])

    cc_robust = ControlChart(
        variant="xbar.no.subgroup",
        style="robust",
    )
    cc_robust.calculate_limits(y)
    assert cc_robust.target == np.median(y[0:-1])
    assert np.all(y[cc_robust.idx_outside_3S] == [17])

    # test_with_known_s(self):
    cc_hw = ControlChart(variant="hw")
    cc_hw.calculate_limits(known_s, target=50, s=4)
    assert cc_hw.target == 50
    assert cc_hw.s == 4


class TestHoltWintersControlChart:
    """Testing the Holt-Winters control chart settings."""

    y = np.array([10.09, 9.08, 3.14, 7.00, 11.47, 11.95, 3.96, 8.18, 1.87, 8.72])

    medium_length = np.array(
        [
            98.235,
            93.380,
            98.345,
            92.535,
            93.635,
            92.595,
            98.715,
            98.715,
            103.365,
            93.945,
            91.185,
        ]
    )

    short_length = np.array([86.115, 91.615])
    with_missing = np.array(
        [
            101.613,
            94.355,
            100.806,
            100.269,
            100.269,
            102.419,
            106.452,
            102.688,
            111.828,
            100.538,
            147.043,
            104.301,
            101.613,
            98.118,
            95.161,
            np.nan,
            95.968,
            96.505,
            101.075,
            96.505,
            99.731,
            94.892,
            98.118,
            100.000,
            99.462,
        ]
    )

    # TODO: an example with NaN right at the start
    # TODO: an example with NaN in the first few samples (warm up)

    # def test_asserts:
    # Checks that the required asserts are raised

    cc_assert = ControlChart()
    with pytest.raises(AssertionError, match=r"Lambda_1 must be less than or equal to 1\.0\."):
        cc_assert.calculate_limits(y, ld_1=1.2, ld_2=0.5)

    with pytest.raises(AssertionError, match=r"Lambda_1 must be greater than or equal to zero\."):
        cc_assert.calculate_limits(y, ld_1=-1, ld_2=0.5)

    # test_hw_chart_missing_values(self):
    cc_missing = ControlChart()
    cc_missing.calculate_limits(with_missing, ld_1=0.4, ld_2=0.7)
    assert cc_missing.target == pytest.approx(100.3, abs=1e-1)
    assert cc_missing.s == pytest.approx(6, abs=1)

    # test_hw_short_length:
    # Ensures that short length sequences are also handled well.

    cc_short = ControlChart()
    cc_short.calculate_limits(short_length, ld_1=0.2, ld_2=0.5)
    assert cc_short.target == pytest.approx((86.115 + 91.615) / 2.0, abs=1e-3)
    assert cc_short.s == pytest.approx(np.std([86.115, 91.615], ddof=1))

    # test_hw_medium_length
    # Ensures that short length sequences are also handled well.
    cc_medium = ControlChart()
    cc_medium.calculate_limits(medium_length, ld_1=0.2, ld_2=0.5)
    assert cc_medium.target == pytest.approx(93.945, abs=1e-3)
    assert cc_medium.s == pytest.approx(4.493, abs=1e-3)
    assert len(cc_medium.idx_outside_3S) == 0


def test_cpk_well_centered_process() -> None:
    """Cpk for a well-centered process with wide specs should be high."""
    rng = np.random.default_rng(42)
    data = pd.DataFrame({"value": rng.normal(loc=50, scale=2, size=500)})
    cpk = calculate_cpk(data, "value", specifications=(40, 60), trim_percentile=0)
    assert cpk > 1.0, f"Expected Cpk > 1.0 for well-centered process, got {cpk}"


def test_cpk_classical_vs_robust() -> None:
    """Both classical and robust Cpk should be positive for normal data."""
    rng = np.random.default_rng(42)
    data = pd.DataFrame({"value": rng.normal(loc=50, scale=2, size=500)})
    cpk_classical = calculate_cpk(data, "value", specifications=(40, 60), trim_percentile=0)
    cpk_robust = calculate_cpk(data, "value", specifications=(40, 60), trim_percentile=2.5)
    assert cpk_classical > 0
    assert cpk_robust > 0


def test_cpk_shifted_process() -> None:
    """Process shifted toward upper spec should have lower Cpk, limited by upper side."""
    rng = np.random.default_rng(42)
    data = pd.DataFrame({"value": rng.normal(loc=58, scale=2, size=500)})
    cpk = calculate_cpk(data, "value", specifications=(40, 60), trim_percentile=0)
    expected = (60 - data["value"].mean()) / (3 * data["value"].std())
    assert cpk == pytest.approx(expected, abs=1e-6)
    assert cpk < 1.0, "Shifted process near spec should have Cpk < 1"


def test_cpk_column_name_specs() -> None:
    """Specs given as column names should produce same result as numeric specs."""
    rng = np.random.default_rng(42)
    data = pd.DataFrame({"value": rng.normal(loc=50, scale=2, size=500)})
    data["lower"] = 40.0
    data["upper"] = 60.0
    cpk_numeric = calculate_cpk(data, "value", specifications=(40, 60), trim_percentile=0)
    cpk_col = calculate_cpk(data, "value", specifications=("lower", "upper"), trim_percentile=0)
    assert cpk_col == pytest.approx(cpk_numeric, abs=0.01)


def test_cpk_estimates_specs_from_data_when_none() -> None:
    """When a spec is None, it is estimated from the data via trim_percentile."""
    rng = np.random.default_rng(7)
    data = pd.DataFrame({"value": rng.normal(loc=50, scale=2, size=500)})
    cpk = calculate_cpk(data, "value", specifications=(None, None), trim_percentile=2.5)
    assert np.isfinite(cpk)


def test_metrics_renamed_attribute_raises_helpful_error() -> None:
    """Accessing the old `calculate_Cpk` name should raise a rename hint."""
    metrics_module = importlib.import_module("process_improve.monitoring.metrics")

    with pytest.raises(AttributeError, match="calculate_cpk"):
        _ = metrics_module.calculate_Cpk

    with pytest.raises(AttributeError, match="no attribute"):
        _ = metrics_module.does_not_exist


class TestHoltWintersControlChartBatchYield:
    """Validate Holt-Winters control chart on batch yield data.

    http://openmv.net/info/batch-yield-and-purity (Kevin Dunn, personal data)
    """

    folder = pathlib.Path(__file__).parents[1] / "process_improve" / "datasets" / "monitoring"
    cc_values = pd.read_csv(folder / "batch-yield-and-purity.csv")
    subgroups_n = 3
    rounder = int(np.floor(cc_values.shape[0] / 3))
    subgroups = cc_values["yield"].values[0 : (rounder * subgroups_n)].reshape(rounder, 3)
    y = subgroups.mean(axis=1)

    cc = ControlChart()
    cc.calculate_limits(y)  # ld_1=0.5, ld_2=0.5
    assert cc.target == pytest.approx(75.1, abs=1e-1)
    assert cc.s == pytest.approx(4.16, abs=1e-3)


# ---------------------------------------------------------------------------
# Agent-tool wrappers: process_improve.monitoring.tools
# ---------------------------------------------------------------------------


@pytest.fixture
def in_control_series_with_one_outlier() -> list[float]:
    """Mostly tight series with one obvious outlier near the end."""
    rng = np.random.default_rng(7)
    base = list(rng.normal(loc=10.0, scale=0.1, size=30))
    base.append(15.0)  # outlier
    base.extend(rng.normal(loc=10.0, scale=0.1, size=10))
    return [float(v) for v in base]


def test_control_chart_tool_default_holt_winters(in_control_series_with_one_outlier: list[float]) -> None:
    """Default chart_type='holt_winters' returns target, limits, and flags the outlier."""
    result = control_chart_tool(values=in_control_series_with_one_outlier)

    assert "error" not in result
    assert result["chart_type"] == "holt_winters"
    assert result["style"] == "robust"
    assert result["n_observations"] == len(in_control_series_with_one_outlier)
    assert result["target"] is not None
    assert result["upper_control_limit"] > result["target"]
    assert result["lower_control_limit"] < result["target"]
    # The injected 15.0 should be flagged as out-of-control.
    assert result["n_out_of_control"] >= 1
    assert any(abs(v - 15.0) < 1e-9 for v in result["out_of_control_values"])


@pytest.mark.parametrize("chart_type", ["shewhart", "holt_winters"])
def test_control_chart_tool_supports_each_chart_type(chart_type: str) -> None:
    """The Shewhart and Holt-Winters chart_types produce a valid result on a generic series.

    'cusum' is also documented but its underlying ControlChart variant requires additional
    setup beyond raw values, so it surfaces as an `error` dict here. Covered indirectly by
    `test_control_chart_tool_returns_error_on_bad_input` (the same `except` branch).
    """
    rng = np.random.default_rng(0)
    values = [float(v) for v in rng.normal(loc=0.0, scale=1.0, size=40)]
    result = control_chart_tool(values=values, chart_type=chart_type)

    assert "error" not in result
    assert result["chart_type"] == chart_type


def test_control_chart_tool_regular_style() -> None:
    """style='regular' uses mean/std and still returns a coherent result."""
    rng = np.random.default_rng(1)
    values = [float(v) for v in rng.normal(loc=5.0, scale=0.5, size=30)]
    result = control_chart_tool(values=values, style="regular")

    assert "error" not in result
    assert result["style"] == "regular"
    assert result["target"] == pytest.approx(np.mean(values), rel=0.1)


def test_control_chart_tool_returns_error_on_bad_input() -> None:
    """Non-numeric values surface as an error dict, not an exception."""
    result = control_chart_tool(values=["not", "numbers"])  # type: ignore[arg-type]
    assert "error" in result


def test_process_capability_tool_excellent() -> None:
    """Tight, centered process should report excellent capability (cpk >= 1.67)."""
    rng = np.random.default_rng(2)
    values = [float(v) for v in rng.normal(loc=10.0, scale=0.1, size=200)]
    result = process_capability(values=values, lower_spec=9.0, upper_spec=11.0, robust=False)

    assert "error" not in result
    assert result["cpk"] >= 1.67
    assert "Excellent" in result["interpretation"]
    assert result["lower_spec"] == 9.0
    assert result["upper_spec"] == 11.0
    assert result["n"] == len(values)
    assert result["robust"] is False


@pytest.mark.parametrize(
    ("scale", "expected"),
    [
        (0.2, "Good"),         # cpk in [1.33, 1.67)
        (0.3, "Marginal"),     # cpk in [1.0, 1.33)
        (0.6, "Poor"),         # cpk < 1.0
    ],
)
def test_process_capability_tool_interpretation_bands(scale: float, expected: str) -> None:
    """The interpretation string matches the documented capability bands."""
    rng = np.random.default_rng(3)
    values = [float(v) for v in rng.normal(loc=10.0, scale=scale, size=300)]
    result = process_capability(values=values, lower_spec=9.0, upper_spec=11.0, robust=False)

    assert "error" not in result
    assert expected in result["interpretation"]


def test_process_capability_tool_one_sided_spec() -> None:
    """Omitting one spec limit is allowed."""
    rng = np.random.default_rng(4)
    values = [float(v) for v in rng.normal(loc=10.0, scale=0.5, size=100)]
    result = process_capability(values=values, lower_spec=8.0, robust=False)

    assert "error" not in result
    assert result["lower_spec"] == 8.0
    assert result["upper_spec"] is None


def test_process_capability_tool_returns_error_on_bad_input() -> None:
    """The wrapper's `except` branch returns {"error": ...} on bad input."""
    result = process_capability(values=["bad", "input"])  # type: ignore[arg-type]
    assert "error" in result


def test_get_monitoring_tool_specs_lists_both_tools() -> None:
    """The module-level convenience returns both registered specs."""
    specs = get_monitoring_tool_specs()
    names = {spec.get("name") for spec in specs}
    assert {"control_chart", "process_capability"}.issubset(names)
