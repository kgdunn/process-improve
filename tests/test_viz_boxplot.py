"""Tests for the generic boxplot visualization tool."""

from __future__ import annotations

import json

import numpy as np
import pytest

from process_improve.tool_spec import execute_tool_call, get_tool_specs
from process_improve.visualization.charts.boxplot import BoxPlot, BoxStats
from process_improve.visualization.tools import boxplot

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def grouped_data() -> list[dict]:
    """Two groups A (with outlier 50) and B (no outliers)."""
    return [
        {"row_id": f"obs_{i}", "lot": "A", "y": v}
        for i, v in enumerate([10, 11, 12, 13, 14, 15, 50])
    ] + [
        {"row_id": f"obs_{i + 100}", "lot": "B", "y": v}
        for i, v in enumerate([5, 6, 7, 8, 9])
    ]


# ---------------------------------------------------------------------------
# Tool-registration
# ---------------------------------------------------------------------------


def test_boxplot_registered() -> None:
    """The boxplot tool is discovered by the global registry."""
    names = {s["name"] for s in get_tool_specs()}
    assert "boxplot" in names


def test_boxplot_dispatchable_via_execute_tool_call(grouped_data: list[dict]) -> None:
    """execute_tool_call routes to the boxplot function."""
    result = execute_tool_call(
        "boxplot",
        {"data": grouped_data, "value_columns": ["y"], "group_by": "lot"},
    )
    assert result["plot_type"] == "boxplot"


# ---------------------------------------------------------------------------
# Return shape
# ---------------------------------------------------------------------------


def test_return_shape(grouped_data: list[dict]) -> None:
    res = boxplot(
        data=grouped_data,
        value_columns=["y"],
        group_by="lot",
        id_column="row_id",
        link_group="L1",
    )
    assert set(res.keys()) == {
        "plot_type", "title", "data", "plotly", "echarts",
        "link_group", "point_ids",
    }
    assert res["plot_type"] == "boxplot"
    assert res["link_group"] == "L1"
    assert isinstance(res["data"]["quartiles"], list)
    assert isinstance(res["data"]["outliers"], list)


def test_output_json_serialisable(grouped_data: list[dict]) -> None:
    res = boxplot(data=grouped_data, value_columns=["y"], group_by="lot")
    json.dumps(res)  # raises if any non-JSON-safe objects slipped through


# ---------------------------------------------------------------------------
# Quartile math
# ---------------------------------------------------------------------------


def test_quartiles_match_numpy_percentile() -> None:
    """Quartiles must agree with numpy.percentile to machine epsilon."""
    values = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
    data = [{"y": v} for v in values]
    res = boxplot(data=data, value_columns=["y"])
    qs = res["data"]["quartiles"][0]["q_stats"]
    expected_q1, expected_med, expected_q3 = np.percentile(values, [25, 50, 75])
    assert qs[1] == pytest.approx(float(expected_q1))
    assert qs[2] == pytest.approx(float(expected_med))
    assert qs[3] == pytest.approx(float(expected_q3))


def test_outliers_flagged_beyond_1p5_iqr() -> None:
    """Values beyond 1.5*IQR are flagged; inliers are not."""
    values = [10, 11, 12, 13, 14, 15, 50]  # 50 is an outlier
    data = [{"row_id": f"r{i}", "y": v} for i, v in enumerate(values)]
    res = boxplot(data=data, value_columns=["y"], id_column="row_id")
    outliers = res["data"]["outliers"]
    assert len(outliers) == 1
    assert outliers[0]["value"] == 50.0
    assert outliers[0]["id"] == "r6"
    # Whiskers fall on the inlier extrema.
    lower, _q1, _med, _q3, upper = res["data"]["quartiles"][0]["q_stats"]
    assert lower == 10.0
    assert upper == 15.0


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


def test_grouped_emits_one_box_per_group(grouped_data: list[dict]) -> None:
    res = boxplot(data=grouped_data, value_columns=["y"], group_by="lot")
    groups = [q["group"] for q in res["data"]["quartiles"]]
    assert groups == ["A", "B"]  # insertion order preserved


def test_ungrouped_emits_one_box_per_column() -> None:
    data = [
        {"x": 1, "z": 10},
        {"x": 2, "z": 20},
        {"x": 3, "z": 30},
    ]
    res = boxplot(data=data, value_columns=["x", "z"])
    groups = [q["group"] for q in res["data"]["quartiles"]]
    assert groups == ["x", "z"]


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


def test_backend_echarts_omits_plotly(grouped_data: list[dict]) -> None:
    res = boxplot(
        data=grouped_data, value_columns=["y"], group_by="lot", backend="echarts",
    )
    assert res["plotly"] is None
    assert res["echarts"] is not None


def test_backend_plotly_omits_echarts(grouped_data: list[dict]) -> None:
    res = boxplot(
        data=grouped_data, value_columns=["y"], group_by="lot", backend="plotly",
    )
    assert res["echarts"] is None
    assert res["plotly"] is not None


# ---------------------------------------------------------------------------
# Rendered ECharts / Plotly shape
# ---------------------------------------------------------------------------


def test_echarts_has_boxplot_and_scatter_series(grouped_data: list[dict]) -> None:
    res = boxplot(
        data=grouped_data, value_columns=["y"], group_by="lot",
        id_column="row_id", show_points=True,
    )
    types = [s["type"] for s in res["echarts"]["series"]]
    assert "boxplot" in types
    assert "scatter" in types  # outlier overlay
    box_series = next(s for s in res["echarts"]["series"] if s["type"] == "boxplot")
    # ECharts boxplot data is the raw five-number summary per category.
    assert box_series["data"] == [[10.0, 11.5, 13.0, 14.5, 15.0], [5.0, 6.0, 7.0, 8.0, 9.0]]


def test_echarts_x_axis_uses_group_categories(grouped_data: list[dict]) -> None:
    res = boxplot(data=grouped_data, value_columns=["y"], group_by="lot")
    x_axis = res["echarts"]["xAxis"]
    assert x_axis["type"] == "category"
    assert x_axis["data"] == ["A", "B"]


def test_plotly_is_box_trace(grouped_data: list[dict]) -> None:
    res = boxplot(data=grouped_data, value_columns=["y"], group_by="lot")
    assert res["plotly"]["data"][0]["type"] == "box"


# ---------------------------------------------------------------------------
# Linking metadata
# ---------------------------------------------------------------------------


def test_link_group_injects_brush_and_marker(grouped_data: list[dict]) -> None:
    res = boxplot(
        data=grouped_data, value_columns=["y"], group_by="lot",
        id_column="row_id", link_group="lots",
    )
    option = res["echarts"]
    assert option["__link_group"] == "lots"
    assert "brush" in option
    assert "brush" in option["toolbox"]["feature"]


def test_link_group_point_ids_carry_outlier_id(grouped_data: list[dict]) -> None:
    res = boxplot(
        data=grouped_data, value_columns=["y"], group_by="lot",
        id_column="row_id", link_group="lots",
    )
    assert res["point_ids"] == ["obs_6"]


def test_no_link_group_no_brush(grouped_data: list[dict]) -> None:
    res = boxplot(data=grouped_data, value_columns=["y"], group_by="lot")
    option = res["echarts"]
    assert "__link_group" not in option
    assert "brush" not in option
    assert res["point_ids"] == []


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_empty_data_returns_error() -> None:
    assert "error" in boxplot(data=[], value_columns=["y"])


def test_missing_value_column_returns_error() -> None:
    res = boxplot(data=[{"x": 1}], value_columns=["nope"])
    assert "error" in res
    assert "nope" in res["error"]


def test_missing_group_by_returns_error() -> None:
    res = boxplot(data=[{"y": 1}], value_columns=["y"], group_by="missing")
    assert "error" in res
    assert "missing" in res["error"]


# ---------------------------------------------------------------------------
# Chart class direct usage
# ---------------------------------------------------------------------------


def test_chart_class_round_trip() -> None:
    """BoxPlot is usable directly from notebooks without the tool wrapper."""
    chart = BoxPlot(
        boxes=[
            BoxStats(group="A", q_stats=[1, 2, 3, 4, 5]),
            BoxStats(group="B", q_stats=[2, 3, 4, 5, 6]),
        ],
        title="demo",
        y_title="y",
        x_title="group",
    )
    spec = chart.to_spec()
    assert spec.plot_type == "boxplot"
    assert len(spec.panels) == 1
    # Rendering works without raising.
    assert chart.to_echarts()["series"][0]["type"] == "boxplot"
    assert chart.to_plotly()["data"][0]["type"] == "box"
