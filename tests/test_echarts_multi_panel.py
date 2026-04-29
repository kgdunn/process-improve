"""Tests for the multi-panel rendering path in EChartsAdapter.

The ``_multi_panel`` method (visualization/adapters/echarts_adapter.py
lines 138-188) and the constraint-region annotation arm (lines 432-447)
were previously uncovered: every existing test built a single-panel
ChartSpec and never used a constraint region.
"""

from __future__ import annotations

from process_improve.visualization.adapters.echarts_adapter import EChartsAdapter
from process_improve.visualization.spec import (
    Annotation,
    ChartSpec,
    Encoding,
    LayerSpec,
    PanelSpec,
)
from process_improve.visualization.types import AnnotationType, MarkType


def _scatter_panel(title: str = "panel") -> PanelSpec:
    """Build a small single-layer scatter panel."""
    layer = LayerSpec(
        mark=MarkType.scatter,
        data=[{"x": i, "y": 2 * i + 1} for i in range(5)],
        x=Encoding(field="x", title="X"),
        y=Encoding(field="y", title="Y"),
        name="trace",
    )
    return PanelSpec(layers=[layer], title=title, x_title="X", y_title="Y")


# ---------------------------------------------------------------------------
# Multi-panel layout
# ---------------------------------------------------------------------------


def test_two_panel_grid_renders() -> None:
    """A 2-panel ChartSpec should fan out into two grids / xAxes / yAxes."""
    spec = ChartSpec(
        panels=[_scatter_panel("p1"), _scatter_panel("p2")],
        title="Two panels",
        layout="grid",
        columns=2,
    )
    option = EChartsAdapter().render(spec)

    assert option["title"]["text"] == "Two panels"
    # Multi-panel uses lists of grids / axes.
    assert isinstance(option["grid"], list)
    assert len(option["grid"]) == 2
    assert isinstance(option["xAxis"], list)
    assert len(option["xAxis"]) == 2
    assert isinstance(option["yAxis"], list)
    assert len(option["yAxis"]) == 2
    # Each series should be tagged with the right xAxisIndex / yAxisIndex.
    for idx, series in enumerate(option["series"]):
        assert series["xAxisIndex"] == idx
        assert series["yAxisIndex"] == idx


def test_three_panel_grid_two_columns() -> None:
    """A 3-panel grid with 2 columns should still render all 3 panels."""
    spec = ChartSpec(
        panels=[_scatter_panel(f"p{i}") for i in range(3)],
        title="Three panels",
        layout="grid",
        columns=2,
    )
    option = EChartsAdapter().render(spec)
    assert len(option["grid"]) == 3
    assert len(option["series"]) == 3


def test_multi_panel_with_annotations() -> None:
    """Multi-panel rendering should also place annotations on each panel."""
    panel = _scatter_panel("p")
    panel.annotations = [
        Annotation(
            annotation_type=AnnotationType.reference_line,
            value=1.5,
            axis="y",
            label="threshold",
        ),
    ]
    spec = ChartSpec(
        panels=[panel, _scatter_panel("p2")],
        title="With annotations",
    )
    option = EChartsAdapter().render(spec)
    # The first panel's series should carry the markLine.
    assert "markLine" in option["series"][0]


# ---------------------------------------------------------------------------
# Constraint-region annotation
# ---------------------------------------------------------------------------


def test_constraint_region_x_only() -> None:
    """A constraint_region with only x bounds should add an xAxis markArea."""
    panel = _scatter_panel("p")
    panel.annotations = [
        Annotation(
            annotation_type=AnnotationType.constraint_region,
            value=None,
            label="forbidden-x",
            style={"x_min": 1.0, "x_max": 3.0},
        ),
    ]
    spec = ChartSpec(panels=[panel], title="Constraint X")
    option = EChartsAdapter().render(spec)

    series = option["series"][0]
    assert "markArea" in series
    data = series["markArea"]["data"]
    assert data[0][0]["xAxis"] == 1.0
    assert data[0][1]["xAxis"] == 3.0


def test_constraint_region_y_only() -> None:
    """A constraint_region with only y bounds should add a yAxis markArea."""
    panel = _scatter_panel("p")
    panel.annotations = [
        Annotation(
            annotation_type=AnnotationType.constraint_region,
            value=None,
            label="forbidden-y",
            style={"y_min": 0.0, "y_max": 2.5},
        ),
    ]
    spec = ChartSpec(panels=[panel], title="Constraint Y")
    option = EChartsAdapter().render(spec)
    series = option["series"][0]
    data = series["markArea"]["data"]
    assert data[0][0]["yAxis"] == 0.0
    assert data[0][1]["yAxis"] == 2.5


def test_constraint_region_both_axes() -> None:
    """A constraint_region with both x and y bounds should add two markAreas."""
    panel = _scatter_panel("p")
    panel.annotations = [
        Annotation(
            annotation_type=AnnotationType.constraint_region,
            value=None,
            label="forbidden-xy",
            style={"x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1},
        ),
    ]
    spec = ChartSpec(panels=[panel], title="Constraint XY")
    option = EChartsAdapter().render(spec)
    series = option["series"][0]
    assert len(series["markArea"]["data"]) == 2


# ---------------------------------------------------------------------------
# Reference band annotations
# ---------------------------------------------------------------------------


def test_reference_band_y() -> None:
    """A reference_band annotation should populate a yAxis markArea."""
    panel = _scatter_panel("p")
    panel.annotations = [
        Annotation(
            annotation_type=AnnotationType.reference_band,
            value=0.0,
            value_end=1.0,
            axis="y",
            label="band",
        ),
    ]
    option = EChartsAdapter().render(ChartSpec(panels=[panel]))
    series = option["series"][0]
    assert series["markArea"]["data"][0][0]["yAxis"] == 0.0
    assert series["markArea"]["data"][0][1]["yAxis"] == 1.0


def test_reference_band_x() -> None:
    """A reference_band annotation on the x-axis should populate xAxis markArea."""
    panel = _scatter_panel("p")
    panel.annotations = [
        Annotation(
            annotation_type=AnnotationType.reference_band,
            value=0.0,
            value_end=2.0,
            axis="x",
            label="x-band",
        ),
    ]
    option = EChartsAdapter().render(ChartSpec(panels=[panel]))
    series = option["series"][0]
    assert series["markArea"]["data"][0][0]["xAxis"] == 0.0


# ---------------------------------------------------------------------------
# Empty / fallback rendering
# ---------------------------------------------------------------------------


def test_empty_chart_spec() -> None:
    """A ChartSpec with zero panels should still produce a valid option dict."""
    option = EChartsAdapter().render(ChartSpec(panels=[], title="Empty"))
    assert option["title"]["text"] == "Empty"
    assert option["series"] == []


def test_render_panel_directly() -> None:
    """``render_panel`` should produce the same option as a single-panel spec."""
    panel = _scatter_panel("p")
    option = EChartsAdapter().render_panel(panel)
    assert option["title"]["text"] == "p"
    assert len(option["series"]) == 1


def test_link_group_injects_brush() -> None:
    """A link_group on the spec should inject brush + __link_group keys."""
    spec = ChartSpec(panels=[_scatter_panel("p")], link_group="grp1")
    option = EChartsAdapter().render(spec)
    assert option["__link_group"] == "grp1"
    assert "brush" in option


# ---------------------------------------------------------------------------
# Bar / scatter / heatmap / wireframe series builders
# ---------------------------------------------------------------------------


def test_bar_series_with_layer_color() -> None:
    """A bar layer with a single ``color`` should set itemStyle.color."""
    layer = LayerSpec(
        mark=MarkType.bar,
        data=[{"x": "A", "y": 1.0}, {"x": "B", "y": 2.0}],
        x=Encoding(field="x", title="X"),
        y=Encoding(field="y", title="Y"),
        color="#ff0000",
    )
    panel = PanelSpec(layers=[layer])
    option = EChartsAdapter().render(ChartSpec(panels=[panel]))
    assert option["series"][0]["itemStyle"] == {"color": "#ff0000"}


def test_bar_series_with_error_bars() -> None:
    """A bar layer with ``style['error_y']`` should add a markLine."""
    layer = LayerSpec(
        mark=MarkType.bar,
        data=[{"x": "A", "y": 10.0}, {"x": "B", "y": 20.0}],
        x=Encoding(field="x", title="X"),
        y=Encoding(field="y", title="Y"),
        style={"error_y": [1.0, 2.0]},
    )
    panel = PanelSpec(layers=[layer])
    option = EChartsAdapter().render(ChartSpec(panels=[panel]))
    series = option["series"][0]
    assert "markLine" in series
    assert len(series["markLine"]["data"]) == 2


def test_bar_series_skips_zero_and_none_errors() -> None:
    """Per-point ``error_y`` entries that are None or zero should be skipped."""
    layer = LayerSpec(
        mark=MarkType.bar,
        data=[{"x": "A", "y": 1.0}, {"x": "B", "y": 2.0}, {"x": "C", "y": 3.0}],
        x=Encoding(field="x", title="X"),
        y=Encoding(field="y", title="Y"),
        style={"error_y": [None, 0.0, 0.5]},
    )
    panel = PanelSpec(layers=[layer])
    option = EChartsAdapter().render(ChartSpec(panels=[panel]))
    series = option["series"][0]
    # Only the third bar (err=0.5) should appear in markLine data.
    assert len(series["markLine"]["data"]) == 1


def test_scatter_series_with_per_point_colors() -> None:
    """A scatter layer with ``style['colors']`` should embed per-point itemStyle."""
    layer = LayerSpec(
        mark=MarkType.scatter,
        data=[{"x": 1, "y": 2}, {"x": 3, "y": 4}],
        x=Encoding(field="x", title="X"),
        y=Encoding(field="y", title="Y"),
        style={"colors": ["#ff0000", "#00ff00"]},
    )
    panel = PanelSpec(layers=[layer])
    option = EChartsAdapter().render(ChartSpec(panels=[panel]))
    points = option["series"][0]["data"]
    assert points[0]["itemStyle"]["color"] == "#ff0000"
    assert points[1]["itemStyle"]["color"] == "#00ff00"


def test_heatmap_series() -> None:
    """A heatmap layer should produce [x, y, z] triples."""
    layer = LayerSpec(
        mark=MarkType.heatmap,
        data=[],
        style={
            "z_matrix": [[1.0, 2.0], [3.0, 4.0]],
            "x_grid": [10, 20],
            "y_grid": [100, 200],
        },
    )
    panel = PanelSpec(layers=[layer])
    option = EChartsAdapter().render(ChartSpec(panels=[panel]))
    series = option["series"][0]
    assert series["type"] == "heatmap"
    # 2 x 2 grid -> 4 cells.
    assert len(series["data"]) == 4


def test_wireframe_series_3d() -> None:
    """A wireframe layer should switch the chart into 3D mode."""
    layer = LayerSpec(
        mark=MarkType.wireframe,
        data=[{"x": 1, "y": 2, "z": 3}, {"x": 4, "y": 5, "z": 6}],
        x=Encoding(field="x", title="X"),
        y=Encoding(field="y", title="Y"),
        z=Encoding(field="z", title="Z"),
    )
    panel = PanelSpec(layers=[layer])
    option = EChartsAdapter().render(ChartSpec(panels=[panel]))
    assert option["series"][0]["type"] == "scatter3D"
    assert "xAxis3D" in option


def test_boxplot_series_with_color() -> None:
    """A boxplot layer with a color should set itemStyle.color."""
    layer = LayerSpec(
        mark=MarkType.boxplot,
        data=[{"q_stats": [0.0, 1.0, 2.0, 3.0, 4.0]}],
        color="#abc123",
    )
    panel = PanelSpec(layers=[layer])
    option = EChartsAdapter().render(ChartSpec(panels=[panel]))
    series = option["series"][0]
    assert series["type"] == "boxplot"
    assert series["itemStyle"] == {"color": "#abc123"}


def test_line_series_basic() -> None:
    """A line layer should build a line series with paired (x, y) data."""
    layer = LayerSpec(
        mark=MarkType.line,
        data=[{"x": i, "y": i * i} for i in range(4)],
        x=Encoding(field="x", title="X"),
        y=Encoding(field="y", title="Y"),
        color="#0066cc",
        style={"smooth": True, "show_points": True, "dash": "dot"},
    )
    panel = PanelSpec(layers=[layer])
    option = EChartsAdapter().render(ChartSpec(panels=[panel]))
    series = option["series"][0]
    assert series["type"] == "line"
    assert series["smooth"] is True
    # The dash mapping should turn "dot" into "dotted".
    assert series["lineStyle"]["type"] == "dotted"
