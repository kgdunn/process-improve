"""Tests for the ChartSpec IR helpers and adapter annotation rendering."""

from __future__ import annotations

from process_improve.visualization.adapters import PlotlyAdapter
from process_improve.visualization.spec import (
    Annotation,
    ChartSpec,
    Encoding,
    LayerSpec,
    PanelSpec,
    constraint_region,
    significance_threshold,
)
from process_improve.visualization.types import AnnotationType, MarkType


def _scatter_panel(annotations: list[Annotation]) -> PanelSpec:
    layer = LayerSpec(
        mark=MarkType.scatter,
        data=[{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}],
        x=Encoding(field="x"),
        y=Encoding(field="y"),
    )
    return PanelSpec(layers=[layer], annotations=annotations)


class TestChartSpecSerialisation:
    """ChartSpec.to_dict / to_data_dict and annotation factories."""

    def test_to_dict_converts_enums_to_strings(self) -> None:
        spec = ChartSpec(panels=[_scatter_panel([])], plot_type="scatter")
        raw = spec.to_dict()
        assert raw["plot_type"] == "scatter"
        # The MarkType enum should be serialised to its string value.
        assert raw["panels"][0]["layers"][0]["mark"] == "scatter"

    def test_constraint_region_factory(self) -> None:
        ann = constraint_region(x_min=0.0, x_max=1.0, label="No-go")
        assert ann.annotation_type == AnnotationType.constraint_region
        assert ann.style["x_min"] == 0.0
        assert ann.label == "No-go"

    def test_significance_threshold_factory(self) -> None:
        ann = significance_threshold(2.5, name="SME")
        assert ann.annotation_type == AnnotationType.significance_threshold
        assert "SME" in ann.label


class TestPlotlyAnnotationRendering:
    """Exercise the reference-band and constraint-region adapter paths."""

    def test_reference_band_on_y_axis(self) -> None:
        band = Annotation(
            annotation_type=AnnotationType.reference_band,
            axis="y",
            value=0.2,
            value_end=0.8,
        )
        spec = ChartSpec(panels=[_scatter_panel([band])])
        result = PlotlyAdapter().render(spec)
        assert "data" in result

    def test_reference_band_on_x_axis(self) -> None:
        band = Annotation(
            annotation_type=AnnotationType.reference_band,
            axis="x",
            value=0.2,
            value_end=0.8,
        )
        spec = ChartSpec(panels=[_scatter_panel([band])])
        result = PlotlyAdapter().render(spec)
        assert "data" in result

    def test_constraint_region_renders(self) -> None:
        region = constraint_region(x_min=0.0, x_max=0.5, y_min=0.0, y_max=0.5)
        spec = ChartSpec(panels=[_scatter_panel([region])])
        result = PlotlyAdapter().render(spec)
        assert "data" in result

    def test_incomplete_reference_band_is_skipped(self) -> None:
        # value_end is None: the band cannot be drawn and is silently skipped.
        band = Annotation(annotation_type=AnnotationType.reference_band, axis="y", value=0.2)
        spec = ChartSpec(panels=[_scatter_panel([band])])
        result = PlotlyAdapter().render(spec)
        assert "data" in result


def _contour_panel(style: dict, *, color: str | None = None, opacity: float = 1.0) -> PanelSpec:
    layer = LayerSpec(
        mark=MarkType.contour,
        data=[],
        x=Encoding(field="x"),
        y=Encoding(field="y"),
        name="surface",
        color=color,
        opacity=opacity,
        style={"x_grid": [0.0, 1.0], "y_grid": [0.0, 1.0], "z_matrix": [[0.0, 1.0], [1.0, 2.0]], **style},
    )
    return PanelSpec(layers=[layer])


def _first_trace(panel: PanelSpec) -> dict:
    rendered = PlotlyAdapter().render(ChartSpec(panels=[panel]))
    return rendered["data"][0]


class TestPlotlyContourStyling:
    """The contour trace must honour the style keys it is handed.

    These keys were previously accepted into the spec and then discarded by the
    adapter, so an overlay of several responses rendered as filled surfaces all
    in one colorscale.
    """

    def test_defaults_are_unchanged_when_no_style_given(self) -> None:
        """A bare contour keeps the previous appearance."""
        trace = _first_trace(_contour_panel({}))
        assert trace["contours"]["showlabels"] is True
        assert trace["colorscale"] is not None
        assert trace["showscale"] is True

    def test_colorscale_and_z_limits_are_forwarded(self) -> None:
        trace = _first_trace(
            _contour_panel({"colorscale": [[0.0, "#000000"], [1.0, "#FFFFFF"]], "zmin": 0.0, "zmax": 1.0})
        )
        assert [list(stop) for stop in trace["colorscale"]] == [[0.0, "#000000"], [1.0, "#FFFFFF"]]
        assert trace["zmin"] == 0.0
        assert trace["zmax"] == 1.0

    def test_explicit_contour_levels_are_forwarded(self) -> None:
        """Pinning start/end/size is how specification limits become contours."""
        trace = _first_trace(_contour_panel({"contours": {"start": 30.0, "end": 50.0, "size": 20.0}}))
        assert trace["contours"]["start"] == 30.0
        assert trace["contours"]["end"] == 50.0
        assert trace["contours"]["size"] == 20.0

    def test_line_coloring_uses_the_layer_colour(self) -> None:
        """Overlaid responses stay distinguishable only if each keeps its colour."""
        trace = _first_trace(_contour_panel({"contours_coloring": "lines"}, color="#123456"))
        assert trace["contours"]["coloring"] == "lines"
        assert trace["line"]["color"] == "#123456"
        # A per-response colour bar would be meaningless, so it is suppressed.
        assert trace["showscale"] is False

    def test_showscale_and_opacity_are_forwarded(self) -> None:
        trace = _first_trace(_contour_panel({"showscale": False}, opacity=0.35))
        assert trace["showscale"] is False
        assert trace["opacity"] == 0.35

    def test_ncontours_is_forwarded_and_omitted_when_unset(self) -> None:
        assert _first_trace(_contour_panel({"ncontours": 8}))["ncontours"] == 8
        # Passing None would override plotly's own auto-ranging, so it is dropped.
        assert "ncontours" not in _first_trace(_contour_panel({}))
