"""Tests for the ChartSpec IR helpers and adapter annotation rendering."""

from __future__ import annotations

import pytest

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
    """Exercise the reference-band and constraint-region adapter paths.

    Assertions pin the rendered ``layout.shapes`` / ``layout.annotations``
    structure, following the style of :class:`TestPlotlyContourStyling`.
    """

    def test_reference_band_on_y_axis(self) -> None:
        band = Annotation(
            annotation_type=AnnotationType.reference_band,
            axis="y",
            value=0.2,
            value_end=0.8,
        )
        spec = ChartSpec(panels=[_scatter_panel([band])])
        result = PlotlyAdapter().render(spec)
        shapes = result["layout"]["shapes"]
        assert len(shapes) == 1
        shape = shapes[0]
        assert shape["type"] == "rect"
        # A y-axis band spans data coordinates on y and the full x domain.
        assert shape["y0"] == 0.2
        assert shape["y1"] == 0.8
        assert shape["yref"] == "y"
        assert shape["xref"] == "x domain"
        assert shape["fillcolor"] == "rgba(37, 99, 235, 0.1)"

    def test_reference_band_on_x_axis(self) -> None:
        band = Annotation(
            annotation_type=AnnotationType.reference_band,
            axis="x",
            value=0.2,
            value_end=0.8,
        )
        spec = ChartSpec(panels=[_scatter_panel([band])])
        result = PlotlyAdapter().render(spec)
        shapes = result["layout"]["shapes"]
        assert len(shapes) == 1
        shape = shapes[0]
        assert shape["type"] == "rect"
        # An x-axis band spans data coordinates on x and the full y domain.
        assert shape["x0"] == 0.2
        assert shape["x1"] == 0.8
        assert shape["xref"] == "x"
        assert shape["yref"] == "y domain"
        assert shape["fillcolor"] == "rgba(37, 99, 235, 0.1)"

    def test_constraint_region_renders(self) -> None:
        region = constraint_region(x_min=0.0, x_max=0.5, y_min=0.0, y_max=0.5)
        spec = ChartSpec(panels=[_scatter_panel([region])])
        result = PlotlyAdapter().render(spec)
        shapes = result["layout"]["shapes"]
        # Both bound pairs are given, so an x-rect and a y-rect are drawn.
        assert len(shapes) == 2
        x_rect, y_rect = shapes
        assert x_rect["x0"] == 0.0
        assert x_rect["x1"] == 0.5
        assert x_rect["xref"] == "x"
        assert x_rect["yref"] == "y domain"
        assert y_rect["y0"] == 0.0
        assert y_rect["y1"] == 0.5
        assert y_rect["yref"] == "y"
        assert y_rect["xref"] == "x domain"
        assert all(shape["fillcolor"] == "rgba(220, 38, 38, 0.15)" for shape in shapes)
        # Each rect carries the region label as a text annotation.
        labels = [ann["text"] for ann in result["layout"]["annotations"]]
        assert labels == ["Infeasible", "Infeasible"]

    def test_incomplete_reference_band_is_skipped(self) -> None:
        # value_end is None: the band cannot be drawn and is silently skipped,
        # so the layout has exactly the shapes of a spec without the band.
        band = Annotation(annotation_type=AnnotationType.reference_band, axis="y", value=0.2)
        spec = ChartSpec(panels=[_scatter_panel([band])])
        result = PlotlyAdapter().render(spec)
        baseline = PlotlyAdapter().render(ChartSpec(panels=[_scatter_panel([])]))
        assert result["layout"].get("shapes") == baseline["layout"].get("shapes")
        assert not result["layout"].get("shapes")


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


def _layer(mark: MarkType, style: dict | None = None) -> LayerSpec:
    """Return a two-point layer of the given mark, with optional style keys."""
    return LayerSpec(
        mark=mark,
        data=[{"x": 0.0, "y": 1.0}, {"x": 1.0, "y": 3.0}],
        x=Encoding(field="x"),
        y=Encoding(field="y"),
        style=style or {},
    )


class TestPlotlyLayersAndPanels:
    """Traces and panel layouts that no other test renders."""

    def test_render_panel_matches_a_one_panel_spec(self) -> None:
        """Rendering a panel on its own gives the figure a one-panel chart would."""
        panel = PanelSpec(layers=[_layer(MarkType.scatter)], title="Yield", x_title="Time")
        adapter = PlotlyAdapter()
        assert adapter.render_panel(panel) == adapter.render(ChartSpec(panels=[panel]))
        assert adapter.render_panel(panel)["layout"]["title"]["text"] == "Yield"

    def test_an_annotation_lands_on_its_own_subplot(self) -> None:
        """In a two-panel chart, a threshold on the second panel references that panel's axes."""
        plain = PanelSpec(layers=[_layer(MarkType.scatter)])
        marked = PanelSpec(layers=[_layer(MarkType.line)], annotations=[significance_threshold(2.0, name="SME")])
        result = PlotlyAdapter().render(ChartSpec(panels=[plain, marked]))
        (shape,) = result["layout"]["shapes"]
        assert shape["y0"] == shape["y1"] == 2.0
        assert shape["yref"] == "y2"
        assert "SME" in [ann["text"] for ann in result["layout"]["annotations"]][-1]

    def test_a_heatmap_layer_takes_its_grid_from_the_style(self) -> None:
        style = {"x_grid": [0.0, 1.0], "y_grid": [5.0, 6.0], "z_matrix": [[1.0, 2.0], [3.0, 4.0]]}
        trace = _first_trace(PanelSpec(layers=[_layer(MarkType.heatmap, style)]))
        assert trace["type"] == "heatmap"
        assert list(trace["y"]) == [5.0, 6.0]
        assert [list(row) for row in trace["z"]] == [[1.0, 2.0], [3.0, 4.0]]

    @pytest.mark.parametrize(
        ("error_y", "expected"),
        [
            pytest.param([0.0, None], None, id="all-zero-bars-draw-no-error-bars"),
            pytest.param([-0.5, None], [0.5, 0.0], id="magnitudes-with-missing-as-zero"),
        ],
    )
    def test_bar_error_bars_are_drawn_only_when_some_are_nonzero(self, error_y: list, expected: list | None) -> None:
        trace = _first_trace(PanelSpec(layers=[_layer(MarkType.bar, {"error_y": error_y})]))
        if expected is None:
            assert "error_y" not in trace
        else:
            assert list(trace["error_y"]["array"]) == expected

    def test_a_scatter_edge_colour_outlines_the_markers(self) -> None:
        """Markers read on any background when they carry an edge: colour and width are forwarded."""
        trace = _first_trace(PanelSpec(layers=[_layer(MarkType.scatter, {"edge_color": "#222222"})]))
        assert trace["marker"]["line"] == {"color": "#222222", "width": 1}


class TestPlotlyIncompleteAnnotations:
    """An annotation without the values it needs draws only what it can."""

    def test_a_reference_line_without_a_value_is_skipped(self) -> None:
        line = Annotation(annotation_type=AnnotationType.reference_line, axis="y", value=None, label="target")
        result = PlotlyAdapter().render(ChartSpec(panels=[_scatter_panel([line])]))
        assert not result["layout"].get("shapes")

    @pytest.mark.parametrize(
        ("bounds", "data_ref", "domain_ref"),
        [
            pytest.param({"y_min": 0.0, "y_max": 0.5}, ("y0", "yref", "y"), ("xref", "x domain"), id="y-bounds-only"),
            pytest.param({"x_min": 0.0, "x_max": 0.5}, ("x0", "xref", "x"), ("yref", "y domain"), id="x-bounds-only"),
        ],
    )
    def test_a_constraint_region_with_one_pair_of_bounds_draws_one_rect(
        self, bounds: dict, data_ref: tuple[str, str, str], domain_ref: tuple[str, str]
    ) -> None:
        result = PlotlyAdapter().render(ChartSpec(panels=[_scatter_panel([constraint_region(**bounds)])]))
        (shape,) = result["layout"]["shapes"]
        start_key, ref_key, ref = data_ref
        assert shape[start_key] == 0.0
        assert shape[ref_key] == ref
        assert shape[domain_ref[0]] == domain_ref[1]


class TestPlotlyUnimplementedMembers:
    """Declared-but-unimplemented spec members raise instead of silently degrading."""

    def test_mark_type_area_raises_not_implemented(self) -> None:
        layer = LayerSpec(
            mark=MarkType.area,
            data=[{"x": 0.0, "y": 0.0}],
            x=Encoding(field="x"),
            y=Encoding(field="y"),
        )
        spec = ChartSpec(panels=[PanelSpec(layers=[layer])])
        with pytest.raises(NotImplementedError, match="area"):
            PlotlyAdapter().render(spec)

    def test_annotation_type_label_raises_not_implemented(self) -> None:
        ann = Annotation(annotation_type=AnnotationType.label, value=1.0, label="note")
        spec = ChartSpec(panels=[_scatter_panel([ann])])
        with pytest.raises(NotImplementedError, match="label"):
            PlotlyAdapter().render(spec)
