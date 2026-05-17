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
