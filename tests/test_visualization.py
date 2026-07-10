"""Tests for Tool 6: visualize_doe and all DOE plot types.

Covers ChartSpec IR construction, Plotly adapter output, ECharts adapter
output, JSON round-trip serialisability, and the public ``visualize_doe``
entry point.
"""

from __future__ import annotations

import json
import math
from typing import Any

import pytest

from process_improve.experiments.visualization import main_effects_plot, visualize_doe
from process_improve.experiments.visualization.plots.optimization_plots import (
    _composite_desirability,
    _desirability_maximize,
    _desirability_minimize,
    _desirability_target,
    _individual_desirability,
)
from process_improve.experiments.visualization.plots.registry import (
    create_plot,
    get_available_plot_types,
)
from process_improve.visualization.adapters import EChartsAdapter, PlotlyAdapter
from process_improve.visualization.spec import ChartSpec
from process_improve.visualization.types import DOEPlotType

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def two_factor_effects() -> dict[str, float]:
    """Effects dict for a 2^2 design."""
    return {"A": 5.25, "B": -3.50, "A:B": 1.75}


@pytest.fixture
def three_factor_effects() -> dict[str, float]:
    """Effects dict for a 2^3 design."""
    return {"A": 8.0, "B": -4.0, "C": 2.0, "A:B": 1.5, "A:C": -0.5, "B:C": 0.3, "A:B:C": 0.1}


@pytest.fixture
def lenth_data() -> dict[str, Any]:
    """Lenth method results."""
    return {
        "ME": 3.0,
        "SME": 5.0,
        "PSE": 1.2,
        "effects": [
            {"term": "A", "effect": 8.0, "active_ME": True, "active_SME": True},
            {"term": "B", "effect": -4.0, "active_ME": True, "active_SME": False},
            {"term": "C", "effect": 2.0, "active_ME": False, "active_SME": False},
            {"term": "A:B", "effect": 1.5, "active_ME": False, "active_SME": False},
        ],
    }


@pytest.fixture
def coefficients_2f() -> list[dict[str, Any]]:
    """Coefficients for a 2-factor model."""
    return [
        {"term": "Intercept", "coefficient": 40.0},
        {"term": "A", "coefficient": 5.25},
        {"term": "B", "coefficient": -3.50},
        {"term": "A:B", "coefficient": 1.75},
    ]


@pytest.fixture
def coefficients_3f() -> list[dict[str, Any]]:
    """Coefficients for a 3-factor model."""
    return [
        {"term": "Intercept", "coefficient": 40.0},
        {"term": "A", "coefficient": 5.25},
        {"term": "B", "coefficient": -3.50},
        {"term": "C", "coefficient": 2.00},
        {"term": "A:B", "coefficient": 1.75},
        {"term": "A:C", "coefficient": -0.50},
        {"term": "B:C", "coefficient": 0.30},
    ]


@pytest.fixture
def quadratic_coefficients() -> list[dict[str, Any]]:
    """Coefficients for a quadratic RSM model."""
    return [
        {"term": "Intercept", "coefficient": 80.0},
        {"term": "A", "coefficient": 5.0},
        {"term": "B", "coefficient": -3.0},
        {"term": "I(A ** 2)", "coefficient": -4.0},
        {"term": "I(B ** 2)", "coefficient": -2.5},
        {"term": "A:B", "coefficient": 1.5},
    ]


@pytest.fixture
def design_data_2f() -> list[dict[str, Any]]:
    """Design data for a 2^2 replicated design."""
    return [
        {"A": -1, "B": -1, "y": 28},
        {"A": 1, "B": -1, "y": 36},
        {"A": -1, "B": 1, "y": 18},
        {"A": 1, "B": 1, "y": 31},
        {"A": -1, "B": -1, "y": 25},
        {"A": 1, "B": -1, "y": 32},
        {"A": -1, "B": 1, "y": 19},
        {"A": 1, "B": 1, "y": 30},
    ]


@pytest.fixture
def design_data_3f() -> list[dict[str, Any]]:
    """Design data for a 2^3 design."""
    return [
        {"A": -1, "B": -1, "C": -1, "y": 22},
        {"A": 1, "B": -1, "C": -1, "y": 32},
        {"A": -1, "B": 1, "C": -1, "y": 15},
        {"A": 1, "B": 1, "C": -1, "y": 29},
        {"A": -1, "B": -1, "C": 1, "y": 24},
        {"A": 1, "B": -1, "C": 1, "y": 35},
        {"A": -1, "B": 1, "C": 1, "y": 17},
        {"A": 1, "B": 1, "C": 1, "y": 31},
    ]


@pytest.fixture
def residual_diagnostics() -> dict[str, Any]:
    """Residual diagnostics from a model fit."""
    fitted = [28.5, 34.0, 18.5, 30.5, 28.5, 34.0, 18.5, 30.5]
    residuals = [-0.5, 2.0, -0.5, 0.5, -3.5, -2.0, 0.5, -0.5]
    return {"fitted_values": fitted, "residuals": residuals}


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_all_plot_types_registered(self) -> None:
        """Every DOEPlotType enum member should be available."""
        available = get_available_plot_types()
        for pt in DOEPlotType:
            assert pt.value in available, f"{pt.value} not registered"

    def test_create_plot_returns_base_plot(self, two_factor_effects: dict) -> None:
        plot = create_plot(
            "pareto",
            analysis_results={"effects": two_factor_effects},
        )
        assert hasattr(plot, "to_spec")

    def test_create_plot_unknown_type(self) -> None:
        with pytest.raises(ValueError, match="Unknown plot_type"):
            create_plot("nonexistent_plot_type")

    def test_factor_name_extraction_against_design_data(self) -> None:
        """SEC-33 (#282) sub-item 6: ``BasePlot._get_factor_names`` cross-
        references formula tokens against the supplied ``design_data``
        instead of relying on a static reserved-word blocklist.

        The previous blocklist ``{"I", "np", "power"}`` missed common
        statsmodels transforms (Q, center, standardize, ...). Whenever
        ``design_data`` is present, the correct test is "does this token
        actually correspond to a column in the design?" -- which is what
        the test below pins.
        """
        # ``np.power(A, 2) + center(B)`` -- statsmodels' newer transform
        # spellings. ``np``, ``power``, and ``center`` are *not* columns;
        # the cross-reference filter must drop them and keep only A, B.
        plot = create_plot(
            "pareto",  # any concrete plot type works; we exercise the base helper.
            analysis_results={
                "effects": {"A": 1.0, "B": -2.0},
                "model_summary": {
                    "formula": "y ~ np.power(A, 2) + center(B) + A:B",
                },
            },
            design_data=[{"A": -1, "B": -1, "y": 0.0}],
        )
        names = plot._get_factor_names()
        assert "A" in names
        assert "B" in names
        # Transforms that aren't columns must be filtered out.
        for noise in ("np", "power", "center", "y"):
            assert noise not in names


# ---------------------------------------------------------------------------
# Significance plots
# ---------------------------------------------------------------------------


class TestParetoPlot:
    def test_spec_structure(self, two_factor_effects: dict) -> None:
        plot = create_plot("pareto", analysis_results={"effects": two_factor_effects})
        spec = plot.to_spec()
        assert isinstance(spec, ChartSpec)
        assert spec.plot_type == "pareto"
        assert len(spec.panels) == 1
        # Should have bar + cumulative line layers
        assert len(spec.panels[0].layers) == 2

    def test_with_lenth_thresholds(self, two_factor_effects: dict, lenth_data: dict) -> None:
        plot = create_plot(
            "pareto",
            analysis_results={"effects": two_factor_effects, "lenth_method": lenth_data},
        )
        spec = plot.to_spec()
        # Should have threshold annotations
        assert len(spec.panels[0].annotations) >= 1

    def test_plotly_output(self, two_factor_effects: dict) -> None:
        plot = create_plot("pareto", analysis_results={"effects": two_factor_effects})
        fig_dict = plot.to_plotly()
        assert "data" in fig_dict
        assert "layout" in fig_dict

    def test_echarts_output(self, two_factor_effects: dict) -> None:
        plot = create_plot("pareto", analysis_results={"effects": two_factor_effects})
        config = plot.to_echarts()
        assert "series" in config

    def test_json_serialisable(self, two_factor_effects: dict) -> None:
        result = visualize_doe(plot_type="pareto", analysis_results={"effects": two_factor_effects})
        json_str = json.dumps(result)
        assert json_str  # Should not raise

    def test_empty_effects(self) -> None:
        plot = create_plot("pareto", analysis_results={"effects": {}})
        spec = plot.to_spec()
        assert "no effects" in spec.title.lower() or spec.plot_type == "pareto"

    def test_no_error_bars_without_uncertainty(self, two_factor_effects: dict) -> None:
        """Bar layer should not carry error_y when no uncertainty info supplied."""
        plot = create_plot("pareto", analysis_results={"effects": two_factor_effects})
        spec = plot.to_spec()
        bar_layer = spec.panels[0].layers[0]
        assert "error_y" not in bar_layer.style

    def test_error_bars_from_effect_std_errors(self, two_factor_effects: dict) -> None:
        """``effect_std_errors`` should drive per-bar error bars sorted with the bars."""
        std_errors = {"A": 0.5, "B": 0.4, "A:B": 0.3}
        plot = create_plot(
            "pareto",
            analysis_results={
                "effects": two_factor_effects,
                "effect_std_errors": std_errors,
            },
        )
        spec = plot.to_spec()
        bar_layer = spec.panels[0].layers[0]
        names = [row["name"] for row in bar_layer.data]
        expected = [std_errors[n] for n in names]
        assert bar_layer.style["error_y"] == pytest.approx(expected)

    def test_error_bars_fall_back_to_lenth_pse(
        self,
        three_factor_effects: dict,
        lenth_data: dict,
    ) -> None:
        """Without explicit std errors, Lenth's PSE should populate uniform error bars."""
        plot = create_plot(
            "pareto",
            analysis_results={"effects": three_factor_effects, "lenth_method": lenth_data},
        )
        spec = plot.to_spec()
        bar_layer = spec.panels[0].layers[0]
        errors = bar_layer.style["error_y"]
        assert len(errors) == len(three_factor_effects)
        assert all(e == pytest.approx(lenth_data["PSE"]) for e in errors)

    def test_plotly_renders_error_bars(self, two_factor_effects: dict) -> None:
        """Plotly bar trace should expose ``error_y`` when uncertainty is present."""
        plot = create_plot(
            "pareto",
            analysis_results={
                "effects": two_factor_effects,
                "effect_std_errors": {"A": 0.5, "B": 0.4, "A:B": 0.3},
            },
        )
        fig_dict = plot.to_plotly()
        bar_traces = [t for t in fig_dict["data"] if t.get("type") == "bar"]
        assert bar_traces
        err = bar_traces[0].get("error_y")
        assert err is not None
        assert err.get("visible") is True
        assert len(err["array"]) == 3

    def test_echarts_renders_error_bars_alongside_thresholds(
        self,
        two_factor_effects: dict,
        lenth_data: dict,
    ) -> None:
        """ECharts bar series should carry markLine pairs for error bars,
        and Lenth threshold annotations must coexist with them.
        """
        plot = create_plot(
            "pareto",
            analysis_results={
                "effects": two_factor_effects,
                "effect_std_errors": {"A": 0.5, "B": 0.4, "A:B": 0.3},
                "lenth_method": lenth_data,
            },
        )
        config = plot.to_echarts()
        bar_series = next(s for s in config["series"] if s["type"] == "bar")
        mark_data = bar_series["markLine"]["data"]
        # Three error-bar pairs (each is [start, end]) + ME + SME annotations
        pair_entries = [d for d in mark_data if isinstance(d, list)]
        threshold_entries = [d for d in mark_data if isinstance(d, dict)]
        assert len(pair_entries) == 3
        assert len(threshold_entries) >= 1


class TestHalfNormalPlot:
    def test_spec_structure(self, three_factor_effects: dict) -> None:
        plot = create_plot("half_normal", analysis_results={"effects": three_factor_effects})
        spec = plot.to_spec()
        assert spec.plot_type == "half_normal"
        assert len(spec.panels) == 1

    def test_with_lenth(self, three_factor_effects: dict, lenth_data: dict) -> None:
        plot = create_plot(
            "half_normal",
            analysis_results={"effects": three_factor_effects, "lenth_method": lenth_data},
        )
        spec = plot.to_spec()
        # Should have scatter + reference + label layers
        assert len(spec.panels[0].layers) >= 2


class TestDanielPlot:
    def test_spec_structure(self, three_factor_effects: dict) -> None:
        plot = create_plot("daniel", analysis_results={"effects": three_factor_effects})
        spec = plot.to_spec()
        assert spec.plot_type == "daniel"
        assert len(spec.panels) == 1


# ---------------------------------------------------------------------------
# Effect plots
# ---------------------------------------------------------------------------


class TestMainEffectsPlot:
    def test_spec_structure(self, design_data_2f: list) -> None:
        plot = create_plot("main_effects", design_data=design_data_2f, response_column="y")
        spec = plot.to_spec()
        assert spec.plot_type == "main_effects"
        # One line per factor
        assert len(spec.panels[0].layers) == 2

    def test_no_data(self) -> None:
        plot = create_plot("main_effects")
        spec = plot.to_spec()
        assert "no data" in spec.title.lower()


class TestMainEffectsPlotConvenience:
    """Tests for the standalone ``main_effects_plot`` convenience function."""

    def test_from_dataframe(self, design_data_2f: list) -> None:
        import pandas as pd
        import plotly.graph_objects as go

        df = pd.DataFrame(design_data_2f)
        fig = main_effects_plot(df, response_column="y")
        assert isinstance(fig, go.Figure)
        # One trace per factor (A and B)
        assert len(fig.data) >= 2

    def test_from_model(self) -> None:
        import plotly.graph_objects as go

        from process_improve.experiments import c, gather, lm

        A = c(-1, +1, -1, +1)
        B = c(-1, -1, +1, +1)
        y = c(52, 74, 62, 80, name="y")
        expt = gather(A=A, B=B, y=y, title="ME convenience test")
        model = lm("y ~ A + B", expt)

        fig = main_effects_plot(model)
        assert isinstance(fig, go.Figure)
        # A and B factors → 2 line traces
        assert len(fig.data) >= 2

    def test_factors_subset(self, design_data_3f: list) -> None:
        import pandas as pd

        df = pd.DataFrame(design_data_3f)
        fig = main_effects_plot(df, response_column="y", factors_to_plot=["A", "C"])
        # Two factors plotted → at least two line traces
        assert len(fig.data) >= 2

    def test_from_model_with_explicit_factors(self) -> None:
        """Passing factors_to_plot with a Model should override the level-1 list."""
        import plotly.graph_objects as go

        from process_improve.experiments import c, gather, lm

        A = c(-1, +1, -1, +1, -1, +1, -1, +1)
        B = c(-1, -1, +1, +1, -1, -1, +1, +1)
        C = c(-1, -1, -1, -1, +1, +1, +1, +1)
        y = c(22, 32, 15, 29, 24, 35, 17, 31, name="y")
        expt = gather(A=A, B=B, C=C, y=y, title="ME factor subset")
        model = lm("y ~ A + B + C", expt)

        fig = main_effects_plot(model, factors_to_plot=["A"])
        assert isinstance(fig, go.Figure)

    def test_dataframe_missing_response_column(self, design_data_2f: list) -> None:
        import pandas as pd

        df = pd.DataFrame(design_data_2f)
        with pytest.raises(ValueError, match="response_column"):
            main_effects_plot(df)

    def test_response_column_not_in_data(self, design_data_2f: list) -> None:
        import pandas as pd

        df = pd.DataFrame(design_data_2f)
        with pytest.raises(ValueError, match="not found in data columns"):
            main_effects_plot(df, response_column="missing")

    def test_invalid_input_type(self) -> None:
        with pytest.raises(TypeError, match="Model or a pandas DataFrame"):
            main_effects_plot([1, 2, 3])  # type: ignore[arg-type]


class TestInteractionPlot:
    def test_spec_structure(self, design_data_2f: list) -> None:
        plot = create_plot(
            "interaction",
            design_data=design_data_2f,
            response_column="y",
            factors_to_plot=["A", "B"],
        )
        spec = plot.to_spec()
        assert spec.plot_type == "interaction"

    def test_need_two_factors(self, design_data_2f: list) -> None:
        plot = create_plot(
            "interaction",
            design_data=design_data_2f,
            response_column="y",
            factors_to_plot=["A"],
        )
        spec = plot.to_spec()
        assert "need at least 2" in spec.title.lower()


class TestPerturbationPlot:
    def test_spec_structure(self, coefficients_2f: list) -> None:
        plot = create_plot(
            "perturbation",
            analysis_results={"coefficients": coefficients_2f},
            factors_to_plot=["A", "B"],
        )
        spec = plot.to_spec()
        assert spec.plot_type == "perturbation"
        # One line per factor
        assert len(spec.panels[0].layers) >= 2


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------


class TestResidualsVsFittedPlot:
    def test_spec_structure(self, residual_diagnostics: dict) -> None:
        plot = create_plot(
            "residuals_vs_fitted",
            analysis_results={"residual_diagnostics": residual_diagnostics},
        )
        spec = plot.to_spec()
        assert spec.plot_type == "residuals_vs_fitted"
        assert len(spec.panels[0].layers) == 1
        # Should have zero reference line annotation
        assert len(spec.panels[0].annotations) >= 1


class TestNormalProbabilityPlot:
    def test_spec_structure(self, residual_diagnostics: dict) -> None:
        plot = create_plot(
            "normal_probability",
            analysis_results={"residual_diagnostics": residual_diagnostics},
        )
        spec = plot.to_spec()
        assert spec.plot_type == "normal_probability"
        # Scatter + reference line
        assert len(spec.panels[0].layers) == 2


class TestResidualsVsOrderPlot:
    def test_spec_structure(self, residual_diagnostics: dict) -> None:
        plot = create_plot(
            "residuals_vs_order",
            analysis_results={"residual_diagnostics": residual_diagnostics},
        )
        spec = plot.to_spec()
        assert spec.plot_type == "residuals_vs_order"


class TestBoxCoxPlot:
    def test_spec_with_positive_response(self, design_data_2f: list) -> None:
        plot = create_plot("box_cox", design_data=design_data_2f, response_column="y")
        spec = plot.to_spec()
        assert spec.plot_type == "box_cox"
        assert "optimal_lambda" in spec.metadata


# ---------------------------------------------------------------------------
# Surface plots
# ---------------------------------------------------------------------------


class TestContourPlot:
    def test_spec_structure(self, coefficients_2f: list) -> None:
        plot = create_plot(
            "contour",
            analysis_results={"coefficients": coefficients_2f},
            factors_to_plot=["A", "B"],
        )
        spec = plot.to_spec()
        assert spec.plot_type == "contour"
        assert len(spec.panels) == 1

    def test_quadratic_model(self, quadratic_coefficients: list) -> None:
        plot = create_plot(
            "contour",
            analysis_results={"coefficients": quadratic_coefficients},
            factors_to_plot=["A", "B"],
        )
        spec = plot.to_spec()
        layer = spec.panels[0].layers[0]
        z = layer.style["z_matrix"]
        assert len(z) == 50  # Grid resolution
        assert len(z[0]) == 50

    def test_zero_reference_lines(self, coefficients_2f: list) -> None:
        """Issue #5: solid black H/V lines through the origin."""
        plot = create_plot(
            "contour",
            analysis_results={"coefficients": coefficients_2f},
            factors_to_plot=["A", "B"],
        )
        spec = plot.to_spec()
        anns = spec.panels[0].annotations
        axes = sorted(a.axis for a in anns if a.value == 0.0)
        assert axes == ["x", "y"]
        # Lines should be solid and dark
        for ann in anns:
            assert ann.style.get("dash") == "solid"

    def test_full_factor_labels_on_axes(self, coefficients_2f: list) -> None:
        """Issue #15: full variable names on contour axes."""
        plot = create_plot(
            "contour",
            analysis_results={"coefficients": coefficients_2f},
            factors_to_plot=["A", "B"],
            factor_labels={"A": "Temperature", "B": "Pressure"},
        )
        spec = plot.to_spec()
        panel = spec.panels[0]
        assert "Temperature" in panel.x_title
        assert "Pressure" in panel.y_title
        # Symbol is preserved for traceability
        assert "(A)" in panel.x_title
        assert "(B)" in panel.y_title

    def test_equal_aspect_hint(self, coefficients_2f: list) -> None:
        """Issues #14 and #22: 1:1 aspect ratio hint."""
        plot = create_plot(
            "contour",
            analysis_results={"coefficients": coefficients_2f},
            factors_to_plot=["A", "B"],
        )
        spec = plot.to_spec()
        assert spec.panels[0].backend_hints.get("equal_aspect") is True

        # And the Plotly adapter applies it as a scaleanchor
        fig = plot.to_plotly()
        layout = fig["layout"]
        # scaleanchor is set on yaxis when equal_aspect is requested
        assert layout.get("yaxis", {}).get("scaleanchor") == "x"

    def test_design_points_overlay(self, coefficients_2f: list, design_data_2f: list) -> None:
        """Issue #11: experimental points overlaid with jittered replicates."""
        plot = create_plot(
            "contour",
            analysis_results={"coefficients": coefficients_2f},
            design_data=design_data_2f,
            response_column="y",
            factors_to_plot=["A", "B"],
        )
        spec = plot.to_spec()
        layers = spec.panels[0].layers
        assert len(layers) == 2  # contour + scatter overlay
        scatter = layers[1]
        assert scatter.mark.value == "scatter"
        assert len(scatter.data) == len(design_data_2f)
        # Replicated runs should have distinct (jittered) coordinates
        coords = {(round(r["x"], 6), round(r["y"], 6)) for r in scatter.data}
        assert len(coords) == len(design_data_2f)

    def test_design_point_hover_lists_all_factors(
        self,
        coefficients_3f: list,
        design_data_3f: list,
    ) -> None:
        """Issue #23: hover text exposes all factor levels for the run."""
        plot = create_plot(
            "contour",
            analysis_results={"coefficients": coefficients_3f},
            design_data=design_data_3f,
            response_column="y",
            factors_to_plot=["A", "B"],
            hold_values={"C": 0.0},
        )
        spec = plot.to_spec()
        scatter = spec.panels[0].layers[1]
        hover = scatter.data[0]["hover"]
        assert "A" in hover
        assert "B" in hover
        assert "C" in hover  # held factor still shown
        assert "y" in hover  # response shown

    def test_no_design_points_without_data(self, coefficients_2f: list) -> None:
        """Without design_data, only the contour layer is emitted."""
        plot = create_plot(
            "contour",
            analysis_results={"coefficients": coefficients_2f},
            factors_to_plot=["A", "B"],
        )
        spec = plot.to_spec()
        assert len(spec.panels[0].layers) == 1

    def test_plotly_contour_labels_enabled(self, coefficients_2f: list) -> None:
        """Issue #10: contour line labels visible (Plotly ``showlabels``)."""
        plot = create_plot(
            "contour",
            analysis_results={"coefficients": coefficients_2f},
            factors_to_plot=["A", "B"],
        )
        fig = plot.to_plotly()
        contour_trace = next(t for t in fig["data"] if t.get("type") == "contour")
        assert contour_trace["contours"]["showlabels"] is True


class TestSurface3DPlot:
    def test_spec_structure(self, quadratic_coefficients: list) -> None:
        plot = create_plot(
            "surface_3d",
            analysis_results={"coefficients": quadratic_coefficients},
            factors_to_plot=["A", "B"],
        )
        spec = plot.to_spec()
        assert spec.plot_type == "surface_3d"
        assert spec.metadata.get("requires_gl") is True


class TestPredictionVariancePlot:
    def test_spec_structure(self, design_data_2f: list) -> None:
        plot = create_plot(
            "prediction_variance",
            design_data=design_data_2f,
            response_column="y",
            factors_to_plot=["A", "B"],
        )
        spec = plot.to_spec()
        assert spec.plot_type == "prediction_variance"


# ---------------------------------------------------------------------------
# Cube plot
# ---------------------------------------------------------------------------


class TestCubePlot:
    def test_spec_structure(self, coefficients_3f: list) -> None:
        plot = create_plot(
            "cube_plot",
            analysis_results={"coefficients": coefficients_3f},
            factors_to_plot=["A", "B", "C"],
        )
        spec = plot.to_spec()
        assert spec.plot_type == "cube_plot"
        assert spec.metadata.get("requires_gl") is True
        assert len(spec.metadata["vertices"]) == 8

    def test_needs_3_factors(self, coefficients_2f: list) -> None:
        plot = create_plot(
            "cube_plot",
            analysis_results={"coefficients": coefficients_2f},
            factors_to_plot=["A", "B"],
        )
        spec = plot.to_spec()
        assert "need exactly 3" in spec.title.lower()

    def test_raw_data_fallback(self, design_data_3f: list) -> None:
        plot = create_plot(
            "cube_plot",
            design_data=design_data_3f,
            response_column="y",
            factors_to_plot=["A", "B", "C"],
        )
        spec = plot.to_spec()
        assert spec.plot_type == "cube_plot"


# ---------------------------------------------------------------------------
# Square plot
# ---------------------------------------------------------------------------


class TestSquarePlot:
    def test_spec_structure(self, coefficients_2f: list) -> None:
        plot = create_plot(
            "square_plot",
            analysis_results={"coefficients": coefficients_2f},
            factors_to_plot=["A", "B"],
        )
        spec = plot.to_spec()
        assert spec.plot_type == "square_plot"
        assert len(spec.metadata["vertices"]) == 4

    def test_needs_2_factors(self) -> None:
        plot = create_plot(
            "square_plot",
            analysis_results={"coefficients": [{"term": "Intercept", "coefficient": 5.0}]},
            factors_to_plot=["A"],
        )
        spec = plot.to_spec()
        assert "need exactly 2" in spec.title.lower()

    def test_raw_data_fallback(self, design_data_2f: list) -> None:
        plot = create_plot(
            "square_plot",
            design_data=design_data_2f,
            response_column="y",
            factors_to_plot=["A", "B"],
        )
        spec = plot.to_spec()
        assert spec.plot_type == "square_plot"
        values = [v["value"] for v in spec.metadata["vertices"]]
        assert len(values) == 4
        assert all(isinstance(v, float) for v in values)

    def test_vertex_values_from_coefficients(self, coefficients_2f: list) -> None:
        """Predicted values should follow y = b0 + bA*A + bB*B + bAB*A*B."""
        plot = create_plot(
            "square_plot",
            analysis_results={"coefficients": coefficients_2f},
            factors_to_plot=["A", "B"],
        )
        spec = plot.to_spec()
        # Vertex order follows itertools.product([-1, 1], repeat=2):
        # (-1,-1), (-1,1), (1,-1), (1,1)
        expected = [
            40.0 + 5.25 * (-1) + (-3.50) * (-1) + 1.75 * (-1) * (-1),
            40.0 + 5.25 * (-1) + (-3.50) * (1) + 1.75 * (-1) * (1),
            40.0 + 5.25 * (1) + (-3.50) * (-1) + 1.75 * (1) * (-1),
            40.0 + 5.25 * (1) + (-3.50) * (1) + 1.75 * (1) * (1),
        ]
        got = [v["value"] for v in spec.metadata["vertices"]]
        for g, e in zip(got, expected):  # noqa: B905
            assert abs(g - e) < 1e-9


# ---------------------------------------------------------------------------
# Optimisation plots
# ---------------------------------------------------------------------------


class TestDesirabilityContourPlot:
    def test_spec_structure(self, quadratic_coefficients: list) -> None:
        plot = create_plot(
            "desirability_contour",
            analysis_results={
                "coefficients": quadratic_coefficients,
                "optimization": {
                    "responses": [
                        {
                            "coefficients": quadratic_coefficients,
                            "goal": "maximize",
                            "low": 60.0,
                            "high": 90.0,
                        },
                    ],
                },
            },
            factors_to_plot=["A", "B"],
        )
        spec = plot.to_spec()
        assert spec.plot_type == "desirability_contour"

    def test_single_response_fallback(self, quadratic_coefficients: list) -> None:
        """Falls back to top-level coefficients when no optimization key."""
        plot = create_plot(
            "desirability_contour",
            analysis_results={"coefficients": quadratic_coefficients},
            factors_to_plot=["A", "B"],
        )
        spec = plot.to_spec()
        assert spec.plot_type == "desirability_contour"


class TestOverlayPlot:
    def test_spec_structure(self, quadratic_coefficients: list, coefficients_2f: list) -> None:
        plot = create_plot(
            "overlay",
            analysis_results={
                "optimization": {
                    "responses": [
                        {"name": "Yield", "coefficients": quadratic_coefficients},
                        {"name": "Purity", "coefficients": coefficients_2f},
                    ],
                },
            },
            factors_to_plot=["A", "B"],
        )
        spec = plot.to_spec()
        assert spec.plot_type == "overlay"
        assert len(spec.panels[0].layers) == 2


class TestRidgeTracePlot:
    def test_spec_structure(self, quadratic_coefficients: list) -> None:
        plot = create_plot(
            "ridge_trace",
            analysis_results={"coefficients": quadratic_coefficients},
            factors_to_plot=["A", "B"],
        )
        spec = plot.to_spec()
        assert spec.plot_type == "ridge_trace"
        assert len(spec.panels) == 2  # Response + factor panels


class TestSteepestAscentPathPlot:
    def test_spec_with_coefficients(self, coefficients_2f: list) -> None:
        plot = create_plot(
            "steepest_ascent_path",
            analysis_results={"coefficients": coefficients_2f},
            factors_to_plot=["A", "B"],
        )
        spec = plot.to_spec()
        assert spec.plot_type == "steepest_ascent_path"
        assert len(spec.panels) == 2
        assert spec.metadata["n_steps"] > 0

    def test_spec_with_precomputed_path(self) -> None:
        path = {
            "direction": "ascent",
            "direction_vector": {"A": 0.8, "B": 0.6},
            "step_size": 0.5,
            "steps": [
                {"step": 0, "coded": {"A": 0.0, "B": 0.0}, "predicted_response": 40.0},
                {"step": 1, "coded": {"A": 0.4, "B": 0.3}, "predicted_response": 43.5},
                {"step": 2, "coded": {"A": 0.8, "B": 0.6}, "predicted_response": 47.0},
            ],
        }
        plot = create_plot(
            "steepest_ascent_path",
            analysis_results={"steepest_path": path},
        )
        spec = plot.to_spec()
        assert spec.plot_type == "steepest_ascent_path"


# ---------------------------------------------------------------------------
# Design quality plots
# ---------------------------------------------------------------------------


class TestFDSPlot:
    def test_spec_structure(self, design_data_2f: list) -> None:
        plot = create_plot("fds_plot", design_data=design_data_2f, response_column="y")
        spec = plot.to_spec()
        assert spec.plot_type == "fds_plot"
        assert "median_spv" in spec.metadata


class TestPowerCurvePlot:
    def test_spec_structure(self, design_data_2f: list) -> None:
        plot = create_plot("power_curve", design_data=design_data_2f, response_column="y")
        spec = plot.to_spec()
        assert spec.plot_type == "power_curve"
        assert "n_runs" in spec.metadata


# ---------------------------------------------------------------------------
# Adapter tests
# ---------------------------------------------------------------------------


class TestPlotlyAdapter:
    def test_renders_pareto(self, two_factor_effects: dict) -> None:
        plot = create_plot("pareto", analysis_results={"effects": two_factor_effects})
        spec = plot.to_spec()
        result = PlotlyAdapter().render(spec)
        assert "data" in result
        assert "layout" in result
        assert len(result["data"]) >= 2  # Bar + line

    def test_renders_contour(self, coefficients_2f: list) -> None:
        plot = create_plot(
            "contour",
            analysis_results={"coefficients": coefficients_2f},
            factors_to_plot=["A", "B"],
        )
        spec = plot.to_spec()
        result = PlotlyAdapter().render(spec)
        assert "data" in result

    def test_multi_panel(self, quadratic_coefficients: list) -> None:
        plot = create_plot(
            "ridge_trace",
            analysis_results={"coefficients": quadratic_coefficients},
            factors_to_plot=["A", "B"],
        )
        spec = plot.to_spec()
        result = PlotlyAdapter().render(spec)
        assert "data" in result


class TestEChartsAdapter:
    def test_renders_pareto(self, two_factor_effects: dict) -> None:
        plot = create_plot("pareto", analysis_results={"effects": two_factor_effects})
        spec = plot.to_spec()
        result = EChartsAdapter().render(spec)
        assert "series" in result

    def test_renders_contour(self, coefficients_2f: list) -> None:
        plot = create_plot(
            "contour",
            analysis_results={"coefficients": coefficients_2f},
            factors_to_plot=["A", "B"],
        )
        spec = plot.to_spec()
        result = EChartsAdapter().render(spec)
        assert "series" in result


# ---------------------------------------------------------------------------
# Public API: visualize_doe
# ---------------------------------------------------------------------------


class TestVisualizeDoe:
    def test_returns_both_backends(self, two_factor_effects: dict) -> None:
        result = visualize_doe(
            plot_type="pareto",
            analysis_results={"effects": two_factor_effects},
        )
        assert "plotly" in result
        assert "echarts" in result
        assert result["plot_type"] == "pareto"

    def test_plotly_only(self, two_factor_effects: dict) -> None:
        result = visualize_doe(
            plot_type="pareto",
            analysis_results={"effects": two_factor_effects},
            backend="plotly",
        )
        assert "plotly" in result
        assert result.get("echarts") is None

    def test_echarts_only(self, two_factor_effects: dict) -> None:
        result = visualize_doe(
            plot_type="pareto",
            analysis_results={"effects": two_factor_effects},
            backend="echarts",
        )
        assert result.get("plotly") is None
        assert "echarts" in result

    def test_json_round_trip(self, two_factor_effects: dict) -> None:
        result = visualize_doe(
            plot_type="pareto",
            analysis_results={"effects": two_factor_effects},
        )
        json_str = json.dumps(result)
        parsed = json.loads(json_str)
        assert parsed["plot_type"] == "pareto"

    def test_unknown_plot_type(self) -> None:
        with pytest.raises(ValueError, match="nonexistent"):
            visualize_doe(plot_type="nonexistent")

    @pytest.mark.parametrize("plot_type", [
        "pareto", "half_normal", "daniel",
    ])
    def test_significance_plots(self, plot_type: str, three_factor_effects: dict, lenth_data: dict) -> None:
        result = visualize_doe(
            plot_type=plot_type,
            analysis_results={"effects": three_factor_effects, "lenth_method": lenth_data},
        )
        assert result["plot_type"] == plot_type
        assert json.dumps(result)

    @pytest.mark.parametrize("plot_type", [
        "residuals_vs_fitted", "normal_probability", "residuals_vs_order",
    ])
    def test_diagnostic_plots(self, plot_type: str, residual_diagnostics: dict) -> None:
        result = visualize_doe(
            plot_type=plot_type,
            analysis_results={"residual_diagnostics": residual_diagnostics},
        )
        assert result["plot_type"] == plot_type
        assert json.dumps(result)

    def test_contour_plot(self, coefficients_2f: list) -> None:
        result = visualize_doe(
            plot_type="contour",
            analysis_results={"coefficients": coefficients_2f},
            factors_to_plot=["A", "B"],
        )
        assert result["plot_type"] == "contour"
        assert json.dumps(result)

    def test_surface_3d_plot(self, quadratic_coefficients: list) -> None:
        result = visualize_doe(
            plot_type="surface_3d",
            analysis_results={"coefficients": quadratic_coefficients},
            factors_to_plot=["A", "B"],
        )
        assert result["plot_type"] == "surface_3d"

    def test_cube_plot(self, coefficients_3f: list) -> None:
        result = visualize_doe(
            plot_type="cube_plot",
            analysis_results={"coefficients": coefficients_3f},
            factors_to_plot=["A", "B", "C"],
        )
        assert result["plot_type"] == "cube_plot"

    def test_main_effects_plot(self, design_data_2f: list) -> None:
        result = visualize_doe(
            plot_type="main_effects",
            design_data=design_data_2f,
            response_column="y",
        )
        assert result["plot_type"] == "main_effects"

    def test_interaction_plot(self, design_data_2f: list) -> None:
        result = visualize_doe(
            plot_type="interaction",
            design_data=design_data_2f,
            response_column="y",
            factors_to_plot=["A", "B"],
        )
        assert result["plot_type"] == "interaction"


# ---------------------------------------------------------------------------
# Tool spec integration
# ---------------------------------------------------------------------------


class TestToolSpecIntegration:
    def test_tool_registered(self) -> None:
        """visualize_doe should be in the tool registry."""
        from process_improve.experiments.tools import get_experiments_tool_specs

        specs = get_experiments_tool_specs()
        names = [s["name"] for s in specs]
        assert "visualize_doe" in names

    def test_execute_tool_call(self) -> None:
        from process_improve.tool_spec import execute_tool_call

        result = execute_tool_call("visualize_doe", {
            "plot_type": "pareto",
            "analysis_results": {"effects": {"A": 5.0, "B": -3.0}},
        })
        assert "error" not in result
        assert result["plot_type"] == "pareto"


# ---------------------------------------------------------------------------
# Desirability helper functions (optimization_plots.py)
# ---------------------------------------------------------------------------


class TestDesirabilityHelpers:
    """Unit tests for the pure desirability functions in optimization_plots."""

    def test_maximize_boundaries(self) -> None:
        assert _desirability_maximize(0.0, low=1.0, high=2.0) == 0.0
        assert _desirability_maximize(3.0, low=1.0, high=2.0) == 1.0
        assert _desirability_maximize(1.5, low=1.0, high=2.0) == pytest.approx(0.5)

    def test_minimize_boundaries(self) -> None:
        assert _desirability_minimize(0.5, low=1.0, high=2.0) == 1.0
        assert _desirability_minimize(2.5, low=1.0, high=2.0) == 0.0
        assert _desirability_minimize(1.5, low=1.0, high=2.0) == pytest.approx(0.5)

    def test_target_all_branches(self) -> None:
        # Outside [low, high] on either side -> 0
        assert _desirability_target(0.5, low=1.0, target=2.0, high=3.0) == 0.0
        assert _desirability_target(3.5, low=1.0, target=2.0, high=3.0) == 0.0
        # Below target: rising ramp
        assert _desirability_target(1.5, low=1.0, target=2.0, high=3.0) == pytest.approx(0.5)
        # Above target: falling ramp
        assert _desirability_target(2.5, low=1.0, target=2.0, high=3.0) == pytest.approx(0.5)

    def test_individual_desirability_goal_dispatch(self) -> None:
        assert _individual_desirability(0.75, {"goal": "maximize", "low": 0.0, "high": 1.0}) == pytest.approx(0.75)
        assert _individual_desirability(0.25, {"goal": "minimize", "low": 0.0, "high": 1.0}) == pytest.approx(0.75)
        target_goal = {"goal": "target", "low": 0.0, "target": 0.5, "high": 1.0}
        assert _individual_desirability(0.25, target_goal) == pytest.approx(0.5)
        # Unknown goal type falls through to 0.0
        assert _individual_desirability(0.5, {"goal": "unknown_goal"}) == 0.0

    def test_composite_empty_list(self) -> None:
        assert _composite_desirability([]) == 0.0

    def test_composite_any_zero(self) -> None:
        assert _composite_desirability([0.9, 0.0, 0.8]) == 0.0

    def test_composite_importance_weighting(self) -> None:
        """Importance weights should change the geometric mean."""
        unweighted = _composite_desirability([0.25, 1.0])
        weighted = _composite_desirability([0.25, 1.0], importances=[3.0, 1.0])
        assert unweighted == pytest.approx(0.25**0.5)
        assert weighted == pytest.approx(0.25**0.75)
        assert weighted != pytest.approx(unweighted)


# ---------------------------------------------------------------------------
# Optimisation plot edge cases
# ---------------------------------------------------------------------------


class TestDesirabilityContourEdgeCases:
    def test_no_optimisation_data(self) -> None:
        plot = create_plot("desirability_contour")
        spec = plot.to_spec()
        assert "no optimisation data" in spec.title.lower()

    def test_empty_optimization_key_and_no_coefficients(self) -> None:
        """An optimization dict without responses and no top-level coefficients."""
        plot = create_plot("desirability_contour", analysis_results={"optimization": {}})
        spec = plot.to_spec()
        assert "no optimisation data" in spec.title.lower()

    def test_single_factor(self, coefficients_2f: list) -> None:
        plot = create_plot(
            "desirability_contour",
            analysis_results={"coefficients": coefficients_2f},
            factors_to_plot=["A"],
        )
        spec = plot.to_spec()
        assert "need at least 2" in spec.title.lower()


class TestOverlayPlotEdgeCases:
    def test_no_analysis_results(self) -> None:
        plot = create_plot("overlay")
        spec = plot.to_spec()
        assert "no response data" in spec.title.lower()

    def test_missing_optimization_key(self, coefficients_2f: list) -> None:
        plot = create_plot("overlay", analysis_results={"coefficients": coefficients_2f})
        spec = plot.to_spec()
        assert "no response data" in spec.title.lower()

    def test_single_factor(self, coefficients_2f: list) -> None:
        plot = create_plot(
            "overlay",
            analysis_results={
                "optimization": {"responses": [{"name": "Yield", "coefficients": coefficients_2f}]},
            },
            factors_to_plot=["A"],
        )
        spec = plot.to_spec()
        assert "need at least 2" in spec.title.lower()

    def test_response_with_empty_coefficients_skipped(self, coefficients_2f: list) -> None:
        plot = create_plot(
            "overlay",
            analysis_results={
                "optimization": {
                    "responses": [
                        {"name": "Yield", "coefficients": coefficients_2f},
                        {"name": "NoModel", "coefficients": []},
                    ],
                },
            },
            factors_to_plot=["A", "B"],
        )
        spec = plot.to_spec()
        # Only one contour layer: the empty-coefficient response is skipped
        assert len(spec.panels[0].layers) == 1
        assert spec.metadata["n_responses"] == 2

    def test_constraint_labels_from_bounds(self, coefficients_2f: list) -> None:
        """Responses carrying low/high bounds produce constraint annotations."""
        plot = create_plot(
            "overlay",
            analysis_results={
                "optimization": {
                    "responses": [
                        {"name": "Yield", "coefficients": coefficients_2f, "low": 30.0, "high": 50.0},
                    ],
                },
            },
            factors_to_plot=["A", "B"],
        )
        spec = plot.to_spec()
        anns = spec.panels[0].annotations
        assert len(anns) == 1
        assert anns[0].label == "Yield: [30.00, 50.00]"


class TestRidgeTracePlotEdgeCases:
    def test_no_coefficients(self) -> None:
        plot = create_plot("ridge_trace")
        spec = plot.to_spec()
        assert "no coefficients" in spec.title.lower()

    def test_no_factors(self) -> None:
        plot = create_plot(
            "ridge_trace",
            analysis_results={"coefficients": [{"term": "Intercept", "coefficient": 1.0}]},
        )
        spec = plot.to_spec()
        assert "no factors" in spec.title.lower()

    def test_singular_matrix_branch(self) -> None:
        """The mu grid includes -10 exactly, making B + mu*I singular there.

        With quadratic coefficients of 10.0 for both factors, the matrix
        ``b_mat + (-10) * I`` is exactly zero, so ``np.linalg.solve`` raises
        LinAlgError and the loop must continue to the next mu value.
        """
        coefficients = [
            {"term": "A", "coefficient": 1.0},
            {"term": "B", "coefficient": 1.0},
            {"term": "I(A ** 2)", "coefficient": 10.0},
            {"term": "I(B ** 2)", "coefficient": 10.0},
        ]
        plot = create_plot(
            "ridge_trace",
            analysis_results={"coefficients": coefficients},
            factors_to_plot=["A", "B"],
        )
        spec = plot.to_spec()
        assert spec.plot_type == "ridge_trace"
        assert len(spec.panels) == 2
        # All 30 radii still produce a response trace despite the singular mu
        assert len(spec.panels[0].layers[0].data) == 30

    def test_zero_linear_terms_keep_centre_solution(self) -> None:
        """With b = 0 the solved x is the zero vector; factor traces stay at 0."""
        coefficients = [
            {"term": "Intercept", "coefficient": 5.0},
            {"term": "I(A ** 2)", "coefficient": -2.0},
            {"term": "I(B ** 2)", "coefficient": -3.0},
        ]
        plot = create_plot(
            "ridge_trace",
            analysis_results={"coefficients": coefficients},
            factors_to_plot=["A", "B"],
        )
        spec = plot.to_spec()
        factor_panel = spec.panels[1]
        for layer in factor_panel.layers:
            assert all(row["coded_level"] == 0.0 for row in layer.data)


class TestSteepestAscentPathEdgeCases:
    def test_no_coefficients(self) -> None:
        plot = create_plot("steepest_ascent_path")
        spec = plot.to_spec()
        assert "no path data" in spec.title.lower()

    def test_coefficients_without_factors(self, coefficients_2f: list) -> None:
        """Coefficients present but no factor names derivable anywhere."""
        plot = create_plot(
            "steepest_ascent_path",
            analysis_results={"coefficients": coefficients_2f},
        )
        spec = plot.to_spec()
        assert "no path data" in spec.title.lower()

    def test_all_zero_linear_coefficients(self) -> None:
        coefficients = [
            {"term": "Intercept", "coefficient": 40.0},
            {"term": "A", "coefficient": 0.0},
            {"term": "B", "coefficient": 0.0},
        ]
        plot = create_plot(
            "steepest_ascent_path",
            analysis_results={"coefficients": coefficients},
            factors_to_plot=["A", "B"],
        )
        spec = plot.to_spec()
        assert "no path data" in spec.title.lower()

    def test_precomputed_path_with_empty_steps(self) -> None:
        plot = create_plot(
            "steepest_ascent_path",
            analysis_results={"steepest_path": {"direction": "ascent", "steps": []}},
        )
        spec = plot.to_spec()
        assert "no steps computed" in spec.title.lower()

    def test_actual_response_overlay(self) -> None:
        """Steps carrying actual_response add a scatter overlay of observations."""
        path = {
            "direction": "ascent",
            "direction_vector": {"A": 1.0},
            "step_size": 0.5,
            "steps": [
                {"step": 0, "coded": {"A": 0.0}, "predicted_response": 40.0, "actual_response": 39.5},
                {"step": 1, "coded": {"A": 0.5}, "predicted_response": 43.0},
                {"step": 2, "coded": {"A": 1.0}, "predicted_response": 46.0, "actual_response": 45.2},
            ],
        }
        plot = create_plot(
            "steepest_ascent_path",
            analysis_results={"steepest_path": path},
        )
        spec = plot.to_spec()
        resp_layers = spec.panels[0].layers
        assert len(resp_layers) == 2
        actual_layer = resp_layers[1]
        assert actual_layer.name == "Actual"
        # Only the two steps with actual_response contribute points
        assert len(actual_layer.data) == 2
        assert actual_layer.data[0]["actual"] == pytest.approx(39.5)


# ---------------------------------------------------------------------------
# Effect plot edge cases
# ---------------------------------------------------------------------------


class TestMainEffectsPlotEdgeCases:
    def test_infer_response_from_y_column(self, design_data_2f: list) -> None:
        """Without response_column the 'y' column is used as the response."""
        plot = create_plot("main_effects", design_data=design_data_2f)
        spec = plot.to_spec()
        assert spec.plot_type == "main_effects"
        assert "Mean y" in spec.panels[0].y_title

    def test_infer_response_falls_back_to_last_column(self) -> None:
        data = [
            {"A": -1, "B": -1, "conversion": 28},
            {"A": 1, "B": -1, "conversion": 36},
            {"A": -1, "B": 1, "conversion": 18},
            {"A": 1, "B": 1, "conversion": 31},
        ]
        plot = create_plot("main_effects", design_data=data, factors_to_plot=["A", "B"])
        spec = plot.to_spec()
        assert "Mean conversion" in spec.panels[0].y_title

    def test_response_column_not_in_data(self, design_data_2f: list) -> None:
        plot = create_plot("main_effects", design_data=design_data_2f, response_column="missing")
        spec = plot.to_spec()
        assert "no response column" in spec.title.lower()

    def test_unknown_factor_skipped(self, design_data_2f: list) -> None:
        plot = create_plot(
            "main_effects",
            design_data=design_data_2f,
            response_column="y",
            factors_to_plot=["A", "Z"],
        )
        spec = plot.to_spec()
        # Z is not a data column: only the A layer is produced
        assert len(spec.panels[0].layers) == 1
        assert spec.panels[0].layers[0].name == "A"

    def test_intercept_only_formula_falls_back_to_columns(self, design_data_2f: list) -> None:
        """A 'y ~ 1' formula yields no factor names; columns are used instead."""
        plot = create_plot(
            "main_effects",
            design_data=design_data_2f,
            response_column="y",
            analysis_results={"model_summary": {"formula": "y ~ 1"}},
        )
        spec = plot.to_spec()
        names = sorted(layer.name for layer in spec.panels[0].layers)
        assert names == ["A", "B"]


class TestInteractionPlotEdgeCases:
    def test_no_data(self) -> None:
        plot = create_plot("interaction")
        spec = plot.to_spec()
        assert "no data" in spec.title.lower()

    def test_bad_response_column(self, design_data_2f: list) -> None:
        plot = create_plot(
            "interaction",
            design_data=design_data_2f,
            response_column="missing",
            factors_to_plot=["A", "B"],
        )
        spec = plot.to_spec()
        assert "no response column" in spec.title.lower()

    def test_factors_not_in_data(self, design_data_2f: list) -> None:
        plot = create_plot(
            "interaction",
            design_data=design_data_2f,
            response_column="y",
            factors_to_plot=["A", "Z"],
        )
        spec = plot.to_spec()
        assert "not in data" in spec.title.lower()

    def test_infer_response_from_y_column(self, design_data_2f: list) -> None:
        plot = create_plot(
            "interaction",
            design_data=design_data_2f,
            factors_to_plot=["A", "B"],
        )
        spec = plot.to_spec()
        assert spec.plot_type == "interaction"
        assert "Mean y" in spec.panels[0].y_title

    def test_infer_response_falls_back_to_last_column(self) -> None:
        data = [
            {"A": -1, "B": -1, "conversion": 28},
            {"A": 1, "B": -1, "conversion": 36},
            {"A": -1, "B": 1, "conversion": 18},
            {"A": 1, "B": 1, "conversion": 31},
        ]
        plot = create_plot("interaction", design_data=data, factors_to_plot=["A", "B"])
        spec = plot.to_spec()
        assert "Mean conversion" in spec.panels[0].y_title


class TestPerturbationPlotEdgeCases:
    def test_no_coefficients(self) -> None:
        plot = create_plot("perturbation")
        spec = plot.to_spec()
        assert "no coefficients" in spec.title.lower()

    def test_no_factors(self) -> None:
        plot = create_plot(
            "perturbation",
            analysis_results={"coefficients": [{"term": "Intercept", "coefficient": 1.0}]},
        )
        spec = plot.to_spec()
        assert "no factors" in spec.title.lower()

    def test_quadratic_term_with_hold_values(self, quadratic_coefficients: list) -> None:
        """Quadratic I(A ** 2) terms curve the sweep; other factors sit at hold values.

        For factor A at the sweep endpoints, with B held at 0.5:
        y(-1) = 80 - 5 - 3*0.5 - 4 + 1.5*(-1)*0.5 = 68.75
        y(+1) = 80 + 5 - 3*0.5 - 4 + 1.5*(+1)*0.5 = 80.25
        (the I(B ** 2) term contributes 0 because hold values are keyed by
        factor name, not by transformed term name).
        """
        plot = create_plot(
            "perturbation",
            analysis_results={"coefficients": quadratic_coefficients},
            factors_to_plot=["A", "B"],
            hold_values={"B": 0.5},
        )
        spec = plot.to_spec()
        layer_a = next(layer for layer in spec.panels[0].layers if layer.name == "A")
        assert layer_a.data[0]["predicted"] == pytest.approx(68.75)
        assert layer_a.data[-1]["predicted"] == pytest.approx(80.25)


# ---------------------------------------------------------------------------
# Significance plot edge cases
# ---------------------------------------------------------------------------


class TestHalfNormalPlotEdgeCases:
    def test_empty_effects(self) -> None:
        plot = create_plot("half_normal", analysis_results={"effects": {}})
        spec = plot.to_spec()
        assert "no effects data" in spec.title.lower()

    def test_single_effect(self) -> None:
        plot = create_plot("half_normal", analysis_results={"effects": {"A": 5.0}})
        spec = plot.to_spec()
        assert spec.plot_type == "half_normal"
        ref_layer = spec.panels[0].layers[1]
        # Degenerate reference line from (0, 0) to (1, |effect|)
        assert ref_layer.data[0] == {"quantile": 0, "abs_effect": 0}
        assert ref_layer.data[1]["abs_effect"] == pytest.approx(5.0)

    def test_all_but_one_significant(self, lenth_data: dict) -> None:
        """With only one non-significant effect, the fallback line is used."""
        lenth = dict(lenth_data)
        lenth["effects"] = [{"term": "A", "effect": 8.0, "active_ME": True}]
        plot = create_plot(
            "half_normal",
            analysis_results={"effects": {"A": 8.0, "B": -4.0}, "lenth_method": lenth},
        )
        spec = plot.to_spec()
        ref_layer = spec.panels[0].layers[1]
        # Fallback: line through origin up to the largest absolute effect
        assert ref_layer.data[0]["abs_effect"] == 0
        assert ref_layer.data[1]["abs_effect"] == pytest.approx(8.0)

    def test_lenth_without_me_key(self) -> None:
        """A lenth dict without ME produces no threshold annotation."""
        plot = create_plot(
            "half_normal",
            analysis_results={
                "effects": {"A": 8.0, "B": -4.0},
                "lenth_method": {"effects": [{"term": "A", "effect": 8.0, "active_ME": True}]},
            },
        )
        spec = plot.to_spec()
        assert len(spec.panels[0].annotations) == 0


class TestDanielPlotEdgeCases:
    def test_empty_effects(self) -> None:
        plot = create_plot("daniel", analysis_results={"effects": {}})
        spec = plot.to_spec()
        assert "no effects data" in spec.title.lower()

    def test_single_effect(self) -> None:
        plot = create_plot("daniel", analysis_results={"effects": {"A": 5.0}})
        spec = plot.to_spec()
        assert spec.plot_type == "daniel"
        ref_layer = spec.panels[0].layers[1]
        # Degenerate horizontal reference line at the single effect value
        assert [row["quantile"] for row in ref_layer.data] == [-2, 2]
        assert all(row["effect"] == pytest.approx(5.0) for row in ref_layer.data)


class TestParetoPlotEdgeCases:
    def test_lenth_with_only_me(self, two_factor_effects: dict) -> None:
        plot = create_plot(
            "pareto",
            analysis_results={"effects": two_factor_effects, "lenth_method": {"ME": 3.0}},
        )
        spec = plot.to_spec()
        assert len(spec.panels[0].annotations) == 1

    def test_lenth_with_only_sme(self, two_factor_effects: dict) -> None:
        plot = create_plot(
            "pareto",
            analysis_results={"effects": two_factor_effects, "lenth_method": {"SME": 5.0}},
        )
        spec = plot.to_spec()
        anns = spec.panels[0].annotations
        assert len(anns) == 1
        assert "SME" in (anns[0].label or "")

    def test_std_errors_with_no_matching_terms(self, two_factor_effects: dict) -> None:
        """A std-error dict that matches no term yields no error bars."""
        plot = create_plot(
            "pareto",
            analysis_results={
                "effects": two_factor_effects,
                "effect_std_errors": {"Z": 1.0},
            },
        )
        spec = plot.to_spec()
        assert "error_y" not in spec.panels[0].layers[0].style


# ---------------------------------------------------------------------------
# Diagnostic plot edge cases
# ---------------------------------------------------------------------------


class TestDiagnosticPlotEdgeCases:
    @pytest.mark.parametrize("plot_type", [
        "residuals_vs_fitted", "normal_probability", "residuals_vs_order",
    ])
    def test_empty_diagnostics_guard(self, plot_type: str) -> None:
        plot = create_plot(plot_type, analysis_results={})
        spec = plot.to_spec()
        assert "no data" in spec.title.lower()


class TestBoxCoxPlotEdgeCases:
    def test_insufficient_values(self) -> None:
        data = [{"y": 5.0}, {"y": 6.0}]
        plot = create_plot("box_cox", design_data=data, response_column="y")
        spec = plot.to_spec()
        assert "insufficient data" in spec.title.lower()

    def test_non_positive_response(self) -> None:
        data = [{"y": 5.0}, {"y": 0.0}, {"y": 6.0}, {"y": 7.0}]
        plot = create_plot("box_cox", design_data=data, response_column="y")
        spec = plot.to_spec()
        assert "requires all positive" in spec.title.lower()

    def test_response_reconstructed_from_fitted_and_residuals(
        self,
        residual_diagnostics: dict,
    ) -> None:
        """Without design data the response is rebuilt as fitted + residuals."""
        plot = create_plot(
            "box_cox",
            analysis_results={"residual_diagnostics": residual_diagnostics},
        )
        spec = plot.to_spec()
        assert spec.plot_type == "box_cox"
        assert "optimal_lambda" in spec.metadata

    def test_response_column_missing_from_design_data(self) -> None:
        """A response column absent from the design columns yields no values."""
        data = [{"z": 5.0}, {"z": 6.0}, {"z": 7.0}]
        plot = create_plot("box_cox", design_data=data, response_column="y")
        spec = plot.to_spec()
        assert "insufficient data" in spec.title.lower()

    def test_mismatched_fitted_and_residual_lengths(self) -> None:
        plot = create_plot(
            "box_cox",
            analysis_results={
                "residual_diagnostics": {
                    "fitted_values": [10.0, 11.0, 12.0],
                    "residuals": [0.5, -0.5],
                },
            },
        )
        spec = plot.to_spec()
        assert "insufficient data" in spec.title.lower()


# ---------------------------------------------------------------------------
# Surface plot edge cases
# ---------------------------------------------------------------------------


class TestSurfacePlotEdgeCases:
    def test_contour_no_coefficients(self) -> None:
        plot = create_plot("contour")
        spec = plot.to_spec()
        assert "no coefficients" in spec.title.lower()

    def test_contour_single_factor(self, coefficients_2f: list) -> None:
        plot = create_plot(
            "contour",
            analysis_results={"coefficients": coefficients_2f},
            factors_to_plot=["A"],
        )
        spec = plot.to_spec()
        assert "need at least 2" in spec.title.lower()

    def test_surface_3d_no_coefficients(self) -> None:
        plot = create_plot("surface_3d")
        spec = plot.to_spec()
        assert "no coefficients" in spec.title.lower()

    def test_surface_3d_single_factor(self, coefficients_2f: list) -> None:
        plot = create_plot(
            "surface_3d",
            analysis_results={"coefficients": coefficients_2f},
            factors_to_plot=["A"],
        )
        spec = plot.to_spec()
        assert "need at least 2" in spec.title.lower()

    def test_no_scatter_overlay_when_rows_lack_plotted_factors(
        self,
        coefficients_2f: list,
    ) -> None:
        """Design rows without the plotted factor columns produce no overlay."""
        plot = create_plot(
            "contour",
            analysis_results={"coefficients": coefficients_2f},
            design_data=[{"X": 1.0, "y": 2.0}],
            response_column="y",
            factors_to_plot=["A", "B"],
        )
        spec = plot.to_spec()
        assert len(spec.panels[0].layers) == 1

    def test_hover_with_non_numeric_values(self, coefficients_2f: list) -> None:
        """Non-numeric factor and response values fall back to str() in hover."""
        design_data = [
            {"A": -1, "B": -1, "batch": "low", "y": "n/a"},
            {"A": 1, "B": 1, "batch": "high", "y": "n/a"},
        ]
        plot = create_plot(
            "contour",
            analysis_results={"coefficients": coefficients_2f},
            design_data=design_data,
            response_column="y",
            factors_to_plot=["A", "B"],
        )
        spec = plot.to_spec()
        scatter = spec.panels[0].layers[1]
        hover = scatter.data[0]["hover"]
        assert "batch: low" in hover
        assert "y: n/a" in hover

    def test_overlay_without_response_column(self, coefficients_2f: list) -> None:
        """Design points without a response column omit the response field."""
        design_data = [{"A": -1, "B": -1}, {"A": 1, "B": 1}]
        plot = create_plot(
            "contour",
            analysis_results={"coefficients": coefficients_2f},
            design_data=design_data,
            factors_to_plot=["A", "B"],
        )
        spec = plot.to_spec()
        scatter = spec.panels[0].layers[1]
        assert len(scatter.data) == 2
        assert all("response" not in point for point in scatter.data)
        assert "A" in scatter.data[0]["hover"]


class TestPredictionVarianceEdgeCases:
    def test_no_design_data(self) -> None:
        plot = create_plot("prediction_variance")
        spec = plot.to_spec()
        assert "no design data" in spec.title.lower()

    def test_single_factor(self, design_data_2f: list) -> None:
        plot = create_plot(
            "prediction_variance",
            design_data=design_data_2f,
            response_column="y",
            factors_to_plot=["A"],
        )
        spec = plot.to_spec()
        assert "need at least 2" in spec.title.lower()

    def test_three_factor_design_with_hold_values(self, design_data_3f: list) -> None:
        """The third (non-plotted) factor is held at its hold value.

        Factors are derived from the data columns so that the model matrix
        includes C, which is then pinned at 0.5 while A and B are swept.
        """
        plot = create_plot(
            "prediction_variance",
            design_data=design_data_3f,
            response_column="y",
            hold_values={"C": 0.5},
        )
        spec = plot.to_spec()
        assert spec.plot_type == "prediction_variance"
        z = spec.panels[0].layers[0].style["z_matrix"]
        assert len(z) == 40
        assert len(z[0]) == 40


# ---------------------------------------------------------------------------
# Design quality plot edge cases
# ---------------------------------------------------------------------------


class TestFDSPlotEdgeCases:
    def test_no_design_data(self) -> None:
        plot = create_plot("fds_plot")
        spec = plot.to_spec()
        assert "no design data" in spec.title.lower()

    def test_single_factor(self, design_data_2f: list) -> None:
        plot = create_plot(
            "fds_plot",
            design_data=design_data_2f,
            response_column="y",
            factors_to_plot=["A"],
        )
        spec = plot.to_spec()
        assert "need at least 2" in spec.title.lower()


class TestPowerCurvePlotEdgeCases:
    def test_no_design_information(self) -> None:
        plot = create_plot("power_curve")
        spec = plot.to_spec()
        assert "no design information" in spec.title.lower()

    def test_design_info_from_model_summary(self) -> None:
        plot = create_plot(
            "power_curve",
            analysis_results={"model_summary": {"n_obs": 8, "n_params": 4}},
        )
        spec = plot.to_spec()
        assert spec.plot_type == "power_curve"
        assert spec.metadata["n_runs"] == 8
        assert spec.metadata["n_terms"] == 4
        assert spec.metadata["residual_df"] == 4

    def test_model_summary_with_zero_counts(self) -> None:
        plot = create_plot(
            "power_curve",
            analysis_results={"model_summary": {"n_obs": 0, "n_params": 0}},
        )
        spec = plot.to_spec()
        assert "no design information" in spec.title.lower()


# ---------------------------------------------------------------------------
# Cube and square plot edge cases
# ---------------------------------------------------------------------------


class TestCubePlotEdgeCases:
    def test_no_data_source(self) -> None:
        plot = create_plot("cube_plot", factors_to_plot=["A", "B", "C"])
        spec = plot.to_spec()
        assert "cannot compute vertex values" in spec.title.lower()

    def test_raw_data_fallback_with_missing_vertex(self, design_data_3f: list) -> None:
        """A vertex absent from the raw data gets a NaN value."""
        # Drop the (-1, -1, -1) run: the first product([-1, 1], repeat=3) vertex
        data = design_data_3f[1:]
        plot = create_plot(
            "cube_plot",
            design_data=data,
            response_column="y",
            factors_to_plot=["A", "B", "C"],
        )
        spec = plot.to_spec()
        vertices = spec.metadata["vertices"]
        assert vertices[0]["levels"] == [-1, -1, -1]
        assert math.isnan(vertices[0]["value"])
        # The remaining vertices are still populated
        assert all(not math.isnan(v["value"]) for v in vertices[1:])

    def test_raw_data_missing_factor_column(self, design_data_2f: list) -> None:
        """Design data lacking the C column cannot supply vertex values."""
        plot = create_plot(
            "cube_plot",
            design_data=design_data_2f,
            response_column="y",
            factors_to_plot=["A", "B", "C"],
        )
        spec = plot.to_spec()
        assert "cannot compute vertex values" in spec.title.lower()


class TestSquarePlotEdgeCases:
    def test_no_data_source(self) -> None:
        plot = create_plot("square_plot", factors_to_plot=["A", "B"])
        spec = plot.to_spec()
        assert "cannot compute vertex values" in spec.title.lower()

    def test_raw_data_fallback_with_missing_vertex(self, design_data_2f: list) -> None:
        """A vertex absent from the raw data gets a NaN value."""
        data = [r for r in design_data_2f if not (r["A"] == -1 and r["B"] == -1)]
        plot = create_plot(
            "square_plot",
            design_data=data,
            response_column="y",
            factors_to_plot=["A", "B"],
        )
        spec = plot.to_spec()
        vertices = spec.metadata["vertices"]
        assert vertices[0]["levels"] == [-1, -1]
        assert math.isnan(vertices[0]["value"])
        assert all(not math.isnan(v["value"]) for v in vertices[1:])

    def test_raw_data_missing_factor_column(self) -> None:
        """Design data lacking the B column cannot supply vertex values."""
        data = [{"A": -1, "y": 10.0}, {"A": 1, "y": 12.0}]
        plot = create_plot(
            "square_plot",
            design_data=data,
            response_column="y",
            factors_to_plot=["A", "B"],
        )
        spec = plot.to_spec()
        assert "cannot compute vertex values" in spec.title.lower()


# ---------------------------------------------------------------------------
# Registry factor-name inference edge cases
# ---------------------------------------------------------------------------


class TestFactorNameInference:
    def test_formula_without_design_data_uses_blocklist(self) -> None:
        """With no design data the static reserved-word blocklist applies."""
        plot = create_plot(
            "pareto",
            analysis_results={
                "effects": {"A": 1.0, "B": -2.0},
                "model_summary": {"formula": "y ~ A + np.power(B, 2) + I(A ** 2)"},
            },
        )
        names = plot._get_factor_names()
        assert "A" in names
        assert "B" in names
        for noise in ("np", "power", "I"):
            assert noise not in names

    def test_design_data_with_only_response_column(self) -> None:
        plot = create_plot(
            "pareto",
            analysis_results={"effects": {}},
            design_data=[{"y": 1.0}],
            response_column="y",
        )
        assert plot._get_factor_names() == []

    def test_formula_without_tilde_falls_back_to_design_data(self) -> None:
        plot = create_plot(
            "pareto",
            analysis_results={
                "effects": {"A": 1.0},
                "model_summary": {"formula": "not-a-formula"},
            },
            design_data=[{"A": -1, "y": 0.0}],
            response_column="y",
        )
        assert plot._get_factor_names() == ["A"]
