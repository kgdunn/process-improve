"""Tests for Tool 6: visualize_doe and all DOE plot types.

Covers ChartSpec IR construction, Plotly adapter output, ECharts adapter
output, JSON round-trip serialisability, and the public ``visualize_doe``
entry point.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from process_improve.experiments.visualization import visualize_doe
from process_improve.experiments.visualization.adapters import EChartsAdapter, PlotlyAdapter
from process_improve.experiments.visualization.plots.registry import (
    create_plot,
    get_available_plot_types,
)
from process_improve.experiments.visualization.spec import ChartSpec
from process_improve.experiments.visualization.types import DOEPlotType

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
    def test_all_20_plot_types_registered(self) -> None:
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
