"""Tests that exercise the LLM tool wrappers in ``experiments/tools.py``.

The wrappers themselves are thin try/except shells around already-tested
APIs, but they were previously uncovered because no test invoked them
through ``execute_tool_call``. Each test here drives one wrapper end to
end (success path) and, where cheap, also exercises its except-branch.
"""

from __future__ import annotations

# Import the experiments tools module so its @tool_spec registrations run.
import process_improve.experiments.tools  # noqa: F401
from process_improve.tool_spec import execute_tool_call

# ---------------------------------------------------------------------------
# create_factorial_design
# ---------------------------------------------------------------------------


class TestCreateFactorialDesign:
    def test_basic_three_factor(self) -> None:
        """Three-factor full factorial should return 8 runs."""
        result = execute_tool_call(
            "create_factorial_design",
            {"n_factors": 3, "factor_names": ["Temperature", "Pressure", "Time"]},
        )
        assert "error" not in result
        assert result["n_runs"] == 8
        assert result["n_factors"] == 3
        # Names may have a "[coded]" suffix - just check the supplied stems are present.
        joined = " ".join(result["factor_names"])
        for stem in ("Temperature", "Pressure", "Time"):
            assert stem in joined

    def test_default_factor_names(self) -> None:
        """Without explicit names the wrapper should still succeed."""
        result = execute_tool_call("create_factorial_design", {"n_factors": 2})
        assert "error" not in result
        assert result["n_runs"] == 4
        assert len(result["factor_names"]) == 2

    def test_invalid_n_factors_returns_error(self) -> None:
        """A bad n_factors should be reported via the error key, not raised."""
        # n_factors of 0 will trip pyDOE3 / the underlying call.
        result = execute_tool_call("create_factorial_design", {"n_factors": 0})
        assert "error" in result


# ---------------------------------------------------------------------------
# fit_linear_model
# ---------------------------------------------------------------------------


class TestFitLinearModel:
    def test_two_factor_factorial_fit(self) -> None:
        """A 2^2 factorial should fit cleanly."""
        result = execute_tool_call(
            "fit_linear_model",
            {
                "formula": "y ~ A*B",
                "data": [
                    {"A": -1, "B": -1, "y": 28.0},
                    {"A": 1, "B": -1, "y": 36.0},
                    {"A": -1, "B": 1, "y": 18.0},
                    {"A": 1, "B": 1, "y": 31.0},
                ],
            },
        )
        assert "error" not in result
        assert "coefficients" in result
        assert "r2" in result
        assert "summary_text" in result
        assert isinstance(result["summary_text"], str)

    def test_invalid_formula_returns_error(self) -> None:
        """A formula that references an absent column should return an error dict."""
        result = execute_tool_call(
            "fit_linear_model",
            {"formula": "y ~ MISSING", "data": [{"A": -1, "y": 1.0}, {"A": 1, "y": 2.0}]},
        )
        assert "error" in result


# ---------------------------------------------------------------------------
# generate_design
# ---------------------------------------------------------------------------


class TestGenerateDesign:
    def test_full_factorial(self) -> None:
        """A 3-factor full factorial via the tool should give 8 + 3 center pts."""
        result = execute_tool_call(
            "generate_design",
            {
                "factors": [
                    {"name": "A", "low": 0, "high": 10},
                    {"name": "B", "low": 0, "high": 10},
                    {"name": "C", "low": 0, "high": 10},
                ],
                "design_type": "full_factorial",
                "center_points": 0,
            },
        )
        assert "error" not in result
        assert result["n_runs"] == 8
        assert result["n_factors"] == 3
        assert "design_coded" in result
        assert "design_actual" in result
        assert "run_order" in result

    def test_fractional_factorial_with_resolution(self) -> None:
        """A 5-factor resolution III fractional factorial fires the optional-metadata
        branches in the wrapper (generators / defining_relation / resolution).
        """
        result = execute_tool_call(
            "generate_design",
            {
                "factors": [{"name": n, "low": 0, "high": 10} for n in "ABCDE"],
                "design_type": "fractional_factorial",
                "resolution": 3,
                "center_points": 0,
            },
        )
        assert "error" not in result
        assert result["n_runs"] == 8
        # At least one of the optional-metadata branches should fire.
        assert "generators" in result or "defining_relation" in result or "resolution" in result

    def test_ccd_alpha_branch(self) -> None:
        """A CCD design exercises the alpha-output branch."""
        result = execute_tool_call(
            "generate_design",
            {
                "factors": [
                    {"name": "T", "low": 150, "high": 200, "units": "degC"},
                    {"name": "P", "low": 1, "high": 5, "units": "bar"},
                ],
                "design_type": "ccd",
                "alpha": "rotatable",
                "center_points": 0,
            },
        )
        assert "error" not in result
        assert "alpha" in result

    def test_invalid_factor_returns_error(self) -> None:
        """A continuous factor without low/high should be reported as an error."""
        result = execute_tool_call(
            "generate_design",
            {"factors": [{"name": "broken"}]},
        )
        assert "error" in result


# ---------------------------------------------------------------------------
# evaluate_design
# ---------------------------------------------------------------------------


class TestEvaluateDesign:
    def test_d_efficiency_of_simple_factorial(self) -> None:
        """Evaluate D-efficiency on a 2^2 factorial via the tool wrapper."""
        result = execute_tool_call(
            "evaluate_design",
            {
                "design_matrix": [
                    {"A": -1, "B": -1},
                    {"A": 1, "B": -1},
                    {"A": -1, "B": 1},
                    {"A": 1, "B": 1},
                ],
                "metric": "d_efficiency",
                "model": "interactions",
            },
        )
        assert "error" not in result
        # d_efficiency should appear somewhere in the result dict.
        flat = str(result)
        assert "d_efficiency" in flat or "D-efficiency" in flat

    def test_invalid_metric_returns_error(self) -> None:
        """An unknown metric name should be reported as an error."""
        result = execute_tool_call(
            "evaluate_design",
            {
                "design_matrix": [
                    {"A": -1, "B": -1},
                    {"A": 1, "B": -1},
                    {"A": -1, "B": 1},
                    {"A": 1, "B": 1},
                ],
                "metric": "definitely_not_a_real_metric_name",
            },
        )
        assert "error" in result


# ---------------------------------------------------------------------------
# analyze_experiment
# ---------------------------------------------------------------------------


class TestAnalyzeExperiment:
    def test_anova_on_factorial(self) -> None:
        """ANOVA on a replicated 2^2 design should succeed."""
        result = execute_tool_call(
            "analyze_experiment",
            {
                "design_matrix": [
                    {"A": -1, "B": -1, "y": 28.0},
                    {"A": 1, "B": -1, "y": 36.0},
                    {"A": -1, "B": 1, "y": 18.0},
                    {"A": 1, "B": 1, "y": 31.0},
                    {"A": -1, "B": -1, "y": 27.5},
                    {"A": 1, "B": -1, "y": 36.5},
                    {"A": -1, "B": 1, "y": 17.5},
                    {"A": 1, "B": 1, "y": 31.5},
                ],
                "response_column": "y",
                "model": "y ~ A*B",
                "analysis_type": "anova",
            },
        )
        assert "error" not in result

    def test_invalid_response_column_returns_error(self) -> None:
        """Referencing a missing response column should return an error dict."""
        result = execute_tool_call(
            "analyze_experiment",
            {
                "design_matrix": [{"A": -1, "y": 1.0}, {"A": 1, "y": 2.0}],
                "response_column": "MISSING",
                "analysis_type": "anova",
            },
        )
        assert "error" in result


# ---------------------------------------------------------------------------
# optimize_responses
# ---------------------------------------------------------------------------


class TestOptimizeResponses:
    def test_stationary_point_quadratic(self) -> None:
        """Stationary-point optimisation on a small quadratic model."""
        result = execute_tool_call(
            "optimize_responses",
            {
                "fitted_models": [
                    {
                        "response_name": "yield",
                        "factor_names": ["A", "B"],
                        "coefficients": [
                            {"term": "Intercept", "coefficient": 40.0},
                            {"term": "A", "coefficient": 5.25},
                            {"term": "B", "coefficient": -2.0},
                            {"term": "I(A ** 2)", "coefficient": -3.0},
                            {"term": "I(B ** 2)", "coefficient": -1.5},
                            {"term": "A:B", "coefficient": 1.5},
                        ],
                    }
                ],
                "method": "stationary_point",
            },
        )
        assert "error" not in result

    def test_invalid_method_returns_error(self) -> None:
        """Unknown method should be reported as an error."""
        result = execute_tool_call(
            "optimize_responses",
            {
                "fitted_models": [
                    {
                        "response_name": "y",
                        "factor_names": ["A"],
                        "coefficients": [
                            {"term": "Intercept", "coefficient": 1.0},
                            {"term": "A", "coefficient": 2.0},
                        ],
                    }
                ],
                "method": "definitely_not_a_method",
            },
        )
        assert "error" in result


# ---------------------------------------------------------------------------
# augment_design
# ---------------------------------------------------------------------------


class TestAugmentDesign:
    def test_add_center_points(self) -> None:
        """Adding center points to a 2^2 design should succeed."""
        result = execute_tool_call(
            "augment_design",
            {
                "existing_design": [
                    {"A": -1.0, "B": -1.0},
                    {"A": 1.0, "B": -1.0},
                    {"A": -1.0, "B": 1.0},
                    {"A": 1.0, "B": 1.0},
                ],
                "augmentation_type": "add_center_points",
                "n_additional_runs": 3,
            },
        )
        assert "error" not in result

    def test_invalid_augmentation_type_returns_error(self) -> None:
        """An unknown augmentation type should return an error dict."""
        result = execute_tool_call(
            "augment_design",
            {
                "existing_design": [
                    {"A": -1.0, "B": -1.0},
                    {"A": 1.0, "B": 1.0},
                ],
                "augmentation_type": "not_a_real_type",
            },
        )
        assert "error" in result


# ---------------------------------------------------------------------------
# visualize_doe
# ---------------------------------------------------------------------------


class TestVisualizeDoe:
    def test_pareto_from_effects(self) -> None:
        """A pareto plot from a small effects dict should render."""
        result = execute_tool_call(
            "visualize_doe",
            {
                "plot_type": "pareto",
                "analysis_results": {
                    "effects": {"A": 5.2, "B": -3.1, "A:B": 1.0},
                },
            },
        )
        assert "error" not in result

    def test_invalid_plot_type_returns_error(self) -> None:
        """An unknown plot type should be reported as an error."""
        result = execute_tool_call(
            "visualize_doe",
            {"plot_type": "definitely_not_a_plot", "analysis_results": {}},
        )
        assert "error" in result


# ---------------------------------------------------------------------------
# doe_knowledge
# ---------------------------------------------------------------------------


class TestDoeKnowledge:
    def test_concept_query(self) -> None:
        """A concept query should return a non-error dict."""
        result = execute_tool_call(
            "doe_knowledge",
            {"query": "What is design resolution?", "topic": "statistical_concepts"},
        )
        assert "error" not in result

    def test_design_selection_with_context(self) -> None:
        """A design-selection query with context should succeed."""
        result = execute_tool_call(
            "doe_knowledge",
            {
                "query": "screening 7 factors 15 runs",
                "topic": "design_selection",
                "context": {
                    "n_factors": 7,
                    "budget": 15,
                    "goal": "screening",
                },
            },
        )
        assert "error" not in result


# ---------------------------------------------------------------------------
# recommend_strategy
# ---------------------------------------------------------------------------


class TestRecommendStrategy:
    def test_simple_strategy(self) -> None:
        """A simple strategy request should succeed."""
        result = execute_tool_call(
            "recommend_strategy",
            {
                "factors": [
                    {"name": "Temperature", "low": 150, "high": 200, "units": "degC"},
                    {"name": "Pressure", "low": 1, "high": 5, "units": "bar"},
                    {"name": "Time", "low": 10, "high": 60, "units": "min"},
                ],
                "responses": [{"name": "Yield", "goal": "maximize"}],
                "budget": 30,
                "domain": "general",
            },
        )
        assert "error" not in result

    def test_invalid_factor_returns_error(self) -> None:
        """A continuous factor missing low/high should be reported as an error."""
        result = execute_tool_call(
            "recommend_strategy",
            {"factors": [{"name": "broken"}]},
        )
        assert "error" in result
