"""Tests that exercise the LLM tool wrappers in ``experiments/tools.py``.

The wrappers themselves are thin try/except shells around already-tested
APIs, but they were previously uncovered because no test invoked them
through ``execute_tool_call``. Each test here drives one wrapper end to
end (success path) and, where cheap, also exercises its except-branch.
"""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest

# execute_tool_call calls discover_tools(), which imports
# process_improve.experiments.tools and triggers all @tool_spec
# registrations - no explicit import is needed here.
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
        """A bad n_factors is rejected by the pydantic Field constraint."""
        from process_improve.tool_safety import ToolInputInvalidError

        with pytest.raises(ToolInputInvalidError):
            execute_tool_call("create_factorial_design", {"n_factors": 0})


# ---------------------------------------------------------------------------
# fit_linear_model
# ---------------------------------------------------------------------------


_SAFE_FIT_DATA = [
    {"A": -1, "B": -1, "y": 28.0},
    {"A": 1, "B": -1, "y": 36.0},
    {"A": -1, "B": 1, "y": 18.0},
    {"A": 1, "B": 1, "y": 31.0},
]


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

    def test_main_effects_only(self) -> None:
        """A '+' main-effects formula should still fit."""
        result = execute_tool_call(
            "fit_linear_model",
            {
                "formula": "y ~ A + B",
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

    # ---- SEC-01: patsy-formula code-execution guard -----------------------
    # Patsy evaluates formula terms as Python, so an untrusted formula is an
    # RCE vector. The wrapper now rejects anything that is not a plain
    # Wilkinson formula over the data columns, *before* it reaches patsy, so
    # these never hit patsy's eval (which also avoids the Python 3.13
    # traceback INTERNALERROR seen previously).

    def test_rce_formula_is_rejected_without_side_effect(self, tmp_path) -> None:
        """A malicious formula must return an error and must not execute code."""
        sentinel = tmp_path / "pwned"
        malicious = f"y ~ A + I(__import__('os').system('touch {sentinel}'))"
        result = execute_tool_call(
            "fit_linear_model",
            {"formula": malicious, "data": _SAFE_FIT_DATA},
        )
        assert "error" in result
        assert not sentinel.exists(), "formula was evaluated - RCE guard failed"

    def test_unknown_identifier_is_rejected(self) -> None:
        """A formula referencing a non-column name (e.g. numpy) is rejected."""
        result = execute_tool_call(
            "fit_linear_model",
            {"formula": "y ~ A + np", "data": _SAFE_FIT_DATA},
        )
        assert "error" in result
        assert "unknown name" in result["error"]


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

    def test_end_to_end_on_real_ldpe_dataset(self) -> None:
        """analyze_experiment runs end-to-end via execute_tool_call on the real LDPE data."""
        csv_path = (
            pathlib.Path(__file__).parents[1]
            / "src" / "process_improve" / "datasets" / "multivariate" / "LDPE" / "LDPE.csv"
        )
        ldpe = pd.read_csv(csv_path, index_col=0)
        factors_and_response = ["Tin", "Tmax1", "z1", "Conv"]
        design_matrix = [
            {key: float(value) for key, value in row.items()}
            for row in ldpe[factors_and_response].to_dict(orient="records")
        ]

        result = execute_tool_call(
            "analyze_experiment",
            {
                "design_matrix": design_matrix,
                "response_column": "Conv",
                "model": "main_effects",
                "analysis_type": ["coefficients", "residual_diagnostics"],
            },
        )

        assert "error" not in result, result
        summary = result["model_summary"]
        assert summary["n_obs"] == len(ldpe)
        assert 0.0 <= summary["r_squared"] <= 1.0
        assert len(result["coefficients"]) > 0


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
        """Unknown method is rejected by the pydantic Literal."""
        from process_improve.tool_safety import ToolInputInvalidError

        with pytest.raises(ToolInputInvalidError):
            execute_tool_call(
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
        """An unknown augmentation type is rejected by the pydantic Literal."""
        from process_improve.tool_safety import ToolInputInvalidError

        with pytest.raises(ToolInputInvalidError):
            execute_tool_call(
                "augment_design",
                {
                    "existing_design": [
                        {"A": -1.0, "B": -1.0},
                        {"A": 1.0, "B": 1.0},
                    ],
                    "augmentation_type": "not_a_real_type",
                },
            )


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
        """An unknown plot type is rejected by the pydantic Literal."""
        from process_improve.tool_safety import ToolInputInvalidError

        with pytest.raises(ToolInputInvalidError):
            execute_tool_call(
                "visualize_doe",
                {"plot_type": "definitely_not_a_plot", "analysis_results": {}},
            )


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
