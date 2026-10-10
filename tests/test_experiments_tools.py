"""Tests that exercise the LLM tool wrappers in ``experiments/tools.py``.

The wrappers themselves are thin try/except shells around already-tested
APIs, but they were previously uncovered because no test invoked them
through ``execute_tool_call``. Each test here drives one wrapper end to
end (success path) and, where cheap, also exercises its except-branch.
"""

from __future__ import annotations

import json
import pathlib

import pandas as pd
import pytest

from process_improve.experiments import datasets

# execute_tool_call calls discover_tools(), which imports
# process_improve.experiments.tools and triggers all @tool_spec
# registrations - no explicit import is needed here.
from process_improve.tool_safety import ToolInputInvalidError
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
        assert result["factor_names"] == ["Temperature", "Pressure", "Time"]
        assert set(result["design"][0]) == {"Temperature", "Pressure", "Time"}

    def test_default_factor_names(self) -> None:
        """Without explicit names the wrapper should still succeed."""
        result = execute_tool_call("create_factorial_design", {"n_factors": 2})
        assert "error" not in result
        assert result["n_runs"] == 4
        assert result["factor_names"] == ["A", "B"]
        assert set(result["design"][0]) == {"A", "B"}

    @pytest.mark.parametrize("names", [["T", "P"], ["T", "P", "Q", "R"], ["T", "T", "P"]])
    def test_names_must_match_n_factors(self, names: list[str]) -> None:
        """Two names with n_factors=3 used to give a 2-factor design reported as 3 factors."""
        from process_improve.tool_safety import ToolInputInvalidError

        with pytest.raises(ToolInputInvalidError):
            execute_tool_call("create_factorial_design", {"n_factors": 3, "factor_names": names})

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
                "n_center_points": 0,
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
                "n_center_points": 0,
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
                "n_center_points": 0,
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

    def test_split_plot_with_a_named_whole_plot_column(self) -> None:
        """The tool passes ``whole_plot`` through: BHH's corrosion heats, temperature tested on 3 df."""
        rows = datasets.corrosion().drop(columns="Position").astype({"Temperature": str}).to_dict("records")
        result = execute_tool_call(
            "analyze_experiment",
            {
                "design_matrix": rows,
                "response_column": "Resistance",
                "analysis_type": "split_plot",
                "whole_plot": "Heat",
            },
        )
        tests = {row["source"]: row for row in result["split_plot"]["tests"]}
        assert tests["Temperature"]["df_denominator"] == pytest.approx(3.0)
        assert result["split_plot"]["significant_terms"] == ["Coating", "Temperature:Coating"]

    def test_coded_coefficients_hand_off_to_optimize_responses(self) -> None:
        """An agent analysing actual units passes coding='coded', and the optimum comes back in actual units."""
        rows = [
            {"T": t, "P": p, "y": y}
            for t, p, y in [
                (150, 1, 10.0),
                (200, 1, 18.0),
                (150, 3, 13.0),
                (200, 3, 25.0),
                (175, 2, 16.0),
                (175, 2, 16.8),
            ]
        ]
        request = {"design_matrix": rows, "response_column": "y", "model": "y ~ T * P", "analysis_type": "coefficients"}
        goal = [{"goal": "maximize", "low": 10, "high": 25}]
        actual = execute_tool_call("analyze_experiment", request)
        refused = execute_tool_call("optimize_responses", {"fitted_models": [actual], "goals": goal})
        assert "coding='coded'" in refused["error"]
        coded = execute_tool_call("analyze_experiment", {**request, "coding": "coded"})
        best = execute_tool_call("optimize_responses", {"fitted_models": [coded], "goals": goal})
        assert best["desirability"]["optimal_actual"] == pytest.approx({"T": 200.0, "P": 3.0})

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
            / "src"
            / "process_improve"
            / "datasets"
            / "multivariate"
            / "LDPE"
            / "LDPE.csv"
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


@pytest.mark.parametrize(
    ("tool", "args"),
    [
        ("generate_design", {"random_seed": -1}),
        ("generate_design", {"random_state": -1}),
        ("augment_design", {"random_state": -1}),
        ("evaluate_design", {"random_state": -1}),
        ("optimize_responses", {"random_state": -1}),
    ],
)
def test_negative_seed_is_rejected_by_the_schema(tool: str, args: dict) -> None:
    """A negative seed reached numpy and came back as 'expected non-negative integer'."""
    base = {
        "generate_design": {"factors": [{"name": "A", "low": 0, "high": 1}, {"name": "B", "low": 0, "high": 1}]},
        "augment_design": {"existing_design": [{"A": -1, "B": -1}, {"A": 1, "B": 1}], "augmentation_type": "foldover"},
        "evaluate_design": {"design_matrix": [{"A": -1, "B": -1}, {"A": 1, "B": 1}]},
        "optimize_responses": {
            "fitted_models": [
                {"factor_names": ["A"], "coefficients": [{"term": "A", "coefficient": 1.0}], "response_name": "y"}
            ],
            "method": "steepest_ascent",
        },
    }[tool]
    with pytest.raises(ToolInputInvalidError):
        execute_tool_call(tool, {**base, **args})


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

    @staticmethod
    def _rising_plane_call(**extra: object) -> dict:
        """Maximize a plane rising with A, so the optimum sits at the upper bound."""
        return execute_tool_call(
            "optimize_responses",
            {
                "fitted_models": [
                    {
                        "response_name": "y",
                        "factor_names": ["A", "B"],
                        "coefficients": [
                            {"term": "Intercept", "coefficient": 0.0},
                            {"term": "A", "coefficient": 1.0},
                        ],
                    }
                ],
                "goals": [{"response": "y", "goal": "maximize", "low": -5.0, "high": 5.0}],
                "method": "desirability",
                **extra,
            },
        )

    def test_search_bounds_as_a_single_pair(self) -> None:
        """Over the wire the region arrives as a JSON list, not a tuple."""
        result = self._rising_plane_call(search_bounds=[-1.41, 1.41])
        assert "error" not in result
        assert result["desirability"]["optimal_coded"]["A"] == pytest.approx(1.41, abs=1e-4)

    def test_search_bounds_per_factor(self) -> None:
        """A mapping widens one factor and leaves the other at the cube."""
        result = self._rising_plane_call(search_bounds={"A": [-2.0, 2.0]})
        assert "error" not in result
        assert result["desirability"]["optimal_coded"]["A"] == pytest.approx(2.0, abs=1e-4)

    def test_search_bounds_default_is_the_cube(self) -> None:
        """Omitting it leaves the previous behaviour in place."""
        result = self._rising_plane_call()
        assert result["desirability"]["optimal_coded"]["A"] == pytest.approx(1.0, abs=1e-4)

    @pytest.mark.parametrize("bounds", [[-1.5], [-1.5, 1.5, 99.0], {"A": [-2.0]}, {"A": [-2.0, 2.0, 3.0]}])
    def test_search_bounds_must_be_a_pair(self, bounds: object) -> None:
        """One number escaped as an IndexError; three had the third silently dropped."""
        with pytest.raises(ToolInputInvalidError):
            self._rising_plane_call(search_bounds=bounds)

    def test_malformed_search_bounds_returns_an_error(self) -> None:
        """A reversed pair is reported, not silently accepted."""
        result = self._rising_plane_call(search_bounds=[1.0, -1.0])
        assert "low < high" in result["error"]

    def test_analyze_experiment_result_feeds_optimize_responses(self) -> None:
        """The documented generate -> analyze -> optimize pipeline, with no hand-editing of the result."""
        design = execute_tool_call(
            "generate_design",
            {
                "factors": [{"name": "T", "low": 150, "high": 200}, {"name": "P", "low": 1, "high": 5}],
                "design_type": "ccd",
            },
        )
        rows = [
            dict(r, y=40 + 5 * r["T"] - 2 * r["P"] - 3 * r["T"] ** 2 - 1.5 * r["P"] ** 2 + 1.5 * r["T"] * r["P"])
            for r in design["design_coded"]
        ]
        fitted = execute_tool_call(
            "analyze_experiment",
            {"design_matrix": rows, "response_column": "y", "model": "quadratic", "analysis_type": ["coefficients"]},
        )
        assert fitted["response_name"] == "y"
        assert fitted["factor_names"] == ["T", "P"]
        result = execute_tool_call("optimize_responses", {"fitted_models": [fitted], "method": "stationary_point"})
        assert result["stationary_point"]["classification"] == "maximum"
        # Solve 2 B x = -b for b = (5, -2), B = [[-3, 0.75], [0.75, -1.5]].
        assert result["stationary_point"]["stationary_point_coded"]["T"] == pytest.approx(16 / 21)

    def test_model_without_factor_names_names_the_key(self) -> None:
        """The error says what is missing, not just the bare key."""
        result = execute_tool_call(
            "optimize_responses",
            {
                "fitted_models": [{"coefficients": [{"term": "Intercept", "coefficient": 1.0}]}],
                "method": "stationary_point",
            },
        )
        assert "has no 'factor_names'" in result["error"]

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


# ---------------------------------------------------------------------------
# trade_off_table
# ---------------------------------------------------------------------------


class TestTradeOffTable:
    def test_default_table_matches_the_course_notes(self) -> None:
        """The default grid is the trade-off table from the course notes."""
        result = execute_tool_call("trade_off_table", {})
        assert "error" not in result
        assert result["table"]["8"]["7"] == "2^(7-4) III"
        assert result["table"]["16"]["5"] == "2^(5-1) V"
        assert result["table"]["16"]["3"] == "2^3 (twice)"

    def test_cells_carry_the_generators(self) -> None:
        """Each existing cell reports how the design is built."""
        result = execute_tool_call("trade_off_table", {"runs": [8], "factors": [5]})
        (cell,) = result["cells"]
        assert cell["exists"] is True
        assert cell["generators"] == ["D=AB", "E=AC"]
        assert cell["roman"] == "III"
        assert cell["n_generators"] == 2

    def test_impossible_cell_is_blank_with_a_reason(self) -> None:
        """8 runs cannot hold 9 factors; the cell is blank and says why."""
        result = execute_tool_call("trade_off_table", {"runs": [8], "factors": [9]})
        assert result["table"]["8"]["9"] == ""
        (cell,) = result["cells"]
        assert cell["exists"] is False
        assert "cannot accommodate" in cell["reason"]

    def test_existing_design_beyond_the_search_is_not_reported_impossible(self) -> None:
        """2^(12-7) designs exist; the cell used to read exists: False, 'too many factors for the budget'."""
        result = execute_tool_call("trade_off_table", {"runs": [32, 64], "factors": [12]})
        result["cells"] += execute_tool_call("trade_off_table", {"runs": [64], "factors": [11]})["cells"]
        cells = {(c["runs"], c["factors"]): c for c in result["cells"]}
        assert cells[32, 12]["exists"] is None
        assert result["table"]["32"]["12"] == "?"
        assert "The design exists" in cells[32, 12]["reason"]
        assert cells[64, 11]["exists"] is True
        assert cells[64, 11]["label"] == "2^(11-5) IV"

    def test_over_budget_cell_reports_replication(self) -> None:
        """A budget larger than the full factorial is replication, not an error."""
        result = execute_tool_call("trade_off_table", {"runs": [32], "factors": [3]})
        (cell,) = result["cells"]
        assert cell["label"] == "2^3 (4 times)"
        assert cell["n_replicates"] == 4
        assert cell["resolution"] is None

    def test_output_is_json_serialisable(self) -> None:
        """Tool output crosses the MCP boundary, so it must serialise."""
        result = execute_tool_call("trade_off_table", {"runs": [8, 16], "factors": [4, 5]})
        assert json.loads(json.dumps(result)) == result

    def test_non_power_of_two_runs_returns_error(self) -> None:
        result = execute_tool_call("trade_off_table", {"runs": [7], "factors": [5]})
        assert "power of 2" in result["error"]

    def test_out_of_range_factors_returns_error(self) -> None:
        result = execute_tool_call("trade_off_table", {"runs": [8], "factors": [99]})
        assert "between 2 and 12" in result["error"]

    def test_unknown_key_is_rejected_by_the_schema(self) -> None:
        """``extra="forbid"`` closes the kwarg-injection vector (SEC-15)."""
        with pytest.raises(ToolInputInvalidError):
            execute_tool_call("trade_off_table", {"runs": [8], "bogus": 1})

    def test_registered_in_the_experiments_tool_specs(self) -> None:
        from process_improve.experiments.tools import get_experiments_tool_specs

        specs = {s["name"]: s for s in get_experiments_tool_specs()}
        assert "trade_off_table" in specs
        assert specs["trade_off_table"]["input_schema"]["additionalProperties"] is False


def test_generate_design_tool_offers_every_design_type() -> None:
    """The tool's design_type choices are exactly generate_design's registry (omars was missing)."""
    import typing

    from process_improve.experiments._tools.generate_design import GenerateDesignInput
    from process_improve.experiments.designs import _DESIGN_REGISTRY

    annotation = GenerateDesignInput.model_fields["design_type"].annotation
    literal = next(arg for arg in typing.get_args(annotation) if typing.get_origin(arg) is typing.Literal)
    assert set(typing.get_args(literal)) == set(_DESIGN_REGISTRY)


def test_generate_design_tool_enforces_constraints_and_returns_the_region() -> None:
    from process_improve.experiments._tools.generate_design import GenerateDesignInput, generate_design_tool

    spec = GenerateDesignInput(
        factors=[{"name": "T", "low": 100, "high": 150}, {"name": "D", "low": 20, "high": 60}],
        design_type="i_optimal",
        budget=10,
        model_type="quadratic",
        constraints=["3*T + 5*D <= 600"],
    )
    out = generate_design_tool(spec)
    assert "error" not in out, out
    assert all(3 * r["T"] + 5 * r["D"] <= 600 + 1e-9 for r in out["design_actual"])
    assert out["metadata"]["region"]["constraints"][0]["expression"] == "3*T + 5*D <= 600"
    import json

    json.dumps(out)  # JSON-serialisable for MCP transport


def test_region_round_trips_through_the_tools() -> None:
    """generate_design -> evaluate_design -> optimize_responses, all inside the constrained region."""
    from process_improve.experiments._tools.evaluate_design import EvaluateDesignInput, evaluate_design_tool
    from process_improve.experiments._tools.generate_design import GenerateDesignInput, generate_design_tool
    from process_improve.experiments._tools.optimize_responses import OptimizeResponsesInput, optimize_responses_tool

    factors = [{"name": "T", "low": 100, "high": 150}, {"name": "D", "low": 20, "high": 60}]
    design = generate_design_tool(
        GenerateDesignInput(
            factors=factors,
            design_type="d_optimal",
            budget=10,
            model_type="quadratic",
            constraints=["3*T + 5*D <= 600"],
        )
    )
    region = design["metadata"]["region"]
    coded = [{k: r[k] for k in ("T", "D")} for r in design["design_coded"]]
    inside = evaluate_design_tool(
        EvaluateDesignInput(design_matrix=coded, model="quadratic", metric="g_efficiency", region=region)
    )
    cube = evaluate_design_tool(
        EvaluateDesignInput(design_matrix=coded, model="quadratic", metric="g_efficiency", region="cuboidal")
    )
    assert inside["g_efficiency"] > 10 * cube["g_efficiency"]

    model = {
        "response_name": "y",
        "factor_names": ["T", "D"],
        "coefficients": [
            {"term": "Intercept", "coefficient": 60.0},
            {"term": "T", "coefficient": 8.0},
            {"term": "D", "coefficient": 6.0},
            {"term": "I(T ** 2)", "coefficient": -2.0},
            {"term": "I(D ** 2)", "coefficient": -3.0},
        ],
    }
    best = optimize_responses_tool(
        OptimizeResponsesInput(
            fitted_models=[model],
            goals=[{"response": "y", "goal": "maximize", "low": 40, "high": 75}],
            factor_ranges={"T": {"low": 100, "high": 150}, "D": {"low": 20, "high": 60}},
            region=region,
        )
    )
    optimum = best["desirability"]["optimal_actual"]
    assert best["desirability"]["within_region"]
    assert 3 * optimum["T"] + 5 * optimum["D"] <= 600 + 1e-6
