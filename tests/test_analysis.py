"""Tests for the analyze_experiment() API (Tool 3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments.analysis import (
    AnalysisResult,
    analyze_experiment,
    build_formula,
    _compute_adequate_precision,
    _compute_pred_r_squared,
    _run_lenth_method,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _two_factor_data() -> pd.DataFrame:
    """Unreplicated 2^2 factorial with response."""
    return pd.DataFrame({
        "A": [-1, 1, -1, 1],
        "B": [-1, -1, 1, 1],
        "y": [28, 36, 18, 31],
    })


def _two_factor_replicated() -> pd.DataFrame:
    """2^2 factorial with 2 replicates."""
    return pd.DataFrame({
        "A": [-1, 1, -1, 1, -1, 1, -1, 1],
        "B": [-1, -1, 1, 1, -1, -1, 1, 1],
        "y": [28, 36, 18, 31, 27, 34, 19, 30],
    })


def _two_factor_with_center() -> pd.DataFrame:
    """2^2 factorial with center points."""
    return pd.DataFrame({
        "A": [-1, 1, -1, 1, 0, 0, 0],
        "B": [-1, -1, 1, 1, 0, 0, 0],
        "y": [28, 36, 18, 31, 25, 26, 24],
    })


def _three_factor_data() -> pd.DataFrame:
    """2^3 factorial with response."""
    return pd.DataFrame({
        "A": [-1, 1, -1, 1, -1, 1, -1, 1],
        "B": [-1, -1, 1, 1, -1, -1, 1, 1],
        "C": [-1, -1, -1, -1, 1, 1, 1, 1],
        "y": [550, 669, 604, 650, 633, 642, 601, 635],
    })


# ---------------------------------------------------------------------------
# Formula builder
# ---------------------------------------------------------------------------


class TestBuildFormula:
    def test_main_effects(self) -> None:
        f = build_formula("y", ["A", "B"], "main_effects")
        assert f == "y ~ A + B"

    def test_interactions_default(self) -> None:
        f = build_formula("y", ["A", "B"])
        assert "**" in f or ":" in f  # patsy interaction syntax

    def test_quadratic(self) -> None:
        f = build_formula("y", ["A", "B"], "quadratic")
        assert "I(A ** 2)" in f
        assert "I(B ** 2)" in f

    def test_explicit_formula_passthrough(self) -> None:
        f = build_formula("y", ["A", "B"], "y ~ A + B + A:B")
        assert f == "y ~ A + B + A:B"

    def test_none_defaults_to_interactions(self) -> None:
        f = build_formula("y", ["A", "B"], None)
        assert f == build_formula("y", ["A", "B"], "interactions")


# ---------------------------------------------------------------------------
# Model summary (always returned)
# ---------------------------------------------------------------------------


class TestModelSummary:
    def test_basic_summary_keys(self) -> None:
        df = _two_factor_data()
        result = analyze_experiment(df, response_column="y", model="main_effects", analysis_type="coefficients")
        summary = result["model_summary"]
        assert "r_squared" in summary
        assert "r_squared_adj" in summary
        assert "r_squared_pred" in summary
        assert "adequate_precision" in summary
        assert "formula" in summary
        assert summary["n_obs"] == 4

    def test_r_squared_range(self) -> None:
        df = _two_factor_replicated()
        result = analyze_experiment(df, response_column="y", analysis_type="anova")
        r2 = result["model_summary"]["r_squared"]
        assert 0 <= r2 <= 1


# ---------------------------------------------------------------------------
# ANOVA
# ---------------------------------------------------------------------------


class TestAnova:
    def test_anova_returns_table(self) -> None:
        df = _two_factor_replicated()
        result = analyze_experiment(df, response_column="y", analysis_type="anova")
        assert "anova_table" in result
        table = result["anova_table"]
        assert len(table) > 0
        assert "source" in table[0]
        assert "p_value" in table[0]

    def test_anova_main_effects_model(self) -> None:
        df = _two_factor_replicated()
        result = analyze_experiment(
            df, response_column="y", model="main_effects", analysis_type="anova"
        )
        sources = [r["source"] for r in result["anova_table"]]
        assert "A" in sources
        assert "B" in sources


# ---------------------------------------------------------------------------
# Effects
# ---------------------------------------------------------------------------


class TestEffects:
    def test_effects_values(self) -> None:
        df = _two_factor_data()
        result = analyze_experiment(
            df, response_column="y", model="main_effects", analysis_type="effects"
        )
        effects = result["effects"]
        # For coded ±1 factors, effect = 2 * coefficient
        assert "A" in effects
        assert "B" in effects

    def test_effects_are_twice_coefficients(self) -> None:
        df = _two_factor_data()
        result = analyze_experiment(
            df, response_column="y", model="main_effects",
            analysis_type=["effects", "coefficients"],
        )
        for coef in result["coefficients"]:
            if coef["term"] == "A":
                a_coef = coef["coefficient"]
        assert abs(result["effects"]["A"] - 2 * a_coef) < 1e-10


# ---------------------------------------------------------------------------
# Coefficients
# ---------------------------------------------------------------------------


class TestCoefficients:
    def test_coefficients_structure(self) -> None:
        df = _two_factor_data()
        result = analyze_experiment(
            df, response_column="y", model="main_effects", analysis_type="coefficients"
        )
        coeffs = result["coefficients"]
        assert len(coeffs) > 0
        first = coeffs[0]
        assert "term" in first
        assert "coefficient" in first
        assert "std_error" in first
        assert "t_value" in first
        assert "p_value" in first
        assert "ci_low" in first
        assert "ci_high" in first


# ---------------------------------------------------------------------------
# Significance
# ---------------------------------------------------------------------------


class TestSignificance:
    def test_significance_split(self) -> None:
        df = _two_factor_replicated()
        result = analyze_experiment(
            df, response_column="y", model="main_effects", analysis_type="significance"
        )
        assert "significant_terms" in result
        assert "not_significant_terms" in result
        assert result["significance_level"] == 0.05

    def test_custom_alpha(self) -> None:
        df = _two_factor_replicated()
        result = analyze_experiment(
            df, response_column="y", model="main_effects",
            analysis_type="significance", significance_level=0.10,
        )
        assert result["significance_level"] == 0.10


# ---------------------------------------------------------------------------
# Residual diagnostics
# ---------------------------------------------------------------------------


class TestResidualDiagnostics:
    def test_diagnostics_keys(self) -> None:
        df = _two_factor_replicated()
        result = analyze_experiment(
            df, response_column="y", model="main_effects",
            analysis_type="residual_diagnostics",
        )
        diag = result["residual_diagnostics"]
        assert "shapiro_wilk" in diag
        assert "durbin_watson" in diag
        assert "breusch_pagan" in diag
        assert "cooks_distance" in diag
        assert "leverage" in diag
        assert "residuals" in diag
        assert "fitted_values" in diag

    def test_cooks_distance_length(self) -> None:
        df = _two_factor_replicated()
        result = analyze_experiment(
            df, response_column="y", model="main_effects",
            analysis_type="residual_diagnostics",
        )
        n = len(df)
        assert len(result["residual_diagnostics"]["cooks_distance"]) == n


# ---------------------------------------------------------------------------
# Lack of fit
# ---------------------------------------------------------------------------


class TestLackOfFit:
    def test_lof_with_replicates(self) -> None:
        df = _two_factor_replicated()
        result = analyze_experiment(
            df, response_column="y", model="main_effects",
            analysis_type="lack_of_fit",
        )
        lof = result["lack_of_fit"]
        assert "f_statistic" in lof
        assert "p_value" in lof
        assert "significant" in lof

    def test_lof_without_replicates_errors(self) -> None:
        df = _two_factor_data()
        result = analyze_experiment(
            df, response_column="y", model="main_effects",
            analysis_type="lack_of_fit",
        )
        lof = result["lack_of_fit"]
        assert "error" in lof


# ---------------------------------------------------------------------------
# Curvature test
# ---------------------------------------------------------------------------


class TestCurvatureTest:
    def test_curvature_with_center_points(self) -> None:
        df = _two_factor_with_center()
        result = analyze_experiment(
            df, response_column="y", model="main_effects",
            analysis_type="curvature_test",
        )
        ct = result["curvature_test"]
        assert "center_point_mean" in ct
        assert "factorial_point_mean" in ct
        assert "t_statistic" in ct
        assert "p_value" in ct
        assert ct["n_center_points"] == 3
        assert ct["n_factorial_points"] == 4

    def test_curvature_without_center_points(self) -> None:
        df = _two_factor_data()
        result = analyze_experiment(
            df, response_column="y", model="main_effects",
            analysis_type="curvature_test",
        )
        assert "error" in result["curvature_test"]


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------


class TestModelSelection:
    def test_backward_selection(self) -> None:
        df = _three_factor_data()
        result = analyze_experiment(
            df, response_column="y", model="main_effects",
            analysis_type="model_selection",
        )
        ms = result["model_selection"]
        assert "selected_formula" in ms
        assert "criterion" in ms
        assert ms["direction"] == "backward"


# ---------------------------------------------------------------------------
# Box-Cox
# ---------------------------------------------------------------------------


class TestBoxCox:
    def test_box_cox_positive_data(self) -> None:
        df = _two_factor_replicated()
        result = analyze_experiment(
            df, response_column="y", analysis_type="box_cox",
        )
        bc = result["box_cox"]
        assert "lambda" in bc
        assert "recommendation" in bc
        assert len(bc["transformed_values"]) == len(df)

    def test_box_cox_negative_data(self) -> None:
        df = _two_factor_data().copy()
        df["y"] = [-1, -2, -3, -4]
        result = analyze_experiment(
            df, response_column="y", analysis_type="box_cox",
        )
        assert "error" in result["box_cox"]


# ---------------------------------------------------------------------------
# Lenth's method
# ---------------------------------------------------------------------------


class TestLenthMethod:
    def test_lenth_unreplicated(self) -> None:
        df = _three_factor_data()
        result = analyze_experiment(
            df, response_column="y", model="main_effects",
            analysis_type="lenth_method",
        )
        lm_result = result["lenth_method"]
        assert "PSE" in lm_result
        assert "ME" in lm_result
        assert "SME" in lm_result
        assert len(lm_result["effects"]) > 0

    def test_lenth_has_active_flags(self) -> None:
        df = _three_factor_data()
        result = analyze_experiment(
            df, response_column="y", model="main_effects",
            analysis_type="lenth_method",
        )
        for eff in result["lenth_method"]["effects"]:
            assert "active_ME" in eff
            assert "active_SME" in eff


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------


class TestConfidenceIntervals:
    def test_ci_structure(self) -> None:
        df = _two_factor_replicated()
        result = analyze_experiment(
            df, response_column="y", model="main_effects",
            analysis_type="confidence_intervals",
        )
        assert "confidence_intervals" in result
        assert result["confidence_level"] == 0.95
        ci = result["confidence_intervals"]
        assert len(ci) > 0
        assert "ci_low" in ci[0]
        assert "ci_high" in ci[0]


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


class TestPrediction:
    def test_prediction_with_new_points(self) -> None:
        df = _two_factor_replicated()
        new = pd.DataFrame({"A": [0, 0.5], "B": [0, -0.5]})
        result = analyze_experiment(
            df, response_column="y", model="main_effects",
            analysis_type="prediction", new_points=new,
        )
        preds = result["predictions"]
        assert len(preds) == 2
        assert "predicted" in preds[0]
        assert "pi_low" in preds[0]
        assert "pi_high" in preds[0]

    def test_prediction_without_new_points_errors(self) -> None:
        df = _two_factor_replicated()
        result = analyze_experiment(
            df, response_column="y", analysis_type="prediction",
        )
        assert "error" in result["prediction"]


# ---------------------------------------------------------------------------
# Confirmation test
# ---------------------------------------------------------------------------


class TestConfirmationTest:
    def test_confirmation_pass(self) -> None:
        df = _two_factor_replicated()
        new = pd.DataFrame({"A": [0], "B": [0]})
        result = analyze_experiment(
            df, response_column="y", model="main_effects",
            analysis_type="confirmation_test",
            new_points=new, observed_at_new=[28.0],
        )
        ct = result["confirmation_test"]
        assert "results" in ct
        assert "all_within_PI" in ct
        assert len(ct["results"]) == 1

    def test_confirmation_missing_args(self) -> None:
        df = _two_factor_replicated()
        result = analyze_experiment(
            df, response_column="y", analysis_type="confirmation_test",
        )
        assert "error" in result["confirmation_test"]


# ---------------------------------------------------------------------------
# Multiple analysis types at once
# ---------------------------------------------------------------------------


class TestMultipleAnalyses:
    def test_multiple_types(self) -> None:
        df = _two_factor_replicated()
        result = analyze_experiment(
            df, response_column="y", model="main_effects",
            analysis_type=["anova", "coefficients", "effects"],
        )
        assert "anova_table" in result
        assert "coefficients" in result
        assert "effects" in result
        assert "model_summary" in result


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


class TestTransforms:
    def test_log_transform(self) -> None:
        df = _two_factor_replicated()
        result = analyze_experiment(
            df, response_column="y", transform="log", analysis_type="coefficients",
        )
        assert "coefficients" in result

    def test_sqrt_transform(self) -> None:
        df = _two_factor_replicated()
        result = analyze_experiment(
            df, response_column="y", transform="sqrt", analysis_type="coefficients",
        )
        assert "coefficients" in result


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_unknown_analysis_type_raises(self) -> None:
        df = _two_factor_data()
        with pytest.raises(ValueError, match="Unknown analysis_type"):
            analyze_experiment(df, response_column="y", analysis_type="bogus")

    def test_missing_response_column_raises(self) -> None:
        df = _two_factor_data()
        with pytest.raises(ValueError, match="not found"):
            analyze_experiment(df, response_column="missing")

    def test_no_response_arg_raises(self) -> None:
        df = pd.DataFrame({"A": [-1, 1], "B": [-1, 1]})
        with pytest.raises(ValueError, match="Must provide"):
            analyze_experiment(df)


# ---------------------------------------------------------------------------
# Responses as separate argument
# ---------------------------------------------------------------------------


class TestSeparateResponses:
    def test_responses_as_series(self) -> None:
        df = pd.DataFrame({"A": [-1, 1, -1, 1], "B": [-1, -1, 1, 1]})
        y = pd.Series([28, 36, 18, 31], name="y")
        result = analyze_experiment(df, responses=y, analysis_type="coefficients")
        assert "coefficients" in result

    def test_responses_as_dataframe(self) -> None:
        df = pd.DataFrame({"A": [-1, 1, -1, 1], "B": [-1, -1, 1, 1]})
        y = pd.DataFrame({"y": [28, 36, 18, 31]})
        result = analyze_experiment(df, responses=y, analysis_type="coefficients")
        assert "coefficients" in result
