"""Tests for the analyze_experiment() API (Tool 3)."""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import Factor, generate_design
from process_improve.experiments.analysis import (
    analyze_experiment,
    build_formula,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _two_factor_data() -> pd.DataFrame:
    """Unreplicated 2^2 factorial with response."""
    return pd.DataFrame(
        {
            "A": [-1, 1, -1, 1],
            "B": [-1, -1, 1, 1],
            "y": [28, 36, 18, 31],
        }
    )


def _two_factor_replicated() -> pd.DataFrame:
    """2^2 factorial with 2 replicates."""
    return pd.DataFrame(
        {
            "A": [-1, 1, -1, 1, -1, 1, -1, 1],
            "B": [-1, -1, 1, 1, -1, -1, 1, 1],
            "y": [28, 36, 18, 31, 27, 34, 19, 30],
        }
    )


def _two_factor_with_center() -> pd.DataFrame:
    """2^2 factorial with center points."""
    return pd.DataFrame(
        {
            "A": [-1, 1, -1, 1, 0, 0, 0],
            "B": [-1, -1, 1, 1, 0, 0, 0],
            "y": [28, 36, 18, 31, 25, 26, 24],
        }
    )


def _three_factor_data() -> pd.DataFrame:
    """2^3 factorial with response."""
    return pd.DataFrame(
        {
            "A": [-1, 1, -1, 1, -1, 1, -1, 1],
            "B": [-1, -1, 1, 1, -1, -1, 1, 1],
            "C": [-1, -1, -1, -1, 1, 1, 1, 1],
            "y": [550, 669, 604, 650, 633, 642, 601, 635],
        }
    )


# ---------------------------------------------------------------------------
# Formula builder
# ---------------------------------------------------------------------------


class TestBuildFormula:
    def test_main_effects(self) -> None:
        """Verify main_effects model produces additive-only formula."""
        f = build_formula("y", ["A", "B"], "main_effects")
        assert f == "y ~ A + B"

    def test_interactions_default(self) -> None:
        """Verify default model includes interaction terms."""
        f = build_formula("y", ["A", "B"])
        assert "**" in f or ":" in f  # patsy interaction syntax

    def test_quadratic(self) -> None:
        """Verify quadratic model includes squared terms."""
        f = build_formula("y", ["A", "B"], "quadratic")
        assert "I(A ** 2)" in f
        assert "I(B ** 2)" in f

    def test_explicit_formula_passthrough(self) -> None:
        """Verify explicit formula string passes through unchanged."""
        f = build_formula("y", ["A", "B"], "y ~ A + B + A:B")
        assert f == "y ~ A + B + A:B"

    def test_none_defaults_to_interactions(self) -> None:
        """Verify None model defaults to interactions."""
        f = build_formula("y", ["A", "B"], None)
        assert f == build_formula("y", ["A", "B"], "interactions")


# ---------------------------------------------------------------------------
# Model summary (always returned)
# ---------------------------------------------------------------------------


class TestModelSummary:
    def test_basic_summary_keys(self) -> None:
        """Verify model summary contains expected keys."""
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
        """Verify R-squared is between 0 and 1."""
        df = _two_factor_replicated()
        result = analyze_experiment(df, response_column="y", analysis_type="anova")
        r2 = result["model_summary"]["r_squared"]
        assert 0 <= r2 <= 1


class TestRankDeficiency:
    """A rank-deficient model is still fitted by the pseudo-inverse, so the caller has
    to be told that some of the reported coefficients are not determined by the data.
    """

    @staticmethod
    def _foldover_design() -> pd.DataFrame:
        """Five-run foldover in two factors: [H; -H; 0]. Its full second-order model
        has 6 terms but rank 5, because mirror-image pairs repeat the even columns.
        """
        levels = [(1, 1), (1, -1), (-1, -1), (-1, 1), (0, 0)]
        df = pd.DataFrame(levels, columns=["A", "B"])
        df["y"] = [10.0, 12.0, 11.0, 9.0, 13.0]
        return df

    def test_full_rank_model_is_silent(self) -> None:
        df = self._foldover_design()
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = analyze_experiment(df, response_column="y", model="A + B", analysis_type="coefficients")
        summary = result["model_summary"]
        assert summary["rank_deficient"] is False
        assert summary["model_rank"] == summary["n_terms"] == 3

    def test_rank_deficient_model_warns_and_reports(self) -> None:
        df = self._foldover_design()
        model = "A + B + I(A**2) + I(B**2) + A:B"
        with pytest.warns(RuntimeWarning, match="not estimable"):
            result = analyze_experiment(df, response_column="y", model=model, analysis_type="coefficients")
        summary = result["model_summary"]
        assert summary["n_terms"] == 6
        assert summary["model_rank"] == 5
        assert summary["rank_deficient"] is True
        # A coefficient is still reported for every requested term.
        assert len(result["coefficients"]) == 6


# ---------------------------------------------------------------------------
# ANOVA
# ---------------------------------------------------------------------------


class TestAnova:
    def test_anova_returns_table(self) -> None:
        """Verify ANOVA returns a table with source and p-value."""
        df = _two_factor_replicated()
        result = analyze_experiment(df, response_column="y", analysis_type="anova")
        assert "anova_table" in result
        table = result["anova_table"]
        assert len(table) > 0
        assert "source" in table[0]
        assert "p_value" in table[0]

    def test_anova_main_effects_model(self) -> None:
        """Verify ANOVA table includes main effect sources."""
        df = _two_factor_replicated()
        result = analyze_experiment(df, response_column="y", model="main_effects", analysis_type="anova")
        sources = [r["source"] for r in result["anova_table"]]
        assert "A" in sources
        assert "B" in sources


# ---------------------------------------------------------------------------
# Effects
# ---------------------------------------------------------------------------


class TestEffects:
    def test_effects_values(self) -> None:
        """Verify effects dict contains factor names."""
        df = _two_factor_data()
        result = analyze_experiment(df, response_column="y", model="main_effects", analysis_type="effects")
        effects = result["effects"]
        # For coded ±1 factors, effect = 2 * coefficient
        assert "A" in effects
        assert "B" in effects

    def test_effects_are_twice_coefficients(self) -> None:
        """Verify effects equal 2x the coefficients for coded factors."""
        df = _two_factor_data()
        result = analyze_experiment(
            df,
            response_column="y",
            model="main_effects",
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
        """Verify coefficient records contain all expected fields."""
        df = _two_factor_data()
        result = analyze_experiment(df, response_column="y", model="main_effects", analysis_type="coefficients")
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
        """Verify significance analysis splits terms at alpha=0.05."""
        df = _two_factor_replicated()
        result = analyze_experiment(df, response_column="y", model="main_effects", analysis_type="significance")
        assert "significant_terms" in result
        assert "not_significant_terms" in result
        assert result["significance_level"] == 0.05

    def test_custom_alpha(self) -> None:
        """Verify custom alpha is reflected in results."""
        df = _two_factor_replicated()
        result = analyze_experiment(
            df,
            response_column="y",
            model="main_effects",
            analysis_type="significance",
            significance_level=0.10,
        )
        assert result["significance_level"] == 0.10


# ---------------------------------------------------------------------------
# Residual diagnostics
# ---------------------------------------------------------------------------


class TestResidualDiagnostics:
    def test_diagnostics_keys(self) -> None:
        """Verify residual diagnostics contain all expected keys."""
        df = _two_factor_replicated()
        result = analyze_experiment(
            df,
            response_column="y",
            model="main_effects",
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
        """Verify Cook's distance has one value per observation."""
        df = _two_factor_replicated()
        result = analyze_experiment(
            df,
            response_column="y",
            model="main_effects",
            analysis_type="residual_diagnostics",
        )
        n = len(df)
        assert len(result["residual_diagnostics"]["cooks_distance"]) == n


# ---------------------------------------------------------------------------
# Lack of fit
# ---------------------------------------------------------------------------


class TestLackOfFit:
    def test_lof_with_replicates(self) -> None:
        """Verify lack-of-fit test runs with replicated data."""
        df = _two_factor_replicated()
        result = analyze_experiment(
            df,
            response_column="y",
            model="main_effects",
            analysis_type="lack_of_fit",
        )
        lof = result["lack_of_fit"]
        assert "f_statistic" in lof
        assert "p_value" in lof
        assert "significant" in lof

    def test_lof_without_replicates_errors(self) -> None:
        """Verify lack-of-fit returns error without replicates."""
        df = _two_factor_data()
        result = analyze_experiment(
            df,
            response_column="y",
            model="main_effects",
            analysis_type="lack_of_fit",
        )
        lof = result["lack_of_fit"]
        assert "error" in lof


# ---------------------------------------------------------------------------
# Curvature test
# ---------------------------------------------------------------------------


class TestCurvatureTest:
    def test_curvature_with_center_points(self) -> None:
        """Verify curvature test runs with center points."""
        df = _two_factor_with_center()
        result = analyze_experiment(
            df,
            response_column="y",
            model="main_effects",
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
        """Verify curvature test returns error without center points."""
        df = _two_factor_data()
        result = analyze_experiment(
            df,
            response_column="y",
            model="main_effects",
            analysis_type="curvature_test",
        )
        assert "error" in result["curvature_test"]


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------


class TestModelSelection:
    def test_backward_selection(self) -> None:
        """Verify backward model selection returns a formula."""
        df = _three_factor_data()
        result = analyze_experiment(
            df,
            response_column="y",
            model="main_effects",
            analysis_type="model_selection",
        )
        ms = result["model_selection"]
        assert "selected_formula" in ms
        assert "criterion" in ms
        assert ms["direction"] == "backward"

    def test_candidates_follow_the_requested_model(self) -> None:
        """A main-effects request never selects an interaction, whatever the data say (#639)."""
        ms = analyze_experiment(
            _three_factor_data(), response_column="y", model="main_effects", analysis_type="model_selection"
        )["model_selection"]
        assert all(":" not in term for term in ms["selected_terms"])
        assert ms["candidate_model"] == "main_effects"

    def test_interaction_and_its_parents_are_found_by_backward_search(self) -> None:
        rng = np.random.default_rng(1)
        x = pd.DataFrame(list(itertools.product([-1.0, 1.0], repeat=4)), columns=list("ABCD"))
        y = 5 + 2 * x.A + 1.5 * x.B + 1.2 * x.A * x.B + rng.normal(0, 0.3, len(x))
        ms = analyze_experiment(x, pd.Series(y, name="y"), model="interactions", analysis_type="model_selection")[
            "model_selection"
        ]
        assert ms["selected_formula"] == "y ~ A + B + A:B"
        assert ms["direction"] == "backward"

    def test_heredity_never_leaves_an_orphan_interaction(self) -> None:
        """A:B alone in the truth: strong heredity brings in A and B, or drops A:B."""
        rng = np.random.default_rng(3)
        x = pd.DataFrame(list(itertools.product([-1.0, 1.0], repeat=3)) * 2, columns=list("ABC"))
        y = 3 * x.A * x.B + rng.normal(0, 0.2, len(x))
        terms = analyze_experiment(x, pd.Series(y, name="y"), model="interactions", analysis_type="model_selection")[
            "model_selection"
        ]["selected_terms"]
        for term in terms:
            assert set(term.split(":")) <= set(terms)

    @pytest.mark.parametrize("seed", range(5))
    def test_supersaturated_design_finds_the_active_factors(self, seed: int) -> None:
        """16 factors in 12 runs: forward search under AICc finds X00 and X05 (#639)."""
        factors = [Factor(name=f"X{i:02d}", low=-1, high=1) for i in range(16)]
        x = generate_design(factors, design_type="supersaturated", budget=12).design_actual[[f.name for f in factors]]
        y = 10 + 4 * x["X00"] - 3 * x["X05"] + np.random.default_rng(seed).normal(0, 0.5, len(x))
        ms = analyze_experiment(x, pd.Series(y, name="y"), model="main_effects", analysis_type="model_selection")[
            "model_selection"
        ]
        assert ms["direction"] == "forward"
        assert ms["criterion"] == "aicc"
        assert {"X00", "X05"} <= set(ms["selected_terms"])
        assert len(ms["selected_terms"]) <= 5

    @pytest.mark.parametrize("criterion", ["aic", "bic"])
    def test_other_criteria(self, criterion: str) -> None:
        from process_improve.experiments._analyses.model_selection import _run_model_selection

        rng = np.random.default_rng(2)
        x = pd.DataFrame(list(itertools.product([-1.0, 1.0], repeat=3)) * 3, columns=list("ABC"))
        x["y"] = 4 + 3 * x.A - 2 * x.C + rng.normal(0, 0.3, len(x))
        ms = _run_model_selection(x, "y", list("ABC"), "main_effects", criterion)["model_selection"]
        assert ms["criterion"] == criterion
        assert ms["selected_terms"] == ["A", "C"]

    def test_squares_need_their_main_effect(self) -> None:
        """A quadratic search adds I(A ** 2) only once A is in (strong heredity)."""
        rng = np.random.default_rng(4)
        levels = [-1.0, 0.0, 1.0]
        x = pd.DataFrame(list(itertools.product(levels, repeat=2)) * 2, columns=list("AB"))
        y = 10 + 2 * x.A - 3 * x.A**2 + rng.normal(0, 0.2, len(x))
        ms = analyze_experiment(x, pd.Series(y, name="y"), model="quadratic", analysis_type="model_selection")[
            "model_selection"
        ]
        assert "I(A ** 2)" in ms["selected_terms"]
        assert "A" in ms["selected_terms"]

    def test_a_formula_model_is_searched_as_interactions_with_a_note(self) -> None:
        ms = analyze_experiment(
            _three_factor_data(), response_column="y", model="A + B + A:B", analysis_type="model_selection"
        )["model_selection"]
        assert ms["candidate_model"] == "interactions"
        assert "searched as 'interactions'" in ms["note"]

    def test_unknown_criterion(self) -> None:
        from process_improve.experiments._analyses.model_selection import _run_model_selection

        with pytest.raises(ValueError, match="aicc, aic, bic"):
            _run_model_selection(_three_factor_data(), "y", ["A", "B", "C"], "main_effects", "cp")


# ---------------------------------------------------------------------------
# Box-Cox
# ---------------------------------------------------------------------------


class TestBoxCox:
    def test_box_cox_positive_data(self) -> None:
        """Verify Box-Cox transform runs on positive data."""
        df = _two_factor_replicated()
        result = analyze_experiment(
            df,
            response_column="y",
            analysis_type="box_cox",
        )
        bc = result["box_cox"]
        assert "lambda" in bc
        assert "recommendation" in bc
        assert len(bc["transformed_values"]) == len(df)

    def test_box_cox_negative_data(self) -> None:
        """Verify Box-Cox returns error for negative data."""
        df = _two_factor_data().copy()
        df["y"] = [-1, -2, -3, -4]
        result = analyze_experiment(
            df,
            response_column="y",
            analysis_type="box_cox",
        )
        assert "error" in result["box_cox"]


# ---------------------------------------------------------------------------
# Lenth's method
# ---------------------------------------------------------------------------


class TestLenthMethod:
    def test_lenth_unreplicated(self) -> None:
        """Verify Lenth's method returns PSE, ME, and SME."""
        df = _three_factor_data()
        result = analyze_experiment(
            df,
            response_column="y",
            model="main_effects",
            analysis_type="lenth_method",
        )
        lm_result = result["lenth_method"]
        assert "PSE" in lm_result
        assert "ME" in lm_result
        assert "SME" in lm_result
        assert len(lm_result["effects"]) > 0

    def test_lenth_has_active_flags(self) -> None:
        """Verify Lenth effects include active flags."""
        df = _three_factor_data()
        result = analyze_experiment(
            df,
            response_column="y",
            model="main_effects",
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
        """Verify confidence interval records have ci_low and ci_high."""
        df = _two_factor_replicated()
        result = analyze_experiment(
            df,
            response_column="y",
            model="main_effects",
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
        """Verify predictions with new points include intervals."""
        df = _two_factor_replicated()
        new = pd.DataFrame({"A": [0, 0.5], "B": [0, -0.5]})
        result = analyze_experiment(
            df,
            response_column="y",
            model="main_effects",
            analysis_type="prediction",
            new_points=new,
        )
        preds = result["predictions"]
        assert len(preds) == 2
        assert "predicted" in preds[0]
        assert "pi_low" in preds[0]
        assert "pi_high" in preds[0]

    def test_prediction_without_new_points_errors(self) -> None:
        """Verify prediction without new_points returns error."""
        df = _two_factor_replicated()
        result = analyze_experiment(
            df,
            response_column="y",
            analysis_type="prediction",
        )
        assert "error" in result["prediction"]


# ---------------------------------------------------------------------------
# Confirmation test
# ---------------------------------------------------------------------------


class TestConfirmationTest:
    def test_confirmation_pass(self) -> None:
        """Verify confirmation test returns results with PI check."""
        df = _two_factor_replicated()
        new = pd.DataFrame({"A": [0], "B": [0]})
        result = analyze_experiment(
            df,
            response_column="y",
            model="main_effects",
            analysis_type="confirmation_test",
            new_points=new,
            observed_at_new=[28.0],
        )
        ct = result["confirmation_test"]
        assert "results" in ct
        assert "all_within_PI" in ct
        assert len(ct["results"]) == 1

    def test_confirmation_missing_args(self) -> None:
        """Verify confirmation test returns error when args missing."""
        df = _two_factor_replicated()
        result = analyze_experiment(
            df,
            response_column="y",
            analysis_type="confirmation_test",
        )
        assert "error" in result["confirmation_test"]


# ---------------------------------------------------------------------------
# Multiple analysis types at once
# ---------------------------------------------------------------------------


class TestMultipleAnalyses:
    def test_multiple_types(self) -> None:
        """Verify multiple analysis types run in a single call."""
        df = _two_factor_replicated()
        result = analyze_experiment(
            df,
            response_column="y",
            model="main_effects",
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
        """Verify log transform produces coefficients."""
        df = _two_factor_replicated()
        result = analyze_experiment(
            df,
            response_column="y",
            transform="log",
            analysis_type="coefficients",
        )
        assert "coefficients" in result

    def test_sqrt_transform(self) -> None:
        """Verify sqrt transform produces coefficients."""
        df = _two_factor_replicated()
        result = analyze_experiment(
            df,
            response_column="y",
            transform="sqrt",
            analysis_type="coefficients",
        )
        assert "coefficients" in result


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_unknown_analysis_type_raises(self) -> None:
        """Verify unknown analysis type raises ValueError."""
        df = _two_factor_data()
        with pytest.raises(ValueError, match="Unknown analysis_type"):
            analyze_experiment(df, response_column="y", analysis_type="bogus")

    def test_missing_response_column_raises(self) -> None:
        """Verify missing response column raises ValueError."""
        df = _two_factor_data()
        with pytest.raises(ValueError, match="not found"):
            analyze_experiment(df, response_column="missing")

    def test_no_response_arg_raises(self) -> None:
        """Verify omitting response argument raises ValueError."""
        df = pd.DataFrame({"A": [-1, 1], "B": [-1, 1]})
        with pytest.raises(ValueError, match="Must provide"):
            analyze_experiment(df)


# ---------------------------------------------------------------------------
# Responses as separate argument
# ---------------------------------------------------------------------------


class TestSeparateResponses:
    def test_responses_as_series(self) -> None:
        """Verify responses can be passed as a separate Series."""
        df = pd.DataFrame({"A": [-1, 1, -1, 1], "B": [-1, -1, 1, 1]})
        y = pd.Series([28, 36, 18, 31], name="y")
        result = analyze_experiment(df, responses=y, analysis_type="coefficients")
        assert "coefficients" in result

    def test_responses_as_dataframe(self) -> None:
        """Verify responses can be passed as a separate DataFrame."""
        df = pd.DataFrame({"A": [-1, 1, -1, 1], "B": [-1, -1, 1, 1]})
        y = pd.DataFrame({"y": [28, 36, 18, 31]})
        result = analyze_experiment(df, responses=y, analysis_type="coefficients")
        assert "coefficients" in result


def test_analyze_experiment_ignores_runorder_and_block_columns() -> None:
    """RunOrder / Block bookkeeping columns must not become model factors.

    A DesignResult frame carries a "RunOrder" (and optionally "Block") column;
    when the whole frame is passed with the response joined, these must not be
    crossed with the factors. RunOrder is dropped; the blocks enter once, as a
    fixed-effect contrast, never in an interaction.
    """
    df = pd.DataFrame(
        {
            "RunOrder": [1, 2, 3, 4, 5, 6, 7, 8],
            "Block": [1, 1, 1, 1, 2, 2, 2, 2],
            "A": [-1, 1, -1, 1, -1, 1, -1, 1],
            "B": [-1, -1, 1, 1, -1, -1, 1, 1],
            "y": [10, 14, 8, 12, 11, 15, 9, 13],
        }
    )
    res = analyze_experiment(df, response_column="y", model="interactions", analysis_type=["coefficients"])
    terms = [c["term"] for c in res["coefficients"]]
    assert not any("RunOrder" in t for t in terms)
    assert [t for t in terms if "Block" in t] == ["Block1"]
    # The real factors and their interaction are still present.
    assert "A" in terms
    assert "B" in terms
    assert "A:B" in terms


def test_a_response_named_yield_is_analysed() -> None:
    """'yield' is the commonest response in chemistry and a Python keyword, which patsy cannot name."""
    from process_improve.experiments import analyze_experiment

    rng = np.random.default_rng(0)
    x = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1], [0, 0], [0, 0]], dtype=float)
    data = pd.DataFrame(x, columns=["A", "B"])
    data["yield"] = 50 + 3 * x[:, 0] - 2 * x[:, 1] + rng.normal(0, 0.1, len(x))
    out = analyze_experiment(data, response_column="yield", model="main_effects", analysis_type=["coefficients"])
    assert out["model_summary"]["formula"].startswith("yield ~")
    coefficients = {c["term"]: c["coefficient"] for c in out["coefficients"]}
    assert coefficients["A"] == pytest.approx(3, abs=0.2)


def test_a_factor_named_with_a_keyword_raises_clearly() -> None:
    from process_improve.experiments import analyze_experiment

    data = pd.DataFrame({"lambda": [-1.0, 1.0, -1.0, 1.0], "y": [1.0, 2.0, 1.5, 2.5]})
    with pytest.raises(ValueError, match="keyword"):
        analyze_experiment(data, response_column="y", model="main_effects")


# ---------------------------------------------------------------------------
# Review findings: statistics checked against the textbook definitions
# ---------------------------------------------------------------------------


def _full_factorial(k: int, replicates: int = 1) -> np.ndarray:
    """Two-level full factorial in coded units, stacked ``replicates`` times."""
    return np.vstack([np.array(list(itertools.product([-1.0, 1.0], repeat=k)))] * replicates)


class TestEffectsCoding:
    """Effects are the change from the low to the high level, whatever units the factors are in."""

    def test_categorical_factor_effect_is_the_difference_of_level_means(self) -> None:
        df = pd.DataFrame({"A": [-1, 1, -1, 1], "B": ["lo", "lo", "hi", "hi"], "y": [28.0, 36.0, 18.0, 31.0]})
        result = analyze_experiment(df, response_column="y", model="main_effects", analysis_type="effects")
        coding = result["effects_coding"]["B"]
        high = df.y[coding["high"] == df.B].mean()
        low = df.y[coding["low"] == df.B].mean()
        assert result["effects"]["B"] == pytest.approx(high - low)
        assert abs(result["effects"]["B"]) == pytest.approx(7.5)

    def test_actual_unit_factor_effect_is_slope_times_range(self) -> None:
        df = pd.DataFrame({"A": [150, 170, 150, 170], "B": [-1, -1, 1, 1], "y": [28.0, 36.0, 18.0, 31.0]})
        result = analyze_experiment(df, response_column="y", model="main_effects", analysis_type="effects")
        assert result["effects"]["A"] == pytest.approx(10.5)  # mean at 170 minus mean at 150
        assert result["effects"]["B"] == pytest.approx(-7.5)
        assert result["effects_coding"] == {"A": {"low": 150.0, "high": 170.0}}

    def test_lenth_uses_the_same_coded_effects(self) -> None:
        x = _full_factorial(3)
        df = pd.DataFrame(x * 10 + 100, columns=list("ABC"))  # actual units 90 and 110
        df["y"] = 50 + 4 * x[:, 0] + np.array([0.1, -0.2, 0.3, 0.0, -0.1, 0.2, -0.3, 0.1])
        lenth = analyze_experiment(df, response_column="y", model="main_effects", analysis_type="lenth_method")
        effects = {e["term"]: e["effect"] for e in lenth["lenth_method"]["effects"]}
        assert effects["A"] == pytest.approx(8.0, abs=0.2)

    def test_a_three_level_categorical_has_no_single_effect(self) -> None:
        df = pd.DataFrame({"A": [-1, 1] * 3, "M": ["a", "a", "b", "b", "c", "c"], "y": [1.0, 2, 3, 4, 5, 7]})
        with pytest.raises(ValueError, match="two-level"):
            analyze_experiment(df, response_column="y", model="main_effects", analysis_type="effects")


class TestBlocks:
    """A Block column enters the model as a fixed effect (Montgomery, DAE chapter 7)."""

    @staticmethod
    def _blocked() -> pd.DataFrame:
        x = _full_factorial(3, replicates=2)
        df = pd.DataFrame(x, columns=list("ABC"))
        df["Block"] = [1] * 8 + [2] * 8
        rng = np.random.default_rng(0)
        df["y"] = 20 + 1.0 * df.A + 0.6 * df.B + np.where(df.Block == 2, 6.0, 0.0) + 0.5 * rng.standard_normal(16)
        return df

    def test_block_variation_is_removed_from_the_error(self) -> None:
        result = analyze_experiment(
            self._blocked(), response_column="y", model="main_effects", analysis_type=["anova", "significance"]
        )
        rows = {r["source"]: r for r in result["anova_table"]}
        assert rows["Block"]["df"] == 1
        assert rows["Block"]["p_value"] < 1e-6
        assert sum(r["df"] for r in result["anova_table"]) == 15
        assert set(result["significant_terms"]) == {"A", "B"}

    def test_three_blocks_are_one_anova_row_with_two_df(self) -> None:
        df = pd.DataFrame(_full_factorial(2, replicates=3), columns=["A", "B"])
        df["Block"] = ["a"] * 4 + ["b"] * 4 + ["c"] * 4
        df["y"] = 10 + 2 * df.A + df.Block.map({"a": 0.0, "b": 3.0, "c": -1.0}) + np.linspace(-0.2, 0.2, 12)
        rows = {
            r["source"]: r for r in analyze_experiment(df, response_column="y", model="main_effects")["anova_table"]
        }
        assert rows["Block"]["df"] == 2
        assert sum(r["df"] for r in rows.values()) == 11

    def test_prediction_without_a_block_is_for_the_average_block(self) -> None:
        df = self._blocked()
        result = analyze_experiment(
            df,
            response_column="y",
            model="main_effects",
            analysis_type="prediction",
            new_points=pd.DataFrame({"A": [0.0], "B": [0.0], "C": [0.0]}),
        )
        assert result["predictions"][0]["predicted"] == pytest.approx(df.y.mean())

    def test_a_mixture_model_warns_that_blocks_are_not_modelled(self) -> None:
        df = pd.DataFrame(
            [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)] * 2,
            columns=["x1", "x2", "x3"],
        )
        df["Block"] = [1] * 6 + [2] * 6
        df["y"] = 10 * df.x1 + 12 * df.x2 + 8 * df.x3 + np.arange(12) * 0.01
        with pytest.warns(UserWarning, match="blocks are not added"):
            analyze_experiment(df, response_column="y", model="scheffe_linear")


class TestTransformedResponse:
    """The model is fitted on the transformed response; every comparison is made on that scale."""

    @staticmethod
    def _lognormal() -> pd.DataFrame:
        rng = np.random.default_rng(2)
        x = _full_factorial(3, replicates=2)
        df = pd.DataFrame(x, columns=list("ABC"))
        df["y"] = np.exp(3 + 0.5 * x[:, 0] + 0.3 * x[:, 1] + 0.05 * rng.standard_normal(len(x)))
        return df

    def test_confirmation_run_at_the_true_mean_is_inside_the_interval(self) -> None:
        result = analyze_experiment(
            self._lognormal(),
            response_column="y",
            model="main_effects",
            transform="log",
            analysis_type=["prediction", "confirmation_test"],
            new_points=pd.DataFrame({"A": [1.0], "B": [1.0], "C": [0.0]}),
            observed_at_new=[float(np.exp(3.8))],
        )
        confirmation = result["confirmation_test"]
        assert confirmation["all_within_PI"] is True
        assert confirmation["results"][0]["observed_transformed"] == pytest.approx(3.8)
        assert confirmation["scale"] == "log of the response"
        assert result["prediction_scale"] == "log of the response"

    def test_box_cox_lambda_is_reported_and_reused(self) -> None:
        result = analyze_experiment(
            self._lognormal(),
            response_column="y",
            model="main_effects",
            transform="box_cox",
            analysis_type="confirmation_test",
            new_points=pd.DataFrame({"A": [1.0], "B": [1.0], "C": [0.0]}),
            observed_at_new=[float(np.exp(3.8))],
        )
        lmbda = result["model_summary"]["box_cox_lambda"]
        assert result["model_summary"]["transform"] == "box_cox"
        assert result["confirmation_test"]["results"][0]["observed_transformed"] == pytest.approx(
            np.expm1(lmbda * 3.8) / lmbda
        )

    @pytest.mark.parametrize(
        ("transform", "values", "match"),
        [
            ("Log", [28.0, 36, 18, 31, 29, 27], "Unknown transform"),
            ("log", [28.0, 36, -18, 31, 29, 27], "positive"),
            ("log", [28.0, 0, 18, 31, 29, 27], "positive"),
            ("box_cox", [28.0, 36, -18, 31, 29, 27], "positive"),
            ("sqrt", [28.0, 36, -18, 31, 29, 27], "non-negative"),
        ],
    )
    def test_a_transform_that_cannot_apply_raises(self, transform: str, values: list[float], match: str) -> None:
        df = pd.DataFrame({"A": [-1, 1, -1, 1, 0, 0], "B": [-1, -1, 1, 1, 0, 0], "y": values})
        with pytest.raises(ValueError, match=match):
            analyze_experiment(df, response_column="y", model="main_effects", transform=transform)


class TestCurvaturePureError:
    """Montgomery, DAE section 6.8: F = SS_curvature / MS_pure_error on (1, nC - 1) df."""

    @staticmethod
    def _example_6_6(shift: float = 0.0) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "A": [-1, 1, -1, 1, 0, 0, 0, 0, 0],
                "B": [-1, -1, 1, 1, 0, 0, 0, 0, 0],
                "y": [39.3, 40.9, 40.0, 41.5, *(np.array([40.3, 40.5, 40.7, 40.2, 40.6]) + shift)],
            }
        )

    def test_montgomery_example_6_6(self) -> None:
        result = analyze_experiment(self._example_6_6(), response_column="y", analysis_type="curvature_test")
        curvature = result["curvature_test"]
        assert curvature["ss_curvature"] == pytest.approx(0.0027, abs=1e-4)
        assert curvature["ms_pure_error"] == pytest.approx(0.043, abs=1e-4)
        assert curvature["df_pure_error"] == 4
        assert curvature["F_statistic"] == pytest.approx(0.063, abs=1e-3)
        assert curvature["significant"] is False

    def test_real_curvature_is_detected_whatever_the_model(self) -> None:
        df = self._example_6_6(shift=1.0)
        for model in ("main_effects", "interactions"):
            curvature = analyze_experiment(df, response_column="y", model=model, analysis_type="curvature_test")[
                "curvature_test"
            ]
            assert curvature["F_statistic"] == pytest.approx(55.36, abs=0.01)
            assert curvature["p_value"] == pytest.approx(0.00174, abs=1e-5)
            assert curvature["significant"] is True

    def test_significance_level_sets_the_flag(self) -> None:
        curvature = analyze_experiment(
            self._example_6_6(shift=1.0), response_column="y", analysis_type="curvature_test", significance_level=1e-4
        )["curvature_test"]
        assert curvature["significant"] is False
        assert curvature["significance_level"] == 1e-4


class TestMixtureAnalysis:
    """Cornell's yarn-elongation data (Experiments with Mixtures, 3rd ed., Table 2.3)."""

    @staticmethod
    def _yarn() -> pd.DataFrame:
        points = {
            (1, 0, 0): [11.0, 12.4],
            (0.5, 0.5, 0): [15.0, 14.8, 16.1],
            (0, 1, 0): [8.8, 10.0],
            (0, 0.5, 0.5): [10.0, 9.7, 11.8],
            (0, 0, 1): [16.8, 16.0],
            (0.5, 0, 0.5): [17.7, 16.4, 16.6],
        }
        return pd.DataFrame([(*p, y) for p, ys in points.items() for y in ys], columns=["x1", "x2", "x3", "y"])

    def test_cox_effects_include_the_blending_terms(self) -> None:
        # Fitted model 11.7 x1 + 9.4 x2 + 16.4 x3 + 19.0 x1x2 + 11.4 x1x3 - 9.6 x2x3: the response
        # change from the opposite face's midpoint to the vertex is 1.2, -7.5 and 1.1.
        result = analyze_experiment(
            self._yarn(), response_column="y", model="scheffe_quadratic", analysis_type="effects"
        )
        assert result["effects"] == pytest.approx({"x1": 1.2, "x2": -7.5, "x3": 1.1}, abs=1e-9)
        assert result["effect_range"] == pytest.approx({c: [0.0, 1.0] for c in ("x1", "x2", "x3")})

    def test_the_default_model_for_mixture_data_is_scheffe(self) -> None:
        result = analyze_experiment(self._yarn(), response_column="y", analysis_type="anova")
        assert result["model_summary"]["model"] == "scheffe_quadratic"
        assert result["model_summary"]["rank_deficient"] is False
        assert result["anova_table"][0]["source"] == "Linear mixture"

    def test_notes_do_not_overwrite_each_other_and_inestimable_terms_are_listed(self) -> None:
        df = self._yarn()
        df = df[(df[["x1", "x2", "x3"]] > 0).sum(axis=1) == 1].iloc[:4]  # vertices only: x1:x2:x3 inestimable
        df = pd.concat([df, self._yarn().iloc[2:5], self._yarn().iloc[9:12], self._yarn().iloc[14:]], ignore_index=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = analyze_experiment(
                df,
                response_column="y",
                model="scheffe_special_cubic",
                analysis_type=["anova", "significance", "effects", "lenth_method"],
            )
        assert "linear blending terms are tested together" in result["anova_note"]
        assert "Cox direction" in result["effects_note"]
        assert "Lenth" in result["lenth_note"]
        assert "note" not in result
        assert result["not_estimable_terms"] == ["x1:x2:x3"]


class TestAliasChainsInTests:
    """Exactly aliased terms are tested once, so the ANOVA degrees of freedom add up to N - 1."""

    def test_replicated_half_fraction(self) -> None:
        x = _full_factorial(3)
        x = np.vstack([np.c_[x, x.prod(axis=1)]] * 2)  # 2^(4-1), D = ABC, twice
        rng = np.random.default_rng(3)
        df = pd.DataFrame(x, columns=list("ABCD"))
        df["y"] = 10 + 3 * x[:, 0] + 2 * x[:, 0] * x[:, 1] + rng.standard_normal(16)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = analyze_experiment(df, response_column="y", analysis_type=["anova", "significance", "effects"])
        table = result["anova_table"]
        assert sum(r["df"] for r in table) == 15
        assert sum(r["sum_sq"] for r in table) == pytest.approx(((df.y - df.y.mean()) ** 2).sum())
        assert result["significant_terms"] == ["A", "A:B + C:D"]
        assert {r["source"] for r in table} - {"Residual"} == set(result["effects"])

    def test_anova_reports_its_sum_of_squares_type_and_mean_squares(self) -> None:
        result = analyze_experiment(_two_factor_with_center(), response_column="y", model="main_effects")
        assert result["anova_type"] == "II"
        for row in result["anova_table"]:
            assert row["mean_sq"] == pytest.approx(row["sum_sq"] / row["df"])


class TestSaturatedModel:
    def test_significance_and_residual_tests_say_there_is_no_error(self) -> None:
        df = pd.DataFrame(_full_factorial(3), columns=list("ABC"))
        df["y"] = [60, 72, 54, 68, 52, 83, 45, 80.0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = analyze_experiment(
                df, response_column="y", model="(A+B+C)**3", analysis_type=["significance", "residual_diagnostics"]
            )
        assert result["significant_terms"] == []
        assert len(result["not_estimable_terms"]) == 7
        assert "lenth_method" in result["significance_note"]
        diagnostics = result["residual_diagnostics"]
        assert diagnostics["shapiro_wilk"]["p_value"] is None
        assert diagnostics["breusch_pagan"]["p_value"] is None
        assert "no residual degrees of freedom" in diagnostics["note"]


class TestLenthDefinition:
    """Lenth (1989): PSE trims strictly below 2.5 s0, and SME uses gamma = (1 + 0.95**(1/m)) / 2."""

    def test_pse_and_sme_match_lenth(self) -> None:
        x = _full_factorial(3)
        df = pd.DataFrame(x, columns=list("ABC"))
        effects = {"A": 1.0, "B": 1.0, "C": 1.0, "A:B": 2.0, "A:C": 2.0, "B:C": 2.0, "A:B:C": 8.0}
        df["y"] = sum(e / 2 * np.prod([df[f] for f in term.split(":")], axis=0) for term, e in effects.items())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lenth = analyze_experiment(df, response_column="y", model="(A+B+C)**3", analysis_type="lenth_method")[
                "lenth_method"
            ]
        # s0 = 1.5 * 2 = 3, so 8 > 2.5 s0 is trimmed: PSE = 1.5 * median(1, 1, 1, 2, 2, 2).
        assert lenth["PSE"] == pytest.approx(2.25)
        assert lenth["SME"] / lenth["PSE"] == pytest.approx(9.008, abs=1e-3)  # Lenth's table, m = 7

    def test_an_effect_exactly_at_the_cutoff_is_trimmed(self, monkeypatch) -> None:
        """The trim is strict (|c| < 2.5 s0), checked on exact effects.

        Fitted coefficients carry rounding noise, which decides the side of an effect placed
        exactly on the cut-off, so the boundary is tested with the coefficients given directly.
        """
        from types import SimpleNamespace

        from process_improve.experiments._analyses import lenth as lenth_module

        effects = pd.Series([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 7.5], index=["A", "B", "C", "A:B", "A:C", "B:C", "A:B:C"])
        monkeypatch.setattr(lenth_module, "estimable_effects", lambda _fit: SimpleNamespace(coefficients=effects / 2))
        out = lenth_module._run_lenth_method(None)["lenth_method"]
        # s0 = 1.5 * 2 = 3, and 7.5 = 2.5 s0 exactly: not strictly below, so it is trimmed.
        assert out["PSE"] == 2.25


class TestReviewedInputs:
    def test_lack_of_fit_uses_the_significance_level(self) -> None:
        a = 1.414
        df = pd.DataFrame(
            {
                "A": [-1, 1, -1, 1, -a, a, 0, 0, 0, 0, 0, 0, 0],
                "B": [-1, -1, 1, 1, 0, 0, -a, a, 0, 0, 0, 0, 0],
                "y": [76.5, 78.0, 77.0, 79.5, 75.6, 78.4, 77.0, 78.5, 79.9, 80.3, 80.0, 79.7, 79.8],
            }
        )
        lof = analyze_experiment(
            df, response_column="y", model="main_effects", analysis_type="lack_of_fit", significance_level=1e-6
        )["lack_of_fit"]
        assert lof["p_value"] < 0.01
        assert lof["significant"] is False

    def test_missing_responses_are_dropped_before_every_analysis(self) -> None:
        a = 1.414
        df = pd.DataFrame(
            {
                "A": [-1, 1, -1, 1, -a, a, 0, 0, 0, 0, 0, 0, 0],
                "B": [-1, -1, 1, 1, 0, 0, -a, a, 0, 0, 0, 0, 0],
                "y": [76.5, 78.0, 77.0, 79.5, 75.6, 78.4, 77.0, 78.5, np.nan, 80.3, 80.0, 79.7, 79.8],
            }
        )
        with pytest.warns(UserWarning, match="1 run"):
            result = analyze_experiment(
                df, response_column="y", model="quadratic", analysis_type=["lack_of_fit", "model_selection"]
            )
        assert result["model_summary"]["n_obs"] == 12
        assert result["lack_of_fit"]["df_pure_error"] == 3
        assert result["lack_of_fit"]["df_lack_of_fit"] == 3
        assert np.isfinite(result["model_selection"]["criterion_value"])

    def test_confirmation_needs_one_observation_per_point(self) -> None:
        new = pd.DataFrame({"A": [0.5, -0.5], "B": [0.5, 0.0]})
        for observed in ([30.0], [30.0, 31.0, 99.0]):
            with pytest.raises(ValueError, match="one observation per new point"):
                analyze_experiment(
                    _two_factor_with_center(),
                    response_column="y",
                    model="main_effects",
                    analysis_type="confirmation_test",
                    new_points=new,
                    observed_at_new=observed,
                )

    def test_a_formula_for_another_response_raises(self) -> None:
        df = _two_factor_with_center().assign(z=[1.0, 2, 3, 4, 5, 6, 7])
        with pytest.raises(ValueError, match="response being analysed is 'y'"):
            analyze_experiment(df, response_column="y", model="z ~ A + B")

    def test_a_formula_uses_only_the_columns_it_names(self) -> None:
        df = _two_factor_replicated().assign(z=np.arange(8.0))
        lof = analyze_experiment(df, response_column="y", model="y ~ A + B", analysis_type="lack_of_fit")["lack_of_fit"]
        assert lof["df_pure_error"] == 4  # replicates over A and B; z, not in the formula, does not split them

    def test_a_formula_may_name_a_keyword_response(self) -> None:
        df = _two_factor_with_center().rename(columns={"y": "yield"})
        result = analyze_experiment(df, response_column="yield", model="yield ~ A + B")
        assert result["model_summary"]["formula"] == "yield ~ A + B"

    def test_formula_term_cap(self) -> None:
        names = [chr(65 + i) for i in range(8)]
        rng = np.random.default_rng(0)
        df = pd.DataFrame(rng.choice([-1.0, 1.0], size=(32, 8)), columns=names)
        df["y"] = rng.standard_normal(32)
        with pytest.raises(ValueError, match="max_formula_terms"):
            analyze_experiment(df, response_column="y", model="(" + "+".join(names) + ")**8")

    def test_an_unnamed_series_takes_the_response_column_name(self) -> None:
        df = _two_factor_data()[["A", "B"]]
        series = pd.Series([28.0, 36, 18, 31])
        result = analyze_experiment(df, series, response_column="y", model="main_effects", analysis_type="effects")
        assert result["effects"]["A"] == pytest.approx(10.5)
        with pytest.raises(ValueError, match="unnamed Series"):
            analyze_experiment(df, series, analysis_type="effects")

    def test_quadratic_model_skips_squares_of_categorical_factors(self) -> None:
        df = pd.DataFrame({"A": [-1, 1, -1, 1, 0, 0], "M": ["a", "b", "a", "b", "a", "b"], "y": [1.0, 2, 3, 5, 2, 3]})
        result = analyze_experiment(df, response_column="y", model="quadratic")
        assert result["model_summary"]["formula"] == "y ~ (A + M) ** 2 + I(A ** 2)"


class TestBoxCoxProfile:
    """Box and Cox (1964): lambda maximises the profile likelihood of the fitted model."""

    def test_additive_data_need_no_transform(self) -> None:
        rng = np.random.default_rng(1)
        x = _full_factorial(3, replicates=2)
        df = pd.DataFrame(x, columns=list("ABC"))
        df["y"] = 20 + 15 * x[:, 0] + 3 * x[:, 1] + 0.5 * rng.standard_normal(len(x))
        box_cox = analyze_experiment(df, response_column="y", model="main_effects", analysis_type="box_cox")["box_cox"]
        assert box_cox["lambda"] == pytest.approx(0.977, abs=0.002)
        assert box_cox["lambda_ci"][0] < 1.0 < box_cox["lambda_ci"][1]
        assert box_cox["recommendation"].startswith("no transform")

    def test_multiplicative_data_recommend_a_log(self) -> None:
        df = TestTransformedResponse._lognormal()
        box_cox = analyze_experiment(df, response_column="y", model="main_effects", analysis_type="box_cox")["box_cox"]
        assert box_cox["lambda_ci"][0] < 0.0 < box_cox["lambda_ci"][1]
        assert box_cox["recommendation"] == "log transform"


class TestModelSelectionCriterion:
    """AICc of Hurvich and Tsai (1989), and Scheffe models searched without an intercept."""

    def test_aicc_counts_the_error_variance(self) -> None:
        import statsmodels.formula.api as smf

        from process_improve.experiments._analyses import model_selection as ms

        x = _full_factorial(3)
        df = pd.DataFrame(x, columns=list("ABC"))
        df["y"] = 5 + 2 * x[:, 0] + x[:, 1] + 0.3 * np.random.default_rng(0).standard_normal(8)
        score = ms._Scorer(df, df.y.to_numpy(), ms._candidate_terms(list("ABC"), "main_effects"), "aicc")
        for included in (frozenset({"A", "B"}), frozenset({"A", "B", "C"})):
            n, p = 8, 1 + len(included)
            ssr = smf.ols("y ~ " + " + ".join(sorted(included)), df).fit().ssr
            hurvich_tsai = n * np.log(ssr / n) + n * (n + p) / (n - p - 2) + n * np.log(2 * np.pi)
            assert score(included) == pytest.approx(hurvich_tsai)
        saturated = ms._Scorer(df, df.y.to_numpy(), ms._candidate_terms(list("ABC"), "interactions"), "aicc")
        assert saturated(frozenset({"A", "B", "C", "A:B", "A:C"})) == np.inf  # n - p - 2 = 0

    def test_scheffe_model_selection_keeps_the_linear_blending_terms_and_no_intercept(self) -> None:
        df = TestMixtureAnalysis._yarn()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = analyze_experiment(
                df, response_column="y", model="scheffe_quadratic", analysis_type="model_selection"
            )
        chosen = result["model_selection"]
        assert chosen["selected_formula"].startswith("y ~ -1 + x1 + x2 + x3")
        assert chosen["candidate_model"] == "scheffe_quadratic"
        assert "note" not in chosen
        assert chosen["n_terms"] == len(chosen["selected_terms"])

    def test_n_terms_counts_the_intercept_like_model_summary(self) -> None:
        result = analyze_experiment(
            _three_factor_data(), response_column="y", model="main_effects", analysis_type="model_selection"
        )
        chosen = result["model_selection"]
        assert chosen["n_terms"] == len(chosen["selected_terms"]) + 1


# ---------------------------------------------------------------------------
# The coefficients' scale: coding="actual", "coded", or ranges (#513)
# ---------------------------------------------------------------------------


def _reactor(kelvin: bool = False) -> pd.DataFrame:
    """Return a 2^2 factorial in actual units with two centre runs: T in degC (or K), P in bar."""
    T = np.array([150.0, 200.0, 150.0, 200.0, 175.0, 175.0])
    return pd.DataFrame(
        {"T": T + 273.15 if kelvin else T, "P": [1, 1, 3, 3, 2, 2], "y": [10.0, 18.0, 13.0, 25.0, 16.0, 16.8]}
    )


def _terms(result: dict, key: str = "coefficient") -> dict[str, float]:
    return {row["term"]: row[key] for row in result["coefficients"]}


class TestCoding:
    """``coding`` sets the scale of the coefficients and their confidence intervals."""

    def test_actual_is_the_default_and_is_the_fit_as_given(self) -> None:
        result = analyze_experiment(_reactor(), response_column="y", model="y ~ T * P", analysis_type="coefficients")
        assert result["coding"] == "actual"
        assert _terms(result)["T"] == pytest.approx(0.12)
        assert result["factor_coding"] == {"T": {"low": 150.0, "high": 200.0}, "P": {"low": 1.0, "high": 3.0}}

    def test_coded_coefficients_are_half_the_effects(self) -> None:
        result = analyze_experiment(
            _reactor(),
            response_column="y",
            model="y ~ T * P",
            analysis_type=["coefficients", "effects"],
            coding="coded",
        )
        assert result["coding"] == "coded"
        coefficients = _terms(result)
        for term, effect in result["effects"].items():
            assert coefficients[term] == pytest.approx(effect / 2)

    def test_a_coded_main_effect_is_tested_at_the_centre(self) -> None:
        """On the actual scale P is tested at 0 degC, and misses what the ANOVA finds."""
        kwargs = {"response_column": "y", "model": "y ~ T * P", "analysis_type": ["coefficients", "anova"]}
        actual = analyze_experiment(_reactor(), **kwargs)
        coded = analyze_experiment(_reactor(), coding="coded", **kwargs)
        anova_p = next(row["p_value"] for row in coded["anova_table"] if row["source"] == "P")
        assert _terms(coded, "p_value")["P"] == pytest.approx(anova_p)
        assert _terms(actual)["P"] < 0 < _terms(coded)["P"]
        assert _terms(actual, "p_value")["P"] > 0.05 > anova_p

    def test_the_unit_of_temperature_does_not_change_coded_coefficients(self) -> None:
        kwargs = {"response_column": "y", "model": "y ~ T * P", "analysis_type": "coefficients"}
        celsius, kelvin = (analyze_experiment(_reactor(k), coding="coded", **kwargs) for k in (False, True))
        assert _terms(kelvin, "p_value") == pytest.approx(_terms(celsius, "p_value"))
        actual = [_terms(analyze_experiment(_reactor(k), **kwargs), "p_value")["P"] for k in (False, True)]
        assert actual[0] != pytest.approx(actual[1], rel=0.1)

    def test_a_range_puts_minus_and_plus_one_at_its_levels(self) -> None:
        """A central composite design's extremes are its axial runs; a range codes from the cube instead."""
        x = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1], [-1.414, 0], [1.414, 0], [0, -1.414], [0, 1.414], [0, 0]])
        y = 60 + 4 * x[:, 0] + 2 * x[:, 1] - 3 * x[:, 0] ** 2 - x[:, 1] ** 2 + x[:, 0] * x[:, 1]
        y = y + np.array([0.3, -0.2, 0.1, -0.1, 0.2, -0.3, 0.1, 0.2, -0.1])
        coded_data = pd.DataFrame({"T": x[:, 0], "P": x[:, 1], "y": y})
        actual_data = coded_data.assign(T=175 + 25 * coded_data["T"], P=2 + coded_data["P"])
        cube = {"T": {"low": 150, "high": 200}, "P": {"low": 1, "high": 3}}
        kwargs = {"response_column": "y", "model": "quadratic", "analysis_type": "coefficients"}
        expected = _terms(analyze_experiment(coded_data, **kwargs))
        assert _terms(analyze_experiment(actual_data, coding=cube, **kwargs)) == pytest.approx(expected)
        axial = _terms(analyze_experiment(actual_data, coding="coded", **kwargs))
        assert axial["T"] == pytest.approx(1.414 * expected["T"])

    def test_coded_data_is_left_as_it_is(self) -> None:
        kwargs = {"response_column": "y", "model": "interactions", "analysis_type": "coefficients"}
        coded = analyze_experiment(_two_factor_replicated(), coding="coded", **kwargs)
        assert coded["coefficients"] == analyze_experiment(_two_factor_replicated(), **kwargs)["coefficients"]
        assert "factor_coding" not in coded

    def test_categorical_factors(self) -> None:
        """Two levels are coded -1/+1 as for the effects; more levels get sum-to-zero contrasts."""
        T = [150.0, 200.0] * 6
        y = [10.0, 18.0, 12.0, 21.0, 9.0, 16.0, 11.0, 19.0, 12.0, 22.0, 8.0, 15.0]
        kwargs = {"response_column": "y", "model": "interactions", "analysis_type": "coefficients", "coding": "coded"}
        two = analyze_experiment(pd.DataFrame({"T": T, "S": ["Dry", "Wet"] * 3 + ["Wet", "Dry"] * 3, "y": y}), **kwargs)
        assert two["factor_coding"]["S"] == {"low": "Dry", "high": "Wet"}
        assert "S" in _terms(two)
        three = analyze_experiment(pd.DataFrame({"T": T, "S": ["a", "a", "b", "b", "c", "c"] * 2, "y": y}), **kwargs)
        assert {"S[S.a]", "S[S.b]", "T:S[S.a]", "T:S[S.b]"} <= set(_terms(three))
        assert _terms(three)["Intercept"] == pytest.approx(np.mean(y))  # balanced: the grand mean
        assert "S" not in three["factor_coding"]

    def test_confidence_intervals_follow_the_coding(self) -> None:
        result = analyze_experiment(
            _reactor(),
            response_column="y",
            model="y ~ T * P",
            analysis_type=["coefficients", "confidence_intervals"],
            coding="coded",
        )
        intervals = {row["term"]: (row["ci_low"], row["ci_high"]) for row in result["confidence_intervals"]}
        for row in result["coefficients"]:
            assert intervals[row["term"]] == pytest.approx((row["ci_low"], row["ci_high"]))

    def test_other_analyses_do_not_change(self) -> None:
        kwargs = {"response_column": "y", "model": "y ~ T * P", "analysis_type": ["anova", "effects"]}
        actual = analyze_experiment(_reactor(), **kwargs)
        coded = analyze_experiment(_reactor(), coding="coded", **kwargs)
        assert coded["anova_table"] == actual["anova_table"]
        assert coded["effects"] == actual["effects"]

    def test_a_model_that_changes_when_coded_warns(self) -> None:
        """``T:P`` without ``P`` is a different model once P is centred."""
        with pytest.warns(UserWarning, match="fits the data differently"):
            analyze_experiment(
                _reactor(), response_column="y", model="y ~ T + T:P", analysis_type="coefficients", coding="coded"
            )
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            analyze_experiment(
                _reactor(), response_column="y", model="y ~ T * P", analysis_type="coefficients", coding="coded"
            )

    def test_a_mixture_model_has_no_coded_scale(self) -> None:
        mixture = TestMixtureAnalysis._yarn()
        with pytest.raises(ValueError, match="mixture model has no coded scale"):
            analyze_experiment(
                mixture, response_column="y", model="scheffe_quadratic", analysis_type="coefficients", coding="coded"
            )
        actual = analyze_experiment(
            mixture, response_column="y", model="scheffe_quadratic", analysis_type="coefficients"
        )
        assert actual["coding"] == "actual"
        assert "factor_coding" not in actual  # proportions are not to be recoded by optimize_responses

    @pytest.mark.parametrize(
        ("coding", "match"),
        [
            ("nonsense", "must be 'actual', 'coded'"),
            (3, "must be 'actual', 'coded'"),
            ({"Q": {"low": 1, "high": 2}}, "not a factor"),
            ({"T": {"low": 200, "high": 150}}, "finite low < high"),
            ({"T": (150, 200)}, "finite low < high"),
            ({"T": {"low": 150, "high": np.inf}}, "finite low < high"),
        ],
    )
    def test_invalid_coding(self, coding: object, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            analyze_experiment(
                _reactor(), response_column="y", model="y ~ T * P", analysis_type="coefficients", coding=coding
            )
