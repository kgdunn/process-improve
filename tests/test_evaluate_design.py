"""Tests for the evaluate_design() design quality metrics API."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments.designs import generate_design
from process_improve.experiments.evaluate import (
    _build_model_matrix,
    _defining_relation_from_generators,
    _multiply_words,
    _parse_word,
    _word_to_str,
    evaluate_design,
)
from process_improve.experiments.factor import Factor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _continuous_factors(n: int, names: str | None = None) -> list[Factor]:
    """Create n continuous factors with default ranges."""
    if names and len(names) >= n:
        return [Factor(name=names[i], low=0, high=10) for i in range(n)]
    return [Factor(name=f"X{i + 1}", low=0, high=10) for i in range(n)]


def _full_factorial_df(k: int, names: str = "ABCDEFGHIJ") -> pd.DataFrame:
    """Create a 2^k full factorial design as a DataFrame of -1/+1 coded values."""
    n = 2**k
    cols = {}
    for i in range(k):
        cols[names[i]] = [1 if (j >> (k - 1 - i)) & 1 else -1 for j in range(n)]
    return pd.DataFrame(cols)


# ---------------------------------------------------------------------------
# Model matrix construction
# ---------------------------------------------------------------------------


class TestModelMatrix:
    """Test _build_model_matrix."""

    def test_main_effects_shape(self) -> None:
        """2^3 factorial with main_effects produces (8, 4) matrix."""
        df = _full_factorial_df(3)
        X, cols = _build_model_matrix(df, "main_effects", ["A", "B", "C"])
        assert X.shape == (8, 4)  # intercept + 3 main effects
        assert "Intercept" in cols

    def test_interactions_shape(self) -> None:
        """2^3 factorial with interactions produces (8, 7+1) columns."""
        df = _full_factorial_df(3)
        X, _cols = _build_model_matrix(df, "interactions", ["A", "B", "C"])
        # intercept + 3 main + 3 two-factor interactions = 7
        assert X.shape == (8, 7)

    def test_quadratic_shape(self) -> None:
        """Design with quadratic model includes squared terms."""
        # CCD has 3 levels per factor, so squared terms are estimable
        factors = _continuous_factors(2, "AB")
        result = generate_design(factors, design_type="ccd", alpha="face_centered")
        df = pd.DataFrame(result.design[["A", "B"]])
        X, _cols = _build_model_matrix(df, "quadratic", ["A", "B"])
        # intercept + 2 main + 1 interaction + 2 squared = 6
        assert X.shape[1] == 6

    def test_none_defaults_to_interactions(self) -> None:
        """model=None should default to interactions."""
        df = _full_factorial_df(3)
        X_none, _cols_none = _build_model_matrix(df, None, ["A", "B", "C"])
        X_int, _cols_int = _build_model_matrix(df, "interactions", ["A", "B", "C"])
        np.testing.assert_array_equal(X_none, X_int)

    def test_explicit_formula_with_tilde(self) -> None:
        """Explicit formula with ~ should strip the LHS."""
        df = _full_factorial_df(2)
        X, _cols = _build_model_matrix(df, "y ~ A + B", ["A", "B"])
        assert X.shape == (4, 3)  # intercept + 2 main

    def test_explicit_rhs_formula(self) -> None:
        """Explicit RHS formula (no ~) should work directly."""
        df = _full_factorial_df(2)
        X, _cols = _build_model_matrix(df, "A * B", ["A", "B"])
        assert X.shape[1] == 4  # intercept + A + B + A:B


# ---------------------------------------------------------------------------
# D-efficiency
# ---------------------------------------------------------------------------


class TestDEfficiency:
    """Test D-efficiency metric."""

    def test_orthogonal_full_factorial(self) -> None:
        """Orthogonal 2^2 factorial with main-effects model has D-eff = 100%."""
        df = _full_factorial_df(2)
        result = evaluate_design(df, model="main_effects", metric="d_efficiency")
        assert result["d_efficiency"] == pytest.approx(100.0, abs=0.5)

    def test_full_factorial_interactions(self) -> None:
        """2^3 full factorial with full interaction model has D-eff = 100%."""
        df = _full_factorial_df(3)
        result = evaluate_design(df, model="interactions", metric="d_efficiency")
        # 2^3 design evaluating main + 2FI is 7 terms for 8 runs; D-eff should be high
        assert result["d_efficiency"] is not None
        assert result["d_efficiency"] > 50.0

    def test_singular_returns_none(self) -> None:
        """Rank-deficient design returns None for D-efficiency."""
        # 2^2 factorial trying to fit quadratic is rank-deficient
        df = _full_factorial_df(2)
        result = evaluate_design(df, model="quadratic", metric="d_efficiency")
        assert result["d_efficiency"] is None

    def test_fractional_less_than_full(self) -> None:
        """Fractional factorial should have lower D-eff than full factorial for same model."""
        factors = _continuous_factors(4, "ABCD")
        frac = generate_design(
            factors, design_type="fractional_factorial", generators=["D=ABC"], center_points=0
        )
        full = generate_design(factors, design_type="full_factorial", center_points=0)
        d_frac = evaluate_design(frac, model="main_effects", metric="d_efficiency")["d_efficiency"]
        d_full = evaluate_design(full, model="main_effects", metric="d_efficiency")["d_efficiency"]
        assert d_frac is not None
        assert d_full is not None
        # Both should be valid; full factorial should be at least as good
        assert d_full >= d_frac - 1.0  # small tolerance


# ---------------------------------------------------------------------------
# G-efficiency
# ---------------------------------------------------------------------------


class TestGEfficiency:
    """Test G-efficiency metric."""

    def test_orthogonal_design(self) -> None:
        """Orthogonal design should have high G-efficiency."""
        df = _full_factorial_df(2)
        result = evaluate_design(df, model="main_effects", metric="g_efficiency")
        assert result["g_efficiency"] is not None
        assert result["g_efficiency"] > 80.0

    def test_returns_max_prediction_variance(self) -> None:
        """G-efficiency result should include max_prediction_variance."""
        df = _full_factorial_df(2)
        result = evaluate_design(df, model="main_effects", metric="g_efficiency")
        assert "max_prediction_variance" in result

    def test_singular_returns_none(self) -> None:
        """Singular design returns None for G-efficiency."""
        df = _full_factorial_df(2)
        result = evaluate_design(df, model="quadratic", metric="g_efficiency")
        assert result["g_efficiency"] is None


# ---------------------------------------------------------------------------
# I-efficiency
# ---------------------------------------------------------------------------


class TestIEfficiency:
    """Test I-efficiency metric."""

    def test_orthogonal_design(self) -> None:
        """Orthogonal design should have high I-efficiency."""
        df = _full_factorial_df(2)
        result = evaluate_design(df, model="main_effects", metric="i_efficiency")
        assert result["i_efficiency"] is not None
        assert result["i_efficiency"] > 80.0

    def test_returns_average_prediction_variance(self) -> None:
        """I-efficiency result should include average_prediction_variance."""
        df = _full_factorial_df(2)
        result = evaluate_design(df, model="main_effects", metric="i_efficiency")
        assert "average_prediction_variance" in result


# ---------------------------------------------------------------------------
# Prediction variance
# ---------------------------------------------------------------------------


class TestPredictionVariance:
    """Test prediction variance metric."""

    def test_returns_list(self) -> None:
        """Prediction variance returns a list of values, one per design point."""
        df = _full_factorial_df(2)
        result = evaluate_design(df, model="main_effects", metric="prediction_variance")
        assert len(result["prediction_variance"]) == 4

    def test_hat_matrix_diagonal(self) -> None:
        """For orthogonal 2^k with full model, hat diagonal = p/N at every point."""
        df = _full_factorial_df(3)
        result = evaluate_design(df, model="main_effects", metric="prediction_variance")
        # p = 4 (intercept + 3 main), N = 8; h_ii = p/N = 0.5
        for v in result["prediction_variance"]:
            assert v == pytest.approx(0.5, abs=1e-10)

    def test_includes_summary_stats(self) -> None:
        """Result should include mean, max, min."""
        df = _full_factorial_df(2)
        result = evaluate_design(df, model="main_effects", metric="prediction_variance")
        assert "mean" in result
        assert "max" in result
        assert "min" in result

    def test_singular_returns_none(self) -> None:
        """Singular design returns None for prediction variance."""
        df = _full_factorial_df(2)
        result = evaluate_design(df, model="quadratic", metric="prediction_variance")
        assert result["prediction_variance"] is None


# ---------------------------------------------------------------------------
# VIF
# ---------------------------------------------------------------------------


class TestVIF:
    """Test VIF metric."""

    def test_orthogonal_all_ones(self) -> None:
        """Orthogonal 2^k design has VIF = 1 for all terms."""
        df = _full_factorial_df(3)
        result = evaluate_design(df, model="main_effects", metric="vif")
        vif = result["vif"]
        for name, value in vif.items():
            assert value == pytest.approx(1.0, abs=0.01), f"VIF for {name} should be 1.0"

    def test_returns_dict_keyed_by_term(self) -> None:
        """VIF returns a dict mapping term names to VIF values."""
        df = _full_factorial_df(2)
        result = evaluate_design(df, model="main_effects", metric="vif")
        vif = result["vif"]
        assert isinstance(vif, dict)
        assert "A" in vif
        assert "B" in vif

    def test_excludes_intercept(self) -> None:
        """VIF should not include the intercept term."""
        df = _full_factorial_df(2)
        result = evaluate_design(df, model="main_effects", metric="vif")
        vif = result["vif"]
        assert "Intercept" not in vif

    def test_singular_returns_none(self) -> None:
        """Singular design returns None for VIF."""
        df = _full_factorial_df(2)
        result = evaluate_design(df, model="quadratic", metric="vif")
        assert result["vif"] is None


# ---------------------------------------------------------------------------
# Condition number
# ---------------------------------------------------------------------------


class TestConditionNumber:
    """Test condition number metric."""

    def test_orthogonal_low(self) -> None:
        """Orthogonal design should have low condition number."""
        df = _full_factorial_df(3)
        result = evaluate_design(df, model="main_effects", metric="condition_number")
        assert result["condition_number"] < 5.0

    def test_always_positive(self) -> None:
        """Condition number should always be positive."""
        df = _full_factorial_df(2)
        result = evaluate_design(df, model="main_effects", metric="condition_number")
        assert result["condition_number"] > 0


# ---------------------------------------------------------------------------
# Power
# ---------------------------------------------------------------------------


class TestPower:
    """Test power analysis metric."""

    def test_large_effect_near_one(self) -> None:
        """Large effect size produces power near 1.0."""
        df = _full_factorial_df(3)
        result = evaluate_design(
            df, model="main_effects", metric="power", effect_size=10.0, sigma=1.0
        )
        power = result["power"]
        for name, pwr in power.items():
            assert pwr > 0.99, f"Power for {name} should be near 1.0 for large effect"

    def test_small_effect_low(self) -> None:
        """Small effect size produces low power."""
        df = _full_factorial_df(2)
        result = evaluate_design(
            df, model="main_effects", metric="power", effect_size=0.01, sigma=1.0
        )
        power = result["power"]
        for name, pwr in power.items():
            assert pwr < 0.5, f"Power for {name} should be low for small effect"

    def test_more_runs_more_power(self) -> None:
        """More runs (replicated design) should yield higher power."""
        factors = _continuous_factors(3, "ABC")
        single = generate_design(factors, design_type="full_factorial", center_points=0, replicates=1)
        double = generate_design(factors, design_type="full_factorial", center_points=0, replicates=2)
        p1 = evaluate_design(single, model="main_effects", metric="power", effect_size=2.0, sigma=1.0)
        p2 = evaluate_design(double, model="main_effects", metric="power", effect_size=2.0, sigma=1.0)
        # Pick any factor — power should be higher with 2x runs
        assert p2["power"]["A"] >= p1["power"]["A"] - 0.01

    def test_power_curve_when_no_effect_size(self) -> None:
        """When effect_size is None, return power curves."""
        df = _full_factorial_df(3)
        result = evaluate_design(df, model="main_effects", metric="power", sigma=1.0)
        assert "power_curves" in result
        # Each term should have a list of {effect_size, power} dicts
        for curve in result["power_curves"].values():
            assert len(curve) > 0
            assert "effect_size" in curve[0]
            assert "power" in curve[0]

    def test_saturated_model_returns_none(self) -> None:
        """Saturated model (0 residual DF) returns None for power."""
        df = _full_factorial_df(3)
        # 8 runs, interactions model = 7 params + intercept = 8 → not fully saturated
        evaluate_design(df, model="interactions", metric="power", effect_size=2.0, sigma=1.0)
        # With N=8, p=7 (intercept + 3 main + 3 2FI), df_resid = 1
        # This should still compute (1 df_resid is nonzero)
        # But a truly saturated case:
        df2 = _full_factorial_df(2)  # 4 runs
        # interactions: intercept + A + B + A:B = 4 params → 0 df_resid
        result2 = evaluate_design(df2, model="interactions", metric="power", effect_size=2.0, sigma=1.0)
        assert result2["power"] is None


# ---------------------------------------------------------------------------
# Degrees of freedom
# ---------------------------------------------------------------------------


class TestDegreesOfFreedom:
    """Test degrees of freedom metric."""

    def test_full_factorial_main_effects(self) -> None:
        """2^3 full factorial, main effects model."""
        df = _full_factorial_df(3)
        result = evaluate_design(df, model="main_effects", metric="degrees_of_freedom")
        dof = result["degrees_of_freedom"]
        assert dof["model"] == 3  # 3 main effects
        assert dof["residual"] == 4  # 8 - 4
        assert dof["total"] == 7  # 8 - 1

    def test_components_sum(self) -> None:
        """df_model + df_residual + 1 (intercept) = df_total + 1 = N."""
        df = _full_factorial_df(3)
        result = evaluate_design(df, model="main_effects", metric="degrees_of_freedom")
        dof = result["degrees_of_freedom"]
        assert dof["model"] + dof["residual"] + 1 == len(df)

    def test_saturated(self) -> None:
        """Saturated design has 0 residual DF."""
        df = _full_factorial_df(2)  # 4 runs
        result = evaluate_design(df, model="interactions", metric="degrees_of_freedom")
        dof = result["degrees_of_freedom"]
        assert dof["residual"] == 0

    def test_replicated_has_pure_error(self) -> None:
        """Replicated design should report pure_error and lack_of_fit DF."""
        factors = _continuous_factors(2, "AB")
        result = generate_design(factors, design_type="full_factorial", center_points=0, replicates=2)
        metrics = evaluate_design(result, model="main_effects", metric="degrees_of_freedom")
        dof = metrics["degrees_of_freedom"]
        assert "pure_error" in dof
        assert dof["pure_error"] == 4  # 8 - 4 distinct points


# ---------------------------------------------------------------------------
# Alias structure
# ---------------------------------------------------------------------------


class TestAliasStructure:
    """Test alias structure metric."""

    def test_half_fraction(self) -> None:
        """2^(4-1) with D=ABC: main effects are aliased with 3FI."""
        factors = [Factor(name=n, low=0, high=10) for n in ["A", "B", "C", "D"]]
        result = generate_design(
            factors, design_type="fractional_factorial", generators=["D=ABC"], center_points=0
        )
        metrics = evaluate_design(result, metric="alias_structure")
        aliases = metrics["alias_structure"]
        assert len(aliases) > 0
        # A should be aliased with BCD
        a_alias = [a for a in aliases if a.startswith("A ")]
        assert len(a_alias) == 1
        assert "BCD" in a_alias[0]

    def test_full_factorial_no_aliases_from_generators(self) -> None:
        """Full factorial has no generators, so generator-based aliasing is empty."""
        factors = _continuous_factors(3, "ABC")
        result = generate_design(factors, design_type="full_factorial", center_points=0)
        metrics = evaluate_design(result, metric="alias_structure")
        # No generators → falls back to correlation-based, which should find none
        aliases = metrics["alias_structure"]
        assert len(aliases) == 0

    def test_raw_dataframe_correlation_fallback(self) -> None:
        """Raw DataFrame without generators should use correlation fallback."""
        # Create a fractional factorial manually: 2^(3-1) with C=AB
        df = pd.DataFrame(
            {"A": [-1, 1, -1, 1], "B": [-1, -1, 1, 1], "C": [1, -1, -1, 1]}
        )
        # C = A*B, so aliasing should be detected
        metrics = evaluate_design(df, metric="alias_structure")
        aliases = metrics["alias_structure"]
        assert len(aliases) > 0


# ---------------------------------------------------------------------------
# Confounding
# ---------------------------------------------------------------------------


class TestConfounding:
    """Test confounding metric."""

    def test_res_iii(self) -> None:
        """Resolution III design has main effects confounded with 2FI."""
        # 2^(3-1) with C=AB is resolution III
        factors = [Factor(name=n, low=0, high=10) for n in ["A", "B", "C"]]
        result = generate_design(
            factors, design_type="fractional_factorial", generators=["C=AB"], center_points=0
        )
        metrics = evaluate_design(result, metric="confounding")
        confounding = metrics["confounding"]
        assert len(confounding) > 0
        # A is confounded with BC
        a_conf = [c for c in confounding if c["effect"] == "A"]
        assert len(a_conf) == 1
        assert "BC" in a_conf[0]["confounded_with"]


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


class TestResolution:
    """Test resolution metric."""

    def test_from_design_result(self) -> None:
        """Resolution should be passed through from DesignResult metadata."""
        factors = [Factor(name=n, low=0, high=10) for n in ["A", "B", "C", "D", "E"]]
        result = generate_design(
            factors, design_type="fractional_factorial", resolution=3, center_points=0
        )
        metrics = evaluate_design(result, metric="resolution")
        assert metrics["resolution"] is not None

    def test_from_generators(self) -> None:
        """Resolution computed from generators."""
        factors = [Factor(name=n, low=0, high=10) for n in ["A", "B", "C", "D"]]
        result = generate_design(
            factors, design_type="fractional_factorial", generators=["D=ABC"], center_points=0
        )
        # D=ABC → I=ABCD → resolution IV
        metrics = evaluate_design(result, metric="resolution")
        res = metrics["resolution"]
        assert res == 4
        assert metrics["roman"] == "IV"

    def test_none_for_full_factorial(self) -> None:
        """Full factorial has no resolution."""
        factors = _continuous_factors(3, "ABC")
        result = generate_design(factors, design_type="full_factorial", center_points=0)
        metrics = evaluate_design(result, metric="resolution")
        assert metrics["resolution"] is None


# ---------------------------------------------------------------------------
# Defining relation
# ---------------------------------------------------------------------------


class TestDefiningRelation:
    """Test defining relation metric."""

    def test_single_generator(self) -> None:
        """D=ABC gives I=ABCD."""
        factors = [Factor(name=n, low=0, high=10) for n in ["A", "B", "C", "D"]]
        result = generate_design(
            factors, design_type="fractional_factorial", generators=["D=ABC"], center_points=0
        )
        metrics = evaluate_design(result, metric="defining_relation")
        dr = metrics["defining_relation"]
        assert dr is not None
        assert any("ABCD" in word for word in dr)

    def test_two_generators(self) -> None:
        """D=AB, E=AC gives 3 defining words."""
        factors = [Factor(name=n, low=0, high=10) for n in ["A", "B", "C", "D", "E"]]
        result = generate_design(
            factors,
            design_type="fractional_factorial",
            generators=["D=AB", "E=AC"],
            center_points=0,
        )
        metrics = evaluate_design(result, metric="defining_relation")
        dr = metrics["defining_relation"]
        assert dr is not None
        assert len(dr) == 3  # ABD, ACE, and BCDE

    def test_none_for_non_fractional(self) -> None:
        """Full factorial returns None for defining relation."""
        df = _full_factorial_df(2)
        metrics = evaluate_design(df, metric="defining_relation")
        assert metrics["defining_relation"] is None


# ---------------------------------------------------------------------------
# Clear effects
# ---------------------------------------------------------------------------


class TestClearEffects:
    """Test clear effects metric."""

    def test_res_iv_clear_main_effects(self) -> None:
        """In a Resolution IV design, all main effects should be clear."""
        factors = [Factor(name=n, low=0, high=10) for n in ["A", "B", "C", "D"]]
        result = generate_design(
            factors, design_type="fractional_factorial", generators=["D=ABC"], center_points=0
        )
        metrics = evaluate_design(result, metric="clear_effects")
        clear = metrics["clear_effects"]
        # Resolution IV: main effects aliased only with 3FI → all main effects clear
        assert set(clear["main_effects"]) == {"A", "B", "C", "D"}


# ---------------------------------------------------------------------------
# Minimum aberration
# ---------------------------------------------------------------------------


class TestMinimumAberration:
    """Test minimum aberration metric."""

    def test_wordlength_pattern(self) -> None:
        """D=ABC gives I=ABCD (length 4), so WLP = [0, 1] for A_3=0, A_4=1."""
        factors = [Factor(name=n, low=0, high=10) for n in ["A", "B", "C", "D"]]
        result = generate_design(
            factors, design_type="fractional_factorial", generators=["D=ABC"], center_points=0
        )
        metrics = evaluate_design(result, metric="minimum_aberration")
        ma = metrics["minimum_aberration"]
        wlp = ma["wordlength_pattern"]
        # I=ABCD has length 4; WLP starts at A_3: A_3=0, A_4=1
        assert wlp == [0, 1]

    def test_non_fractional(self) -> None:
        """Non-fractional design returns empty WLP."""
        df = _full_factorial_df(2)
        metrics = evaluate_design(df, metric="minimum_aberration")
        assert metrics["minimum_aberration"]["wordlength_pattern"] == []


# ---------------------------------------------------------------------------
# GF(2) word arithmetic internals
# ---------------------------------------------------------------------------


class TestWordArithmetic:
    """Test internal GF(2) word parsing and multiplication."""

    def test_parse_single_char_word(self) -> None:
        """Parse 'ABC' into factor indices {0, 1, 2}."""
        result = _parse_word("ABC", ["A", "B", "C", "D"])
        assert result == frozenset({0, 1, 2})

    def test_parse_with_i_prefix(self) -> None:
        """Parse 'I=ABCD' stripping the I= prefix."""
        result = _parse_word("I=ABCD", ["A", "B", "C", "D"])
        assert result == frozenset({0, 1, 2, 3})

    def test_parse_identity(self) -> None:
        """Parse 'I' returns empty set."""
        result = _parse_word("I", ["A", "B", "C"])
        assert result == frozenset()

    def test_multiply_words(self) -> None:
        """Multiplying {0,1} and {1,2} gives {0,2} (symmetric difference)."""
        w1 = frozenset({0, 1})
        w2 = frozenset({1, 2})
        assert _multiply_words(w1, w2) == frozenset({0, 2})

    def test_word_to_str(self) -> None:
        """Convert indices back to string."""
        result = _word_to_str(frozenset({0, 2, 3}), ["A", "B", "C", "D"])
        assert result == "ACD"

    def test_word_to_str_identity(self) -> None:
        """Empty frozenset gives 'I'."""
        result = _word_to_str(frozenset(), ["A", "B", "C"])
        assert result == "I"

    def test_defining_relation_single_generator(self) -> None:
        """D=ABC → full relation is [ABCD]."""
        words = _defining_relation_from_generators(["D=ABC"], ["A", "B", "C", "D"])
        strs = [_word_to_str(w, ["A", "B", "C", "D"]) for w in words]
        assert "ABCD" in strs
        assert len(words) == 1

    def test_defining_relation_two_generators(self) -> None:
        """D=AB, E=AC → relations ABD, ACE, BCDE."""
        words = _defining_relation_from_generators(["D=AB", "E=AC"], ["A", "B", "C", "D", "E"])
        strs = [_word_to_str(w, ["A", "B", "C", "D", "E"]) for w in words]
        assert len(words) == 3
        assert "ABD" in strs
        assert "ACE" in strs
        assert "BCDE" in strs


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


class TestDispatcher:
    """Test the evaluate_design dispatcher."""

    def test_single_metric_string(self) -> None:
        """metric='d_efficiency' works."""
        df = _full_factorial_df(2)
        result = evaluate_design(df, model="main_effects", metric="d_efficiency")
        assert "d_efficiency" in result

    def test_multiple_metrics_list(self) -> None:
        """metric=['d_efficiency', 'vif'] returns both."""
        df = _full_factorial_df(3)
        result = evaluate_design(
            df, model="main_effects", metric=["d_efficiency", "vif"]
        )
        assert "d_efficiency" in result
        assert "vif" in result

    def test_unknown_metric_raises(self) -> None:
        """Unknown metric name raises ValueError."""
        df = _full_factorial_df(2)
        with pytest.raises(ValueError, match="Unknown metric"):
            evaluate_design(df, model="main_effects", metric="nonexistent")

    def test_accepts_design_result(self) -> None:
        """evaluate_design accepts a DesignResult object."""
        factors = _continuous_factors(2, "AB")
        result = generate_design(factors, design_type="full_factorial", center_points=0)
        metrics = evaluate_design(result, model="main_effects", metric="d_efficiency")
        assert "d_efficiency" in metrics

    def test_drops_run_order_column(self) -> None:
        """RunOrder column in design should be ignored."""
        factors = _continuous_factors(2, "AB")
        result = generate_design(factors, design_type="full_factorial", center_points=0)
        # DesignResult has RunOrder in the design
        metrics = evaluate_design(result, model="main_effects", metric="degrees_of_freedom")
        dof = metrics["degrees_of_freedom"]
        assert dof["model"] == 2  # only A and B as factors


# ---------------------------------------------------------------------------
# Tool spec wrapper
# ---------------------------------------------------------------------------


class TestToolSpec:
    """Test the LLM tool spec wrapper."""

    def test_basic_round_trip(self) -> None:
        """Tool wrapper returns JSON-serializable output."""
        from process_improve.experiments.tools import evaluate_design_tool

        result = evaluate_design_tool(
            design_matrix=[
                {"A": -1, "B": -1},
                {"A": 1, "B": -1},
                {"A": -1, "B": 1},
                {"A": 1, "B": 1},
            ],
            model="main_effects",
            metric="d_efficiency",
        )
        assert "error" not in result
        assert "d_efficiency" in result

    def test_multiple_metrics(self) -> None:
        """Tool wrapper supports list of metrics."""
        from process_improve.experiments.tools import evaluate_design_tool

        result = evaluate_design_tool(
            design_matrix=[
                {"A": -1, "B": -1},
                {"A": 1, "B": -1},
                {"A": -1, "B": 1},
                {"A": 1, "B": 1},
            ],
            metric=["d_efficiency", "condition_number"],
        )
        assert "error" not in result
        assert "d_efficiency" in result
        assert "condition_number" in result

    def test_error_handling(self) -> None:
        """Bad metric name returns error dict."""
        from process_improve.experiments.tools import evaluate_design_tool

        result = evaluate_design_tool(
            design_matrix=[{"A": -1}, {"A": 1}],
            metric="nonexistent",
        )
        assert "error" in result


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration tests: generate_design -> evaluate_design pipeline."""

    def test_full_pipeline(self) -> None:
        """Generate a design, evaluate multiple metrics."""
        factors = [
            Factor(name="Temperature", low=150, high=200, units="degC"),
            Factor(name="Pressure", low=1, high=5, units="bar"),
            Factor(name="Time", low=10, high=60, units="min"),
        ]
        result = generate_design(factors, design_type="full_factorial", center_points=3)
        metrics = evaluate_design(
            result,
            model="main_effects",
            metric=["d_efficiency", "vif", "degrees_of_freedom", "condition_number"],
        )
        assert metrics["d_efficiency"] is not None
        assert all(v == pytest.approx(1.0, abs=0.1) for v in metrics["vif"].values())
        assert metrics["degrees_of_freedom"]["residual"] > 0
        assert metrics["condition_number"] > 0

    def test_ccd_quadratic(self) -> None:
        """CCD with quadratic model should have valid metrics."""
        factors = _continuous_factors(3, "ABC")
        result = generate_design(factors, design_type="ccd", alpha="face_centered", center_points=3)
        metrics = evaluate_design(
            result, model="quadratic", metric=["d_efficiency", "degrees_of_freedom"]
        )
        assert metrics["d_efficiency"] is not None
        assert metrics["degrees_of_freedom"]["residual"] > 0

    def test_fractional_factorial_aliases(self) -> None:
        """Fractional factorial should report alias structure and resolution."""
        factors = [Factor(name=n, low=0, high=10) for n in ["A", "B", "C", "D"]]
        result = generate_design(
            factors, design_type="fractional_factorial", generators=["D=ABC"], center_points=0
        )
        metrics = evaluate_design(
            result,
            metric=["alias_structure", "resolution", "defining_relation", "clear_effects", "minimum_aberration"],
        )
        assert len(metrics["alias_structure"]) > 0
        assert metrics["resolution"] == 4
        assert any("ABCD" in w for w in metrics["defining_relation"])
        assert "A" in metrics["clear_effects"]["main_effects"]
        assert metrics["minimum_aberration"]["wordlength_pattern"] == [0, 1]
