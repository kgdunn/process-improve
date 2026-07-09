"""Tests for the evaluate_design() design quality metrics API."""

from __future__ import annotations

from typing import ClassVar

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
        X, cols, _ = _build_model_matrix(df, "main_effects", ["A", "B", "C"])
        assert X.shape == (8, 4)  # intercept + 3 main effects
        assert "Intercept" in cols

    def test_interactions_shape(self) -> None:
        """2^3 factorial with interactions produces (8, 7+1) columns."""
        df = _full_factorial_df(3)
        X, _cols, _ = _build_model_matrix(df, "interactions", ["A", "B", "C"])
        # intercept + 3 main + 3 two-factor interactions = 7
        assert X.shape == (8, 7)

    def test_quadratic_shape(self) -> None:
        """Design with quadratic model includes squared terms."""
        # CCD has 3 levels per factor, so squared terms are estimable
        factors = _continuous_factors(2, "AB")
        result = generate_design(factors, design_type="ccd", alpha="face_centered")
        df = pd.DataFrame(result.design[["A", "B"]])
        X, _cols, _ = _build_model_matrix(df, "quadratic", ["A", "B"])
        # intercept + 2 main + 1 interaction + 2 squared = 6
        assert X.shape[1] == 6

    def test_none_defaults_to_interactions(self) -> None:
        """model=None should default to interactions."""
        df = _full_factorial_df(3)
        X_none, _cols_none, _ = _build_model_matrix(df, None, ["A", "B", "C"])
        X_int, _cols_int, _ = _build_model_matrix(df, "interactions", ["A", "B", "C"])
        np.testing.assert_array_equal(X_none, X_int)

    def test_explicit_formula_with_tilde(self) -> None:
        """Explicit formula with ~ should strip the LHS."""
        df = _full_factorial_df(2)
        X, _cols, _ = _build_model_matrix(df, "y ~ A + B", ["A", "B"])
        assert X.shape == (4, 3)  # intercept + 2 main

    def test_explicit_rhs_formula(self) -> None:
        """Explicit RHS formula (no ~) should work directly."""
        df = _full_factorial_df(2)
        X, _cols, _ = _build_model_matrix(df, "A * B", ["A", "B"])
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
        frac = generate_design(factors, design_type="fractional_factorial", generators=["D=ABC"], center_points=0)
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
        result = evaluate_design(df, model="main_effects", metric="power", effect_size=10.0, sigma=1.0)
        power = result["power"]
        for name, pwr in power.items():
            assert pwr > 0.99, f"Power for {name} should be near 1.0 for large effect"

    def test_small_effect_low(self) -> None:
        """Small effect size produces low power."""
        df = _full_factorial_df(2)
        result = evaluate_design(df, model="main_effects", metric="power", effect_size=0.01, sigma=1.0)
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
        # Pick any factor - power should be higher with 2x runs
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
        result = generate_design(factors, design_type="fractional_factorial", generators=["D=ABC"], center_points=0)
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
        df = pd.DataFrame({"A": [-1, 1, -1, 1], "B": [-1, -1, 1, 1], "C": [1, -1, -1, 1]})
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
        result = generate_design(factors, design_type="fractional_factorial", generators=["C=AB"], center_points=0)
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
        result = generate_design(factors, design_type="fractional_factorial", resolution=3, center_points=0)
        metrics = evaluate_design(result, metric="resolution")
        assert metrics["resolution"] is not None

    def test_from_generators(self) -> None:
        """Resolution computed from generators."""
        factors = [Factor(name=n, low=0, high=10) for n in ["A", "B", "C", "D"]]
        result = generate_design(factors, design_type="fractional_factorial", generators=["D=ABC"], center_points=0)
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
        result = generate_design(factors, design_type="fractional_factorial", generators=["D=ABC"], center_points=0)
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
        result = generate_design(factors, design_type="fractional_factorial", generators=["D=ABC"], center_points=0)
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
        result = generate_design(factors, design_type="fractional_factorial", generators=["D=ABC"], center_points=0)
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
        result = evaluate_design(df, model="main_effects", metric=["d_efficiency", "vif"])
        assert "d_efficiency" in result
        assert "vif" in result

    def test_unknown_metric_raises(self) -> None:
        """Unknown metric name raises ValueError."""
        df = _full_factorial_df(2)
        with pytest.raises(ValueError, match="Unknown metric"):
            evaluate_design(df, model="main_effects", metric="nonexistent")

    def test_optimality_suffix_aliases_resolve(self) -> None:
        """The opposite suffix is accepted and keyed under the canonical name."""
        df = _full_factorial_df(3)
        # d_optimality -> d_efficiency (an _efficiency metric requested by _optimality)
        via_alias = evaluate_design(df, model="main_effects", metric="d_optimality")
        via_canon = evaluate_design(df, model="main_effects", metric="d_efficiency")
        assert "d_efficiency" in via_alias
        assert "d_optimality" not in via_alias
        assert via_alias["d_efficiency"] == via_canon["d_efficiency"]
        # a_efficiency -> a_optimality. Requesting the _efficiency spelling is now
        # valid (previously an unknown-metric error); a_optimality's own output
        # already carries both a_optimality and a_efficiency keys.
        via_alias_a = evaluate_design(df, model="main_effects", metric="a_efficiency")
        assert "a_optimality" in via_alias_a

    def test_metric_alias_in_list(self) -> None:
        """Aliases resolve inside a metric list alongside canonical names."""
        df = _full_factorial_df(3)
        result = evaluate_design(df, model="main_effects", metric=["d_optimality", "vif"])
        assert "d_efficiency" in result
        assert "vif" in result

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
        from process_improve.tool_spec import execute_tool_call

        result = execute_tool_call(
            "evaluate_design",
            {
                "design_matrix": [
                    {"A": -1, "B": -1},
                    {"A": 1, "B": -1},
                    {"A": -1, "B": 1},
                    {"A": 1, "B": 1},
                ],
                "model": "main_effects",
                "metric": "d_efficiency",
            },
        )
        assert "error" not in result
        assert "d_efficiency" in result

    def test_multiple_metrics(self) -> None:
        """Tool wrapper supports list of metrics."""
        from process_improve.tool_spec import execute_tool_call

        result = execute_tool_call(
            "evaluate_design",
            {
                "design_matrix": [
                    {"A": -1, "B": -1},
                    {"A": 1, "B": -1},
                    {"A": -1, "B": 1},
                    {"A": 1, "B": 1},
                ],
                "metric": ["d_efficiency", "condition_number"],
            },
        )
        assert "error" not in result
        assert "d_efficiency" in result
        assert "condition_number" in result

    def test_error_handling(self) -> None:
        """Bad metric name returns error dict."""
        from process_improve.tool_spec import execute_tool_call

        result = execute_tool_call(
            "evaluate_design",
            {"design_matrix": [{"A": -1}, {"A": 1}], "metric": "nonexistent"},
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
        metrics = evaluate_design(result, model="quadratic", metric=["d_efficiency", "degrees_of_freedom"])
        assert metrics["d_efficiency"] is not None
        assert metrics["degrees_of_freedom"]["residual"] > 0

    def test_fractional_cube_ccd_quadratic_full_rank(self) -> None:
        """A fractional-cube CCD is full-rank for the full second-order model."""
        factors = _continuous_factors(5, "ABCDE")
        result = generate_design(factors, design_type="ccd", cube="fractional", alpha="face_centered", center_points=6)
        metrics = evaluate_design(result, model="quadratic", metric=["d_efficiency", "vif", "power"])
        assert metrics["d_efficiency"] is not None
        assert metrics["vif"] is not None

    def test_fractional_factorial_aliases(self) -> None:
        """Fractional factorial should report alias structure and resolution."""
        factors = [Factor(name=n, low=0, high=10) for n in ["A", "B", "C", "D"]]
        result = generate_design(factors, design_type="fractional_factorial", generators=["D=ABC"], center_points=0)
        metrics = evaluate_design(
            result,
            metric=["alias_structure", "resolution", "defining_relation", "clear_effects", "minimum_aberration"],
        )
        assert len(metrics["alias_structure"]) > 0
        assert metrics["resolution"] == 4
        assert any("ABCD" in w for w in metrics["defining_relation"])
        assert "A" in metrics["clear_effects"]["main_effects"]
        assert metrics["minimum_aberration"]["wordlength_pattern"] == [0, 1]


# ---------------------------------------------------------------------------
# New model-aware metrics: A/E-optimality, correlation, alias matrix, FDS
# ---------------------------------------------------------------------------

# Five-factor pure main-effects-plus-quadratics model (11 terms incl. intercept).
_FACTORS_5 = list("ABCDE")
_PURE_QUADRATIC_5 = "+".join(_FACTORS_5) + "+" + "+".join(f"I({n}**2)" for n in _FACTORS_5)


def _rsm_factors() -> list[Factor]:
    """Five continuous factors coded on [-1, 1]."""
    return [Factor(name=n, low=-1, high=1) for n in _FACTORS_5]


def _coded(result: object) -> pd.DataFrame:
    """Extract the coded factor columns of a DesignResult as a plain DataFrame."""
    df = pd.DataFrame(result.design)  # type: ignore[attr-defined]
    return df[[c for c in df.columns if c in _FACTORS_5]].reset_index(drop=True).astype(float)


# A 25-run OMARS design for five factors, reproducing the published reference
# row (A=2.34, E=0.93, max|r|=0.00, I=0.51, G=0.84).  It is the conference-matrix
# foldover [C; -C; C; -C; 0] (a doubled DSD core plus one centre run) and is
# verified by ``is_omars``.  The enumerated OMARS catalogue is not shipped with
# the package; constructive generation of OMARS designs for small factor counts
# is planned, at which point this hand-embedded fixture can be replaced.
_OMARS_25 = pd.DataFrame(
    [
        [0, 1, 1, 1, 1], [-1, 0, -1, 1, 1], [-1, 1, 1, -1, 0], [-1, 1, -1, 0, -1],
        [1, 1, -1, -1, 1], [1, 1, 0, 1, -1], [0, -1, -1, -1, -1], [1, 0, 1, -1, -1],
        [1, -1, -1, 1, 0], [1, -1, 1, 0, 1], [-1, -1, 1, 1, -1], [-1, -1, 0, -1, 1],
        [0, 1, 1, 1, 1], [-1, 0, -1, 1, 1], [-1, 1, 1, -1, 0], [-1, 1, -1, 0, -1],
        [1, 1, -1, -1, 1], [1, 1, 0, 1, -1], [0, -1, -1, -1, -1], [1, 0, 1, -1, -1],
        [1, -1, -1, 1, 0], [1, -1, 1, 0, 1], [-1, -1, 1, 1, -1], [-1, -1, 0, -1, 1],
        [0, 0, 0, 0, 0],
    ],
    columns=_FACTORS_5,
    dtype=float,
)


class TestAOptimality:
    """Test A-optimality (trace of the inverse information matrix)."""

    def test_orthogonal_design(self) -> None:
        """Orthogonal 2^3 design: A = trace((X'X)^-1) is positive and small."""
        df = _full_factorial_df(3)
        result = evaluate_design(df, model="main_effects", metric="a_optimality")
        # Orthogonal: X'X = 8 I, so trace((X'X)^-1) = 4 / 8 = 0.5.
        assert result["a_optimality"] == pytest.approx(0.5, abs=1e-9)
        assert result["a_efficiency"] is not None

    def test_singular_returns_none(self) -> None:
        """Rank-deficient design returns None for A-optimality."""
        df = _full_factorial_df(2)
        result = evaluate_design(df, model="quadratic", metric="a_optimality")
        assert result["a_optimality"] is None


class TestEOptimality:
    """Test E-optimality (smallest eigenvalue of X'X)."""

    def test_orthogonal_design(self) -> None:
        """Orthogonal 2^3 design: every eigenvalue of X'X equals N = 8."""
        df = _full_factorial_df(3)
        result = evaluate_design(df, model="main_effects", metric="e_optimality")
        assert result["e_optimality"] == pytest.approx(8.0, abs=1e-9)

    def test_reported_even_when_singular(self) -> None:
        """E-optimality is defined (near zero) even for a rank-deficient design."""
        df = _full_factorial_df(2)
        result = evaluate_design(df, model="quadratic", metric="e_optimality")
        assert result["e_optimality"] == pytest.approx(0.0, abs=1e-9)


class TestCorrelation:
    """Test the residualised second-order correlation summary."""

    def test_keys_present(self) -> None:
        """Correlation result exposes max/mean |r|, matrix, and terms."""
        df = _coded(generate_design(_rsm_factors(), design_type="box_behnken", center_points=6))
        result = evaluate_design(df, model=_PURE_QUADRATIC_5, metric="correlation")
        corr = result["correlation"]
        assert {"max_abs_r", "mean_abs_r", "matrix", "terms"} <= set(corr)
        assert 0.0 <= corr["max_abs_r"] <= 1.0

    def test_omars_second_order_orthogonal(self) -> None:
        """The OMARS fixture has an orthogonal second-order block (max|r| = 0)."""
        result = evaluate_design(_OMARS_25, model=_PURE_QUADRATIC_5, metric="correlation")
        assert result["correlation"]["max_abs_r"] == pytest.approx(0.0, abs=1e-9)


class TestAliasMatrix:
    """Test the general alias (bias) matrix."""

    def test_keys_present(self) -> None:
        """Alias-matrix result exposes the matrix, terms, and summary norms."""
        df = _coded(generate_design(_rsm_factors(), design_type="box_behnken", center_points=6))
        result = evaluate_design(df, model=_PURE_QUADRATIC_5, metric="alias_matrix")
        am = result["alias_matrix"]
        assert {"matrix", "model_terms", "alias_terms", "max_abs", "max_abs_main_effect_rows", "frobenius_norm"} <= set(
            am
        )
        # Box-Behnken aliases no two-factor interactions into the pure-quadratic model.
        assert am["max_abs"] == pytest.approx(0.0, abs=1e-9)

    def test_main_effect_rows_unbiased_for_omars(self) -> None:
        """OMARS keeps main effects clear of the omitted 2fi (main-row alias = 0)."""
        result = evaluate_design(_OMARS_25, model=_PURE_QUADRATIC_5, metric="alias_matrix")
        assert result["alias_matrix"]["max_abs_main_effect_rows"] == pytest.approx(0.0, abs=1e-9)


class TestFDS:
    """Test the fraction-of-design-space (FDS) curve."""

    def test_payload_and_scaling(self) -> None:
        """FDS reports region metadata, quantiles, I/G, and the xN SPV variants."""
        df = _coded(generate_design(_rsm_factors(), design_type="box_behnken", center_points=6))
        result = evaluate_design(df, model=_PURE_QUADRATIC_5, metric="fds", random_seed=1)
        fds = result["fds"]
        assert fds["region"] == "cuboidal"
        assert fds["include_vertices"] is True
        assert fds["random_seed"] == 1
        n = df.shape[0]
        assert fds["scaled_average_prediction_variance"] == pytest.approx(fds["average_prediction_variance"] * n)
        assert fds["scaled_max_prediction_variance"] == pytest.approx(fds["max_prediction_variance"] * n)
        # The maximum prediction variance is at least the region average.
        assert fds["max_prediction_variance"] >= fds["average_prediction_variance"]

    def test_quantiles_monotone(self) -> None:
        """FDS quantiles are non-decreasing in the fraction of design space."""
        df = _coded(generate_design(_rsm_factors(), design_type="box_behnken", center_points=6))
        fds = evaluate_design(df, model=_PURE_QUADRATIC_5, metric="fds", random_seed=1)["fds"]
        keys = sorted(fds["quantiles"], key=float)
        values = [fds["quantiles"][k] for k in keys]
        assert values == sorted(values)


class TestReducedFormulaRegression:
    """Regression: explicit reduced formulas must not crash the region metrics."""

    def test_i_g_efficiency_explicit_pure_quadratic(self) -> None:
        """i/g efficiency for an 11-term reduced formula no longer raises (size 11 vs 21)."""
        df = _coded(generate_design(_rsm_factors(), design_type="box_behnken", center_points=6))
        result = evaluate_design(
            df, model=_PURE_QUADRATIC_5, metric=["i_efficiency", "g_efficiency"], random_seed=1
        )
        assert result["i_efficiency"] is not None
        assert result["g_efficiency"] is not None
        assert result["average_prediction_variance"] > 0
        assert result["max_prediction_variance"] >= result["average_prediction_variance"]


class TestAggregate:
    """Test metric='all' and the evaluate_all wrapper."""

    def test_metric_all_includes_every_metric(self) -> None:
        """metric='all' returns keys for every registered metric family."""
        df = _coded(generate_design(_rsm_factors(), design_type="box_behnken", center_points=6))
        result = evaluate_design(df, model=_PURE_QUADRATIC_5, metric="all", effect_size=1.0, random_seed=1)
        for key in ("d_efficiency", "a_optimality", "e_optimality", "correlation", "alias_matrix", "fds", "power"):
            assert key in result

    def test_evaluate_all_matches_metric_all(self) -> None:
        """evaluate_all is a thin wrapper over evaluate_design(metric='all')."""
        from process_improve.experiments.evaluate import evaluate_all

        df = _coded(generate_design(_rsm_factors(), design_type="box_behnken", center_points=6))
        a = evaluate_all(df, model=_PURE_QUADRATIC_5, effect_size=1.0, random_seed=1)
        b = evaluate_design(df, model=_PURE_QUADRATIC_5, metric="all", effect_size=1.0, random_seed=1)
        assert set(a) == set(b)
        assert a["a_optimality"] == pytest.approx(b["a_optimality"])


class TestGoldenMetrics:
    """Golden-value regression against an independent numpy/scipy implementation.

    Five factors on the 11-term main-effects-plus-pure-quadratics model, all
    designs confined to [-1, 1]^5, region cuboidal with uniform sampling plus
    the 2^5 vertices (random_seed=1).  Stable/deterministic metrics (A, E,
    max|r|, region-average I, VIF, alias, power) are asserted tightly; the
    Monte-Carlo region maximum (G) is asserted with a modest tolerance.
    """

    # design name -> (A, E, max|r|, region-I, region-G, max VIF,
    #                 alias max|A|, main-row alias, power main, power quad)
    GOLDEN: ClassVar[dict[str, tuple[float, ...]]] = {
        "BBD": (1.05, 2.54, 0.15, 0.18, 0.84, 1.20, 0.00, 0.00, 0.97, 0.82),
        "CCD": (2.39, 2.00, 0.75, 0.31, 0.77, 3.20, 0.00, 0.00, 0.98, 0.32),
        "OMARS": (2.34, 0.93, 0.00, 0.51, 0.84, 1.00, 1.00, 0.00, 0.99, 0.46),
        "DSD": (3.70, 0.85, 0.13, 0.71, 1.05, 1.05, 1.09, 0.00, 0.42, 0.15),
    }

    def _design(self, name: str) -> pd.DataFrame:
        if name == "BBD":
            return _coded(generate_design(_rsm_factors(), design_type="box_behnken", center_points=6))
        if name == "CCD":
            return _coded(
                generate_design(
                    _rsm_factors(), design_type="ccd", cube="fractional", alpha="face_centered", center_points=6
                )
            )
        if name == "DSD":
            return _coded(generate_design(_rsm_factors(), design_type="dsd"))
        return _OMARS_25

    @pytest.mark.parametrize("name", ["BBD", "CCD", "OMARS", "DSD"])
    def test_golden_row(self, name: str) -> None:
        df = self._design(name)
        gA, gE, gR, gI, gG, gVIF, gAlias, gMain, gPM, gPQ = self.GOLDEN[name]
        res = evaluate_design(
            df,
            model=_PURE_QUADRATIC_5,
            metric=["a_optimality", "e_optimality", "correlation", "fds", "vif", "alias_matrix", "power"],
            effect_size=1.0,
            random_seed=1,
            n_samples=100_000,
        )
        assert res["a_optimality"] == pytest.approx(gA, abs=0.01)
        assert res["e_optimality"] == pytest.approx(gE, abs=0.01)
        assert res["correlation"]["max_abs_r"] == pytest.approx(gR, abs=0.01)
        assert res["fds"]["average_prediction_variance"] == pytest.approx(gI, abs=0.01)
        assert res["fds"]["max_prediction_variance"] == pytest.approx(gG, abs=0.05)
        assert max(res["vif"].values()) == pytest.approx(gVIF, abs=0.01)
        assert res["alias_matrix"]["max_abs"] == pytest.approx(gAlias, abs=0.01)
        assert res["alias_matrix"]["max_abs_main_effect_rows"] == pytest.approx(gMain, abs=0.01)
        power_main = np.mean([res["power"][n] for n in _FACTORS_5])
        power_quad = np.mean([res["power"][c] for c in res["power"] if c.startswith("I(")])
        assert power_main == pytest.approx(gPM, abs=0.01)
        assert power_quad == pytest.approx(gPQ, abs=0.01)

    def test_omars_fixture_is_valid(self) -> None:
        """The embedded 25-run OMARS fixture really is an OMARS design."""
        from process_improve.experiments.designs_omars import is_omars

        assert _OMARS_25.shape == (25, 5)
        assert is_omars(_OMARS_25.to_numpy())


# ---------------------------------------------------------------------------
# Edge-branch coverage for the new region / metric machinery
# ---------------------------------------------------------------------------


class TestRegionAndMetricEdgeCases:
    """Cover the error/edge branches of the new model-aware metrics."""

    def test_spherical_region(self) -> None:
        """region='spherical' samples the circumscribing ball and returns finite I/G."""
        df = _coded(generate_design(_rsm_factors(), design_type="box_behnken", center_points=6))
        result = evaluate_design(
            df, model=_PURE_QUADRATIC_5, metric="fds", region="spherical", n_samples=2000, random_seed=1
        )
        assert result["fds"]["region"] == "spherical"
        assert result["fds"]["max_prediction_variance"] >= result["fds"]["average_prediction_variance"] > 0

    def test_unknown_region_raises(self) -> None:
        """An unknown region name raises ValueError."""
        df = _coded(generate_design(_rsm_factors(), design_type="box_behnken", center_points=6))
        with pytest.raises(ValueError, match="Unknown region"):
            evaluate_design(df, model=_PURE_QUADRATIC_5, metric="fds", region="bogus")

    def test_include_vertices_false(self) -> None:
        """include_vertices=False omits the cube corners from the region sample."""
        df = _coded(generate_design(_rsm_factors(), design_type="box_behnken", center_points=6))
        fds = evaluate_design(
            df, model=_PURE_QUADRATIC_5, metric="fds", include_vertices=False, n_samples=2000, random_seed=1
        )["fds"]
        assert fds["include_vertices"] is False
        assert fds["max_prediction_variance"] > 0

    def test_i_efficiency_singular_returns_none(self) -> None:
        """i_efficiency is None for a rank-deficient design."""
        df = _full_factorial_df(2)
        result = evaluate_design(df, model="quadratic", metric="i_efficiency")
        assert result["i_efficiency"] is None

    def test_alias_and_fds_singular_return_none(self) -> None:
        """alias_matrix and fds are None for a rank-deficient design."""
        df = _full_factorial_df(2)
        result = evaluate_design(df, model="quadratic", metric=["alias_matrix", "fds"])
        assert result["alias_matrix"] is None
        assert result["fds"] is None

    def test_correlation_fewer_than_two_second_order(self) -> None:
        """A main-effects model has no second-order block: max|r| is 0 with a note."""
        df = _full_factorial_df(3)
        result = evaluate_design(df, model="main_effects", metric="correlation")
        corr = result["correlation"]
        assert corr["max_abs_r"] == 0.0
        assert "note" in corr

    def test_alias_matrix_all_interactions_present(self) -> None:
        """When every 2fi is already in the model there is nothing to alias against."""
        df = _full_factorial_df(3)
        # ``interactions`` includes all two-factor interactions, so the omitted set is empty.
        result = evaluate_design(df, model="interactions", metric=["alias_matrix", "correlation"])
        am = result["alias_matrix"]
        assert am["alias_terms"] == []
        assert am["max_abs"] == 0.0
        # ``correlation`` exercises the ":" interaction branch of the term classifier.
        assert result["correlation"]["max_abs_r"] >= 0.0

    def test_block_column_in_factor_names_is_dropped(self) -> None:
        """A raw DataFrame carrying a Block column drops it from the factor set."""
        df = _full_factorial_df(2)
        df = df.assign(Block=1)
        result = evaluate_design(df, model="main_effects", metric="degrees_of_freedom")
        assert result["degrees_of_freedom"]["model"] == 2  # only A and B count as factors


class TestFDSResolution:
    """Test the tunable-resolution FDS curve."""

    def test_default_is_coarse_quantiles_only(self) -> None:
        """With fds_resolution=None the output is the backward-compatible summary."""
        df = _coded(generate_design(_rsm_factors(), design_type="box_behnken", center_points=6))
        fds = evaluate_design(df, model=_PURE_QUADRATIC_5, metric="fds", random_seed=1)["fds"]
        assert "curve" not in fds
        assert len(fds["quantiles"]) == 11
        assert fds["fds_resolution"] is None

    def test_dense_curve(self) -> None:
        """fds_resolution=200 returns length-200 monotone arrays with min/max endpoints."""
        df = _coded(generate_design(_rsm_factors(), design_type="box_behnken", center_points=6))
        n = df.shape[0]
        fds = evaluate_design(
            df, model=_PURE_QUADRATIC_5, metric="fds", fds_resolution=200, random_seed=1
        )["fds"]
        curve = fds["curve"]
        pv = np.asarray(curve["prediction_variance"])
        frac = np.asarray(curve["fraction"])
        scaled = np.asarray(curve["scaled_prediction_variance"])
        assert pv.shape == frac.shape == scaled.shape == (200,)
        assert frac[0] == 0.0
        assert frac[-1] == pytest.approx(1.0)
        assert np.all(np.diff(pv) >= -1e-12)  # non-decreasing
        assert pv[-1] == pytest.approx(fds["max_prediction_variance"])
        assert pv[0] == pytest.approx(pv.min())
        assert np.allclose(scaled, pv * n)
        # The coarse quantile summary is still present for backward compatibility.
        assert len(fds["quantiles"]) == 11

    def test_resolution_below_two_raises(self) -> None:
        """fds_resolution must be at least 2."""
        df = _coded(generate_design(_rsm_factors(), design_type="box_behnken", center_points=6))
        with pytest.raises(ValueError, match="fds_resolution must be at least 2"):
            evaluate_design(df, model=_PURE_QUADRATIC_5, metric="fds", fds_resolution=1)

    def test_max_reproducible_and_published_values(self) -> None:
        """A fixed (n_samples, seed) is reproducible; 120k/seed-1 hits the published maxima."""
        bbd = _coded(generate_design(_rsm_factors(), design_type="box_behnken", center_points=6))
        first = evaluate_design(bbd, model=_PURE_QUADRATIC_5, metric="fds", n_samples=120_000, random_seed=1)
        second = evaluate_design(bbd, model=_PURE_QUADRATIC_5, metric="fds", n_samples=120_000, random_seed=1)
        assert first["fds"]["max_prediction_variance"] == second["fds"]["max_prediction_variance"]
        assert first["fds"]["max_prediction_variance"] == pytest.approx(0.84, abs=0.01)


# ---------------------------------------------------------------------------
# Mixed-level (categorical + continuous) factor support
# ---------------------------------------------------------------------------
import importlib.util  # noqa: E402

_HAS_PYOPTEX = importlib.util.find_spec("pyoptex") is not None
_needs_pyoptex = pytest.mark.skipif(not _HAS_PYOPTEX, reason="optimal designs require pyoptex")


def test_build_model_matrix_mixed_quadratic_is_partial_rsm() -> None:
    """A quadratic model with a categorical factor squares only the numeric factors.

    The categorical column stays a label column that patsy contrast-codes; it is
    never manually dummy-expanded (which would create singular cross terms) and
    never squared (a category has no square).
    """
    df = pd.DataFrame({
        "cat": ["A", "B", "C", "A", "B", "C"],
        "x1": [-1.0, 0.0, 1.0, 1.0, -1.0, 0.0],
        "x2": [1.0, -1.0, 0.0, 0.0, 1.0, -1.0],
    })
    _X, names, _info = _build_model_matrix(df, "quadratic", ["cat", "x1", "x2"])
    flat = [n.replace(" ", "") for n in names]
    # No square of the categorical factor.
    assert not any("I(cat**2)" in n for n in flat)
    # Categorical contrasts and continuous squares are present.
    assert any(n.startswith("cat[") for n in flat)
    assert "I(x1**2)" in flat
    assert "I(x2**2)" in flat


@_needs_pyoptex
def test_generate_and_evaluate_mixed_level_design() -> None:
    """End-to-end: generate an optimal mixed-level design and evaluate it.

    generate_design must return a usable DesignResult with the categorical
    factor as labels, and evaluate_design must return finite quality metrics
    (D/I/G efficiency, condition number, degrees of freedom, FDS) rather than
    None / inf as it did when the categorical was dummy-expanded.
    """
    np.random.seed(42)  # noqa: NPY002  # pyoptex coordinate exchange reads the global RNG
    factors = [
        Factor(name="catalyst", type="categorical", levels=["A", "B", "C", "D"]),
        Factor(name="temp", type="continuous", low=50, high=90),
        Factor(name="conc", type="continuous", low=0.1, high=0.5),
        Factor(name="rate", type="continuous", low=1, high=9),
    ]
    res = generate_design(factors, design_type="i_optimal", budget=44, model_type="quadratic")
    # The categorical is carried as labels in both the coded and actual designs.
    assert set(res.design["catalyst"].unique()) <= {"A", "B", "C", "D"}
    assert set(res.design_actual["catalyst"].unique()) <= {"A", "B", "C", "D"}

    design = res.design.drop(columns=["RunOrder"])
    metrics = evaluate_design(
        design,
        model="quadratic",
        metric=["d_efficiency", "i_efficiency", "g_efficiency", "condition_number",
                "degrees_of_freedom", "fds"],
    )
    assert metrics["d_efficiency"] is not None
    assert metrics["d_efficiency"] > 0
    assert metrics["i_efficiency"] is not None
    assert metrics["g_efficiency"] is not None
    assert np.isfinite(metrics["condition_number"])
    assert metrics["degrees_of_freedom"]["model"] > 0
    assert metrics["fds"] is not None
    assert metrics["fds"]["average_prediction_variance"] > 0
