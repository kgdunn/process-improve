"""Tests for the generate_design() unified design generation API."""

from __future__ import annotations

import numpy as np
import pytest

from process_improve.experiments.designs import _auto_select, generate_design
from process_improve.experiments.factor import Constraint, Factor, FactorType

_HAS_PYOPTEX = False
try:
    import pyoptex  # noqa: F401

    _HAS_PYOPTEX = True
except ImportError:
    pass

_skip_no_pyoptex = pytest.mark.skipif(not _HAS_PYOPTEX, reason="pyoptex not installed")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _continuous_factors(n: int, names: str | None = None) -> list[Factor]:
    """Create n continuous factors with default ranges."""
    if names and len(names) >= n:
        return [Factor(name=names[i], low=0, high=10) for i in range(n)]
    return [Factor(name=f"X{i + 1}", low=0, high=10) for i in range(n)]


# ---------------------------------------------------------------------------
# Factor / Constraint validation
# ---------------------------------------------------------------------------


class TestFactor:
    """Test the Factor Pydantic model validation."""

    def test_continuous_factor(self) -> None:
        """Continuous factor should store low, high, center, and range."""
        f = Factor(name="T", low=150, high=200, units="degC")
        assert f.type == FactorType.continuous
        assert f.center == 175.0
        assert f.range == (150, 200)

    def test_continuous_requires_low_high(self) -> None:
        """Continuous factor without low/high should raise ValueError."""
        with pytest.raises(ValueError, match="require 'low' and 'high'"):
            Factor(name="T")

    def test_continuous_low_ge_high(self) -> None:
        """Continuous factor with low >= high should raise ValueError."""
        with pytest.raises(ValueError, match="must be less than"):
            Factor(name="T", low=200, high=100)

    def test_categorical_factor(self) -> None:
        """Categorical factor should store levels."""
        f = Factor(name="Cat", type="categorical", levels=["A", "B", "C"])
        assert f.type == FactorType.categorical
        assert f.levels == ["A", "B", "C"]

    def test_categorical_requires_levels(self) -> None:
        """Categorical factor without levels should raise ValueError."""
        with pytest.raises(ValueError, match="require 'levels'"):
            Factor(name="Cat", type="categorical")

    def test_mixture_factor_defaults(self) -> None:
        """Mixture factor should default to low=0, high=1."""
        f = Factor(name="x1", type="mixture")
        assert f.low == 0.0
        assert f.high == 1.0

    def test_constraint(self) -> None:
        """Constraint should default to type='linear'."""
        c = Constraint(expression="A + B <= 1.0")
        assert c.type == "linear"


# ---------------------------------------------------------------------------
# DesignResult
# ---------------------------------------------------------------------------


class TestDesignResult:
    """Test the DesignResult container."""

    def test_repr(self) -> None:
        """DesignResult repr should include type, runs, and factors."""
        factors = _continuous_factors(2, "AB")
        result = generate_design(factors, design_type="full_factorial", center_points=0)
        r = repr(result)
        assert "full_factorial" in r
        assert "runs=4" in r
        assert "factors=2" in r


# ---------------------------------------------------------------------------
# Full Factorial
# ---------------------------------------------------------------------------


class TestFullFactorial:
    """Test full factorial design generation."""

    def test_2_factors(self) -> None:
        """2-factor full factorial should produce 4 runs."""
        factors = _continuous_factors(2, "AB")
        result = generate_design(factors, design_type="full_factorial", center_points=0)
        assert result.n_runs == 4
        assert result.n_factors == 2
        assert result.design_type == "full_factorial"
        assert result.factor_names == ["A", "B"]

    def test_3_factors(self) -> None:
        """3-factor full factorial should produce 8 runs."""
        factors = _continuous_factors(3, "ABC")
        result = generate_design(factors, design_type="full_factorial", center_points=0)
        assert result.n_runs == 8

    def test_with_center_points(self) -> None:
        """Full factorial with center points should add extra runs."""
        factors = _continuous_factors(2, "AB")
        result = generate_design(factors, design_type="full_factorial", center_points=3)
        assert result.n_runs == 7  # 4 factorial + 3 center

    def test_coded_values(self) -> None:
        """Coded design values should be -1 or +1."""
        factors = _continuous_factors(2, "AB")
        result = generate_design(factors, design_type="full_factorial", center_points=0)
        for col in result.factor_names:
            vals = result.design[col].unique()
            assert set(vals) <= {-1.0, 1.0}

    def test_actual_values(self) -> None:
        """Actual design values should match factor low/high."""
        factors = [Factor(name="T", low=100, high=200), Factor(name="P", low=1, high=5)]
        result = generate_design(factors, design_type="full_factorial", center_points=0)
        t_vals = set(result.design_actual["T"].unique())
        p_vals = set(result.design_actual["P"].unique())
        assert t_vals == {100.0, 200.0}
        assert p_vals == {1.0, 5.0}

    def test_reproducible_randomization(self) -> None:
        """Same seed should produce the same run order."""
        factors = _continuous_factors(3, "ABC")
        r1 = generate_design(factors, design_type="full_factorial", random_seed=123, center_points=0)
        r2 = generate_design(factors, design_type="full_factorial", random_seed=123, center_points=0)
        assert r1.run_order == r2.run_order
        # run_order should be a permutation
        assert sorted(r1.run_order) == list(range(1, r1.n_runs + 1))

    def test_different_seeds_different_order(self) -> None:
        """Different seeds should produce different run orders."""
        factors = _continuous_factors(4)  # 16 runs to reduce chance of collision
        r1 = generate_design(factors, design_type="full_factorial", random_seed=1, center_points=0)
        r2 = generate_design(factors, design_type="full_factorial", random_seed=999, center_points=0)
        # run_order should be a permutation of original rows (1-based)
        assert sorted(r1.run_order) == list(range(1, r1.n_runs + 1))
        assert sorted(r2.run_order) == list(range(1, r2.n_runs + 1))
        # With 16 runs, extremely unlikely to be the same with different seeds
        assert r1.run_order != r2.run_order

    def test_replication(self) -> None:
        """Replication should double the number of runs."""
        factors = _continuous_factors(2, "AB")
        result = generate_design(
            factors, design_type="full_factorial", center_points=0, replicates=2
        )
        assert result.n_runs == 8  # 4 * 2

    def test_blocking(self) -> None:
        """Blocking should add a Block column."""
        factors = _continuous_factors(2, "AB")
        result = generate_design(
            factors, design_type="full_factorial", center_points=0, blocks=2
        )
        assert result.blocks is not None
        assert set(result.blocks) == {1, 2}
        assert "Block" in result.design.columns


# ---------------------------------------------------------------------------
# Fractional Factorial
# ---------------------------------------------------------------------------


class TestFractionalFactorial:
    """Test fractional factorial design generation."""

    def test_5_factors_res_3(self) -> None:
        """5-factor resolution III fractional factorial should have 8 runs."""
        factors = _continuous_factors(5, "ABCDE")
        result = generate_design(
            factors, design_type="fractional_factorial", resolution=3, center_points=0
        )
        assert result.n_runs == 8  # 2^(5-2)
        assert result.n_factors == 5

    def test_coded_values(self) -> None:
        """Fractional factorial coded values should be -1 or +1."""
        factors = _continuous_factors(5, "ABCDE")
        result = generate_design(
            factors, design_type="fractional_factorial", resolution=3, center_points=0
        )
        for col in result.factor_names:
            vals = result.design[col].unique()
            assert set(vals) <= {-1.0, 1.0}

    def test_with_generators(self) -> None:
        """Explicit generators should produce the specified confounding pattern."""
        factors = [Factor(name=n, low=0, high=10) for n in ["A", "B", "C", "D"]]
        result = generate_design(
            factors,
            design_type="fractional_factorial",
            generators=["D=ABC"],
            center_points=0,
        )
        assert result.n_runs == 8  # 2^3 = 8 (since D=ABC is a 2^(4-1))
        assert result.generators == ["D=ABC"]


# ---------------------------------------------------------------------------
# Plackett-Burman
# ---------------------------------------------------------------------------


class TestPlackettBurman:
    """Test Plackett-Burman screening design."""

    def test_7_factors(self) -> None:
        """7-factor PB should produce 8 runs."""
        factors = _continuous_factors(7)
        result = generate_design(factors, design_type="plackett_burman", center_points=0)
        assert result.n_runs == 8
        assert result.n_factors == 7

    def test_11_factors(self) -> None:
        """11-factor PB should produce 12 runs."""
        factors = _continuous_factors(11)
        result = generate_design(factors, design_type="plackett_burman", center_points=0)
        assert result.n_runs == 12

    def test_coded_values(self) -> None:
        """PB coded values should be -1 or +1."""
        factors = _continuous_factors(7)
        result = generate_design(factors, design_type="plackett_burman", center_points=0)
        for col in result.factor_names:
            vals = result.design[col].unique()
            assert set(vals) <= {-1.0, 1.0}


# ---------------------------------------------------------------------------
# Box-Behnken
# ---------------------------------------------------------------------------


class TestBoxBehnken:
    """Test Box-Behnken response surface design."""

    def test_3_factors(self) -> None:
        """3-factor BB should produce 15 runs (with 3 center points)."""
        factors = _continuous_factors(3, "ABC")
        result = generate_design(factors, design_type="box_behnken", center_points=3)
        assert result.n_runs == 15
        assert result.n_factors == 3

    def test_4_factors(self) -> None:
        """4-factor BB should produce 27 runs (with 3 center points)."""
        factors = _continuous_factors(4)
        result = generate_design(factors, design_type="box_behnken", center_points=3)
        assert result.n_factors == 4
        assert result.n_runs == 27

    def test_values_in_range(self) -> None:
        """BB coded values should be within [-1, +1]."""
        factors = _continuous_factors(3, "ABC")
        result = generate_design(factors, design_type="box_behnken", center_points=3)
        for col in result.factor_names:
            vals = result.design[col].values
            assert np.all(np.abs(vals) <= 1.0 + 1e-10)

    def test_requires_3_factors(self) -> None:
        """BB should require at least 3 factors."""
        factors = _continuous_factors(2, "AB")
        with pytest.raises(ValueError, match="at least 3"):
            generate_design(factors, design_type="box_behnken")


# ---------------------------------------------------------------------------
# CCD
# ---------------------------------------------------------------------------


class TestCCD:
    """Test Central Composite Design."""

    def test_2_factors_rotatable(self) -> None:
        """Rotatable CCD should have alpha > 1."""
        factors = _continuous_factors(2, "AB")
        result = generate_design(factors, design_type="ccd", alpha="rotatable")
        assert result.n_factors == 2
        assert result.alpha is not None
        assert result.alpha > 1.0

    def test_3_factors(self) -> None:
        """3-factor CCD should produce more than 14 runs."""
        factors = _continuous_factors(3, "ABC")
        result = generate_design(factors, design_type="ccd", center_points=3)
        assert result.n_factors == 3
        assert result.n_runs > 14

    def test_face_centered(self) -> None:
        """Face-centered CCD should have alpha == 1."""
        factors = _continuous_factors(2, "AB")
        result = generate_design(factors, design_type="ccd", alpha="face_centered")
        assert result.alpha == pytest.approx(1.0)

    def test_axial_points_present(self) -> None:
        """Rotatable CCD should have values outside [-1, +1]."""
        factors = _continuous_factors(2, "AB")
        result = generate_design(factors, design_type="ccd", alpha="rotatable")
        max_abs = max(result.design[col].abs().max() for col in result.factor_names)
        assert max_abs > 1.0


# ---------------------------------------------------------------------------
# DSD (Definitive Screening Design)
# ---------------------------------------------------------------------------


class TestDSD:
    """Test Definitive Screening Design."""

    def test_3_factors(self) -> None:
        """DSD for 3 factors (odd) should produce 7 runs."""
        factors = _continuous_factors(3, "ABC")
        result = generate_design(factors, design_type="dsd", center_points=0)
        assert result.n_runs == 7

    def test_4_factors(self) -> None:
        """DSD for 4 factors (even) should produce 11 runs."""
        factors = _continuous_factors(4)
        result = generate_design(factors, design_type="dsd", center_points=0)
        assert result.n_runs == 11

    def test_requires_3_factors(self) -> None:
        """DSD should require at least 3 factors."""
        factors = _continuous_factors(2, "AB")
        with pytest.raises(ValueError, match="at least 3"):
            generate_design(factors, design_type="dsd")

    def test_values_in_range(self) -> None:
        """DSD coded values should be within [-1, +1]."""
        factors = _continuous_factors(5)
        result = generate_design(factors, design_type="dsd", center_points=0)
        for col in result.factor_names:
            vals = result.design[col].values
            assert np.all(np.abs(vals) <= 1.0 + 1e-10)


# ---------------------------------------------------------------------------
# D-Optimal
# ---------------------------------------------------------------------------


class TestDOptimal:
    """Test D-optimal design generation."""

    def test_basic(self) -> None:
        """D-optimal should select the requested number of points."""
        factors = _continuous_factors(2, "AB")
        result = generate_design(
            factors, design_type="d_optimal", budget=5, center_points=0
        )
        assert result.n_runs == 5
        assert result.n_factors == 2

    def test_default_budget(self) -> None:
        """D-optimal default budget should be 2*k + 1."""
        factors = _continuous_factors(3, "ABC")
        result = generate_design(factors, design_type="d_optimal", center_points=0)
        assert result.n_runs == 7

    @_skip_no_pyoptex
    def test_pyoptex_backend(self) -> None:
        """D-optimal via pyoptex should report the backend in metadata."""
        factors = _continuous_factors(2, "AB")
        result = generate_design(
            factors, design_type="d_optimal", budget=8, center_points=0
        )
        assert result.metadata.get("backend") == "pyoptex"
        assert result.metadata.get("metric_value") is not None

    def test_coded_values_in_range(self) -> None:
        """D-optimal coded values should be within [-1, +1]."""
        factors = _continuous_factors(3, "ABC")
        result = generate_design(factors, design_type="d_optimal", budget=10, center_points=0)
        for col in result.factor_names:
            vals = result.design[col].values
            assert np.all(np.abs(vals) <= 1.0 + 1e-10)


# ---------------------------------------------------------------------------
# I-Optimal (pyoptex)
# ---------------------------------------------------------------------------


@_skip_no_pyoptex
class TestIOptimal:
    """Test I-optimal design generation (requires pyoptex)."""

    def test_basic(self) -> None:
        """I-optimal should produce the requested number of runs."""
        factors = _continuous_factors(2, "AB")
        result = generate_design(
            factors, design_type="i_optimal", budget=8, center_points=0
        )
        assert result.n_runs == 8
        assert result.n_factors == 2
        assert result.metadata.get("backend") == "pyoptex"
        assert result.metadata.get("optimality_criterion") == "i_optimal"

    def test_3_factors(self) -> None:
        """I-optimal with 3 factors should work."""
        factors = _continuous_factors(3, "ABC")
        result = generate_design(
            factors, design_type="i_optimal", budget=10, center_points=0
        )
        assert result.n_runs == 10
        assert result.n_factors == 3


# ---------------------------------------------------------------------------
# A-Optimal (pyoptex)
# ---------------------------------------------------------------------------


@_skip_no_pyoptex
class TestAOptimal:
    """Test A-optimal design generation (requires pyoptex)."""

    def test_basic(self) -> None:
        """A-optimal should produce the requested number of runs."""
        factors = _continuous_factors(2, "AB")
        result = generate_design(
            factors, design_type="a_optimal", budget=8, center_points=0
        )
        assert result.n_runs == 8
        assert result.n_factors == 2
        assert result.metadata.get("backend") == "pyoptex"
        assert result.metadata.get("optimality_criterion") == "a_optimal"

    def test_3_factors(self) -> None:
        """A-optimal with 3 factors should work."""
        factors = _continuous_factors(3, "ABC")
        result = generate_design(
            factors, design_type="a_optimal", budget=10, center_points=0
        )
        assert result.n_runs == 10


# ---------------------------------------------------------------------------
# Split-plot (hard-to-change factors via pyoptex)
# ---------------------------------------------------------------------------


@_skip_no_pyoptex
class TestSplitPlot:
    """Test split-plot design generation with hard-to-change factors."""

    def test_hard_to_change_preserves_grouping(self) -> None:
        """Hard-to-change factor should be constant within whole plots."""
        factors = _continuous_factors(3, "ABC")
        result = generate_design(
            factors,
            design_type="d_optimal",
            budget=12,
            hard_to_change=["A"],
            center_points=0,
        )
        assert result.n_runs == 12
        assert result.metadata.get("hard_to_change") == ["A"]

        # Check that A is grouped: within each 3-run block, A should be constant
        a_vals = result.design["A"].values
        n_whole_plots = max(4, 12 // 3)
        runs_per_plot = 12 // n_whole_plots
        for plot_idx in range(n_whole_plots):
            start = plot_idx * runs_per_plot
            end = start + runs_per_plot
            block = a_vals[start:end]
            assert len(set(block)) == 1, f"A not constant in whole plot {plot_idx}: {block}"

    def test_split_plot_metadata(self) -> None:
        """Split-plot design should report hard_to_change in metadata."""
        factors = _continuous_factors(2, "AB")
        result = generate_design(
            factors,
            design_type="d_optimal",
            budget=8,
            hard_to_change=["A"],
            center_points=0,
        )
        assert result.metadata.get("hard_to_change") == ["A"]
        assert result.metadata.get("backend") == "pyoptex"


# ---------------------------------------------------------------------------
# Mixture
# ---------------------------------------------------------------------------


class TestMixture:
    """Test mixture design generation."""

    def test_3_components(self) -> None:
        """3-component simplex-centroid should produce 7 points."""
        factors = [Factor(name=f"x{i}", type="mixture") for i in range(3)]
        result = generate_design(factors, design_type="mixture")
        assert result.n_factors == 3
        assert result.n_runs == 7

    def test_rows_sum_to_one(self) -> None:
        """All mixture design rows should sum to 1."""
        factors = [Factor(name=f"x{i}", type="mixture") for i in range(3)]
        result = generate_design(factors, design_type="mixture")
        for i in range(result.n_runs):
            row_sum = sum(result.design_actual[f.name].iloc[i] for f in factors)
            assert row_sum == pytest.approx(1.0, abs=1e-10)

    def test_2_components(self) -> None:
        """2-component simplex-centroid should produce 3 points."""
        factors = [Factor(name=f"x{i}", type="mixture") for i in range(2)]
        result = generate_design(factors, design_type="mixture")
        assert result.n_factors == 2
        assert result.n_runs == 3

    def test_requires_2_components(self) -> None:
        """Mixture design should require at least 2 components."""
        factors = [Factor(name="x1", type="mixture")]
        with pytest.raises(ValueError, match="at least 2"):
            generate_design(factors, design_type="mixture")


# ---------------------------------------------------------------------------
# Auto-selection
# ---------------------------------------------------------------------------


class TestAutoSelect:
    """Test automatic design type selection."""

    def test_small_full_factorial(self) -> None:
        """Auto-select should pick full_factorial for <= 5 factors with no budget."""
        factors = _continuous_factors(3, "ABC")
        result = _auto_select(factors, budget=None, constraints=None, hard_to_change=None)
        assert result == "full_factorial"

    def test_mixture_factors(self) -> None:
        """Auto-select should pick mixture for all-mixture factors."""
        factors = [Factor(name=f"x{i}", type="mixture") for i in range(3)]
        result = _auto_select(factors, budget=None, constraints=None, hard_to_change=None)
        assert result == "mixture"

    def test_constraints_trigger_d_optimal(self) -> None:
        """Auto-select should pick d_optimal when constraints present."""
        factors = _continuous_factors(3, "ABC")
        constraints = [Constraint(expression="A + B <= 1")]
        result = _auto_select(factors, budget=None, constraints=constraints, hard_to_change=None)
        assert result == "d_optimal"

    def test_hard_to_change_triggers_d_optimal(self) -> None:
        """Auto-select should pick d_optimal when hard-to-change factors present."""
        factors = _continuous_factors(3, "ABC")
        result = _auto_select(factors, budget=None, constraints=None, hard_to_change=["A"])
        assert result == "d_optimal"

    def test_many_factors_small_budget(self) -> None:
        """Auto-select should pick plackett_burman for many factors with small budget."""
        factors = _continuous_factors(10)
        result = _auto_select(factors, budget=21, constraints=None, hard_to_change=None)
        assert result == "plackett_burman"

    def test_budget_allows_half_fraction(self) -> None:
        """Auto-select should pick fractional_factorial when budget allows half-fraction."""
        factors = _continuous_factors(5, "ABCDE")
        result = _auto_select(factors, budget=16, constraints=None, hard_to_change=None)
        assert result == "fractional_factorial"

    def test_auto_select_through_generate_design(self) -> None:
        """generate_design with no design_type should auto-select."""
        factors = _continuous_factors(2, "AB")
        result = generate_design(factors)
        assert result.design_type == "full_factorial"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Test error cases."""

    def test_empty_factors(self) -> None:
        """Empty factor list should raise ValueError."""
        with pytest.raises(ValueError, match=r"[Aa]t least one factor"):
            generate_design([])

    def test_unknown_design_type(self) -> None:
        """Unknown design_type should raise ValueError."""
        factors = _continuous_factors(2, "AB")
        with pytest.raises(ValueError, match="Unknown design_type"):
            generate_design(factors, design_type="nonexistent")

    @_skip_no_pyoptex
    def test_i_optimal_works_with_pyoptex(self) -> None:
        """I-optimal should work when pyoptex is available."""
        factors = _continuous_factors(2, "AB")
        result = generate_design(factors, design_type="i_optimal", budget=6, center_points=0)
        assert result.n_runs == 6


# ---------------------------------------------------------------------------
# Integration: generate design -> fit model
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration tests: generate a design and verify it works with lm()."""

    def test_full_factorial_to_lm(self) -> None:
        """Generate a full factorial, add response, fit model, check coefficients."""
        from process_improve.experiments.models import lm  # noqa: PLC0415
        from process_improve.experiments.structures import c, gather  # noqa: PLC0415

        factors = [Factor(name="A", low=0, high=10), Factor(name="B", low=0, high=10)]
        result = generate_design(factors, design_type="full_factorial", center_points=0)

        # Add a fake response: y = 10 + 5*A + 3*B
        design = result.design.copy()
        design["y"] = 10 + 5 * design["A"] + 3 * design["B"]

        # Fit a model
        a_col = c(*design["A"].tolist(), name="A")
        b_col = c(*design["B"].tolist(), name="B")
        y_col = c(*design["y"].tolist(), name="y")
        expt = gather(A=a_col, B=b_col, y=y_col)
        model = lm("y ~ A + B", expt)

        params = model.get_parameters(drop_intercept=False)
        assert params["A"] == pytest.approx(5.0, abs=1e-6)
        assert params["B"] == pytest.approx(3.0, abs=1e-6)
        assert params["Intercept"] == pytest.approx(10.0, abs=1e-6)

    def test_ccd_has_axial_and_center(self) -> None:
        """CCD should contain cube, axial, and center points."""
        factors = _continuous_factors(2, "AB")
        result = generate_design(factors, design_type="ccd", alpha="rotatable", center_points=4)

        design = result.design
        # Should have points at 0,0 (center)
        center_mask = (design["A"] == 0) & (design["B"] == 0)
        assert center_mask.sum() >= 2

        # Should have axial points (values beyond +/-1)
        max_val = max(design["A"].abs().max(), design["B"].abs().max())
        assert max_val > 1.0


# ---------------------------------------------------------------------------
# Tool spec wrapper
# ---------------------------------------------------------------------------


class TestToolSpec:
    """Test the LLM tool spec wrapper for generate_design."""

    def test_generate_design_tool(self) -> None:
        """Tool wrapper should return design as list of dicts."""
        from process_improve.experiments.tools import generate_design_tool  # noqa: PLC0415

        result = generate_design_tool(
            factors=[
                {"name": "T", "low": 150, "high": 200, "units": "degC"},
                {"name": "P", "low": 1, "high": 5, "units": "bar"},
            ],
            design_type="full_factorial",
            center_points=0,
        )
        assert "error" not in result
        assert result["n_runs"] == 4
        assert result["design_type"] == "full_factorial"
        assert len(result["design_coded"]) == 4
        assert len(result["design_actual"]) == 4

    def test_generate_design_tool_auto(self) -> None:
        """Tool wrapper with no design_type should auto-select."""
        from process_improve.experiments.tools import generate_design_tool  # noqa: PLC0415

        result = generate_design_tool(
            factors=[
                {"name": "A", "low": 0, "high": 10},
                {"name": "B", "low": 0, "high": 10},
                {"name": "C", "low": 0, "high": 10},
            ],
        )
        assert "error" not in result
        assert result["design_type"] == "full_factorial"

    def test_generate_design_tool_error(self) -> None:
        """Tool wrapper should return error dict on invalid input."""
        from process_improve.experiments.tools import generate_design_tool  # noqa: PLC0415

        result = generate_design_tool(factors=[{"name": "T"}])  # missing low/high
        assert "error" in result
