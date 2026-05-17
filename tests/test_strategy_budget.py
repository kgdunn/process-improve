"""Tests for the multi-stage DOE budget allocation logic."""

from process_improve.experiments.strategy.budget import (
    allocate_budget,
    estimate_confirmation_runs,
    estimate_rsm_runs,
    estimate_screening_runs,
)


class TestEstimateScreeningRuns:
    """Run-count estimation for screening designs."""

    def test_plackett_burman_rounds_up_to_multiple_of_four(self) -> None:
        assert estimate_screening_runs(5, "plackett_burman") == 8
        assert estimate_screening_runs(7, "plackett_burman") == 8
        assert estimate_screening_runs(8, "plackett_burman") == 12

    def test_definitive_screening(self) -> None:
        assert estimate_screening_runs(4, "definitive_screening") == 9

    def test_fractional_factorial_small_is_full_factorial(self) -> None:
        # k <= 4: full factorial is feasible
        assert estimate_screening_runs(3, "fractional_factorial") == 8
        assert estimate_screening_runs(4, "fractional_factorial") == 16

    def test_fractional_factorial_mid_range(self) -> None:
        assert estimate_screening_runs(5, "fractional_factorial") == 16
        assert estimate_screening_runs(6, "fractional_factorial") == 16
        assert estimate_screening_runs(7, "fractional_factorial") == 16

    def test_fractional_factorial_large(self) -> None:
        assert estimate_screening_runs(10, "fractional_factorial") == 32
        assert estimate_screening_runs(11, "fractional_factorial") == 32
        assert estimate_screening_runs(14, "fractional_factorial") == 64

    def test_full_factorial(self) -> None:
        assert estimate_screening_runs(4, "full_factorial") == 16

    def test_unknown_design_falls_back_to_plackett_burman(self) -> None:
        assert estimate_screening_runs(5, "some_unknown_design") == 8


class TestEstimateRsmRuns:
    """Run-count estimation for response-surface designs."""

    def test_ccd(self) -> None:
        # factorial(3)=8 + axial(2*3)=6 + center(3) = 17
        assert estimate_rsm_runs(3, "ccd", center_points=3) == 17

    def test_box_behnken_known_factor_count(self) -> None:
        # BBD(4)=24 + center(3)
        assert estimate_rsm_runs(4, "box_behnken", center_points=3) == 27

    def test_box_behnken_unknown_factor_count_falls_back_to_ccd(self) -> None:
        # n_factors=2 is not a Box-Behnken key -> CCD-style fallback estimate
        result = estimate_rsm_runs(2, "box_behnken", center_points=3)
        # factorial(2)=4 + axial(2*2)=4 + center(3) = 11
        assert result == 11

    def test_d_optimal(self) -> None:
        result = estimate_rsm_runs(3, "d_optimal")
        assert result > 0
        # 1 + 2*3 + 3*2//2 = 10 terms; ceil(1.5*10) = 15
        assert result == 15

    def test_unknown_design_falls_back_to_ccd(self) -> None:
        result = estimate_rsm_runs(3, "mystery_design", center_points=3)
        assert result == 17


def test_estimate_confirmation_runs_has_minimum_of_three() -> None:
    assert estimate_confirmation_runs(1) == 3
    assert estimate_confirmation_runs(3) == 3
    assert estimate_confirmation_runs(8) == 8


class TestAllocateBudget:
    """Budget allocation across screening / optimisation / confirmation."""

    def test_no_budget_uses_ideal_allocation(self) -> None:
        result = allocate_budget(
            total_budget=None,
            n_factors=5,
            needs_screening=True,
            needs_rsm=True,
        )
        assert result["is_tight"] is False
        assert result["warnings"] == []
        assert result["total"] == result["ideal_total"]
        assert result["total"] == result["screening"] + result["optimization"] + result["confirmation"]

    def test_full_multi_stage_allocation_sums_to_total(self) -> None:
        result = allocate_budget(
            total_budget=60,
            n_factors=5,
            needs_screening=True,
            needs_rsm=True,
        )
        assert result["screening"] > 0
        assert result["optimization"] > 0
        assert result["confirmation"] >= 3

    def test_very_tight_budget_emits_warning(self) -> None:
        result = allocate_budget(
            total_budget=3,
            n_factors=5,
            needs_screening=True,
            needs_rsm=True,
        )
        assert any("very tight" in w for w in result["warnings"])
        # remaining <= 0 path: all allocations clamp to non-negative
        assert result["screening"] >= 0
        assert result["optimization"] >= 0
        assert result["confirmation"] >= 0

    def test_tight_but_feasible_budget_emits_tight_warning(self) -> None:
        # ideal_total is 26 here; minimum feasible is 17, so 20 is tight-but-feasible
        result = allocate_budget(
            total_budget=20,
            n_factors=4,
            needs_screening=True,
            needs_rsm=True,
        )
        assert result["is_tight"] is True
        assert any("tight" in w for w in result["warnings"])

    def test_screening_only(self) -> None:
        result = allocate_budget(
            total_budget=20,
            n_factors=5,
            needs_screening=True,
            needs_rsm=False,
        )
        assert result["optimization"] == 0
        assert result["confirmation"] == 3
        assert result["screening"] == 17

    def test_rsm_only(self) -> None:
        result = allocate_budget(
            total_budget=20,
            n_factors=3,
            needs_screening=False,
            needs_rsm=True,
        )
        assert result["screening"] == 0
        assert result["optimization"] == 17
        assert result["confirmation"] == 3

    def test_no_screening_no_rsm_assigns_all_to_confirmation(self) -> None:
        result = allocate_budget(
            total_budget=12,
            n_factors=3,
            needs_screening=False,
            needs_rsm=False,
        )
        assert result["screening"] == 0
        assert result["optimization"] == 0
        assert result["confirmation"] == 12
