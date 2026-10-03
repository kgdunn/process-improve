"""Tests for supersaturated designs (Lin's half-fraction of a Hadamard matrix)."""

from __future__ import annotations

import numpy as np
import pytest

from process_improve.experiments import Factor, generate_design
from process_improve.experiments.designs_supersaturated import (
    dispatch_supersaturated,
    e_s2,
    e_s2_lower_bound,
    hadamard,
    n_fully_aliased,
)


def _factors(k: int) -> list[Factor]:
    return [Factor(name=f"X{i + 1}", low=0, high=10) for i in range(k)]


@pytest.mark.parametrize("order", [*range(4, 92, 4), *range(96, 116, 4)])
def test_hadamard_orders(order: int) -> None:
    """Every order up to 112 except 92 is built (28 needs GF(27), 36 Paley II); the first column is all +1."""
    h = hadamard(order)
    assert h is not None
    np.testing.assert_array_equal(h.T @ h, order * np.eye(order))
    assert (h[:, 0] == 1).all()


def test_hadamard_order_92_is_not_built() -> None:
    """Order 92 needs Williamson's construction, which is not implemented."""
    assert hadamard(92) is None


@pytest.mark.parametrize(("k", "budget"), [(10, 6), (16, 10), (22, 12), (30, 16), (40, 22)])
def test_designs_are_balanced_supersaturated_and_unaliased(k: int, budget: int) -> None:
    design, meta = dispatch_supersaturated(_factors(k), budget)
    assert design.shape == (budget, k)
    assert budget < k + 1  # more factors than the runs can estimate
    assert set(np.unique(design)) == {-1.0, 1.0}
    np.testing.assert_array_equal(design.sum(axis=0), 0)  # every column balanced
    assert meta["n_fully_aliased_pairs"] == n_fully_aliased(design) == 0
    assert meta["e_s2"] == pytest.approx(e_s2(design))
    assert meta["e_s2"] >= e_s2_lower_bound(budget, k) - 1e-9


def test_lin_designs_reach_the_e_s2_bound_where_known() -> None:
    """Lin's 6-run design for 10 factors and 12-run design for 22 factors are E(s^2)-optimal."""
    for k, budget in [(10, 6), (22, 12)]:
        _design, meta = dispatch_supersaturated(_factors(k), budget)
        assert meta["e_s2_efficiency"] == pytest.approx(1.0)


def test_paley_matrix_avoids_identical_columns_at_32_runs() -> None:
    """The Kronecker-built order-32 Hadamard leaves identical columns in its half-fraction; Paley's does not."""
    _design, meta = dispatch_supersaturated(_factors(30), 16)
    assert meta["n_fully_aliased_pairs"] == 0


def test_without_budget_the_smallest_unaliased_size_is_chosen() -> None:
    """14 factors fit in 8 runs only with aliased pairs (order 16), so 10 runs (order 20) are chosen."""
    design, meta = dispatch_supersaturated(_factors(14), None)
    assert design.shape == (10, 14)
    assert meta["n_fully_aliased_pairs"] == 0


def test_a_budget_forcing_aliasing_warns() -> None:
    with pytest.warns(UserWarning, match="10 runs avoid that"):
        _design, meta = dispatch_supersaturated(_factors(14), 8)
    assert meta["n_fully_aliased_pairs"] > 0


class TestGenerateDesign:
    def test_auto_selected_when_runs_are_fewer_than_effects(self) -> None:
        """10 factors in 6 runs has an unaliased design; 12 in 8 does not, so that falls back to Plackett-Burman."""
        result = generate_design(_factors(10), budget=6)
        assert result.design_type == "supersaturated"
        assert result.n_runs == 6  # no centre points added
        assert generate_design(_factors(12), budget=8).design_type == "plackett_burman"
        assert set(result.design_actual["X1"]) == {0.0, 10.0}

    def test_explicit(self) -> None:
        result = generate_design(_factors(18), design_type="supersaturated")
        assert result.n_runs == 10
        assert result.metadata["e_s2_efficiency"] > 0.8


class TestErrors:
    def test_budget_that_is_not_supersaturated(self) -> None:
        with pytest.raises(ValueError, match="plackett_burman"):
            dispatch_supersaturated(_factors(6), 8)

    def test_budget_with_no_hadamard_matrix(self) -> None:
        with pytest.raises(ValueError, match="run counts available"):
            dispatch_supersaturated(_factors(20), 7)

    def test_categorical_factor(self) -> None:
        factors = [*_factors(4), Factor(name="C", type="categorical", levels=["a", "b"])]
        with pytest.raises(ValueError, match="continuous factors"):
            dispatch_supersaturated(factors, 4)


class TestReviewFixes:
    @pytest.mark.parametrize("k", [3, 4, 5])
    def test_no_budget_refuses_factor_counts_without_an_unaliased_design(self, k: int) -> None:
        """3 to 5 factors used to get 4 or 6 runs (k + 1 or more: not supersaturated), with a negative bound."""
        with pytest.raises(ValueError, match="plackett_burman"):
            dispatch_supersaturated(_factors(k))

    def test_available_run_counts_stay_below_k_plus_one(self) -> None:
        with pytest.raises(ValueError, match=r"available are \[4\]"):
            dispatch_supersaturated(_factors(5), 3)

    @pytest.mark.parametrize("k", [6, 7, 8])
    def test_six_run_designs_with_every_abs_s_equal_to_two_are_fully_efficient(self, k: int) -> None:
        """In 6 runs (6 = 2 mod 4) every |s_ij| >= 2, so E(s^2) = 4 is optimal; it was reported as 36-77%."""
        design, meta = dispatch_supersaturated(_factors(k), 6)
        s = design.T @ design
        assert np.all(np.abs(s[np.triu_indices(k, 1)]) == 2)
        assert meta["e_s2_lower_bound"] == pytest.approx(4.0)
        assert meta["e_s2_efficiency"] == pytest.approx(1.0)

    def test_bound_is_never_negative(self) -> None:
        assert e_s2_lower_bound(6, 4) == 0.0

    @pytest.mark.parametrize(("k", "budget", "runs"), [(10, 7, 6), (10, 9, 6), (22, 13, 12)])
    def test_auto_selection_uses_the_largest_supersaturated_design_under_the_budget(
        self, k: int, budget: int, runs: int
    ) -> None:
        result = generate_design(_factors(k), budget=budget)
        assert result.design_type == "supersaturated"
        assert result.n_runs == runs

    def test_whole_float_budget_is_accepted_and_a_fractional_one_refused(self) -> None:
        assert generate_design(_factors(10), budget=6.0).n_runs == 6
        assert dispatch_supersaturated(_factors(10), 6.0)[0].shape == (6, 10)
        with pytest.raises(ValueError, match="whole number"):
            dispatch_supersaturated(_factors(10), 6.5)

    def test_aliasing_warning_can_be_caught_and_names_no_impossible_run_count(self) -> None:
        with pytest.warns(UserWarning, match="plackett_burman"):
            _design, meta = dispatch_supersaturated(_factors(4), 4)
        assert meta["n_fully_aliased_pairs"] == 1
