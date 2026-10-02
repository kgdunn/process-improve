"""Tests for supersaturated designs (Lin's half-fraction of a Hadamard matrix)."""

from __future__ import annotations

import logging

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


@pytest.mark.parametrize("order", [4, 8, 12, 16, 20, 24, 28, 32, 40, 44, 48, 60, 68, 72, 84])
def test_hadamard_orders(order: int) -> None:
    """pyDOE3 and Paley's I + C between them cover these orders; the first column is all +1."""
    h = hadamard(order)
    if order == 28:  # neither construction reaches 28 (27 is not prime, pyDOE3 has no 28)
        assert h is None
        return
    assert h is not None
    np.testing.assert_array_equal(h.T @ h, order * np.eye(order))
    assert (h[:, 0] == 1).all()


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


def test_a_budget_forcing_aliasing_warns(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        _design, meta = dispatch_supersaturated(_factors(14), 8)
    assert meta["n_fully_aliased_pairs"] > 0
    assert "10 runs avoid that" in caplog.text


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
