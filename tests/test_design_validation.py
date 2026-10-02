"""Inputs that ``generate_design`` refuses rather than turning into a wrong design."""

from __future__ import annotations

import pytest

from process_improve.experiments import generate_design
from process_improve.experiments.factor import Factor


def test_duplicate_factor_names_raise() -> None:
    """Two factors named 'A' collapsed into one column of a two-factor design."""
    factors = [Factor(name="A", low=0, high=10), Factor(name="A", low=5, high=9)]
    with pytest.raises(ValueError, match="unique"):
        generate_design(factors, "full_factorial")


@pytest.mark.parametrize(
    "design_type", ["full_factorial", "ccd", "box_behnken", "plackett_burman", "dsd", "fractional_factorial", "omars"]
)
def test_mixture_components_in_a_box_family_raise(design_type: str) -> None:
    """These families returned coded +/-1 rows labelled as proportions (row sums from -3 to 3)."""
    factors = [Factor(name=f"x{i}", type="mixture", low=0.1, high=0.6) for i in range(3)]
    with pytest.raises(ValueError, match="sum to 1"):
        generate_design(factors, design_type)


@pytest.mark.parametrize("design_type", ["mixture", "d_optimal", "maximin"])
def test_mixture_components_in_a_mixture_family_sum_to_one(design_type: str) -> None:
    factors = [Factor(name=f"x{i}", type="mixture", low=0.1, high=0.6) for i in range(3)]
    actual = generate_design(factors, design_type, budget=10).design_actual[["x0", "x1", "x2"]]
    assert actual.sum(axis=1).round(9).eq(1).all()
