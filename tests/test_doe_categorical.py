"""Categorical factors, general full factorials and Taguchi arrays in ``generate_design``."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from process_improve.experiments import generate_design
from process_improve.experiments.factor import Factor


def _c(name: str) -> Factor:
    return Factor(name=name, low=0, high=10)


def _k(name: str, levels: list[str]) -> Factor:
    return Factor(name=name, type="categorical", levels=levels)


def _strength_two(columns: np.ndarray) -> bool:
    for i, j in itertools.combinations(range(columns.shape[1]), 2):
        _, counts = np.unique(columns[:, [i, j]], axis=0, return_counts=True)
        if counts.min() != counts.max():
            return False
    return True


@pytest.mark.parametrize("design_type", ["fractional_factorial", "plackett_burman", "full_factorial", "dsd"])
def test_two_level_categorical_factor_is_labelled(design_type: str) -> None:
    """These families failed with 'All values must be present in levels'."""
    factors = [_c("a"), _c("b"), _c("c"), _c("d"), _k("cat", ["old", "new"])]
    result = generate_design(factors, design_type)
    assert set(result.design_actual["cat"]) == {"old", "new"}


def test_dsd_with_categorical_keeps_main_effects_clear_of_second_order() -> None:
    """Jones and Nachtsheim (2013) DSD-augment: main effects orthogonal to every second-order column."""
    factors = [_c("a"), _c("b"), _c("c"), _k("p", ["x", "y"]), _k("q", ["x", "y"])]
    result = generate_design(factors, "dsd")
    d = result.design
    x = np.column_stack(
        [
            d[["a", "b", "c"]].to_numpy(dtype=float),
            np.where(d["p"] == "y", 1.0, -1.0),
            np.where(d["q"] == "y", 1.0, -1.0),
        ]
    )
    second = [x[:, i] * x[:, j] for i in range(5) for j in range(i, 5) if not (i == j and i >= 3)]
    assert np.abs(x.T @ np.column_stack(second)).max() == 0
    assert result.n_runs == 14  # 2 * 6 + 2 centre runs


@pytest.mark.parametrize("design_type", ["ccd", "box_behnken", "omars", "supersaturated", "latin_hypercube"])
def test_families_without_categorical_support_say_so(design_type: str) -> None:
    factors = [_c("a"), _c("b"), _c("c"), _k("cat", ["old", "new"])]
    with pytest.raises(ValueError, match="needs continuous factors"):
        generate_design(factors, design_type, budget=4)


def test_three_level_categorical_in_a_two_level_family_says_so() -> None:
    factors = [_c("a"), _c("b"), _k("cat", ["x", "y", "z"])]
    with pytest.raises(ValueError, match="two-level categorical factors only"):
        generate_design(factors, "fractional_factorial")


def test_general_full_factorial() -> None:
    """Every combination of every factor's levels: 2 x 3 x 3 = 18 runs, each exactly once."""
    factors = [_c("a"), _k("b", ["p", "q", "r"]), Factor(name="t", low=10, high=30, levels=[10, 20, 30])]
    result = generate_design(factors, "full_factorial", n_center_points=0)
    combos = set(map(tuple, result.design_actual[["a", "b", "t"]].to_numpy().tolist()))
    assert len(combos) == result.n_runs == 18


def test_continuous_levels_outside_the_range_raise() -> None:
    with pytest.raises(ValueError, match="must lie within"):
        generate_design([_c("a"), Factor(name="t", low=10, high=30, levels=[10, 40])], "full_factorial")


@pytest.mark.parametrize("k", range(2, 16))
def test_taguchi_any_number_of_two_level_factors(k: int) -> None:
    """pyDOE3 needed one level list per array column, so 4 to 6 factors (L8) failed."""
    result = generate_design([_c(f"x{i}") for i in range(k)], "taguchi")
    x = result.design[[f"x{i}" for i in range(k)]].to_numpy(dtype=float)
    assert _strength_two(x)
    assert not np.all(x == 0, axis=1).any()  # no centre points
    assert result.n_runs == {2: 4, 3: 4}.get(k, 8 if k <= 7 else 12 if k <= 11 else 16)


def test_taguchi_mixed_levels_use_a_wider_column() -> None:
    """A 2-level factor rides a 6-level column of L18, which keeps every pair balanced."""
    factors = [_c("a"), _k("b", ["p", "q", "r"]), Factor(name="t", low=10, high=30, levels=[10, 20, 30])]
    result = generate_design(factors, "taguchi")
    assert result.metadata["orthogonal_array"].startswith("L18")
    codes = result.design_actual[["a", "b", "t"]].apply(lambda col: col.astype("category").cat.codes)
    assert _strength_two(codes.to_numpy())
