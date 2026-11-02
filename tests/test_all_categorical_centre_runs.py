"""A factorial whose factors are all categorical gets no centre runs unless they are asked for (#788 A5).

A categorical factor has no centre, so the three default "centre runs" of such a design only repeat
cells it already has: a 3 x 3 factorial had 12 runs, with three of its diagonal cells twice.
"""

from __future__ import annotations

import warnings

import pytest

from process_improve.experiments import Factor, generate_design


def categorical(name: str, n_levels: int) -> Factor:
    return Factor(name=name, type="categorical", levels=[f"{name}{i}" for i in range(1, n_levels + 1)])


def runs(factors: list[Factor], **options) -> tuple[int, int]:
    """Return the number of runs of the design, and how many of them repeat an earlier run."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # no pure error without centre runs: wanted here
        design = generate_design(factors, **options).design
    cells = design[[f.name for f in factors]].astype(str).agg("|".join, axis=1)
    return len(design), int(cells.duplicated().sum())


@pytest.mark.parametrize(
    ("levels", "n_runs"),
    [((3, 3), 9), ((2, 3), 6), ((2, 2), 4), ((2, 2, 3), 12)],
)
def test_an_all_categorical_factorial_has_each_cell_once(levels, n_runs):
    factors = [categorical(name, k) for name, k in zip("ABC", levels, strict=False)]
    assert runs(factors, design_type="full_factorial") == (n_runs, 0)


def test_the_design_chosen_for_categorical_factors_has_no_centre_runs_either():
    assert runs([categorical("A", 3), categorical("B", 3)]) == (9, 0)


def test_centre_runs_asked_for_are_still_added():
    assert runs([categorical("A", 3), categorical("B", 3)], design_type="full_factorial", n_center_points=3) == (12, 3)


def test_a_design_with_a_numeric_factor_keeps_its_three_centre_runs():
    """With a numeric factor there is a centre to run at, so the default is unchanged."""
    factors = [Factor(name="T", low=10, high=20), categorical("B", 3)]
    n_runs, repeats = runs(factors, design_type="full_factorial")
    assert (n_runs, repeats) == (9, 0), "6 corners and 3 centre runs, one at each level of B"
