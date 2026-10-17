"""Splitting runs between two levels that differ in noise or in cost."""

from __future__ import annotations

import pytest

from process_improve.experiments import allocate_runs


def test_equal_noise_and_equal_cost_split_the_runs_evenly():
    found = allocate_runs(variance_a=1, variance_b=1, budget=12)
    assert (found["best"]["runs_a"], found["best"]["runs_b"]) == (6, 6)
    assert found["best"]["variance"] == pytest.approx(1 / 3)
    assert found["best"]["efficiency"] == pytest.approx(100.0)


def test_a_noisier_level_takes_more_runs_in_proportion_to_its_standard_deviation():
    """Nine times the variance is three times the standard deviation: three times the runs (Neyman)."""
    found = allocate_runs(variance_a=1, variance_b=9, budget=12)
    assert (found["best"]["runs_a"], found["best"]["runs_b"]) == (3, 9)
    assert found["best"]["variance"] == pytest.approx(4 / 3)


def test_a_dearer_level_takes_fewer_runs():
    """Four times the cost at B halves B's share relative to A: n_A / n_B near sqrt(4) = 2."""
    found = allocate_runs(variance_a=1, variance_b=1, cost_b=4, budget=30)
    assert (found["best"]["runs_a"], found["best"]["runs_b"]) == (10, 5)


def test_every_split_is_listed_with_its_efficiency_relative_to_the_best():
    found = allocate_runs(variance_a=1, variance_b=1, budget=4)
    assert [(s["runs_a"], s["runs_b"]) for s in found["splits"]] == [(1, 3), (2, 2), (3, 1)]
    assert [round(s["efficiency"], 1) for s in found["splits"]] == [75.0, 100.0, 75.0]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"variance_a": 0, "variance_b": 1}, "positive"),
        ({"variance_a": 1, "variance_b": 1, "cost_a": -1}, "positive"),
        ({"variance_a": 1, "variance_b": 1, "budget": 1}, "does not buy"),
    ],
)
def test_a_split_that_cannot_be_made_is_refused(kwargs, message):
    with pytest.raises(ValueError, match=message):
        allocate_runs(**kwargs)
