"""Tests for the prediction from marginal means (Taguchi's additive model)."""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import marginal_means_prediction
from process_improve.experiments.reference_catalogue import reference_experiments


def _full_factorial(k: int) -> pd.DataFrame:
    return pd.DataFrame(list(itertools.product([1, 2], repeat=k)), columns=list("ABC"[:k]))


def test_on_a_balanced_design_the_main_effects_are_the_least_squares_ones() -> None:
    """In a 2^3 every level, and every pair of levels, occurs equally often, so no warning, and OLS agrees."""
    runs = _full_factorial(3)
    y = np.array([3.0, 5.0, 4.0, 9.0, 6.0, 7.0, 8.0, 15.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        predicted = marginal_means_prediction(runs, y, runs)
    x = np.column_stack([np.ones(8), (runs.to_numpy() == 2).astype(float)])
    least_squares = x @ np.linalg.lstsq(x, y, rcond=None)[0]
    np.testing.assert_allclose(predicted.to_numpy(), least_squares)


def test_an_interaction_adds_its_cell_mean_less_the_two_marginal_means_plus_the_grand_mean() -> None:
    """With A x B in a 2^2, the prediction at a cell is that cell's own mean."""
    runs = pd.concat([_full_factorial(2)] * 2, ignore_index=True)
    y = pd.Series([10.0, 12.0, 14.0, 20.0, 12.0, 14.0, 16.0, 22.0])
    predicted = marginal_means_prediction(runs, y, _full_factorial(2), interactions=[("A", "B")])
    assert predicted.tolist() == pytest.approx([11.0, 13.0, 15.0, 21.0])
    additive = marginal_means_prediction(runs, y, {"A": 2, "B": 2})
    assert additive.index.tolist() == [0]
    assert additive.name == "predicted"
    assert additive.iloc[0] == pytest.approx(15.0 + (18.0 - 15.0) + (17.0 - 15.0))


def test_a_run_without_its_response_is_left_out_of_every_mean() -> None:
    runs = _full_factorial(2)
    with_gap = marginal_means_prediction(pd.concat([runs, runs.iloc[[0]]]), [1.0, 2.0, 3.0, 4.0, np.nan], runs)
    np.testing.assert_allclose(with_gap, marginal_means_prediction(runs, [1.0, 2.0, 3.0, 4.0], runs))


def test_levels_balanced_one_at_a_time_but_not_in_pairs_are_warned_of() -> None:
    """A and B each have two runs at each level, but never meet at (1, 2) or (2, 1): B's mean carries A's effect."""
    runs = pd.DataFrame({"A": [1, 1, 2, 2], "B": [1, 1, 2, 2]})
    with pytest.warns(UserWarning, match="not balanced"):
        marginal_means_prediction(runs, [1.0, 2.0, 3.0, 4.0], {"A": 1, "B": 1})


@pytest.mark.dataset
def test_schmidt_and_launsbys_lunar_concrete_prediction_is_reproduced_and_warned_of() -> None:
    """Equation [8] on 18 of 80 combinations: Table 2 prints 3893.40 for run 1 and 2214.17 for run 5."""
    entry = next(e for e in reference_experiments("schmidt-launsby-2000") if e["name"] == "schmidt-lunar-concrete")
    table = pd.DataFrame(entry["runs"]["rows"], columns=entry["runs"]["columns"])
    with pytest.warns(UserWarning, match="not balanced"):
        predicted = marginal_means_prediction(
            table, table["actual"], table[list("CEWPA")], interactions=[("W", "E"), ("E", "P"), ("W", "P")]
        )
    assert predicted.iloc[[0, 4]].round(2).tolist() == [3893.40, 2214.17]


def test_a_setting_or_factor_the_runs_do_not_have_is_refused() -> None:
    runs, y = _full_factorial(2), [1.0, 2.0, 3.0, 4.0]
    with pytest.raises(ValueError, match="responses has 3 values; design_matrix has 4 runs"):
        marginal_means_prediction(runs, y[:3], runs)
    with pytest.raises(ValueError, match=r"factors \['C'\] of `at` are not columns"):
        marginal_means_prediction(runs, y, {"A": 1, "C": 1})
    with pytest.raises(ValueError, match="must be a pair of the factors"):
        marginal_means_prediction(runs, y, runs, interactions=[("A", "C")])
    with pytest.raises(ValueError, match="No run has"):
        marginal_means_prediction(runs, y, {"A": 3, "B": 1})
