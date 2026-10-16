"""Measures read off a design's coded runs: its models, the cost of losing a run, its projections, its fingerprint."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from process_improve.experiments import Factor, design_measures, generate_design
from process_improve.experiments.design_measures import (
    design_classes,
    estimable_models,
    fingerprint,
    leverages,
    model_matrix,
    projectivity,
    quadratic_projection,
    relabelling,
    run_loss,
    same_design,
)

SQUARE = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]], dtype=float)


def coded(design_type: str, k: int, **options) -> np.ndarray:
    """Return a design the library makes, its runs in coded units in standard order."""
    factors = [Factor(name="ABCDEFGHJKLMN"[i], low=-1, high=1) for i in range(k)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # a definitive screening design's error degrees of freedom: not tested here
        result = generate_design(factors, design_type=design_type, **options)
    runs = result.design[[f.name for f in factors]].to_numpy(dtype=float)
    return runs[np.argsort(result.run_order)]


# --------------------------------------------------------------------------- the model matrix
def test_the_model_matrix_adds_the_terms_of_each_model_in_turn():
    assert model_matrix(SQUARE, ["numeric"] * 2, "main_effects").shape == (4, 3)
    assert model_matrix(SQUARE, ["numeric"] * 2, "interactions").shape == (4, 4)
    assert model_matrix(SQUARE, ["numeric"] * 2, "quadratic").shape == (4, 6)


def test_a_categorical_factor_enters_with_a_column_per_extra_level():
    runs = np.array([[s, t] for s in (-1.0, 0.0, 1.0) for t in (-1.0, 1.0)])
    matrix = model_matrix(runs, ["categorical", "numeric"], "quadratic")
    assert matrix.shape == (6, 1 + 2 + 1 + 2 + 1), "intercept, two supplier columns, T, their products, T squared"
    assert estimable_models(runs, ["categorical", "numeric"]) == {
        "main_effects": True,
        "interactions": True,
        "quadratic": False,
    }


def test_a_mixture_or_a_missing_setting_has_no_model_matrix():
    simplex = np.eye(3)
    assert model_matrix(simplex, ["mixture"] * 3, "main_effects") is None
    assert set(estimable_models(simplex, ["mixture"] * 3).values()) == {None}
    gap = np.array([[-1, 1], [1, np.nan], [-1, -1], [1, 1]])
    assert set(estimable_models(gap, ["numeric"] * 2).values()) == {None}


def test_a_two_level_factorial_estimates_interactions_but_no_squares():
    assert estimable_models(SQUARE, ["numeric"] * 2) == {"main_effects": True, "interactions": True, "quadratic": False}
    assert estimable_models(SQUARE[:, :1], ["categorical"]) == {
        "main_effects": True,
        "interactions": None,
        "quadratic": None,
    }


# --------------------------------------------------------------------------- losing a run
def test_leverages_add_up_to_the_number_of_coefficients():
    runs = coded("d_optimal", 3, model_type="quadratic", budget=14)
    matrix = model_matrix(runs, ["numeric"] * 3, "quadratic")
    assert leverages(matrix).sum() == pytest.approx(matrix.shape[1])


def test_losing_a_run_scales_the_determinant_by_one_less_its_leverage():
    """The claim the measure rests on: det(X'X) without run i is det(X'X) times (1 - h_i)."""
    runs = coded("d_optimal", 3, model_type="quadratic", budget=14)
    matrix = model_matrix(runs, ["numeric"] * 3, "quadratic")
    h = leverages(matrix)
    whole = np.linalg.det(matrix.T @ matrix)
    for i in range(len(matrix)):
        rest = np.delete(matrix, i, axis=0)
        assert np.linalg.det(rest.T @ rest) == pytest.approx(whole * (1 - h[i]), rel=1e-6, abs=1e-6)
    found = run_loss(runs, ["numeric"] * 3, "quadratic")
    assert found["worst_d_ratio"] == pytest.approx((1 - h.max()) ** (1 / matrix.shape[1]), abs=1e-4)


def test_no_run_of_a_saturated_plackett_burman_design_can_be_lost():
    runs = coded("plackett_burman", 11, n_center_points=0)
    found = run_loss(runs, ["numeric"] * 11, "main_effects")
    assert (found["survives"], found["critical_runs"], found["worst_d_ratio"]) == (False, 12, 0.0)
    assert found["leverage"] == [1.0] * 12


def test_a_center_run_lets_a_two_by_two_factorial_lose_any_run():
    with_center = np.vstack([SQUARE, [0, 0]])
    found = run_loss(with_center, ["numeric"] * 2, "interactions")
    assert (found["survives"], found["critical_runs"]) == (True, 0)
    assert found["leverage"] == [0.95, 0.95, 0.95, 0.95, 0.2]


def test_a_model_the_design_does_not_estimate_has_no_run_loss():
    assert run_loss(SQUARE, ["numeric"] * 2, "quadratic") is None
    assert run_loss(np.eye(3), ["mixture"] * 3, "main_effects") is None


# --------------------------------------------------------------------------- projections
def test_every_three_factors_of_a_definitive_screening_design_fit_the_full_quadratic():
    assert quadratic_projection(coded("dsd", 6, n_center_points=1), ["numeric"] * 6) == 3


def test_a_central_composite_design_projects_onto_every_factor_at_once():
    assert quadratic_projection(coded("ccd", 3, alpha="face_centered", n_center_points=3), ["numeric"] * 3) == 3


def test_a_two_level_design_fits_no_square_and_a_mixture_has_no_projection():
    assert quadratic_projection(SQUARE, ["numeric"] * 2) == 0
    assert quadratic_projection(np.eye(3), ["mixture"] * 3) is None
    assert quadratic_projection(SQUARE, ["categorical"] * 2) is None


def test_a_capped_search_returns_the_largest_size_it_checked_in_full():
    runs = coded("dsd", 6, n_center_points=1)
    assert quadratic_projection(runs, ["numeric"] * 6, most=6) == 1, "6 one-factor projections, but 15 two-factor ones"


@pytest.mark.parametrize(
    ("runs", "found"),
    [
        (np.array([[-1, -1, 1], [1, -1, -1], [-1, 1, -1], [1, 1, 1]], dtype=float), 2),
        (coded("full_factorial", 3, n_center_points=0), 3),
        (coded("plackett_burman", 11, n_center_points=0), 3),
        (coded("fractional_factorial", 7, resolution=4, n_center_points=0), 3),
        (coded("fractional_factorial", 5, resolution=5, n_center_points=3), 4),
    ],
    ids=["half-fraction", "full-factorial", "plackett-burman-12", "resolution-IV", "resolution-V-with-centers"],
)
def test_projectivity_is_the_most_factors_every_projection_holds_in_full(runs, found):
    assert projectivity(runs) == found


def test_a_design_with_three_levels_has_no_projectivity():
    assert projectivity(coded("dsd", 6, n_center_points=1)) is None
    assert projectivity(np.array([[-1, np.nan], [1, 1]])) is None


# --------------------------------------------------------------------------- the same design
def _shuffled(runs: np.ndarray, seed: int) -> np.ndarray:
    """Return the design with its runs and factors put in another order, and some factors' signs swapped."""
    rng = np.random.default_rng(seed)
    out = runs[rng.permutation(len(runs))][:, rng.permutation(runs.shape[1])]
    return out * rng.choice([-1, 1], size=runs.shape[1])


@pytest.mark.parametrize(
    "runs",
    [
        coded("fractional_factorial", 6, resolution=4, n_center_points=0),
        coded("plackett_burman", 7, n_center_points=0),
        coded("ccd", 3, alpha="rotatable", n_center_points=2),
    ],
    ids=["fraction", "plackett-burman", "ccd"],
)
def test_a_reordered_relabelled_design_is_the_same_design(runs):
    other = _shuffled(runs, seed=7)
    assert fingerprint(other) == fingerprint(runs)
    assert same_design(runs, other)


def test_two_fractions_of_different_resolution_are_not_the_same_design():
    four = coded("fractional_factorial", 6, resolution=4, n_center_points=0)
    three = coded("fractional_factorial", 6, generators=["E=AB", "F=CD"], n_center_points=0)
    assert len(four) == len(three) == 16
    assert fingerprint(four) != fingerprint(three)
    assert not same_design(four, three)


def test_designs_of_other_sizes_or_with_a_gap_are_not_the_same_design():
    assert not same_design(SQUARE, SQUARE[:3])
    assert not same_design(SQUARE, np.array([[-1, -1], [1, -1], [-1, 1], [1, np.nan]]))
    assert fingerprint(np.array([[np.nan]])) is None


def test_a_relabelling_can_use_some_columns_of_an_orthogonal_array():
    """Three factors assigned to columns of the L8, as a source assigns them, the others left for interactions."""
    array = coded("plackett_burman", 7, n_center_points=0)
    chosen = array[:, [4, 1, 6]] * [1, -1, 1]
    found = relabelling(chosen, array)
    assert found is not None
    np.testing.assert_array_equal(
        np.sort(chosen, axis=0), np.sort(np.column_stack([array[:, c] * s for c, s in found]), axis=0)
    )


def test_a_search_that_runs_out_of_steps_finds_nothing():
    runs = coded("fractional_factorial", 6, resolution=4, n_center_points=0)
    assert relabelling(_shuffled(runs, seed=3), runs, most=1) is None


def test_designs_are_grouped_into_classes_of_one_design(monkeypatch):
    runs = coded("fractional_factorial", 6, resolution=4, n_center_points=0)
    other = coded("fractional_factorial", 6, generators=["E=AB", "F=CD"], n_center_points=0)
    classes = design_classes({"a": runs, "b": _shuffled(runs, seed=1), "c": other, "gap": np.full((2, 2), np.nan)})
    assert classes["a"] == classes["b"] != classes["c"]
    assert "gap" not in classes
    # Two designs that share a fingerprint but are not one go in classes of their own.
    monkeypatch.setattr(design_measures, "fingerprint", lambda _runs: "same")
    assert design_classes({"a": runs, "c": other, "b": _shuffled(runs, seed=1)}) == {
        "a": "same",
        "c": "same-2",
        "b": "same",
    }
