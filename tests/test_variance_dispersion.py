"""Variance dispersion over spheres about a design's centre, and rotatability (#790 C4).

The book says a Box-Behnken design "is rotatable for k = 4 (and k = 7)" and nearly rotatable for
the other factor counts (the catalogue's pid-book entries box-behnken-k3 to -k7), and a central
composite design is rotatable when its axial distance is the fourth root of its factorial runs.
These tests hold the function to those statements, to evaluate_design's prediction variance at the
runs, and to dense evaluations of the variance on a circle and on a sphere.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from patsy import build_design_matrices

from process_improve.experiments import Factor, evaluate_design, generate_design, variance_dispersion
from process_improve.experiments.evaluate import _build_model_matrix
from process_improve.experiments.variance_dispersion import _sphere_moments


def factors(k: int) -> list[Factor]:
    return [Factor(name=f"x{i + 1}", low=-1, high=1) for i in range(k)]


def box_behnken(k: int):
    return generate_design(factors(k), design_type="box_behnken", n_center_points=3)


def ccd(k: int, alpha: str | float, n_center_points: int = 5):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # the axial runs go past the ranges, as asked
        return generate_design(factors(k), design_type="ccd", n_center_points=n_center_points, alpha=alpha)


def scaled_variance(design, model: str, points: np.ndarray) -> np.ndarray:
    """Return N f(x)' (X'X)^-1 f(x) at each point, straight from the model matrix."""
    names = list(design.factor_names)
    X, _, info = _build_model_matrix(design.design[names].astype(float), model, names)
    terms = np.asarray(build_design_matrices([info], pd.DataFrame(points, columns=names))[0])
    return len(X) * np.einsum("ij,jk,ik->i", terms, np.linalg.inv(X.T @ X), terms)


# --------------------------------------------------------------------------- rotatability
@pytest.mark.parametrize(("k", "rotatable"), [(3, False), (4, True), (5, False), (6, False), (7, True)])
def test_a_box_behnken_design_is_rotatable_in_four_and_seven_factors(k, rotatable):
    found = variance_dispersion(box_behnken(k), radii=6)
    assert found["rotatable"] is rotatable
    relative = np.array(found["spread"]) / np.array(found["mean"])
    assert bool(relative.max() < 1e-12) is rotatable


@pytest.mark.parametrize("k", [2, 3, 4])
def test_a_central_composite_design_is_rotatable_at_the_fourth_root_of_its_cube_runs(k):
    assert variance_dispersion(ccd(k, "rotatable"), radii=6)["rotatable"]
    assert not variance_dispersion(ccd(k, "face_centered"), radii=6)["rotatable"]
    assert not variance_dispersion(ccd(k, 1.5), radii=6)["rotatable"]


def test_a_two_level_factorial_is_rotatable_for_main_effects_and_not_with_the_interaction():
    """For main effects v = 1 + |x|^2 for the 2^2 factorial; with the interaction, v adds x1^2 x2^2."""
    design = generate_design(factors(2), design_type="full_factorial", n_center_points=0)
    main = variance_dispersion(design, model="main_effects", radii=[0.0, 1.0, np.sqrt(2)])
    assert main["rotatable"]
    assert main["mean"] == pytest.approx([1.0, 2.0, 3.0])
    both = variance_dispersion(design, model="interactions", radii=[1.0])
    assert not both["rotatable"]
    assert (both["minimum"][0], both["mean"][0], both["maximum"][0]) == pytest.approx((2.0, 2.125, 2.25))


# --------------------------------------------------------------------------- the values themselves
def test_the_values_at_the_runs_are_evaluate_design_s():
    """The rotatable CCD in 2 factors with 5 centre runs: Var(y-hat) is 0.2 at the centre, 0.625 at a corner."""
    design = ccd(2, "rotatable")
    at_runs = evaluate_design(design, model="quadratic", metric="prediction_variance")
    found = variance_dispersion(design, radii=[0.0, np.sqrt(2)])
    assert found["n_runs"] == 13
    assert found["minimum"] == pytest.approx([13 * at_runs["min"], 13 * at_runs["max"]])
    assert found["minimum"] == pytest.approx([2.6, 8.125])


def test_on_a_circle_the_extremes_and_the_mean_are_those_of_a_fine_grid_of_angles():
    """A face-centred CCD in 2 factors, on circles of radius 0.5, 1 and 1.4; the grid's mean is exact."""
    design = ccd(2, "face_centered")
    angles = np.linspace(0, 2 * np.pi, 20_000, endpoint=False)
    found = variance_dispersion(design, radii=[0.5, 1.0, 1.4])
    for j, radius in enumerate(found["radius"]):
        values = scaled_variance(design, "quadratic", radius * np.column_stack([np.cos(angles), np.sin(angles)]))
        assert found["mean"][j] == pytest.approx(values.mean(), rel=1e-12)
        assert found["minimum"][j] == pytest.approx(values.min(), rel=1e-7)
        assert found["maximum"][j] == pytest.approx(values.max(), rel=1e-7)


def test_on_a_sphere_the_extremes_reach_past_a_dense_sample():
    """The Box-Behnken design in 3 factors: 200,000 directions on the unit sphere fall between them."""
    design = box_behnken(3)
    directions = np.random.default_rng(5).normal(size=(200_000, 3))
    values = scaled_variance(design, "quadratic", directions / np.linalg.norm(directions, axis=1, keepdims=True))
    found = variance_dispersion(design, radii=[1.0])
    assert found["minimum"][0] <= values.min() + 1e-12
    assert found["maximum"][0] >= values.max() - 1e-12
    assert found["maximum"][0] == pytest.approx(values.max(), rel=1e-4)
    assert found["mean"][0] == pytest.approx(values.mean(), rel=2e-3)


@pytest.mark.parametrize(
    ("exponents", "average"),
    [((0, 0), 1.0), ((2, 0), 1 / 2), ((4, 0), 3 / 8), ((2, 2), 1 / 8), ((1, 1), 0.0), ((3, 1), 0.0)],
)
def test_the_average_of_a_monomial_over_the_unit_circle(exponents, average):
    """Over the circle, the averages of cos^4 and cos^2 sin^2 are 3/8 and 1/8."""
    assert _sphere_moments(np.array([exponents]))[0] == pytest.approx(average)


def test_the_average_of_a_monomial_over_the_unit_sphere_in_three_dimensions():
    """E[u1^2] = 1/3, E[u1^4] = 3/15, E[u1^2 u2^2] = 1/15, E[u1^2 u2^2 u3^2] = 1/105."""
    exponents = np.array([[2, 0, 0], [4, 0, 0], [2, 2, 0], [2, 2, 2]])
    assert _sphere_moments(exponents) == pytest.approx([1 / 3, 3 / 15, 1 / 15, 1 / 105])


# --------------------------------------------------------------------------- radii and inputs
def test_a_number_of_radii_spans_the_centre_to_the_furthest_run():
    found = variance_dispersion(box_behnken(3), radii=5)
    assert found["design_radius"] == pytest.approx(np.sqrt(2))
    assert found["radius"] == pytest.approx(np.linspace(0, np.sqrt(2), 5))
    centre = (found["minimum"][0], found["mean"][0], found["maximum"][0])
    assert centre == pytest.approx((5.0, 5.0, 5.0)), "15 runs, 3 at the centre: N Var(b0) = 15 / 3"
    assert all(lo <= mean <= hi for lo, mean, hi in zip(found["minimum"], found["mean"], found["maximum"], strict=True))


def test_a_data_frame_with_a_run_order_and_a_block_gives_the_same_values():
    design = box_behnken(3)
    frame = design.design.assign(Block=1)
    assert variance_dispersion(frame, radii=4) == variance_dispersion(design, radii=4)


def test_the_directions_are_seeded():
    design = box_behnken(5)
    assert variance_dispersion(design, radii=3, random_state=7) == variance_dispersion(design, radii=3, random_state=7)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"radii": 0}, "at least 1"),
        ({"radii": [-0.5, 1.0]}, "finite numbers, 0 or more"),
        ({"radii": [np.nan]}, "finite numbers, 0 or more"),
        ({"radii": []}, "finite numbers, 0 or more"),
        ({"n_directions": 0}, "n_directions must be"),
        ({"n_directions": True}, "n_directions must be"),
        ({"model": "x1 + x2 + x3 + I(x1 ** 4)"}, "not polynomials of degree 3 or less"),
    ],
)
def test_settings_that_cannot_be_used_are_refused(kwargs, message):
    with pytest.raises(ValueError, match=message):
        variance_dispersion(box_behnken(3), **kwargs)


def test_a_design_that_cannot_estimate_the_model_is_refused():
    design = generate_design(factors(3), design_type="full_factorial", n_center_points=0)
    with pytest.raises(ValueError, match="cannot estimate the model's 10 terms"):
        variance_dispersion(design, model="quadratic")


def test_a_categorical_factor_is_refused():
    frame = pd.DataFrame({"x1": [-1.0, 1.0, -1.0, 1.0], "kind": ["a", "a", "b", "b"]})
    with pytest.raises(ValueError, match=r"need numeric factors; \['kind'\] are categorical"):
        variance_dispersion(frame, model="main_effects")


def test_settings_in_actual_units_are_warned_about():
    frame = pd.DataFrame({"T": [240.0, 260.0, 240.0, 260.0, 250.0], "P": [1.0, 1.0, 2.0, 2.0, 1.5]})
    with pytest.warns(UserWarning, match="expects factor settings in coded units"):
        variance_dispersion(frame, model="main_effects", radii=2)


def test_a_missing_setting_is_refused():
    frame = pd.DataFrame({"x1": [-1.0, 1.0, np.nan, 1.0], "x2": [-1.0, -1.0, 1.0, 1.0]})
    with pytest.raises(ValueError, match="finite number"):
        variance_dispersion(frame, model="main_effects")
