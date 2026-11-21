"""All-bias designs (Box and Draper 2007, Appendices 6B and 7H; #972 V2, V3) and the optimum spread (V4).

The design is all-bias when its pure second moment equals the region's. These tests hold the function
to Box and Draper's tables and, independently, to that condition: on the corners and centre runs of the
design it builds, and on points drawn uniformly over the region.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy import optimize

from process_improve.experiments.designs_response_surface import all_bias_ccd, all_bias_distance, optimum_spread


@pytest.mark.parametrize(
    ("region", "distances"),
    [("spherical", [0.500, 0.559, 0.612, 0.661, 0.707]), ("cuboidal", [0.577, 0.645, 0.707, 0.764, 0.816])],
)
def test_two_factors_give_box_and_drapers_table(region, distances):
    """Table 6B: k = 2, four cube runs and 0 to 4 centre runs."""
    found = [all_bias_distance(2, 4, n_center, region=region) for n_center in range(5)]
    assert found == pytest.approx(distances, abs=5e-4)


@pytest.mark.parametrize(("k", "cube_runs", "n_center"), [(2, 4, 0), (3, 8, 3), (4, 8, 2), (5, 32, 4)])
@pytest.mark.parametrize("region", ["spherical", "cuboidal"])
def test_the_design_matches_the_regions_second_moment(k, cube_runs, n_center, region):
    """The mean of x_i^2 over the design's runs equals the region's pure second moment."""
    distance = all_bias_distance(k, cube_runs, n_center, region=region)
    corners = np.array(list(itertools.product([-distance, distance], repeat=k)))[:cube_runs]
    runs = np.vstack([corners, np.zeros((n_center, k))])
    region_moment = 1 / (k + 2) if region == "spherical" else 1 / 3
    assert np.mean(runs[:, 0] ** 2) == pytest.approx(region_moment)


@pytest.mark.parametrize("k", [2, 3, 5])
def test_the_spheres_second_moment_is_one_over_k_plus_two(k):
    """Points drawn uniformly in the unit ball have E[x_1^2] = 1/(k + 2), the moment the function uses."""
    rng = np.random.default_rng(0)
    directions = rng.standard_normal((200_000, k))
    points = directions / np.linalg.norm(directions, axis=1, keepdims=True) * rng.random((200_000, 1)) ** (1 / k)
    assert all_bias_distance(k, 1, 0) ** 2 == pytest.approx(np.mean(points[:, 0] ** 2), rel=1e-2)


def test_expand_spreads_the_design():
    assert all_bias_distance(3, 8, 3, expand=1.15) == pytest.approx(0.603, abs=5e-4)
    assert all_bias_distance(3, 8, 3, expand=1.15) == pytest.approx(1.15 * all_bias_distance(3, 8, 3))


@pytest.mark.parametrize(
    ("args", "options", "message"),
    [
        ((2, 4, 0), {"region": "cube"}, "region must be"),
        ((0, 4, 0), {}, "at least 1"),
        ((2, 4, -1), {}, "at least 0"),
        ((2, 4, 0), {"expand": 0.0}, "expand positive"),
    ],
)
def test_bad_arguments_are_refused(args, options, message):
    with pytest.raises(ValueError, match=message):
        all_bias_distance(*args, **options)


@pytest.mark.parametrize(
    ("region", "row"),
    [("spherical", (2.67, 0.577, 0.816)), ("cuboidal", (2.66, 0.738, 0.830))],
)
def test_two_factors_give_box_and_drapers_composite_table(region, row):
    """Table 7H: k = 2 with four cube runs."""
    found = all_bias_ccd(2, 4, region=region)
    assert [found["n_center_points"], found["cube_distance"], found["axial_distance"]] == pytest.approx(row, abs=5e-3)
    assert found["n_center_points_rounded_up"] == 3


@pytest.mark.parametrize(("k", "cube_runs"), [(2, 4), (3, 8), (4, 16), (5, 16), (6, 32)])
@pytest.mark.parametrize(
    ("region", "moments"),
    [
        ("spherical", lambda k: (1 / (k + 2), 3 / ((k + 2) * (k + 4)), 1 / ((k + 2) * (k + 4)))),
        ("cuboidal", lambda _k: (1 / 3, 1 / 5, 1 / 9)),
    ],
)
def test_the_composite_design_matches_the_regions_moments(k, cube_runs, region, moments):
    """The design's [ii], [iiii] and [iijj] moments, with its real number of centre runs, equal the region's."""
    found = all_bias_ccd(k, cube_runs, region=region)
    a, b, n_runs = found["cube_distance"], found["axial_distance"], cube_runs + 2 * k + found["n_center_points"]
    design = ((cube_runs * a**2 + 2 * b**2) / n_runs, (cube_runs * a**4 + 2 * b**4) / n_runs, cube_runs * a**4 / n_runs)
    assert design == pytest.approx(moments(k))


def test_the_rounded_centre_runs_are_never_negative_and_a_whole_number_stays():
    assert all_bias_ccd(6, 16, region="cuboidal")["n_center_points"] == pytest.approx(-0.28, abs=5e-3)
    assert all_bias_ccd(6, 16, region="cuboidal")["n_center_points_rounded_up"] == 0
    assert all_bias_ccd(4, 16)["n_center_points_rounded_up"] == 3  # exactly 3 in the sphere, not taken up to 4


def test_expand_spreads_the_composite_design_but_not_its_centre_runs():
    plain, spread = all_bias_ccd(3, 8), all_bias_ccd(3, 8, expand=1.15)
    assert spread["cube_distance"] == pytest.approx(1.15 * plain["cube_distance"])
    assert spread["axial_distance"] == pytest.approx(1.15 * plain["axial_distance"])
    assert spread["n_center_points"] == plain["n_center_points"]


@pytest.mark.parametrize(
    ("args", "options", "message"),
    [
        ((2, 4), {"region": "sphere"}, "region must be"),
        ((2, 0), {}, "at least 1"),
        ((2, 4), {"expand": -1}, "positive"),
    ],
)
def test_bad_composite_arguments_are_refused(args, options, message):
    with pytest.raises(ValueError, match=message):
        all_bias_ccd(*args, **options)


@pytest.mark.parametrize(
    ("weight", "spreads"),
    [("uniform", [1.00, 1.02, 1.04, 1.08, 1.14, 1.25]), ("normal", [1.00, 1.09, 1.16, 1.26, 1.41])],
)
def test_the_optimum_spread_gives_box_and_drapers_table_13_1(weight, spreads):
    """Table 13.1, at V/(V + B) = 0, 0.2, 0.333, 0.5, 0.667 and 0.8 (the normal weight's 0.2 left out: 1.0525)."""
    shares = [0, 0.2, 0.333, 0.5, 0.667, 0.8] if weight == "uniform" else [0, 0.333, 0.5, 0.667, 0.8]
    assert [optimum_spread(weight, share) for share in shares] == pytest.approx(spreads, abs=5e-3)


@pytest.mark.parametrize(("weight", "excess"), [("uniform", 0.8), ("normal", 2.0)])
@pytest.mark.parametrize("share", [0.1, 0.5, 0.9])
def test_the_optimum_spread_minimizes_variance_plus_bias(weight, excess, share):
    """At the bias it implies, no other spread gives a smaller V + B, and V is that share of the total."""
    t_best = optimum_spread(weight, share) ** 2
    scale = 1 / (2 * t_best**2 * (t_best - 1))  # alpha^2 mu2^2 at which t_best is the optimum

    def total(t: float) -> float:
        return 1 + 1 / t + scale * (excess + (t - 1) ** 2)

    found = optimize.minimize_scalar(total, bounds=(1.0, 10 * t_best), method="bounded", options={"xatol": 1e-10})
    assert found.x == pytest.approx(t_best, rel=1e-4)
    assert (1 + 1 / t_best) / total(t_best) == pytest.approx(share)


@pytest.mark.parametrize(("weight", "share"), [("triangular", 0.5), ("uniform", 1.0), ("normal", -0.1)])
def test_bad_optimum_spread_arguments_are_refused(weight, share):
    with pytest.raises(ValueError, match=r"weight must be|variance_share must be"):
        optimum_spread(weight, share)
