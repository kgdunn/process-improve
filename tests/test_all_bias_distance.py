"""The all-bias distance of a two-level first-order design (Box and Draper 2007, Appendix 6B; #972 V2).

The design is all-bias when its pure second moment equals the region's. These tests hold the function
to Box and Draper's tables and, independently, to that condition: on the corners and centre runs of the
design it builds, and on points drawn uniformly over the region.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from process_improve.experiments.designs_response_surface import all_bias_distance


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
