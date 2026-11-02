"""Space-filling designs: review fixes for the uniform and maximin Latin hypercubes and the run count."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import pdist
from scipy.stats import qmc

from process_improve.experiments import Factor, generate_design
from process_improve.experiments.designs_space_filling import space_filling_design


def _box(k: int) -> list[Factor]:
    return [Factor(name=f"x{i}", low=0, high=10) for i in range(k)]


def test_uniform_design_sits_on_the_slice_centres() -> None:
    """Fang's uniform designs are U-type: every coordinate at (2i - 1) / 2n, the jitter of a scrambled LHS removed."""
    n = 10
    for seed in range(5):
        points, meta = space_filling_design(_box(2), n, "uniform", random_state=seed)
        unit = (points + 1.0) / 2.0
        np.testing.assert_allclose(np.sort(unit, axis=0), np.tile((2 * np.arange(1, n + 1) - 1) / (2 * n), (2, 1)).T)
        # The metadata value is CD^2, the squared centred discrepancy scipy returns.
        assert meta["centered_l2_discrepancy"] == pytest.approx(qmc.discrepancy(unit, method="CD"))


def test_uniform_design_discrepancy_is_lower_than_with_jitter() -> None:
    metas = [space_filling_design(_box(2), 10, "uniform", random_state=s)[1] for s in range(10)]
    cd2 = [m["centered_l2_discrepancy"] for m in metas]
    # The scrambled (jittered) version averaged 0.0046 over seeds; the centred one about 0.0030.
    assert np.mean(cd2) < 0.0035


def test_one_factor_uniform_design_is_the_midpoints() -> None:
    points, _ = space_filling_design(_box(1), 10, "uniform", random_state=3)
    np.testing.assert_allclose(np.sort(points[:, 0]), (2 * np.arange(1, 11) - 1) / 10 - 1)


def test_maximin_lhs_keeps_swapping_until_it_stops_improving() -> None:
    """A fixed 200 * k swap attempts left the 50-run, 5-factor design at a smallest distance of about 0.86."""
    distances = [space_filling_design(_box(5), 50, "maximin_lhs", random_state=s)[1]["min_distance"] for s in range(3)]
    assert min(distances) > 0.98


def test_maximin_lhs_stays_a_centred_latin_hypercube() -> None:
    points, meta = space_filling_design(_box(3), 15, "maximin_lhs", random_state=0)
    unit = np.sort((points + 1.0) / 2.0, axis=0)
    np.testing.assert_allclose(unit, np.tile((2 * np.arange(1, 16) - 1) / 30, (3, 1)).T)
    assert meta["min_distance"] == pytest.approx(pdist(points).min())


@pytest.mark.parametrize("n_runs", [1, 0, -3, 7.5])
@pytest.mark.parametrize("method", ["latin_hypercube", "maximin_lhs", "maximin", "sobol"])
def test_run_count_must_be_a_whole_number_of_at_least_two(method: str, n_runs: float) -> None:
    with pytest.raises(ValueError, match="at least 2"):
        space_filling_design(_box(2), n_runs, method, random_state=0)  # type: ignore[arg-type]


def test_whole_float_run_count_is_accepted() -> None:
    assert generate_design(_box(2), design_type="latin_hypercube", budget=7.0).n_runs == 7  # type: ignore[arg-type]
