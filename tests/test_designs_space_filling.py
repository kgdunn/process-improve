"""Tests for space-filling designs."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import pdist
from scipy.stats import qmc

from process_improve.experiments import Constraint, DesignRegion, Factor, generate_design
from process_improve.experiments.designs_space_filling import (
    ANY_REGION,
    BOX_ONLY,
    _to_simplex,
    space_filling_design,
)

BOX = [Factor(name=n, low=0, high=10) for n in "ABC"]
HEAT_FACTORS = [Factor(name="T", low=100, high=150), Factor(name="D", low=20, high=60)]
HEAT = [Constraint(expression="3*T + 5*D <= 600")]
MIX = [Factor(name=n, type="mixture", low=0.1, high=0.6) for n in ("x1", "x2", "x3")]
CAP = [Constraint(expression="x1 + x2 <= 0.8")]


def _random_lhs(n: int, k: int, seed: int) -> np.ndarray:
    return 2 * qmc.LatinHypercube(d=k, rng=seed).random(n) - 1


def _is_latin_hypercube(points: np.ndarray) -> bool:
    """Each factor's range, cut into n equal slices, has exactly one run per slice."""
    n = len(points)
    slices = np.floor((points + 1) / 2 * n).clip(0, n - 1)
    return all(sorted(slices[:, j]) == list(range(n)) for j in range(points.shape[1]))


class TestBoxMethods:
    @pytest.mark.parametrize("method", BOX_ONLY)
    def test_latin_hypercube_property(self, method: str) -> None:
        points, _ = space_filling_design(BOX, 15, method, random_state=0)
        assert points.shape == (15, 3)
        assert _is_latin_hypercube(points)

    def test_maximin_lhs_spreads_runs_further_than_random_lhs(self) -> None:
        points, meta = space_filling_design(BOX, 15, "maximin_lhs", random_state=0)
        typical = np.median([pdist(_random_lhs(15, 3, s)).min() for s in range(30)])
        assert meta["min_distance"] == pytest.approx(pdist(points).min())
        assert meta["min_distance"] > 1.5 * typical

    def test_uniform_design_has_low_discrepancy(self) -> None:
        _, meta = space_filling_design(BOX, 15, "uniform", random_state=0)
        typical = np.median([qmc.discrepancy((_random_lhs(15, 3, s) + 1) / 2) for s in range(30)])
        assert meta["centered_l2_discrepancy"] < typical

    @pytest.mark.parametrize("method", ["sobol", "halton"])
    def test_sequences_on_the_box(self, method: str) -> None:
        points, meta = space_filling_design(BOX, 20, method, random_state=1)
        assert points.shape == (20, 3)
        assert np.abs(points).max() <= 1
        assert meta["centered_l2_discrepancy"] > 0

    def test_maximin_has_the_largest_smallest_distance(self) -> None:
        spread = {m: space_filling_design(BOX, 20, m, random_state=0)[1]["min_distance"] for m in ANY_REGION}
        assert max(spread, key=spread.get) == "maximin"


class TestConstrainedAndMixtureRegions:
    @pytest.mark.parametrize("method", ANY_REGION)
    def test_constrained_box(self, method: str) -> None:
        points, meta = space_filling_design(HEAT_FACTORS, 15, method, HEAT, random_state=0)
        assert DesignRegion(HEAT_FACTORS, HEAT).feasible(points).all()
        assert meta["constraints_enforced"] is True
        assert "centered_l2_discrepancy" not in meta  # defined on the cube only

    @pytest.mark.parametrize("method", ANY_REGION)
    def test_constrained_mixture(self, method: str) -> None:
        points, _ = space_filling_design(MIX, 12, method, CAP, random_state=0)
        np.testing.assert_allclose(points.sum(axis=1), 1.0)
        assert DesignRegion(MIX, CAP).feasible(points).all()

    def test_simplex_map_is_uniform(self) -> None:
        """Gaps between sorted uniform coordinates are uniform on the simplex: each has mean 1/q."""
        blends = _to_simplex(qmc.Sobol(d=2, rng=0).random_base2(12))
        np.testing.assert_allclose(blends.sum(axis=1), 1.0)
        np.testing.assert_allclose(blends.mean(axis=0), 1 / 3, atol=0.01)

    def test_reproducible(self) -> None:
        first, _ = space_filling_design(MIX, 10, "maximin", CAP, random_state=7)
        second, _ = space_filling_design(MIX, 10, "maximin", CAP, random_state=7)
        np.testing.assert_array_equal(first, second)


class TestGenerateDesign:
    def test_default_size_and_no_centre_points(self) -> None:
        result = generate_design(BOX, design_type="maximin_lhs", random_seed=0)
        assert result.n_runs == 30  # 10 runs per factor
        assert result.design_actual["A"].between(0, 10).all()

    def test_constrained_design_records_its_region(self) -> None:
        result = generate_design(HEAT_FACTORS, design_type="sobol", budget=12, constraints=HEAT)
        assert result.metadata["region"]["kind"] == "box"
        heat = 3 * result.design_actual["T"] + 5 * result.design_actual["D"]
        assert (heat <= 600 + 1e-9).all()

    def test_mixture_design_in_proportions(self) -> None:
        result = generate_design(MIX, design_type="maximin", budget=10, constraints=CAP)
        np.testing.assert_allclose(result.design_actual[["x1", "x2", "x3"]].sum(axis=1), 1.0)


class TestErrors:
    @pytest.mark.parametrize("method", BOX_ONLY)
    def test_box_only_method_in_a_constrained_region(self, method: str) -> None:
        with pytest.raises(ValueError, match="plain factor box only"):
            space_filling_design(HEAT_FACTORS, 10, method, HEAT)

    def test_box_only_method_for_a_mixture(self) -> None:
        with pytest.raises(ValueError, match="plain factor box only"):
            space_filling_design(MIX, 10, "latin_hypercube")

    def test_categorical_factor(self) -> None:
        factors = [*BOX, Factor(name="C", type="categorical", levels=["a", "b"])]
        with pytest.raises(ValueError, match="categorical level has no distance"):
            space_filling_design(factors, 10, "sobol")

    def test_unknown_method(self) -> None:
        with pytest.raises(ValueError, match="Unknown space-filling method"):
            space_filling_design(BOX, 10, "random")
