"""Every design family against the property that defines it in the literature.

One class per design family, in the order of the cheat sheet that compares Python DoE
packages. Each test builds the design through the public ``generate_design`` and checks
the defining property directly on the run sheet, so a family that silently returned
some other design would fail here.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from scipy.stats import qmc

from process_improve.experiments import generate_design
from process_improve.experiments.designs_supersaturated import e_s2, e_s2_lower_bound
from process_improve.experiments.factor import Constraint, Factor


def _factors(k: int) -> list[Factor]:
    return [Factor(name=f"x{i}", low=0, high=10) for i in range(k)]


def _coded(result, k: int) -> np.ndarray:
    return result.design[[f"x{i}" for i in range(k)]].to_numpy(dtype=float)


def _corners(x: np.ndarray) -> np.ndarray:
    return x[~np.all(x == 0, axis=1)]


class TestSupersaturated:
    """Lin (1993): a half fraction of a Hadamard matrix, balanced columns, E(s^2) near its bound."""

    @pytest.mark.parametrize(("k", "n"), [(10, 6), (22, 12), (30, 16)])
    def test_balanced_and_e_s2_near_the_bound(self, k: int, n: int) -> None:
        x = _coded(generate_design(_factors(k), "supersaturated", budget=n), k)
        assert x.shape == (n, k)
        assert np.all(x.sum(axis=0) == 0)
        assert e_s2(x) <= 1.25 * e_s2_lower_bound(n, k)

    def test_twelve_runs_reach_the_bound(self) -> None:
        """For N = 12 and k = 22 Lin's design attains the Nguyen / Tang-Wu bound exactly."""
        x = _coded(generate_design(_factors(22), "supersaturated", budget=12), 22)
        assert math.isclose(e_s2(x), e_s2_lower_bound(12, 22))


class TestPlackettBurman:
    """Plackett and Burman (1946): N a multiple of 4, columns orthogonal and balanced."""

    @pytest.mark.parametrize("k", [3, 7, 11, 19, 23, 27, 35, 43])
    def test_columns_orthogonal(self, k: int) -> None:
        x = _corners(_coded(generate_design(_factors(k), "plackett_burman", n_center_points=0), k))
        n = len(x)
        assert n % 4 == 0
        assert n == 4 * (k // 4 + 1)
        np.testing.assert_array_equal(x.T @ x, n * np.eye(k))


class TestFractionalFactorial:
    """Minimum-aberration fractions (Chen, Sun and Wu 1993): resolution and word-length pattern."""

    @pytest.mark.parametrize(("k", "resolution", "runs"), [(5, 5, 16), (6, 4, 16), (7, 4, 16), (8, 4, 16), (9, 3, 16)])
    def test_resolution_and_run_count(self, k: int, resolution: int, runs: int) -> None:
        result = generate_design(_factors(k), "fractional_factorial", resolution=resolution, n_center_points=0)
        x = _coded(result, k)
        assert len(x) == runs
        assert result.resolution >= resolution
        # The resolution is the length of the shortest word whose column product is constant.
        shortest = min(
            len(w)
            for r in range(1, k + 1)
            for w in itertools.combinations(range(k), r)
            if abs(np.prod(x[:, list(w)], axis=1).sum()) == len(x)
        )
        assert shortest == result.resolution


class TestFullFactorial:
    """Every combination of levels exactly once, two-level and mixed-level."""

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_two_level(self, k: int) -> None:
        x = _corners(_coded(generate_design(_factors(k), "full_factorial", n_center_points=0), k))
        assert len({tuple(r) for r in x}) == len(x) == 2**k

    def test_mixed_level(self) -> None:
        factors = [
            Factor(name="a", low=0, high=1),
            Factor(name="b", type="categorical", levels=["p", "q", "r"]),
            Factor(name="c", low=0, high=4, levels=[0, 1, 2, 3, 4]),
        ]
        actual = generate_design(factors, "full_factorial", n_center_points=0).design_actual
        assert len({tuple(r) for r in actual[["a", "b", "c"]].to_numpy().tolist()}) == len(actual) == 30


class TestDefinitiveScreening:
    """Jones and Nachtsheim (2011): foldover of a conference matrix plus a centre run."""

    @pytest.mark.parametrize("k", [4, 5, 9, 10, 14, 16, 21, 26, 27])
    def test_defining_properties(self, k: int) -> None:
        x = _coded(generate_design(_factors(k), "dsd"), k)
        assert np.all(x.sum(axis=0) == 0)  # foldover
        assert np.all(np.isin(x, (-1, 0, 1)))
        second = np.column_stack([x[:, i] * x[:, j] for i in range(k) for j in range(i, k)])
        assert np.abs(x.T @ second).max() == 0  # main effects clear of all second-order terms
        gram = x.T @ x
        np.testing.assert_array_equal(gram, np.diag(np.diag(gram)))  # main effects orthogonal


class TestOMARS:
    """Nunez Ares and Goos (2020): orthogonal main effects, clear of second-order effects."""

    @pytest.mark.parametrize("k", [3, 4, 6])
    def test_omars_property(self, k: int) -> None:
        x = _coded(generate_design(_factors(k), "omars"), k)
        gram = x.T @ x
        np.testing.assert_array_equal(gram, np.diag(np.diag(gram)))
        second = np.column_stack([x[:, i] * x[:, j] for i in range(k) for j in range(i, k)])
        assert np.abs(x.T @ second).max() == 0


class TestTaguchi:
    """Orthogonal arrays of strength 2: every pair of columns shows every level pair equally often."""

    @pytest.mark.parametrize("k", [3, 5, 7, 11, 15])
    def test_strength_two(self, k: int) -> None:
        x = _coded(generate_design(_factors(k), "taguchi"), k)
        for i, j in itertools.combinations(range(k), 2):
            _, counts = np.unique(x[:, [i, j]], axis=0, return_counts=True)
            assert len(counts) == 4
            assert counts.min() == counts.max()


class TestOptimal:
    """Each criterion beats the others on its own measure, on a constrained quadratic problem."""

    @pytest.fixture(scope="class")
    def designs(self) -> dict:
        factors = [Factor(name="T", low=100, high=150), Factor(name="D", low=20, high=60)]
        constraints = [Constraint(expression="3*T + 5*D <= 600")]
        return {
            c: generate_design(factors, c, budget=10, model_type="quadratic", constraints=constraints)
            for c in ("d_optimal", "a_optimal", "e_optimal", "i_optimal")
        }

    def test_runs_are_feasible(self, designs: dict) -> None:
        for result in designs.values():
            actual = result.design_actual
            assert np.all(3 * actual["T"] + 5 * actual["D"] <= 600 + 1e-9)

    @staticmethod
    def _information(result) -> np.ndarray:
        x = result.design[["T", "D"]].to_numpy(dtype=float)
        model = np.column_stack([np.ones(len(x)), x, x[:, 0] * x[:, 1], x**2])
        return model.T @ model

    def test_each_criterion_wins_on_its_own_measure(self, designs: dict) -> None:
        info = {c: self._information(r) for c, r in designs.items()}
        log_det = {c: np.linalg.slogdet(m)[1] for c, m in info.items()}
        trace_inv = {c: np.trace(np.linalg.inv(m)) for c, m in info.items()}
        lam_min = {c: np.linalg.eigvalsh(m)[0] for c, m in info.items()}
        assert max(log_det, key=log_det.get) == "d_optimal"
        assert min(trace_inv, key=trace_inv.get) == "a_optimal"
        assert max(lam_min, key=lam_min.get) == "e_optimal"


class TestBoxBehnken:
    """Box and Behnken (1960): published run counts, three levels, no corner runs."""

    @pytest.mark.parametrize(("k", "runs"), [(3, 12), (4, 24), (5, 40), (6, 48), (7, 56)])
    def test_published(self, k: int, runs: int) -> None:
        x = _corners(_coded(generate_design(_factors(k), "box_behnken", n_center_points=0), k))
        assert len(x) == runs
        assert not np.any(np.all(np.abs(x) == 1, axis=1))


class TestCentralComposite:
    """Box and Wilson (1951): 2^k cube, 2k axial runs at +/-alpha; rotatable alpha = (2^k)^(1/4)."""

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_rotatable(self, k: int) -> None:
        result = generate_design(_factors(k), "ccd", alpha="rotatable", n_center_points=0)
        x = _coded(result, k)
        alpha = (2**k) ** 0.25
        assert len(x) == 2**k + 2 * k
        assert math.isclose(np.abs(x).max(), alpha)
        # Rotatability: fourth moments [iiii] = 3 [iijj] (Box and Hunter 1957).
        assert math.isclose((x[:, 0] ** 4).sum(), 3 * (x[:, 0] ** 2 * x[:, 1] ** 2).sum())


class TestMixture:
    """Simplex lattice and centroid sizes (Scheffe 1958, 1963); extreme vertices inside the bounds."""

    @pytest.mark.parametrize("q", [3, 4, 5])
    def test_simplex_centroid(self, q: int) -> None:
        factors = [Factor(name=f"x{i}", type="mixture") for i in range(q)]
        actual = generate_design(factors, "mixture", model_type="special_cubic").design_actual
        x = actual[[f"x{i}" for i in range(q)]].to_numpy(dtype=float)
        np.testing.assert_allclose(x.sum(axis=1), 1.0)
        assert len({tuple(np.round(r, 9)) for r in x}) == 2**q - 1

    def test_extreme_vertices_respect_the_bounds(self) -> None:
        factors = [
            Factor(name="a", type="mixture", low=0.1, high=0.6),
            Factor(name="b", type="mixture", low=0.2, high=0.7),
            Factor(name="c", type="mixture", low=0.1, high=0.5),
        ]
        actual = generate_design(factors, "mixture").design_actual
        x = actual[["a", "b", "c"]].to_numpy(dtype=float)
        np.testing.assert_allclose(x.sum(axis=1), 1.0)
        assert np.all(x >= np.array([0.1, 0.2, 0.1]) - 1e-9)
        assert np.all(x <= np.array([0.6, 0.7, 0.5]) + 1e-9)


class TestSpaceFilling:
    """Latin hypercube stratification, maximin distance, discrepancy and Sobol balance."""

    def test_latin_hypercube_one_run_per_slice(self) -> None:
        x = _coded(generate_design(_factors(4), "latin_hypercube", budget=20, random_state=3), 4)
        for j in range(4):
            slices = np.floor((x[:, j] + 1) / 2 * 20).astype(int)
            assert sorted(slices) == list(range(20))

    def test_maximin_spreads_further_than_a_plain_hypercube(self) -> None:
        lhs = _coded(generate_design(_factors(3), "latin_hypercube", budget=20, random_state=3), 3)
        maximin = _coded(generate_design(_factors(3), "maximin", budget=20, random_state=3), 3)

        def min_distance(p: np.ndarray) -> float:
            return min(np.linalg.norm(a - b) for a, b in itertools.combinations(p, 2))

        assert min_distance(maximin) > min_distance(lhs)

    def test_uniform_has_lower_discrepancy_than_a_plain_hypercube(self) -> None:
        lhs = _coded(generate_design(_factors(3), "latin_hypercube", budget=20, random_state=3), 3)
        uniform = _coded(generate_design(_factors(3), "uniform", budget=20, random_state=3), 3)
        assert qmc.discrepancy((uniform + 1) / 2) < qmc.discrepancy((lhs + 1) / 2)

    def test_sobol_points_are_balanced(self) -> None:
        """The first 2^m Sobol points put 2^(m-1) points in each half of every factor's range."""
        x = _coded(generate_design(_factors(3), "sobol", budget=16, random_state=3), 3)
        assert np.all((x > 0).sum(axis=0) == 8)


@pytest.mark.slow
@pytest.mark.parametrize(("k", "budget"), [(4, 13), (4, 17)])
def test_omars_budget_below_the_full_model_size_still_gives_omars(k: int, budget: int) -> None:
    """A foldover estimates the full second-order model only from k^2 + k + 1 runs (21 for 4 factors).

    A smaller budget used to be refused as having no error degrees of freedom; it now
    gets an OMARS design sized for main effects plus pure quadratics.
    """
    result = generate_design(_factors(k), "omars", budget=budget)
    x = _coded(result, k)
    assert result.n_runs == budget
    assert result.metadata["model"] == "main_quadratic"
    second = np.column_stack([x[:, i] * x[:, j] for i in range(k) for j in range(i, k)])
    assert np.abs(x.T @ second).max() < 1e-9


class TestDSDFakeFactors:
    """Jones and Nachtsheim (2017): a budget above the minimal DSD adds fake factors for error degrees of freedom."""

    @pytest.mark.parametrize(("budget", "runs", "fake"), [(None, 13, 0), (16, 13, 0), (17, 17, 2), (24, 21, 4)])
    def test_budget_adds_fake_factors(self, budget: int | None, runs: int, fake: int) -> None:
        result = generate_design(_factors(6), "dsd", budget=budget)
        assert result.n_runs == runs
        assert result.metadata["fake_factors"] == fake
        x = _coded(result, 6)
        second = np.column_stack([x[:, i] * x[:, j] for i in range(6) for j in range(i, 6)])
        assert np.abs(x.T @ second).max() == 0

    def test_budget_below_the_minimal_design_raises(self) -> None:
        with pytest.raises(ValueError, match="needs at least 13 runs"):
            generate_design(_factors(6), "dsd", budget=11)
