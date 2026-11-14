"""Candidates of a region on many factors whose grid misses it: vertices found by linear programming."""

from __future__ import annotations

import numpy as np
import pytest

from process_improve.experiments import DesignRegion, Factor, generate_design
from process_improve.experiments import designs_constrained as engine
from process_improve.experiments.designs_constrained import (
    _sampled_vertices,
    build_candidates,
    model_matrix,
    region_with_rules,
)
from process_improve.experiments.factor import as_constraints


def _buckets(n_buckets: int, per_bucket: int) -> tuple[list[Factor], list[str], list[list[str]]]:
    """Return ingredients in buckets: each bucket's total from 5 to 20, and the total of them all at most 60%."""
    names = [[f"B{b + 1}_{i + 1}" for i in range(per_bucket)] for b in range(n_buckets)]
    flat = [name for bucket in names for name in bucket]
    factors = [Factor(name=name, low=0, high=20, units="%") for name in flat]
    limits = [f"5 <= {' + '.join(bucket)} <= 20" for bucket in names]
    limits.append(f"{' + '.join(flat)} <= {12 * n_buckets}")
    return factors, limits, names


def _rows(k: int, total: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the unit-length rows of ``x1 + ... + xk <= total`` and ``x1 + x2 >= -1``, in coded units."""
    lower = np.zeros(k)
    lower[:2] = -1.0
    return (
        np.vstack([np.ones(k) / np.sqrt(k), lower / np.sqrt(2.0)]),
        np.array([total / np.sqrt(k), 1.0 / np.sqrt(2.0)]),
    )


@pytest.fixture
def few_programmes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Solve 128 linear programmes, not 1024: enough for these regions, and fast."""
    monkeypatch.setattr(engine, "_SAMPLED_VERTICES", 128)


@pytest.mark.usefixtures("few_programmes")
class TestSampledVertices:
    def test_every_point_is_inside_the_region(self) -> None:
        a_mat, b_vec = _rows(6, 1.0)
        points = _sampled_vertices(a_mat, b_vec)
        assert len(points) > 6
        assert np.all(np.abs(points) <= 1.0)
        assert np.all(points @ a_mat.T <= b_vec + 1e-9)

    def test_the_points_do_not_all_lie_on_one_face(self) -> None:
        # Directions drawn alike in every coordinate end where the total is at its limit, and there the
        # intercept cannot be told from the sum of the main effects.
        a_mat, b_vec = _rows(12, -6.0)
        points = _sampled_vertices(a_mat, b_vec)
        assert np.linalg.matrix_rank(np.column_stack([np.ones(len(points)), points])) == 13
        assert np.ptp(points.sum(axis=1)) > 1.0

    def test_points_between_the_vertices_are_included(self) -> None:
        a_mat, b_vec = _rows(4, 1.0)
        points = _sampled_vertices(a_mat, b_vec)
        slack = np.minimum(1.0 - np.abs(points).max(axis=1), (b_vec - points @ a_mat.T).min(axis=1))
        assert np.any(slack > 1e-6)  # a point on no boundary: the mean of the vertices, and some midpoints

    def test_no_points_are_repeated(self) -> None:
        points = _sampled_vertices(*_rows(3, 1.0))  # few vertices, found many times over
        assert len(np.unique(points.round(10), axis=0)) == len(points)

    def test_an_empty_region_gives_no_points(self) -> None:
        a_mat = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        points = _sampled_vertices(a_mat, np.array([-0.5, -0.5]))  # x1 <= -0.5 and x1 >= 0.5
        assert points.shape == (0, 3)

    def test_the_points_do_not_depend_on_a_random_state(self) -> None:
        first, second = _sampled_vertices(*_rows(6, 1.0)), _sampled_vertices(*_rows(6, 1.0))
        np.testing.assert_array_equal(first, second)


@pytest.mark.usefixtures("few_programmes")
class TestCandidates:
    def test_a_grid_that_misses_the_region_is_filled_from_the_vertices(self) -> None:
        factors, limits, _ = _buckets(5, 3)
        region = region_with_rules(factors, as_constraints(limits) or [])
        coded, cats, counts = build_candidates(region, model_type="main_effects")
        assert counts["vertices_sampled"] is True
        assert counts["n_candidates"] == len(coded) > 16
        assert np.all(region.slack(coded, cats) <= 1e-9)
        assert np.linalg.matrix_rank(model_matrix(region, coded, cats, "main_effects")) == 16

    def test_a_region_the_grid_covers_keeps_its_candidates(self) -> None:
        # 14 factors have 16384 corners, more than are listed, yet half of the grid is inside the region.
        factors = [Factor(name=f"X{i}", low=0, high=10) for i in range(14)]
        region = region_with_rules(factors, as_constraints([" + ".join(f.name for f in factors) + " <= 70"]) or [])
        _coded, _cats, counts = build_candidates(region, model_type="main_effects")
        assert counts["n_vertex_points"] == 0
        assert "vertices_sampled" not in counts

    def test_a_region_whose_vertices_are_listed_is_not_sampled(self) -> None:
        factors = [Factor(name="A", low=0, high=10), Factor(name="B", low=0, high=10)]
        region = region_with_rules(factors, as_constraints(["A + B <= 0.5"]) or [])  # a corner the grid misses
        _coded, _cats, counts = build_candidates(region, n_levels=2, model_type="main_effects")
        assert counts["n_vertex_points"] == 3
        assert "vertices_sampled" not in counts

    def test_limits_that_conflict_still_leave_no_candidate(self) -> None:
        factors, limits, names = _buckets(5, 3)
        limits.append(f"{' + '.join(names[0])} >= 25")  # above the bucket's own limit of 20
        with pytest.raises(ValueError, match="No candidate point satisfies all the constraints"):
            generate_design(factors, budget=20, model_type="main_effects", constraints=limits, random_state=1)

    def test_the_region_can_be_sampled_from_those_candidates(self) -> None:
        factors, limits, _ = _buckets(5, 3)
        region = DesignRegion(factors, limits)
        points = region.sample(200, np.random.default_rng(3))
        assert points.shape == (200, 15)
        assert region.feasible(points).all()


@pytest.mark.slow
class TestDesigns:
    def test_group_totals_on_many_factors_get_a_design(self) -> None:
        factors, limits, names = _buckets(5, 3)
        result = generate_design(factors, budget=30, model_type="main_effects", constraints=limits, random_state=1)
        assert result.metadata["vertices_sampled"] is True
        flat = [name for bucket in names for name in bucket]
        amounts = result.design_actual[flat].to_numpy(dtype=float)
        totals = amounts.reshape(30, 5, 3).sum(axis=2)
        assert totals.min() >= 5 - 1e-6
        assert totals.max() <= 20 + 1e-6
        assert amounts.sum(axis=1).max() <= 60 + 1e-6
        assert np.linalg.matrix_rank(np.column_stack([np.ones(30), amounts])) == 16
        again = generate_design(factors, budget=30, model_type="main_effects", constraints=limits, random_state=1)
        np.testing.assert_array_equal(again.design_actual[flat].to_numpy(dtype=float), amounts)

    def test_a_quadratic_model_is_supported(self) -> None:
        factors, limits, names = _buckets(3, 2)  # 6 factors: forced to sample by a small limit on listing
        flat = [name for bucket in names for name in bucket]
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(engine, "_MAX_VERTEX_CANDIDATES", 4)
            patch.setattr(engine, "_FEW_CANDIDATES_PER_COEFFICIENT", 10**6)
            result = generate_design(factors, budget=34, model_type="quadratic", constraints=limits, random_state=1)
        assert result.metadata["vertices_sampled"] is True
        amounts = result.design_actual[flat].to_numpy(dtype=float)
        assert DesignRegion(factors, limits).feasible((amounts - 10.0) / 10.0, tol=1e-6).all()
