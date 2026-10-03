"""Mixture designs: review fixes for the default sizes, budgets and the vertex enumeration."""

from __future__ import annotations

import itertools
import logging

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import Constraint, Factor, generate_design
from process_improve.experiments.designs_constrained import ConstrainedOptions
from process_improve.experiments.designs_mixture import _simplex_lattice
from process_improve.experiments.designs_mixture_constrained import (
    _adjacency,
    _face_rank,
    constrained_mixture_design,
    extreme_vertices,
    mixture_candidates,
    mixture_inequalities,
    scheffe_matrix,
)


def _components(q: int, low: float = 0.0, high: float = 1.0) -> list[Factor]:
    return [Factor(name=f"x{i + 1}", type="mixture", low=low, high=high) for i in range(q)]


class TestFullSimplex:
    def test_special_cubic_with_a_small_budget_can_be_fitted(self) -> None:
        """6 runs for a 7-term special cubic gave the 6-run {3, 2} lattice, of rank 6."""
        factors = _components(3)
        result = generate_design(factors, design_type="mixture", budget=6, model_type="scheffe_special_cubic")
        x = result.design_actual[["x1", "x2", "x3"]].to_numpy(dtype=float)
        assert np.linalg.matrix_rank(scheffe_matrix(x, "scheffe_special_cubic")) == 7
        assert result.metadata["budget_requested"] == 6

    def test_budget_below_the_model_is_recorded(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            result = generate_design(_components(3), design_type="mixture", budget=2)
        assert result.n_runs == 6
        assert result.metadata["budget_requested"] == 2
        assert "raising the budget to 6" in caplog.text

    @pytest.mark.parametrize(("q", "runs"), [(8, 9), (10, 11), (13, 14)])
    def test_linear_model_gets_the_vertices_and_centroid(self, q: int, runs: int) -> None:
        """No longer the 2^q - 1 simplex centroid: 1023 runs for a 10-term linear model."""
        result = generate_design(_components(q), design_type="mixture", model_type="scheffe_linear")
        assert result.n_runs == runs
        assert result.metadata["method"] == "simplex_vertices_plus_centroid"

    def test_three_components_keep_the_simplex_centroid(self) -> None:
        result = generate_design(_components(3), design_type="mixture")
        assert result.metadata["method"] == "simplex_centroid"
        assert result.n_runs == 7

    def test_large_lattice_is_listed_directly(self) -> None:
        """The {13, 2} lattice has 91 points; (2 + 1)**13 tuples used to be refused."""
        lattice = _simplex_lattice(13, 2)
        assert lattice.shape == (91, 13)
        np.testing.assert_allclose(lattice.sum(axis=1), 1.0)
        assert len({tuple(r) for r in lattice}) == 91
        assert generate_design(_components(13), design_type="mixture", budget=100).n_runs <= 100


class TestVertexEnumeration:
    def test_vertices_do_not_depend_on_how_a_constraint_is_scaled(self) -> None:
        factors = _components(6)
        counts = []
        for scale in ["1", "0.003"]:
            cons = [Constraint(expression=f"{scale}*x{i + 1} >= {scale}*0.1") for i in range(5)]
            counts.append(len(extreme_vertices(*mixture_inequalities(factors, cons))))
        assert counts == [6, 6]

    def test_scaled_general_constraints_keep_every_vertex(self) -> None:
        """Five constraints with coefficients near 1e-5: 13 of the 15 vertices were found."""
        cons = [
            Constraint(expression=e)
            for e in [
                "9.5e-05*x1 + 1.4e-05*x2 + 9.5e-05*x3 + 3.1e-05*x4 + 4.2e-05*x5 <= 7.13851e-05",
                "4.1e-05*x1 + 5.5e-05*x2 + 3e-06*x3 + 7.5e-05*x4 + 5.4e-05*x5 <= 4.64866e-05",
                "7.9e-05*x1 + 3e-05*x2 + 4.5e-05*x3 + 1.3e-05*x4 + 4e-05*x5 <= 4.01728e-05",
                "2.6e-05*x1 + 7.5e-05*x2 + 2.8e-05*x3 + 4.9e-05*x4 + 9.8e-05*x5 <= 7.80829e-05",
                "7.2e-05*x1 + 5.4e-05*x2 + 2.8e-05*x3 + 1.6e-05*x4 + 9.7e-05*x5 <= 5.58034e-05",
            ]
        ]
        a_mat, b_vec = mixture_inequalities(_components(5), cons)
        assert len(extreme_vertices(a_mat * 1e-3, b_vec * 1e-3)) == 15

    def test_redundant_bounds_do_not_block_ten_components(self) -> None:
        """23 rows for 10 components needed 817,190 solves and were refused; the x_i <= 1 rows are redundant."""
        factors = _components(10)
        cons = [Constraint(expression=f"x1 + x{i} <= 0.6") for i in range(2, 5)]
        assert len(extreme_vertices(*mixture_inequalities(factors, cons))) == 40
        result = generate_design(factors, design_type="mixture", constraints=cons, model_type="scheffe_linear")
        assert result.metadata["n_vertices"] == 40

    def test_adjacency_matches_the_rank_test_on_every_pair(self) -> None:
        flare = [
            Factor(name="x1", type="mixture", low=0.4, high=0.6),
            Factor(name="x2", type="mixture", low=0.1, high=0.5),
            Factor(name="x3", type="mixture", low=0.1, high=0.5),
            Factor(name="x4", type="mixture", low=0.03, high=0.08),
        ]
        a_mat, b_vec = mixture_inequalities(flare, None)
        vertices = extreme_vertices(a_mat, b_vec)
        active = np.abs(vertices @ a_mat.T - b_vec) <= 1e-9
        brute = np.zeros((len(vertices), len(vertices)), dtype=bool)
        for i, j in itertools.combinations(range(len(vertices)), 2):
            brute[i, j] = brute[j, i] = _face_rank(a_mat, active[i] & active[j]) == 3
        np.testing.assert_array_equal(_adjacency(active, a_mat), brute)
        assert len(mixture_candidates(a_mat, b_vec)["edge_midpoint"]) == brute.sum() // 2


class TestDesignSize:
    def test_many_bounded_components_get_a_model_sized_design(self) -> None:
        """Six components in (0.05, 0.35): vertices + edges + centroid is 211 runs for 21 terms."""
        result = generate_design(_components(6, 0.05, 0.35), design_type="mixture", model_type="scheffe_quadratic")
        assert result.n_runs == 21 + 5
        assert result.metadata["n_runs_vertices_edges_centroid"] == 211

    def test_small_region_keeps_the_vertices_edges_and_centroid(self) -> None:
        flare = [
            Factor(name="x1", type="mixture", low=0.4, high=0.6),
            Factor(name="x2", type="mixture", low=0.1, high=0.5),
            Factor(name="x3", type="mixture", low=0.1, high=0.5),
            Factor(name="x4", type="mixture", low=0.03, high=0.08),
        ]
        result = generate_design(flare, design_type="mixture", model_type="scheffe_quadratic")
        assert result.n_runs == 21
        assert result.metadata["method"] == "extreme_vertices"

    def test_user_candidates_are_what_n_candidates_counts(self) -> None:
        factors = _components(3)
        blends = pd.DataFrame(_simplex_lattice(3, 4), columns=["x1", "x2", "x3"])
        design, meta = constrained_mixture_design(
            factors, None, [], ConstrainedOptions(model_type="scheffe_quadratic", candidates=blends), 0
        )
        assert len(design) == 6 + 3  # n_terms + 3 runs without a budget
        assert meta["n_candidates"] == meta["n_candidates_supplied"] == 15
