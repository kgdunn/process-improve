"""Optimal and constrained designs: review fixes for the exchange engines and their metadata."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import Constraint, Factor
from process_improve.experiments.designs_constrained import ConstrainedOptions, constrained_optimal_design
from process_improve.experiments.designs_optimal import dispatch_d_optimal
from process_improve.experiments.optimal import point_exchange


def test_point_exchange_reaches_the_half_fraction_from_every_seed() -> None:
    """Four runs from the 3^3 grid, first-order model: the 2^(3-1) half fraction, det(X'X) = 4^4."""
    grid = pd.DataFrame(list(itertools.product([-1, 0, 1], repeat=3)), columns=list("abc"))
    values = [point_exchange(grid, 4, random_state=seed)[1] for seed in range(20)]
    # A single pass stopped at det(X'X) = 64 or below for about half of all seeds.
    np.testing.assert_allclose(values, -np.log(4.0**4))


def _box(k: int) -> list[Factor]:
    return [Factor(name=f"x{i}", low=-1, high=1) for i in range(k)]


class TestCandidateExchange:
    def test_restarts_reach_the_a_optimal_design_in_the_candidate_set(self) -> None:
        """Three factors, quadratic model, 14 runs: trace 2.30 is in the grid; greedy starts only found 2.39 or 2.42."""
        traces = [
            constrained_optimal_design(
                _box(3), 14, [], ConstrainedOptions(model_type="quadratic", criterion="a_optimal"), random_state=s
            )[1]["trace_criterion"]
            for s in range(8)
        ]
        assert min(traces) == pytest.approx(2.30, abs=1e-3)
        assert np.mean(np.isclose(traces, 2.30, atol=1e-3)) >= 0.5

    def test_d_optimal_design_without_squares_uses_the_corners(self) -> None:
        """det(X'X) is convex in each coordinate of such a model, so the 2-level grid holds an optimum."""
        design, meta = constrained_optimal_design(_box(5), 18, [], ConstrainedOptions(model_type="interactions"), 0)
        assert meta["n_levels"] == 2
        assert set(np.unique(design)) == {-1.0, 1.0}
        assert meta["log_det_information"] == pytest.approx(45.7477, abs=1e-3)

    def test_unknown_model_type_is_refused(self) -> None:
        with pytest.raises(ValueError, match="main_effects, interactions, quadratic"):
            constrained_optimal_design(_box(3), 10, [], ConstrainedOptions(model_type="quad"), 0)


class TestConstrainedCandidates:
    def test_vertex_where_two_constraints_meet_is_a_candidate(self) -> None:
        """The constraints a + 2b <= 1.1 and 2a + b <= 1.1 meet at (0.367, 0.367), off every grid line."""
        factors = [Factor(name="a", low=-1, high=1), Factor(name="b", low=-1, high=1)]
        cons = [Constraint(expression="a + 2*b <= 1.1"), Constraint(expression="2*a + b <= 1.1")]
        _, meta = constrained_optimal_design(factors, 4, cons, ConstrainedOptions(model_type="interactions"), 0)
        vertex = np.array([[1.1 / 3, 1.1 / 3]])
        with_vertex = pd.DataFrame(np.vstack([_grid(5), vertex]), columns=["a", "b"])
        _, best = constrained_optimal_design(
            factors, 4, cons, ConstrainedOptions(model_type="interactions", candidates=with_vertex), 0
        )
        # Without the vertex the design reached a D-efficiency of 0.94.
        assert meta["log_det_information"] == pytest.approx(best["log_det_information"], abs=1e-6)
        assert meta["n_vertex_points"] >= 1

    def test_fixed_runs_outside_the_region_are_counted(self) -> None:
        factors = [Factor(name="a", low=-1, high=1), Factor(name="b", low=-1, high=1)]
        _, meta = dispatch_d_optimal(
            factors,
            8,
            constraints=[Constraint(expression="a + b <= 0")],
            fixed_runs=pd.DataFrame({"a": [1.0, -1.0], "b": [1.0, -1.0]}),
        )
        assert meta["n_fixed_runs"] == 2
        assert meta["n_fixed_runs_outside_region"] == 1


def _grid(levels: int) -> np.ndarray:
    values = np.linspace(-1, 1, levels)
    return np.array(list(itertools.product(values, values)))
