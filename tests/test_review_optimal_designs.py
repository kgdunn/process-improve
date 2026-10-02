"""Optimal and constrained designs: review fixes for the exchange engines and their metadata."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import Constraint, Factor, designs_optimal
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


class TestDispatchOptimal:
    def test_unknown_hard_to_change_name_is_refused(self) -> None:
        with pytest.raises(ValueError, match="not factors of the design"):
            dispatch_d_optimal(_box(3), 12, hard_to_change=["Zzz"])

    def test_budget_below_the_model_is_recorded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(designs_optimal, "_PYOPTEX_AVAILABLE", False)
        design, meta = dispatch_d_optimal(_box(3), 5, model_type="interactions")
        assert len(design) == 7  # intercept, 3 main effects, 3 interactions
        assert meta["budget_requested"] == 5

    def test_default_budget_leaves_room_beyond_the_fixed_runs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Eight fixed runs and no budget: the default 2k + 1 = 7 used to be refused as too small."""
        monkeypatch.setattr(designs_optimal, "_PYOPTEX_AVAILABLE", False)
        fixed = pd.DataFrame(list(itertools.product([-1.0, 1.0], repeat=3)), columns=["x0", "x1", "x2"])
        design, meta = dispatch_d_optimal(_box(3), None, model_type="interactions", fixed_runs=fixed)
        assert len(design) > len(fixed)
        assert meta["n_fixed_runs"] == 8
        assert "budget_requested" not in meta


@pytest.fixture
def pyoptex() -> None:
    pytest.importorskip("pyoptex")
    if not designs_optimal._PYOPTEX_AVAILABLE:
        pytest.skip("pyoptex is not importable")


@pytest.mark.usefixtures("pyoptex")
class TestPyoptexBackend:
    def test_quadratic_model_offers_interior_levels(self) -> None:
        """With only {-1, 0, 1} the 6-run, 2-factor A-optimal trace was 5.0; 4.185 with the levels +/-0.5."""
        design, meta = designs_optimal._run_pyoptex(
            _box(2), "a_optimal", 6, designs_optimal._PyoptexOptions(model_type="quadratic", random_state=0)
        )
        assert meta["trace_criterion"] == pytest.approx(4.185, abs=1e-3)
        assert {0.5, -0.5} & set(np.round(np.asarray(design, dtype=float).ravel(), 6))

    def test_criterion_is_reported_on_the_exchange_scale(self) -> None:
        """For D, ``metric_value`` is det(X'X)^(1/p); ``log_det_information`` matches the exchange's."""
        _, meta = designs_optimal._run_pyoptex(
            _box(2), "d_optimal", 6, designs_optimal._PyoptexOptions(model_type="quadratic", random_state=1)
        )
        _, exchange = constrained_optimal_design(_box(2), 6, [], ConstrainedOptions(model_type="quadratic"), 1)
        assert meta["log_det_information"] == pytest.approx(exchange["log_det_information"], abs=1e-6)
        assert meta["metric_value"] == pytest.approx(np.exp(meta["log_det_information"] / 6))

    def test_split_plot_has_enough_whole_plots_and_records_them(self) -> None:
        """Two hard-to-change factors, quadratic, 16 runs: 5 whole plots cannot estimate their 6 terms."""
        design, meta = designs_optimal._dispatch_optimal(
            "d_optimal", designs_optimal._OptimalRequest(_box(4), 16, ["x0", "x1"], None, "quadratic", None, 0)
        )
        assert meta["n_whole_plots"] >= 7
        assert len(meta["whole_plot"]) == 16
        assert meta["whole_plot_variance_ratio"] == 0.5
        frame = pd.DataFrame(np.asarray(design, dtype=float)[:, :2], columns=["x0", "x1"])
        frame["plot"] = meta["whole_plot"]
        spread = frame.groupby("plot")[["x0", "x1"]].agg(lambda v: v.max() - v.min())
        assert (spread == 0).all().all()  # hard-to-change factors are constant within each whole plot

    def test_fixed_runs_short_of_rank_raise_the_budget(self) -> None:
        """Six identical fixed runs and a budget of 8 failed inside pyoptex with 'rank collinearity'."""
        factors = [Factor(name="a", low=10, high=20), Factor(name="b", low=0, high=1)]
        fixed = pd.DataFrame({"a": [0.0] * 6, "b": [0.0] * 6})
        design, meta = designs_optimal._run_pyoptex(
            factors, "d_optimal", 8, designs_optimal._PyoptexOptions(model_type="interactions", fixed_runs=fixed)
        )
        assert len(design) == 6 + 3
        assert meta["budget_requested"] == 8
