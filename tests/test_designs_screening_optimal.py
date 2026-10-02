"""Direct-dispatch tests for screening and optimal design generators.

These exercise the lower-level ``dispatch_*`` functions, including code
paths that the high-level ``generate_design`` API does not reach.
"""

from __future__ import annotations

import numpy as np
import pytest

from process_improve.experiments import designs_optimal
from process_improve.experiments.designs_constrained import MAX_CANDIDATES
from process_improve.experiments.designs_optimal import (
    dispatch_a_optimal,
    dispatch_d_optimal,
    dispatch_i_optimal,
)
from process_improve.experiments.designs_screening import (
    dispatch_fractional_factorial,
    dispatch_taguchi,
)
from process_improve.experiments.factor import Factor


@pytest.fixture
def no_pyoptex(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the no-pyoptex code paths regardless of installation.

    The dispatch functions read the module-level ``_PYOPTEX_AVAILABLE`` flag
    at call time, so patching the attribute is sufficient to exercise the
    built-in candidate-exchange backend in any environment.
    """
    monkeypatch.setattr(designs_optimal, "_PYOPTEX_AVAILABLE", False)


def _continuous(n: int) -> list[Factor]:
    return [Factor(name=f"X{i + 1}", low=0, high=10) for i in range(n)]


class TestFractionalFactorialDispatch:
    """Lower-level fractional factorial dispatch."""

    def test_default_is_the_half_fraction(self) -> None:
        """No resolution and no generators gives the half fraction 2^(k-1), of resolution k."""
        coded, meta = dispatch_fractional_factorial(_continuous(6))
        assert coded.shape == (32, 6)
        assert meta["resolution"] == 6
        assert meta["generators_used"] == ["X6=X1X2X3X4X5"]

    def test_default_reports_the_resolution_reached(self) -> None:
        """Seven factors used to give 32 runs labelled resolution V, which were resolution IV."""
        coded, meta = dispatch_fractional_factorial(_continuous(7))
        assert coded.shape == (64, 7)
        assert meta["resolution"] == 7

    def test_explicit_resolution(self) -> None:
        coded, meta = dispatch_fractional_factorial(_continuous(5), resolution=3)
        assert meta["resolution"] == 3
        assert coded.shape[1] == 5


class TestTaguchiDispatch:
    """Lower-level Taguchi dispatch, including categorical factors."""

    def test_categorical_factors_use_level_indices(self) -> None:
        """Categorical factors contribute level-index columns to the OA."""
        factors = [
            Factor(name="A", type="categorical", levels=["lo", "hi"]),
            Factor(name="B", type="categorical", levels=["lo", "hi"]),
            Factor(name="C", type="categorical", levels=["lo", "hi"]),
        ]
        coded, meta = dispatch_taguchi(factors)
        assert coded.shape[1] == 3
        assert "orthogonal_array" in meta

    def test_no_orthogonal_array_raises(self) -> None:
        """A factor with more levels than any standard OA supports raises."""
        factors = [Factor(name="Big", type="categorical", levels=[str(i) for i in range(40)])]
        with pytest.raises(ValueError, match="No standard Taguchi orthogonal array"):
            dispatch_taguchi(factors)


@pytest.mark.usefixtures("no_pyoptex")
class TestDOptimalDispatch:
    """D-optimal dispatch without pyoptex: the built-in candidate exchange."""

    def test_fallback_returns_design_and_metadata(self) -> None:
        design, meta = dispatch_d_optimal(_continuous(3), budget=8)
        assert design.shape[1] == 3
        assert meta["backend"] == "candidate_exchange"
        assert meta["optimality_criterion"] == "d_optimal"
        assert isinstance(meta["log_det_information"], float)

    def test_default_budget(self) -> None:
        design, _meta = dispatch_d_optimal(_continuous(2))
        assert design.shape[0] > 0

    def test_constraints_are_enforced(self) -> None:
        """Constraints route to the constrained exchange and every run satisfies them.

        This held regardless of pyoptex: the constrained path is built in. It replaced
        a warning that constraints were noted but not enforced.
        """
        from process_improve.experiments.factor import Constraint

        constraints = [Constraint(expression="X1 + X2 <= 10")]
        design, meta = dispatch_d_optimal(_continuous(2), budget=6, constraints=constraints, random_state=0)
        assert meta["constraints_enforced"] is True
        assert meta["backend"] == "candidate_exchange"
        actual = 5.0 + 5.0 * design  # coded [-1, 1] -> actual [0, 10]
        assert (actual.sum(axis=1) <= 10 + 1e-9).all()

    def test_hard_to_change_ignored_flag_without_pyoptex(self) -> None:
        """Without pyoptex the split-plot request is dropped and recorded on the meta."""
        _design, meta = dispatch_d_optimal(_continuous(2), budget=6, hard_to_change=["X1"])
        assert meta.get("hard_to_change_ignored") == ["X1"]

    def test_hard_to_change_without_pyoptex_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("WARNING"):
            dispatch_d_optimal(_continuous(2), budget=6, hard_to_change=["X1"])
        assert any("pyoptex is not installed" in rec.getMessage() for rec in caplog.records)
        assert any("pip install pyoptex" in rec.getMessage() for rec in caplog.records)

    @pytest.mark.slow
    def test_candidate_grid_never_exceeds_the_cap(self) -> None:
        """No candidate set above MAX_CANDIDATES is allocated (SEC-19 / #268): coarser levels, or a sample."""
        _design, meta = dispatch_d_optimal(_continuous(16), budget=40, model_type="main_effects")
        assert meta["n_candidates"] <= MAX_CANDIDATES
        _design, meta = dispatch_d_optimal(_continuous(12), budget=100, model_type="quadratic")
        assert meta["grid_sampled"]
        assert meta["n_candidates"] <= MAX_CANDIDATES

    def test_budget_clamped_to_minimum_model_size(self) -> None:
        """A budget below the model size is raised to it, so the model is estimable.

        The floor used to be ``k + 1``, the size of a main-effects model, even
        for the default ``model_type="interactions"``: three factors give
        1 + 3 + 3 = 7 coefficients, not 4.
        """
        design, _meta = dispatch_d_optimal(_continuous(3), budget=2)
        assert design.shape[0] == 7
        assert np.linalg.matrix_rank(np.column_stack([np.ones(design.shape[0]), design])) == 4

    def test_budget_above_the_number_of_candidates_replicates_runs(self) -> None:
        """The exchange may pick a candidate twice, so a large budget gives replicates, not a short design."""
        design, _meta = dispatch_d_optimal(_continuous(2), budget=50)
        assert design.shape == (50, 2)
        assert len(np.unique(design, axis=0)) < 50

    @pytest.mark.parametrize(("model_type", "n_runs"), [("main_effects", 8), ("interactions", 8), ("quadratic", 10)])
    def test_model_type_sets_the_budget_floor_on_the_fallback_path(self, model_type: str, n_runs: int) -> None:
        """A quadratic model over three factors has 10 coefficients, so a budget of 8 is raised to 10."""
        design, meta = dispatch_d_optimal(_continuous(3), budget=8, model_type=model_type)
        assert design.shape == (n_runs, 3)
        assert meta["backend"] == "candidate_exchange"

    def test_the_criterion_follows_the_model(self) -> None:
        """The old point-exchange fallback scored a first-order model whatever was asked for.

        A quadratic model needs a third level on every factor to estimate its squares.
        """
        design, _meta = dispatch_d_optimal(_continuous(3), budget=14, model_type="quadratic", random_state=0)
        assert all(len(np.unique(design[:, j].round(9))) >= 3 for j in range(3))


@pytest.mark.usefixtures("no_pyoptex")
class TestOptimalWithoutPyoptex:
    """I-optimal and A-optimal designs no longer need pyoptex."""

    @pytest.mark.parametrize(
        ("dispatch", "name"), [(dispatch_i_optimal, "i_optimal"), (dispatch_a_optimal, "a_optimal")]
    )
    def test_candidate_exchange_builds_the_design(self, dispatch: object, name: str) -> None:
        design, meta = dispatch(_continuous(3), budget=10, model_type="quadratic", random_state=0)  # type: ignore[operator]
        assert design.shape == (10, 3)
        assert meta["backend"] == "candidate_exchange"
        assert meta["optimality_criterion"] == name
        assert meta["trace_criterion"] > 0
