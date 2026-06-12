"""Direct-dispatch tests for screening and optimal design generators.

These exercise the lower-level ``dispatch_*`` functions, including code
paths that the high-level ``generate_design`` API does not reach.
"""

from __future__ import annotations

import importlib.util

import pytest

from process_improve.experiments.designs_optimal import (
    _run_point_exchange_fallback,
    dispatch_a_optimal,
    dispatch_d_optimal,
    dispatch_i_optimal,
)
from process_improve.experiments.designs_screening import (
    dispatch_fractional_factorial,
    dispatch_taguchi,
)
from process_improve.experiments.factor import Factor

_HAS_PYOPTEX = importlib.util.find_spec("pyoptex") is not None

_skip_with_pyoptex = pytest.mark.skipif(_HAS_PYOPTEX, reason="behaviour differs when pyoptex installed")


def _continuous(n: int) -> list[Factor]:
    return [Factor(name=f"X{i + 1}", low=0, high=10) for i in range(n)]


class TestFractionalFactorialDispatch:
    """Lower-level fractional factorial dispatch."""

    def test_default_resolution_when_unspecified(self) -> None:
        """No resolution and no generators picks the default min(k, 5) resolution."""
        coded, meta = dispatch_fractional_factorial(_continuous(6))
        assert coded.shape[1] == 6
        assert meta["resolution"] == 5

    def test_default_resolution_capped_at_five(self) -> None:
        coded, meta = dispatch_fractional_factorial(_continuous(7))
        assert coded.shape[1] == 7
        assert meta["resolution"] == 5

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


class TestDOptimalDispatch:
    """D-optimal dispatch fallback paths (no pyoptex required)."""

    @_skip_with_pyoptex
    def test_fallback_returns_design_and_metadata(self) -> None:
        design, meta = dispatch_d_optimal(_continuous(3), budget=8)
        assert design.shape[1] == 3
        assert meta["backend"] == "point_exchange_fallback"

    @_skip_with_pyoptex
    def test_default_budget(self) -> None:
        design, _meta = dispatch_d_optimal(_continuous(2))
        assert design.shape[0] > 0

    def test_constraints_emit_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Passing constraints logs an experimental-feature warning."""
        from process_improve.experiments.factor import Constraint

        constraints = [Constraint(expression="X1 + X2 <= 10")]
        with caplog.at_level("WARNING"):
            dispatch_d_optimal(_continuous(2), budget=6, constraints=constraints)
        assert any("Constraint enforcement" in rec.message for rec in caplog.records)

    @_skip_with_pyoptex
    def test_hard_to_change_without_pyoptex_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("WARNING"):
            dispatch_d_optimal(_continuous(2), budget=6, hard_to_change=["X1"])
        assert any("pyoptex is not installed" in rec.message for rec in caplog.records)


class TestPointExchangeFallback:
    """Direct tests of the no-pyoptex D-optimal fallback.

    These call ``_run_point_exchange_fallback`` directly, so they exercise the
    fallback regardless of whether pyoptex is installed (the function does not
    consult ``_PYOPTEX_AVAILABLE``).
    """

    def test_returns_design_and_d_optimality(self) -> None:
        """Fallback returns a k-column matrix and a finite d_optimality score."""
        import numpy as np

        k = 3
        design, meta = _run_point_exchange_fallback(_continuous(k), budget=8)
        assert design.shape[1] == k
        # point_exchange is a heuristic that grows from k rows up to the cap,
        # so the row count lands in [k, budget] rather than being exact.
        assert k <= design.shape[0] <= 8
        assert meta["backend"] == "point_exchange_fallback"
        assert np.isfinite(meta["d_optimality"])

    def test_n_points_floor_avoids_singular_request(self) -> None:
        """A budget below k+1 is raised so point_exchange does not raise.

        Without the ``max(n_points, k + 1)`` floor, a budget of 2 with 4
        factors would ask point_exchange for fewer points than columns and
        raise a ValueError. The floor keeps the request estimable.
        """
        design, _meta = _run_point_exchange_fallback(_continuous(4), budget=2)
        assert design.shape[1] == 4
        assert design.shape[0] >= 4  # point_exchange always starts with k rows

    def test_budget_capped_to_candidate_set(self) -> None:
        """A budget larger than the 3**k candidate set is capped to its size."""
        # 2 factors -> 3**2 = 9 candidate rows; an over-large budget cannot
        # select more than the 9 available candidates.
        design, _meta = _run_point_exchange_fallback(_continuous(2), budget=50)
        assert design.shape[1] == 2
        assert design.shape[0] <= 9

    def test_sec19_cap_rejects_too_many_factors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The 3**k candidate set is refused past the SEC-19 factor cap."""
        from process_improve.config import settings

        monkeypatch.setattr(settings, "max_factors_combinatorial", 3)
        with pytest.raises(ValueError, match="SEC-19 cap"):
            _run_point_exchange_fallback(_continuous(5), budget=8)


@_skip_with_pyoptex
class TestDOptimalFallbackModelType:
    """model_type is accepted on the fallback dispatch path (no pyoptex)."""

    @pytest.mark.parametrize("model_type", ["main_effects", "interactions", "quadratic"])
    def test_model_type_accepted(self, model_type: str) -> None:
        design, meta = dispatch_d_optimal(_continuous(3), budget=8, model_type=model_type)
        assert design.shape[1] == 3
        assert meta["backend"] == "point_exchange_fallback"


@_skip_with_pyoptex
class TestOptimalRequiresPyoptex:
    """I-optimal and A-optimal raise a clear error without pyoptex."""

    def test_i_optimal_raises_import_error(self) -> None:
        with pytest.raises(ImportError, match="pyoptex"):
            dispatch_i_optimal(_continuous(3), budget=8)

    def test_a_optimal_raises_import_error(self) -> None:
        with pytest.raises(ImportError, match="pyoptex"):
            dispatch_a_optimal(_continuous(3), budget=8)
