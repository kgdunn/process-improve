"""Direct-dispatch tests for screening and optimal design generators.

These exercise the lower-level ``dispatch_*`` functions, including code
paths that the high-level ``generate_design`` API does not reach.
"""

from __future__ import annotations

import importlib.util

import pytest

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

    def test_constraints_recorded_as_not_enforced_in_meta(self) -> None:
        """Passing constraints records constraints_enforced=False on the result meta.

        Enforcement is not implemented, so the flag lets a caller detect that the
        returned design does not honour the constraints, rather than relying only
        on an easy-to-miss log line.
        """
        from process_improve.experiments.factor import Constraint

        constraints = [Constraint(expression="X1 + X2 <= 10")]
        _design, meta = dispatch_d_optimal(_continuous(2), budget=6, constraints=constraints)
        assert meta.get("constraints_enforced") is False

    @_skip_with_pyoptex
    def test_hard_to_change_ignored_flag_without_pyoptex(self) -> None:
        """Without pyoptex the split-plot request is dropped and recorded on the meta."""
        _design, meta = dispatch_d_optimal(_continuous(2), budget=6, hard_to_change=["X1"])
        assert meta.get("hard_to_change_ignored") == ["X1"]

    @_skip_with_pyoptex
    def test_hard_to_change_without_pyoptex_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("WARNING"):
            dispatch_d_optimal(_continuous(2), budget=6, hard_to_change=["X1"])
        assert any("pyoptex is not installed" in rec.message for rec in caplog.records)


@_skip_with_pyoptex
class TestOptimalRequiresPyoptex:
    """I-optimal and A-optimal raise a clear error without pyoptex."""

    def test_i_optimal_raises_import_error(self) -> None:
        with pytest.raises(ImportError, match="pyoptex"):
            dispatch_i_optimal(_continuous(3), budget=8)

    def test_a_optimal_raises_import_error(self) -> None:
        with pytest.raises(ImportError, match="pyoptex"):
            dispatch_a_optimal(_continuous(3), budget=8)


class TestOptimalMissingExtraMessage:
    """The not-installed error points at the ``[pyoptex]`` extra.

    Forcing the availability flag off lets this run regardless of whether
    pyoptex is present in the test environment, so the remediation message is
    always covered.
    """

    def test_i_optimal_error_names_the_extra(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from process_improve.experiments import designs_optimal

        monkeypatch.setattr(designs_optimal, "_PYOPTEX_AVAILABLE", False)
        with pytest.raises(ImportError, match=r"process-improve\[pyoptex\]"):
            dispatch_i_optimal(_continuous(3), budget=8)

    def test_a_optimal_error_names_the_extra(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from process_improve.experiments import designs_optimal

        monkeypatch.setattr(designs_optimal, "_PYOPTEX_AVAILABLE", False)
        with pytest.raises(ImportError, match=r"process-improve\[pyoptex\]"):
            dispatch_a_optimal(_continuous(3), budget=8)
