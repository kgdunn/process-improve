"""SEC-09: exception handling no longer swallows silently or leaks internals."""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import augment


class TestSafeEvaluateLogsAndNarrows:
    def _design(self) -> pd.DataFrame:
        return pd.DataFrame({"A": [-1, 1, -1, 1], "B": [-1, -1, 1, 1]})

    def test_expected_failure_returns_empty_and_logs(self, monkeypatch, caplog) -> None:
        def _boom(*_args, **_kwargs):
            raise ValueError("not applicable to this design")

        monkeypatch.setattr(augment, "evaluate_design", _boom)
        with caplog.at_level(logging.WARNING):
            result = augment._safe_evaluate(self._design(), generators=None)
        assert result == {}
        assert "Design evaluation skipped" in caplog.text

    def test_unexpected_error_propagates(self, monkeypatch) -> None:
        def _boom(*_args, **_kwargs):
            raise RuntimeError("genuine bug, should not be swallowed")

        monkeypatch.setattr(augment, "evaluate_design", _boom)
        with pytest.raises(RuntimeError, match="genuine bug"):
            augment._safe_evaluate(self._design(), generators=None)


class TestBreuschPaganFailureLogged:
    def test_failure_is_logged_and_not_raised(self, monkeypatch, caplog) -> None:
        import statsmodels.api as sm

        from process_improve.experiments.analysis import _run_residual_diagnostics

        def _raise(*_args, **_kwargs):
            raise ValueError("singular exog")

        # The diagnostic imports het_breuschpagan from this module at call time,
        # so patching the module attribute is picked up.
        monkeypatch.setattr("statsmodels.stats.diagnostic.het_breuschpagan", _raise)

        # Fit with pandas inputs so resid/fittedvalues are Series (the function
        # accesses ``.values`` on them).
        x = sm.add_constant(pd.DataFrame({"x": np.arange(1.0, 8.0)}))
        y = pd.Series([1.0, 2.0, 2.5, 4.0, 4.5, 6.0, 7.0], name="y")
        ols_result = sm.OLS(y, x).fit()

        with caplog.at_level(logging.WARNING):
            out = _run_residual_diagnostics(ols_result)

        assert "Breusch-Pagan" in caplog.text
        bp = out["residual_diagnostics"]["breusch_pagan"]
        assert bp["statistic"] is None
        assert bp["p_value"] is None


class TestMcpErrorSanitisation:
    def test_unexpected_error_is_generic_no_leak(self) -> None:
        pytest.importorskip("mcp")
        from process_improve.mcp_server import _serialise_tool_error

        internal_detail = "/home/app/internal/path/module.py line 42"
        out = _serialise_tool_error(RuntimeError(internal_detail), "fit_pca")
        payload = json.loads(out)
        assert payload == {"error": "internal error while executing tool", "tool": "fit_pca"}
        assert internal_detail not in out
