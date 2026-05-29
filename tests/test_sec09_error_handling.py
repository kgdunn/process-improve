"""SEC-09: exception handling no longer swallows silently or leaks internals."""

from __future__ import annotations

import json
import logging

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


class TestMcpErrorSanitisation:
    def test_unexpected_error_is_generic_no_leak(self) -> None:
        pytest.importorskip("mcp")
        from process_improve.mcp_server import _serialise_tool_error

        internal_detail = "/home/app/internal/path/module.py line 42"
        out = _serialise_tool_error(RuntimeError(internal_detail), "fit_pca")
        payload = json.loads(out)
        assert payload == {"error": "internal error while executing tool", "tool": "fit_pca"}
        assert internal_detail not in out
