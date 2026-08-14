"""Regression tests for the 2026-08 repo-wide correctness audit: infrastructure.

Each test pins a specific defect found and fixed in the audit: the Settings
cache, strict boolean env parsing, JSON gaps in clean(), discover_tools
error scoping, and thread-safety of the safe tool-execution path.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from process_improve.config import Settings
from process_improve.tool_spec import _import_tool_module, clean


class TestSettings:
    def test_bool_env_rejects_typos(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A typo in the security-relevant safe-mode knob previously failed
        OPEN (silently returned False).
        """
        monkeypatch.setenv("PROCESS_IMPROVE_MCP_SAFE_MODE", "treu")
        with pytest.raises(ValueError, match="not a valid boolean"):
            _ = Settings().mcp_safe_mode

    def test_bool_env_accepts_explicit_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PROCESS_IMPROVE_MCP_SAFE_MODE", "off")
        assert Settings().mcp_safe_mode is False

    def test_value_is_really_cached_on_first_access(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Setdefault evaluated the env read on EVERY access, so a knob that
        had been served successfully raised later if the env var went bad.
        """
        monkeypatch.setenv("PROCESS_IMPROVE_MAX_CELLS", "123")
        settings = Settings()
        assert settings.max_cells == 123
        monkeypatch.setenv("PROCESS_IMPROVE_MAX_CELLS", "not-a-number")
        # Still served from the cache; no ValueError, no re-read.
        assert settings.max_cells == 123

    def test_non_positive_limits_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PROCESS_IMPROVE_TOOL_TIMEOUT", "-5")
        with pytest.raises(ValueError, match="must be positive"):
            _ = Settings().tool_timeout
        monkeypatch.setenv("PROCESS_IMPROVE_MAX_MEMORY_MB", "0")
        with pytest.raises(ValueError, match="must be positive"):
            _ = Settings().max_memory_mb


class TestClean:
    def test_numpy_bool_is_serialisable(self) -> None:
        result = clean({"significant": np.bool_(True), "flags": np.array([1, 2]) > 1})
        json.dumps(result)
        assert result["significant"] is True
        assert result["flags"] == [False, True]

    def test_numpy_scalar_dict_keys_are_unwrapped(self) -> None:
        """A dict keyed by np.int64 (e.g. pandas groupby labels) previously
        survived clean() and made json.dumps raise TypeError.
        """
        result = clean({np.int64(3): "a", np.str_("k"): np.float64(1.5)})
        json.dumps(result)
        assert result == {3: "a", "k": 1.5}

    def test_sets_and_generic_scalars(self) -> None:
        result = clean({"s": {np.int64(1), np.int64(2)}, "c": np.complex128(1 + 0j)})
        json.dumps({"s": result["s"]})
        assert sorted(result["s"]) == [1, 2]


class TestDiscoverTools:
    def test_missing_first_party_module_raises(self) -> None:
        """A typo'd or renamed process_improve module is a real bug and must
        propagate; previously it was logged as a 'missing dependency' and the
        whole tool category silently vanished.
        """
        with pytest.raises(ModuleNotFoundError):
            _import_tool_module("process_improve.no_such_tools_module")

    def test_missing_third_party_module_is_tolerated(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("WARNING"):
            _import_tool_module("no_such_third_party_pkg_xyz.tools")
        assert any("not loaded" in record.message for record in caplog.records)


class TestSafeExecutionConcurrency:
    @pytest.mark.slow
    def test_concurrent_safe_calls_do_not_kill_each_other(self) -> None:
        """Two threads previously shared (and tore down) one module pool: one
        thread's teardown SIGKILLed the worker running the other thread's
        task, surfacing as a bogus 'likely exceeded memory limit' error.
        """
        from process_improve.tool_safety import safe_execute_tool_call
        from process_improve.tool_spec import discover_tools

        discover_tools()

        def one_call(_: int) -> dict:
            return safe_execute_tool_call(
                "confidence_interval",
                {"values": [10.0, 11.0, 12.0, 10.5, 9.5, 10.2]},
                timeout=60.0,
            )

        with ThreadPoolExecutor(max_workers=4) as tp:
            results = list(tp.map(one_call, range(8)))
        for result in results:
            assert "lower" in result
            assert "upper" in result
