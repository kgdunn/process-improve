"""SEC-11: discover_tools no longer silently swallows every ImportError.

A missing optional dependency (ModuleNotFoundError) is tolerated but logged; a
genuine bug inside a tools module (any other ImportError, e.g. a bad name) now
propagates instead of making the whole tool category vanish silently.
"""

from __future__ import annotations

import importlib
import logging

import pytest

import process_improve.tool_spec as ts

_TARGET = "process_improve.univariate.tools"


class TestDiscoverToolsImportHandling:
    def test_missing_dependency_is_logged_and_tolerated(self, monkeypatch, caplog) -> None:
        real_import = importlib.import_module

        def fake(name: str, *args, **kwargs):
            if name == _TARGET:
                raise ModuleNotFoundError("No module named 'some_optional_dep'", name="some_optional_dep")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(ts, "_discovery_done", False)
        monkeypatch.setattr(importlib, "import_module", fake)

        with caplog.at_level(logging.WARNING):
            ts.discover_tools()

        assert "not loaded" in caplog.text
        assert _TARGET in caplog.text
        # Discovery still completed for the other modules.
        assert ts._discovery_done is True

    def test_unexpected_import_error_propagates(self, monkeypatch) -> None:
        real_import = importlib.import_module

        def fake(name: str, *args, **kwargs):
            if name == _TARGET:
                raise ImportError("cannot import name 'foo' from 'process_improve.univariate.tools'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(ts, "_discovery_done", False)
        monkeypatch.setattr(importlib, "import_module", fake)

        with pytest.raises(ImportError, match="cannot import name"):
            ts.discover_tools()
