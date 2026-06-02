"""(c) Kevin Dunn, 2010-2026. MIT License.

Central configuration for ``process-improve``.

Closes **ENG-09** (#291) -- "configuration sprawl: env vars at import
time, magic numbers, no central config" -- and **ENG-27** (#309) --
"``tool_safety.py`` reads env vars at import time".

The :data:`settings` singleton is the single place every other module
reads configurable knobs from. Each knob:

* has a documented default,
* can be overridden by the corresponding environment variable
  (``PROCESS_IMPROVE_*``), and
* is read **on first access**, not at import time. Tests can call
  :meth:`Settings.reload` to re-read after mutating the environment;
  callers can override a single knob in code via
  ``settings.tool_timeout = 5.0`` (the setter writes through and
  bypasses the cache).

Usage
-----

>>> from process_improve.config import settings
>>> settings.tool_timeout
10.0
>>> settings.mcp_safe_mode
False

Override via environment::

    PROCESS_IMPROVE_TOOL_TIMEOUT=30 python -m process_improve.mcp_server

Override in code (e.g. in a test)::

    from process_improve.config import settings
    monkeypatch.setenv("PROCESS_IMPROVE_TOOL_TIMEOUT", "5")
    settings.reload()
    assert settings.tool_timeout == 5.0

Why a class rather than module-level globals?
---------------------------------------------

Module-level globals are read once at import; a test that mutates
``os.environ`` after import has no effect. The :class:`Settings` class
caches values on first access and exposes ``reload()`` for the test
case. That matches what every other module's behaviour *should* be.

The class deliberately does **not** depend on ``pydantic-settings``
(or any new third-party package). ``pydantic`` is already a hard dep
(`ENG-04 <https://github.com/kgdunn/process-improve/issues/286>`_ is
the open decision about whether to commit to it everywhere); a plain
class with explicit env-var reads keeps that decision unfettered.
"""

from __future__ import annotations

import os
from typing import Any, Final


def _read_env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {name}={raw!r} is not a valid float."
        ) from exc


def _read_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {name}={raw!r} is not a valid integer."
        ) from exc


def _read_env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Knob registry
# ---------------------------------------------------------------------------

#: Default knob values. Kept here so ``Settings.reload()`` knows what to
#: fall back to when an env var is unset, and so :data:`DEFAULTS` is the
#: single place to read the canonical defaults from.
DEFAULTS: Final[dict[str, Any]] = {
    "tool_timeout": 10.0,
    "max_cells": 1_000_000,
    "max_string": 100_000,
    "max_depth": 10,
    "max_memory_mb": 1024,
    "mcp_safe_mode": False,
}

#: Mapping from knob name to environment-variable name. ``tool_safety``'s
#: original env-var contract is preserved verbatim so existing deployments
#: keep working without modification.
ENV_VAR_NAMES: Final[dict[str, str]] = {
    "tool_timeout": "PROCESS_IMPROVE_TOOL_TIMEOUT",
    "max_cells": "PROCESS_IMPROVE_MAX_CELLS",
    "max_string": "PROCESS_IMPROVE_MAX_STRING",
    "max_depth": "PROCESS_IMPROVE_MAX_DEPTH",
    "max_memory_mb": "PROCESS_IMPROVE_MAX_MEMORY_MB",
    "mcp_safe_mode": "PROCESS_IMPROVE_MCP_SAFE_MODE",
}


class Settings:
    """Single-instance configuration store.

    Every attribute is a knob; reads are cached after the first access.
    Call :meth:`reload` after mutating ``os.environ`` (typically inside a
    test fixture); call :meth:`override` to set a single knob from code.
    """

    __slots__ = ("_cache",)

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Typed properties (one per knob).
    # We list them explicitly rather than using ``__getattr__`` so IDEs
    # and ``help()`` can show them.
    # ------------------------------------------------------------------

    @property
    def tool_timeout(self) -> float:
        """Wall-clock seconds budget for a single tool call."""
        return self._cache.setdefault(
            "tool_timeout",
            _read_env_float(ENV_VAR_NAMES["tool_timeout"], DEFAULTS["tool_timeout"]),
        )

    @tool_timeout.setter
    def tool_timeout(self, value: float) -> None:
        self._cache["tool_timeout"] = float(value)

    @property
    def max_cells(self) -> int:
        """Maximum number of numeric leaves anywhere in a tool input payload."""
        return self._cache.setdefault(
            "max_cells",
            _read_env_int(ENV_VAR_NAMES["max_cells"], DEFAULTS["max_cells"]),
        )

    @max_cells.setter
    def max_cells(self, value: int) -> None:
        self._cache["max_cells"] = int(value)

    @property
    def max_string(self) -> int:
        """Maximum length of any single string in a tool input payload."""
        return self._cache.setdefault(
            "max_string",
            _read_env_int(ENV_VAR_NAMES["max_string"], DEFAULTS["max_string"]),
        )

    @max_string.setter
    def max_string(self, value: int) -> None:
        self._cache["max_string"] = int(value)

    @property
    def max_depth(self) -> int:
        """Maximum nesting depth of any tool input payload."""
        return self._cache.setdefault(
            "max_depth",
            _read_env_int(ENV_VAR_NAMES["max_depth"], DEFAULTS["max_depth"]),
        )

    @max_depth.setter
    def max_depth(self, value: int) -> None:
        self._cache["max_depth"] = int(value)

    @property
    def max_memory_mb(self) -> int:
        """Per-subprocess RSS cap for tool execution (MiB)."""
        return self._cache.setdefault(
            "max_memory_mb",
            _read_env_int(ENV_VAR_NAMES["max_memory_mb"], DEFAULTS["max_memory_mb"]),
        )

    @max_memory_mb.setter
    def max_memory_mb(self, value: int) -> None:
        self._cache["max_memory_mb"] = int(value)

    @property
    def mcp_safe_mode(self) -> bool:
        """Whether the MCP server should treat its transport as untrusted.

        When ``True``, every tool call goes through
        :func:`process_improve.tool_safety.safe_execute_tool_call`
        (validation, subprocess isolation, memory cap).
        """
        return self._cache.setdefault(
            "mcp_safe_mode",
            _read_env_bool(ENV_VAR_NAMES["mcp_safe_mode"], DEFAULTS["mcp_safe_mode"]),
        )

    @mcp_safe_mode.setter
    def mcp_safe_mode(self, value: bool) -> None:
        self._cache["mcp_safe_mode"] = bool(value)

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def reload(self) -> None:
        """Drop the cache so the next attribute access re-reads from env."""
        self._cache.clear()

    def as_dict(self) -> dict[str, Any]:
        """Return a snapshot of every knob's current value.

        Triggers a read of every knob (populating the cache); useful in
        ``--verbose`` startup banners and for printing the effective
        configuration.
        """
        return {name: getattr(self, name) for name in DEFAULTS}


#: The single module-level :class:`Settings` instance. Every other
#: module imports this object directly.
settings: Settings = Settings()


__all__ = ["DEFAULTS", "ENV_VAR_NAMES", "Settings", "settings"]
