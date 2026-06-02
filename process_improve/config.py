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
    # Tool-safety knobs (pre-existing).
    "tool_timeout": 10.0,
    "max_cells": 1_000_000,
    "max_string": 100_000,
    "max_depth": 10,
    "max_memory_mb": 1024,
    "mcp_safe_mode": False,
    # SEC-19 DoS caps (added in this PR). Each is "comfortably above
    # any legitimate use" and rejects payloads designed to exhaust CPU
    # or memory in algorithms with poor input-size scaling.
    "max_factors_combinatorial": 15,  # 2**15 = 32k rows for full_factorial
    "max_regression_points": 5_000,   # repeated_median_slope is O(N^2)
    "max_matrix_rows": 10_000,        # fit_pca / fit_pls / data inputs
    "max_matrix_cols": 500,           # SVD on 500 columns is the practical limit
    "max_formula_chars": 4_096,       # fit_linear_model formula string
    "max_formula_terms": 100,         # expanded patsy RHS terms
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
    "max_factors_combinatorial": "PROCESS_IMPROVE_MAX_FACTORS_COMBINATORIAL",
    "max_regression_points": "PROCESS_IMPROVE_MAX_REGRESSION_POINTS",
    "max_matrix_rows": "PROCESS_IMPROVE_MAX_MATRIX_ROWS",
    "max_matrix_cols": "PROCESS_IMPROVE_MAX_MATRIX_COLS",
    "max_formula_chars": "PROCESS_IMPROVE_MAX_FORMULA_CHARS",
    "max_formula_terms": "PROCESS_IMPROVE_MAX_FORMULA_TERMS",
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
    # SEC-19 DoS caps. Each is the maximum value the underlying
    # algorithm will accept; anything bigger raises a clear
    # ``ValueError`` / ``ToolInputInvalidError`` at the boundary.
    # ------------------------------------------------------------------

    @property
    def max_factors_combinatorial(self) -> int:
        """Maximum ``k`` for combinatorial design generators (``ff2n``,
        ``fullfact``, simplex centroid / lattice). Default 15 caps
        ``2**k`` rows at ~32 KiB of memory per cell.
        """
        return self._cache.setdefault(
            "max_factors_combinatorial",
            _read_env_int(
                ENV_VAR_NAMES["max_factors_combinatorial"],
                DEFAULTS["max_factors_combinatorial"],
            ),
        )

    @max_factors_combinatorial.setter
    def max_factors_combinatorial(self, value: int) -> None:
        self._cache["max_factors_combinatorial"] = int(value)

    @property
    def max_regression_points(self) -> int:
        """Maximum ``len(x)`` / ``len(y)`` for the O(N^2) regression
        kernels (``repeated_median_slope`` etc.).
        """
        return self._cache.setdefault(
            "max_regression_points",
            _read_env_int(
                ENV_VAR_NAMES["max_regression_points"],
                DEFAULTS["max_regression_points"],
            ),
        )

    @max_regression_points.setter
    def max_regression_points(self, value: int) -> None:
        self._cache["max_regression_points"] = int(value)

    @property
    def max_matrix_rows(self) -> int:
        """Maximum row count for ``data`` / ``x_data`` matrix inputs to
        ``fit_pca`` / ``fit_pls`` / ``detect_multivariate_outliers``.
        """
        return self._cache.setdefault(
            "max_matrix_rows",
            _read_env_int(ENV_VAR_NAMES["max_matrix_rows"], DEFAULTS["max_matrix_rows"]),
        )

    @max_matrix_rows.setter
    def max_matrix_rows(self, value: int) -> None:
        self._cache["max_matrix_rows"] = int(value)

    @property
    def max_matrix_cols(self) -> int:
        """Maximum column count for matrix inputs to the multivariate tools."""
        return self._cache.setdefault(
            "max_matrix_cols",
            _read_env_int(ENV_VAR_NAMES["max_matrix_cols"], DEFAULTS["max_matrix_cols"]),
        )

    @max_matrix_cols.setter
    def max_matrix_cols(self, value: int) -> None:
        self._cache["max_matrix_cols"] = int(value)

    @property
    def max_formula_chars(self) -> int:
        """Maximum length (chars) of a model-formula string accepted by
        ``fit_linear_model`` and ``analyze_experiment``.
        """
        return self._cache.setdefault(
            "max_formula_chars",
            _read_env_int(
                ENV_VAR_NAMES["max_formula_chars"], DEFAULTS["max_formula_chars"]
            ),
        )

    @max_formula_chars.setter
    def max_formula_chars(self, value: int) -> None:
        self._cache["max_formula_chars"] = int(value)

    @property
    def max_formula_terms(self) -> int:
        """Maximum number of terms after patsy expansion of a model formula."""
        return self._cache.setdefault(
            "max_formula_terms",
            _read_env_int(
                ENV_VAR_NAMES["max_formula_terms"], DEFAULTS["max_formula_terms"]
            ),
        )

    @max_formula_terms.setter
    def max_formula_terms(self, value: int) -> None:
        self._cache["max_formula_terms"] = int(value)

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
