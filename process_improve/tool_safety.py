"""(c) Kevin Dunn, 2010-2026. MIT License.

Safe execution wrapper for process-improve tool calls.

Adds the four guard rails needed to expose the tool registry over an
untrusted transport (public MCP server, hosted REST API, etc.):

1. Input-size validation (reject oversize arrays/strings before work).
2. Wall-clock timeout via subprocess isolation.
3. Memory cap per subprocess (POSIX; best-effort on Windows).
4. Structured error types so callers can distinguish failure modes.

The in-process :func:`process_improve.tool_spec.execute_tool_call` is
left untouched for callers that trust their input (notebooks, tests,
the stdio MCP server running on the user's own machine). Hosted
callers should use :func:`safe_execute_tool_call` instead.

Environment variables (all optional):

- ``PROCESS_IMPROVE_TOOL_TIMEOUT`` -- seconds, default 10
- ``PROCESS_IMPROVE_MAX_CELLS`` -- max numeric leaves in input, default 1_000_000
- ``PROCESS_IMPROVE_MAX_STRING`` -- max chars in any single string, default 100_000
- ``PROCESS_IMPROVE_MAX_DEPTH`` -- max nested dict/list depth, default 10
- ``PROCESS_IMPROVE_MAX_MEMORY_MB`` -- per-subprocess RSS cap, default 1024
"""

from __future__ import annotations

import contextlib
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from concurrent.futures.process import BrokenProcessPool
from typing import Any

# Prefer fork where available: the worker inherits the parent's imported
# numpy/registry/etc., which makes startup and tool-dispatch much cheaper
# than re-importing on every spawn. Falls back to the platform default
# (spawn) on Windows, where fork is not supported.
try:
    _DEFAULT_MP_CONTEXT: multiprocessing.context.BaseContext | None = multiprocessing.get_context("fork")
except ValueError:
    _DEFAULT_MP_CONTEXT = None

# ---------------------------------------------------------------------------
# Defaults (read once at import time; tests can override via env before import)
# ---------------------------------------------------------------------------

DEFAULT_TIMEOUT_S: float = float(os.environ.get("PROCESS_IMPROVE_TOOL_TIMEOUT", "10"))
DEFAULT_MAX_CELLS: int = int(os.environ.get("PROCESS_IMPROVE_MAX_CELLS", "1000000"))
DEFAULT_MAX_STRING: int = int(os.environ.get("PROCESS_IMPROVE_MAX_STRING", "100000"))
DEFAULT_MAX_DEPTH: int = int(os.environ.get("PROCESS_IMPROVE_MAX_DEPTH", "10"))
DEFAULT_MEMORY_MB: int = int(os.environ.get("PROCESS_IMPROVE_MAX_MEMORY_MB", "1024"))

# Keys whose numeric value scales the cost of the underlying algorithm.
# A malicious caller can otherwise request a huge SVD or a long ESD loop
# with a tiny input array.
_SCALAR_CAPS: dict[str, float] = {
    "n_components": 50,
    "max_outliers_to_detect": 20,
    "n_iter": 10_000,
    "max_iter": 10_000,
    "n_boot": 1_000,
    "n_permutations": 1_000,
    "budget": 10_000,
}


# ---------------------------------------------------------------------------
# Structured errors
# ---------------------------------------------------------------------------


class ToolSafetyError(Exception):
    """Base class for safety-related tool-execution failures."""

    code: str = "tool_safety_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of this error."""
        return {"error": self.code, "message": str(self), "details": self.details}


class ToolInputTooLargeError(ToolSafetyError):
    """Input exceeded an allowed size limit (cells, string length, depth)."""

    code = "input_too_large"


class ToolInputInvalidError(ToolSafetyError):
    """Input failed structural validation (unexpected types, bad shape)."""

    code = "input_invalid"


class ToolTimeoutError(ToolSafetyError):
    """Tool call exceeded the wall-clock timeout."""

    code = "timeout"


class ToolMemoryExceededError(ToolSafetyError):
    """Subprocess was killed, most likely by the memory limit."""

    code = "memory_exceeded"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _count_numeric_leaves(value: Any, depth: int, max_depth: int) -> int:  # noqa: ANN401
    if depth > max_depth:
        raise ToolInputTooLargeError(
            f"Input nesting depth exceeds limit of {max_depth}",
            details={"limit": max_depth},
        )
    if isinstance(value, (list, tuple)):
        return sum(_count_numeric_leaves(v, depth + 1, max_depth) for v in value)
    if isinstance(value, dict):
        return sum(_count_numeric_leaves(v, depth + 1, max_depth) for v in value.values())
    if isinstance(value, (int, float, bool)):
        return 1
    # Strings, None, and anything else do not count as numeric cells.
    return 0


def _check_strings(value: Any, limit: int, depth: int, max_depth: int) -> None:  # noqa: ANN401
    if depth > max_depth:
        # Depth check is already handled by _count_numeric_leaves; keep this
        # function simple so it can be reused independently.
        raise ToolInputTooLargeError(
            f"Input nesting depth exceeds limit of {max_depth}",
            details={"limit": max_depth},
        )
    if isinstance(value, str):
        if len(value) > limit:
            raise ToolInputTooLargeError(
                f"String length {len(value)} exceeds limit of {limit}",
                details={"limit": limit, "observed": len(value)},
            )
        return
    if isinstance(value, (list, tuple)):
        for v in value:
            _check_strings(v, limit, depth + 1, max_depth)
    elif isinstance(value, dict):
        for v in value.values():
            _check_strings(v, limit, depth + 1, max_depth)


def validate_input(
    tool_input: dict[str, Any],
    *,
    max_cells: int = DEFAULT_MAX_CELLS,
    max_string: int = DEFAULT_MAX_STRING,
    max_depth: int = DEFAULT_MAX_DEPTH,
    scalar_caps: dict[str, float] | None = None,
) -> None:
    """Raise :class:`ToolInputTooLargeError` if *tool_input* breaks any limit.

    Parameters
    ----------
    tool_input:
        The ``input`` dict that would be passed as keyword arguments to
        the tool function.
    max_cells:
        Maximum number of numeric leaves anywhere in the payload.
    max_string:
        Maximum length of any single string value.
    max_depth:
        Maximum nesting depth for dicts/lists.
    scalar_caps:
        Override the default per-key numeric caps (see :data:`_SCALAR_CAPS`).
    """
    if not isinstance(tool_input, dict):
        raise ToolInputInvalidError(
            f"Tool input must be a dict, got {type(tool_input).__name__}",
        )

    caps = {**_SCALAR_CAPS, **(scalar_caps or {})}
    for key, limit in caps.items():
        if key in tool_input:
            observed = tool_input[key]
            if isinstance(observed, (int, float)) and not isinstance(observed, bool) and observed > limit:
                raise ToolInputTooLargeError(
                    f"Parameter {key!r}={observed} exceeds limit of {limit}",
                    details={"key": key, "limit": limit, "observed": observed},
                )

    _check_strings(tool_input, max_string, 0, max_depth)

    cells = _count_numeric_leaves(tool_input, 0, max_depth)
    if cells > max_cells:
        raise ToolInputTooLargeError(
            f"Input contains {cells} numeric cells; limit is {max_cells}",
            details={"limit": max_cells, "observed": cells},
        )


# ---------------------------------------------------------------------------
# Subprocess worker
# ---------------------------------------------------------------------------


def _apply_memory_limit(memory_mb: int) -> None:
    """Apply an address-space limit to the current process (POSIX)."""
    try:
        import resource  # noqa: PLC0415  POSIX-only
    except ImportError:
        return  # Windows: no-op
    limit_bytes = memory_mb * 1024 * 1024
    # Some sandboxes disallow raising the hard limit; ignore silently.
    with contextlib.suppress(ValueError, OSError):
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))


def _pool_initializer(memory_mb: int) -> None:
    """Run inside each worker process before it accepts tasks.

    Imports run *before* the memory cap is applied so that large
    data files loaded at import time (e.g. pyDOE3's orthogonal-array
    tables) do not count against the per-call budget.
    """
    # Warm the registry so the first real call doesn't pay discovery cost.
    from process_improve.tool_spec import discover_tools  # noqa: PLC0415

    discover_tools()
    _apply_memory_limit(memory_mb)


def _worker_run(tool_name: str, tool_input: dict[str, Any]) -> Any:  # noqa: ANN401
    """Target function executed inside a worker process."""
    from process_improve.tool_spec import execute_tool_call  # noqa: PLC0415

    return execute_tool_call(tool_name, tool_input)


# ---------------------------------------------------------------------------
# Pool management
# ---------------------------------------------------------------------------


_pool: ProcessPoolExecutor | None = None
_pool_memory_mb: int | None = None


def get_pool(memory_mb: int = DEFAULT_MEMORY_MB, max_workers: int = 1) -> ProcessPoolExecutor:
    """Return a lazily-initialised module-level :class:`ProcessPoolExecutor`.

    The pool is recreated if ``memory_mb`` changes (e.g. tests override it).
    """
    global _pool, _pool_memory_mb  # noqa: PLW0603
    if _pool is None or _pool_memory_mb != memory_mb:
        shutdown_pool()
        kwargs: dict[str, Any] = {
            "max_workers": max_workers,
            "initializer": _pool_initializer,
            "initargs": (memory_mb,),
        }
        if _DEFAULT_MP_CONTEXT is not None:
            kwargs["mp_context"] = _DEFAULT_MP_CONTEXT
        _pool = ProcessPoolExecutor(**kwargs)
        _pool_memory_mb = memory_mb
    return _pool


def shutdown_pool() -> None:
    """Shut down the module-level pool, if any. Safe to call repeatedly."""
    global _pool, _pool_memory_mb  # noqa: PLW0603
    if _pool is not None:
        _pool.shutdown(wait=False, cancel_futures=True)
        _pool = None
        _pool_memory_mb = None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def safe_execute_tool_call(  # noqa: PLR0913
    tool_name: str,
    tool_input: dict[str, Any],
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
    max_cells: int = DEFAULT_MAX_CELLS,
    max_string: int = DEFAULT_MAX_STRING,
    max_depth: int = DEFAULT_MAX_DEPTH,
    memory_mb: int = DEFAULT_MEMORY_MB,
    executor: ProcessPoolExecutor | None = None,
) -> Any:  # noqa: ANN401
    """Execute a tool call with input validation, timeout, and memory cap.

    Parameters
    ----------
    tool_name, tool_input:
        Same meaning as :func:`process_improve.tool_spec.execute_tool_call`.
    timeout:
        Wall-clock seconds. On overrun, the current subprocess is terminated
        and a fresh pool is started; :class:`ToolTimeoutError` is raised.
    max_cells, max_string, max_depth:
        Input-size limits. See :func:`validate_input`.
    memory_mb:
        RSS cap applied to the worker subprocess via ``RLIMIT_AS`` (POSIX).
        On overrun the subprocess dies and :class:`ToolMemoryExceededError`
        is raised.
    executor:
        Optional caller-provided pool. When *None* (default) a module-level
        singleton is used.

    Raises
    ------
    ToolInputInvalidError, ToolInputTooLargeError:
        Synchronous rejection before any subprocess work.
    ToolTimeoutError:
        Wall-clock overrun.
    ToolMemoryExceededError:
        Worker subprocess died unexpectedly (likely OOM).
    ValueError:
        Unknown tool name (propagated from ``execute_tool_call``).
    """
    validate_input(
        tool_input,
        max_cells=max_cells,
        max_string=max_string,
        max_depth=max_depth,
    )

    pool = executor if executor is not None else get_pool(memory_mb=memory_mb)
    future = pool.submit(_worker_run, tool_name, tool_input)

    try:
        return future.result(timeout=timeout)
    except FuturesTimeoutError as exc:
        # Kill the whole pool so the runaway worker cannot hold the CPU.
        if executor is None:
            shutdown_pool()
        raise ToolTimeoutError(
            f"Tool {tool_name!r} exceeded {timeout}s timeout",
            details={"tool_name": tool_name, "timeout": timeout},
        ) from exc
    except BrokenProcessPool as exc:
        if executor is None:
            shutdown_pool()
        raise ToolMemoryExceededError(
            f"Tool {tool_name!r} worker died (likely exceeded memory limit of {memory_mb} MB)",
            details={"tool_name": tool_name, "memory_mb": memory_mb},
        ) from exc
    except MemoryError as exc:
        # RLIMIT_AS caused an allocator to fail inside the worker; the worker
        # stays alive, but the tool could not complete. Surface as a
        # structured error so hosted callers can distinguish this from bugs.
        raise ToolMemoryExceededError(
            f"Tool {tool_name!r} exceeded memory limit of {memory_mb} MB",
            details={"tool_name": tool_name, "memory_mb": memory_mb},
        ) from exc
