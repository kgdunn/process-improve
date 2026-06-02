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

Configuration
-------------

Every knob is read lazily from :mod:`process_improve.config.settings`,
which in turn picks up the corresponding ``PROCESS_IMPROVE_*``
environment variable on first access (ENG-09 / ENG-27). Tests can
override a knob in-process via ``settings.tool_timeout = 5.0``; CI
deployments can still set env vars at startup. See
``process_improve/config.py`` for the canonical defaults table.
"""

from __future__ import annotations

import contextlib
import multiprocessing
import sys
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from concurrent.futures.process import BrokenProcessPool
from typing import Any

from process_improve.config import settings

# Prefer ``fork`` on Linux: the worker inherits the parent's imported
# numpy/registry/etc., which makes startup and tool-dispatch much cheaper
# than re-importing on every spawn. macOS is excluded because Apple's
# Accelerate framework (used by numpy) is not fork-safe and Python 3.13
# emits a DeprecationWarning when a multi-threaded parent forks; Windows
# does not support fork at all. On those platforms we fall back to the
# platform default (spawn).
_DEFAULT_MP_CONTEXT: multiprocessing.context.BaseContext | None = None
if sys.platform.startswith("linux"):
    try:
        _DEFAULT_MP_CONTEXT = multiprocessing.get_context("fork")
    except ValueError:
        _DEFAULT_MP_CONTEXT = None

# ---------------------------------------------------------------------------
# Legacy default-name compatibility shims (ENG-22: deprecate, remove in v2.0).
# ---------------------------------------------------------------------------
# Some callers imported these by name (``from process_improve.tool_safety
# import DEFAULT_TIMEOUT_S``). The new home for the actual values is
# ``process_improve.config.settings``; the shims below stay so existing
# imports keep resolving, but resolve to whatever the settings object
# currently reports rather than freezing the value at import time.


def __getattr__(name: str) -> Any:  # noqa: ANN401
    """Forward legacy ``DEFAULT_*`` reads to :data:`settings`.

    Lets ``from process_improve.tool_safety import DEFAULT_TIMEOUT_S``
    keep working, but the value is no longer frozen at module import:
    a test that overrides ``settings.tool_timeout`` will see the new
    value through both the new and the legacy name.
    """
    legacy_to_settings = {
        "DEFAULT_TIMEOUT_S": "tool_timeout",
        "DEFAULT_MAX_CELLS": "max_cells",
        "DEFAULT_MAX_STRING": "max_string",
        "DEFAULT_MAX_DEPTH": "max_depth",
        "DEFAULT_MEMORY_MB": "max_memory_mb",
    }
    if name in legacy_to_settings:
        return getattr(settings, legacy_to_settings[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Keys whose numeric value scales the cost of the underlying algorithm.
# A malicious caller can otherwise request a huge SVD or a long ESD loop
# with a tiny input array. SEC-19 (#268) extends the historic six keys
# with caps for the design / experiment cost knobs.
_SCALAR_CAPS: dict[str, float] = {
    "n_components": 50,
    "max_outliers_to_detect": 20,
    "n_iter": 10_000,
    "max_iter": 10_000,
    "n_boot": 1_000,
    "n_permutations": 1_000,
    "budget": 10_000,
    # SEC-19 additions.
    "n_steps": 100,
    "n_additional_runs": 500,
    "center_points": 50,
    "replicates": 50,
    "n_factors": 15,  # mirrors settings.max_factors_combinatorial
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
    max_cells: int | None = None,
    max_string: int | None = None,
    max_depth: int | None = None,
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

    # Resolve None sentinels from settings at call time (ENG-09 / ENG-27).
    if max_cells is None:
        max_cells = settings.max_cells
    if max_string is None:
        max_string = settings.max_string
    if max_depth is None:
        max_depth = settings.max_depth

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
# JSON-schema enforcement
# ---------------------------------------------------------------------------
#
# Each tool ships an ``input_schema`` (JSON Schema), but the dispatch path
# passes the input straight through as ``**kwargs`` without checking it. For an
# untrusted transport that means types, bounds, enums and ``required`` are
# unenforced and unknown keys reach the function. ``validate_against_schema``
# enforces the subset of JSON Schema actually used by the tool specs. It is
# deliberately dependency-free (no ``jsonschema``) so it adds no install/lockfile
# churn.

# A JSON "number" accepts ints; "integer" rejects floats. ``bool`` is excluded
# from the numeric types because ``True``/``False`` are ints in Python.
_JSON_TYPE_CHECKS: dict[str, Callable[[Any], bool]] = {
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
}


def _validate_value(key: str, value: Any, subschema: dict[str, Any]) -> None:  # noqa: ANN401
    """Validate a single parameter *value* against its property *subschema*."""
    # Optional parameters may be supplied as an explicit ``null``; the tool
    # function then falls back to its own default, so do not type-check None.
    if value is None:
        return

    expected = subschema.get("type")
    check = _JSON_TYPE_CHECKS.get(expected) if isinstance(expected, str) else None
    if check is not None and not check(value):
        raise ToolInputInvalidError(
            f"Parameter {key!r} must be of type {expected!r}, got {type(value).__name__}.",
            details={"key": key, "expected_type": expected, "observed_type": type(value).__name__},
        )

    enum = subschema.get("enum")
    if enum is not None and value not in enum:
        raise ToolInputInvalidError(
            f"Parameter {key!r}={value!r} is not one of {enum}.",
            details={"key": key, "enum": list(enum)},
        )

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = subschema.get("minimum")
        if minimum is not None and value < minimum:
            raise ToolInputInvalidError(
                f"Parameter {key!r}={value} is below the minimum of {minimum}.",
                details={"key": key, "minimum": minimum, "observed": value},
            )
        maximum = subschema.get("maximum")
        if maximum is not None and value > maximum:
            raise ToolInputInvalidError(
                f"Parameter {key!r}={value} exceeds the maximum of {maximum}.",
                details={"key": key, "maximum": maximum, "observed": value},
            )

    if isinstance(value, list):
        min_items = subschema.get("minItems")
        if min_items is not None and len(value) < min_items:
            raise ToolInputInvalidError(
                f"Parameter {key!r} has {len(value)} items; minimum is {min_items}.",
                details={"key": key, "min_items": min_items, "observed": len(value)},
            )
        max_items = subschema.get("maxItems")
        if max_items is not None and len(value) > max_items:
            raise ToolInputInvalidError(
                f"Parameter {key!r} has {len(value)} items; maximum is {max_items}.",
                details={"key": key, "max_items": max_items, "observed": len(value)},
            )


def validate_against_schema(tool_input: dict[str, Any], schema: dict[str, Any]) -> None:
    """Validate *tool_input* against a tool's JSON ``input_schema``.

    Enforces the subset of JSON Schema used by process-improve tool specs:
    the object's ``required`` keys must be present, parameters not declared in
    ``properties`` are rejected (no additional keys), and each supplied value is
    checked against its property ``type``, ``enum``, ``minimum``/``maximum``
    (numbers) and ``minItems``/``maxItems`` (arrays).

    Parameters
    ----------
    tool_input:
        The ``input`` dict that would be passed as keyword arguments to the tool.
    schema:
        The tool's ``input_schema`` (a JSON Schema object).

    Raises
    ------
    ToolInputInvalidError
        On any missing-required, unknown-key, type, enum, bound, or item-count
        violation.
    """
    if schema.get("type") != "object":
        return

    properties: dict[str, Any] = schema.get("properties", {})
    required = schema.get("required", [])

    for req_key in required:
        if req_key not in tool_input:
            raise ToolInputInvalidError(
                f"Missing required parameter {req_key!r}.",
                details={"key": req_key, "required": list(required)},
            )

    if properties:
        unknown = sorted(set(tool_input) - set(properties))
        if unknown:
            raise ToolInputInvalidError(
                f"Unknown parameter(s): {unknown}. Allowed: {sorted(properties)}.",
                details={"unknown": unknown, "allowed": sorted(properties)},
            )

    for key, value in tool_input.items():
        subschema = properties.get(key)
        if isinstance(subschema, dict):
            _validate_value(key, value, subschema)


def _lookup_input_schema(tool_name: str) -> dict[str, Any] | None:
    """Return the ``input_schema`` for *tool_name*, or None if not registered."""
    from process_improve.tool_spec import get_tool_specs  # noqa: PLC0415

    specs = get_tool_specs(names=[tool_name])
    return specs[0]["input_schema"] if specs else None


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


def get_pool(memory_mb: int | None = None, max_workers: int = 1) -> ProcessPoolExecutor:
    """Return a lazily-initialised module-level :class:`ProcessPoolExecutor`.

    The pool is recreated if ``memory_mb`` changes (e.g. tests override it).
    Passing ``None`` resolves the cap from :data:`settings.max_memory_mb` at
    call time (ENG-09 / ENG-27).
    """
    if memory_mb is None:
        memory_mb = settings.max_memory_mb
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


def _terminate_workers(pool: ProcessPoolExecutor) -> None:
    """Force-kill a pool's worker processes (best-effort).

    ``ProcessPoolExecutor.shutdown(wait=False, cancel_futures=True)`` does not
    interrupt a worker that is *already running* a task: ``cancel_futures`` only
    drops not-yet-started futures. So a runaway tool (infinite loop, pathological
    input) would keep a CPU core busy after a timeout. Here we reach into the
    CPython-internal process table and terminate each worker, escalating to
    ``kill()`` if ``terminate()`` does not take. Guarded with ``suppress`` so that
    an internal-API change degrades to the old (non-killing) behaviour instead of
    raising.
    """
    processes = getattr(pool, "_processes", None)
    if not processes:
        return
    workers = list(processes.values())
    for proc in workers:
        with contextlib.suppress(Exception):
            if proc.is_alive():
                proc.terminate()
    for proc in workers:
        with contextlib.suppress(Exception):
            proc.join(timeout=1.0)
            if proc.is_alive():
                proc.kill()


def shutdown_pool() -> None:
    """Shut down the module-level pool, if any. Safe to call repeatedly.

    Worker processes are force-terminated first so a runaway task cannot keep
    holding a CPU after the executor is torn down.
    """
    global _pool, _pool_memory_mb  # noqa: PLW0603
    if _pool is not None:
        _terminate_workers(_pool)
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
    timeout: float | None = None,
    max_cells: int | None = None,
    max_string: int | None = None,
    max_depth: int | None = None,
    memory_mb: int | None = None,
    executor: ProcessPoolExecutor | None = None,
) -> Any:  # noqa: ANN401
    """Execute a tool call with input validation, timeout, and memory cap.

    Parameters
    ----------
    tool_name, tool_input:
        Same meaning as :func:`process_improve.tool_spec.execute_tool_call`.
    timeout:
        Wall-clock seconds. On overrun the runaway worker is force-terminated
        (``terminate()`` then ``kill()``) so it cannot keep holding a CPU, and
        :class:`ToolTimeoutError` is raised.
    max_cells, max_string, max_depth:
        Input-size limits. See :func:`validate_input`.
    memory_mb:
        RSS cap applied to the worker subprocess via ``RLIMIT_AS`` (POSIX).
        On overrun the subprocess dies and :class:`ToolMemoryExceededError`
        is raised.
    executor:
        Optional caller-provided pool. When *None* (default) a module-level
        singleton is used and is recycled after every call, so each call runs
        in a fresh worker with isolated process-global state and reclaimed
        memory. A caller-provided executor is never recycled or terminated by
        this function - the caller owns its lifecycle.

    Raises
    ------
    ToolInputInvalidError, ToolInputTooLargeError:
        Synchronous rejection before any subprocess work. ``ToolInputInvalidError``
        also covers JSON-schema violations (wrong type, out-of-bounds value,
        bad enum, missing required key, or an unknown parameter).
    ToolTimeoutError:
        Wall-clock overrun.
    ToolMemoryExceededError:
        Worker subprocess died unexpectedly (likely OOM).
    ValueError:
        Unknown tool name (propagated from ``execute_tool_call``).
    """
    # Resolve None sentinels from settings at call time (ENG-09 / ENG-27).
    # Tests can monkey-patch ``settings.tool_timeout = 5.0`` and the next
    # call will see the override.
    if timeout is None:
        timeout = settings.tool_timeout
    if memory_mb is None:
        memory_mb = settings.max_memory_mb
    # ``max_cells`` / ``max_string`` / ``max_depth`` resolve inside
    # ``validate_input`` from the same source.

    validate_input(
        tool_input,
        max_cells=max_cells,
        max_string=max_string,
        max_depth=max_depth,
    )

    # Enforce the tool's declared JSON schema (types, bounds, enums, required,
    # no unknown keys). Done after the size check so an oversize payload is still
    # reported as ToolInputTooLargeError. Unknown tools fall through to the
    # worker, which raises the canonical "Unknown tool" ValueError.
    schema = _lookup_input_schema(tool_name)
    if schema is not None:
        validate_against_schema(tool_input, schema)

    pool = executor if executor is not None else get_pool(memory_mb=memory_mb)
    future = pool.submit(_worker_run, tool_name, tool_input)

    try:
        return future.result(timeout=timeout)
    except FuturesTimeoutError as exc:
        raise ToolTimeoutError(
            f"Tool {tool_name!r} exceeded {timeout}s timeout",
            details={"tool_name": tool_name, "timeout": timeout},
        ) from exc
    except BrokenProcessPool as exc:
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
    finally:
        # Recycle the module-managed pool after every call: each call then runs in
        # a fresh worker with clean process-global state (RNG, matplotlib, cached
        # imports) and reclaimed memory, and any worker that overran the timeout
        # or hit the memory cap is force-terminated here rather than left holding
        # a CPU. A caller-provided executor is left untouched - the caller owns
        # its lifecycle.
        if executor is None:
            shutdown_pool()
