"""Tests for the safe-execution wrapper around the tool registry.

Covers:

- Input-size validation (cells, string length, depth, scalar caps).
- Happy-path round-trip through the subprocess pool.
- Wall-clock timeout.
- Memory-cap breach.
- Unknown-tool handling.

The subprocess-based tests rely on ``fork`` so that the worker inherits
the parent's in-test ``@tool_spec`` registrations. Fork is preferred
only on Linux (see ``tool_safety._DEFAULT_MP_CONTEXT``); on macOS and
Windows these tests are skipped.
"""

from __future__ import annotations

import contextlib
import os
import sys
import time

import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict, Field

from process_improve.tool_safety import (
    ToolInputInvalidError,
    ToolInputTooLargeError,
    ToolMemoryExceededError,
    ToolSafetyError,
    ToolTimeoutError,
    _apply_memory_limit,
    _count_numeric_leaves,
    _lookup_input_model,
    _pool_initializer,
    _terminate_workers,
    _validate_against_model,
    _worker_run,
    get_pool,
    safe_execute_tool_call,
    shutdown_pool,
    validate_input,
)
from process_improve.tool_spec import _TOOL_REGISTRY, tool_spec

# Subprocess tests in this file register @tool_spec tools inline; those
# registrations only survive into the worker when the pool is forked.
# ``tool_safety`` only opts into fork on Linux, so skip elsewhere.
_skip_if_not_linux = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="Subprocess safety tests require a fork-based pool (Linux only)",
)


# ---------------------------------------------------------------------------
# Synthetic tools registered only for these tests. Fork inherits the registry,
# so the worker process sees them without any extra plumbing.
# ---------------------------------------------------------------------------


class _EchoInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: float


@tool_spec(
    name="_safety_test_echo",
    description="Return the input unchanged. Test-only.",
    input_model=_EchoInput,
)
def _safety_test_echo(spec: _EchoInput) -> dict:
    return {"value": spec.value}


class _SleepInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seconds: float


@tool_spec(
    name="_safety_test_sleep",
    description="Sleep for the given number of seconds. Test-only.",
    input_model=_SleepInput,
)
def _safety_test_sleep(spec: _SleepInput) -> dict:
    time.sleep(spec.seconds)
    return {"slept": spec.seconds}


class _MemoryBombInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    megabytes: int


@tool_spec(
    name="_safety_test_memory_bomb",
    description="Allocate the given number of megabytes. Test-only.",
    input_model=_MemoryBombInput,
)
def _safety_test_memory_bomb(spec: _MemoryBombInput) -> dict:
    # Allocate a NumPy array to force a real RSS increase.
    size = int(spec.megabytes) * 1024 * 1024 // 8
    _ = np.zeros(size, dtype=np.float64)
    return {"allocated_mb": spec.megabytes}


class _BusyLoopInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


@tool_spec(
    name="_safety_test_busy_loop",
    description="Spin on the CPU forever. Test-only.",
    input_model=_BusyLoopInput,
)
def _safety_test_busy_loop(spec: _BusyLoopInput) -> dict:
    while True:
        pass


def _pid_alive(pid: int) -> bool:
    """Return True if a process with *pid* still exists."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@pytest.fixture(autouse=True)
def _ensure_registered() -> None:
    """Test tools must live in the registry so fork children can see them."""
    assert "_safety_test_echo" in _TOOL_REGISTRY
    assert "_safety_test_sleep" in _TOOL_REGISTRY
    assert "_safety_test_memory_bomb" in _TOOL_REGISTRY


@pytest.fixture(autouse=True)
def _cleanup_pool() -> None:
    yield
    shutdown_pool()


# ---------------------------------------------------------------------------
# validate_input: synchronous, in-process
# ---------------------------------------------------------------------------


class TestValidateInput:
    def test_accepts_small_payload(self) -> None:
        validate_input({"x": [1, 2, 3], "name": "hi"})

    def test_rejects_non_dict(self) -> None:
        with pytest.raises(ToolInputInvalidError):
            validate_input([1, 2, 3])  # type: ignore[arg-type]

    def test_rejects_oversized_array(self) -> None:
        with pytest.raises(ToolInputTooLargeError) as exc_info:
            validate_input({"data": list(range(100))}, max_cells=10)
        assert exc_info.value.code == "input_too_large"
        assert exc_info.value.details["limit"] == 10

    def test_rejects_long_string(self) -> None:
        with pytest.raises(ToolInputTooLargeError):
            validate_input({"name": "x" * 100}, max_string=10)

    def test_rejects_excess_depth(self) -> None:
        deeply: dict = {"a": {}}
        cur = deeply["a"]
        for _ in range(15):
            cur["nested"] = {}
            cur = cur["nested"]
        with pytest.raises(ToolInputTooLargeError):
            validate_input(deeply, max_depth=5)

    def test_rejects_excess_scalar_cap(self) -> None:
        with pytest.raises(ToolInputTooLargeError) as exc_info:
            validate_input({"n_components": 10_000})
        assert exc_info.value.details["key"] == "n_components"

    def test_booleans_not_counted_against_scalar_caps(self) -> None:
        # bool is a subclass of int; caps should not fire for it.
        validate_input({"n_components": True})

    def test_nested_strings_checked(self) -> None:
        with pytest.raises(ToolInputTooLargeError):
            validate_input({"batch": [{"label": "x" * 200}]}, max_string=10)


# ---------------------------------------------------------------------------
# _validate_against_model + _lookup_input_model: SEC-04, synchronous, in-process
#
# The legacy bespoke ``validate_against_schema`` JSON-schema walker was
# removed alongside the legacy ``input_schema=`` parameter (ENG-04 / ENG-10
# cleanup). Validation now lives on the pydantic ``BaseModel`` attached to
# each tool: types, bounds, enums, ``required`` fields, and "no unknown keys"
# (via ``ConfigDict(extra="forbid")``) all come from ``model_validate``. The
# tests below pin that surface through the helpers ``safe_execute_tool_call``
# actually uses.
# ---------------------------------------------------------------------------


class _DemoInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: list[float] = Field(..., min_length=3, max_length=5)
    n_components: int = Field(..., ge=1)
    conf_level: float = Field(0.95, ge=0.8, le=0.999)
    method: str | None = Field(None, pattern="^[ab]$")
    name: str | None = None


class TestValidateAgainstModel:
    def test_valid_input_passes(self) -> None:
        _validate_against_model(
            "demo",
            {"data": [1, 2, 3], "n_components": 2, "conf_level": 0.95, "method": "a"},
            _DemoInput,
        )

    def test_missing_required_rejected(self) -> None:
        with pytest.raises(ToolInputInvalidError, match="n_components"):
            _validate_against_model("demo", {"data": [1, 2, 3]}, _DemoInput)

    def test_unknown_key_rejected(self) -> None:
        with pytest.raises(ToolInputInvalidError, match="bogus"):
            _validate_against_model(
                "demo", {"data": [1, 2, 3], "n_components": 2, "bogus": 1}, _DemoInput
            )

    def test_wrong_type_rejected(self) -> None:
        with pytest.raises(ToolInputInvalidError, match="n_components"):
            _validate_against_model(
                "demo", {"data": [1, 2, 3], "n_components": "two"}, _DemoInput
            )

    def test_below_minimum_rejected(self) -> None:
        with pytest.raises(ToolInputInvalidError, match="greater than or equal to 1"):
            _validate_against_model(
                "demo", {"data": [1, 2, 3], "n_components": 0}, _DemoInput
            )

    def test_above_maximum_rejected(self) -> None:
        with pytest.raises(ToolInputInvalidError, match=r"less than or equal to 0\.999"):
            _validate_against_model(
                "demo",
                {"data": [1, 2, 3], "n_components": 2, "conf_level": 2.0},
                _DemoInput,
            )

    def test_too_few_items_rejected(self) -> None:
        with pytest.raises(ToolInputInvalidError, match="at least 3 items"):
            _validate_against_model("demo", {"data": [1, 2], "n_components": 2}, _DemoInput)

    def test_too_many_items_rejected(self) -> None:
        with pytest.raises(ToolInputInvalidError, match="at most 5 items"):
            _validate_against_model(
                "demo", {"data": [1, 2, 3, 4, 5, 6], "n_components": 2}, _DemoInput
            )

    def test_bad_enum_rejected(self) -> None:
        with pytest.raises(ToolInputInvalidError, match="method"):
            _validate_against_model(
                "demo", {"data": [1, 2, 3], "n_components": 2, "method": "z"}, _DemoInput
            )

    def test_explicit_null_for_optional_allowed(self) -> None:
        _validate_against_model(
            "demo",
            {"data": [1, 2, 3], "n_components": 2, "name": None},
            _DemoInput,
        )

    def test_lookup_input_model_unknown_returns_none(self) -> None:
        assert _lookup_input_model("definitely_not_a_registered_tool") is None

    def test_lookup_input_model_known_returns_model(self) -> None:
        model = _lookup_input_model("_safety_test_echo")
        assert model is not None
        assert issubclass(model, BaseModel)
        assert "value" in model.model_fields


# ---------------------------------------------------------------------------
# _terminate_workers: SEC-02 helper, no real subprocesses (runs everywhere)
# ---------------------------------------------------------------------------


class _FakeProc:
    """A multiprocessing.Process double recording terminate()/kill() calls.

    ``alive`` is a list of booleans consumed by successive ``is_alive()`` calls
    (first in the terminate loop, then after ``join`` in the kill loop).
    """

    def __init__(self, alive: list[bool]) -> None:
        self._alive = alive
        self.terminated = False
        self.killed = False

    def is_alive(self) -> bool:
        return self._alive.pop(0) if self._alive else False

    def terminate(self) -> None:
        self.terminated = True

    def join(self, timeout: float | None = None) -> None:
        pass

    def kill(self) -> None:
        self.killed = True


class _FakePool:
    def __init__(self, procs: list[_FakeProc]) -> None:
        self._processes = dict(enumerate(procs))


class TestTerminateWorkers:
    def test_no_processes_attr_is_noop(self) -> None:
        # A pool object without a _processes table is handled gracefully.
        _terminate_workers(object())  # type: ignore[arg-type]

    def test_empty_process_table_is_noop(self) -> None:
        _terminate_workers(_FakePool([]))  # type: ignore[arg-type]

    def test_alive_worker_is_terminated_then_killed(self) -> None:
        # Still alive after terminate() + join -> escalate to kill().
        proc = _FakeProc(alive=[True, True])
        _terminate_workers(_FakePool([proc]))  # type: ignore[arg-type]
        assert proc.terminated
        assert proc.killed

    def test_terminate_suffices_when_worker_exits(self) -> None:
        # Alive at first, dead after terminate() -> no kill().
        proc = _FakeProc(alive=[True, False])
        _terminate_workers(_FakePool([proc]))  # type: ignore[arg-type]
        assert proc.terminated
        assert not proc.killed

    def test_already_dead_worker_untouched(self) -> None:
        proc = _FakeProc(alive=[False, False])
        _terminate_workers(_FakePool([proc]))  # type: ignore[arg-type]
        assert not proc.terminated
        assert not proc.killed

    def test_cpython_pool_still_exposes_processes_attribute(self) -> None:
        """SEC-31 (#280) regression guard.

        ``_terminate_workers`` reaches into ``ProcessPoolExecutor._processes``
        to enumerate workers it can ``terminate()`` / ``kill()`` after a
        timeout. That attribute is a CPython implementation detail; if a
        future Python release renames it, the blanket ``contextlib.suppress``
        in ``_terminate_workers`` would silently degrade the timeout
        guarantee back to the pre-SEC-02 behaviour (a runaway worker would
        keep a CPU after ``ToolTimeoutError``).

        Asserting the attribute exists at the supported Python versions
        means a future upgrade fails CI loudly instead of regressing
        invisibly.
        """
        from concurrent.futures import ProcessPoolExecutor

        pool = ProcessPoolExecutor(max_workers=1)
        try:
            assert hasattr(pool, "_processes"), (
                "ProcessPoolExecutor lost the _processes attribute on this "
                "Python version. Update _terminate_workers in tool_safety.py "
                "before bumping the supported Python range."
            )
        finally:
            pool.shutdown(wait=False, cancel_futures=True)


# ---------------------------------------------------------------------------
# Subprocess-based tests
# ---------------------------------------------------------------------------


@_skip_if_not_linux
class TestSafeExecuteToolCall:
    def test_happy_path_round_trip(self) -> None:
        result = safe_execute_tool_call("_safety_test_echo", {"value": 42}, timeout=10)
        assert result == {"value": 42}

    def test_unknown_tool_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown tool"):
            safe_execute_tool_call("no_such_tool", {}, timeout=5)

    def test_input_validation_runs_before_subprocess(self) -> None:
        with pytest.raises(ToolInputTooLargeError):
            safe_execute_tool_call(
                "_safety_test_echo",
                {"value": 1, "extra": list(range(10_000))},
                timeout=10,
                max_cells=100,
            )

    def test_schema_violation_rejected_before_subprocess(self) -> None:
        # SEC-04: wrong type for a declared parameter is rejected synchronously,
        # before any worker runs.
        with pytest.raises(ToolInputInvalidError):
            safe_execute_tool_call("_safety_test_echo", {"value": "not-a-number"}, timeout=10)

    def test_schema_unknown_key_rejected_before_subprocess(self) -> None:
        # Pydantic ``extra="forbid"`` rejects undeclared keys; the error message
        # references the extra-forbidden discriminator and the offending key.
        with pytest.raises(ToolInputInvalidError, match="extra_forbidden"):
            safe_execute_tool_call("_safety_test_echo", {"value": 1, "rogue": 2}, timeout=10)

    def test_timeout_raises_structured_error(self) -> None:
        with pytest.raises(ToolTimeoutError) as exc_info:
            safe_execute_tool_call("_safety_test_sleep", {"seconds": 5}, timeout=0.2)
        assert exc_info.value.code == "timeout"
        assert exc_info.value.details["tool_name"] == "_safety_test_sleep"

    def test_timeout_force_terminates_runaway_worker(self) -> None:
        # SEC-02: a CPU-bound runaway must actually be killed on timeout, not
        # left spinning. Warm the module pool (workers spawn lazily) so we can
        # record its worker pids, then run an infinite-loop tool through the same
        # pool and confirm those workers are gone afterwards.
        pool = get_pool()
        pool.submit(_worker_run, "_safety_test_echo", {"value": 0}).result(timeout=10)
        pids = list(getattr(pool, "_processes", {}).keys())
        assert pids, "expected at least one worker process"

        with pytest.raises(ToolTimeoutError):
            safe_execute_tool_call("_safety_test_busy_loop", {}, timeout=0.3)

        deadline = time.time() + 5
        while time.time() < deadline and any(_pid_alive(p) for p in pids):
            time.sleep(0.05)
        assert not any(_pid_alive(p) for p in pids), "runaway worker still alive after timeout"

    def test_module_pool_recycled_after_each_call(self) -> None:
        # SEC-03: the module-managed pool is recycled after every call so each
        # call runs in a fresh, isolated worker.
        import process_improve.tool_safety as ts

        result = safe_execute_tool_call("_safety_test_echo", {"value": 1}, timeout=10)
        assert result == {"value": 1}
        assert ts._pool is None

    def test_error_has_json_serialisable_dict(self) -> None:
        err: ToolSafetyError = ToolTimeoutError("boom", details={"tool_name": "x", "timeout": 1})
        payload = err.to_dict()
        assert payload["error"] == "timeout"
        assert payload["message"] == "boom"
        assert payload["details"]["timeout"] == 1

    @pytest.mark.skipif(sys.platform == "win32", reason="RLIMIT_AS is POSIX-only")
    def test_memory_cap_kills_runaway_worker(self) -> None:
        # Ask for far more memory than the cap allows; the subprocess should die.
        with pytest.raises(ToolMemoryExceededError) as exc_info:
            safe_execute_tool_call(
                "_safety_test_memory_bomb",
                {"megabytes": 2048},
                timeout=10,
                memory_mb=128,
            )
        assert exc_info.value.code == "memory_exceeded"


# ---------------------------------------------------------------------------
# Lower-level helpers, exercised in-process
# ---------------------------------------------------------------------------


class TestHelpers:
    """Direct tests for the internal building blocks."""

    def test_count_numeric_leaves_rejects_excess_depth(self) -> None:
        """_count_numeric_leaves raises once nesting exceeds max_depth."""
        nested: dict = {"a": {"b": {"c": {"d": 1}}}}
        with pytest.raises(ToolInputTooLargeError):
            _count_numeric_leaves(nested, depth=0, max_depth=2)

    def test_count_numeric_leaves_counts_scalars(self) -> None:
        assert _count_numeric_leaves({"x": [1, 2], "y": 3}, depth=0, max_depth=10) == 3
        assert _count_numeric_leaves("a string", depth=0, max_depth=10) == 0

    @pytest.mark.skipif(sys.platform == "win32", reason="RLIMIT_AS is POSIX-only")
    def test_apply_memory_limit_is_safe_with_generous_cap(self) -> None:
        """A very large cap leaves the running process unharmed."""
        import resource

        original = resource.getrlimit(resource.RLIMIT_AS)
        try:
            # 1 TB cap: far above this process, so nothing is actually constrained.
            _apply_memory_limit(1024 * 1024)
        finally:
            with contextlib.suppress(ValueError, OSError):
                resource.setrlimit(resource.RLIMIT_AS, original)

    def test_worker_run_dispatches_to_registry(self) -> None:
        """_worker_run executes a registered tool in the current process."""
        assert _worker_run("_safety_test_echo", {"value": 99}) == {"value": 99}

    @pytest.mark.skipif(sys.platform == "win32", reason="RLIMIT_AS is POSIX-only")
    def test_pool_initializer_warms_registry(self) -> None:
        """_pool_initializer discovers tools and applies the memory cap."""
        import resource

        original = resource.getrlimit(resource.RLIMIT_AS)
        try:
            _pool_initializer(1024 * 1024)
        finally:
            with contextlib.suppress(ValueError, OSError):
                resource.setrlimit(resource.RLIMIT_AS, original)

    @_skip_if_not_linux
    def test_get_pool_returns_cached_instance(self) -> None:
        """Repeated get_pool calls with the same memory cap reuse one pool."""
        first = get_pool(memory_mb=256)
        second = get_pool(memory_mb=256)
        assert first is second
