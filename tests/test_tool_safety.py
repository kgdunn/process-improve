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

import json
import os
import subprocess
import sys
import textwrap
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
    _count_numeric_leaves,
    _lookup_input_model,
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
            _validate_against_model("demo", {"data": [1, 2, 3], "n_components": 2, "bogus": 1}, _DemoInput)

    def test_wrong_type_rejected(self) -> None:
        with pytest.raises(ToolInputInvalidError, match="n_components"):
            _validate_against_model("demo", {"data": [1, 2, 3], "n_components": "two"}, _DemoInput)

    def test_below_minimum_rejected(self) -> None:
        with pytest.raises(ToolInputInvalidError, match="greater than or equal to 1"):
            _validate_against_model("demo", {"data": [1, 2, 3], "n_components": 0}, _DemoInput)

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
            _validate_against_model("demo", {"data": [1, 2, 3, 4, 5, 6], "n_components": 2}, _DemoInput)

    def test_bad_enum_rejected(self) -> None:
        with pytest.raises(ToolInputInvalidError, match="method"):
            _validate_against_model("demo", {"data": [1, 2, 3], "n_components": 2, "method": "z"}, _DemoInput)

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


#: Applying an address-space limit is checked in a child process, never in the test
#: runner. RLIMIT_AS starts at infinity here, and lowering it is permitted while raising
#: it back is not ("not allowed to raise maximum limit"), so a `finally` that restores
#: the original silently fails and leaves the runner capped for the rest of the session.
#: Worse for a test: once the cap is in place, asserting that the function applied it is
#: satisfied whether or not the function did anything. A child process sheds the limit on
#: exit, so the assertion means what it says.
_CAP_MB = 1024 * 1024  # 1 TB: a real change from infinity, far above anything in use.


def _address_space_in_child(body: str) -> dict:
    """Run `body` in a fresh interpreter and return the JSON dict it prints."""
    # Assembled by concatenation, not by interpolating into an indented f-string: only
    # the first line of an interpolated block picks up the surrounding indent, which
    # leaves `textwrap.dedent` nothing in common to strip and the child with an
    # IndentationError.
    source = "import json, resource\n" + textwrap.dedent(body).strip() + "\nprint(json.dumps(result))\n"
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-c", source], capture_output=True, text=True, check=True
    )
    return json.loads(completed.stdout)


def _platform_allows_lowering_address_space() -> bool:
    """Report whether a plain `setrlimit` works here, independent of the code under test.

    Probing separately keeps the skip honest: if this returns True and the function under
    test still leaves the limit alone, that is a real failure rather than a platform quirk.
    """
    probe = _address_space_in_child(
        f"""
        cap = {_CAP_MB} * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
            result = {{"allowed": True}}
        except (ValueError, OSError):
            result = {{"allowed": False}}
        """
    )
    return bool(probe["allowed"])


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

    def test_timeout_force_terminates_runaway_worker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # SEC-02: a CPU-bound runaway must actually be killed on timeout, not
        # left spinning. The default path now runs each call in a PRIVATE pool
        # (audit fix: the shared module pool was not thread-safe), so capture
        # the worker pids at teardown time via _terminate_workers and confirm
        # they are gone afterwards.
        seen_pids: list[int] = []

        def capture_and_terminate(pool) -> None:
            seen_pids.extend(getattr(pool, "_processes", None) or {})
            _terminate_workers(pool)

        monkeypatch.setattr("process_improve.tool_safety._terminate_workers", capture_and_terminate)

        with pytest.raises(ToolTimeoutError):
            safe_execute_tool_call("_safety_test_busy_loop", {}, timeout=0.3)

        assert seen_pids, "expected at least one worker process"
        deadline = time.time() + 5
        while time.time() < deadline and any(_pid_alive(p) for p in seen_pids):
            time.sleep(0.05)
        assert not any(_pid_alive(p) for p in seen_pids), "runaway worker still alive after timeout"

    def test_default_path_uses_a_private_per_call_pool(self) -> None:
        # SEC-03 + audit fix: each default-path call runs in its own fresh
        # worker (clean process-global state), and never touches the shared
        # module-level pool, so concurrent calls cannot tear down each
        # other's workers.
        from process_improve import tool_safety as ts

        ts.shutdown_pool()
        result = safe_execute_tool_call("_safety_test_echo", {"value": 1}, timeout=10)
        assert result == {"value": 1}
        assert ts._pool_state is None

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
    def test_apply_memory_limit_applies_the_cap_it_was_given(self) -> None:
        """The cap reaches RLIMIT_AS, checked in a child process.

        The body used to call `_apply_memory_limit` and assert nothing, so it passed
        whether or not the limit was applied. It cannot be checked in the runner: see the
        note on `_address_space_in_child`.
        """
        if not _platform_allows_lowering_address_space():
            pytest.skip("this environment does not permit lowering RLIMIT_AS")

        observed = _address_space_in_child(
            f"""
            from process_improve.tool_safety import _apply_memory_limit
            _apply_memory_limit({_CAP_MB})
            result = {{"soft": resource.getrlimit(resource.RLIMIT_AS)[0]}}
            """
        )

        assert observed["soft"] == _CAP_MB * 1024 * 1024

    def test_worker_run_dispatches_to_registry(self) -> None:
        """_worker_run executes a registered tool in the current process."""
        assert _worker_run("_safety_test_echo", {"value": 99}) == {"value": 99}

    @pytest.mark.skipif(sys.platform == "win32", reason="RLIMIT_AS is POSIX-only")
    def test_pool_initializer_warms_registry_and_applies_the_cap(self) -> None:
        """Both halves of what `_pool_initializer` promises, checked in a child process.

        The body used to assert neither, so it passed even if the initializer discovered
        nothing and set no limit.
        """
        if not _platform_allows_lowering_address_space():
            pytest.skip("this environment does not permit lowering RLIMIT_AS")

        observed = _address_space_in_child(
            f"""
            from process_improve.tool_safety import _pool_initializer
            from process_improve.tool_spec import _TOOL_REGISTRY
            _pool_initializer({_CAP_MB})
            result = {{
                "soft": resource.getrlimit(resource.RLIMIT_AS)[0],
                "n_tools": len(_TOOL_REGISTRY),
                "has_known_tool": "robust_regression" in _TOOL_REGISTRY,
            }}
            """
        )

        assert observed["soft"] == _CAP_MB * 1024 * 1024
        assert observed["n_tools"] > 0
        assert observed["has_known_tool"]

    @_skip_if_not_linux
    def test_get_pool_returns_cached_instance(self) -> None:
        """Repeated get_pool calls with the same memory cap reuse one pool."""
        first = get_pool(memory_mb=256)
        second = get_pool(memory_mb=256)
        assert first is second
