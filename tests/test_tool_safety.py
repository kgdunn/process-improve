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

import sys
import time

import numpy as np
import pytest

from process_improve.tool_safety import (
    ToolInputInvalidError,
    ToolInputTooLargeError,
    ToolMemoryExceededError,
    ToolSafetyError,
    ToolTimeoutError,
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


@tool_spec(
    name="_safety_test_echo",
    description="Return the input unchanged. Test-only.",
    input_schema={"json": {"type": "object", "properties": {"value": {"type": "number"}}, "required": ["value"]}},
)
def _safety_test_echo(*, value: float) -> dict:
    return {"value": value}


@tool_spec(
    name="_safety_test_sleep",
    description="Sleep for the given number of seconds. Test-only.",
    input_schema={"json": {"type": "object", "properties": {"seconds": {"type": "number"}}, "required": ["seconds"]}},
)
def _safety_test_sleep(*, seconds: float) -> dict:
    time.sleep(seconds)
    return {"slept": seconds}


@tool_spec(
    name="_safety_test_memory_bomb",
    description="Allocate the given number of megabytes. Test-only.",
    input_schema={
        "json": {"type": "object", "properties": {"megabytes": {"type": "integer"}}, "required": ["megabytes"]},
    },
)
def _safety_test_memory_bomb(*, megabytes: int) -> dict:
    # Allocate a NumPy array to force a real RSS increase.
    size = int(megabytes) * 1024 * 1024 // 8
    _ = np.zeros(size, dtype=np.float64)
    return {"allocated_mb": megabytes}


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

    def test_timeout_raises_structured_error(self) -> None:
        with pytest.raises(ToolTimeoutError) as exc_info:
            safe_execute_tool_call("_safety_test_sleep", {"seconds": 5}, timeout=0.2)
        assert exc_info.value.code == "timeout"
        assert exc_info.value.details["tool_name"] == "_safety_test_sleep"

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
