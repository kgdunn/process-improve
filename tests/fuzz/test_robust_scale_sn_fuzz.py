"""Fuzz the simplest MCP tool, as a template for the rest.

``robust_scale_sn`` accepts a single parameter (``values: list of
numbers``). We throw arbitrary float lists at it via the
``execute_tool_call`` dispatch path (matching what the MCP server
would do) and assert:

1. The call returns a dict that JSON-serialises cleanly.
2. Or it raises one of the documented exception types -- never an
   uncaught ``Exception`` (which would carry library-internal
   strings into the MCP response; see SEC-18 / #267).

When generalising to every registered tool, walk
``_TOOL_REGISTRY``, derive a hypothesis strategy from each tool's
``input_schema``, and apply the same assertion.
"""

from __future__ import annotations

import json

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from process_improve.tool_spec import execute_tool_call

# The documented "expected" exception classes for a tool wrapper.
# Anything outside this set means a wrapper either (a) leaks an
# implementation exception or (b) raises something undocumented
# -- both are bugs.
_EXPECTED = (ValueError, TypeError, KeyError)


_finite_floats = st.floats(allow_nan=False, allow_infinity=False, width=64)


@given(values=st.lists(_finite_floats, min_size=0, max_size=200))
@settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
def test_robust_scale_sn_does_not_leak_unexpected_exceptions(values: list[float]) -> None:
    """No matter what finite-float list we send, the dispatch path
    returns either a JSON-serialisable result or a documented error.
    """
    try:
        result = execute_tool_call("robust_scale_sn", {"values": values})
    except _EXPECTED:
        # Documented validation failure. Acceptable.
        return
    except pytest.fail.Exception:  # type: ignore[attr-defined]
        # Defensive: hypothesis-internal failure should propagate.
        raise
    except Exception as exc:  # noqa: BLE001
        pytest.fail(
            f"robust_scale_sn raised an undocumented exception "
            f"{type(exc).__name__}: {exc!r} on input length "
            f"{len(values)}. This would leak through "
            f"_serialise_tool_error and into an MCP response. "
            f"Narrow the tool wrapper's except clause."
        )
    else:
        # Reachable only when execute_tool_call returned cleanly; ``result``
        # is guaranteed bound here. The ``else`` clause also keeps CodeQL
        # from flagging it as potentially uninitialized.
        assert isinstance(result, dict)
        # ``clean()`` should already have stripped numpy types, so this
        # round-trip is the strongest assertion we can cheaply make.
        json.dumps(result)
