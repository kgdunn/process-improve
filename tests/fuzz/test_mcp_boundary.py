"""MCP-boundary fuzz suite (ENG-15, acceptance item 4).

Generalises ``test_robust_scale_sn_fuzz`` to *every* registered ``@tool_spec``
tool: for each tool we derive a hypothesis strategy from its published
``input_schema`` and throw schema-shaped (and deliberately ragged) payloads at
the ``execute_tool_call`` dispatch path - exactly what the MCP server does.

The invariant under test is the one that would have caught SEC-14..SEC-21:
**a tool wrapper must never leak an undocumented exception**. Each call must
either return a JSON-serialisable ``dict`` or raise one of the documented
boundary exceptions; anything else means a wrapper's ``except`` clause is too
narrow and an implementation-internal error would reach the caller.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from process_improve.tool_safety import ToolInputInvalidError
from process_improve.tool_spec import discover_tools, execute_tool_call, get_tool_specs

# Documented boundary exceptions. Anything outside this set is a leak (a bug in
# the tool wrapper), per ENG-11 / SEC-18.
_EXPECTED = (ValueError, TypeError, KeyError, ToolInputInvalidError)

_finite_floats = st.floats(allow_nan=False, allow_infinity=False, width=64)

discover_tools()
_SPECS = sorted(get_tool_specs(), key=lambda s: s["name"])


def _scalar_strategy(schema: dict[str, Any]) -> st.SearchStrategy:  # noqa: C901, PLR0911
    """Map a JSON-schema fragment to a small, bounded hypothesis strategy."""
    if "enum" in schema:
        return st.sampled_from(schema["enum"])
    # pydantic emits unions as anyOf/oneOf (e.g. ``int | None``).
    for key in ("anyOf", "oneOf"):
        if key in schema:
            return st.one_of([_scalar_strategy(sub) for sub in schema[key]] or [st.none()])
    json_type = schema.get("type")
    if isinstance(json_type, list):  # ``"type": ["integer", "null"]``
        return st.one_of([_scalar_strategy({**schema, "type": t}) for t in json_type])
    if json_type == "number":
        return _finite_floats
    if json_type == "integer":
        return st.integers(min_value=-1000, max_value=1000)
    if json_type == "string":
        return st.text(max_size=24)
    if json_type == "boolean":
        return st.booleans()
    if json_type == "null":
        return st.none()
    if json_type == "array":
        items = schema.get("items") or {}
        return st.lists(_scalar_strategy(items) if items else _finite_floats, max_size=8)
    if json_type == "object":
        return st.dictionaries(
            st.text(max_size=8),
            st.one_of(_finite_floats, st.integers(-100, 100), st.text(max_size=8), st.booleans()),
            max_size=4,
        )
    # Unknown / untyped: a grab-bag, including the adversarial ``None``.
    return st.one_of(st.none(), _finite_floats, st.integers(-100, 100), st.text(max_size=8), st.booleans())


def _payload_strategy(input_schema: dict[str, Any]) -> st.SearchStrategy:
    """Build a strategy producing payloads with the schema's keys and fuzzed values."""
    props: dict[str, Any] = input_schema.get("properties", {})
    if not props:
        return st.just({})
    return st.fixed_dictionaries({name: _scalar_strategy(sub) for name, sub in props.items()})


@pytest.mark.parametrize("spec", _SPECS, ids=[s["name"] for s in _SPECS])
@given(data=st.data())
@settings(
    max_examples=40,
    derandomize=True,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_tool_never_leaks_unexpected_exception(spec: dict[str, Any], data: st.DataObject) -> None:
    """No schema-shaped payload makes a tool leak an undocumented exception."""
    name = spec["name"]
    payload = data.draw(_payload_strategy(spec.get("input_schema", {})))
    try:
        result = execute_tool_call(name, payload)
    except _EXPECTED:
        return  # documented validation / value failure - acceptable
    except pytest.fail.Exception:  # type: ignore[attr-defined]
        raise  # let hypothesis/pytest failures propagate
    except Exception as exc:  # noqa: BLE001
        pytest.fail(
            f"Tool {name!r} leaked an undocumented {type(exc).__name__}: {exc!r} "
            f"on payload {payload!r}. This would reach the MCP caller; narrow the "
            f"wrapper's except clause (ENG-11 / SEC-18)."
        )
    else:
        assert isinstance(result, dict)
        json.dumps(result, default=str)
