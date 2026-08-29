"""Tests for the MCP server (``process_improve.mcp_server``).

Covers two areas:

- ENG-30 (#312): the MCP dispatch path offloads blocking tool execution to a
  worker thread, so a slow tool call does not block other calls on the same
  (async) server.
- #506: the server must publish, for every registered tool, exactly the
  ``input_schema`` that ``get_tool_specs()`` reports; signature introspection
  of the generic ``(**kwargs)`` handler used to discard it entirely.

The MCP optional dependency is only present with the ``[mcp]`` extra
(``uv sync --dev --all-extras`` installs it), so these tests skip when it is
not installed.
"""

from __future__ import annotations

import asyncio
import json
import threading

import pytest

pytest.importorskip("mcp.server.mcpserver")

from process_improve import mcp_server
from process_improve.tool_safety import ToolTimeoutError
from process_improve.tool_spec import get_tool_specs

_SLEEP = 0.5


# ---------------------------------------------------------------------------
# Handler dispatch (ENG-30)
# ---------------------------------------------------------------------------


def test_single_call_returns_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """A single dispatch returns the tool result serialised as JSON (unchanged behaviour)."""
    monkeypatch.setattr(mcp_server, "execute_tool_call", lambda name, payload: {"name": name, "echo": payload})
    handler = mcp_server._make_tool_handler("echo")

    out = asyncio.run(handler(value=5))

    assert '"name": "echo"' in out
    assert '"value": 5' in out


def test_non_dict_result_is_stringified(monkeypatch: pytest.MonkeyPatch) -> None:
    """A tool returning a non-dict is passed through ``str()``, not ``json.dumps``."""
    monkeypatch.setattr(mcp_server, "execute_tool_call", lambda _name, _payload: 42)
    handler = mcp_server._make_tool_handler("scalar")

    assert asyncio.run(handler()) == "42"


def test_concurrent_dispatch_does_not_block(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two slow tool calls run concurrently rather than serialising on the event loop.

    Drive-by robustness fix from #513: the previous version asserted on wall
    clock (``elapsed < 1.7 * _SLEEP``), which flakes on loaded CI runners. A
    barrier both calls must reach while in flight proves the overlap directly,
    with no timing race: a serialised dispatch never has two calls in flight,
    so the barrier times out and raises ``BrokenBarrierError``.
    """
    barrier = threading.Barrier(2, timeout=20 * _SLEEP)

    def fake_exec(tool_name: str, _payload: dict) -> dict:
        barrier.wait()  # Blocks until BOTH calls are in flight at once.
        return {"tool": tool_name, "ok": True}

    monkeypatch.setattr(mcp_server, "execute_tool_call", fake_exec)
    handler = mcp_server._make_tool_handler("slow")

    async def run_two() -> list[str]:
        return await asyncio.gather(handler(), handler())

    results = asyncio.run(run_two())

    assert len(results) == 2
    assert all('"ok": true' in r for r in results)


def test_tool_safety_error_returns_curated_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``ToolSafetyError`` is serialised via its curated ``to_dict()`` payload."""

    def raise_timeout(_name: str, _payload: dict) -> dict:
        raise ToolTimeoutError("tool call timed out", details={"timeout_s": 10})

    monkeypatch.setattr(mcp_server, "execute_tool_call", raise_timeout)
    handler = mcp_server._make_tool_handler("slow")

    out = json.loads(asyncio.run(handler()))

    assert out == {"error": "timeout", "message": "tool call timed out", "details": {"timeout_s": 10}}


def test_unexpected_error_is_not_leaked(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unexpected exception yields a generic message, not the exception's own text."""

    def boom(_name: str, _payload: dict) -> dict:
        msg = "secret /internal/path detail"
        raise RuntimeError(msg)

    monkeypatch.setattr(mcp_server, "execute_tool_call", boom)
    handler = mcp_server._make_tool_handler("fragile")

    out = json.loads(asyncio.run(handler()))

    assert out == {"error": "internal error while executing tool", "tool": "fragile"}


def test_safe_mode_routes_through_safe_execute(monkeypatch: pytest.MonkeyPatch) -> None:
    """With ``settings.mcp_safe_mode`` on, dispatch goes through ``safe_execute_tool_call``."""
    calls: list[str] = []

    def fake_safe(name: str, _payload: dict) -> dict:
        calls.append(name)
        return {"safe": True}

    monkeypatch.setattr(mcp_server, "safe_execute_tool_call", fake_safe)
    monkeypatch.setattr(mcp_server.settings, "mcp_safe_mode", True)
    handler = mcp_server._make_tool_handler("guarded")

    out = asyncio.run(handler())

    assert calls == ["guarded"]
    assert '"safe": true' in out


# ---------------------------------------------------------------------------
# Schema publication (#506)
# ---------------------------------------------------------------------------


def test_published_schemas_match_registry_exactly() -> None:
    """Every registered tool's published ``inputSchema`` equals the registry's ``input_schema``.

    This is the acceptance criterion of #506: the schema an MCP client sees
    (``tools/list``) must be the one ``get_tool_specs()`` computed from the
    tool's pydantic input model - types, required vs optional, enums, bounds,
    and descriptions included - not an empty ``(**kwargs)`` schema.
    """
    server = mcp_server.create_server()
    published = {tool.name: tool for tool in asyncio.run(server.list_tools())}
    specs = {spec["name"]: spec for spec in get_tool_specs()}

    assert set(published) == set(specs)
    assert len(specs) > 0
    for name, spec in specs.items():
        assert published[name].input_schema == spec["input_schema"], f"schema mismatch for tool {name!r}"
        assert published[name].description == spec["description"], f"description mismatch for tool {name!r}"


def test_optional_and_union_parameters_survive() -> None:
    """``anyOf`` unions and the required/optional split reach the published schemas intact.

    Pydantic emits ``anyOf`` (with no top-level ``"type"`` key) for unions such
    as ``int | None``; the old introspection-based registration collapsed every
    such parameter to a bare string. Assert the published schemas still carry
    them, so this test fails if registration ever regresses to introspection.
    """
    server = mcp_server.create_server()
    published = [tool.input_schema for tool in asyncio.run(server.list_tools())]

    n_with_any_of = sum(1 for schema in published if "anyOf" in json.dumps(schema))
    assert n_with_any_of > 0, "expected at least one published schema with an anyOf union"

    n_with_required = sum(1 for schema in published if schema.get("required"))
    assert n_with_required > 0, "expected at least one published schema with required parameters"


def test_call_tool_dispatches_raw_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real server call path hands the request arguments to ``execute_tool_call`` unchanged.

    Registration bypasses pydantic argument-model validation (the registry's
    own ``execute_tool_call`` validates against the tool's real input model),
    so the arguments must reach the dispatcher exactly as the client sent them.
    """
    seen: dict[str, object] = {}

    def fake_exec(name: str, payload: dict) -> dict:
        seen["name"] = name
        seen["payload"] = payload
        return {"ok": True}

    monkeypatch.setattr(mcp_server, "execute_tool_call", fake_exec)
    server = mcp_server.create_server()
    tool_name = get_tool_specs()[0]["name"]
    arguments = {"alpha": 1, "beta": [1.0, None, "x"], "gamma": {"nested": True}}

    result = asyncio.run(server.call_tool(tool_name, arguments))

    assert seen == {"name": tool_name, "payload": arguments}
    assert json.loads(result.content[0].text) == {"ok": True}


def test_main_runs_created_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """``main()`` creates the server and hands control to its ``run()`` loop."""
    ran: list[bool] = []

    class _FakeServer:
        def run(self) -> None:
            ran.append(True)

    monkeypatch.setattr(mcp_server, "create_server", _FakeServer)
    mcp_server.main()

    assert ran == [True]
