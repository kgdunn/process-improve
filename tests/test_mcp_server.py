"""ENG-30 (#312): the MCP dispatch path offloads blocking tool execution to a
worker thread, so a slow tool call does not block other calls on the same
(async) server.

The MCP optional dependency is only present with the ``[mcp]`` extra, so these
tests skip when it is not installed.
"""

from __future__ import annotations

import asyncio
import time

import pytest

pytest.importorskip("mcp.server.fastmcp")

from process_improve import mcp_server

_SLEEP = 0.5


def test_single_call_returns_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """A single dispatch returns the tool result serialised as JSON (unchanged behaviour)."""
    monkeypatch.setattr(mcp_server, "execute_tool_call", lambda name, payload: {"name": name, "echo": payload})
    handler = mcp_server._make_tool_handler("echo")

    out = asyncio.run(handler(value=5))

    assert '"name": "echo"' in out
    assert '"value": 5' in out


def test_concurrent_dispatch_does_not_block(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two slow tool calls run concurrently rather than serialising on the event loop."""

    def fake_exec(tool_name: str, _payload: dict) -> dict:
        time.sleep(_SLEEP)
        return {"tool": tool_name, "ok": True}

    monkeypatch.setattr(mcp_server, "execute_tool_call", fake_exec)
    handler = mcp_server._make_tool_handler("slow")

    async def run_two() -> tuple[list[str], float]:
        start = time.perf_counter()
        results = await asyncio.gather(handler(), handler())
        return results, time.perf_counter() - start

    results, elapsed = asyncio.run(run_two())

    assert len(results) == 2
    assert all('"ok": true' in r for r in results)
    # Offloaded to threads, the two ``_SLEEP``-second calls overlap; a blocking
    # (serialised) dispatch would take ~2 * _SLEEP.
    assert elapsed < 1.7 * _SLEEP, f"dispatch appears serialised: {elapsed:.3f}s for two {_SLEEP}s calls"
