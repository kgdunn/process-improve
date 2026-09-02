"""(c) Kevin Dunn, 2010-2026. MIT License.

MCP (Model Context Protocol) server for process-improve.

Exposes all ``@tool_spec``-decorated functions as MCP tools, making
them instantly available to Claude Desktop, Cursor, VS Code Copilot,
and any other MCP-compatible client.

Usage
-----

Run directly::

    python -m process_improve.mcp_server

Or via the installed entry-point::

    process-improve-mcp

Configuration for Claude Desktop (``claude_desktop_config.json``)::

    {
        "mcpServers": {
            "process-improve": {
                "command": "process-improve-mcp"
            }
        }
    }
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.tools import Tool
from mcp.server.mcpserver.utilities.func_metadata import ArgModelBase, FuncMetadata

from process_improve.config import settings
from process_improve.tool_safety import ToolSafetyError, safe_execute_tool_call
from process_improve.tool_spec import discover_tools, execute_tool_call, get_tool_specs

logger = logging.getLogger(__name__)

_INSTRUCTIONS = (
    "Process improvement tools: robust statistics, multivariate analysis (PCA/PLS), "
    "control charts, designed experiments, batch process analysis, and regression. "
    "All tools accept JSON inputs and return JSON outputs."
)

# Opt-in safety. The default (stdio on the user's own machine) keeps the
# fast in-process path so local Claude Desktop / Cursor integrations don't
# pay subprocess overhead. Set ``PROCESS_IMPROVE_MCP_SAFE_MODE=1`` when the
# server is fronted by HTTP or otherwise reachable from untrusted clients.
# Reads via ``settings`` so tests can override at runtime (ENG-09 / ENG-27).


def _serialise_tool_error(exc: Exception, tool_name: str) -> str:
    """Return a JSON error string that does not leak internal detail.

    An unexpected exception's message may carry internal detail (filesystem
    paths, library internals). Over an untrusted MCP transport that is an
    information-disclosure risk, so the full traceback is logged server-side and
    only a generic message is returned to the caller. (Structured
    :class:`ToolSafetyError`s, which have a curated payload, are handled by the
    caller before reaching here.)
    """
    logger.error("Tool %r raised an unexpected error", tool_name, exc_info=exc)
    return json.dumps({"error": "internal error while executing tool", "tool": tool_name})


class _SchemaPassthroughMetadata(FuncMetadata):
    """Hand-built ``FuncMetadata`` that skips pydantic argument validation.

    Each tool's published ``inputSchema`` comes from the ``@tool_spec``
    registry, not from a Python signature, so there is no pydantic argument
    model here to validate against. The arguments are passed through unchanged
    to :func:`process_improve.tool_spec.execute_tool_call`, which validates
    them against the tool's real pydantic input model (unknown keys raise
    ``ToolInputInvalidError`` there; see SEC-15).
    """

    def validate_arguments(self, arguments_to_validate: dict[str, Any]) -> dict[str, Any]:
        """Return a shallow copy of the arguments without validating them.

        Parameters
        ----------
        arguments_to_validate : dict[str, Any]
            The raw ``arguments`` dict from the MCP ``tools/call`` request.

        Returns
        -------
        dict[str, Any]
            The same key/value pairs, handed to the handler as ``**kwargs``.
        """
        return dict(arguments_to_validate)


def _make_tool_handler(tool_name: str) -> Callable[..., Awaitable[str]]:
    """Build the async MCP handler for ``tool_name``.

    ENG-30: the tool execution path is synchronous (in safe mode it blocks on a
    ``ProcessPoolExecutor`` future; otherwise it runs the tool in-process). To
    avoid blocking the MCP server's event loop - which would serialise
    concurrent requests when the server is fronted by HTTP / SSE - the blocking
    call is offloaded to a worker thread via ``run_in_executor``. Single-call
    behaviour is unchanged.
    """

    async def handler(**kwargs: Any) -> str:  # noqa: ANN401
        """Run the registered tool and return its result as a JSON string."""
        loop = asyncio.get_running_loop()
        sync_call = safe_execute_tool_call if settings.mcp_safe_mode else execute_tool_call
        try:
            result = await loop.run_in_executor(None, sync_call, tool_name, kwargs)
            if isinstance(result, dict):
                return json.dumps(result, indent=2, default=str)
            return str(result)
        except ToolSafetyError as exc:
            # Structured safety errors carry a curated, non-sensitive payload.
            return json.dumps(exc.to_dict())
        except Exception as exc:  # noqa: BLE001
            return _serialise_tool_error(exc, tool_name)

    return handler


def _build_tool(spec: dict[str, Any]) -> Tool:
    """Build one MCP ``Tool`` whose published ``inputSchema`` is the registry's schema.

    The tool is constructed directly (not via ``Tool.from_function``) so the
    ``parameters`` field - which the server publishes verbatim as the tool's
    ``inputSchema`` - is exactly ``spec["input_schema"]``: parameter types,
    required vs optional, enums, bounds, and ``anyOf`` unions all survive
    (issue #506). Signature introspection of the generic ``(**kwargs)``
    handler could not synthesise any of that.

    Parameters
    ----------
    spec : dict[str, Any]
        One entry from :func:`process_improve.tool_spec.get_tool_specs`, with
        ``"name"``, ``"description"``, and ``"input_schema"`` keys.

    Returns
    -------
    Tool
        The MCP tool registration object, ready to hand to ``MCPServer``.
    """
    return Tool(
        fn=_make_tool_handler(spec["name"]),
        name=spec["name"],
        title=None,
        description=spec["description"],
        parameters=spec["input_schema"],
        fn_metadata=_SchemaPassthroughMetadata(arg_model=ArgModelBase),
        is_async=True,
        context_kwarg=None,
        annotations=None,
    )


def create_server() -> MCPServer:
    """Create the MCP server with every ``@tool_spec`` tool registered.

    Returns
    -------
    MCPServer
        A server whose ``list_tools()`` publishes, for each registered tool,
        the same ``input_schema`` that
        :func:`process_improve.tool_spec.get_tool_specs` reports.
    """
    discover_tools()
    specs = get_tool_specs()
    logger.info("Registering %d tools with MCP server", len(specs))
    return MCPServer(
        "process-improve",
        instructions=_INSTRUCTIONS,
        tools=[_build_tool(spec) for spec in specs],
    )


def main() -> None:
    """Entry point for the MCP server."""
    create_server().run()


if __name__ == "__main__":
    main()
