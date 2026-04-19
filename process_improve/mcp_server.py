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

import json
import logging
import os
from typing import Any

from mcp.server.fastmcp import FastMCP

from process_improve.tool_safety import ToolSafetyError, safe_execute_tool_call
from process_improve.tool_spec import discover_tools, execute_tool_call, get_tool_specs

logger = logging.getLogger(__name__)

# Opt-in safety. The default (stdio on the user's own machine) keeps the
# fast in-process path so local Claude Desktop / Cursor integrations don't
# pay subprocess overhead. Set ``PROCESS_IMPROVE_MCP_SAFE_MODE=1`` when the
# server is fronted by HTTP or otherwise reachable from untrusted clients.
_SAFE_MODE = os.environ.get("PROCESS_IMPROVE_MCP_SAFE_MODE", "0").lower() in {"1", "true", "yes"}

mcp = FastMCP(
    "process-improve",
    instructions=(
        "Process improvement tools: robust statistics, multivariate analysis (PCA/PLS), "
        "control charts, designed experiments, batch process analysis, and regression. "
        "All tools accept JSON inputs and return JSON outputs."
    ),
)


def _register_all_tools() -> None:
    """Register every ``@tool_spec`` tool as an MCP tool."""
    discover_tools()
    specs = get_tool_specs()
    logger.info("Registering %d tools with MCP server", len(specs))

    for spec in specs:
        tool_name = spec["name"]
        tool_description = spec["description"]
        tool_schema = spec["input_schema"]

        # Build parameter list from the JSON schema properties
        _create_mcp_tool(tool_name, tool_description, tool_schema)


def _create_mcp_tool(
    tool_name: str,
    tool_description: str,
    tool_schema: dict[str, Any],
) -> None:
    """Create and register a single MCP tool that delegates to execute_tool_call."""

    # Define an async handler that calls through to our tool registry
    async def handler(**kwargs: Any) -> str:  # noqa: ANN401
        try:
            result = safe_execute_tool_call(tool_name, kwargs) if _SAFE_MODE else execute_tool_call(tool_name, kwargs)
            if isinstance(result, dict):
                return json.dumps(result, indent=2, default=str)
            return str(result)
        except ToolSafetyError as exc:
            return json.dumps(exc.to_dict())
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"error": str(exc)})

    # Set proper function metadata for FastMCP
    handler.__name__ = tool_name
    handler.__doc__ = tool_description
    handler.__qualname__ = tool_name

    # Build parameter annotations from JSON schema for FastMCP introspection
    properties = tool_schema.get("properties", {})
    required = set(tool_schema.get("required", []))

    # Create type annotations mapping
    type_map = {
        "number": float,
        "integer": int,
        "string": str,
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    annotations: dict[str, Any] = {}
    defaults: dict[str, Any] = {}

    for param_name, param_spec in properties.items():
        param_type = param_spec.get("type", "string")
        annotations[param_name] = type_map.get(param_type, Any)
        if param_name not in required:
            # Provide a sentinel default for optional params
            defaults[param_name] = None

    handler.__annotations__ = annotations
    if defaults:
        handler.__defaults__ = tuple(defaults.values())

    # Register with FastMCP using the low-level add_tool
    mcp.tool(name=tool_name, description=tool_description)(handler)


def main() -> None:
    """Entry point for the MCP server."""
    _register_all_tools()
    mcp.run()


if __name__ == "__main__":
    main()
