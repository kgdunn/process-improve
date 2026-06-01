"""Fuzz tests for the MCP boundary.

Hypothesis-driven smoke tests that throw schema-conforming random
inputs at each registered tool and assert that no *uncaught*
exception escapes -- only the structured ``ToolSafetyError`` family
plus the tool's own documented exceptions.

Why this matters: ``mcp_server._serialise_tool_error`` redacts
unhandled exceptions, but a single ``except Exception: return
{"error": str(e)}`` inside a per-tool wrapper (SEC-18 / #267)
short-circuits that redaction. A fuzz pass that produces a stack
trace inside a tool wrapper is an information-disclosure
regression even if the tool's "happy path" still works.

Scaffolding landed in ENG-15. The first example uses
``robust_scale_sn`` -- the simplest registered tool (one
parameter, list of numbers). A follow-up generalises this to walk
the whole ``_TOOL_REGISTRY`` using a JSON-Schema -> hypothesis
strategy converter (the hardest part is supporting ``oneOf`` and
nested ``items``).

Run locally with::

    pytest tests/fuzz -o "addopts="
"""
