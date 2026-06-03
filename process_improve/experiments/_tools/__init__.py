# (c) Kevin Dunn, 2010-2026. MIT License.
"""Per-tool MCP wrappers for designed experiments (ENG-02).

Each module in this package defines exactly one agent-callable tool: a pydantic
input model, a ``@tool_spec``-decorated wrapper function, and a ``_register(...)``
call. The shared registration primitives live here (rather than in
``experiments.tools``) so the per-tool modules and the ``experiments.tools``
aggregator can both import them without an import cycle.
"""

from __future__ import annotations

import logging

import numpy as np
from patsy import PatsyError

# Keep the logger name byte-identical to the pre-split ``experiments.tools``
# module so log output and any name-based filtering are unaffected.
logger = logging.getLogger("process_improve.experiments.tools")

# Per the ENG-11 error-handling style guide, every tool wrapper narrows
# its ``except`` to this canonical set so that anything *outside* this
# set propagates up to ``mcp_server._serialise_tool_error`` and gets
# redacted before reaching the caller (SEC-18 / #267).
_TOOL_EXPECTED_EXCEPTIONS: tuple[type[BaseException], ...] = (
    ValueError,
    TypeError,
    KeyError,
    np.linalg.LinAlgError,
    PatsyError,
)

_EXPERIMENTS_TOOL_NAMES: list[str] = []


def _register(name: str) -> None:
    _EXPERIMENTS_TOOL_NAMES.append(name)
