"""(c) Kevin Dunn, 2010-2026. MIT License.

Tool-call-first infrastructure for process-improve.

Provides the ``@tool_spec`` decorator, a global registry of all decorated
functions, and helpers for Anthropic-compatible tool-use integrations.

Quick start
-----------

Import the decorated tools and pass the specs to the Anthropic client::

    import anthropic
    from process_improve.tool_spec import get_tool_specs, execute_tool_call

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        tools=get_tool_specs(),
        messages=[{"role": "user", "content": "Are there outliers in [1,2,3,100]?"}],
    )

    # Dispatch tool calls from the response
    for block in response.content:
        if block.type == "tool_use":
            result = execute_tool_call(block.name, block.input)
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Maps tool name -> decorated callable.  Populated by ``@tool_spec``.
_TOOL_REGISTRY: dict[str, Callable[..., Any]] = {}

#: Whether ``discover_tools()`` has already run.
_discovery_done: bool = False


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


def tool_spec(
    name: str,
    description: str,
    input_schema: dict[str, Any],
    examples: str = "",
    category: str = "",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Mark a function as an agent-callable tool.

    Parameters
    ----------
    name:
        Unique tool name exposed to the LLM (snake_case).
    description:
        Natural-language description of what the tool does and when to use it.
        Good descriptions contain:
        * What the tool computes.
        * What the inputs represent.
        * When *not* to use it (if applicable).
    input_schema:
        A dict with a single key ``"json"`` whose value is a valid `JSON Schema
        <https://json-schema.org>`_ object describing the tool's parameters.
        This mirrors the Anthropic ``input_schema`` field directly.
    examples:
        Optional string with one or more natural-language -> tool-call mappings
        (plain text, no special format required).  Appended to ``description``
        so the LLM can see worked examples inside the tool spec.
    category:
        Optional category string (e.g. ``"univariate"``, ``"multivariate"``).
        Used for filtering with :func:`get_tool_specs`.

    Returns
    -------
    Callable
        The original function, unchanged except for an attached ``_tool_spec``
        attribute and registration in :data:`_TOOL_REGISTRY`.

    Examples
    --------
    ::

        @tool_spec(
            name="add_numbers",
            description="Add two numbers together.",
            input_schema={"json": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            }},
            examples='# "What is 2 + 3?" -> ``add_numbers(a=2, b=3)``',
            category="math",
        )
        def add_numbers(*, a: float, b: float) -> dict:
            return {"result": a + b}
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        full_description = description
        if examples:
            full_description = f"{description}\n\nExamples\n--------\n{examples}"

        spec: dict[str, Any] = {
            "name": name,
            "description": full_description,
            "input_schema": input_schema["json"],
        }
        if category:
            spec["category"] = category

        func._tool_spec = spec  # type: ignore[attr-defined]
        _TOOL_REGISTRY[name] = func
        return func

    return decorator


# ---------------------------------------------------------------------------
# Serialisation helper
# ---------------------------------------------------------------------------


def clean(value: Any) -> Any:  # noqa: PLR0911, ANN401
    """Recursively convert numpy scalars / arrays to plain Python types.

    All ``tools.py`` modules should call ``clean(result)`` before returning
    so that every tool output is JSON-serialisable.
    """
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        v = float(value)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    return value


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_tools() -> None:
    """Import all ``tools.py`` modules to populate the tool registry.

    This is called lazily on the first :func:`get_tool_specs` invocation.
    It is safe to call multiple times (subsequent calls are no-ops).
    """
    global _discovery_done  # noqa: PLW0603
    if _discovery_done:
        return

    import contextlib  # noqa: PLC0415
    import importlib  # noqa: PLC0415

    for module in [
        "process_improve.univariate.tools",
        "process_improve.multivariate.tools",
        "process_improve.monitoring.tools",
        "process_improve.regression.tools",
        "process_improve.bivariate.tools",
        "process_improve.experiments.tools",
        "process_improve.batch.tools",
    ]:
        with contextlib.suppress(ImportError):
            importlib.import_module(module)

    _discovery_done = True


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_tool_specs(
    names: list[str] | None = None,
    category: str | None = None,
) -> list[dict[str, Any]]:
    """Return tool specs in the format expected by the Anthropic ``tools=`` parameter.

    Parameters
    ----------
    names:
        Optional allow-list of tool names to include.  When *None* (default)
        all registered tools are returned.
    category:
        Optional category filter (e.g. ``"univariate"``).  When provided, only
        tools whose ``category`` matches are returned.

    Returns
    -------
    list[dict]
        Each dict has keys ``"name"``, ``"description"``, and
        ``"input_schema"`` as required by the Anthropic API.
    """
    discover_tools()
    registry = _TOOL_REGISTRY
    if names is not None:
        registry = {k: v for k, v in registry.items() if k in names}
    specs = [func._tool_spec for func in registry.values()]  # type: ignore[attr-defined]
    if category is not None:
        specs = [s for s in specs if s.get("category") == category]
    return specs


def execute_tool_call(tool_name: str, tool_input: dict[str, Any]) -> Any:  # noqa: ANN401
    """Dispatch a single tool call from an Anthropic ``tool_use`` content block.

    Parameters
    ----------
    tool_name:
        The ``name`` field from the ``tool_use`` block.
    tool_input:
        The ``input`` dict from the ``tool_use`` block.

    Returns
    -------
    Any
        Whatever the tool function returns (typically a JSON-serialisable
        ``dict``).

    Raises
    ------
    ValueError
        If *tool_name* is not in the registry.
    """
    discover_tools()
    if tool_name not in _TOOL_REGISTRY:
        available = sorted(_TOOL_REGISTRY)
        raise ValueError(f"Unknown tool {tool_name!r}. Available tools: {available}")
    return _TOOL_REGISTRY[tool_name](**tool_input)
