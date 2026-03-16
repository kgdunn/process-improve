"""(c) Kevin Dunn, 2010-2025. MIT License.

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

from typing import Any, Callable

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Maps tool name -> decorated callable.  Populated by ``@tool_spec``.
_TOOL_REGISTRY: dict[str, Callable[..., Any]] = {}


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


def tool_spec(
    name: str,
    description: str,
    input_schema: dict[str, Any],
    examples: str = "",
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
        Optional string with one or more natural-language → tool-call mappings
        (plain text, no special format required).  Appended to ``description``
        so the LLM can see worked examples inside the tool spec.

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
        func._tool_spec = spec  # type: ignore[attr-defined]
        _TOOL_REGISTRY[name] = func
        return func

    return decorator


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_tool_specs(names: list[str] | None = None) -> list[dict[str, Any]]:
    """Return tool specs in the format expected by the Anthropic ``tools=`` parameter.

    Parameters
    ----------
    names:
        Optional allow-list of tool names to include.  When *None* (default)
        all registered tools are returned.

    Returns
    -------
    list[dict]
        Each dict has keys ``"name"``, ``"description"``, and
        ``"input_schema"`` as required by the Anthropic API.
    """
    registry = _TOOL_REGISTRY
    if names is not None:
        registry = {k: v for k, v in registry.items() if k in names}
    return [func._tool_spec for func in registry.values()]  # type: ignore[attr-defined]


def execute_tool_call(tool_name: str, tool_input: dict[str, Any]) -> Any:
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
    if tool_name not in _TOOL_REGISTRY:
        available = sorted(_TOOL_REGISTRY)
        raise ValueError(f"Unknown tool {tool_name!r}. Available tools: {available}")
    return _TOOL_REGISTRY[tool_name](**tool_input)
