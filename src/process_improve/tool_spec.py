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

import contextlib
import logging
import sys
import threading
import warnings
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ValidationError

from process_improve._tool_registry import _TOOL_REGISTRY, clean, tool_spec

logger = logging.getLogger(__name__)

#: Serialises ``warnings.catch_warnings``, which swaps module-global state and is not
#: thread-safe: two tool calls on different threads could otherwise restore each other's
#: filters. A build with context-aware warnings (``sys.flags.context_aware_warnings``,
#: Python 3.14) keeps that state per thread and needs no lock.
_WARNINGS_LOCK = threading.Lock()

#: Whether ``discover_tools()`` has already run.
_discovery_done: bool = False


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _import_tool_module(module: str) -> None:
    """Import a single ``tools.py`` module for discovery.

    A genuinely missing module - typically an uninstalled optional third-party
    dependency that the tools module imports - is tolerated but logged, so the
    dropped tool category is visible rather than silent. Any other
    :class:`ImportError` (for example a bad name imported inside the module) is a
    real bug and is allowed to propagate rather than being silently swallowed,
    which would make the whole tool category vanish without a trace.
    """
    import importlib  # noqa: PLC0415

    try:
        importlib.import_module(module)
    except ModuleNotFoundError as exc:
        # ModuleNotFoundError is raised for ANY unresolvable module path,
        # including a typo inside the tools module itself or a renamed
        # internal module after a refactor. Only tolerate it when the module
        # that is actually missing (exc.name) is a third-party dependency;
        # a missing first-party module is a real bug and must propagate,
        # exactly as the docstring above promises.
        missing = exc.name or ""
        if missing.split(".")[0] == "process_improve":
            raise
        logger.warning("Tool module %r not loaded (missing dependency %r): %s", module, missing, exc)


def discover_tools() -> None:
    """Import all ``tools.py`` modules to populate the tool registry.

    This is called lazily on the first :func:`get_tool_specs` invocation.
    It is safe to call multiple times (subsequent calls are no-ops).
    """
    global _discovery_done  # noqa: PLW0603
    if _discovery_done:
        return

    for module in [
        "process_improve.univariate.tools",
        "process_improve.multivariate.tools",
        "process_improve.monitoring.tools",
        "process_improve.regression.tools",
        "process_improve.bivariate.tools",
        "process_improve.experiments.tools",
        "process_improve.batch.tools",
        "process_improve.visualization.tools",
        "process_improve.simulation.tools",
        "process_improve.sensory.tools",
        "process_improve.recipes",
        "process_improve.workbench.tools",
    ]:
        _import_tool_module(module)

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
        ``"input_schema"`` as required by the Anthropic API.  Tools that
        opt in via ``rng=`` on the decorator also carry an ``"rng"`` key
        describing their reproducibility contract; see :func:`tool_spec`.
    """
    discover_tools()
    registry = _TOOL_REGISTRY
    if names is not None:
        registry = {k: v for k, v in registry.items() if k in names}
    specs = [func._tool_spec for func in registry.values()]  # type: ignore[attr-defined]
    if category is not None:
        specs = [s for s in specs if s.get("category") == category]
    return specs


#: Packages whose own warnings are numerical noise to a caller of the tools, such as a saturated
#: fit's division by zero; the library says what they mean in its own warnings. The browser app
#: leaves out the same packages (``web/bootstrap.py``).
_FOREIGN_PACKAGES = ("numpy", "scipy", "statsmodels", "pandas", "patsy", "sklearn")


def _foreign(filename: str) -> bool:
    """Say whether a warning was raised inside one of the numerical packages."""
    path = filename.replace("\\", "/")
    return any(f"/{package}/" in path for package in _FOREIGN_PACKAGES)


def _call_recording_warnings(func: Callable[..., Any], parsed: BaseModel) -> tuple[Any, list[warnings.WarningMessage]]:
    """Run ``func(parsed)``; return its result and each warning it raised, once, in the order raised.

    Warnings raised inside numpy, scipy, statsmodels, pandas, patsy or scikit-learn are left out.
    """
    context_aware = getattr(sys.flags, "context_aware_warnings", False)
    guard = contextlib.nullcontext() if context_aware else _WARNINGS_LOCK
    with guard, warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = func(parsed)
    unique: dict[tuple[type[Warning], str], warnings.WarningMessage] = {}
    for record in caught:
        if not _foreign(record.filename):
            unique.setdefault((record.category, str(record.message)), record)
    return result, list(unique.values())


def execute_tool_call(tool_name: str, tool_input: dict[str, Any]) -> Any:  # noqa: ANN401
    """Dispatch a single tool call from an Anthropic ``tool_use`` content block.

    The input dict is validated via
    ``input_model.model_validate(tool_input)``. Unknown keys raise
    ``ToolInputInvalidError`` (closes the SEC-15 ``confirmed=True``
    kwarg-injection at the schema layer). The parsed pydantic model is
    passed to the tool function as a single positional argument.

    A warning the tool raises would otherwise reach only the server's stderr, which a
    model calling the tool never reads. Each one is returned in the result instead,
    under ``library_warnings`` as ``{"category", "message"}`` records, once each; the
    numerical packages' own warnings, such as a division by zero, are left out. A
    tool whose result is not a dict has its warnings raised again, as they were. The
    warnings are recorded under a lock, so tool calls on different threads run one at
    a time, except on a build with context-aware warnings.

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
        ``dict``), with ``library_warnings`` added to a dict when the tool
        raised any warning.

    Raises
    ------
    ValueError
        If *tool_name* is not in the registry.
    ToolInputInvalidError
        If the input fails pydantic validation.
    """
    discover_tools()
    if tool_name not in _TOOL_REGISTRY:
        available = sorted(_TOOL_REGISTRY)
        raise ValueError(f"Unknown tool {tool_name!r}. Available tools: {available}")
    func = _TOOL_REGISTRY[tool_name]
    model_cls: type[BaseModel] = func._input_model  # type: ignore[attr-defined]
    try:
        parsed = model_cls.model_validate(tool_input)
    except ValidationError as exc:
        raise ToolInputInvalidError(
            f"Input to tool {tool_name!r} failed validation: {exc}",
            details={"tool_name": tool_name, "errors": exc.errors()},
        ) from exc
    result, raised = _call_recording_warnings(func, parsed)
    if raised and isinstance(result, dict):
        records = [{"category": w.category.__name__, "message": str(w.message)} for w in raised]
        return {**result, "library_warnings": records}
    for w in raised:
        warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
    return result


# ---------------------------------------------------------------------------
# Safety wrapper re-exports
# ---------------------------------------------------------------------------
# Callers that expose the registry over an untrusted transport should use
# ``safe_execute_tool_call`` from ``process_improve.tool_safety`` instead of
# ``execute_tool_call``. The names are re-exported here for discoverability.


from process_improve.tool_safety import (  # noqa: E402
    ToolInputInvalidError,
    ToolInputTooLargeError,
    ToolMemoryExceededError,
    ToolSafetyError,
    ToolTimeoutError,
    safe_execute_tool_call,
)

__all__ = [
    "ToolInputInvalidError",
    "ToolInputTooLargeError",
    "ToolMemoryExceededError",
    "ToolSafetyError",
    "ToolTimeoutError",
    "clean",
    "discover_tools",
    "execute_tool_call",
    "get_tool_specs",
    "safe_execute_tool_call",
    "tool_spec",
]
