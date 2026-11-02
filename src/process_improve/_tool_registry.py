"""(c) Kevin Dunn, 2010-2026. MIT License.

The ``@tool_spec`` decorator, the registry it fills, and ``clean``.

These are the parts of :mod:`process_improve.tool_spec` that every ``tools.py``
module imports. They live apart from tool discovery, which names every
``tools.py`` module, so that a tool module depends on this file alone and not,
through discovery, on every other tool module. ``tool_spec.py`` re-exports all
of it, and callers import it from there.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Maps tool name -> decorated callable.  Populated by ``@tool_spec``.
_TOOL_REGISTRY: dict[str, Callable[..., Any]] = {}


# ---------------------------------------------------------------------------
# RNG metadata validation
# ---------------------------------------------------------------------------


def _validate_rng_metadata(name: str, rng: dict[str, Any]) -> None:
    """Reject ``rng`` payloads that don't match the published contract.

    The contract is intentionally narrow so downstream consumers (the
    factorial reproducible-export service) can introspect specs without
    defensive type-checking. See ``CLAUDE.md`` for the full schema.
    """
    if not isinstance(rng, dict):
        raise TypeError(f"@tool_spec(name={name!r}): 'rng' must be a dict, got {type(rng).__name__}.")
    if "uses_rng" not in rng or not isinstance(rng["uses_rng"], bool):
        raise ValueError(f"@tool_spec(name={name!r}): 'rng' must have a boolean 'uses_rng' key.")
    allowed_keys = {"uses_rng", "seed_param", "default_seed", "note"}
    extra = set(rng) - allowed_keys
    if extra:
        raise ValueError(
            f"@tool_spec(name={name!r}): unknown keys in 'rng': {sorted(extra)}. Allowed: {sorted(allowed_keys)}."
        )
    if not rng["uses_rng"]:
        # Deterministic tools shouldn't carry a seed_param / default_seed.
        if rng.get("seed_param") is not None or rng.get("default_seed") is not None:
            raise ValueError(
                f"@tool_spec(name={name!r}): 'seed_param' / 'default_seed' must be omitted when uses_rng is False."
            )
        return
    seed_param = rng.get("seed_param")
    if seed_param is not None and not isinstance(seed_param, str):
        raise TypeError(
            f"@tool_spec(name={name!r}): 'seed_param' must be a string or None, got {type(seed_param).__name__}."
        )
    default_seed = rng.get("default_seed")
    if default_seed is not None and not isinstance(default_seed, int):
        raise TypeError(
            f"@tool_spec(name={name!r}): 'default_seed' must be an int or None, got {type(default_seed).__name__}."
        )
    if seed_param is None and default_seed is not None:
        raise ValueError(f"@tool_spec(name={name!r}): 'default_seed' requires a 'seed_param' name.")


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


def tool_spec(  # noqa: PLR0913
    name: str,
    description: str,
    *,
    input_model: type[BaseModel],
    examples: str = "",
    category: str = "",
    rng: dict[str, Any] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Mark a function as an agent-callable tool.

    Parameters
    ----------
    name:
        Unique tool name exposed to the LLM (snake_case).
    description:
        Natural-language description of what the tool does and when to use it.
    input_model:
        A :class:`pydantic.BaseModel` subclass describing the tool's
        input. The model is the single source of truth for both the
        function's Python signature and the MCP JSON Schema (derived
        via :meth:`BaseModel.model_json_schema`). Every input model
        **must** set ``model_config = ConfigDict(extra="forbid")`` so
        unknown keys are rejected (closes the SEC-15
        kwarg-injection vector at the schema layer).

        The decorated function then receives the parsed model as its
        single positional argument::

            class FooInput(BaseModel):
                model_config = ConfigDict(extra="forbid")
                values: list[float] = Field(..., min_length=2)

            @tool_spec(name="foo", description="...", input_model=FooInput)
            def foo(spec: FooInput) -> dict:
                return {"n": len(spec.values)}
    examples:
        Optional natural-language -> tool-call mappings appended to
        ``description`` so the LLM sees worked examples.
    category:
        Optional category string (e.g. ``"univariate"``). Used for
        filtering with :func:`get_tool_specs`.
    rng:
        Optional reproducibility metadata; see :func:`_validate_rng_metadata`
        for the contract.

    Returns
    -------
    Callable
        The original function, with an attached ``_tool_spec`` dict
        and ``_input_model`` attribute.
    """
    if rng is not None:
        _validate_rng_metadata(name, rng)

    if not (isinstance(input_model, type) and issubclass(input_model, BaseModel)):
        raise TypeError(
            f"@tool_spec(name={name!r}): input_model must be a pydantic BaseModel subclass; got {input_model!r}."
        )
    extra_policy = input_model.model_config.get("extra")
    if extra_policy != "forbid":
        raise ValueError(
            f"@tool_spec(name={name!r}): input_model {input_model.__name__} "
            "must set model_config = ConfigDict(extra='forbid') so unknown "
            "kwargs are rejected (ENG-04 / SEC-15 contract)."
        )

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        """Attach the assembled tool spec to ``func`` and return it unchanged."""
        full_description = description
        if examples:
            full_description = f"{description}\n\nExamples\n--------\n{examples}"

        spec: dict[str, Any] = {
            "name": name,
            "description": full_description,
            "input_schema": input_model.model_json_schema(),
        }
        if category:
            spec["category"] = category
        if rng is not None:
            spec["rng"] = dict(rng)

        func._tool_spec = spec  # type: ignore[attr-defined]
        func._input_model = input_model  # type: ignore[attr-defined]
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
        # Keys need unwrapping too: json.dumps rejects numpy scalar keys (a
        # dict keyed by np.int64 group labels from a pandas groupby is the
        # common case), and previously they leaked through untouched. Only
        # numpy scalars are unwrapped - other key types pass through so
        # hashability is never at risk.
        return {(k.item() if isinstance(k, np.generic) else k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [clean(v) for v in value]
    # np.bool_ subclasses neither np.integer nor Python bool, and json.dumps
    # rejects it; it must be tested before the numeric branches.
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        v = float(value)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        # Catch-all for the remaining numpy scalar types (datetime64,
        # complex, str_, ...): unwrap to the closest Python equivalent.
        return clean(value.item())
    return value
