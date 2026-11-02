"""(c) Kevin Dunn, 2010-2026. MIT License.

Reusable analysis-recipe framework.

An *analysis recipe* is a predefined, step-by-step workflow an LLM agent can
follow when a user's request matches a known analytical scenario (intake,
panel processing, relating to covariates, ...). Recipes reference existing agent
tools by name so the agent chains calls deterministically instead of improvising
the order.

The framework is package-wide and domain-agnostic: any subpackage may define its
own recipes and register them. The sensory subpackage is the first consumer (see
:mod:`process_improve.sensory.recipes`).

Adding recipes for a subpackage
-------------------------------
1. Create ``process_improve/<subpackage>/recipes.py``.
2. Build :class:`AnalysisRecipe` instances and pass each to
   :func:`register_recipe`.
3. Add ``"process_improve.<subpackage>.recipes"`` to ``_RECIPE_MODULES`` below so
   :func:`discover_recipes` imports it.

No changes to the agent tool layer are required: the single, general
``select_analysis_recipe`` tool matches across every registered recipe.
"""

from __future__ import annotations

import importlib
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from process_improve.tool_spec import clean, tool_spec

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecipeStep:
    """One step in an analysis recipe the agent should execute.

    ``tools`` is stored as a tuple, whatever sequence was passed in, and ``arg_hints`` as a
    copy of the mapping passed in.

    Attributes
    ----------
    order : int
        1-based position of the step in the recipe.
    directive : str
        Natural-language instruction telling the agent what to do.
    tools : tuple of str
        Names of agent tools this step may call (empty for prose-only steps
        such as interpretation or data assembly).
    arg_hints : dict
        Optional ``{parameter: "where the value comes from"}`` hints, for
        example ``{"score_min": "0", "mode": "observational"}``.
    """

    order: int
    directive: str
    tools: Sequence[str] = ()
    # shortcut: a dict, which frozen=True does not protect, because a read-only mappingproxy cannot be
    # pickled or deep-copied; use an immutable mapping if a caller comes to rely on it. A dict has no
    # hash, so it is left out of the hash (equal steps still hash equal).
    arg_hints: dict[str, str] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        """Store ``tools`` as a tuple and ``arg_hints`` as a copy, so no caller's list or dict is shared."""
        object.__setattr__(self, "tools", tuple(self.tools))
        object.__setattr__(self, "arg_hints", dict(self.arg_hints))


@dataclass(frozen=True)
class AnalysisRecipe:
    """A reusable, multi-step analysis workflow for the agent.

    The recipe is hashable, and its sequences cannot be changed in place: each is stored as a
    tuple, whatever sequence was passed in.

    Attributes
    ----------
    key : str
        Unique snake_case identifier.
    title : str
        Human-readable name.
    summary : str
        One-paragraph description of what the recipe does and when to use it.
    domain : str
        The subpackage the recipe belongs to (e.g. ``"sensory"``).
    cue_phrases : tuple of str
        Lower-case substrings; each one found in a user's request scores the
        recipe one point during matching.
    inputs_needed : tuple of str
        What the agent must resolve from the user before running, each with a
        short example.
    stages : tuple of RecipeStep
        The ordered steps. Empty for a planned (not yet available) recipe.
    status : str
        ``"available"`` (default) or ``"planned"`` for parked future work.
    """

    key: str
    title: str
    summary: str
    domain: str
    cue_phrases: Sequence[str]
    inputs_needed: Sequence[str]
    stages: Sequence[RecipeStep]
    status: str = "available"
    #: ``cue_phrases`` in the form :func:`select_recipe` compares, computed once here rather than per query.
    _canonical_cues: tuple[str, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Store each sequence as a tuple, and canonicalise the cue phrases once."""
        for name in ("cue_phrases", "inputs_needed", "stages"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "_canonical_cues", tuple(_canonicalise(phrase) for phrase in self.cue_phrases))

    def to_payload(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dict the agent can consume."""
        payload: dict[str, Any] = {
            "recipe_key": self.key,
            "title": self.title,
            "summary": self.summary,
            "domain": self.domain,
            "status": self.status,
            "inputs_needed": list(self.inputs_needed),
            "stages": [
                {
                    "step": s.order,
                    "directive": s.directive,
                    **({"tools": list(s.tools)} if s.tools else {}),
                    **({"arg_hints": dict(s.arg_hints)} if s.arg_hints else {}),
                }
                for s in self.stages
            ],
        }
        if self.status != "available":
            payload["note"] = (
                "This recipe is planned and not yet runnable; tell the user the workflow is "
                "coming rather than attempting it."
            )
        return payload


# ---------------------------------------------------------------------------
# Registry and discovery (mirrors process_improve.tool_spec)
# ---------------------------------------------------------------------------

#: Maps recipe key -> registered recipe. Populated by ``register_recipe``.
_RECIPE_REGISTRY: dict[str, AnalysisRecipe] = {}

#: Whether ``discover_recipes()`` has already run.
_recipes_discovered: bool = False

#: Subpackage recipe modules imported by ``discover_recipes``. Append new
#: ``process_improve.<subpackage>.recipes`` modules here as they are added.
_RECIPE_MODULES: list[str] = [
    "process_improve.sensory.recipes",
    "process_improve.simulation.recipes",
]


def register_recipe(recipe: AnalysisRecipe) -> AnalysisRecipe:
    """Register *recipe* in the global catalog and return it.

    Raises
    ------
    ValueError
        If a recipe with the same ``key`` is already registered.
    """
    if recipe.key in _RECIPE_REGISTRY:
        raise ValueError(f"A recipe with key {recipe.key!r} is already registered.")
    _RECIPE_REGISTRY[recipe.key] = recipe
    return recipe


def _import_recipe_module(module: str) -> None:
    """Import one subpackage recipe module, tolerating a missing dependency."""
    try:
        importlib.import_module(module)
    except ModuleNotFoundError as exc:
        logger.warning("Recipe module %r not loaded (missing dependency): %s", module, exc)


def discover_recipes() -> None:
    """Import every subpackage recipe module to populate the registry.

    Called lazily by the public query helpers. Safe to call repeatedly
    (subsequent calls are no-ops). A genuinely missing module is logged rather
    than raised, mirroring :func:`process_improve.tool_spec.discover_tools`.
    """
    global _recipes_discovered  # noqa: PLW0603
    if _recipes_discovered:
        return
    for module in _RECIPE_MODULES:
        _import_recipe_module(module)
    _recipes_discovered = True


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

_WORD_PATTERN = re.compile(r"\w+")


def _canonicalise(text: str) -> str:
    """Lower-case and collapse to single-spaced word tokens."""
    return " ".join(_WORD_PATTERN.findall(text.lower()))


def select_recipe(query: str) -> AnalysisRecipe | None:
    """Return the best-matching recipe for *query*, or ``None``.

    Scoring is intentionally simple: each cue phrase that appears as a substring
    of the canonicalised query contributes one point. The highest-scoring recipe
    wins (ties broken by registration order); a minimum score of one is required
    to match. Replace with embedding similarity if the catalog grows beyond a few
    dozen recipes.
    """
    discover_recipes()
    canonical = _canonicalise(query)
    best: AnalysisRecipe | None = None
    best_score = 0
    for recipe in _RECIPE_REGISTRY.values():
        score = sum(1 for phrase in recipe._canonical_cues if phrase in canonical)
        if score > best_score:
            best_score = score
            best = recipe
    return best


def list_recipes() -> list[AnalysisRecipe]:
    """Return every registered recipe (registration order)."""
    discover_recipes()
    return list(_RECIPE_REGISTRY.values())


def get_recipe(key: str) -> AnalysisRecipe | None:
    """Return the registered recipe with *key*, or ``None``."""
    discover_recipes()
    return _RECIPE_REGISTRY.get(key)


# ---------------------------------------------------------------------------
# Agent tool
# ---------------------------------------------------------------------------


class _RecipeQuery(BaseModel):
    """Input contract for ``select_analysis_recipe``."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        ...,
        min_length=1,
        description="The user's request in natural language; the best-matching analysis recipe is returned.",
    )


@tool_spec(
    name="select_analysis_recipe",
    description=(
        "Match a free-text request to a predefined, step-by-step analysis recipe (a guided workflow that "
        "chains existing process-improve tools in the right order). Use this first when a user asks an "
        "open-ended 'how do I analyse this' question, then follow the returned stages. "
        "Returns: {ok: true, matched: bool, recipe, available}. 'recipe' is the matched recipe payload "
        "(recipe_key, title, summary, domain, status, inputs_needed, and ordered 'stages' each with a "
        "directive plus optional tools/arg_hints; a planned recipe also carries a 'note') or null when "
        "nothing matches. 'available' always lists every registered recipe as {recipe_key, title, "
        "summary, domain, status} so you can offer a choice even when matched is false."
    ),
    input_model=_RecipeQuery,
    category="recipes",
)
def select_analysis_recipe(spec: _RecipeQuery) -> dict:
    """Return the best-matching recipe payload plus the full catalogue."""
    match = select_recipe(spec.query)
    available = [
        {
            "recipe_key": r.key,
            "title": r.title,
            "summary": r.summary,
            "domain": r.domain,
            "status": r.status,
        }
        for r in list_recipes()
    ]
    return clean(
        {
            "ok": True,
            "matched": match is not None,
            "recipe": match.to_payload() if match is not None else None,
            "available": available,
        }
    )


__all__ = [
    "AnalysisRecipe",
    "RecipeStep",
    "get_recipe",
    "list_recipes",
    "register_recipe",
    "select_analysis_recipe",
    "select_recipe",
]
