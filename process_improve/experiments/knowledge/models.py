"""Dataclass models for DOE knowledge graph nodes.

Each class maps directly to one YAML file in the ``data/`` directory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Node types
# ---------------------------------------------------------------------------


@dataclass
class DesignTypeNode:
    """A DOE design type (e.g. full_factorial, plackett_burman, ccd)."""

    id: str
    display_name: str
    category: str  # factorial, screening, response_surface, optimal, mixture
    description: dict[str, str]  # detail_level -> text
    suitable_for: list[str] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)
    min_runs: dict[str, Any] = field(default_factory=dict)
    supports_models: list[str] = field(default_factory=list)
    can_augment_to: list[dict[str, str]] = field(default_factory=list)
    advantages: list[str] = field(default_factory=list)
    disadvantages: list[str] = field(default_factory=list)
    common_misconceptions: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)


@dataclass
class DecisionRuleNode:
    """A context-aware design selection rule."""

    id: str
    topic: str
    description: str
    conditions: list[dict[str, Any]]
    recommend: dict[str, Any]  # primary + alternatives
    explanation: dict[str, str]  # detail_level -> text


@dataclass
class DiagnosticNode:
    """A residual diagnostic pattern (e.g. funnel residuals -> heteroscedasticity)."""

    id: str
    display_name: str
    visual_pattern: str
    indicates: list[dict[str, str]]  # problem + description
    remedies: list[dict[str, str]]  # action + detail_level
    severity: str = "moderate"


@dataclass
class InterpretationGuide:
    """Guidance for interpreting DOE output (ANOVA, plots, model adequacy)."""

    id: str
    topic: str
    title: str
    content: dict[str, str]  # detail_level -> text
    related_questions: list[int] = field(default_factory=list)


@dataclass
class ConceptNode:
    """A statistical concept definition (replication, resolution, etc.)."""

    id: str
    title: str
    topic: str
    content: dict[str, str]  # detail_level -> text
    related_to: list[str] = field(default_factory=list)
    related_questions: list[int] = field(default_factory=list)


@dataclass
class WorkedExample:
    """A worked example demonstrating a DOE concept or workflow."""

    id: str
    title: str
    demonstrates: list[str]  # concept IDs
    topic: str
    scenario: str
    steps: list[str]
    related_questions: list[int] = field(default_factory=list)
    detail_level: str = "intermediate"


# ---------------------------------------------------------------------------
# Top-level container
# ---------------------------------------------------------------------------


@dataclass
class KnowledgeGraph:
    """Container for all loaded knowledge nodes and adjacency indices."""

    design_types: dict[str, DesignTypeNode] = field(default_factory=dict)
    decision_rules: list[DecisionRuleNode] = field(default_factory=list)
    diagnostics: dict[str, DiagnosticNode] = field(default_factory=dict)
    interpretation_guides: dict[str, InterpretationGuide] = field(default_factory=dict)
    concepts: dict[str, ConceptNode] = field(default_factory=dict)
    worked_examples: dict[str, WorkedExample] = field(default_factory=dict)

    # Indices built at load time
    topic_index: dict[str, list[str]] = field(default_factory=dict)
    keyword_index: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    # keyword -> list of (node_type, node_id) tuples
