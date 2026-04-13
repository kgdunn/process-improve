# (c) Kevin Dunn, 2010-2026. MIT License.

"""Pydantic models for the DOE strategy recommender.

Defines the input specification (``DOEProblemSpec``), the output
(``ExperimentalStrategy``, ``ExperimentalStage``, ``TransitionRule``),
and supporting types (``DomainType``, ``PriorKnowledge``).
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from process_improve.experiments.factor import Constraint, Factor, Response


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DomainType(str, Enum):
    """Application domain for domain-specific strategy adjustments."""

    pharma_formulation = "pharma_formulation"
    fermentation = "fermentation"
    food_science = "food_science"
    extraction = "extraction"
    analytical_method = "analytical_method"
    cell_culture = "cell_culture"
    bioprocess = "bioprocess"
    general = "general"


# ---------------------------------------------------------------------------
# Prior knowledge
# ---------------------------------------------------------------------------


class PriorKnowledge(BaseModel):
    """Parsed prior knowledge with a confidence score.

    Parameters
    ----------
    raw_text : str
        The original free-text description provided by the user.
    confidence : float
        Confidence score between 0.0 (no knowledge) and 1.0 (confirmed).
    known_significant_factors : list[str]
        Factor names identified as significant in the prior knowledge.
    known_ranges_reliable : bool
        Whether the user's factor ranges are informed by prior data.
    has_supporting_data : bool
        Whether the prior knowledge is backed by experimental data.
    """

    raw_text: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    known_significant_factors: list[str] = Field(default_factory=list)
    known_ranges_reliable: bool = False
    has_supporting_data: bool = False


# ---------------------------------------------------------------------------
# Strategy output models
# ---------------------------------------------------------------------------


class TransitionRule(BaseModel):
    """Rule governing the transition between consecutive experimental stages.

    Parameters
    ----------
    condition : str
        Human-readable condition, e.g. ``"2-5 significant factors identified"``.
    action : str
        Action to take when the condition is met, e.g. ``"proceed_to_rsm"``.
    fallback : str
        Action if the condition is not met, e.g. ``"broaden_factor_ranges"``.
    """

    condition: str
    action: str
    fallback: str


class ExperimentalStage(BaseModel):
    """One stage in a multi-stage experimental strategy.

    Parameters
    ----------
    stage_number : int
        1-based stage index.
    stage_name : str
        Human-readable name, e.g. ``"Screening"``, ``"Optimization"``.
    design_type : str
        Design type key, e.g. ``"plackett_burman"``, ``"ccd"``, ``"bbd"``.
    design_params : dict
        Design-specific parameters (resolution, center_points, alpha, etc.).
    factors : list[str]
        Factor names involved in this stage.
    estimated_runs : int
        Estimated number of experimental runs.
    purpose : str
        Brief description of what this stage accomplishes.
    success_criteria : dict
        Criteria for deeming this stage successful.
    transition_rules : list[TransitionRule]
        Rules governing the transition to the next stage.
    """

    stage_number: int
    stage_name: str
    design_type: str
    design_params: dict[str, Any] = Field(default_factory=dict)
    factors: list[str] = Field(default_factory=list)
    estimated_runs: int = 0
    purpose: str = ""
    success_criteria: dict[str, Any] = Field(default_factory=dict)
    transition_rules: list[TransitionRule] = Field(default_factory=list)


class ExperimentalStrategy(BaseModel):
    """Complete multi-stage experimental strategy recommendation.

    Parameters
    ----------
    strategy_id : str
        Deterministic hash of the input specification.
    stages : list[ExperimentalStage]
        Ordered list of experimental stages.
    total_estimated_runs : int
        Sum of estimated runs across all stages.
    budget_allocation : dict[str, int]
        Stage name to allocated run count mapping.
    assumptions : list[str]
        Key assumptions underlying the recommendation.
    risks : list[str]
        Risks and potential issues with the strategy.
    alternative_strategies : list[str]
        Brief descriptions of alternative approaches.
    domain : str
        The domain used for domain-specific adjustments.
    detail_level : str
        The detail level used for explanations.
    reasoning : list[str]
        Step-by-step explanation of the decision logic.
    """

    strategy_id: str = ""
    stages: list[ExperimentalStage] = Field(default_factory=list)
    total_estimated_runs: int = 0
    budget_allocation: dict[str, int] = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    alternative_strategies: list[str] = Field(default_factory=list)
    domain: str = "general"
    detail_level: str = "intermediate"
    reasoning: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Input specification
# ---------------------------------------------------------------------------


class DOEProblemSpec(BaseModel):
    """Validated input specification for the strategy recommender.

    Wraps all inputs into a single object for pipeline processing.

    Parameters
    ----------
    factors : list[Factor]
        All candidate experimental factors.
    responses : list[Response]
        Response variables with optimisation goals.
    budget : int or None
        Total run budget across all stages.
    constraints : list[Constraint] or None
        Factor-space constraints.
    hard_to_change_factors : list[str] or None
        Factor names that are expensive to reset between runs.
    prior_knowledge : PriorKnowledge or None
        Parsed prior knowledge with confidence score.
    existing_data_summary : dict or None
        Summary of any existing experimental data.
    domain : DomainType
        Application domain.
    detail_level : str
        ``"novice"`` or ``"intermediate"``.
    """

    factors: list[Factor]
    responses: list[Response] = Field(default_factory=list)
    budget: int | None = None
    constraints: list[Constraint] | None = None
    hard_to_change_factors: list[str] | None = None
    prior_knowledge: PriorKnowledge | None = None
    existing_data_summary: dict[str, Any] | None = None
    domain: DomainType = DomainType.general
    detail_level: Literal["novice", "intermediate"] = "intermediate"

    @property
    def n_factors(self) -> int:
        """Total number of factors."""
        return len(self.factors)

    @property
    def factor_names(self) -> list[str]:
        """Ordered list of factor names."""
        return [f.name for f in self.factors]

    @property
    def n_continuous(self) -> int:
        """Number of continuous factors."""
        return sum(1 for f in self.factors if f.type.value == "continuous")

    @property
    def n_categorical(self) -> int:
        """Number of categorical factors."""
        return sum(1 for f in self.factors if f.type.value == "categorical")

    @property
    def n_mixture(self) -> int:
        """Number of mixture factors."""
        return sum(1 for f in self.factors if f.type.value == "mixture")

    @property
    def has_mixture(self) -> bool:
        """Whether any mixture factors are present."""
        return self.n_mixture > 0

    @property
    def has_hard_to_change(self) -> bool:
        """Whether any hard-to-change factors are specified."""
        return bool(self.hard_to_change_factors)

    @property
    def has_constraints(self) -> bool:
        """Whether any constraints are specified."""
        return bool(self.constraints)

    @property
    def goal_includes_optimization(self) -> bool:
        """Whether any response has an optimisation goal."""
        return len(self.responses) > 0
