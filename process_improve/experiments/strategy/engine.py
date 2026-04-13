# (c) Kevin Dunn, 2010-2026. MIT License.

"""Deterministic rule engine for DOE strategy recommendation.

Implements ~50 decision rules from Montgomery, NIST, and Stat-Ease SCOR
to recommend multi-stage experimental strategies.  No LLM or randomness —
identical inputs always produce identical outputs.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from process_improve.experiments.factor import Constraint, Factor, FactorType, Response
from process_improve.experiments.strategy.budget import (
    allocate_budget,
    estimate_confirmation_runs,
    estimate_rsm_runs,
    estimate_screening_runs,
)
from process_improve.experiments.strategy.domain_templates import get_domain_template
from process_improve.experiments.strategy.models import (
    DOEProblemSpec,
    DomainType,
    ExperimentalStage,
    ExperimentalStrategy,
    PriorKnowledge,
    TransitionRule,
)

# ---------------------------------------------------------------------------
# Prior knowledge parsing
# ---------------------------------------------------------------------------

_HIGH_KEYWORDS = re.compile(
    r"\b(confirmed|validated|published|well[-\s]established|proven|known\s+to\s+be\s+significant)\b",
    re.IGNORECASE,
)
_MEDIUM_KEYWORDS = re.compile(
    r"\b(literature\s+suggests?|preliminary\s+data|pilot\s+study|some\s+evidence|reported)\b",
    re.IGNORECASE,
)
_LOW_KEYWORDS = re.compile(
    r"\b(suspect|expected|based\s+on\s+theory|similar\s+process|assume)\b",
    re.IGNORECASE,
)
_NO_KEYWORDS = re.compile(
    r"\b(no\s+prior|first\s+time|unknown|exploratory|no\s+data|no\s+knowledge)\b",
    re.IGNORECASE,
)
_SIGNIFICANT_FACTOR_PATTERN = re.compile(
    r"(\w[\w\s]*?)\s+(?:is|are)\s+(?:known\s+to\s+be\s+)?(?:significant|important|key|critical)",
    re.IGNORECASE,
)


def _parse_prior_knowledge(
    text: str | None,
    factor_names: list[str],
) -> PriorKnowledge:
    """Map free-text prior knowledge to a structured confidence level."""
    if not text or not text.strip():
        return PriorKnowledge(raw_text="", confidence=0.0)

    text = text.strip()

    # Score based on keyword matching
    confidence = 0.0
    has_supporting_data = False

    if _NO_KEYWORDS.search(text):
        confidence = 0.1
    elif _LOW_KEYWORDS.search(text):
        confidence = 0.4
    elif _MEDIUM_KEYWORDS.search(text):
        confidence = 0.7
        has_supporting_data = "data" in text.lower() or "study" in text.lower()
    elif _HIGH_KEYWORDS.search(text):
        confidence = 0.9
        has_supporting_data = True
    else:
        # No clear keywords — assign moderate-low confidence
        confidence = 0.3

    # Extract factor names mentioned near significance keywords
    known_factors: list[str] = []
    for match in _SIGNIFICANT_FACTOR_PATTERN.finditer(text):
        candidate = match.group(1).strip()
        # Match against actual factor names (case-insensitive substring)
        for fn in factor_names:
            if fn.lower() in candidate.lower() or candidate.lower() in fn.lower():
                if fn not in known_factors:
                    known_factors.append(fn)

    return PriorKnowledge(
        raw_text=text,
        confidence=confidence,
        known_significant_factors=known_factors,
        known_ranges_reliable=confidence >= 0.6,
        has_supporting_data=has_supporting_data,
    )


# ---------------------------------------------------------------------------
# Problem classification
# ---------------------------------------------------------------------------


def _classify_problem(spec: DOEProblemSpec) -> dict[str, Any]:
    """Classify the DOE problem into categories for rule matching."""
    n = spec.n_factors
    prior_conf = spec.prior_knowledge.confidence if spec.prior_knowledge else 0.0

    # Budget tightness
    budget_per_factor = spec.budget / n if spec.budget and n > 0 else float("inf")
    is_tight = budget_per_factor < 4
    is_very_tight = budget_per_factor < 2.5

    return {
        "n_factors": n,
        "n_continuous": spec.n_continuous,
        "n_categorical": spec.n_categorical,
        "n_mixture": spec.n_mixture,
        "has_mixture": spec.has_mixture,
        "has_hard_to_change": spec.has_hard_to_change,
        "has_constraints": spec.has_constraints,
        "prior_confidence": prior_conf,
        "budget": spec.budget,
        "budget_per_factor": budget_per_factor,
        "is_tight_budget": is_tight,
        "is_very_tight_budget": is_very_tight,
        "has_existing_data": spec.existing_data_summary is not None,
    }


# ---------------------------------------------------------------------------
# Screening design selection
# ---------------------------------------------------------------------------


def _select_screening_design(
    spec: DOEProblemSpec,
    classification: dict[str, Any],
    template: dict[str, Any],
) -> ExperimentalStage | None:
    """Select the appropriate screening design based on decision rules."""
    n = classification["n_factors"]
    reasoning: list[str] = []

    # Rule: 2 or fewer factors — no screening needed
    if n <= 2:
        return None

    # Rule: High prior confidence — skip screening
    if classification["prior_confidence"] >= 0.8:
        return None

    # Rule: Mixture factors only — use mixture design path
    if spec.has_mixture and spec.n_continuous == 0:
        design_type = "simplex_lattice"
        runs = max(n + 1, 6)  # Minimum for simplex lattice
        reasoning.append(f"All {n} factors are mixture components → simplex lattice design.")
        return ExperimentalStage(
            stage_number=1,
            stage_name="Screening",
            design_type=design_type,
            design_params={"model": "scheffe_linear"},
            factors=spec.factor_names,
            estimated_runs=runs,
            purpose="Screen mixture components to identify significant proportions.",
            success_criteria={"min_significant_factors": 1},
            transition_rules=_screening_transition_rules(),
        )

    # Rule: 3-5 factors, all continuous — full/fractional factorial
    if 3 <= n <= 5 and spec.n_continuous == n:
        if n <= 4:
            design_type = "full_factorial"
            runs = 2**n + 3  # + center points
            reasoning.append(f"{n} continuous factors → full factorial (2^{n} = {2**n} runs + 3 center points).")
        else:
            design_type = "fractional_factorial"
            runs = estimate_screening_runs(n, "fractional_factorial") + 3
            reasoning.append(f"{n} factors → fractional factorial + center points.")

        return ExperimentalStage(
            stage_number=1,
            stage_name="Screening",
            design_type=design_type,
            design_params={"center_points": 3, "resolution": 4 if n >= 5 else None},
            factors=spec.factor_names,
            estimated_runs=runs,
            purpose=f"Screen {n} factors to identify significant main effects and interactions.",
            success_criteria={"min_significant_factors": 1},
            transition_rules=_screening_transition_rules(),
        )

    # Rule: 6+ factors — PB, DSD, or fractional factorial
    # Sub-rule: Domain or user prefers curvature detection → DSD
    prefer_curvature = template.get("prefer_curvature_detection", False)
    domain_pref = template.get("screening_preference")

    if classification["is_very_tight_budget"]:
        design_type = "definitive_screening"
        runs = estimate_screening_runs(n, "definitive_screening")
        reasoning.append(f"Very tight budget with {n} factors → DSD ({runs} runs) combines screening + curvature.")
    elif prefer_curvature or (domain_pref == "definitive_screening"):
        design_type = "definitive_screening"
        runs = estimate_screening_runs(n, "definitive_screening")
        reasoning.append(f"Domain prefers curvature detection → DSD ({runs} runs).")
    elif classification["prior_confidence"] >= 0.6:
        # Medium confidence — DSD to screen + detect curvature
        design_type = "definitive_screening"
        runs = estimate_screening_runs(n, "definitive_screening")
        reasoning.append(f"Moderate prior confidence ({classification['prior_confidence']:.1f}) → DSD for dual purpose.")
    elif domain_pref == "plackett_burman" or (n >= 6 and not classification["is_tight_budget"]):
        design_type = "plackett_burman"
        runs = estimate_screening_runs(n, "plackett_burman")
        reasoning.append(f"{n} factors with adequate budget → PB design ({runs} runs).")
    elif domain_pref == "fractional_factorial":
        design_type = "fractional_factorial"
        runs = estimate_screening_runs(n, "fractional_factorial") + 3
        reasoning.append(f"Domain prefers fractional factorial → Res IV fraction ({runs} runs).")
    else:
        # Default for 6+ factors
        design_type = "plackett_burman"
        runs = estimate_screening_runs(n, "plackett_burman")
        reasoning.append(f"{n} factors → PB design ({runs} runs) for efficient screening.")

    params: dict[str, Any] = {}
    if design_type == "fractional_factorial":
        params["resolution"] = 4
        params["center_points"] = 3
    elif design_type == "plackett_burman":
        params["center_points"] = 0  # PB typically without center points
    elif design_type == "definitive_screening":
        params["fake_factor"] = n % 2 == 0  # DSD needs odd factor count

    return ExperimentalStage(
        stage_number=1,
        stage_name="Screening",
        design_type=design_type,
        design_params=params,
        factors=spec.factor_names,
        estimated_runs=runs,
        purpose=f"Screen {n} candidate factors to identify the vital few.",
        success_criteria={"min_significant_factors": 1, "max_significant_factors": 5},
        transition_rules=_screening_transition_rules(),
    )


def _screening_transition_rules() -> list[TransitionRule]:
    """Standard transition rules after a screening stage."""
    return [
        TransitionRule(
            condition="0-1 significant factors identified",
            action="broaden_factor_ranges",
            fallback="check_measurement_system",
        ),
        TransitionRule(
            condition="2-5 significant factors identified",
            action="proceed_to_optimization",
            fallback="proceed_to_optimization",
        ),
        TransitionRule(
            condition="6+ significant factors identified",
            action="sub_group_factors",
            fallback="run_additional_screening",
        ),
        TransitionRule(
            condition="Curvature detected via center points",
            action="augment_to_ccd",
            fallback="proceed_to_optimization",
        ),
    ]


# ---------------------------------------------------------------------------
# RSM design selection
# ---------------------------------------------------------------------------


def _select_rsm_design(
    spec: DOEProblemSpec,
    classification: dict[str, Any],
    template: dict[str, Any],
    has_screening: bool,
) -> ExperimentalStage | None:
    """Select the RSM optimisation design."""
    if not spec.goal_includes_optimization and classification["n_factors"] > 5:
        return None

    # Determine RSM factor count (screening narrows to ~3)
    if has_screening:
        n_rsm = min(classification["n_factors"], 3)
    else:
        n_rsm = classification["n_factors"]

    if n_rsm < 2:
        return None

    domain_pref = template.get("rsm_preference")
    center_points = template.get("min_center_points", 3)

    # Rule: Constraints present → D-optimal
    if classification["has_constraints"]:
        design_type = "d_optimal"
        runs = estimate_rsm_runs(n_rsm, "d_optimal")
        purpose = "D-optimal RSM design for constrained factor space."
    # Rule: Sequential buildup from factorial base → CCD
    elif has_screening and domain_pref in ("ccd", "ccd_face_centered", None):
        if domain_pref == "ccd_face_centered":
            design_type = "ccd_face_centered"
        else:
            design_type = "ccd"
        runs = estimate_rsm_runs(n_rsm, design_type, center_points)
        purpose = "CCD augments the factorial base from screening with axial + center points."
    # Rule: Fresh start → BBD (fewer runs, avoids corners)
    elif domain_pref == "box_behnken" or (not has_screening and 3 <= n_rsm <= 7):
        design_type = "box_behnken"
        runs = estimate_rsm_runs(n_rsm, "box_behnken", center_points)
        purpose = "BBD for response surface modeling — fewer runs, avoids extreme corners."
    # Rule: Default — CCD
    else:
        design_type = "ccd"
        runs = estimate_rsm_runs(n_rsm, "ccd", center_points)
        purpose = "CCD for full quadratic model with rotatability."

    params: dict[str, Any] = {"center_points": center_points}
    if "ccd" in design_type:
        params["alpha"] = "face_centered" if design_type == "ccd_face_centered" else "rotatable"

    stage_number = 2 if has_screening else 1
    factor_label = f"the {n_rsm} significant factors" if has_screening else f"all {n_rsm} factors"

    return ExperimentalStage(
        stage_number=stage_number,
        stage_name="Optimization",
        design_type=design_type,
        design_params=params,
        factors=spec.factor_names[:n_rsm],  # Placeholder: actual factors determined after screening
        estimated_runs=runs,
        purpose=f"Fit quadratic response surface model for {factor_label}. {purpose}",
        success_criteria={"min_r_squared": 0.7, "adequate_precision": 4.0},
        transition_rules=[
            TransitionRule(
                condition="Model is adequate (R² > 0.7, adequate precision > 4)",
                action="proceed_to_confirmation",
                fallback="augment_design_or_transform_response",
            ),
            TransitionRule(
                condition="Saddle point detected",
                action="perform_ridge_analysis",
                fallback="proceed_to_confirmation",
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Confirmation stage
# ---------------------------------------------------------------------------


def _build_confirmation_stage(
    spec: DOEProblemSpec,
    stage_number: int,
    min_confirmation: int = 3,
) -> ExperimentalStage:
    """Build the confirmation stage (always included)."""
    n_runs = estimate_confirmation_runs(min_confirmation)
    return ExperimentalStage(
        stage_number=stage_number,
        stage_name="Confirmation",
        design_type="replicates_at_optimum",
        design_params={"n_replicates": n_runs},
        factors=spec.factor_names,
        estimated_runs=n_runs,
        purpose=(
            "Run replicates at the predicted optimum to verify the model predictions. "
            "Compare observed vs. predicted using a confirmation test (prediction interval check)."
        ),
        success_criteria={"observed_within_prediction_interval": True},
        transition_rules=[
            TransitionRule(
                condition="Observed values within prediction intervals",
                action="accept_optimum",
                fallback="investigate_discrepancy",
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Hard-to-change factor wrapping
# ---------------------------------------------------------------------------


def _apply_split_plot(
    stages: list[ExperimentalStage],
    spec: DOEProblemSpec,
) -> list[ExperimentalStage]:
    """Wrap stages in split-plot structure if hard-to-change factors present."""
    if not spec.has_hard_to_change:
        return stages

    htc = spec.hard_to_change_factors or []
    updated: list[ExperimentalStage] = []
    for stage in stages:
        if stage.stage_name in ("Screening", "Optimization"):
            new_params = dict(stage.design_params)
            new_params["split_plot"] = True
            new_params["whole_plot_factors"] = htc
            new_params["subplot_factors"] = [f for f in stage.factors if f not in htc]
            stage = stage.model_copy(update={"design_params": new_params})
        updated.append(stage)

    return updated


# ---------------------------------------------------------------------------
# Prior knowledge adjustments
# ---------------------------------------------------------------------------


def _apply_prior_knowledge(
    stages: list[ExperimentalStage],
    spec: DOEProblemSpec,
) -> list[ExperimentalStage]:
    """Adjust stages based on prior knowledge confidence."""
    if not spec.prior_knowledge or spec.prior_knowledge.confidence < 0.8:
        return stages

    pk = spec.prior_knowledge

    # High confidence with supporting data → skip screening
    if pk.confidence >= 0.8 and pk.has_supporting_data:
        stages = [s for s in stages if s.stage_name != "Screening"]
        # Re-number remaining stages
        for i, stage in enumerate(stages):
            stages[i] = stage.model_copy(update={"stage_number": i + 1})

    return stages


# ---------------------------------------------------------------------------
# Budget adjustment
# ---------------------------------------------------------------------------


def _apply_budget_constraints(
    stages: list[ExperimentalStage],
    spec: DOEProblemSpec,
    budget_alloc: dict[str, Any],
) -> tuple[list[ExperimentalStage], list[str]]:
    """Adjust stage run counts to fit within budget. Returns (stages, warnings)."""
    warnings: list[str] = list(budget_alloc.get("warnings", []))

    if spec.budget is None:
        return stages, warnings

    # Map stage names to budget keys
    stage_budget_map = {
        "Screening": "screening",
        "Optimization": "optimization",
        "Confirmation": "confirmation",
    }

    updated: list[ExperimentalStage] = []
    for stage in stages:
        budget_key = stage_budget_map.get(stage.stage_name, "")
        alloc = budget_alloc.get(budget_key, stage.estimated_runs)
        if alloc < stage.estimated_runs:
            stage = stage.model_copy(update={"estimated_runs": max(alloc, 3)})
        updated.append(stage)

    return updated, warnings


# ---------------------------------------------------------------------------
# Strategy ID
# ---------------------------------------------------------------------------


def _compute_strategy_id(spec: DOEProblemSpec) -> str:
    """Compute a deterministic strategy ID from the input spec."""
    canonical = json.dumps(
        {
            "factors": sorted(spec.factor_names),
            "n_factors": spec.n_factors,
            "responses": sorted(r.name for r in spec.responses),
            "budget": spec.budget,
            "domain": spec.domain.value,
            "htc": sorted(spec.hard_to_change_factors or []),
        },
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Reasoning / assumptions / risks / alternatives
# ---------------------------------------------------------------------------


def _build_assumptions(spec: DOEProblemSpec, has_screening: bool) -> list[str]:
    """Build the list of assumptions for the strategy."""
    assumptions = [
        "Factor ranges are set wide enough to detect effects.",
        "Measurement system is adequate (repeatability << effect sizes).",
        "Runs are randomised to avoid confounding with lurking variables.",
    ]
    if has_screening:
        assumptions.append("Screening will identify 2-4 significant factors for optimisation.")
        assumptions.append("Effect sparsity: only a few factors dominate the response.")
        assumptions.append("Effect heredity: interactions are only important if parent main effects are active.")
    if spec.prior_knowledge and spec.prior_knowledge.confidence >= 0.6:
        assumptions.append("Prior knowledge is reliable and applicable to the current experimental conditions.")
    return assumptions


def _build_risks(spec: DOEProblemSpec, classification: dict[str, Any], budget_warnings: list[str]) -> list[str]:
    """Build the list of risks for the strategy."""
    risks = list(budget_warnings)
    if classification["is_tight_budget"]:
        risks.append("Tight budget may result in underpowered designs with low effect detection probability.")
    if spec.has_hard_to_change:
        risks.append("Hard-to-change factors require split-plot analysis; standard ANOVA gives incorrect p-values.")
    if spec.has_mixture:
        risks.append("Mixture constraints require specialised designs and Scheffe polynomial models.")
    if classification["n_factors"] >= 8:
        risks.append("With 8+ factors, screening may miss important interactions (resolution III/IV limitation).")
    if not risks:
        risks.append("Standard risks: ensure randomisation, verify measurement system, check for outliers.")
    return risks


def _build_alternatives(spec: DOEProblemSpec, classification: dict[str, Any]) -> list[str]:
    """Suggest alternative strategies."""
    alternatives: list[str] = []
    n = classification["n_factors"]

    if n >= 6:
        alternatives.append(
            f"Definitive Screening Design ({2 * n + 1} runs) to combine screening and curvature detection."
        )
    if n <= 5:
        alternatives.append(f"Full factorial 2^{n} ({2**n} runs) if budget allows complete information.")
    if n >= 4:
        alternatives.append("I-optimal design for better prediction variance at the cost of simpler interpretation.")
    if spec.has_mixture:
        alternatives.append("D-optimal mixture design if simplex lattice is too restrictive.")

    return alternatives


def _build_reasoning(
    spec: DOEProblemSpec,
    classification: dict[str, Any],
    stages: list[ExperimentalStage],
    template: dict[str, Any],
) -> list[str]:
    """Build step-by-step reasoning for the strategy."""
    reasoning: list[str] = []
    n = classification["n_factors"]
    domain = spec.domain.value

    reasoning.append(f"Problem: {n} factors, {len(spec.responses)} response(s), domain={domain}.")

    if spec.budget:
        reasoning.append(f"Budget: {spec.budget} total runs ({spec.budget / n:.1f} runs per factor).")
    else:
        reasoning.append("No budget constraint — recommending ideal allocation.")

    if spec.prior_knowledge and spec.prior_knowledge.confidence > 0:
        reasoning.append(
            f"Prior knowledge confidence: {spec.prior_knowledge.confidence:.1f}. "
            + (
                "Skipping screening — going directly to RSM."
                if spec.prior_knowledge.confidence >= 0.8
                else "Using prior knowledge to inform design choices."
            )
        )

    for stage in stages:
        reasoning.append(
            f"Stage {stage.stage_number} ({stage.stage_name}): "
            f"{stage.design_type}, {stage.estimated_runs} runs. {stage.purpose}"
        )

    domain_notes = template.get("notes", {}).get(spec.detail_level, "")
    if domain_notes:
        reasoning.append(f"Domain note ({domain}): {domain_notes}")

    return reasoning


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def recommend_strategy(
    *,
    factors: list[Factor],
    responses: list[Response] | None = None,
    budget: int | None = None,
    constraints: list[Constraint] | None = None,
    hard_to_change_factors: list[str] | None = None,
    prior_knowledge: str | None = None,
    existing_data: Any = None,  # noqa: ANN401 — DataFrame or None
    domain: str | None = None,
    detail_level: str = "intermediate",
) -> dict[str, Any]:
    """Recommend a multi-stage experimental strategy.

    Given a DOE problem description, apply deterministic decision rules
    to recommend a staged experimental plan (screening → optimisation →
    confirmation).

    Parameters
    ----------
    factors : list[Factor]
        All candidate experimental factors.
    responses : list[Response] or None
        Response variables with optimisation goals.
    budget : int or None
        Total run budget across all stages.  ``None`` = no constraint.
    constraints : list[Constraint] or None
        Factor-space constraints (linear or nonlinear).
    hard_to_change_factors : list[str] or None
        Factor names that are expensive to reset between runs.
    prior_knowledge : str or None
        Free-text description of what the user already knows.
    existing_data : DataFrame or None
        Prior experimental data (summary extracted internally).
    domain : str or None
        Application domain (e.g. ``"fermentation"``).  Defaults to ``"general"``.
    detail_level : str
        ``"novice"`` or ``"intermediate"`` (default).

    Returns
    -------
    dict
        JSON-serialisable dictionary with the ``ExperimentalStrategy`` fields.

    Examples
    --------
    >>> from process_improve.experiments.factor import Factor, Response
    >>> factors = [Factor(name=chr(65+i), low=0, high=100) for i in range(7)]
    >>> result = recommend_strategy(factors=factors, budget=40, domain="fermentation")
    >>> result["total_estimated_runs"] <= 40
    True
    """
    # --- Validate inputs ---
    if not factors:
        raise ValueError("At least one factor is required.")
    if budget is not None and budget <= 0:
        raise ValueError(f"Budget must be a positive integer, got {budget}.")
    if detail_level not in ("novice", "intermediate"):
        raise ValueError(f"detail_level must be 'novice' or 'intermediate', got {detail_level!r}.")

    domain_enum = DomainType.general
    if domain:
        try:
            domain_enum = DomainType(domain)
        except ValueError:
            valid = [d.value for d in DomainType]
            raise ValueError(f"Unknown domain {domain!r}. Valid domains: {valid}") from None

    # --- Parse prior knowledge ---
    factor_names = [f.name for f in factors]
    pk = _parse_prior_knowledge(prior_knowledge, factor_names)

    # --- Summarise existing data ---
    data_summary = None
    if existing_data is not None:
        try:
            data_summary = {
                "n_rows": len(existing_data),
                "columns": list(existing_data.columns),
            }
        except (AttributeError, TypeError):
            data_summary = None

    # --- Build problem spec ---
    spec = DOEProblemSpec(
        factors=factors,
        responses=responses or [],
        budget=budget,
        constraints=constraints,
        hard_to_change_factors=hard_to_change_factors,
        prior_knowledge=pk,
        existing_data_summary=data_summary,
        domain=domain_enum,
        detail_level=detail_level,
    )

    # --- Get domain template ---
    template = get_domain_template(domain_enum.value)

    # --- Classify problem ---
    classification = _classify_problem(spec)

    # --- Determine stages ---
    stages: list[ExperimentalStage] = []

    # Screening stage
    screening = _select_screening_design(spec, classification, template)
    has_screening = screening is not None
    if screening:
        stages.append(screening)

    # RSM optimisation stage
    rsm = _select_rsm_design(spec, classification, template, has_screening)
    if rsm:
        stages.append(rsm)

    # Confirmation stage (always)
    min_conf = template.get("min_confirmation", 3)
    confirmation = _build_confirmation_stage(spec, len(stages) + 1, min_conf)
    stages.append(confirmation)

    # --- Apply modifiers ---
    stages = _apply_split_plot(stages, spec)
    stages = _apply_prior_knowledge(stages, spec)

    # --- Budget allocation ---
    needs_screening = any(s.stage_name == "Screening" for s in stages)
    needs_rsm = any(s.stage_name == "Optimization" for s in stages)
    screening_design = next((s.design_type for s in stages if s.stage_name == "Screening"), "plackett_burman")
    rsm_design = next((s.design_type for s in stages if s.stage_name == "Optimization"), "box_behnken")

    budget_alloc = allocate_budget(
        total_budget=budget,
        n_factors=spec.n_factors,
        needs_screening=needs_screening,
        needs_rsm=needs_rsm,
        screening_design=screening_design,
        rsm_design=rsm_design,
        domain_weights=template.get("budget_weights"),
        min_confirmation=min_conf,
        center_points=template.get("min_center_points", 3),
    )

    stages, budget_warnings = _apply_budget_constraints(stages, spec, budget_alloc)

    # --- Re-number stages ---
    for i, stage in enumerate(stages):
        stages[i] = stage.model_copy(update={"stage_number": i + 1})

    # --- Assemble strategy ---
    total_runs = sum(s.estimated_runs for s in stages)
    budget_dict = {s.stage_name: s.estimated_runs for s in stages}

    strategy = ExperimentalStrategy(
        strategy_id=_compute_strategy_id(spec),
        stages=stages,
        total_estimated_runs=total_runs,
        budget_allocation=budget_dict,
        assumptions=_build_assumptions(spec, has_screening),
        risks=_build_risks(spec, classification, budget_warnings),
        alternative_strategies=_build_alternatives(spec, classification),
        domain=domain_enum.value,
        detail_level=detail_level,
        reasoning=_build_reasoning(spec, classification, stages, template),
    )

    return strategy.model_dump()
