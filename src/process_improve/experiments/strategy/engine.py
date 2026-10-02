# (c) Kevin Dunn, 2010-2026. MIT License.

"""Deterministic rule engine for DOE strategy recommendation.

Implements ~50 decision rules from Montgomery, NIST, and Stat-Ease SCOR
to recommend multi-stage experimental strategies.  No LLM or randomness -
identical inputs always produce identical outputs.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Literal, cast

from process_improve.experiments.designs_response_surface import dsd_run_count
from process_improve.experiments.factor import Constraint, Factor, Response
from process_improve.experiments.strategy.budget import (
    allocate_budget,
    estimate_confirmation_runs,
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
    # SEC-29 (#278): bounded the capture group so a multi-KB whitespace
    # payload can no longer trigger O(n^2) regex matching. ``\w+(?:\s\w+){0,4}``
    # is the longest realistic factor name we expect ("two-stage reactor
    # temperature") and matches in linear time.
    r"\b(\w+(?:\s\w+){0,4})\s+(?:is|are)\s+(?:known\s+to\s+be\s+)?(?:significant|important|key|critical)\b",
    re.IGNORECASE,
)

# SEC-29 (#278): cap the length of the free-text prior so a caller
# cannot send a multi-MB string that swamps the regex engine even with
# the bounded pattern above. 4 KiB comfortably accommodates any
# realistic prior-knowledge note and slams the door on payload DoS.
_PRIOR_KNOWLEDGE_MAX_CHARS = 4096


def _parse_prior_knowledge(
    text: str | None,
    factor_names: list[str],
) -> PriorKnowledge:
    """Map free-text prior knowledge to a structured confidence level.

    The strongest evidence cue in the text sets the confidence: confirmed, validated or
    published results (0.9) outrank reported or preliminary data (0.7), which outrank
    suspicion or theory (0.4), which outrank a statement of no prior knowledge (0.1). A
    weaker cue elsewhere in the text, such as one factor whose role is "unknown", does
    not lower the confidence that the strongest cue sets.
    """
    if not text or not text.strip():
        return PriorKnowledge(raw_text="", confidence=0.0)

    text = text.strip()
    if len(text) > _PRIOR_KNOWLEDGE_MAX_CHARS:
        raise ValueError(
            f"prior_knowledge text is {len(text)} characters; the maximum "
            f"is {_PRIOR_KNOWLEDGE_MAX_CHARS}. Trim the input to its "
            "relevant prose."
        )

    # Score based on keyword matching, strongest evidence first
    confidence = 0.0
    has_supporting_data = False

    if _HIGH_KEYWORDS.search(text):
        confidence = 0.9
        has_supporting_data = True
    elif _MEDIUM_KEYWORDS.search(text):
        confidence = 0.7
        has_supporting_data = "data" in text.lower() or "study" in text.lower()
    elif _LOW_KEYWORDS.search(text):
        confidence = 0.4
    elif _NO_KEYWORDS.search(text):
        confidence = 0.1
    else:
        # No clear keywords - assign moderate-low confidence
        confidence = 0.3

    # Extract factor names mentioned near significance keywords
    known_factors: list[str] = []
    for match in _SIGNIFICANT_FACTOR_PATTERN.finditer(text):
        candidate = match.group(1).strip()
        # Match against actual factor names (case-insensitive substring)
        for fn in factor_names:
            if (fn.lower() in candidate.lower() or candidate.lower() in fn.lower()) and fn not in known_factors:
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
# Run counts
# ---------------------------------------------------------------------------

#: Design types whose run count is their ``budget``; they are chosen by an optimality criterion.
_OPTIMAL_TYPES = frozenset({"d_optimal", "i_optimal", "a_optimal", "e_optimal"})

#: Runs beyond the model's coefficient count in the smallest optimal design the planner
#: proposes, so the fitted model keeps some residual degrees of freedom.
_MIN_RESIDUAL_RUNS = 3

#: Multiple of the model's coefficient count an optimal design is given when the budget allows.
_OPTIMAL_RUNS_PER_TERM = 1.5


def _n_parameters(factors: list[Factor], model_type: str) -> int:
    """Return the number of coefficients in ``model_type`` over ``factors``.

    The Scheffe models count the mixture blending terms; the others count as the
    optimal designs do (a categorical factor with ``L`` levels takes ``L - 1`` columns).
    """
    q = len(factors)
    if model_type == "scheffe_linear":
        return q
    if model_type == "scheffe_quadratic":
        return q * (q + 1) // 2
    if model_type == "scheffe_special_cubic":
        return q * (q + 1) // 2 + math.comb(q, 3)
    from process_improve.experiments.designs_optimal import _n_model_parameters  # noqa: PLC0415

    return _n_model_parameters(factors, model_type)


def _optimal_runs(factors: list[Factor], model_type: str) -> int:
    """Return the runs the planner gives an optimal design: 1.5 times the model's coefficients."""
    return math.ceil(_OPTIMAL_RUNS_PER_TERM * _n_parameters(factors, model_type))


def _on_full_simplex(factors: list[Factor], constraints: list[Any] | None) -> bool:
    """Whether mixture components span the whole simplex (no bound inside (0, 1), no constraint)."""
    bounded = any((f.low or 0.0) > 0.0 or (f.high if f.high is not None else 1.0) < 1.0 for f in factors)
    return not bounded and not constraints


def _built_runs(factors: list[Factor], design_type: str, params: dict[str, Any]) -> int:
    """Return the number of runs ``generate_design(factors, design_type=design_type, **params)`` gives.

    The ``budget`` sets the size of an optimal design (never fewer runs than its model
    has coefficients), of a supersaturated design and of a mixture design in a bounded or
    constrained region. The classical designs are built, which is quick, so their count
    is exactly the one ``generate_design`` gives.
    """
    if design_type in _OPTIMAL_TYPES:
        return max(int(params["budget"]), _n_parameters(factors, params.get("model_type", "interactions")))
    if design_type == "supersaturated" or (
        design_type == "mixture" and not _on_full_simplex(factors, params.get("constraints"))
    ):
        return int(params["budget"])
    from process_improve.experiments.designs import generate_design  # noqa: PLC0415

    return generate_design(factors, design_type=design_type, **params).n_runs


def _factors_named(spec: DOEProblemSpec, names: list[str]) -> list[Factor]:
    """Return the factors called ``names``, in the order the problem lists them."""
    return [f for f in spec.factors if f.name in names]


def _constraint_params(spec: DOEProblemSpec) -> dict[str, Any]:
    """Return the ``constraints`` keyword for a stage that honours them (empty without constraints)."""
    return {"constraints": [c.model_dump() for c in spec.constraints]} if spec.constraints else {}


# ---------------------------------------------------------------------------
# Screening design selection
# ---------------------------------------------------------------------------


def _mixture_screening_stage(spec: DOEProblemSpec, n: int) -> ExperimentalStage:
    """Screening stage when all factors are mixture components.

    A budget of ``max(q + 1, 6)`` blends gives the {q, 2} simplex lattice on the full
    simplex (the simplex centroid for two or three components, whose ``2^q - 1`` blends fit
    that budget), and a D-optimal choice of that many blends for the linear Scheffe model
    when the components have bounds or constraints.
    """
    params: dict[str, Any] = {"model_type": "scheffe_linear", "budget": max(n + 1, 6), **_constraint_params(spec)}
    return ExperimentalStage(
        stage_number=1,
        stage_name="Screening",
        design_type="mixture",
        design_params=params,
        factors=spec.factor_names,
        estimated_runs=_built_runs(spec.factors, "mixture", params),
        purpose="Screen mixture components to identify significant proportions.",
        success_criteria={"min_significant_factors": 1},
        transition_rules=_screening_transition_rules(),
    )


def _optimal_screening_stage(spec: DOEProblemSpec, n: int) -> ExperimentalStage:
    """Screening stage with a categorical factor of more than two levels: a D-optimal main-effects design.

    The two-level designs (factorials, Plackett-Burman, DSD) take two-level categorical
    factors only.
    """
    params = {"model_type": "main_effects", "budget": _optimal_runs(spec.factors, "main_effects")}
    return ExperimentalStage(
        stage_number=1,
        stage_name="Screening",
        design_type="d_optimal",
        design_params=params,
        factors=spec.factor_names,
        estimated_runs=_built_runs(spec.factors, "d_optimal", params),
        purpose=(
            f"Screen {n} factors with a D-optimal main-effects design, which places categorical factors "
            "at all their levels."
        ),
        success_criteria={"min_significant_factors": 1},
        transition_rules=_screening_transition_rules(),
    )


def _factorial_screening_stage(spec: DOEProblemSpec, n: int) -> ExperimentalStage:
    """Screening stage for 3-5 all-continuous factors (full or fractional factorial)."""
    design_type = "full_factorial" if n <= 4 else "fractional_factorial"
    params = {"n_center_points": 3, "resolution": 4 if n >= 5 else None}
    return ExperimentalStage(
        stage_number=1,
        stage_name="Screening",
        design_type=design_type,
        design_params=params,
        factors=spec.factor_names,
        estimated_runs=_built_runs(spec.factors, design_type, params),
        purpose=f"Screen {n} factors to identify significant main effects and interactions.",
        success_criteria={"min_significant_factors": 1},
        transition_rules=_screening_transition_rules(),
    )


def _supersaturated_runs(n: int, classification: dict[str, Any]) -> int | None:
    """Most runs within a budget below ``n + 1`` that give an unaliased supersaturated design, if any.

    Such a budget cannot estimate every main effect, so the screening design has to
    be supersaturated: the factors must all be continuous, run at two levels.
    """
    from process_improve.experiments.designs_supersaturated import supersaturated_available  # noqa: PLC0415

    budget = classification["budget"]
    if budget is None or budget >= n + 1 or classification["n_continuous"] != n:
        return None
    return next((runs for runs in range(int(budget), 3, -1) if supersaturated_available(n, runs)), None)


def _large_factor_screening_choice(n: int, classification: dict[str, Any], template: dict[str, Any]) -> tuple[str, int]:
    """Choose (design_type, estimated_runs) when the factorial/mixture rules do not apply.

    A budget below ``n + 1`` gets a supersaturated design when one exists for it.
    Otherwise the original ordered decision chain holds: definitive screening (very
    tight budget, curvature preference, or moderate prior confidence) takes
    precedence, then Plackett-Burman (explicit preference or 6+ factors with adequate
    budget), then an explicit fractional-factorial preference, with Plackett-Burman as
    the default. The returned key is a ``generate_design`` design type.
    """
    if (runs := _supersaturated_runs(n, classification)) is not None:
        return "supersaturated", runs
    prefer_curvature = template.get("prefer_curvature_detection", False)
    domain_pref = template.get("screening_preference")

    if (
        classification["is_very_tight_budget"]
        or prefer_curvature
        or domain_pref == "definitive_screening"
        or classification["prior_confidence"] >= 0.6
    ):
        return "dsd", estimate_screening_runs(n, "definitive_screening")
    if domain_pref == "plackett_burman" or (n >= 6 and not classification["is_tight_budget"]):
        return "plackett_burman", estimate_screening_runs(n, "plackett_burman")
    if domain_pref == "fractional_factorial":
        return "fractional_factorial", estimate_screening_runs(n, "fractional_factorial") + 3
    return "plackett_burman", estimate_screening_runs(n, "plackett_burman")


def _screening_design_params(design_type: str, runs: int) -> dict[str, Any]:
    """Build the ``generate_design`` keyword arguments for a large-factor screening stage.

    A definitive screening design needs none: ``generate_design`` handles an even
    number of factors itself.
    """
    if design_type == "fractional_factorial":
        return {"resolution": 4, "n_center_points": 3}
    if design_type == "plackett_burman":
        return {"n_center_points": 0}  # PB typically without center points
    if design_type == "supersaturated":
        return {"budget": runs}
    return {}


def _select_screening_design(
    spec: DOEProblemSpec,
    classification: dict[str, Any],
    template: dict[str, Any],
) -> ExperimentalStage | None:
    """Select the appropriate screening design based on decision rules."""
    n = classification["n_factors"]

    # Rule: 2 or fewer factors - no screening needed.
    if n <= 2:
        return None
    # Rule: high prior confidence - skip screening.
    if classification["prior_confidence"] >= 0.8:
        return None
    # Rule: mixture factors only - use the mixture design path.
    if spec.has_mixture and spec.n_continuous == 0:
        return _mixture_screening_stage(spec, n)
    # Rule: a categorical factor with more than two levels - the two-level designs cannot place it.
    if any(f.type.value == "categorical" and len(f.levels or []) > 2 for f in spec.factors):
        return _optimal_screening_stage(spec, n)
    # Rule: 3-5 factors, all continuous - full/fractional factorial.
    if 3 <= n <= 5 and spec.n_continuous == n:
        return _factorial_screening_stage(spec, n)

    # Remaining cases (typically 6+ factors): Plackett-Burman, DSD, or fractional factorial.
    design_type, runs = _large_factor_screening_choice(n, classification, template)
    params = _screening_design_params(design_type, runs)
    return ExperimentalStage(
        stage_number=1,
        stage_name="Screening",
        design_type=design_type,
        design_params=params,
        factors=spec.factor_names,
        estimated_runs=_built_runs(spec.factors, design_type, params),
        purpose=f"Screen {n} candidate factors to identify the vital few.",
        success_criteria={"min_significant_factors": 1, "max_significant_factors": 5},
        transition_rules=_screening_transition_rules(),
    )


def _screening_transition_rules() -> list[TransitionRule]:
    """Return standard transition rules after a screening stage."""
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


def _mixture_optimization_stage(spec: DOEProblemSpec, stage_number: int) -> ExperimentalStage:
    """Quadratic Scheffe stage for an all-mixture problem.

    Components cannot be dropped after screening the way process factors can (the
    proportions must still sum to 1), and a CCD or Box-Behnken design in the components
    would not give proportions, so every component stays in a mixture design. On the full
    simplex that is the simplex centroid (``2^q - 1`` blends); in a bounded or constrained
    region, a D-optimal choice of 1.5 times the model's coefficients.
    """
    params: dict[str, Any] = {"model_type": "scheffe_quadratic", **_constraint_params(spec)}
    if not _on_full_simplex(spec.factors, spec.constraints):
        params["budget"] = _optimal_runs(spec.factors, "scheffe_quadratic")
    return ExperimentalStage(
        stage_number=stage_number,
        stage_name="Optimization",
        design_type="mixture",
        design_params=params,
        factors=spec.factor_names,
        estimated_runs=_built_runs(spec.factors, "mixture", params),
        purpose="Fit a quadratic Scheffe blending model over the mixture region, to locate the best blend.",
        success_criteria={"min_r_squared": 0.7, "adequate_precision": 4.0},
        transition_rules=[
            TransitionRule(
                condition="Model is adequate (R² > 0.7, adequate precision > 4)",
                action="proceed_to_confirmation",
                fallback="augment_design_or_transform_response",
            ),
        ],
    )


def _rsm_factor_names(spec: DOEProblemSpec, has_screening: bool) -> list[str]:
    """Factors the optimisation stage is planned over.

    Without screening these are the factors the prior knowledge names as significant when
    that knowledge is strong enough to skip screening, and otherwise every factor. After
    screening the stage needs the factors screening finds; until then it lists three
    placeholders, the factors the prior knowledge names first.
    """
    pk = spec.prior_knowledge
    known = [name for name in spec.factor_names if pk and name in pk.known_significant_factors]
    if not has_screening:
        return known if known and pk is not None and pk.confidence >= 0.8 else spec.factor_names
    ordered = known + [name for name in spec.factor_names if name not in known]
    return ordered[: min(spec.n_factors, 3)]


def _optimal_rsm_design(spec: DOEProblemSpec, factors: list[Factor]) -> tuple[str, dict[str, Any], str]:
    """D-optimal quadratic design: for constraints, categorical factors or a single factor."""
    params: dict[str, Any] = {"model_type": "quadratic", "budget": _optimal_runs(factors, "quadratic")}
    params.update(_constraint_params(spec))
    if spec.has_constraints:
        reason = "D-optimal RSM design over the constrained factor space."
    elif len(factors) == 1:
        reason = "A three-level D-optimal design (low, centre and high settings, replicated) fits the quadratic."
    else:
        reason = (
            "D-optimal RSM design: a CCD or Box-Behnken design needs continuous factors, so the categorical "
            "factors enter as main effects and interactions."
        )
    return "d_optimal", params, reason


def _rsm_design(
    spec: DOEProblemSpec,
    factors: list[Factor],
    template: dict[str, Any],
    has_screening: bool,
) -> tuple[str, dict[str, Any], str]:
    """Choose the optimisation design over ``factors``: (design_type, design_params, reason)."""
    n_rsm = len(factors)
    if spec.has_constraints or n_rsm == 1 or any(f.type.value == "categorical" for f in factors):
        return _optimal_rsm_design(spec, factors)

    domain_pref = template.get("rsm_preference")
    params: dict[str, Any] = {"n_center_points": template.get("min_center_points", 3)}
    # Box-Behnken designs exist for 3 to 7 factors.
    if has_screening and domain_pref in ("ccd", "ccd_face_centered", None):
        face_centered = domain_pref == "ccd_face_centered"
        reason = "CCD augments the factorial base from screening with axial + center points."
    elif 3 <= n_rsm <= 7 and (domain_pref == "box_behnken" or not has_screening):
        return "box_behnken", params, "BBD for response surface modeling - fewer runs, avoids extreme corners."
    else:
        # A domain that prefers Box-Behnken avoids settings beyond the factor ranges.
        face_centered = domain_pref in ("ccd_face_centered", "box_behnken")
        reason = (
            "Face-centred CCD keeps every run within the factor ranges."
            if face_centered
            else "CCD for full quadratic model with rotatability."
        )
    params["alpha"] = "face_centered" if face_centered else "rotatable"
    if n_rsm >= 6:
        params["cube"] = "fractional"  # a resolution V half fraction, not the full 2^k cube
    return "ccd", params, reason


def _select_rsm_design(
    spec: DOEProblemSpec,
    classification: dict[str, Any],
    template: dict[str, Any],
    has_screening: bool,
) -> ExperimentalStage | None:
    """Select the RSM optimisation design."""
    if not spec.goal_includes_optimization and classification["n_factors"] > 5:
        return None
    stage_number = 2 if has_screening else 1
    if spec.has_mixture and spec.n_continuous == 0:
        return _mixture_optimization_stage(spec, stage_number=stage_number)

    names = _rsm_factor_names(spec, has_screening)
    factors = _factors_named(spec, names)
    design_type, params, reason = _rsm_design(spec, factors, template, has_screening)
    n_rsm = len(names)
    if has_screening:
        factor_label = (
            f"the {n_rsm} significant factors. The listed factors are placeholders: run this stage on "
            "the factors screening finds significant"
        )
    else:
        factor_label = f"all {n_rsm} factors" if n_rsm == spec.n_factors else f"the {n_rsm} known significant factors"

    return ExperimentalStage(
        stage_number=stage_number,
        stage_name="Optimization",
        design_type=design_type,
        design_params=params,
        factors=names,
        estimated_runs=_built_runs(factors, design_type, params),
        purpose=f"Fit quadratic response surface model for {factor_label}. {reason}",
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
    """Give each stage with a hard-to-change factor a split-plot design.

    ``generate_design`` builds split-plot structure only in the optimal families, so such
    a stage becomes a D-optimal design of the same size, for main effects when screening
    and the quadratic model when optimising, with ``hard_to_change`` naming its
    whole-plot factors.
    """
    if not spec.has_hard_to_change:
        return stages

    htc = set(spec.hard_to_change_factors or [])
    updated: list[ExperimentalStage] = []
    for stage in stages:
        whole_plot = [name for name in stage.factors if name in htc]
        if stage.stage_name not in ("Screening", "Optimization") or not whole_plot:
            updated.append(stage)
            continue
        if stage.design_type in _OPTIMAL_TYPES:
            params = {**stage.design_params, "hard_to_change": whole_plot}
        else:
            model_type = "main_effects" if stage.stage_name == "Screening" else "quadratic"
            params = {
                "model_type": model_type,
                "budget": stage.estimated_runs,
                "hard_to_change": whole_plot,
                **_constraint_params(spec),
            }
        factors = _factors_named(spec, stage.factors)
        purpose = (
            f"{stage.purpose} Split-plot: {', '.join(whole_plot)} change only between whole plots, "
            "so the runs come from a D-optimal split-plot design."
        )
        updated.append(
            stage.model_copy(
                update={
                    "design_type": "d_optimal",
                    "design_params": params,
                    "estimated_runs": _built_runs(factors, "d_optimal", params),
                    "purpose": purpose,
                }
            )
        )

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

#: Stage name to the key ``allocate_budget`` uses for its share.
_STAGE_BUDGET_KEYS = {"Screening": "screening", "Optimization": "optimization", "Confirmation": "confirmation"}

#: Classical screening designs a tight budget may replace with a saturated Plackett-Burman design.
_SHRINKABLE_SCREENING = frozenset({"full_factorial", "fractional_factorial", "plackett_burman", "dsd"})


def _stage_variants(stage: ExperimentalStage, factors: list[Factor]) -> list[tuple[str, dict[str, Any]]]:
    """Versions of ``stage`` from its ideal design down to the smallest the planner accepts.

    Screening drops its centre points, then becomes a Plackett-Burman design. Optimisation
    keeps two centre points (one degree of freedom for pure error), then becomes a D-optimal
    quadratic design with three runs more than the model has coefficients, as does any
    optimal design. Confirmation drops to three replicates.
    """
    design_type, params = stage.design_type, stage.design_params
    variants: list[tuple[str, dict[str, Any]]] = [(design_type, params)]
    if stage.stage_name == "Confirmation":
        variants.append((design_type, {**params, "n_replicates": 3}))
    elif design_type in _OPTIMAL_TYPES or (design_type == "mixture" and "budget" in params):
        smallest = _n_parameters(factors, params.get("model_type", "interactions")) + _MIN_RESIDUAL_RUNS
        variants.append((design_type, {**params, "budget": smallest}))
    elif stage.stage_name == "Screening" and design_type in _SHRINKABLE_SCREENING:
        if params.get("n_center_points"):
            variants.append((design_type, {**params, "n_center_points": 0}))
        variants.append(("plackett_burman", {"n_center_points": 0}))
    elif stage.stage_name == "Optimization" and design_type in ("ccd", "box_behnken"):
        if params.get("n_center_points", 3) > 2:
            variants.append((design_type, {**params, "n_center_points": 2}))
        smallest = _n_parameters(factors, "quadratic") + _MIN_RESIDUAL_RUNS
        variants.append(("d_optimal", {"model_type": "quadratic", "budget": smallest}))
    return variants


def _variant_runs(stage: ExperimentalStage, spec: DOEProblemSpec) -> list[tuple[str, dict[str, Any], int]]:
    """Each variant of ``stage`` with its run count, keeping only those that save runs."""
    factors = _factors_named(spec, stage.factors)
    out: list[tuple[str, dict[str, Any], int]] = []
    for design_type, params in _stage_variants(stage, factors):
        if design_type == "replicates_at_optimum":
            runs = int(params["n_replicates"])
        else:
            runs = _built_runs(factors, design_type, params)
        if not out or runs < out[-1][2]:
            out.append((design_type, params, runs))
    return out


def _single_dsd_plan(stages: list[ExperimentalStage], spec: DOEProblemSpec) -> list[ExperimentalStage] | None:
    """One definitive screening design in place of separate screening and optimisation stages.

    Returns None when the plan has no such pair, or when a DSD cannot take the factors
    (fewer than three, any not continuous, or hard-to-change factors needing a split plot).
    """
    names = {s.stage_name for s in stages}
    if (
        not {"Screening", "Optimization"} <= names
        or spec.n_factors < 3
        or spec.n_continuous != spec.n_factors
        or spec.has_hard_to_change
    ):
        return None
    dsd = ExperimentalStage(
        stage_number=1,
        stage_name="Screening and optimization",
        design_type="dsd",
        design_params={},
        factors=spec.factor_names,
        estimated_runs=_built_runs(spec.factors, "dsd", {}),
        purpose=(
            f"Screen all {spec.n_factors} factors and detect curvature in one definitive screening design; "
            "it estimates the full quadratic model in up to three active factors."
        ),
        success_criteria={"min_significant_factors": 1, "max_significant_factors": 3},
        transition_rules=_screening_transition_rules(),
    )
    confirmation = next(s for s in stages if s.stage_name == "Confirmation")
    smallest = confirmation.model_copy(update={"design_params": {"n_replicates": 3}, "estimated_runs": 3})
    return [dsd, smallest]


def _apply_budget_constraints(
    stages: list[ExperimentalStage],
    spec: DOEProblemSpec,
    budget_alloc: dict[str, Any],
) -> tuple[list[ExperimentalStage], list[str]]:
    """Change the stages' designs until the plan fits the budget. Returns (stages, warnings).

    Each step takes the stage furthest over its share of the budget (from
    ``allocate_budget``) to its next smaller design, so ``estimated_runs`` is always what
    the stage's ``design_params`` build. When every stage is as small as it goes and the
    plan still does not fit, one definitive screening design replaces the screening and
    optimisation stages if that fits; otherwise the warning says the budget is too small.
    """
    budget = spec.budget
    ideal_total = sum(s.estimated_runs for s in stages)
    if budget is None or ideal_total <= budget:
        return stages, []

    options = [_variant_runs(stage, spec) for stage in stages]
    level = [0] * len(stages)

    def runs(i: int) -> int:
        return options[i][level[i]][2]

    while sum(runs(i) for i in range(len(stages))) > budget:
        shrinkable = [i for i in range(len(stages)) if level[i] + 1 < len(options[i])]
        if not shrinkable:
            break
        share = [budget_alloc.get(_STAGE_BUDGET_KEYS.get(s.stage_name, ""), 0) for s in stages]
        level[max(shrinkable, key=lambda i: runs(i) - share[i])] += 1

    fitted: list[ExperimentalStage] = []
    changes: list[str] = []
    for i, stage in enumerate(stages):
        design_type, params, n_runs = options[i][level[i]]
        if level[i]:
            changes.append(
                f"{stage.stage_name}: {stage.design_type} ({stage.estimated_runs} runs) "
                f"became {design_type} ({n_runs} runs)"
            )
        fitted.append(
            stage.model_copy(update={"design_type": design_type, "design_params": params, "estimated_runs": n_runs})
        )
    smallest_total = sum(s.estimated_runs for s in fitted)
    if smallest_total <= budget:
        return fitted, [
            (
                f"Budget of {budget} runs is below the {ideal_total} runs of the ideal plan, so the designs were "
                f"reduced to fit: {'; '.join(changes)}."
            )
        ]

    single = _single_dsd_plan(stages, spec)
    single_total = sum(s.estimated_runs for s in single) if single is not None else smallest_total
    if single is not None and single_total <= budget:
        return single, [
            (
                f"Budget of {budget} runs cannot hold separate screening and optimisation stages (at least "
                f"{smallest_total} runs), so one definitive screening design screens the factors and detects "
                "curvature; it fits a full quadratic model only if at most three factors are active."
            )
        ]
    if single is not None and single_total < smallest_total:
        fitted, smallest_total = single, single_total
    return fitted, [
        (
            f"Budget of {budget} runs is below the smallest plan the rules give ({smallest_total} runs, with "
            "every stage reduced as far as it goes). Raise the budget or study fewer factors; "
            "total_estimated_runs reports the runs the stages need."
        )
    ]


# ---------------------------------------------------------------------------
# Strategy ID
# ---------------------------------------------------------------------------


def _compute_strategy_id(spec: DOEProblemSpec) -> str:
    """Compute a deterministic strategy ID from every input that can change the strategy.

    The factors are hashed in order with their types, ranges and levels, because the
    order picks the optimisation stage's placeholder factors.
    """
    canonical = json.dumps(
        {
            "factors": [f.model_dump(mode="json") for f in spec.factors],
            "responses": sorted(
                (r.model_dump(mode="json") for r in spec.responses), key=lambda r: json.dumps(r, sort_keys=True)
            ),
            "budget": spec.budget,
            "constraints": [c.model_dump(mode="json") for c in spec.constraints or []],
            "prior_knowledge": spec.prior_knowledge.raw_text if spec.prior_knowledge else "",
            "existing_data": spec.existing_data_summary,
            "domain": spec.domain.value,
            "detail_level": spec.detail_level,
            "htc": sorted(spec.hard_to_change_factors or []),
        },
        sort_keys=True,
        default=str,
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


def _build_risks(
    spec: DOEProblemSpec,
    classification: dict[str, Any],
    budget_warnings: list[str],
    template: dict[str, Any],
) -> list[str]:
    """Build the list of risks for the strategy, ending with the domain's special considerations."""
    risks = list(budget_warnings)
    if classification["is_tight_budget"]:
        risks.append("Tight budget may result in underpowered designs with low effect detection probability.")
    if spec.has_hard_to_change:
        risks.append("Hard-to-change factors require split-plot analysis; standard ANOVA gives incorrect p-values.")
    if spec.has_mixture:
        risks.append("Mixture constraints require specialised designs and Scheffe polynomial models.")
    if classification["n_factors"] >= 8:
        risks.append("With 8+ factors, screening may miss important interactions (resolution III/IV limitation).")
    risks.extend(template.get("special_considerations", []))
    if not risks:
        risks.append("Standard risks: ensure randomisation, verify measurement system, check for outliers.")
    return risks


def _build_alternatives(spec: DOEProblemSpec, classification: dict[str, Any]) -> list[str]:
    """Suggest alternative strategies."""
    alternatives: list[str] = []
    n = classification["n_factors"]

    if n >= 6:
        alternatives.append(
            f"Definitive Screening Design ({dsd_run_count(n)} runs) to combine screening and curvature detection."
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
        reasoning.append("No budget constraint - recommending ideal allocation.")

    if spec.prior_knowledge and spec.prior_knowledge.confidence > 0:
        reasoning.append(
            f"Prior knowledge confidence: {spec.prior_knowledge.confidence:.1f}. "
            + (
                "Skipping screening - going directly to RSM."
                if spec.prior_knowledge.confidence >= 0.8
                else "Using prior knowledge to inform design choices."
            )
        )

    reasoning.extend(
        f"Stage {stage.stage_number} ({stage.stage_name}): "
        f"{stage.design_type}, {stage.estimated_runs} runs. {stage.purpose}"
        for stage in stages
    )

    domain_notes = template.get("notes", {}).get(spec.detail_level, "")
    if domain_notes:
        reasoning.append(f"Domain note ({domain}): {domain_notes}")
    extra_stages = template.get("extra_stages", [])
    if extra_stages:
        names = ", ".join(stage.replace("_", " ") for stage in extra_stages)
        reasoning.append(f"Studies in this domain usually add stages that are not scheduled here: {names}.")

    return reasoning


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _refuse_unplannable(factors: list[Factor], hard_to_change_factors: list[str] | None) -> None:
    """Raise for inputs no stage can be built for, naming what to change.

    Mixture components with process factors are refused by ``generate_design``, a split
    plot cannot hold mixture components, and a hard-to-change name must be a factor.
    """
    names = [f.name for f in factors]
    is_mixture = [f.type.value == "mixture" for f in factors]
    if any(is_mixture) and not all(is_mixture):
        process = [name for name, mixture in zip(names, is_mixture, strict=True) if not mixture]
        raise ValueError(
            f"Mixture-process problems (mixture components together with process factors {process}) are not "
            "supported: generate_design cannot build their stages. Plan the mixture components and the "
            "process factors as separate studies, or cross a mixture design with a process design."
        )
    unknown = [name for name in hard_to_change_factors or [] if name not in names]
    if unknown:
        raise ValueError(f"hard_to_change_factors {unknown} are not among the factors {names}.")
    if hard_to_change_factors and any(is_mixture):
        raise ValueError("Hard-to-change factors (a split-plot design) are not supported for mixture components.")


def recommend_strategy(  # noqa: C901, PLR0913
    *,
    factors: list[Factor],
    responses: list[Response] | None = None,
    budget: int | None = None,
    constraints: list[Constraint] | None = None,
    hard_to_change_factors: list[str] | None = None,
    prior_knowledge: str | None = None,
    existing_data: Any = None,  # noqa: ANN401 - DataFrame or None
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
    _refuse_unplannable(factors, hard_to_change_factors)

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
        detail_level=cast("Literal['novice', 'intermediate']", detail_level),
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
        n_center_points=template.get("min_center_points", 3),
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
        risks=_build_risks(spec, classification, budget_warnings, template),
        alternative_strategies=_build_alternatives(spec, classification),
        domain=domain_enum.value,
        detail_level=detail_level,
        reasoning=_build_reasoning(spec, classification, stages, template),
    )

    return strategy.model_dump()
