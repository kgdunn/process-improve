# (c) Kevin Dunn, 2010-2026. MIT License.

"""Budget allocation logic for multi-stage DOE strategies.

Implements the **25-40-55-15 framework**:
    - Screening:     25-40 % of total budget
    - Optimisation:  40-55 %
    - Confirmation:  5-15 % (minimum 3 runs)

Sources:
    - Montgomery, *Design and Analysis of Experiments*, 10th ed. (25% rule)
    - Stat-Ease SCOR framework
    - NIST Engineering Statistics Handbook section 5.3.3
"""

from __future__ import annotations

import math
from typing import Any

# ---------------------------------------------------------------------------
# Run estimation look-up tables
# ---------------------------------------------------------------------------

# Box-Behnken run counts (excluding center points)
_BBD_RUNS: dict[int, int] = {
    3: 12,
    4: 24,
    5: 40,
    6: 48,
    7: 56,
}

# CCD factorial portion: 2^k for k <= 5, 2^(k-1) for k >= 6
_CCD_FACTORIAL_RUNS: dict[int, int] = {
    2: 4,
    3: 8,
    4: 16,
    5: 32,
    6: 32,  # half-fraction
    7: 64,  # half-fraction
}


# ---------------------------------------------------------------------------
# Run estimation functions
# ---------------------------------------------------------------------------


def estimate_screening_runs(n_factors: int, design_type: str) -> int:
    """Estimate the number of runs for a screening design.

    Parameters
    ----------
    n_factors : int
        Number of factors to screen.
    design_type : str
        One of ``"plackett_burman"``, ``"definitive_screening"``,
        ``"fractional_factorial"``, ``"full_factorial"``.

    Returns
    -------
    int
        Estimated run count including center points.
    """
    if design_type == "plackett_burman":
        # Next multiple of 4 >= k + 1
        n = n_factors + 1
        return int(math.ceil(n / 4) * 4)

    if design_type == "definitive_screening":
        return 2 * n_factors + 1

    if design_type == "fractional_factorial":
        # Smallest 2^(k-p) with resolution >= IV
        if n_factors <= 4:
            return 2**n_factors  # full factorial feasible
        if n_factors == 5:
            return 16  # 2^(5-1) = 16, resolution V
        if n_factors == 6:
            return 16  # 2^(6-2) = 16, resolution IV
        if n_factors == 7:
            return 16  # 2^(7-3) = 16, resolution IV (with minimum aberration)
        # k >= 8: 2^(k-p) where p gives resolution >= IV
        # Conservative: use 32 runs for 8-11 factors, 64 for 12+
        if n_factors <= 11:
            return 32
        return 64

    if design_type == "full_factorial":
        return 2**n_factors

    # Fallback: PB estimate
    n = n_factors + 1
    return int(math.ceil(n / 4) * 4)


def estimate_rsm_runs(n_factors: int, design_type: str, center_points: int = 3) -> int:
    """Estimate the number of runs for an RSM design.

    Parameters
    ----------
    n_factors : int
        Number of factors (typically 2-5 after screening).
    design_type : str
        One of ``"ccd"``, ``"box_behnken"``, ``"ccd_face_centered"``,
        ``"d_optimal"``.
    center_points : int
        Number of center point replicates (default 3).

    Returns
    -------
    int
        Estimated run count.
    """
    if design_type in ("ccd", "ccd_face_centered"):
        factorial = _CCD_FACTORIAL_RUNS.get(n_factors, 2**n_factors)
        axial = 2 * n_factors
        return factorial + axial + center_points

    if design_type == "box_behnken":
        base = _BBD_RUNS.get(n_factors, 0)
        if base == 0:
            # BBD not defined for this factor count; fall back to CCD estimate
            factorial = _CCD_FACTORIAL_RUNS.get(n_factors, 2**n_factors)
            return factorial + 2 * n_factors + center_points
        return base + center_points

    if design_type == "d_optimal":
        # Rough estimate: 1.5x the number of model terms for a quadratic model
        # Quadratic model terms: 1 + k + k(k-1)/2 + k = 1 + 2k + k(k-1)/2
        n_terms = 1 + 2 * n_factors + n_factors * (n_factors - 1) // 2
        return int(math.ceil(1.5 * n_terms))

    # Fallback: CCD estimate
    factorial = _CCD_FACTORIAL_RUNS.get(n_factors, 2**n_factors)
    return factorial + 2 * n_factors + center_points


def estimate_confirmation_runs(min_runs: int = 3) -> int:
    """Return the number of confirmation runs.

    Parameters
    ----------
    min_runs : int
        Minimum confirmation runs (default 3).

    Returns
    -------
    int
        Confirmation run count (always at least 3).
    """
    return max(3, min_runs)


# ---------------------------------------------------------------------------
# Budget allocation
# ---------------------------------------------------------------------------


def allocate_budget(
    total_budget: int | None,
    n_factors: int,
    needs_screening: bool,
    needs_rsm: bool,
    screening_design: str = "plackett_burman",
    rsm_design: str = "box_behnken",
    domain_weights: dict[str, float] | None = None,
    min_confirmation: int = 3,
    center_points: int = 3,
) -> dict[str, Any]:
    """Allocate a total run budget across experimental stages.

    Parameters
    ----------
    total_budget : int or None
        Total runs across all stages.  If ``None``, computes an ideal budget.
    n_factors : int
        Total number of candidate factors.
    needs_screening : bool
        Whether a screening stage is needed.
    needs_rsm : bool
        Whether an RSM optimisation stage is needed.
    screening_design : str
        Preferred screening design type.
    rsm_design : str
        Preferred RSM design type.
    domain_weights : dict or None
        Stage-to-fraction mapping from the domain template.
    min_confirmation : int
        Minimum confirmation runs (domain-dependent).
    center_points : int
        Center points for RSM design.

    Returns
    -------
    dict
        Keys: ``"screening"``, ``"optimization"``, ``"confirmation"``,
        ``"total"``, ``"ideal_total"``, ``"is_tight"``, ``"warnings"``.
    """
    weights = domain_weights or {"screening": 0.30, "optimization": 0.50, "confirmation": 0.10}
    warnings: list[str] = []

    # Estimate ideal runs for each stage
    ideal_screening = estimate_screening_runs(n_factors, screening_design) if needs_screening else 0
    # Assume screening reduces to ~3 significant factors for RSM
    n_rsm_factors = min(n_factors, 3) if needs_screening else n_factors
    ideal_rsm = estimate_rsm_runs(n_rsm_factors, rsm_design, center_points) if needs_rsm else 0
    ideal_confirmation = estimate_confirmation_runs(min_confirmation)
    ideal_total = ideal_screening + ideal_rsm + ideal_confirmation

    if total_budget is None:
        # No budget constraint: use ideal allocation
        return {
            "screening": ideal_screening,
            "optimization": ideal_rsm,
            "confirmation": ideal_confirmation,
            "total": ideal_total,
            "ideal_total": ideal_total,
            "is_tight": False,
            "warnings": [],
        }

    # Check if budget is feasible
    minimum_feasible = 0
    if needs_screening:
        minimum_feasible += max(n_factors + 1, 4)  # Absolute minimum: k+1 runs
    if needs_rsm:
        minimum_feasible += n_rsm_factors * 2 + 3  # Bare minimum RSM
    minimum_feasible += 3  # Minimum confirmation

    is_tight = total_budget < ideal_total
    is_very_tight = total_budget < minimum_feasible

    if is_very_tight:
        warnings.append(
            f"Budget of {total_budget} runs is very tight for {n_factors} factors. "
            f"Minimum feasible is ~{minimum_feasible} runs. "
            "Consider a single Definitive Screening Design to combine screening and curvature detection."
        )

    # Allocate proportionally, respecting minimums
    if needs_screening and needs_rsm:
        # Full multi-stage allocation
        screening_share = weights.get("screening", 0.30)
        rsm_share = weights.get("optimization", 0.50)
        conf_share = weights.get("confirmation", 0.10)

        alloc_confirmation = max(min_confirmation, int(total_budget * conf_share))
        remaining = total_budget - alloc_confirmation

        if remaining <= 0:
            alloc_confirmation = min_confirmation
            remaining = total_budget - alloc_confirmation

        # Split remaining between screening and RSM
        total_share = screening_share + rsm_share
        alloc_screening = max(n_factors + 1, int(remaining * screening_share / total_share))
        alloc_rsm = remaining - alloc_screening

        # Ensure RSM has minimum runs
        if alloc_rsm < n_rsm_factors * 2 + 3:
            alloc_rsm = min(n_rsm_factors * 2 + 3, remaining)
            alloc_screening = remaining - alloc_rsm

        if alloc_screening < n_factors + 1:
            alloc_screening = n_factors + 1
            alloc_rsm = remaining - alloc_screening
    elif needs_screening:
        alloc_screening = total_budget - min_confirmation
        alloc_rsm = 0
        alloc_confirmation = min_confirmation
    elif needs_rsm:
        alloc_screening = 0
        alloc_rsm = total_budget - min_confirmation
        alloc_confirmation = min_confirmation
    else:
        alloc_screening = 0
        alloc_rsm = 0
        alloc_confirmation = total_budget

    # Clamp all values to be non-negative
    alloc_screening = max(0, alloc_screening)
    alloc_rsm = max(0, alloc_rsm)
    alloc_confirmation = max(0, alloc_confirmation)

    if is_tight and not is_very_tight:
        warnings.append(
            f"Budget of {total_budget} runs is tight (ideal is {ideal_total}). "
            "Designs may have reduced power or fewer center points."
        )

    return {
        "screening": alloc_screening,
        "optimization": alloc_rsm,
        "confirmation": alloc_confirmation,
        "total": alloc_screening + alloc_rsm + alloc_confirmation,
        "ideal_total": ideal_total,
        "is_tight": is_tight,
        "warnings": warnings,
    }
