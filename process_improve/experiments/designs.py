# (c) Kevin Dunn, 2010-2026. MIT License.

"""Unified design generation: ``generate_design()`` dispatcher.

This module provides a single entry point for creating any standard
experimental design.  It dispatches to specialised modules based on
``design_type`` and applies common post-processing (center points,
replication, randomization, coded/actual mapping).

Examples
--------
>>> from process_improve.experiments import generate_design, Factor
>>> factors = [
...     Factor(name="Temperature", low=150, high=200, units="degC"),
...     Factor(name="Pressure", low=1, high=5, units="bar"),
... ]
>>> result = generate_design(factors, design_type="full_factorial")
>>> result.n_runs
7
>>> result.design
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from pyDOE3 import ff2n

from process_improve.experiments.designs_utils import build_design_result
from process_improve.experiments.factor import Constraint, DesignResult, Factor, FactorType

# ---------------------------------------------------------------------------
# Dispatch handlers — each returns (coded_matrix, metadata_dict)
# ---------------------------------------------------------------------------


def _dispatch_full_factorial(
    factors: list[Factor],
    **kwargs: Any,  # noqa: ANN401
) -> tuple[np.ndarray, dict]:
    """Full 2^k factorial using pyDOE3.ff2n (returns -1/+1)."""
    k = len(factors)
    coded_matrix = ff2n(k)
    return coded_matrix, {}


def _dispatch_fractional_factorial(
    factors: list[Factor],
    **kwargs: Any,  # noqa: ANN401
) -> tuple[np.ndarray, dict]:
    from process_improve.experiments.designs_screening import dispatch_fractional_factorial  # noqa: PLC0415

    return dispatch_fractional_factorial(
        factors,
        resolution=kwargs.get("resolution"),
        generators=kwargs.get("generators"),
    )


def _dispatch_plackett_burman(
    factors: list[Factor],
    **kwargs: Any,  # noqa: ANN401
) -> tuple[np.ndarray, dict]:
    from process_improve.experiments.designs_screening import dispatch_plackett_burman  # noqa: PLC0415

    return dispatch_plackett_burman(factors)


def _dispatch_box_behnken(
    factors: list[Factor],
    **kwargs: Any,  # noqa: ANN401
) -> tuple[np.ndarray, dict]:
    from process_improve.experiments.designs_response_surface import dispatch_box_behnken  # noqa: PLC0415

    return dispatch_box_behnken(factors, center_points=kwargs.get("center_points", 3))


def _dispatch_ccd(
    factors: list[Factor],
    **kwargs: Any,  # noqa: ANN401
) -> tuple[np.ndarray, dict]:
    from process_improve.experiments.designs_response_surface import dispatch_ccd  # noqa: PLC0415

    return dispatch_ccd(
        factors,
        center_points=kwargs.get("center_points", 3),
        alpha=kwargs.get("alpha"),
    )


def _dispatch_dsd(
    factors: list[Factor],
    **kwargs: Any,  # noqa: ANN401
) -> tuple[np.ndarray, dict]:
    from process_improve.experiments.designs_response_surface import dispatch_dsd  # noqa: PLC0415

    return dispatch_dsd(factors)


def _dispatch_d_optimal(
    factors: list[Factor],
    **kwargs: Any,  # noqa: ANN401
) -> tuple[np.ndarray, dict]:
    from process_improve.experiments.designs_optimal import dispatch_d_optimal  # noqa: PLC0415

    return dispatch_d_optimal(
        factors,
        budget=kwargs.get("budget"),
        hard_to_change=kwargs.get("hard_to_change"),
        constraints=kwargs.get("constraints"),
        model_type=kwargs.get("model_type", "interactions"),
    )


def _dispatch_i_optimal(
    factors: list[Factor],
    **kwargs: Any,  # noqa: ANN401
) -> tuple[np.ndarray, dict]:
    from process_improve.experiments.designs_optimal import dispatch_i_optimal  # noqa: PLC0415

    return dispatch_i_optimal(
        factors,
        budget=kwargs.get("budget"),
        hard_to_change=kwargs.get("hard_to_change"),
        constraints=kwargs.get("constraints"),
        model_type=kwargs.get("model_type", "interactions"),
    )


def _dispatch_a_optimal(
    factors: list[Factor],
    **kwargs: Any,  # noqa: ANN401
) -> tuple[np.ndarray, dict]:
    from process_improve.experiments.designs_optimal import dispatch_a_optimal  # noqa: PLC0415

    return dispatch_a_optimal(
        factors,
        budget=kwargs.get("budget"),
        hard_to_change=kwargs.get("hard_to_change"),
        constraints=kwargs.get("constraints"),
        model_type=kwargs.get("model_type", "interactions"),
    )


def _dispatch_mixture(
    factors: list[Factor],
    **kwargs: Any,  # noqa: ANN401
) -> tuple[np.ndarray, dict]:
    from process_improve.experiments.designs_mixture import dispatch_mixture  # noqa: PLC0415

    return dispatch_mixture(factors, budget=kwargs.get("budget"))


def _dispatch_taguchi(
    factors: list[Factor],
    **kwargs: Any,  # noqa: ANN401
) -> tuple[np.ndarray, dict]:
    from process_improve.experiments.designs_screening import dispatch_taguchi  # noqa: PLC0415

    return dispatch_taguchi(factors)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_DESIGN_REGISTRY: dict[str, Callable[..., tuple[np.ndarray, dict]]] = {
    "full_factorial": _dispatch_full_factorial,
    "fractional_factorial": _dispatch_fractional_factorial,
    "plackett_burman": _dispatch_plackett_burman,
    "box_behnken": _dispatch_box_behnken,
    "ccd": _dispatch_ccd,
    "dsd": _dispatch_dsd,
    "d_optimal": _dispatch_d_optimal,
    "i_optimal": _dispatch_i_optimal,
    "a_optimal": _dispatch_a_optimal,
    "mixture": _dispatch_mixture,
    "taguchi": _dispatch_taguchi,
}


# ---------------------------------------------------------------------------
# Auto-selection
# ---------------------------------------------------------------------------


def _auto_select(
    factors: list[Factor],
    budget: int | None,
    constraints: list[Constraint] | None,
    hard_to_change: list[str] | None,
) -> str:
    """Choose the best design type based on factors, budget, and constraints.

    Parameters
    ----------
    factors : list[Factor]
        Factor specifications.
    budget : int or None
        Maximum number of runs the experimenter can afford.
    constraints : list[Constraint] or None
        Factor-space constraints.
    hard_to_change : list[str] or None
        Names of hard-to-change factors.

    Returns
    -------
    str
        Selected design type key.
    """
    k_mixture = sum(1 for f in factors if f.type == FactorType.mixture)
    k = len(factors) - k_mixture

    # Mixture factors dominate
    if k_mixture > 0 and k_mixture == len(factors):
        return "mixture"

    # Constraints or split-plot -> D-optimal
    if constraints or hard_to_change:
        return "d_optimal"

    effective_budget = budget if budget is not None else float("inf")

    if k <= 5 and effective_budget >= 2**k:
        return "full_factorial"

    if k >= 6 and effective_budget <= 2 * k + 1:
        return "plackett_burman"

    if effective_budget >= 2 ** (k - 1):
        return "fractional_factorial"

    return "d_optimal"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate_design(  # noqa: PLR0913
    factors: list[Factor],
    design_type: str | None = None,
    budget: int | None = None,
    center_points: int = 3,
    replicates: int = 1,
    blocks: int | None = None,
    resolution: int | None = None,
    generators: list[str] | None = None,
    alpha: str | float | None = None,
    constraints: list[Constraint] | None = None,
    hard_to_change: list[str] | None = None,
    random_seed: int = 42,
) -> DesignResult:
    """Generate an experimental design matrix.

    Parameters
    ----------
    factors : list[Factor]
        Factor specifications.  Each ``Factor`` has a *name*, *type*
        (``"continuous"``, ``"categorical"``, ``"mixture"``), *low*/*high*
        bounds (for continuous), *levels* (for categorical), and optional
        *units*.
    design_type : str or None
        One of ``"full_factorial"``, ``"fractional_factorial"``,
        ``"plackett_burman"``, ``"box_behnken"``, ``"ccd"``, ``"dsd"``,
        ``"d_optimal"``, ``"i_optimal"``, ``"a_optimal"``, ``"mixture"``,
        ``"taguchi"``.
        If ``None``, the design type is chosen automatically based on the
        factor count, budget, and constraints.
    budget : int or None
        Maximum number of runs the experimenter can afford.
    center_points : int
        Number of center-point replicates (default 3).  For designs that
        embed their own center points (CCD, Box-Behnken), this parameter
        controls the count within the design structure.
    replicates : int
        Number of full replicates of the design (default 1 = no replication).
    blocks : int or None
        Number of blocks.
    resolution : int or None
        Desired minimum resolution for fractional factorials (III=3, IV=4, V=5).
    generators : list[str] or None
        Explicit generators for fractional factorials,
        e.g. ``["D=ABC", "E=AC"]``.
    alpha : str, float, or None
        Axial distance for CCD designs: ``"rotatable"``,
        ``"face_centered"``, ``"orthogonal"``, or a numeric value.
    constraints : list[Constraint] or None
        Constraints on the factor space.
    hard_to_change : list[str] or None
        Names of hard-to-change factors (triggers split-plot structure).
    random_seed : int
        Seed for reproducible randomization (default 42).

    Returns
    -------
    DesignResult
        Contains ``design`` (coded ``Expt``), ``design_actual`` (actual-units
        ``Expt``), ``run_order``, and design metadata (generators, defining
        relation, resolution, etc.).

    Raises
    ------
    ValueError
        If *design_type* is unknown, or if factor/budget constraints
        cannot be satisfied.

    Examples
    --------
    >>> from process_improve.experiments import generate_design, Factor
    >>> factors = [
    ...     Factor(name="T", low=150, high=200, units="degC"),
    ...     Factor(name="P", low=1, high=5, units="bar"),
    ... ]
    >>> result = generate_design(factors, design_type="full_factorial")
    >>> result.design_actual
    """
    # --- Validate ----------------------------------------------------------
    if not factors:
        raise ValueError("At least one factor must be provided.")

    if design_type is None:
        design_type = _auto_select(factors, budget, constraints, hard_to_change)

    if design_type not in _DESIGN_REGISTRY:
        raise ValueError(
            f"Unknown design_type={design_type!r}.  "
            f"Choose from: {', '.join(sorted(_DESIGN_REGISTRY))}."
        )

    # --- Dispatch ----------------------------------------------------------
    dispatch_fn = _DESIGN_REGISTRY[design_type]

    # Build kwargs for the dispatch handler
    dispatch_kwargs: dict[str, Any] = {
        "budget": budget,
        "center_points": center_points,
        "resolution": resolution,
        "generators": generators,
        "alpha": alpha,
        "hard_to_change": hard_to_change,
        "constraints": constraints,
    }

    coded_matrix, meta = dispatch_fn(factors, **dispatch_kwargs)

    # --- Determine center-point handling -----------------------------------
    # Designs that embed their own center points (CCD, Box-Behnken)
    # already include them; don't add more.
    designs_with_embedded_centers = {"ccd", "box_behnken", "dsd", "mixture", "d_optimal", "i_optimal", "a_optimal"}

    # Optimal designs from pyoptex produce a pre-optimized run order
    # (especially important for split-plot).  Skip randomization for these.
    optimal_designs = {"d_optimal", "i_optimal", "a_optimal"}
    if design_type in optimal_designs and meta.get("backend") == "pyoptex":
        random_seed = None  # signal to build_design_result to skip randomization
    extra_center_points = 0 if design_type in designs_with_embedded_centers else center_points

    # Mixture designs return proportions (actual units), not coded
    is_actual = design_type == "mixture"

    # Extract resolution/generators/defining_relation from metadata
    result_generators = generators or meta.get("generators_used")
    result_resolution = resolution or meta.get("resolution")
    result_alpha = meta.pop("alpha_value", None)

    return build_design_result(
        coded_matrix=coded_matrix,
        factors=factors,
        design_type=design_type,
        center_points=extra_center_points,
        replicates=replicates,
        blocks=blocks,
        random_seed=random_seed,
        generators=result_generators,
        defining_relation=meta.get("defining_relation"),
        resolution=result_resolution,
        alpha=result_alpha,
        metadata=meta,
        is_actual=is_actual,
    )
