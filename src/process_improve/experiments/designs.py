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

import functools
import itertools
import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

try:
    from pyDOE3 import ff2n
except ImportError:  # pragma: no cover - exercised via env-without-pyDOE3
    from process_improve._extras import _MissingExtra

    ff2n = _MissingExtra("pyDOE3", "expt")  # type: ignore[assignment]

from process_improve._random import resolve_deprecated_seed
from process_improve.experiments.designs_utils import build_design_result, categorical_codes
from process_improve.experiments.factor import Constraint, DesignResult, Factor, FactorType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dispatch handlers - each returns (coded_matrix, metadata_dict)
# ---------------------------------------------------------------------------


def _factor_codes(factor: Factor) -> np.ndarray:
    """Coded settings a full factorial gives ``factor``.

    -1 and +1 for a continuous factor; its ``levels`` (actual values inside
    ``[low, high]``) coded to ``[-1, 1]`` when given; the level codes of a categorical
    factor (see :func:`~process_improve.experiments.designs_utils.categorical_codes`).
    """
    if factor.type == FactorType.categorical:
        return categorical_codes(len(factor.levels or []))
    if not factor.levels:
        return np.array([-1.0, 1.0])
    low, high = float(factor.low), float(factor.high)  # type: ignore[arg-type]
    values = np.unique(np.asarray(factor.levels, dtype=float))
    if values.min() < low or values.max() > high:
        raise ValueError(f"Factor {factor.name!r}: levels {values.tolist()} must lie within low={low} and high={high}.")
    return (values - (low + high) / 2.0) / ((high - low) / 2.0)


def _dispatch_full_factorial(
    factors: list[Factor],
    **kwargs: Any,  # noqa: ANN401
) -> tuple[np.ndarray, dict]:
    """Full factorial: every combination of every factor's levels, first factor changing fastest.

    Two-level factors give pyDOE3's ``ff2n`` 2^k design; a continuous factor with
    ``levels`` or a categorical factor with more than two levels gives the general
    (mixed-level) full factorial.
    """
    sets = [_factor_codes(f) for f in factors]
    if all(len(codes) == 2 and codes.tolist() == [-1.0, 1.0] for codes in sets):
        return ff2n(len(factors)), {}
    grid = np.array(list(itertools.product(*sets[::-1])))[:, ::-1]
    return grid, {"levels_per_factor": [len(codes) for codes in sets]}


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

    return dispatch_box_behnken(factors, n_center_points=kwargs.get("n_center_points", 3))


def _dispatch_ccd(
    factors: list[Factor],
    **kwargs: Any,  # noqa: ANN401
) -> tuple[np.ndarray, dict]:
    from process_improve.experiments.designs_response_surface import dispatch_ccd  # noqa: PLC0415

    return dispatch_ccd(
        factors,
        n_center_points=kwargs.get("n_center_points", 3),
        alpha=kwargs.get("alpha"),
        cube=kwargs.get("cube", "full"),
        generators=kwargs.get("generators"),
        resolution=kwargs.get("resolution"),
    )


def _dispatch_dsd(
    factors: list[Factor],
    **kwargs: Any,  # noqa: ANN401
) -> tuple[np.ndarray, dict]:
    from process_improve.experiments.designs_response_surface import dispatch_dsd  # noqa: PLC0415

    return dispatch_dsd(factors)


def _dispatch_omars(
    factors: list[Factor],
    **kwargs: Any,  # noqa: ANN401
) -> tuple[np.ndarray, dict]:
    # With a run budget, reach the ILP enumerator (a larger OMARS member that
    # leaves error degrees of freedom for a full second-order model); without a
    # budget, return the minimal conference-foldover member, which is identical
    # to the definitive screening design.  This is a thin wrapper over
    # ``design_type="omars_ilp"`` so both spellings reach the same enumerator.
    if kwargs.get("budget") is not None:
        return _dispatch_omars_ilp(factors, **kwargs)

    from process_improve.experiments.designs_omars import dispatch_omars  # noqa: PLC0415

    return dispatch_omars(factors)


def _dispatch_omars_ilp(
    factors: list[Factor],
    **kwargs: Any,  # noqa: ANN401
) -> tuple[np.ndarray, dict]:
    from process_improve.experiments.designs_omars_ilp import _dispatch_omars_ilp as _run  # noqa: PLC0415

    return _run(factors, budget=kwargs.get("budget"), random_state=kwargs.get("random_state"))


def _dispatch_optimal_family(
    criterion: str,
    factors: list[Factor],
    **kwargs: Any,  # noqa: ANN401
) -> tuple[np.ndarray, dict]:
    from process_improve.experiments.designs_optimal import _dispatch_optimal, _OptimalRequest  # noqa: PLC0415

    request = _OptimalRequest(
        factors,
        budget=kwargs.get("budget"),
        hard_to_change=kwargs.get("hard_to_change"),
        constraints=kwargs.get("constraints"),
        model_type=kwargs.get("model_type", "interactions"),
        fixed_runs=kwargs.get("fixed_runs"),
        random_state=kwargs.get("random_state"),
        candidates=kwargs.get("candidates"),
        backend=kwargs.get("backend", "auto"),
    )
    return _dispatch_optimal(criterion, request)


def _dispatch_supersaturated(
    factors: list[Factor],
    **kwargs: Any,  # noqa: ANN401
) -> tuple[np.ndarray, dict]:
    from process_improve.experiments.designs_supersaturated import dispatch_supersaturated  # noqa: PLC0415

    return dispatch_supersaturated(factors, budget=kwargs.get("budget"))


def _dispatch_space_filling(
    method: str,
    factors: list[Factor],
    **kwargs: Any,  # noqa: ANN401
) -> tuple[np.ndarray, dict]:
    from process_improve.experiments.designs_space_filling import space_filling_design  # noqa: PLC0415

    return space_filling_design(
        factors, kwargs.get("budget"), method, kwargs.get("constraints"), kwargs.get("random_state")
    )


def _dispatch_mixture(
    factors: list[Factor],
    **kwargs: Any,  # noqa: ANN401
) -> tuple[np.ndarray, dict]:
    from process_improve.experiments.designs_mixture import dispatch_mixture  # noqa: PLC0415

    return dispatch_mixture(
        factors,
        budget=kwargs.get("budget"),
        constraints=kwargs.get("constraints"),
        model_type=kwargs.get("model_type", "interactions"),
        random_state=kwargs.get("random_state"),
    )


def _dispatch_taguchi(
    factors: list[Factor],
    **kwargs: Any,  # noqa: ANN401
) -> tuple[np.ndarray, dict]:
    from process_improve.experiments.designs_screening import dispatch_taguchi  # noqa: PLC0415

    return dispatch_taguchi(factors)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Space-filling design types (see designs_space_filling.py); none adds centre points.
_SPACE_FILLING = ("latin_hypercube", "maximin_lhs", "uniform", "sobol", "halton", "maximin")

#: Design types chosen by an optimality criterion; the only ones that take fixed_runs or candidates.
_OPTIMAL_FAMILIES = frozenset({"d_optimal", "i_optimal", "a_optimal", "e_optimal"})


#: Design types that accept categorical factors: the optimal families with any number of
#: levels, the full factorial and Taguchi arrays with the levels the design provides,
#: and the two-level designs with two-level categorical factors.
_CATEGORICAL_ANY_LEVELS = frozenset({"full_factorial", "taguchi", *_OPTIMAL_FAMILIES})
_CATEGORICAL_TWO_LEVELS = frozenset({"fractional_factorial", "plackett_burman", "dsd"})


def _refuse_unsupported_categorical(factors: list[Factor], design_type: str) -> None:
    """Raise one clear message when ``design_type`` cannot place a categorical factor."""
    categorical = [f for f in factors if f.type == FactorType.categorical]
    if not categorical or design_type in _CATEGORICAL_ANY_LEVELS:
        return
    if design_type in _CATEGORICAL_TWO_LEVELS:
        many = [f.name for f in categorical if len(f.levels or []) != 2]
        if not many:
            return
        raise ValueError(
            f"design_type={design_type!r} takes two-level categorical factors only; {many} have more levels. "
            "Use 'full_factorial', 'taguchi' or an optimal design ('d_optimal', 'i_optimal')."
        )
    raise ValueError(
        f"design_type={design_type!r} needs continuous factors, but {[f.name for f in categorical]} are "
        "categorical. Use 'd_optimal' or 'i_optimal' (any levels), 'dsd', 'fractional_factorial' or "
        "'plackett_burman' (two-level categorical factors), 'full_factorial' or 'taguchi'."
    )


def _refuse_mixture_process(factors: list[Factor]) -> None:
    """Raise one clear error for mixture components mixed with process factors, which no engine here supports.

    The proportions must sum to 1 while the process factors range freely, so neither
    the mixture engine nor the factor-box engines can place the runs. Crossing a
    mixture design with a design in the process factors is the classical answer.
    """
    is_mixture = [f.type == FactorType.mixture for f in factors]
    if any(is_mixture) and not all(is_mixture):
        process = [f.name for f, m in zip(factors, is_mixture, strict=True) if not m]
        raise ValueError(
            f"Mixture-process designs (mixture components together with process factors {process}) are not "
            "supported. Generate a mixture design for the components and a design for the process factors, "
            "then cross them: every blend at every process setting, e.g. "
            "mixture.design_actual.merge(process.design_actual, how='cross')."
        )


_DESIGN_REGISTRY: dict[str, Callable[..., tuple[np.ndarray, dict]]] = {
    "full_factorial": _dispatch_full_factorial,
    "fractional_factorial": _dispatch_fractional_factorial,
    "plackett_burman": _dispatch_plackett_burman,
    "box_behnken": _dispatch_box_behnken,
    "ccd": _dispatch_ccd,
    "dsd": _dispatch_dsd,
    "omars": _dispatch_omars,
    "omars_ilp": _dispatch_omars_ilp,
    "d_optimal": functools.partial(_dispatch_optimal_family, "d_optimal"),
    "i_optimal": functools.partial(_dispatch_optimal_family, "i_optimal"),
    "a_optimal": functools.partial(_dispatch_optimal_family, "a_optimal"),
    "e_optimal": functools.partial(_dispatch_optimal_family, "e_optimal"),
    "mixture": _dispatch_mixture,
    "taguchi": _dispatch_taguchi,
    "supersaturated": _dispatch_supersaturated,
    **{name: functools.partial(_dispatch_space_filling, name) for name in _SPACE_FILLING},
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

    return _auto_select_by_budget(factors, k, budget if budget is not None else float("inf"))


def _auto_select_by_budget(factors: list[Factor], k: int, budget: float) -> str:
    """Pick the unconstrained design family that fits ``budget`` runs for ``k`` process factors."""
    if k <= 5 and budget >= 2**k:
        return "full_factorial"
    # Fewer runs than main effects: a supersaturated design screens them all, when one exists
    # for this budget without fully aliased factors; otherwise fall through as before.
    if all(f.type == FactorType.continuous for f in factors):
        from process_improve.experiments.designs_supersaturated import supersaturated_available  # noqa: PLC0415

        if supersaturated_available(k, budget):
            return "supersaturated"
    if k >= 6 and budget <= 2 * k + 1:
        return "plackett_burman"
    if budget >= 2 ** (k - 1):
        return "fractional_factorial"
    return "d_optimal"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate_design(  # noqa: PLR0913
    factors: list[Factor],
    design_type: str | None = None,
    budget: int | None = None,
    n_center_points: int = 3,
    n_replicates: int = 1,
    n_blocks: int | None = None,
    resolution: int | None = None,
    generators: list[str] | None = None,
    alpha: str | float | None = None,
    cube: str = "full",
    constraints: list[Constraint] | None = None,
    hard_to_change: list[str] | None = None,
    model_type: str = "interactions",
    fixed_runs: pd.DataFrame | None = None,
    random_seed: int | None = None,
    candidates: pd.DataFrame | None = None,
    random_state: int | np.random.Generator | None = 42,
    backend: str = "auto",
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
        ``"omars"``, ``"omars_ilp"``, ``"d_optimal"``, ``"i_optimal"``,
        ``"a_optimal"``, ``"e_optimal"``, ``"mixture"``, ``"taguchi"``, ``"supersaturated"``, and the
        space-filling types ``"latin_hypercube"``, ``"maximin_lhs"``, ``"uniform"``, ``"sobol"``,
        ``"halton"`` and ``"maximin"`` (``budget`` runs, default ``10 * k``).
        If ``None``, the design type is chosen automatically based on the
        factor count, budget, and constraints; an automatically chosen fractional
        factorial for six or more factors is the smallest one of resolution IV.
    budget : int or None
        Maximum number of runs the experimenter can afford.
    n_center_points : int
        Number of center-point replicates (default 3).  For designs that
        embed their own center points (CCD, Box-Behnken), this parameter
        controls the count within the design structure.
    n_replicates : int
        Number of full replicates of the design (default 1 = no replication).
    n_blocks : int or None
        Number of blocks.
    resolution : int or None
        Desired minimum resolution for fractional factorials (III=3, IV=4, V=5).
        The design is the minimum-aberration fraction with the fewest runs that
        reaches it. Without *resolution* or *generators*, a fractional factorial
        is the half fraction 2^(k-1), of resolution k. The result reports the
        resolution the design achieves.
    generators : list[str] or None
        Explicit generators for fractional factorials,
        e.g. ``["D=ABC", "E=AC"]``.
    alpha : str, float, or None
        Axial distance for CCD designs: ``"rotatable"``, ``"face_centered"``,
        ``"inscribed"``, ``"orthogonal"`` (the default), or a positive number. Any
        other value raises ``ValueError``.
    cube : str
        For CCD designs, how to build the cube (factorial) portion:
        ``"full"`` (default) uses the complete 2^k factorial; ``"fractional"``
        uses a resolution-V (or higher) fractional factorial, keeping the run
        count practical for k >= 5.  When ``"fractional"`` and *generators* is
        given, those generators define the cube; otherwise a minimum-aberration
        half-fraction is chosen automatically.
    constraints : list[Constraint] or None
        Inequalities on the continuous factors, in actual units, e.g.
        ``Constraint(expression="3*T + 5*D <= 600")``. Honoured by the optimal
        families (``"d_optimal"``, chosen automatically when constraints are given,
        ``"i_optimal"``, ``"a_optimal"`` and ``"e_optimal"``), whose runs are selected
        from a candidate set of feasible points; by ``"mixture"`` (linear constraints
        in the proportions); and by the ``"sobol"``, ``"halton"`` and ``"maximin"``
        space-filling designs. Other design types do not enforce them and set
        ``metadata["constraints_enforced"] = False``.
    hard_to_change : list[str] or None
        Names of hard-to-change factors (triggers split-plot structure).
    model_type : str
        Model the optimal designs (``"d_optimal"``, ``"i_optimal"``,
        ``"a_optimal"``) are built for: ``"main_effects"``, ``"interactions"``
        (default), or ``"quadratic"``.  With a categorical factor present,
        ``"quadratic"`` builds a partial response-surface model (quadratics on
        the continuous factors only; the categorical enters as a main effect
        plus its interactions), since a categorical factor has no square.
        Ignored by the classical (non-optimal) design families.
    fixed_runs : pandas.DataFrame or None
        Runs to hold fixed while the optimizer fills the rest (design augmentation), for the
        optimal families only (``"d_optimal"``, ``"i_optimal"``, ``"a_optimal"``, ``"e_optimal"``). One row per
        fixed run, one column per factor, in the same coding as the returned design: continuous
        factors in coded ``[-1, 1]`` units, categorical factors as level labels.
        The fixed runs occupy the first rows of the result and ``budget`` counts them, so
        ``budget`` must exceed ``len(fixed_runs)``. A common use is to seed a centre point. Raises
        ``ValueError`` if given for a non-optimal ``design_type``.
    random_seed : int or None
        Deprecated since 1.97.0 and removed in 2.0; use ``random_state``.
    candidates : pandas.DataFrame or None
        The settings the runs must be chosen from, for the optimal families only: for
        example historical operating points, the discrete settings a piece of
        equipment offers, or blends that can actually be made up. One row per
        candidate, one column per factor, in actual units (categorical factors as
        labels, mixtures as proportions). Replaces the generated candidate grid, so it
        also defines the region I-optimality averages over. Rows that break a
        constraint are dropped. A candidate may be chosen more than once, and
        ``metadata["selected_candidates"]`` counts the picks per index label.
    random_state : int, numpy.random.Generator or None
        Seeds the run-order randomisation and any random search (default 42, so the
        same call gives the same design). ``None`` draws fresh entropy; see
        :doc:`/development/reproducibility`.
    backend : {"auto", "exchange", "pyoptex"}
        Engine for the optimal families. ``"auto"`` (default) uses the built-in
        candidate exchange, and the optional pyoptex package only for a split-plot
        design (``hard_to_change``), so the same call gives the same design whether or
        not pyoptex is installed. ``"pyoptex"`` asks for pyoptex's coordinate exchange.

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
    random_state = resolve_deprecated_seed(random_state, random_seed, "generate_design")
    if not factors:
        raise ValueError("At least one factor must be provided.")

    if design_type is None:
        # A candidate set means "choose runs from these", which only the optimal families do.
        design_type = (
            "d_optimal" if candidates is not None else _auto_select(factors, budget, constraints, hard_to_change)
        )
        if design_type == "fractional_factorial" and resolution is None and generators is None and len(factors) >= 6:
            # Screening many factors: the smallest resolution IV fraction (main effects clear of
            # two-factor interactions), not the half fraction (512 runs for 10 factors).
            resolution = 4

    if design_type not in _DESIGN_REGISTRY:
        raise ValueError(f"Unknown design_type={design_type!r}.  Choose from: {', '.join(sorted(_DESIGN_REGISTRY))}.")

    families = ", ".join(sorted(_OPTIMAL_FAMILIES))
    if candidates is not None and design_type not in _OPTIMAL_FAMILIES:
        raise ValueError(
            f"candidates is only supported for the optimal design families ({families}); "
            f"got design_type={design_type!r}."
        )
    if fixed_runs is not None and design_type not in _OPTIMAL_FAMILIES:
        raise ValueError(
            f"fixed_runs (design augmentation) is only supported for the optimal design families ({families}); "
            f"got design_type={design_type!r}."
        )
    _refuse_mixture_process(factors)
    _refuse_unsupported_categorical(factors, design_type)

    # --- Dispatch ----------------------------------------------------------
    dispatch_fn = _DESIGN_REGISTRY[design_type]

    # Build kwargs for the dispatch handler
    dispatch_kwargs: dict[str, Any] = {
        "budget": budget,
        "n_center_points": n_center_points,
        "resolution": resolution,
        "generators": generators,
        "alpha": alpha,
        "cube": cube,
        "hard_to_change": hard_to_change,
        "constraints": constraints,
        "model_type": model_type,
        "fixed_runs": fixed_runs,
        "random_state": random_state,
        "candidates": candidates,
        "backend": backend,
    }

    coded_matrix, meta = dispatch_fn(factors, **dispatch_kwargs)
    if constraints and not meta.get("constraints_enforced"):
        # Only the constrained D-optimal and mixture paths honour constraints; say so on the result.
        logger.warning("design_type=%r does not enforce constraints; use design_type='d_optimal'.", design_type)
        meta["constraints_enforced"] = False
    all_mixture = all(f.type == FactorType.mixture for f in factors)
    if all_mixture or meta.get("constraints_enforced"):
        # Record the region the runs were placed in, so evaluate_design and
        # optimize_responses can work over the same region (see DesignRegion).
        from process_improve.experiments.region import DesignRegion  # noqa: PLC0415

        meta["region"] = DesignRegion(factors, constraints if meta.get("constraints_enforced") else None).to_dict()

    # --- Determine center-point handling -----------------------------------
    # Designs that embed their own center points (CCD, Box-Behnken)
    # already include them; don't add more.
    designs_with_embedded_centers = {
        "ccd",
        "box_behnken",
        "dsd",
        "omars",
        "omars_ilp",
        "mixture",
        "supersaturated",  # the point is the fewest runs; centre points would spend them on nothing
        "taguchi",  # an orthogonal array is complete as it stands
        *_OPTIMAL_FAMILIES,
        *_SPACE_FILLING,
    }

    # Optimal designs from pyoptex produce a pre-optimized run order
    # (especially important for split-plot).  Skip randomization for these.
    randomize = not (design_type in _OPTIMAL_FAMILIES and meta.get("backend") == "pyoptex")
    extra_center_points = 0 if design_type in designs_with_embedded_centers else n_center_points

    # Mixture designs return proportions (actual units), not coded
    is_actual = design_type == "mixture" or all_mixture

    # Extract resolution/generators/defining_relation from metadata
    result_generators = generators or meta.get("generators_used")
    # The design's own resolution, which can exceed the minimum asked for.
    result_resolution = meta.get("resolution") or resolution
    result_alpha = meta.pop("alpha_value", None)

    return build_design_result(
        coded_matrix=coded_matrix,
        factors=factors,
        design_type=design_type,
        n_center_points=extra_center_points,
        n_replicates=n_replicates,
        n_blocks=n_blocks,
        random_state=random_state,
        randomize=randomize,
        generators=result_generators,
        defining_relation=meta.get("defining_relation"),
        resolution=result_resolution,
        alpha=result_alpha,
        metadata=meta,
        is_actual=is_actual,
        n_leading_fixed=int(meta.get("n_fixed_runs", 0)),
    )
