# (c) Kevin Dunn, 2010-2026. MIT License.

"""Optimal designs: D-optimal, I-optimal, A-optimal.

Uses ``pyoptex`` (coordinate exchange) when available for high-quality
optimal designs with support for split-plot structures.  Falls back to the
built-in ``point_exchange()`` in ``optimal.py`` for D-optimal when
``pyoptex`` is not installed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pyDOE3 import fullfact

from process_improve.experiments.optimal import point_exchange

if TYPE_CHECKING:
    from process_improve.experiments.factor import Constraint, Factor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# pyoptex availability check
# ---------------------------------------------------------------------------

_PYOPTEX_AVAILABLE = False
try:
    from pyoptex.doe.fixed_structure import (
        Factor as PyoptexFactor,
    )
    from pyoptex.doe.fixed_structure import (
        RandomEffect,
        create_fixed_structure_design,
        create_parameters,
        default_fn,
    )
    from pyoptex.doe.fixed_structure.metric import Aopt, Dopt, Iopt
    from pyoptex.utils.model import model2Y2X, partial_rsm_names

    _PYOPTEX_AVAILABLE = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# pyoptex adapter layer
# ---------------------------------------------------------------------------

#: Map our model-type strings to pyoptex's partial_rsm_names keywords.
_PYOPTEX_MODEL_MAP = {
    "main_effects": "lin",
    "interactions": "tfi",
    "quadratic": "quad",
}

#: Map our optimality-criterion strings to pyoptex metric constructors.
_PYOPTEX_METRIC_MAP: dict = {}
if _PYOPTEX_AVAILABLE:
    _PYOPTEX_METRIC_MAP = {
        "d_optimal": Dopt,
        "i_optimal": Iopt,
        "a_optimal": Aopt,
    }


def _convert_factors_to_pyoptex(
    factors: list[Factor],
    hard_to_change: list[str] | None = None,
    n_runs: int | None = None,
    n_whole_plots: int | None = None,
) -> list:
    """Translate our ``Factor`` objects into ``pyoptex.doe.fixed_structure.Factor`` objects.

    Parameters
    ----------
    factors : list[Factor]
        Our Factor specifications.
    hard_to_change : list[str] or None
        Names of hard-to-change factors.  When provided a ``RandomEffect``
        is created that groups consecutive runs into whole plots.
    n_runs : int or None
        Total number of runs (needed when building the split-plot structure).
    n_whole_plots : int or None
        Number of whole plots.  Defaults to ``max(4, n_runs // 3)`` when
        *hard_to_change* is given.

    Returns
    -------
    list[pyoptex Factor]
    """
    from process_improve.experiments.factor import FactorType  # noqa: PLC0415

    random_effect = None
    if hard_to_change and n_runs:
        if n_whole_plots is None:
            n_whole_plots = max(4, n_runs // 3)
        # Build a balanced whole-plot assignment: e.g. 12 runs, 4 plots → [0,0,0, 1,1,1, 2,2,2, 3,3,3]
        runs_per_plot = max(1, n_runs // n_whole_plots)
        z_array = np.repeat(np.arange(n_whole_plots), runs_per_plot)
        # Pad if n_runs doesn't divide evenly
        if len(z_array) < n_runs:
            z_array = np.concatenate([z_array, np.full(n_runs - len(z_array), n_whole_plots - 1)])
        random_effect = RandomEffect(z_array[:n_runs], ratio=0.5)

    htc_set = set(hard_to_change) if hard_to_change else set()
    pyoptex_factors = []
    for f in factors:
        if f.type == FactorType.categorical:
            pf = PyoptexFactor(
                f.name,
                random_effect if f.name in htc_set else None,
                type="categorical",
                levels=f.levels,
            )
        else:
            pf = PyoptexFactor(
                f.name,
                random_effect if f.name in htc_set else None,
                type="continuous",
            )
        pyoptex_factors.append(pf)
    return pyoptex_factors


def _run_pyoptex(  # noqa: PLR0913
    factors: list[Factor],
    criterion: str,
    budget: int,
    model_type: str = "interactions",
    hard_to_change: list[str] | None = None,
    n_tries: int = 10,
) -> tuple[np.ndarray, dict]:
    """Run pyoptex's coordinate-exchange optimizer.

    Parameters
    ----------
    factors : list[Factor]
        Our Factor specifications.
    criterion : str
        One of ``"d_optimal"``, ``"i_optimal"``, ``"a_optimal"``.
    budget : int
        Number of runs in the design.
    model_type : str
        One of ``"main_effects"``, ``"interactions"``, ``"quadratic"``.
    hard_to_change : list[str] or None
        Names of hard-to-change factors (triggers split-plot structure).
    n_tries : int
        Number of random restarts for the coordinate exchange algorithm.

    Returns
    -------
    tuple[np.ndarray, dict]
        Coded design matrix (-1 / +1 for continuous, labels for categorical)
        and metadata dict.
    """
    pyoptex_factors = _convert_factors_to_pyoptex(
        factors,
        hard_to_change=hard_to_change,
        n_runs=budget,
    )

    # Build model matrix
    rsm_key = _PYOPTEX_MODEL_MAP.get(model_type, "tfi")
    model_spec = partial_rsm_names({f.name: rsm_key for f in factors})
    y2x = model2Y2X(model_spec, pyoptex_factors)

    # Select metric
    metric_cls = _PYOPTEX_METRIC_MAP[criterion]
    metric = metric_cls()

    fn = default_fn(pyoptex_factors, metric, y2x)
    params = create_parameters(pyoptex_factors, fn, nruns=budget)
    design_df, state = create_fixed_structure_design(params, n_tries=n_tries)

    meta = {
        "optimality_criterion": criterion,
        "metric_value": float(state.metric),
        "model_type": model_type,
        "backend": "pyoptex",
    }
    if hard_to_change:
        meta["hard_to_change"] = hard_to_change

    return design_df.values, meta


# ---------------------------------------------------------------------------
# Fallback: point-exchange from candidate set (no pyoptex needed)
# ---------------------------------------------------------------------------


def _run_point_exchange_fallback(
    factors: list[Factor],
    budget: int,
) -> tuple[np.ndarray, dict]:
    """D-optimal via built-in point exchange on a 3-level candidate set.

    This is the fallback when pyoptex is not installed.

    Parameters
    ----------
    factors : list[Factor]
        Factor specifications.
    budget : int
        Number of runs to select.

    Returns
    -------
    tuple[np.ndarray, dict]
    """
    k = len(factors)
    # Build candidate set: 3-level full factorial (-1, 0, +1)
    candidates_raw = fullfact([3] * k)
    candidates = candidates_raw - 1.0

    candidates_df = pd.DataFrame(candidates, columns=[f.name for f in factors])
    n_points = min(budget, candidates_df.shape[0])
    n_points = max(n_points, k + 1)

    design_df, d_opt = point_exchange(candidates_df, number_points=n_points)
    return design_df.values, {"d_optimality": float(d_opt), "backend": "point_exchange_fallback"}


# ---------------------------------------------------------------------------
# Public dispatch functions
# ---------------------------------------------------------------------------


def dispatch_d_optimal(
    factors: list[Factor],
    budget: int | None = None,
    hard_to_change: list[str] | None = None,
    constraints: list[Constraint] | None = None,
    model_type: str = "interactions",
) -> tuple[np.ndarray, dict]:
    """Generate a D-optimal design.

    Uses pyoptex's coordinate-exchange algorithm when available (much better
    quality, supports split-plot).  Falls back to the built-in point-exchange
    when pyoptex is not installed.

    Parameters
    ----------
    factors : list[Factor]
        Factor specifications.
    budget : int or None
        Number of runs.  Defaults to ``2 * n_factors + 1``.
    hard_to_change : list[str] or None
        Names of hard-to-change factors (triggers split-plot via pyoptex).
    constraints : list[Constraint] or None
        Factor-space constraints (logged as warning; not yet enforced).
    model_type : str
        Model assumption: ``"main_effects"``, ``"interactions"``, or ``"quadratic"``.

    Returns
    -------
    tuple[np.ndarray, dict]
    """
    k = len(factors)
    if budget is None:
        budget = 2 * k + 1

    if constraints:
        logger.warning(
            "Constraint enforcement in optimal designs is experimental. "
            "Constraints are noted but may not be fully enforced. "
            "Consider filtering infeasible runs manually."
        )

    if _PYOPTEX_AVAILABLE:
        return _run_pyoptex(
            factors,
            criterion="d_optimal",
            budget=budget,
            model_type=model_type,
            hard_to_change=hard_to_change,
        )

    if hard_to_change:
        logger.warning(
            "pyoptex is not installed — hard_to_change factors will be ignored. "
            "Install with: pip install pyoptex"
        )
    return _run_point_exchange_fallback(factors, budget)


def dispatch_i_optimal(
    factors: list[Factor],
    budget: int | None = None,
    hard_to_change: list[str] | None = None,
    constraints: list[Constraint] | None = None,
    model_type: str = "interactions",
) -> tuple[np.ndarray, dict]:
    """Generate an I-optimal design (minimizes average prediction variance).

    Requires pyoptex.

    Parameters
    ----------
    factors : list[Factor]
        Factor specifications.
    budget : int or None
        Number of runs.  Defaults to ``2 * n_factors + 1``.
    hard_to_change : list[str] or None
        Names of hard-to-change factors.
    constraints : list[Constraint] or None
        Factor-space constraints.
    model_type : str
        Model assumption.

    Returns
    -------
    tuple[np.ndarray, dict]

    Raises
    ------
    ImportError
        If pyoptex is not installed.
    """
    if not _PYOPTEX_AVAILABLE:
        raise ImportError(
            "I-optimal design generation requires pyoptex. "
            "Install with: pip install pyoptex"
        )

    k = len(factors)
    if budget is None:
        budget = 2 * k + 1

    return _run_pyoptex(
        factors,
        criterion="i_optimal",
        budget=budget,
        model_type=model_type,
        hard_to_change=hard_to_change,
    )


def dispatch_a_optimal(
    factors: list[Factor],
    budget: int | None = None,
    hard_to_change: list[str] | None = None,
    constraints: list[Constraint] | None = None,
    model_type: str = "interactions",
) -> tuple[np.ndarray, dict]:
    """Generate an A-optimal design (minimizes trace of variance matrix).

    Requires pyoptex.

    Parameters
    ----------
    factors : list[Factor]
        Factor specifications.
    budget : int or None
        Number of runs.  Defaults to ``2 * n_factors + 1``.
    hard_to_change : list[str] or None
        Names of hard-to-change factors.
    constraints : list[Constraint] or None
        Factor-space constraints.
    model_type : str
        Model assumption.

    Returns
    -------
    tuple[np.ndarray, dict]

    Raises
    ------
    ImportError
        If pyoptex is not installed.
    """
    if not _PYOPTEX_AVAILABLE:
        raise ImportError(
            "A-optimal design generation requires pyoptex. "
            "Install with: pip install pyoptex"
        )

    k = len(factors)
    if budget is None:
        budget = 2 * k + 1

    return _run_pyoptex(
        factors,
        criterion="a_optimal",
        budget=budget,
        model_type=model_type,
        hard_to_change=hard_to_change,
    )
