# (c) Kevin Dunn, 2010-2026. MIT License.

"""Optimal designs: D-optimal, I-optimal, A-optimal.

Uses ``pyoptex`` (coordinate exchange) when available for high-quality
optimal designs with support for split-plot structures.  Falls back to the
built-in ``point_exchange()`` in ``optimal.py`` for D-optimal when
``pyoptex`` is not installed.  ``pyoptex`` is not a process-improve extra
because it pins ``plotly~=5.24`` (< 6), which conflicts with this project's
``plotly>=6.5.2``; install it separately (``pip install pyoptex``) in its own
environment. Without it, I-/A-optimal and ``hard_to_change`` split-plot
requests are unavailable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

try:
    from pyDOE3 import fullfact
except ImportError:  # pragma: no cover - exercised via env-without-pyDOE3
    from process_improve._extras import _MissingExtra
    fullfact = _MissingExtra("pyDOE3", "expt")  # type: ignore[assignment]

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
except ImportError:  # pragma: no cover - exercised via env-without-pyoptex
    pass

#: Remediation hint shown when pyoptex is required but not installed. pyoptex is
#: deliberately not a process-improve extra: its latest release pins
#: ``plotly~=5.24`` (< 6), which conflicts with this project's ``plotly>=6.5.2``
#: floor, so the two cannot share an environment. Install it separately.
_PYOPTEX_INSTALL_HINT = (
    "Install it separately with `pip install pyoptex` (note: pyoptex pins "
    "plotly<6, which conflicts with this project's plotly>=6.5.2, so it cannot "
    "be co-installed with the 'plotting'/'all' extras; use a separate "
    "environment)."
)

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
if _PYOPTEX_AVAILABLE:  # pragma: no branch - false only in env-without-pyoptex
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


@dataclass
class _PyoptexOptions:
    """Optional knobs for :func:`_run_pyoptex`.

    Pulled into a dataclass so the function signature stays at four
    parameters (ENG-25 / #307: removes one ``noqa: PLR0913``).
    """

    model_type: str = "interactions"
    hard_to_change: list[str] | None = None
    n_tries: int = 10
    fixed_runs: pd.DataFrame | None = None


def _prepare_prior_runs(fixed_runs: pd.DataFrame, factors: list[Factor], budget: int) -> pd.DataFrame:
    """Validate and format fixed runs for pyoptex's ``prior=`` augmentation.

    ``fixed_runs`` holds runs to keep fixed while the coordinate exchange fills the remaining
    ``budget - len(fixed_runs)`` runs. It must be in the same coding as the returned design:
    continuous factors in coded ``[-1, 1]`` units, categorical factors as level labels. Columns
    other than the factor names are ignored, and the input is not mutated.

    Parameters
    ----------
    fixed_runs : pandas.DataFrame
        One row per fixed run, one column per factor.
    factors : list[Factor]
        The design factors.
    budget : int
        Total number of runs (fixed plus optimized).

    Returns
    -------
    pandas.DataFrame
        A copy holding only the factor columns, in factor order, ready to pass to pyoptex.
    """
    from process_improve.experiments.factor import FactorType  # noqa: PLC0415

    if not isinstance(fixed_runs, pd.DataFrame):
        raise TypeError("fixed_runs must be a pandas DataFrame with one column per factor.")
    names = [f.name for f in factors]
    missing = [n for n in names if n not in fixed_runs.columns]
    if missing:
        raise ValueError(f"fixed_runs is missing columns for factors: {missing}.")
    n_fixed = len(fixed_runs)
    if n_fixed == 0:
        raise ValueError("fixed_runs is empty; omit it instead of passing an empty frame.")
    if n_fixed >= budget:
        raise ValueError(
            f"fixed_runs has {n_fixed} runs but budget is {budget}; budget must exceed the number "
            f"of fixed runs so at least one run is optimized."
        )
    prior = fixed_runs.loc[:, names].reset_index(drop=True).copy()
    for f in factors:
        col = prior[f.name]
        if f.type == FactorType.categorical:
            levels = list(f.levels or [])
            unknown = sorted(set(col.astype(str)) - {str(lv) for lv in levels})
            if unknown:
                raise ValueError(
                    f"fixed_runs has unknown levels for categorical factor {f.name!r}: {unknown}. "
                    f"Valid levels: {levels}."
                )
        else:
            vals = pd.to_numeric(col, errors="coerce")
            if vals.isna().any():
                raise ValueError(f"fixed_runs has non-numeric values for continuous factor {f.name!r}.")
            if (vals.abs() > 1.0 + 1e-9).any():
                raise ValueError(
                    f"fixed_runs values for continuous factor {f.name!r} must be in coded [-1, 1], "
                    f"the same coding as the returned design."
                )
            prior[f.name] = vals.astype(float)
    return prior


def _run_pyoptex(
    factors: list[Factor],
    criterion: str,
    budget: int,
    options: _PyoptexOptions | None = None,
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
    options : _PyoptexOptions or None
        Optional knobs (model_type, hard_to_change, n_tries). Defaults
        are interactions / no hard-to-change / 10 restarts.

    Returns
    -------
    tuple[np.ndarray, dict]
        Coded design matrix (-1 / +1 for continuous, labels for categorical)
        and metadata dict.
    """
    opts = options if options is not None else _PyoptexOptions()
    model_type = opts.model_type
    hard_to_change = opts.hard_to_change
    n_tries = opts.n_tries

    pyoptex_factors = _convert_factors_to_pyoptex(
        factors,
        hard_to_change=hard_to_change,
        n_runs=budget,
    )

    # Build the model per factor. A categorical factor has no pure-quadratic
    # term (its square is undefined and, once indicator-coded, idempotent), so a
    # "quadratic" request becomes a partial response-surface model: quadratics on
    # the continuous factors, main-effect-plus-interactions on the categorical
    # ones. This is the standard second-order-with-categorical model and avoids
    # the rank collinearity a uniform x**2 would create.
    from process_improve.experiments.factor import FactorType  # noqa: PLC0415

    rsm_key = _PYOPTEX_MODEL_MAP.get(model_type, "tfi")

    def _factor_rsm_key(factor: Factor) -> str:
        if rsm_key == "quad" and factor.type == FactorType.categorical:
            return "tfi"
        return rsm_key

    model_spec = partial_rsm_names({f.name: _factor_rsm_key(f) for f in factors})
    y2x = model2Y2X(model_spec, pyoptex_factors)

    # Select metric
    metric_cls = _PYOPTEX_METRIC_MAP[criterion]
    metric = metric_cls()

    prior = None
    if opts.fixed_runs is not None:
        prior = _prepare_prior_runs(opts.fixed_runs, factors, budget)

    fn = default_fn(pyoptex_factors, metric, y2x)
    params = create_parameters(pyoptex_factors, fn, nruns=budget, prior=prior)
    design_df, state = create_fixed_structure_design(params, n_tries=n_tries)

    meta = {
        "optimality_criterion": criterion,
        "metric_value": float(state.metric),
        "model_type": model_type,
        "backend": "pyoptex",
    }
    if hard_to_change:
        meta["hard_to_change"] = hard_to_change
    if prior is not None:
        meta["n_fixed_runs"] = len(prior)

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
    # SEC-19 (#268): a 3-level full factorial allocates 3**k candidate
    # rows. Cap k against the central setting so a request for
    # ``len(factors) >= 20`` (3.5B rows) is rejected before allocation.
    from process_improve.config import settings  # noqa: PLC0415

    if k > settings.max_factors_combinatorial:
        raise ValueError(
            f"d-optimal fallback would build a 3**{k} candidate set; "
            f"that exceeds the SEC-19 cap of "
            f"{settings.max_factors_combinatorial} factors. "
            "Increase settings.max_factors_combinatorial if intentional."
        )
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


def dispatch_d_optimal(  # noqa: PLR0913
    factors: list[Factor],
    budget: int | None = None,
    hard_to_change: list[str] | None = None,
    constraints: list[Constraint] | None = None,
    model_type: str = "interactions",
    fixed_runs: pd.DataFrame | None = None,
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

    if fixed_runs is not None and not _PYOPTEX_AVAILABLE:
        raise ImportError(f"fixed_runs (design augmentation) requires pyoptex. {_PYOPTEX_INSTALL_HINT}")

    if constraints:
        logger.warning(
            "Constraint enforcement in optimal designs is experimental. "
            "Constraints are noted but may not be fully enforced. "
            "Consider filtering infeasible runs manually."
        )

    if _PYOPTEX_AVAILABLE:
        matrix, meta = _run_pyoptex(
            factors,
            criterion="d_optimal",
            budget=budget,
            options=_PyoptexOptions(model_type=model_type, hard_to_change=hard_to_change, fixed_runs=fixed_runs),
        )
        # Record that constraints were not enforced so the DesignResult carries
        # the fact programmatically, not only as an easy-to-miss log line.
        if constraints:
            meta["constraints_enforced"] = False
        return matrix, meta

    if hard_to_change:
        logger.warning(
            "pyoptex is not installed - hard_to_change factors will be ignored. %s",
            _PYOPTEX_INSTALL_HINT,
        )
    matrix, meta = _run_point_exchange_fallback(factors, budget)
    if constraints:
        meta["constraints_enforced"] = False
    if hard_to_change:
        # The randomized fallback cannot honour the split-plot request; surface
        # it on the result rather than only in the log.
        meta["hard_to_change_ignored"] = list(hard_to_change)
    return matrix, meta


def dispatch_i_optimal(  # noqa: PLR0913
    factors: list[Factor],
    budget: int | None = None,
    hard_to_change: list[str] | None = None,
    constraints: list[Constraint] | None = None,
    model_type: str = "interactions",
    fixed_runs: pd.DataFrame | None = None,
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
        raise ImportError(f"I-optimal design generation requires pyoptex. {_PYOPTEX_INSTALL_HINT}")

    k = len(factors)
    if budget is None:
        budget = 2 * k + 1

    matrix, meta = _run_pyoptex(
        factors,
        criterion="i_optimal",
        budget=budget,
        options=_PyoptexOptions(model_type=model_type, hard_to_change=hard_to_change, fixed_runs=fixed_runs),
    )
    if constraints:
        meta["constraints_enforced"] = False
    return matrix, meta


def dispatch_a_optimal(  # noqa: PLR0913
    factors: list[Factor],
    budget: int | None = None,
    hard_to_change: list[str] | None = None,
    constraints: list[Constraint] | None = None,
    model_type: str = "interactions",
    fixed_runs: pd.DataFrame | None = None,
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
        raise ImportError(f"A-optimal design generation requires pyoptex. {_PYOPTEX_INSTALL_HINT}")

    k = len(factors)
    if budget is None:
        budget = 2 * k + 1

    matrix, meta = _run_pyoptex(
        factors,
        criterion="a_optimal",
        budget=budget,
        options=_PyoptexOptions(model_type=model_type, hard_to_change=hard_to_change, fixed_runs=fixed_runs),
    )
    if constraints:
        meta["constraints_enforced"] = False
    return matrix, meta
