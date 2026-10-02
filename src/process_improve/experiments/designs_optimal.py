# (c) Kevin Dunn, 2010-2026. MIT License.

"""Optimal designs: D-optimal, I-optimal, A-optimal, E-optimal.

Two engines, chosen per request by :func:`_dispatch_optimal`:

- ``pyoptex`` (coordinate exchange), when installed and no constraints are given.
  It also builds split-plot designs for ``hard_to_change`` factors. ``pyoptex`` is
  not a process-improve extra because it pins ``plotly~=5.24`` (< 6), which
  conflicts with this project's ``plotly>=6.5.2``; install it separately
  (``pip install pyoptex``) in its own environment.
- The built-in candidate exchange in ``designs_constrained.py`` otherwise: a
  Fedorov exchange over a grid of candidate points for D, I or A-optimality, with
  constraints, categorical factors and fixed runs, but no split-plot structure.

Mixture factors always use ``designs_mixture_constrained.py``.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from process_improve._random import check_random_state

if TYPE_CHECKING:
    from collections.abc import Iterator

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


def _n_model_parameters(factors: list[Factor], model_type: str) -> int:
    """Return the number of coefficients in the model an optimal design is asked for.

    A design with fewer runs than this cannot estimate the model at all: the
    model matrix is rank deficient no matter which points are chosen. The count
    is the estimability floor every backend is held to.

    Each factor contributes one encoded column, except a categorical factor with
    ``L`` levels, which contributes ``L - 1``. The model then has an intercept,
    the main effects, the two-factor interactions (for ``"interactions"`` and
    ``"quadratic"``), and one pure-quadratic term per non-categorical factor
    (for ``"quadratic"`` only; a categorical factor has no square, which is why
    ``_run_pyoptex`` drops it to a ``"tfi"`` model per factor).

    Parameters
    ----------
    factors : list[Factor]
        Factor specifications.
    model_type : str
        ``"main_effects"``, ``"interactions"`` or ``"quadratic"``, as validated by
        ``_dispatch_optimal``.

    Returns
    -------
    int
        The number of model coefficients.
    """
    from process_improve.experiments.factor import FactorType  # noqa: PLC0415

    widths = [len(f.levels) - 1 if f.type == FactorType.categorical and f.levels else 1 for f in factors]
    n_parameters = 1 + sum(widths)  # intercept + main effects
    rsm_key = _PYOPTEX_MODEL_MAP.get(model_type, "tfi")
    if rsm_key in {"tfi", "quad"}:
        n_parameters += sum(widths[i] * widths[j] for i in range(len(widths)) for j in range(i + 1, len(widths)))
    if rsm_key == "quad":
        n_parameters += sum(1 for f in factors if f.type != FactorType.categorical)
    return n_parameters


def _floor_budget_at_model_size(factors: list[Factor], budget: int, model_type: str) -> int:
    """Raise ``budget`` to the model's parameter count, warning when it does.

    Both backends need this floor and neither imposed it consistently. pyoptex
    failed with an upstream message about "rank collinearity" between model
    components, which names neither the budget nor the model; the point-exchange
    fallback clamped to one run per factor, one short of the intercept, and even
    the default ``budget = 2 * n_factors + 1`` falls below the parameter count
    of an interactions model from five factors up.
    """
    n_parameters = _n_model_parameters(factors, model_type)
    if budget < n_parameters:
        logger.warning(
            "A budget of %d run(s) cannot estimate a '%s' model over %d factor(s), which has %d "
            "coefficients; raising the budget to %d. Reduce the model or the number of factors to "
            "run fewer experiments.",
            budget,
            model_type,
            len(factors),
            n_parameters,
            n_parameters,
        )
        return n_parameters
    return budget


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
    random_state: int | np.random.Generator | None = None


@contextlib.contextmanager
def _global_numpy_seed(rng: np.random.Generator) -> Iterator[None]:
    """Seed numpy's global RNG from ``rng`` for the duration of the block, then restore the caller's state.

    pyoptex draws its coordinate-exchange restarts from the global RNG, which a
    caller's ``random_state`` cannot otherwise reach. Restoring the state afterwards
    means the design neither depends on nor disturbs any global seeding the caller
    has done.
    """
    saved = np.random.get_state()  # noqa: NPY002
    np.random.seed(int(rng.integers(2**32)))  # noqa: NPY002 - pyoptex reads the legacy global RNG
    try:
        yield
    finally:
        np.random.set_state(saved)  # noqa: NPY002


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
    with _global_numpy_seed(check_random_state(opts.random_state)):
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
# Routing: pyoptex, or the built-in candidate exchange
# ---------------------------------------------------------------------------


@dataclass
class _OptimalRequest:
    """Everything an optimal-design request carries, whichever criterion it is for."""

    factors: list[Factor]
    budget: int | None
    hard_to_change: list[str] | None
    constraints: list[Constraint] | None
    model_type: str
    fixed_runs: pd.DataFrame | None
    random_state: int | np.random.Generator | None
    candidates: pd.DataFrame | None = None
    backend: str = "auto"


def _dispatch_mixture_optimal(criterion: str, req: _OptimalRequest) -> tuple[np.ndarray, dict]:
    """Optimal mixture design over the (possibly constrained) simplex, for a Scheffé model."""
    from process_improve.experiments.designs_constrained import ConstrainedOptions  # noqa: PLC0415
    from process_improve.experiments.designs_mixture_constrained import (  # noqa: PLC0415
        constrained_mixture_design,
        scheffe_matrix,
    )

    if req.fixed_runs is not None or req.hard_to_change:
        raise ValueError("fixed_runs and hard_to_change are not supported for mixture designs.")
    n_terms = scheffe_matrix(np.ones((1, len(req.factors))), req.model_type).shape[1]
    options = ConstrainedOptions(model_type=req.model_type, criterion=criterion, candidates=req.candidates)
    return constrained_mixture_design(
        req.factors, req.budget or n_terms + 3, req.constraints, options, req.random_state
    )


def _pyoptex_blocker(criterion: str, req: _OptimalRequest) -> str | None:
    """Why pyoptex cannot build this request, or ``None`` when it can."""
    if req.constraints:
        return "constraints are given (pyoptex does not enforce them)"
    if req.candidates is not None:
        return "a candidate set is given"
    if criterion in ("e_optimal", "g_optimal", "k_optimal"):
        return f"pyoptex has no {criterion[0].upper()}-optimality criterion"
    if not _PYOPTEX_AVAILABLE:
        return f"pyoptex is not installed. {_PYOPTEX_INSTALL_HINT}"
    return None


def _dispatch_optimal(criterion: str, req: _OptimalRequest) -> tuple[np.ndarray, dict]:
    """Route a D-, I- or A-optimal request to the backend that can honour it.

    - Mixture factors go to the constrained-simplex engine with a Scheffé model.
    - By default the built-in candidate exchange runs
      (:func:`~process_improve.experiments.designs_constrained.constrained_optimal_design`),
      which handles every criterion, constraints, candidate sets, categorical factors and
      fixed runs, so the design does not depend on whether pyoptex is installed.
    - pyoptex's coordinate exchange runs for split-plot designs (``hard_to_change``), when
      it is installed and nothing it cannot honour is asked for, or for
      ``backend="pyoptex"``. Otherwise ``hard_to_change`` is ignored and recorded.
    """
    from process_improve.experiments.factor import FactorType  # noqa: PLC0415

    if req.factors and all(f.type == FactorType.mixture for f in req.factors):
        return _dispatch_mixture_optimal(criterion, req)
    if req.model_type not in _PYOPTEX_MODEL_MAP:
        raise ValueError(
            f"model_type={req.model_type!r} is not supported for {criterion}; choose from "
            f"{', '.join(_PYOPTEX_MODEL_MAP)} (Scheffé models apply to mixture components only)."
        )

    budget = req.budget if req.budget is not None else 2 * len(req.factors) + 1
    budget = _floor_budget_at_model_size(req.factors, budget, req.model_type)

    if req.backend not in ("auto", "exchange", "pyoptex"):
        raise ValueError(f"backend must be 'auto', 'exchange' or 'pyoptex'; got {req.backend!r}.")
    blocker = _pyoptex_blocker(criterion, req)
    if req.backend == "pyoptex" and blocker is not None:
        raise ValueError(f"backend='pyoptex' cannot be used here: {blocker}.")
    # The built-in exchange is the default, so a design does not depend on which optional
    # packages are installed; pyoptex runs for split-plot structure or when asked for.
    use_pyoptex = req.backend == "pyoptex" or (req.backend == "auto" and bool(req.hard_to_change) and blocker is None)
    if not use_pyoptex:
        from process_improve.experiments.designs_constrained import (  # noqa: PLC0415
            ConstrainedOptions,
            constrained_optimal_design,
        )

        prior = _prepare_prior_runs(req.fixed_runs, req.factors, budget) if req.fixed_runs is not None else None
        options = ConstrainedOptions(
            model_type=req.model_type, criterion=criterion, fixed_runs=prior, candidates=req.candidates
        )
        matrix, meta = constrained_optimal_design(req.factors, budget, req.constraints or [], options, req.random_state)
        if req.hard_to_change:
            reason = blocker or "backend='exchange' was requested"
            logger.warning("hard_to_change factors (split-plot) are ignored because %s", reason)
            meta["hard_to_change_ignored"] = list(req.hard_to_change)
        return matrix, meta

    return _run_pyoptex(
        req.factors,
        criterion=criterion,
        budget=budget,
        options=_PyoptexOptions(
            model_type=req.model_type,
            hard_to_change=req.hard_to_change,
            fixed_runs=req.fixed_runs,
            random_state=req.random_state,
        ),
    )


# ---------------------------------------------------------------------------
# Public dispatch functions
# ---------------------------------------------------------------------------

_DISPATCH_PARAMETERS = """
    Parameters
    ----------
    factors : list[Factor]
        Factor specifications. All-mixture factors get a Scheffé-model design over
        the (possibly constrained) simplex.
    budget : int or None
        Number of runs.  Defaults to ``2 * n_factors + 1``.
    hard_to_change : list[str] or None
        Names of hard-to-change factors: a split-plot design via pyoptex. Ignored,
        and recorded as ``hard_to_change_ignored``, when constraints are given or
        pyoptex is not installed.
    constraints : list[Constraint] or None
        Inequalities in actual units over the continuous factors, e.g.
        ``"3*T + 5*D <= 600"``. Enforced: every returned run satisfies them.
    model_type : str
        Model assumption: ``"main_effects"``, ``"interactions"``, or ``"quadratic"``.
    fixed_runs : pd.DataFrame or None
        Runs to hold fixed while the optimizer fills the rest (design
        augmentation). Occupies the first rows of the returned design and counts
        towards ``budget``.
    random_state : int, numpy.random.Generator or None
        Seed for the candidate exchange's random starts.
    candidates : pd.DataFrame or None
        Settings to choose the runs from, in actual units (proportions for a
        mixture), one column per factor. Replaces the generated grid; rows breaking
        a constraint are dropped. ``metadata["selected_candidates"]`` counts how often
        each row (by index label) was picked.
    backend : {"auto", "exchange", "pyoptex"}
        ``"auto"`` uses the built-in exchange, and pyoptex only for ``hard_to_change``.

    Returns
    -------
    tuple[np.ndarray, dict]
        The coded design and its metadata; ``metadata["backend"]`` names the engine
        (``"pyoptex"`` or ``"candidate_exchange"``).
"""


def dispatch_d_optimal(  # noqa: PLR0913
    factors: list[Factor],
    budget: int | None = None,
    hard_to_change: list[str] | None = None,
    constraints: list[Constraint] | None = None,
    model_type: str = "interactions",
    fixed_runs: pd.DataFrame | None = None,
    random_state: int | np.random.Generator | None = None,
    candidates: pd.DataFrame | None = None,
    backend: str = "auto",
) -> tuple[np.ndarray, dict]:
    """Generate a D-optimal design (maximises ``det(X'X)``, the precision of the coefficients jointly)."""
    req = _OptimalRequest(
        factors, budget, hard_to_change, constraints, model_type, fixed_runs, random_state, candidates, backend
    )
    return _dispatch_optimal("d_optimal", req)


def dispatch_i_optimal(  # noqa: PLR0913
    factors: list[Factor],
    budget: int | None = None,
    hard_to_change: list[str] | None = None,
    constraints: list[Constraint] | None = None,
    model_type: str = "interactions",
    fixed_runs: pd.DataFrame | None = None,
    random_state: int | np.random.Generator | None = None,
    candidates: pd.DataFrame | None = None,
    backend: str = "auto",
) -> tuple[np.ndarray, dict]:
    """Generate an I-optimal design (minimises the average prediction variance over the region)."""
    req = _OptimalRequest(
        factors, budget, hard_to_change, constraints, model_type, fixed_runs, random_state, candidates, backend
    )
    return _dispatch_optimal("i_optimal", req)


def dispatch_a_optimal(  # noqa: PLR0913
    factors: list[Factor],
    budget: int | None = None,
    hard_to_change: list[str] | None = None,
    constraints: list[Constraint] | None = None,
    model_type: str = "interactions",
    fixed_runs: pd.DataFrame | None = None,
    random_state: int | np.random.Generator | None = None,
    candidates: pd.DataFrame | None = None,
    backend: str = "auto",
) -> tuple[np.ndarray, dict]:
    """Generate an A-optimal design (minimises the summed variance of the coefficients)."""
    req = _OptimalRequest(
        factors, budget, hard_to_change, constraints, model_type, fixed_runs, random_state, candidates, backend
    )
    return _dispatch_optimal("a_optimal", req)


for _fn in (dispatch_d_optimal, dispatch_i_optimal, dispatch_a_optimal):
    _fn.__doc__ = (_fn.__doc__ or "") + "\n" + _DISPATCH_PARAMETERS
