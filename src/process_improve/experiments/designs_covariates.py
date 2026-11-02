# (c) Kevin Dunn, 2010-2026. MIT License.
r"""Optimal designs on experimental units with known covariates, each unit used once.

Some experiments run on units that carry a measured property the experimenter cannot set: a
sample's composition, a field's slope, or the time at which a run is made. The units come from a
list, each can be used once, and the factors are set for each run on top. The model has the
covariates' main effects beside the factors' terms, and a design is judged on :math:`M = X^T X` of
the whole model (Goos and Jones, *Optimal Design of Experiments: A Case Study Approach*, chapter 9).

The units used and the factors' settings are found together by coordinate exchange: each run's
unit is swapped for the unused unit, and each run's setting of each factor for the level, that
improves the criterion most, until a pass changes nothing; the best of several random starts is
kept. A change replaces one row of the model matrix, so it changes :math:`M` by two outer products.

A time trend is the case where every unit is used: the units are the run positions, with the time
of each as its covariate, and the search places the factors' settings in time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from process_improve._random import check_random_state
from process_improve.experiments.designs_split_plot import (
    _IMPROVEMENT,
    _MAX_PASSES,
    _RIDGE,
    CRITERIA,
    _build_model,
    _criterion_value,
    _Model,
    _scores,
    _settings,
)
from process_improve.experiments.designs_utils import categorical_effect_columns, sorted_level_ranks

if TYPE_CHECKING:
    from process_improve.experiments.factor import Factor


@dataclass(frozen=True)
class CovariateOptions:
    """How a design on units with covariates is searched for.

    Attributes
    ----------
    criterion : str
        One of ``"d_optimal"``, ``"a_optimal"`` and ``"i_optimal"``.
    model_type : str
        The factors' terms: ``"main_effects"``, ``"interactions"`` or ``"quadratic"``. The
        covariates enter by their main effects.
    n_starts : int
        Random starts; the best design of them is kept.
    """

    criterion: str = "d_optimal"
    model_type: str = "main_effects"
    n_starts: int = 10


@dataclass(frozen=True)
class _Problem:
    """The factors' model, each unit's covariate columns, and the criterion with its moments."""

    model: _Model
    covariates: np.ndarray  # (n_units, c)
    moments: np.ndarray  # (p, p), the factor box and the units' covariates together
    criterion: str


@dataclass
class _State:
    """A design during the search: each run's unit and levels, the model rows, M and its criterion."""

    units: np.ndarray
    levels: np.ndarray
    rows: np.ndarray
    information: np.ndarray
    score: float


def covariate_design(
    units: pd.DataFrame,
    factors: list[Factor],
    n_runs: int,
    options: CovariateOptions,
    random_state: int | np.random.Generator | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Choose units, each at most once, and set the factors for each, optimally for a model with the covariates.

    Parameters
    ----------
    units : pandas.DataFrame
        The candidate units, one per row, with a column per covariate: numeric covariates enter
        the model as they are given (code them to -1 and +1 for a criterion on the coded scale),
        and the others as categorical, effect-coded. The index names the units.
    factors : list[Factor]
        The factors set for each run: continuous (coded from -1 to +1) or categorical.
    n_runs : int
        The number of runs, at most the number of units. With every unit used, the search only
        sets the factors: a time trend, the units being the run positions.
    options : CovariateOptions
        The criterion, the factors' model and the number of starts.
    random_state : int, numpy.random.Generator or None
        Seeds the random starts.

    Returns
    -------
    tuple[pandas.DataFrame, dict]
        The design, one run per row: the unit's covariates (indexed by the unit, in the order
        of ``units``) and the factors' settings (continuous coded, categorical as labels); and
        metadata: ``optimality_criterion``, ``model_type``, ``backend`` (``"covariate_exchange"``),
        ``units`` (the units used), ``n_starts``, and ``log_det_information`` for D or
        ``trace_criterion`` for A and I. The I criterion averages over the factor box and over
        the candidate units.

    Raises
    ------
    ValueError
        If the criterion is not one of :data:`CRITERIA`, there are no units, ``n_runs`` exceeds
        their number or is below the model's number of terms, a factor shares a covariate's name,
        or no choice of units estimates the model.

    Examples
    --------
    A linear trend in time over eight runs: the 2^3 placed so that every main effect is
    orthogonal to the trend.

    >>> import numpy as np
    >>> import pandas as pd
    >>> from process_improve.experiments import Factor
    >>> times = pd.DataFrame({"time": np.linspace(-1, 1, 8)})
    >>> factors = [Factor(name=n, low=-1, high=1) for n in ("x1", "x2", "x3")]
    >>> design, meta = covariate_design(times, factors, 8, CovariateOptions(), random_state=1)
    >>> bool(np.allclose(design[["x1", "x2", "x3"]].T @ design["time"], 0.0))
    True
    """
    _check(units, factors, n_runs, options)
    model = _build_model(factors, options.model_type)
    covariates = _covariate_columns(units)
    problem = _Problem(model, covariates, _moments(model, covariates), options.criterion)
    if n_runs < len(model.parts) + covariates.shape[1]:
        msg = f"{n_runs} runs cannot estimate the {len(model.parts) + covariates.shape[1]} terms of the model."
        raise ValueError(msg)
    rng = check_random_state(random_state)
    best: _State | None = None
    for _ in range(options.n_starts):
        state = _climb(problem, _start(problem, len(units), n_runs, rng), rng)
        if best is None or state.score < best.score:
            best = state
    assert best is not None  # n_starts >= 1, checked above
    information = best.rows.T @ best.rows
    if np.linalg.matrix_rank(information) < information.shape[0]:
        msg = f"No choice of {n_runs} of the {len(units)} units estimates the model."
        raise ValueError(msg)
    settings = pd.DataFrame(_settings(model, best.levels, factors), columns=[f.name for f in factors])
    order = np.argsort(best.units, kind="stable")
    design = units.iloc[best.units[order]].copy()
    for name in settings.columns:
        design[name] = settings[name].to_numpy()[order]
    meta = {
        "optimality_criterion": options.criterion,
        "model_type": options.model_type,
        "backend": "covariate_exchange",
        "units": design.index.tolist(),
        "n_starts": options.n_starts,
        **_criterion_value(information, problem.moments, options.criterion),
    }
    return design, meta


def _check(units: pd.DataFrame, factors: list[Factor], n_runs: int, options: CovariateOptions) -> None:
    """Refuse a criterion, a unit list or a run count the search cannot work with."""
    if options.criterion not in CRITERIA:
        msg = f"A design on units with covariates is searched for on {list(CRITERIA)}; got {options.criterion!r}."
        raise ValueError(msg)
    if units.empty or not 1 <= n_runs <= len(units) or options.n_starts < 1:
        msg = f"Choose from 1 to {len(units)} of the units, with at least one start; got n_runs={n_runs}."
        raise ValueError(msg)
    shared = sorted({f.name for f in factors} & set(units.columns))
    if shared:
        msg = f"The factors {shared} are also covariates of the units; a column is one or the other."
        raise ValueError(msg)


def _covariate_columns(units: pd.DataFrame) -> np.ndarray:
    """Return the covariates' model columns: numeric ones as given, the others effect-coded."""
    columns = []
    for name in units.columns:
        values = units[name]
        if pd.api.types.is_numeric_dtype(values):
            columns.append(values.to_numpy(dtype=float)[:, None])
        else:
            levels = sorted(pd.unique(values).tolist(), key=str)
            ranks = sorted_level_ranks(levels)
            codes = ranks[[levels.index(v) for v in values]]
            columns.append(categorical_effect_columns(codes, len(levels)))
    return np.hstack(columns) if columns else np.empty((len(units), 0))


def _moments(model: _Model, covariates: np.ndarray) -> np.ndarray:
    """Return the averages of the products of the model columns: over the factor box, and over the units."""
    first = model.moments[:, 0]  # the averages of the factors' columns, the first column being 1
    mean = covariates.mean(axis=0)
    second = covariates.T @ covariates / len(covariates)
    return np.block([[model.moments, np.outer(first, mean)], [np.outer(mean, first), second]])


def _start(problem: _Problem, n_units: int, n_runs: int, rng: np.random.Generator) -> _State:
    """Return a random design: distinct units and random levels."""
    units = rng.choice(n_units, size=n_runs, replace=False)
    levels = np.column_stack([rng.integers(len(t), size=n_runs) for t in problem.model.tables])
    rows = np.hstack([problem.model.rows(levels), problem.covariates[units]])
    information = rows.T @ rows
    return _State(units, levels, rows, information, float(_score(problem, information, n_runs)))


def _score(problem: _Problem, information: np.ndarray, n_runs: int) -> np.ndarray:
    return _scores(information, problem.criterion, problem.moments, _RIDGE * n_runs)


def _try(state: _State, problem: _Problem, run: int, rows: np.ndarray) -> int | None:
    """Return which of the candidate rows for ``run`` improves the criterion most, if one does."""
    old = state.rows[run]
    candidates = state.information - np.outer(old, old) + np.einsum("li,lj->lij", rows, rows)
    found = _score(problem, candidates, len(state.units))
    best = int(np.argmin(found))
    if found[best] < state.score - _IMPROVEMENT * max(1.0, abs(state.score)):
        state.rows[run], state.information, state.score = rows[best], candidates[best], float(found[best])
        return best
    return None


def _swap(state: _State, problem: _Problem, run: int) -> bool:
    """Exchange ``run``'s unit with another run's, if one exchange improves the criterion; return whether it did.

    For a time trend this moves a run to another time. It changes two rows of the model matrix,
    which a change of one coordinate cannot do without losing the balance of a factor first.
    """
    others = np.delete(np.arange(len(state.units)), run)
    width = len(problem.model.parts)
    mine = np.hstack(
        [np.repeat(state.rows[run, :width][None], len(others), axis=0), problem.covariates[state.units[others]]]
    )
    theirs = np.hstack(
        [state.rows[others, :width], np.repeat(problem.covariates[state.units[run]][None], len(others), 0)]
    )
    old_pairs = np.einsum("li,lj->lij", state.rows[others], state.rows[others]) + np.outer(
        state.rows[run], state.rows[run]
    )
    new_pairs = np.einsum("li,lj->lij", mine, mine) + np.einsum("li,lj->lij", theirs, theirs)
    candidates = state.information - old_pairs + new_pairs
    found = _score(problem, candidates, len(state.units))
    best = int(np.argmin(found))
    if found[best] >= state.score - _IMPROVEMENT * max(1.0, abs(state.score)):
        return False
    other = others[best]
    state.units[run], state.units[other] = state.units[other], state.units[run]
    state.rows[run], state.rows[other] = mine[best], theirs[best]
    state.information, state.score = candidates[best], float(found[best])
    return True


def _improve_run(state: _State, problem: _Problem, run: int) -> bool:
    """Try every change to one run: an unused unit, another run's unit, each factor's levels."""
    improved = False
    unused = np.setdiff1d(np.arange(len(problem.covariates)), state.units)
    if len(unused):
        factor_part = np.repeat(state.rows[run, : len(problem.model.parts)][None], len(unused), axis=0)
        choice = _try(state, problem, run, np.hstack([factor_part, problem.covariates[unused]]))
        if choice is not None:
            state.units[run], improved = unused[choice], True
    improved = _swap(state, problem, run) or improved
    for j, table in enumerate(problem.model.tables):
        trial = np.repeat(state.levels[run][None], len(table), axis=0)
        trial[:, j] = np.arange(len(table))
        own = np.repeat(problem.covariates[state.units[run]][None], len(table), axis=0)
        choice = _try(state, problem, run, np.hstack([problem.model.rows(trial), own]))
        if choice is not None:
            state.levels[run, j], improved = choice, True
    return improved


def _climb(problem: _Problem, state: _State, rng: np.random.Generator) -> _State:
    """Improve the runs one at a time, in a random order, until a pass improves nothing."""
    for _ in range(_MAX_PASSES):
        improved = False
        for run in rng.permutation(len(state.units)):
            improved = _improve_run(state, problem, run) or improved
        if not improved:
            break
    return state
