# (c) Kevin Dunn, 2010-2026. MIT License.
r"""Split-plot optimal designs, found by coordinate exchange on the GLS information matrix.

In a split-plot design the hard-to-change factors are set once for each *whole plot*, a group of
consecutive runs, and the easy-to-change factors are reset for every run. The runs of a whole plot
share its random effect, so their responses are correlated: :math:`V = I + \eta Z Z^T` in units of
the run-to-run variance, with :math:`\eta` the ratio of the whole-plot variance to it and :math:`Z`
the runs' whole-plot indicators. A design is judged on :math:`M = X^T V^{-1} X`, the information of
the generalized least squares estimates (Goos and Jones, *Optimal Design of Experiments: A Case
Study Approach*, chapter 10):

- D-optimal: the largest :math:`|M|`;
- A-optimal: the smallest :math:`\operatorname{tr}(M^{-1})`;
- I-optimal: the smallest :math:`\operatorname{tr}(M^{-1} W)`, the variance of a prediction averaged
  over the factor box, :math:`W` holding the averages of the products of the model terms there.

The search is a coordinate exchange. From a random start, each whole plot's setting of each
hard-to-change factor, then each run's setting of each easy-to-change factor, is replaced by the
level that improves the criterion most, until a pass changes nothing; the best of several starts
is kept. :math:`V^{-1}` is block diagonal, so :math:`M` is a sum over the whole plots,

.. math::

    M = \sum_w X_w^T X_w - c_w (X_w^T \mathbf{1})(\mathbf{1}^T X_w), \qquad
    c_w = \frac{\eta}{1 + m_w \eta},

for a whole plot of :math:`m_w` runs, and a change replaces one whole plot's term.

A two-way split-plot (strip-plot) design has two such groupings, crossed: rows, whose factors are
set once for the runs of a row, and columns, whose factors are set once for the runs of a column
(:func:`strip_plot_design`). Then :math:`V = I + \eta_r Z_r Z_r^T + \eta_c Z_c Z_c^T`, and the
search updates :math:`M` by the runs a change touches.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from process_improve._random import check_random_state
from process_improve.experiments.designs_utils import categorical_effect_columns, sorted_level_ranks
from process_improve.experiments.factor import FactorType

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

    from process_improve.experiments.factor import Factor
    from process_improve.experiments.region import DesignRegion

logger = logging.getLogger(__name__)

#: The criteria the search optimizes.
CRITERIA = ("d_optimal", "a_optimal", "i_optimal")

#: Levels a continuous factor may take: three for a model without squares, five with them, since
#: A- and I-optimal designs for a quadratic model use interior levels (as the pyoptex backend does).
_LINEAR_LEVELS = np.array([-1.0, 0.0, 1.0])
_QUADRATIC_LEVELS = np.linspace(-1.0, 1.0, 5)

#: Averages over [-1, 1] of x**0 to x**4, the products of a continuous factor's terms 1, x and x**2.
_UNIFORM_MOMENTS = np.array([1.0, 0.0, 1.0 / 3.0, 0.0, 1.0 / 5.0])

#: Most passes over the coordinates in one start.
_MAX_PASSES = 100

#: A change is kept when it lowers the criterion by more than this share of its size.
_IMPROVEMENT = 1e-10

#: Ridge, per run, added to M while searching, so a singular start has a finite criterion.
_RIDGE = 1e-9

#: Random starts drawn, at most, for one that estimates the model.
_START_DRAWS = 50


@dataclass(frozen=True)
class SplitPlotOptions:
    r"""How a split-plot design is searched for.

    Attributes
    ----------
    criterion : str
        One of :data:`CRITERIA`.
    model_type : str
        ``"main_effects"``, ``"interactions"`` or ``"quadratic"``. A categorical factor has no
        square, so ``"quadratic"`` squares the continuous factors only.
    n_whole_plots : int
        The number of whole plots. The runs are shared out in order, the last whole plot
        taking any that are left over, as the pyoptex backend does.
    variance_ratio : float
        :math:`\eta`, the whole-plot variance over the run-to-run variance.
    n_starts : int
        Random starts; the best design of them is kept.
    region : DesignRegion or None
        The limits on the settings every run keeps to, a limit on some levels of a categorical
        factor among them; None for the whole box. The search starts from runs inside them and
        takes no exchange that leaves them, so a design's runs lie on the grid of levels inside the
        limits, not on the limits' own corners.
    fixed_runs : pandas.DataFrame or None
        Runs kept as given, one column per factor (continuous factors coded, categorical factors
        as their labels), validated as ``generate_design`` validates them. They are the design's
        first runs, so they fill its first whole plots in order. A whole plot that holds one keeps
        its hard-to-change settings for all its runs, as the pyoptex backend does, and the search
        chooses the rest. Fixed runs that share a whole plot must share its hard-to-change
        settings. A fixed run may lie off the grid of levels, or outside the limits (with a
        warning).
    """

    criterion: str = "d_optimal"
    model_type: str = "interactions"
    n_whole_plots: int = 4
    variance_ratio: float = 0.5
    n_starts: int = 10
    region: DesignRegion | None = None
    fixed_runs: pd.DataFrame | None = None


@dataclass(frozen=True)
class _Model:
    """The model's columns as products of one part of each factor, with what the search needs."""

    tables: tuple[np.ndarray, ...]  # per factor: (n_levels, n_parts) the value of each part at each level
    parts: np.ndarray  # (p, k) which part of each factor a model column takes
    moments: np.ndarray  # (p, p) averages of the products of the columns over the factor box
    settings: tuple[list, ...]  # per factor: the coded value or the label of each level
    choices: tuple[int, ...]  # per factor: the levels the search may give a run, the first of its levels

    def rows(self, levels: np.ndarray) -> np.ndarray:
        """Return the model columns of runs given as level indices, one run per row (any leading shape)."""
        x = np.ones((*levels.shape[:-1], len(self.parts)))
        for j, table in enumerate(self.tables):
            x *= table[levels[..., j]][..., self.parts[:, j]]
        return x


@dataclass(frozen=True)
class _Layout:
    """Each run's whole plot (from 0), the hard-to-change factors' columns, and the fixed runs' levels."""

    whole_plot: np.ndarray
    hard: tuple[int, ...]
    fixed: np.ndarray  # (n_fixed, k) the levels of the runs kept as given, which are the design's first

    @property
    def pinned(self) -> np.ndarray:
        """Return the whole plots that hold a fixed run, whose hard-to-change settings are therefore fixed too."""
        return np.unique(self.whole_plot[: len(self.fixed)])

    def pin(self, levels: np.ndarray) -> np.ndarray:
        """Set the fixed runs, and the hard-to-change levels of their whole plots' other runs to theirs."""
        levels[: len(self.fixed)] = self.fixed
        for w in self.pinned:
            runs = self.whole_plot == w
            levels[np.ix_(runs, self.hard)] = levels[np.argmax(runs), list(self.hard)]  # its first run is fixed
        return levels


def split_plot_design(
    factors: list[Factor],
    hard_to_change: list[str],
    n_runs: int,
    options: SplitPlotOptions,
    random_state: int | np.random.Generator | None = None,
) -> tuple[np.ndarray, dict]:
    r"""Search for a split-plot design that is optimal on the GLS information :math:`X^T V^{-1} X`.

    Parameters
    ----------
    factors : list[Factor]
        Continuous factors (coded from -1 to +1) and categorical factors.
    hard_to_change : list[str]
        The factors set once for each whole plot.
    n_runs : int
        The number of runs.
    options : SplitPlotOptions
        The criterion, the model, the whole plots, the variance ratio and the number of starts.
    random_state : int, numpy.random.Generator or None
        Seeds the random starts.

    Returns
    -------
    tuple[numpy.ndarray, dict]
        The design, one run per row in whole-plot order (continuous factors coded, categorical
        factors as their labels), and metadata: ``optimality_criterion``, ``model_type``,
        ``backend`` (``"split_plot_exchange"``), ``hard_to_change``, ``whole_plot`` (each run's
        whole plot, from 0), ``n_whole_plots``, ``whole_plot_variance_ratio``, ``n_starts``, and
        the criterion of the design on :math:`M = X^T V^{-1} X`: ``log_det_information`` for D,
        ``trace_criterion`` (:math:`\operatorname{tr}(M^{-1})` or :math:`\operatorname{tr}(M^{-1} W)`)
        for A and I.

    Raises
    ------
    ValueError
        If the criterion is not one of :data:`CRITERIA`, the variance ratio is not a finite number
        of at least 0, there are fewer runs than whole plots or than model terms, or no design of
        this structure can estimate the model.
    """
    _check(factors, n_runs, options)
    model = _build_model(factors, options.model_type)
    if n_runs < len(model.parts):
        msg = f"{n_runs} runs cannot estimate the {len(model.parts)} terms of the {options.model_type} model."
        raise ValueError(msg)
    model, fixed = _with_fixed_runs(model, factors, options.fixed_runs, n_runs)
    hard_names = set(hard_to_change)
    hard = tuple(j for j, f in enumerate(factors) if f.name in hard_names)
    layout = _layout(_whole_plots(n_runs, options.n_whole_plots), hard, fixed, factors)
    rng = check_random_state(random_state)
    feasible = _feasibility(model, factors, options.region)
    best_levels, best_score = None, np.inf
    for _ in range(options.n_starts):
        start = _estimable_start(model, lambda: _feasible_start(model, layout, rng, feasible))
        levels, score = _climb(model, start, layout, options, rng, feasible)
        if score < best_score:
            best_levels, best_score = levels, score
    assert best_levels is not None  # n_starts >= 1, checked above
    information = _information(model.rows(best_levels), layout.whole_plot, options.variance_ratio)
    if np.linalg.matrix_rank(information) < len(model.parts):
        msg = (
            f"No split-plot design of {n_runs} runs in {options.n_whole_plots} whole plots estimates the "
            f"{options.model_type} model; give more runs or more whole plots."
        )
        raise ValueError(msg)
    meta = {
        "optimality_criterion": options.criterion,
        "model_type": options.model_type,
        "backend": "split_plot_exchange",
        "hard_to_change": list(hard_to_change),
        "whole_plot": layout.whole_plot.tolist(),
        "n_whole_plots": options.n_whole_plots,
        "whole_plot_variance_ratio": options.variance_ratio,
        "n_starts": options.n_starts,
        **({"constraints_enforced": True} if feasible is not None else {}),
        **_fixed_run_metadata(layout.fixed, feasible),
        **_criterion_value(information, model.moments, options.criterion),
    }
    return _settings(model, best_levels, factors), meta


def _check(factors: list[Factor], n_runs: int, options: SplitPlotOptions) -> None:
    """Refuse a criterion, a variance ratio or a run count the search cannot work with."""
    if options.criterion not in CRITERIA:
        msg = f"A split-plot design is searched for on {list(CRITERIA)}; got criterion={options.criterion!r}."
        raise ValueError(msg)
    ratio = options.variance_ratio
    if not isinstance(ratio, (int, float)) or not np.isfinite(ratio) or ratio < 0:
        msg = f"The whole-plot variance ratio must be a finite number of at least 0; got {ratio!r}."
        raise ValueError(msg)
    if not 1 <= options.n_whole_plots <= n_runs or options.n_starts < 1 or not factors:
        msg = (
            f"A split-plot design needs factors, at least one start, and from 1 to {n_runs} whole plots for "
            f"{n_runs} runs; got {options.n_whole_plots} whole plots and {options.n_starts} starts."
        )
        raise ValueError(msg)


def _build_model(factors: list[Factor], model_type: str) -> _Model:
    """Write the model's columns as products of one part of each factor: 1, x or x**2, or a level contrast.

    A continuous factor's parts are 1, x and x**2; a categorical factor's are 1 and its
    effect-coded (sum-to-zero) contrasts, against the last level in sorted order, as the
    candidate exchange and ``evaluate_design`` code it.
    """
    tables, moments, settings, mains = [], [], [], []
    for f in factors:
        if f.type == FactorType.categorical:
            labels = list(f.levels or [])
            contrasts = categorical_effect_columns(sorted_level_ranks(labels), len(labels))
            table = np.column_stack([np.ones(len(labels)), contrasts])
            tables.append(table)
            moments.append(table.T @ table / len(labels))  # the levels equally often
            settings.append(labels)
            mains.append(list(range(1, len(labels))))
        else:
            values = _QUADRATIC_LEVELS if model_type == "quadratic" else _LINEAR_LEVELS
            tables.append(np.column_stack([np.ones_like(values), values, values**2]))
            moments.append(_UNIFORM_MOMENTS[np.add.outer(np.arange(3), np.arange(3))])
            settings.append(values.tolist())
            mains.append([1])
    k = len(factors)
    columns = [np.zeros(k, dtype=int)]
    for j, own in enumerate(mains):
        columns.extend(_one_part(k, {j: part}) for part in own)
    if model_type in ("interactions", "quadratic"):
        for i, j in itertools.combinations(range(k), 2):
            columns.extend(_one_part(k, {i: a, j: b}) for a in mains[i] for b in mains[j])
    if model_type == "quadratic":
        columns.extend(_one_part(k, {j: 2}) for j, f in enumerate(factors) if f.type != FactorType.categorical)
    parts = np.array(columns)
    joint = np.ones((len(parts), len(parts)))
    for j, moment in enumerate(moments):
        joint *= moment[np.ix_(parts[:, j], parts[:, j])]  # the factors vary independently over the box
    return _Model(tuple(tables), parts, joint, tuple(settings), tuple(len(table) for table in tables))


def _with_fixed_runs(
    model: _Model, factors: list[Factor], fixed_runs: pd.DataFrame | None, n_runs: int
) -> tuple[_Model, np.ndarray]:
    """Return the model with a level for each fixed setting off the grid, and the fixed runs as levels.

    A fixed run is kept as given, so a continuous setting between the grid's levels becomes a level of
    its own, after the grid's: ``model.choices`` keeps the search to the grid for every other run.

    Raises
    ------
    ValueError
        If the fixed runs are not as ``generate_design`` takes them, or span less of the model than the
        other runs can make up.
    """
    if fixed_runs is None:
        return model, np.zeros((0, len(factors)), dtype=int)
    from process_improve.experiments.designs_constrained import refuse_budget_short_of_fixed_runs  # noqa: PLC0415
    from process_improve.experiments.designs_optimal import _prepare_prior_runs  # noqa: PLC0415

    prior = _prepare_prior_runs(fixed_runs, factors, n_runs)
    tables, settings = list(model.tables), list(model.settings)
    levels = np.zeros((len(prior), len(factors)), dtype=int)
    for j, f in enumerate(factors):
        for r, value in enumerate(prior[f.name]):
            if f.type == FactorType.categorical:
                levels[r, j] = [str(label) for label in settings[j]].index(str(value))
                continue
            known = np.flatnonzero(np.isclose(settings[j], value))
            if len(known):
                levels[r, j] = known[0]
                continue
            settings[j] = [*settings[j], float(value)]  # off the grid: a level of its own
            tables[j] = np.vstack([tables[j], [1.0, value, value**2]])
            levels[r, j] = len(settings[j]) - 1
    model = replace(model, tables=tuple(tables), settings=tuple(settings))
    refuse_budget_short_of_fixed_runs(model.rows(levels), len(model.parts), n_runs)
    return model, levels


def _layout(whole_plot: np.ndarray, hard: tuple[int, ...], fixed: np.ndarray, factors: list[Factor]) -> _Layout:
    """Return the layout, the fixed runs first; refuse fixed runs that share a whole plot but not its settings."""
    for w in np.unique(whole_plot[: len(fixed)]):
        sharing = fixed[whole_plot[: len(fixed)] == w]
        for j in hard:
            if len(set(sharing[:, j])) > 1:
                msg = (
                    f"The fixed runs that fall in whole plot {w + 1} set the hard-to-change factor "
                    f"{factors[j].name!r} to different levels. The fixed runs are the design's first runs, "
                    f"{np.sum(whole_plot == 0)} to a whole plot, and the runs of a whole plot share its "
                    "hard-to-change settings: order the fixed runs so that those sharing a whole plot share them."
                )
                raise ValueError(msg)
    return _Layout(whole_plot, hard, fixed)


def _fixed_run_metadata(fixed: np.ndarray, feasible: Callable[[np.ndarray], np.ndarray] | None) -> dict:
    """Return how many runs were kept as given and, with limits, how many of them lie outside, as a warning says."""
    if not len(fixed):
        return {}
    meta = {"n_fixed_runs": len(fixed)}
    if feasible is not None and (n_outside := int((~feasible(fixed)).sum())):
        logger.warning("%d fixed run(s) lie outside the constrained region; they are kept as given.", n_outside)
        meta["n_fixed_runs_outside_region"] = n_outside
    return meta


def _one_part(k: int, chosen: dict[int, int]) -> np.ndarray:
    """Return a model column's parts: the chosen part of the factors named, the constant 1 of the rest."""
    parts = np.zeros(k, dtype=int)
    for j, part in chosen.items():
        parts[j] = part
    return parts


def _whole_plots(n_runs: int, n_whole_plots: int) -> np.ndarray:
    """Share the runs out among the whole plots in order, the last whole plot taking any left over."""
    per_plot = n_runs // n_whole_plots
    return np.minimum(np.arange(n_runs) // per_plot, n_whole_plots - 1)


def _random_start(model: _Model, layout: _Layout, rng: np.random.Generator) -> np.ndarray:
    """Return random levels: one for each whole plot of a hard-to-change factor, one for each run of the rest.

    The fixed runs, and the hard-to-change levels of their whole plots, are then set to theirs.
    """
    levels = np.column_stack([rng.integers(n, size=len(layout.whole_plot)) for n in model.choices])
    for j in layout.hard:
        levels[:, j] = rng.integers(model.choices[j], size=layout.whole_plot.max() + 1)[layout.whole_plot]
    return layout.pin(levels)


def _feasibility(
    model: _Model, factors: list[Factor], region: DesignRegion | None
) -> Callable[[np.ndarray], np.ndarray] | None:
    """Return the test of runs given as level indices (any leading shape) against the limits; None without any."""
    if region is None or not region.constraints:
        return None
    numeric = [j for j, f in enumerate(factors) if f.type != FactorType.categorical]
    labelled = [j for j, f in enumerate(factors) if f.type == FactorType.categorical]
    values = [np.asarray(column, dtype=object if j in labelled else float) for j, column in enumerate(model.settings)]

    def feasible(levels: np.ndarray) -> np.ndarray:
        flat = levels.reshape(-1, levels.shape[-1])
        coded = np.column_stack([values[j][flat[:, j]] for j in numeric]) if numeric else np.zeros((len(flat), 0))
        labels = {factors[j].name: values[j][flat[:, j]] for j in labelled} or None
        return region.feasible(coded, labels=labels).reshape(levels.shape[:-1])

    return feasible


def _feasible_start(
    model: _Model, layout: _Layout, rng: np.random.Generator, feasible: Callable[[np.ndarray], np.ndarray] | None
) -> np.ndarray:
    """Return random levels, as :func:`_random_start` draws them, with every run inside the limits.

    Each whole plot's hard-to-change levels are drawn, then each of its runs' other levels until the run is
    inside; a whole plot whose runs cannot all be is drawn again. A fixed run is kept as given, inside or
    not, and so are the hard-to-change levels of its whole plot.
    """
    levels = _random_start(model, layout, rng)
    if feasible is None:
        return levels
    easy = [j for j in range(len(model.choices)) if j not in layout.hard]
    for w in range(layout.whole_plot.max() + 1):
        rows = np.flatnonzero(layout.whole_plot == w)
        free = rows[rows >= len(layout.fixed)]
        drawn = [] if len(free) < len(rows) else layout.hard  # a fixed run's whole plot keeps its settings
        for _ in range(_START_DRAWS):
            for j in drawn:
                levels[rows, j] = rng.integers(model.choices[j])
            for r in free:
                for _ in range(_START_DRAWS):
                    if feasible(levels[r])[()]:
                        break
                    levels[r, easy] = [rng.integers(model.choices[j]) for j in easy]
            if feasible(levels[free]).all():
                break
        else:
            msg = "No whole plot of runs inside the limits on the settings was found; check that the limits leave room."
            raise ValueError(msg)
    return levels


def _estimable_start(model: _Model, draw: Callable[[], np.ndarray]) -> np.ndarray:
    """Return a start from ``draw``, drawn again up to ``_START_DRAWS`` times until it estimates the model.

    From a singular start the search works on M plus the small ridge, whose condition number is
    about 1e9, so the criteria of its first exchanges differ in the eighth digit from one BLAS
    kernel to another. The search then follows a different path, and ends at a different design,
    on a different processor. From a start that estimates the model the criteria agree far closer
    than the improvement a change must make.
    """
    levels = draw()
    for _ in range(_START_DRAWS - 1):
        if np.linalg.matrix_rank(model.rows(levels)) == len(model.parts):
            break
        levels = draw()
    return levels


def _plot_terms(rows: np.ndarray, shrink: float) -> np.ndarray:
    """Return a whole plot's share of M: X'X less ``shrink`` times the outer product of its column sums."""
    totals = rows.sum(axis=-2)
    return np.einsum("...ri,...rj->...ij", rows, rows) - shrink * totals[..., :, None] * totals[..., None, :]


def _information(rows: np.ndarray, whole_plot: np.ndarray, ratio: float) -> np.ndarray:
    """Return :math:`M = X^T V^{-1} X` of runs whose model rows are given, summed over the whole plots."""
    information = np.zeros((rows.shape[1], rows.shape[1]))
    for w in range(whole_plot.max() + 1):
        mine = rows[whole_plot == w]
        information += _plot_terms(mine, ratio / (1.0 + len(mine) * ratio))
    return information


def _scores(information: np.ndarray, criterion: str, moments: np.ndarray, ridge: float) -> np.ndarray:
    """Return each information matrix's criterion, lower being better: -log|M|, tr(M^-1) or tr(M^-1 W)."""
    regular = information + ridge * np.eye(information.shape[-1])
    if criterion == "d_optimal":
        sign, log_det = np.linalg.slogdet(regular)
        return np.where(sign > 0, -log_det, np.inf)
    inverse = np.linalg.inv(regular)
    if criterion == "a_optimal":
        return np.trace(inverse, axis1=-2, axis2=-1)
    return np.einsum("...ij,ji->...", inverse, moments)


def _climb(
    model: _Model,
    levels: np.ndarray,
    layout: _Layout,
    options: SplitPlotOptions,
    rng: np.random.Generator,
    feasible: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[np.ndarray, float]:
    """Exchange coordinates from a start until a pass improves nothing; return the levels and their criterion.

    With limits (``feasible``), an exchange that puts a run of the whole plot outside them is not taken. A
    fixed run is never changed, nor are the hard-to-change levels of a whole plot that holds one.
    """
    plots = [np.flatnonzero(layout.whole_plot == w) for w in range(layout.whole_plot.max() + 1)]
    free = [np.flatnonzero(rows >= len(layout.fixed)) for rows in plots]  # each whole plot's runs to search
    pinned = set(layout.pinned.tolist())
    shrinks = [options.variance_ratio / (1.0 + len(rows) * options.variance_ratio) for rows in plots]
    terms = [_plot_terms(model.rows(levels[rows]), s) for rows, s in zip(plots, shrinks, strict=True)]
    information = np.sum(terms, axis=0)
    ridge = _RIDGE * len(layout.whole_plot)
    score = float(_scores(information, options.criterion, model.moments, ridge))
    easy = [j for j in range(levels.shape[1]) if j not in layout.hard]
    # Each coordinate is (whole plot, factor, the positions in the whole plot of the runs it sets).
    for _ in range(_MAX_PASSES):
        moves = [
            (w, j, np.arange(len(plots[w])))
            for w in rng.permutation(len(plots))
            if w not in pinned
            for j in layout.hard
        ]
        moves += [(w, j, np.array([r])) for w in rng.permutation(len(plots)) for r in free[w] for j in easy]
        improved = False
        for w, j, runs in moves:
            options_here = np.repeat(levels[plots[w]][None], model.choices[j], axis=0)
            options_here[:, runs, j] = np.arange(model.choices[j])[:, None]
            candidates = _plot_terms(model.rows(options_here), shrinks[w])
            scores = _scores(information - terms[w] + candidates, options.criterion, model.moments, ridge)
            if feasible is not None:
                scores = np.where(feasible(options_here[:, free[w]]).all(axis=1), scores, np.inf)
            best = int(np.argmin(scores))
            if scores[best] < score - _IMPROVEMENT * max(1.0, abs(score)):
                levels[plots[w]] = options_here[best]
                information += candidates[best] - terms[w]
                terms[w], score, improved = candidates[best], float(scores[best]), True
        if not improved:
            break
    return levels, score


def _criterion_value(information: np.ndarray, moments: np.ndarray, criterion: str) -> dict[str, float]:
    """Return the design's criterion on M under the candidate exchange's key."""
    if criterion == "d_optimal":
        return {"log_det_information": float(np.linalg.slogdet(information)[1])}
    return {"trace_criterion": float(_scores(information, criterion, moments, 0.0))}


def _settings(model: _Model, levels: np.ndarray, factors: list[Factor]) -> np.ndarray:
    """Return the runs as settings: coded values of the continuous factors, labels of the categorical ones."""
    columns = [np.asarray(model.settings[j], dtype=object)[levels[:, j]] for j in range(len(factors))]
    design = np.column_stack(columns)
    if all(f.type != FactorType.categorical for f in factors):
        return design.astype(float)
    return design


# --------------------------------------------------------------------------- two-way split-plot designs
@dataclass(frozen=True)
class StripPlotLayout:
    """The structure of a two-way split-plot (strip-plot) design.

    Attributes
    ----------
    row_factors, column_factors : tuple of str
        The factors set once for each row and once for each column of the grid, as the two
        stages of a process set them (lots, then curing cycles, say). Any other factor is set
        for each run.
    cells : tuple of (int, int)
        Each run's row and column, from 0.
    """

    row_factors: tuple[str, ...]
    column_factors: tuple[str, ...]
    cells: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class StripPlotOptions:
    r"""How a two-way split-plot design is searched for.

    Attributes
    ----------
    criterion : str
        One of :data:`CRITERIA`.
    model_type : str
        ``"main_effects"``, ``"interactions"`` or ``"quadratic"``.
    row_variance_ratio, column_variance_ratio : float
        :math:`\eta_r` and :math:`\eta_c`: the row and the column variance over the run-to-run
        variance.
    n_starts : int
        Random starts; the best design of them is kept.
    """

    criterion: str = "d_optimal"
    model_type: str = "main_effects"
    row_variance_ratio: float = 1.0
    column_variance_ratio: float = 1.0
    n_starts: int = 10


@dataclass(frozen=True)
class _Grid:
    """For each factor, the groups of runs that share its setting; and the inverse of the runs' covariance."""

    groups: tuple[tuple[np.ndarray, ...], ...]
    weight: np.ndarray


def strip_plot_design(
    factors: list[Factor],
    layout: StripPlotLayout,
    options: StripPlotOptions,
    random_state: int | np.random.Generator | None = None,
) -> tuple[np.ndarray, dict]:
    r"""Search for a two-way split-plot (strip-plot) design that is optimal on :math:`X^T V^{-1} X`.

    The runs sit in the cells of a grid. A row's factors are set once for all its runs and a
    column's once for all of its runs; any other factor is set for each run. The rows and the
    columns are crossed random effects, :math:`V = I + \eta_r Z_r Z_r^T + \eta_c Z_c Z_c^T` in
    units of the run-to-run variance (Goos and Jones, chapter 11). The search is the coordinate
    exchange of :func:`split_plot_design`: each row's setting of each row factor, each column's
    setting of each column factor and each run's setting of the other factors is replaced by the
    level that improves the criterion most, until a pass changes nothing.

    Parameters
    ----------
    factors : list[Factor]
        Continuous factors (coded from -1 to +1) and categorical factors.
    layout : StripPlotLayout
        The row factors, the column factors and the cell of each run.
    options : StripPlotOptions
        The criterion, the model, the two variance ratios and the number of starts.
    random_state : int, numpy.random.Generator or None
        Seeds the random starts.

    Returns
    -------
    tuple[numpy.ndarray, dict]
        The design, one run per row in the order of ``layout.cells`` (continuous factors coded,
        categorical factors as their labels), and metadata: ``optimality_criterion``,
        ``model_type``, ``backend`` (``"strip_plot_exchange"``), ``row_factors``,
        ``column_factors``, ``row`` and ``column`` (each run's), ``row_variance_ratio``,
        ``column_variance_ratio``, ``n_starts``, and ``log_det_information`` for D or
        ``trace_criterion`` for A and I, on :math:`M = X^T V^{-1} X`.

    Raises
    ------
    ValueError
        If the criterion is not one of :data:`CRITERIA`, a variance ratio is not a finite number of
        at least 0, a row or column factor is not a factor or is both, a cell is not a pair of
        whole numbers from 0, there are fewer runs than model terms, or no design on this grid can
        estimate the model.

    Examples
    --------
    Goos and Jones's twelve runs in four rows and four columns (Table 11.9), each row in three
    columns: x1 and x2 set by row, x3 by column.

    >>> from process_improve.experiments import Factor
    >>> cells = tuple((r, (r + c) % 4) for r in range(4) for c in range(3))
    >>> layout = StripPlotLayout(("x1", "x2"), ("x3",), cells)
    >>> design, meta = strip_plot_design(
    ...     [Factor(name=n, low=-1, high=1) for n in ("x1", "x2", "x3")], layout, StripPlotOptions(), random_state=1
    ... )
    >>> design.shape, round(meta["log_det_information"], 3)
    ((12, 3), 3.767)
    """
    cells = _check_strip(factors, layout, options)
    model = _build_model(factors, options.model_type)
    if len(cells) < len(model.parts):
        msg = f"{len(cells)} runs cannot estimate the {len(model.parts)} terms of the {options.model_type} model."
        raise ValueError(msg)
    grid = _grid(factors, layout, cells, options)
    rng = check_random_state(random_state)
    best_levels, best_score = None, np.inf
    for _ in range(options.n_starts):
        start = _estimable_start(model, lambda: _grid_start(model, grid, len(cells), rng))
        levels, score = _climb_grid(model, start, grid, options.criterion, rng)
        if score < best_score:
            best_levels, best_score = levels, score
    assert best_levels is not None  # n_starts >= 1, checked above
    rows = model.rows(best_levels)
    information = rows.T @ grid.weight @ rows
    if np.linalg.matrix_rank(information) < len(model.parts):
        msg = f"No design on this grid of {len(cells)} runs estimates the {options.model_type} model."
        raise ValueError(msg)
    meta = {
        "optimality_criterion": options.criterion,
        "model_type": options.model_type,
        "backend": "strip_plot_exchange",
        "row_factors": list(layout.row_factors),
        "column_factors": list(layout.column_factors),
        "row": cells[:, 0].tolist(),
        "column": cells[:, 1].tolist(),
        "row_variance_ratio": options.row_variance_ratio,
        "column_variance_ratio": options.column_variance_ratio,
        "n_starts": options.n_starts,
        **_criterion_value(information, model.moments, options.criterion),
    }
    return _settings(model, best_levels, factors), meta


def _check_strip(factors: list[Factor], layout: StripPlotLayout, options: StripPlotOptions) -> np.ndarray:
    """Refuse options, factors or cells the search cannot work with; return the cells as an array."""
    if options.criterion not in CRITERIA:
        msg = f"A two-way split-plot design is searched for on {list(CRITERIA)}; got criterion={options.criterion!r}."
        raise ValueError(msg)
    for ratio in (options.row_variance_ratio, options.column_variance_ratio):
        if not isinstance(ratio, (int, float)) or not np.isfinite(ratio) or ratio < 0:
            msg = f"A variance ratio must be a finite number of at least 0; got {ratio!r}."
            raise ValueError(msg)
    names = {f.name for f in factors}
    staged = [*layout.row_factors, *layout.column_factors]
    if not factors or options.n_starts < 1 or not set(staged) <= names or len(set(staged)) < len(staged):
        msg = (
            f"The row factors {list(layout.row_factors)} and the column factors {list(layout.column_factors)} must "
            f"be different factors of {sorted(names)}, and the search needs at least one start."
        )
        raise ValueError(msg)
    cells = np.asarray(layout.cells)
    if cells.ndim != 2 or cells.shape[1:] != (2,) or len(cells) == 0 or cells.dtype.kind not in "iu" or cells.min() < 0:
        msg = "The cells must be (row, column) pairs of whole numbers from 0, one for each run."
        raise ValueError(msg)
    return cells


def _grid(factors: list[Factor], layout: StripPlotLayout, cells: np.ndarray, options: StripPlotOptions) -> _Grid:
    """Return the runs that share each factor's setting, and V^-1 for the rows and columns as random effects."""

    def by(column: int) -> tuple[np.ndarray, ...]:
        return tuple(np.flatnonzero(cells[:, column] == value) for value in np.unique(cells[:, column]))

    each_run = tuple(np.array([i]) for i in range(len(cells)))
    groups = tuple(
        by(0) if f.name in layout.row_factors else by(1) if f.name in layout.column_factors else each_run
        for f in factors
    )
    same_row = (cells[:, 0][:, None] == cells[:, 0][None, :]).astype(float)
    same_column = (cells[:, 1][:, None] == cells[:, 1][None, :]).astype(float)
    covariance = (
        np.eye(len(cells)) + options.row_variance_ratio * same_row + options.column_variance_ratio * same_column
    )
    return _Grid(groups, np.linalg.inv(covariance))


def _grid_start(model: _Model, grid: _Grid, n_runs: int, rng: np.random.Generator) -> np.ndarray:
    """Return random levels, one for each group of runs that shares a factor's setting."""
    levels = np.zeros((n_runs, len(grid.groups)), dtype=int)
    for j, groups in enumerate(grid.groups):
        for runs in groups:
            levels[runs, j] = rng.integers(len(model.tables[j]))
    return levels


def _climb_grid(
    model: _Model, levels: np.ndarray, grid: _Grid, criterion: str, rng: np.random.Generator
) -> tuple[np.ndarray, float]:
    """Exchange coordinates on the grid until a pass improves nothing; return the levels and their criterion.

    A change to the runs S changes the model rows by D there, and :math:`M = X^T W X` by
    :math:`D^T (WX)_S + (WX)_S^T D + D^T W_{SS} D`, so a trial costs little more than the runs it
    changes. M and WX are worked afresh at the start of each pass.
    """
    ridge = _RIDGE * len(levels)
    coordinates = [(j, runs) for j, groups in enumerate(grid.groups) for runs in groups]
    rows = model.rows(levels)
    score = float(_scores(rows.T @ grid.weight @ rows, criterion, model.moments, ridge))
    for _ in range(_MAX_PASSES):
        weighted = grid.weight @ rows
        information = rows.T @ weighted
        improved = False
        for i in rng.permutation(len(coordinates)):
            j, runs = coordinates[i]
            trial = np.repeat(levels[runs][None], len(model.tables[j]), axis=0)
            trial[:, :, j] = np.arange(len(model.tables[j]))[:, None]
            change = model.rows(trial) - rows[runs]
            cross = np.einsum("lri,rj->lij", change, weighted[runs])
            own = np.einsum("lri,rs,lsj->lij", change, grid.weight[np.ix_(runs, runs)], change)
            candidates = information + cross + cross.transpose(0, 2, 1) + own
            found = _scores(candidates, criterion, model.moments, ridge)
            best = int(np.argmin(found))
            if found[best] < score - _IMPROVEMENT * max(1.0, abs(score)):
                levels[runs] = trial[best]
                rows[runs] += change[best]
                weighted += grid.weight[:, runs] @ change[best]
                information, score, improved = candidates[best], float(found[best]), True
        if not improved:
            break
    return levels, score
