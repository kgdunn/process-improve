# (c) Kevin Dunn, 2010-2026. MIT License.
r"""Split-plot optimal designs, found by coordinate exchange on the GLS information matrix.

In a split-plot design the hard-to-change factors are set once for each *whole plot*, a group of
consecutive runs, and the easy-to-change factors are reset for every run. The runs of a whole plot
share its random effect, so their responses are correlated: :math:`V = I + \eta Z Z^T` in units of
the run-to-run variance, with :math:`\eta` the ratio of the whole-plot variance to it and :math:`Z`
the runs' whole-plot indicators. A design is judged on :math:`M = X^T V^{-1} X`, the information of
the generalised least squares estimates (Goos and Jones, *Optimal Design of Experiments: A Case
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
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from process_improve._random import check_random_state
from process_improve.experiments.designs_utils import categorical_effect_columns, sorted_level_ranks
from process_improve.experiments.factor import FactorType

if TYPE_CHECKING:
    from process_improve.experiments.factor import Factor

#: The criteria the search optimises.
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
    """

    criterion: str = "d_optimal"
    model_type: str = "interactions"
    n_whole_plots: int = 4
    variance_ratio: float = 0.5
    n_starts: int = 10


@dataclass(frozen=True)
class _Model:
    """The model's columns as products of one part of each factor, with what the search needs."""

    tables: tuple[np.ndarray, ...]  # per factor: (n_levels, n_parts) the value of each part at each level
    parts: np.ndarray  # (p, k) which part of each factor a model column takes
    moments: np.ndarray  # (p, p) averages of the products of the columns over the factor box
    settings: tuple[list, ...]  # per factor: the coded value or the label of each level

    def rows(self, levels: np.ndarray) -> np.ndarray:
        """Return the model columns of runs given as level indices, one run per row (any leading shape)."""
        x = np.ones((*levels.shape[:-1], len(self.parts)))
        for j, table in enumerate(self.tables):
            x *= table[levels[..., j]][..., self.parts[:, j]]
        return x


@dataclass(frozen=True)
class _Layout:
    """Each run's whole plot (from 0), and the hard-to-change factors' columns."""

    whole_plot: np.ndarray
    hard: tuple[int, ...]


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
    hard_names = set(hard_to_change)
    layout = _Layout(
        _whole_plots(n_runs, options.n_whole_plots), tuple(j for j, f in enumerate(factors) if f.name in hard_names)
    )
    rng = check_random_state(random_state)
    best_levels, best_score = None, np.inf
    for _ in range(options.n_starts):
        levels, score = _climb(model, _random_start(model, layout, rng), layout, options, rng)
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
    return _Model(tuple(tables), parts, joint, tuple(settings))


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
    """Return random levels: one for each whole plot of a hard-to-change factor, one for each run of the rest."""
    n_levels = [len(table) for table in model.tables]
    levels = np.column_stack([rng.integers(n, size=len(layout.whole_plot)) for n in n_levels])
    for j in layout.hard:
        levels[:, j] = rng.integers(n_levels[j], size=layout.whole_plot.max() + 1)[layout.whole_plot]
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
    model: _Model, levels: np.ndarray, layout: _Layout, options: SplitPlotOptions, rng: np.random.Generator
) -> tuple[np.ndarray, float]:
    """Exchange coordinates from a start until a pass improves nothing; return the levels and their criterion."""
    plots = [np.flatnonzero(layout.whole_plot == w) for w in range(layout.whole_plot.max() + 1)]
    shrinks = [options.variance_ratio / (1.0 + len(rows) * options.variance_ratio) for rows in plots]
    terms = [_plot_terms(model.rows(levels[rows]), s) for rows, s in zip(plots, shrinks, strict=True)]
    information = np.sum(terms, axis=0)
    ridge = _RIDGE * len(layout.whole_plot)
    score = float(_scores(information, options.criterion, model.moments, ridge))
    easy = [j for j in range(levels.shape[1]) if j not in layout.hard]
    # Each coordinate is (whole plot, factor, the positions in the whole plot of the runs it sets).
    for _ in range(_MAX_PASSES):
        moves = [(w, j, np.arange(len(plots[w]))) for w in rng.permutation(len(plots)) for j in layout.hard]
        moves += [
            (w, j, np.array([r])) for w in rng.permutation(len(plots)) for r in range(len(plots[w])) for j in easy
        ]
        improved = False
        for w, j, runs in moves:
            options_here = np.repeat(levels[plots[w]][None], len(model.tables[j]), axis=0)
            options_here[:, runs, j] = np.arange(len(model.tables[j]))[:, None]
            candidates = _plot_terms(model.rows(options_here), shrinks[w])
            scores = _scores(information - terms[w] + candidates, options.criterion, model.moments, ridge)
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
