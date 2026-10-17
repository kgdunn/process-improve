# (c) Kevin Dunn, 2010-2026. MIT License.
r"""Run orders that are robust to a time trend: how far each effect is confounded with drift, and how to avoid it.

When the runs of an experiment are made one after another, a slow drift in the process (a
catalyst ageing, a fouling heat exchanger, an instrument warming up) adds a smooth function of
time to every response. An effect whose column rises and falls with that function is biased by
it. A run order is *trend robust* when the model's columns are orthogonal to the time trends:
then the drift adds nothing to the estimated effects.

The trends are the orthogonal polynomials of the run index :math:`t = 1, \dots, N`: the linear
trend (the centred time count :math:`t - \bar t`), the quadratic trend, the cubic trend, each
orthogonal to the lower ones and to the intercept. A term's exposure to a trend :math:`z` is the
cosine of the angle between its model column :math:`x` and :math:`z`,

.. math::

    \cos\theta = \frac{x^\top z}{\lVert x \rVert \, \lVert z \rVert},

0 when the column is orthogonal to the trend (90 degrees) and 1 when it follows it exactly. This is
the "absolute correlation" Núñez Ares and Goos (2019) tabulate; for a column that does not average
zero, such as a quadratic term's, it is smaller than Pearson's correlation, which centres the column
first.

:func:`trend_robust_run_order` searches for an order of the runs that makes these cosines as
small as possible, the main effects against the linear trend first.

Examples
--------
>>> import numpy as np
>>> from process_improve.experiments.run_order import run_order_trends, trend_robust_run_order
>>> design = np.array([[a, b] for a in (-1, 1) for b in (-1, 1)] * 2)  # a 2^2 twice, in standard order
>>> run_order_trends(design, model="main_effects", degree=1)["summary"]["linear"]["main_effects"]["max_cosine"]
0.43643578047198456
>>> found = trend_robust_run_order(design, model="main_effects", degree=1, random_state=1)
>>> found["trends"]["summary"]["linear"]["main_effects"]["max_cosine"] < 1e-12
True
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd

from process_improve._random import check_random_state

#: The model shorthands, and the classes of term each holds.
RUN_ORDER_MODELS = ("main_effects", "interactions", "quadratic")
TREND_NAMES = ("linear", "quadratic", "cubic", "quartic")
_CLASSES = {"main_effect": "main_effects", "interaction": "interactions", "quadratic": "quadratics"}
_ZERO = 1e-12


def time_trends(n_runs: int, degree: int = 3) -> np.ndarray:
    r"""Return the orthogonal polynomial time trends of the run index, one column per degree, each of unit length.

    Parameters
    ----------
    n_runs : int
        The number of runs, :math:`N`.
    degree : int
        The highest degree, 1 (linear) to 4 (quartic), and at most :math:`N - 1`.

    Returns
    -------
    numpy.ndarray
        An :math:`N \times` ``degree`` array whose column k is the trend of degree k + 1: orthogonal
        to the intercept and to the trends of lower degree. Its sign makes it rise at the last run.
    """
    if not 1 <= degree <= min(len(TREND_NAMES), n_runs - 1):
        raise ValueError(f"degree must be 1 to {min(len(TREND_NAMES), n_runs - 1)} for {n_runs} runs; got {degree}.")
    t = np.arange(n_runs, dtype=float) - (n_runs - 1) / 2
    basis, _ = np.linalg.qr(np.vander(t, degree + 1, increasing=True))
    trends = basis[:, 1:]
    return trends * np.sign(trends[-1])


def _frame(design: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    """Return the factor columns as a frame of floats, named ``x1``, ``x2``, ... for an array."""
    frame = design if isinstance(design, pd.DataFrame) else pd.DataFrame(np.asarray(design, dtype=float))
    if not isinstance(design, pd.DataFrame):
        frame.columns = [f"x{j + 1}" for j in range(frame.shape[1])]
    return frame.astype(float)


def model_columns(frame: pd.DataFrame, model: str = "quadratic") -> tuple[list[str], list[str], np.ndarray]:
    """Return the model's terms, each one's class and their columns, without the intercept.

    ``"main_effects"`` holds the factors; ``"interactions"`` adds the products of two;
    ``"quadratic"`` adds the square of each factor with three or more levels.
    """
    if model not in RUN_ORDER_MODELS:
        raise ValueError(f"model must be one of {RUN_ORDER_MODELS}; got {model!r}.")
    names = list(frame.columns)
    values = frame.to_numpy(dtype=float)
    terms, kinds, columns = list(names), ["main_effect"] * len(names), [values[:, j] for j in range(len(names))]
    if model != "main_effects":
        for a, b in itertools.combinations(range(len(names)), 2):
            terms.append(f"{names[a]}:{names[b]}")
            kinds.append("interaction")
            columns.append(values[:, a] * values[:, b])
    if model == "quadratic":
        for j, name in enumerate(names):
            if frame[name].round(9).nunique() >= 3:  # a two-level factor has no curvature to estimate
                terms.append(f"I({name}**2)")
                kinds.append("quadratic")
                columns.append(values[:, j] ** 2)
    return terms, kinds, np.column_stack(columns)


def _cosines(columns: np.ndarray, trends: np.ndarray) -> np.ndarray:
    """Return the cosine between each model column (row) and each trend (column); 0 for a column of zeros."""
    lengths = np.linalg.norm(columns, axis=0)
    dots = columns.T @ trends
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(lengths[:, None] > _ZERO, dots / lengths[:, None], 0.0)


def run_order_trends(
    design: pd.DataFrame | np.ndarray,
    model: str = "quadratic",
    degree: int = 3,
    order: list[int] | None = None,
) -> dict[str, Any]:
    r"""Measure how far each term of a model is confounded with the time trends of a run order.

    Parameters
    ----------
    design : DataFrame or numpy.ndarray
        The runs in coded units, one row per run, in the order they are made: one column per
        factor (pass ``runs[["A", "B"]]`` for some columns of a larger frame).
    model : {"main_effects", "interactions", "quadratic"}
        The model whose terms are measured. Default ``"quadratic"``, the full second-order model.
    degree : int
        The highest degree of time trend: 1 (linear) to 4. Default 3, as Núñez Ares and Goos (2019).
    order : list[int] or None
        The order to make the runs in, as 0-based row positions of *design*; the rows' own order
        by default.

    Returns
    -------
    dict[str, Any]
        - ``trends``: the names of the trends measured, ``"linear"`` first;
        - ``terms``: each term with its ``kind`` (``"main_effect"``, ``"interaction"`` or
          ``"quadratic"``) and, by trend, its ``cosine`` with it (signed) and the ``angle``
          between them in degrees;
        - ``summary``: by trend and by class of term (``"main_effects"``, ``"interactions"``,
          ``"quadratics"``), the mean and largest absolute cosine, ``mean_cosine`` and
          ``max_cosine``, and the mean and largest deviation from orthogonality in degrees,
          ``mean_deviation`` and ``max_deviation``;
        - ``time_count_dot``: each term's dot product with the time count :math:`t - \bar t`,
          the linear trend in its natural units;
        - ``level_changes``: how often each factor changes level from one run to the next, and
          ``total_level_changes``, their sum: what the order costs to run.
    """
    frame = _frame(design)
    if order is not None:
        if sorted(order) != list(range(len(frame))):
            raise ValueError(f"order must hold each of the {len(frame)} row positions 0 to {len(frame) - 1} once.")
        frame = frame.iloc[list(order)].reset_index(drop=True)
    terms, kinds, columns = model_columns(frame, model)
    trends = time_trends(len(frame), degree)
    cosines = _cosines(columns, trends)
    angles = np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))
    names = list(TREND_NAMES[:degree])
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for k, trend in enumerate(names):
        summary[trend] = {}
        for kind, label in _CLASSES.items():
            picked = np.abs(cosines[[i for i, c in enumerate(kinds) if c == kind], k])
            if picked.size:
                deviation = np.degrees(np.arcsin(np.clip(picked, 0.0, 1.0)))
                summary[trend][label] = {
                    "mean_cosine": float(picked.mean()),
                    "max_cosine": float(picked.max()),
                    "mean_deviation": float(deviation.mean()),
                    "max_deviation": float(deviation.max()),
                }
    count = np.arange(len(frame), dtype=float) - (len(frame) - 1) / 2
    changes = {name: int(np.count_nonzero(np.diff(frame[name].to_numpy()) != 0)) for name in frame.columns}
    return {
        "trends": names,
        "terms": [
            {
                "term": term,
                "kind": kind,
                "cosine": dict(zip(names, map(float, cosines[i]), strict=True)),
                "angle": dict(zip(names, map(float, angles[i]), strict=True)),
            }
            for i, (term, kind) in enumerate(zip(terms, kinds, strict=True))
        ],
        "summary": summary,
        "time_count_dot": {term: float(columns[:, i] @ count) for i, term in enumerate(terms)},
        "level_changes": changes,
        "total_level_changes": int(sum(changes.values())),
    }


def _priorities(kinds: list[str], degree: int) -> list[np.ndarray]:
    """Return the (term, trend) cells of each priority: by trend degree, the main effects and then the rest."""
    main = np.array([k == "main_effect" for k in kinds])
    cells = []
    for k in range(degree):
        for rows in (main, ~main):
            mask = np.zeros((len(kinds), degree), dtype=bool)
            mask[rows, k] = True
            if mask.any():
                cells.append(mask)
    return cells


def _score(dots: np.ndarray, lengths: np.ndarray, masks: list[np.ndarray]) -> np.ndarray:
    """Return the sum of squared cosines in each priority, the most important first."""
    squared = (dots / np.where(lengths > _ZERO, lengths, 1.0)[:, None]) ** 2
    return np.array([squared[mask].sum() for mask in masks])


def _better(a: np.ndarray, b: np.ndarray) -> bool:
    """Say whether score *a* beats *b*: smaller at the first priority where they differ by more than rounding."""
    for x, y in zip(a, b, strict=True):
        if abs(x - y) > 1e-10:
            return bool(x < y)
    return False


def _descend(columns: np.ndarray, trends: np.ndarray, start: np.ndarray, masks: list[np.ndarray]) -> np.ndarray:
    """Improve an order by exchanging two runs at a time, taking the best exchange, until none helps."""
    order = start.copy()
    lengths = np.linalg.norm(columns, axis=0)
    rows = columns[order]
    dots = rows.T @ trends
    score = _score(dots, lengths, masks)
    while True:
        best, best_pair, best_dots = score, None, None
        for i in range(len(order) - 1):
            # Exchanging the runs at times i and j adds (x_i - x_j) (z_j - z_i)' to the dot products.
            step = rows[i] - rows[i + 1 :]
            gap = trends[i + 1 :] - trends[i]
            for offset in range(len(step)):
                if not step[offset].any():
                    continue
                trial = dots + np.outer(step[offset], gap[offset])
                trial_score = _score(trial, lengths, masks)
                if _better(trial_score, best):
                    best, best_pair, best_dots = trial_score, (i, i + 1 + offset), trial
        if best_pair is None:
            return order
        i, j = best_pair
        order[[i, j]] = order[[j, i]]
        rows[[i, j]] = rows[[j, i]]
        dots, score = best_dots, best


def trend_robust_run_order(
    design: pd.DataFrame | np.ndarray,
    model: str = "quadratic",
    degree: int = 3,
    search: tuple[int, int] = (5, 50),
    random_state: int | np.random.Generator | None = None,
) -> dict[str, Any]:
    """Search for an order of a design's runs that keeps the model's terms orthogonal to the time trends.

    The terms are taken in priority, a trend of lower degree first, since a slow drift is most
    often close to linear: the main effects against the linear trend, then the interactions and
    quadratic terms against it, then the main effects against the quadratic trend, and so on. An
    order is better than another when its sum of squared cosines is smaller at the first priority
    where the two differ.

    The search is an iterated local search. From each of several random orders, two runs at a
    time are exchanged, the best exchange first, until no exchange helps. The order found is then
    disturbed many times by a few random exchanges and improved again, and a disturbed order
    is kept when it is no worse. The best order found is returned. The search is a heuristic: it
    does not prove that a better order does not exist, as the integer program of Núñez Ares and
    Goos (2019) does, and it may stop short of an order that is orthogonal where one exists.

    Parameters
    ----------
    design : DataFrame or numpy.ndarray
        The runs in coded units, one row per run, in any order: one column per factor.
    model : {"main_effects", "interactions", "quadratic"}
        The model whose terms should be orthogonal to the trends.
    degree : int
        The highest degree of time trend to guard against, 1 (linear) to 4.
    search : tuple[int, int]
        The effort of the search: the number of random starting orders (default 5), and how many
        times each local optimum is disturbed and improved again (default 50).
    random_state : int, numpy.random.Generator or None
        Seeds the starting orders; see :doc:`/development/reproducibility`.

    Returns
    -------
    dict[str, Any]
        ``order``, the 0-based row positions of *design* in the order to run them; ``design``, the
        runs in that order (a frame); and ``trends``, :func:`run_order_trends` of that order.
    """
    rng = check_random_state(random_state)
    n_starts, n_kicks = search
    if n_starts < 1 or n_kicks < 0:
        raise ValueError(f"search needs at least 1 start and 0 or more kicks; got {search}.")
    frame = _frame(design)
    _, kinds, columns = model_columns(frame, model)
    trends = time_trends(len(frame), degree)
    masks = _priorities(kinds, degree)
    lengths = np.linalg.norm(columns, axis=0)

    def score_of(order: np.ndarray) -> np.ndarray:
        return _score(columns[order].T @ trends, lengths, masks)

    best_order, best_score = None, None
    for _ in range(n_starts):
        order = _descend(columns, trends, rng.permutation(len(frame)), masks)
        score = score_of(order)
        for _ in range(n_kicks):
            trial = order.copy()
            for _ in range(int(rng.integers(2, 5))):  # a few random exchanges, to leave the local optimum
                i, j = rng.choice(len(frame), size=2, replace=False)
                trial[[i, j]] = trial[[j, i]]
            trial = _descend(columns, trends, trial, masks)
            trial_score = score_of(trial)
            if not _better(score, trial_score):  # no worse: move on, across a plateau too
                order, score = trial, trial_score
        if best_score is None or _better(score, best_score):
            best_order, best_score = order, score
    assert best_order is not None  # n_starts >= 1 ran the loop
    chosen = [int(i) for i in best_order]
    return {
        "order": chosen,
        "design": frame.iloc[chosen].reset_index(drop=True),
        "trends": run_order_trends(frame, model=model, degree=degree, order=chosen),
    }
