r"""Measures read off a design's coded runs: what it estimates, what losing a run costs, what its projections hold.

Each function takes a design as its runs in coded units (one row per run, one column per factor)
and the kind of each factor (``"numeric"``, ``"categorical"`` or ``"mixture"``), and needs no fitted
model and no responses: these are properties of the design, known before it is run.

- :func:`estimable_models`: which of main effects, two-factor interactions and the full quadratic
  the design estimates in full.
- :func:`run_loss`: whether the design still estimates its model if any one run is lost, and the
  D-efficiency left after the worst loss. Losing run *i* scales :math:`\det(X^\top X)` by
  :math:`1 - h_i`, where :math:`h_i` is the run's leverage, so one hat matrix answers it for every
  run at once.
- :func:`quadratic_projection` and :func:`projectivity`: the most factors such that every projection
  onto that many still estimates the full quadratic (any design), or still holds every combination
  of their two levels (a two-level design).
- :func:`fingerprint` and :func:`same_design`: whether two designs are one design, with their runs in
  another order, their factors in another order, or a factor's levels swapped.
- :func:`fds_curve`: the fraction-of-design-space curve, the prediction variance that each fraction
  of the cube does not exceed.

Examples
--------
>>> import numpy as np
>>> from process_improve.experiments.design_measures import run_loss
>>> square = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]], dtype=float)
>>> run_loss(square, ["numeric", "numeric"], "main_effects")["survives"]
True
>>> run_loss(square, ["numeric", "numeric"], "interactions")["survives"]
False
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import Counter
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from process_improve._random import check_random_state

if TYPE_CHECKING:
    from collections.abc import Sequence

Model = Literal["main_effects", "interactions", "quadratic"]

#: The models a design is described by, each including the ones before it, as the browser app names them.
MODELS: tuple[Model, ...] = ("main_effects", "interactions", "quadratic")

#: The most factor subsets a projection measure checks for one projection size: past it, the measure
#: stops and reports the largest size it checked in full, a lower bound.
MOST_SUBSETS = 20_000

#: The most steps :func:`relabelling` takes before giving up: past it, two designs are not called one.
MOST_STEPS = 200_000

#: A leverage this close to 1 is 1: the run cannot be lost.
_ONE = 1 - 1e-9

#: The points drawn uniformly from the cube that a fraction-of-design-space curve is read from.
FDS_SAMPLES = 4096


# --------------------------------------------------------------------------- the model matrix
def _columns(values: np.ndarray, kind: str) -> np.ndarray:
    """Return a factor's columns in the model matrix: its coded values, or one indicator per extra level."""
    if kind != "categorical":
        return values[:, None]
    levels = np.unique(values)
    if len(levels) < 2:  # a factor held at one level adds nothing to estimate
        return np.empty((len(values), 0))
    return np.column_stack([values == level for level in levels[1:]]).astype(float)


def model_matrix(coded: np.ndarray, kinds: Sequence[str], model: Model) -> np.ndarray | None:
    """Return a design's model matrix for a model in coded units, or ``None`` when the model does not apply.

    The intercept, then each factor's main effect (a categorical factor's as one indicator column per
    level after its first), then, for ``"interactions"`` and ``"quadratic"``, the products of every
    pair of factors' columns, then, for ``"quadratic"``, the square of every numeric factor.

    Parameters
    ----------
    coded : np.ndarray
        The runs in coded units, one column per factor.
    kinds : Sequence[str]
        Each factor's type: ``"numeric"``, ``"categorical"`` or ``"mixture"``.
    model : {"main_effects", "interactions", "quadratic"}
        The model.

    Returns
    -------
    np.ndarray or None
        The model matrix; ``None`` for a mixture design, whose models are Scheffé's, or when a
        setting is missing.
    """
    if "mixture" in kinds or np.isnan(coded).any():
        return None
    blocks = [_columns(coded[:, j], kind) for j, kind in enumerate(kinds)]
    parts = [np.ones((len(coded), 1)), *blocks]
    if model in {"interactions", "quadratic"}:
        parts += [(a[:, :, None] * b[:, None, :]).reshape(len(coded), -1) for a, b in itertools.combinations(blocks, 2)]
    if model == "quadratic":
        parts += [coded[:, [j]] ** 2 for j, kind in enumerate(kinds) if kind == "numeric"]
    return np.column_stack(parts)


def full_rank(matrix: np.ndarray) -> bool:
    """Say whether a model matrix estimates every one of its coefficients: as many runs as columns, and full rank."""
    return bool(matrix.shape[0] >= matrix.shape[1] and np.linalg.matrix_rank(matrix) == matrix.shape[1])


def estimable_models(coded: np.ndarray, kinds: Sequence[str]) -> dict[str, bool | None]:
    """Say which models a design estimates in full: main effects, two-factor interactions, the full quadratic.

    Each model includes the ones before it (see :func:`model_matrix`). A model is estimable when its
    model matrix has full column rank.

    Parameters
    ----------
    coded : np.ndarray
        The runs in coded units, one column per factor.
    kinds : Sequence[str]
        Each factor's type: ``"numeric"``, ``"categorical"`` or ``"mixture"``.

    Returns
    -------
    dict[str, bool | None]
        For each name in :data:`MODELS`, whether the design estimates it. Every value is ``None``
        for a mixture design, whose models are Scheffé's, or when a setting is missing. The
        interactions are ``None`` for a single factor; the quadratic is ``None`` when no factor
        is numeric, and ``False`` when a numeric factor has fewer than three levels.

    Examples
    --------
    >>> import numpy as np
    >>> square = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]])
    >>> estimable_models(square, ["numeric", "numeric"])
    {'main_effects': True, 'interactions': True, 'quadratic': False}
    """
    out: dict[str, bool | None] = dict.fromkeys(MODELS)
    coded = np.asarray(coded, dtype=float)
    for model in MODELS:
        matrix = model_matrix(coded, kinds, model)
        if matrix is None:
            return out
        out[model] = full_rank(matrix)
    if len(kinds) < 2:
        out["interactions"] = None
    if "numeric" not in kinds:
        out["quadratic"] = None
    return out


# --------------------------------------------------------------------------- losing a run
def leverages(matrix: np.ndarray) -> np.ndarray:
    r"""Return each run's leverage, the diagonal of the hat matrix, for a model matrix of full column rank.

    Parameters
    ----------
    matrix : np.ndarray
        The model matrix, one row per run.

    Returns
    -------
    np.ndarray
        :math:`h_i = x_i^\top (X^\top X)^{-1} x_i` for each run, from 0 to 1; they add up to the
        number of columns.
    """
    q, _ = np.linalg.qr(matrix)
    return np.sum(q**2, axis=1)


def run_loss(coded: np.ndarray, kinds: Sequence[str], model: Model) -> dict[str, Any] | None:
    r"""Say what losing any one run costs a design: whether it still estimates the model, and how well.

    Deleting run *i* from the model matrix :math:`X` scales :math:`\det(X^\top X)` by
    :math:`1 - h_i`, where :math:`h_i` is the run's leverage. So the design still estimates the model
    after losing run *i* exactly when :math:`h_i < 1`, and keeps :math:`(1 - h_i)^{1/p}` of its
    D-efficiency, with *p* coefficients. A saturated design (as many runs as coefficients) has every
    :math:`h_i = 1`: no run can be lost.

    Parameters
    ----------
    coded : np.ndarray
        The runs in coded units, one column per factor.
    kinds : Sequence[str]
        Each factor's type: ``"numeric"``, ``"categorical"`` or ``"mixture"``.
    model : {"main_effects", "interactions", "quadratic"}
        The model the design is to estimate.

    Returns
    -------
    dict or None
        ``survives`` (whether the model is estimable after losing any one run), ``critical_runs``
        (how many runs cannot be lost), ``worst_d_ratio`` (the D-efficiency left after the worst loss,
        relative to the whole design's), and ``leverage`` (each run's). ``None`` when the design does
        not estimate the model, or the model does not apply (see :func:`model_matrix`).

    Examples
    --------
    >>> import numpy as np
    >>> square = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1], [0, 0]], dtype=float)
    >>> found = run_loss(square, ["numeric", "numeric"], "interactions")
    >>> found["survives"], found["critical_runs"], found["leverage"]
    (True, 0, [0.95, 0.95, 0.95, 0.95, 0.2])
    >>> run_loss(square[:4], ["numeric", "numeric"], "interactions")["critical_runs"]
    4
    """
    matrix = model_matrix(np.asarray(coded, dtype=float), kinds, model)
    if matrix is None or not full_rank(matrix):
        return None
    h = np.clip(leverages(matrix), 0, 1)
    worst = float(h.max())
    return {
        "model": model,
        "survives": worst < _ONE,
        "critical_runs": int((h >= _ONE).sum()),
        "worst_d_ratio": round(max(0.0, 1 - worst) ** (1 / matrix.shape[1]), 4),
        "leverage": [round(float(v), 4) for v in h],
    }


# --------------------------------------------------------------------------- the prediction variance
def fds_curve(
    coded: np.ndarray,
    kinds: Sequence[str],
    model: Model,
    *,
    n_points: int = 21,
    random_state: int | np.random.Generator | None = 0,
) -> list[float] | None:
    r"""Return a design's fraction-of-design-space curve: the prediction variance at each fraction of the cube.

    The variance of the model's prediction at a point :math:`\mathbf{x}` of the cube
    :math:`[-1, 1]^k` is :math:`\mathbf{f}(\mathbf{x})^\top (X^\top X)^{-1} \mathbf{f}(\mathbf{x})`
    in units of :math:`\sigma^2`, the run-to-run noise variance, where :math:`\mathbf{f}(\mathbf{x})`
    is the model's row at the point (see :func:`model_matrix`). The curve gives, for each fraction
    from 0 to 1, the variance that this fraction of the cube's points does not exceed: its value at
    0.5 is the median prediction variance over the cube, and at 1 the largest found. The variance is
    not scaled by the number of runs, so a design is read at its own size, as
    :func:`~process_improve.experiments.compare.compare_designs` reads it.

    Parameters
    ----------
    coded : np.ndarray
        The runs in coded units, one column per factor.
    kinds : Sequence[str]
        Each factor's type; the curve is drawn for designs whose factors are all ``"numeric"``.
    model : {"main_effects", "interactions", "quadratic"}
        The model whose predictions are read.
    n_points : int, default 21
        The fractions, spaced evenly from 0 to 1, at which the curve is given.
    random_state : int, np.random.Generator or None, default 0
        Seeds the :data:`FDS_SAMPLES` points drawn uniformly from the cube, so the same design gives
        the same curve.

    Returns
    -------
    list[float] or None
        The prediction variance at each fraction, rising, to four significant figures; ``None``
        when a factor is not numeric, a setting is missing, or the design does not estimate the model.

    Examples
    --------
    >>> import numpy as np
    >>> square = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]], dtype=float)
    >>> curve = fds_curve(square, ["numeric", "numeric"], "main_effects", n_points=3)
    >>> curve[0], curve[-1] <= 0.75
    (0.25, True)
    """
    coded = np.asarray(coded, dtype=float)
    if any(kind != "numeric" for kind in kinds):
        return None
    matrix = model_matrix(coded, kinds, model)
    if matrix is None or not full_rank(matrix):
        return None
    points = check_random_state(random_state).uniform(-1.0, 1.0, size=(FDS_SAMPLES, coded.shape[1]))
    rows = cast("np.ndarray", model_matrix(points, kinds, model))  # numeric points with no gap: never None
    variance = np.sum(rows * np.linalg.solve(matrix.T @ matrix, rows.T).T, axis=1)
    return [float(f"{v:.4g}") for v in np.quantile(variance, np.linspace(0.0, 1.0, n_points))]


# --------------------------------------------------------------------------- projections
def quadratic_projection(coded: np.ndarray, kinds: Sequence[str], most: int = MOST_SUBSETS) -> int | None:
    """Return the most factors such that every projection onto that many estimates the full quadratic in them.

    Projecting a design onto *p* of its factors keeps every run and drops the other factors' columns.
    When only *p* factors turn out to matter, a design whose every *p*-factor projection estimates the
    full quadratic (intercept, main effects, two-factor interactions and squares in those *p*) can fit
    their response surface from the runs already made: the property definitive screening and OMARS
    designs are built for. Only the numeric factors are projected onto.

    Parameters
    ----------
    coded : np.ndarray
        The runs in coded units, one column per factor.
    kinds : Sequence[str]
        Each factor's type: ``"numeric"``, ``"categorical"`` or ``"mixture"``.
    most : int, default MOST_SUBSETS
        The most projections checked for one size. Past it the search stops, and the largest size
        checked in full is returned: a lower bound.

    Returns
    -------
    int or None
        The largest such *p*; 0 when a single factor has fewer than three levels. ``None`` for a
        mixture design, one with a missing setting, or one with no numeric factor.

    Examples
    --------
    >>> import numpy as np
    >>> from process_improve.experiments import Factor, generate_design
    >>> factors = [Factor(name=n, low=-1, high=1) for n in "ABCDEF"]
    >>> dsd = generate_design(factors, design_type="dsd", n_center_points=3)
    >>> quadratic_projection(dsd.design[list("ABCDEF")].to_numpy(dtype=float), ["numeric"] * 6)
    3
    """
    coded = np.asarray(coded, dtype=float)
    numeric = [j for j, kind in enumerate(kinds) if kind == "numeric"]
    if "mixture" in kinds or np.isnan(coded).any() or not numeric:
        return None
    best = 0
    for p in range(1, len(numeric) + 1):
        if math.comb(len(numeric), p) > most:
            break
        for subset in itertools.combinations(numeric, p):
            matrix = model_matrix(coded[:, list(subset)], ["numeric"] * p, "quadratic")
            if matrix is None or not full_rank(matrix):
                return best
        best = p
    return best


def projectivity(coded: np.ndarray, most: int = MOST_SUBSETS) -> int | None:
    """Return a two-level design's projectivity: the most factors whose every projection holds all their combinations.

    A design of projectivity *P* holds, for every choice of *P* factors, every one of the
    :math:`2^P` combinations of their levels: a full factorial in any *P* factors, possibly
    replicated (Box and Tyssedal, 1996). A regular fraction of resolution *R* has projectivity
    *R* - 1. Center runs and any other run off the cube are set aside: the measure is of the
    design's two-level runs.

    Parameters
    ----------
    coded : np.ndarray
        The runs in coded units, one column per factor.
    most : int, default MOST_SUBSETS
        The most projections checked for one size; past it, the largest size checked in full is
        returned, a lower bound.

    Returns
    -------
    int or None
        The projectivity; ``None`` when a setting is missing, or the runs on the cube are not a
        two-level design in every factor.

    Examples
    --------
    >>> import numpy as np
    >>> half = np.array([[-1, -1, 1], [1, -1, -1], [-1, 1, -1], [1, 1, 1]], dtype=float)
    >>> projectivity(half)
    2
    """
    coded = np.asarray(coded, dtype=float)
    if np.isnan(coded).any():
        return None
    cube = coded[np.all(np.isclose(np.abs(coded), 1), axis=1)]
    if len(cube) < 2 or any(len(np.unique(column)) != 2 for column in cube.T):
        return None
    k = cube.shape[1]
    best = 0
    for p in range(1, k + 1):
        if math.comb(k, p) > most:
            break
        for subset in itertools.combinations(range(k), p):
            if len({tuple(row) for row in cube[:, list(subset)]}) < 2**p:
                return best
        best = p
    return best


# --------------------------------------------------------------------------- the same design
def run_counts(matrix: np.ndarray) -> Counter:
    """Return how often each run occurs, its settings rounded to 6 decimals: a design's runs, whatever their order."""
    return Counter(map(tuple, np.round(matrix, 6)))


def relabelling(
    first: np.ndarray, second: np.ndarray, *, signs: bool = True, most: int = MOST_STEPS
) -> list[tuple[int, int]] | None:
    """Return, for each column of ``first``, the column of ``second`` and the sign that make their runs one.

    A design keeps its properties when its runs or its columns are put in another order, or a
    column's levels are swapped (its signs changed, in coded units). The search assigns the columns
    of ``first`` in turn, and drops an assignment as soon as the runs on the columns assigned so far
    differ. ``first`` may use only some of the columns of ``second``, as factors are assigned to some
    columns of an orthogonal array and the others are left for interactions.

    Parameters
    ----------
    first, second : np.ndarray
        Two designs in coded units, with the same number of runs.
    signs : bool, default True
        Whether a column's signs may be changed.
    most : int, default MOST_STEPS
        The most assignments tried; past it the search gives up and returns ``None``.

    Returns
    -------
    list of (int, int) or None
        For each column of ``first``, the column of ``second`` and the sign (1 or -1); ``None`` when
        no relabelling makes the runs one, or none was found within ``most`` steps.
    """
    chosen: list[tuple[int, int]] = []
    steps = 0

    def search(j: int) -> bool:
        nonlocal steps
        if j == first.shape[1]:
            return True
        used = {c for c, _ in chosen}
        for column, sign in itertools.product(range(second.shape[1]), (1, -1) if signs else (1,)):
            steps += 1
            if column in used or steps > most:
                continue
            chosen.append((column, sign))
            relabelled = np.column_stack([second[:, c] * s for c, s in chosen])
            if run_counts(first[:, : j + 1]) == run_counts(relabelled) and search(j + 1):
                return True
            chosen.pop()
        return False

    fits = len(first) == len(second) and first.shape[1] <= second.shape[1]
    return list(chosen) if fits and search(0) else None


def same_design(first: np.ndarray, second: np.ndarray) -> bool:
    """Say whether two designs are one: the same runs, once runs, factors and each factor's two signs are reordered.

    Parameters
    ----------
    first, second : np.ndarray
        Two designs in coded units.

    Returns
    -------
    bool
        Whether a relabelling of ``second`` (see :func:`relabelling`) gives the runs of ``first``,
        with the same number of factors.
    """
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if first.shape != second.shape or np.isnan(first).any() or np.isnan(second).any():
        return False
    return relabelling(first, second) is not None


def fingerprint(coded: np.ndarray) -> str | None:
    """Return a short hash of a design that does not change when its runs or factors are reordered or a sign swapped.

    The hash is of the number of runs and factors, each factor's levels (as their sizes, sorted),
    and, for every set of one to four factors, the size of the sum over the runs of the product of
    their settings (their J-characteristic), sorted; and the same sums of their squares, for one or
    two factors. Two designs that are one (see :func:`same_design`) have the same fingerprint; two
    with the same fingerprint are very likely one, and :func:`same_design` says for sure.

    Parameters
    ----------
    coded : np.ndarray
        The runs in coded units, one column per factor.

    Returns
    -------
    str or None
        Sixteen hexadecimal digits; ``None`` when a setting is missing.

    Examples
    --------
    >>> import numpy as np
    >>> square = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]], dtype=float)
    >>> fingerprint(square) == fingerprint(-square[::-1, ::-1])
    True
    """
    x = np.round(np.asarray(coded, dtype=float), 6)
    if np.isnan(x).any():
        return None
    n, k = x.shape
    parts: list[Any] = [n, k, sorted(sorted(Counter(np.abs(column).tolist()).items()) for column in x.T)]
    for order in range(1, min(k, 4) + 1):
        if math.comb(k, order) > MOST_SUBSETS:
            break
        sums = (np.prod(x[:, list(s)], axis=1).sum() for s in itertools.combinations(range(k), order))
        parts.append(sorted(round(abs(float(v)), 4) for v in sums))
    for order in range(1, min(k, 2) + 1):
        sums = (np.prod(x[:, list(s)] ** 2, axis=1).sum() for s in itertools.combinations(range(k), order))
        parts.append(sorted(round(float(v), 4) for v in sums))
    return hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:16]


def design_classes(designs: dict[str, np.ndarray]) -> dict[str, str]:
    """Group designs that are one design (see :func:`same_design`): return each design's class, by its name.

    Designs are first grouped by their :func:`fingerprint`; within a group, each design joins the
    first class whose first member it is the same design as, or starts a class of its own.

    Parameters
    ----------
    designs : dict[str, np.ndarray]
        Each design's coded runs, by its name. A design with a missing setting is left out.

    Returns
    -------
    dict[str, str]
        Each design's class: its fingerprint, followed by ``-2``, ``-3``, ... for a second or later
        class of designs that share a fingerprint but are not one.
    """
    groups: dict[str, list[str]] = {}
    for name, coded in designs.items():
        if (found := fingerprint(coded)) is not None:
            groups.setdefault(found, []).append(name)
    classes: dict[str, str] = {}
    for found, names in groups.items():
        firsts: list[str] = []
        for name in names:
            match = next((f for f in firsts if same_design(designs[name], designs[f])), None)
            if match is None:
                firsts.append(name)
                match = name
            index = firsts.index(match)
            classes[name] = found if index == 0 else f"{found}-{index + 1}"
    return classes


# --------------------------------------------------------------------------- what the runs are made of
def _zero_count(column: np.ndarray) -> int:
    """Return how many entries of a column are zero."""
    return int(np.isclose(column, 0).sum())


def _same_or_none(counts: list[int]) -> int | None:
    """Return the one value every count shares, or ``None`` when they differ (or there are none)."""
    return counts[0] if counts and all(c == counts[0] for c in counts) else None


def design_structure(
    coded: np.ndarray, names: Sequence[str] | None = None, blocks: np.ndarray | None = None
) -> dict[str, Any]:
    """Describe what a design's runs are made of: levels, middle-level zeros, centre runs, mirror pairs, replicates.

    These are the counts a three-level screening design is described by (Núñez Ares and Goos,
    2020): how often each factor sits at its middle level, and how often each product of two
    three-level factors is zero, both counted over the runs that are not centre runs (``n0ME`` and
    ``n0IE``). A centre run has every three-level factor at 0; a two-level factor, which has no
    middle level, may take any value in it.

    Parameters
    ----------
    coded : np.ndarray
        The runs in coded units, one column per factor.
    names : Sequence[str] or None
        The factors' names; ``"x1"``, ``"x2"``, ... by default.
    blocks : np.ndarray or None
        Each run's block label, to count the runs in each block and say whether every factor is
        balanced across the blocks (the same average level in every block, so the blocks are
        orthogonal to the main effects).

    Returns
    -------
    dict[str, Any]
        - ``n_runs``, ``n_factors``, ``n_distinct_runs``;
        - ``levels``, each factor's distinct levels, and ``level_counts``, by level (written as
          ``"-1"``, ``"0"``, ``"1"``), how often each factor takes it;
        - ``n_two_level``, ``n_three_level``: how many factors have two and three levels;
        - ``n_center_runs``: the runs with every three-level factor at 0;
        - ``zeros``: ``main_effects``, each three-level factor's zeros, and ``interactions``, each
          product of two three-level factors' zeros, both over the runs that are not centre runs,
          with ``n0ME`` and ``n0IE``, the value they all share (``None`` when they differ), and
          ``zeros_with_center_runs``, the same counted over every run;
        - ``mirror_pairs``: how many pairs of the runs that are not centre runs are each other's
          negative, and ``foldover``, whether every such run is in one;
        - ``replicated``: the groups of runs (0-based) with the same settings, each of two or more;
        - with ``blocks``, ``block_sizes`` (runs per block) and ``blocks_balanced``.

    Examples
    --------
    >>> import numpy as np
    >>> dsd = np.array([[0, 1, 1], [0, -1, -1], [1, 0, -1], [-1, 0, 1], [1, 1, 0], [-1, -1, 0], [0, 0, 0]])
    >>> found = design_structure(dsd)
    >>> found["n_center_runs"], found["zeros"]["n0ME"], found["mirror_pairs"], found["foldover"]
    (1, 2, 3, True)
    """
    coded = np.asarray(coded, dtype=float)
    n_runs, n_factors = coded.shape
    names = list(names) if names is not None else [f"x{j + 1}" for j in range(n_factors)]
    rounded = np.round(coded, 9)
    levels = {name: sorted(set(rounded[:, j].tolist())) for j, name in enumerate(names)}
    three = [j for j, name in enumerate(names) if len(levels[name]) == 3 and 0.0 in levels[name]]
    centre = np.all(np.isclose(coded[:, three], 0), axis=1) if three else np.zeros(n_runs, dtype=bool)
    rest = coded[~centre]
    main = {names[j]: _zero_count(rest[:, j]) for j in three}
    pairs = {
        f"{names[a]}:{names[b]}": _zero_count(rest[:, a] * rest[:, b]) for a, b in itertools.combinations(three, 2)
    }
    with_centre_main = {names[j]: _zero_count(coded[:, j]) for j in three}
    with_centre_pairs = {
        f"{names[a]}:{names[b]}": _zero_count(coded[:, a] * coded[:, b]) for a, b in itertools.combinations(three, 2)
    }
    remaining = Counter(map(tuple, np.round(rest, 9)))
    mirror = 0
    for row in list(remaining):
        negative = tuple(0.0 - v + 0.0 for v in row)
        while remaining[row] > 0 and remaining[negative] > 0 and (negative != row or remaining[row] > 1):
            remaining[row] -= 1
            remaining[negative] -= 1
            mirror += 1
    groups: dict[tuple[float, ...], list[int]] = {}
    for i, row in enumerate(map(tuple, rounded)):
        groups.setdefault(row, []).append(i)
    out: dict[str, Any] = {
        "n_runs": n_runs,
        "n_factors": n_factors,
        "n_distinct_runs": len(groups),
        "levels": levels,
        "level_counts": {
            f"{level:g}": {name: int(np.isclose(coded[:, j], level).sum()) for j, name in enumerate(names)}
            for level in sorted(set(rounded.ravel().tolist()))
        },
        "n_two_level": sum(len(v) == 2 for v in levels.values()),
        "n_three_level": sum(len(v) == 3 for v in levels.values()),
        "n_center_runs": int(centre.sum()),
        "zeros": {
            "main_effects": main,
            "interactions": pairs,
            "n0ME": _same_or_none(list(main.values())),
            "n0IE": _same_or_none(list(pairs.values())),
        },
        "zeros_with_center_runs": {
            "main_effects": with_centre_main,
            "interactions": with_centre_pairs,
            "n0ME": _same_or_none(list(with_centre_main.values())),
            "n0IE": _same_or_none(list(with_centre_pairs.values())),
        },
        "mirror_pairs": mirror,
        "foldover": bool(len(rest) > 0 and 2 * mirror == len(rest)),
        "replicated": [runs for runs in groups.values() if len(runs) > 1],
    }
    if blocks is not None:
        labels = np.asarray(blocks)
        sizes = Counter(labels.tolist())
        means = [coded[labels == label].mean(axis=0) for label in sizes]
        out["block_sizes"] = dict(sizes)
        out["blocks_balanced"] = bool(np.allclose(means, coded.mean(axis=0)))
    return out
