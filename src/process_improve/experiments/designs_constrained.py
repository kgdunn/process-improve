# (c) Kevin Dunn, 2010-2026. MIT License.

"""Optimal designs (D, I, A and E) over a constrained factor region, by candidate exchange.

A ``Constraint`` such as ``"3*T + 5*D <= 600"`` cuts a corner off the factor box.
Classical designs cannot follow that cut, so the design is chosen from a set of
feasible *candidate* points instead, in three steps:

1. **Candidate set.** A grid over the box, plus the points where each constraint
   boundary crosses a grid line (found by bisection) and the vertices where linear
   constraints meet each other or the box, minus every infeasible point. The boundary
   points matter: a D-optimal design pushes its runs to the edge of the region, and
   without them the nearest grid point can sit well inside it.
2. **Model matrix.** Each candidate is expanded into the columns of the model the
   design is for (intercept, main effects, two-factor interactions, squares).
3. **Exchange.** A Fedorov exchange swaps design runs for candidates while the
   criterion improves (see :class:`Criterion`: the determinant of the information
   matrix ``X'X`` for D, a weighted trace of its inverse for A and I, its smallest
   eigenvalue for E), with several random starts. For D, A and I each run in turn
   takes its best swap, with ``(X'X)^-1`` kept current by rank-one updates.

Constraint expressions are parsed into a small arithmetic tree and evaluated with
numpy. Nothing is passed to ``eval``, so an expression from an untrusted caller can
only compute a number.
"""

from __future__ import annotations

import ast
import itertools
import logging
import math
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.stats import qmc

from process_improve._random import check_random_state
from process_improve.experiments._uniform_sampling import UniformSampler
from process_improve.experiments.factor import FactorType

if TYPE_CHECKING:
    from process_improve.experiments.factor import Constraint, Factor

logger = logging.getLogger(__name__)

#: Model types the candidate exchange builds a model matrix for.
_MODEL_TYPES = ("main_effects", "interactions", "quadratic")
#: Upper limit on the candidate grid; a larger grid is refused before allocation.
MAX_CANDIDATES = 100_000
#: Points taken from a grid of requested levels too large to list (a power of two, for the Sobol sequence): the
#: exchange's cost grows with this. An automatic grid is sampled down to :func:`_candidate_cap` instead.
_SAMPLED_CANDIDATES = 2**15
#: Longest constraint expression accepted, which also bounds the parse depth.
MAX_EXPRESSION_LENGTH = 500
#: Slack when testing ``g(x) <= 0``, so points found on a boundary are kept.
_FEASIBILITY_TOL = 1e-9
_BISECTION_STEPS = 50
_MAX_EXCHANGES = 500
#: Rank-one updates of ``M^-1`` between full re-factorisations, which bound the drift of the updates.
_REFACTOR_EVERY = 50
#: ``M^-1`` is also rebuilt when the largest candidate variance falls below this fraction of its value at the last
#: rebuild, as it does while a singular design is filled in: the rounding error scales with the old, larger values.
_COLLAPSE = 1e-3
#: Most passes over the design rows the row-wise exchange makes.
_MAX_PASSES = 100
_N_STARTS = 5
#: Most random starts of the D, A and I exchange, and the work (candidates x coefficients x free runs, summed over
#: the starts) that sets how many a problem gets between ``_N_STARTS`` and this (see :func:`_n_starts`).
_MAX_STARTS = 50
_START_WORK = 400_000_000
#: A greedy start step picks at random among candidates whose variance is within this fraction of the largest.
_GREEDY_SLACK = 0.05

# ---------------------------------------------------------------------------
# 1. Constraint expressions -> vectorised inequality functions g(x) <= 0
# ---------------------------------------------------------------------------

_Env = Mapping[str, np.ndarray]
_Evaluator = Callable[[_Env], Any]

_BINARY_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: np.add,
    ast.Sub: np.subtract,
    ast.Mult: np.multiply,
    ast.Div: np.divide,
    ast.Pow: np.power,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {ast.USub: np.negative, ast.UAdd: np.positive}
_FUNCTIONS: dict[str, Callable[[Any], Any]] = {
    "abs": np.abs,
    "sqrt": np.sqrt,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
}
#: For each allowed comparison, whether ``g`` is ``rhs - lhs`` (True) or ``lhs - rhs``.
_FLIP: dict[type[ast.cmpop], bool] = {ast.Lt: False, ast.LtE: False, ast.Gt: True, ast.GtE: True}


def _compile(node: ast.expr, names: set[str]) -> _Evaluator:
    """Turn one arithmetic node into a function of the factor values.

    Only numbers, factor names, ``+ - * / **``, unary signs and the functions in
    ``_FUNCTIONS`` are accepted; anything else raises ``ValueError``.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float) and not isinstance(node.value, bool):
        number = float(node.value)
        return lambda _env: number
    if isinstance(node, ast.Name):
        name = node.id
        if name not in names:
            raise ValueError(f"Unknown name {name!r}; constraints may use the continuous factors {sorted(names)}.")
        return lambda env: env[name]
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        fn, lhs, rhs = _BINARY_OPS[type(node.op)], _compile(node.left, names), _compile(node.right, names)
        return lambda env: fn(lhs(env), rhs(env))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        unary, operand = _UNARY_OPS[type(node.op)], _compile(node.operand, names)
        return lambda env: unary(operand(env))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and len(node.args) == 1
        and not node.keywords
    ):
        func, argument = _FUNCTIONS[node.func.id], _compile(node.args[0], names)
        return lambda env: func(argument(env))
    raise ValueError(f"Unsupported syntax in constraint: {ast.unparse(node)!r}.")


def _difference(lhs: _Evaluator, rhs: _Evaluator) -> _Evaluator:
    """Return ``g = lhs - rhs``, which is feasible where ``g <= 0``."""
    return lambda env: np.asarray(lhs(env), dtype=float) - np.asarray(rhs(env), dtype=float)


def parse_constraint(expression: str, names: set[str]) -> list[_Evaluator]:
    """Parse a constraint into inequality functions, each feasible where ``g(x) <= 0``.

    Parameters
    ----------
    expression : str
        An inequality in actual units, e.g. ``"3*T + 5*D <= 600"``. A chained
        comparison such as ``"400 <= 3*T + 5*D <= 600"`` gives two inequalities.
        Strict and non-strict comparisons are treated alike, since the boundary
        of a continuous region has no volume.
    names : set[str]
        Names that may appear in the expression (the continuous factors).

    Returns
    -------
    list[Callable]
        One function per comparison. Each takes a mapping from factor name to an
        array of actual values and returns an array ``g``.

    Raises
    ------
    ValueError
        For an expression that is too long, is not a comparison, uses ``==`` or
        ``!=``, or uses a name or syntax outside the allowed set.
    """
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ValueError(f"Constraint expression is longer than {MAX_EXPRESSION_LENGTH} characters.")
    try:
        tree = ast.parse(expression, mode="eval").body
    except SyntaxError as err:
        raise ValueError(f"Constraint {expression!r} is not a valid expression: {err.msg}.") from err
    compare = tree if isinstance(tree, ast.Compare) else None
    if compare is None:
        raise ValueError(f"Constraint {expression!r} must be an inequality, e.g. 'A + B <= 10'.")
    terms = [_compile(compare.left, names)] + [_compile(c, names) for c in compare.comparators]

    inequalities = []
    for op, lhs, rhs in zip(compare.ops, terms[:-1], terms[1:], strict=True):
        flip = _FLIP.get(type(op))
        if flip is None:
            raise ValueError(
                f"Constraint {expression!r}: only <, <=, >, >= are supported. An equality leaves "
                "a region with no volume to place runs in; for proportions summing to 1, use "
                "mixture factors."
            )
        inequalities.append(_difference(rhs, lhs) if flip else _difference(lhs, rhs))
    return inequalities


# ---------------------------------------------------------------------------
# 2. Candidate set: grid + boundary crossings, filtered to the feasible region
# ---------------------------------------------------------------------------


@dataclass
class _Region:
    """The factor box, split into continuous and categorical parts."""

    continuous: list[Factor]
    categorical: list[Factor]
    inequalities: list[_Evaluator]

    def actual(self, coded: np.ndarray) -> dict[str, np.ndarray]:
        """Map coded continuous columns to ``{name: actual values}``."""
        return {
            f.name: f.center + coded[:, j] * (f.high - f.low) / 2.0  # type: ignore[operator]
            for j, f in enumerate(self.continuous)
        }

    def slack(self, coded: np.ndarray) -> np.ndarray:
        """Return the largest ``g`` per row: the point is feasible where this is ``<= 0``."""
        env = self.actual(coded)
        g = [np.broadcast_to(ineq(env), (coded.shape[0],)) for ineq in self.inequalities]
        return np.max(g, axis=0) if g else np.full(coded.shape[0], -np.inf)


#: The automatic grid holds at most ``max(_MIN_CANDIDATE_CAP, _CANDIDATES_PER_COEFFICIENT * p)`` candidates for a
#: model with ``p`` coefficients (and never more than ``MAX_CANDIDATES``); a larger grid is sampled down to that.
_MIN_CANDIDATE_CAP = 5000
_CANDIDATES_PER_COEFFICIENT = 200
#: Criteria on the prediction variance over the region, which place runs inside it as well as on its edges.
_VARIANCE_CRITERIA = ("i_optimal", "g_optimal")


def _candidate_cap(region: _Region, model_type: str) -> int:
    """Most candidates the automatic grid may hold, which grows with the number of model coefficients."""
    n_parameters = model_matrix(
        region, np.zeros((1, len(region.continuous))), np.zeros((1, len(region.categorical)), dtype=int), model_type
    ).shape[1]
    return min(MAX_CANDIDATES, max(_MIN_CANDIDATE_CAP, _CANDIDATES_PER_COEFFICIENT * n_parameters))


def _level_choices(model_type: str, criterion: str) -> list[int]:
    """Grid resolutions to try, finest first.

    A quadratic model needs 3 levels; for criteria that push runs to the edges of the
    region (D, A, E, K) an even level count adds no centre and only displaces the
    middle level, so 4 is skipped. Models without squared terms may use 2 levels.
    """
    if model_type != "quadratic":
        return [5, 4, 3, 2]
    return [5, 4, 3] if criterion in _VARIANCE_CRITERIA else [5, 3]


def _grid_levels(
    region: _Region, n_levels: int | None, model_type: str, criterion: str = "d_optimal"
) -> tuple[int, bool]:
    """Pick the continuous grid resolution, and whether the grid is too large to list in full.

    With ``n_levels`` given, that grid is used, and sampled when it has more than
    ``MAX_CANDIDATES`` points. Otherwise the finest of :func:`_level_choices` whose grid
    stays within :func:`_candidate_cap` is used, and the coarsest is sampled when none
    does (see :func:`_lattice_points`). The cap grows with the model, so the exchange's
    cost does too, and not with the number of grid levels: 5 levels on 7 factors are
    78,125 points, and 3 levels lose nothing for a D-optimal quadratic design on a box.
    """
    n_cat = int(np.prod([len(f.levels or []) for f in region.categorical]))
    if n_levels is not None:
        if n_levels < 2:
            raise ValueError("n_levels must be at least 2.")
        return n_levels, n_levels ** len(region.continuous) * n_cat > MAX_CANDIDATES
    cap = _candidate_cap(region, model_type)
    choices = _level_choices(model_type, criterion)
    for n in choices:
        if n ** len(region.continuous) * n_cat <= cap:
            return n, False
    return choices[-1], True


def _lattice_points(shape: list[int], k_cont: int, n_sampled: int | None, edges: bool = False) -> np.ndarray:
    """Index vectors of the candidate grid: all of it (``n_sampled`` None), or ``n_sampled`` well-spread points of it.

    The sample is the unscrambled Sobol sequence rounded down onto the grid: evenly
    spread over the grid, and deterministic without drawing on any random generator.
    With ``edges`` (for a 3-level grid), a continuous coordinate takes the middle level
    with probability ``1 / k_cont`` rather than 1/3, so a typical point has one factor at
    its middle level and the rest at the extremes: those are the points a D-optimal
    quadratic design is built from, and a uniform sample of a large grid holds few of
    them (2% of the points of the 10-factor grid have no factor at its middle level).
    A sampled grid on an odd number of levels also holds the centre and the face
    centres (one factor at an extreme, the rest at the middle), the points a quadratic
    model leans on and a random sample would rarely contain.
    """
    if n_sampled is None:
        return np.indices(shape).reshape(len(shape), -1).T
    unit = qmc.Sobol(d=len(shape), scramble=False).random_base2(math.ceil(math.log2(n_sampled)))[:n_sampled]
    idx = np.minimum((unit * shape).astype(int), np.array(shape) - 1)
    if edges and k_cont:
        p_middle = min(1.0 / 3.0, 1.0 / k_cont)
        cont = unit[:, :k_cont]
        idx[:, :k_cont] = np.where(cont < (1.0 - p_middle) / 2, 0, np.where(cont < (1.0 + p_middle) / 2, 1, 2))
    levels = shape[0] if k_cont else 0
    if levels % 2:
        middle = np.full((2 * k_cont + 1, len(shape)), levels // 2)
        middle[:, k_cont:] = 0
        for axis in range(k_cont):
            middle[1 + 2 * axis, axis], middle[2 + 2 * axis, axis] = 0, levels - 1
        idx = np.vstack([middle, idx])
    _, first = np.unique(idx, axis=0, return_index=True)
    return idx[np.sort(first)]


def _boundary_points(region: _Region, grid: np.ndarray, cat_idx: np.ndarray, step: float) -> tuple:
    """Find where each constraint boundary crosses a grid edge, by vectorised bisection.

    For every edge between neighbouring grid points along one axis where one end is
    feasible for a constraint and the other is not, the crossing point is located on
    that edge. The feasible side of the final bracket is returned, so the point
    satisfies the constraint it came from; other constraints are checked afterwards.
    """
    points, cats = [np.empty((0, grid.shape[1]))], [np.empty((0, cat_idx.shape[1]), dtype=int)]
    for axis in range(grid.shape[1]):
        lower = grid[:, axis] < 1.0 - step / 2
        start, cat = grid[lower], cat_idx[lower]
        direction = np.zeros(grid.shape[1])
        direction[axis] = step
        for ineq in region.inequalities:
            g_lo = ineq(region.actual(start)) <= 0
            g_hi = ineq(region.actual(start + direction)) <= 0
            cross = np.broadcast_to(g_lo != g_hi, (start.shape[0],))
            if not cross.any():
                continue
            base, lo_feasible = start[cross], np.broadcast_to(g_lo, cross.shape)[cross]
            t_feas, t_infeas = np.where(lo_feasible, 0.0, 1.0), np.where(lo_feasible, 1.0, 0.0)
            for _ in range(_BISECTION_STEPS):
                t_mid = (t_feas + t_infeas) / 2
                ok = ineq(region.actual(base + t_mid[:, None] * direction)) <= 0
                t_feas, t_infeas = np.where(ok, t_mid, t_feas), np.where(ok, t_infeas, t_mid)
            points.append(base + t_feas[:, None] * direction)
            cats.append(cat[cross])
    return np.vstack(points), np.vstack(cats)


#: Most linear systems solved to find the vertices of a region cut by linear constraints.
_MAX_VERTEX_SYSTEMS = 200_000


def _coded_inequality(region: _Region, g: _Evaluator) -> Callable[[np.ndarray], np.ndarray]:
    """``g`` as a function of coded points ``(n, k)``."""
    return lambda x: np.broadcast_to(np.asarray(g(region.actual(np.atleast_2d(x))), dtype=float), (len(x),))


def _polytope_vertices(region: _Region) -> np.ndarray:
    """Vertices of the box cut by the region's linear constraints, in coded units.

    A vertex is a point where ``k`` of the hyperplanes (box faces and linear constraint
    boundaries) meet; each choice of ``k`` is solved, and the solutions inside the
    region are kept. These are where D-optimal runs go, and two constraints usually
    meet away from every grid line, where neither the grid nor the boundary crossings
    reach (Atkinson, Donev and Tobias 2007, ch. 12). Non-linear constraints only filter
    the result. Returns no points when there is no linear constraint, or when there
    would be more than ``_MAX_VERTEX_SYSTEMS`` systems to solve.
    """
    from process_improve.experiments._uniform_sampling import _affine_form  # noqa: PLC0415

    k = len(region.continuous)
    ones = np.ones(k)
    forms = [_affine_form(_coded_inequality(region, g), -ones, ones) for g in region.inequalities]
    rows = [f for f in forms if f is not None]
    if not rows or k == 0 or math.comb(2 * k + len(rows), k) > _MAX_VERTEX_SYSTEMS:
        return np.empty((0, k))
    a_mat = np.vstack([np.eye(k), -np.eye(k), *[a[None, :] for a, _ in rows]])
    b_vec = np.concatenate([ones, ones, [b for _, b in rows]])
    norms = np.linalg.norm(a_mat, axis=1)
    keep = norms > 0
    a_mat, b_vec = a_mat[keep] / norms[keep, None], b_vec[keep] / norms[keep]
    subsets = np.array(list(itertools.combinations(range(len(a_mat)), k)), dtype=int).reshape(-1, k)
    systems, rhs = a_mat[subsets], b_vec[subsets]
    solvable = np.abs(np.linalg.det(systems)) > 1e-10
    points = np.linalg.solve(systems[solvable], rhs[solvable][..., None])[..., 0]
    points = points[np.all(points @ a_mat.T <= b_vec + 1e-9, axis=1)]
    points = np.clip(points, -1.0, 1.0)
    return np.unique(points.round(12), axis=0)


def _with_every_level(points: np.ndarray, cat_shape: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Pair every point with every combination of categorical levels, unless that is more than ``MAX_CANDIDATES``."""
    combos = np.indices(cat_shape).reshape(len(cat_shape), -1).T if cat_shape else np.empty((1, 0), dtype=int)
    if len(points) * len(combos) > MAX_CANDIDATES:
        return np.empty((0, points.shape[1])), np.empty((0, len(cat_shape)), dtype=int)
    return np.repeat(points, len(combos), axis=0), np.tile(combos, (len(points), 1))


def build_candidates(
    region: _Region, n_levels: int | None = None, model_type: str = "quadratic", criterion: str = "d_optimal"
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return the feasible candidate points in coded units.

    The candidates are the grid (see :func:`_grid_levels`), the points where each
    constraint boundary crosses a grid edge (:func:`_boundary_points`), and the
    vertices of the region cut by its linear constraints (:func:`_polytope_vertices`),
    each with every categorical level combination.

    Parameters
    ----------
    region : _Region
        Factors and constraints.
    n_levels : int or None
        Grid levels per continuous factor; ``None`` picks them (see :func:`_grid_levels`).
    model_type : str
        The model the design is for: without squared terms, 2 levels may be used.
    criterion : str
        The optimality criterion the design is for: I- and G-optimality may use 4 levels
        on a quadratic model, the others go from 5 levels to 3 (see :func:`_level_choices`).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, dict]
        Coded continuous values ``(n, k_cont)``, categorical level indices
        ``(n, k_cat)``, and counts for the metadata.
    """
    levels, sampled = _grid_levels(region, n_levels, model_type, criterion)
    shape = [levels] * len(region.continuous) + [len(f.levels or []) for f in region.categorical]
    k_cont = len(region.continuous)
    n_sampled = None
    if sampled:
        n_sampled = _SAMPLED_CANDIDATES if n_levels is not None else _candidate_cap(region, model_type)
    edges = n_levels is None and levels == 3 and model_type == "quadratic" and criterion not in _VARIANCE_CRITERIA
    idx = _lattice_points(shape, k_cont, n_sampled, edges)
    if sampled:
        logger.info(
            "The %d-level grid on %d factors is too large to list; sampling %d points.", levels, k_cont, len(idx)
        )
    grid = np.linspace(-1.0, 1.0, levels)[idx[:, :k_cont]]
    cat_idx = idx[:, k_cont:]

    extra, extra_cat = _boundary_points(region, grid, cat_idx, step=2.0 / (levels - 1))
    corners, corner_cat = _with_every_level(_polytope_vertices(region), shape[k_cont:])
    coded = np.vstack([grid, extra, corners])
    cats = np.vstack([cat_idx, extra_cat, corner_cat])

    feasible = region.slack(coded) <= _FEASIBILITY_TOL
    coded, cats = coded[feasible], cats[feasible]
    # Drop duplicates (a boundary point can coincide with a grid point).
    _, unique = np.unique(np.hstack([coded.round(9), cats]), axis=0, return_index=True)
    unique.sort()
    counts = {
        "n_levels": levels,
        "n_grid_points": grid.shape[0],
        "grid_sampled": sampled,
        "n_boundary_points": extra.shape[0],
        "n_vertex_points": corners.shape[0],
        "n_candidates": unique.size,
    }
    return coded[unique], cats[unique], counts


# ---------------------------------------------------------------------------
# 3. Model matrix and Fedorov exchange
# ---------------------------------------------------------------------------


def model_matrix(region: _Region, coded: np.ndarray, cats: np.ndarray, model_type: str) -> np.ndarray:
    """Expand points into model columns: intercept, main effects, interactions, squares.

    A categorical factor with ``L`` levels contributes ``L - 1`` indicator columns.
    The D-criterion does not depend on which full-rank coding is used, so the choice
    of reference level does not change the selected design.
    """
    blocks = [coded[:, [j]] for j in range(coded.shape[1])]
    for j, f in enumerate(region.categorical):
        blocks.append((cats[:, [j]] == np.arange(1, len(f.levels or []))).astype(float))

    columns = [np.ones((coded.shape[0], 1)), *blocks]
    if model_type in {"interactions", "quadratic"}:
        columns.extend(
            np.einsum("ni,nj->nij", blocks[i], blocks[j]).reshape(coded.shape[0], -1)
            for i in range(len(blocks))
            for j in range(i + 1, len(blocks))
        )
    if model_type == "quadratic":
        columns.append(coded**2)
    return np.hstack(columns)


#: Candidate rows handled at a time when a quadratic form is taken over the whole candidate set.
_CHUNK_ROWS = 8192


def _quadratic_forms(rows: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """``r' matrix r`` for every row ``r`` of ``rows``, a block of rows at a time to bound the temporaries."""
    out = np.empty(len(rows))
    for start in range(0, len(rows), _CHUNK_ROWS):
        block = rows[start : start + _CHUNK_ROWS]
        out[start : start + _CHUNK_ROWS] = np.einsum("ij,ij->i", block @ matrix, block)
    return out


def _greedy_start(f_cand: np.ndarray, f_fixed: np.ndarray, n_free: int, rng: np.random.Generator) -> np.ndarray:
    """Build a non-singular starting design by adding a high-variance candidate each step.

    Each step picks at random among the candidates whose prediction variance is within
    ``_GREEDY_SLACK`` of the largest. On a grid many candidates tie for the largest
    variance; always taking the first of them gave nearly the same start every time,
    so the restarts of the exchange explored almost nothing (one or two distinct local
    optima from five starts). Adding a run is a rank-one update of ``M^-1`` and of the
    variance vector, so a step costs one product of the candidate matrix with a vector.
    Both are rebuilt from ``M`` every ``_REFACTOR_EVERY`` steps, and whenever the largest
    variance has fallen by ``_COLLAPSE`` since the last rebuild: the ridge makes the first
    variances about ``1e6`` times the final ones, and the rounding error they leave behind
    would otherwise swamp the final values.
    """
    p = f_cand.shape[1]
    info = f_fixed.T @ f_fixed + 1e-6 * np.eye(p)  # small ridge so the first steps are defined
    rows = [int(rng.integers(f_cand.shape[0]))]
    info += np.outer(f_cand[rows[0]], f_cand[rows[0]])
    m_inv = np.linalg.inv(info)
    variance = _quadratic_forms(f_cand, m_inv)
    refactored, peak = 0, float(variance.max())
    for step in range(1, n_free):
        near = np.flatnonzero(variance >= (1.0 - _GREEDY_SLACK) * variance.max())
        best = int(rng.choice(near))
        rows.append(best)
        added = f_cand[best]
        info += np.outer(added, added)
        if step == n_free - 1:
            break
        m_inv_added = m_inv @ added
        scale = 1.0 / (1.0 + float(added @ m_inv_added))
        variance -= scale * (f_cand @ m_inv_added) ** 2
        m_inv -= scale * np.outer(m_inv_added, m_inv_added)
        if step - refactored >= _REFACTOR_EVERY or variance.max() < _COLLAPSE * peak:
            m_inv = np.linalg.inv(info)
            variance = _quadratic_forms(f_cand, m_inv)
            refactored, peak = step, float(variance.max())
    return np.array(rows)


@dataclass(frozen=True)
class Criterion:
    """The optimality criterion an exchange maximises.

    ``"d_optimal"`` maximises ``log det(M)``, with ``M = X'X``. ``"a_optimal"`` and
    ``"i_optimal"`` minimise ``trace(M^-1 W)``: with ``W = I`` that is the summed
    variance of the coefficients (A), and with ``W`` the average of ``f(x) f(x)'``
    over the region it is the average prediction variance over the region (I), which
    is what ``evaluate_design`` reports as I-efficiency. ``"e_optimal"`` maximises the
    smallest eigenvalue of ``M``, which bounds the variance of the worst-estimated
    linear combination of the coefficients.

    Use :meth:`d`, :meth:`a`, :meth:`i` or :meth:`e` to build one.
    """

    name: str
    weights: np.ndarray | None = None
    region_rows: np.ndarray | None = None

    @classmethod
    def d(cls) -> Criterion:
        """D-optimality."""
        return cls("d_optimal")

    @classmethod
    def a(cls, n_parameters: int) -> Criterion:
        """A-optimality for a model with ``n_parameters`` coefficients."""
        return cls("a_optimal", np.eye(n_parameters))

    @classmethod
    def i(cls, region_rows: np.ndarray) -> Criterion:
        """I-optimality, with the moment matrix estimated from model rows sampled uniformly in the region."""
        return cls("i_optimal", region_rows.T @ region_rows / len(region_rows))

    @classmethod
    def e(cls) -> Criterion:
        """E-optimality."""
        return cls("e_optimal")

    @classmethod
    def g(cls, region_rows: np.ndarray) -> Criterion:
        """G-optimality: the largest prediction variance over the model rows ``region_rows``."""
        return cls("g_optimal", region_rows=region_rows)

    @classmethod
    def k(cls) -> Criterion:
        """K-optimality: the condition number of ``X'X`` (Ye and Zhou 2013)."""
        return cls("k_optimal")

    def value(self, info: np.ndarray) -> float:
        """Score an information matrix; higher is better.

        ``log det(M)`` (D), ``-trace(M^-1 W)`` (A, I), ``lambda_min`` (E), minus the largest
        prediction variance over the region (G), and ``-log(lambda_max / lambda_min)`` (K).
        """
        if self.name in ("e_optimal", "k_optimal", "g_optimal"):
            return _spectral_or_minimax_value(self, info)
        if self.weights is None:
            sign, logdet = np.linalg.slogdet(info)
            return float(logdet) if sign > 0 else -np.inf
        if np.linalg.matrix_rank(info) < info.shape[0]:
            return -np.inf
        return -float(np.trace(np.linalg.solve(info, self.weights)))

    @property
    def n_phases(self) -> int:
        """Exchange phases: E-optimality climbs ``phi_p`` first, then polishes ``lambda_min``."""
        return 2 if self.name == "e_optimal" else 1

    def best_swap(
        self, info: np.ndarray, f_design: np.ndarray, f_cand: np.ndarray, phase: int = 0
    ) -> tuple[int, int, float]:
        """Return ``(i, j, gain)`` for the best single swap: design row ``i`` out, candidate ``j`` in."""
        if self.name == "e_optimal":
            return _best_e_swap(info, f_design, f_cand, polish=phase > 0)
        if self.name == "k_optimal":
            return _best_k_swap(info, f_design, f_cand)
        gains = self.swap_gains(np.linalg.pinv(info), f_design, f_cand)
        i, j = np.unravel_index(np.argmax(gains), gains.shape)
        return int(i), int(j), float(gains[i, j])

    def swap_gains(self, m_inv: np.ndarray, f_design: np.ndarray, f_cand: np.ndarray) -> np.ndarray:
        """Return the gain of every swap (design run ``i`` out, candidate ``j`` in), shape ``(n_design, n_cand)``.

        With ``A = M^-1`` and ``d(a, b) = f(a)' A f(b)``, D-optimality multiplies the
        determinant by ``1 + d(j) - d(i) - [d(i) d(j) - d(i, j)**2]`` (Fedorov 1972).
        For the trace criteria, two Sherman-Morrison updates (add ``j``, then remove
        ``i``) with ``b(a, b) = f(a)' A W A f(b)`` give the new trace in closed form::

            s = 1 + d(j)
            trace_new = trace - b(j)/s + [b(i) - 2 d(i,j) b(i,j)/s + d(i,j)**2 b(j)/s**2] / [1 - d(i) + d(i,j)**2/s]

        so both kinds are scored for all pairs in a few matrix products.
        """
        a_design, a_cand = f_design @ m_inv, f_cand @ m_inv
        d_i = np.einsum("ij,ij->i", a_design, f_design)[:, None]
        d_j = np.einsum("ij,ij->i", a_cand, f_cand)[None, :]
        d_ij = a_design @ f_cand.T
        if self.weights is None:
            return _d_gain(d_i, d_j, d_ij)
        b_design, b_cand = a_design @ self.weights, a_cand @ self.weights
        b_i = np.einsum("ij,ij->i", b_design, a_design)[:, None]
        b_j = np.einsum("ij,ij->i", b_cand, a_cand)[None, :]
        b_ij = b_design @ a_cand.T
        return _trace_gain((d_i, d_j, d_ij), (b_i, b_j, b_ij))


def _d_gain(d_i: float | np.ndarray, d_j: np.ndarray, d_ij: np.ndarray) -> np.ndarray:
    """Return the relative change in ``det(M)`` from swapping run ``i`` for candidate ``j``."""
    return d_j - d_i - (d_i * d_j - d_ij**2)


def _trace_gain(d: tuple, b: tuple) -> np.ndarray:
    """Return the fall in ``trace(M^-1 W)`` from swapping run ``i`` for candidate ``j`` (:meth:`Criterion.swap_gains`).

    ``d`` holds ``(d(i), d(j), d(i, j))`` and ``b`` holds ``(b(i), b(j), b(i, j))``, as
    scalars or arrays that broadcast against each other.
    """
    (d_i, d_j, d_ij), (b_i, b_j, b_ij) = d, b
    s = 1.0 + d_j
    denominator = 1.0 - d_i + d_ij**2 / s
    numerator = b_i - 2.0 * d_ij * b_ij / s + d_ij**2 * b_j / s**2
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = b_j / s - numerator / denominator
    return np.where(denominator > 1e-10, gain, -np.inf)


#: Criteria whose swap gains have a closed form (:meth:`Criterion.swap_gains`), climbed by :func:`_row_exchange`.
_CLOSED_FORM_CRITERIA = ("d_optimal", "a_optimal", "i_optimal")


def _regularised_inverse(info: np.ndarray) -> np.ndarray:
    """``M^-1``, with a small ridge when ``M`` is singular so swaps that fill the missing directions score highest."""
    eigenvalues = np.linalg.eigvalsh(info)
    largest = max(float(eigenvalues[-1]), 1.0)
    ridge = 0.0 if eigenvalues[0] > 1e-10 * largest else 1e-8 * largest
    return np.linalg.inv(info + ridge * np.eye(len(info)))


#: Entries of the (design rows x candidates) block of swap-gain terms the row-wise exchange keeps.
_BLOCK_ENTRIES = 2**20
#: Most design rows in such a block: each swap updates the terms of the rows after it.
_BLOCK_ROWS = 32


@dataclass
class _RowTerms:
    """Swap-gain terms of a block of design rows: ``d(i)``, ``d(i, j)``, and ``b(i)``, ``b(i, j)`` for a trace."""

    d_i: np.ndarray
    d_ij: np.ndarray
    b_i: np.ndarray | None = None
    b_ij: np.ndarray | None = None

    def follow(self, x: np.ndarray, step: tuple, start: int) -> None:
        """Bring rows ``start:`` (model rows ``x``) up to date after one rank-one update ``step`` of ``A`` and ``B``.

        With ``A <- A - s u u'``, ``d(i, j)`` falls by ``s (u'x)(F u)`` and ``d(i)`` by
        ``s (u'x)^2``; ``B`` changes by two cross terms and a square term, and so do
        ``b(i, j)`` and ``b(i)``. Each costs a few passes over the block, where a fresh
        score would multiply the block by the whole candidate matrix again.
        """
        u, g, s, vbv, f_u, f_g = step
        ux = x @ u
        self.d_i[start:] -= s * ux**2
        self.d_ij[start:] -= s * np.multiply.outer(ux, f_u)
        if self.b_i is not None and self.b_ij is not None:
            gx = x @ g
            self.b_i[start:] += s * (s * vbv * ux**2 - 2.0 * ux * gx)
            self.b_ij[start:] += s * (
                np.multiply.outer(s * vbv * ux - gx, f_u) - np.multiply.outer(ux, f_g)  # type: ignore[arg-type]
            )


class _ExchangeState:
    """``M^-1`` of a design and the candidates' variance terms, kept current through each swap.

    For D-optimality the state is ``A = M^-1`` and ``d(j) = f(j)' A f(j)`` for every
    candidate; the trace criteria also keep ``B = A W A`` and ``b(j) = f(j)' B f(j)``.
    A swap is two rank-one (Sherman-Morrison) updates, adding the candidate and then
    removing the run. Each costs one product of the candidate matrix with a vector (two
    for a trace) instead of a new inverse and a new pass over every (run, candidate)
    pair; the removal reuses the products the swap was scored with. The state is rebuilt
    from ``X'X`` every ``_REFACTOR_EVERY`` swaps, and when the largest variance collapses
    (see ``_COLLAPSE``), to bound rounding drift.
    """

    def __init__(self, f_cand: np.ndarray, f_fixed: np.ndarray, rows: np.ndarray, weights: np.ndarray | None) -> None:
        self.f_cand, self.f_fixed, self.rows, self.weights = f_cand, f_fixed, rows.copy(), weights
        self.refactor()

    def refactor(self) -> None:
        """Rebuild ``A``, ``d`` (and ``B``, ``b``) from the design's information matrix."""
        x = np.vstack([self.f_fixed, self.f_cand[self.rows]])
        self.m_inv = _regularised_inverse(x.T @ x)
        self.variance = _quadratic_forms(self.f_cand, self.m_inv)
        if self.weights is not None:
            self.b_mat = self.m_inv @ self.weights @ self.m_inv
            self.b_variance = _quadratic_forms(self.f_cand, self.b_mat)
        self.n_updates, self.peak = 0, float(self.variance.max())

    def terms(self, block: np.ndarray) -> _RowTerms:
        """Return the swap-gain terms of design rows ``block`` against every candidate, one matrix product each."""
        x = self.f_cand[self.rows[block]]
        x_a = x @ self.m_inv
        terms = _RowTerms(np.einsum("ij,ij->i", x_a, x), x_a @ self.f_cand.T)
        if self.weights is not None:
            x_b = x @ self.b_mat
            terms.b_i, terms.b_ij = np.einsum("ij,ij->i", x_b, x), x_b @ self.f_cand.T
        return terms

    def best_swaps(self, terms: _RowTerms, rows: slice) -> tuple[np.ndarray, np.ndarray]:
        """Return the best candidate for block ``rows`` and the gain of each swap (:meth:`Criterion.swap_gains`).

        For D the gain ``d(j) (1 - d(i)) + d(i, j)^2 - d(i)`` is ranked without its last
        term, in one temporary, where the general form takes several.
        """
        d_i = terms.d_i[rows]
        index = np.arange(len(d_i))
        if self.weights is None:
            score = np.square(terms.d_ij[rows])
            score += np.multiply.outer(1.0 - d_i, self.variance)
            best = np.argmax(score, axis=1)
            return best, score[index, best] - d_i
        gains = _trace_gain(
            (d_i[:, None], self.variance[None, :], terms.d_ij[rows]),
            (terms.b_i[rows, None], self.b_variance[None, :], terms.b_ij[rows]),  # type: ignore[index]
        )
        best = np.argmax(gains, axis=1)
        return best, gains[index, best]

    def _rank_one(
        self, v: np.ndarray, sign: float, f_u: np.ndarray | None = None, f_g: np.ndarray | None = None
    ) -> tuple:
        """Add (``sign = 1``) or remove (``sign = -1``) the model row ``v``: ``A <- A - s (A v)(A v)'``.

        ``f_u = F A v`` and ``f_g = F B v`` are computed unless given. Returns the step
        ``(A v, B v, s, v' B v, f_u, f_g)``, from which the products of another row with
        the updated ``A`` and ``B`` follow without touching ``F`` again (:meth:`_RowTerms.follow`).
        """
        u = self.m_inv @ v
        s = sign / (1.0 + sign * float(v @ u))
        f_u = self.f_cand @ u if f_u is None else f_u
        g, vbv = None, 0.0
        if self.weights is not None:  # B = A W A takes two cross terms and one square term
            g = self.b_mat @ v
            vbv = float(v @ g)
            f_g = self.f_cand @ g if f_g is None else f_g
            self.b_variance += s * (s * vbv * f_u**2 - 2.0 * f_u * f_g)
            cross = np.outer(u, g)
            self.b_mat += s * (s * vbv * np.outer(u, u) - cross - cross.T)
        self.variance -= s * f_u**2
        self.m_inv -= s * np.outer(u, u)
        return u, g, s, vbv, f_u, f_g

    def swap(self, i: int, j: int, terms: _RowTerms, r: int) -> list[tuple] | None:
        """Replace design row ``i`` by candidate ``j``; row ``r`` of ``terms`` holds its scored terms.

        Returns the two rank-one steps made, or None when the state was rebuilt instead
        (and other rows' terms must be scored afresh).
        """
        removed = self.f_cand[self.rows[i]]
        self.rows[i] = j
        self.n_updates += 1
        if self.n_updates >= _REFACTOR_EVERY:
            self.refactor()
            return None
        added = self._rank_one(self.f_cand[j], 1.0)
        u, g, s, vbv, f_u, f_g = added
        # The removed row's products with the updated A and B, from the ones it was scored with.
        u_a = float(u @ removed)
        f_a, f_ba = terms.d_ij[r] - s * u_a * f_u, None
        if terms.b_ij is not None and g is not None:
            f_ba = terms.b_ij[r] - s * (f_u * float(g @ removed) + f_g * u_a) + s * s * vbv * u_a * f_u
        steps = [added, self._rank_one(removed, -1.0, f_a, f_ba)]
        if self.variance.max() < _COLLAPSE * self.peak:
            self.refactor()
            return None
        return steps


def _climb_block(state: _ExchangeState, block: np.ndarray) -> bool:
    """Swap each row of ``block`` in turn for its best candidate, when that helps; return whether any was swapped.

    The block's terms come from one matrix product. A swap changes ``A``, and the terms
    of the rows after it follow by rank-one updates (:meth:`_RowTerms.follow`), so every
    row is scored against the current design without multiplying by the candidates
    again. Rows are scored a few at a time, one after a swap and twice as many after each
    group with none, so frequent swaps do not rescore rows that a later swap changes.
    """
    terms, r, size, swapped = state.terms(block), 0, 1, False
    while r < len(block):
        best, gain = state.best_swaps(terms, slice(r, r + size))
        better = np.flatnonzero(gain > 1e-9)  # also refuses NaN
        if not len(better):
            r, size = r + size, 2 * size
            continue
        r += int(better[0])
        steps = state.swap(int(block[r]), int(best[better[0]]), terms, r)
        r, size, swapped = r + 1, 1, True
        if r == len(block):
            break
        if steps is None:  # the state was rebuilt: score the rest afresh
            terms, block, r = state.terms(block[r:]), block[r:], 0
            continue
        rest = state.f_cand[state.rows[block[r:]]]
        for step in steps:
            terms.follow(rest, step, r)
    return swapped


def _row_exchange(f_cand: np.ndarray, f_fixed: np.ndarray, rows: np.ndarray, criterion: Criterion) -> np.ndarray:
    """Swap each design row for its best candidate in turn: the modified Fedorov exchange (Cook and Nachtsheim 1980).

    Each row is scored against every candidate and its best improving swap is made at
    once, where the full Fedorov step scored all ``n x N`` (run, candidate) pairs, and
    rebuilt ``X'X`` and its inverse, to make one swap. Passes repeat until none improves
    the criterion. Rows are scored in blocks of up to ``_BLOCK_ROWS`` rows and
    ``_BLOCK_ENTRIES`` terms (:func:`_climb_block`).
    """
    state = _ExchangeState(f_cand, f_fixed, rows, criterion.weights)
    size = int(np.clip(_BLOCK_ENTRIES // max(len(f_cand), 1), 1, _BLOCK_ROWS))
    for _ in range(_MAX_PASSES):
        improved = False
        for start in range(0, len(rows), size):
            improved |= _climb_block(state, np.arange(start, min(len(rows), start + size)))
        if not improved:
            break
    return state.rows


#: Exponent of Kiefer's ``phi_p``, the smooth stand-in for ``lambda_min`` that ranks E-optimal swaps.
_E_P = 8
#: Work allowed per E-optimal iteration for exact scoring, in (swaps x coefficients^2): all swaps when they fit.
_E_EXACT_WORK = 4_000_000


def _phi_p(eigenvalues: np.ndarray) -> np.ndarray:
    """Kiefer's ``phi_p = (sum lambda_k^-p)^(-1/p)`` along the last axis of ascending eigenvalues.

    It sits just below ``lambda_min`` and rises whenever any small eigenvalue rises,
    so it separates designs that tie on ``lambda_min``. Written through the ratios
    ``lambda_min / lambda_k <= 1`` so the powers cannot overflow.
    """
    floor = 1e-12 * max(float(np.max(eigenvalues)), 1.0)
    lam = np.maximum(eigenvalues, floor)
    ratio = lam[..., :1] / lam
    return lam[..., 0] * np.sum(ratio**_E_P, axis=-1) ** (-1.0 / _E_P)


def _exact_e_scores(
    info: np.ndarray, f_design: np.ndarray, f_cand: np.ndarray, pairs: tuple[np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """``lambda_min`` and ``phi_p`` after each swap in ``pairs`` (design rows, candidate rows), in chunks."""
    p = info.shape[0]
    chunk = max(1, 2_000_000 // p**2)  # about 16 MB of matrices at a time
    lam_min, phi = [], []
    for start in range(0, len(pairs[0]), chunk):
        a = f_design[pairs[0][start : start + chunk]]
        b = f_cand[pairs[1][start : start + chunk]]
        swapped = info[None] - a[:, :, None] * a[:, None, :] + b[:, :, None] * b[:, None, :]
        eigenvalues = np.linalg.eigvalsh(swapped)
        lam_min.append(eigenvalues[:, 0])
        phi.append(_phi_p(eigenvalues))
    return np.concatenate(lam_min), np.concatenate(phi)


def _best_e_swap(
    info: np.ndarray, f_design: np.ndarray, f_cand: np.ndarray, *, polish: bool = False
) -> tuple[int, int, float]:
    """Best swap for E-optimality: rank every swap by the gradient of ``phi_p``, then score the best exactly.

    The exchange runs in two phases. The first climbs ``phi_p`` itself, a smooth
    stand-in for ``lambda_min`` that can trade a little of the smallest eigenvalue for
    a lot of the next ones, and so escapes designs where no swap raises
    ``lambda_min`` alone. The second (``polish=True``) then takes only swaps that
    raise ``lambda_min``, or hold it and raise ``phi_p``. Each phase climbs one
    objective, so neither can cycle. Two things make ``lambda_min`` hard to climb one
    swap at a time:

    - **Ties.** On a +/-1 grid thousands of swaps share a first-order estimate, so a
      short list of "most promising" swaps can miss the one that helps. The list
      scored exactly is therefore large (every swap when ``_E_EXACT_WORK`` allows),
      and batched eigenvalue solves keep that cheap.
    - **Repeated eigenvalues.** If ``lambda_min`` has multiplicity 2 or more, no single
      swap can raise it: adding ``b b'`` gives ``lambda_1(M + b b') <= lambda_2(M)``
      (Weyl interlacing). A swap that keeps ``lambda_min`` and raises ``phi_p``
      splits the repeated eigenvalue, so a later swap can lift it.

    Swaps are ranked by the first-order change in ``phi_p``, which weights each
    eigen-direction by ``(lambda_min / lambda_k)^(p+1)``: the directions at or near the
    minimum count fully, the rest hardly at all.

    Returns
    -------
    tuple[int, int, float]
        Design row out, candidate row in, and the gain: in ``phi_p`` in the first phase;
        when polishing, in ``lambda_min`` when it rises, otherwise in ``phi_p`` with
        ``lambda_min`` held. A gain of 0 means no swap helps.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(info)
    current, current_phi = float(eigenvalues[0]), float(_phi_p(eigenvalues))
    weights = (max(current, 1e-12) / np.maximum(eigenvalues, 1e-12)) ** (_E_P + 1)
    added = (f_cand @ eigenvectors) ** 2 @ weights
    removed = (f_design @ eigenvectors) ** 2 @ weights
    estimate = added[None, :] - removed[:, None]

    n_exact = max(2000, _E_EXACT_WORK // info.shape[0] ** 2)
    order = np.argsort(estimate, axis=None)[::-1][:n_exact]
    rows_out, rows_in = np.unravel_index(order, estimate.shape)
    pairs = (rows_out, rows_in)
    lam_min, phi = _exact_e_scores(info, f_design, f_cand, pairs)

    phi_tol = 1e-9 * max(1.0, abs(current_phi))
    if not polish:
        best = int(np.argmax(phi))
        gain = float(phi[best] - current_phi)
        return (int(pairs[0][best]), int(pairs[1][best]), gain) if gain > phi_tol else (0, 0, 0.0)
    tol = 1e-9 * max(1.0, abs(current))
    if (lam_min > current + tol).any():
        best = int(np.argmax(np.where(lam_min > current + tol, lam_min + 1e-12 * phi, -np.inf)))
        return int(pairs[0][best]), int(pairs[1][best]), float(lam_min[best] - current)
    holds = lam_min >= current - tol
    if holds.any():
        best = int(np.argmax(np.where(holds, phi, -np.inf)))
        gain = float(phi[best] - current_phi)
        if gain > phi_tol:
            return int(pairs[0][best]), int(pairs[1][best]), gain
    return 0, 0, 0.0


def _spectral_or_minimax_value(criterion: Criterion, info: np.ndarray) -> float:
    """``lambda_min`` (E), ``-log(lambda_max / lambda_min)`` (K), or minus the largest prediction variance (G)."""
    if criterion.name == "g_optimal":
        if np.linalg.matrix_rank(info) < info.shape[0]:
            return -np.inf
        return -float(_prediction_variance(criterion.region_rows, np.linalg.inv(info)).max())  # type: ignore[arg-type]
    eigenvalues = np.linalg.eigvalsh(info)
    if criterion.name == "e_optimal":
        return float(eigenvalues[0])
    if eigenvalues[0] <= 1e-12 * max(float(eigenvalues[-1]), 1e-300):
        return -np.inf
    return -float(np.log(eigenvalues[-1] / eigenvalues[0]))


def _prediction_variance(rows: np.ndarray, m_inv: np.ndarray) -> np.ndarray:
    """``f(x)' M^-1 f(x)`` for every model row ``f(x)`` in ``rows``."""
    return np.einsum("ij,jk,ik->i", rows, m_inv, rows)


def _swap_eigenvalues(info: np.ndarray, f_design: np.ndarray, f_cand: np.ndarray, pairs: tuple) -> np.ndarray:
    """Ascending eigenvalues of ``M - a a' + b b'`` for each swap in ``pairs`` (design rows, candidate rows)."""
    p = info.shape[0]
    chunk = max(1, 2_000_000 // p**2)
    out = []
    for start in range(0, len(pairs[0]), chunk):
        a = f_design[pairs[0][start : start + chunk]]
        b = f_cand[pairs[1][start : start + chunk]]
        out.append(np.linalg.eigvalsh(info[None] - a[:, :, None] * a[:, None, :] + b[:, :, None] * b[:, None, :]))
    return np.concatenate(out)


def _best_k_swap(info: np.ndarray, f_design: np.ndarray, f_cand: np.ndarray) -> tuple[int, int, float]:
    """Best swap for K-optimality (smallest condition number of ``X'X``).

    Every swap is ranked by the first-order change in ``log(lambda_min) - log(lambda_max)``,
    ``((v' b)^2 - (v' a)^2) / lambda`` along the extreme eigenvectors ``v``, and the most
    promising ones are scored exactly with batched eigenvalue solves, as for E-optimality.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(info)
    lam_min, lam_max = max(float(eigenvalues[0]), 1e-12), float(eigenvalues[-1])
    current = -np.log(lam_max / lam_min)

    def pull(rows: np.ndarray) -> np.ndarray:
        return (rows @ eigenvectors[:, 0]) ** 2 / lam_min - (rows @ eigenvectors[:, -1]) ** 2 / lam_max

    estimate = pull(f_cand)[None, :] - pull(f_design)[:, None]
    n_exact = max(2000, _E_EXACT_WORK // info.shape[0] ** 2)
    order = np.argsort(estimate, axis=None)[::-1][:n_exact]
    pairs = np.unravel_index(order, estimate.shape)
    eig = _swap_eigenvalues(info, f_design, f_cand, pairs)
    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.where(eig[:, 0] > 1e-12 * eig[:, -1], -np.log(eig[:, -1] / eig[:, 0]), -np.inf)
    best = int(np.argmax(score))
    gain = float(score[best] - current)
    return (int(pairs[0][best]), int(pairs[1][best]), gain) if gain > 1e-12 else (0, 0, 0.0)


#: Reweighting rounds of the I-lambda search for G-optimality, and the size of its final polish.
_G_ROUNDS = 12
_G_POLISH_CANDIDATES = 400
_G_POLISH_PASSES = 20


def _g_polish(f_cand: np.ndarray, f_fixed: np.ndarray, rows: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Exact exchange on the largest prediction variance over ``region``: take the best single swap while it helps.

    With ``A = (M - a a')^-1`` after removing design row ``a``, adding candidate ``b`` gives
    ``d(r) = r' A r - (r' A b)^2 / (1 + b' A b)`` at every region row ``r`` at once.
    """
    rows = rows.copy()
    for _ in range(_G_POLISH_PASSES):
        x = np.vstack([f_fixed, f_cand[rows]])
        m_inv = np.linalg.inv(x.T @ x)
        current = float(_prediction_variance(region, m_inv).max())
        pool = np.arange(len(f_cand))
        if len(pool) > _G_POLISH_CANDIDATES:  # the runs that help most sit where the variance is largest
            pool = np.argsort(_prediction_variance(f_cand, m_inv))[::-1][:_G_POLISH_CANDIDATES]
        best = (current - 1e-9 * current, -1, -1)
        for i, row in enumerate(rows):
            a = f_cand[row]
            denominator = 1.0 - a @ m_inv @ a
            if denominator <= 1e-10:
                continue
            m_inv_a = m_inv @ a
            removed = m_inv + np.outer(m_inv_a, m_inv_a) / denominator
            region_removed = region @ removed
            base = np.einsum("ij,ij->i", region_removed, region)
            b = f_cand[pool]
            cross = region_removed @ b.T
            scale = 1.0 + np.einsum("ij,jk,ik->i", b, removed, b)
            worst = (base[:, None] - cross**2 / scale[None, :]).max(axis=0)
            j = int(np.argmin(worst))
            if worst[j] < best[0]:
                best = (float(worst[j]), i, int(pool[j]))
        if best[1] < 0:
            break
        rows[best[1]] = best[2]
    return rows


def _g_exchange(
    f_cand: np.ndarray, n_free: int, f_fixed: np.ndarray, rng: np.random.Generator, criterion: Criterion
) -> tuple[np.ndarray, float]:
    """G-optimal design by I-lambda optimality (Hernandez and Nachtsheim 2018), then an exact polish.

    The largest prediction variance over the region is hard to lower one swap at a time,
    so each round runs the closed-form trace exchange for a weighted average of the
    variance (I-lambda), and Lawson's rule then moves weight towards the region points
    where the variance is largest (``w <- w * d``). The rounds converge on the minimax
    design; the best design met is kept and polished by exact swaps on the maximum.
    """
    region = criterion.region_rows
    if region is None:
        raise ValueError("G-optimality needs the region's model rows.")
    best_rows, best_value = np.empty(0, dtype=int), -np.inf
    for _ in range(_N_STARTS):
        rows = _greedy_start(f_cand, f_fixed, n_free, rng)
        weights = np.full(len(region), 1.0 / len(region))
        start_rows, start_value = rows.copy(), -np.inf
        for _ in range(_G_ROUNDS):
            rows = _climb(f_cand, f_fixed, rows, Criterion("i_optimal", (region * weights[:, None]).T @ region))
            x = np.vstack([f_fixed, f_cand[rows]])
            if np.linalg.matrix_rank(x) < x.shape[1]:
                break
            variance = _prediction_variance(region, np.linalg.inv(x.T @ x))
            if -variance.max() > start_value:
                start_rows, start_value = rows.copy(), -float(variance.max())
            weights = weights * variance
            weights /= weights.sum()
        if np.isfinite(start_value):
            start_rows = _g_polish(f_cand, f_fixed, start_rows, region)
        x = np.vstack([f_fixed, f_cand[start_rows]])
        value = criterion.value(x.T @ x)
        if value > best_value or len(best_rows) == 0:
            best_rows, best_value = start_rows, value
    return best_rows, best_value


def criterion_metadata(criterion: Criterion, value: float) -> dict[str, float]:
    """Report the criterion value under a name that says what it is."""
    if criterion.name == "d_optimal":
        return {"log_det_information": value}
    if criterion.name == "e_optimal":
        return {"min_eigenvalue": value}
    if criterion.name == "g_optimal":
        return {"max_prediction_variance": -value}
    if criterion.name == "k_optimal":
        return {"condition_number": float(np.exp(-value))}
    # trace((X'X)^-1 W): the summed coefficient variance (A), or the average
    # prediction variance over the region in units of sigma^2 (I).
    return {"trace_criterion": -value}


def _n_starts(criterion: Criterion, f_cand: np.ndarray, n_free: int) -> int:
    """Random starts for the exchange: more for a small problem, so long as their total work stays near ``_START_WORK``.

    The work of one start grows with candidates x coefficients x free runs. E and K
    score every swap at each step and keep ``_N_STARTS``.
    """
    if criterion.name not in _CLOSED_FORM_CRITERIA:
        return _N_STARTS
    work = f_cand.shape[0] * f_cand.shape[1] * max(n_free, 1)
    return int(np.clip(_START_WORK // work, _N_STARTS, _MAX_STARTS))


def fedorov_exchange(
    f_cand: np.ndarray,
    n_free: int,
    f_fixed: np.ndarray,
    rng: np.random.Generator,
    criterion: Criterion | None = None,
) -> tuple[np.ndarray, float]:
    """Choose ``n_free`` candidate rows that maximise ``criterion`` (D-optimality by default).

    For D, A and I each design run in turn is swapped for the candidate that improves
    the criterion most (:func:`_row_exchange`, the modified Fedorov exchange of Cook and
    Nachtsheim 1980), with the gains in the closed form of :meth:`Criterion.swap_gains`
    (Fedorov 1972). E and K make the best single swap over all pairs at each step. Rows in
    ``f_fixed`` stay in the design and are never swapped out. Candidates may repeat,
    which gives replicated runs where the criterion wants them. The best of several
    random starts is kept: ``_N_STARTS`` for E and K, and up to ``_MAX_STARTS`` for a
    small D, A or I problem (see :func:`_n_starts`).

    Returns
    -------
    tuple[np.ndarray, float]
        Selected candidate indices and the criterion value of the full design (see
        :meth:`Criterion.value`). G-optimality runs its own search, :func:`_g_exchange`.
    """
    criterion = criterion if criterion is not None else Criterion.d()
    if criterion.name == "g_optimal":
        return _g_exchange(f_cand, n_free, f_fixed, rng, criterion)
    best_rows, best_value = np.empty(0, dtype=int), -np.inf
    for rows in _climbed_starts(f_cand, n_free, f_fixed, rng, criterion):
        x = np.vstack([f_fixed, f_cand[rows]])
        value = criterion.value(x.T @ x)
        if value > best_value or len(best_rows) == 0:  # keep a design even if every start is singular
            best_rows, best_value = rows.copy(), value
    return best_rows, best_value


def _climbed_starts(
    f_cand: np.ndarray, n_free: int, f_fixed: np.ndarray, rng: np.random.Generator, criterion: Criterion
) -> Iterator[np.ndarray]:
    """Each random start of the exchange (see :func:`_n_starts`), climbed to a local optimum of ``criterion``."""
    for _ in range(_n_starts(criterion, f_cand, n_free)):
        yield _climb(f_cand, f_fixed, _greedy_start(f_cand, f_fixed, n_free, rng), criterion)


def _climb(f_cand: np.ndarray, f_fixed: np.ndarray, rows: np.ndarray, criterion: Criterion) -> np.ndarray:
    """Improve ``rows`` by swaps while ``criterion`` improves; return the rows.

    D, A and I (whose swaps have a closed form) use the row-wise exchange,
    :func:`_row_exchange`. E and K make the best single swap at each step, in each of
    their phases.
    """
    if criterion.name in _CLOSED_FORM_CRITERIA:
        return _row_exchange(f_cand, f_fixed, rows, criterion)
    rows = rows.copy()
    for phase in range(criterion.n_phases):
        for _ in range(_MAX_EXCHANGES):
            x = np.vstack([f_fixed, f_cand[rows]])
            i, j, gain = criterion.best_swap(x.T @ x, f_cand[rows], f_cand, phase)
            if not gain > 1e-9:  # also stops on NaN
                break
            rows[i] = j
    return rows


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


@dataclass
class ConstrainedOptions:
    """Optional settings for :func:`constrained_optimal_design` and the constrained mixture design.

    Parameters
    ----------
    model_type : str
        ``"main_effects"``, ``"interactions"`` or ``"quadratic"`` (a mixture design maps
        these to Scheffé models).
    criterion : str
        ``"d_optimal"`` (default), ``"i_optimal"``, ``"a_optimal"``, ``"e_optimal"``,
        ``"g_optimal"`` or ``"k_optimal"``.
    fixed_runs : pandas.DataFrame or None
        Runs kept in the design (continuous in coded units, categorical as labels),
        already validated by the caller. They count towards the budget.
    n_levels : int or None
        Grid levels per continuous factor. ``None`` picks the finest of 5, 4 and 3
        (and 2, for a model without squared terms; 4 is skipped for a quadratic model
        unless the criterion is I or G) that keeps the grid within
        ``max(5000, 200 p)`` points for ``p`` model coefficients, and samples the
        coarsest grid down to that when none does.
    candidates : pandas.DataFrame or None
        Settings the runs must be chosen from, in actual units (proportions for a
        mixture), one column per factor, instead of a generated grid. Rows that
        break a constraint are dropped, and the I-optimality average is taken over
        the remaining rows.
    """

    model_type: str = "interactions"
    criterion: str = "d_optimal"
    fixed_runs: pd.DataFrame | None = None
    n_levels: int | None = None
    candidates: pd.DataFrame | None = None


def _fixed_rows(region: _Region, fixed_runs: pd.DataFrame | None, model_type: str, n_columns: int) -> np.ndarray:
    """Model rows for the fixed runs (none when ``fixed_runs`` is None), warning about any outside the region."""
    if fixed_runs is None:
        return np.empty((0, n_columns))
    fixed_coded = fixed_runs[[f.name for f in region.continuous]].to_numpy(dtype=float)
    labels = {f.name: [str(lv) for lv in f.levels or []] for f in region.categorical}
    fixed_cats = np.array(
        [[labels[f.name].index(str(v)) for v in fixed_runs[f.name]] for f in region.categorical], dtype=int
    ).T.reshape(len(fixed_runs), len(region.categorical))
    n_outside = int((region.slack(fixed_coded) > _FEASIBILITY_TOL).sum())
    if n_outside:
        logger.warning("%d fixed run(s) lie outside the constrained region; they are kept as given.", n_outside)
    return model_matrix(region, fixed_coded, fixed_cats, model_type)


def _fixed_run_metadata(region: _Region, fixed_runs: pd.DataFrame, requested_budget: int, budget: int) -> dict:
    """Metadata on the fixed runs: how many, how many lie outside the region, and any budget they raised."""
    meta = {"n_fixed_runs": len(fixed_runs)}
    coded = fixed_runs[[f.name for f in region.continuous]].to_numpy(dtype=float)
    n_outside = int((region.slack(coded) > _FEASIBILITY_TOL).sum())
    if n_outside:
        meta["n_fixed_runs_outside_region"] = n_outside
    if budget != requested_budget:
        meta["budget_requested"] = requested_budget
    return meta


def _user_candidates(region: _Region, candidates: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict, list]:
    """Code a user's candidate settings, drop the infeasible and repeated ones.

    Returns
    -------
    tuple
        Coded continuous values, categorical level indices, counts for the
        metadata, and the ``candidates`` index label of each kept row.
    """
    names = [f.name for f in region.continuous] + [f.name for f in region.categorical]
    missing = [n for n in names if n not in candidates.columns]
    if missing:
        raise ValueError(f"candidates is missing columns for factors: {missing}.")
    actual = candidates[[f.name for f in region.continuous]].apply(pd.to_numeric, errors="coerce")
    if actual.isna().any().any():
        raise ValueError("candidates has missing or non-numeric values in a continuous factor column.")
    low = np.array([f.low for f in region.continuous], dtype=float)
    high = np.array([f.high for f in region.continuous], dtype=float)
    coded = (actual.to_numpy(dtype=float) - (low + high) / 2.0) / ((high - low) / 2.0)
    cats = np.zeros((len(candidates), len(region.categorical)), dtype=int)
    for j, f in enumerate(region.categorical):
        labels = [str(lv) for lv in f.levels or []]
        values = candidates[f.name].astype(str)
        unknown = sorted(set(values) - set(labels))
        if unknown:
            raise ValueError(f"candidates has unknown levels {unknown} for categorical factor {f.name!r}.")
        cats[:, j] = [labels.index(v) for v in values]
    if (np.abs(coded) > 1.0 + 1e-9).any():
        logger.warning("Some candidates lie outside the factors' low/high range; they are kept (coded beyond +/-1).")

    feasible = region.slack(coded) <= _FEASIBILITY_TOL
    _, first = np.unique(np.hstack([coded.round(9), cats]), axis=0, return_index=True)
    keep = np.zeros(len(coded), dtype=bool)
    keep[first] = True
    keep &= feasible
    counts = {
        "n_candidates_supplied": len(candidates),
        "n_candidates_infeasible": int((~feasible).sum()),
        "n_candidates": int(keep.sum()),
    }
    return coded[keep], cats[keep], counts, list(candidates.index[keep])


def _candidate_pool(
    region: _Region, opts: ConstrainedOptions, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, dict, list | None, Callable[[], np.ndarray]]:
    """Return the candidates (a generated grid, or the user's), and the rows I-optimality averages over."""
    if opts.candidates is not None:
        coded, cats, counts, labels = _user_candidates(region, opts.candidates)
        return coded, cats, counts, labels, lambda: model_matrix(region, coded, cats, opts.model_type)
    n_levels = opts.n_levels
    corners_suffice = opts.criterion == "d_optimal" and opts.model_type != "quadratic" and not region.inequalities
    if n_levels is None and corners_suffice:
        # Without squared terms each model row is affine in every single coordinate, so det(X'X) is a convex
        # quadratic in it and is largest at -1 or +1: some exact D-optimal design uses only the box's corners.
        n_levels = 2
    coded, cats, counts = build_candidates(region, n_levels, opts.model_type, opts.criterion)
    return coded, cats, counts, None, lambda: _uniform_rows(region, opts.model_type, rng, coded)


def selection_counts(labels: list, rows: np.ndarray) -> dict[str, int]:
    """Count how often each supplied candidate (by index label) was chosen."""
    chosen = pd.Series([labels[r] for r in rows]).value_counts(sort=False)
    return {str(label): int(n) for label, n in chosen.items()}


def _budget_for_fixed_runs(f_fixed: np.ndarray, n_parameters: int, budget: int) -> int:
    """Raise ``budget`` so the free runs can supply the rank the fixed runs lack, warning when it does.

    Fixed runs that repeat a point (three centre runs, say) carry less information than
    their count: the floor on the budget counts runs, not rank, so it can leave too few
    free runs to estimate the model at all.
    """
    n_fixed = len(f_fixed)
    missing_rank = n_parameters - (np.linalg.matrix_rank(f_fixed) if n_fixed else 0)
    if budget - n_fixed >= missing_rank:
        return budget
    logger.warning(
        "The %d fixed run(s) span only %d of the %d model coefficients, so at least %d more run(s) are "
        "needed; raising the budget from %d to %d.",
        n_fixed,
        n_parameters - missing_rank,
        n_parameters,
        missing_rank,
        budget,
        n_fixed + missing_rank,
    )
    return n_fixed + missing_rank


#: Uniform draws used to estimate the region's moment matrix for I-optimality.
_N_MOMENT_SAMPLES = 20_000


def _uniform_rows(region: _Region, model_type: str, rng: np.random.Generator, seeds: np.ndarray) -> np.ndarray:
    """Model rows at points drawn uniformly from the feasible region.

    Thin regions are handled by hit-and-run started from the feasible candidates
    (``seeds``), so the moment matrix covers the whole region, not the few points a
    rejection pass happens to hit.
    """
    k = len(region.continuous)

    def coded_inequality(g: Callable) -> Callable[[np.ndarray], np.ndarray]:
        return lambda x: np.broadcast_to(np.asarray(g(region.actual(x)), dtype=float), (len(x),))

    sampler = UniformSampler(
        [coded_inequality(g) for g in region.inequalities],
        (np.full(k, -1.0), np.full(k, 1.0)),
        lambda m: rng.uniform(-1.0, 1.0, size=(m, k)),
        seeds=lambda: seeds,
    )
    coded = sampler.draw(_N_MOMENT_SAMPLES, rng)
    cats = np.empty((len(coded), 0), dtype=int)
    if region.categorical:
        cats = np.column_stack([rng.integers(len(f.levels or []), size=len(coded)) for f in region.categorical])
    return model_matrix(region, coded, cats, model_type)


#: Uniform draws added to the candidates for the G-criterion's maximum (the candidates hold the boundary points).
_N_G_REGION_SAMPLES = 3000


def make_criterion(
    name: str,
    n_parameters: int,
    region_rows: Callable[[], np.ndarray],
    candidate_rows: np.ndarray | None = None,
) -> Criterion:
    """Build the :class:`Criterion` called ``name``.

    ``region_rows`` (model rows drawn uniformly in the region) is only evaluated for
    I- and G-optimality. G takes its maximum over those rows plus ``candidate_rows``,
    which hold the vertices and boundary points where the worst case usually sits.
    """
    if name == "d_optimal":
        return Criterion.d()
    if name == "a_optimal":
        return Criterion.a(n_parameters)
    if name == "i_optimal":
        return Criterion.i(region_rows())
    if name == "e_optimal":
        return Criterion.e()
    if name == "k_optimal":
        return Criterion.k()
    if name == "g_optimal":
        sample = region_rows()[:_N_G_REGION_SAMPLES]
        rows = sample if candidate_rows is None else np.vstack([candidate_rows, sample])
        return Criterion.g(np.unique(rows.round(12), axis=0))
    raise ValueError(
        f"Unknown criterion {name!r}; choose 'd_optimal', 'i_optimal', 'a_optimal', 'e_optimal', 'g_optimal' or "
        "'k_optimal'."
    )


def _check_factors_and_model(factors: list[Factor], model_type: str) -> None:
    """Refuse mixture components and a model type the model matrix does not build."""
    if model_type not in _MODEL_TYPES:
        raise ValueError(f"model_type={model_type!r} is not supported; choose from {', '.join(_MODEL_TYPES)}.")
    if any(f.type == FactorType.mixture for f in factors):
        raise ValueError(
            "Mixture components need the mixture engine: give only mixture factors, and generate_design routes "
            "them there. Mixture-process designs are not supported."
        )


def constrained_optimal_design(
    factors: list[Factor],
    budget: int,
    constraints: list[Constraint],
    options: ConstrainedOptions | None = None,
    random_state: int | np.random.Generator | None = None,
) -> tuple[np.ndarray, dict]:
    """Generate a D-, I-, A- or E-optimal design from a candidate set, every chosen run satisfying ``constraints``.

    This is also the optimal-design backend when pyoptex is not installed, with
    ``constraints`` empty. Runs in ``options.fixed_runs`` are kept as given, even
    outside the region (with a warning); ``metadata["n_fixed_runs_outside_region"]``
    counts them.

    Parameters
    ----------
    factors : list[Factor]
        Continuous and categorical factors. Mixture factors are not supported here.
    budget : int
        Total number of runs, including ``fixed_runs``. When the fixed runs leave too
        few free runs to estimate the model, it is raised (with a warning) and the value
        asked for is recorded in ``metadata["budget_requested"]``.
    constraints : list[Constraint]
        Inequalities in actual units over the continuous factors, e.g.
        ``Constraint(expression="3*T + 5*D <= 600")``. May be empty.
    options : ConstrainedOptions or None
        Model type, criterion, fixed runs and grid resolution; defaults to a
        D-optimal design for an interactions model, no fixed runs and an automatic grid.
    random_state : int, numpy.random.Generator or None
        Seed for the random starts of the exchange.

    Returns
    -------
    tuple[np.ndarray, dict]
        The design in coded units (categorical factors as labels) and metadata.

    Raises
    ------
    ValueError
        If ``model_type`` is unknown, a constraint cannot be parsed, no candidate point
        is feasible, or the feasible region cannot support the requested model.
    """
    rng = check_random_state(random_state)
    opts = options if options is not None else ConstrainedOptions()
    model_type, fixed_runs = opts.model_type, opts.fixed_runs
    requested_budget = budget
    _check_factors_and_model(factors, model_type)

    continuous = [f for f in factors if f.type != FactorType.categorical]
    categorical = [f for f in factors if f.type == FactorType.categorical]
    names = {f.name for f in continuous}
    inequalities = [g for c in constraints for g in parse_constraint(c.expression, names)]
    region = _Region(continuous, categorical, inequalities)

    coded, cats, counts, labels, region_rows = _candidate_pool(region, opts, rng)
    if counts["n_candidates"] == 0:
        raise ValueError("No candidate point satisfies all the constraints; check them for conflicts.")

    f_cand = model_matrix(region, coded, cats, model_type)
    f_fixed = _fixed_rows(region, fixed_runs, model_type, f_cand.shape[1])
    n_fixed = len(f_fixed)

    if np.linalg.matrix_rank(np.vstack([f_fixed, f_cand])) < f_cand.shape[1]:
        raise ValueError(
            f"The feasible region ({counts['n_candidates']} candidate points) cannot support a "
            f"'{model_type}' model with {f_cand.shape[1]} coefficients. Use a simpler model, supply "
            "candidates, or loosen the constraints."
        )

    p = f_cand.shape[1]
    budget = _budget_for_fixed_runs(f_fixed, p, budget)
    criterion = make_criterion(opts.criterion, p, region_rows, f_cand)
    rows, value = fedorov_exchange(f_cand, budget - n_fixed, f_fixed, rng, criterion)
    if len(rows) != budget - n_fixed or np.linalg.matrix_rank(np.vstack([f_fixed, f_cand[rows]])) < p:
        raise ValueError(
            f"No design of {budget} runs from these candidates can estimate the '{model_type}' model "
            f"({p} coefficients). Use a simpler model or more distinct candidate points."
        )

    design = pd.DataFrame(coded[rows], columns=[f.name for f in continuous])
    for j, f in enumerate(categorical):
        design[f.name] = np.asarray(f.levels, dtype=object)[cats[rows, j]]
    if fixed_runs is not None:
        design = pd.concat([fixed_runs[design.columns].reset_index(drop=True), design], ignore_index=True)
    design = design[[f.name for f in factors]]

    meta = {
        "backend": "candidate_exchange",
        "optimality_criterion": opts.criterion,
        "model_type": model_type,
        **counts,
    }
    if constraints:
        meta["constraints"] = [c.expression for c in constraints]
        meta["constraints_enforced"] = True
    meta.update(criterion_metadata(criterion, value))
    if fixed_runs is not None:
        meta.update(_fixed_run_metadata(region, fixed_runs, requested_budget, budget))
    if labels is not None:
        meta["candidate_source"] = "user"
        meta["selected_candidates"] = selection_counts(labels, rows)
    values = design.to_numpy() if categorical else design.to_numpy(dtype=float)
    return values, meta
