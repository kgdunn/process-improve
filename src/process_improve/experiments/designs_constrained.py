# (c) Kevin Dunn, 2010-2026. MIT License.

"""D-optimal designs over a constrained factor region.

A ``Constraint`` such as ``"3*T + 5*D <= 600"`` cuts a corner off the factor box.
Classical designs cannot follow that cut, so the design is chosen from a set of
feasible *candidate* points instead, in three steps:

1. **Candidate set.** A grid over the box, plus the points where each constraint
   boundary crosses a grid line (found by bisection), minus every infeasible point.
   The boundary points matter: a D-optimal design pushes its runs to the edge of
   the region, and without them the nearest grid point can sit well inside it.
2. **Model matrix.** Each candidate is expanded into the columns of the model the
   design is for (intercept, main effects, two-factor interactions, squares).
3. **Exchange.** A Fedorov exchange swaps design runs for candidates while the
   determinant of the information matrix ``X'X`` grows, with several random starts.

Constraint expressions are parsed into a small arithmetic tree and evaluated with
numpy. Nothing is passed to ``eval``, so an expression from an untrusted caller can
only compute a number.
"""

from __future__ import annotations

import ast
import logging
from collections.abc import Callable, Mapping
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

#: Upper limit on the candidate grid; a larger grid is refused before allocation.
MAX_CANDIDATES = 100_000
#: Points taken from a grid too large to list (a power of two, for the Sobol sequence): the exchange's cost grows
#: with this.
_SAMPLED_CANDIDATES = 2**15
#: Longest constraint expression accepted, which also bounds the parse depth.
MAX_EXPRESSION_LENGTH = 500
#: Slack when testing ``g(x) <= 0``, so points found on a boundary are kept.
_FEASIBILITY_TOL = 1e-9
_BISECTION_STEPS = 50
_MAX_EXCHANGES = 500
_N_STARTS = 5

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


def _grid_levels(region: _Region, n_levels: int | None, model_type: str) -> tuple[int, bool]:
    """Pick the continuous grid resolution, and whether the grid is too large to list in full.

    The finest of 5, 4 and 3 levels whose full grid stays under ``MAX_CANDIDATES``
    points is used; models without squared terms may also drop to 2 levels, which is
    all they need. When even the coarsest grid is too large (11 or more factors at 3
    levels, 17 or more at 2), the grid is sampled instead (see :func:`_lattice_points`).
    """
    n_cat = int(np.prod([len(f.levels or []) for f in region.categorical]))
    coarsest = 3 if model_type == "quadratic" else 2
    choices = [n_levels] if n_levels is not None else list(range(5, coarsest - 1, -1))
    for n in choices:
        if n < 2:
            raise ValueError("n_levels must be at least 2.")
        if n ** len(region.continuous) * n_cat <= MAX_CANDIDATES:
            return n, False
    return choices[-1], True


def _lattice_points(shape: list[int], k_cont: int, sampled: bool) -> np.ndarray:
    """Index vectors of the candidate grid: all of it, or ``_SAMPLED_CANDIDATES`` well-spread points of it.

    The sample is the unscrambled Sobol sequence rounded down onto the grid: evenly
    spread over the grid, and deterministic without drawing on any random generator.
    A sampled grid on an odd number of levels also holds the centre and the face
    centres (one factor at an extreme, the rest at the middle), the points a quadratic
    model leans on and a random sample would rarely contain.
    """
    if not sampled:
        return np.indices(shape).reshape(len(shape), -1).T
    unit = qmc.Sobol(d=len(shape), scramble=False).random_base2(int(np.log2(_SAMPLED_CANDIDATES)))
    idx = np.minimum((unit * shape).astype(int), np.array(shape) - 1)
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


def build_candidates(
    region: _Region, n_levels: int | None = None, model_type: str = "quadratic"
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return the feasible candidate points in coded units.

    Parameters
    ----------
    region : _Region
        Factors and constraints.
    n_levels : int or None
        Grid levels per continuous factor; ``None`` picks them (see :func:`_grid_levels`).
    model_type : str
        The model the design is for: without squared terms, 2 levels may be used.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, dict]
        Coded continuous values ``(n, k_cont)``, categorical level indices
        ``(n, k_cat)``, and counts for the metadata.
    """
    levels, sampled = _grid_levels(region, n_levels, model_type)
    shape = [levels] * len(region.continuous) + [len(f.levels or []) for f in region.categorical]
    k_cont = len(region.continuous)
    idx = _lattice_points(shape, k_cont, sampled)
    if sampled:
        logger.info(
            "The %d-level grid on %d factors is too large to list; sampling %d points.", levels, k_cont, len(idx)
        )
    grid = np.linspace(-1.0, 1.0, levels)[idx[:, :k_cont]]
    cat_idx = idx[:, k_cont:]

    extra, extra_cat = _boundary_points(region, grid, cat_idx, step=2.0 / (levels - 1))
    coded = np.vstack([grid, extra])
    cats = np.vstack([cat_idx, extra_cat])

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


def _greedy_start(f_cand: np.ndarray, f_fixed: np.ndarray, n_free: int, rng: np.random.Generator) -> np.ndarray:
    """Build a non-singular starting design by adding the highest-variance candidate each step."""
    p = f_cand.shape[1]
    info = f_fixed.T @ f_fixed + 1e-6 * np.eye(p)  # small ridge so the first steps are defined
    rows = [int(rng.integers(f_cand.shape[0]))]
    info += np.outer(f_cand[rows[0]], f_cand[rows[0]])
    for _ in range(n_free - 1):
        variance = np.einsum("ij,ij->i", f_cand @ np.linalg.inv(info), f_cand)
        best = int(np.argmax(variance))
        rows.append(best)
        info += np.outer(f_cand[best], f_cand[best])
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

    def value(self, info: np.ndarray) -> float:
        """Score an information matrix; higher is better (``log det``, ``-trace(M^-1 W)``, or ``lambda_min``)."""
        if self.name == "e_optimal":
            return float(np.linalg.eigvalsh(info)[0])
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
            return d_j - d_i - (d_i * d_j - d_ij**2)
        b_design, b_cand = a_design @ self.weights, a_cand @ self.weights
        b_i = np.einsum("ij,ij->i", b_design, a_design)[:, None]
        b_j = np.einsum("ij,ij->i", b_cand, a_cand)[None, :]
        b_ij = b_design @ a_cand.T
        s = 1.0 + d_j
        denominator = 1.0 - d_i + d_ij**2 / s
        numerator = b_i - 2.0 * d_ij * b_ij / s + d_ij**2 * b_j / s**2
        with np.errstate(divide="ignore", invalid="ignore"):
            gain = b_j / s - numerator / denominator
        return np.where(denominator > 1e-10, gain, -np.inf)


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


def criterion_metadata(criterion: Criterion, value: float) -> dict[str, float]:
    """Report the criterion value under a name that says what it is."""
    if criterion.name == "d_optimal":
        return {"log_det_information": value}
    if criterion.name == "e_optimal":
        return {"min_eigenvalue": value}
    # trace((X'X)^-1 W): the summed coefficient variance (A), or the average
    # prediction variance over the region in units of sigma^2 (I).
    return {"trace_criterion": -value}


def fedorov_exchange(
    f_cand: np.ndarray,
    n_free: int,
    f_fixed: np.ndarray,
    rng: np.random.Generator,
    criterion: Criterion | None = None,
) -> tuple[np.ndarray, float]:
    """Choose ``n_free`` candidate rows that maximise ``criterion`` (D-optimality by default).

    Each iteration makes the single swap (design run ``i`` out, candidate ``j`` in)
    that improves the criterion most, scored for all pairs at once by
    :meth:`Criterion.swap_gains` (Fedorov 1972; Cook and Nachtsheim 1980). Rows in
    ``f_fixed`` stay in the design and are never swapped out. Candidates may repeat,
    which gives replicated runs where the criterion wants them. The best of
    ``_N_STARTS`` random starts is kept.

    Returns
    -------
    tuple[np.ndarray, float]
        Selected candidate indices and the criterion value of the full design:
        ``log det(X'X)`` for D, ``-trace((X'X)^-1 W)`` for A and I.
    """
    criterion = criterion if criterion is not None else Criterion.d()
    best_rows, best_value = np.empty(0, dtype=int), -np.inf
    for _ in range(_N_STARTS):
        rows = _greedy_start(f_cand, f_fixed, n_free, rng)
        for phase in range(criterion.n_phases):
            for _ in range(_MAX_EXCHANGES):
                x = np.vstack([f_fixed, f_cand[rows]])
                i, j, gain = criterion.best_swap(x.T @ x, f_cand[rows], f_cand, phase)
                if not gain > 1e-9:  # also stops on NaN
                    break
                rows[i] = j
        x = np.vstack([f_fixed, f_cand[rows]])
        value = criterion.value(x.T @ x)
        if value > best_value or len(best_rows) == 0:  # keep a design even if every start is singular
            best_rows, best_value = rows.copy(), value
    return best_rows, best_value


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
        ``"d_optimal"`` (default), ``"i_optimal"``, ``"a_optimal"`` or ``"e_optimal"``.
    fixed_runs : pandas.DataFrame or None
        Runs kept in the design (continuous in coded units, categorical as labels),
        already validated by the caller. They count towards the budget.
    n_levels : int or None
        Grid levels per continuous factor. ``None`` picks the finest of 5, 4 and 3
        (and 2, for a model without squared terms) that keeps the grid under
        ``MAX_CANDIDATES`` points, and samples the coarsest grid when none does.
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
    coded, cats, counts = build_candidates(region, opts.n_levels, opts.model_type)
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


def make_criterion(name: str, n_parameters: int, region_rows: Callable[[], np.ndarray]) -> Criterion:
    """Build the :class:`Criterion` called ``name``; ``region_rows`` is only evaluated for I-optimality."""
    if name == "d_optimal":
        return Criterion.d()
    if name == "a_optimal":
        return Criterion.a(n_parameters)
    if name == "i_optimal":
        return Criterion.i(region_rows())
    if name == "e_optimal":
        return Criterion.e()
    raise ValueError(f"Unknown criterion {name!r}; choose 'd_optimal', 'i_optimal', 'a_optimal' or 'e_optimal'.")


def constrained_optimal_design(
    factors: list[Factor],
    budget: int,
    constraints: list[Constraint],
    options: ConstrainedOptions | None = None,
    random_state: int | np.random.Generator | None = None,
) -> tuple[np.ndarray, dict]:
    """Generate a D-, I-, A- or E-optimal design from a candidate set, with every run satisfying ``constraints``.

    This is also the optimal-design backend when pyoptex is not installed, with
    ``constraints`` empty.

    Parameters
    ----------
    factors : list[Factor]
        Continuous and categorical factors. Mixture factors are not supported here.
    budget : int
        Total number of runs, including ``fixed_runs``.
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
        If a constraint cannot be parsed, no candidate point is feasible, or the
        feasible region cannot support the requested model.
    """
    rng = check_random_state(random_state)
    opts = options if options is not None else ConstrainedOptions()
    model_type, fixed_runs = opts.model_type, opts.fixed_runs
    if any(f.type == FactorType.mixture for f in factors):
        raise ValueError(
            "Mixture components need the mixture engine: give only mixture factors, and generate_design routes "
            "them there. Mixture-process designs are not supported."
        )

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
    criterion = make_criterion(opts.criterion, p, region_rows)
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
    if n_fixed:
        meta["n_fixed_runs"] = n_fixed
    if labels is not None:
        meta["candidate_source"] = "user"
        meta["selected_candidates"] = selection_counts(labels, rows)
    values = design.to_numpy() if categorical else design.to_numpy(dtype=float)
    return values, meta
