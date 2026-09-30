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

from process_improve._random import check_random_state
from process_improve.experiments.factor import FactorType

if TYPE_CHECKING:
    from process_improve.experiments.factor import Constraint, Factor

logger = logging.getLogger(__name__)

#: Upper limit on the candidate grid; a larger grid is refused before allocation.
MAX_CANDIDATES = 100_000
#: Longest constraint expression accepted, which also bounds the parse depth.
MAX_EXPRESSION_LENGTH = 500
#: Slack when testing ``g(x) <= 0``, so points found on a boundary are kept.
_FEASIBILITY_TOL = 1e-9
_BISECTION_STEPS = 50
_MAX_EXCHANGES = 500

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
    match node:
        case ast.Constant(value=value) if isinstance(value, int | float) and not isinstance(value, bool):
            number = float(value)
            return lambda _env: number
        case ast.Name(id=name):
            if name not in names:
                raise ValueError(f"Unknown name {name!r}; constraints may use the continuous factors {sorted(names)}.")
            return lambda env: env[name]
        case ast.BinOp(left=left, op=op, right=right) if type(op) in _BINARY_OPS:
            fn, lhs, rhs = _BINARY_OPS[type(op)], _compile(left, names), _compile(right, names)
            return lambda env: fn(lhs(env), rhs(env))
        case ast.UnaryOp(op=op, operand=operand) if type(op) in _UNARY_OPS:
            unary, inner = _UNARY_OPS[type(op)], _compile(operand, names)
            return lambda env: unary(inner(env))
        case ast.Call(func=ast.Name(id=fname), args=[arg], keywords=[]) if fname in _FUNCTIONS:
            func, inner = _FUNCTIONS[fname], _compile(arg, names)
            return lambda env: func(inner(env))
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
    match tree:
        case ast.Compare(left=left, ops=ops, comparators=comparators):
            terms = [_compile(left, names)] + [_compile(c, names) for c in comparators]
        case _:
            raise ValueError(f"Constraint {expression!r} must be an inequality, e.g. 'A + B <= 10'.")

    inequalities = []
    for op, lhs, rhs in zip(ops, terms[:-1], terms[1:], strict=True):
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


def _grid_levels(region: _Region, n_levels: int | None) -> int:
    """Pick the continuous grid resolution: the finest of 5, 4, 3 under ``MAX_CANDIDATES``."""
    n_cat = int(np.prod([len(f.levels or []) for f in region.categorical]))
    choices = [n_levels] if n_levels is not None else [5, 4, 3]
    for n in choices:
        if n < 2:
            raise ValueError("n_levels must be at least 2.")
        if n ** len(region.continuous) * n_cat <= MAX_CANDIDATES:
            return n
    raise ValueError(
        f"A candidate grid with {choices[-1]} levels on {len(region.continuous)} continuous factor(s) "
        f"exceeds {MAX_CANDIDATES} points. Reduce the number of factors or n_levels."
    )


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


def build_candidates(region: _Region, n_levels: int | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return the feasible candidate points in coded units.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, dict]
        Coded continuous values ``(n, k_cont)``, categorical level indices
        ``(n, k_cat)``, and counts for the metadata.
    """
    levels = _grid_levels(region, n_levels)
    shape = [levels] * len(region.continuous) + [len(f.levels or []) for f in region.categorical]
    idx = np.indices(shape).reshape(len(shape), -1).T
    k_cont = len(region.continuous)
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


def fedorov_exchange(
    f_cand: np.ndarray,
    n_free: int,
    f_fixed: np.ndarray,
    rng: np.random.Generator,
    n_starts: int = 5,
) -> tuple[np.ndarray, float]:
    """Choose ``n_free`` candidate rows that maximise ``log det(X'X)``.

    Each iteration makes the single swap (design run ``i`` out, candidate ``j`` in)
    that increases the determinant most. For information matrix ``M``, with
    ``d(a, b) = f(a)' M^-1 f(b)``, the swap multiplies ``det(M)`` by ``1 + delta``::

        delta = d(j) - d(i) - [d(i) d(j) - d(i, j)**2]

    (Fedorov 1972; Cook and Nachtsheim 1980), so all swaps are scored in one matrix
    product. Rows in ``f_fixed`` stay in the design and are never swapped out.
    Candidates may repeat, which gives replicated runs where the criterion wants them.

    Returns
    -------
    tuple[np.ndarray, float]
        Selected candidate indices and ``log det(X'X)`` of the full design.
    """
    best_rows, best_logdet = np.empty(0, dtype=int), -np.inf
    for _ in range(n_starts):
        rows = _greedy_start(f_cand, f_fixed, n_free, rng)
        for _ in range(_MAX_EXCHANGES):
            x = np.vstack([f_fixed, f_cand[rows]])
            m_inv = np.linalg.pinv(x.T @ x)
            v_design = f_cand[rows] @ m_inv
            d_design = np.einsum("ij,ij->i", v_design, f_cand[rows])
            d_cand = np.einsum("ij,ij->i", f_cand @ m_inv, f_cand)
            d_cross = v_design @ f_cand.T
            delta = d_cand[None, :] - d_design[:, None] - (np.outer(d_design, d_cand) - d_cross**2)
            i, j = np.unravel_index(np.argmax(delta), delta.shape)
            if delta[i, j] <= 1e-9:
                break
            rows[i] = j
        x = np.vstack([f_fixed, f_cand[rows]])
        sign, logdet = np.linalg.slogdet(x.T @ x)
        if sign > 0 and logdet > best_logdet:
            best_rows, best_logdet = rows.copy(), float(logdet)
    return best_rows, best_logdet


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def constrained_d_optimal(  # noqa: PLR0913
    factors: list[Factor],
    budget: int,
    constraints: list[Constraint],
    model_type: str = "interactions",
    fixed_runs: pd.DataFrame | None = None,
    n_levels: int | None = None,
    random_state: int | np.random.Generator | None = None,
) -> tuple[np.ndarray, dict]:
    """Generate a D-optimal design whose runs all satisfy ``constraints``.

    Parameters
    ----------
    factors : list[Factor]
        Continuous and categorical factors. Mixture factors are not supported here.
    budget : int
        Total number of runs, including ``fixed_runs``.
    constraints : list[Constraint]
        Inequalities in actual units over the continuous factors, e.g.
        ``Constraint(expression="3*T + 5*D <= 600")``.
    model_type : str
        ``"main_effects"``, ``"interactions"`` or ``"quadratic"``.
    fixed_runs : pandas.DataFrame or None
        Runs kept in the design (continuous in coded units, categorical as labels),
        already validated by the caller. They count towards ``budget``.
    n_levels : int or None
        Grid levels per continuous factor. ``None`` picks 5, 4 or 3, whichever is
        the finest that keeps the grid under ``MAX_CANDIDATES`` points.
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
    if any(f.type == FactorType.mixture for f in factors):
        raise ValueError("Constraints on mixture factors are not supported by the constrained D-optimal design.")

    continuous = [f for f in factors if f.type != FactorType.categorical]
    categorical = [f for f in factors if f.type == FactorType.categorical]
    names = {f.name for f in continuous}
    inequalities = [g for c in constraints for g in parse_constraint(c.expression, names)]
    region = _Region(continuous, categorical, inequalities)

    coded, cats, counts = build_candidates(region, n_levels)
    if counts["n_candidates"] == 0:
        raise ValueError("No point in the factor box satisfies all the constraints; check them for conflicts.")

    f_cand = model_matrix(region, coded, cats, model_type)
    f_fixed, n_fixed = np.empty((0, f_cand.shape[1])), 0
    if fixed_runs is not None:
        fixed_coded = fixed_runs[[f.name for f in continuous]].to_numpy(dtype=float)
        labels = {f.name: [str(lv) for lv in f.levels or []] for f in categorical}
        fixed_cats = np.array(
            [[labels[f.name].index(str(v)) for v in fixed_runs[f.name]] for f in categorical], dtype=int
        ).T.reshape(len(fixed_runs), len(categorical))
        f_fixed, n_fixed = model_matrix(region, fixed_coded, fixed_cats, model_type), len(fixed_runs)
        n_outside = int((region.slack(fixed_coded) > _FEASIBILITY_TOL).sum())
        if n_outside:
            logger.warning("%d fixed run(s) lie outside the constrained region; they are kept as given.", n_outside)

    if np.linalg.matrix_rank(np.vstack([f_fixed, f_cand])) < f_cand.shape[1]:
        raise ValueError(
            f"The feasible region ({counts['n_candidates']} candidate points) cannot support a "
            f"'{model_type}' model with {f_cand.shape[1]} coefficients. Use a simpler model, a finer "
            "grid (n_levels), or loosen the constraints."
        )

    rows, logdet = fedorov_exchange(f_cand, budget - n_fixed, f_fixed, rng)

    design = pd.DataFrame(coded[rows], columns=[f.name for f in continuous])
    for j, f in enumerate(categorical):
        design[f.name] = np.asarray(f.levels, dtype=object)[cats[rows, j]]
    if fixed_runs is not None:
        design = pd.concat([fixed_runs[design.columns].reset_index(drop=True), design], ignore_index=True)
    design = design[[f.name for f in factors]]

    meta = {
        "backend": "constrained_exchange",
        "optimality_criterion": "d_optimal",
        "model_type": model_type,
        "constraints": [c.expression for c in constraints],
        "constraints_enforced": True,
        "log_det_information": logdet,
        **counts,
    }
    if n_fixed:
        meta["n_fixed_runs"] = n_fixed
    values = design.to_numpy() if categorical else design.to_numpy(dtype=float)
    return values, meta
