# (c) Kevin Dunn, 2010-2026. MIT License.

"""Integer-programming generator for OMARS designs.

The constructive generator in :mod:`process_improve.experiments.designs_omars`
(``dispatch_omars``) only builds the minimal conference-foldover member of the
OMARS family (``2k + 1`` / ``2k + 3`` runs).  That design is saturated for a
full second-order model, so :func:`process_improve.experiments.analyze_omars`
has no error degrees of freedom to work with.  This module builds *larger*
OMARS designs that leave error degrees of freedom, by selecting runs with an
integer linear program (ILP).

Method
------
Every design here is a **foldover** ``[H; -H; 0]``: a half-design ``H``, its
mirror image ``-H``, and a single centre run.  The foldover structure makes
three of the four OMARS-defining conditions hold automatically:

* balance - ``h`` and ``-h`` cancel, so every main-effect column sums to zero;
* main effects clear of the two-factor interactions - ``x_i x_a x_b`` is an odd
  function, so its contributions from ``h`` and ``-h`` cancel;
* main effects clear of the pure quadratics - ``x_i x_j^2`` is odd in ``x_i``,
  so those contributions cancel too;

and the centre run makes every pure quadratic estimable (each ``x_i^2`` column
takes the value 0 there).  The only condition that is *not* automatic is the
mutual orthogonality of the main effects, which is linear in the binary
"include this half-run" variables ``s_r``: for each pair ``i < j``,
``sum_r (x[r,i] x[r,j]) s_r = 0``.  The run count is ``2 * sum_r s_r + 1``.

So the ILP selects a half-design from the ``(3**k - 1) / 2`` distinct non-mirror
three-level runs subject to a handful of linear equalities - only ``k(k-1)/2``
of them - which keeps it tractable up to seven factors.  Because the
coefficients are integers, the equalities are exact; the floating-point
:func:`is_omars` re-check only guards against mistakes.  A pure feasibility
solve, however, returns an arbitrary OMARS design that is usually far from the
most efficient member.  To search for a high-quality design the solve is
repeated with random linear objectives (a multistart): each random objective
sends the solver to a different vertex of the feasibility polytope, so the
retained designs span the high-D-efficiency / low-A members.  The best is then
chosen by a satisficing-and-dominance rule over D-efficiency and the maximum
second-order correlation, following the selection philosophy of Nunez Ares and
Goos (2020).  This makes the generator competitive with their enumerated
catalogue without consulting it.

This realises, for OMARS designs, the integer-programming construction of Nunez
Ares and Goos (2020); the ILP-over-design-points framing is shared with their
trend-robust run-order work (Nunez Ares and Goos, 2019).  An exhaustively
enumerated OMARS catalogue exists but is unlicensed and is not redistributed
here.  Only the (dominant) foldover OMARS family is generated; the rarer
non-foldover members are a documented future extension.

References
----------
* Nunez Ares, J. and Goos, P. (2020).  "Enumeration and multicriteria
  selection of orthogonal minimally aliased response surface designs."
  *Technometrics*, 62(1):21-36.
* Nunez Ares, J. and Goos, P. (2019).  "An integer linear programming
  approach to find trend-robust run orders of experimental designs."
  *Journal of Quality Technology*.
"""

from __future__ import annotations

import itertools
import math
import time
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from process_improve.experiments.designs_omars import _second_order_terms, is_omars

try:
    import pulp

    _PULP_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised via env-without-pulp
    _PULP_AVAILABLE = False

if TYPE_CHECKING:
    from process_improve.experiments.factor import DesignResult, Factor

# Selection criteria understood by :func:`generate_omars`.
_CRITERIA = ("dominance", "d_efficiency", "min_second_order_correlation", "a_optimal")

# Analysis models a design can be *sized* for.  The OMARS construction is
# identical either way - the main effects stay clear of every second-order term
# (quadratics and interactions both) - so this choice does not change the design
# family.  It only sets how many runs the design must have to leave error
# degrees of freedom, and which model matrix the D-efficiency is read from.
# "full_second_order" keeps room for all two-factor interactions;
# "main_quadratic" drops them from the analysis model, so it admits smaller
# designs (for example a thirteen-run, four-factor OMARS) that can still fit the
# main effects and pure quadratics with error df to spare.
_MODELS = ("full_second_order", "main_quadratic")

# Attributes that ``satisfice`` thresholds may constrain.  ``d_efficiency`` is a
# lower bound (higher is better); ``max_second_order_correlation`` is an upper
# bound (lower is better).
_SATISFICE_KEYS = ("d_efficiency", "max_second_order_correlation")

# Early-stop the randomized multistart once this many consecutive solves fail to
# turn up a new distinct design: the feasible set is effectively exhausted (small
# factor counts) and further solves only repeat designs already retained.
_RESTART_PATIENCE = 25

# The exhaustive search enumerates every feasible half-design multiset (counts
# per sign class, replication allowed) when the class is small enough.  The caps
# below bound the half-design size per factor count; beyond them, or past the
# leaf budget, the search falls back to the randomized multistart.  The caps are
# calibrated so the worst in-cap cell enumerates in seconds on ordinary hardware.
_ENUM_MAX_HALF = {3: 18, 4: 12}
_ENUM_MAX_LEAVES = 4_000_000
# Batch size for the vectorised scoring of enumerated count vectors.
_ENUM_SCORE_CHUNK = 65_536


@dataclass
class _Candidate:
    """A single feasible OMARS design found by the ILP, with its quality metrics."""

    coded: np.ndarray
    n_runs: int
    half_indices: list[int]
    d_efficiency: float
    a_optimality: float
    max_second_order_correlation: float
    solver_status: str


@dataclass
class OmarsSearchReport:
    """Diagnostics from the ILP search, recorded on ``DesignResult.metadata``.

    Attributes
    ----------
    n_factors : int
        Number of factors.
    half_pool_size : int
        Number of distinct non-mirror three-level runs the ILP chose from.
    n_restarts : int
        Number of randomized-objective ILP solves the multistart was allowed
        (the actual count can be lower when early-stopping ends it).
    ilp_iterations : int
        Number of ILP solves (the outer search iterations): the minimize-size
        probe, the baseline feasibility solve, and every randomized-objective
        restart.
    feasible_designs : int
        Number of distinct verified OMARS designs found and ranked.
    run_size : int
        Run count of the winning design.
    total_solve_seconds : float
        Cumulative wall-clock time spent inside the ILP solver.
    search_mode : str
        ``"exhaustive"`` when the feasible design class was enumerated in full
        (the selection is then exact), ``"multistart"`` when the randomized
        ILP multistart was used (the selection is then the best design found,
        which may miss the optimum).
    enumerated_designs : int
        Number of feasible designs enumerated on the exhaustive path (0 on the
        multistart path).
    """

    n_factors: int = 0
    half_pool_size: int = 0
    n_restarts: int = 0
    ilp_iterations: int = 0
    feasible_designs: int = 0
    run_size: int = 0
    total_solve_seconds: float = 0.0
    search_mode: str = ""
    enumerated_designs: int = 0


def _half_pool(n_factors: int) -> np.ndarray:
    """Return the distinct non-mirror three-level runs (one per ``+/-`` pair).

    These are the candidate half-runs: every nonzero run of the ``3**k`` grid
    whose first nonzero coordinate is ``+1``.  The full foldover design adds the
    mirror ``-H`` and a centre run.
    """
    grid = itertools.product((-1.0, 0.0, 1.0), repeat=n_factors)
    reps = []
    for run in grid:
        run_array = np.asarray(run, dtype=float)
        nonzero = np.flatnonzero(run_array)
        if nonzero.size and run_array[nonzero[0]] > 0:
            reps.append(run_array)
    return np.array(reps, dtype=float)


def _foldover(half: np.ndarray) -> np.ndarray:
    """Assemble the foldover design ``[H; -H; 0]`` from a half-design ``H``."""
    return np.vstack([half, -half, np.zeros((1, half.shape[1]))])


def _model_matrix(coded: np.ndarray, model: str = "full_second_order") -> np.ndarray:
    """Model matrix the design is sized for: ``[1 | main effects | second-order terms]``.

    For ``model="main_quadratic"`` the two-factor interactions are dropped,
    leaving ``[1 | main effects | pure quadratics]``.
    """
    second_order, names = _second_order_terms(coded)
    if model == "main_quadratic":
        keep = [t for t, name in enumerate(names) if "^2" in name]
        second_order = second_order[:, keep]
    return np.column_stack([np.ones(coded.shape[0]), coded, second_order])


def _min_half_runs(n_factors: int, model: str = "full_second_order") -> int:
    r"""Smallest half-design size at which the sizing model becomes estimable.

    In a foldover ``[H; -H; 0]`` every second-order term is an **even** function,
    so the quadratic and interaction columns of ``H`` and ``-H`` are identical.
    The even block therefore has at most ``h + 1`` distinct rows, against
    ``1 + k(k+1)/2`` columns for the full second-order model (an intercept, ``k``
    pure quadratics and ``k(k-1)/2`` interactions), or ``1 + k`` for
    ``"main_quadratic"``.  The main effects live in the odd block and contribute
    ``k`` more, so

    .. math::

        \mathrm{rank}(X) = k + \min(h + 1, \text{even-block columns})

    and the model is estimable only once ``h`` reaches the even-block column
    count minus one.
    """
    if model == "main_quadratic":
        return n_factors
    return n_factors * (n_factors + 1) // 2


def _min_runs(n_factors: int, model: str = "full_second_order") -> int:
    """Smallest (odd) run count at which the sizing model is estimable.

    ``k**2 + k + 1`` for the full second-order model, ``2k + 1`` for
    ``"main_quadratic"``.  See :func:`_min_half_runs` for the derivation.
    """
    return 2 * _min_half_runs(n_factors, model) + 1


def _model_rank(coded: np.ndarray, model: str = "full_second_order") -> int:
    """Rank of the sizing-model matrix of a coded design."""
    return int(np.linalg.matrix_rank(_model_matrix(coded, model)))


def _d_efficiency(coded: np.ndarray, model: str = "full_second_order") -> float:
    """D-efficiency of the sizing model: ``100 * |X'X|^(1/p) / n``.

    Returns ``0.0`` for a rank-deficient model matrix.  Without that guard
    ``slogdet`` reports a finite log-determinant for an exactly singular
    integer Gram matrix (floating-point round-off away from zero), which would
    make an unusable design look merely mediocre.  This mirrors the rank guard
    in :func:`~process_improve.experiments.evaluate._compute_d_efficiency` and
    in :func:`_a_optimality` below.
    """
    model_matrix = _model_matrix(coded, model)
    n_runs, n_params = model_matrix.shape
    if n_runs < n_params or np.linalg.matrix_rank(model_matrix) < n_params:
        return 0.0
    sign, log_det = np.linalg.slogdet(model_matrix.T @ model_matrix)
    if sign <= 0:
        # Belt and braces.  A full-rank X'X is positive definite, but
        # ``matrix_rank`` decides rank against an SVD tolerance, so a Gram
        # matrix can clear the guard above and still be too ill-conditioned for
        # ``slogdet`` to return a positive sign.
        return 0.0
    return float(100.0 * math.exp(log_det / n_params) / n_runs)


def _a_optimality(coded: np.ndarray, model: str = "full_second_order") -> float:
    """A-optimality of the sizing model: ``trace((X'X)^-1)``, the summed coefficient variance.

    Lower is better.  Returns ``inf`` for a rank-deficient model matrix (the
    coefficients are then not jointly estimable).
    """
    model_matrix = _model_matrix(coded, model)
    n_runs, n_params = model_matrix.shape
    if n_runs < n_params or np.linalg.matrix_rank(model_matrix) < n_params:
        return float("inf")
    return float(np.trace(np.linalg.inv(model_matrix.T @ model_matrix)))


def _full_second_order_params(n_factors: int) -> int:
    """Return the column count of the full second-order model (including the intercept)."""
    return 1 + 2 * n_factors + n_factors * (n_factors - 1) // 2


def _model_params(n_factors: int, model: str) -> int:
    """Column count (including the intercept) of the model a design is sized for.

    ``"full_second_order"`` counts ``1 + 2k + k(k-1)/2`` (intercept, main
    effects, pure quadratics, and the two-factor interactions).
    ``"main_quadratic"`` counts ``1 + 2k`` (intercept, main effects, and pure
    quadratics only), because the two-factor interactions are not in the
    analysis model.
    """
    if model == "main_quadratic":
        return 1 + 2 * n_factors
    return _full_second_order_params(n_factors)


def solve_omars_ilp(  # noqa: PLR0913
    half_pool: np.ndarray,
    *,
    n_half: int | None = None,
    half_bounds: tuple[int, int] | None = None,
    minimize_size: bool = False,
    objective: np.ndarray | None = None,
    exclude_solutions: list[list[int]] | None = None,
    solver_options: dict[str, Any] | None = None,
) -> tuple[np.ndarray | None, str, list[int]]:
    """Select a half-design from *half_pool* and return the foldover OMARS design.

    Exactly one of *n_half* (exact half count) or *half_bounds* (inclusive
    ``(min, max)`` half count) sets the size constraint.  The returned design
    has ``2 * n_half + 1`` runs.

    Parameters
    ----------
    half_pool : np.ndarray
        Candidate half-runs of shape ``(n_candidates, n_factors)``, coded to
        ``{-1, 0, +1}`` (see :func:`_half_pool`).
    n_half : int, optional
        Exact number of half-runs to select.
    half_bounds : tuple[int, int], optional
        Inclusive ``(min, max)`` half-run count.
    minimize_size : bool, optional
        When ``True`` the objective minimises the half-run count (smallest
        feasible design); otherwise the solve is a pure feasibility search.
    objective : np.ndarray, optional
        Per-candidate linear cost of shape ``(n_candidates,)``.  When given, the
        solver minimises ``sum_r objective[r] * s_r`` instead of running a pure
        feasibility (or minimise-size) search.  A random objective drives the
        solver to a different vertex of the OMARS-feasibility polytope, which is
        how :func:`generate_omars` samples diverse, high-quality designs.  Takes
        precedence over *minimize_size*.
    exclude_solutions : list[list[int]], optional
        Previously found half-index sets to forbid via no-good cuts.
    solver_options : dict, optional
        ``{"msg": bool, "time_limit": int seconds}``.

    Returns
    -------
    tuple
        ``(design or None, solver_status, chosen_half_indices)``.  ``None`` means
        the solver returned no feasible selection.

    Raises
    ------
    ImportError
        If PuLP (the ``ilp`` extra) is not installed.
    """
    if not _PULP_AVAILABLE:
        from process_improve._extras import require_extra  # noqa: PLC0415

        raise require_extra("pulp", "ilp")

    options = solver_options or {}
    n_candidates, n_factors = half_pool.shape

    # PuLP 3.x deprecates the LpVariable constructor and PULP_CBC_CMD in favour
    # of a 4.0 API; both still work and keep us compatible back to pulp 2.8, so
    # silence the (very repetitive) deprecation noise here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        s = [pulp.LpVariable(f"s_{r}", cat="Binary") for r in range(n_candidates)]
    problem = pulp.LpProblem("omars_foldover", pulp.LpMinimize)
    if objective is not None:
        problem += pulp.lpSum(float(objective[r]) * s[r] for r in range(n_candidates)), "objective"
    else:
        problem += (pulp.lpSum(s) if minimize_size else 0), "objective"

    # Main-effect orthogonality over the half-design (the only non-automatic
    # OMARS condition; balance and clear-of-second-order hold by the foldover).
    for i, j in itertools.combinations(range(n_factors), 2):
        coefficients = half_pool[:, i] * half_pool[:, j]
        nonzero = np.flatnonzero(coefficients)
        if nonzero.size:
            problem.addConstraint(pulp.lpSum(float(coefficients[r]) * s[r] for r in nonzero) == 0, f"me_orth_{i}_{j}")

    if n_half is not None:
        problem += pulp.lpSum(s) == n_half, "half_size"
    elif half_bounds is not None:
        low, high = half_bounds
        problem += pulp.lpSum(s) >= low, "half_size_lo"
        problem += pulp.lpSum(s) <= high, "half_size_hi"
    else:  # pragma: no cover - defensive: caller always sets one
        raise ValueError("solve_omars_ilp requires either n_half or half_bounds.")

    for cut, excluded in enumerate(exclude_solutions or []):
        problem += pulp.lpSum(s[r] for r in excluded) <= len(excluded) - 1, f"nogood_{cut}"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        solver = pulp.PULP_CBC_CMD(msg=bool(options.get("msg", False)), timeLimit=int(options.get("time_limit", 60)))
        problem.solve(solver)
    status = pulp.LpStatus[problem.status]

    # Accept only a genuine feasible/optimal integer solution (sol_status > 0);
    # a time-limited "no solution found" leaves the variables meaningless.
    if problem.sol_status <= 0:
        return None, status, []
    chosen = [r for r in range(n_candidates) if (s[r].value() or 0) > 0.5]
    if not chosen:
        return None, status, []
    return _foldover(half_pool[chosen]), status, chosen


def _half_bounds(
    n_runs_range: tuple[int, int] | None,
    n_params: int,
    half_pool_size: int,
    min_half: int,
    center_runs: int = 1,
) -> tuple[int, int]:
    """Inclusive ``(min, max)`` half-run window for a usable design.

    Two floors apply, and the window starts at whichever binds harder:

    * **Estimability**: ``min_half`` half-runs, from :func:`_min_half_runs`.
      Below this the sizing model is rank-deficient and cannot be fitted at
      all, whatever the parameter count says.
    * **Error degrees of freedom**: the total run count ``2h + center_runs``
      must exceed ``n_params``.

    For the full second-order model estimability is the binding floor; for
    ``"main_quadratic"`` the error-df floor usually is.  *n_runs_range* is in
    total runs, centre runs included.
    """
    floor_half = max(1, min_half, (n_params - center_runs) // 2 + 1)
    if n_runs_range is not None:
        low, high = n_runs_range
        half_low = max(floor_half, math.ceil((low - center_runs) / 2))
        half_high = max(half_low, (high - center_runs) // 2)
    else:
        half_low = floor_half
        half_high = half_low + 6
    return half_low, max(half_low, min(half_high, half_pool_size))


def _is_dominated(candidate: _Candidate, others: list[_Candidate]) -> bool:
    """Pareto dominance on (D-efficiency up, max second-order correlation down)."""
    for other in others:
        if other is candidate:
            continue
        not_worse = (
            other.d_efficiency >= candidate.d_efficiency
            and other.max_second_order_correlation <= candidate.max_second_order_correlation
        )
        strictly_better = (
            other.d_efficiency > candidate.d_efficiency
            or other.max_second_order_correlation < candidate.max_second_order_correlation
        )
        if not_worse and strictly_better:
            return True
    return False


def _satisfice(candidates: list[_Candidate], thresholds: dict[str, float]) -> list[_Candidate]:
    """Keep only the designs meeting every acceptability threshold.

    ``d_efficiency`` is treated as a minimum (higher is better) and
    ``max_second_order_correlation`` as a maximum (lower is better).
    """
    unknown = set(thresholds) - set(_SATISFICE_KEYS)
    if unknown:
        msg = f"satisfice keys must be a subset of {_SATISFICE_KEYS}, got unknown {sorted(unknown)}."
        raise ValueError(msg)
    d_min = thresholds.get("d_efficiency")
    correlation_max = thresholds.get("max_second_order_correlation")
    return [
        candidate
        for candidate in candidates
        if (d_min is None or candidate.d_efficiency >= d_min)
        and (correlation_max is None or candidate.max_second_order_correlation <= correlation_max)
    ]


def _select(candidates: list[_Candidate], criterion: str) -> _Candidate:
    """Pick the winning design under the requested multicriteria rule."""
    if criterion == "d_efficiency":
        return max(candidates, key=lambda c: (c.d_efficiency, -c.n_runs))
    if criterion == "min_second_order_correlation":
        return min(candidates, key=lambda c: (c.max_second_order_correlation, -c.d_efficiency, c.n_runs))
    if criterion == "a_optimal":
        # Minimum summed coefficient variance trace((X'X)^-1); ties broken towards
        # the smaller, lower-aliasing design.
        return min(candidates, key=lambda c: (c.a_optimality, c.n_runs, c.max_second_order_correlation))
    # "dominance": keep the Pareto front, then prefer the smallest, most efficient design.
    front = [c for c in candidates if not _is_dominated(c, candidates)] or candidates
    return min(front, key=lambda c: (c.n_runs, -c.d_efficiency, c.max_second_order_correlation))


def _sparsity(coded: np.ndarray) -> tuple[int, int]:
    """Return the OMARS sparsity pair ``(n_ME0, n_IE0)``.

    ``n_ME0`` is the number of zeros in a main-effect column and ``n_IE0`` the
    number of zeros in a two-factor-interaction column, each reported as the
    minimum across the relevant columns.
    """
    n_me0 = int(np.min(np.sum(np.abs(coded) < 0.5, axis=0)))
    second_order, names = _second_order_terms(coded)
    interaction_cols = [t for t, name in enumerate(names) if "*" in name]
    # There is always at least one interaction column here (k >= 3).
    n_ie0 = int(np.min([np.sum(np.abs(second_order[:, t]) < 0.5) for t in interaction_cols])) if interaction_cols else 0
    return n_me0, n_ie0


def _max_second_order_correlation_metric(coded: np.ndarray, tol: float = 1e-9) -> float:
    """Largest absolute pairwise correlation among the second-order columns of a design.

    Unlike the descriptive statistic in
    :func:`process_improve.experiments.designs_omars.omars_properties` (which
    skips constant columns), this selection metric returns ``inf`` when any
    second-order column is constant: such a column belongs to a term the design
    cannot estimate, and skipping it would flatter exactly the degenerate
    designs a correlation-minimising selection would then crown as winners.
    """
    second_order, _ = _second_order_terms(coded)
    if second_order.shape[1] < 2:
        return 0.0
    centered = second_order - second_order.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=0)
    if np.any(norms <= tol):
        return float("inf")
    unit = centered / norms
    corr = unit.T @ unit
    off_diagonal = corr - np.diag(np.diag(corr))
    return float(np.abs(off_diagonal).max())


def _enumerate_feasible_counts(  # noqa: C901, PLR0915
    pool: np.ndarray, n_half: int, max_leaves: int
) -> tuple[np.ndarray, bool]:
    """Enumerate every feasible half-design multiset of size *n_half*.

    A foldover OMARS design is determined by how many times each sign class in
    *pool* appears in the half-design (replication allowed), so the design class
    at a fixed size is the set of count vectors ``m >= 0`` with
    ``sum(m) = n_half`` that satisfy the pairwise main-effect orthogonality
    equalities ``sum_r (x_ri * x_rj) m_r = 0``.  This walks that set with a
    depth-first search over the counts, pruning on the running pair balances.

    Returns ``(count_matrix, overflow)``: an integer array of shape
    ``(n_designs, len(pool))`` with columns in the row order of *pool*, and an
    ``overflow`` flag that is ``True`` when the *max_leaves* budget was hit
    (the enumeration is then incomplete and must not be used).
    """
    n_rows, n_factors = pool.shape
    pairs = list(itertools.combinations(range(n_factors), 2))
    n_pairs = len(pairs)
    # Deterministic order: widest-support rows first, so every orthogonality
    # constraint closes out (and prunes) as early as possible; the k singleton
    # rows, which touch no constraint, form the tail and are expanded
    # vectorised below rather than recursed over.
    support = (pool != 0).sum(axis=1)
    order = sorted(range(n_rows), key=lambda r: (-int(support[r]), tuple(-pool[r])))
    coeff: list[list[tuple[int, int]]] = []
    for r in order:
        row = pool[r]
        coeff.append([(p, int(row[i] * row[j])) for p, (i, j) in enumerate(pairs) if row[i] * row[j] != 0])
    tail_start = n_rows - n_factors
    last_touch = [-1] * n_pairs
    for pos, entries in enumerate(coeff):
        for p, _ in entries:
            last_touch[p] = pos
    # Per DFS level: the constraints whose last touching row was just passed
    # (they must have closed at zero) and those still open (prunable by the
    # remaining budget).  Checking only these keeps the per-node work small.
    closing_at = [[p for p in range(n_pairs) if last_touch[p] == pos - 1] for pos in range(tail_start + 1)]
    open_at = [[p for p in range(n_pairs) if last_touch[p] >= pos] for pos in range(tail_start + 1)]

    prefixes: list[tuple[tuple[int, ...], int]] = []
    n_leaves = 0
    counts = [0] * tail_start
    balance = [0] * n_pairs
    overflow = False
    # Different prefixes converge on the same (position, remaining, balances)
    # state; a state whose subtree yielded no feasible design once will yield
    # none again, so dead states are memoized and skipped on revisits.
    dead: set[tuple[int, ...]] = set()

    def tail_leaves(remaining: int) -> int:
        return math.comb(remaining + n_factors - 1, n_factors - 1)

    def rec(pos: int, remaining: int) -> None:  # noqa: C901
        nonlocal overflow, n_leaves
        if overflow:
            return
        for p in closing_at[pos]:
            if balance[p]:
                return
        if pos == tail_start:
            n_leaves += tail_leaves(remaining)
            if n_leaves > max_leaves:
                overflow = True
                return
            prefixes.append((tuple(counts), remaining))
            return
        for p in open_at[pos]:
            if abs(balance[p]) > remaining:
                return
        key = (pos, remaining, *(balance[p] for p in open_at[pos]))
        if key in dead:
            return
        before = len(prefixes)
        entries = coeff[pos]
        for c in range(remaining, -1, -1):
            counts[pos] = c
            for p, v in entries:
                balance[p] += c * v
            rec(pos + 1, remaining - c)
            for p, v in entries:
                balance[p] -= c * v
        counts[pos] = 0
        if len(prefixes) == before and not overflow:
            dead.add(key)

    rec(0, n_half)
    if overflow:
        return np.empty((0, n_rows), dtype=np.int16), True

    # Expand the unconstrained singleton tail: distribute each prefix's leftover
    # budget over the n_factors singleton rows in every possible way.
    comp_cache: dict[int, np.ndarray] = {}

    def compositions(total: int) -> np.ndarray:
        cached = comp_cache.get(total)
        if cached is None:
            rows = [
                [total - sum(parts), *parts]
                for parts in itertools.product(range(total + 1), repeat=n_factors - 1)
                if sum(parts) <= total
            ]
            cached = np.asarray(rows, dtype=np.int16)
            comp_cache[total] = cached
        return cached

    blocks = []
    for prefix, remaining in prefixes:
        tail = compositions(remaining)
        head = np.tile(np.asarray(prefix, dtype=np.int16), (tail.shape[0], 1))
        blocks.append(np.hstack([head, tail]))
    if not blocks:
        return np.empty((0, n_rows), dtype=np.int16), False
    ordered = np.vstack(blocks)
    # Map the DFS ordering back to pool row order.
    inverse = np.argsort(order)
    return ordered[:, inverse], False


def _score_count_vectors(
    count_matrix: np.ndarray,
    pool: np.ndarray,
    center_runs: int,
    model: str,
    tol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score enumerated count vectors: D-efficiency, A-optimality, max second-order correlation.

    Works from the count vectors alone, without materialising any design: for a
    foldover the model Gram matrix is a count-weighted sum of per-sign-class
    contributions plus the centre-run block, and the second-order columns are
    even functions, so their Gram doubles per half-row.  Scores match
    :func:`_d_efficiency`, :func:`_a_optimality` and
    :func:`_max_second_order_correlation_metric` evaluated on the materialised
    design (foldover plus ``center_runs - 1`` appended centre rows).
    """
    n_rows = pool.shape[0]
    n_leaves = count_matrix.shape[0]
    n_half_total = int(count_matrix[0].sum()) if n_leaves else 0
    n_total = 2 * n_half_total + center_runs

    second_order, so_names = _second_order_terms(pool)
    if model == "main_quadratic":
        keep = [t for t, name in enumerate(so_names) if "^2" in name]
        model_even = second_order[:, keep]
    else:
        model_even = second_order
    # Model rows for +h and -h: [1 | +-x | even terms]; their outer products sum.
    u_plus = np.column_stack([np.ones(n_rows), pool, model_even])
    u_minus = np.column_stack([np.ones(n_rows), -pool, model_even])
    n_params = u_plus.shape[1]
    b_flat = (np.einsum("ri,rj->rij", u_plus, u_plus) + np.einsum("ri,rj->rij", u_minus, u_minus)).reshape(n_rows, -1)
    b_center = np.zeros((n_params, n_params))
    b_center[0, 0] = 1.0

    q = second_order.shape[1]
    so_gram_flat = 2.0 * np.einsum("ri,rj->rij", second_order, second_order).reshape(n_rows, -1)
    so_colsum = 2.0 * second_order

    d_eff = np.empty(n_leaves)
    a_opt = np.empty(n_leaves)
    max_corr = np.empty(n_leaves)
    for start in range(0, n_leaves, _ENUM_SCORE_CHUNK):
        chunk = count_matrix[start : start + _ENUM_SCORE_CHUNK].astype(float)
        n_chunk = chunk.shape[0]
        gram = (chunk @ b_flat).reshape(n_chunk, n_params, n_params)
        gram += center_runs * b_center
        eig = np.linalg.eigvalsh(gram)
        singular = eig[:, 0] <= tol * np.maximum(1.0, eig[:, -1])
        with np.errstate(divide="ignore", invalid="ignore"):
            log_det = np.where(singular, -np.inf, np.log(np.where(eig > 0, eig, 1.0)).sum(axis=1))
            d_chunk = np.where(singular, 0.0, 100.0 * np.exp(log_det / n_params) / n_total)
            a_chunk = np.where(singular, np.inf, (1.0 / np.where(eig > 0, eig, 1.0)).sum(axis=1))

        so_gram = (chunk @ so_gram_flat).reshape(n_chunk, q, q)
        colsum = chunk @ so_colsum
        centered = so_gram - colsum[:, :, None] * colsum[:, None, :] / n_total
        variances = np.einsum("lii->li", centered)
        constant = (variances <= tol).any(axis=1)
        safe_var = np.where(variances > tol, variances, 1.0)
        scale = np.sqrt(safe_var[:, :, None] * safe_var[:, None, :])
        corr = np.abs(centered / scale)
        idx = np.arange(q)
        corr[:, idx, idx] = 0.0
        corr_chunk = np.where(constant, np.inf, corr.max(axis=(1, 2)))

        d_eff[start : start + n_chunk] = d_chunk
        a_opt[start : start + n_chunk] = a_chunk
        max_corr[start : start + n_chunk] = corr_chunk
    return d_eff, a_opt, max_corr


def _pick_exhaustive_winner(d_eff: np.ndarray, a_opt: np.ndarray, max_corr: np.ndarray, criterion: str) -> int:
    """Index of the winning count vector, mirroring the tie-breaks of :func:`_select`.

    All enumerated designs share the same run count, so the run-size terms of
    the :func:`_select` tie-break tuples drop out.
    """
    if criterion == "d_efficiency":
        keys = (max_corr, -d_eff)
    elif criterion == "min_second_order_correlation":
        keys = (-d_eff, max_corr)
    elif criterion == "a_optimal":
        keys = (max_corr, a_opt)
    else:  # "dominance": the Pareto-front member with the highest D-efficiency.
        keys = (max_corr, -d_eff)
    # np.lexsort sorts by the last key first.
    return int(np.lexsort(keys)[0])


def _search_best_omars(  # noqa: C901, PLR0912, PLR0913, PLR0915
    factors: list[Factor],
    *,
    n_runs: int | None,
    n_runs_range: tuple[int, int] | None,
    selection_criterion: str,
    satisfice: dict[str, float] | None,
    n_restarts: int,
    model: str,
    solver_options: dict[str, Any] | None,
    tol: float,
    verify: bool,
    random_seed: int,
    center_runs: int = 1,
) -> tuple[np.ndarray, dict]:
    """Run the design search and return ``(coded_matrix, metadata)`` for the winner.

    *n_runs*, *n_runs_range*, and every reported run count are **totals**,
    centre runs included.  The returned matrix is a foldover design and
    contains exactly one centre run; callers append the remaining
    ``center_runs - 1`` centre rows during post-processing, but all quality
    metrics here are computed with those rows included, so the metadata
    describes the design the caller receives.
    """
    from process_improve.config import settings  # noqa: PLC0415

    if selection_criterion not in _CRITERIA:
        msg = f"selection_criterion must be one of {_CRITERIA}, got {selection_criterion!r}."
        raise ValueError(msg)
    if model not in _MODELS:
        msg = f"model must be one of {_MODELS}, got {model!r}."
        raise ValueError(msg)
    n_factors = len(factors)
    if n_factors < 3:
        raise ValueError("OMARS designs require at least 3 factors.")
    if n_factors > settings.max_factors_combinatorial:
        msg = (
            f"{n_factors} factors exceeds the combinatorial cap "
            f"max_factors_combinatorial={settings.max_factors_combinatorial} (SEC-19); the 3**k candidate pool "
            "would be too large."
        )
        raise ValueError(msg)

    n_params = _model_params(n_factors, model)
    target_half: int | None = None
    if n_runs is not None:
        if n_runs <= n_params:
            msg = (
                f"n_runs={n_runs} leaves no error degrees of freedom: the {model} model has "
                f"{n_params} parameters, so n_runs (the total run count, centre runs included) "
                f"must exceed {n_params}."
            )
            raise ValueError(msg)
        if n_runs <= center_runs or (n_runs - center_runs) % 2 != 0:
            msg = (
                f"n_runs={n_runs} is incompatible with center_runs={center_runs}: a foldover OMARS "
                f"design has n_runs = 2*h + center_runs runs (h half-runs, their h mirrors, and the "
                f"centre runs), so n_runs - center_runs must be a positive even number. "
                f"Try n_runs={n_runs + 1} or n_runs={n_runs - 1}."
            )
            raise ValueError(msg)
        min_runs = 2 * _min_half_runs(n_factors, model) + center_runs
        if n_runs < min_runs:
            msg = (
                f"n_runs={n_runs} cannot estimate the {model} model for {n_factors} factors with "
                f"center_runs={center_runs}: a foldover design repeats its second-order terms across "
                f"H and -H, so the model matrix stays rank-deficient below {min_runs} runs. Use "
                f'n_runs >= {min_runs}, or model="main_quadratic" to size for main effects and pure '
                "quadratics only."
            )
            raise ValueError(msg)
        target_half = (n_runs - center_runs) // 2

    pool = _half_pool(n_factors)
    report = OmarsSearchReport(n_factors=n_factors, half_pool_size=pool.shape[0], n_restarts=n_restarts)
    candidates: list[_Candidate] = []
    seen: set[frozenset[int]] = set()
    extra_centers = np.zeros((center_runs - 1, n_factors))

    def _solve(**solve_kwargs: Any) -> tuple[np.ndarray | None, str, list[int]]:  # noqa: ANN401
        started = time.perf_counter()
        result = solve_omars_ilp(pool, solver_options=solver_options, **solve_kwargs)
        report.ilp_iterations += 1
        report.total_solve_seconds += time.perf_counter() - started
        return result

    def _record(coded: np.ndarray, indices: list[int], status: str) -> bool:
        """Verify and retain a distinct OMARS design; return True if it was new."""
        key = frozenset(indices)
        if key in seen:
            return False
        seen.add(key)
        if verify and not is_omars(coded, tol=tol):
            return False
        # Score the design the caller will actually receive: the foldover plus
        # the extra centre runs appended during post-processing.
        scored = np.vstack([coded, extra_centers])
        candidates.append(
            _Candidate(
                coded=coded,
                n_runs=scored.shape[0],
                half_indices=indices,
                d_efficiency=_d_efficiency(scored, model),
                a_optimality=_a_optimality(scored, model),
                max_second_order_correlation=_max_second_order_correlation_metric(scored, tol=tol),
                solver_status=status,
            )
        )
        return True

    # Find the target half-size: pinned exactly, or the smallest feasible size in
    # the window (the minimize-size solution becomes the first candidate).
    if target_half is None:
        coded, status, indices = _solve(
            half_bounds=_half_bounds(
                n_runs_range, n_params, pool.shape[0], _min_half_runs(n_factors, model), center_runs
            ),
            minimize_size=True,
        )
        if coded is not None:
            target_half = len(indices)
            _record(coded, indices, status)

    # Exhaustive path: when the design class at this size is small enough,
    # enumerate every feasible half-design multiset (replication allowed) and
    # pick the winner exactly.  The binary multistart below cannot even reach
    # designs that repeat a half-run, so this is what makes the selection
    # criteria live up to their names (issues #497, #498, #499).
    exhausted = False
    if target_half is not None and target_half <= _ENUM_MAX_HALF.get(n_factors, 0):
        count_matrix, overflow = _enumerate_feasible_counts(pool, target_half, _ENUM_MAX_LEAVES)
        n_enumerated = count_matrix.shape[0]
        if not overflow:
            if n_enumerated == 0:
                target = f"n_runs={n_runs}" if n_runs is not None else f"n_runs_range={n_runs_range}"
                msg = (
                    f"No feasible OMARS design exists for {target} with center_runs={center_runs} "
                    "(exhaustive enumeration). Try a different n_runs or a wider n_runs_range."
                )
                raise ValueError(msg)
            d_eff, a_opt, max_corr = _score_count_vectors(count_matrix, pool, center_runs, model, tol=tol)
            keep = np.ones(n_enumerated, dtype=bool)
            if satisfice:
                _satisfice([], satisfice)  # validate the threshold keys
                d_min = satisfice.get("d_efficiency")
                correlation_max = satisfice.get("max_second_order_correlation")
                if d_min is not None:
                    keep &= d_eff >= d_min
                if correlation_max is not None:
                    keep &= max_corr <= correlation_max
                if not keep.any():
                    finite_corr = max_corr[np.isfinite(max_corr)]
                    best_corr = float(finite_corr.min()) if finite_corr.size else float("inf")
                    msg = (
                        f"No feasible OMARS design met the satisfice thresholds {satisfice}. "
                        f"The best among {n_enumerated} enumerated design(s) reached "
                        f"d_efficiency={float(d_eff.max()):.3f} and "
                        f"max_second_order_correlation={best_corr:.3f}. Relax the thresholds "
                        "or widen n_runs_range."
                    )
                    raise ValueError(msg)
            kept_idx = np.flatnonzero(keep)
            local = _pick_exhaustive_winner(d_eff[kept_idx], a_opt[kept_idx], max_corr[kept_idx], selection_criterion)
            best = int(kept_idx[local])
            counts = count_matrix[best]
            coded = _foldover(np.repeat(pool, counts, axis=0))
            half_indices = [int(r) for r in np.repeat(np.arange(pool.shape[0]), counts)]
            candidates = [
                _Candidate(
                    coded=coded,
                    n_runs=2 * target_half + center_runs,
                    half_indices=half_indices,
                    d_efficiency=float(d_eff[best]),
                    a_optimality=float(a_opt[best]),
                    max_second_order_correlation=float(max_corr[best]),
                    solver_status="Enumerated",
                )
            ]
            if verify and not is_omars(coded, tol=tol):  # pragma: no cover - defensive
                msg = "Exhaustive OMARS enumeration produced a design that failed the is_omars re-check."
                raise RuntimeError(msg)
            report.search_mode = "exhaustive"
            report.enumerated_designs = n_enumerated
            report.feasible_designs = n_enumerated
            exhausted = True

    if not exhausted and target_half is not None:
        report.run_size = 2 * target_half + center_runs
        report.search_mode = "multistart"

        # A plain feasibility solve guarantees at least one design at this size,
        # even when n_restarts is 0 or every random objective turns out degenerate.
        coded, status, indices = _solve(n_half=target_half)
        if coded is not None:
            _record(coded, indices, status)

        # Randomized-objective multistart.  Each random linear objective sends the
        # solver to a different vertex of the OMARS-feasibility polytope, so the
        # retained set spans the high-D-efficiency / low-A members a pure
        # feasibility search never reaches.  Deterministic for a fixed random_seed.
        # Early-stop once the feasible set stops yielding new designs.
        rng = np.random.default_rng(random_seed)
        stall = 0
        for _ in range(n_restarts):
            if stall >= _RESTART_PATIENCE:
                break
            coded, status, indices = _solve(n_half=target_half, objective=rng.standard_normal(pool.shape[0]))
            if coded is not None and _record(coded, indices, status):
                stall = 0
            else:
                stall += 1

    if not exhausted:
        report.feasible_designs = len(candidates)
    if not candidates:
        target = f"n_runs={n_runs}" if n_runs is not None else f"n_runs_range={n_runs_range}"
        msg = (
            f"No feasible OMARS design was found for {target}. "
            "Try a different n_runs (n_runs - center_runs must be a positive even number), "
            "or a wider n_runs_range."
        )
        raise ValueError(msg)

    # Satisfice first (drop designs below the acceptability thresholds), then
    # pick from the survivors by dominance / the chosen criterion.  On the
    # exhaustive path the thresholds were already applied to the full
    # enumeration, so the single retained winner passes them by construction.
    eligible = candidates
    if satisfice and not exhausted:
        eligible = _satisfice(candidates, satisfice)
        if not eligible:
            best_d = max(c.d_efficiency for c in candidates)
            best_corr = min(c.max_second_order_correlation for c in candidates)
            msg = (
                f"No feasible OMARS design met the satisfice thresholds {satisfice}. "
                f"The best among {len(candidates)} candidate(s) reached d_efficiency={best_d:.3f} and "
                f"max_second_order_correlation={best_corr:.3f}. Relax the thresholds, raise max_candidates, "
                "or widen n_runs_range."
            )
            raise ValueError(msg)

    winner = _select(eligible, selection_criterion)
    report.run_size = winner.n_runs
    metadata = {
        "family": "omars_ilp",
        "construction": "foldover_ilp_selection",
        "foldover": True,
        "half_pool_size": pool.shape[0],
        "n_runs_selected": winner.n_runs,
        "sizing_model": model,
        "model_params": n_params,
        "full_second_order_params": _full_second_order_params(n_factors),
        "model_rank": _model_rank(winner.coded, model),
        "expected_error_df": winner.n_runs - _model_rank(winner.coded, model),
        "min_runs_for_model": 2 * _min_half_runs(n_factors, model) + center_runs,
        "sparsity": _sparsity(winner.coded),
        "selection_criterion": selection_criterion,
        "satisfice": dict(satisfice) if satisfice else None,
        "d_efficiency": winner.d_efficiency,
        "a_optimality": winner.a_optimality,
        "max_second_order_correlation": winner.max_second_order_correlation,
        "solver": (solver_options or {}).get("solver", "pulp"),
        "solver_status": winner.solver_status,
        "search_mode": report.search_mode,
        "omars_verified": is_omars(winner.coded, tol=tol),
        "omars_search": report,
    }
    return winner.coded, metadata


def generate_omars(  # noqa: PLR0913
    factors: list[Factor],
    *,
    n_runs: int | None = None,
    n_runs_range: tuple[int, int] | None = None,
    selection_criterion: str = "dominance",
    satisfice: dict[str, float] | None = None,
    center_runs: int = 1,
    n_restarts: int = 50,
    max_candidates: int = 6,
    model: str = "full_second_order",
    solver_options: dict[str, Any] | None = None,
    tol: float = 1e-9,
    random_seed: int = 42,
    verify: bool = True,
) -> DesignResult:
    """Generate a foldover OMARS design by exhaustive or integer-programming run selection.

    Builds a three-level OMARS design large enough to leave error degrees of
    freedom for the chosen analysis *model*, so it can be analysed with
    :func:`process_improve.experiments.analyze_omars`.  The design is a foldover
    ``[H; -H; 0]`` (half-runs, their mirrors, and a centre run) plus any further
    centre runs, for ``2*h + center_runs`` runs in total.  Regardless of
    *model*, the design is a genuine OMARS design: the main effects stay
    orthogonal to every second-order term (quadratics and interactions alike).

    Parameters
    ----------
    factors : list[Factor]
        At least three continuous factors.
    n_runs : int, optional
        Exact total run size of the returned design, centre runs included.
        ``n_runs - center_runs`` must be a positive even number (the half-runs
        and their mirrors), and *n_runs* must exceed the number of parameters in
        the chosen *model* (``1 + 2k + k(k-1)/2`` for ``"full_second_order"``,
        ``1 + 2k`` for ``"main_quadratic"``).  If ``None`` a size is chosen
        automatically.
    n_runs_range : tuple[int, int], optional
        Inclusive ``(min, max)`` total-run-size window to search when *n_runs*
        is ``None``; the smallest feasible size is used.
    selection_criterion : {"dominance", "d_efficiency", "min_second_order_correlation", "a_optimal"}
        How to choose among the feasible designs.  When the design class at the
        chosen size is small enough (currently up to four factors at moderate
        sizes), the search enumerates it exhaustively and the selection is exact
        for the stated objective; the metadata reports
        ``search_mode="exhaustive"``.  Otherwise the criterion selects the best
        design among those found by the randomized multistart
        (``search_mode="multistart"``), which may miss the optimum.
        ``"dominance"`` (default) keeps the Pareto front on D-efficiency and the
        maximum second-order correlation, then prefers the smallest, most
        efficient design.  ``"a_optimal"`` selects the design with the lowest
        summed coefficient variance ``trace((X'X)^-1)`` of the sizing model
        (lower prediction variance on average), which is the natural choice when
        the design is judged on precision rather than on aliasing.  A design
        containing a constant second-order column (a term the design cannot
        estimate) scores ``inf`` on the correlation metric, so it is never
        selected by ``"min_second_order_correlation"`` when an alternative with
        every term present exists.
    satisfice : dict, optional
        Acceptability thresholds applied *before* selection: a design is kept
        only if it clears every threshold.  Supported keys are
        ``"d_efficiency"`` (a minimum, higher is better) and
        ``"max_second_order_correlation"`` (a maximum, lower is better), for
        example ``{"d_efficiency": 5.0, "max_second_order_correlation": 0.7}``.
        A ``ValueError`` is raised if no enumerated design meets the thresholds.
    center_runs : int, optional
        Number of centre runs in the design (at least one; the foldover already
        contributes one).  Centre runs count towards *n_runs*: asking for
        ``n_runs=17, center_runs=3`` returns 17 rows, 3 of them centre runs.
        Default 1.
    n_restarts : int, optional
        Number of randomized-objective ILP solves used to search for a
        high-quality design.  Each restart drives the solver to a different
        feasible OMARS design; the best one (by *selection_criterion*) is kept.
        Higher values explore more of the feasible set and approach the
        catalogue-optimal designs more closely, at a roughly linear cost in
        runtime.  The search early-stops once the feasible set stops yielding new
        designs, so small factor counts finish quickly regardless.  Default 50,
        which reaches catalogue-competitive D-efficiency for up to seven factors.
        Deterministic for a fixed *random_seed*.
    max_candidates : int, optional
        Legacy alias retained for backward compatibility.  It now sets a floor on
        *n_restarts* (the effective restart budget is ``max(n_restarts,
        max_candidates)``), so calls that raised it to enumerate more designs
        still explore at least that many.  Default 6.
    model : {"full_second_order", "main_quadratic"}, optional
        The analysis model the design is sized for.  ``"full_second_order"``
        (default) leaves room for every two-factor interaction, so the smallest
        feasible design must exceed ``1 + 2k + k(k-1)/2`` runs.
        ``"main_quadratic"`` sizes for only the main effects and pure quadratics
        (``1 + 2k`` parameters), admitting smaller designs such as a
        thirteen-run, four-factor OMARS; the interactions are still present in
        the design and confined to the second-order block, they are simply not
        part of the model the run count is chosen for.  The D-efficiency reported
        in the metadata is read from this same model.
    solver_options : dict, optional
        Passed to the solver: ``{"msg": bool, "time_limit": int seconds}``.
    tol : float, optional
        Tolerance for the floating-point :func:`is_omars` re-check.
    random_seed : int, optional
        Seed for both the randomized-objective search (which design is found) and
        the run-order randomisation of the returned design.  A fixed seed makes
        the whole call reproducible.
    verify : bool, optional
        When ``True`` (default) every selected design is re-checked with
        :func:`is_omars` before it is accepted.

    Returns
    -------
    DesignResult
        The OMARS design, with ILP provenance and search diagnostics under
        ``metadata`` (``family``, ``sparsity``, ``omars_search`` report, ...).

    Raises
    ------
    ValueError
        If fewer than three factors are given, the factor count exceeds the
        combinatorial cap, *model* is not recognised, *n_runs* is too small or
        incompatible with *center_runs*, or no feasible design is found.
    ImportError
        If PuLP (the ``ilp`` extra) is not installed.

    Examples
    --------
    >>> from process_improve.experiments import Factor, generate_omars, analyze_omars
    >>> factors = [Factor(name=n, low=-1, high=1) for n in "ABCDE"]
    >>> result = generate_omars(factors)              # doctest: +SKIP
    >>> result.metadata["omars_verified"]             # doctest: +SKIP
    True
    """
    from process_improve.experiments.designs_utils import build_design_result  # noqa: PLC0415
    from process_improve.experiments.factor import FactorType  # noqa: PLC0415

    categorical = [f.name for f in factors if f.type == FactorType.categorical]
    if categorical:
        raise ValueError(
            "OMARS designs require continuous factors; got categorical factor(s): "
            f"{categorical}. OMARS is built from three-level quantitative contrasts. "
            "For a mixed-level study use an optimal design (generate_design(..., "
            "design_type='i_optimal'))."
        )

    if center_runs < 1:
        raise ValueError("center_runs must be at least 1.")

    coded, metadata = _search_best_omars(
        factors,
        n_runs=n_runs,
        n_runs_range=n_runs_range,
        selection_criterion=selection_criterion,
        satisfice=satisfice,
        n_restarts=max(n_restarts, max_candidates),
        model=model,
        solver_options=solver_options,
        tol=tol,
        verify=verify,
        random_seed=random_seed,
        center_runs=center_runs,
    )
    return build_design_result(
        coded_matrix=coded,
        factors=factors,
        design_type="omars",
        n_center_points=center_runs - 1,
        random_seed=random_seed,
        metadata=metadata,
    )


def _dispatch_omars_ilp(factors: list[Factor], **kwargs: Any) -> tuple[np.ndarray, dict]:  # noqa: ANN401
    """Registry handler: ``generate_design(design_type="omars_ilp", budget=N)``.

    Returns the raw coded matrix (with its single centre run) and metadata;
    :func:`process_improve.experiments.generate_design` handles post-processing.
    """
    return _search_best_omars(
        factors,
        n_runs=kwargs.get("budget"),
        n_runs_range=None,
        selection_criterion="dominance",
        satisfice=None,
        n_restarts=50,
        model="full_second_order",
        solver_options=None,
        tol=1e-9,
        verify=True,
        random_seed=42,
        center_runs=1,
    )
