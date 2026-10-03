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

and the centre run puts every factor at its middle level.  The condition that
is *not* automatic is the mutual orthogonality of the main effects, which is
linear in the binary "include this half-run" variables ``s_r``: for each pair
``i < j``, ``sum_r (x[r,i] x[r,j]) s_r = 0``.  Each factor must also reach an
outer level in some half-run, ``sum_r |x[r,i]| s_r >= 1``; otherwise its column
is all zeros, orthogonal but not three-level.  The run count is
``2 * sum_r s_r + 1``.

So the ILP selects a half-design from the ``(3**k - 1) / 2`` distinct non-mirror
three-level runs subject to a handful of linear constraints - ``k(k-1)/2``
equalities and ``k`` coverage rows - which keeps it tractable up to seven
factors.  The ILP is solved with HiGHS, through ``scipy.optimize.milp``.
Because the coefficients are integers, the constraints are exact; every
selection the solver returns is re-checked exactly, and the floating-point
:func:`is_omars` re-check only guards against mistakes.  A pure feasibility solve, however, returns an
arbitrary OMARS design that is usually far from the most efficient member.  To
search for a high-quality design the solve is repeated with random linear
objectives (a multistart): each random objective steers the solver towards a
different feasible design, so the retained designs span the high-D-efficiency
/ low-A members.  Each of these solves stops after a fixed number of
branch-and-bound nodes, which ends it at the same point on every run, so the
search is reproducible for a fixed seed.  Designs from which the sizing model
cannot be estimated are set aside.  The best of the rest is then
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
import logging
import math
import operator
import os
import re
import threading
import time
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, OptimizeResult, milp

from process_improve._random import check_random_state
from process_improve.experiments.designs_omars import _second_order_terms, is_omars

logger = logging.getLogger(__name__)

# HiGHS thread-pool bookkeeping for _run_milp: whether this thread's pool was
# already started multi-threaded by other code, and whether this process is a
# fork child, whose inherited pool has no threads behind it.
_thread_state = threading.local()
_fork_state = {"in_child": False}


def _after_fork_in_child() -> None:
    _fork_state["in_child"] = True
    _thread_state.multi_threaded = False


if hasattr(os, "register_at_fork"):  # absent on Windows
    os.register_at_fork(after_in_child=_after_fork_in_child)

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
# Relative eigenvalue threshold below which an enumerated design's model Gram
# matrix counts as singular (rank-deficient).
_SINGULAR_RTOL = 1e-9

# Keys accepted in ``solver_options``, and their defaults.  ``node_limit`` caps
# the branch-and-bound nodes of each randomized-objective solve.  A node budget
# ends a solve at the same point on every run, so a fixed seed reproduces the
# design; a wall-clock limit would not.  ``time_limit`` is only a safety cap.
_SOLVER_OPTION_KEYS = ("msg", "time_limit", "node_limit")
_DEFAULT_TIME_LIMIT = 60.0
_DEFAULT_NODE_LIMIT = 100
# HiGHS stores the node limit in a C int.
_MAX_NODE_LIMIT = 2**31 - 1

# ``solver_status`` labels.  ``"Node limit"`` and ``"Time limit"`` carry the best
# design found when the limit stopped the solve, so they are usable designs
# whose optimality is not proven.
_STATUS_OPTIMAL = "Optimal"
_STATUS_NODE_LIMIT = "Node limit"
_STATUS_TIME_LIMIT = "Time limit"
_STATUS_INFEASIBLE = "Infeasible"
_STATUS_NOT_SOLVED = "Not Solved"
_STATUS_ENUMERATED = "Enumerated"
# HiGHS model-status codes, as scipy reports them in ``"(HiGHS Status N: ...)"``.
# The scipy status integer alone is ambiguous: a node limit is status 1 on
# scipy < 1.15 and status 4 from 1.15 on, where 4 also means "solver error".
_HIGHS_STATUS_LABELS = {
    7: _STATUS_OPTIMAL,
    8: _STATUS_INFEASIBLE,
    9: _STATUS_INFEASIBLE,  # "unbounded or infeasible"; the binaries are bounded
    13: _STATUS_TIME_LIMIT,
    14: _STATUS_NODE_LIMIT,  # iteration limit; HiGHS < 1.8 reports the node limit this way
    16: _STATUS_NODE_LIMIT,  # solution limit; HiGHS >= 1.8 reports the node limit this way
}
_HIGHS_STATUS_PATTERN = re.compile(r"HiGHS Status (\d+)")
# HiGHS model status 0 ("Not Set"): the solve never ran.  See _run_milp.
_HIGHS_NOT_SET = 0


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
        Number of distinct verified OMARS designs found (enumerated, on the
        exhaustive path).  The ``rank_deficient_designs`` among them are set
        aside; the rest are ranked.
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
    node_limit : int or None
        Branch-and-bound node budget of each randomized-objective solve
        (``None`` means unlimited).
    time_limit : float
        Wall-clock cap, in seconds, on each ILP solve.
    node_limited_solves : int
        Number of ILP solves that the node budget stopped before optimality
        was proven.  These are expected and deterministic.
    time_limited_solves : int
        Number of ILP solves that the wall-clock cap stopped.  When this is
        nonzero the search depends on machine speed, so a fixed
        ``random_seed`` may not reproduce the design.
    rank_deficient_designs : int
        Number of distinct OMARS designs found, but set aside because the
        sizing model is not estimable from them (rank below the parameter
        count).
    size_proven_minimal : bool or None
        ``True`` when the returned run size was proven to be the smallest
        feasible one in the window.  ``False`` when a limit stopped the
        minimise-size solve first, or when no design at the proven minimum could
        estimate the model and the search moved up.  ``None`` when the run size
        was pinned by ``n_runs`` or lies beyond the distinct half-runs (see
        *n_runs_range* in :func:`generate_omars`).
    run_sizes_searched : int
        Number of run sizes searched.  More than one means that no design at
        the smallest feasible size could estimate the sizing model, so the
        search moved up the window.
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
    node_limit: int | None = _DEFAULT_NODE_LIMIT
    time_limit: float = _DEFAULT_TIME_LIMIT
    node_limited_solves: int = 0
    time_limited_solves: int = 0
    rank_deficient_designs: int = 0
    size_proven_minimal: bool | None = None
    run_sizes_searched: int = 0


@dataclass(frozen=True)
class _SolverSettings:
    """Validated ``solver_options``: see :func:`_solver_settings`."""

    msg: bool
    time_limit: float
    node_limit: int | None


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
    ``(min, max)`` half count) sets the size constraint; *n_half* wins if both
    are given.  The returned design has ``2 * h + 1`` runs for ``h`` selected
    half-runs.  The integer program is solved with HiGHS, through
    ``scipy.optimize.milp``.

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
        solver towards a different feasible OMARS design, which is how
        :func:`generate_omars` samples diverse, high-quality designs.  Takes
        precedence over *minimize_size*.  Only solves with an *objective* are
        subject to the ``node_limit`` solver option.
    exclude_solutions : list[list[int]], optional
        Previously found half-index sets to forbid via no-good cuts.
    solver_options : dict, optional
        Any of the keys:

        * ``"msg"`` (bool, default ``False``): print the HiGHS log to standard
          output.
        * ``"time_limit"`` (float, default ``60.0``): wall-clock cap in seconds
          on the solve; ``math.inf`` removes it.  A solve stopped by this cap
          depends on machine speed.
        * ``"node_limit"`` (int or None, default ``100``): branch-and-bound
          node budget for a solve with an *objective*.  The budget stops the
          solve at the same point on every run, so results stay
          reproducible; ``None`` solves to proven optimality.

    Returns
    -------
    tuple
        ``(design or None, solver_status, chosen_half_indices)``.  ``None`` means
        the solver returned no feasible selection.  *solver_status* is one of
        ``"Optimal"``, ``"Node limit"`` or ``"Time limit"`` (a limit stopped the
        solve; the best selection found is returned), ``"Infeasible"``, or
        ``"Not Solved"``.

    Raises
    ------
    TypeError
        If *solver_options* is not a dict, or one of its values has the wrong
        type.
    ValueError
        If neither *n_half* nor *half_bounds* is given, or *solver_options* has
        an unknown key or an out-of-range value.
    """
    settings = _solver_settings(solver_options)
    size_bounds = _size_bounds(n_half, half_bounds)
    n_candidates = half_pool.shape[0]
    _check_exclusions(exclude_solutions, n_candidates)

    options: dict[str, Any] = {"disp": settings.msg, "time_limit": settings.time_limit}
    if objective is not None:
        cost = np.asarray(objective, dtype=float)
        if settings.node_limit is not None:
            options["node_limit"] = settings.node_limit
    else:
        # Minimise-size and feasibility solves close at the root node, so they
        # need no node budget, and a node-limited minimise-size solve could not
        # prove that its size is the smallest.
        cost = np.full(n_candidates, 1.0 if minimize_size else 0.0)

    constraints = _selection_constraints(half_pool, size_bounds, exclude_solutions)
    result = _run_milp(cost, constraints, options)
    status = _status_label(result)
    logger.debug(
        "OMARS ILP solve: %s, %s nodes, %d candidates.", status, getattr(result, "mip_node_count", None), n_candidates
    )
    if result.x is None:
        return None, status, []
    # The coverage rows make an empty selection infeasible, so chosen is never
    # empty here; _check_selection would reject one.
    chosen = [int(r) for r in np.flatnonzero(result.x > 0.5)]
    _check_selection(half_pool, chosen, size_bounds, exclude_solutions)
    return _foldover(half_pool[chosen]), status, chosen


def _solver_settings(solver_options: dict[str, Any] | None) -> _SolverSettings:
    """Validate *solver_options* and fill in the defaults."""
    if solver_options is None:
        solver_options = {}
    if not isinstance(solver_options, dict):
        msg = f"solver_options must be a dict or None, got {type(solver_options).__name__}."
        raise TypeError(msg)
    unknown = sorted(str(key) for key in set(solver_options) - set(_SOLVER_OPTION_KEYS))
    if unknown:
        msg = f"solver_options accepts only the keys {list(_SOLVER_OPTION_KEYS)}, got unknown {unknown}."
        raise ValueError(msg)
    msg_flag = solver_options.get("msg", False)
    if not isinstance(msg_flag, (bool, np.bool_)):
        msg = f"solver_options['msg'] must be True or False, got {msg_flag!r}."
        raise TypeError(msg)
    return _SolverSettings(
        msg=bool(msg_flag),
        time_limit=_checked_time_limit(solver_options.get("time_limit", _DEFAULT_TIME_LIMIT)),
        node_limit=_checked_node_limit(solver_options.get("node_limit", _DEFAULT_NODE_LIMIT)),
    )


def _checked_time_limit(value: object) -> float:
    """Return ``solver_options["time_limit"]`` as positive seconds (``inf`` allowed)."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        msg = f"solver_options['time_limit'] must be a number of seconds, got {value!r}."
        raise TypeError(msg)
    try:
        seconds = float(value)
    except OverflowError:  # an int too large for a float: no practical limit
        seconds = math.inf
    if math.isnan(seconds) or seconds <= 0:
        msg = f"solver_options['time_limit'] must be positive, got {value!r}."
        raise ValueError(msg)
    return seconds


def _checked_node_limit(value: object) -> int | None:
    """Return ``solver_options["node_limit"]`` as a positive int, or ``None`` for no limit."""
    if value is None:
        return None
    if isinstance(value, bool):
        msg = f"solver_options['node_limit'] must be an integer or None, got {value!r}."
        raise TypeError(msg)
    try:
        # operator.index accepts Python and NumPy integers and rejects floats,
        # which HiGHS would refuse with an opaque error.
        nodes = operator.index(value)  # type: ignore[arg-type]
    except TypeError:
        msg = f"solver_options['node_limit'] must be an integer or None, got {value!r}."
        raise TypeError(msg) from None
    if not 1 <= nodes <= _MAX_NODE_LIMIT:
        msg = f"solver_options['node_limit'] must be between 1 and {_MAX_NODE_LIMIT}, got {value!r}."
        raise ValueError(msg)
    return nodes


def _check_exclusions(exclude_solutions: list[list[int]] | None, n_candidates: int) -> None:
    """Each excluded selection must be a non-empty list of pool row indices."""
    for excluded in exclude_solutions or []:
        rows = np.asarray(excluded)
        valid = rows.ndim == 1 and rows.size > 0 and np.issubdtype(rows.dtype, np.integer)
        if not valid or rows.min() < 0 or rows.max() >= n_candidates:
            msg = (
                "exclude_solutions entries must be non-empty lists of half-pool row indices in "
                f"[0, {n_candidates}), got {excluded!r}."
            )
            raise ValueError(msg)


def _size_bounds(n_half: int | None, half_bounds: tuple[int, int] | None) -> tuple[int, int]:
    """Inclusive ``(min, max)`` half-run count from *n_half* or *half_bounds*."""
    if n_half is not None:
        return n_half, n_half
    if half_bounds is not None:
        low, high = half_bounds
        return low, high
    raise ValueError("solve_omars_ilp requires either n_half or half_bounds.")


def _selection_constraints(
    half_pool: np.ndarray, size_bounds: tuple[int, int], exclude_solutions: list[list[int]] | None
) -> list[LinearConstraint]:
    """Linear constraints on the binary "include this half-run" variables.

    The foldover makes balance and clear-of-second-order automatic, and its
    centre run puts every factor at its middle level.  Two OMARS conditions
    remain, and both are linear in ``s_r``:

    * main-effect orthogonality: each factor pair ``i < j`` contributes the
      equality ``sum_r x[r, i] x[r, j] s_r = 0``;
    * every factor reaches an outer level: ``sum_r |x[r, i]| s_r >= 1``, or its
      column is all zeros, which is trivially orthogonal but not three-level.

    The coefficients are integers, so the constraints are exact.  Then come the
    size window and one no-good cut per excluded selection (see
    :func:`_no_good_row`).
    """
    n_candidates, n_factors = half_pool.shape
    constraints = []
    pairs = list(itertools.combinations(range(n_factors), 2))
    if pairs:
        orthogonality = np.array([half_pool[:, i] * half_pool[:, j] for i, j in pairs])
        constraints.append(LinearConstraint(orthogonality, 0.0, 0.0))
    constraints.append(LinearConstraint(np.abs(half_pool).T, 1.0, np.inf))
    constraints.append(LinearConstraint(np.ones((1, n_candidates)), *size_bounds))
    for excluded in exclude_solutions or []:
        row = _no_good_row(n_candidates, excluded)
        constraints.append(LinearConstraint(row[np.newaxis, :], -np.inf, len(excluded) - 1))
    return constraints


def _no_good_row(n_candidates: int, excluded: list[int]) -> np.ndarray:
    """Coefficients of the no-good cut ``sum_{r in excluded} s_r <= len(excluded) - 1``."""
    # np.add.at counts a repeated index twice, as the sum over *excluded* does.
    row = np.zeros(n_candidates)
    np.add.at(row, np.asarray(excluded, dtype=int), 1.0)
    return row


def _run_milp(cost: np.ndarray, constraints: list[LinearConstraint], options: dict[str, Any]) -> OptimizeResult:
    """Solve the binary selection problem with HiGHS on a single thread.

    HiGHS keeps one thread pool per calling thread, sized at that thread's
    first solve, and a process forked after a multi-threaded HiGHS solve hangs
    in its own next solve.  Solving with ``threads=1`` keeps this module from
    creating such a pool.  If earlier code on this thread already started HiGHS
    with more threads, a ``threads=1`` request is refused with model status
    "Not Set" before any work is done:

    * in the process that owns that pool, the solve is repeated without
      ``threads``, and later solves on the thread skip the refused attempt;
    * in a forked child, the inherited pool has no threads behind it and a
      solve on it would hang, so the pool is reset first (see
      :func:`_reset_highs_scheduler`).  If that is not possible the "Not Set"
      result is returned, which reads as ``"Not Solved"``.
    """
    n_candidates = cost.shape[0]

    def solve(extra: dict[str, Any]) -> OptimizeResult:
        # milp pops keys out of the dict it is given, so pass a fresh one.
        with warnings.catch_warnings():
            # scipy forwards "threads" to HiGHS but warns that it does not know it.
            warnings.filterwarnings("ignore", message="Unrecognized options detected", category=RuntimeWarning)
            return milp(
                cost,
                integrality=np.ones(n_candidates),
                bounds=Bounds(0, 1),
                constraints=constraints,
                options={**options, **extra},
            )

    if getattr(_thread_state, "multi_threaded", False):
        return solve({})
    result = solve({"threads": 1})
    if result.x is not None or _highs_status(result) != _HIGHS_NOT_SET:
        return result
    if _fork_state["in_child"]:
        if _reset_highs_scheduler():
            return solve({"threads": 1})
        logger.warning("HiGHS's thread pool was inherited from the parent process and cannot be reset; not solving.")
        return result
    logger.info("HiGHS already runs multi-threaded on this thread; solving without threads=1 from now on.")
    _thread_state.multi_threaded = True
    return solve({})


def _reset_highs_scheduler() -> bool:
    """Discard the HiGHS thread pool a forked child inherited; return True on success.

    The reset lives in SciPy's private HiGHS bindings (present in SciPy 1.15 to
    1.18), so its absence is handled rather than assumed.
    """
    try:
        from scipy.optimize._highspy import _core  # noqa: PLC0415
    except ImportError:
        return False
    highs = getattr(_core, "_Highs", None)
    reset = getattr(highs, "resetGlobalScheduler", None)
    if reset is None:
        return False
    reset(False)
    return True


def _highs_status(result: OptimizeResult) -> int | None:
    """HiGHS model-status code quoted in a ``milp`` result message, if any."""
    match = _HIGHS_STATUS_PATTERN.search(str(getattr(result, "message", "")))
    return int(match.group(1)) if match else None


def _status_label(result: OptimizeResult) -> str:
    """Map a ``milp`` result to a ``solver_status`` label."""
    code = _highs_status(result)
    if code is not None:
        return _HIGHS_STATUS_LABELS.get(code, _STATUS_NOT_SOLVED)
    return _STATUS_OPTIMAL if result.status == 0 else _STATUS_NOT_SOLVED


def _check_selection(
    half_pool: np.ndarray, chosen: list[int], size_bounds: tuple[int, int], exclude_solutions: list[list[int]] | None
) -> None:
    """Re-check a solver selection against the constraints it was solved under.

    The solver works to floating-point tolerances; this check is exact for the
    integer-coded pool, so a selection that slipped through a tolerance is
    caught here instead of reaching the caller as a non-OMARS design.
    """
    selected = half_pool[chosen]
    gram = selected.T @ selected
    off_diagonal = gram - np.diag(np.diag(gram))
    tolerance = 1e-9 * max(1.0, float(np.abs(selected).sum()))
    problems = []
    if np.abs(off_diagonal).max(initial=0.0) > tolerance:
        problems.append("the main effects are not orthogonal")
    if not np.all(np.abs(selected).sum(axis=0) > 0):
        problems.append("a factor never leaves its middle level")
    if not size_bounds[0] <= len(chosen) <= size_bounds[1]:
        problems.append(f"{len(chosen)} half-runs lie outside {size_bounds}")
    problems.extend(
        f"an excluded selection {sorted(excluded)} was returned"
        for excluded in exclude_solutions or []
        if _no_good_row(half_pool.shape[0], excluded)[chosen].sum() > len(excluded) - 1
    )
    if problems:
        msg = f"internal: the HiGHS selection violates its constraints ({'; '.join(problems)}). Please report this."
        raise RuntimeError(msg)


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
        # A fixed relative threshold, not the caller's is_omars tolerance: the
        # Gram matrix is integer, so a singular one has eigenvalues at round-off.
        singular = eig[:, 0] <= _SINGULAR_RTOL * eig[:, -1]
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
    solver_settings = _solver_settings(solver_options)
    report = OmarsSearchReport(
        n_factors=n_factors,
        half_pool_size=pool.shape[0],
        n_restarts=n_restarts,
        node_limit=solver_settings.node_limit,
        time_limit=solver_settings.time_limit,
    )
    candidates: list[_Candidate] = []
    seen: set[frozenset[int]] = set()
    extra_centers = np.zeros((center_runs - 1, n_factors))
    half_window: tuple[int, int] | None = None
    last_status = _STATUS_NOT_SOLVED

    def _solve(**solve_kwargs: Any) -> tuple[np.ndarray | None, str, list[int]]:  # noqa: ANN401
        nonlocal last_status
        started = time.perf_counter()
        result = solve_omars_ilp(pool, solver_options=solver_options, **solve_kwargs)
        report.ilp_iterations += 1
        report.total_solve_seconds += time.perf_counter() - started
        last_status = result[1]
        if last_status == _STATUS_NODE_LIMIT:
            report.node_limited_solves += 1
        elif last_status == _STATUS_TIME_LIMIT:
            report.time_limited_solves += 1
            logger.info("An OMARS ILP solve stopped at its %.3g s time limit.", solver_settings.time_limit)
        return result

    def _record(coded: np.ndarray, indices: list[int], status: str) -> bool:
        """Verify and retain a distinct, estimable OMARS design; return True if it was new."""
        key = frozenset(indices)
        if key in seen:
            return False
        seen.add(key)
        if verify and not is_omars(coded, tol=tol):
            return False
        # Score the design the caller will actually receive: the foldover plus
        # the extra centre runs appended during post-processing.
        scored = np.vstack([coded, extra_centers])
        if _model_rank(scored, model) < n_params:
            # A valid OMARS design that cannot fit the sizing model.  Its
            # D-efficiency (0) and correlation (often finite) would let some
            # criteria crown it, so it never enters the ranking.
            report.rank_deficient_designs += 1
            return False
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
    # the window (the minimize-size solution becomes the first candidate).  The
    # solver's selection has already passed the exact constraint check in
    # solve_omars_ilp, so its size is a genuinely feasible one.
    enum_cap = _ENUM_MAX_HALF.get(n_factors, 0)
    if target_half is not None and target_half > max(pool.shape[0], enum_cap):
        msg = (
            f"n_runs={n_runs} needs {target_half} half-runs, but there are only {pool.shape[0]} distinct "
            f"three-level half-runs for {n_factors} factors, and repeating half-runs is supported only "
            + (
                f"up to {enum_cap} half-runs ({2 * enum_cap + center_runs} runs). "
                if enum_cap
                else "for three and four factors. "
            )
            + "Use a smaller n_runs."
        )
        raise ValueError(msg)
    probed_half: int | None = None
    probe_rank_deficient = 0
    requested_high = 0
    if n_runs is None and n_runs_range is not None and n_runs_range[0] > n_runs_range[1]:
        msg = f"n_runs_range must be (min, max) with min <= max, got {tuple(n_runs_range)}."
        raise ValueError(msg)
    if target_half is None:
        half_window = _half_bounds(n_runs_range, n_params, pool.shape[0], _min_half_runs(n_factors, model), center_runs)
        requested_high = half_window[0] + 6 if n_runs_range is None else (n_runs_range[1] - center_runs) // 2
        if n_runs_range is not None and requested_high < half_window[0]:
            msg = (
                f"n_runs_range={tuple(n_runs_range)} ends below {2 * half_window[0] + center_runs} runs, the smallest "
                f"size at which the {model} model is estimable with error degrees of freedom for "
                f"{n_factors} factors and center_runs={center_runs}."
            )
            raise ValueError(msg)
    if target_half is None and half_window is not None and half_window[0] > pool.shape[0]:
        # Every size in the window needs more distinct half-runs than the pool
        # has, so a selection without repeats cannot reach it; only the
        # exhaustive search, which repeats half-runs, can.
        high = min(requested_high, enum_cap)
        if high < half_window[0]:
            msg = (
                f"The run sizes in n_runs_range={n_runs_range} need more than the {pool.shape[0]} distinct "
                f"three-level half-runs for {n_factors} factors, and repeating half-runs is not supported "
                "there. Ask for fewer runs."
            )
            raise ValueError(msg)
        target_half = half_window[0]
        half_window = (half_window[0], high)
    elif target_half is None and half_window is not None:
        coded, status, indices = _solve(half_bounds=half_window, minimize_size=True)
        if coded is not None:
            target_half = probed_half = len(indices)
            report.size_proven_minimal = status == _STATUS_OPTIMAL
            if not report.size_proven_minimal:
                logger.info(
                    "The OMARS minimise-size solve stopped early (%s); %d runs may not be the smallest.",
                    status,
                    2 * target_half + center_runs,
                )
            _record(coded, indices, status)
            probe_rank_deficient = report.rank_deficient_designs

    # Search the target size.  When the size was chosen automatically and no
    # design there can estimate the sizing model, move up one half-run at a
    # time within the window: the smallest feasible size is not always the
    # smallest usable one.  A pinned n_runs is searched alone.
    exhausted = False
    if target_half is None:
        sizes: list[int] = []
    elif n_runs is not None or half_window is None:
        sizes = [target_half]
    else:
        sizes = list(range(target_half, half_window[1] + 1))
    for half in sizes:
        if half != sizes[0]:
            logger.info(
                "No OMARS design at %d runs can estimate the %s model; trying %d runs.",
                2 * half - 2 + center_runs,
                model,
                2 * half + center_runs,
            )
        target_half = half
        report.run_sizes_searched += 1
        target = _describe_target(n_runs, 2 * half + center_runs)

        # Exhaustive path: when the design class at this size is small enough,
        # enumerate every feasible half-design multiset (replication allowed) and
        # pick the winner exactly.  The binary multistart below cannot even reach
        # designs that repeat a half-run, so this is what makes the selection
        # criteria live up to their names (issues #497, #498, #499).
        if half <= _ENUM_MAX_HALF.get(n_factors, 0):
            count_matrix, overflow = _enumerate_feasible_counts(pool, half, _ENUM_MAX_LEAVES)
            if not overflow:
                # The enumeration replaces whatever the solver found at this
                # size, including the minimise-size design and its count.
                candidates.clear()
                if half == probed_half:
                    report.rank_deficient_designs -= probe_rank_deficient
                # Keep only designs in which every factor reaches an outer level,
                # the condition the ILP's coverage rows impose.
                count_matrix = count_matrix[(count_matrix @ np.abs(pool) > 0).all(axis=1)]
                n_enumerated = count_matrix.shape[0]
                report.search_mode = "exhaustive"
                report.enumerated_designs += n_enumerated
                report.feasible_designs += n_enumerated
                if n_enumerated == 0:
                    if len(sizes) > 1:
                        continue
                    msg = (
                        f"No feasible OMARS design exists at {target} with center_runs={center_runs} "
                        "(exhaustive enumeration). Try a different n_runs."
                    )
                    raise ValueError(msg)
                d_eff, a_opt, max_corr = _score_count_vectors(count_matrix, pool, center_runs, model, tol=tol)
                # d_eff is exactly 0 for a singular model matrix: such a design
                # cannot fit the sizing model and never enters the ranking.
                keep = d_eff > 0.0
                report.rank_deficient_designs += int(n_enumerated - keep.sum())
                if not keep.any():
                    continue
                if satisfice:
                    _satisfice([], satisfice)  # validate the threshold keys
                    d_min = satisfice.get("d_efficiency")
                    correlation_max = satisfice.get("max_second_order_correlation")
                    if d_min is not None:
                        keep &= d_eff >= d_min
                    if correlation_max is not None:
                        keep &= max_corr <= correlation_max
                    if not keep.any():
                        finite_corr = max_corr[(d_eff > 0.0) & np.isfinite(max_corr)]
                        best_corr = float(finite_corr.min()) if finite_corr.size else float("inf")
                        msg = (
                            f"No feasible OMARS design met the satisfice thresholds {satisfice}. "
                            f"The best among {n_enumerated} enumerated design(s) reached "
                            f"d_efficiency={float(d_eff.max()):.3f} and "
                            f"max_second_order_correlation={best_corr:.3f}. Relax the thresholds, "
                            "or ask for more runs (a larger n_runs, or a higher lower bound in n_runs_range)."
                        )
                        raise ValueError(msg)
                kept_idx = np.flatnonzero(keep)
                local = _pick_exhaustive_winner(
                    d_eff[kept_idx], a_opt[kept_idx], max_corr[kept_idx], selection_criterion
                )
                best = int(kept_idx[local])
                counts = count_matrix[best]
                coded = _foldover(np.repeat(pool, counts, axis=0))
                half_indices = [int(r) for r in np.repeat(np.arange(pool.shape[0]), counts)]
                candidates = [
                    _Candidate(
                        coded=coded,
                        n_runs=2 * half + center_runs,
                        half_indices=half_indices,
                        d_efficiency=float(d_eff[best]),
                        a_optimality=float(a_opt[best]),
                        max_second_order_correlation=float(max_corr[best]),
                        solver_status=_STATUS_ENUMERATED,
                    )
                ]
                if verify and not is_omars(coded, tol=tol):  # pragma: no cover - defensive
                    msg = "Exhaustive OMARS enumeration produced a design that failed the is_omars re-check."
                    raise RuntimeError(msg)
                exhausted = True
                break

        report.search_mode = "multistart"
        # A plain feasibility solve guarantees at least one design at this size,
        # even when n_restarts is 0 or every random objective turns out degenerate.
        coded, status, indices = _solve(n_half=half)
        if coded is not None:
            _record(coded, indices, status)

        # Randomized-objective multistart.  Each random linear objective steers
        # the solver towards a different feasible OMARS design, so the retained
        # set spans the high-D-efficiency / low-A members a pure feasibility
        # search never reaches.  The node budget ends each solve at the same
        # point on every run, so the search is deterministic for a fixed
        # random_seed.  Early-stop once the feasible set stops yielding new
        # designs.
        rng = np.random.default_rng(random_seed)
        stall = 0
        for _ in range(n_restarts):
            if stall >= _RESTART_PATIENCE:
                break
            coded, status, indices = _solve(n_half=half, objective=rng.standard_normal(pool.shape[0]))
            if coded is not None and _record(coded, indices, status):
                stall = 0
            else:
                stall += 1
        if candidates:
            break

    if not exhausted:
        report.feasible_designs = len(candidates) + report.rank_deficient_designs
    if not candidates:
        if target_half is None:
            window = half_window or (0, 0)
            target = (
                f"n_runs_range={tuple(n_runs_range)}"
                if n_runs_range is not None
                else _describe_target(n_runs, 2 * window[0] + center_runs, 2 * window[1] + center_runs)
            )
            msg = (
                f"No feasible OMARS design was found at {target} with center_runs={center_runs}: the "
                f"minimise-size solve ended with status {last_status!r}. {_no_design_advice(last_status)}"
            )
        elif report.rank_deficient_designs:
            exhausted_all = report.search_mode == "exhaustive"
            target = _describe_target(n_runs, 2 * sizes[0] + center_runs, 2 * sizes[-1] + center_runs)
            # An exhaustive search saw every design, so more restarts cannot help.
            advice = "Ask for more runs." if exhausted_all else "Raise n_restarts, or ask for more runs."
            msg = (
                f"{report.rank_deficient_designs} OMARS design(s) were found at {target}, but the {model} "
                f"model cannot be estimated from any of them (model matrix rank below {n_params}). {advice}"
            )
        else:
            target = _describe_target(n_runs, 2 * sizes[0] + center_runs, 2 * sizes[-1] + center_runs)
            msg = (
                f"No feasible OMARS design was found at {target} with center_runs={center_runs}: the last "
                f"solve ended with status {last_status!r}. {_no_design_advice(last_status)}"
            )
        raise ValueError(msg)

    if probed_half is not None and target_half != probed_half:
        # The smallest feasible size was proven, but the design comes from a
        # larger one, so the returned size is not proven minimal.
        report.size_proven_minimal = False

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
                f"max_second_order_correlation={best_corr:.3f}. Relax the thresholds, raise n_restarts, "
                "or ask for more runs (a larger n_runs, or a higher lower bound in n_runs_range)."
            )
            raise ValueError(msg)

    winner = _select(eligible, selection_criterion)
    report.run_size = winner.n_runs
    # Describe the design the caller receives: the foldover plus the extra centre runs.
    received = np.vstack([winner.coded, extra_centers])
    received_rank = _model_rank(received, model)
    metadata = {
        "family": "omars_ilp",
        "construction": "foldover_ilp_selection",
        "foldover": True,
        "half_pool_size": pool.shape[0],
        "n_runs_selected": winner.n_runs,
        "sizing_model": model,
        "model_params": n_params,
        "full_second_order_params": _full_second_order_params(n_factors),
        "model_rank": received_rank,
        "expected_error_df": winner.n_runs - received_rank,
        "min_runs_for_model": 2 * _min_half_runs(n_factors, model) + center_runs,
        "sparsity": _sparsity(received),
        "selection_criterion": selection_criterion,
        "satisfice": dict(satisfice) if satisfice else None,
        "d_efficiency": winner.d_efficiency,
        "a_optimality": winner.a_optimality,
        "max_second_order_correlation": winner.max_second_order_correlation,
        "solver": "highs",
        "solver_status": winner.solver_status,
        "search_mode": report.search_mode,
        "omars_verified": is_omars(received, tol=tol),
        "omars_search": report,
    }
    return winner.coded, metadata


def _no_design_advice(status: str) -> str:
    """Return what a caller can change when a solve ended with *status* and no design."""
    if status == _STATUS_INFEASIBLE:
        return (
            "No selection of distinct half-runs satisfies the OMARS constraints at this size; "
            "try a different n_runs or n_runs_range."
        )
    return "Raise solver_options['time_limit'] or solver_options['node_limit'], or try a different n_runs."


def _describe_target(n_runs: int | None, low: int, high: int | None = None) -> str:
    """Name the run size a search was asked for, for error messages."""
    if n_runs is not None:
        return f"n_runs={n_runs}"
    if high is None or high == low:
        return f"{low} runs"
    return f"{low} to {high} runs"


def generate_omars(  # noqa: PLR0913
    factors: list[Factor],
    *,
    n_runs: int | None = None,
    n_runs_range: tuple[int, int] | None = None,
    selection_criterion: str = "dominance",
    satisfice: dict[str, float] | None = None,
    center_runs: int = 1,
    n_restarts: int = 50,
    max_candidates: int | None = None,
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
        is ``None``.  The search starts at the smallest feasible size and moves
        up the window only if no design at a size can estimate the *model*.  A
        window that ends below the smallest estimable size raises
        ``ValueError``.  Sizes needing more half-runs than the
        ``(3**k - 1) / 2`` distinct ones are reachable only by repeating
        half-runs, which the exhaustive search does for three and four
        factors.
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
        high-quality design.  Each restart steers the solver towards a different
        feasible OMARS design; the best one (by *selection_criterion*) is kept.
        Higher values explore more of the feasible set and approach the
        catalogue-optimal designs more closely, at a roughly linear cost in
        runtime; ``0`` runs only the plain feasibility solve at each size.  The
        search early-stops once the feasible set stops yielding new designs, so
        small factor counts finish quickly regardless.  The budget
        applies at each run size the search visits.  Default 50,
        which reaches catalogue-competitive D-efficiency for up to seven factors.
        Deterministic for a fixed *random_seed*, as long as no solve hits
        ``solver_options["time_limit"]`` (see *random_seed*).
    max_candidates : int, optional
        Legacy alias retained for backward compatibility.  When given, it sets a
        floor on *n_restarts* (the effective restart budget is ``max(n_restarts,
        max_candidates)``), so calls that raised it to enumerate more designs
        still explore at least that many.  Default ``None``: no floor.
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
        Settings for the HiGHS solves, any of: ``"msg"`` (bool, default
        ``False``) prints the HiGHS log to standard output; ``"time_limit"``
        (float seconds, default ``60.0``) caps each solve's wall-clock time;
        ``"node_limit"`` (int or None, default ``100``) caps the
        branch-and-bound nodes of each randomized-objective solve, and
        ``None`` solves each one to proven optimality.  Unknown keys raise
        ``ValueError``.  See :func:`solve_omars_ilp`.
    tol : float, optional
        Tolerance for the floating-point :func:`is_omars` re-check.
    random_seed : int, optional
        Seed for both the randomized-objective search (which design is found) and
        the run-order randomisation of the returned design.  A fixed seed makes
        the whole call reproducible on a given SciPy version: the node budget,
        not the clock, ends each solve.  A solve stopped by the wall-clock
        ``time_limit`` instead depends on machine speed;
        ``metadata["omars_search"].time_limited_solves`` counts them.  Another
        SciPy release ships another HiGHS, which may return a different design
        for the same seed.
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
        incompatible with *center_runs*, *n_restarts* is negative,
        *solver_options* has an unknown key or an out-of-range value, or no
        feasible design from which the sizing model can be estimated is found.
    TypeError
        If *solver_options* is not a dict, or one of its values has the wrong
        type.

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
    if n_restarts < 0:
        raise ValueError(f"n_restarts must be >= 0; got {n_restarts}.")

    coded, metadata = _search_best_omars(
        factors,
        n_runs=n_runs,
        n_runs_range=n_runs_range,
        selection_criterion=selection_criterion,
        satisfice=satisfice,
        n_restarts=n_restarts if max_candidates is None else max(n_restarts, max_candidates),
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
    :func:`process_improve.experiments.generate_design` handles post-processing,
    appending the other ``center_runs - 1`` centre runs.

    The budget is an upper bound: the design is the largest foldover within it,
    ``2h + center_runs`` runs, so an even budget with one centre run gives one run
    fewer. It is sized for the full second-order model when that size reaches the
    ``k**2 + k + 1`` runs (with one centre run) a foldover needs to estimate it, and
    otherwise for main effects plus pure quadratics, so any budget of at least
    ``2k + 2 + center_runs`` runs (15 runs for 6 factors and one centre run) gives an
    OMARS design.
    """
    budget = kwargs.get("budget")
    center_runs = int(kwargs.get("center_runs", 1))
    k = len(factors)
    n_runs = budget
    if budget is not None and (budget - center_runs) % 2:
        n_runs = budget - 1  # a foldover has 2h + center_runs runs
    # A foldover estimates the full second-order model only from k**2 + k + center_runs runs (see _min_half_runs).
    full_size = 2 * _min_half_runs(k, "full_second_order") + center_runs
    model = "full_second_order" if n_runs is None or n_runs >= full_size else "main_quadratic"
    random_state = kwargs.get("random_state", 42)
    seed = random_state if isinstance(random_state, int) else int(check_random_state(random_state).integers(2**31))
    try:
        designed, meta = _search_best_omars(
            factors,
            n_runs=n_runs,
            n_runs_range=None,
            selection_criterion="dominance",
            satisfice=None,
            n_restarts=50,
            model=model,
            solver_options=None,
            tol=1e-9,
            verify=True,
            random_seed=seed,
            center_runs=center_runs,
        )
    except ValueError as exc:
        if budget is None:
            raise
        msg = (
            f"No OMARS design fits budget={budget} with {center_runs} centre run(s): the largest foldover "
            f"within it has n_runs={n_runs} runs. {exc}"
        )
        raise ValueError(msg) from exc
    return designed, {**meta, "model": model}
