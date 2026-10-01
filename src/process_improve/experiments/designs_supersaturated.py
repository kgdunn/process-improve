# (c) Kevin Dunn, 2010-2026. MIT License.

"""Supersaturated two-level designs: more factors than runs.

A supersaturated design screens ``k`` factors in ``n < k + 1`` runs. The columns
cannot all be orthogonal, so the design is chosen to keep them as close to orthogonal
as possible, measured by ``E(s^2)``: the average of ``s_ij^2`` over all pairs of
columns, with ``s_ij = x_i' x_j``. It only works under effect sparsity, when a few of
the many factors are active, and the analysis has to search for them (stepwise
selection, the lasso, or the Dantzig selector), not fit them all at once.

Construction (Lin 1993): take a normalised Hadamard matrix of order ``N`` (first
column all +1), pick one of its other columns as the *branching column*, and keep the
``N / 2`` rows where it is +1. The remaining ``N - 2`` columns are balanced (half +1,
half -1) in those rows, and nearly orthogonal. Each choice of branching column gives
a different design, so all are tried and the one with the smallest ``E(s^2)``, then the
smallest ``max |s_ij|``, is kept. When fewer than ``N - 2`` factors are needed, the
columns that contribute most to ``E(s^2)`` are dropped one at a time.

Hadamard matrices come from pyDOE3's Plackett-Burman construction, and, for orders
``N`` where ``N - 1`` is a prime congruent to 3 mod 4, from Paley's construction
``H = I + C`` with ``C`` the skew conference matrix (which adds 44, 60, 68, 72, 84).
"""

from __future__ import annotations

import contextlib
import logging
import math
from typing import TYPE_CHECKING

import numpy as np

from process_improve.experiments.factor import FactorType

if TYPE_CHECKING:
    from process_improve.experiments.factor import Factor

logger = logging.getLogger(__name__)

#: Largest Hadamard order searched when no run budget is given.
_MAX_ORDER = 200


def hadamard(order: int) -> np.ndarray | None:
    """Return a normalised Hadamard matrix of ``order`` (first column all +1), or ``None`` if none is built here."""
    from pyDOE3 import pbdesign  # noqa: PLC0415

    from process_improve.experiments.designs_response_surface import (  # noqa: PLC0415
        _is_prime,
        _paley_conference_matrix,
    )

    candidates = []
    q = order - 1
    if q % 4 == 3 and _is_prime(q):
        # Paley first: its cyclic structure gives half-fractions without identical columns,
        # where the Kronecker-built (Sylvester-type) matrices for 16, 24, 32 always have some.
        candidates.append(np.eye(order) + _paley_conference_matrix(q))
    with contextlib.suppress(AssertionError, ValueError, IndexError):  # pyDOE3 asserts on unsupported orders
        candidates.append(np.column_stack([np.ones(order), pbdesign(order - 1)]))
    for h in candidates:
        if h.shape == (order, order) and np.array_equal(h.T @ h, order * np.eye(order)):
            return h * h[:, [0]]  # multiply rows so the first column is all +1
    return None


def e_s2(design: np.ndarray) -> float:
    """``E(s^2)``: the mean of ``(x_i' x_j)^2`` over all pairs of distinct columns."""
    s = design.T @ design
    upper = s[np.triu_indices_from(s, k=1)]
    return float(np.mean(upper**2))


def e_s2_lower_bound(n_runs: int, n_factors: int) -> float:
    """Return the lower bound on ``E(s^2)`` for balanced two-level designs (Nguyen 1996; Tang and Wu 1997)."""
    return n_runs**2 * (n_factors - n_runs + 1) / ((n_factors - 1) * (n_runs - 1))


def n_fully_aliased(design: np.ndarray) -> int:
    """Count column pairs that are identical or opposite (``|s_ij| = n``): their effects cannot be told apart."""
    s = np.abs(design.T @ design)
    return int((s[np.triu_indices_from(s, k=1)] == design.shape[0]).sum())


def _drop_worst_columns(columns: np.ndarray, k: int) -> np.ndarray:
    """Remove columns until ``k`` remain: fully aliased ones first, then the one with the largest summed ``s^2``."""
    n = columns.shape[0]
    keep = list(range(columns.shape[1]))
    while len(keep) > k:
        s = columns[:, keep].T @ columns[:, keep]
        np.fill_diagonal(s, 0)
        aliased = (np.abs(s) == n).sum(axis=0)
        keep.pop(int(np.lexsort(((s**2).sum(axis=0), aliased))[-1]))
    return columns[:, keep]


def _lin_half_fraction(h: np.ndarray, k: int) -> np.ndarray:
    """Best Lin half-fraction of ``h`` with ``k`` columns, over every choice of branching column."""
    order = h.shape[0]
    best, best_key = None, (np.inf, np.inf, np.inf)
    for branch in range(1, order):
        rows = h[:, branch] > 0
        others = [c for c in range(1, order) if c != branch]
        design = _drop_worst_columns(h[rows][:, others], k)
        s = design.T @ design
        key = (n_fully_aliased(design), e_s2(design), float(np.abs(s[np.triu_indices_from(s, k=1)]).max()))
        if key < best_key:
            best, best_key = design, key
    if best is None:  # unreachable for order >= 4, which always offers a branching column
        raise RuntimeError("No branching column was available.")
    return best


def _smallest_unaliased(k: int) -> np.ndarray:
    """Lin design for ``k`` factors from the smallest Hadamard order without fully aliased pairs.

    Falls back to the smallest order with enough columns when every order up to
    ``_MAX_ORDER`` leaves some pair aliased.
    """
    first = None
    for order in range(4, _MAX_ORDER + 1, 4):
        if order - 2 < k or (h := hadamard(order)) is None:
            continue
        design = _lin_half_fraction(h, k)
        if not n_fully_aliased(design):
            return design
        first = design if first is None else first
    if first is None:
        raise ValueError(f"No Hadamard matrix of order up to {_MAX_ORDER} is available for {k} factors.")
    return first


def dispatch_supersaturated(factors: list[Factor], budget: int | None = None) -> tuple[np.ndarray, dict]:
    """Generate a supersaturated design: ``k`` two-level factors in fewer than ``k + 1`` runs.

    Parameters
    ----------
    factors : list[Factor]
        Continuous factors, run at their low and high levels (coded -1 / +1).
    budget : int or None
        Number of runs. It must be even, below ``k + 1``, and ``2 * budget`` must be a
        Hadamard order built here (4, 8, 12, ..., 40, 44, 48, 60, 64, ...). ``None``
        picks the smallest order ``N`` with ``N - 2 >= k`` whose half-fraction has no
        fully aliased pair of factors, giving ``N / 2`` runs. A budget that forces
        fully aliased pairs is honoured, with a warning naming the run count that
        avoids them.

    Returns
    -------
    tuple[np.ndarray, dict]
        The coded design and metadata: ``e_s2``, its lower bound and their ratio
        ``e_s2_efficiency`` (1 is the best a balanced design can do), ``max_abs_s``,
        and the Hadamard order used.

    Raises
    ------
    ValueError
        For non-continuous factors, fewer than 3 factors, a budget that is not
        supersaturated (``budget >= k + 1``: use ``"plackett_burman"``), or a budget
        with no Hadamard matrix of order ``2 * budget``.
    """
    if any(f.type != FactorType.continuous for f in factors):
        raise ValueError("Supersaturated designs are for continuous factors run at two levels.")
    k = len(factors)
    if k < 3:
        raise ValueError("A supersaturated design needs at least 3 factors.")

    if budget is not None:
        if budget >= k + 1:
            raise ValueError(
                f"{budget} runs can estimate all {k} main effects; a supersaturated design is for fewer than "
                f"{k + 1} runs. Use design_type='plackett_burman' instead."
            )
        h = hadamard(2 * budget) if budget % 2 == 0 else None
        if h is None or 2 * budget - 2 < k:
            feasible = [n // 2 for n in range(4, 2 * k + 3, 4) if n - 2 >= k and hadamard(n) is not None]
            raise ValueError(
                f"No supersaturated design with {budget} runs for {k} factors here; the run counts available are "
                f"{feasible or 'none below k + 1'}."
            )
        design = _lin_half_fraction(h, k)
    else:
        design = _smallest_unaliased(k)

    if n_fully_aliased(design):
        clean = _smallest_unaliased(k)
        logger.warning(
            "%d pair(s) of factors are fully aliased in %d runs; %d runs avoid that.",
            n_fully_aliased(design),
            design.shape[0],
            clean.shape[0],
        )
    n_runs = design.shape[0]
    s = design.T @ design
    value, bound = e_s2(design), e_s2_lower_bound(n_runs, k)
    meta = {
        "method": "lin_half_fraction",
        "hadamard_order": 2 * n_runs,
        "e_s2": value,
        "e_s2_lower_bound": bound,
        "e_s2_efficiency": bound / value if value > 0 else 1.0,
        "max_abs_s": int(np.abs(s[np.triu_indices_from(s, k=1)]).max()),
        "n_fully_aliased_pairs": n_fully_aliased(design),
        "n_column_pairs": math.comb(k, 2),
        "note": (
            f"{k} factors in {n_runs} runs: main effects are partially aliased. Analyse with a selection "
            "method (stepwise, lasso) under effect sparsity, not by fitting every factor."
        ),
    }
    return design, meta
