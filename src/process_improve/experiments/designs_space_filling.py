# (c) Kevin Dunn, 2010-2026. MIT License.

"""Space-filling designs: points spread evenly through the region, for any model.

Model-based designs place runs where a chosen polynomial is best estimated. A
space-filling design assumes no model: it spreads the runs out, which suits computer
experiments, Gaussian-process or machine-learning surrogates, and exploratory work
where the shape of the response is unknown.

Six methods, in two groups:

- **On the factor box only** (projections onto each factor are what they control):

  - ``"latin_hypercube"``: each factor's range is cut into ``n`` equal slices, and
    each slice is used exactly once (McKay, Beckman and Conover 1979).
  - ``"maximin_lhs"``: a Latin hypercube whose smallest distance between two runs
    is made as large as possible, by swapping values within a column while the
    Morris-Mitchell criterion improves.
  - ``"uniform"``: a Latin hypercube with minimal centred L2 discrepancy, the
    criterion of uniform designs (Fang and Wang).

- **On any region**, including constrained boxes and constrained mixtures:

  - ``"sobol"`` and ``"halton"``: scrambled low-discrepancy sequences. In a
    constrained region the sequence is mapped to the region (onto the simplex for a
    mixture) and its infeasible points are skipped, keeping the order of the rest.
  - ``"maximin"``: runs chosen from a dense uniform sample of the region, plus its
    boundary points, by a farthest-point build followed by exchanges that raise the
    smallest distance between runs.

Distances are measured in design units: coded ``[-1, 1]`` for a box, proportions for
a mixture.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.stats import qmc

from process_improve._random import check_random_state
from process_improve.experiments.factor import FactorType
from process_improve.experiments.region import DesignRegion

if TYPE_CHECKING:
    from process_improve.experiments.factor import Constraint, Factor

#: Methods that work only on the plain factor box.
BOX_ONLY = ("latin_hypercube", "maximin_lhs", "uniform")
#: Methods that also work in constrained and mixture regions.
ANY_REGION = ("sobol", "halton", "maximin")
SPACE_FILLING_METHODS = BOX_ONLY + ANY_REGION

_MORRIS_MITCHELL_P = 15
_LHS_STARTS = 20
_EXCHANGE_PASSES = 10


# ---------------------------------------------------------------------------
# Box-only methods, on the unit cube
# ---------------------------------------------------------------------------


def _phi_p(points: np.ndarray) -> float:
    """Morris-Mitchell criterion: a smooth stand-in for the smallest distance (lower is better)."""
    return float(np.sum(pdist(points) ** -_MORRIS_MITCHELL_P) ** (1.0 / _MORRIS_MITCHELL_P))


def _maximin_lhs(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Latin hypercube with a large smallest distance: best of several starts, then in-column swaps.

    Swapping two values within one column keeps every slice of every factor used
    exactly once, so the design stays a Latin hypercube while ``phi_p`` falls.
    """
    starts = [qmc.LatinHypercube(d=k, rng=rng).random(n) for _ in range(_LHS_STARTS)]
    design = min(starts, key=_phi_p)
    current = _phi_p(design)
    for _ in range(200 * k):
        column, (i, j) = rng.integers(k), rng.choice(n, 2, replace=False)
        design[[i, j], column] = design[[j, i], column]
        trial = _phi_p(design)
        if trial < current:
            current = trial
        else:
            design[[i, j], column] = design[[j, i], column]  # undo
    return design


def _unit_cube_design(method: str, n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Return ``n`` points in ``[0, 1]^k`` for one of the box-only methods."""
    if method == "latin_hypercube":
        return qmc.LatinHypercube(d=k, rng=rng).random(n)
    if method == "uniform":
        return qmc.LatinHypercube(d=k, optimization="random-cd", rng=rng).random(n)
    return _maximin_lhs(n, k, rng)


# ---------------------------------------------------------------------------
# Methods for any region
# ---------------------------------------------------------------------------


def _to_simplex(u: np.ndarray) -> np.ndarray:
    """Map ``(n, q - 1)`` points of the unit cube to the simplex: the gaps between sorted coordinates.

    The map carries the uniform distribution to the uniform distribution, so a
    low-discrepancy sequence on the cube stays well spread on the simplex.
    """
    edges = np.sort(np.column_stack([np.zeros(len(u)), u, np.ones(len(u))]), axis=1)
    return np.diff(edges, axis=1)


def _sequence_in_region(region: DesignRegion, n: int, method: str, rng: np.random.Generator) -> np.ndarray:
    """Return the first ``n`` feasible points of a scrambled Sobol or Halton sequence mapped onto the region.

    Sobol points are drawn in powers of two (doubling each time), which keeps the
    sequence's balance properties and avoids scipy's warning.
    """
    k = len(region.names)
    dim = k - 1 if region.kind == "mixture" else k
    engine = (qmc.Sobol if method == "sobol" else qmc.Halton)(d=dim, rng=rng)
    low = np.array([b[0] for b in region.bounds])
    kept: list[np.ndarray] = []
    size = int(2 ** np.ceil(np.log2(max(n, 64))))
    while sum(len(p) for p in kept) < n:
        u = engine.random_base2(int(np.log2(size))) if method == "sobol" else engine.random(size)
        points = low + (1.0 - low.sum()) * _to_simplex(u) if region.kind == "mixture" else 2.0 * u - 1.0
        kept.append(points[region.feasible(points)])
        if engine.num_generated > 2**22 and not sum(len(p) for p in kept):
            raise ValueError("No point of the sequence fell inside the region; check the constraints.")
        size = engine.num_generated  # double the total each round
    return np.vstack(kept)[:n]


def _maximin_in_region(region: DesignRegion, n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` points from a dense sample of the region, chosen to make the smallest distance large.

    Farthest-point build: start from a random point, then repeatedly add the pool
    point farthest from those chosen. Then exchange: replace a run by a pool point
    whenever that raises the smallest distance in the design.
    """
    pool = region.sample(max(4000, 100 * n), rng)
    with contextlib.suppress(ValueError):  # a box too large for a boundary grid: the uniform sample alone
        pool = np.vstack([pool, region.support_points()])
    chosen = [int(rng.integers(len(pool)))]
    nearest = cdist(pool, pool[chosen]).ravel()
    for _ in range(n - 1):
        chosen.append(int(np.argmax(nearest)))
        nearest = np.minimum(nearest, cdist(pool, pool[[chosen[-1]]]).ravel())

    rows = np.array(chosen)
    for _ in range(_EXCHANGE_PASSES):
        improved = False
        for i in range(n):
            others = np.delete(rows, i)
            current = pdist(pool[rows]).min()
            gap = cdist(pool, pool[others]).min(axis=1)  # each pool point's distance to the other runs
            within = pdist(pool[others]).min() if len(others) > 1 else np.inf
            candidate = int(np.argmax(gap))
            if min(gap[candidate], within) > current + 1e-12:
                rows[i], improved = candidate, True
        if not improved:
            break
    return pool[rows]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def space_filling_design(
    factors: list[Factor],
    n_runs: int | None,
    method: str,
    constraints: list[Constraint] | None = None,
    random_state: int | np.random.Generator | None = None,
) -> tuple[np.ndarray, dict]:
    """Generate a space-filling design.

    Parameters
    ----------
    factors : list[Factor]
        Continuous factors, or mixture components. Categorical factors are refused:
        distance between levels has no meaning.
    n_runs : int or None
        Number of runs. ``None`` uses ``10 * k``, the usual starting size for fitting a
        Gaussian-process surrogate (Loeppky, Sacks and Welch 2009).
    method : str
        One of :data:`SPACE_FILLING_METHODS`; see the module docstring.
    constraints : list[Constraint] or None
        Inequalities in actual units (proportions for a mixture). Only the
        ``"sobol"``, ``"halton"`` and ``"maximin"`` methods accept them.
    random_state : int, numpy.random.Generator or None
        Seed for the scrambling, the sampling and the swaps.

    Returns
    -------
    tuple[np.ndarray, dict]
        Coded points (proportions for a mixture) and metadata: the smallest and the
        mean nearest-neighbour distance between runs, and on the plain box the
        centred L2 discrepancy (lower is more uniform).

    Raises
    ------
    ValueError
        For an unknown method, a categorical factor, or a box-only method asked for
        a constrained or mixture region.
    """
    if method not in SPACE_FILLING_METHODS:
        raise ValueError(f"Unknown space-filling method {method!r}; choose from {', '.join(SPACE_FILLING_METHODS)}.")
    if any(f.type == FactorType.categorical for f in factors):
        raise ValueError(
            "Space-filling designs need continuous or mixture factors; a categorical level has no distance."
        )
    rng = check_random_state(random_state)
    region = DesignRegion(factors, constraints)
    on_box = region.kind == "box" and not region.is_constrained
    if method in BOX_ONLY and not on_box:
        raise ValueError(
            f"{method!r} works on the plain factor box only; for a constrained or mixture region use "
            "'maximin', 'sobol' or 'halton'."
        )
    k = len(region.names)
    n = n_runs if n_runs is not None else 10 * k

    if method in BOX_ONLY:
        points = 2.0 * _unit_cube_design(method, n, k, rng) - 1.0
    elif method == "maximin":
        points = _maximin_in_region(region, n, rng)
    else:
        points = _sequence_in_region(region, n, method, rng)

    distances = cdist(points, points) + np.diag(np.full(n, np.inf))
    meta: dict = {
        "method": method,
        "min_distance": float(distances.min()),
        "mean_nearest_neighbour_distance": float(distances.min(axis=1).mean()),
    }
    if on_box:
        meta["centered_l2_discrepancy"] = float(qmc.discrepancy((points + 1.0) / 2.0, method="CD"))
    if constraints:
        meta["constraints"] = [c.expression for c in constraints]
        meta["constraints_enforced"] = True
    return points, meta
