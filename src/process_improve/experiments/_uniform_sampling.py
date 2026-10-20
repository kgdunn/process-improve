# (c) Kevin Dunn, 2010-2026. MIT License.

"""Uniform sampling of a region given by bounds and inequalities, however thin it is.

Kept free of other ``process_improve.experiments`` imports so that both the region
(:mod:`~process_improve.experiments.region`) and the design engines that need
uniform samples can use it without importing each other.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc

_TOL = 1e-9
#: Draws rejection sampling may spend before handing over to hit-and-run.
_REJECTION_DRAWS = 2_000_000
#: Parallel hit-and-run chains, and the shrinkage attempts per step.
_CHAINS = 1000
_SHRINK_STEPS = 60
#: Extra points on which a fitted affine form must reproduce a constraint, and the
#: largest dimension whose box corners are all checked as well.
_AFFINE_CHECKS = 1024
_CORNER_DIMENSIONS = 10

_Inequality = Callable[[np.ndarray], np.ndarray]


def _linear_row(a: np.ndarray, b: float) -> _Inequality:
    """Return ``g(x) = x @ a - b``, feasible where ``<= 0``."""
    return lambda x: np.atleast_2d(x) @ a - b


def _probe_points(low: np.ndarray, high: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Points to fit an affine form on, and more points to check it on, spread through the box ``[low, high]``.

    The fit uses the centre of the box, a step of a quarter of the range from it along
    each axis, and the first point of the unscrambled Halton sequence: ``k + 2`` points
    of which the first ``k + 1`` are affinely independent, so an affine form is always
    recovered exactly. (The first ``k + 2`` Halton points alone are not: in a base larger
    than ``k + 2`` the coordinates are ``i / p``, so from 6 factors several columns are
    proportional and every linear constraint failed the test.) The check adds the box
    corners (up to ``2**_CORNER_DIMENSIONS`` of them) and ``_AFFINE_CHECKS`` pseudo-random
    points from a fixed seed, so the probes reach every part of the box, its edges
    included. Nothing depends on the caller's random generator.

    Returns the fit points, the pseudo-random checks and the corners, in that order.
    """
    k = len(low)
    span = np.where(np.isfinite(high - low), high - low, 1.0)
    base = np.where(np.isfinite(low), low, 0.0)
    steps = np.vstack([np.full((1, k), 0.5), 0.5 + 0.25 * np.eye(k), qmc.Halton(d=k, scramble=False).random(2)[1:]])
    randoms = np.random.default_rng(0).random((_AFFINE_CHECKS, k))
    corners = np.empty((0, k))
    if k <= _CORNER_DIMENSIONS:
        corners = np.array(list(itertools.product((0.0, 1.0), repeat=k))).reshape(2**k, k)  # one corner when k is 0
    return base + span * steps, base + span * randoms, base + span * corners


def _affine_form(
    g: _Inequality, low: np.ndarray, high: np.ndarray, offsets: np.ndarray | None = None
) -> tuple[np.ndarray, float] | None:
    """Return ``(a, b)`` with ``g(x) = a @ x - b`` for every ``x``, or ``None`` when ``g`` is not affine.

    The form is fitted on ``k + 2`` points and must then reproduce ``g`` on the box
    corners and on many more points spread through the box (see :func:`_probe_points`).
    This is a numerical test, not a proof: a piecewise-linear constraint whose kink lies
    in a sliver that no probe reaches would pass it. The corners catch the common case,
    a kink near a face of the box (``abs(a - 0.95) >= 0.01``). Misclassifying a
    constraint never yields an infeasible point, only a part of the region the
    hit-and-run chains do not visit.

    The fit may miss ``g`` by ``1e-9`` of the largest value ``g`` takes, or by the
    rounding of the terms ``g`` adds up, ``1e-13`` of their size, whichever is more: so
    the test does not depend on the units of ``g``. (The ``1e-9`` used to be added in the
    limit's own units, so a curved limit whose values stay below ``1e-9``, as on a factor
    whose range is ``1e-9`` of its units, passed for straight.) ``offsets`` says where the
    values behind each coordinate sit, as ``centre / half-range`` of a factor that ``g``
    reads in coded units, where its terms can be far larger than its values: ``t``
    near ``1.7e9`` with a range of an hour carries a rounding of ``5e-7``. None means
    the coordinates themselves.

    A constraint that cannot be evaluated somewhere in the box (``sqrt`` or ``log`` of
    a negative number, a division by zero) is not affine there: any probe value that is
    not finite, and any fitted coefficient that overflows, gives ``None``. (Before this
    was checked, an infinite value made the tolerance infinite and a NaN made the test
    false, so such a limit passed and its secant plane stood in for it.)
    """
    k = len(low)
    fit, randoms, corners = _probe_points(low, high)
    points = np.vstack([fit, randoms, corners])
    with np.errstate(all="ignore"):
        values = np.broadcast_to(np.asarray(g(points), dtype=float), (len(points),))
    if not np.all(np.isfinite(values)):
        return None
    design = np.column_stack([points, np.ones(len(points))])
    coef = np.linalg.lstsq(design[: len(fit)], values[: len(fit)], rcond=None)[0]
    shift = np.zeros(k) if offsets is None else np.asarray(offsets, dtype=float)
    with np.errstate(all="ignore"):
        error = float(np.max(np.abs(design @ coef - values)))
        # The terms in the values behind the coordinates: a @ (x + shift) and the constant left over.
        terms = float(np.max(np.abs((points + shift) * coef[:k]).sum(axis=1))) + abs(coef[k] - coef[:k] @ shift)
        tolerance = max(1e-9 * float(np.max(np.abs(values))), 1e-13 * terms)
    if not error <= tolerance:  # written so that a NaN fails the test too, as from a coefficient that overflowed
        return None
    return coef[:k], -float(coef[k])


def _compares_numbers_only(a: np.ndarray, b: float) -> bool:
    """Say whether the straight limit ``a @ x <= b`` compares numbers only: its gradient is below 1e-9 of ``b``.

    The gradient and the constant are in the limit's own units, so the test does not depend
    on them: ``X <= 0.5e-9`` on a range of ``1e-9`` is a limit, though its gradient is
    ``5e-10``. (Compared with ``1e-9`` itself, it was taken for one that holds everywhere.)
    """
    return math.hypot(*a) <= 1e-9 * abs(b)


def _spread(g: _Inequality, low: np.ndarray, high: np.ndarray) -> float:
    """Return the largest finite ``|g|`` at the corners of the box and at its fit points, or 1 when that is 0.

    A curved limit's tolerance is ``_TOL`` times this, ``1e-9`` of the values it takes over
    the box, whatever its units; a straight limit's is a distance (see ``tolerances`` in
    :class:`UniformSampler`). The corners are left out above ``_CORNER_DIMENSIONS`` factors.
    """
    fit, _, corners = _probe_points(low, high)
    points = np.vstack([fit, corners])
    with np.errstate(all="ignore"):
        values = np.abs(np.broadcast_to(np.asarray(g(points), dtype=float), (len(points),)))
    largest = float(values[np.isfinite(values)].max(initial=0.0))
    return largest if largest > 0 else 1.0


def _box_chord(x: np.ndarray, d: np.ndarray, low: np.ndarray, high: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per row, the range of ``t`` for which ``x + t d`` stays inside the bounds ``[low, high]``."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (low - x) / d, (high - x) / d
    moving = d != 0
    return (
        np.where(moving, np.minimum(t1, t2), -np.inf).max(axis=1),
        np.where(moving, np.maximum(t1, t2), np.inf).min(axis=1),
    )


@dataclass
class UniformSampler:
    """Uniform sampling of a region, however small a part of its bounding set it is.

    Rejection from ``propose`` (uniform on a set that holds the region) is exact and
    fast while the region is a sizeable part of that set. When it is too thin, the
    points found so far, plus ``seeds()``, start parallel hit-and-run chains, whose
    cost does not depend on how thin the region is.

    Each hit-and-run step picks a random direction, drawn from the shape of the points
    known to lie in the region, and a uniform point on the chord through the current
    point. The chord is cut exactly by the bounds and by every
    affine constraint (``a @ x <= b`` limits ``t`` in closed form); only non-linear
    constraints are handled by shrinking the interval after a miss, as in Neal's
    slice sampler, which keeps the uniform distribution invariant even on a
    non-convex region.

    Parameters
    ----------
    inequalities : list of callables
        ``g(x)`` of ``(m, k)`` points; a point is in the region where every ``g <= 0``.
    bounds : tuple[np.ndarray, np.ndarray]
        Per-column lower and upper limits that hold the region.
    propose : callable
        ``propose(m)`` returns ``m`` points uniform on a set holding the region.
    seeds : callable or None
        Returns known feasible points (a candidate grid, extreme vertices), used to
        start the chains when rejection finds none.
    on_simplex : bool
        The points are proportions summing to 1; directions keep that sum.
    tolerances : np.ndarray or None
        Per inequality, what the tolerance ``_TOL`` on its value is multiplied by, so a
        point passes here as it passes the caller's own test: the length of a straight
        limit's gradient makes the tolerance a distance, whatever the limit's units. None
        keeps ``_TOL`` in each limit's own units.
    """

    inequalities: list[_Inequality]
    bounds: tuple[np.ndarray, np.ndarray]
    propose: Callable[[int], np.ndarray]
    seeds: Callable[[], np.ndarray] | None = None
    on_simplex: bool = False
    tolerances: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Split the constraints into affine rows ``a @ x <= b`` and the rest."""
        forms = [_affine_form(g, *self.bounds) for g in self.inequalities]
        rows = [f for f in forms if f is not None]
        k = len(self.bounds[0])
        self._a = np.array([a for a, _ in rows]).reshape(len(rows), k)
        self._b = np.array([b for _, b in rows])
        widen = np.ones(len(self.inequalities)) if self.tolerances is None else np.asarray(self.tolerances, dtype=float)
        self._reach = _TOL * widen.reshape(len(self.inequalities))
        self._nonlinear = [
            (g, reach) for g, f, reach in zip(self.inequalities, forms, self._reach, strict=True) if f is None
        ]

    def feasible(self, x: np.ndarray) -> np.ndarray:
        """Boolean per row: inside the bounds, on the simplex if one, and meeting every constraint."""
        low, high = self.bounds
        ok = np.all((x >= low - _TOL) & (x <= high + _TOL), axis=1)
        if self.on_simplex:
            ok &= np.abs(x.sum(axis=1) - 1.0) <= 1e-6
        for g, reach in zip(self.inequalities, self._reach, strict=True):
            ok &= np.asarray(g(x)) <= reach
        return ok

    def draw(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Return ``n`` points, ``(n, k)``.

        From hit-and-run the points are dependent along each chain but uniform in
        distribution, which is what averages over the region need.

        Raises
        ------
        ValueError
            If no feasible point is found at all.
        """
        found, enough = self._rejection(n)
        if enough:
            return found[:n]
        points = self._known_points(found)
        x = self._starts(found, points, rng, min(_CHAINS, n))
        shape = self._direction_shape(points)
        for _ in range(20 * x.shape[1] + 100):  # burn-in, away from the seeds
            x = self._step(x, rng, shape)
        collected: list[np.ndarray] = []
        while sum(len(c) for c in collected) < n:
            for _ in range(x.shape[1]):  # thin by the dimension
                x = self._step(x, rng, shape)
            collected.append(x.copy())
        return np.vstack(collected)[:n]

    def _rejection(self, n: int) -> tuple[np.ndarray, bool]:
        """Draw by rejection until ``n`` points are kept, or until that is projected to exceed the draw budget."""
        batch = min(max(n, 10_000), _REJECTION_DRAWS)
        kept, drawn, accepted = [], 0, 0
        while accepted < n:
            points = self.propose(batch)
            kept.append(points[self.feasible(points)])
            drawn, accepted = drawn + batch, accepted + len(kept[-1])
            if n * drawn / max(accepted, 1) > _REJECTION_DRAWS:  # projected total draws
                break
        return np.vstack(kept), accepted >= n

    def _known_points(self, found: np.ndarray) -> np.ndarray:
        """Return the rejection hits and the feasible ``seeds()``: every point known to lie in the region."""
        points = found
        if self.seeds is not None and (extra := np.atleast_2d(self.seeds())).size:
            points = np.vstack([points, extra[self.feasible(extra)]])
        if not len(points):
            raise ValueError("No point inside the region was found; the constraints may leave no feasible settings.")
        return points

    def _starts(self, found: np.ndarray, points: np.ndarray, rng: np.random.Generator, n_chains: int) -> np.ndarray:
        """Chain starts: rejection hits when there are any, else random convex combinations of the known points.

        Rejection hits are uniform on the region, and hit-and-run keeps the uniform
        distribution, so chains started from them are uniform at every step, however
        slowly they mix; on a region made of separate pieces the chains are then shared
        between the pieces in proportion to their size. Seeds sit on the boundary
        (vertices, edges), where most directions have a chord of length zero, so without
        hits a random weighting of all the known points is used: it lies inside a convex
        region, and where it does not (a non-convex region), the chain starts at one of
        the points instead.
        """
        if len(found):
            return found[rng.integers(len(found), size=n_chains)]
        mixed = rng.dirichlet(np.ones(len(points)), size=n_chains) @ points
        picked = points[rng.integers(len(points), size=n_chains)]
        return np.where(self.feasible(mixed)[:, None], mixed, picked)

    def _direction_shape(self, points: np.ndarray) -> np.ndarray:
        """Return the square root of the covariance the step directions are drawn with: the known points' spread.

        Isotropic directions mix very slowly in a long, thin region, since almost every
        chord is short. Drawing them from the region's own shape (here, the covariance
        of the points known to lie in it) rounds the region, so most chords run along
        it. Any fixed direction distribution that gives ``d`` and ``-d`` the same weight
        leaves the uniform distribution invariant. A small isotropic part keeps every
        direction possible, and makes the matrix positive definite.

        The square root is the Cholesky factor: any ``L`` with ``L @ L.T`` equal to the
        covariance gives the same Gaussian directions, and this one is unique, so the
        chains do not depend on the signs an eigensolver happens to return. (Eigenvectors'
        signs differed between native LAPACK and Pyodide, and averages over the samples
        differed in the third or fourth figure.) The chains still magnify differences in
        the last bit of the platforms' linear algebra: in 6 or more factors their points
        drift apart, by up to about 0.002 in coded units over thousands of steps, and
        averages over them agree to about 7 to 11 figures.
        """
        k = points.shape[1]
        cov = np.cov(points, rowvar=False).reshape(k, k) if len(points) > 1 else np.zeros((k, k))
        scale = float(np.trace(cov)) / k
        if not np.isfinite(scale) or scale <= 0:
            return np.eye(k)
        return np.linalg.cholesky(cov + 1e-3 * scale * np.eye(k))

    def _chord(self, x: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Range of ``t`` keeping ``x + t d`` inside the bounds and every affine constraint."""
        t_lo, t_hi = _box_chord(x, d, *self.bounds)
        if len(self._b):
            rate = d @ self._a.T  # how fast each constraint's left side grows along d
            slack = np.maximum(self._b - x @ self._a.T, 0.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                limit = slack / rate
            t_hi = np.minimum(t_hi, np.where(rate > 0, limit, np.inf).min(axis=1))
            t_lo = np.maximum(t_lo, np.where(rate < 0, limit, -np.inf).max(axis=1))
        return np.minimum(t_lo, 0.0), np.maximum(t_hi, 0.0)

    def _step(self, x: np.ndarray, rng: np.random.Generator, shape: np.ndarray | None = None) -> np.ndarray:
        """Move every chain once: a random direction, a uniform point on its chord, shrinking on a miss.

        Directions are Gaussian with covariance ``shape @ shape.T`` (isotropic by default).
        """
        d = rng.standard_normal(x.shape)
        if shape is not None:
            d = d @ shape.T
        if self.on_simplex:
            d -= d.mean(axis=1, keepdims=True)  # stay on the plane where the proportions sum to 1
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        t_lo, t_hi = self._chord(x, d)
        pending = np.arange(len(x))
        for _ in range(_SHRINK_STEPS):
            t = rng.uniform(t_lo[pending], t_hi[pending])
            trial = x[pending] + t[:, None] * d[pending]
            ok = np.ones(len(trial), dtype=bool)
            for g, reach in self._nonlinear:
                ok &= np.asarray(g(trial)) <= reach
            x[pending[ok]] = trial[ok]
            t, pending = t[~ok], pending[~ok]
            t_lo[pending] = np.where(t < 0, t, t_lo[pending])  # shrink towards the current point (t = 0)
            t_hi[pending] = np.where(t >= 0, t, t_hi[pending])
            if not len(pending):
                break
        return x
