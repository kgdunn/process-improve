# (c) Kevin Dunn, 2010-2026. MIT License.

"""Uniform sampling of a region given by bounds and inequalities, however thin it is.

Kept free of other ``process_improve.experiments`` imports so that both the region
(:mod:`~process_improve.experiments.region`) and the design engines that need
uniform samples can use it without importing each other.
"""

from __future__ import annotations

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
#: Extra random points on which a fitted affine form must reproduce a constraint.
_AFFINE_CHECKS = 30

_Inequality = Callable[[np.ndarray], np.ndarray]


def _linear_row(a: np.ndarray, b: float) -> _Inequality:
    """Return ``g(x) = x @ a - b``, feasible where ``<= 0``."""
    return lambda x: np.atleast_2d(x) @ a - b


def _affine_form(g: _Inequality, low: np.ndarray, high: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Return ``(a, b)`` with ``g(x) = a @ x - b`` for every ``x``, or ``None`` when ``g`` is not affine.

    The form is fitted on ``k + 1`` random points and must then reproduce ``g`` on
    ``_AFFINE_CHECKS`` more. A non-affine constraint (``T * D <= 900``, say) that
    passes that check would have to match a hyperplane at points in general
    position, which it cannot. The points are the unscrambled Halton sequence: spread
    through the box, and deterministic without a random generator.
    """
    k = len(low)
    span = np.where(np.isfinite(high - low), high - low, 1.0)
    points = low + span * qmc.Halton(d=k, scramble=False).random(k + 2 + _AFFINE_CHECKS)[1:]
    values = np.asarray(g(points), dtype=float)
    design = np.column_stack([points, np.ones(len(points))])
    coef = np.linalg.lstsq(design, values, rcond=None)[0]
    if np.max(np.abs(design @ coef - values)) > 1e-9 * (1.0 + np.max(np.abs(values))):
        return None
    return coef[:k], -float(coef[k])


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

    Each hit-and-run step picks a random direction and a uniform point on the chord
    through the current point. The chord is cut exactly by the bounds and by every
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
    """

    inequalities: list[_Inequality]
    bounds: tuple[np.ndarray, np.ndarray]
    propose: Callable[[int], np.ndarray]
    seeds: Callable[[], np.ndarray] | None = None
    on_simplex: bool = False

    def __post_init__(self) -> None:
        """Split the constraints into affine rows ``a @ x <= b`` and the rest."""
        forms = [_affine_form(g, *self.bounds) for g in self.inequalities]
        rows = [f for f in forms if f is not None]
        k = len(self.bounds[0])
        self._a = np.array([a for a, _ in rows]).reshape(len(rows), k)
        self._b = np.array([b for _, b in rows])
        self._nonlinear = [g for g, f in zip(self.inequalities, forms, strict=True) if f is None]

    def feasible(self, x: np.ndarray) -> np.ndarray:
        """Boolean per row: inside the bounds, on the simplex if one, and meeting every constraint."""
        low, high = self.bounds
        ok = np.all((x >= low - _TOL) & (x <= high + _TOL), axis=1)
        if self.on_simplex:
            ok &= np.abs(x.sum(axis=1) - 1.0) <= 1e-6
        for g in self.inequalities:
            ok &= np.asarray(g(x)) <= _TOL
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
        x = self._starts(found, rng, min(_CHAINS, n))
        for _ in range(20 * x.shape[1] + 100):  # burn-in, away from the seeds
            x = self._step(x, rng)
        collected: list[np.ndarray] = []
        while sum(len(c) for c in collected) < n:
            for _ in range(x.shape[1]):  # thin by the dimension
                x = self._step(x, rng)
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

    def _starts(self, found: np.ndarray, rng: np.random.Generator, n_chains: int) -> np.ndarray:
        """Chain starts: random convex combinations of the rejection hits and the seeds.

        Seeds sit on the boundary (vertices, edges), where most directions have a chord
        of length zero, so a chain started there barely moves. A random weighting of
        all of them lies inside a convex region; where it does not (a non-convex
        region), the chain starts at one of the points instead.
        """
        starts = found
        if self.seeds is not None and (extra := np.atleast_2d(self.seeds())).size:
            starts = np.vstack([starts, extra[self.feasible(extra)]])
        if not len(starts):
            raise ValueError("No point inside the region was found; the constraints may leave no feasible settings.")
        mixed = rng.dirichlet(np.ones(len(starts)), size=n_chains) @ starts
        picked = starts[rng.integers(len(starts), size=n_chains)]
        return np.where(self.feasible(mixed)[:, None], mixed, picked)

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

    def _step(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Move every chain once: a random direction, a uniform point on its chord, shrinking on a miss."""
        d = rng.standard_normal(x.shape)
        if self.on_simplex:
            d -= d.mean(axis=1, keepdims=True)  # stay on the plane where the proportions sum to 1
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        t_lo, t_hi = self._chord(x, d)
        pending = np.arange(len(x))
        for _ in range(_SHRINK_STEPS):
            t = rng.uniform(t_lo[pending], t_hi[pending])
            trial = x[pending] + t[:, None] * d[pending]
            ok = np.ones(len(trial), dtype=bool)
            for g in self._nonlinear:
                ok &= np.asarray(g(trial)) <= _TOL
            x[pending[ok]] = trial[ok]
            t, pending = t[~ok], pending[~ok]
            t_lo[pending] = np.where(t < 0, t, t_lo[pending])  # shrink towards the current point (t = 0)
            t_hi[pending] = np.where(t >= 0, t, t_hi[pending])
            if not len(pending):
                break
        return x
