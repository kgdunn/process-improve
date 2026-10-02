# (c) Kevin Dunn, 2010-2026. MIT License.

"""The experimental region: the set of factor settings a design, an evaluation and an optimum may use.

A :class:`DesignRegion` joins the factors and their constraints into one object, so
the same region can drive the three places it matters:

- :func:`~process_improve.experiments.generate_design` places runs inside it;
- :func:`~process_improve.experiments.evaluate_design` averages (I-efficiency) and
  maximises (G-efficiency) the prediction variance over it, not over the full box;
- :func:`~process_improve.experiments.optimize_responses` searches for an optimum
  inside it, so the recommended settings are ones the plant can run.

Two kinds of region exist, and each works in its own *design units*:

- ``"box"``: continuous (and categorical) factors, in coded units ``[-1, 1]``, with
  optional inequality constraints written in actual units;
- ``"mixture"``: mixture components, as proportions summing to 1, with optional
  component bounds and linear inequality constraints.

``generate_design`` records the region it used in ``DesignResult.metadata["region"]``
as a plain dict, which :meth:`DesignRegion.from_dict` turns back into a region.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np

from process_improve.experiments._uniform_sampling import _TOL, UniformSampler, _Inequality, _linear_row
from process_improve.experiments.factor import Constraint, Factor, FactorType


class DesignRegion:
    """Factors plus constraints, with feasibility tests and uniform sampling in design units.

    Parameters
    ----------
    factors : list[Factor]
        Either all mixture components, or continuous and categorical factors.
    constraints : list[Constraint] or None
        Inequalities in actual units (proportions for mixtures).

    Examples
    --------
    >>> region = DesignRegion(
    ...     [Factor(name="T", low=100, high=150), Factor(name="D", low=20, high=60)],
    ...     [Constraint(expression="3*T + 5*D <= 600")],
    ... )
    >>> region.feasible(np.array([[1.0, 1.0], [-1.0, -1.0]]))  # coded (150, 60) and (100, 20)
    array([False,  True])
    """

    def __init__(self, factors: list[Factor], constraints: list[Constraint] | None = None) -> None:
        self.factors = list(factors)
        self.constraints = list(constraints or [])
        is_mixture = [f.type == FactorType.mixture for f in self.factors]
        if any(is_mixture) and not all(is_mixture):
            raise ValueError(
                "A region is either all mixture components or none; mixture-process regions are not supported."
            )
        self.kind: Literal["box", "mixture"] = "mixture" if all(is_mixture) and self.factors else "box"
        #: Names of the numeric columns the region acts on, in order.
        self.names = [f.name for f in self.factors if f.type != FactorType.categorical]
        self._inequalities = self._build_inequalities()

    # -- construction ---------------------------------------------------------

    def _build_inequalities(self) -> list[_Inequality]:
        """Return functions of design-unit points (n, k), each feasible where ``<= 0``."""
        if self.kind == "mixture":
            from process_improve.experiments.designs_mixture_constrained import mixture_inequalities  # noqa: PLC0415

            a_mat, b_vec = mixture_inequalities(self.factors, self.constraints)
            self._a, self._b = a_mat, b_vec
            return [_linear_row(a_row, float(b_val)) for a_row, b_val in zip(a_mat, b_vec, strict=True)]

        from process_improve.experiments.designs_constrained import parse_constraint  # noqa: PLC0415

        continuous = [f for f in self.factors if f.type != FactorType.categorical]
        low = np.array([f.low for f in continuous], dtype=float)
        high = np.array([f.high for f in continuous], dtype=float)
        centre, half = (low + high) / 2.0, (high - low) / 2.0
        names = {f.name for f in continuous}
        gs = [g for c in self.constraints for g in parse_constraint(c.expression, names)]
        self._actual_inequalities = gs

        def wrap(g: Callable) -> _Inequality:
            def coded(x: np.ndarray) -> np.ndarray:
                actual = centre + np.atleast_2d(x) * half
                env = {n: actual[:, j] for j, n in enumerate(self.names)}
                return np.broadcast_to(np.asarray(g(env), dtype=float), (actual.shape[0],))

            return coded

        return [wrap(g) for g in gs]

    @property
    def is_constrained(self) -> bool:
        """True unless the region is the plain box or the full simplex."""
        if self.kind == "mixture":
            return bool(self.constraints) or any(f.low > 0 or f.high < 1 for f in self.factors)  # type: ignore[operator]
        return bool(self.constraints)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly description, as stored in ``DesignResult.metadata["region"]``."""
        return {
            "kind": self.kind,
            "factors": [f.model_dump(mode="json") for f in self.factors],
            "constraints": [c.model_dump(mode="json") for c in self.constraints],
        }

    @classmethod
    def from_dict(cls, spec: dict[str, Any]) -> DesignRegion:
        """Rebuild a region from :meth:`to_dict` output."""
        return cls([Factor(**f) for f in spec["factors"]], [Constraint(**c) for c in spec.get("constraints", [])])

    def __repr__(self) -> str:
        """Return a short summary."""
        expressions = [c.expression for c in self.constraints]
        return f"DesignRegion(kind={self.kind!r}, factors={self.names}, constraints={expressions})"

    # -- queries --------------------------------------------------------------

    @property
    def inequalities(self) -> list[_Inequality]:
        """Functions ``g(x)`` of design-unit points ``(n, k)``; a point is feasible where every ``g <= 0``."""
        return list(self._inequalities)

    @property
    def bounds(self) -> list[tuple[float, float]]:
        """Per-column bounds in design units: ``(-1, 1)`` for a box, ``(low, high)`` for a mixture."""
        if self.kind == "mixture":
            return [(float(f.low), float(f.high)) for f in self.factors]  # type: ignore[arg-type]
        return [(-1.0, 1.0)] * len(self.names)

    def feasible(self, x: np.ndarray, tol: float = _TOL) -> np.ndarray:
        """Return a boolean per row of ``x`` (design units): inside the bounds and every constraint."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        low, high = np.array(self.bounds).T
        ok = np.all((x >= low - tol) & (x <= high + tol), axis=1)
        if self.kind == "mixture":
            ok &= np.abs(x.sum(axis=1) - 1.0) <= 1e-6
        for g in self._inequalities:
            ok &= g(x) <= tol
        return ok

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` points uniformly from the region, in design units.

        A box region is sampled from the coded cube. A mixture region is sampled from
        the smallest simplex holding the lower bounds (the L-pseudocomponent simplex),
        where a flat Dirichlet draw is uniform, so tight lower bounds cost nothing.
        When the constraints leave too small a part of that set for rejection to be
        cheap, hit-and-run chains started from the region's support points take over
        (see :class:`UniformSampler`).

        Raises
        ------
        ValueError
            If no point of the region can be found.
        """
        k = len(self.names)
        if self.kind == "mixture":
            low = np.array([f.low for f in self.factors], dtype=float)

            def propose(m: int) -> np.ndarray:
                return low + (1.0 - low.sum()) * rng.dirichlet(np.ones(k), size=m)
        else:

            def propose(m: int) -> np.ndarray:
                return rng.uniform(-1.0, 1.0, size=(m, k))

        bounds = tuple(np.array(self.bounds, dtype=float).T)
        sampler = UniformSampler(
            self._inequalities, bounds, propose, self.seed_points, on_simplex=self.kind == "mixture"
        )
        return sampler.draw(n, rng)

    def seed_points(self) -> np.ndarray:
        """Return :meth:`support_points`, or none for a box too large for a boundary grid."""
        try:
            return self.support_points()
        except ValueError:
            return np.empty((0, len(self.names)))

    def support_points(self) -> np.ndarray:
        """Return points on the region's boundary, in design units: where worst-case variance sits.

        For a mixture these are the extreme vertices and edge midpoints; for a box, the
        feasible grid points and the points where each constraint crosses a grid edge.
        """
        if self.kind == "mixture":
            from process_improve.experiments.designs_mixture_constrained import mixture_candidates  # noqa: PLC0415

            candidates = mixture_candidates(self._a, self._b)
            return np.vstack([candidates["vertex"], candidates["edge_midpoint"]])

        from process_improve.experiments.designs_constrained import _Region, build_candidates  # noqa: PLC0415

        continuous = [f for f in self.factors if f.type != FactorType.categorical]
        coded, _cats, _ = build_candidates(_Region(continuous, [], self._actual_inequalities), n_levels=3)
        return coded
