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

A box region also describes itself, in actual units, for someone writing limits on
the settings: the share of the box it keeps (:meth:`~DesignRegion.volume_fraction`,
exact for straight limits), whether anything is left (:meth:`~DesignRegion.is_empty`),
the range each factor can still reach (:meth:`~DesignRegion.implied_bounds`), a central
setting to hold other factors at (:meth:`~DesignRegion.centroid`), the limits that remove
nothing (:meth:`~DesignRegion.redundant_constraints`), each limit's affine form
(:meth:`~DesignRegion.linear_forms`), its corners (:meth:`~DesignRegion.vertices`), and
pictures of it on any two factors: its shadow (:meth:`~DesignRegion.shadow`), a slice
through it (:meth:`~DesignRegion.slice_polygon`, :meth:`~DesignRegion.slice_lines`) and
the slack of a grid (:meth:`~DesignRegion.slice_slack`). Every answer is deterministic,
and fast enough to recompute on each edit of 8 limits on 8 factors.
"""

from __future__ import annotations

import ast
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Literal, cast

import numpy as np
from scipy.optimize import linprog
from scipy.stats import qmc

from process_improve.experiments._region_geometry import (
    box_share,
    box_vertices,
    clip_square,
    convex_polygon,
    points_inside,
    round_significant,
)
from process_improve.experiments._uniform_sampling import _TOL, UniformSampler, _affine_form, _Inequality, _linear_row
from process_improve.experiments.factor import Constraint, Factor, FactorType, as_constraints

#: Sobol points behind :meth:`DesignRegion.volume_fraction` and the searches of a curved region.
_N_SAMPLES = 2**14
#: Points spread inside the polytope of the linear constraints, to search a curved region with.
_N_INSIDE = 2**12
#: Cells per axis of a shadow mask, and grid points per axis of a slice mask.
_SHADOW_CELLS = 40
_SLICE_POINTS = 61
#: Significant figures of the numbers reported in actual units, and the relative size below which a
#: coefficient or a coordinate is taken to be zero.
_DIGITS = 12
_SNAP = 1e-9
#: A setting reported in actual units is rounded no coarser than this share of its factor's range, which
#: takes more than 12 figures for a range narrow next to its values, up to 16 figures (see ``_actual``).
_RESOLUTION = 1e-10
_MOST_DIGITS = 16
#: ``scipy.optimize.linprog`` status of a problem with no feasible point.
_INFEASIBLE = 2


class DesignRegion:
    """Factors plus constraints, with feasibility tests and uniform sampling in design units.

    Parameters
    ----------
    factors : list[Factor]
        Either all mixture components, or continuous and categorical factors.
    constraints : Sequence[Constraint | dict | str] or None
        Inequalities in actual units (proportions for mixtures): ``Constraint`` objects,
        their dict form (as :meth:`to_dict` writes it) or bare expressions such as
        ``"3*T + 5*D <= 600"``, mixed freely.

    Examples
    --------
    >>> region = DesignRegion(
    ...     [Factor(name="T", low=100, high=150), Factor(name="D", low=20, high=60)],
    ...     ["3*T + 5*D <= 600"],
    ... )
    >>> region.feasible(np.array([[1.0, 1.0], [-1.0, -1.0]]))  # coded (150, 60) and (100, 20)
    array([False,  True])
    """

    def __init__(self, factors: list[Factor], constraints: Sequence[Constraint | dict | str] | None = None) -> None:
        self.factors = list(factors)
        self.constraints: list[Constraint] = as_constraints(constraints) or []
        is_mixture = [f.type == FactorType.mixture for f in self.factors]
        if any(is_mixture) and not all(is_mixture):
            raise ValueError(
                "A region is either all mixture components or none; mixture-process regions are not supported."
            )
        self.kind: Literal["box", "mixture"] = "mixture" if all(is_mixture) and self.factors else "box"
        #: Names of the numeric columns the region acts on, in order.
        self.names = [f.name for f in self.factors if f.type != FactorType.categorical]
        numeric = [f for f in self.factors if f.type != FactorType.categorical]
        self._low = np.array([f.low for f in numeric], dtype=float)
        self._high = np.array([f.high for f in numeric], dtype=float)
        self._centre, self._half = (self._low + self._high) / 2.0, (self._high - self._low) / 2.0
        #: For each inequality, the index of the constraint it came from (a chain gives several).
        self._limit_of: list[int] = []
        self._inequalities = self._build_inequalities()
        self._cache: dict[str, Any] = {}

    # -- construction ---------------------------------------------------------

    def _build_inequalities(self) -> list[_Inequality]:
        """Return functions of design-unit points (n, k), each feasible where ``<= 0``."""
        if self.kind == "mixture":
            from process_improve.experiments.designs_mixture_constrained import mixture_inequalities  # noqa: PLC0415

            a_mat, b_vec = mixture_inequalities(self.factors, self.constraints)
            self._a, self._b = a_mat, b_vec
            return [_linear_row(a_row, float(b_val)) for a_row, b_val in zip(a_mat, b_vec, strict=True)]

        from process_improve.experiments.designs_constrained import parse_constraint  # noqa: PLC0415

        centre, half = self._centre, self._half
        per_limit = [parse_constraint(c.expression, set(self.names)) for c in self.constraints]
        gs = [g for group in per_limit for g in group]
        self._limit_of = [i for i, group in enumerate(per_limit) for _ in group]
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
        """Rebuild a region from :meth:`to_dict` output.

        The constraints go through the constructor, so they may take any form it accepts
        (``Constraint`` dicts as :meth:`to_dict` writes them, or bare expressions), and a
        missing or ``None`` entry means no constraints.

        Parameters
        ----------
        spec : dict[str, Any]
            ``{"factors": [...], "constraints": [...]}``; a ``"kind"`` entry is ignored,
            since the factors decide it.

        Returns
        -------
        DesignRegion
            The region.
        """
        return cls([Factor(**f) for f in spec["factors"]], spec.get("constraints") or [])

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
        """Return a boolean per row of ``x`` (design units): inside the bounds and every constraint.

        Parameters
        ----------
        x : np.ndarray
            Points ``(n, k)`` in design units (coded for a box, proportions for a mixture).
        tol : float
            How far a point may lie outside. For a straight limit of a box region whose
            coded gradient is longer than 1 it is a distance in coded units (the limit's
            value divided by that length); otherwise it is in the limit's own units, as for
            a curved limit. Either way a point within ``tol`` of a straight limit's
            boundary, in coded units, passes, however the limit is scaled. So do the
            corners the region reports (:meth:`vertices`, :meth:`shadow`,
            :meth:`slice_polygon`): their rounding moves each coordinate by at most
            ``1e-10`` in coded units.

        Returns
        -------
        np.ndarray
            A boolean per row.
        """
        x = np.atleast_2d(np.asarray(x, dtype=float))
        low, high = np.array(self.bounds, dtype=float).reshape(-1, 2).T
        ok = np.all((x >= low - tol) & (x <= high + tol), axis=1)
        if self.kind == "mixture":
            ok &= np.abs(x.sum(axis=1) - 1.0) <= 1e-6
        for g, widen in zip(self._inequalities, self._widen(), strict=True):
            ok &= g(x) <= tol * widen
        return ok

    def _widen(self) -> np.ndarray:
        """Return, per inequality, what the tolerance on its value is multiplied by (cached).

        For a straight limit of a box region it is the length of its coded gradient when
        above 1, which turns the tolerance into a distance; else 1, which keeps the
        tolerance of a gently sloped or curved limit in its own units.
        """
        if "widen" not in self._cache:
            scale = self._linear().scale if self.kind == "box" else [1.0] * len(self._inequalities)
            self._cache["widen"] = np.maximum(np.asarray(scale, dtype=float), 1.0).reshape(len(self._inequalities))
        return self._cache["widen"]

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` points uniformly from the region, in design units: an ``(n, 0)`` array without continuous factors.

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
        if k == 0:  # only categorical factors: no continuous dimension to sample
            return np.empty((n, 0))
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

    # -- diagnostics of a box region (actual units in, actual units out) -------

    def _require_box(self, method: str) -> None:
        """Raise for a mixture region, where a box diagnostic has no meaning."""
        if self.kind == "mixture":
            raise ValueError(
                f"DesignRegion.{method}() describes a box of factor settings; this region holds mixture "
                "proportions that sum to 1, where it does not apply."
            )

    def _linear(self) -> _Rows:
        """Return the inequalities in coded units, split into unit-length affine rows and the rest (cached)."""
        if "rows" not in self._cache:
            self._cache["rows"] = _Rows.of(self._inequalities, self._limit_of, len(self.names))
        return self._cache["rows"]

    def _vertices_coded(self) -> np.ndarray | None:
        """Return the vertices of the box cut by the affine rows, in coded units, or None when too many (cached)."""
        if "vertices" not in self._cache:
            rows = self._linear()
            try:
                self._cache["vertices"] = box_vertices(rows.a, rows.b)
            except ValueError as err:
                self._cache["vertices"], self._cache["too_many"] = None, str(err)
        return self._cache["vertices"]

    def _unit_sample(self, n_samples: int) -> np.ndarray:
        """Return ``n_samples`` unscrambled Sobol points, moved to their cells' midpoints, in coded units."""
        k = len(self.names)
        if k == 0:
            return np.zeros((1, 0))
        m = max(0, math.ceil(math.log2(max(n_samples, 1))))
        unit = qmc.Sobol(d=k, scramble=False).random_base2(m)[:n_samples]
        return 2.0 * (unit + 0.5 / 2**m) - 1.0

    def _cloud(self) -> np.ndarray:
        """Return the points a curved region is searched with (cached).

        They are a Sobol sample of the box, and the vertices of the polytope the linear
        constraints leave with points spread through it, which land inside a region too
        thin for the box sample to reach. (The support points would add the crossings of a
        grid's edges with the curved boundaries, but cost over 100 ms with 8 factors and 8
        constraints, against a few milliseconds for these.)
        """
        if "cloud" not in self._cache:
            parts = [self._unit_sample(_N_SAMPLES)]
            vertices = self._vertices_coded()
            if vertices is not None and len(vertices):
                parts += [vertices, points_inside(vertices, _N_INSIDE)]
            self._cache["cloud"] = np.vstack(parts)
        return self._cache["cloud"]

    def _cloud_slack(self) -> np.ndarray:
        """Return every inequality at every :meth:`_cloud` point, ``(points, inequalities)`` (cached)."""
        if "cloud_slack" not in self._cache:
            cloud = self._cloud()
            columns = [g(cloud) for g in self._inequalities]
            self._cache["cloud_slack"] = np.column_stack(columns) if columns else np.empty((len(cloud), 0))
        return self._cache["cloud_slack"]

    def _cloud_allowed(self) -> np.ndarray:
        """Return the :meth:`_cloud` points that meet every constraint (cached; all lie in the box)."""
        if "allowed" not in self._cache:
            self._cache["allowed"] = self._cloud()[np.all(self._cloud_slack() <= _TOL * self._widen(), axis=1)]
        return self._cache["allowed"]

    def _actual(self, coded: np.ndarray, columns: list[int] | None = None) -> np.ndarray:
        """Return coded values in actual units, exact at the box faces and rounded as :meth:`vertices` describes.

        Twelve figures alone are not enough for a factor whose range is narrow next to its
        values: they move a time stamp near ``1.7e9`` with a range of an hour by up to
        ``0.005``, ``3e-6`` in coded units, which takes a corner on a limit's boundary well
        outside the tolerance of :meth:`feasible`.
        """
        cols = list(range(len(self.names))) if columns is None else columns
        low, high, centre, half = self._low[cols], self._high[cols], self._centre[cols], self._half[cols]
        coded = np.asarray(coded, dtype=float)
        actual = np.where(coded == -1.0, low, np.where(coded == 1.0, high, centre + coded * half))
        actual = np.where(np.abs(actual) <= _SNAP * (high - low), 0.0, actual)
        with np.errstate(divide="ignore", invalid="ignore"):
            places = np.ceil(-np.log10(_RESOLUTION * (high - low)))  # decimal places that resolve 1e-10 of the range
            figures = np.floor(np.log10(np.abs(actual))) + 1 + places  # -inf for a zero, which keeps 12 figures
        figures = np.clip(np.nan_to_num(figures, nan=_DIGITS), _DIGITS, _MOST_DIGITS)
        return round_significant(actual, figures.astype(int))

    def _columns(self, x: str, y: str) -> tuple[int, int]:
        """Return the positions of the factors ``x`` and ``y``, which must be two different numeric factors."""
        for name in (x, y):
            if name not in self.names:
                raise ValueError(f"{name!r} is not a numeric factor of this region; it has {self.names}.")
        if x == y:
            raise ValueError(f"A picture of the region needs two different factors; got {x!r} twice.")
        return self.names.index(x), self.names.index(y)

    def _coded_point(self, at: Mapping[str, float] | None, skip: tuple[int, int]) -> np.ndarray:
        """Return the coded point at the values ``at`` (actual units), each factor not given at its centre."""
        point = np.zeros(len(self.names))
        for name, value in (at or {}).items():
            if name not in self.names:
                raise ValueError(f"{name!r} is not a numeric factor of this region; it has {self.names}.")
            if not math.isfinite(float(value)):
                raise ValueError(f"The value of {name!r} must be a finite number; got {value!r}.")
            j = self.names.index(name)
            if j not in skip:
                point[j] = (float(value) - self._centre[j]) / self._half[j]
        return point

    def volume_fraction(self, n_samples: int = 2**14) -> float:
        """Return the share of the factor box that the constraints leave: 1 without constraints, 0 when empty.

        With straight constraints only, the share is exact, to rounding. The factors are
        split into the groups the constraints tie together (a factor no constraint names
        keeps its whole range), and the share is the product of the groups' shares: in
        closed form for a group under one constraint, a chained one such as ``"400 <= 3*T +
        5*D <= 600"`` included, and from the volume of the polytope's convex hull for a group
        of up to five factors under several (see
        :func:`~process_improve.experiments._region_geometry.box_share`).

        Otherwise (a curved constraint, or six factors or more that several constraints tie
        together), the share is counted over ``n_samples`` points of the unscrambled Sobol
        sequence (moved to the midpoints of their cells). Beside a curved constraint, the
        share the straight ones leave is still exact, and the points count only what the
        curved ones take from it: the share is that exact share times the fraction of the
        points inside the straight constraints that meet every constraint. Else (or when the
        straight constraints leave too thin a region for any point to lie in it) it is the
        fraction of all the points that meet every constraint. The points are fixed, so the
        result is the same on every run and on every platform; with the default ``2**14``
        points it is within about 0.005 of the exact share for the regions of a few factors
        that limits describe.

        Parameters
        ----------
        n_samples : int
            Number of points, when the share is sampled; a power of two keeps the sequence
            balanced.

        Returns
        -------
        float
            The share of the box, between 0 and 1.

        Raises
        ------
        ValueError
            For a mixture region, or ``n_samples`` below 1.

        Examples
        --------
        >>> T, D = Factor(name="T", low=100, high=150), Factor(name="D", low=20, high=60)
        >>> round(DesignRegion([T, D], ["3*T + 5*D <= 600"]).volume_fraction(), 3)  # exactly 0.625
        0.625
        """
        self._require_box("volume_fraction")
        if n_samples < 1:
            raise ValueError(f"n_samples must be at least 1; got {n_samples}.")
        if not self.constraints:
            return 1.0
        exact = self._exact_share()
        if exact is not None:
            return exact
        sample = self._unit_sample(n_samples)
        allowed = self._sample_allowed() if n_samples == _N_SAMPLES else self.feasible(sample)
        rows = self._linear()
        straight = self._straight_share()
        inside = np.all(sample @ rows.a.T <= rows.b + _TOL, axis=1)
        if straight is None or not inside.any():
            return float(np.mean(allowed))
        return straight * float(np.mean(allowed[inside]))

    def _exact_share(self) -> float | None:
        """Return the exact share of the box with straight constraints only, or None when it is sampled."""
        return None if self._linear().nonlinear else self._straight_share()

    def _straight_share(self) -> float | None:
        """Return the exact share of the box its straight constraints leave, or None when not had fast (cached)."""
        if "straight_share" not in self._cache:
            rows = self._linear()
            self._cache["straight_share"] = box_share(rows.a, rows.b)
        return self._cache["straight_share"]

    def _sample_allowed(self) -> np.ndarray:
        """Return whether each point of the default Sobol sample is allowed (cached)."""
        if "sample_allowed" not in self._cache:
            self._cache["sample_allowed"] = self.feasible(self._unit_sample(_N_SAMPLES))
        return self._cache["sample_allowed"]

    def centroid(self) -> dict[str, float] | None:
        """Return a central setting of the region, in actual units: where to hold factors that are not plotted.

        It is the mean of the allowed points of the Sobol sample behind
        :meth:`volume_fraction` (of the points :meth:`is_empty` searches with, when the
        region is too thin for that sample), so it is the region's centroid to within the
        sample's accuracy. A region that is not convex, such as a ring, can have its
        centroid outside it; the allowed point nearest the centroid is returned then, so
        the result can always be run.

        Returns
        -------
        dict[str, float] or None
            ``{name: value}`` for every numeric factor, rounded as :meth:`vertices`
            describes; ``None`` when no allowed point is found (see :meth:`is_empty`).

        Raises
        ------
        ValueError
            For a mixture region.

        Examples
        --------
        >>> factors = [Factor(name="A", low=-1, high=1), Factor(name="B", low=-1, high=1)]
        >>> middle = DesignRegion(factors, ["A + B >= 1"]).centroid()  # the triangle's centroid is (2/3, 2/3)
        >>> {name: round(value, 2) for name, value in middle.items()}
        {'A': 0.66, 'B': 0.66}
        """
        self._require_box("centroid")
        allowed = self._unit_sample(_N_SAMPLES)[self._sample_allowed()]
        if not len(allowed):
            allowed = self._cloud_allowed()
        if not len(allowed):
            return None
        middle = allowed.mean(axis=0)
        if not self.feasible(middle[None, :])[0]:
            middle = allowed[np.argmin(np.sum((allowed - middle) ** 2, axis=1))]
        return {name: float(v) for name, v in zip(self.names, self._actual(middle), strict=True)}

    def is_empty(self) -> bool:
        """Return True when no setting inside the factor ranges meets every constraint.

        The linear constraints are decided exactly, from the vertices of the box they cut
        (a linear program when there are too many vertices to list). When some
        constraints are curved, a setting is searched for among a Sobol sample of the
        box and points spread through the polytope the linear constraints leave, its
        vertices included; a region so thin that none of these falls inside it is
        reported empty.

        Returns
        -------
        bool
            Whether the region holds no setting at all.

        Raises
        ------
        ValueError
            For a mixture region.
        """
        self._require_box("is_empty")
        rows = self._linear()
        vertices = self._vertices_coded()
        if vertices is not None and not len(vertices):
            return True
        if vertices is None and _lp_extreme(np.zeros(len(self.names)), rows.a, rows.b) is None:
            return True
        if not rows.nonlinear:
            return False
        return not len(self._cloud_allowed())

    def implied_bounds(self) -> dict[str, tuple[float, float]] | None:
        """Return the range each factor can still take, in actual units, given the linear constraints.

        A limit can keep a factor inside its own range everywhere: with ``T`` from 150 to
        200 and ``P`` from 1 to 3, ``T + 25*P <= 215`` keeps ``T`` at or below 190 (``P`` is
        at least 1) and ``P`` at or below 2.6 (``T`` is at least 150). These bounds are
        the smallest and largest value of each factor over the region cut by the linear
        constraints, from its vertices (or a linear program per bound when there are too
        many vertices). Curved constraints are left out, so with any of them the bounds
        are outer bounds: no setting lies outside them, but the extremes may not be
        reachable.

        Returns
        -------
        dict[str, tuple[float, float]] or None
            ``{name: (low, high)}`` for every numeric factor, in factor order, rounded as
            :meth:`vertices` describes; ``None`` when the linear constraints leave no
            setting.

        Raises
        ------
        ValueError
            For a mixture region.

        Examples
        --------
        >>> factors = [Factor(name="T", low=150, high=200), Factor(name="P", low=1, high=3)]
        >>> DesignRegion(factors, ["T + 25*P <= 215"]).implied_bounds()
        {'T': (150.0, 190.0), 'P': (1.0, 2.6)}
        """
        self._require_box("implied_bounds")
        k = len(self.names)
        vertices = self._vertices_coded()
        if vertices is not None:
            if not len(vertices):
                return None
            lows, highs = vertices.min(axis=0), vertices.max(axis=0)
        else:
            rows = self._linear()
            if _lp_extreme(np.zeros(k), rows.a, rows.b) is None:
                return None
            lows, highs = -np.ones(k), np.ones(k)
            for j, unit in enumerate(np.eye(k)):  # the region is not empty, so each program has a solution
                lows[j] = cast("np.ndarray", _lp_extreme(unit, rows.a, rows.b))[j]
                highs[j] = cast("np.ndarray", _lp_extreme(-unit, rows.a, rows.b))[j]
        lows, highs = self._actual(np.clip(lows, -1, 1)), self._actual(np.clip(highs, -1, 1))
        return {name: (float(lo), float(hi)) for name, lo, hi in zip(self.names, lows, highs, strict=True)}

    def redundant_constraints(self) -> list[int]:
        """Return the indices of the constraints that remove nothing the others do not already remove.

        Each constraint is tested against the region left by all the others: a linear one
        by a linear program per inequality (its largest value over that region), a curved
        one, or one the linear test cannot clear because some other constraint is curved,
        by checking that it removes none of the points that :meth:`is_empty` searches
        with (a Sobol sample of the box, and points spread through the polytope of the
        linear constraints) that the others keep. Two copies of the same constraint are
        both reported, since each removes nothing given the other; removing both is not
        safe.

        Returns
        -------
        list[int]
            Indices into ``constraints``, in order.

        Raises
        ------
        ValueError
            For a mixture region.
        """
        self._require_box("redundant_constraints")
        return [i for i in range(len(self.constraints)) if self._removes_nothing(i)]

    def _removes_nothing(self, i: int) -> bool:
        """Return whether constraint ``i`` removes no setting that the other constraints keep."""
        rows = self._linear()
        mine = rows.limit == i
        curved_here = i in rows.nonlinear_limit
        if not curved_here:
            others_a, others_b = rows.a[~mine], rows.b[~mine]
            for a_row, b_val in zip(rows.a[mine], rows.b[mine], strict=True):
                highest = _lp_extreme(-a_row, others_a, others_b)
                if highest is not None and a_row @ highest > b_val + _TOL:
                    break
            else:
                return True
            if not rows.nonlinear:  # the linear programs were exact
                return False
        slack = self._cloud_slack()
        limit_of = np.array(self._limit_of, dtype=int)
        within = slack <= _TOL * self._widen()  # NaN, where a limit is undefined, is not within: it removes
        kept_by_others = np.all(within[:, limit_of != i], axis=1)
        removed_here = np.any(~within[:, limit_of == i], axis=1)
        return not bool(np.any(kept_by_others & removed_here))

    def curved_constraints(self) -> list[int]:
        """Return the indices of the constraints that are curved: not affine over the factor box.

        A constraint is straight when each of its comparisons is affine over the box, by
        the test the design engines use, so ``"400 <= 3*T + 5*D <= 600"`` and ``"T <= 2*D"``
        are straight, and so is ``"abs(T) <= 3"`` over a box where ``T`` keeps one sign. The
        diagnostics treat the straight ones exactly and the curved ones by sampling (see
        :meth:`is_empty`); :meth:`implied_bounds` leaves the curved ones out, and
        :meth:`vertices` refuses a region that has any.

        Returns
        -------
        list[int]
            Indices into ``constraints``, in order.

        Raises
        ------
        ValueError
            For a mixture region.

        Examples
        --------
        >>> T, D = Factor(name="T", low=100, high=150), Factor(name="D", low=20, high=60)
        >>> DesignRegion([T, D], ["3*T + 5*D <= 600", "T * D <= 7000", "T <= 2*D"]).curved_constraints()
        [1]
        """
        self._require_box("curved_constraints")
        return sorted(set(self._linear().nonlinear_limit))

    def linear_forms(self) -> list[tuple[dict[str, float], float | None, float | None] | None]:
        """Return each constraint as ``(coefficients, low, high)`` in actual units, or None where it is not affine.

        ``"3*T + 5*D <= 600"`` gives ``({"T": 3.0, "D": 5.0}, None, 600.0)`` and
        ``"400 <= 3*T + 5*D <= 600"`` gives ``({"T": 3.0, "D": 5.0}, 400.0, 600.0)``: the
        constraint reads ``low <= sum(coefficient * factor) <= high``, with ``None`` for a
        side it does not bound. The factors stay on the side they were written on, so
        ``"600 >= 3*T + 5*D"`` gives the same form, and ``"T <= 2*D"`` gives
        ``({"T": 1.0, "D": -2.0}, None, 0.0)``. Each comparison is tested for being
        affine over the factor box, the test the design engines use. Its form is then read
        off the expression in exact arithmetic, with each number taken as written, so
        ``"TK + 0.01*P <= 340"`` gives ``0.01`` and not ``0.00999999999999``; where the
        expression is affine over the box only (``abs(T)`` with ``T`` positive), it is read
        off a fit there. A coefficient whose effect over the box is below ``1e-9`` of the
        largest is set to zero, its factor held at the centre of its range, where its term's
        value joins the bound; and every number is rounded to 12 significant figures, so
        floating-point residue such as ``1.65e-14`` or ``250.00000000000009`` does not
        appear.

        Returns
        -------
        list
            One entry per constraint, in order: the tuple above, or ``None`` for a
            constraint that is not affine, compares only numbers, or chains two different
            combinations of factors (``"A <= B <= 2"``).
        """
        from process_improve.experiments.designs_constrained import _compile, _factor_names  # noqa: PLC0415

        names = set(self.names)
        out: list[tuple[dict[str, float], float | None, float | None] | None] = []
        for constraint in self.constraints:
            # parse_constraint checked, when the region was built, that the expression is a comparison.
            compare = cast("ast.Compare", ast.parse(constraint.expression, mode="eval").body)
            terms = [compare.left, *compare.comparators]
            pairs = []
            for op, left, right in zip(compare.ops, terms[:-1], terms[1:], strict=True):
                g_left, g_right = _compile(left, names), _compile(right, names)
                mirrored = not _factor_names(left) and bool(_factor_names(right))
                upper = isinstance(op, ast.Lt | ast.LtE) != mirrored
                first, second = (g_right, g_left) if mirrored else (g_left, g_right)
                written = (right, left) if mirrored else (left, right)
                pairs.append((self._affine_actual(first, second, written), upper))
            out.append(_combine_forms(pairs, self.names))
        return out

    def _affine_actual(
        self, first: Callable, second: Callable, written: tuple[ast.expr, ast.expr] | None = None
    ) -> tuple[np.ndarray, float] | None:
        """Return ``first - second`` as ``(c, r)``, with ``first - second = c @ X - r`` in actual units, or None.

        ``written`` holds the two sides as parsed: when both are affine by their syntax,
        the form is read off them exactly rather than off the fit in coded units, whose
        digits run out when a constant is large next to a coefficient's effect over the box.
        """
        centre, half = self._centre, self._half

        def coded(x: np.ndarray) -> np.ndarray:
            actual = centre + np.atleast_2d(x) * half
            env = {n: actual[:, j] for j, n in enumerate(self.names)}
            value = np.asarray(first(env), dtype=float) - np.asarray(second(env), dtype=float)
            return np.broadcast_to(value, (actual.shape[0],))

        ones = np.ones(len(self.names))
        form = _affine_form(coded, -ones, ones)
        if form is None:
            return None
        exact = None if written is None else _exact_difference(*written, self.names)
        if exact is not None:
            coef, rhs = exact
            effect = np.abs(coef) * half
            dropped = effect <= _SNAP * effect.max(initial=0.0)
            # A term dropped for its effect is held at its factor's centre, as the fit below and the coded rows
            # behind vertices() hold it: its value there moves into the bound. (Held at zero, it moved the bound
            # of A + B <= 6 with A from 1 to 1 + 1e-9 from 4.9999999995 to 6.)
            rhs -= float(coef[dropped] @ centre[dropped])
            coef = np.where(dropped, 0.0, coef)
        else:
            a, b = form
            a = np.where(np.abs(a) <= _SNAP * (np.abs(a).sum() + abs(b)), 0.0, a)  # effect on g below 1e-9 of its size
            coef = a / half
            rhs = b + float(coef @ centre)
            # A fitted bound of zero carries the fit's residue; the bound read off the expression has none.
            scale = float(np.abs(coef) @ np.maximum(np.abs(self._low), np.abs(self._high))) + abs(rhs)
            rhs = 0.0 if abs(rhs) <= _SNAP * scale else rhs
        return round_significant(coef, _DIGITS), float(round_significant(np.array(rhs), _DIGITS))

    def vertices(self) -> np.ndarray:
        """Return the vertices of the region, in actual units, when every constraint is linear.

        The region is then a convex polytope: the box cut by the constraints. Its vertices
        are the extreme settings, where a D-optimal design for a main-effects model puts
        its runs (one for a quadratic model also uses edge midpoints and face centres).
        They are found by cutting the box one constraint at a time (see
        :func:`~process_improve.experiments._region_geometry.box_vertices`), so the work
        grows with the number of vertices rather than with the ways of choosing ``k``
        boundaries.

        Returns
        -------
        np.ndarray
            ``(n, k)`` vertices, one column per numeric factor, without duplicates,
            sorted row by row in coded order; no rows when the region is empty. Each
            value is rounded to 12 significant figures, so that ``250.00000000000009``
            reads ``250.0``, or to more where its factor's range is narrow next to its
            values (a year, an absolute pressure, a time stamp): to the decimal place of
            ``1e-10`` times the range, up to 16 figures, about what a double holds.
            Every vertex then passes :meth:`feasible`.

        Raises
        ------
        ValueError
            For a mixture region, a region with a curved constraint, or a region with
            too many vertices to list (more than 65536).
        """
        self._require_box("vertices")
        rows = self._linear()
        if rows.nonlinear:
            curved = sorted({self.constraints[i].expression for i in rows.nonlinear_limit})
            raise ValueError(f"The region has curved constraints {curved}, so it has no vertices to list.")
        vertices = self._vertices_coded()
        if vertices is None:
            raise ValueError(self._cache["too_many"])
        return self._actual(vertices)

    def shadow(self, x: str, y: str) -> dict[str, Any]:
        """Return the region's shadow on the factors ``x`` and ``y``: the settings of the two that some run can use.

        A pair ``(x, y)`` is in the shadow when at least one setting of the other factors
        makes it allowed: the region seen from above, along the other factors' axes. For
        a region whose constraints are all linear the shadow is exact: the convex hull of
        the vertices' projections. Otherwise it is a mask of 40 by 40 cells, each marked
        where an allowed point falls, among those :meth:`is_empty` searches with.

        Parameters
        ----------
        x, y : str
            Two different numeric factors.

        Returns
        -------
        dict
            ``{"kind": "polygon", "points": [[x, y], ...]}`` in actual units: the corners
            counter-clockwise from the one with the smallest ``x``; two points when the
            shadow is a segment, one when it is a point, none when the region is empty.
            Or ``{"kind": "mask", "x": [...], "y": [...], "z": [[0 or 1, ...], ...]}``:
            the 40 cell centres along each axis, and ``z[i][j]`` for the cell at
            ``(x[j], y[i])``.

        Raises
        ------
        ValueError
            For a mixture region, or when ``x`` and ``y`` are not two different numeric
            factors.
        """
        self._require_box("shadow")
        ix, iy = self._columns(x, y)
        vertices = None if self._linear().nonlinear else self._vertices_coded()
        if vertices is None:
            return self._shadow_mask(ix, iy)
        corners = convex_polygon(vertices[:, [ix, iy]])
        return {"kind": "polygon", "points": self._actual(corners, [ix, iy]).tolist()}

    def _shadow_mask(self, ix: int, iy: int) -> dict[str, Any]:
        """Return the shadow on columns ``ix``, ``iy`` as a mask of cells holding an allowed point of the cloud."""
        allowed = self._cloud_allowed()
        cells = np.clip(np.floor((allowed[:, [ix, iy]] + 1.0) / 2.0 * _SHADOW_CELLS), 0, _SHADOW_CELLS - 1).astype(int)
        z = np.zeros((_SHADOW_CELLS, _SHADOW_CELLS), dtype=int)
        z[cells[:, 1], cells[:, 0]] = 1
        centres = -1.0 + (2.0 * np.arange(_SHADOW_CELLS) + 1.0) / _SHADOW_CELLS
        return {
            "kind": "mask",
            "x": self._actual(centres, [ix] * _SHADOW_CELLS).tolist(),
            "y": self._actual(centres, [iy] * _SHADOW_CELLS).tolist(),
            "z": z.tolist(),
        }

    def slice_polygon(self, x: str, y: str, at: Mapping[str, float] | None = None) -> dict[str, Any]:
        """Return the settings of ``x`` and ``y`` that are allowed with every other factor held at ``at``.

        For linear constraints this is exact: the ``x``-``y`` rectangle of the two
        factors' ranges, clipped by each constraint's half-plane at the held values
        (Sutherland-Hodgman). With a curved constraint it is a mask on a 61 by 61 grid
        over the rectangle.

        Parameters
        ----------
        x, y : str
            Two different numeric factors.
        at : Mapping[str, float] or None
            Values (actual units) of the other numeric factors; a factor not given is held
            at the centre of its range, and entries for ``x`` and ``y`` are ignored. A
            value outside a factor's range leaves nothing to run.

        Returns
        -------
        dict
            ``{"kind": "polygon", "points": [[x, y], ...]}`` in actual units, as for
            :meth:`shadow` (no points when nothing can be run there), or
            ``{"kind": "mask", "x": [...], "y": [...], "z": [[0 or 1, ...], ...]}`` with
            the 61 grid values along each axis and ``z[i][j]`` for the point
            ``(x[j], y[i])``.

        Raises
        ------
        ValueError
            For a mixture region, when ``x`` and ``y`` are not two different numeric
            factors, or when ``at`` names a factor the region does not have or gives a
            value that is not a finite number.
        """
        self._require_box("slice_polygon")
        ix, iy = self._columns(x, y)
        point = self._coded_point(at, (ix, iy))
        rows = self._linear()
        if rows.nonlinear:
            grid = np.linspace(-1.0, 1.0, _SLICE_POINTS)
            xs, ys = self._actual(grid, [ix] * _SLICE_POINTS), self._actual(grid, [iy] * _SLICE_POINTS)
            allowed = self.slice_slack(x, y, xs[None, :], ys[:, None], self._at_actual(point)) <= 0
            return {"kind": "mask", "x": xs.tolist(), "y": ys.tolist(), "z": allowed.astype(int).tolist()}
        others = [j for j in range(len(self.names)) if j not in (ix, iy)]
        if np.any(np.abs(point[others]) > 1.0 + _TOL):
            return {"kind": "polygon", "points": []}
        half_planes = []
        for a_row, b_val in zip(rows.a, rows.b, strict=True):
            c, d = a_row[[ix, iy]], b_val - a_row[others] @ point[others]
            size = float(np.linalg.norm(c))
            if size <= _SNAP:
                if d < -_TOL:
                    return {"kind": "polygon", "points": []}
                continue
            half_planes.append((c / size, d / size))
        corners = convex_polygon(clip_square(half_planes))
        return {"kind": "polygon", "points": self._actual(corners, [ix, iy]).tolist()}

    def _at_actual(self, point: np.ndarray) -> dict[str, float]:
        """Return a coded point as ``{name: actual value}`` (not rounded: it is passed straight back in)."""
        return {n: float(c + p * h) for n, c, p, h in zip(self.names, self._centre, point, self._half, strict=True)}

    def slice_lines(self, x: str, y: str, at: Mapping[str, float] | None = None) -> list[dict[str, Any]]:
        """Return where each linear constraint's boundary crosses the ``x``-``y`` rectangle, the others held at ``at``.

        These are the lines a picture of the region draws and numbers: for a constraint
        in ``x`` and ``y`` alone the line does not depend on ``at``; for one that also
        names other factors it is the line in the slice at ``at`` (see
        :meth:`slice_polygon`). Each boundary is clipped to the rectangle of the two
        factors' ranges only, not by the other constraints.

        Parameters
        ----------
        x, y : str
            Two different numeric factors.
        at : Mapping[str, float] or None
            Values (actual units) of the other numeric factors, as for :meth:`slice_polygon`.

        Returns
        -------
        list[dict]
            ``{"limit": i, "points": [[x1, y1], [x2, y2]]}`` in actual units, one per
            inequality whose boundary crosses the rectangle, in constraint order; a
            chained constraint such as ``"400 <= 3*T + 5*D <= 600"`` can give two, both
            with its index ``i``. A boundary that misses the rectangle, only touches a
            corner of it, or does not involve ``x`` or ``y`` gives none, and curved
            constraints give none.

        Raises
        ------
        ValueError
            For a mixture region, when ``x`` and ``y`` are not two different numeric
            factors, or when ``at`` names a factor the region does not have or gives a
            value that is not a finite number.
        """
        self._require_box("slice_lines")
        ix, iy = self._columns(x, y)
        point = self._coded_point(at, (ix, iy))
        others = [j for j in range(len(self.names)) if j not in (ix, iy)]
        rows = self._linear()
        lines = []
        for a_row, b_val, limit in zip(rows.a, rows.b, rows.limit, strict=True):
            c, d = a_row[[ix, iy]], b_val - a_row[others] @ point[others]
            size = float(np.linalg.norm(c))
            if size <= _SNAP:
                continue
            ends = convex_polygon(clip_square([(c / size, d / size), (-c / size, -d / size)]))
            if len(ends) == 2:  # a segment, not a corner the boundary only touches
                lines.append({"limit": int(limit), "points": self._actual(ends, [ix, iy]).tolist()})
        return lines

    def slice_slack(
        self, x: str, y: str, x_grid: np.ndarray, y_grid: np.ndarray, at: Mapping[str, float] | None = None
    ) -> np.ndarray:
        """Return how far each grid point is from being allowed: ``<= 0`` exactly where it can be run.

        Each value is the largest slack over the factor ranges and the constraints, with
        the other factors held at ``at``. A linear constraint's slack is its signed
        distance from its boundary in coded units, so it varies linearly over the grid and
        a contour of it at zero draws a straight limit straight; a curved constraint's
        slack is its own value (the left side minus the right), and ``+inf`` where it cannot
        be evaluated. A value within the default tolerance of :meth:`feasible` (``1e-9``,
        as a coded distance for a steep straight limit) is returned as zero, so ``<= 0``
        here agrees with :meth:`feasible` at every point.

        Parameters
        ----------
        x, y : str
            Two different numeric factors.
        x_grid, y_grid : np.ndarray
            Values of ``x`` and ``y`` in actual units; they are broadcast together, so the
            two arrays from ``np.meshgrid`` (or a row and a column) give a grid.
        at : Mapping[str, float] or None
            Values (actual units) of the other numeric factors, each one not given held at
            the centre of its range; entries for ``x`` and ``y`` are ignored.

        Returns
        -------
        np.ndarray
            The slack, of the broadcast shape of ``x_grid`` and ``y_grid``.

        Raises
        ------
        ValueError
            For a mixture region, when ``x`` and ``y`` are not two different numeric
            factors, or when ``at`` names a factor the region does not have or gives a
            value that is not a finite number.
        """
        self._require_box("slice_slack")
        ix, iy = self._columns(x, y)
        xs, ys = np.broadcast_arrays(np.asarray(x_grid, dtype=float), np.asarray(y_grid, dtype=float))
        coded = np.tile(self._coded_point(at, (ix, iy)), (xs.size, 1))
        coded[:, ix] = (xs.ravel() - self._centre[ix]) / self._half[ix]
        coded[:, iy] = (ys.ravel() - self._centre[iy]) / self._half[iy]
        return self._slack(coded).reshape(xs.shape)

    def _slack(self, coded: np.ndarray) -> np.ndarray:
        """Return the largest slack per coded point, over the box and every inequality, zero within the tolerance."""
        box = np.abs(coded) - 1.0
        box_slack = np.max(np.where(np.abs(box) <= _TOL, 0.0, box), axis=1, initial=-np.inf)
        return np.maximum(box_slack, self._limit_slack(coded))

    def _limit_slack(self, coded: np.ndarray) -> np.ndarray:
        """Return the largest slack per coded point over the inequalities alone, zero within the tolerance.

        A linear inequality's slack is its signed distance from its boundary in coded
        units, a curved one's its own value, and ``+inf`` where a curved one cannot be
        evaluated (``sqrt`` or ``log`` of a negative number): :meth:`feasible` cannot run
        such a setting either, so ``> 0`` and ``<= 0`` both agree with it. ``-inf`` without
        inequalities. Leaving the box out keeps the slack of straight limits linear up to
        the box's faces, so a contour of it at zero is straight there too.
        """
        rows = self._linear()
        parts = []
        for g, scale, widen in zip(self._inequalities, rows.scale, self._widen(), strict=True):
            with np.errstate(invalid="ignore", divide="ignore"):
                value = g(coded)
            value = np.where(np.isnan(value), np.inf, value)
            parts.append(np.where(np.abs(value) <= _TOL * widen, 0.0, value / scale)[:, None])
        if not parts:
            return np.full(len(coded), -np.inf)
        return np.max(np.hstack(parts), axis=1)


@dataclass(frozen=True)
class _Rows:
    """A box region's inequalities in coded units: unit-length affine rows ``a @ x <= b``, and the rest.

    Attributes
    ----------
    a, b : np.ndarray
        The affine rows ``(r, k)`` and right-hand sides ``(r,)``. An inequality that
        compares numbers only is left out when it holds, and kept as ``0 @ x <= -1`` when
        it does not.
    limit : np.ndarray
        For each row, the index of the constraint it came from.
    scale : list[float]
        For each inequality (affine or not, in order), the length of its coded gradient
        (1 for a curved or constant one): dividing by it turns a value into a distance.
    nonlinear, nonlinear_limit : list
        The inequalities that are not affine over the box, and their constraints' indices.
    """

    a: np.ndarray
    b: np.ndarray
    limit: np.ndarray
    scale: list[float]
    nonlinear: list[_Inequality]
    nonlinear_limit: list[int]

    @classmethod
    def of(cls, inequalities: list[_Inequality], limit_of: list[int], k: int) -> _Rows:
        """Classify ``inequalities`` (coded functions) with the test the design engines use."""
        ones = np.ones(k)
        a_rows, b_rows, limits, scale, nonlinear, nonlinear_limit = [], [], [], [], [], []
        for g, limit in zip(inequalities, limit_of, strict=True):
            form = _affine_form(g, -ones, ones) if k else (np.zeros(0), -float(g(np.zeros((1, 0)))[0]))
            # The length without squaring: np.linalg.norm squares, which overflows from a length of 1e154, so
            # T <= 1e200 (whose fit carries noise near 1e184) and 1e200*T <= 1.2e202 were taken for curved.
            size = math.hypot(*form[0]) if form is not None else math.inf
            if not math.isfinite(size):  # not affine (a length past 1.8e308 would have overflowed the fit too)
                nonlinear.append(g)
                nonlinear_limit.append(limit)
                scale.append(1.0)
                continue
            a, b = cast("tuple[np.ndarray, float]", form)
            if size <= _SNAP * (1.0 + abs(b)):  # compares numbers only
                scale.append(1.0)
                if b < -_TOL:
                    a_rows.append(np.zeros(k))
                    b_rows.append(-1.0)
                    limits.append(limit)
                continue
            a = np.where(np.abs(a) <= _SNAP * size, 0.0, a)
            size = math.hypot(*a)
            scale.append(size)
            a_rows.append(a / size)
            b_rows.append(b / size)
            limits.append(limit)
        a_mat = np.array(a_rows, dtype=float).reshape(len(a_rows), k)
        return cls(a_mat, np.array(b_rows, dtype=float), np.array(limits, dtype=int), scale, nonlinear, nonlinear_limit)


_Exact = tuple[dict[str, Fraction], Fraction]
#: Most bits a power of constants may give and still be computed exactly: a limit of 500 characters can nest
#: powers, as in ``(((0.5**64)**64)**64)**64``, whose exact value takes a billion bits. Past this, floating point.
_EXACT_BITS = 4096


def _exact_difference(first: ast.expr, second: ast.expr, names: list[str]) -> tuple[np.ndarray, float] | None:
    """Return ``first - second`` as ``(c, r)`` with ``first - second = c @ X - r``, exactly, or None.

    None unless both sides are affine by their syntax (see :func:`_exact_affine`).
    """
    left, right = _exact_affine(first), _exact_affine(second)
    if left is None or right is None:
        return None
    coefficients = np.array([float(left[0].get(n, 0) - right[0].get(n, 0)) for n in names])
    return coefficients, float(right[1] - left[1])


def _exact_affine(node: ast.expr) -> _Exact | None:
    """Return a parsed expression as ``({name: coefficient}, constant)`` in exact arithmetic, or None.

    Numbers are taken as written (``Fraction("0.01")``, not the double nearest it), and
    ``+``, ``-``, a product with a constant, a division by one and a whole power of a
    number (up to the 64th, and up to :data:`_EXACT_BITS` bits) are kept exact. Another
    part without factors (``2**0.5``, ``sqrt(2)``, ``(0.5**64)**64``) is evaluated in
    floating point. Anything else (a product of factors, a function of one) gives None.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float) and not isinstance(node.value, bool):
        finite = math.isfinite(node.value)
        return ({}, Fraction(node.value if isinstance(node.value, int) else repr(node.value))) if finite else None
    if isinstance(node, ast.Name):
        return {node.id: Fraction(1)}, Fraction(0)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
        inner = _exact_affine(node.operand)
        return inner if inner is None or isinstance(node.op, ast.UAdd) else _scaled(inner, Fraction(-1))
    if isinstance(node, ast.BinOp):
        left, right = _exact_affine(node.left), _exact_affine(node.right)
        return None if left is None or right is None else _exact_operation(node.op, left, right)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and len(node.args) == 1:
        argument = _exact_affine(node.args[0])
        return None if argument is None or _varies(argument) else _exact_function(node.func.id, argument[1])
    return None


def _varies(form: _Exact) -> bool:
    """Return whether an exact affine form depends on a factor."""
    return any(form[0].values())


def _exact_operation(op: ast.operator, left: _Exact, right: _Exact) -> _Exact | None:
    """Return ``left op right`` for two exact affine forms, or None when the result is not affine."""
    if isinstance(op, ast.Add | ast.Sub):
        sign = 1 if isinstance(op, ast.Add) else -1
        zero = Fraction(0)
        coefficients = {n: left[0].get(n, zero) + sign * right[0].get(n, zero) for n in set(left[0]) | set(right[0])}
        return coefficients, left[1] + sign * right[1]
    if isinstance(op, ast.Mult) and not (_varies(left) and _varies(right)):
        return _scaled(right, left[1]) if not _varies(left) else _scaled(left, right[1])
    if isinstance(op, ast.Div) and not _varies(right) and right[1] != 0:
        return _scaled(left, 1 / right[1])
    if isinstance(op, ast.Pow) and not _varies(left) and not _varies(right):
        base, power = left[1], right[1]
        size = max(base.numerator.bit_length(), base.denominator.bit_length())
        whole = power.denominator == 1 and abs(power) <= 64
        if whole and abs(power) * size <= _EXACT_BITS and (base != 0 or power > 0):
            return {}, base ** int(power)
        return _constant(lambda: float(base) ** float(power))
    return None


def _exact_function(name: str, value: Fraction) -> _Exact | None:
    """Return one of the allowed functions of a constant: ``abs`` exactly, the others in floating point."""
    if name == "abs":
        return {}, abs(value)
    function = {"sqrt": math.sqrt, "exp": math.exp, "log": math.log, "log10": math.log10}.get(name)
    return None if function is None else _constant(lambda: function(float(value)))


def _scaled(form: _Exact, factor: Fraction) -> _Exact:
    """Return an exact affine form multiplied by a constant."""
    return {n: c * factor for n, c in form[0].items()}, form[1] * factor


def _constant(evaluate: Callable[[], Any]) -> _Exact | None:
    """Return a constant computed in floating point as an exact form, or None where it is not a finite number."""
    try:
        value = evaluate()
    except (ValueError, OverflowError, ZeroDivisionError):
        return None
    return ({}, Fraction(repr(value))) if isinstance(value, float) and math.isfinite(value) else None


def _lp_extreme(c: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    """Return the point of ``{x in [-1, 1]^k : a @ x <= b}`` minimising ``c @ x``, or None when there is none."""
    k = len(c)
    if k == 0:
        return np.zeros(0) if np.all(b >= -_TOL) else None
    result = linprog(
        c, A_ub=a if len(b) else None, b_ub=b if len(b) else None, bounds=[(-1.0, 1.0)] * k, method="highs"
    )
    return None if result.status == _INFEASIBLE else np.asarray(result.x)


def _combine_forms(
    pairs: list[tuple[tuple[np.ndarray, float] | None, bool]], names: list[str]
) -> tuple[dict[str, float], float | None, float | None] | None:
    """Join the comparisons of one constraint into ``(coefficients, low, high)``, or None when they do not share one.

    Each pair is ``((c, r), upper)``: the comparison reads ``c @ X <= r`` when ``upper``,
    else ``c @ X >= r``; ``None`` in place of ``(c, r)`` marks a comparison that is not affine.
    """
    coefficients: np.ndarray | None = None
    low: float | None = None
    high: float | None = None
    for form, upper in pairs:
        if form is None:
            return None
        c, r = form
        if not np.any(c):  # compares numbers: drop it when it holds
            if (r >= 0) if upper else (r <= 0):
                continue
            return None
        if coefficients is not None and not np.array_equal(c, coefficients):
            return None
        coefficients = c
        if upper:
            high = r if high is None else min(high, r)
        else:
            low = r if low is None else max(low, r)
    if coefficients is None:
        return None
    return {n: float(v) for n, v in zip(names, coefficients, strict=True) if v != 0}, low, high
