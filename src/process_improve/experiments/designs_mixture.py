# (c) Kevin Dunn, 2010-2026. MIT License.

"""Mixture designs: simplex-lattice and simplex-centroid points sized to the Scheffé model.

For mixture experiments the factor levels represent proportions that must
sum to 1.  These designs operate directly in actual proportions rather
than coded -1/+1 units.
"""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING

import numpy as np

from process_improve.experiments.designs_mixture_constrained import scheffe_model
from process_improve.experiments.factor import FactorType

if TYPE_CHECKING:
    from process_improve.experiments.factor import Constraint, Factor


#: Most components blended in one run of the default design, by Scheffé model: the blends
#: each model's terms need (pure components, binary and ternary blends).
_BLEND_SIZE = {"scheffe_linear": 1, "scheffe_quadratic": 2, "scheffe_special_cubic": 3}
#: Default design names, by largest blend; the full simplex centroid is named as such.
_DEFAULT_METHOD = {
    1: "simplex_vertices_plus_centroid",
    2: "simplex_lattice_degree_2_plus_centroid",
    3: "simplex_centroid_ternary_plus_centroid",
}


def dispatch_mixture(
    factors: list[Factor],
    budget: int | None = None,
    constraints: list[Constraint] | None = None,
    model_type: str = "interactions",
    random_state: int | np.random.Generator | None = None,
) -> tuple[np.ndarray, dict]:
    """Generate a mixture design.

    On the full simplex (every component bounded by 0 and 1, no constraints) the
    design is sized to the Scheffé model: the pure components, the binary 50:50 blends
    for a quadratic model, and the ternary 1/3 blends for a special cubic one, each
    plus the overall centroid (Cornell 2002). These are the simplex-centroid points
    that blend at most one, two or three components; with three components the
    quadratic and special cubic designs are the full 7-run simplex centroid. A
    *budget* smaller than that design gives a D-optimal subset of the candidate
    blends instead (see below), so the design never exceeds the budget unless the
    budget is below the number of model terms; that budget is raised to the term
    count, with a warning, and recorded in ``metadata["budget_requested"]``.

    When any component has a bound inside (0, 1), or *constraints* are given, the
    region is a polytope inside the simplex and the design comes from
    :func:`~process_improve.experiments.designs_mixture_constrained.constrained_mixture_design`:
    vertices, edge midpoints and centroid without a budget, a D-optimal subset of the
    candidate blends with one.

    Parameters
    ----------
    factors : list[Factor]
        Mixture factors (proportions summing to 1); ``low``/``high`` are component bounds.
    budget : int or None
        Maximum number of runs.
    constraints : list[Constraint] or None
        Linear inequalities in the proportions, e.g. ``"x1 + x2 <= 0.7"``.
    model_type : str
        Scheffé model the design is built for: ``"scheffe_linear"`` (``"main_effects"``
        maps here), ``"scheffe_quadratic"`` (``"interactions"`` and ``"quadratic"`` map
        here) or ``"scheffe_special_cubic"``.
    random_state : int, numpy.random.Generator or None
        Seed for the exchange of a D-optimal design.

    Returns
    -------
    tuple[np.ndarray, dict]
        Design matrix (in proportions, not coded) and metadata.
    """

    k = len(factors)
    if k < 2:
        raise ValueError("Mixture designs require at least 2 components.")
    if any(f.type != FactorType.mixture for f in factors):
        raise ValueError("A mixture design needs every factor to be a mixture component (type='mixture').")
    model = scheffe_model(model_type)
    bounded = any((f.low or 0.0) > 0.0 or (f.high if f.high is not None else 1.0) < 1.0 for f in factors)
    blend_size = _BLEND_SIZE[model]
    n_default = sum(math.comb(k, r) for r in range(1, min(blend_size, k) + 1)) + (blend_size < k)
    if bounded or constraints or (budget is not None and budget < n_default):
        from process_improve.experiments.designs_constrained import ConstrainedOptions  # noqa: PLC0415
        from process_improve.experiments.designs_mixture_constrained import (  # noqa: PLC0415
            constrained_mixture_design,
        )

        return constrained_mixture_design(factors, budget, constraints, ConstrainedOptions(model), random_state)

    matrix = _simplex_centroid(k, max_components=blend_size)
    method = "simplex_centroid" if blend_size >= k - 1 else _DEFAULT_METHOD[blend_size]
    return matrix, {"method": method, "model_type": model}


def _simplex_lattice(k: int, degree: int = 2) -> np.ndarray:
    """Generate a {k, degree} simplex-lattice design.

    The lattice consists of all points where each component takes values
    from {0, 1/degree, 2/degree, ..., 1} and all components sum to 1. They are
    listed directly, as the ways of sharing ``degree`` equal parts among the ``k``
    components, so the work grows with the ``C(k + degree - 1, degree)`` points, not
    with ``(degree + 1) ** k``.

    Parameters
    ----------
    k : int
        Number of mixture components.
    degree : int
        Degree of the lattice (typically 2 or 3).

    Returns
    -------
    np.ndarray
        Design matrix of shape (n_points, k) with rows summing to 1.

    Raises
    ------
    ValueError
        If ``k`` exceeds ``settings.max_factors_combinatorial``, or the lattice has
        more than 1,000,000 points (SEC-19 #268).
    """
    from process_improve.config import settings  # noqa: PLC0415

    if k > settings.max_factors_combinatorial:
        raise ValueError(
            f"_simplex_lattice: k={k} exceeds the SEC-19 cap of "
            f"{settings.max_factors_combinatorial}. "
            "Increase settings.max_factors_combinatorial if intentional."
        )
    n_points = math.comb(k + degree - 1, degree)
    if n_points > 1_000_000:
        raise ValueError(
            f"_simplex_lattice: the {{{k}, {degree}}} lattice has {n_points} points, "
            "more than the 1M cap. Reduce k or degree."
        )
    points = np.zeros((n_points, k))
    for row, parts in enumerate(itertools.combinations_with_replacement(range(k), degree)):
        np.add.at(points[row], list(parts), 1.0 / degree)
    return points


def _simplex_centroid(k: int, max_components: int | None = None) -> np.ndarray:
    """Generate a simplex-centroid design for *k* components.

    Includes:
    - k vertices (pure components)
    - k*(k-1)/2 binary midpoints
    - k*(k-1)*(k-2)/6 ternary centroids
    - ... up to the overall centroid (1/k, ..., 1/k)

    Parameters
    ----------
    k : int
        Number of mixture components.
    max_components : int or None
        Keep only the blends of at most this many components, plus the overall
        centroid. ``None`` keeps every blend: the full design.

    Returns
    -------
    np.ndarray
        Design matrix of shape (2^k - 1, k) for the full design.

    Raises
    ------
    ValueError
        If the full design is asked for with ``k`` above
        ``settings.max_factors_combinatorial`` (SEC-19 #268): it has ``2**k - 1``
        rows, and ``k=40`` would allocate ~1 TiB. A partial design is refused
        above 1,000,000 rows instead.
    """
    from process_improve.config import settings  # noqa: PLC0415

    sizes = list(range(1, k + 1))
    if max_components is not None and max_components < k - 1:
        sizes = [*range(1, max_components + 1), k]
        n_rows = sum(math.comb(k, r) for r in sizes)
        if n_rows > 1_000_000:
            raise ValueError(f"_simplex_centroid: {n_rows} rows exceed the 1M cap. Reduce k or max_components.")
    elif k > settings.max_factors_combinatorial:
        raise ValueError(
            f"_simplex_centroid: k={k} exceeds the SEC-19 cap of "
            f"{settings.max_factors_combinatorial}. A k={k} centroid design "
            f"has 2**{k} - 1 rows. Increase settings.max_factors_combinatorial "
            "if intentional."
        )
    points: list[list[float]] = []

    # Generate the non-empty subsets of {0, 1, ..., k-1} of the sizes kept
    for r in sizes:
        for subset in itertools.combinations(range(k), r):
            point = [0.0] * k
            proportion = 1.0 / r
            for idx in subset:
                point[idx] = proportion
            points.append(point)

    return np.array(points)
