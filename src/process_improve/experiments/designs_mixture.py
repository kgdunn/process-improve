# (c) Kevin Dunn, 2010-2026. MIT License.

"""Mixture designs: simplex-lattice and simplex-centroid.

For mixture experiments the factor levels represent proportions that must
sum to 1.  These designs operate directly in actual proportions rather
than coded -1/+1 units.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from process_improve.experiments.factor import Factor


def dispatch_mixture(
    factors: list[Factor],
    budget: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Generate a mixture design (simplex-lattice or simplex-centroid).

    Selects the design type automatically:
    - If *budget* is large enough for a simplex-centroid, use that.
    - Otherwise fall back to a simplex-lattice of degree 2 or 3.

    Parameters
    ----------
    factors : list[Factor]
        Mixture factors (proportions summing to 1).
    budget : int or None
        Maximum number of runs.

    Returns
    -------
    tuple[np.ndarray, dict]
        Design matrix (in proportions, not coded) and metadata.
    """
    k = len(factors)
    if k < 2:
        raise ValueError("Mixture designs require at least 2 components.")

    # Simplex-centroid has 2^k - 1 points
    n_centroid = 2**k - 1

    if budget is not None and budget < n_centroid:
        # Use simplex-lattice degree 2 (has k*(k+1)/2 points)
        matrix = _simplex_lattice(k, degree=2)
        method = "simplex_lattice_degree_2"
    else:
        matrix = _simplex_centroid(k)
        method = "simplex_centroid"

    return matrix, {"method": method}


def _simplex_lattice(k: int, degree: int = 2) -> np.ndarray:
    """Generate a {k, degree} simplex-lattice design.

    The lattice consists of all points where each component takes values
    from {0, 1/degree, 2/degree, ..., 1} and all components sum to 1.

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
        If ``(degree + 1) ** k`` exceeds 1,000,000 combinations
        (SEC-19 #268). ``itertools.product`` would otherwise iterate
        ~5 x 10^11 tuples for ``k=15, degree=5``.
    """
    from process_improve.config import settings  # noqa: PLC0415

    if k > settings.max_factors_combinatorial:
        raise ValueError(
            f"_simplex_lattice: k={k} exceeds the SEC-19 cap of "
            f"{settings.max_factors_combinatorial}. "
            "Increase settings.max_factors_combinatorial if intentional."
        )
    estimated_tuples = (degree + 1) ** k
    if estimated_tuples > 1_000_000:
        raise ValueError(
            f"_simplex_lattice: (degree+1)**k = {estimated_tuples} tuples "
            "exceeds the 1M iteration cap. Reduce k or degree."
        )
    grid_values = [i / degree for i in range(degree + 1)]
    points = [list(combo) for combo in itertools.product(grid_values, repeat=k) if abs(sum(combo) - 1.0) < 1e-10]
    return np.array(points)


def _simplex_centroid(k: int) -> np.ndarray:
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

    Returns
    -------
    np.ndarray
        Design matrix of shape (2^k - 1, k).

    Raises
    ------
    ValueError
        If ``k`` exceeds ``settings.max_factors_combinatorial``
        (SEC-19 #268). The design has ``2**k - 1`` rows; ``k=40``
        would allocate ~1 TiB.
    """
    from process_improve.config import settings  # noqa: PLC0415

    if k > settings.max_factors_combinatorial:
        raise ValueError(
            f"_simplex_centroid: k={k} exceeds the SEC-19 cap of "
            f"{settings.max_factors_combinatorial}. A k={k} centroid design "
            f"has 2**{k} - 1 rows. Increase settings.max_factors_combinatorial "
            "if intentional."
        )
    points: list[list[float]] = []

    # Generate all non-empty subsets of {0, 1, ..., k-1}
    for r in range(1, k + 1):
        for subset in itertools.combinations(range(k), r):
            point = [0.0] * k
            proportion = 1.0 / r
            for idx in subset:
                point[idx] = proportion
            points.append(point)

    return np.array(points)
