# (c) Kevin Dunn, 2010-2026. MIT License.

"""Optimal designs: D-optimal and I-optimal.

D-optimal wraps the existing ``point_exchange()`` in ``optimal.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pyDOE3 import fullfact

from process_improve.experiments.optimal import point_exchange

if TYPE_CHECKING:
    from process_improve.experiments.factor import Factor


def dispatch_d_optimal(
    factors: list[Factor],
    budget: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Generate a D-optimal design by selecting from a candidate set.

    Builds a candidate set from all combinations of factor levels (including
    center points and axial points for continuous factors), then uses the
    existing ``point_exchange`` algorithm to select an optimal subset.

    Parameters
    ----------
    factors : list[Factor]
        Factor specifications.
    budget : int or None
        Number of runs to select.  Defaults to ``2 * n_factors + 1``.

    Returns
    -------
    tuple[np.ndarray, dict]
        Selected design matrix in coded units and metadata with
        ``d_optimality`` value.
    """
    k = len(factors)
    if budget is None:
        budget = 2 * k + 1

    # Build candidate set: 3-level full factorial (-1, 0, +1) for each factor
    levels_per_factor = [3] * k
    candidates_raw = fullfact(levels_per_factor)
    # fullfact returns 0-based: map 0->-1, 1->0, 2->+1
    candidates = candidates_raw - 1.0

    candidates_df = pd.DataFrame(candidates, columns=[f.name for f in factors])

    n_points = min(budget, candidates_df.shape[0])
    n_points = max(n_points, k + 1)  # need at least k+1 points for estimability

    design_df, d_opt = point_exchange(candidates_df, number_points=n_points)

    return design_df.values, {"d_optimality": float(d_opt)}


def dispatch_i_optimal(
    factors: list[Factor],
    budget: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Generate an I-optimal design (minimizes average prediction variance).

    .. note::
        I-optimal design generation requires a specialized algorithm
        (V-optimal coordinate exchange).  This is not yet implemented.
        Currently raises ``NotImplementedError``.

    Parameters
    ----------
    factors : list[Factor]
        Factor specifications.
    budget : int or None
        Number of runs.

    Raises
    ------
    NotImplementedError
        Always, until a proper I-optimal algorithm is implemented.
    """
    raise NotImplementedError(
        "I-optimal design generation is not yet implemented. "
        "Consider using 'd_optimal' as an alternative."
    )
