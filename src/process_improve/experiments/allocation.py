# (c) Kevin Dunn, 2010-2026. MIT License.
r"""How to split runs between the two levels of a factor, when the levels differ in noise or in cost.

To estimate the difference between two levels A and B from :math:`n_A` and :math:`n_B` runs, the
variance of the estimate is :math:`\sigma_A^2 / n_A + \sigma_B^2 / n_B`. With equal noise and equal
cost the even split is best; a noisier level deserves more runs, and a dearer one fewer. The best
split of a budget :math:`c_A n_A + c_B n_B \le C` puts :math:`n_A / n_B` near
:math:`(\sigma_A / \sqrt{c_A}) / (\sigma_B / \sqrt{c_B})` (Neyman's allocation), and
:func:`allocate_runs` finds it among the whole-run splits.

Examples
--------
>>> from process_improve.experiments.allocation import allocate_runs
>>> found = allocate_runs(variance_a=1, variance_b=9, budget=12)
>>> found["best"]["runs_a"], found["best"]["runs_b"]
(3, 9)
"""

from __future__ import annotations

import math
from typing import Any


def allocate_runs(
    variance_a: float,
    variance_b: float,
    cost_a: float = 1.0,
    cost_b: float = 1.0,
    budget: float = 12.0,
) -> dict[str, Any]:
    r"""Tabulate every whole-run split of a budget between two levels, and find the most precise.

    For each number of runs at level A, from one up, level B gets as many runs as the rest of the
    budget buys (at least one). Each split is scored by the variance of the estimated difference
    between the levels, :math:`\sigma_A^2 / n_A + \sigma_B^2 / n_B`, and by its efficiency, the
    smallest variance of any split over its own, in percent.

    Parameters
    ----------
    variance_a, variance_b : float
        The noise variance of a run at level A and at level B (any common units).
    cost_a, cost_b : float
        The cost of one run at each level. Default 1, so the budget is a run count.
    budget : float
        The total cost the runs may take, :math:`c_A n_A + c_B n_B \le` ``budget``.

    Returns
    -------
    dict[str, Any]
        ``splits``, one dict per split with ``runs_a``, ``runs_b``, ``variance`` and ``efficiency``
        (percent), in order of ``runs_a``; and ``best``, the split of least variance (the first
        of them, should two tie).

    Raises
    ------
    ValueError
        For a variance or cost that is not positive, or a budget too small for one run at each level.
    """
    if min(variance_a, variance_b) <= 0 or min(cost_a, cost_b) <= 0:
        raise ValueError("The variances and costs must be positive.")
    if budget < cost_a + cost_b:
        raise ValueError(f"A budget of {budget} does not buy one run at each level ({cost_a} + {cost_b}).")
    splits = []
    runs_a = 1
    while budget - cost_a * runs_a >= cost_b:
        runs_b = math.floor((budget - cost_a * runs_a) / cost_b + 1e-9)
        splits.append({"runs_a": runs_a, "runs_b": runs_b, "variance": variance_a / runs_a + variance_b / runs_b})
        runs_a += 1
    least = min(s["variance"] for s in splits)
    for s in splits:
        s["efficiency"] = 100.0 * (least / s["variance"])
    best = next(s for s in splits if s["variance"] == least)
    return {"splits": splits, "best": dict(best)}
