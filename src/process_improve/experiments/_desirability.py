# (c) Kevin Dunn, 2010-2026. MIT License.
"""Derringer-Suich desirability functions, shared by the optimizer and the plots.

Two quantities in this module are easy to confuse, so they are named apart:

*weight*
    The exponent that shapes an *individual* desirability ramp, per response.
    A weight of 1 gives a linear ramp from 0 to 1 across the acceptable range.
    A weight above 1 concentrates desirability near the good end, so only
    values close to the target score well. A weight below 1 flattens the ramp,
    so anything inside the range scores nearly as well as the target.

*importance*
    The exponent applied when the individual desirabilities are combined into
    the composite. It sets how much one response counts relative to another,
    and it has no effect on any individual ramp.

This module was previously duplicated in ``optimization.py`` and in
``visualization/plots/optimization_plots.py``, and the two copies drifted.
It now holds the union of their behaviour: asymmetric target weights from the
plotting copy, and the strict argument checking from the optimizer copy.

References
----------
Derringer, G. and Suich, R. (1980). "Simultaneous Optimization of Several
Response Variables", *Journal of Quality Technology*, **12**, 214-219.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "composite_desirability",
    "desirability_maximize",
    "desirability_minimize",
    "desirability_target",
    "individual_desirability",
]


def desirability_maximize(y: float, low: float, high: float, weight: float = 1.0) -> float:
    """One-sided desirability for maximisation.

    Parameters
    ----------
    y : float
        Predicted response value.
    low : float
        Value at or below which the response is unacceptable, d = 0.
    high : float
        Value at or above which the response is fully acceptable, d = 1.
    weight : float
        Exponent shaping the ramp between *low* and *high*. See the module
        docstring for how this differs from importance.

    Returns
    -------
    float
        Desirability in [0, 1].
    """
    if y <= low:
        return 0.0
    if y >= high:
        return 1.0
    return ((y - low) / (high - low)) ** weight


def desirability_minimize(y: float, low: float, high: float, weight: float = 1.0) -> float:
    """One-sided desirability for minimisation.

    Parameters
    ----------
    y : float
        Predicted response value.
    low : float
        Value at or below which the response is fully acceptable, d = 1.
    high : float
        Value at or above which the response is unacceptable, d = 0.
    weight : float
        Exponent shaping the ramp between *low* and *high*.

    Returns
    -------
    float
        Desirability in [0, 1].
    """
    if y <= low:
        return 1.0
    if y >= high:
        return 0.0
    return ((high - y) / (high - low)) ** weight


def desirability_target(  # noqa: PLR0913
    y: float,
    low: float,
    target: float,
    high: float,
    weight_low: float = 1.0,
    weight_high: float = 1.0,
) -> float:
    """Two-sided desirability for a target value.

    Ramps up from *low* to *target*, then back down from *target* to *high*.
    Values outside ``[low, high]`` score zero.

    Parameters
    ----------
    y : float
        Predicted response value.
    low, target, high : float
        Lower bound, most desirable value, and upper bound.
    weight_low : float
        Exponent shaping the rising side, below the target.
    weight_high : float
        Exponent shaping the falling side, above the target.

    Returns
    -------
    float
        Desirability in [0, 1].
    """
    if y <= low or y >= high:
        return 0.0
    if y <= target:
        return ((y - low) / (target - low)) ** weight_low
    return ((high - y) / (high - target)) ** weight_high


def individual_desirability(y: float, goal: dict[str, Any]) -> float:
    """Desirability of one predicted response, given its goal.

    Parameters
    ----------
    y : float
        Predicted response value.
    goal : dict
        Keys: ``goal`` (one of ``"maximize"``, ``"minimize"``, ``"target"``),
        ``low``, ``high``, ``target`` (required for a target goal), ``weight``,
        and optionally ``weight_high`` to make a target goal asymmetric. When
        ``weight_high`` is absent, ``weight`` applies to both sides.

    Returns
    -------
    float
        Desirability in [0, 1].

    Raises
    ------
    ValueError
        If ``goal`` is missing, unrecognised, or a target goal has no target.
    """
    if "goal" not in goal:
        msg = "Goal dict is missing the required 'goal' key. Use 'maximize', 'minimize', or 'target'."
        raise ValueError(msg)

    goal_type = goal["goal"]
    low = goal.get("low", 0.0)
    high = goal.get("high", 1.0)
    weight = goal.get("weight", 1.0)

    if goal_type == "maximize":
        return desirability_maximize(y, low, high, weight)
    if goal_type == "minimize":
        return desirability_minimize(y, low, high, weight)
    if goal_type == "target":
        if "target" not in goal:
            msg = "A goal of 'target' requires a 'target' value in the goal dict."
            raise ValueError(msg)
        return desirability_target(y, low, goal["target"], high, weight, goal.get("weight_high", weight))

    msg = f"Unknown goal type: {goal_type!r}. Use 'maximize', 'minimize', or 'target'."
    raise ValueError(msg)


def composite_desirability(d_values: list[float], importances: list[float] | None = None) -> float:
    """Weighted geometric mean of the individual desirabilities.

    ``D = (d1^w1 * d2^w2 * ... * dk^wk) ^ (1 / sum(wi))``

    A single zero drives the composite to zero, whatever the other responses
    do. That is the defining property of the geometric mean here: a setting
    that fails one specification outright cannot be rescued by excelling
    elsewhere.

    Parameters
    ----------
    d_values : list of float
        Individual desirabilities, each in [0, 1].
    importances : list of float, optional
        Relative importance per response. Defaults to equal importance.

    Returns
    -------
    float
        Composite desirability in [0, 1]. Zero for an empty input, for any
        zero individual value, or when the importances sum to zero.
    """
    if not d_values:
        return 0.0

    weights = importances if importances is not None else [1.0] * len(d_values)

    if any(d == 0.0 for d in d_values):
        return 0.0

    log_d = sum(w * np.log(d) for d, w in zip(d_values, weights, strict=True))
    w_sum = sum(weights)
    if w_sum == 0:
        return 0.0
    return float(np.exp(log_d / w_sum))
