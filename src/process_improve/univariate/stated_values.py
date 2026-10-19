# (c) Kevin Dunn, 2010-2026. MIT License.
r"""Judge results against values stated before the data: a known mean, a new observation, a norm.

Box, Hunter and Hunter (chapters 2 and 6) set results against values that were known, or stated, before
the data were taken:

- **A variance about a known mean.** When the mean :math:`\eta` is known, the squared deviations from
  it estimate :math:`\sigma^2` on all :math:`n` degrees of freedom, :math:`\sum (y - \eta)^2 / n`: no
  degree of freedom is spent estimating the mean.
- **One new observation.** A single new value :math:`y` is set against the stated mean and scaled by
  the standard deviation of earlier data, :math:`t = (y - \eta)/s`, on the degrees of freedom of
  :math:`s`: :math:`n` when :math:`s` is taken about the known mean, :math:`n - 1` when about the
  earlier data's own average.
- **A grand mean against a norm.** In a one-way layout the sum of squares about a norm :math:`\eta_0`
  splits into three parts,

  .. math::

      \sum_{ij} (y_{ij} - \eta_0)^2 = N(\bar y - \eta_0)^2 + \sum_i n_i (\bar y_i - \bar y)^2
          + \sum_{ij} (y_{ij} - \bar y_i)^2,

  on 1, :math:`a - 1` and :math:`N - a` degrees of freedom, and the grand mean is tested against the
  norm with the residual's mean square, which the differences between the groups do not inflate.

Examples
--------
Box, Hunter and Hunter's five cars (Exercise 2.12), whose mean is known to be 25.0, and a new car of
the same kind that gives 28.2:

>>> cars = [27.8, 24.3, 22.8, 26.0, 24.2]
>>> round(variance_about(cars, 25.0)["variance"], 3)
2.962
>>> round(ttest_new_observation(28.2, 25.0, cars)["t_value"], 2)
1.86
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from process_improve.univariate.classical import _p_values


def variance_about(x: Sequence[float] | np.ndarray, mean: float) -> dict[str, float]:
    r"""Return the variance of observations about a known mean, on all :math:`n` degrees of freedom.

    Parameters
    ----------
    x : Sequence[float] or numpy.ndarray
        The observations.
    mean : float
        The mean :math:`\eta`, known before the data: 0 for differences that average to zero, for
        example.

    Returns
    -------
    dict[str, float]
        ``variance`` (:math:`\sum (y - \eta)^2 / n`), ``std_dev`` (its square root) and ``df``
        (:math:`n`).

    Raises
    ------
    ValueError
        If there are no observations, or a value or the mean is not a finite number.
    """
    values = _finite(x, "x")
    deviations = values - _finite_number(mean, "mean")
    variance = float(deviations @ deviations / len(values))
    return {"variance": variance, "std_dev": math.sqrt(variance), "df": len(values)}


def ttest_new_observation(
    y: float, mean: float, x: Sequence[float] | np.ndarray, mean_known: bool = True
) -> dict[str, float]:
    r"""Test one new observation against a stated mean, scaled by the standard deviation of earlier data.

    :math:`t = (y - \eta)/s`, with :math:`s` from the earlier observations *x*: about the stated mean
    on :math:`n` degrees of freedom when the mean is known, or about their own average on
    :math:`n - 1` when it is not.

    Parameters
    ----------
    y : float
        The new observation.
    mean : float
        The stated mean :math:`\eta`.
    x : Sequence[float] or numpy.ndarray
        Earlier observations of the same kind, which give :math:`s`.
    mean_known : bool
        Whether *mean* is the known mean of the earlier observations, so that :math:`s` is taken
        about it.

    Returns
    -------
    dict[str, float]
        ``t_value``, ``df``, ``std_dev`` (:math:`s`), the two-sided ``p_value`` and the one-sided
        ``p_value_greater`` and ``p_value_less``.

    Raises
    ------
    ValueError
        If a value is not a finite number, or *x* leaves no degree of freedom for :math:`s`.
    """
    values = _finite(x, "x")
    if mean_known:
        spread = variance_about(values, mean)
        std_dev, df = spread["std_dev"], spread["df"]
    else:
        if len(values) < 2:
            msg = "A standard deviation about the earlier data's own average needs at least two of them."
            raise ValueError(msg)
        std_dev, df = float(values.std(ddof=1)), len(values) - 1
    t_value = (_finite_number(y, "y") - _finite_number(mean, "mean")) / std_dev
    return {"t_value": t_value, "df": df, "std_dev": std_dev, **_p_values(t_value, stats.t(df))}


def grand_mean_test(y: Sequence[float] | np.ndarray, groups: Sequence[Hashable], value: float) -> dict[str, Any]:
    r"""Split a one-way layout's sum of squares about a stated value, and test the grand mean against it.

    Parameters
    ----------
    y : Sequence[float] or numpy.ndarray
        The observations.
    groups : Sequence[Hashable]
        The group of each observation (a treatment, a diet, a supplier).
    value : float
        The norm :math:`\eta_0`, stated before the data.

    Returns
    -------
    dict[str, Any]
        - ``sum_of_squares`` and ``df``, each with the keys ``total`` (about the norm), ``mean``
          (:math:`N(\bar y - \eta_0)^2`, 1 degree of freedom), ``groups`` (between the group averages,
          :math:`a - 1`) and ``residual`` (within the groups, :math:`N - a`); the first is the sum of
          the other three;
        - ``grand_mean``, :math:`\bar y`, and ``std_error``, :math:`\sqrt{s^2 / N}` with :math:`s^2` the
          residual mean square;
        - ``t_value`` (:math:`(\bar y - \eta_0)` over its standard error) on ``df_resid``
          (:math:`N - a`) degrees of freedom, with the two-sided ``p_value`` and the one-sided
          ``p_value_greater`` and ``p_value_less``.

    Raises
    ------
    ValueError
        If *y* and *groups* differ in length, a value is not a finite number, or the groups leave
        no residual degree of freedom (every group a single observation).

    Examples
    --------
    Box, Hunter and Hunter's blood coagulation times of 24 animals on four diets (Table 6.1), against
    the norm of 63:

    >>> times = [62, 60, 63, 59, 63, 67, 71, 64, 65, 66, 68, 66, 71, 67, 68, 68, 56, 62, 60, 61, 63, 64, 63, 59]
    >>> diets = list("AAAA" "BBBBBB" "CCCCCC" "DDDDDDDD")
    >>> found = grand_mean_test(times, diets, 63)
    >>> {part: round(ss) for part, ss in found["sum_of_squares"].items()}
    {'total': 364, 'mean': 24, 'groups': 228, 'residual': 112}
    >>> round(found["t_value"], 2), found["df_resid"]
    (2.07, 20)
    """
    values = _finite(y, "y")
    labels = pd.Series(list(groups))
    if len(labels) != len(values):
        msg = f"y and groups must have the same length; got {len(values)} and {len(labels)}."
        raise ValueError(msg)
    norm = _finite_number(value, "value")
    n, n_groups = len(values), labels.nunique()
    if n <= n_groups:
        msg = f"{n} observations in {n_groups} groups leave no residual degree of freedom to test against."
        raise ValueError(msg)
    group_means = pd.Series(values).groupby(labels.to_numpy()).transform("mean").to_numpy()
    grand_mean = float(values.mean())
    sums = {
        "total": float(np.sum((values - norm) ** 2)),
        "mean": n * (grand_mean - norm) ** 2,
        "groups": float(np.sum((group_means - grand_mean) ** 2)),
        "residual": float(np.sum((values - group_means) ** 2)),
    }
    df_resid = n - n_groups
    std_error = math.sqrt(sums["residual"] / df_resid / n)
    t_value = (grand_mean - norm) / std_error
    return {
        "sum_of_squares": sums,
        "df": {"total": n, "mean": 1, "groups": n_groups - 1, "residual": df_resid},
        "grand_mean": grand_mean,
        "std_error": std_error,
        "t_value": t_value,
        "df_resid": df_resid,
        **_p_values(t_value, stats.t(df_resid)),
    }


def _finite(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    """Return *values* as a one-dimensional float array, refusing an empty one or one that is not all finite."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        msg = f"{name} must be a non-empty sequence of finite numbers."
        raise ValueError(msg)
    return array


def _finite_number(value: float, name: str) -> float:
    """Return *value* as a float, refusing one that is not a finite number."""
    number = float(value)
    if not math.isfinite(number):
        msg = f"{name} must be a finite number; got {value!r}."
        raise ValueError(msg)
    return number
