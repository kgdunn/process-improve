# (c) Kevin Dunn, 2010-2026. MIT License.
r"""Records whose neighbours in time are correlated: a first-order moving average, and the variance of a difference.

Successive batches of a process are often not independent: a high yield tends to follow a low one,
or another high one. Two consequences matter when such a record is used as a reference.

- **The variance of a difference of averages.** With a lag-1 autocorrelation :math:`\rho_1` and
  none beyond, the variance of the difference of two adjacent averages of :math:`n` consecutive
  observations is :math:`(2\sigma^2/n)\,[1 + \rho_1 (2n - 3)/n]`, not the :math:`2\sigma^2/n` of
  independent observations (Box, Hunter and Hunter, equation 18.5). A negative :math:`\rho_1`
  shrinks it, so a difference judged as if the observations were independent looks less
  significant than it is.
- **A model of the record.** The first-order moving average
  :math:`y_t = \mu + \varepsilon_t - \theta\varepsilon_{t-1}` has a lag-1 autocorrelation
  :math:`-\theta/(1 + \theta^2)` and none beyond it, which is what such a record shows.

Examples
--------
Box, Hunter and Hunter's 210 batch yields have :math:`s^2 = 8.30` and :math:`r_1 = -0.29`; for two
averages of 10 batches:

>>> from process_improve.univariate import difference_variance
>>> round(difference_variance(10, variance=8.30), 3), round(difference_variance(10, -0.29, 8.30), 3)
(1.66, 0.842)
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np

#: The ways :func:`fit_ma1` estimates the model.
MA1_METHODS = ("moments", "likelihood")


def difference_variance(n: int, rho1: float = 0.0, variance: float = 1.0, *, adjacent: bool = True) -> float:
    r"""Return the variance of the difference of two averages of ``n`` consecutive observations.

    The observations have variance :math:`\sigma^2` and lag-1 autocorrelation :math:`\rho_1`, with
    none beyond lag 1 (a first-order moving average has that pattern). The variance of one average
    is :math:`(\sigma^2/n)\,[1 + 2\rho_1(n - 1)/n]`. Two adjacent averages, the second block
    starting where the first ends, are themselves correlated through the one pair of neighbours that
    straddles them, so the variance of their difference is

    .. math::

        \frac{2\sigma^2}{n}\left[1 + \rho_1 \frac{2n - 3}{n}\right],

    which is Box, Hunter and Hunter's :math:`1 + 1.7\rho_1` for :math:`n = 10` (equation 18.5). Two
    averages far apart are not correlated, and the bracket is :math:`1 + 2\rho_1(n - 1)/n`.

    Parameters
    ----------
    n : int
        The number of observations in each average.
    rho1 : float
        The lag-1 autocorrelation of the observations, from -0.5 to 0.5: no process with
        autocorrelation at lag 1 only has one larger in size. 0 for independent observations.
    variance : float
        The variance :math:`\sigma^2` of one observation; 1 gives the variance in units of it.
    adjacent : bool
        True when the second average's observations follow the first's directly, as in a
        reference set of successive averages; False for two averages far apart.

    Returns
    -------
    float
        The variance of the difference of the two averages.

    Raises
    ------
    ValueError
        If ``n`` is not a whole number of at least 1, ``rho1`` is outside -0.5 to 0.5, or the
        variance is negative or not a finite number.
    """
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        msg = f"n, the number of observations in each average, must be a whole number of at least 1; got {n!r}."
        raise ValueError(msg)
    if not -0.5 <= rho1 <= 0.5:  # the bounds of a lag-1 autocorrelation with none beyond it
        msg = f"rho1 must be from -0.5 to 0.5, as a lag-1 autocorrelation with none beyond is; got {rho1}."
        raise ValueError(msg)
    if not (math.isfinite(variance) and variance >= 0):
        msg = f"The variance must be a finite number, 0 or more; got {variance}."
        raise ValueError(msg)
    pairs = 2 * n - 3 if adjacent else 2 * (n - 1)  # neighbours within the blocks, less the pair across them
    return 2 * variance / n * (1 + rho1 * pairs / n)


def fit_ma1(y: Sequence[float], method: str = "moments") -> dict[str, Any]:
    r"""Fit a first-order moving average, :math:`y_t = \mu + \varepsilon_t - \theta\varepsilon_{t-1}`, to a record.

    ``method="moments"`` matches the model to the record's lag-1 autocorrelation :math:`r_1` and
    variance :math:`s^2`, as Box, Hunter and Hunter do (section 18.2): the model's lag-1
    autocorrelation is :math:`-\theta/(1 + \theta^2)`, so :math:`\theta` is the root of
    :math:`r_1\theta^2 + \theta + r_1 = 0` between -1 and 1, and
    :math:`\sigma_\varepsilon^2 = s^2/(1 + \theta^2)`. No such root exists when :math:`r_1` is 0.5
    or more in size.

    ``method="likelihood"`` maximizes the likelihood of the whole record (statsmodels' ARIMA of
    order (0, 0, 1)), which uses every autocorrelation, not only the first, and gives a standard
    error for :math:`\theta`. The two estimates can differ a good deal on a short record.

    Parameters
    ----------
    y : Sequence[float]
        The record, in time order: at least three observations, not all the same.
    method : str
        ``"moments"`` or ``"likelihood"``.

    Returns
    -------
    dict[str, Any]
        ``mean`` (:math:`\hat\mu`), ``theta``, ``sigma2`` (:math:`\hat\sigma_\varepsilon^2`),
        ``r1`` (the record's lag-1 autocorrelation), ``theta_std_error`` (None for moments),
        ``n`` and ``method``.

    Raises
    ------
    ValueError
        If the method is not one of :data:`MA1_METHODS`, the record has fewer than three
        observations, a value is not a finite number, every value is the same, or (by moments) the
        lag-1 autocorrelation is 0.5 or more in size.

    Examples
    --------
    The first twelve of Box, Hunter and Hunter's batch yields:

    >>> found = fit_ma1([85.5, 81.7, 80.6, 84.7, 88.2, 84.9, 81.8, 84.9, 85.2, 81.9, 89.4, 79.0])
    >>> round(found["r1"], 3), round(found["theta"], 3), round(found["sigma2"], 2)
    (-0.337, 0.388, 8.19)
    """
    if method not in MA1_METHODS:
        msg = f"method must be one of {MA1_METHODS}; got {method!r}."
        raise ValueError(msg)
    values = np.asarray(y, dtype=float)
    if values.ndim != 1 or len(values) < 3 or not np.all(np.isfinite(values)):
        msg = "The record must be at least three finite numbers, in time order."
        raise ValueError(msg)
    centred = values - values.mean()
    total = float(centred @ centred)
    if total == 0:
        msg = "Every value of the record is the same, so it has no autocorrelation to fit."
        raise ValueError(msg)
    r1 = float(centred[:-1] @ centred[1:]) / total
    if method == "likelihood":
        return _ma1_likelihood(values, r1)
    if abs(r1) >= 0.5:  # a first-order moving average has a lag-1 autocorrelation of less than 0.5 in size
        msg = (
            f"The record's lag-1 autocorrelation is {r1:.3f}; a first-order moving average has one of less than "
            "0.5 in size, so none matches it. Use method='likelihood', or a model of higher order."
        )
        raise ValueError(msg)
    theta = 0.0 if r1 == 0 else (-1 + math.sqrt(1 - 4 * r1**2)) / (2 * r1)
    variance = float(values.var(ddof=1))
    return {
        "mean": float(values.mean()),
        "theta": theta,
        "sigma2": variance / (1 + theta**2),
        "r1": r1,
        "theta_std_error": None,
        "n": len(values),
        "method": "moments",
    }


def _ma1_likelihood(values: np.ndarray, r1: float) -> dict[str, Any]:
    r"""Fit the first-order moving average by maximum likelihood, with statsmodels' ARIMA of order (0, 0, 1).

    statsmodels writes the model with :math:`+\theta' \varepsilon_{t-1}`, so :math:`\theta = -\theta'`.
    """
    from statsmodels.tsa.arima.model import ARIMA  # noqa: PLC0415 - loaded only for this method

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # statsmodels' notes on the starting values it chose
        fit = ARIMA(values, order=(0, 0, 1)).fit()
    params = dict(zip(fit.param_names, np.asarray(fit.params, dtype=float), strict=True))
    errors = dict(zip(fit.param_names, np.asarray(fit.bse, dtype=float), strict=True))
    return {
        "mean": params["const"],
        "theta": -params["ma.L1"],
        "sigma2": params["sigma2"],
        "r1": r1,
        "theta_std_error": errors["ma.L1"],
        "n": len(values),
        "method": "likelihood",
    }
