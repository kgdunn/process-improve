# (c) Kevin Dunn, 2010-2026. MIT License.
r"""Error transmission: how the errors of measured quantities pass into a quantity worked out from them.

A quantity :math:`y = f(x_1, \dots, x_k)` worked out from measured quantities carries their
errors. When each input varies a little about its mean :math:`\mu_i`, with standard deviation
:math:`\sigma_i`, and the errors are independent, the straight line that touches :math:`f` at the
means (its local linearization) gives the variance of :math:`y`:

.. math::

    \operatorname{Var}(y) \approx \sum_i \left(\frac{\partial f}{\partial x_i}\right)^2 \sigma_i^2 .

Each term is the variance that one input transmits, so the terms say which measurement to improve
first (Box, Hunter and Hunter, section 17.2). The approximation holds while :math:`f` is nearly
straight over a few standard deviations of each input. The linearity check works :math:`f` out at
:math:`\mu_i - 3\sigma_i` and :math:`\mu_i + 3\sigma_i`, the other inputs at their means, and sets
each change beside the straight line's :math:`\pm 3\sigma_i\,\partial f/\partial x_i`.

Examples
--------
The volume of a sphere from its diameter, :math:`V = \pi d^3 / 6`, measured with a standard
deviation of 0.01 about 2.0:

>>> import math
>>> from process_improve.univariate import error_transmission
>>> found = error_transmission(lambda d: math.pi * d**3 / 6, means={"d": 2.0}, std_devs={"d": 0.01})
>>> round(found["value"], 4), round(found["slopes"]["d"], 4), round(found["std_dev"], 4)
(4.1888, 6.2832, 0.0628)
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

#: How far each input moves for the linearity check, in its own standard deviations.
CHECK_SDS = 3.0

#: The step of the central difference that gives each slope, as a share of the input's standard deviation.
STEP = 1e-4


def error_transmission(
    func: Callable[..., float], means: Mapping[str, float], std_devs: Mapping[str, float]
) -> dict[str, Any]:
    r"""Return the variance a function transmits from the errors of its inputs, by local linearization.

    Each slope :math:`\partial f/\partial x_i` is a central difference at the means, over a step of
    :data:`STEP` of the input's standard deviation (or of its mean's size, for an input known
    exactly). The transmitted variance is :math:`\sum_i (\partial f/\partial x_i)^2 \sigma_i^2`,
    for inputs whose errors are independent.

    Parameters
    ----------
    func : Callable[..., float]
        The quantity worked out from the inputs, called with each input by name:
        ``func(d=2.0, h=5.0)``.
    means : Mapping[str, float]
        Each input's mean: the point the function is linearized at.
    std_devs : Mapping[str, float]
        Each input's standard deviation, for the same inputs; 0 for an input known exactly.

    Returns
    -------
    dict[str, Any]
        - ``value``: the function at the means.
        - ``variance`` and ``std_dev``: the transmitted variance, and its square root.
        - ``slopes``: each input's partial derivative at the means.
        - ``contributions``: the variance each input transmits, :math:`(\partial f/\partial x_i)^2 \sigma_i^2`.
        - ``shares``: each contribution as a fraction of the variance; empty when the variance is 0.
        - ``linearity``: for each input with a standard deviation above 0, the function at
          :math:`\mu_i - 3\sigma_i` and :math:`\mu_i + 3\sigma_i` (``low`` and ``high``), the
          changes from ``value`` there (``down`` and ``up``), and the straight line's change,
          :math:`3\sigma_i\,\partial f/\partial x_i` (``linear``). When ``-down`` and ``up``
          are both close to ``linear``, the function is close to a straight line over the input's
          spread, and the variance can be trusted.

    Raises
    ------
    ValueError
        If ``means`` and ``std_devs`` name different inputs, a mean is not a finite number, a
        standard deviation is negative or not a finite number, or the function is not a finite
        number at a point it is worked out at.

    Examples
    --------
    The variance of a product, :math:`y = x_1 x_2`, is close to
    :math:`x_2^2 \sigma_1^2 + x_1^2 \sigma_2^2`:

    >>> found = error_transmission(lambda a, b: a * b, means={"a": 3.0, "b": 4.0}, std_devs={"a": 0.1, "b": 0.2})
    >>> round(found["variance"], 6), {k: round(v, 6) for k, v in found["contributions"].items()}
    (0.52, {'a': 0.16, 'b': 0.36})
    """
    names = list(means)
    if set(std_devs) != set(names):
        msg = f"means and std_devs must name the same inputs; got {sorted(names)} and {sorted(std_devs)}."
        raise ValueError(msg)
    mu = {name: float(means[name]) for name in names}
    sigma = {name: float(std_devs[name]) for name in names}
    if not all(math.isfinite(value) for value in mu.values()):
        msg = f"Each mean must be a finite number; got {mu}."
        raise ValueError(msg)
    if not all(math.isfinite(value) and value >= 0 for value in sigma.values()):
        msg = f"Each standard deviation must be a finite number, 0 or more; got {sigma}."
        raise ValueError(msg)

    def at(**moved: float) -> float:
        point = {**mu, **moved}
        value = float(func(**point))
        if not math.isfinite(value):
            msg = f"The function is {value} at {point}: each value it is worked out at must be a finite number."
            raise ValueError(msg)
        return value

    value = at()
    slopes = {}
    for name in names:
        step = STEP * (sigma[name] if sigma[name] > 0 else max(abs(mu[name]), 1.0))
        slopes[name] = (at(**{name: mu[name] + step}) - at(**{name: mu[name] - step})) / (2 * step)
    contributions = {name: slopes[name] ** 2 * sigma[name] ** 2 for name in names}
    variance = math.fsum(contributions.values())
    linearity = {}
    for name in (n for n in names if sigma[n] > 0):
        low = at(**{name: mu[name] - CHECK_SDS * sigma[name]})
        high = at(**{name: mu[name] + CHECK_SDS * sigma[name]})
        linearity[name] = {
            "low": low,
            "high": high,
            "down": low - value,
            "up": high - value,
            "linear": CHECK_SDS * sigma[name] * slopes[name],
        }
    return {
        "value": value,
        "variance": variance,
        "std_dev": math.sqrt(variance),
        "slopes": slopes,
        "contributions": contributions,
        "shares": {name: part / variance for name, part in contributions.items()} if variance > 0 else {},
        "linearity": linearity,
    }
