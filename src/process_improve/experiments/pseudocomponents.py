# (c) Kevin Dunn, 2010-2026. MIT License.
r"""Pseudocomponents: a mixture region with lower or upper bounds, put back on the whole simplex.

When each component :math:`i` of a mixture must be at least :math:`L_i`, the blends that remain form
a smaller simplex of the same shape inside the original (Cornell; Goos and Jones, section 6.3.4).
The L-pseudocomponents

.. math::

    x'_i = \frac{x_i - L_i}{T - \sum_j L_j}, \qquad x_i = L_i + \left(T - \sum_j L_j\right) x'_i,

with :math:`T` the mixture's total (1 for proportions, 100 for percentages), map that smaller
simplex onto the whole one: a component at its lower bound has :math:`x'_i = 0`, and
:math:`x'_i = 1` is the blend with every other component at its lower bound. A design built for
the full simplex, a simplex lattice or centroid, is then made in pseudocomponents and converted
back, and a Scheffe model fitted in pseudocomponents is better conditioned than one fitted in the
proportions of a small region.

Upper bounds :math:`U_i` cut out an inverted simplex, the U-simplex, with the U-pseudocomponents
(Crosier 1984; Box and Draper 2007, Section 17.3)

.. math::

    u_i = \frac{U_i - x_i}{U - T}, \qquad x_i = U_i - (U - T)\,u_i, \qquad U = \sum_j U_j.

A component at its upper bound has :math:`u_i = 0`. The U-simplex lies wholly inside the mixture
simplex only when :math:`U - U_{\min} \le T` (:func:`u_simplex_check`); otherwise part of it
holds blends with a negative proportion.

Examples
--------
Lower bounds of 0.2, 0.2 and 0: the pure first pseudocomponent is the blend (0.8, 0.2, 0.0).

>>> from_pseudocomponents([1.0, 0.0, 0.0], [0.2, 0.2, 0.0]).tolist()
[0.8, 0.2, 0.0]
>>> to_pseudocomponents([0.5, 0.3, 0.2], [0.2, 0.2, 0.0]).round(4).tolist()
[0.5, 0.1667, 0.3333]

Upper bounds of 0.50, 0.45 and 0.40: the pure first U-pseudocomponent is (0.15, 0.45, 0.40).

>>> from_pseudocomponents([1.0, 0.0, 0.0], upper=[0.50, 0.45, 0.40]).round(4).tolist()
[0.15, 0.45, 0.4]
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

Blends = pd.DataFrame | np.ndarray | Sequence[float] | Sequence[Sequence[float]]
Bounds = Mapping[str, float] | Sequence[float]


def to_pseudocomponents(
    blends: Blends, lower: Bounds | None = None, total: float = 1.0, *, upper: Bounds | None = None
) -> Blends:
    r"""Return blends as L-pseudocomponents, or as U-pseudocomponents when *upper* is given.

    :math:`x'_i = (x_i - L_i)/(T - \sum_j L_j)` from the lower bounds, or
    :math:`u_i = (U_i - x_i)/(\sum_j U_j - T)` from the upper bounds.

    Parameters
    ----------
    blends : pandas.DataFrame, numpy.ndarray or Sequence
        The blends, a component per column (a single blend may be one sequence).
    lower : Mapping[str, float] or Sequence[float], optional
        The lower bound of each component: by column name for a DataFrame (a component not named
        has the bound 0), or one per column. Give this or *upper*, not both.
    total : float
        The mixture's total, :math:`T`: 1 for proportions, 100 for percentages.
    upper : Mapping[str, float] or Sequence[float], optional
        The upper bound of each component, as *lower* takes them (a component not named has the
        bound *total*).

    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        The pseudocomponents, in the shape and type of ``blends`` (a DataFrame keeps its index and
        columns; a sequence comes back as an array).

    Raises
    ------
    ValueError
        If not exactly one of *lower* and *upper* is given, a bound is negative or not a finite
        number, the bounds leave no room (:math:`\sum_j L_j \ge T`, or :math:`\sum_j U_j \le T`),
        the bounds do not match the components, or a blend has a value that is not a finite number.
    TypeError
        If the bounds name the components but ``blends`` is not a DataFrame.
    """
    values, bounds, span, sign = _prepare(blends, lower, upper, total)
    return _like(blends, sign * (values - bounds) / span)


def from_pseudocomponents(
    pseudocomponents: Blends, lower: Bounds | None = None, total: float = 1.0, *, upper: Bounds | None = None
) -> Blends:
    r"""Return L- or U-pseudocomponents as blends.

    :math:`x_i = L_i + (T - \sum_j L_j)\,x'_i` from the lower bounds, or
    :math:`x_i = U_i - (\sum_j U_j - T)\,u_i` from the upper bounds.

    Parameters
    ----------
    pseudocomponents : pandas.DataFrame, numpy.ndarray or Sequence
        The pseudocomponents, a component per column (a single blend may be one sequence).
    lower : Mapping[str, float] or Sequence[float], optional
        The lower bound of each component, as :func:`to_pseudocomponents` takes them.
    total : float
        The mixture's total, :math:`T`.
    upper : Mapping[str, float] or Sequence[float], optional
        The upper bound of each component, for U-pseudocomponents.

    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        The blends, in the shape and type of ``pseudocomponents``.

    Raises
    ------
    ValueError
        As :func:`to_pseudocomponents`.
    """
    values, bounds, span, sign = _prepare(pseudocomponents, lower, upper, total)
    return _like(pseudocomponents, bounds + sign * span * values)


def u_simplex_check(upper: Sequence[float], total: float = 1.0) -> dict[str, float | bool]:
    r"""Return whether the U-simplex of upper bounds lies wholly inside the mixture simplex.

    The upper bounds :math:`U_i` cut out an inverted simplex whose vertices put every component but
    one at its bound. It lies inside the mixture simplex when no vertex needs a negative proportion,
    that is when :math:`U - U_{\min} \le T`, with :math:`U = \sum_j U_j` (Box and Draper 2007,
    Eq. 17.3.2). Only then are the U-pseudocomponents of :func:`to_pseudocomponents` a whole simplex
    of blends; otherwise the region is the part of the U-simplex inside the mixture simplex.

    Parameters
    ----------
    upper : Sequence[float]
        The upper bound of each component.
    total : float
        The mixture's total, :math:`T`.

    Returns
    -------
    dict
        ``upper_sum``, :math:`U`; ``upper_sum_less_smallest``, :math:`U - U_{\min}`; and ``inside``,
        True when the U-simplex lies wholly inside the mixture simplex.

    Raises
    ------
    ValueError
        As :func:`to_pseudocomponents` for the bounds.

    Examples
    --------
    Box and Draper's Figure 17.4a, upper bounds 0.50, 0.45 and 0.40:

    >>> check = u_simplex_check([0.50, 0.45, 0.40])
    >>> round(check["upper_sum"], 2), round(check["upper_sum_less_smallest"], 2), check["inside"]
    (1.35, 0.95, True)
    """
    bounds = np.asarray(upper, dtype=float).ravel()
    _prepare(bounds, None, upper, total)  # validates the bounds as the conversions do
    upper_sum = float(bounds.sum())
    less_smallest = upper_sum - float(bounds.min())
    return {
        "upper_sum": upper_sum,
        "upper_sum_less_smallest": less_smallest,
        "inside": bool(less_smallest <= total + 1e-12 * max(1.0, abs(total))),
    }


def _prepare(
    blends: Blends, lower: Bounds | None, upper: Bounds | None, total: float
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return the values as an array, each column's bound, the room the bounds leave and the direction.

    The direction is 1 for lower bounds and -1 for upper bounds, so one formula serves both.
    """
    if (lower is None) == (upper is None):
        msg = "Give the lower bounds or the upper bounds of the components, not both and not neither."
        raise ValueError(msg)
    values = np.asarray(blends, dtype=float)
    if values.ndim not in (1, 2) or not np.all(np.isfinite(values)):
        msg = "The blends must be finite numbers, a component per column."
        raise ValueError(msg)
    side, given = ("lower", lower) if lower is not None else ("upper", upper)
    bounds = _bounds(blends, given, values.shape[-1], side, default=0.0 if side == "lower" else float(total))
    if side == "lower":
        span, sign = float(total) - float(bounds.sum()), 1.0
    else:
        span, sign = float(bounds.sum()) - float(total), -1.0
    if not np.isfinite(span) or span <= 0:
        relation = "leaves no room in" if side == "lower" else "is not above"
        msg = f"The {side} bounds add to {bounds.sum():g}, which {relation} a total of {total:g}."
        raise ValueError(msg)
    return values, bounds, span, sign


def _bounds(blends: Blends, given: Bounds | None, k: int, side: str, default: float) -> np.ndarray:
    """Return one bound per column, from a mapping by name or a sequence."""
    if isinstance(given, Mapping):
        if not isinstance(blends, pd.DataFrame):
            msg = (
                f"{side.capitalize()} bounds by name need the blends as a DataFrame with the components' names "
                "as columns."
            )
            raise TypeError(msg)
        unknown = sorted(set(map(str, given)) - {str(c) for c in blends.columns})
        if unknown:
            msg = f"{side} names {unknown}, which are not columns of the blends {list(blends.columns)}."
            raise ValueError(msg)
        named = {str(name): float(bound) for name, bound in given.items()}
        bounds = np.array([named.get(str(column), default) for column in blends.columns])
    else:
        bounds = np.asarray(given, dtype=float).ravel()
        if bounds.size != k:
            msg = f"{side} must give one bound per component, {k}; got {bounds.size}."
            raise ValueError(msg)
    if not np.all(np.isfinite(bounds)) or np.any(bounds < 0):
        msg = f"Each {side} bound must be a finite number, 0 or more; got {bounds.tolist()}."
        raise ValueError(msg)
    return bounds


def _like(template: Blends, values: np.ndarray) -> Blends:
    """Return ``values`` as ``template`` was given: a DataFrame keeps its index and columns."""
    if isinstance(template, pd.DataFrame):
        return pd.DataFrame(values, index=template.index, columns=template.columns)
    return values
