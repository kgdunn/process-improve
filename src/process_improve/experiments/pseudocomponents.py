# (c) Kevin Dunn, 2010-2026. MIT License.
r"""L-pseudocomponents: a mixture region with lower bounds, put back on the whole simplex.

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

Examples
--------
Lower bounds of 0.2, 0.2 and 0: the pure first pseudocomponent is the blend (0.8, 0.2, 0.0).

>>> from_pseudocomponents([1.0, 0.0, 0.0], [0.2, 0.2, 0.0]).tolist()
[0.8, 0.2, 0.0]
>>> to_pseudocomponents([0.5, 0.3, 0.2], [0.2, 0.2, 0.0]).round(4).tolist()
[0.5, 0.1667, 0.3333]
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

Blends = pd.DataFrame | np.ndarray | Sequence[float] | Sequence[Sequence[float]]


def to_pseudocomponents(blends: Blends, lower: Mapping[str, float] | Sequence[float], total: float = 1.0) -> Blends:
    r"""Return blends as L-pseudocomponents, :math:`x'_i = (x_i - L_i)/(T - \sum_j L_j)`.

    Parameters
    ----------
    blends : pandas.DataFrame, numpy.ndarray or Sequence
        The blends, a component per column (a single blend may be one sequence).
    lower : Mapping[str, float] or Sequence[float]
        The lower bound of each component: by column name for a DataFrame (a component not named
        has the bound 0), or one per column.
    total : float
        The mixture's total, :math:`T`: 1 for proportions, 100 for percentages.

    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        The pseudocomponents, in the shape and type of ``blends`` (a DataFrame keeps its index and
        columns; a sequence comes back as an array).

    Raises
    ------
    ValueError
        If a bound is negative or not a finite number, the bounds leave no room
        (:math:`\sum_j L_j \ge T`), ``lower`` does not match the components, or a blend has a value
        that is not a finite number.
    TypeError
        If ``lower`` names the components but ``blends`` is not a DataFrame.
    """
    values, bounds, span = _prepare(blends, lower, total)
    return _like(blends, (values - bounds) / span)


def from_pseudocomponents(
    pseudocomponents: Blends, lower: Mapping[str, float] | Sequence[float], total: float = 1.0
) -> Blends:
    r"""Return L-pseudocomponents as blends, :math:`x_i = L_i + (T - \sum_j L_j)\,x'_i`.

    Parameters
    ----------
    pseudocomponents : pandas.DataFrame, numpy.ndarray or Sequence
        The pseudocomponents, a component per column (a single blend may be one sequence).
    lower : Mapping[str, float] or Sequence[float]
        The lower bound of each component, as :func:`to_pseudocomponents` takes them.
    total : float
        The mixture's total, :math:`T`.

    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        The blends, in the shape and type of ``pseudocomponents``.

    Raises
    ------
    ValueError
        As :func:`to_pseudocomponents`.
    """
    values, bounds, span = _prepare(pseudocomponents, lower, total)
    return _like(pseudocomponents, bounds + span * values)


def _prepare(
    blends: Blends, lower: Mapping[str, float] | Sequence[float], total: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return the values as an array, the bound of each column, and the room the bounds leave."""
    values = np.asarray(blends, dtype=float)
    if values.ndim not in (1, 2) or not np.all(np.isfinite(values)):
        msg = "The blends must be finite numbers, a component per column."
        raise ValueError(msg)
    k = values.shape[-1]
    if isinstance(lower, Mapping):
        if not isinstance(blends, pd.DataFrame):
            msg = "Lower bounds by name need the blends as a DataFrame with the components' names as columns."
            raise TypeError(msg)
        unknown = sorted(set(map(str, lower)) - {str(c) for c in blends.columns})
        if unknown:
            msg = f"lower names {unknown}, which are not columns of the blends {list(blends.columns)}."
            raise ValueError(msg)
        named = {str(name): float(bound) for name, bound in lower.items()}
        bounds = np.array([named.get(str(column), 0.0) for column in blends.columns])
    else:
        bounds = np.asarray(lower, dtype=float).ravel()
        if bounds.size != k:
            msg = f"lower must give one bound per component, {k}; got {bounds.size}."
            raise ValueError(msg)
    if not np.all(np.isfinite(bounds)) or np.any(bounds < 0):
        msg = f"Each lower bound must be a finite number, 0 or more; got {bounds.tolist()}."
        raise ValueError(msg)
    span = float(total) - float(bounds.sum())
    if not np.isfinite(span) or span <= 0:
        msg = f"The lower bounds add to {bounds.sum():g}, which leaves no room in a total of {total:g}."
        raise ValueError(msg)
    return values, bounds, span


def _like(template: Blends, values: np.ndarray) -> Blends:
    """Return ``values`` as ``template`` was given: a DataFrame keeps its index and columns."""
    if isinstance(template, pd.DataFrame):
        return pd.DataFrame(values, index=template.index, columns=template.columns)
    return values
