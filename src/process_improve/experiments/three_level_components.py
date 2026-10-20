# (c) Kevin Dunn, 2010-2026. MIT License.
r"""The orthogonal components of the interactions of three-level factors.

In a :math:`3^k` factorial each main effect has two degrees of freedom, and an interaction of
:math:`m` factors has :math:`2^m`. Arithmetic modulo 3 splits the interaction into
:math:`2^{m-1}` *components* of two degrees of freedom each (Montgomery, *Design and Analysis of
Experiments*, section 9.2; Davies, *The Design and Analysis of Industrial Experiments*, chapter 9).
The component :math:`AB^2` groups the runs by :math:`x_A + 2 x_B` (mod 3), each factor's level
written 0, 1 or 2, and its sum of squares is that of the three group totals:

.. math::

    SS = \sum_{u=0}^{2} \frac{T_u^2}{n_u} - \frac{T^2}{N}.

A component and its square group the runs alike, so each is written with its first exponent 1, as
in :mod:`~process_improve.experiments.designs_three_level`. In a full factorial, replicated or not,
the components of an interaction add up to its sum of squares.

Yates named the components of a two-factor interaction I and J, and those of a three-factor
interaction W, X, Y and Z: :math:`I(AB) = AB^2` and :math:`J(AB) = AB`, and

.. math::

    W(ABC) = I[I(AB) \times C] = AB^2C^2, \quad X(ABC) = J[I(AB) \times C] = AB^2C, \quad
    Y(ABC) = I[J(AB) \times C] = ABC^2, \quad Z(ABC) = J[J(AB) \times C] = ABC.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from process_improve.experiments.designs_three_level import MODULUS, _component

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Yates's names of a three-factor interaction's components, by the exponents of its second and
#: third factors: W = I[I], X = J[I], Y = I[J], Z = J[J].
_YATES_THREE: dict[tuple[int, ...], str] = {(2, 2): "W", (2, 1): "X", (1, 2): "Y", (1, 1): "Z"}


def interaction_components(data: pd.DataFrame, response: str, factors: Sequence[str]) -> pd.DataFrame:
    r"""Return each main effect and each component of interaction of three-level factors, with its sum of squares.

    Parameters
    ----------
    data : pandas.DataFrame
        One row per run: the factors' settings and the response.
    response : str
        The response column.
    factors : sequence of str
        The three-level factors, in the order the components are written in. A factor's three
        settings are its levels 0, 1 and 2: the categories in order for a categorical column,
        otherwise its values sorted, lowest first. The components of an interaction depend on
        this order; their total does not.

    Returns
    -------
    pandas.DataFrame
        One row per main effect and per component, main effects first and then the
        interactions of two, three and more factors: ``effect`` (``"AB"``), ``component``
        (``"AB2"``, Montgomery's form, or ``"Temp*Feed^2"`` when a name is longer than one
        letter), ``yates`` (``"I(AB)"`` or ``"W(ABC)"``, and empty for an interaction of more
        than three factors, which Yates did not name), ``df`` (2) and ``sum_of_squares``.

    Raises
    ------
    ValueError
        If the response or a factor is not a column, a factor is named twice or does not have
        exactly three settings, or the response is missing or not finite in a run.

    Examples
    --------
    Montgomery's tool life (Example 5.5): two runs at each of the nine settings of the angle and
    the speed. The two components of their interaction add up to its 61.33.

    >>> tool = pd.DataFrame({
    ...     "A": [15, 15, 15, 15, 15, 15, 20, 20, 20, 20, 20, 20, 25, 25, 25, 25, 25, 25],
    ...     "B": [125, 125, 150, 150, 175, 175] * 3,
    ...     "y": [-2, -1, -3, 0, 2, 3, 0, 2, 1, 3, 4, 6, -1, 0, 5, 6, 0, -1],
    ... })
    >>> table = interaction_components(tool, "y", ["A", "B"])
    >>> table[["component", "yates"]].to_numpy().tolist()
    [['A', ''], ['B', ''], ['AB', 'J(AB)'], ['AB2', 'I(AB)']]
    >>> table["sum_of_squares"].round(3).tolist()
    [24.333, 25.333, 33.333, 28.0]
    """
    names = list(factors)
    levels, y = _levels(data, response, names)
    rows: list[dict[str, object]] = []
    for size in range(1, len(names) + 1):
        for chosen in itertools.combinations(range(len(names)), size):
            effect = _effect(chosen, names)
            rows.extend(
                {
                    "effect": effect,
                    "component": _component(word, names),
                    "yates": _yates(word, chosen, effect),
                    "df": MODULUS - 1,
                    "sum_of_squares": _sum_of_squares(levels @ word % MODULUS, y),
                }
                for word in _words(len(names), chosen)
            )
    return pd.DataFrame(rows, columns=["effect", "component", "yates", "df", "sum_of_squares"])


def _levels(data: pd.DataFrame, response: str, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return each run's levels of the factors as 0, 1 and 2, and the response."""
    missing = [c for c in [response, *names] if c not in data.columns]
    if missing or len(set(names)) != len(names) or not names:
        msg = f"interaction_components needs the response and distinct factors as columns; missing {missing}."
        raise ValueError(msg)
    y = pd.to_numeric(data[response], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(y)):
        msg = f"The response {response!r} must be a finite number in every run."
        raise ValueError(msg)
    columns = []
    for name in names:
        column = data[name]
        settings = list(column.cat.categories) if isinstance(column.dtype, pd.CategoricalDtype) else sorted(set(column))
        if len(settings) != MODULUS or column.isna().any():
            msg = f"Factor {name!r} must have exactly three settings, its levels 0, 1 and 2; it has {settings}."
            raise ValueError(msg)
        columns.append(column.map({s: i for i, s in enumerate(settings)}).to_numpy(dtype=int))
    return np.column_stack(columns), y


def _words(k: int, chosen: tuple[int, ...]) -> list[np.ndarray]:
    """Return the components of the interaction of the chosen factors: first exponent 1, the others 1 or 2."""
    words = []
    for powers in itertools.product((1, 2), repeat=len(chosen) - 1):
        word = np.zeros(k, dtype=int)
        word[list(chosen)] = (1, *powers)
        words.append(word)
    return words


def _effect(chosen: tuple[int, ...], names: list[str]) -> str:
    """Name an effect by its factors: ``AB``, or ``Temp*Feed`` when a name is longer than one letter."""
    return ("" if all(len(n) == 1 for n in names) else "*").join(names[j] for j in chosen)


def _yates(word: np.ndarray, chosen: tuple[int, ...], effect: str) -> str:
    """Return Yates's name of a component of a two- or three-factor interaction, else an empty string."""
    exponents = tuple(int(word[j]) for j in chosen[1:])
    if len(chosen) == 2:  # Yates named the components of interactions of two and three factors
        return f"{'I' if exponents == (2,) else 'J'}({effect})"
    if len(chosen) == 3:
        return f"{_YATES_THREE[exponents]}({effect})"
    return ""


def _sum_of_squares(groups: np.ndarray, y: np.ndarray) -> float:
    """Return the sum of squares of the three group totals: sum of T_u^2 / n_u, less T^2 / N."""
    totals = np.bincount(groups, weights=y, minlength=MODULUS)
    counts = np.bincount(groups, minlength=MODULUS)
    kept = counts > 0
    return float(np.sum(totals[kept] ** 2 / counts[kept]) - y.sum() ** 2 / len(y))
