# (c) Kevin Dunn, 2010-2026. MIT License.
r"""Evolutionary operation (EVOP): the calculation sheet of a :math:`2^2` design with a centre point.

In evolutionary operation a full-scale process runs, cycle after cycle, at five conditions close to
its usual one: the centre (point 1) and the corners of a small square in two factors A and B
(point 2 with A and B both low, 3 both high, 4 A high and B low, 5 A low and B high). The changes
are small enough to keep the product saleable, and repeating the cycle lets their effects emerge
from the noise. After each cycle the calculation sheet updates:

- each point's sum and average over the cycles so far;
- the effects of A and B, their interaction, and the change in mean (the corners against the centre);
- an estimate of the noise, from the range of the differences between each point's previous average
  and its new reading, times :math:`f_{k,n} = \sqrt{(n-1)/n} / d_2(k)` (see :func:`evop_f`);
- the two-standard-deviation error limits of the averages, the effects and the change in mean.

The method and its sheet are Box and Draper's (*Evolutionary Operation*, 1969); Montgomery works an
example in section 11.7.

Examples
--------
Montgomery's Example 11.7, three cycles of yields at points 1 to 5, the noise factor rounded to two
decimals as his Table 11.27 prints it:

>>> from process_improve.monitoring import evop_worksheet
>>> cycles = [[84.5, 84.2, 84.9, 84.5, 84.3], [84.9, 84.6, 85.9, 83.5, 84.0], [85.0, 84.0, 86.6, 84.9, 85.2]]
>>> board = evop_worksheet(cycles, factors=("T", "P"), f_decimals=2)[-1]
>>> {name: round(effect, 2) for name, effect in board["effects"].items()}
{'T': 0.67, 'P': 0.87, 'T:P': 0.63, 'change_in_mean': -0.07}
>>> round(board["average_s"], 2), {name: round(limit, 2) for name, limit in board["limits"].items()}
(0.58, {'averages': 0.67, 'effects': 0.67, 'change_in_mean': 0.6})
"""

from __future__ import annotations

import functools
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import integrate, stats

#: The points of the design in the sheet's order: the centre, then the corners as (A, B) in coded units.
POINTS = ((0, 0), (-1, -1), (1, 1), (1, -1), (-1, 1))

#: Two standard deviations of the change in mean, (y2 + y3 + y4 + y5 - 4 y1) / 5, in units of sigma / sqrt(n):
#: its variance is (4 + 16) sigma^2 / (25 n). Box and Draper's sheets print 1.78.
CHANGE_IN_MEAN_LIMIT = 2 * math.sqrt(20 / 25)


def range_d2(k: int) -> float:
    r"""Return :math:`d_2(k)`, the mean range of ``k`` independent standard normal observations.

    It is :math:`\int_{-\infty}^{\infty} [1 - \Phi(x)^k - (1 - \Phi(x))^k]\,dx`: 1.128 for two
    observations, 2.326 for five. A range of ``k`` observations divided by it estimates their
    standard deviation.

    Parameters
    ----------
    k : int
        The number of observations in the range, 2 or more.

    Returns
    -------
    float
        The mean range, in units of the observations' standard deviation.

    Raises
    ------
    ValueError
        If ``k`` is not a whole number of at least 2.
    """
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 2:  # a range needs two observations
        msg = f"k, the number of observations in the range, must be a whole number of at least 2; got {k!r}."
        raise ValueError(msg)
    return _mean_range(int(k))


@functools.cache
def _mean_range(k: int) -> float:
    """Integrate the mean range of ``k`` standard normal observations, once for each ``k``.

    The cache sits behind :func:`range_d2`'s check: a cache keyed on ``k`` alone would answer 2.0,
    which equals 2, without checking it.
    """
    value, _ = integrate.quad(lambda x: 1 - stats.norm.cdf(x) ** k - stats.norm.sf(x) ** k, -np.inf, np.inf)
    return float(value)


def evop_f(k: int, n: int, decimals: int | None = None) -> float:
    r"""Return :math:`f_{k,n} = \sqrt{(n-1)/n} / d_2(k)`, which turns the range of a cycle's differences into a sigma.

    In cycle :math:`n`, each of the ``k`` differences between a point's previous average and its new
    reading has variance :math:`\sigma^2 (1 + 1/(n-1)) = \sigma^2 n/(n-1)`. Their range, divided by
    :math:`d_2(k)` and multiplied by :math:`\sqrt{(n-1)/n}`, estimates :math:`\sigma`.

    Montgomery's Table 11.27 prints it to two decimals, and matches that rounding in all but two
    entries: :math:`f_{5,9}` and :math:`f_{10,3}`, 0.4054 and 0.2653, print as 0.40 and 0.26. All of
    its entries are reproduced with :math:`d_2` rounded to two decimals and :math:`\sqrt{(n-1)/n}`
    to three.

    Parameters
    ----------
    k : int
        The number of points in the design, whose differences give the range: 5 for a :math:`2^2`
        design with a centre point.
    n : int
        The cycle, 2 or more: the first cycle has no previous average to differ from.
    decimals : int or None
        Round :math:`f_{k,n}` to this many decimals, as a printed table does; None leaves it unrounded.

    Returns
    -------
    float
        :math:`f_{k,n}`.

    Raises
    ------
    ValueError
        If ``n`` is not a whole number of at least 2, or ``k`` is not one of at least 2.
    """
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 2:  # a difference needs a previous cycle
        msg = f"n, the cycle, must be a whole number of at least 2; got {n!r}."
        raise ValueError(msg)
    value = math.sqrt((n - 1) / n) / range_d2(k)
    return value if decimals is None else round(value, decimals)


def evop_worksheet(
    cycles: Sequence[Sequence[float]], factors: tuple[str, str] = ("A", "B"), f_decimals: int | None = None
) -> list[dict[str, Any]]:
    r"""Work the EVOP calculation sheet of a :math:`2^2` design with a centre point, one cycle after another.

    Parameters
    ----------
    cycles : Sequence[Sequence[float]]
        The readings, a row per cycle in the order they were run, each row the five points in the
        order of :data:`POINTS`: the centre, then A and B both low, both high, A high and B low, A
        low and B high.
    factors : tuple[str, str]
        The names of the two factors, A and B, which name the effects.
    f_decimals : int or None
        Round :math:`f_{k,n}` to this many decimals, as a printed table does; None uses it unrounded.

    Returns
    -------
    list[dict[str, Any]]
        A sheet per cycle, the last of them the information board:

        - ``cycle``, :math:`n`; ``readings``, the cycle's five readings;
        - ``sums`` and ``averages`` of each point over the cycles so far;
        - ``effects``: the main effects of A and B, ``(y3 + y4 - y2 - y5) / 2`` and
          ``(y3 + y5 - y2 - y4) / 2``, their interaction ``"A:B"``, ``(y2 + y3 - y4 - y5) / 2``, and
          ``change_in_mean``, ``(y2 + y3 + y4 + y5 - 4 y1) / 5``, all from the averages;
        - from the second cycle: ``differences``, each point's previous average less its new
          reading, their ``range``, ``s`` (the range times :math:`f_{5,n}`), the running ``sum_s``
          of these and its average ``average_s`` over the cycles that gave one, and ``limits``,
          two standard deviations of the ``averages`` and the ``effects``,
          :math:`2 S/\sqrt{n}`, and of the ``change_in_mean``, :math:`1.79\,S/\sqrt{n}`, with
          :math:`S` the average. In the first cycle these are None.

    Raises
    ------
    ValueError
        If there is no cycle, a cycle does not have five readings, a reading is not a finite number,
        or the factors are not two different names (neither of them ``"change_in_mean"``).
    """
    readings = _readings(cycles)
    if len(factors) != 2 or len(set(factors)) != 2 or "change_in_mean" in factors:
        msg = f"factors must be two different names, neither of them 'change_in_mean'; got {factors!r}."
        raise ValueError(msg)
    a, b = factors
    sheets = []
    sum_s = 0.0
    for index, row in enumerate(readings):
        n = index + 1
        sums = readings[:n].sum(axis=0)
        averages = sums / n
        y1, y2, y3, y4, y5 = averages.tolist()
        sheet: dict[str, Any] = {
            "cycle": n,
            "readings": row.tolist(),
            "sums": sums.tolist(),
            "averages": averages.tolist(),
            "effects": {
                a: (y3 + y4 - y2 - y5) / 2,
                b: (y3 + y5 - y2 - y4) / 2,
                f"{a}:{b}": (y2 + y3 - y4 - y5) / 2,
                "change_in_mean": (y2 + y3 + y4 + y5 - 4 * y1) / 5,
            },
            "differences": None,
            "range": None,
            "s": None,
            "sum_s": None,
            "average_s": None,
            "limits": None,
        }
        if n > 1:
            differences = readings[: n - 1].mean(axis=0) - row
            spread = float(differences.max() - differences.min())
            s = spread * evop_f(len(POINTS), n, f_decimals)
            sum_s += s
            average_s = sum_s / (n - 1)
            sheet |= {
                "differences": differences.tolist(),
                "range": spread,
                "s": s,
                "sum_s": sum_s,
                "average_s": average_s,
                "limits": {
                    "averages": 2 * average_s / math.sqrt(n),
                    "effects": 2 * average_s / math.sqrt(n),
                    "change_in_mean": CHANGE_IN_MEAN_LIMIT * average_s / math.sqrt(n),
                },
            }
        sheets.append(sheet)
    return sheets


def _readings(cycles: Sequence[Sequence[float]]) -> np.ndarray:
    """Return the cycles as an array with a row of five finite readings per cycle, or refuse them."""
    shape = f"a row of {len(POINTS)} readings per cycle, one cycle at least"
    try:
        readings = np.asarray(cycles, dtype=float)
    except (TypeError, ValueError) as error:  # a ragged table, or a reading that is not a number
        msg = f"cycles must be {shape}, each reading a number."
        raise ValueError(msg) from error
    if readings.ndim != 2 or readings.shape[0] < 1 or readings.shape[1] != len(POINTS):
        msg = f"cycles must be {shape}; got shape {readings.shape}."
        raise ValueError(msg)
    if not np.all(np.isfinite(readings)):
        msg = "Each reading must be a finite number."
        raise ValueError(msg)
    return readings
