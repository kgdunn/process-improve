# (c) Kevin Dunn, 2010-2026. MIT License.

"""The share of the coded box ``[-1, 1]^k`` that curved limits leave, by quadrature (#810 item 1).

The factors are split into the groups the limits tie together. A group with no curved limit
has its exact share (see :func:`~process_improve.experiments._region_geometry.box_share`).
A group with one has its share as an integral over one of its factors, the hub, when every
limit of the group that names two factors names the hub, and none names three:

- the limits on the hub alone leave it a set of intervals (:func:`_intervals`);
- at each value of the hub, every other factor of the group is limited on its own, so the
  share at that value is the product of each one's allowed length along a line across the
  box (:func:`_lengths`): its straight limits leave an interval, found exactly
  (:func:`_bounds`), and its curved limits are searched for across that interval;
- that product bends where a limit meets an end of such an interval. Those values of the
  hub are found, exactly for the straight limits (:func:`_straight_kinks`) and by the same
  search for the curved ones, and the product is integrated between them by Gauss-Legendre
  quadrature, on panels halved only where it still bends (:func:`_integral`).

A group of one or two factors always has a hub, and so does ``T + 25*P <= 250`` beside
``T*F <= 3600``, with ``T``. The share is then good to about 1e-10, where a fixed sample of
16384 points is within about 0.005. Like the rest of the region's geometry, it uses only
numpy, so it runs unchanged under Pyodide.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable

import numpy as np

from process_improve.experiments._region_geometry import _groups, box_share, round_significant

#: A function of coded points ``(n, k)``; a point meets it where it is at most 0.
Inequality = Callable[[np.ndarray], np.ndarray]
#: A curved limit, with the columns it names.
Curved = tuple[Inequality, np.ndarray]
#: A straight limit ``row @ x <= rhs``.
Row = tuple[np.ndarray, float]
#: Where the points at ``values`` along lines ``lines`` lie in the box: ``place(values, lines)``, coded ``(n, k)``.
Place = Callable[[np.ndarray, np.ndarray], np.ndarray]

#: Points of the grid that first cuts a line. A stretch that a curved limit allows, or removes, narrower than 1/128
#: of the line can fall between two of the grid's points and be missed.
_GRID = 129
#: Parts a bracket of a boundary is cut into at each search, and the searches: 16**13 takes a step of the grid
#: past the spacing of doubles near 1.
_SECTIONS, _SEARCHES = 16, 13
#: Gauss-Legendre points per panel, and the panels an interval of the hub starts with.
_GAUSS_POINTS, _START_PANELS = 8, 8
#: A panel is settled when its integral and the sum of its halves' differ by at most this times the length it is
#: integrated over, or once it has been halved this many times, to about the spacing of doubles.
_SETTLED, _MOST_HALVINGS = 1e-12, 48
#: Open panels past which every panel is settled as it is, so a limit that bends the integrand everywhere costs a
#: few passes, not one panel more each time.
_MOST_PANELS = 512
#: Significant figures the share is given to.
_DIGITS = 10


def columns_named(g: Inequality, points: np.ndarray) -> np.ndarray:
    """Return the columns ``g`` depends on: each one whose values, moved among ``points``, change ``g``'s.

    Found by evaluating rather than by reading the limit, so a name written in any form the
    region accepts counts. A column ``g`` reads only in a corner no point reaches is missed.
    """
    with np.errstate(all="ignore"):
        values = g(points)
        moved = []
        for j in range(points.shape[1]):
            shifted = points.copy()
            shifted[:, j] = np.roll(points[:, j], 1)
            if not np.array_equal(g(shifted), values, equal_nan=True):
                moved.append(j)
    return np.array(moved, dtype=int)


def curved_share(a: np.ndarray, b: np.ndarray, curved: list[Curved]) -> float | None:
    """Return the share of the box where ``a @ x <= b`` and every curved ``g(x) <= 0``, or None when not had here.

    Parameters
    ----------
    a, b : np.ndarray
        The straight limits as unit-length rows ``(r, k)`` in coded units, and their right-hand sides.
    curved : list[Curved]
        Each curved limit with the columns it names (see :func:`columns_named`).

    Returns
    -------
    float or None
        The share, to 10 significant figures. None when a group of factors with a curved limit
        has no hub, a curved limit names no column, or the other groups have no exact share.
    """
    named = a.any(axis=1)
    if np.any(~named & (b < 0)):  # a limit no point meets
        return 0.0
    if any(not columns.size for _, columns in curved):
        return None
    a, b = a[named], b[named]
    pattern = np.vstack([a != 0, *(np.isin(np.arange(a.shape[1]), columns) for _, columns in curved)])
    share, straight = 1.0, np.zeros(len(a), dtype=bool)
    for columns, members in _groups(pattern.astype(float)):
        rows = members[members < len(a)]
        if rows.size == members.size:
            straight[rows] = True
            continue
        part = _hub_share(a[rows], b[rows], [curved[m - len(a)] for m in members[members >= len(a)]], columns)
        if part is None:
            return None
        share *= part
    rest = box_share(a[straight], b[straight]) if straight.any() else 1.0
    return None if rest is None else float(round_significant(np.array([share * rest]), _DIGITS)[0])


def _hub_share(a: np.ndarray, b: np.ndarray, curved: list[Curved], columns: np.ndarray) -> float | None:
    """Return one group's share of its columns' box, as an integral over its hub; None when it has none."""
    row_names = [np.flatnonzero(row) for row in a]
    names = row_names + [named for _, named in curved]
    if any(named.size > 2 for named in names):  # a hub and one other factor at most
        return None
    pairs = [set(named.tolist()) for named in names if named.size == 2]
    hubs = set.intersection(*pairs) if pairs else set(columns.tolist())
    if not hubs:
        return None
    hub, k = min(hubs), a.shape[1]

    def limits_on(column: int, *, alone: bool) -> tuple[list[Row], list[Inequality]]:
        """Return the straight rows and the curved limits that name ``column``: it only, when ``alone``."""
        rows = [
            (row, float(rhs))
            for row, rhs, named in zip(a, b, row_names, strict=True)
            if column in named and (named.size == 1 or not alone)
        ]
        return rows, [g for g, named in curved if column in named and (named.size == 1 or not alone)]

    hub_rows, hub_curved = limits_on(hub, alone=True)
    low, high = _bounds(hub_rows, hub, None, np.zeros(1))
    intervals = _intervals(hub_curved, _placed(k, hub), float(low[0]), float(high[0]))
    others = {int(c): limits_on(int(c), alone=False) for c in columns if c != hub}
    if not others:
        return sum(end - start for start, end in intervals) / 2
    kinks = sorted({x for column, (rows, limits) in others.items() for x in _kinks(rows, limits, k, hub, column)})

    def product(at: np.ndarray) -> np.ndarray:
        share = np.ones(at.size)
        for column, (rows, limits) in others.items():
            share *= _lengths(limits, _placed(k, hub, column, at), *_bounds(rows, column, hub, at)) / 2
        return share

    pieces = [
        (start, end)
        for low_end, high_end in intervals
        for start, end in itertools.pairwise([low_end, *(x for x in kinks if low_end < x < high_end), high_end])
    ]
    return _integral(product, pieces) / 2


def _placed(
    k: int, hub: int, column: int | None = None, at: np.ndarray | None = None, side: tuple[float, float] | None = None
) -> Place:
    """Return where the points of lines lie: along the hub, or along ``column`` at the hub's values ``at``.

    Along the hub, ``column`` (when given) follows the straight bound ``side``, the line
    ``column = side[1] - side[0] * hub``, so a curved limit's crossings there are where it
    meets that bound.
    """

    def place(values: np.ndarray, lines: np.ndarray) -> np.ndarray:
        points = np.zeros((values.size, k))
        if at is None:
            points[:, hub] = values
            if column is not None and side is not None:
                points[:, column] = side[1] - side[0] * values
        else:
            points[:, hub], points[:, column] = at[lines], values
        return points

    return place


def _bounds(rows: list[Row], column: int, hub: int | None, at: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return, at each value of the hub, the interval of ``column`` within ``[-1, 1]`` that straight ``rows`` leave.

    Each row names ``column``, and the hub or nothing else, so it bounds the column on one side.
    An interval that is empty comes back with its ends equal.
    """
    low, high = np.full(at.size, -1.0), np.full(at.size, 1.0)
    for row, rhs in rows:
        bound = (rhs - (row[hub] * at if hub is not None else 0.0)) / row[column]
        if row[column] > 0:
            high = np.minimum(high, bound)
        else:
            low = np.maximum(low, bound)
    return low, np.maximum(high, low)


def _straight_kinks(rows: list[Row], column: int, hub: int) -> list[float]:
    """Return the values of the hub where a straight bound of ``column`` meets another one, or an end of ``[-1, 1]``."""
    lines = [(rhs / row[column], row[hub] / row[column]) for row, rhs in rows] + [(-1.0, 0.0), (1.0, 0.0)]
    return [(c - d) / (s - t) for (c, s), (d, t) in itertools.combinations(lines, 2) if s != t]


def _kinks(rows: list[Row], limits: list[Inequality], k: int, hub: int, column: int) -> list[float]:
    """Return the values of the hub where the allowed length of ``column`` can bend.

    Where its straight bounds meet (see :func:`_straight_kinks`), and where a curved limit
    meets one of them: the ends of the intervals of the hub over which the limits hold along it.
    """
    found = _straight_kinks(rows, column, hub)
    for c, s in [(rhs / row[column], row[hub] / row[column]) for row, rhs in rows] + [(-1.0, 0.0), (1.0, 0.0)]:
        found += [x for ends in _intervals(limits, _placed(k, hub, column, side=(s, c)), -1.0, 1.0) for x in ends]
    return found


def _boundaries(
    limits: list[Inequality], place: Place, low: np.ndarray, high: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return where lines from ``low`` to ``high`` cross the boundary of the curved ``limits``.

    Each line is cut at a grid of points. Where two neighbouring points differ, the step
    between them is searched: at each search, the limits are evaluated at 15 points across
    the bracket, all brackets in one call, and the bracket shrinks to the sixteenth where the
    boundary first appears, until it is found to the last bit. Returns the grid, whether each
    of its points meets the limits, and for each crossing its line, its step of the grid,
    where it is, and whether the limits are met above it.
    """

    def meets(values: np.ndarray, lines: np.ndarray) -> np.ndarray:
        points = place(values, lines)
        ok = np.ones(values.size, dtype=bool)
        with np.errstate(all="ignore"):
            for g in limits:
                ok &= g(points) <= 0
        return ok

    grid = low[:, None] + (high - low)[:, None] * np.linspace(0.0, 1.0, _GRID)
    inside = meets(grid.ravel(), np.repeat(np.arange(low.size), _GRID)).reshape(grid.shape)
    line, step = np.nonzero(inside[:, :-1] != inside[:, 1:])
    start, end, rising = grid[line, step], grid[line, step + 1], inside[line, step + 1]
    across, crossing = np.arange(1, _SECTIONS) / _SECTIONS, np.arange(line.size)
    for _ in range(_SEARCHES if line.size else 0):
        points = start[:, None] + (end - start)[:, None] * across
        above = meets(points.ravel(), np.repeat(line, across.size)).reshape(points.shape) == rising[:, None]
        first = np.where(above.any(axis=1), above.argmax(axis=1), across.size)  # the first point past the boundary
        start, end = (
            np.where(first > 0, points[crossing, np.maximum(first - 1, 0)], start),
            np.where(first < across.size, points[crossing, np.minimum(first, across.size - 1)], end),
        )
    return grid, inside, line, step, (start + end) / 2, rising


def _lengths(limits: list[Inequality], place: Place, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Return the length of each line from ``low`` to ``high`` that meets ``limits``."""
    if not limits:
        return high - low
    grid, inside, line, step, boundary, rising = _boundaries(limits, place, low, high)
    lengths = (np.diff(grid, axis=1) * (inside[:, :-1] & inside[:, 1:])).sum(axis=1)
    np.add.at(lengths, line, np.where(rising, grid[line, step + 1] - boundary, boundary - grid[line, step]))
    return lengths


def _intervals(limits: list[Inequality], place: Place, low: float, high: float) -> list[tuple[float, float]]:
    """Return the intervals of one line, from ``low`` to ``high``, where ``limits`` hold."""
    if high <= low or not limits:
        return [(low, high)] if high > low else []
    _, inside, _, _, boundary, rising = _boundaries(limits, place, np.array([low]), np.array([high]))
    ends = [low, *boundary.tolist(), high]
    opens = [bool(inside[0, 0]), *rising.tolist()]  # whether the limits are met just after each end
    return [(ends[i], ends[i + 1]) for i in range(len(ends) - 1) if opens[i]]


def _integral(f: Callable[[np.ndarray], np.ndarray], pieces: list[tuple[float, float]]) -> float:
    """Return the integral of a vectorised ``f`` over the intervals ``pieces``.

    Gauss-Legendre on panels: a panel settles when its integral and the sum of its two
    halves' agree to :data:`_SETTLED` times the pieces' length, so the smooth stretches
    settle at once, and only the panels where ``f`` still bends are halved. A panel's halves
    become the next pass's panels, with their integrals already known, and every open panel
    of every piece is evaluated in one call of ``f`` per pass.
    """
    nodes, weights = np.polynomial.legendre.leggauss(_GAUSS_POINTS)

    def gauss(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        x = (starts + ends)[..., None] / 2 + (ends - starts)[..., None] / 2 * nodes
        return (f(x.ravel()).reshape(x.shape) @ weights) * (ends - starts) / 2

    if not pieces:
        return 0.0
    edges = np.array([np.linspace(start, end, _START_PANELS + 1) for start, end in pieces])
    starts, ends, total = edges[:, :-1].ravel(), edges[:, 1:].ravel(), 0.0
    tolerance = _SETTLED * sum(end - start for start, end in pieces)
    wholes = gauss(starts, ends)
    halvings = 0
    while starts.size:
        middle = (starts + ends) / 2
        left, right = gauss(np.stack([starts, middle]), np.stack([middle, ends]))
        settled = (
            (np.abs(wholes - left - right) <= tolerance) | (halvings == _MOST_HALVINGS) | (wholes.size > _MOST_PANELS)
        )
        total += float((left + right)[settled].sum())
        kept = ~settled
        starts, ends = np.concatenate([starts[kept], middle[kept]]), np.concatenate([middle[kept], ends[kept]])
        wholes, halvings = np.concatenate([left[kept], right[kept]]), halvings + 1
    return total
