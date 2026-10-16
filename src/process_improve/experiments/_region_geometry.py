# (c) Kevin Dunn, 2010-2026. MIT License.

"""Exact geometry of the coded box ``[-1, 1]^k`` cut by linear inequalities.

The helpers here work in coded units on rows ``a @ x <= b`` whose ``a`` has unit length,
so one tolerance (a distance) serves every row. They are kept free of other
``process_improve.experiments`` imports, and use only numpy and ``scipy.spatial``, so
they run unchanged under Pyodide.

- :func:`box_vertices`: every vertex of the cut box, by the double-description method;
- :func:`clip_square`: the square ``[-1, 1]^2`` clipped by half-planes (Sutherland-Hodgman);
- :func:`convex_polygon`: the counter-clockwise hull of points in the plane, or the one
  or two points a degenerate hull reduces to;
- :func:`points_inside`: fixed points spread through the hull of given vertices;
- :func:`round_significant`: numbers rounded to a number of significant figures, so the
  last-digit noise of floating point does not reach the reader.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.stats import qmc

#: Distance (coded units) within which a point counts as on a boundary.
TOL = 1e-9
#: Most vertices the double-description method keeps before it gives up.
MAX_VERTICES = 2**16
#: Pairs of vertices tested for adjacency at once (bounds the memory of the test).
_PAIR_CHUNK = 512
#: Entries of the (kept x removed) count of shared boundaries computed at once: 32 MB of float64.
_PRODUCT_CELLS = 2**22
#: The square ``[-1, 1]^2``, counter-clockwise from its lower-left corner.
_SQUARE = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])


def round_significant(values: np.ndarray, digits: int = 12) -> np.ndarray:
    """Round every value to ``digits`` significant figures.

    Parameters
    ----------
    values : np.ndarray
        Finite numbers.
    digits : int
        Significant figures to keep.

    Returns
    -------
    np.ndarray
        The rounded values, as floats; zeros stay zero.

    Examples
    --------
    >>> round_significant(np.array([250.00000000000009, 1.65e-14, -0.30000000000000004])).tolist()
    [250.0, 1.65e-14, -0.3]
    """
    v = np.asarray(values, dtype=float)
    with np.errstate(divide="ignore"):
        magnitude = np.floor(np.log10(np.abs(v)))
    exponent = digits - 1 - np.where(np.isfinite(magnitude), magnitude, 0.0)
    power = 10.0 ** np.abs(exponent)
    # Dividing by an exact power of ten rounds once, so a whole number of units in the last kept digit comes
    # back as the double nearest that decimal (multiplying by 10**-n would not: 10**-n is inexact).
    return np.where(exponent >= 0, np.round(v * power) / power, np.round(v / power) * power)


def box_vertices(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vertices of ``{x in [-1, 1]^k : a @ x <= b}``, by the double-description method.

    The box's ``2**k`` corners are cut by one row at a time. A cut keeps the vertices on
    its side and adds the point where it crosses each edge joining a kept vertex to a
    removed one. Two vertices are joined by an edge when no third vertex lies on every
    boundary the two share (the combinatorial test of Fukuda and Prodon 1996), which also
    holds at degenerate vertices, where more than ``k`` boundaries meet, and when a cut
    leaves a region of lower dimension. The work grows with the number of vertices, not
    with the number of ways to choose ``k`` boundaries.

    Parameters
    ----------
    a : np.ndarray
        Rows ``(m, k)`` of unit length (a zero row with ``b = -1`` stands for a row that
        no point meets).
    b : np.ndarray
        Right-hand sides ``(m,)``.

    Returns
    -------
    np.ndarray
        Vertices ``(n, k)`` in coded units, without duplicates, sorted row by row; no rows
        when the region is empty.

    Raises
    ------
    ValueError
        When the box has more than ``MAX_VERTICES`` corners, or a cut would leave more
        than that many vertices.

    References
    ----------
    Fukuda, K. and Prodon, A. (1996) Double description method revisited. In
    *Combinatorics and Computer Science*, LNCS 1120, 91-111.
    """
    m, k = a.shape
    if 2**k > MAX_VERTICES:
        raise ValueError(f"A box in {k} factors has {2**k} corners; vertices are listed for up to {MAX_VERTICES}.")
    vertices = np.array(list(itertools.product((-1.0, 1.0), repeat=k)), dtype=float).reshape(2**k, k)
    # Which boundary each vertex lies on: the k upper faces, the k lower faces, then the m rows.
    active = np.hstack([vertices == 1.0, vertices == -1.0, np.zeros((len(vertices), m), dtype=bool)])
    for row in range(m):
        vertices, active = _cut(vertices, active, a[row], float(b[row]), 2 * k + row)
        if not len(vertices):
            break
        if len(vertices) > MAX_VERTICES:
            raise ValueError(f"The region has more than {MAX_VERTICES} vertices; they are not listed.")
    vertices = np.where(np.abs(np.abs(vertices) - 1.0) <= 1e-12, np.sign(vertices), vertices)
    vertices = np.where(np.abs(vertices) <= 1e-12, 0.0, vertices)
    return np.unique(vertices.round(12), axis=0)


def _cut(
    vertices: np.ndarray, active: np.ndarray, a_row: np.ndarray, b_val: float, column: int
) -> tuple[np.ndarray, np.ndarray]:
    """Cut the polytope with vertices ``vertices`` by ``a_row @ x <= b_val`` (one step of :func:`box_vertices`)."""
    slack = vertices @ a_row - b_val
    outside, inside = slack > TOL, slack < -TOL
    active[~outside & ~inside, column] = True
    if not outside.any():
        return vertices, active
    keep = ~outside
    inner, outer = np.flatnonzero(inside), np.flatnonzero(outside)
    k = vertices.shape[1]
    weights = active.astype(float)
    # The boundaries each (kept, removed) pair has in common, a block of kept vertices at a time: the whole
    # (kept x removed) matrix would take gigabytes from 15 factors, where a cut through the centre leaves
    # tens of thousands of vertices on each side.
    rows_per_block = max(1, _PRODUCT_CELLS // len(outer))
    removed = weights[outer].T
    u_parts, w_parts = [np.zeros(0, dtype=int)], [np.zeros(0, dtype=int)]
    for start in range(0, len(inner), rows_per_block):
        block = inner[start : start + rows_per_block]
        i, j = np.nonzero(weights[block] @ removed >= k - 1)
        u_parts.append(block[i])
        w_parts.append(outer[j])
    u, w = np.concatenate(u_parts), np.concatenate(w_parts)
    common = active[u] & active[w]
    # A vertex on exactly k boundaries (a simple vertex) shares k - 1 independent ones with any vertex it is
    # joined to, and those k - 1 cut out a line holding just the two: the third-vertex test is needed only
    # when both ends of the pair are degenerate.
    on_more = active.sum(axis=1) > k
    check = np.flatnonzero(on_more[u] & on_more[w])
    joined = np.ones(len(u), dtype=bool)
    for start in range(0, len(check), _PAIR_CHUNK):
        rows = check[start : start + _PAIR_CHUNK]
        joined[rows] = _no_third_vertex(weights, common[rows])
    u, w, common = u[joined], w[joined], common[joined]
    t = slack[u] / (slack[u] - slack[w])
    crossings = vertices[u] + t[:, None] * (vertices[w] - vertices[u])
    common[:, column] = True
    return np.vstack([vertices[keep], crossings]), np.vstack([active[keep], common])


def _no_third_vertex(weights: np.ndarray, common: np.ndarray) -> np.ndarray:
    """Return, for each pair's shared boundaries, whether only the pair's own two vertices lie on all of them."""
    need = common.sum(axis=1)
    holders = (weights @ common.T.astype(float)) >= need  # (vertices, pairs)
    return holders.sum(axis=0) == 2


def clip_square(rows: list[tuple[np.ndarray, float]]) -> np.ndarray:
    """Clip the square ``[-1, 1]^2`` by each half-plane ``c @ p <= d`` in turn (Sutherland-Hodgman).

    Parameters
    ----------
    rows : list[tuple[np.ndarray, float]]
        Half-planes, each ``(c, d)`` with ``c`` of unit length.

    Returns
    -------
    np.ndarray
        The corners ``(n, 2)`` of what is left, counter-clockwise, possibly with repeated
        points where the region has shrunk to a segment or a point; no rows when nothing
        is left. Pass it through :func:`convex_polygon` for a clean outline.
    """
    polygon = _SQUARE
    for c, d in rows:
        slack = polygon @ c - d
        nxt = np.roll(np.arange(len(polygon)), -1)
        kept: list[np.ndarray] = []
        for here, there in zip(range(len(polygon)), nxt, strict=True):
            if slack[here] <= TOL:
                kept.append(polygon[here])
            if (slack[here] < -TOL and slack[there] > TOL) or (slack[here] > TOL and slack[there] < -TOL):
                t = slack[here] / (slack[here] - slack[there])
                kept.append(polygon[here] + t * (polygon[there] - polygon[here]))
        polygon = np.array(kept).reshape(-1, 2)
        if not len(polygon):
            break
    return polygon


def convex_polygon(points: np.ndarray) -> np.ndarray:
    """Return the convex hull of points in the plane, counter-clockwise, or the segment or point it reduces to.

    Parameters
    ----------
    points : np.ndarray
        Points ``(n, 2)``, in units where the region spans about 1 (coded units).

    Returns
    -------
    np.ndarray
        ``(m, 2)``: the hull's corners counter-clockwise starting from the one with the
        smallest first coordinate (then second); the two ends of a segment, in the same
        order, when every point lies on one line; the single point when all coincide;
        no rows for no points.
    """
    if not len(points):
        return np.empty((0, 2))
    unique = np.unique(np.asarray(points, dtype=float).round(12), axis=0)
    if len(unique) == 1:
        return unique
    centred = unique - unique.mean(axis=0)
    _, spread, axes = np.linalg.svd(centred, full_matrices=False)
    if spread[1] <= 1e-9 * spread[0]:
        return _segment(unique, axes[0])
    try:
        ring = unique[ConvexHull(unique).vertices]  # counter-clockwise in two dimensions
    except QhullError:  # flatter than the test above could tell
        return _segment(unique, axes[0])
    ring = _without_straight_corners(ring)
    first = np.lexsort((ring[:, 1], ring[:, 0]))[0]
    return np.roll(ring, -first, axis=0)


def _without_straight_corners(ring: np.ndarray) -> np.ndarray:
    """Return a hull's corners without those where it turns by less than ``1e-9`` (a point on an edge).

    Coordinates rounded to 12 decimals, or computed with rounding error, can move a point
    of an edge off it by ``1e-13``, and the hull then lists it as a corner.
    """
    while len(ring) > 3:  # a triangle keeps its corners
        before, after = ring - np.roll(ring, 1, axis=0), np.roll(ring, -1, axis=0) - ring
        turn = before[:, 0] * after[:, 1] - before[:, 1] * after[:, 0]
        straight = np.abs(turn) <= 1e-9 * np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1)
        if not straight.any():
            break
        ring = np.delete(ring, int(np.argmax(straight)), axis=0)
    return ring


def _segment(points: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Return the two extreme points of ``points`` along ``direction``, the lexicographically smaller first."""
    along = points @ direction
    ends = points[[int(np.argmin(along)), int(np.argmax(along))]]
    return ends[np.lexsort((ends[:, 1], ends[:, 0]))]


def points_inside(vertices: np.ndarray, n_points: int) -> np.ndarray:
    """Return ``n_points`` fixed points spread through the convex hull of ``vertices``.

    Each point is uniform on a simplex of ``k + 1`` of the vertices (by Caratheodory's
    theorem every point of the hull lies in one), with the vertices and the weights read
    off the unscrambled Sobol sequence: no random generator and no transcendental
    function, so the points are the same on every platform. They are not uniform on the
    hull; they are for finding where a thin region is, not for averaging over it.

    Parameters
    ----------
    vertices : np.ndarray
        ``(n, k)`` points, at least one.
    n_points : int
        Number of points to return.

    Returns
    -------
    np.ndarray
        ``(n_points, k)`` points inside the hull.
    """
    count, k = vertices.shape
    m = min(k + 1, count)
    power = max(0, int(np.ceil(np.log2(max(n_points, 1)))))
    unit = qmc.Sobol(d=2 * m - 1, scramble=False).random_base2(power)[:n_points] + 0.5 / 2**power
    picks = np.minimum((unit[:, :m] * count).astype(int), count - 1)
    cuts = np.sort(unit[:, m:], axis=1)  # spacings of sorted uniforms are uniform on the simplex
    weights = np.diff(np.hstack([np.zeros((len(unit), 1)), cuts, np.ones((len(unit), 1))]), axis=1)
    return np.einsum("nm,nmk->nk", weights, vertices[picks])
