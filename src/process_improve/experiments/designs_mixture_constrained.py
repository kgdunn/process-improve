# (c) Kevin Dunn, 2010-2026. MIT License.

"""Mixture designs over a constrained simplex: extreme vertices and D-optimal selection.

Component bounds (``Factor(type="mixture", low=0.1, high=0.6)``) and linear constraints
(``"x1 + x2 <= 0.7"``) cut the simplex down to a convex polytope. The designs here are
built from the geometry of that polytope, in three steps:

1. **Extreme vertices.** Inside the plane ``sum(x) = 1`` a vertex is a point where
   ``q - 1`` constraints are active. Every choice of ``q - 1`` constraints is solved as
   one linear system (all at once, batched), and the feasible solutions are the
   vertices (McLean and Anderson 1966, with general linear constraints as in XVERT).
2. **Candidate points.** The vertices, the midpoints of the edges joining them and the
   points a third of the way along them, the centroid of each constraint face, the
   overall centroid, and the axial check blends halfway between each vertex and the
   centroid.
3. **Selection.** Without a run budget the vertices, the edge midpoints (for a
   quadratic model) or edge third-points (for a full cubic one), the 2-face centroids
   (for a special or full cubic one) and the overall centroid are returned, unless
   that is more than three runs per model term; with a budget, or for such a large
   design, a D-optimal subset is chosen for the Scheffé model by the same Fedorov
   exchange used for constrained process designs.

The Scheffé models have no intercept, since ``sum(x) = 1`` makes an intercept
redundant: ``linear`` has one term per component, ``quadratic`` adds ``x_i x_j``,
``special_cubic`` adds ``x_i x_j x_k``, and the full ``cubic`` adds
``x_i x_j (x_i - x_j)`` beside those.
"""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.stats import qmc

from process_improve._random import check_random_state
from process_improve.experiments._count_rules import UsedSets, count_rules, used_sets, without_count_rules
from process_improve.experiments._uniform_sampling import UniformSampler, _linear_row
from process_improve.experiments.designs_constrained import (
    ConstrainedOptions,
    criterion_metadata,
    fedorov_exchange,
    independent_columns,
    make_criterion,
    parse_constraint,
    piece_vertices,
    search_directions,
    selection_counts,
)
from process_improve.experiments.designs_utils import refuse_budget_below_model

if TYPE_CHECKING:
    from process_improve.experiments.factor import Constraint, Factor

logger = logging.getLogger(__name__)

#: Scheffé model names, and the design model_type names that map onto them.
SCHEFFE_MODELS = ("scheffe_linear", "scheffe_quadratic", "scheffe_special_cubic", "scheffe_cubic")
_MODEL_ALIASES = {
    "main_effects": "scheffe_linear",
    "linear": "scheffe_linear",
    "interactions": "scheffe_quadratic",
    "quadratic": "scheffe_quadratic",
    "special_cubic": "scheffe_special_cubic",
    "cubic": "scheffe_cubic",
}
#: Highest number of components multiplied in one term of each Scheffé model.
SCHEFFE_DEGREE = {"scheffe_linear": 1, "scheffe_quadratic": 2, "scheffe_special_cubic": 3, "scheffe_cubic": 3}
#: Uniform blends that estimate the region's moment matrix for I-optimality.
_N_REGION_SAMPLES = 20_000
#: Largest number of constraint subsets solved when enumerating vertices, and how many are solved at once.
MAX_VERTEX_SUBSETS = 500_000
_VERTEX_CHUNK = 50_000
_TOL = 1e-9


def scheffe_model(model_type: str) -> str:
    """Return the Scheffé model name for a ``model_type``, e.g. ``"quadratic"`` -> ``"scheffe_quadratic"``."""
    name = _MODEL_ALIASES.get(model_type, model_type)
    if name not in SCHEFFE_MODELS:
        raise ValueError(f"Unknown mixture model {model_type!r}; choose from {', '.join(SCHEFFE_MODELS)}.")
    return name


def scheffe_matrix(x: np.ndarray, model: str) -> np.ndarray:
    """Expand proportions ``x`` (n, q) into the columns of a Scheffé model."""
    model = scheffe_model(model)
    q = x.shape[1]
    columns = [x]
    if SCHEFFE_DEGREE[model] >= 2:
        columns += [x[:, [i]] * x[:, [j]] for i, j in itertools.combinations(range(q), 2)]
    if model == "scheffe_cubic":
        columns += [x[:, [i]] * x[:, [j]] * (x[:, [i]] - x[:, [j]]) for i, j in itertools.combinations(range(q), 2)]
    if SCHEFFE_DEGREE[model] == 3:
        columns += [x[:, [i]] * x[:, [j]] * x[:, [k]] for i, j, k in itertools.combinations(range(q), 3)]
    return np.hstack(columns)


def scheffe_formula_rhs(names: list[str], model: str) -> str:
    """Return the patsy right-hand side of a Scheffé model, e.g. ``"-1 + (A + B + C) ** 2"``.

    The full cubic adds ``I(A * B * (A - B))`` for each pair of components to the special cubic.
    """
    model = scheffe_model(model)
    degree = SCHEFFE_DEGREE[model]
    joined = " + ".join(names)
    rhs = f"-1 + {joined}" if degree == 1 else f"-1 + ({joined}) ** {min(degree, len(names))}"
    if model == "scheffe_cubic":
        rhs += "".join(f" + {cubic_difference_term(a, b)}" for a, b in itertools.combinations(names, 2))
    return rhs


def cubic_difference_term(a: str, b: str) -> str:
    """Return the full cubic Scheffé model's term in components ``a`` and ``b``: ``I(a * b * (a - b))``."""
    return f"I({a} * {b} * ({a} - {b}))"


# ---------------------------------------------------------------------------
# The feasible region as linear inequalities A x <= b on the plane sum(x) = 1
# ---------------------------------------------------------------------------


def _linear_coefficients(expression: str, names: list[str]) -> list[tuple[np.ndarray, float]]:
    """Return ``(a, c)`` pairs with ``g(x) = a @ x + c <= 0`` for each inequality in ``expression``.

    The coefficients are read off by evaluating ``g`` at the origin and the unit vectors,
    then checked at points spread through the cube (the unscrambled Halton sequence, so
    no random generator is involved); a constraint that is not affine is refused, because
    the vertex enumeration below relies on flat faces.
    """
    q = len(names)
    probe = np.vstack([np.zeros(q), np.eye(q), qmc.Halton(d=q, scramble=False).random(9)[1:]])
    env = {n: probe[:, j] for j, n in enumerate(names)}
    pairs = []
    for g in parse_constraint(expression, set(names)):
        values = np.broadcast_to(np.asarray(g(env), dtype=float), (probe.shape[0],))
        c = float(values[0])
        a = values[1 : q + 1] - c
        if not np.allclose(probe @ a + c, values, atol=1e-9, rtol=1e-9):
            raise ValueError(
                f"Constraint {expression!r} is not linear in the mixture proportions. Mixture regions "
                "accept linear constraints only, e.g. 'x1 + 2*x2 <= 0.8'."
            )
        pairs.append((a, c))
    return pairs


def mixture_inequalities(factors: list[Factor], constraints: list[Constraint] | None) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(A, b)`` so the feasible mixtures are ``A @ x <= b`` with ``sum(x) = 1``.

    The rows are the component lower bounds, the upper bounds, then one row per
    inequality in ``constraints``.

    Raises
    ------
    ValueError
        If the bounds cannot be met by any mixture (``sum(low) > 1`` or ``sum(high) < 1``).
    """
    names = [f.name for f in factors]
    low = np.array([f.low for f in factors], dtype=float)
    high = np.array([f.high for f in factors], dtype=float)
    if low.sum() > 1 + _TOL or high.sum() < 1 - _TOL:
        raise ValueError(
            f"No mixture satisfies the component bounds: the lower bounds sum to {low.sum():.4g} and "
            f"the upper bounds to {high.sum():.4g}, but proportions must sum to 1."
        )
    q = len(factors)
    rows, rhs = [-np.eye(q), np.eye(q)], [-low, high]
    for constraint in constraints or []:
        for a, c in _linear_coefficients(constraint.expression, names):
            rows.append(a[None, :])
            rhs.append(np.array([-c]))
    return _unit_rows(np.vstack(rows), np.concatenate(rhs))


def _unit_rows(a_mat: np.ndarray, b_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scale each inequality ``a @ x <= b`` so that ``|a| = 1``, leaving the region unchanged.

    The tolerances below are then distances, whatever units a constraint was written in:
    ``0.003*x1 >= 0.0003`` and ``x1 >= 0.1`` give the same row.
    """
    norms = np.linalg.norm(a_mat, axis=1)
    norms = np.where(norms > 0, norms, 1.0)
    return a_mat / norms[:, None], b_vec / norms


def _drop_redundant_rows(a_mat: np.ndarray, b_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove inequalities implied by the others (with ``sum(x) = 1``); the region is unchanged.

    Row ``r`` is implied when the largest ``a_r @ x`` over the region cut by the other
    rows is at most ``b_r``, found by one linear programme per row. Rows are removed one
    at a time, so of two identical rows one is kept. The upper bound ``x_i <= 1`` with
    lower bounds of 0 is the common case: it adds nothing, but would multiply the
    systems the vertex enumeration solves.
    """
    from scipy.optimize import linprog  # noqa: PLC0415

    keep = np.ones(len(a_mat), dtype=bool)
    q = a_mat.shape[1]
    for r in range(len(a_mat)):
        keep[r] = False
        if not keep.any():
            keep[r] = True
            continue
        result = linprog(
            -a_mat[r],
            A_ub=a_mat[keep],
            b_ub=b_vec[keep],
            A_eq=np.ones((1, q)),
            b_eq=[1.0],
            bounds=[(None, None)] * q,
            method="highs",
        )
        implied = result.status == 0 and -result.fun <= b_vec[r] + _TOL
        keep[r] = not implied
    return a_mat[keep], b_vec[keep]


class TooManyVertices(ValueError):  # noqa: N818  a ValueError, as before it had a name of its own
    """Raised by :func:`extreme_vertices` for a region with more constraint subsets than it solves."""


def sampled_vertices(a_mat: np.ndarray, b_vec: np.ndarray) -> np.ndarray:
    """Return vertices of ``{x : A x <= b, sum(x) = 1}`` found by linear programming.

    For a region with too many vertices for :func:`extreme_vertices` to list.

    A linear programme moves a blend as far as it goes in one direction, which ends at a
    vertex, so the directions of
    :func:`~process_improve.experiments.designs_constrained.search_directions` give that
    many vertices, less the repeats.

    Returns
    -------
    np.ndarray
        Vertices ``(n, q)``, each once, in the order found; no rows when the region is empty.
    """
    q = a_mat.shape[1]
    found = []
    for direction in search_directions(q):
        result = linprog(
            direction,
            A_ub=a_mat,
            b_ub=b_vec,
            A_eq=np.ones((1, q)),
            b_eq=[1.0],
            bounds=[(None, None)] * q,
            method="highs-ds",
        )
        if result.status != 0:  # no blend meets every row, whatever the direction
            return np.empty((0, q))
        found.append(result.x)
    points = np.array(found)
    points[np.abs(points) < 1e-12] = 0.0  # no -0.0 or 1e-17 in a design sheet
    return _unique_rows(points)


def extreme_vertices(a_mat: np.ndarray, b_vec: np.ndarray) -> np.ndarray:
    """Enumerate the vertices of ``{x : A x <= b, sum(x) = 1}``.

    Each inequality is first scaled to unit length and the ones implied by the others
    are dropped (see :func:`_drop_redundant_rows`), which leaves the region unchanged.
    Every subset of ``q - 1`` remaining rows is then made active, stacked with the row
    ``sum(x) = 1`` into a ``q`` by ``q`` system, and the systems are solved in batches.
    With every row of unit length, a system's determinant is at most 1 in size, so the
    test for a singular system does not depend on how a constraint was scaled.
    Solutions that satisfy every constraint are vertices.

    Raises
    ------
    TooManyVertices
        If more than ``MAX_VERTEX_SUBSETS`` systems would have to be solved. It is a
        ``ValueError``; :func:`sampled_vertices` finds some of the vertices of such a region.
    """
    a_mat, b_vec = _unit_rows(np.asarray(a_mat, dtype=float), np.asarray(b_vec, dtype=float))
    a_red, b_red = _drop_redundant_rows(a_mat, b_vec)
    m, q = a_red.shape
    n_subsets = math.comb(m, q - 1)
    if n_subsets > MAX_VERTEX_SUBSETS:
        raise TooManyVertices(
            f"Enumerating the vertices needs {n_subsets} linear solves ({m} non-redundant constraints, {q} "
            f"components), more than the limit of {MAX_VERTEX_SUBSETS}. Remove constraints or components."
        )
    found = []
    combos = itertools.combinations(range(m), q - 1)
    while chunk := list(itertools.islice(combos, _VERTEX_CHUNK)):
        # The row count is given, not -1: with one component each subset is empty, and numpy
        # cannot infer a row count from rows of length 0 (#999).
        subsets = np.array(chunk, dtype=int).reshape(len(chunk), q - 1)
        systems = np.concatenate([a_red[subsets], np.full((len(subsets), 1, q), 1.0 / np.sqrt(q))], axis=1)
        rhs = np.concatenate([b_red[subsets], np.full((len(subsets), 1), 1.0 / np.sqrt(q))], axis=1)
        solvable = np.abs(np.linalg.det(systems)) > 1e-12  # every row has unit length, so this is a relative test
        points = np.linalg.solve(systems[solvable], rhs[solvable][..., None])[..., 0]
        found.append(points[np.all(points @ a_mat.T <= b_vec + 1e-9, axis=1)])
    points = np.vstack(found) if found else np.empty((0, q))
    points[np.abs(points) < 1e-12] = 0.0  # no -0.0 or 1e-17 in a design sheet
    return _unique_rows(points)


def _unique_rows(points: np.ndarray) -> np.ndarray:
    """Drop rows equal to an earlier row after rounding, keeping first-seen order."""
    if len(points) == 0:
        return points
    _, first = np.unique(points.round(10), axis=0, return_index=True)
    return points[np.sort(first)]


def _face_rank(a_mat: np.ndarray, shared: np.ndarray) -> int:
    """Rank of the constraints in ``shared`` together with ``sum(x) = 1``: ``q`` minus the face's dimension."""
    return int(np.linalg.matrix_rank(np.vstack([a_mat[shared], np.ones(a_mat.shape[1])])))


def _plane_centroids(vertices: np.ndarray, active: np.ndarray, a_mat: np.ndarray, adjacent: np.ndarray) -> list:
    """Centroids of the region's 2-dimensional faces: the constrained analogue of ternary blends.

    A vertex and two of its neighbours span a 2-face when the constraints active at
    all three leave a plane (rank ``q - 2``). Its centroid averages every vertex on
    which those constraints are active, so each face is found once whichever corner
    it is reached from. Each set of shared constraints is examined once, and one with
    fewer than ``q - 3`` constraints cannot leave a plane, so needs no rank computed.
    """
    q, checked, centroids = a_mat.shape[1], set(), []
    for i in range(len(vertices)):
        for j, k in itertools.combinations(np.flatnonzero(adjacent[i]), 2):
            shared = active[i] & active[j] & active[k]
            key = shared.tobytes()
            if key in checked:  # many triples share one set of constraints: each set is looked at once
                continue
            checked.add(key)
            if shared.sum() < q - 3 or _face_rank(a_mat, shared) != q - 2:
                continue
            centroids.append(vertices[active[:, shared].all(axis=1)].mean(axis=0))
    return centroids


def _adjacency(active: np.ndarray, a_mat: np.ndarray) -> np.ndarray:
    """Which vertices share an edge: the constraints active at both leave a line (rank ``q - 1`` with the sum).

    A pair needs at least ``q - 2`` shared active constraints for that, so only those
    pairs, counted in one matrix product, have their rank computed.
    """
    q = a_mat.shape[1]
    counts = active.astype(int)
    shared_counts = counts @ counts.T
    adjacent = np.zeros(shared_counts.shape, dtype=bool)
    for i, j in zip(*np.nonzero(np.triu(shared_counts >= q - 2, k=1)), strict=True):
        adjacent[i, j] = adjacent[j, i] = _face_rank(a_mat, active[i] & active[j]) == q - 1
    return adjacent


def _uniform_blends(
    a_mat: np.ndarray, b_vec: np.ndarray, low: np.ndarray, vertices: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """20,000 blends drawn uniformly from the constrained simplex ``a_mat @ x <= b_vec``.

    Proposals come from the smallest simplex holding the lower bounds (the
    L-pseudocomponent simplex), where a flat Dirichlet draw is uniform; a region too
    thin for rejection is sampled by hit-and-run from its extreme vertices. Built from
    :class:`~process_improve.experiments._uniform_sampling.UniformSampler` directly,
    not through :class:`~process_improve.experiments.region.DesignRegion`, which
    imports this module.
    """
    q = len(low)
    sampler = UniformSampler(
        [_linear_row(a, float(b)) for a, b in zip(a_mat, b_vec, strict=True)],
        (low, np.ones(q)),
        lambda m: low + (1.0 - low.sum()) * rng.dirichlet(np.ones(q), size=m),
        seeds=lambda: vertices,
        on_simplex=True,
    )
    return sampler.draw(_N_REGION_SAMPLES, rng)


def mixture_candidates(a_mat: np.ndarray, b_vec: np.ndarray) -> dict[str, np.ndarray]:
    """Return candidate blends by kind.

    The kinds are vertices, edge midpoints and the points a third and two thirds of the
    way along each edge (``edge_third``, the points the ``x_i x_j (x_i - x_j)`` terms of
    a full cubic model need), centroids of the 2-dimensional faces (``plane_centroid``,
    the points the ``x_i x_j x_k`` terms of a special cubic model need), centroids of the
    facets (``face_centroid``), the overall centroid, and axial blends halfway from each
    vertex to it. Two vertices share an edge when the
    constraints active at both, together with ``sum(x) = 1``, have rank ``q - 1``: the
    set of points satisfying them is a line.

    A region with too many vertices to list (see :func:`extreme_vertices`) gives some of
    them, found by linear programming, and points between those in place of its edges and
    faces (see :func:`_candidates_from_a_sample`); it has no ``face_centroid`` blends.
    """
    a_mat, b_vec = _unit_rows(np.asarray(a_mat, dtype=float), np.asarray(b_vec, dtype=float))
    try:
        vertices = extreme_vertices(a_mat, b_vec)
    except TooManyVertices:
        return _candidates_from_a_sample(sampled_vertices(a_mat, b_vec))
    if len(vertices) == 0:
        raise ValueError("No mixture satisfies all the constraints; check them for conflicts.")
    q = a_mat.shape[1]
    active = np.abs(vertices @ a_mat.T - b_vec) <= 1e-9
    adjacent = _adjacency(active, a_mat)
    pairs = list(zip(*np.nonzero(np.triu(adjacent)), strict=True))
    edges = [(vertices[i] + vertices[j]) / 2 for i, j in pairs]
    thirds = [(w * vertices[i] + (3 - w) * vertices[j]) / 3 for i, j in pairs for w in (1, 2)]
    faces = [vertices[active[:, r]].mean(axis=0) for r in range(a_mat.shape[0]) if active[:, r].sum() > 2]
    centroid = vertices.mean(axis=0, keepdims=True)
    return {
        "vertex": vertices,
        "edge_midpoint": _unique_rows(np.array(edges).reshape(-1, q)),
        "edge_third": _unique_rows(np.array(thirds).reshape(-1, q)),
        "plane_centroid": _unique_rows(np.array(_plane_centroids(vertices, active, a_mat, adjacent)).reshape(-1, q)),
        "face_centroid": _unique_rows(np.array(faces).reshape(-1, q)),
        "centroid": centroid,
        "axial_blend": (vertices + centroid) / 2,
    }


class _SampledCandidates(dict):
    """Candidate blends by kind, from a sample of the region's vertices and not from all of them."""


def _candidates_from_a_sample(vertices: np.ndarray) -> dict[str, np.ndarray]:
    """Return candidate blends by kind from some of a region's vertices, when not all of them can be listed.

    Which vertices share an edge or a face is not known, so the points between the
    vertices stand in for them: each vertex with the next one found gives a midpoint and
    two third-points, and each with the next two gives the mean of the three. A convex
    region holds all of them, and they give a model's product terms blends of two and of
    three vertices to be estimated from.
    """
    if len(vertices) == 0:
        raise ValueError("No mixture satisfies all the constraints; check them for conflicts.")
    q = vertices.shape[1]
    first, second = vertices[:-1], vertices[1:]
    centroid = vertices.mean(axis=0, keepdims=True)
    return _SampledCandidates(
        {
            "vertex": vertices,
            "edge_midpoint": _unique_rows((first + second) / 2.0),
            "edge_third": _unique_rows(np.vstack([(2.0 * first + second) / 3.0, (first + 2.0 * second) / 3.0])),
            "plane_centroid": _unique_rows((vertices[:-2] + vertices[1:-1] + vertices[2:]) / 3.0),
            "face_centroid": np.empty((0, q)),
            "centroid": centroid,
            "axial_blend": (vertices + centroid) / 2.0,
        }
    )


def mixture_sets(factors: list[Factor], constraints: list[Constraint] | None) -> UsedSets | None:
    """Return which components a blend may use together, or None when no constraint limits how many are used."""
    names = [f.name for f in factors]
    return used_sets(names, factors, count_rules(constraints, factors), dict.fromkeys(names, False))


def candidate_blends(factors: list[Factor], constraints: list[Constraint] | None) -> np.ndarray:
    """Return every candidate blend of a mixture region that a run may use, each once.

    They are the blends of :func:`mixture_candidates` for the region without its limits
    on how many components are used; with such limits, the vertices of each of their
    pieces and the points between them
    (:func:`~process_improve.experiments.designs_constrained.piece_vertices`); less every
    blend with a component between 0 and its ``min_when_used``, or with a number of
    components in use that a limit does not allow.

    Raises
    ------
    ValueError
        If no blend meets the linear constraints.
    """
    a_mat, b_vec = mixture_inequalities(factors, without_count_rules(constraints))
    return _usable_blends(factors, mixture_candidates(a_mat, b_vec), (a_mat, b_vec), mixture_sets(factors, constraints))


def _usable_blends(
    factors: list[Factor], candidates: dict[str, np.ndarray], limits: tuple, sets: UsedSets | None
) -> np.ndarray:
    """See :func:`candidate_blends`: the blends of ``candidates``, the pieces' blends, less those no run may use."""
    kinds = list(candidates.values())
    if sets is not None:
        bounds = [(float(f.low), float(f.high)) for f in factors]  # type: ignore[arg-type]
        kinds.append(piece_vertices(sets, limits[0], limits[1], bounds, sum_to_one=True))
    pool = _unique_rows(np.vstack(kinds))
    usable = ~_traces(factors, pool)
    if sets is not None:
        usable &= sets.allows(pool)
    return pool[usable]


def _traces(factors: list[Factor], blends: np.ndarray) -> np.ndarray:
    """Return, per blend, whether a component lies between 0 and its ``min_when_used``."""
    out = np.zeros(len(blends), dtype=bool)
    for j, f in enumerate(factors):
        if f.min_when_used is not None:
            out |= (blends[:, j] > _TOL) & (blends[:, j] < f.min_when_used - _TOL)
    return out


#: Inestimable columns a warning names before it says there are more.
_LOST_COLUMNS_NAMED = 6


def _column_labels(names: list[str], model: str) -> list[str]:
    """Return a label for each column of :func:`scheffe_matrix`, in its order."""
    labels = list(names)
    if SCHEFFE_DEGREE[model] >= 2:
        labels += [f"{a}:{b}" for a, b in itertools.combinations(names, 2)]
    if model == "scheffe_cubic":
        labels += [cubic_difference_term(a, b) for a, b in itertools.combinations(names, 2)]
    if SCHEFFE_DEGREE[model] == 3:
        labels += [f"{a}:{b}:{c}" for a, b, c in itertools.combinations(names, 3)]
    return labels


def _estimable_columns(factors: list[Factor], pool: np.ndarray, model: str) -> tuple[np.ndarray, list[str]]:
    """Return the columns of the Scheffé model the candidate blends can estimate, and the labels of the others.

    Limits can hold a group of components to a fixed total: five groups that may each
    supply at most 20% of a blend supply exactly 20% each. The components of such a group
    then sum to a constant in every blend, as all the components do, and one column of
    the group cannot be told from the others. The columns are kept in model order while
    each adds rank, so every set kept this way spans the same model over the region, and
    the design does not depend on which column of a group is left out.
    """
    kept = independent_columns(scheffe_matrix(pool, model))
    labels = _column_labels([f.name for f in factors], model)
    lost = [labels[j] for j in sorted(set(range(len(labels))) - set(kept.tolist()))]
    if lost:
        logger.warning(
            "The region leaves %d of the %d columns of the %s model inestimable (%s%s): its limits fix the total of "
            "some components. The design is built for the columns that can be estimated, and "
            "metadata['inestimable_columns'] lists the others.",
            len(lost),
            len(labels),
            model,
            ", ".join(lost[:_LOST_COLUMNS_NAMED]),
            ", ..." if len(lost) > _LOST_COLUMNS_NAMED else "",
        )
    return kept, lost


@dataclass
class _Blends:
    """The candidate blends a design is chosen from, and the columns of its model that they can estimate."""

    pool: np.ndarray
    #: The index label of each blend, when the caller supplied the candidates.
    labels: list | None
    #: The model columns kept, and the labels of the ones the region leaves inestimable.
    columns: np.ndarray
    lost: list[str]
    n_parameters: int
    #: Whether the region's vertices were sampled, not listed.
    sampled: bool

    def metadata(self, rows: np.ndarray | None, supplied: pd.DataFrame | None) -> dict:
        """Return what the design's metadata says of the candidates: sampled, columns lost, the caller's picks."""
        meta: dict = {}
        if self.sampled:
            meta["vertices_sampled"] = True
        if self.lost:
            meta["inestimable_columns"] = self.lost
        if self.labels is not None and supplied is not None and rows is not None:
            meta["candidate_source"] = "user"
            meta["n_candidates_supplied"] = len(supplied)
            meta["selected_candidates"] = selection_counts(self.labels, rows)
        return meta


def _blends(
    factors: list[Factor],
    candidates: dict[str, np.ndarray],
    opts: ConstrainedOptions,
    limits: tuple,
    sets: UsedSets | None,
) -> _Blends:
    """Return the blends a design may use: the caller's or the region's, without those no run may use.

    A blend with a component between 0 and its ``min_when_used`` is left out, and so is
    one that uses more components, or fewer, than a limit on their number allows. The
    region's own candidates then say which columns of the model can be estimated (see
    :func:`_estimable_columns`); the caller's candidates must support the whole model.

    Raises
    ------
    ValueError
        If no candidate blend is left.
    """
    model = scheffe_model(opts.model_type)
    n_parameters = scheffe_matrix(np.ones((1, len(factors))), model).shape[1]
    labels: list | None = None
    if opts.candidates is not None:
        pool, labels = _user_blends(factors, opts.candidates, *limits)
        usable = ~_traces(factors, pool) & (True if sets is None else sets.allows(pool))
        pool, labels = pool[usable], [label for label, keep in zip(labels, usable, strict=True) if keep]
    else:
        pool = _usable_blends(factors, candidates, limits, sets)
    if len(pool) == 0:
        raise ValueError(
            "Every candidate blend has a component between 0 and its min_when_used, or uses a number of components "
            "that a count_nonzero limit does not allow."
        )
    columns = np.arange(n_parameters)
    lost: list[str] = []
    if labels is None and np.linalg.matrix_rank(scheffe_matrix(pool, model)) < n_parameters:
        columns, lost = _estimable_columns(factors, pool, model)
    return _Blends(pool, labels, columns, lost, n_parameters, isinstance(candidates, _SampledCandidates))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _user_blends(
    factors: list[Factor], candidates: pd.DataFrame, a_mat: np.ndarray, b_vec: np.ndarray
) -> tuple[np.ndarray, list]:
    """Return the supplied blends that sum to 1 and satisfy every constraint, with their index labels."""
    names = [f.name for f in factors]
    missing = [n for n in names if n not in candidates.columns]
    if missing:
        raise ValueError(f"candidates is missing columns for mixture components: {missing}.")
    blends = candidates[names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if np.isnan(blends).any():
        raise ValueError("candidates has missing or non-numeric proportions.")
    if not np.allclose(blends.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Every candidate blend must sum to 1 (proportions, not amounts).")
    keep = np.all(blends @ a_mat.T <= b_vec + 1e-9, axis=1)
    if not keep.any():
        raise ValueError("No candidate blend satisfies the component bounds and constraints.")
    return blends[keep], list(candidates.index[keep])


#: Without a budget, a design of more than this many runs per model term is replaced by a
#: D-optimal one with ``_EXTRA_RUNS`` runs beyond the number of terms.
_CLASSICAL_RUNS_PER_TERM = 3
_EXTRA_RUNS = 5
#: Point kinds in the design used without a budget, by Scheffé model.
_EV_DESIGN = {
    "scheffe_linear": ("vertex", "centroid"),
    "scheffe_quadratic": ("vertex", "edge_midpoint", "centroid"),
    "scheffe_special_cubic": ("vertex", "edge_midpoint", "plane_centroid", "centroid"),
    "scheffe_cubic": ("vertex", "edge_third", "plane_centroid", "centroid"),
}


def _budget_for_a_large_region(
    budget: int | None, candidates: dict, model: str, n_parameters: int
) -> tuple[int | None, int | None]:
    """Return ``(budget, n_runs)``: a missing budget stays missing while the design used without one is small.

    ``n_runs`` is the size of that design (vertices, edge midpoints, 2-face centroids
    and the centroid, as the model needs), or ``None`` when a budget was given. Above
    three runs per model term, as with many bounded components, the budget becomes
    ``n_parameters + _EXTRA_RUNS`` for a D-optimal design.
    """
    if budget is not None:
        return budget, None
    n_runs = len(_unique_rows(np.vstack([candidates[kind] for kind in _EV_DESIGN[model]])))
    if n_runs > _CLASSICAL_RUNS_PER_TERM * n_parameters:
        return n_parameters + _EXTRA_RUNS, n_runs
    return None, n_runs


def constrained_mixture_design(
    factors: list[Factor],
    budget: int | None,
    constraints: list[Constraint] | None = None,
    options: ConstrainedOptions | None = None,
    random_state: int | np.random.Generator | None = None,
) -> tuple[np.ndarray, dict]:
    """Generate a mixture design over component bounds and linear constraints.

    Parameters
    ----------
    factors : list[Factor]
        Mixture factors; ``low`` and ``high`` are the component bounds as proportions.
    budget : int or None
        Number of runs. ``None`` returns, for the model, the vertices and the overall
        centroid, plus the edge midpoints for a quadratic model, plus the centroids of
        the 2-dimensional faces for a special cubic one, and the edge third-points and
        2-face centroids for a full cubic one. (This is not McLean and
        Anderson's design, which adds the facet centroids instead.) When that is more
        than three runs per model term, as with many bounded components, a D-optimal
        design of ``n_terms + 5`` runs is returned instead, and
        ``metadata["n_runs_vertices_edges_centroid"]`` records the size avoided. With
        ``options.candidates``, ``None`` means a D-optimal design of ``n_terms + 3``
        runs. With a budget, a D-optimal subset of all candidate blends is chosen;
        blends may be replicated. A budget below the number of model terms raises a
        ``ValueError``.
    constraints : list[Constraint] or None
        Linear inequalities in the proportions, e.g. ``"x1 + x2 <= 0.7"``.
    options : ConstrainedOptions or None
        ``model_type``: ``"scheffe_linear"``, ``"scheffe_quadratic"``,
        ``"scheffe_special_cubic"`` or ``"scheffe_cubic"``, or the process-design names ``"main_effects"``,
        ``"interactions"`` and ``"quadratic"``, which map to linear, quadratic and
        quadratic. ``criterion``: ``"d_optimal"`` (default), ``"i_optimal"`` (average
        prediction variance over the constrained simplex), ``"a_optimal"`` or
        ``"e_optimal"``.
    random_state : int, numpy.random.Generator or None
        Seed for the exchange's random starts.

    Returns
    -------
    tuple[np.ndarray, dict]
        Proportions ``(n_runs, q)`` whose rows sum to 1, and metadata.
        ``n_vertices`` counts the region's vertices and ``n_candidates`` the distinct
        blends the runs were chosen from (the feasible supplied ones, with
        ``options.candidates``).

    Raises
    ------
    ValueError
        If the region is empty, a constraint is not linear, or the region cannot
        support the model (for example a region that is a single blend).
    """
    opts = options if options is not None else ConstrainedOptions(model_type="scheffe_quadratic")
    model = scheffe_model(opts.model_type)
    rng = check_random_state(random_state)
    a_mat, b_vec = mixture_inequalities(factors, without_count_rules(constraints))
    sets = mixture_sets(factors, constraints)
    candidates = mixture_candidates(a_mat, b_vec)
    blends = _blends(factors, candidates, opts, (a_mat, b_vec), sets)
    pool, columns = blends.pool, blends.columns
    left_out = sets is not None or any(f.min_when_used is not None for f in factors)

    def model_rows(points: np.ndarray) -> np.ndarray:
        return scheffe_matrix(points, model)[:, columns]

    def region_rows() -> np.ndarray:
        if blends.labels is not None or left_out:
            # The supplied blends stand for the region. shortcut: so do the candidates when a component can be
            # left out, whose region is made of pieces that no sampler here draws from, upgrade with one.
            return model_rows(pool)
        low = np.array([f.low or 0.0 for f in factors], dtype=float)
        return model_rows(_uniform_blends(a_mat, b_vec, low, candidates["vertex"], rng))

    if budget is None and blends.labels is not None:
        budget = blends.n_parameters + 3
    requested_budget = budget
    budget, n_classical = _budget_for_a_large_region(budget, candidates, model, len(columns))
    if budget is None and (blends.sampled or left_out or blends.lost):
        # The design used without a budget is every vertex, which a sample of them is not, and its centroid
        # blends every component, traces included; nor is it built for a model with columns left out.
        budget = len(columns) + _EXTRA_RUNS
    if budget is None:
        design = _unique_rows(np.vstack([candidates[kind] for kind in _EV_DESIGN[model]]))
        method, chosen = "extreme_vertices", None
    else:
        refuse_budget_below_model(budget, len(columns), model)
        f_pool = model_rows(pool)
        criterion = make_criterion(opts.criterion, len(columns), region_rows, f_pool)
        rows, logdet = fedorov_exchange(f_pool, budget, np.empty((0, len(columns))), rng, criterion)
        design = pool[rows]
        method = f"{opts.criterion}_{'user_candidates' if blends.labels is not None else 'extreme_vertices'}"
        chosen = (rows, {"optimality_criterion": opts.criterion, **criterion_metadata(criterion, logdet)})

    if np.linalg.matrix_rank(model_rows(design)) < len(columns):
        raise ValueError(
            f"The constrained mixture region cannot support a {model} model ({blends.n_parameters} terms): "
            "it has too few distinct blends. Use a simpler model or loosen the constraints."
        )
    meta = {
        "method": method,
        "model_type": model,
        "n_vertices": len(candidates["vertex"]),
        "n_candidates": len(pool),
        "constraints": [c.expression for c in constraints or []],
        "constraints_enforced": True,
    }
    if requested_budget != budget:
        # A budget raised to the model's size, or one set because the default design was too large.
        key = "budget_requested" if requested_budget is not None else "n_runs_vertices_edges_centroid"
        meta[key] = requested_budget if requested_budget is not None else n_classical
    meta.update(blends.metadata(None if chosen is None else chosen[0], opts.candidates))
    if chosen is not None:
        meta.update(chosen[1])
    return design, meta
