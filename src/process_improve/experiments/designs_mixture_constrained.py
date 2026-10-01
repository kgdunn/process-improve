# (c) Kevin Dunn, 2010-2026. MIT License.

"""Mixture designs over a constrained simplex: extreme vertices and D-optimal selection.

Component bounds (``Factor(type="mixture", low=0.1, high=0.6)``) and linear constraints
(``"x1 + x2 <= 0.7"``) cut the simplex down to a convex polytope. The designs here are
built from the geometry of that polytope, in three steps:

1. **Extreme vertices.** Inside the plane ``sum(x) = 1`` a vertex is a point where
   ``q - 1`` constraints are active. Every choice of ``q - 1`` constraints is solved as
   one linear system (all at once, batched), and the feasible solutions are the
   vertices (McLean and Anderson 1966, with general linear constraints as in XVERT).
2. **Candidate points.** The vertices, the midpoints of the edges joining them, the
   centroid of each constraint face, the overall centroid, and the axial check blends
   halfway between each vertex and the centroid.
3. **Selection.** Without a run budget the classical extreme-vertices design is
   returned; with one, a D-optimal subset is chosen for the Scheffé model by the same
   Fedorov exchange used for constrained process designs.

The Scheffé models have no intercept, since ``sum(x) = 1`` makes an intercept
redundant: ``linear`` has one term per component, ``quadratic`` adds ``x_i x_j`` and
``special_cubic`` adds ``x_i x_j x_k``.
"""

from __future__ import annotations

import itertools
import logging
import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from process_improve._random import check_random_state
from process_improve.experiments.designs_constrained import (
    ConstrainedOptions,
    criterion_metadata,
    fedorov_exchange,
    make_criterion,
    parse_constraint,
    selection_counts,
)

if TYPE_CHECKING:
    from process_improve.experiments.factor import Constraint, Factor

logger = logging.getLogger(__name__)

#: Scheffé model names, and the design model_type names that map onto them.
SCHEFFE_MODELS = ("scheffe_linear", "scheffe_quadratic", "scheffe_special_cubic")
_MODEL_ALIASES = {
    "main_effects": "scheffe_linear",
    "linear": "scheffe_linear",
    "interactions": "scheffe_quadratic",
    "quadratic": "scheffe_quadratic",
    "special_cubic": "scheffe_special_cubic",
}
#: Largest number of constraint subsets solved when enumerating vertices.
MAX_VERTEX_SUBSETS = 500_000
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
    if model in {"scheffe_quadratic", "scheffe_special_cubic"}:
        columns += [x[:, [i]] * x[:, [j]] for i, j in itertools.combinations(range(q), 2)]
    if model == "scheffe_special_cubic":
        columns += [x[:, [i]] * x[:, [j]] * x[:, [k]] for i, j, k in itertools.combinations(range(q), 3)]
    return np.hstack(columns)


def scheffe_formula_rhs(names: list[str], model: str) -> str:
    """Return the patsy right-hand side of a Scheffé model, e.g. ``"-1 + (A + B + C) ** 2"``."""
    degree = {"scheffe_linear": 1, "scheffe_quadratic": 2, "scheffe_special_cubic": 3}[scheffe_model(model)]
    joined = " + ".join(names)
    return f"-1 + {joined}" if degree == 1 else f"-1 + ({joined}) ** {min(degree, len(names))}"


# ---------------------------------------------------------------------------
# The feasible region as linear inequalities A x <= b on the plane sum(x) = 1
# ---------------------------------------------------------------------------


def _linear_coefficients(expression: str, names: list[str]) -> list[tuple[np.ndarray, float]]:
    """Return ``(a, c)`` pairs with ``g(x) = a @ x + c <= 0`` for each inequality in ``expression``.

    The coefficients are read off by evaluating ``g`` at the origin and the unit vectors,
    then checked at random points; a constraint that is not affine is refused, because
    the vertex enumeration below relies on flat faces.
    """
    q = len(names)
    probe = np.vstack([np.zeros(q), np.eye(q), np.random.default_rng(0).uniform(0, 1, size=(8, q))])
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
    return np.vstack(rows), np.concatenate(rhs)


def extreme_vertices(a_mat: np.ndarray, b_vec: np.ndarray) -> np.ndarray:
    """Enumerate the vertices of ``{x : A x <= b, sum(x) = 1}``.

    Every subset of ``q - 1`` constraint rows is made active, stacked with the row
    ``sum(x) = 1`` into a ``q`` by ``q`` system, and all systems are solved in one
    batched call. Solutions that satisfy every constraint are vertices.
    """
    m, q = a_mat.shape
    n_subsets = math.comb(m, q - 1)
    if n_subsets > MAX_VERTEX_SUBSETS:
        raise ValueError(
            f"Enumerating the vertices needs {n_subsets} linear solves ({m} constraints, {q} components), "
            f"more than the limit of {MAX_VERTEX_SUBSETS}. Remove redundant constraints or components."
        )
    subsets = np.array(list(itertools.combinations(range(m), q - 1)), dtype=int).reshape(-1, q - 1)
    systems = np.concatenate([a_mat[subsets], np.ones((len(subsets), 1, q))], axis=1)
    rhs = np.concatenate([b_vec[subsets], np.ones((len(subsets), 1))], axis=1)
    solvable = np.abs(np.linalg.det(systems)) > 1e-12
    points = np.linalg.solve(systems[solvable], rhs[solvable][..., None])[..., 0]
    points[np.abs(points) < 1e-12] = 0.0  # no -0.0 or 1e-17 in a design sheet
    feasible = np.all(points @ a_mat.T <= b_vec + 1e-9, axis=1)
    return _unique_rows(points[feasible])


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
    it is reached from.
    """
    q, seen, centroids = a_mat.shape[1], set(), []
    for i in range(len(vertices)):
        for j, k in itertools.combinations(np.flatnonzero(adjacent[i]), 2):
            shared = active[i] & active[j] & active[k]
            if _face_rank(a_mat, shared) != q - 2:
                continue
            members = tuple(np.flatnonzero(active[:, shared].all(axis=1)))
            if members not in seen:
                seen.add(members)
                centroids.append(vertices[list(members)].mean(axis=0))
    return centroids


def mixture_candidates(a_mat: np.ndarray, b_vec: np.ndarray) -> dict[str, np.ndarray]:
    """Return candidate blends by kind.

    The kinds are vertices, edge midpoints, centroids of the 2-dimensional faces
    (``plane_centroid``, the points the ``x_i x_j x_k`` terms of a special cubic model
    need), centroids of the facets (``face_centroid``), the overall centroid, and
    axial blends halfway from each vertex to it. Two vertices share an edge when the
    constraints active at both, together with ``sum(x) = 1``, have rank ``q - 1``: the
    set of points satisfying them is a line.
    """
    vertices = extreme_vertices(a_mat, b_vec)
    if len(vertices) == 0:
        raise ValueError("No mixture satisfies all the constraints; check them for conflicts.")
    q = a_mat.shape[1]
    active = np.abs(vertices @ a_mat.T - b_vec) <= 1e-9
    adjacent = np.zeros((len(vertices), len(vertices)), dtype=bool)
    for i, j in itertools.combinations(range(len(vertices)), 2):
        adjacent[i, j] = adjacent[j, i] = _face_rank(a_mat, active[i] & active[j]) == q - 1
    edges = [(vertices[i] + vertices[j]) / 2 for i, j in zip(*np.nonzero(np.triu(adjacent)), strict=True)]
    faces = [vertices[active[:, r]].mean(axis=0) for r in range(a_mat.shape[0]) if active[:, r].sum() > 2]
    centroid = vertices.mean(axis=0, keepdims=True)
    return {
        "vertex": vertices,
        "edge_midpoint": _unique_rows(np.array(edges).reshape(-1, q)),
        "plane_centroid": _unique_rows(np.array(_plane_centroids(vertices, active, a_mat, adjacent)).reshape(-1, q)),
        "face_centroid": _unique_rows(np.array(faces).reshape(-1, q)),
        "centroid": centroid,
        "axial_blend": (vertices + centroid) / 2,
    }


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


#: Point kinds in the classical extreme-vertices design, by Scheffé model.
_EV_DESIGN = {
    "scheffe_linear": ("vertex", "centroid"),
    "scheffe_quadratic": ("vertex", "edge_midpoint", "centroid"),
    "scheffe_special_cubic": ("vertex", "edge_midpoint", "plane_centroid", "centroid"),
}


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
        Number of runs. ``None`` returns the classical extreme-vertices design for the
        model: vertices and centroid, plus edge midpoints for a quadratic model, plus
        the centroids of the 2-dimensional faces for a special cubic one. With a budget, a D-optimal subset of
        all candidate blends is chosen; blends may be replicated.
    constraints : list[Constraint] or None
        Linear inequalities in the proportions, e.g. ``"x1 + x2 <= 0.7"``.
    options : ConstrainedOptions or None
        ``model_type``: ``"scheffe_linear"``, ``"scheffe_quadratic"`` or
        ``"scheffe_special_cubic"``, or the process-design names ``"main_effects"``,
        ``"interactions"`` and ``"quadratic"``, which map to linear, quadratic and
        quadratic. ``criterion``: ``"d_optimal"`` (default), ``"i_optimal"`` (average
        prediction variance over the constrained simplex) or ``"a_optimal"``.
    random_state : int, numpy.random.Generator or None
        Seed for the exchange's random starts.

    Returns
    -------
    tuple[np.ndarray, dict]
        Proportions ``(n_runs, q)`` whose rows sum to 1, and metadata.

    Raises
    ------
    ValueError
        If the region is empty, a constraint is not linear, or the region cannot
        support the model (for example a region that is a single blend).
    """
    opts = options if options is not None else ConstrainedOptions(model_type="scheffe_quadratic")
    model = scheffe_model(opts.model_type)
    rng = check_random_state(random_state)
    a_mat, b_vec = mixture_inequalities(factors, constraints)
    candidates = mixture_candidates(a_mat, b_vec)
    n_parameters = scheffe_matrix(np.ones((1, len(factors))), model).shape[1]
    labels: list | None = None
    if opts.candidates is not None:
        pool, labels = _user_blends(factors, opts.candidates, a_mat, b_vec)
        budget = budget if budget is not None else n_parameters + 3

        def region_rows() -> np.ndarray:
            return scheffe_matrix(pool, model)  # the supplied blends stand for the region
    else:
        pool = _unique_rows(np.vstack(list(candidates.values())))

        def region_rows() -> np.ndarray:
            from process_improve.experiments.region import DesignRegion  # noqa: PLC0415

            return scheffe_matrix(DesignRegion(factors, constraints).sample(20_000, rng), model)

    if budget is None:
        design = _unique_rows(np.vstack([candidates[kind] for kind in _EV_DESIGN[model]]))
        method, logdet = "extreme_vertices", None
    else:
        if budget < n_parameters:
            logger.warning(
                "A budget of %d run(s) cannot estimate a %s model with %d terms; raising the budget to %d.",
                budget,
                model,
                n_parameters,
                n_parameters,
            )
            budget = n_parameters
        criterion = make_criterion(opts.criterion, n_parameters, region_rows)
        rows, logdet = fedorov_exchange(
            scheffe_matrix(pool, model), budget, np.empty((0, n_parameters)), rng, criterion
        )
        design = pool[rows]
        method = f"{opts.criterion}_{'user_candidates' if labels is not None else 'extreme_vertices'}"

    if np.linalg.matrix_rank(scheffe_matrix(design, model)) < n_parameters:
        raise ValueError(
            f"The constrained mixture region cannot support a {model} model ({n_parameters} terms): "
            "it has too few distinct blends. Use a simpler model or loosen the constraints."
        )
    meta = {
        "method": method,
        "model_type": model,
        "n_vertices": len(candidates["vertex"]),
        "n_candidates": int(sum(len(v) for v in candidates.values())),
        "constraints": [c.expression for c in constraints or []],
        "constraints_enforced": True,
    }
    if labels is not None:
        meta["candidate_source"] = "user"
        meta["selected_candidates"] = selection_counts(labels, rows)
    if logdet is not None:
        meta["optimality_criterion"] = opts.criterion
        meta.update(criterion_metadata(criterion, logdet))
    return design, meta
