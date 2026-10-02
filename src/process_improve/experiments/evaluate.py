# (c) Kevin Dunn, 2010-2026. MIT License.

"""Design evaluation: quality metrics for experimental designs.

Provides :func:`evaluate_design`, which computes properties and quality metrics
of an existing design matrix.  Supported metrics include efficiency values
(D/I/G), prediction variance, VIF, condition number, power analysis, alias
structure, confounding, resolution, defining relation, clear effects, minimum
aberration, moment aberration, and degrees of freedom.

Example
-------
>>> from process_improve.experiments import evaluate_design, generate_design, Factor
>>> factors = [Factor(name="A", low=0, high=10), Factor(name="B", low=0, high=10)]
>>> result = generate_design(factors, design_type="full_factorial", n_center_points=0)
>>> metrics = evaluate_design(result, model="interactions", metric=["d_efficiency", "vif"])
"""

from __future__ import annotations

import contextlib
import itertools
import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from patsy import EvalFactor, ModelDesc, Term, build_design_matrices, dmatrix
from patsy.design_info import DesignInfo
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

from process_improve._random import check_random_state, resolve_deprecated_seed
from process_improve.experiments._moment_aberration import NotTwoLevelError, moment_aberration
from process_improve.experiments.designs_mixture_constrained import SCHEFFE_MODELS, scheffe_formula_rhs
from process_improve.experiments.factor import DesignResult
from process_improve.experiments.models import validate_formula_is_safe, validate_identifier_is_safe
from process_improve.experiments.region import DesignRegion

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal context shared across metric computations
# ---------------------------------------------------------------------------


@dataclass
class _EvalRequest:
    """Caller-supplied evaluation request.

    Bundles every piece of metric-independent input so
    :func:`_build_context` can accept a single dataclass instead of
    nine positional / keyword arguments (ENG-25 / #307: removes one
    ``noqa: PLR0913``).
    """

    design_df: pd.DataFrame
    factor_names: list[str]
    model: str | None
    generators: list[str] | None
    defining_relation: list[str] | None
    resolution: int | None
    effect_size: float | None
    alpha: float
    sigma: float | None
    region: str | DesignRegion = "cuboidal"
    n_samples: int = 100_000
    include_vertices: bool = True
    random_state: int | np.random.Generator | None = 42
    fds_resolution: int | None = None
    design_type: str | None = None


@dataclass
class _EvalContext:
    """Shared evaluation context, computed once and reused by all metrics."""

    X: np.ndarray
    column_names: list[str]
    factor_names: list[str]
    design_df: pd.DataFrame
    design_info: DesignInfo  # patsy DesignInfo of the fitted model matrix
    N: int
    p: int
    XtX: np.ndarray
    XtX_inv: np.ndarray | None  # None when X'X is singular
    is_singular: bool
    generators: list[str] | None
    defining_relation: list[str] | None
    resolution: int | None
    effect_size: float | None
    alpha: float
    sigma: float | None
    region: str | DesignRegion = "cuboidal"
    n_samples: int = 100_000
    include_vertices: bool = True
    random_state: int | np.random.Generator | None = 42
    fds_resolution: int | None = None
    design_type: str | None = None


# ---------------------------------------------------------------------------
# Model matrix construction
# ---------------------------------------------------------------------------

_ROMAN_DIGITS = ((10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"))


def _roman(n: int) -> str:
    """Write a design resolution as a Roman numeral (``2 -> "II"``, ``12 -> "XII"``)."""
    if n < 1:
        return str(n)
    out = ""
    for value, digits in _ROMAN_DIGITS:
        count, n = divmod(n, value)
        out += digits * count
    return out


def _build_model_matrix(
    design_df: pd.DataFrame,
    model: str | None,
    factor_names: list[str],
) -> tuple[np.ndarray, list[str], DesignInfo]:
    """Build the expanded model matrix *X* using patsy.

    Parameters
    ----------
    design_df : DataFrame
        Design matrix with one column per factor (coded units).
    model : str or None
        ``"main_effects"``, ``"interactions"``, ``"quadratic"``, an explicit
        patsy formula, or *None* (defaults to ``"interactions"``).
    factor_names : list[str]
        Ordered factor names.

    Returns
    -------
    X : ndarray of shape (N, p)
        The model matrix including the intercept column.
    column_names : list[str]
        Human-readable names for each column of *X*.
    design_info : patsy.DesignInfo
        The patsy design info describing the expansion.  Pass it to
        :func:`patsy.build_design_matrices` to expand *new* factor-space points
        through the identical model (see :func:`_prediction_variance_of_frame`).
    """
    if model is None:
        model = "interactions"

    # Factor names are interpolated into the formula; reject non-identifiers.
    for name in factor_names:
        validate_identifier_is_safe(name)

    # A categorical factor is carried as a non-numeric (label) column; patsy
    # contrast-codes it automatically, so it must NOT be manually expanded into
    # dummy columns by the caller (that is what creates singular within-factor
    # cross terms). Only quantitative factors get a pure-quadratic term - a
    # categorical has no square - so "quadratic" becomes a partial response
    # surface model when a categorical factor is present.
    numeric_factors = [f for f in factor_names if pd.api.types.is_numeric_dtype(design_df[f])]

    # Map shorthand names to patsy right-hand-side formulas
    joined = " + ".join(factor_names)
    if model == "main_effects":
        rhs = joined
    elif model == "interactions":
        rhs = f"({joined}) ** 2"
    elif model == "quadratic":
        squared = " + ".join(f"I({f} ** 2)" for f in numeric_factors)
        rhs = f"({joined}) ** 2 + {squared}" if squared else f"({joined}) ** 2"
    elif model in SCHEFFE_MODELS:
        rhs = scheffe_formula_rhs(factor_names, model)
    elif "~" in model:
        # Explicit formula with response side - strip LHS
        rhs = model.split("~", 1)[1].strip()
    else:
        # Assume it is already a valid RHS formula
        rhs = model

    # Patsy evaluates each term as Python, so a custom ``model`` is a code-
    # execution vector. Permit only safe column arithmetic, with I()/Q() (SEC-14).
    validate_formula_is_safe(rhs, design_df.columns, allow_transforms=True)
    dm = dmatrix(rhs, design_df, return_type="dataframe")
    X = np.asarray(dm, dtype=float)
    column_names = list(dm.columns)
    return X, column_names, dm.design_info


def _build_context(req: _EvalRequest) -> _EvalContext:
    """Build the shared evaluation context."""
    X, column_names, design_info = _build_model_matrix(req.design_df, req.model, req.factor_names)
    N, p = X.shape
    XtX = X.T @ X

    rank = np.linalg.matrix_rank(XtX)
    is_singular = rank < p
    XtX_inv: np.ndarray | None = None
    if not is_singular:
        XtX_inv = np.linalg.inv(XtX)

    return _EvalContext(
        X=X,
        column_names=column_names,
        factor_names=req.factor_names,
        design_df=req.design_df,
        design_info=design_info,
        N=N,
        p=p,
        XtX=XtX,
        XtX_inv=XtX_inv,
        is_singular=is_singular,
        generators=req.generators,
        defining_relation=req.defining_relation,
        resolution=req.resolution,
        effect_size=req.effect_size,
        alpha=req.alpha,
        sigma=req.sigma,
        region=req.region,
        n_samples=req.n_samples,
        include_vertices=req.include_vertices,
        random_state=req.random_state,
        fds_resolution=req.fds_resolution,
        design_type=req.design_type,
    )


# ---------------------------------------------------------------------------
# Metric implementations
# ---------------------------------------------------------------------------


def _compute_d_efficiency(ctx: _EvalContext) -> dict[str, Any]:
    """D-efficiency: 100 * det(X'X)^(1/p) / N."""
    if ctx.is_singular:
        return {"d_efficiency": None, "note": "Design is rank-deficient for the specified model."}
    sign, logdet = np.linalg.slogdet(ctx.XtX)
    if sign <= 0:
        return {"d_efficiency": None, "note": "X'X has non-positive determinant."}
    d_eff = 100.0 * np.exp(logdet / ctx.p) / ctx.N
    return {"d_efficiency": float(d_eff)}


def _prediction_variance_at_points(X_points: np.ndarray, XtX_inv: np.ndarray) -> np.ndarray:
    """Compute d(x) = x' (X'X)^-1 x for each row of X_points."""
    # (X_points @ XtX_inv) element-wise * X_points, summed per row
    return np.sum((X_points @ XtX_inv) * X_points, axis=1)


def _cube_vertices(k: int) -> np.ndarray:
    """Return all ``2**k`` cube vertices (corners) of ``[-1, 1]^k``."""
    return np.array(list(itertools.product([-1.0, 1.0], repeat=k)), dtype=float)


def _region_points(
    factor_names: list[str],
    region: str,
    n_samples: int,
    include_vertices: bool,
    random_state: int | np.random.Generator | None,
) -> np.ndarray:
    """Sample raw factor-space points over the design region.

    Parameters
    ----------
    factor_names : list[str]
        Ordered factor names (only the count ``k`` matters here).
    region : {"cuboidal", "spherical"}
        ``"cuboidal"`` samples uniformly in ``[-1, 1]^k``; ``"spherical"``
        samples uniformly inside the ball of radius ``sqrt(k)`` (the sphere
        that circumscribes the unit cube).
    n_samples : int
        Number of random interior samples to draw.
    include_vertices : bool
        When *True*, append all ``2**k`` cube vertices to the sample set.  The
        worst-case prediction variance for second-order models very often sits
        at (or near) a corner, so the corners are always represented in the
        G / FDS statistics.
    random_state : int, numpy.random.Generator or None
        Seed for the NumPy random generator (full reproducibility).

    Returns
    -------
    ndarray of shape (M, k)
        The sampled points, with the cube vertices appended last when
        *include_vertices* is set.
    """
    k = len(factor_names)
    rng = check_random_state(random_state)
    if region == "cuboidal":
        pts = rng.uniform(-1.0, 1.0, size=(n_samples, k))
    elif region == "spherical":
        # Uniform in the radius-sqrt(k) ball: direction * radius, radius ~ U^(1/k).
        directions = rng.normal(size=(n_samples, k))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        radii = np.sqrt(k) * rng.uniform(0.0, 1.0, size=(n_samples, 1)) ** (1.0 / k)
        pts = directions * radii
    else:
        raise ValueError(f"Unknown region={region!r}.  Choose 'cuboidal' or 'spherical'.")

    if include_vertices and k > 0:
        pts = np.vstack([pts, _cube_vertices(k)])
    return pts


def _region_prediction_variance(ctx: _EvalContext) -> tuple[np.ndarray, np.ndarray]:
    """Prediction variance ``d(x) = x' (X'X)^-1 x`` over the design region.

    Single source of truth for the region-based metrics (I / G efficiency and
    the FDS curve), sampled with the region settings carried on *ctx*.

    Returns
    -------
    interior : ndarray
        ``d(x)`` at the uniform sample of the region. The region average (the
        I-criterion) and the FDS curve are read from this sample alone: the I-criterion
        is the integral over the region, which boundary points would bias upward.
    boundary : ndarray
        ``d(x)`` at the boundary points added when ``include_vertices`` is set (the cube
        vertices, crossed with every combination of categorical levels; or the support
        points of a :class:`DesignRegion`), where the worst case usually sits. Used only
        for the maximum (G). Empty when ``include_vertices`` is off.
    """
    assert ctx.XtX_inv is not None  # callers guard on ``ctx.is_singular``
    if isinstance(ctx.region, DesignRegion):
        interior_df, boundary_df = _points_in_design_region(ctx, ctx.region)
    else:
        interior_df, boundary_df = _points_in_box_region(ctx, ctx.region)
    return _prediction_variance_of_frame(ctx, interior_df), _prediction_variance_of_frame(ctx, boundary_df)


def _prediction_variance_of_frame(ctx: _EvalContext, points: pd.DataFrame) -> np.ndarray:
    """Expand factor-space points (one column per factor) through the fitted model and return ``d(x)``."""
    assert ctx.XtX_inv is not None
    if points.empty:
        return np.empty(0)
    (expanded,) = build_design_matrices([ctx.design_info], points[ctx.factor_names], return_type="matrix")
    return _prediction_variance_at_points(np.asarray(expanded, dtype=float), ctx.XtX_inv)


def _points_in_box_region(ctx: _EvalContext, region: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Uniform sample and corner points of the cuboidal or spherical region.

    The quantitative factors are sampled over *region* (which is validated); each
    categorical factor (a label column) uniformly over its observed levels. The corners
    are every cube vertex of the quantitative factors crossed with every combination of
    categorical levels, so no (vertex, level) pair is missed.
    """
    cat_levels = {
        f: list(ctx.design_df[f].unique())
        for f in ctx.factor_names
        if not pd.api.types.is_numeric_dtype(ctx.design_df[f])
    }
    cont_names = [f for f in ctx.factor_names if f not in cat_levels]
    rng = check_random_state(ctx.random_state)
    points = _region_points(cont_names, region, ctx.n_samples, include_vertices=False, random_state=rng)
    interior: dict[str, Any] = {f: points[:, j] for j, f in enumerate(cont_names)}
    for f, levels in cat_levels.items():
        interior[f] = rng.choice(np.asarray(levels, dtype=object), size=ctx.n_samples)

    if not ctx.include_vertices:
        return pd.DataFrame(interior), pd.DataFrame()
    axes = [(-1.0, 1.0)] * len(cont_names) + list(cat_levels.values())
    boundary = pd.DataFrame(list(itertools.product(*axes)), columns=cont_names + list(cat_levels))
    return pd.DataFrame(interior), boundary


def _points_in_design_region(ctx: _EvalContext, region: DesignRegion) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Uniform sample and support points of a constrained or mixture region.

    The region's boundary points (extreme vertices and edge midpoints for a mixture;
    constraint crossings and feasible grid corners for a box) stand in for the cube
    vertices, since that is where the worst-case variance of a constrained design sits.
    Categorical factors are sampled uniformly over their observed levels.
    """
    missing = [n for n in region.names if n not in ctx.factor_names]
    if missing:
        raise ValueError(f"The region names factors {missing} that are not columns of the design.")
    rng = check_random_state(ctx.random_state)
    frames = [region.sample(ctx.n_samples, rng)]
    if ctx.include_vertices:
        with contextlib.suppress(ValueError):  # a grid too large for support points: sample only
            frames.append(region.support_points())
    out: list[pd.DataFrame] = []
    for points in frames:
        data: dict[str, Any] = {n: points[:, j] for j, n in enumerate(region.names)}
        for f in ctx.factor_names:
            if f not in data:
                data[f] = rng.choice(ctx.design_df[f].unique(), size=len(points))
        out.append(pd.DataFrame(data))
    return out[0], out[1] if len(out) > 1 else pd.DataFrame()


def _region_label(region: str | DesignRegion) -> str:
    """Name the region in results: the string as given, or ``"mixture"`` / ``"constrained"``."""
    if isinstance(region, str):
        return region
    return region.kind if region.kind == "mixture" else "constrained"


def _resolve_region(
    region: str | DesignRegion | None, design_matrix: pd.DataFrame | DesignResult
) -> str | DesignRegion:
    """Use the caller's region, else the one ``generate_design`` recorded, else the cube."""
    if region is not None:
        return region
    spec = design_matrix.metadata.get("region") if isinstance(design_matrix, DesignResult) else None
    if spec:
        recorded = DesignRegion.from_dict(spec)
        if recorded.kind == "mixture" or recorded.is_constrained:
            return recorded
    return "cuboidal"


def _compute_g_efficiency(ctx: _EvalContext) -> dict[str, Any]:
    """G-efficiency: 100 * p / (N * max prediction variance over design region)."""
    if ctx.is_singular:
        return {"g_efficiency": None, "note": "Design is rank-deficient for the specified model."}

    max_pv = float(np.max(np.concatenate(_region_prediction_variance(ctx))))

    g_eff = 100.0 * ctx.p / (ctx.N * max_pv) if max_pv > 0 else None
    return {
        "g_efficiency": float(g_eff) if g_eff is not None else None,
        "max_prediction_variance": max_pv,
    }


def _compute_average_prediction_variance(ctx: _EvalContext) -> dict[str, Any]:
    """I-criterion: the prediction variance ``f(x)'(X'X)^-1 f(x)`` averaged over the design region.

    In units of the error variance, and lower is better. This is what JMP reports as the
    "average variance of prediction" and what an I-optimal design minimises.
    """
    if ctx.is_singular:
        return {"average_prediction_variance": None, "note": "Design is rank-deficient for the specified model."}
    interior, _boundary = _region_prediction_variance(ctx)
    return {"average_prediction_variance": float(np.mean(interior))}


def _compute_i_efficiency(ctx: _EvalContext) -> dict[str, Any]:
    """Compute the deprecated ``100 * p / (N * average prediction variance)``, which is not bounded by 100."""
    if ctx.is_singular:
        return {"i_efficiency": None, "note": "Design is rank-deficient for the specified model."}

    interior, _boundary = _region_prediction_variance(ctx)
    avg_pv = float(np.mean(interior))

    i_eff = 100.0 * ctx.p / (ctx.N * avg_pv) if avg_pv > 0 else None
    return {
        "i_efficiency": float(i_eff) if i_eff is not None else None,
        "average_prediction_variance": avg_pv,
    }


def _term_order(name: str) -> int:
    """Classify a model-matrix column by its polynomial order.

    Returns ``0`` for the intercept, ``1`` for a main effect, and ``2`` for a
    second-order term (a pure quadratic ``I(x ** 2)`` or a two-factor
    interaction ``x:y``).  Sufficient for the response-surface models this
    module evaluates; higher powers are reported as their interaction depth.
    """
    if name.lower() == "intercept" or name == "1":
        return 0
    if ":" in name:
        return name.count(":") + 1
    if "**" in name or name.startswith("I("):
        return 2
    return 1


def _compute_a_optimality(ctx: _EvalContext) -> dict[str, Any]:
    """A-optimality: ``trace((X'X)^-1)`` (lower is better).

    The trace of the inverse information matrix is the sum (so, up to a
    constant, the average) of the coefficient variances.  ``a_efficiency`` is a
    normalised score in the same spirit as ``d_efficiency`` (higher is better).
    """
    if ctx.is_singular:
        return {"a_optimality": None, "note": "Design is rank-deficient for the specified model."}

    assert ctx.XtX_inv is not None
    trace = float(np.trace(ctx.XtX_inv))
    a_eff = 100.0 * ctx.p / (ctx.N * trace) if trace > 0 else None
    return {
        "a_optimality": trace,
        "a_efficiency": float(a_eff) if a_eff is not None else None,
    }


def _compute_e_optimality(ctx: _EvalContext) -> dict[str, Any]:
    """E-optimality: smallest eigenvalue of ``X'X`` (higher is better).

    The minimum eigenvalue measures how well the worst-estimated direction in
    parameter space is supported by the design.  ``e_efficiency`` scales it by
    the run count for comparison across designs of different size.
    """
    min_eig = float(np.linalg.eigvalsh(ctx.XtX).min())
    return {
        "e_optimality": min_eig,
        "e_efficiency": 100.0 * min_eig / ctx.N if ctx.N > 0 else None,
    }


def _compute_correlation(ctx: _EvalContext) -> dict[str, Any]:
    """Pairwise correlation summary among the model's second-order terms.

    The pure-quadratic columns ``x_i^2`` have a non-zero mean, so a raw Pearson
    correlation between them is inflated by that shared offset and depends on
    the coding.  To get a coding-invariant measure, each second-order column is
    first residualised against the intercept-and-main-effect block (the
    content those columns share by construction) and the correlations are taken
    on the residuals.

    Returns ``max_abs_r``, ``mean_abs_r``, the full correlation ``matrix`` (as
    nested lists), and the ordered ``terms``.
    """
    second_idx = [i for i, c in enumerate(ctx.column_names) if _term_order(c) == 2]
    base_idx = [i for i, c in enumerate(ctx.column_names) if _term_order(c) in (0, 1)]
    if len(second_idx) < 2:
        return {
            "correlation": {
                "max_abs_r": 0.0,
                "mean_abs_r": 0.0,
                "matrix": [[1.0]] if second_idx else [],
                "terms": [ctx.column_names[i] for i in second_idx],
                "note": "Fewer than two second-order terms; no pairwise correlation.",
            }
        }

    second = ctx.X[:, second_idx]
    base = ctx.X[:, base_idx]
    # Residualise the second-order columns against [intercept, main effects].
    resid = second - base @ (np.linalg.pinv(base) @ second)

    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(resid, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    corr = np.atleast_2d(corr)

    m = corr.shape[0]
    iu = np.triu_indices(m, k=1)
    off_diag = np.abs(corr[iu])
    max_abs = float(off_diag.max()) if off_diag.size else 0.0
    mean_abs = float(off_diag.mean()) if off_diag.size else 0.0
    return {
        "correlation": {
            "max_abs_r": max_abs,
            "mean_abs_r": mean_abs,
            "matrix": corr.tolist(),
            "terms": [ctx.column_names[i] for i in second_idx],
        }
    }


def _omitted_two_factor_interactions(ctx: _EvalContext) -> tuple[np.ndarray, list[str]]:
    """Build the two-factor-interaction columns *not* already in the model.

    Returns the ``(N, q)`` matrix of the interaction columns and their names, for
    every factor pair whose interaction is absent from the fitted model. The columns
    are built by patsy next to the model's own terms, so a categorical factor is
    contrast-coded exactly as it is in ``X`` (``"A:C[T.y]"``), and a quantitative
    pair gives the product ``x_a * x_b`` (``"A:B"``).
    """
    present = {frozenset(f.name() for f in term.factors) for term in ctx.design_info.terms}
    omitted = [
        Term([EvalFactor(a), EvalFactor(b)])
        for a, b in itertools.combinations(ctx.factor_names, 2)
        if frozenset((a, b)) not in present
    ]
    if not omitted:
        return np.empty((ctx.N, 0)), []
    dm = dmatrix(ModelDesc([], [*ctx.design_info.terms, *omitted]), ctx.design_df, return_type="dataframe")
    slices = [dm.design_info.term_slices[term] for term in omitted]
    columns = [i for sl in slices for i in range(sl.start, sl.stop)]
    return np.asarray(dm, dtype=float)[:, columns], [dm.columns[i] for i in columns]


def _compute_alias_matrix(ctx: _EvalContext) -> dict[str, Any]:
    """General alias (bias) matrix ``A = (X1'X1)^-1 X1' X2``.

    With the fitted model ``X1`` and a set of potential extra terms ``X2``
    (default: the two-factor interactions not already in the model), the least
    squares estimate of the fitted coefficients is biased by
    ``E[b1] = beta1 + A @ beta2``.  This generalises :func:`alias_structure`
    (which only handles two-level fractional factorials) to any design / model.

    Returns the ``matrix`` (nested lists), the ``model_terms`` (rows) and
    ``alias_terms`` (columns), the worst single bias ``max_abs``, the maximum
    over the main-effect rows ``max_abs_main_effect_rows``, and the Frobenius
    norm ``frobenius_norm``.
    """
    if ctx.is_singular:
        return {"alias_matrix": None, "note": "Design is rank-deficient for the specified model."}

    assert ctx.XtX_inv is not None
    X2, alias_terms = _omitted_two_factor_interactions(ctx)
    if X2.shape[1] == 0:
        return {
            "alias_matrix": {
                "matrix": [],
                "model_terms": list(ctx.column_names),
                "alias_terms": [],
                "max_abs": 0.0,
                "max_abs_main_effect_rows": 0.0,
                "frobenius_norm": 0.0,
                "note": "No two-factor interactions outside the model to alias against.",
            }
        }

    alias = ctx.XtX_inv @ ctx.X.T @ X2
    main_rows = [i for i, c in enumerate(ctx.column_names) if _term_order(c) == 1]
    max_abs = float(np.max(np.abs(alias)))
    max_main = float(np.max(np.abs(alias[main_rows, :]))) if main_rows else 0.0
    return {
        "alias_matrix": {
            "matrix": alias.tolist(),
            "model_terms": list(ctx.column_names),
            "alias_terms": alias_terms,
            "max_abs": max_abs,
            "max_abs_main_effect_rows": max_main,
            "frobenius_norm": float(np.linalg.norm(alias)),
        }
    }


_FDS_QUANTILES = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)


def _compute_fds(ctx: _EvalContext) -> dict[str, Any]:
    """Fraction-of-design-space (FDS) distribution of the prediction variance.

    Samples the scaled prediction variance ``d(x) = x' (X'X)^-1 x`` over the
    whole design region (not just the design points), returning quantiles of
    the curve together with the region average (I / V-optimality) and maximum
    (G-optimality) in ``sigma^2`` units, plus the run-count-scaled SPV variants
    (multiplied by ``N``).  The region settings are echoed back for
    reproducibility.

    When ``ctx.fds_resolution`` is set, a dense ``curve`` sub-dict is added with
    ``fraction``, ``prediction_variance``, and ``scaled_prediction_variance``
    arrays of that length, evaluated on evenly spaced fractions in ``[0, 1]``
    (the endpoints are the minimum and maximum prediction variance) - suitable
    for drawing a smooth FDS plot.  The coarse 11-point ``quantiles`` summary is
    always present for backward compatibility.

    The curve, the quantiles and the average come from the uniform sample of the
    region, since the FDS curve is the distribution over the region; boundary points
    would bias it upward. Its value at fraction 1 is the maximum over the region,
    which the boundary points added by ``include_vertices`` help locate, so the curve
    ends at ``max_prediction_variance``, the value ``g_efficiency`` uses.
    """
    if ctx.is_singular:
        return {"fds": None, "note": "Design is rank-deficient for the specified model."}

    pv, boundary = _region_prediction_variance(ctx)
    pv = np.sort(pv)
    avg = float(pv.mean())
    mx = float(np.max(np.concatenate([pv, boundary])))
    quantile_values = np.quantile(pv, _FDS_QUANTILES)
    quantile_values[-1] = mx  # fraction 1 is the maximum over the region
    quantiles = {f"{q:g}": float(v) for q, v in zip(_FDS_QUANTILES, quantile_values, strict=True)}
    payload: dict[str, Any] = {
        "region": _region_label(ctx.region),
        "n_samples": ctx.n_samples,
        "include_vertices": ctx.include_vertices,
        "random_seed": ctx.random_state if isinstance(ctx.random_state, int) else None,
        "fds_resolution": ctx.fds_resolution,
        "quantiles": quantiles,
        "average_prediction_variance": avg,
        "max_prediction_variance": mx,
        "scaled_average_prediction_variance": avg * ctx.N,
        "scaled_max_prediction_variance": mx * ctx.N,
    }
    if ctx.fds_resolution is not None:
        if ctx.fds_resolution < 2:
            raise ValueError(f"fds_resolution must be at least 2, got {ctx.fds_resolution}.")
        fractions = np.linspace(0.0, 1.0, ctx.fds_resolution)
        curve = np.quantile(pv, fractions)  # non-decreasing; endpoints are min and max
        curve[-1] = mx
        payload["curve"] = {
            "fraction": fractions.tolist(),
            "prediction_variance": curve.tolist(),
            "scaled_prediction_variance": (curve * ctx.N).tolist(),
        }
    return {"fds": payload}


def _compute_prediction_variance(ctx: _EvalContext) -> dict[str, Any]:
    """Prediction variance d(x_i) = x_i' (X'X)^-1 x_i at each design point."""
    if ctx.is_singular:
        return {"prediction_variance": None, "note": "Design is rank-deficient for the specified model."}

    # ``XtX_inv`` is non-None whenever the design is not singular (guarded above).
    assert ctx.XtX_inv is not None
    pv = _prediction_variance_at_points(ctx.X, ctx.XtX_inv)
    pv_list = [float(v) for v in pv]
    return {
        "prediction_variance": pv_list,
        "mean": float(np.mean(pv)),
        "max": float(np.max(pv)),
        "min": float(np.min(pv)),
    }


def _compute_vif(ctx: _EvalContext) -> dict[str, Any]:
    """Variance Inflation Factor for each model term (excluding intercept)."""
    if ctx.is_singular:
        return {"vif": None, "note": "Design is rank-deficient for the specified model."}

    vif_dict: dict[str, float] = {}
    for i, name in enumerate(ctx.column_names):
        if name.lower() == "intercept" or name == "1":
            continue
        vif_val = variance_inflation_factor(ctx.X, i)
        vif_dict[name] = float(vif_val)
    return {"vif": vif_dict}


def _compute_condition_number(ctx: _EvalContext) -> dict[str, float]:
    """Condition number of the model matrix X."""
    cn = float(np.linalg.cond(ctx.X))
    return {"condition_number": cn}


def _compute_power(ctx: _EvalContext) -> dict[str, Any]:
    """Statistical power for detecting each model term."""
    if ctx.is_singular:
        return {"power": None, "note": "Design is rank-deficient for the specified model."}

    sigma = ctx.sigma if ctx.sigma is not None else 1.0
    df_resid = ctx.N - ctx.p

    if df_resid <= 0:
        return {"power": None, "note": "No residual degrees of freedom (saturated model)."}

    assert ctx.XtX_inv is not None  # guaranteed by not is_singular
    diag_inv = np.diag(ctx.XtX_inv)

    if ctx.effect_size is not None:
        # Single power value per term
        power_dict: dict[str, float] = {}
        for i, name in enumerate(ctx.column_names):
            if name.lower() == "intercept" or name == "1":
                continue
            ncp = (ctx.effect_size**2) / (sigma**2 * diag_inv[i])
            f_crit = stats.f.ppf(1.0 - ctx.alpha, dfn=1, dfd=df_resid)
            pwr = 1.0 - stats.ncf.cdf(f_crit, dfn=1, dfd=df_resid, nc=ncp)
            power_dict[name] = float(pwr)
        return {"power": power_dict}

    # No effect_size: generate power curves over a range of effect sizes
    effect_sizes = np.linspace(0.5 * sigma, 3.0 * sigma, 20)
    power_curves: dict[str, list[dict[str, float]]] = {}
    for i, name in enumerate(ctx.column_names):
        if name.lower() == "intercept" or name == "1":
            continue
        curve = []
        for es in effect_sizes:
            ncp = (es**2) / (sigma**2 * diag_inv[i])
            f_crit = stats.f.ppf(1.0 - ctx.alpha, dfn=1, dfd=df_resid)
            pwr = 1.0 - stats.ncf.cdf(f_crit, dfn=1, dfd=df_resid, nc=ncp)
            curve.append({"effect_size": float(es), "power": float(pwr)})
        power_curves[name] = curve
    return {"power_curves": power_curves, "sigma": float(sigma)}


def _spans_constant(X: np.ndarray) -> bool:
    """Whether the constant column lies in the column space of *X*.

    True for a model with an intercept, and for a Scheffé mixture model, whose linear
    blending terms sum to one (so its ANOVA is corrected for the mean, as in Cornell).
    """
    ones = np.ones(X.shape[0])
    fitted = X @ np.linalg.lstsq(X, ones, rcond=None)[0]
    return bool(np.allclose(fitted, ones, atol=1e-8))


def _compute_degrees_of_freedom(ctx: _EvalContext) -> dict[str, Any]:
    """Degrees-of-freedom breakdown, from the rank of ``X``.

    ``model = rank - m``, ``residual = N - rank`` and ``total = N - m``, where ``m`` is
    1 when the model spans the constant (an intercept, or a Scheffé model) and 0
    otherwise. The residual splits into ``pure_error`` (replicated runs, ``N`` minus the
    number of distinct settings) and ``lack_of_fit``; both are always reported, as 0
    when there are no replicates. A rank-deficient model says so in a note.
    """
    rank = int(np.linalg.matrix_rank(ctx.X))
    mean_df = int(_spans_constant(ctx.X))

    # Detect replicates by counting distinct factor-setting rows. Round the
    # quantitative columns to absorb floating-point noise; categorical (label)
    # columns are compared as-is (they cannot be rounded).
    design_sub = ctx.design_df[ctx.factor_names].copy()
    numeric_cols = [c for c in ctx.factor_names if pd.api.types.is_numeric_dtype(design_sub[c])]
    if numeric_cols:
        design_sub[numeric_cols] = design_sub[numeric_cols].round(10)
    n_distinct = len(design_sub.drop_duplicates())

    result: dict[str, Any] = {
        "degrees_of_freedom": {
            "model": rank - mean_df,
            "residual": ctx.N - rank,
            "total": ctx.N - mean_df,
            "pure_error": ctx.N - n_distinct,
            "lack_of_fit": n_distinct - rank,
        }
    }
    if rank < ctx.p:
        result["note"] = f"The model has {ctx.p} columns but rank {rank}; degrees of freedom are counted from the rank."
    return result


# ---------------------------------------------------------------------------
# Alias / confounding metrics (GF(2) arithmetic on generator words)
# ---------------------------------------------------------------------------


def _parse_word(word: str, factor_names: list[str]) -> frozenset[int]:
    """Parse a word like ``"ABCE"`` into a frozenset of factor indices.

    Also handles ``"I=ABCE"`` format (strips the ``I=`` prefix).
    """
    word = word.strip().removeprefix("I=")
    if word == "I":
        return frozenset()

    # Try multi-char factor names first (when names are longer than 1 char)
    name_to_idx = {name: i for i, name in enumerate(factor_names)}

    # If all factor names are single chars, parse character-by-character
    if all(len(n) == 1 for n in factor_names):
        indices = set()
        for ch in word:
            if ch in name_to_idx:
                indices.add(name_to_idx[ch])
        return frozenset(indices)

    # Multi-char names: try to match greedily (longest first)
    remaining = word
    indices = set()
    sorted_names = sorted(name_to_idx.keys(), key=len, reverse=True)
    while remaining:
        matched = False
        for name in sorted_names:
            if remaining.startswith(name):
                indices.add(name_to_idx[name])
                remaining = remaining[len(name) :]
                matched = True
                break
        if not matched:
            remaining = remaining[1:]  # skip unrecognized character
    return frozenset(indices)


def _word_to_str(indices: frozenset[int], factor_names: list[str]) -> str:
    """Convert factor indices back to a word string."""
    if not indices:
        return "I"
    sorted_indices = sorted(indices)
    return "".join(factor_names[i] for i in sorted_indices)


def _multiply_words(w1: frozenset[int], w2: frozenset[int]) -> frozenset[int]:
    """Multiply two words in GF(2) - symmetric difference of factor sets."""
    return w1.symmetric_difference(w2)


_SignedWord = tuple[frozenset[int], int]


def _generator_sign(lhs: str, rhs: str) -> int:
    """Return -1 when exactly one side of a generator is negated (``D=-ABC``), else +1."""
    return -1 if lhs.strip().startswith("-") != rhs.strip().startswith("-") else 1


def _signed_defining_relation(generators: list[str], factor_names: list[str]) -> list[_SignedWord]:
    """Compute the full defining relation, with the sign of every word.

    Each generator like ``"D=ABC"`` produces the word ``ABCD``; ``"D=-ABC"`` produces
    ``-ABCD``, since its runs satisfy ``ABCD = -1``.  The full defining relation is the
    closure under GF(2) multiplication of the generator words (all non-empty subsets);
    the sign of a product is the product of the signs.
    """
    base_words: list[_SignedWord] = []
    for gen in generators:
        lhs, _, rhs = gen.partition("=")
        word = _multiply_words(_parse_word(lhs, factor_names), _parse_word(rhs, factor_names))
        base_words.append((word, _generator_sign(lhs, rhs)))

    signed: dict[frozenset[int], int] = {}
    for r in range(1, len(base_words) + 1):
        for subset in itertools.combinations(base_words, r):
            product: frozenset[int] = frozenset()
            sign = 1
            for w, s in subset:
                product = _multiply_words(product, w)
                sign *= s
            if product:  # exclude identity
                signed[product] = sign

    return sorted(signed.items(), key=lambda ws: (len(ws[0]), sorted(ws[0])))


def _defining_relation_from_generators(generators: list[str], factor_names: list[str]) -> list[frozenset[int]]:
    """Compute the words of the full defining relation from generator strings, without their signs.

    See :func:`_signed_defining_relation`; the word lengths (resolution, wordlength
    pattern) do not depend on the signs.
    """
    return [word for word, _sign in _signed_defining_relation(generators, factor_names)]


def _signed_word_str(word: frozenset[int], sign: int, factor_names: list[str]) -> str:
    """Render a word with a leading ``-`` when its sign is negative."""
    return ("-" if sign < 0 else "") + _word_to_str(word, factor_names)


def _defining_relation_strings(generators: list[str], factor_names: list[str]) -> list[str]:
    """Return the defining relation as ``"I=ABCD"`` / ``"I=-ABCD"`` strings, signs included."""
    return [f"I={_signed_word_str(w, s, factor_names)}" for w, s in _signed_defining_relation(generators, factor_names)]


def _compute_defining_relation(ctx: _EvalContext) -> dict[str, Any]:
    """Compute the defining relation from the generators, else return the recorded one."""
    if ctx.generators:
        return {"defining_relation": _defining_relation_strings(ctx.generators, ctx.factor_names)}
    if ctx.defining_relation:
        return {"defining_relation": ctx.defining_relation}
    return {"defining_relation": None, "note": "No generators available. Not a fractional factorial design."}


def _compute_resolution(ctx: _EvalContext) -> dict[str, Any]:
    """Design resolution = minimum word length in the defining relation."""
    if ctx.resolution is not None:
        return {"resolution": ctx.resolution, "roman": _roman(ctx.resolution)}

    if not ctx.generators:
        return {"resolution": None, "roman": None, "note": "Not a fractional factorial design."}

    words = _defining_relation_from_generators(ctx.generators, ctx.factor_names)
    if not words:
        return {"resolution": None, "roman": None, "note": "No defining relation words found."}

    res = min(len(w) for w in words)
    return {"resolution": res, "roman": _roman(res)}


def _generator_chains_apply(ctx: _EvalContext) -> bool:
    """Whether the generators' alias chains describe the whole design.

    They do for a (fractional) factorial, centre points included. The axial runs of a
    central composite design (or any other runs added to a fractional cube) break part
    of that aliasing, so for those designs the chains are found from the design itself.
    """
    return bool(ctx.generators) and (ctx.design_type is None or "factorial" in ctx.design_type)


def _cube_only_note(ctx: _EvalContext) -> str:
    return (
        f"The generators describe only the fractional cube of this {ctx.design_type!r} design; its other runs "
        "break part of the cube's aliasing, so the aliasing is found from the correlations of the whole design."
    )


def _compute_alias_structure(ctx: _EvalContext) -> dict[str, Any]:
    """Alias structure: which effects are aliased with which others.

    Uses GF(2) arithmetic when the generators describe the whole design; falls back to
    correlation-based detection otherwise.
    """
    if _generator_chains_apply(ctx):
        return _alias_structure_from_generators(ctx)
    result = _alias_structure_from_correlation(ctx)
    if ctx.generators:
        result["note"] = _cube_only_note(ctx)
    return result


def _generator_alias_chains(ctx: _EvalContext) -> list[tuple[frozenset[int], list[_SignedWord]]]:
    """Alias chain (signed aliases) of every main effect and two-factor interaction."""
    assert ctx.generators is not None  # callers check ``_generator_chains_apply``
    names = ctx.factor_names
    words = _signed_defining_relation(ctx.generators, names)
    effects = [frozenset([i]) for i in range(len(names))]
    effects += [frozenset(pair) for pair in itertools.combinations(range(len(names)), 2)]
    chains: list[tuple[frozenset[int], list[_SignedWord]]] = []
    for effect in effects:
        aliases = [(_multiply_words(effect, w), s) for w, s in words]
        aliases.sort(key=lambda a: (len(_word_to_str(a[0], names)), _word_to_str(a[0], names)))
        chains.append((effect, aliases))
    return chains


def _alias_structure_from_generators(ctx: _EvalContext) -> dict[str, Any]:
    """Compute alias chains using GF(2) multiplication against the defining relation."""
    assert ctx.generators is not None  # only reached when the generators apply
    if not _defining_relation_from_generators(ctx.generators, ctx.factor_names):
        return {"alias_structure": []}
    names = ctx.factor_names
    alias_chains = [
        f"{_word_to_str(effect, names)} = " + " + ".join(_signed_word_str(w, s, names) for w, s in aliases)
        for effect, aliases in _generator_alias_chains(ctx)
    ]
    return {"alias_structure": alias_chains}


#: Correlation above which two columns are treated as fully aliased.
_FULL_ALIAS_THRESHOLD = 0.995


def _is_intercept_col(name: str) -> bool:
    """Check if a column name represents the intercept."""
    return name.lower() == "intercept" or name == "1"


def _find_correlated_aliases(
    X_full: np.ndarray, col_names: list[str], col_idx: int, nonzero: np.ndarray, threshold: float
) -> list[tuple[str, str]]:
    """Find columns highly correlated with column *col_idx*."""
    aliases: list[tuple[str, str]] = []
    for j in range(X_full.shape[1]):
        if j == col_idx or not nonzero[j] or _is_intercept_col(col_names[j]):
            continue
        corr = np.corrcoef(X_full[:, col_idx], X_full[:, j])[0, 1]
        if abs(corr) > threshold:
            sign = "+" if corr > 0 else "-"
            aliases.append((sign, col_names[j]))
    return aliases


def _alias_structure_from_correlation(ctx: _EvalContext) -> dict[str, Any]:
    """Detect aliasing via correlation of model matrix columns."""
    if ctx.p <= 1:
        return {"alias_structure": []}

    # Build an expanded model matrix with higher-order terms for alias detection
    factor_str = " + ".join(ctx.factor_names)
    k = len(ctx.factor_names)
    max_order = min(k, 3)
    rhs = f"({factor_str}) ** {max_order}" if max_order >= 3 else f"({factor_str}) ** 2"

    dm = dmatrix(rhs, ctx.design_df, return_type="dataframe")
    X_full = np.asarray(dm, dtype=float)
    col_names = list(dm.columns)

    stddevs = X_full.std(axis=0)
    nonzero = stddevs > np.sqrt(np.finfo(float).eps)

    alias_chains: list[str] = []
    for i in range(X_full.shape[1]):
        if _is_intercept_col(col_names[i]) or not nonzero[i]:
            continue
        aliases = _find_correlated_aliases(X_full, col_names, i, nonzero, threshold=_FULL_ALIAS_THRESHOLD)
        if aliases:
            alias_parts = [f"{sign}{name}" for sign, name in aliases]
            alias_chains.append(f"{col_names[i]} = " + " + ".join(alias_parts))

    return {"alias_structure": alias_chains}


def _compute_confounding(ctx: _EvalContext) -> dict[str, Any]:
    """Confounding structure: pairs of effects that cannot be distinguished."""
    alias_result = _compute_alias_structure(ctx)
    alias_chains = alias_result.get("alias_structure", [])
    if not alias_chains:
        return {"confounding": [], "note": "No confounding detected."}

    confounding_list: list[dict[str, Any]] = []
    for chain in alias_chains:
        if " = " not in chain:
            continue
        parts = chain.split(" = ", 1)
        effect = parts[0].strip()
        aliases_str = parts[1].strip()
        confounded = [a.strip().lstrip("+-") for a in aliases_str.split(" + ")]
        confounding_list.append(
            {
                "effect": effect,
                "confounded_with": confounded,
            }
        )
    return {"confounding": confounding_list}


def _clear_effects_from_correlation(ctx: _EvalContext) -> tuple[list[str], list[str]]:
    """Clear main effects and 2FIs, read from the correlations of the design's own columns.

    Builds every main-effect and two-factor-interaction column (categorical factors
    contrast-coded by patsy) and calls a term clear when none of its columns is fully
    correlated (``|r| > 0.995``) with a column of another main effect or 2FI. A term
    whose column is constant (aliased with the intercept) is not estimable, so not clear.
    """
    names = ctx.factor_names
    rhs = f"({' + '.join(names)}) ** 2" if len(names) > 1 else names[0]
    dm = dmatrix(rhs, ctx.design_df[names], return_type="dataframe")
    X = np.asarray(dm, dtype=float)
    terms: list[tuple[tuple[str, ...], list[int]]] = [
        (tuple(f.name() for f in term.factors), list(range(sl.start, sl.stop)))
        for term, sl in dm.design_info.term_slices.items()
        if term.factors
    ]
    std = X.std(axis=0)
    live = std > np.sqrt(np.finfo(float).eps)
    Z = np.zeros_like(X)
    Z[:, live] = (X[:, live] - X[:, live].mean(axis=0)) / std[live]
    aliased = np.abs(Z.T @ Z / X.shape[0]) > _FULL_ALIAS_THRESHOLD

    clear_main: list[str] = []
    clear_2fi: list[str] = []
    for factors, cols in terms:
        if not live[cols].all():
            continue
        others = [c for other, other_cols in terms if other != factors for c in other_cols if live[c]]
        if aliased[np.ix_(cols, others)].any():
            continue
        (clear_main if len(factors) == 1 else clear_2fi).append(":".join(factors))
    return clear_main, clear_2fi


def _compute_clear_effects(ctx: _EvalContext) -> dict[str, Any]:
    """Identify clear effects per Wu & Hamada's definition (2009, Sec. 5.2).

    An effect (a main effect or a two-factor interaction) is *clear* when it is
    aliased with no other main effect and no two-factor interaction; an effect
    aliased with nothing (a full factorial) is clear. The effect orders are read
    from the sets of factors in each word, so the result does not depend on how
    long the factor names are. Two-factor interactions are named ``"A:B"``.

    With generators that describe the whole design, the aliases come from the
    defining relation; otherwise from the correlations of the design's columns.
    """
    if _generator_chains_apply(ctx):
        names = ctx.factor_names
        clear = [effect for effect, aliases in _generator_alias_chains(ctx) if all(len(w) >= 3 for w, _ in aliases)]
        clear_main = [names[next(iter(e))] for e in clear if len(e) == 1]
        clear_2fi = [":".join(names[i] for i in sorted(e)) for e in clear if len(e) == 2]
        return {"clear_effects": {"main_effects": clear_main, "two_factor_interactions": clear_2fi}}

    clear_main, clear_2fi = _clear_effects_from_correlation(ctx)
    result: dict[str, Any] = {"clear_effects": {"main_effects": clear_main, "two_factor_interactions": clear_2fi}}
    if ctx.generators:
        result["note"] = _cube_only_note(ctx)
    return result


def _compute_minimum_aberration(ctx: _EvalContext) -> dict[str, Any]:
    """Wordlength pattern (A_3, A_4, ...) from the defining relation.

    The pattern starts at ``A_3``, or at the shortest word when a word is shorter
    than three letters (a resolution II or I design), so that word is not hidden.
    """
    if not ctx.generators:
        return {
            "minimum_aberration": {
                "wordlength_pattern": [],
                "note": "Not a fractional factorial design.",
            }
        }

    words = _defining_relation_from_generators(ctx.generators, ctx.factor_names)
    if not words:
        return {
            "minimum_aberration": {
                "wordlength_pattern": [],
                "note": "No defining relation words found.",
            }
        }

    lengths = [len(w) for w in words]
    start = min(3, *lengths)
    pattern_range = range(start, max(lengths) + 1)
    return {
        "minimum_aberration": {
            "wordlength_pattern": [lengths.count(i) for i in pattern_range],
            "wordlength_pattern_labels": [f"A_{i}" for i in pattern_range],
        }
    }


def _compute_moment_aberration(ctx: _EvalContext) -> dict[str, Any]:
    """Moment aberration pattern, strength and resolution (Xu, 2003).

    Unlike ``minimum_aberration``, this needs no generators: it reads the
    design matrix directly. That makes it the metric to reach for when
    checking a design this library did not construct.
    """
    try:
        result = moment_aberration(ctx.design_df[ctx.factor_names])
    except NotTwoLevelError as exc:
        return {"moment_aberration": {"pattern": [], "note": str(exc)}}
    except ValueError as exc:
        logger.debug("moment_aberration declined the design: %s", exc)
        return {"moment_aberration": {"pattern": [], "note": str(exc)}}

    payload = result.to_dict()
    # The design carries a declared resolution when it came from a
    # DesignResult. Disagreement means the matrix is not the design it claims
    # to be, which is exactly what this metric is here to surface.
    if ctx.resolution is not None and ctx.resolution != result.resolution:
        payload["declared_resolution"] = ctx.resolution
        payload["note"] = (
            f"Declared resolution {ctx.resolution} disagrees with the resolution "
            f"{result.resolution} implied by the design matrix."
        )
    return {"moment_aberration": payload}


# ---------------------------------------------------------------------------
# Metric dispatch registry
# ---------------------------------------------------------------------------

_METRIC_REGISTRY: dict[str, Callable[[_EvalContext], Any]] = {
    "d_efficiency": _compute_d_efficiency,
    "average_prediction_variance": _compute_average_prediction_variance,
    "i_efficiency": _compute_i_efficiency,
    "g_efficiency": _compute_g_efficiency,
    "a_optimality": _compute_a_optimality,
    "e_optimality": _compute_e_optimality,
    "correlation": _compute_correlation,
    "alias_matrix": _compute_alias_matrix,
    "fds": _compute_fds,
    "prediction_variance": _compute_prediction_variance,
    "vif": _compute_vif,
    "condition_number": _compute_condition_number,
    "power": _compute_power,
    "degrees_of_freedom": _compute_degrees_of_freedom,
    "alias_structure": _compute_alias_structure,
    "confounding": _compute_confounding,
    "resolution": _compute_resolution,
    "defining_relation": _compute_defining_relation,
    "clear_effects": _compute_clear_effects,
    "minimum_aberration": _compute_minimum_aberration,
    "moment_aberration": _compute_moment_aberration,
}

#: Accepted spelling variants for the optimality-criterion metrics. The registry
#: historically mixed suffixes (``d_efficiency`` / ``i_efficiency`` /
#: ``g_efficiency`` versus ``a_optimality`` / ``e_optimality``), so callers who
#: reach for the "other" suffix (a natural guess, e.g. ``d_optimality``) hit an
#: "unknown metric" error. Each alias resolves to its canonical registry key
#: *before* validation, so both spellings work and the returned dict still keys
#: the result under the canonical name. Aliases are deliberately kept out of
#: ``_METRIC_REGISTRY`` so ``metric="all"`` does not compute anything twice.
#: Metrics kept for compatibility but left out of ``metric="all"``; asking for one warns.
#: ``i_efficiency`` divided ``p / N`` by the average prediction variance, which, unlike
#: D- and G-efficiency, has no upper bound of 100 (181% for a design in a small region).
_DEPRECATED_METRICS: dict[str, str] = {
    "i_efficiency": (
        "metric 'i_efficiency' is deprecated since 1.97.0 and will be removed in 2.0; use "
        "'average_prediction_variance' (the I-criterion, lower is better). The percentage was not bounded by 100."
    ),
}

_METRIC_ALIASES: dict[str, str] = {
    "d_optimality": "d_efficiency",
    "i_optimality": "average_prediction_variance",
    "i_criterion": "average_prediction_variance",
    "g_optimality": "g_efficiency",
    "a_efficiency": "a_optimality",
    "e_efficiency": "e_optimality",
}


def _resolve_metrics(metric: str | list[str]) -> list[str]:
    """Return the canonical metric names asked for; warn for a deprecated one, raise for an unknown one."""
    if metric == "all":
        return [m for m in _METRIC_REGISTRY if m not in _DEPRECATED_METRICS]
    # Resolve accepted spelling variants to their canonical registry keys.
    metrics = [_METRIC_ALIASES.get(m, m) for m in ([metric] if isinstance(metric, str) else metric)]
    unknown = [m for m in metrics if m not in _METRIC_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown metric(s): {unknown}. Available metrics: {sorted(_METRIC_REGISTRY)}")
    for m in metrics:
        if m in _DEPRECATED_METRICS:
            warnings.warn(_DEPRECATED_METRICS[m], category=DeprecationWarning, stacklevel=3)
    return metrics


#: Coded settings beyond this many units are taken as a sign of actual (uncoded) units. A
#: rotatable or orthogonal central composite design stays inside it for any usual size.
_CODED_LIMIT_FLOOR = 3.0


def _validate_inputs(
    design_df: pd.DataFrame,
    factor_names: list[str],
    alpha: float,
    sigma: float | None,
    n_samples: int,
) -> None:
    """Reject inputs that would give NaN or silently wrong metrics; warn on uncoded settings."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie strictly between 0 and 1; got alpha={alpha!r}.")
    if sigma is not None and not sigma > 0:
        raise ValueError(f"sigma must be positive; got sigma={sigma!r}.")
    if n_samples < 1:
        raise ValueError(f"n_samples must be at least 1; got n_samples={n_samples!r}.")

    missing = design_df[factor_names].isna()
    if missing.any().any():
        rows = list(design_df.index[missing.any(axis=1)])
        cols = [c for c in factor_names if missing[c].any()]
        raise ValueError(
            f"The design has missing factor settings in rows {rows} (columns {cols}). "
            "Fill them in or drop those runs before evaluating the design."
        )

    numeric = [f for f in factor_names if pd.api.types.is_numeric_dtype(design_df[f])]
    if numeric:
        limit = max(_CODED_LIMIT_FLOOR, 2 ** (len(numeric) / 4), np.sqrt(len(numeric)))
        largest = design_df[numeric].abs().max()
        uncoded = [f for f in numeric if largest[f] > limit]
        if uncoded:
            warnings.warn(
                f"evaluate_design expects factor settings in coded units (-1 and +1 at the low and high "
                f"levels), but columns {uncoded} reach |x| = {float(largest[uncoded].max()):g}. In actual "
                "units the efficiencies and the region-based metrics are on the wrong scale; pass the "
                "DesignResult, or code the columns first.",
                category=UserWarning,
                stacklevel=3,
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_design(  # noqa: PLR0913
    design_matrix: pd.DataFrame | DesignResult,
    model: str | None = None,
    metric: str | list[str] = "d_efficiency",
    effect_size: float | None = None,
    alpha: float = 0.05,
    sigma: float | None = None,
    region: str | DesignRegion | None = None,
    n_samples: int = 100_000,
    include_vertices: bool = True,
    random_seed: int | None = None,
    fds_resolution: int | None = None,
    random_state: int | np.random.Generator | None = 42,
) -> dict[str, Any]:
    """Compute quality metrics for an experimental design.

    Parameters
    ----------
    design_matrix : DataFrame or DesignResult
        The design to evaluate.  If a :class:`DesignResult` is passed, the
        coded design matrix and any generator / defining-relation metadata
        are extracted automatically.  A DataFrame must be in coded units (-1 and
        +1 at the low and high levels; proportions for mixture components), since
        the efficiencies and the region are defined on that scale; settings far
        outside it raise a ``UserWarning``.  Categorical factors are label columns.
        A run with a missing factor setting raises ``ValueError``.
    model : str or None
        Model type: ``"main_effects"``, ``"interactions"``, ``"quadratic"``, a
        Scheffé mixture model (``"scheffe_linear"``, ``"scheffe_quadratic"``,
        ``"scheffe_special_cubic"``), or an explicit patsy formula.  ``None``
        defaults to ``"scheffe_quadratic"`` over a mixture region and to
        ``"interactions"`` otherwise.
    metric : str or list[str]
        One or more metric names to compute, or the special value ``"all"`` to
        compute every metric.  Valid names: ``"d_efficiency"``,
        ``"average_prediction_variance"`` (the I-criterion), ``"g_efficiency"``, ``"a_optimality"``,
        ``"e_optimality"``, ``"correlation"``, ``"alias_matrix"``, ``"fds"``,
        ``"prediction_variance"``, ``"vif"``, ``"condition_number"``,
        ``"power"``, ``"degrees_of_freedom"``, ``"alias_structure"``,
        ``"confounding"``, ``"resolution"``, ``"defining_relation"``,
        ``"clear_effects"``, ``"minimum_aberration"``, ``"moment_aberration"``.
        ``"i_efficiency"`` still works but is deprecated (since 1.97.0, removed in
        2.0): it divides ``p / N`` by the average prediction variance and so is not
        bounded by 100. The optimality-criterion
        metrics also accept the opposite suffix as an alias (e.g.
        ``"d_optimality"`` for ``"d_efficiency"``, ``"a_efficiency"`` for
        ``"a_optimality"``); the result is keyed under the canonical name.
    effect_size : float or None
        Expected effect size for power calculation.  When *None*, a power
        curve over a range of effect sizes is returned instead.
    alpha : float
        Significance level for power calculation (default 0.05).
    sigma : float or None
        Estimated noise standard deviation.  Defaults to 1.0 when needed
        but not provided.
    region : {"cuboidal", "spherical"}, DesignRegion, or None
        Design region over which the region-based metrics (``average_prediction_variance``,
        ``g_efficiency``, ``fds``) integrate the prediction variance.
        ``"cuboidal"`` is ``[-1, 1]^k``; ``"spherical"`` is the ball of radius
        ``sqrt(k)``. A :class:`~process_improve.experiments.DesignRegion`
        restricts the average and the maximum to settings that satisfy its
        constraints (a box) or lie on its constrained simplex (a mixture): a
        constrained design should not be judged on corners it may not visit.
        ``None`` (default) uses the region a :class:`DesignResult` recorded when
        it was generated with constraints or mixture factors, and the cube
        otherwise.
    n_samples : int
        Number of random samples drawn uniformly over the region (default
        100,000; at least 1).  The region average and the FDS curve are taken
        over this sample.
    include_vertices : bool
        When *True* (default), boundary points are added for the maximum, so the
        worst-case (G) value is represented: for the cuboidal or spherical region
        the ``2**k`` cube vertices of the quantitative factors, crossed with every
        combination of categorical levels; for a :class:`DesignRegion`, its
        support points (extreme vertices and edge midpoints for a mixture).  They
        are used only for the maximum, not for the region average.
    random_seed : int or None
        Deprecated since 1.97.0 and removed in 2.0; use ``random_state``.
    fds_resolution : int or None
        Resolution of the dense FDS curve.  When *None* (default) the ``fds``
        metric returns only the coarse 11-point ``quantiles`` summary.  When set
        (e.g. 200), a ``curve`` sub-dict with length-``fds_resolution``
        ``fraction`` / ``prediction_variance`` / ``scaled_prediction_variance``
        arrays is added for smooth plotting; the endpoints are the minimum and
        maximum prediction variance.
    random_state : int, numpy.random.Generator or None
        Seed for the region sampler (default 42, so repeated calls agree).

    Returns
    -------
    dict[str, Any]
        Results keyed by metric name.  The structure of each value depends
        on the metric - see individual metric documentation.  A metric that
        needs to explain its result (for example a rank-deficient model) adds
        its note to ``result["notes"][metric_name]``.

    Raises
    ------
    ValueError
        If *alpha* is not in (0, 1), *sigma* is not positive, *n_samples* is
        below 1, or a factor setting is missing.

    Examples
    --------
    >>> from process_improve.experiments import evaluate_design, generate_design, Factor
    >>> factors = [Factor(name="A", low=0, high=10), Factor(name="B", low=0, high=10)]
    >>> result = generate_design(factors, design_type="full_factorial", n_center_points=0)
    >>> metrics = evaluate_design(result, model="main_effects", metric="d_efficiency")
    >>> metrics["d_efficiency"]  # doctest: +SKIP
    100.0
    """
    # --- Unpack input ---
    generators: list[str] | None = None
    defining_relation: list[str] | None = None
    resolution: int | None = None
    design_type: str | None = None

    if isinstance(design_matrix, DesignResult):
        design_type = design_matrix.design_type
        generators = design_matrix.generators
        defining_relation = design_matrix.defining_relation
        resolution = design_matrix.resolution
        factor_names = list(design_matrix.factor_names)
        design_df = pd.DataFrame(design_matrix.design)
    else:
        design_df = pd.DataFrame(design_matrix)
        factor_names = list(design_df.columns)

    # Drop non-factor columns
    for col in ["RunOrder", "Block"]:
        if col in design_df.columns and col not in factor_names:
            design_df = design_df.drop(columns=[col])
        elif col in factor_names:
            factor_names.remove(col)
            design_df = design_df.drop(columns=[col])

    _validate_inputs(design_df, factor_names, alpha, sigma, n_samples)
    metrics = _resolve_metrics(metric)
    logger.debug("evaluate_design: model=%r, metrics=%s", model, metrics)

    # --- Region, and the Scheffé default for mixtures (an intercept is redundant there) ---
    region = _resolve_region(region, design_matrix)
    if model is None and isinstance(region, DesignRegion) and region.kind == "mixture":
        model = "scheffe_quadratic"

    # --- Build context ---
    ctx = _build_context(
        _EvalRequest(
            design_df=design_df,
            factor_names=factor_names,
            model=model,
            generators=generators,
            defining_relation=defining_relation,
            resolution=resolution,
            effect_size=effect_size,
            alpha=alpha,
            sigma=sigma,
            region=region,
            n_samples=n_samples,
            include_vertices=include_vertices,
            random_state=resolve_deprecated_seed(random_state, random_seed, "evaluate_design"),
            fds_resolution=fds_resolution,
            design_type=design_type,
        )
    )

    # --- Compute requested metrics; each metric's note is kept under its own name ---
    results: dict[str, Any] = {}
    notes: dict[str, str] = {}
    for m in metrics:
        result = dict(_METRIC_REGISTRY[m](ctx))
        note = result.pop("note", None)
        if note:
            notes[m] = note
        results.update(result)
    if notes:
        results["notes"] = notes

    return results


def evaluate_all(  # noqa: PLR0913
    design_matrix: pd.DataFrame | DesignResult,
    model: str | None = None,
    effect_size: float | None = None,
    alpha: float = 0.05,
    sigma: float | None = None,
    region: str | DesignRegion | None = None,
    n_samples: int = 100_000,
    include_vertices: bool = True,
    random_seed: int | None = None,
    fds_resolution: int | None = None,
    random_state: int | np.random.Generator | None = 42,
) -> dict[str, Any]:
    """Compute *every* available metric for a design in one call.

    Thin convenience wrapper around :func:`evaluate_design` with
    ``metric="all"`` so callers need not enumerate the metric list.  All
    parameters have the same meaning as in :func:`evaluate_design`.

    Returns
    -------
    dict[str, Any]
        Results keyed by metric name (the union of every registered metric).

    See Also
    --------
    evaluate_design : Compute one or more named metrics.
    """
    return evaluate_design(
        design_matrix,
        model=model,
        metric="all",
        effect_size=effect_size,
        alpha=alpha,
        sigma=sigma,
        region=region,
        n_samples=n_samples,
        include_vertices=include_vertices,
        fds_resolution=fds_resolution,
        random_state=resolve_deprecated_seed(random_state, random_seed, "evaluate_all"),
    )
