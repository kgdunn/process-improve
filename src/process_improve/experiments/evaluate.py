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
import re
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from patsy import EvalFactor, ModelDesc, Sum, Term, Treatment, build_design_matrices, dmatrix
from patsy import categorical as patsy_categorical
from patsy.design_info import DesignInfo
from scipy import stats

from process_improve._random import check_random_state
from process_improve.experiments._analyses._shared import WHOLE_PLOT_COL
from process_improve.experiments._moment_aberration import NotTwoLevelError, moment_aberration
from process_improve.experiments.designs_mixture_constrained import SCHEFFE_MODELS, scheffe_formula_rhs, scheffe_model
from process_improve.experiments.designs_mixture_process import ReducedRank, category_levels, mixture_process_model
from process_improve.experiments.designs_three_level import MODULUS
from process_improve.experiments.designs_utils import RESERVED_COLUMN_NAMES
from process_improve.experiments.factor import DesignResult, FactorType
from process_improve.experiments.models import validate_formula_is_safe, validate_identifier_is_safe
from process_improve.experiments.region import DesignRegion
from process_improve.experiments.true_process import _canonical

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
    region: str | DesignRegion | pd.DataFrame = "cuboidal"
    n_samples: int = 100_000
    include_vertices: bool = True
    random_state: int | np.random.Generator | None = 42
    fds_resolution: int | None = None
    design_type: str | None = None
    categorical_coding: str = "effect"
    integration: str = "sample"
    max_search: str = "sample"
    # each random effect's group labels, one array per run, and its variance over the run variance (GLS)
    random_effects: dict[str, tuple[np.ndarray, float]] | None = None
    alias_terms: str = "interactions"  # the terms outside the model the alias matrix is worked against


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
    rank: int  # of X, decided with each column scaled to unit length
    generators: list[str] | None
    defining_relation: list[str] | None
    resolution: int | None
    effect_size: float | None
    alpha: float
    sigma: float | None
    region: str | DesignRegion | pd.DataFrame = "cuboidal"  # a DataFrame: the candidate points themselves
    n_samples: int = 100_000
    include_vertices: bool = True
    random_state: int | np.random.Generator | None = 42
    fds_resolution: int | None = None
    design_type: str | None = None
    is_scheffe: bool = False  # a Scheffé mixture model (no intercept; linear blending terms)
    blending_terms: frozenset[str] = frozenset()  # the linear blending terms of a mixture or mixture-process model
    reduced_rank: bool = False  # every categorical in L - 1 columns: a mixture-process model has no intercept
    categorical_coding: str = "effect"
    integration: str = "sample"  # the region average: "sample" or "exact" (quadrature over the cube)
    max_search: str = "sample"  # the region maximum: "sample" or "optimize" (a local search from the best points)
    alias_terms: str = "interactions"  # "interactions", or "quadratic" for the squares too
    random_effects: dict[str, tuple[np.ndarray, float]] | None = None  # each effect's group labels and variance ratio
    # the model columns kept when the region's rules leave some inestimable (None keeps all), and those lost
    kept_columns: np.ndarray | None = None
    lost_columns: list[str] | None = None

    def information(self, X: np.ndarray) -> np.ndarray:
        """Return ``X'X``, or ``X' V^-1 X`` when the runs share random effects (:func:`_random_effects_information`)."""
        return X.T @ X if not self.random_effects else _random_effects_information(X, self.random_effects)

    @property
    def patsy_data(self) -> dict[str, Any] | pd.DataFrame:
        """The design as patsy reads it: each categorical column carries its contrast."""
        return _patsy_data(self.design_df, self.categorical_coding, self.reduced_rank)

    def model_rows(self, points: pd.DataFrame) -> np.ndarray:
        """Expand factor-space points (one column per factor) into the model's columns, as ``X`` holds them."""
        (expanded,) = build_design_matrices([self.design_info], points[self.factor_names], return_type="matrix")
        rows = np.asarray(expanded, dtype=float)
        return rows if self.kept_columns is None else rows[:, self.kept_columns]


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


#: Codings for a categorical (label) factor; see ``categorical_coding`` in :func:`evaluate_design`.
CATEGORICAL_CODINGS = ("effect", "treatment")


def _patsy_data(
    design_df: pd.DataFrame, categorical_coding: str, reduced_rank: bool = False
) -> dict[str, Any] | pd.DataFrame:
    """Return the data handed to patsy, each categorical (label) column carrying its contrast.

    With ``"effect"`` coding a categorical column gets patsy's ``Sum`` contrast: a
    two-level factor is coded -1 for its first level (in sorted order) and +1 for its
    second (``Sum(omit=0)``), and a factor with more levels has the last level at -1 in
    every column. A ``Block`` column gets plain ``Sum``, as ``analyze_experiment`` codes
    blocks. The contrast travels with the data, not the formula, so the column names
    read ``C[S.y]`` and ``A:C[S.y]``, and a formula that names a contrast itself
    (``C(cat, Treatment)``) overrides it. ``"treatment"`` returns the data unchanged, so
    patsy's default treatment (0/1) coding applies.

    With ``reduced_rank`` (a mixture-process model, which has no intercept) every contrast
    is wrapped in :class:`~process_improve.experiments.designs_mixture_process.ReducedRank`,
    so a categorical factor takes ``L - 1`` columns in every term it appears in.
    """
    if categorical_coding not in CATEGORICAL_CODINGS:
        raise ValueError(
            f"categorical_coding must be one of {', '.join(map(repr, CATEGORICAL_CODINGS))}; "
            f"got {categorical_coding!r}."
        )
    labels = [str(name) for name, column in design_df.items() if not pd.api.types.is_numeric_dtype(column)]
    if not labels or (categorical_coding == "treatment" and not reduced_rank):
        return design_df
    data: dict[str, Any] = {str(name): column for name, column in design_df.items()}
    for name in labels:
        column = design_df[name]
        n_levels = len(column.cat.categories) if isinstance(column.dtype, pd.CategoricalDtype) else column.nunique()
        contrast: Any
        if categorical_coding == "treatment":
            contrast = Treatment()
        else:
            contrast = Sum(omit=0) if n_levels == 2 and name != "Block" else Sum
        if reduced_rank:
            contrast = ReducedRank(contrast() if contrast is Sum else contrast)
        data[name] = patsy_categorical.C(column, contrast)
    return data


def _build_model_matrix(
    design_df: pd.DataFrame,
    model: str | None,
    factor_names: list[str],
    categorical_coding: str = "effect",
    reduced_rank: bool = False,
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
    categorical_coding : {"effect", "treatment"}
        How a categorical (label) column is coded; see :func:`_patsy_data`.
    reduced_rank : bool
        Code every categorical factor in ``L - 1`` columns, as a model without an
        intercept needs; see :func:`_patsy_data`.

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
    # contrast-codes it (see _patsy_data), so it must NOT be manually expanded into
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
    dm = dmatrix(rhs, _patsy_data(design_df, categorical_coding, reduced_rank), return_type="dataframe")
    X = np.asarray(dm, dtype=float)
    column_names = list(dm.columns)
    return X, column_names, dm.design_info


def _unit_length_columns(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``X`` with each column scaled to unit length, and the lengths; a column of zeros stays as it is."""
    lengths = np.linalg.norm(X, axis=0)
    lengths = np.where(lengths > 0, lengths, 1.0)
    return X / lengths, lengths


def _random_effects_information(X: np.ndarray, random_effects: dict[str, tuple[np.ndarray, float]]) -> np.ndarray:
    """Return the generalized least squares information ``X' V^-1 X`` of runs that share random effects.

    ``V = I + sum_k eta_k Z_k Z_k'`` is the runs' covariance in units of the run variance, with
    ``Z_k`` the indicators of the groups of random effect ``k`` (the whole plots of a split-plot
    design, the rows and the columns of a two-way split-plot design) and ``eta_k`` its variance
    over the run variance. With ``Z = [Z_1 ... Z_K]`` and ``D`` the diagonal of the ``eta_k``,
    Woodbury's identity gives ``V^-1 = I - Z (D^-1 + Z'Z)^-1 Z'``, so only a matrix the size of
    the number of groups is inverted. A random effect of ratio zero leaves ``V`` unchanged.
    """
    indicators, ratios = [], []
    for labels, ratio in random_effects.values():
        if ratio > 0:
            codes, _ = pd.factorize(labels)
            indicators.append(np.eye(int(codes.max()) + 1)[codes])
            ratios.append(np.full(indicators[-1].shape[1], 1.0 / ratio))
    if not indicators:
        return X.T @ X
    Z = np.hstack(indicators)
    sums = Z.T @ X
    return X.T @ X - sums.T @ np.linalg.solve(np.diag(np.concatenate(ratios)) + Z.T @ Z, sums)


def _blending_terms(req: _EvalRequest) -> frozenset[str]:
    """Return the model's linear blending terms: a Scheffé model's factors, a mixture-process model's components.

    The components sum to 1, so these terms are not tested against zero.
    """
    if req.model in SCHEFFE_MODELS:
        return frozenset(req.factor_names)
    if isinstance(req.region, DesignRegion) and req.region.kind == "mixture_process":
        return frozenset(f.name for f in req.region.factors if f.type == FactorType.mixture)
    return frozenset()


def _build_context(req: _EvalRequest) -> _EvalContext:
    """Build the shared evaluation context."""
    reduced_rank = isinstance(req.region, DesignRegion) and req.region.kind == "mixture_process"
    X, column_names, design_info = _build_model_matrix(
        req.design_df, req.model, req.factor_names, req.categorical_coding, reduced_rank
    )
    kept = _columns_the_rules_leave(req, X, design_info)
    lost = None
    if kept is not None:
        lost = [name for j, name in enumerate(column_names) if j not in set(kept.tolist())]
        X, column_names = X[:, kept], [column_names[j] for j in kept]
    N, p = X.shape

    def information(Xs: np.ndarray) -> np.ndarray:
        return Xs.T @ Xs if not req.random_effects else _random_effects_information(Xs, req.random_effects)

    XtX = information(X)

    # The rank and (X'X)^-1 are worked on X with unit-length columns, which changes neither. In
    # the factors' own units the columns' lengths can differ by many orders (x1 beside x1^2 x2);
    # X'X squares X's condition number, and a full-rank model could fall under the rank tolerance.
    scaled, lengths = _unit_length_columns(X)
    rank = int(np.linalg.matrix_rank(scaled))
    is_singular = rank < p
    XtX_inv: np.ndarray | None = None
    if not is_singular:
        XtX_inv = np.linalg.inv(information(scaled)) / np.outer(lengths, lengths)

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
        rank=rank,
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
        is_scheffe=req.model in SCHEFFE_MODELS,
        blending_terms=_blending_terms(req),
        reduced_rank=reduced_rank,
        categorical_coding=req.categorical_coding,
        integration=req.integration,
        max_search=req.max_search,
        alias_terms=req.alias_terms,
        random_effects=req.random_effects,
        kept_columns=kept,
        lost_columns=lost,
    )


#: Points of a region with rules on which the model's estimable columns are judged.
_RULES_SAMPLE = 4000


def _columns_the_rules_leave(req: _EvalRequest, X: np.ndarray, design_info: DesignInfo) -> np.ndarray | None:
    """Return the model columns the region's rules leave estimable, or None when every column is.

    A combination of levels the rules do not allow is a missing cell, which costs an
    interaction of the factors it names some of its columns over the whole region, not
    only in this design. The columns are judged on a sample of the region and kept in model
    order while each adds rank, as :func:`generate_design` does; a design that cannot
    estimate even those is still reported rank-deficient.
    """
    from process_improve.experiments.designs_constrained import independent_columns  # noqa: PLC0415

    region = req.region
    if not isinstance(region, DesignRegion) or region.rules is None:
        return None
    if any(name not in req.design_df.columns for name in [f.name for f in region.factors]):
        return None
    sample = region.sample_frame(_RULES_SAMPLE, np.random.default_rng(0))
    seen = np.ones(len(sample), dtype=bool)
    for f in region.factors:
        if f.type == FactorType.categorical:
            seen &= sample[f.name].isin(req.design_df[f.name].unique()).to_numpy()
    (rows,) = build_design_matrices([design_info], sample[seen][req.factor_names], return_type="matrix")
    kept = independent_columns(np.asarray(rows, dtype=float))
    return None if len(kept) == X.shape[1] else kept


# ---------------------------------------------------------------------------
# Metric implementations
# ---------------------------------------------------------------------------


def _compute_d_efficiency(ctx: _EvalContext) -> dict[str, Any]:
    """D-efficiency: 100 * det(X'X)^(1/p) / N."""
    if ctx.is_singular:
        return {"d_efficiency": None, "note": "Design is rank-deficient for the specified model."}
    sign, logdet = np.linalg.slogdet(ctx.XtX)
    # X'X of full rank is positive definite; this only guards an ill-conditioned matrix
    # that clears matrix_rank's tolerance yet gets a non-positive sign from slogdet.
    if sign <= 0:  # pragma: no cover - belt and braces after the rank check
        return {"d_efficiency": None, "note": "X'X has non-positive determinant."}
    d_eff = 100.0 * np.exp(logdet / ctx.p) / ctx.N
    return {"d_efficiency": float(d_eff)}


def _prediction_variance_at_points(X_points: np.ndarray, XtX_inv: np.ndarray) -> np.ndarray:
    """Compute d(x) = x' (X'X)^-1 x for each row of X_points."""
    # (X_points @ XtX_inv) element-wise * X_points, summed per row
    return np.sum((X_points @ XtX_inv) * X_points, axis=1)


def _region_points(
    factor_names: list[str],
    region: str,
    n_samples: int,
    random_state: int | np.random.Generator | None,
) -> np.ndarray:
    """Sample raw factor-space points uniformly over the design region.

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
    random_state : int, numpy.random.Generator or None
        Seed for the NumPy random generator (full reproducibility).

    Returns
    -------
    ndarray of shape (n_samples, k)
        The sampled points. The corners, where the worst-case prediction variance
        usually sits, are added by the caller, crossed with the categorical levels.
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
    return pts


#: The ways of working out the region average and the region maximum of the prediction variance.
_INTEGRATIONS = ("sample", "exact")
_MAX_SEARCHES = ("sample", "optimize")
_ALIAS_TERMS = ("interactions", "quadratic")
#: The values each of evaluate_design's named options takes.
_OPTIONS = {"integration": _INTEGRATIONS, "max_search": _MAX_SEARCHES, "alias_terms": _ALIAS_TERMS}


def _check_options(**options: str) -> None:
    """Refuse an option of :func:`evaluate_design` that is not one of its values, naming the option."""
    for name, value in options.items():
        if value not in _OPTIONS[name]:
            raise ValueError(f"{name} must be one of {_OPTIONS[name]}; got {name}={value!r}.")


#: Gauss-Legendre points per factor for the exact region average: exact for a model of degree up to
#: four in each factor, whose prediction variance is then a polynomial of degree up to eight.
_QUADRATURE_NODES = 5
#: The largest quadrature grid worked: five points per quantitative factor, times the categorical levels.
_QUADRATURE_MOST_POINTS = 250_000
#: Starting points of the compass search for the region maximum, and the step it stops at.
_SEARCH_STARTS = 8
_SEARCH_TOLERANCE = 1e-6


def _categorical_levels(ctx: _EvalContext) -> dict[str, list[Any]]:
    """Return each categorical (label) factor's levels in the design."""
    return {
        f: list(ctx.design_df[f].unique())
        for f in ctx.factor_names
        if not pd.api.types.is_numeric_dtype(ctx.design_df[f])
    }


def _cube_moment_matrix(ctx: _EvalContext) -> np.ndarray | None:
    """Return the cube's moment matrix ``W``, the average of ``f(x) f(x)'`` over ``[-1, 1]^k``.

    Worked by Gauss-Legendre quadrature on the tensor grid of the quantitative factors, crossed
    with every combination of categorical levels (each combination weighted equally). ``None``
    when that grid has more than ``_QUADRATURE_MOST_POINTS`` points.
    """
    categorical = _categorical_levels(ctx)
    continuous = [f for f in ctx.factor_names if f not in categorical]
    n_points = _QUADRATURE_NODES ** len(continuous) * int(np.prod([len(v) for v in categorical.values()] or [1]))
    if n_points > _QUADRATURE_MOST_POINTS:
        return None
    nodes, weights = np.polynomial.legendre.leggauss(_QUADRATURE_NODES)
    weights = weights / 2.0  # average over [-1, 1] rather than integrate
    axes = [nodes] * len(continuous) + [np.arange(len(v)) for v in categorical.values()]
    grid = np.array(list(itertools.product(*axes)), dtype=float).reshape(-1, len(axes))
    weight = np.prod([weights[np.searchsorted(nodes, grid[:, j])] for j in range(len(continuous))], axis=0)
    weight = np.broadcast_to(weight, (len(grid),)) / np.prod([len(v) for v in categorical.values()] or [1])
    frame = {f: grid[:, j] for j, f in enumerate(continuous)}
    for j, (f, levels) in enumerate(categorical.items(), start=len(continuous)):
        frame[f] = np.asarray(levels, dtype=object)[grid[:, j].astype(int)]
    expanded = ctx.model_rows(pd.DataFrame(frame))
    return (expanded * weight[:, None]).T @ expanded


def _region_average(ctx: _EvalContext) -> tuple[float, str | None]:
    """Return the prediction variance averaged over the region (the I-criterion), and a note when exactness fell back.

    With ``integration="exact"`` over the cuboidal region it is ``trace((X'X)^-1 W)``, ``W`` the
    cube's moment matrix; otherwise, and when that is not possible, the mean over the sample.
    """
    assert ctx.XtX_inv is not None
    note = None
    if ctx.integration == "exact" and not isinstance(ctx.region, pd.DataFrame):
        moments = _cube_moment_matrix(ctx) if _is_cube(ctx.region) else None
        if moments is not None:
            return float(np.trace(ctx.XtX_inv @ moments)), None
        note = "The exact average needs the cuboidal region and a quadrature grid of at most 250,000 points: sampled."
    interior, _boundary = _region_prediction_variance(ctx)
    return float(np.mean(interior)), note


def _region_maximum(ctx: _EvalContext) -> float:
    """Return the largest prediction variance over the region: over the sample and boundary points, then searched.

    With ``max_search="optimize"`` over the cuboidal region, a compass search climbs from the
    best ``_SEARCH_STARTS`` points (each keeping its categorical levels), stepping each
    quantitative factor up and down inside ``[-1, 1]`` and halving the step when no move helps.
    """
    interior, boundary = _region_prediction_variance(ctx)
    found = float(np.max(np.concatenate([interior, boundary])))
    categorical = _categorical_levels(ctx)
    continuous = [f for f in ctx.factor_names if f not in categorical]
    if ctx.max_search != "optimize" or not _is_cube(ctx.region) or not continuous:
        return found
    interior_df, boundary_df = _points_in_box_region(ctx, "cuboidal")
    points = pd.concat([interior_df, boundary_df], ignore_index=True)
    values = np.concatenate([interior, boundary])
    starts = points.iloc[np.argsort(values)[::-1][:_SEARCH_STARTS]].reset_index(drop=True)
    return max(found, _compass_search(ctx, starts, continuous))


def _compass_search(ctx: _EvalContext, starts: pd.DataFrame, continuous: list[str]) -> float:
    """Climb the prediction variance from each starting point by a compass search inside the cube."""
    current = starts.copy()
    best = _prediction_variance_of_frame(ctx, current)
    step = 0.5
    k = len(continuous)
    while step > _SEARCH_TOLERANCE:
        # every point's 2k neighbours: one quantitative factor moved up or down by the step
        moves = np.vstack([np.eye(k), -np.eye(k)]) * step
        neighbours = current.loc[current.index.repeat(2 * k)].reset_index(drop=True)
        shift = np.tile(moves, (len(current), 1))
        neighbours[continuous] = np.clip(neighbours[continuous].to_numpy(dtype=float) + shift, -1.0, 1.0)
        values = _prediction_variance_of_frame(ctx, neighbours).reshape(len(current), 2 * k)
        choice = values.argmax(axis=1)
        better = values[np.arange(len(current)), choice] > best + 1e-12
        if not better.any():
            step /= 2.0
            continue
        rows = np.flatnonzero(better)
        current.loc[rows, continuous] = neighbours.loc[rows * 2 * k + choice[rows], continuous].to_numpy(dtype=float)
        best[rows] = values[rows, choice[rows]]
    return float(best.max())


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
    if isinstance(ctx.region, pd.DataFrame):  # a candidate set: every point, each once
        return _prediction_variance_of_frame(ctx, ctx.region), np.empty(0)
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
    return _prediction_variance_at_points(ctx.model_rows(points), ctx.XtX_inv)


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
    points = _region_points(cont_names, region, ctx.n_samples, random_state=rng)
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
    Categorical factors are sampled uniformly over their observed levels. With rules on
    categorical factors, the factors they name take only allowed combinations, each
    equally likely, with the settings each combination allows (see
    :meth:`DesignRegion.sample_frame`).
    """
    missing = [n for n in region.names if n not in ctx.factor_names]
    if missing:
        raise ValueError(f"The region names factors {missing} that are not columns of the design.")
    rng = check_random_state(ctx.random_state)
    frames: list[pd.DataFrame] = []
    if region.rules is None:
        frames.append(pd.DataFrame(region.sample(ctx.n_samples, rng), columns=region.names))
        if ctx.include_vertices:
            with contextlib.suppress(ValueError):  # a grid too large for support points: sample only
                frames.append(pd.DataFrame(region.support_points(), columns=region.names))
    else:
        named = region.names + region.rules.factors
        frames.append(region.sample_frame(ctx.n_samples, rng)[named])
        if ctx.include_vertices:
            with contextlib.suppress(ValueError):  # too many support points: sample only
                frames.append(region.support_frame())
    out: list[pd.DataFrame] = []
    for frame in frames:
        data: dict[str, Any] = {n: frame[n].to_numpy() for n in frame.columns}
        for f in ctx.factor_names:
            if f not in data:
                data[f] = rng.choice(ctx.design_df[f].unique(), size=len(frame))
        points = pd.DataFrame(data)
        # A level the design never ran has no column in the model's coding; its points cannot be predicted.
        seen = np.ones(len(points), dtype=bool)
        for f in region.rules.factors if region.rules is not None else []:
            if f in ctx.design_df.columns:
                seen &= points[f].isin(ctx.design_df[f].unique()).to_numpy()
        out.append(points[seen].reset_index(drop=True))
    return out[0], out[1] if len(out) > 1 else pd.DataFrame()


def _region_label(region: str | DesignRegion | pd.DataFrame) -> str:
    """Name the region in results: the string as given, its kind, ``"constrained"`` for a box, or ``"candidates"``."""
    if isinstance(region, str):
        return region
    if isinstance(region, pd.DataFrame):
        return "candidates"
    return region.kind if region.kind != "box" else "constrained"


def _is_cube(region: str | DesignRegion | pd.DataFrame) -> bool:
    """Say whether the region is the cube ``[-1, 1]^k``, where the exact average and the search apply."""
    return isinstance(region, str) and region == "cuboidal"


def _candidate_points(points: pd.DataFrame, factor_names: list[str]) -> pd.DataFrame:
    """Return the candidate points as a region: a row per point, the design's factors as columns."""
    missing = [f for f in factor_names if f not in points.columns]
    if points.empty or missing:
        found = "no points" if points.empty else f"no column for {missing}"
        raise ValueError(f"A region of candidate points needs a row per point and a column per factor; it has {found}.")
    chosen = points[factor_names].reset_index(drop=True)
    if chosen.isna().to_numpy().any():
        raise ValueError("The candidate points have missing settings; give every factor's setting at every point.")
    return chosen


def _resolve_region(
    region: str | DesignRegion | pd.DataFrame | None,
    design_matrix: pd.DataFrame | DesignResult,
    factor_names: list[str],
) -> str | DesignRegion | pd.DataFrame:
    """Use the caller's region (candidate points checked against the factors), else the one recorded, else the cube."""
    if isinstance(region, pd.DataFrame):
        return _candidate_points(region, factor_names)
    if region is not None:
        return region
    spec = design_matrix.metadata.get("region") if isinstance(design_matrix, DesignResult) else None
    if spec:
        recorded = DesignRegion.from_dict(spec)
        if recorded.kind != "box" or recorded.is_constrained:
            return recorded
    return "cuboidal"


def _mixture_model(model: str | None, design_matrix: pd.DataFrame | DesignResult) -> str:
    """Name the Scheffé model to evaluate a mixture design with.

    The process-model names that ``generate_design`` accepts for a mixture
    (``"quadratic"``, ``"special_cubic"``, ...) map to their Scheffé models, as they do
    there; an explicit formula is kept. Without a model, the one the design was
    generated for is used, else the Scheffé quadratic model.
    """
    if model is None:
        recorded = design_matrix.metadata.get("model_type") if isinstance(design_matrix, DesignResult) else None
        model = recorded or "scheffe_quadratic"
    with contextlib.suppress(ValueError):  # not a model name: an explicit formula
        return scheffe_model(model)
    return model


def _mixture_process_formula(
    model: str | None, design_matrix: pd.DataFrame | DesignResult, region: DesignRegion
) -> str:
    """Return the formula to evaluate a mixture-process design with.

    A mixture-process model name (``"interactions"``, ``"scheffe_quadratic + main_effects"``,
    ...) becomes its formula, as :func:`generate_design` built it; an explicit formula is
    kept. Without a model, the one the design was generated for is used, else the
    ``"interactions"`` model.
    """
    if model is None:
        recorded = design_matrix.metadata.get("model_type") if isinstance(design_matrix, DesignResult) else None
        model = recorded or "interactions"
    mixture = [f.name for f in region.factors if f.type == FactorType.mixture]
    process = [f.name for f in region.factors if f.type != FactorType.mixture]
    try:
        return mixture_process_model(mixture, process, model, category_levels(region.factors)).formula_rhs()
    except ValueError:  # not a model name: an explicit formula
        return model


def _compute_g_efficiency(ctx: _EvalContext) -> dict[str, Any]:
    """G-efficiency: ``100 * p / (N * max d(x))``, the maximum taken over the design region.

    This is the definition of Atkinson, Donev and Tobias and of Montgomery: ``p`` over the
    maximum scaled prediction variance ``N d(x)``. JMP and SAS PROC OPTEX report its
    square root, ``100 * sqrt(p / (N * max d(x)))``, so below 100 the two disagree (82.8
    against 91.0 for a 3^2 factorial under the quadratic model); ``max_prediction_variance``
    is returned so either can be formed.
    """
    if ctx.is_singular:
        return {"g_efficiency": None, "note": "Design is rank-deficient for the specified model."}

    max_pv = _region_maximum(ctx)

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
    average, note = _region_average(ctx)
    return {"average_prediction_variance": average, **({"note": note} if note else {})}


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


#: The largest natural log whose exponential is a finite float.
_LARGEST_LOG = 709.0


def _compute_information_matrix(ctx: _EvalContext) -> dict[str, Any]:
    """Return the information matrix ``M = X'X``, and the criteria read from it.

    Returns the model's ``terms``, ``xtx`` (rows and columns in the order of ``terms``),
    its ``determinant`` and ``log_determinant`` (the determinant of a large design
    overflows), the D-criterion ``d_criterion``, ``|X'X|^(1/p)``, the ``eigenvalues`` and
    the E-criterion ``e_criterion`` (the smallest of them), the ``rank`` of ``X`` beside
    ``n_parameters`` and ``n_runs``, the ``residual_df``, ``N - rank``, and ``is_diagonal``
    (every term orthogonal to every other). For a model the design estimates it adds the
    ``inverse``, whose entries are the coefficients' variances and covariances in units of
    sigma^2, its trace, the A-criterion ``a_criterion``, that trace over ``p``
    (``a_criterion_per_parameter``), each term's ``coefficient_variances``, and the
    correlations of the estimates, ``estimate_correlations`` (term by term, JMP's
    "Correlation of Estimates"), with the largest in size off the diagonal,
    ``max_abs_estimate_correlation``, and the number of pairs of terms whose estimates are
    correlated at all, ``n_correlated_pairs``.
    """
    sign, logdet = np.linalg.slogdet(ctx.XtX)
    estimable = sign > 0 and not ctx.is_singular
    eigenvalues = np.linalg.eigvalsh(ctx.XtX)
    off_diagonal = ctx.XtX - np.diag(np.diag(ctx.XtX))
    scale = max(float(np.abs(ctx.XtX).max()), 1.0)
    result: dict[str, Any] = {
        "terms": list(ctx.column_names),
        "xtx": ctx.XtX.tolist(),
        "determinant": float(np.exp(logdet)) if estimable and logdet < _LARGEST_LOG else (np.inf if estimable else 0.0),
        "log_determinant": float(logdet) if estimable else None,
        "d_criterion": float(np.exp(logdet / ctx.p)) if estimable else 0.0,
        "eigenvalues": eigenvalues.tolist(),
        "e_criterion": float(eigenvalues.min()),
        "rank": ctx.rank,
        "n_parameters": ctx.p,
        "n_runs": ctx.N,
        "residual_df": ctx.N - ctx.rank,
        "is_diagonal": bool(np.all(np.abs(off_diagonal) <= 1e-9 * scale)),
    }
    if not ctx.is_singular:
        assert ctx.XtX_inv is not None
        trace = float(np.trace(ctx.XtX_inv))
        result |= {
            "inverse": ctx.XtX_inv.tolist(),
            "a_criterion": trace,
            "a_criterion_per_parameter": trace / ctx.p,
            "coefficient_variances": dict(zip(ctx.column_names, map(float, np.diag(ctx.XtX_inv)), strict=True)),
        }
        result |= _estimate_correlations(ctx.XtX_inv, ctx.column_names)
    return {"information_matrix": result}


#: Correlations of the estimates smaller than this in size count as zero: rounding, not dependence.
_CORRELATION_TOLERANCE = 1e-8


def _estimate_correlations(inverse: np.ndarray, names: list[str]) -> dict[str, Any]:
    """Return the correlations of the estimates, ``(X'X)^-1`` scaled to a unit diagonal, and their summary."""
    scale = np.sqrt(np.diag(inverse))
    correlation = inverse / np.outer(scale, scale)
    upper = np.abs(correlation[np.triu_indices(len(names), 1)])
    return {
        "estimate_correlations": {
            a: dict(zip(names, map(float, row), strict=True)) for a, row in zip(names, correlation, strict=True)
        },
        "max_abs_estimate_correlation": float(upper.max()) if upper.size else 0.0,
        "n_correlated_pairs": int(np.count_nonzero(upper > _CORRELATION_TOLERANCE)),
    }


def _compute_estimable_interactions(ctx: _EvalContext) -> dict[str, Any]:
    """How many two-factor interactions outside the model the design can estimate beside it.

    The rank the omitted interaction columns add to the model matrix,
    ``rank([X, X2]) - rank(X)``: so many of them can be estimated together with the
    model, though not every choice of that many need be. Returns ``n_estimable``,
    ``n_omitted`` and the omitted interactions' ``terms``.
    """
    X2, terms = _omitted_two_factor_interactions(ctx)
    if not terms:
        return {"estimable_interactions": {"n_estimable": 0, "n_omitted": 0, "terms": []}}
    both, _ = _unit_length_columns(np.hstack([ctx.X, X2]))
    added = int(np.linalg.matrix_rank(both)) - ctx.rank
    return {"estimable_interactions": {"n_estimable": added, "n_omitted": len(terms), "terms": terms}}


def _compute_correlation(ctx: _EvalContext) -> dict[str, Any]:
    """Pairwise correlation summary among the model's second-order terms.

    The pure-quadratic columns ``x_i^2`` have a non-zero mean, so a raw Pearson
    correlation between them is inflated by that shared offset and depends on
    the coding.  To get a coding-invariant measure, each second-order column is
    first residualised against the intercept-and-main-effect block (the
    content those columns share by construction) and the correlations are taken
    on the residuals.

    Returns ``max_abs_r``, ``mean_abs_r``, the full correlation ``matrix`` (as
    nested lists), the ordered ``terms``, and ``by_pair``, the sizes by kind of pair (see
    :func:`_by_pair`): Jones and Nachtsheim's (2011) Table 3 for a definitive screening design
    under the full quadratic model.
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
                "by_pair": {},
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
            "by_pair": _by_pair(corr, [ctx.column_names[i] for i in second_idx]),
        }
    }


#: The kinds of pair of second-order terms, in Jones and Nachtsheim's (2011) notation: two squares
#: (qq, ss), a square and an interaction in its factor (qq, qs), a square and an interaction without
#: it (qq, st), and two interactions (st, uv), with a factor in common or not.
SECOND_ORDER_PAIRS = ("two_squares", "square_and_own_interaction", "square_and_other_interaction", "two_interactions")


def _second_order_factors(name: str) -> tuple[str, str] | None:
    """Return the factors a square (``(A, A)``) or a two-factor interaction (``(A, B)``) multiplies; else ``None``.

    The term is read as the simulation reads one (``true_process._canonical``), once the level of a
    categorical factor's column (``A:C[S.y]``) is set aside.
    """
    try:
        _, factors = _canonical(re.sub(r"\[[^\]]*\]", "", name))
    except ValueError:
        return None
    return (factors[0], factors[1]) if len(factors) == 2 else None


def _second_order_pair(first: tuple[str, str], second: tuple[str, str]) -> str:
    """Name the kind of a pair of second-order terms (see :data:`SECOND_ORDER_PAIRS`)."""
    squares = [a == b for a, b in (first, second)]
    if all(squares):
        return "two_squares"
    if not any(squares):
        return "two_interactions"
    square, interaction = (first, second) if squares[0] else (second, first)
    return "square_and_own_interaction" if square[0] in interaction else "square_and_other_interaction"


def _by_pair(corr: np.ndarray, terms: list[str]) -> dict[str, dict[str, float | int]]:
    """Summarise the sizes of correlations between second-order terms by kind of pair.

    For each kind of :data:`SECOND_ORDER_PAIRS` that the terms make, ``max_abs_r``, ``mean_abs_r``
    and ``n_pairs``, as Jones and Nachtsheim (2011) tabulate them for definitive screening designs
    (their Tables 3 and 4). Two columns of one term, the levels of a categorical factor in an
    interaction, are not a pair: their correlation comes from the coding, not from the design.
    """
    factors = [_second_order_factors(t) for t in terms]
    sizes: dict[str, list[float]] = {}
    for i, j in itertools.combinations(range(len(terms)), 2):
        first, second = factors[i], factors[j]
        if first is None or second is None or set(first) == set(second):
            continue
        sizes.setdefault(_second_order_pair(first, second), []).append(abs(float(corr[i, j])))
    return {
        kind: {"max_abs_r": max(sizes[kind]), "mean_abs_r": float(np.mean(sizes[kind])), "n_pairs": len(sizes[kind])}
        for kind in SECOND_ORDER_PAIRS
        if kind in sizes
    }


def _compute_estimate_correlation(ctx: _EvalContext) -> dict[str, Any]:
    """Correlations between the estimates of the model's second-order terms, from ``(X'X)^-1``.

    Where ``correlation`` compares the second-order *columns* of the design, this compares the
    *estimates* the fit makes of them: ``r_ij = c_ij / sqrt(c_ii c_jj)`` with ``c = (X'X)^-1``,
    the correlations JMP prints as the "correlation of estimates". A large ``|r_ij|`` means the
    data cannot tell the two effects apart: an active square can show up as an interaction, or
    the reverse. ``by_pair`` summarises them by kind of pair (see :func:`_by_pair`), as
    ``correlation`` does for the columns. A model of every main effect and every square gives
    Jones and Nachtsheim's (2011) Table 4 for two squares, and the full quadratic model of three of
    the factors gives its other columns.

    Returns ``terms``, ``matrix`` (the correlations, as nested lists) and ``by_pair``.
    """
    if ctx.is_singular:
        return {"estimate_correlation": None, "note": "Design is rank-deficient for the specified model."}
    assert ctx.XtX_inv is not None  # guaranteed by not is_singular
    second_idx = [i for i, c in enumerate(ctx.column_names) if _term_order(c) == 2]
    terms = [ctx.column_names[i] for i in second_idx]
    if len(second_idx) < 2:
        return {
            "estimate_correlation": {
                "terms": terms,
                "matrix": [[1.0]] if second_idx else [],
                "by_pair": {},
                "note": "Fewer than two second-order terms; no pairwise correlation.",
            }
        }
    inverse = ctx.XtX_inv[np.ix_(second_idx, second_idx)]
    scale = np.sqrt(np.diag(inverse))
    corr = inverse / np.outer(scale, scale)
    return {"estimate_correlation": {"terms": terms, "matrix": corr.tolist(), "by_pair": _by_pair(corr, terms)}}


def _omitted_two_factor_interactions(ctx: _EvalContext) -> tuple[np.ndarray, list[str]]:
    """Build the two-factor-interaction columns *not* already in the model.

    Returns the ``(N, q)`` matrix of the interaction columns and their names, for
    every factor pair whose interaction is absent from the fitted model. The columns
    are built by patsy next to the model's own terms, so a categorical factor is
    contrast-coded exactly as it is in ``X`` (``"A:C[S.y]"``), and a quantitative
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
    dm = dmatrix(ModelDesc([], [*ctx.design_info.terms, *omitted]), ctx.patsy_data, return_type="dataframe")
    slices = [dm.design_info.term_slices[term] for term in omitted]
    columns = [i for sl in slices for i in range(sl.start, sl.stop)]
    return np.asarray(dm, dtype=float)[:, columns], [dm.columns[i] for i in columns]


def _omitted_squares(ctx: _EvalContext) -> tuple[np.ndarray, list[str]]:
    """Build the squared columns of the numeric factors whose square the model leaves out (#797 R4).

    A first-order design run at two levels has every square equal to 1, so each square is
    aliased with the intercept: Davies (11.33) writes b0 -> b0 + b11 + b22 + ... This is what
    curvature does to the mean of such a design. A Scheffe mixture model has no squares.
    """
    if ctx.is_scheffe:
        return np.empty((ctx.N, 0)), []
    present = {factor.name() for term in ctx.design_info.terms for factor in term.factors}
    squares = [
        Term([EvalFactor(f"I({name} ** 2)")])
        for name in ctx.factor_names
        if name not in ctx.blending_terms  # a component's square is in its blending terms (x^2 = x - sum x x_j)
        and pd.api.types.is_numeric_dtype(ctx.design_df[name])
        and EvalFactor(f"I({name} ** 2)").name() not in present
    ]
    if not squares:
        return np.empty((ctx.N, 0)), []
    dm = dmatrix(ModelDesc([], [*ctx.design_info.terms, *squares]), ctx.patsy_data, return_type="dataframe")
    columns = [dm.design_info.term_slices[term].start for term in squares]
    return np.asarray(dm, dtype=float)[:, columns], [dm.columns[i] for i in columns]


def _compute_alias_matrix(ctx: _EvalContext) -> dict[str, Any]:
    """General alias (bias) matrix ``A = (X1'X1)^-1 X1' X2``.

    With the fitted model ``X1`` and a set of potential extra terms ``X2``
    (the two-factor interactions not already in the model, and with
    ``alias_terms="quadratic"`` the squares of the numeric factors too), the least
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
    if ctx.alias_terms == "quadratic":
        squares, square_names = _omitted_squares(ctx)
        X2, alias_terms = np.hstack([X2, squares]), [*alias_terms, *square_names]
    if X2.shape[1] == 0:
        outside = "second-order terms" if ctx.alias_terms == "quadratic" else "two-factor interactions"
        return {
            "alias_matrix": {
                "matrix": [],
                "model_terms": list(ctx.column_names),
                "alias_terms": [],
                "max_abs": 0.0,
                "max_abs_main_effect_rows": 0.0,
                "frobenius_norm": 0.0,
                "note": f"No {outside} outside the model to alias against.",
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
        "n_samples": len(pv),
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
    """Variance inflation factor of each model term (excluding the intercept).

    ``VIF_j = c_jj * sum_i (x_ij - mean_j)^2`` with ``c = (X'X)^-1``. For a model with an
    intercept this is the classical ``1 / (1 - R_j^2)``, with ``R_j^2`` from regressing
    column ``j`` on the other columns. A Scheffé mixture model has no intercept, and its
    linear blending columns sum to one, so the classical form (which centres the columns)
    is undefined for them; the same expression is reported, as JMP does: it is finite
    whenever the model is estimable, and the cross-product terms read as usual.
    """
    if ctx.is_singular:
        return {"vif": None, "note": "Design is rank-deficient for the specified model."}

    assert ctx.XtX_inv is not None
    centred_ss = ((ctx.X - ctx.X.mean(axis=0)) ** 2).sum(axis=0)
    vif = np.diag(ctx.XtX_inv) * centred_ss
    return {"vif": {name: float(vif[i]) for i, name in enumerate(ctx.column_names) if not _is_intercept_col(name)}}


def _compute_condition_number(ctx: _EvalContext) -> dict[str, float]:
    """Condition number of the model matrix X."""
    cn = float(np.linalg.cond(ctx.X))
    return {"condition_number": cn}


def _compute_scaled_condition_number(ctx: _EvalContext) -> dict[str, float]:
    """Condition number of the model matrix X with each column scaled to unit length.

    The scaling removes what the factors' units contribute, so the number measures only how
    nearly the columns are collinear: 1 for orthogonal columns, whatever their units. Belsley,
    Kuh and Welsch scale this way for their condition indexes; RS/Discover prints it as Cond. No.
    """
    return {"scaled_condition_number": float(np.linalg.cond(_unit_length_columns(ctx.X)[0]))}


def _coefficient_power(
    coefficient_over_sigma: float | np.ndarray, c_jj: float, df_resid: int, alpha: float
) -> float | np.ndarray:
    """Power of the 1-df F-test of one coefficient, ``beta_j = 0``, at level *alpha*.

    The noncentrality is ``(beta_j / sigma)**2 / c_jj``, with ``c_jj`` the coefficient's
    diagonal element of ``(X'X)^-1``; *df_resid* is the residual degrees of freedom.
    """
    f_crit = stats.f.ppf(1.0 - alpha, dfn=1, dfd=df_resid)
    ncp = np.asarray(coefficient_over_sigma, dtype=float) ** 2 / c_jj
    return 1.0 - stats.ncf.cdf(f_crit, dfn=1, dfd=df_resid, nc=ncp)


def _compute_power(ctx: _EvalContext) -> dict[str, Any]:
    """Power of the t-test (an F-test on 1 df) of each model coefficient against zero.

    ``effect_size`` is the anticipated *coefficient* in coded units (what JMP calls the
    anticipated coefficient), not the high-minus-low effect: for a two-level factor
    coded -1 / +1 a high-minus-low effect of ``delta`` is a coefficient of ``delta / 2``.
    The noncentrality is ``effect_size**2 / (sigma**2 c_jj)``, with ``c = (X'X)^-1``.

    For a Scheffé mixture model the linear blending coefficients are left out: they are
    the expected responses of the pure components, and testing one against zero is not
    a meaningful hypothesis (the mixture analysis tests them jointly, for equality).
    """
    if ctx.is_singular:
        return {"power": None, "note": "Design is rank-deficient for the specified model."}

    sigma = ctx.sigma if ctx.sigma is not None else 1.0
    df_resid = ctx.N - ctx.p

    if df_resid <= 0:
        return {"power": None, "note": "No residual degrees of freedom (saturated model)."}

    assert ctx.XtX_inv is not None  # guaranteed by not is_singular
    diag_inv = np.diag(ctx.XtX_inv)
    terms = [
        (i, name)
        for i, name in enumerate(ctx.column_names)
        if not _is_intercept_col(name) and name not in ctx.blending_terms
    ]

    def power_at(coefficient: float, i: int) -> float:
        return float(_coefficient_power(coefficient / sigma, diag_inv[i], df_resid, ctx.alpha))

    result: dict[str, Any]
    if ctx.effect_size is not None:
        result = {"power": {name: power_at(ctx.effect_size, i) for i, name in terms}}
    else:
        # No effect_size: power curves over a range of coefficients
        effect_sizes = np.linspace(0.5 * sigma, 3.0 * sigma, 20)
        result = {
            "power_curves": {
                name: [{"effect_size": float(es), "power": power_at(float(es), i)} for es in effect_sizes]
                for i, name in terms
            },
            "sigma": float(sigma),
        }
    if ctx.blending_terms:
        result["note"] = (
            "The linear blending terms are not tested against zero, so they have no power here; the "
            "mixture analysis tests them jointly for equality."
        )
    if not ctx.is_scheffe:
        result.update(_power_alias_exposure(ctx, [name for _, name in terms]))
    return result


#: Jones and Nachtsheim's (2011) power study (their Table 6): each hypothesis the test of one kind
#: of term, in the model it is made in, averaged over every factor or pair of factors it can be asked of.
SCREENING_HYPOTHESES = (
    (1, "linear effect", "the constant and the factor"),
    (2, "quadratic effect", "main effects and the square"),
    (3, "interaction effect", "main effects and the interaction"),
    (4, "linear effect", "full quadratic model of the two factors"),
    (5, "quadratic effect", "full quadratic model of the two factors"),
    (6, "interaction effect", "full quadratic model of the two factors"),
)

#: Table 7's coefficients, in units of sigma, when no ``effect_size`` is given.
_SCREENING_EFFECT_SIZES = (1.0, 2.0, 3.0)


def _screening_tests(x: np.ndarray) -> dict[int, list[tuple[np.ndarray, list[int], set[tuple[int, ...]]]]]:
    """Return each hypothesis's tests: a model matrix, the columns tested, and the second-order terms it holds.

    A square is keyed by its factor ``(a,)``, an interaction by its two ``(a, b)``.
    """
    n, k = x.shape
    ones, pairs = np.ones((n, 1)), list(itertools.combinations(range(k), 2))
    tests: dict[int, list[tuple[np.ndarray, list[int], set[tuple[int, ...]]]]] = {
        h: [] for h, _, _ in SCREENING_HYPOTHESES
    }
    tests[1] = [(np.column_stack([ones, x[:, a]]), [1], set()) for a in range(k)]
    tests[2] = [(np.column_stack([ones, x, x[:, a] ** 2]), [k + 1], {(a,)}) for a in range(k)]
    tests[3] = [(np.column_stack([ones, x, x[:, a] * x[:, b]]), [k + 1], {(a, b)}) for a, b in pairs]
    for a, b in pairs:
        model = np.column_stack([ones, x[:, a], x[:, b], x[:, a] ** 2, x[:, b] ** 2, x[:, a] * x[:, b]])
        held = {(a,), (b,), (a, b)}
        tests[4].append((model, [1, 2], held))
        tests[5].append((model, [3, 4], held))
        tests[6].append((model, [5], held))
    return tests


def _is_aliased(model: np.ndarray, j: int, outside: np.ndarray) -> bool:
    """Say whether a second-order column outside the model is indistinguishable from model column ``j``.

    It is when it lies in the span of the model's columns but not of the others: the fit then
    cannot tell the two effects apart, which Jones and Nachtsheim (2011) call direct aliasing.
    """
    others = np.delete(model, j, axis=1)
    project = others @ np.linalg.pinv(others)
    tested = model[:, j] - project @ model[:, j]
    rest = outside - project @ outside
    tested_norm, rest_norms = np.linalg.norm(tested), np.linalg.norm(rest, axis=0)
    scale = np.linalg.norm(outside, axis=0) + 1.0
    live = rest_norms > 1e-8 * scale
    if not live.any() or tested_norm == 0:
        return False
    cosine = np.abs(tested @ rest[:, live]) / (tested_norm * rest_norms[live])
    return bool((cosine > 1 - 1e-9).any())


def _compute_screening_power(ctx: _EvalContext) -> dict[str, Any]:
    """Average power of the tests of a screening study, Jones and Nachtsheim's (2011) Tables 6 and 7.

    How likely the design is to detect one active term of each kind, in the model an analyst
    would fit to find it (their Table 6): (1) a linear effect, with the constant alone beside it; (2) a
    quadratic effect, the main effects and that square; (3) an interaction, the main effects and that interaction;
    and (4) to (6) a linear effect, a quadratic effect or the interaction in the full quadratic
    model of two factors. Each test is the t test of one coefficient against zero, its power as
    for the ``power`` metric (a coefficient of ``effect_size``, in units of ``sigma``, at level
    ``alpha``; Table 7's 1, 2 and 3 times sigma when no ``effect_size`` is given), and the powers
    are averaged over every factor or pair of factors the test can be asked of. The ``model``
    argument is not used: each hypothesis has its own models.

    A test is not estimable when its model is rank-deficient or leaves no residual degrees of
    freedom, and aliased when a second-order term outside its model cannot be told apart from the
    term tested (every square of a two-level design with centre runs; an interaction of a
    resolution IV fraction). Neither has a power; the averages are over the other tests, and a
    hypothesis none of whose tests remains is reported as ``"aliased"`` or ``"not estimable"``,
    as Table 7 marks them. Only numeric factors can be studied this way.
    """
    if ctx.blending_terms or any(not pd.api.types.is_numeric_dtype(ctx.design_df[f]) for f in ctx.factor_names):
        return {
            "screening_power": None,
            "note": "The screening power study needs numeric factors only; it does not apply to mixture models "
            "or categorical factors.",
        }
    sigma = ctx.sigma if ctx.sigma is not None else 1.0
    sizes = [ctx.effect_size] if ctx.effect_size is not None else [e * sigma for e in _SCREENING_EFFECT_SIZES]
    x = ctx.design_df[ctx.factor_names].to_numpy(dtype=float)
    k = x.shape[1]
    every = {(a,): x[:, a] ** 2 for a in range(k)} | {
        (a, b): x[:, a] * x[:, b] for a, b in itertools.combinations(range(k), 2)
    }
    every_test, hypotheses = _screening_tests(x), []
    for number, effect, model in SCREENING_HYPOTHESES:
        powers, n_tests, n_aliased, n_not_estimable = [], 0, 0, 0
        for matrix, tested, held in every_test[number]:
            n_tests += len(tested)
            scaled, lengths = _unit_length_columns(matrix)
            df_resid = matrix.shape[0] - matrix.shape[1]
            if int(np.linalg.matrix_rank(scaled)) < matrix.shape[1] or df_resid <= 0:
                n_not_estimable += len(tested)
                continue
            inverse = np.linalg.inv(ctx.information(scaled)) / np.outer(lengths, lengths)
            outside = np.column_stack([column for term, column in every.items() if term not in held] or [x[:, :0]])
            for j in tested:
                if _is_aliased(matrix, j, outside):
                    n_aliased += 1
                    continue
                powers.append([float(_coefficient_power(e / sigma, inverse[j, j], df_resid, ctx.alpha)) for e in sizes])
        if not n_tests:
            continue
        status = "estimable" if powers else "aliased" if n_aliased else "not estimable"
        found = np.array(powers) if powers else None
        hypotheses.append(
            {
                "hypothesis": number,
                "effect": effect,
                "model": model,
                "status": status,
                "n_tests": n_tests,
                "n_aliased": n_aliased,
                "n_not_estimable": n_not_estimable,
                "average_power": None if found is None else found.mean(axis=0).tolist(),
                "min_power": None if found is None else found.min(axis=0).tolist(),
            }
        )
    return {
        "screening_power": {"effect_sizes": sizes, "sigma": float(sigma), "alpha": ctx.alpha, "hypotheses": hypotheses}
    }


#: Smallest alias weight treated as a real bias rather than rounding noise.
_ALIAS_TOLERANCE = 1e-8


def _power_alias_exposure(ctx: _EvalContext, tested: list[str]) -> dict[str, Any]:
    """Say which tested coefficients the two-factor interactions outside the model would bias.

    The power assumes the fitted model holds. An active interaction outside it adds
    ``weight * beta`` to each coefficient it is aliased with (``weight`` from the alias
    matrix), which can shrink the effect the test sees below ``effect_size``, and its
    unfitted variation inflates the residual error. In a 12-run Plackett-Burman design
    every main effect carries a third of each interaction among the other factors, and
    the stated power can then be far above what the experiment delivers. Returns an
    empty dict when no tested coefficient is aliased.
    """
    X2, alias_terms = _omitted_two_factor_interactions(ctx)
    if X2.shape[1] == 0:
        return {}
    assert ctx.XtX_inv is not None  # guaranteed: _compute_power returned early for a singular design
    alias = np.abs(ctx.XtX_inv @ ctx.X.T @ X2)
    rows = {name: i for i, name in enumerate(ctx.column_names)}
    exposure = {
        name: {
            "aliased_interactions": int(np.sum(alias[rows[name]] > _ALIAS_TOLERANCE)),
            "max_abs_weight": float(alias[rows[name]].max()),
        }
        for name in tested
        if alias[rows[name]].max() > _ALIAS_TOLERANCE
    }
    if not exposure:
        return {}
    worst = max(v["max_abs_weight"] for v in exposure.values())
    return {
        "alias_exposure": exposure,
        "power_assumes": (
            f"No two-factor interaction outside the model is active. {len(exposure)} of the tested coefficients are "
            f"aliased with interactions among the {len(alias_terms)} left out (weights up to {worst:.3g}; see "
            "alias_matrix): an active one biases them, which can hide a real effect or show a false one, and "
            "inflates the residual error, so the real power can be much lower than stated. A design whose main "
            "effects are orthogonal to the interactions (a DSD, or resolution IV and above) avoids this."
        ),
    }


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
    rank = ctx.rank
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

    dm = dmatrix(rhs, ctx.patsy_data, return_type="dataframe")
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
    # Both alias-structure builders write every chain as "effect = alias + alias ...".
    for chain in alias_chains:
        effect, aliases_str = (part.strip() for part in chain.split(" = ", 1))
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
    dm = dmatrix(
        rhs, _patsy_data(ctx.design_df[names], ctx.categorical_coding, ctx.reduced_rank), return_type="dataframe"
    )
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
                "note": "Not a two-level fractional factorial design.",
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
    "g_efficiency": _compute_g_efficiency,
    "a_optimality": _compute_a_optimality,
    "e_optimality": _compute_e_optimality,
    "information_matrix": _compute_information_matrix,
    "estimable_interactions": _compute_estimable_interactions,
    "correlation": _compute_correlation,
    "estimate_correlation": _compute_estimate_correlation,
    "alias_matrix": _compute_alias_matrix,
    "fds": _compute_fds,
    "prediction_variance": _compute_prediction_variance,
    "vif": _compute_vif,
    "condition_number": _compute_condition_number,
    "scaled_condition_number": _compute_scaled_condition_number,
    "power": _compute_power,
    "screening_power": _compute_screening_power,
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
#: mixes suffixes (``d_efficiency`` / ``g_efficiency`` versus ``a_optimality`` /
#: ``e_optimality``), so callers who reach for the "other" suffix (a natural guess,
#: e.g. ``d_optimality``) hit an "unknown metric" error. Each alias resolves to its
#: canonical registry key *before* validation, so both spellings work and the returned
#: dict still keys the result under the canonical name. Aliases are deliberately kept
#: out of ``_METRIC_REGISTRY`` so ``metric="all"`` does not compute anything twice.
_METRIC_ALIASES: dict[str, str] = {
    "d_optimality": "d_efficiency",
    "i_optimality": "average_prediction_variance",
    "i_criterion": "average_prediction_variance",
    "g_optimality": "g_efficiency",
    "a_efficiency": "a_optimality",
    "e_efficiency": "e_optimality",
}


def _resolve_metrics(metric: str | list[str]) -> list[str]:
    """Return the canonical metric names asked for; raise for an unknown one."""
    if metric == "all":
        return list(_METRIC_REGISTRY)
    # Resolve accepted spelling variants to their canonical registry keys.
    metrics = [_METRIC_ALIASES.get(m, m) for m in ([metric] if isinstance(metric, str) else metric)]
    unknown = [m for m in metrics if m not in _METRIC_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown metric(s): {unknown}. Available metrics: {sorted(_METRIC_REGISTRY)}")
    return metrics


def _random_effect_groups(
    design_df: pd.DataFrame, factor_names: list[str], random_effects: Mapping[str, float] | None
) -> tuple[pd.DataFrame, list[str], dict[str, tuple[np.ndarray, float]] | None, str | None]:
    """Take the columns that label the random effects' groups out of the design, with a note.

    Return the design and factor names without them, each random effect's labels and ratio,
    and the note; raise for a column the design does not have or a ratio below zero.
    """
    if not random_effects:
        return design_df, factor_names, None, None
    groups: dict[str, tuple[np.ndarray, float]] = {}
    for name, ratio in random_effects.items():
        if name not in design_df.columns:
            raise ValueError(f"random_effects names {name!r}, which is not a column of the design.")
        if not np.isfinite(ratio) or ratio < 0:
            raise ValueError(
                f"The variance ratio of random effect {name!r} must be a finite number of at least 0; got {ratio!r}."
            )
        groups[name] = (design_df[name].to_numpy(), float(ratio))
    described = ", ".join(
        f"{name} ({len(pd.unique(labels))} groups, ratio {ratio:g})" for name, (labels, ratio) in groups.items()
    )
    note = (
        "The metrics use the generalized least squares information X' V^-1 X, with V = I + sum_k eta_k Z_k Z_k' "
        f"over the random effects {described}: variances are in units of the run variance. The degrees of freedom "
        "and the power keep the run-level residual degrees of freedom."
    )
    kept = [f for f in factor_names if f not in groups]
    return design_df.drop(columns=list(groups)), kept, groups, note


def _separate_run_columns(
    design_df: pd.DataFrame, factor_names: list[str], model: str | None
) -> tuple[pd.DataFrame, list[str], str | None]:
    """Drop ``RunOrder`` and ``WholePlot``, and keep ``Block`` only when the model formula uses it.

    A formula that names ``Block`` (``"A + B + Block"``) gets it as a categorical
    factor (its values become labels). Otherwise the block column is dropped, and when the design has more than
    one block a note says that the metrics ignore the blocks, which overstates the
    residual degrees of freedom by ``n_blocks - 1``.
    """
    factor_names = [f for f in factor_names if f not in RESERVED_COLUMN_NAMES]
    design_df = design_df.drop(columns=["RunOrder", WHOLE_PLOT_COL], errors="ignore")
    if "Block" not in design_df.columns:
        return design_df, factor_names, None
    if model is not None and re.search(r"\bBlock\b", model):
        return design_df.assign(Block=design_df["Block"].astype(str)), [*factor_names, "Block"], None
    n_blocks = int(design_df["Block"].nunique())
    design_df = design_df.drop(columns=["Block"])
    if n_blocks < 2:
        return design_df, factor_names, None
    note = (
        f"The design has {n_blocks} blocks, which the model leaves out, so the residual degrees of freedom, "
        "power and efficiencies are those of an unblocked design. Write the model as a formula with "
        "'+ Block' to include them as a categorical factor."
    )
    return design_df, factor_names, note


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
    region: str | DesignRegion | pd.DataFrame | None = None,
    n_samples: int = 100_000,
    include_vertices: bool = True,
    fds_resolution: int | None = None,
    random_state: int | np.random.Generator | None = 42,
    categorical_coding: str = "effect",
    integration: str = "sample",
    max_search: str = "sample",
    random_effects: Mapping[str, float] | None = None,
    alias_terms: str = "interactions",
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
        ``"scheffe_special_cubic"``, ``"scheffe_cubic"``), or an explicit patsy formula.  Over a
        mixture region the process-model names map to Scheffé models as in
        :func:`generate_design` (``"interactions"`` and ``"quadratic"`` to
        ``"scheffe_quadratic"``, ``"special_cubic"`` to
        ``"scheffe_special_cubic"``, ``"cubic"`` to ``"scheffe_cubic"``), and
        ``None`` means the model the design was generated for, else
        ``"scheffe_quadratic"``. Over a mixture-process region a mixture-process model
        name (``"interactions"``, ``"scheffe_quadratic + main_effects"``, ...; see
        :func:`~process_improve.experiments.designs_mixture_process.mixture_process_model`)
        becomes that model, and ``None`` means the model the design was generated for,
        else ``"interactions"``. Otherwise ``None`` means
        ``"interactions"``.  A formula may name a ``Block`` column (``"A + B +
        Block"``) to include the blocks as a categorical factor; otherwise the
        blocks are left out and a note says so.
    metric : str or list[str]
        One or more metric names to compute, or the special value ``"all"`` to
        compute every metric.  Valid names: ``"d_efficiency"``,
        ``"average_prediction_variance"`` (the I-criterion), ``"g_efficiency"``, ``"a_optimality"``,
        ``"e_optimality"``, ``"correlation"``, ``"estimate_correlation"`` (the
        correlations between the estimated second-order effects, by kind of pair),
        ``"alias_matrix"``, ``"fds"``,
        ``"prediction_variance"``, ``"vif"``, ``"condition_number"``,
        ``"scaled_condition_number"`` (with each column of the model matrix scaled to
        unit length, so the factors' units do not count), ``"power"``, ``"screening_power"`` (the
        average power of each kind of test of a screening study, Jones and Nachtsheim's Table 7;
        it ignores *model*), ``"degrees_of_freedom"``, ``"alias_structure"``,
        ``"confounding"``, ``"resolution"``, ``"defining_relation"``,
        ``"clear_effects"``, ``"minimum_aberration"``, ``"moment_aberration"``.
        ``"g_efficiency"`` is ``100 * p / (N * max d(x))`` (Atkinson, Donev and
        Tobias; Montgomery), where JMP reports its square root. The
        optimality-criterion metrics also accept the opposite suffix as an alias
        (e.g. ``"d_optimality"`` for ``"d_efficiency"``, ``"i_optimality"`` for
        ``"average_prediction_variance"``, ``"a_efficiency"`` for
        ``"a_optimality"``); the result is keyed under the canonical name.
    effect_size : float or None
        Anticipated size of a model *coefficient* in coded units, for the power
        calculation (JMP's "anticipated coefficient").  For a two-level factor
        coded -1 / +1 this is half the high-minus-low effect.  When *None*, a
        power curve over coefficients from ``0.5 * sigma`` to ``3 * sigma`` is
        returned instead.
    alpha : float
        Significance level for power calculation (default 0.05).
    sigma : float or None
        Estimated noise standard deviation.  Defaults to 1.0 when needed
        but not provided.
    region : {"cuboidal", "spherical"}, DesignRegion, pandas.DataFrame, or None
        Design region over which the region-based metrics (``average_prediction_variance``,
        ``g_efficiency``, ``fds``) integrate the prediction variance.
        ``"cuboidal"`` is ``[-1, 1]^k``; ``"spherical"`` is the ball of radius
        ``sqrt(k)``. A :class:`~process_improve.experiments.DesignRegion`
        restricts the average and the maximum to settings that satisfy its
        constraints (a box), lie on its constrained simplex (a mixture), or both (a
        mixture-process region): a constrained design should not be judged on corners
        it may not visit.
        A DataFrame is a discrete region, such as the candidate points an optimal
        design was chosen from: one row per point and a column per factor, in the
        design's units (coded, for a :class:`DesignResult`). The average and the
        maximum are then taken over exactly those points, each counted once, with
        no sampling.
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
    fds_resolution : int or None
        Resolution of the dense FDS curve.  When *None* (default) the ``fds``
        metric returns only the coarse 11-point ``quantiles`` summary.  When set
        (e.g. 200), a ``curve`` sub-dict with length-``fds_resolution``
        ``fraction`` / ``prediction_variance`` / ``scaled_prediction_variance``
        arrays is added for smooth plotting; the endpoints are the minimum and
        maximum prediction variance.
    random_state : int, numpy.random.Generator or None
        Seed for the region sampler (default 42, so repeated calls agree).
    categorical_coding : {"effect", "treatment"}
        How a categorical (label) factor enters the model. ``"effect"`` (default) is
        sum-to-zero effect coding, the usual convention in DoE software and the coding
        the built-in optimal designs are built in: a two-level factor is -1 and +1 (first
        and second level in sorted order), and a factor with ``L > 2`` levels gets ``L - 1``
        columns, the last level at -1 in every column (names such as ``"C[S.lo]"``).
        ``"treatment"`` is patsy's 0/1 dummy coding against the first level (``"C[T.y]"``),
        which earlier releases used. The A- and E-criteria, VIF, both condition numbers,
        power and the alias matrix depend on the coding; the prediction-variance metrics,
        D rankings and the degrees of freedom do not. A ``Block`` column named in the
        formula is sum-coded under ``"effect"``, as ``analyze_experiment`` codes blocks.
    integration : {"sample", "exact"}
        How the region average of the prediction variance (the I-criterion, in
        ``average_prediction_variance``) is worked out. ``"sample"``
        (default) averages it over the uniform sample. ``"exact"`` integrates it over the
        cuboidal region by Gauss-Legendre quadrature, ``trace((X'X)^-1 W)`` with ``W`` the
        region's moment matrix: exact for a model of degree up to four in each factor, a
        categorical factor averaged over its levels. It falls back to the sample, with a
        note, for another region or a quadrature grid of more than 250,000 points.
    max_search : {"sample", "optimize"}
        How the region maximum of the prediction variance (in ``g_efficiency``) is found.
        ``"sample"`` (default) takes the largest value over the sample and the boundary
        points. ``"optimize"`` then climbs from the best of those points by a compass
        search inside the cube, so a maximum between the vertices is found rather than
        missed: the variance of a design's projection often peaks inside a face or an
        edge. Over a :class:`DesignRegion` it is the sample's maximum.
    random_effects : mapping of str to float, or None
        For runs that share random effects: each key a column of the design that labels a
        run's group (the ``WholePlot`` of a split-plot design; the rows and the columns of a
        two-way split-plot, or strip-plot, design; random blocks), each value that effect's
        variance over the run variance, at least 0 (1 is a common assumption). The columns
        are not factors. Every metric then uses the generalized least squares information
        ``X' V^-1 X``, with ``V = I + sum_k eta_k Z_k Z_k'`` and ``Z_k`` the indicators of
        effect ``k``'s groups, in place of ``X'X``: the variances are in units of the run
        variance, as JMP reports them for split-plot and two-way split-plot designs, and
        ratios of 0 give back the ordinary least squares metrics. The degrees of freedom and
        the power keep the run-level residual degrees of freedom ``N - p``. ``None`` (default)
        treats the runs as independent.
    alias_terms : {"interactions", "quadratic"}
        The terms outside the model that ``alias_matrix`` works out the biases of.
        ``"interactions"`` (default): the two-factor interactions the model leaves out.
        ``"quadratic"``: those and the squares of the numeric factors, the terms of the full
        second-order model, so the matrix shows how curvature biases a first-order design's
        intercept (Davies, 11.33: b0 -> b0 + b11 + b22 + ...).

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
        below 1, a factor setting is missing, *categorical_coding* is unknown, or
        *random_effects* names a column the design lacks or has a negative ratio.

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
        # A three-level fraction's generators work modulo 3, and the relation and alias metrics below
        # modulo 2, so they read its recorded relation and resolution instead.
        generators = None if design_matrix.metadata.get("modulus") == MODULUS else design_matrix.generators
        defining_relation = design_matrix.defining_relation
        resolution = design_matrix.resolution
        factor_names = list(design_matrix.factor_names)
        design_df = pd.DataFrame(design_matrix.design)
    else:
        design_df = pd.DataFrame(design_matrix)
        factor_names = list(design_df.columns)

    design_df, factor_names, groups, groups_note = _random_effect_groups(design_df, factor_names, random_effects)
    design_df, factor_names, block_note = _separate_run_columns(design_df, factor_names, model)
    _validate_inputs(design_df, factor_names, alpha, sigma, n_samples)
    _check_options(integration=integration, max_search=max_search, alias_terms=alias_terms)
    metrics = _resolve_metrics(metric)
    logger.debug("evaluate_design: model=%r, metrics=%s", model, metrics)

    # --- Region, and the Scheffé default for mixtures (an intercept is redundant there) ---
    region = _resolve_region(region, design_matrix, factor_names)
    if isinstance(region, DesignRegion) and region.kind == "mixture":
        model = _mixture_model(model, design_matrix)
    elif isinstance(region, DesignRegion) and region.kind == "mixture_process":
        model = _mixture_process_formula(model, design_matrix, region)

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
            random_state=random_state,
            fds_resolution=fds_resolution,
            design_type=design_type,
            categorical_coding=categorical_coding,
            integration=integration,
            max_search=max_search,
            random_effects=groups,
            alias_terms=alias_terms,
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
    if block_note:
        notes["blocks"] = block_note
    if groups_note:
        notes["random_effects"] = groups_note
    if ctx.lost_columns:
        notes["inestimable_columns"] = (
            f"The region's constraints on categorical factors leave the column(s) {ctx.lost_columns} inestimable "
            "over every allowed combination of levels; the metrics are for the model without them."
        )
    if notes:
        results["notes"] = notes

    return results


def evaluate_all(  # noqa: PLR0913
    design_matrix: pd.DataFrame | DesignResult,
    model: str | None = None,
    effect_size: float | None = None,
    alpha: float = 0.05,
    sigma: float | None = None,
    region: str | DesignRegion | pd.DataFrame | None = None,
    n_samples: int = 100_000,
    include_vertices: bool = True,
    fds_resolution: int | None = None,
    random_state: int | np.random.Generator | None = 42,
    categorical_coding: str = "effect",
    integration: str = "sample",
    max_search: str = "sample",
    random_effects: Mapping[str, float] | None = None,
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
        random_state=random_state,
        categorical_coding=categorical_coding,
        integration=integration,
        max_search=max_search,
        random_effects=random_effects,
    )
