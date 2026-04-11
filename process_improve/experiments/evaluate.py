# (c) Kevin Dunn, 2010-2026. MIT License.

"""Design evaluation: quality metrics for experimental designs.

Provides :func:`evaluate_design`, which computes properties and quality metrics
of an existing design matrix.  Supported metrics include efficiency values
(D/I/G), prediction variance, VIF, condition number, power analysis, alias
structure, confounding, resolution, defining relation, clear effects, minimum
aberration, and degrees of freedom.

Example
-------
>>> from process_improve.experiments import evaluate_design, generate_design, Factor
>>> factors = [Factor(name="A", low=0, high=10), Factor(name="B", low=0, high=10)]
>>> result = generate_design(factors, design_type="full_factorial", center_points=0)
>>> metrics = evaluate_design(result, model="interactions", metric=["d_efficiency", "vif"])
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from patsy import dmatrix
from scipy import stats
from scipy.stats import qmc
from statsmodels.stats.outliers_influence import variance_inflation_factor

from process_improve.experiments.factor import DesignResult

# ---------------------------------------------------------------------------
# Internal context shared across metric computations
# ---------------------------------------------------------------------------


@dataclass
class _EvalContext:
    """Shared evaluation context, computed once and reused by all metrics."""

    X: np.ndarray
    column_names: list[str]
    factor_names: list[str]
    design_df: pd.DataFrame
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


# ---------------------------------------------------------------------------
# Model matrix construction
# ---------------------------------------------------------------------------

_ROMAN = {3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII"}


def _build_model_matrix(
    design_df: pd.DataFrame,
    model: str | None,
    factor_names: list[str],
) -> tuple[np.ndarray, list[str]]:
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
    """
    if model is None:
        model = "interactions"

    # Map shorthand names to patsy right-hand-side formulas
    joined = " + ".join(factor_names)
    if model == "main_effects":
        rhs = joined
    elif model == "interactions":
        rhs = f"({joined}) ** 2"
    elif model == "quadratic":
        squared = " + ".join(f"I({f} ** 2)" for f in factor_names)
        rhs = f"({joined}) ** 2 + {squared}"
    elif "~" in model:
        # Explicit formula with response side — strip LHS
        rhs = model.split("~", 1)[1].strip()
    else:
        # Assume it is already a valid RHS formula
        rhs = model

    dm = dmatrix(rhs, design_df, return_type="dataframe")
    X = np.asarray(dm, dtype=float)
    column_names = list(dm.columns)
    return X, column_names


def _build_context(  # noqa: PLR0913
    design_df: pd.DataFrame,
    factor_names: list[str],
    model: str | None,
    generators: list[str] | None,
    defining_relation: list[str] | None,
    resolution: int | None,
    effect_size: float | None,
    alpha: float,
    sigma: float | None,
) -> _EvalContext:
    """Build the shared evaluation context."""
    X, column_names = _build_model_matrix(design_df, model, factor_names)
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
        factor_names=factor_names,
        design_df=design_df,
        N=N,
        p=p,
        XtX=XtX,
        XtX_inv=XtX_inv,
        is_singular=is_singular,
        generators=generators,
        defining_relation=defining_relation,
        resolution=resolution,
        effect_size=effect_size,
        alpha=alpha,
        sigma=sigma,
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


def _generate_evaluation_grid(factor_names: list[str], n_points: int = 5000) -> np.ndarray:
    """Generate points in [-1, 1]^k for evaluating prediction variance.

    Uses Sobol sequences for efficient space-filling coverage.
    """
    k = len(factor_names)
    sampler = qmc.Sobol(d=k, scramble=True, seed=42)
    # Power-of-2 samples for Sobol balance
    m = max(10, int(np.ceil(np.log2(n_points))))
    points_01 = sampler.random_base2(m)  # in [0, 1]^k
    return points_01 * 2.0 - 1.0  # scale to [-1, 1]^k


def _expand_grid_points(grid_raw: np.ndarray, factor_names: list[str], model: str | None) -> np.ndarray:
    """Expand raw grid points through the model formula to get X_grid."""
    df_grid = pd.DataFrame(grid_raw, columns=factor_names)
    X_grid, _ = _build_model_matrix(df_grid, model, factor_names)
    return X_grid


def _compute_g_efficiency(ctx: _EvalContext) -> dict[str, Any]:
    """G-efficiency: 100 * p / (N * max prediction variance over design region)."""
    if ctx.is_singular:
        return {"g_efficiency": None, "note": "Design is rank-deficient for the specified model."}

    grid_raw = _generate_evaluation_grid(ctx.factor_names)
    # Determine the model string to rebuild for grid points
    model_str = _infer_model_string(ctx)
    X_grid = _expand_grid_points(grid_raw, ctx.factor_names, model_str)
    pv = _prediction_variance_at_points(X_grid, ctx.XtX_inv)
    max_pv = float(np.max(pv))

    g_eff = 100.0 * ctx.p / (ctx.N * max_pv) if max_pv > 0 else None
    return {
        "g_efficiency": float(g_eff) if g_eff is not None else None,
        "max_prediction_variance": max_pv,
    }


def _compute_i_efficiency(ctx: _EvalContext) -> dict[str, Any]:
    """I-efficiency: 100 * p / (N * average prediction variance over design region)."""
    if ctx.is_singular:
        return {"i_efficiency": None, "note": "Design is rank-deficient for the specified model."}

    grid_raw = _generate_evaluation_grid(ctx.factor_names)
    model_str = _infer_model_string(ctx)
    X_grid = _expand_grid_points(grid_raw, ctx.factor_names, model_str)
    pv = _prediction_variance_at_points(X_grid, ctx.XtX_inv)
    avg_pv = float(np.mean(pv))

    i_eff = 100.0 * ctx.p / (ctx.N * avg_pv) if avg_pv > 0 else None
    return {
        "i_efficiency": float(i_eff) if i_eff is not None else None,
        "average_prediction_variance": avg_pv,
    }


def _infer_model_string(ctx: _EvalContext) -> str | None:
    """Reconstruct the model string from the context for grid expansion.

    We need to re-apply the same model transformation to grid points.  Since
    the context stores only the expanded column names, we reconstruct the
    model shorthand by checking the column structure.
    """
    cols = set(ctx.column_names)
    has_interactions = any(":" in c for c in cols)
    has_squared = any("**" in c or c.startswith("I(") for c in cols)

    if has_squared:
        return "quadratic"
    if has_interactions:
        return "interactions"
    return "main_effects"


def _compute_prediction_variance(ctx: _EvalContext) -> dict[str, Any]:
    """Prediction variance d(x_i) = x_i' (X'X)^-1 x_i at each design point."""
    if ctx.is_singular:
        return {"prediction_variance": None, "note": "Design is rank-deficient for the specified model."}

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


def _compute_degrees_of_freedom(ctx: _EvalContext) -> dict[str, Any]:
    """Degrees of freedom breakdown."""
    df_model = ctx.p - 1  # excluding intercept
    df_residual = ctx.N - ctx.p
    df_total = ctx.N - 1

    # Detect replicates by counting distinct rows
    design_rounded = np.round(ctx.design_df[ctx.factor_names].values, decimals=10)
    n_distinct = len(set(map(tuple, design_rounded)))
    has_replicates = n_distinct < ctx.N

    result: dict[str, Any] = {
        "degrees_of_freedom": {
            "model": df_model,
            "residual": df_residual,
            "total": df_total,
        }
    }
    if has_replicates:
        df_pure_error = ctx.N - n_distinct
        df_lack_of_fit = n_distinct - ctx.p if n_distinct > ctx.p else 0
        result["degrees_of_freedom"]["pure_error"] = df_pure_error
        result["degrees_of_freedom"]["lack_of_fit"] = df_lack_of_fit

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
    """Multiply two words in GF(2) — symmetric difference of factor sets."""
    return w1.symmetric_difference(w2)


def _defining_relation_from_generators(
    generators: list[str], factor_names: list[str]
) -> list[frozenset[int]]:
    """Compute the full defining relation from generator strings.

    Each generator like ``"D=ABC"`` produces the word ``ABCD``.  The full
    defining relation is the closure under GF(2) multiplication of all
    generator words and their products (all non-empty subsets).
    """
    # Parse each generator into a defining word
    base_words: list[frozenset[int]] = []
    for gen in generators:
        parts = gen.split("=")
        lhs = parts[0].strip()
        rhs = parts[1].strip() if len(parts) > 1 else ""
        lhs_idx = _parse_word(lhs, factor_names)
        rhs_idx = _parse_word(rhs, factor_names)
        word = _multiply_words(lhs_idx, rhs_idx)
        base_words.append(word)

    # Generate all non-empty subsets and their products
    all_words: set[frozenset[int]] = set()
    for r in range(1, len(base_words) + 1):
        for subset in itertools.combinations(base_words, r):
            product = frozenset()
            for w in subset:
                product = _multiply_words(product, w)
            if product:  # exclude identity
                all_words.add(product)

    return sorted(all_words, key=lambda w: (len(w), sorted(w)))


def _compute_defining_relation(ctx: _EvalContext) -> dict[str, Any]:
    """Compute or return the defining relation."""
    if ctx.defining_relation:
        return {"defining_relation": ctx.defining_relation}

    if not ctx.generators:
        return {"defining_relation": None, "note": "No generators available. Not a fractional factorial design."}

    words = _defining_relation_from_generators(ctx.generators, ctx.factor_names)
    relation = [f"I={_word_to_str(w, ctx.factor_names)}" for w in words]
    return {"defining_relation": relation}


def _compute_resolution(ctx: _EvalContext) -> dict[str, Any]:
    """Design resolution = minimum word length in the defining relation."""
    if ctx.resolution is not None:
        roman = _ROMAN.get(ctx.resolution, str(ctx.resolution))
        return {"resolution": ctx.resolution, "roman": roman}

    if not ctx.generators:
        return {"resolution": None, "roman": None, "note": "Not a fractional factorial design."}

    words = _defining_relation_from_generators(ctx.generators, ctx.factor_names)
    if not words:
        return {"resolution": None, "roman": None, "note": "No defining relation words found."}

    res = min(len(w) for w in words)
    roman = _ROMAN.get(res, str(res))
    return {"resolution": res, "roman": roman}


def _compute_alias_structure(ctx: _EvalContext) -> dict[str, Any]:
    """Alias structure: which effects are aliased with which others.

    Uses GF(2) arithmetic when generators are available; falls back to
    correlation-based detection otherwise.
    """
    if ctx.generators:
        return _alias_structure_from_generators(ctx)
    return _alias_structure_from_correlation(ctx)


def _alias_structure_from_generators(ctx: _EvalContext) -> dict[str, Any]:
    """Compute alias chains using GF(2) multiplication against the defining relation."""
    words = _defining_relation_from_generators(ctx.generators, ctx.factor_names)
    if not words:
        return {"alias_structure": []}

    # Build alias chains for main effects and 2-factor interactions
    alias_chains: list[str] = []
    k = len(ctx.factor_names)

    # Main effects
    for i in range(k):
        effect = frozenset([i])
        effect_name = ctx.factor_names[i]
        aliases = []
        for w in words:
            alias = _multiply_words(effect, w)
            alias_name = _word_to_str(alias, ctx.factor_names)
            aliases.append(alias_name)
        # Sort by word length
        aliases.sort(key=lambda s: (len(s), s))
        chain = f"{effect_name} = " + " + ".join(aliases)
        alias_chains.append(chain)

    # 2-factor interactions
    for i, j in itertools.combinations(range(k), 2):
        effect = frozenset([i, j])
        effect_name = _word_to_str(effect, ctx.factor_names)
        aliases = []
        for w in words:
            alias = _multiply_words(effect, w)
            alias_name = _word_to_str(alias, ctx.factor_names)
            aliases.append(alias_name)
        aliases.sort(key=lambda s: (len(s), s))
        chain = f"{effect_name} = " + " + ".join(aliases)
        alias_chains.append(chain)

    return {"alias_structure": alias_chains}


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
        aliases = _find_correlated_aliases(X_full, col_names, i, nonzero, threshold=0.995)
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
        confounding_list.append({
            "effect": effect,
            "confounded_with": confounded,
        })
    return {"confounding": confounding_list}


def _effect_order(term: str, factor_names: list[str]) -> int:
    """Determine the order of an effect term (1=main, 2=2FI, etc.)."""
    if ":" in term:
        return term.count(":") + 1
    if term in factor_names:
        return 1
    return len(term)  # approximate for single-char factor names


def _compute_clear_effects(ctx: _EvalContext) -> dict[str, Any]:
    """Identify effects whose aliases are all of higher order."""
    alias_result = _compute_alias_structure(ctx)
    alias_chains = alias_result.get("alias_structure", [])

    clear_main: list[str] = []
    clear_2fi: list[str] = []

    for chain in alias_chains:
        if " = " not in chain:
            continue
        effect, aliases_str = chain.split(" = ", 1)
        effect = effect.strip()
        order = _effect_order(effect, ctx.factor_names)

        alias_terms = [a.strip().lstrip("+-") for a in aliases_str.strip().split(" + ")]
        all_higher = all(_effect_order(a, ctx.factor_names) > order for a in alias_terms)

        if all_higher and order == 1:
            clear_main.append(effect)
        elif all_higher and order == 2:
            clear_2fi.append(effect)

    return {
        "clear_effects": {
            "main_effects": clear_main,
            "two_factor_interactions": clear_2fi,
        }
    }


def _compute_minimum_aberration(ctx: _EvalContext) -> dict[str, Any]:
    """Wordlength pattern (A_3, A_4, ...) from the defining relation."""
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
    max_len = max(lengths) if lengths else 0
    # Wordlength pattern: A_i = number of words of length i, starting at i=3
    pattern = [lengths.count(i) for i in range(3, max_len + 1)]

    return {
        "minimum_aberration": {
            "wordlength_pattern": pattern,
            "wordlength_pattern_labels": [f"A_{i}" for i in range(3, max_len + 1)],
        }
    }


# ---------------------------------------------------------------------------
# Metric dispatch registry
# ---------------------------------------------------------------------------

_METRIC_REGISTRY: dict[str, Callable[[_EvalContext], Any]] = {
    "d_efficiency": _compute_d_efficiency,
    "i_efficiency": _compute_i_efficiency,
    "g_efficiency": _compute_g_efficiency,
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
}


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
) -> dict[str, Any]:
    """Compute quality metrics for an experimental design.

    Parameters
    ----------
    design_matrix : DataFrame or DesignResult
        The design to evaluate.  If a :class:`DesignResult` is passed, the
        coded design matrix and any generator / defining-relation metadata
        are extracted automatically.
    model : str or None
        Model type: ``"main_effects"``, ``"interactions"``, ``"quadratic"``,
        or an explicit patsy formula.  ``None`` defaults to ``"interactions"``.
    metric : str or list[str]
        One or more metric names to compute.  Valid names:
        ``"alias_structure"``, ``"confounding"``, ``"resolution"``,
        ``"defining_relation"``, ``"power"``, ``"d_efficiency"``,
        ``"i_efficiency"``, ``"g_efficiency"``, ``"prediction_variance"``,
        ``"degrees_of_freedom"``, ``"vif"``, ``"condition_number"``,
        ``"clear_effects"``, ``"minimum_aberration"``.
    effect_size : float or None
        Expected effect size for power calculation.  When *None*, a power
        curve over a range of effect sizes is returned instead.
    alpha : float
        Significance level for power calculation (default 0.05).
    sigma : float or None
        Estimated noise standard deviation.  Defaults to 1.0 when needed
        but not provided.

    Returns
    -------
    dict[str, Any]
        Results keyed by metric name.  The structure of each value depends
        on the metric — see individual metric documentation.

    Examples
    --------
    >>> from process_improve.experiments import evaluate_design, generate_design, Factor
    >>> factors = [Factor(name="A", low=0, high=10), Factor(name="B", low=0, high=10)]
    >>> result = generate_design(factors, design_type="full_factorial", center_points=0)
    >>> metrics = evaluate_design(result, model="main_effects", metric="d_efficiency")
    >>> metrics["d_efficiency"]  # doctest: +SKIP
    100.0
    """
    # --- Unpack input ---
    generators: list[str] | None = None
    defining_relation: list[str] | None = None
    resolution: int | None = None

    if isinstance(design_matrix, DesignResult):
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

    # --- Normalize metric to list ---
    metrics = [metric] if isinstance(metric, str) else list(metric)

    # Validate metric names
    unknown = [m for m in metrics if m not in _METRIC_REGISTRY]
    if unknown:
        available = sorted(_METRIC_REGISTRY.keys())
        raise ValueError(f"Unknown metric(s): {unknown}. Available metrics: {available}")

    # --- Build context ---
    ctx = _build_context(
        design_df=design_df,
        factor_names=factor_names,
        model=model,
        generators=generators,
        defining_relation=defining_relation,
        resolution=resolution,
        effect_size=effect_size,
        alpha=alpha,
        sigma=sigma,
    )

    # --- Compute requested metrics ---
    results: dict[str, Any] = {}
    for m in metrics:
        result = _METRIC_REGISTRY[m](ctx)
        results.update(result)

    return results
