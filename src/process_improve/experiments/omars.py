# (c) Kevin Dunn, 2010-2026. MIT License.

"""Analysis of data from OMARS designs (orthogonal minimally aliased response surface).

OMARS designs are three-level response surface designs in which the main
effects are mutually orthogonal and orthogonal to the second-order effects,
while the second-order effects (two-factor interactions and quadratic terms)
are only *minimally* aliased with one another.  This orthogonality structure
permits a staged analysis that first resolves the main effects and then,
using a pooled estimate of the error variance, searches the second-order
space for the small number of active interaction and quadratic terms.

:func:`analyze_omars` implements that staged protocol.  It is design-source
agnostic: it accepts any coded design matrix (two- or three-level
quantitative factors) together with a response vector, and does not require
the design to have been produced by any particular generator.

The procedure, summarised:

1. Code each factor to the ``[-1, +1]`` range and centre the response.
2. Estimate the error variance from the residuals of the full second-order
   model (intercept + main effects + all quadratics and interactions).
3. Test the main effects with t-tests at ``alpha_main`` and split them into
   active and inactive sets.
4. Pool the inactive main effects back into the error estimate, increasing
   the error degrees of freedom and (usually) sharpening the variance.
5. Gate the second-order space with an overall F-test at
   ``alpha_second_overall``.
6. If the gate passes, build a candidate set of second-order terms subject to
   the requested heredity rules, then run an exhaustive best-subset search,
   adding terms until the residual second-order variation is no longer
   significant at ``alpha_second_subset``.

References
----------
The staged OMARS analysis methodology is described in the response-surface
design literature on orthogonal minimally aliased designs (Schoen, Eendebak,
Goos, and co-workers).  This is an independent implementation of that
published method.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# Quadratic-heredity options.
_Q_HEREDITY = ("none", "strong")
# Interaction-heredity options.
_I_HEREDITY = ("none", "weak", "strong")


@dataclass
class OmarsResult:
    """Container returned by :func:`analyze_omars`.

    Attributes
    ----------
    success : bool
        ``False`` when the full second-order model leaves no error degrees of
        freedom (the design cannot support the analysis).  All other fields are
        only meaningful when ``success`` is ``True``.
    initial_error_df : int
        Error degrees of freedom from the full second-order model (step 2).
    initial_rmse : float
        Square root of the initial error-variance estimate (step 2).
    active_main_effects : list[str]
        Factor names declared active by the main-effect t-tests (step 3),
        including any forced in via ``force_main_effects``.
    forced_main_effects : list[str]
        Factor names that were forced active despite being statistically
        inactive.
    main_effect_p_values : dict[str, float]
        Two-sided p-value for each factor's main effect.
    updated_error_df : int
        Error degrees of freedom after pooling inactive main effects (step 4).
    updated_rmse : float
        Square root of the pooled error-variance estimate (step 4).
    second_order_overall_p_value : float
        p-value of the overall second-order F-gate (step 5).  ``nan`` when the
        analysis was main-effects only.
    active_interactions : list[str]
        Two-factor interaction terms declared active, labelled ``"A:B"``.
    active_quadratics : list[str]
        Quadratic terms declared active, labelled ``"A^2"``.
    final_p_value : float
        p-value of the final (failed) F-test that terminated the subset search.
        ``nan`` when the second-order gate did not open.
    subset_limit : int
        Maximum number of second-order terms the subset search was allowed to
        consider.
    subset_limit_reason : str
        Human-readable explanation of how ``subset_limit`` was chosen.
    details : dict[str, Any]
        Additional diagnostic values (term ranks, candidate counts, ...).
    """

    success: bool = False
    initial_error_df: int = 0
    initial_rmse: float = float("nan")
    active_main_effects: list[str] = field(default_factory=list)
    forced_main_effects: list[str] = field(default_factory=list)
    main_effect_p_values: dict[str, float] = field(default_factory=dict)
    updated_error_df: int = 0
    updated_rmse: float = float("nan")
    second_order_overall_p_value: float = float("nan")
    active_interactions: list[str] = field(default_factory=list)
    active_quadratics: list[str] = field(default_factory=list)
    final_p_value: float = float("nan")
    subset_limit: int = 0
    subset_limit_reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)


def _code_design(matrix: np.ndarray) -> np.ndarray:
    """Code each column to the ``[-1, +1]`` range from its own min and max.

    A column with levels ``{lo, mid, hi}`` is mapped with centre
    ``(hi + lo) / 2`` and half-range ``(hi - lo) / 2`` so that the extreme
    levels become ``-1`` and ``+1`` and any symmetric middle level becomes
    ``0``.
    """
    coded = np.empty_like(matrix, dtype=float)
    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        hi, lo = col.max(), col.min()
        half = (hi - lo) / 2.0
        if half == 0:
            msg = f"Factor column {j} is constant; it has no range to code."
            raise ValueError(msg)
        coded[:, j] = (col - (hi + lo) / 2.0) / half
    return coded


def _hat(matrix: np.ndarray) -> np.ndarray:
    """Orthogonal projection (hat) matrix onto the column space of ``matrix``."""
    return matrix @ np.linalg.pinv(matrix.T @ matrix) @ matrix.T


def _quadratic_columns(coded: np.ndarray) -> list[int]:
    """Return indices of columns carrying a third (middle) level, i.e. quadratic-capable."""
    return [j for j in range(coded.shape[1]) if np.any(np.abs(coded[:, j]) != 1)]


def _full_second_order(coded: np.ndarray, quad_cols: list[int]) -> np.ndarray:
    """Return the centred matrix of all quadratic and two-factor interaction columns."""
    n_runs, n_factors = coded.shape
    quad_blocks = [(coded[:, j] ** 2).reshape(n_runs, 1) for j in quad_cols]
    inter_blocks = [
        (coded[:, a] * coded[:, b]).reshape(n_runs, 1) for a, b in itertools.combinations(range(n_factors), 2)
    ]
    blocks = quad_blocks + inter_blocks
    if not blocks:
        return np.empty((n_runs, 0))
    full = np.hstack(blocks)
    return full - full.mean(axis=0)


def _heredity_candidates(  # noqa: PLR0913
    coded: np.ndarray,
    names: list[str],
    quad_cols: list[int],
    active_idx: list[int],
    inactive_idx: list[int],
    q_heredity: str,
    i_heredity: str,
) -> tuple[np.ndarray, list[str]]:
    """Build the centred candidate second-order matrix under heredity rules.

    Returns the matrix of candidate columns and the matching list of effect
    labels (``"A^2"`` for quadratics, ``"A:B"`` for interactions).
    """
    n_runs, n_factors = coded.shape
    columns: list[np.ndarray] = []
    labels: list[str] = []

    quad_set = set(quad_cols)
    # When no main effects are active the heredity rules cannot bite.
    if not active_idx:
        q_heredity, i_heredity = "none", "none"

    # Quadratic candidates.
    quad_candidates = [j for j in active_idx if j in quad_set] if q_heredity == "strong" else list(quad_cols)
    for j in quad_candidates:
        columns.append((coded[:, j] ** 2).reshape(n_runs, 1))
        labels.append(f"{names[j]}^2")

    # Interaction candidates.
    if i_heredity == "strong":
        pairs = list(itertools.combinations(sorted(active_idx), 2))
    elif i_heredity == "weak":
        pairs = list(itertools.combinations(sorted(active_idx), 2))
        pairs += [(min(a, b), max(a, b)) for a in active_idx for b in inactive_idx]
    else:
        pairs = list(itertools.combinations(range(n_factors), 2))
    for a, b in pairs:
        columns.append((coded[:, a] * coded[:, b]).reshape(n_runs, 1))
        labels.append(f"{names[a]}:{names[b]}")

    if not columns:
        return np.empty((n_runs, 0)), []

    candidate = np.hstack(columns)
    candidate = candidate - candidate.mean(axis=0)
    return candidate, labels


def _subset_limit(  # noqa: PLR0913
    n_runs: int,
    n_factors: int,
    full_so_rank: int,
    candidate_rank: int,
    n_candidates: int,
    user_limit: int | None,
) -> tuple[int, str]:
    """Decide how many second-order terms the subset search may consider."""
    total_so_terms = n_factors * (n_factors + 1) // 2
    if user_limit is not None:
        limit = min(user_limit, candidate_rank)
        return limit, f"user specified ({user_limit})"
    if total_so_terms == full_so_rank:
        return full_so_rank, "all second-order terms are jointly estimable"
    n_by_4 = n_runs // 4
    if n_candidates <= full_so_rank or candidate_rank <= n_by_4:
        return candidate_rank, "maximum jointly estimable second-order terms"
    return n_by_4, "run size divided by four"


def _residual_ss(design: np.ndarray, response: np.ndarray) -> float:
    """Residual sum of squares of ``response`` regressed on ``design``."""
    beta, _, _, _ = np.linalg.lstsq(design, response, rcond=None)
    residual = response - design @ beta
    return float((residual.T @ residual).item())


def _best_subset_search(  # noqa: PLR0913
    candidate: np.ndarray,
    labels: list[str],
    cy_second: np.ndarray,
    error_variance: float,
    error_df: int,
    full_so_rank: int,
    limit: int,
    alpha: float,
) -> tuple[list[str], float]:
    """Exhaustive best-subset search over the second-order candidates.

    Grows the subset one term at a time, always keeping the lowest-RSS subset of
    the current size, and stops once the residual second-order variation is no
    longer significant at ``alpha`` (or the candidates are exhausted).  Returns
    the active labels and the p-value of the terminating F-test.
    """
    from scipy.stats import f as f_dist  # noqa: PLC0415 - local import keeps scipy optional

    n_runs, n_candidates = candidate.shape
    intercept = np.ones((n_runs, 1))
    best_combo: tuple[int, ...] = ()
    final_p_value = float("nan")

    for size in range(1, limit + 1):
        best_rss = np.inf
        for combo in itertools.combinations(range(n_candidates), size):
            design = np.hstack((intercept, candidate[:, combo]))
            rss = _residual_ss(design, cy_second)
            if rss < best_rss:
                best_rss = rss
                best_combo = combo

        df_left = full_so_rank - size
        if df_left <= 0:
            break
        f_stat = (best_rss / df_left) / error_variance
        f_crit = f_dist.ppf(1 - alpha, dfn=df_left, dfd=error_df)
        final_p_value = float(1 - f_dist.cdf(f_stat, df_left, error_df))
        if size == n_candidates or f_stat < f_crit:
            break

    active = [labels[i] for i in best_combo]
    return active, final_p_value


def analyze_omars(  # noqa: C901, PLR0911, PLR0912, PLR0913, PLR0915
    design_matrix: pd.DataFrame,
    response: pd.Series | pd.DataFrame | np.ndarray,
    *,
    alpha_main: float = 0.05,
    alpha_second_overall: float = 0.20,
    alpha_second_subset: float = 0.20,
    quadratic_heredity: str = "none",
    interaction_heredity: str = "none",
    effects_to_drop: list[str] | None = None,
    force_main_effects: list[str] | None = None,
    second_order: bool = True,
    max_subset_terms: int | None = None,
) -> OmarsResult:
    """Analyse data from an OMARS design with the staged main-then-second-order protocol.

    Parameters
    ----------
    design_matrix : DataFrame
        One column per factor, one row per run.  Factors must be quantitative
        with two or three levels; they are coded internally to ``[-1, +1]``.
        Do not include second-order columns; they are constructed internally.
    response : Series, DataFrame, or ndarray
        The (single) response, one value per run.
    alpha_main : float, optional
        Significance level for the main-effect t-tests (step 3).  Default 0.05.
    alpha_second_overall : float, optional
        Significance level for the overall second-order F-gate (step 5).
        Default 0.20.
    alpha_second_subset : float, optional
        Significance level for the subset-search F-tests (step 6).
        Default 0.20.
    quadratic_heredity : {"none", "strong"}, optional
        Heredity rule for quadratic terms.  ``"strong"`` only considers the
        quadratic of a factor whose main effect is active.  Default ``"none"``.
    interaction_heredity : {"none", "weak", "strong"}, optional
        Heredity rule for two-factor interactions.  ``"strong"`` requires both
        parent main effects to be active; ``"weak"`` requires at least one;
        ``"none"`` considers all interactions.  Default ``"none"``.
    effects_to_drop : list of str, optional
        Second-order effect labels to exclude from the candidate set, e.g.
        ``["A^2", "B:C"]``.
    force_main_effects : list of str, optional
        Factor names to force into the active set even if statistically
        inactive.  Useful for borderline effects.
    second_order : bool, optional
        When ``False`` the analysis stops after the main effects (step 3).
        Default ``True``.
    max_subset_terms : int, optional
        Hard cap on the number of second-order terms the subset search may
        consider.  ``None`` (default) lets the routine choose a cap from the
        design's estimability.

    Returns
    -------
    OmarsResult
        Structured analysis result.  See the class documentation for fields.

    Raises
    ------
    ValueError
        If inputs are malformed (non-numeric, contain missing values,
        mismatched lengths, unknown heredity option, or unknown effect label).

    Examples
    --------
    >>> import pandas as pd
    >>> from process_improve.experiments import analyze_omars
    >>> result = analyze_omars(design, y, quadratic_heredity="strong")  # doctest: +SKIP
    >>> result.active_main_effects                                      # doctest: +SKIP
    ['A', 'C']
    """
    if quadratic_heredity not in _Q_HEREDITY:
        msg = f"quadratic_heredity must be one of {_Q_HEREDITY}, got {quadratic_heredity!r}."
        raise ValueError(msg)
    if interaction_heredity not in _I_HEREDITY:
        msg = f"interaction_heredity must be one of {_I_HEREDITY}, got {interaction_heredity!r}."
        raise ValueError(msg)

    from scipy.stats import f as f_dist  # noqa: PLC0415
    from scipy.stats import t as t_dist  # noqa: PLC0415

    effects_to_drop = list(effects_to_drop or [])
    force_main_effects = list(force_main_effects or [])

    names = [str(c) for c in design_matrix.columns]
    raw = np.asarray(design_matrix.to_numpy(), dtype=float)
    y = np.asarray(response).reshape(-1)
    if raw.shape[0] != y.shape[0]:
        msg = f"design_matrix has {raw.shape[0]} rows but response has {y.shape[0]} values."
        raise ValueError(msg)
    if not np.isfinite(raw).all() or not np.isfinite(y).all():
        msg = "design_matrix and response must not contain missing or infinite values."
        raise ValueError(msg)

    coded = _code_design(raw)
    n_runs, n_factors = coded.shape
    cy = (y - y.mean()).reshape(n_runs, 1)

    quad_cols = _quadratic_columns(coded)
    full_so = _full_second_order(coded, quad_cols)
    intercept = np.ones((n_runs, 1))
    total = np.hstack((intercept, coded, full_so))

    error_projection = np.identity(n_runs) - _hat(total)
    error_df = n_runs - np.linalg.matrix_rank(total)

    result = OmarsResult()
    if error_df <= 0:
        result.success = False
        result.details["reason"] = "full second-order model leaves no error degrees of freedom"
        return result

    variance = float((cy.T @ error_projection @ cy).item()) / error_df
    if variance <= 0:
        result.success = False
        result.details["reason"] = (
            "error variance is zero; the full model fits perfectly so no significance tests are possible"
        )
        return result

    result.success = True
    result.initial_error_df = int(error_df)
    s = float(np.sqrt(variance))
    result.initial_rmse = round(s, 6)

    # --- Step 3: main-effect t-tests -------------------------------------
    xtx_inv = np.linalg.pinv(coded.T @ coded)
    beta_me = (xtx_inv @ coded.T @ cy).reshape(n_factors)
    se_me = np.sqrt(np.diag(xtx_inv))
    t_values = np.abs(beta_me / (s * se_me))
    t_crit = t_dist.ppf(1 - alpha_main / 2, df=error_df)
    p_values = 2 * (1 - t_dist.cdf(t_values, error_df))

    active_idx = [j for j in range(n_factors) if t_values[j] >= t_crit]
    result.main_effect_p_values = {names[j]: float(round(p_values[j], 6)) for j in range(n_factors)}

    # Force requested main effects active.
    forced: list[str] = []
    name_to_idx = {names[j]: j for j in range(n_factors)}
    for fname in force_main_effects:
        if fname not in name_to_idx:
            msg = f"force_main_effects names unknown factor {fname!r}."
            raise ValueError(msg)
        j = name_to_idx[fname]
        if j not in active_idx:
            active_idx.append(j)
            forced.append(fname)
    active_idx.sort()
    inactive_idx = [j for j in range(n_factors) if j not in active_idx]
    result.active_main_effects = [names[j] for j in active_idx]
    result.forced_main_effects = forced

    if not second_order:
        result.updated_error_df = int(error_df)
        result.updated_rmse = result.initial_rmse
        return result

    # --- Step 4: pool inactive main effects into the error estimate ------
    inactive_projection = _hat(coded[:, inactive_idx]) if inactive_idx else np.zeros((n_runs, n_runs))
    pooled_projection = error_projection + inactive_projection
    pooled_df = error_df + len(inactive_idx)
    pooled_variance = float((cy.T @ pooled_projection @ cy).item()) / pooled_df
    result.updated_error_df = int(pooled_df)
    result.updated_rmse = round(float(np.sqrt(pooled_variance)), 6)

    # --- Step 5: overall second-order F-gate -----------------------------
    second_projection = np.identity(n_runs) - _hat(coded) - error_projection
    cy_second = (second_projection @ cy).reshape(n_runs, 1)
    tss = float((cy_second.T @ cy_second).item())
    full_so_rank = int(np.linalg.matrix_rank(full_so)) if full_so.shape[1] else 0
    result.details["full_second_order_rank"] = full_so_rank
    result.details["full_second_order_terms"] = int(full_so.shape[1])

    if full_so_rank == 0:
        result.second_order_overall_p_value = float("nan")
        return result

    f_stat = (tss / full_so_rank) / pooled_variance
    f_crit = f_dist.ppf(1 - alpha_second_overall, dfn=full_so_rank, dfd=pooled_df)
    result.second_order_overall_p_value = round(float(1 - f_dist.cdf(f_stat, full_so_rank, pooled_df)), 6)

    if f_stat < f_crit:
        # Gate stays shut: no active second-order effects.
        return result

    # --- Step 6: heredity-constrained best-subset search -----------------
    candidate, labels = _heredity_candidates(
        coded, names, quad_cols, active_idx, inactive_idx, quadratic_heredity, interaction_heredity
    )
    if effects_to_drop:
        unknown = [lbl for lbl in effects_to_drop if lbl not in labels]
        if unknown:
            msg = f"effects_to_drop names unknown second-order terms {unknown}."
            raise ValueError(msg)
        keep = [i for i, lbl in enumerate(labels) if lbl not in effects_to_drop]
        candidate = candidate[:, keep]
        labels = [labels[i] for i in keep]

    if candidate.shape[1] == 0:
        return result

    candidate_rank = int(np.linalg.matrix_rank(candidate))
    limit, reason = _subset_limit(n_runs, n_factors, full_so_rank, candidate_rank, candidate.shape[1], max_subset_terms)
    result.subset_limit = int(limit)
    result.subset_limit_reason = reason
    result.details["candidate_terms"] = int(candidate.shape[1])
    result.details["candidate_rank"] = candidate_rank

    if limit <= 0:
        return result

    active, final_p = _best_subset_search(
        candidate,
        labels,
        cy_second,
        pooled_variance,
        pooled_df,
        full_so_rank,
        limit,
        alpha_second_subset,
    )
    result.final_p_value = round(final_p, 6) if np.isfinite(final_p) else float("nan")
    result.active_interactions = [lbl for lbl in active if ":" in lbl]
    result.active_quadratics = [lbl for lbl in active if lbl.endswith("^2")]
    return result
