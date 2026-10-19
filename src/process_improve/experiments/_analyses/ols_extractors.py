# (c) Kevin Dunn, 2010-2026. MIT License.
"""Thin extractors over a fitted OLS result: ANOVA, effects, coefficients,
significance, and confidence intervals (ENG-02).

The ANOVA and significance take a fit with one column per alias chain
(:func:`~.aliasing.chain_reduced_fit`) and a ``labels`` map that renames each chain's
leader to the chain, so exactly aliased terms are tested once, as in the effects.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.regression.linear_model import RegressionResultsWrapper

from ._shared import has_error_variance, is_block_term
from .aliasing import estimable_effects

_SATURATED_NOTE = (
    "Saturated model - no residual degrees of freedom, so no term can be tested. "
    "Use analysis_type='lenth_method' for an unreplicated two-level design, or fit fewer terms."
)


def _block_row(ols_result: RegressionResultsWrapper, blocks: tuple[str, ...]) -> dict[str, Any]:
    """One ANOVA row for all the block contrasts together, by the F-test that they are all zero."""
    names = [str(n) for n in ols_result.model.exog_names]
    present = [name for name in blocks if name in names]
    restrictions = np.zeros((len(present), len(names)))
    for row, name in enumerate(present):
        restrictions[row, names.index(name)] = 1.0
    test = ols_result.f_test(restrictions)
    f_value = float(np.squeeze(test.fvalue))
    mse = float(ols_result.mse_resid)
    return {
        "source": "Block",
        "df": len(present),
        "sum_sq": f_value * len(present) * mse,
        "mean_sq": f_value * mse,
        "F": f_value,
        "p_value": float(test.pvalue),
    }


def _run_anova(
    ols_result: RegressionResultsWrapper,
    anova_type: int = 2,
    labels: dict[str, str] | None = None,
    blocks: tuple[str, ...] = (),
) -> dict[str, Any]:
    """ANOVA table via statsmodels, with Type II sums of squares by default.

    Type II tests each term adjusted for every term that does not contain it, so a main
    effect is not adjusted for its own interactions. In an orthogonal design this equals
    the Type III (fully adjusted) test that the coefficient t-tests make; in a
    non-orthogonal one the two can differ, and the type used is reported under
    ``anova_type``. The ``blocks`` contrast columns are tested together, as one
    ``"Block"`` row with one degree of freedom fewer than the number of blocks.
    """
    if ols_result.df_resid <= 0:
        return {"anova_table": [], "anova_note": _SATURATED_NOTE}
    labels = labels or {}
    table = sm.stats.anova_lm(ols_result, typ=anova_type)
    records = []
    for idx, row in table.iterrows():
        if is_block_term(str(idx), blocks):
            continue
        df = float(row.get("df", 0))
        sum_sq = float(row.get("sum_sq", 0))
        records.append(
            {
                "source": labels.get(str(idx), str(idx)),
                "df": int(df),
                "sum_sq": sum_sq,
                "mean_sq": sum_sq / df if df > 0 else None,
                "F": float(row["F"]) if "F" in row and pd.notna(row.get("F")) else None,
                "p_value": (float(row["PR(>F)"]) if "PR(>F)" in row and pd.notna(row.get("PR(>F)")) else None),
            }
        )
    if any(name in ols_result.model.exog_names for name in blocks):
        records.insert(0, _block_row(ols_result, blocks))
    return {"anova_table": records, "anova_type": "I" * anova_type if anova_type <= 3 else str(anova_type)}


def _run_effects(
    ols_result: RegressionResultsWrapper,
    blocks: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Effects of the factors: each coefficient times the span of its coded model column.

    ``ols_result`` must be fitted on factors coded to -1/+1, as
    :func:`analyze_experiment` arranges. An effect is then the change in response
    from the lowest to the highest value of its model column: twice the coefficient
    for a main effect or interaction (Box, Hunter and Hunter, chapter 5), and the
    coefficient itself for a square, whose column runs from 0 to 1.

    Also returns ``effect_std_errors`` (the coefficient standard error times
    the same span) when residual degrees of freedom or a prior sigma are available;
    consumers such as the Pareto plot use this to draw effect-level error bars.

    Exactly aliased terms are reported as one effect per alias chain, named
    ``"A:B + C:D"``, since the design determines only their sum (#16). The
    chains are listed under ``alias_chains``, and terms aliased with the
    intercept under ``confounded_with_mean``; both keys appear only when
    there is aliasing. The ``blocks`` contrasts, and chains they lead, are not
    factor effects and are left out.
    """
    estimable = estimable_effects(ols_result)
    keep = [label for label in estimable.coefficients.index if not is_block_term(label, blocks)]
    spans = estimable.spans[keep]
    result: dict[str, Any] = {"effects": (spans * estimable.coefficients[keep]).to_dict()}
    if estimable.std_errors is not None:
        result["effect_std_errors"] = {str(k): float(spans[k] * estimable.std_errors[k]) for k in keep}
    chains = {label: members for label, members in estimable.chains.items() if label in keep}
    if chains:
        result["alias_chains"] = chains
    if estimable.confounded_with_mean:
        result["confounded_with_mean"] = estimable.confounded_with_mean
    return result


def _standardised(ols_result: RegressionResultsWrapper) -> np.ndarray:
    """Each coefficient in standard deviations of the response per standard deviation of its model column.

    The coefficient times the standard deviation of its column of the model matrix, over the
    standard deviation of the response: unitless, so terms in different units compare. The
    intercept's column does not vary, so its standardised coefficient is zero. Not a number when
    the response does not vary.
    """
    columns = np.asarray(ols_result.model.exog, dtype=float)
    response = np.asarray(ols_result.model.endog, dtype=float)
    if len(response) < 2 or np.ptp(response) == 0:  # a spread needs two values that differ
        return np.full(columns.shape[1], np.nan)
    return np.asarray(ols_result.params, dtype=float) * columns.std(axis=0, ddof=1) / response.std(ddof=1)


def _run_coefficients(ols_result: RegressionResultsWrapper) -> dict[str, Any]:
    """Coefficients with standard errors, t-values, p-values, CIs and standardised coefficients."""
    summary_df = pd.DataFrame(
        {
            "coefficient": ols_result.params,
            "std_error": ols_result.bse,
            "t_value": ols_result.tvalues,
            "p_value": ols_result.pvalues,
            "ci_low": ols_result.conf_int()[0],
            "ci_high": ols_result.conf_int()[1],
            "standardised_coefficient": _standardised(ols_result),
        }
    )
    records = []
    for name, row in summary_df.iterrows():
        records.append(
            {
                "term": str(name),
                "coefficient": float(row["coefficient"]),
                "std_error": float(row["std_error"]),
                "t_value": float(row["t_value"]),
                "p_value": float(row["p_value"]),
                "ci_low": float(row["ci_low"]),
                "ci_high": float(row["ci_high"]),
                "standardised_coefficient": float(row["standardised_coefficient"]),
            }
        )
    return {"coefficients": records}


def _joint_inference(
    ols_result: RegressionResultsWrapper, alpha: float, rename: Callable[[str], str] = str
) -> dict[str, Any]:
    r"""Return the correlations of the estimates, and the contour of their joint confidence region (#789 B10).

    The correlation of two estimates is their covariance over the product of their standard
    errors, so it comes from :math:`(X'X)^{-1}` alone, whatever the error variance. The joint
    :math:`1 - \alpha` confidence region of all :math:`p` coefficients is the set of values whose
    residual sum of squares is at most :math:`S_R (1 + p F / (n - p))`, with :math:`F` the upper
    :math:`\alpha` point on :math:`p` and :math:`n - p` degrees of freedom (Box, Hunter and Hunter,
    eq. 14.45); with a prior sigma it is :math:`S_R + \sigma^2 \chi^2`, on :math:`p` degrees of
    freedom. ``rename`` turns a model column's name into the name the coefficient table shows.
    """
    unscaled = np.asarray(ols_result.normalized_cov_params, dtype=float)
    spread = np.sqrt(np.diag(unscaled))
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = unscaled / np.outer(spread, spread)
    correlation[np.diag_indices_from(correlation)] = np.where(spread > 0, 1.0, np.nan)
    names = [rename(str(name)) for name in ols_result.model.exog_names]
    p, n, ssr = int(ols_result.model.rank), int(ols_result.nobs), float(ols_result.ssr)
    prior = getattr(ols_result, "prior_sigma", None)
    if prior is not None:
        bound = ssr + prior**2 * float(stats.chi2.ppf(1 - alpha, p))
    elif n > p:
        bound = ssr * (1 + p / (n - p) * float(stats.f.ppf(1 - alpha, p, n - p)))
    else:  # no residual to scale the region by
        bound = float("nan")
    return {
        "estimate_correlations": {
            a: {b: float(correlation[i, j]) for j, b in enumerate(names)} for i, a in enumerate(names)
        },
        "joint_confidence_region": {
            "sum_of_squares": bound,
            "residual_sum_of_squares": ssr,
            "confidence_level": 1 - alpha,
        },
    }


def _run_significance(
    ols_result: RegressionResultsWrapper,
    alpha: float = 0.05,
    labels: dict[str, str] | None = None,
    blocks: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Split the terms into significant and not significant, and list those that cannot be tested.

    A term whose p-value is not a number (no residual degrees of freedom, or a term the
    design cannot estimate) is listed under ``not_estimable_terms`` rather than left out
    of both lists. The ``blocks`` contrasts are not listed.
    """
    labels = labels or {}
    pvals = ols_result.pvalues.drop("Intercept", errors="ignore")
    named = [(labels.get(str(n), str(n)), float(p)) for n, p in pvals.items() if not is_block_term(str(n), blocks)]
    result: dict[str, Any] = {
        "significant_terms": [n for n, p in named if p < alpha],
        "not_significant_terms": [n for n, p in named if p >= alpha],
        "not_estimable_terms": [n for n, p in named if not np.isfinite(p)],
        "significance_level": alpha,
    }
    if not has_error_variance(ols_result):
        result["significance_note"] = _SATURATED_NOTE
    return result


def _run_confidence_intervals(ols_result: RegressionResultsWrapper, alpha: float = 0.05) -> dict[str, Any]:
    """Confidence intervals for coefficients."""
    ci = ols_result.conf_int(alpha=alpha)
    records = [
        {
            "term": str(name),
            "ci_low": float(ci.loc[name, 0]),
            "ci_high": float(ci.loc[name, 1]),
        }
        for name in ci.index
    ]
    return {"confidence_intervals": records, "confidence_level": 1.0 - alpha}
