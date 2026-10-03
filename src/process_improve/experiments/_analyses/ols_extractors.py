# (c) Kevin Dunn, 2010-2026. MIT License.
"""Thin extractors over a fitted OLS result: ANOVA, effects, coefficients,
significance, and confidence intervals (ENG-02).

The ANOVA and significance take a fit with one column per alias chain
(:func:`~.aliasing.chain_reduced_fit`) and a ``labels`` map that renames each chain's
leader to the chain, so exactly aliased terms are tested once, as in the effects.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper

from ._shared import is_block_term
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
    """Effects of the factors: twice the coefficient of each coded (-1/+1) model column.

    ``ols_result`` must be fitted on factors coded to -1/+1, as
    :func:`analyze_experiment` arranges, so that twice the coefficient is the change in
    response from the low to the high level (Box, Hunter and Hunter, chapter 5).

    Also returns ``effect_std_errors`` (twice the coefficient standard
    error) when residual degrees of freedom are available; consumers such
    as the Pareto plot use this to draw effect-level error bars.

    Exactly aliased terms are reported as one effect per alias chain, named
    ``"A:B + C:D"``, since the design determines only their sum (#16). The
    chains are listed under ``alias_chains``, and terms aliased with the
    intercept under ``confounded_with_mean``; both keys appear only when
    there is aliasing. The ``blocks`` contrasts, and chains they lead, are not
    factor effects and are left out.
    """
    estimable = estimable_effects(ols_result)
    keep = [label for label in estimable.coefficients.index if not is_block_term(label, blocks)]
    result: dict[str, Any] = {"effects": (2.0 * estimable.coefficients[keep]).to_dict()}
    if estimable.std_errors is not None:
        result["effect_std_errors"] = {str(k): float(2.0 * estimable.std_errors[k]) for k in keep}
    chains = {label: members for label, members in estimable.chains.items() if label in keep}
    if chains:
        result["alias_chains"] = chains
    if estimable.confounded_with_mean:
        result["confounded_with_mean"] = estimable.confounded_with_mean
    return result


def _run_coefficients(ols_result: RegressionResultsWrapper) -> dict[str, Any]:
    """Coefficients with standard errors, t-values, p-values, and CIs."""
    summary_df = pd.DataFrame(
        {
            "coefficient": ols_result.params,
            "std_error": ols_result.bse,
            "t_value": ols_result.tvalues,
            "p_value": ols_result.pvalues,
            "ci_low": ols_result.conf_int()[0],
            "ci_high": ols_result.conf_int()[1],
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
            }
        )
    return {"coefficients": records}


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
    if ols_result.df_resid <= 0:
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
