# (c) Kevin Dunn, 2010-2026. MIT License.
"""Thin extractors over a fitted OLS result: ANOVA, effects, coefficients,
significance, and confidence intervals (ENG-02).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper


def _run_anova(ols_result: RegressionResultsWrapper, anova_type: int = 2) -> dict[str, Any]:
    """ANOVA table via statsmodels."""
    if ols_result.df_resid <= 0:
        return {"anova_table": [], "note": "Saturated model - no residual degrees of freedom for ANOVA."}
    table = sm.stats.anova_lm(ols_result, typ=anova_type)
    records = [
        {
            "source": str(idx),
            "df": int(row.get("df", 0)),
            "sum_sq": float(row.get("sum_sq", 0)),
            "mean_sq": float(row.get("mean_sq", 0)) if "mean_sq" in row else None,
            "F": float(row["F"]) if "F" in row and pd.notna(row.get("F")) else None,
            "p_value": (
                float(row["PR(>F)"]) if "PR(>F)" in row and pd.notna(row.get("PR(>F)")) else None
            ),
        }
        for idx, row in table.iterrows()
    ]
    return {"anova_table": records}


def _run_effects(ols_result: RegressionResultsWrapper) -> dict[str, Any]:
    """Coefficient effects (2x the coefficient for coded ±1 factors).

    Also returns ``effect_std_errors`` (twice the coefficient standard
    error) when residual degrees of freedom are available; consumers such
    as the Pareto plot use this to draw effect-level error bars.
    """
    params = ols_result.params.drop("Intercept", errors="ignore")
    effects = (2.0 * params).to_dict()
    result: dict[str, Any] = {"effects": effects}
    if int(ols_result.df_resid) > 0:
        bse = ols_result.bse.drop("Intercept", errors="ignore")
        result["effect_std_errors"] = {str(k): float(2.0 * v) for k, v in bse.items()}
    return result


def _run_coefficients(ols_result: RegressionResultsWrapper) -> dict[str, Any]:
    """Coefficients with standard errors, t-values, p-values, and CIs."""
    summary_df = pd.DataFrame({
        "coefficient": ols_result.params,
        "std_error": ols_result.bse,
        "t_value": ols_result.tvalues,
        "p_value": ols_result.pvalues,
        "ci_low": ols_result.conf_int()[0],
        "ci_high": ols_result.conf_int()[1],
    })
    records = []
    for name, row in summary_df.iterrows():
        records.append({
            "term": str(name),
            "coefficient": float(row["coefficient"]),
            "std_error": float(row["std_error"]),
            "t_value": float(row["t_value"]),
            "p_value": float(row["p_value"]),
            "ci_low": float(row["ci_low"]),
            "ci_high": float(row["ci_high"]),
        })
    return {"coefficients": records}


def _run_significance(ols_result: RegressionResultsWrapper, alpha: float = 0.05) -> dict[str, Any]:
    """Identify significant and non-significant terms."""
    pvals = ols_result.pvalues.drop("Intercept", errors="ignore")
    significant = [str(n) for n, p in pvals.items() if p < alpha]
    not_significant = [str(n) for n, p in pvals.items() if p >= alpha]
    return {
        "significant_terms": significant,
        "not_significant_terms": not_significant,
        "significance_level": alpha,
    }


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
