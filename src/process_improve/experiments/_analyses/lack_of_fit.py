# (c) Kevin Dunn, 2010-2026. MIT License.
"""Lack-of-fit F-test using pure error from replicated points (ENG-02)."""

from __future__ import annotations

from typing import Any

import pandas as pd
from scipy import stats
from statsmodels.regression.linear_model import RegressionResultsWrapper


def _run_lack_of_fit(
    ols_result: RegressionResultsWrapper,
    design_df: pd.DataFrame,
    response_col: str,
    factor_cols: list[str] | None = None,
) -> dict[str, Any]:
    """Lack-of-fit F-test using pure error from replicated points.

    Separates residual SS into pure-error SS (from n_replicates) and
    lack-of-fit SS.  Custom implementation (~40 lines).

    ``factor_cols`` must be the MODEL's factor columns. When the caller
    passes a full design frame, bookkeeping columns like ``RunOrder`` (unique
    per row) or ``Block`` must be excluded: grouping on them previously made
    every group a singleton, so no generated design ever had detectable
    replicates and the test always reported "No replicated points".
    """
    residuals = ols_result.resid
    y = design_df[response_col]

    if factor_cols is None:
        _non_factor = {response_col, "RunOrder", "Block"}
        factor_cols = [c for c in design_df.columns if c not in _non_factor]
    if not factor_cols:
        return {"lack_of_fit": {"error": "No factor columns found."}}

    # Identify replicate groups by ROUNDED factor values: a coded -> actual ->
    # coded round trip can perturb the last few bits, which would silently
    # split replicates into distinct groups under exact groupby matching.
    group_frame = design_df[factor_cols].copy()
    for col in factor_cols:
        if pd.api.types.is_numeric_dtype(group_frame[col]):
            group_frame[col] = group_frame[col].astype(float).round(8)
    groups = design_df.groupby([group_frame[c] for c in factor_cols], sort=False)

    ss_pure_error = 0.0
    df_pure_error = 0

    for _name, group in groups:
        ni = len(group)
        if ni > 1:
            yi = y.loc[group.index]
            ss_pure_error += float(((yi - yi.mean()) ** 2).sum())
            df_pure_error += ni - 1

    if df_pure_error == 0:
        return {"lack_of_fit": {"error": "No replicated points - cannot test lack of fit."}}

    ss_residual = float((residuals**2).sum())
    df_residual = int(ols_result.df_resid)

    ss_lof = ss_residual - ss_pure_error
    df_lof = df_residual - df_pure_error

    if df_lof <= 0 or ss_pure_error <= 0:
        return {"lack_of_fit": {"error": "Insufficient degrees of freedom for lack-of-fit test."}}

    ms_lof = ss_lof / df_lof
    ms_pe = ss_pure_error / df_pure_error
    f_stat = ms_lof / ms_pe
    p_value = 1.0 - stats.f.cdf(f_stat, df_lof, df_pure_error)

    return {
        "lack_of_fit": {
            "ss_lack_of_fit": ss_lof,
            "df_lack_of_fit": df_lof,
            "ms_lack_of_fit": ms_lof,
            "ss_pure_error": ss_pure_error,
            "df_pure_error": df_pure_error,
            "ms_pure_error": ms_pe,
            "f_statistic": float(f_stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
        }
    }
