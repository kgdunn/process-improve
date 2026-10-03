# (c) Kevin Dunn, 2010-2026. MIT License.
"""Lack-of-fit F-test using pure error from replicated points (ENG-02)."""

from __future__ import annotations

from typing import Any

import pandas as pd
from scipy import stats
from statsmodels.regression.linear_model import RegressionResultsWrapper


def pure_error(design_df: pd.DataFrame, response_col: str, group_cols: list[str]) -> tuple[float, int]:
    """Pure-error sum of squares and degrees of freedom from runs replicated over ``group_cols``.

    Replicates are found on ROUNDED factor values: a coded -> actual -> coded round
    trip can perturb the last few bits, which would silently split replicates into
    distinct groups under exact matching. Rows with a missing response do not count.
    """
    data = design_df[design_df[response_col].notna()]
    y = data[response_col]
    group_frame = data[group_cols].copy()
    for col in group_cols:
        if pd.api.types.is_numeric_dtype(group_frame[col]):
            group_frame[col] = group_frame[col].astype(float).round(8)
    groups = data.groupby([group_frame[c] for c in group_cols], sort=False)

    ss_pure_error = 0.0
    df_pure_error = 0
    for _name, group in groups:
        ni = len(group)
        if ni > 1:
            yi = y.loc[group.index]
            ss_pure_error += float(((yi - yi.mean()) ** 2).sum())
            df_pure_error += ni - 1
    return ss_pure_error, df_pure_error


def _run_lack_of_fit(
    ols_result: RegressionResultsWrapper,
    design_df: pd.DataFrame,
    response_col: str,
    factor_cols: list[str] | None = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Lack-of-fit F-test using pure error from replicated points.

    Separates residual SS into pure-error SS (from n_replicates) and
    lack-of-fit SS.  Custom implementation (~40 lines).

    ``factor_cols`` must be the MODEL's factor columns (plus the block column when
    the model has a block term, so pure error is taken within blocks). When the
    caller passes a full design frame, bookkeeping columns like ``RunOrder`` (unique
    per row) must be excluded: grouping on them previously made every group a
    singleton, so no generated design ever had detectable replicates and the test
    always reported "No replicated points". ``alpha`` sets the ``significant`` flag.
    """
    residuals = ols_result.resid

    if factor_cols is None:
        _non_factor = {response_col, "RunOrder", "Block"}
        factor_cols = [c for c in design_df.columns if c not in _non_factor]
    if not factor_cols:
        return {"lack_of_fit": {"error": "No factor columns found."}}

    ss_pure_error, df_pure_error = pure_error(design_df, response_col, factor_cols)

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
    p_value = float(stats.f.sf(f_stat, df_lof, df_pure_error))

    return {
        "lack_of_fit": {
            "ss_lack_of_fit": ss_lof,
            "df_lack_of_fit": df_lof,
            "ms_lack_of_fit": ms_lof,
            "ss_pure_error": ss_pure_error,
            "df_pure_error": df_pure_error,
            "ms_pure_error": ms_pe,
            "f_statistic": float(f_stat),
            "p_value": p_value,
            "significant": p_value < alpha,
            "significance_level": alpha,
        }
    }
