# (c) Kevin Dunn, 2010-2026. MIT License.
"""Curvature test: center-point mean vs factorial-point mean (ENG-02)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.linear_model import RegressionResultsWrapper


def _run_curvature_test(
    ols_result: RegressionResultsWrapper,
    design_df: pd.DataFrame,
    response_col: str,
    factor_cols: list[str],
) -> dict[str, Any]:
    """Curvature test: compare center point mean vs factorial point mean.

    Custom implementation (~20 lines).
    """
    factors = design_df[factor_cols]
    y = design_df[response_col]

    # Center points: rows where ALL factors == 0
    is_center = (factors == 0).all(axis=1)
    n_center = int(is_center.sum())

    if n_center == 0:
        return {"curvature_test": {"error": "No center points in design."}}

    # Factorial points: rows where ALL factors are ±1
    is_factorial = factors.abs().eq(1).all(axis=1)
    n_factorial = int(is_factorial.sum())

    if n_factorial == 0:
        return {"curvature_test": {"error": "No factorial points found."}}

    y_center_mean = float(y[is_center].mean())
    y_factorial_mean = float(y[is_factorial].mean())
    difference = y_center_mean - y_factorial_mean

    mse = float(ols_result.mse_resid)
    se_diff = np.sqrt(mse * (1.0 / n_center + 1.0 / n_factorial))
    t_stat = difference / se_diff if se_diff > 0 else 0.0
    df_resid = int(ols_result.df_resid)
    p_value = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), df_resid))

    return {
        "curvature_test": {
            "center_point_mean": y_center_mean,
            "factorial_point_mean": y_factorial_mean,
            "difference": difference,
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "n_center_points": n_center,
            "n_factorial_points": n_factorial,
        }
    }
