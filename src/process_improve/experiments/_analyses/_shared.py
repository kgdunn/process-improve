# (c) Kevin Dunn, 2010-2026. MIT License.
"""Always-on model-summary helpers used by :func:`analyze_experiment` (ENG-02)."""

from __future__ import annotations

import numpy as np
from statsmodels.regression.linear_model import RegressionResultsWrapper


def _compute_pred_r_squared(ols_result: RegressionResultsWrapper) -> float:
    """Predicted R² from PRESS residuals (~5 lines)."""
    influence = ols_result.get_influence()
    press_residuals = influence.resid_press
    ss_press = float(np.sum(press_residuals**2))
    ss_total = float(np.sum((ols_result.model.endog - ols_result.model.endog.mean()) ** 2))
    if ss_total == 0:
        return 0.0
    return 1.0 - ss_press / ss_total


def _compute_adequate_precision(ols_result: RegressionResultsWrapper) -> float:
    """Adequate precision: signal-to-noise ratio (~10 lines)."""
    predicted = ols_result.fittedvalues.values
    mse = float(ols_result.mse_resid)
    n = len(predicted)
    p = ols_result.df_model + 1

    max_pred = float(np.max(predicted))
    min_pred = float(np.min(predicted))
    signal = max_pred - min_pred

    # Average variance of prediction
    avg_var = p * mse / n
    noise = np.sqrt(avg_var)

    if noise == 0:
        return float("inf")
    return signal / noise
