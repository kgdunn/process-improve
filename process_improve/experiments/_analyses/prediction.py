# (c) Kevin Dunn, 2010-2026. MIT License.
"""Prediction intervals and confirmation-run testing (ENG-02)."""

from __future__ import annotations

from typing import Any

import pandas as pd
from statsmodels.regression.linear_model import RegressionResultsWrapper


def _run_prediction(
    ols_result: RegressionResultsWrapper,
    new_points: pd.DataFrame,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Predictions with confidence and prediction intervals."""
    pred = ols_result.get_prediction(new_points)
    summary = pred.summary_frame(alpha=alpha)

    records = [
        {
            "predicted": float(row["mean"]),
            "ci_low": float(row["mean_ci_lower"]),
            "ci_high": float(row["mean_ci_upper"]),
            "pi_low": float(row["obs_ci_lower"]),
            "pi_high": float(row["obs_ci_upper"]),
        }
        for _i, row in summary.iterrows()
    ]
    return {"predictions": records}


def _run_confirmation_test(
    ols_result: RegressionResultsWrapper,
    new_points: pd.DataFrame,
    observed: list[float],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Compare observed confirmation runs against predicted values with PI."""
    pred = ols_result.get_prediction(new_points)
    summary = pred.summary_frame(alpha=alpha)

    results = []
    for i, obs in enumerate(observed):
        row = summary.iloc[i]
        pi_low = float(row["obs_ci_lower"])
        pi_high = float(row["obs_ci_upper"])
        predicted = float(row["mean"])
        within_pi = pi_low <= obs <= pi_high
        results.append({
            "observed": obs,
            "predicted": predicted,
            "pi_low": pi_low,
            "pi_high": pi_high,
            "within_PI": within_pi,
        })

    all_pass = all(r["within_PI"] for r in results)
    return {
        "confirmation_test": {
            "results": results,
            "all_within_PI": all_pass,
            "confidence_level": 1.0 - alpha,
        }
    }
