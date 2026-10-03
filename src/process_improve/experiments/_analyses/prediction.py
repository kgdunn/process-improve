# (c) Kevin Dunn, 2010-2026. MIT License.
"""Prediction intervals and confirmation-run testing (ENG-02).

When the response was transformed, the model, and so every prediction and interval,
is on the transformed scale. The results say so under ``scale``, and a confirmation
run is put on that scale with the same transform before it is compared.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import RegressionResultsWrapper

from .transforms import to_model_scale


def _scale_label(transform_info: dict[str, Any] | None) -> str:
    """Describe the scale the model's predictions are on."""
    info = transform_info or {}
    transform = info.get("transform")
    if transform is None:
        return "response"
    if transform == "box_cox":
        return f"box_cox(lambda={float(info['box_cox_lambda']):.4g}) of the response"
    return f"{transform} of the response"


def _run_prediction(
    ols_result: RegressionResultsWrapper,
    new_points: pd.DataFrame,
    alpha: float = 0.05,
    transform_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Predictions with confidence and prediction intervals, on the scale the model was fitted on."""
    summary = ols_result.get_prediction(new_points).summary_frame(alpha=alpha)

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
    return {"predictions": records, "prediction_scale": _scale_label(transform_info)}


def _run_confirmation_test(
    ols_result: RegressionResultsWrapper,
    new_points: pd.DataFrame,
    observed: list[float],
    alpha: float = 0.05,
    transform_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare observed confirmation runs with the prediction interval at each new point.

    ``observed`` is on the response's own scale; with a transformed response it is
    transformed the same way before the comparison, and both are reported.
    """
    observed_raw = np.asarray(observed, dtype=float).reshape(-1)
    if len(observed_raw) != len(new_points):
        raise ValueError(
            f"observed_at_new has {len(observed_raw)} values but new_points has {len(new_points)} rows; "
            "give one observation per new point."
        )
    observed_model = to_model_scale(observed_raw, transform_info or {}, what="observed_at_new")
    summary = ols_result.get_prediction(new_points).summary_frame(alpha=alpha)
    transformed = (transform_info or {}).get("transform") is not None

    results = []
    for i, obs in enumerate(observed_raw):
        row = summary.iloc[i]
        pi_low = float(row["obs_ci_lower"])
        pi_high = float(row["obs_ci_upper"])
        on_scale = float(observed_model[i])
        record: dict[str, Any] = {
            "observed": float(obs),
            "predicted": float(row["mean"]),
            "pi_low": pi_low,
            "pi_high": pi_high,
            "within_PI": bool(pi_low <= on_scale <= pi_high),
        }
        if transformed:
            record["observed_transformed"] = on_scale
        results.append(record)

    return {
        "confirmation_test": {
            "results": results,
            "all_within_PI": all(r["within_PI"] for r in results),
            "confidence_level": 1.0 - alpha,
            "scale": _scale_label(transform_info),
        }
    }
