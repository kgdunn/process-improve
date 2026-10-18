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
from scipy import stats
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


def _groups_of_settings(new_points: pd.DataFrame) -> list[list[int]]:
    """Return the positions of the new points grouped by identical settings, in order of first appearance."""
    groups: dict[tuple[Any, ...], list[int]] = {}
    for i, row in enumerate(new_points.itertuples(index=False, name=None)):
        key = tuple(round(v, 9) if isinstance(v, float) else v for v in row)
        groups.setdefault(key, []).append(i)
    return list(groups.values())


def _mean_tests(
    ols_result: RegressionResultsWrapper,
    new_points: pd.DataFrame,
    summary: pd.DataFrame,
    observed: tuple[np.ndarray, np.ndarray],
    alpha: float,
) -> list[dict[str, Any]]:
    """Test the mean of the runs at each distinct setting against the prediction interval for such a mean.

    The interval for the mean of ``m`` new runs at a point is
    ``predicted +/- t(df) sqrt(se_fit**2 + s**2 / m)``, with ``se_fit`` the standard
    error of the fitted mean there and ``s**2`` the residual mean square. ``observed``
    holds the runs on the response's own scale and on the model's.
    """
    raw, on_scale = observed
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, ols_result.df_resid)) if ols_result.df_resid > 0 else np.nan
    tests = []
    for members in _groups_of_settings(new_points):
        first = summary.iloc[members[0]]
        predicted = float(first["mean"])
        half = t_crit * float(np.sqrt(first["mean_se"] ** 2 + ols_result.scale / len(members)))
        observed_mean = float(on_scale[members].mean())
        record: dict[str, Any] = {
            "settings": new_points.iloc[members[0]].to_dict(),
            "n_runs": len(members),
            "observed_mean": float(raw[members].mean()),
            "predicted": predicted,
            "pi_low": predicted - half,
            "pi_high": predicted + half,
            "within_PI": bool(abs(observed_mean - predicted) <= half) if np.isfinite(half) else None,
        }
        if on_scale is not raw:
            record["observed_mean_transformed"] = observed_mean
        tests.append(record)
    return tests


def _run_confirmation_test(
    ols_result: RegressionResultsWrapper,
    new_points: pd.DataFrame,
    observed: list[float],
    alpha: float = 0.05,
    transform_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare observed confirmation runs with the model's prediction intervals.

    ``observed`` is on the response's own scale; with a transformed response it is
    transformed the same way before the comparison, and both are reported.

    Each run is compared with the prediction interval for one new run (``results``,
    ``all_within_PI``). The verdict, ``confirmed``, instead compares the mean of the runs
    at each distinct setting with the prediction interval for the mean of that many runs
    (``mean_tests``). Requiring every run to fall inside its own 95% interval fails a
    correct model about one time in seven with three runs (``1 - 0.95**3``, a little
    less as the runs share the fitted mean), and more often with more runs; the test of
    the mean keeps the stated 5% false alarm per setting. ``confirmed`` is None when the
    model has no residual degrees of freedom, so no interval exists.
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

    mean_tests = _mean_tests(
        ols_result, new_points, summary, (observed_raw, observed_model if transformed else observed_raw), alpha
    )
    verdicts = [test["within_PI"] for test in mean_tests]
    return {
        "confirmation_test": {
            "confirmed": None if None in verdicts else all(verdicts),
            "mean_tests": mean_tests,
            "results": results,
            "all_within_PI": all(r["within_PI"] for r in results),
            "confidence_level": 1.0 - alpha,
            "scale": _scale_label(transform_info),
        }
    }
