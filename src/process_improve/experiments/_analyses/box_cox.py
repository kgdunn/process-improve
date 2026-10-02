# (c) Kevin Dunn, 2010-2026. MIT License.
"""Box-Cox transformation diagnostic (ENG-02).

The power ``lambda`` is chosen as Box and Cox (1964) and Montgomery (*Design and
Analysis of Experiments*, section 15.1.1) choose it: by maximising the profile
likelihood of the fitted model, which is the same as minimising the residual sum of
squares of the geometric-mean scaled response ``(y**lambda - 1) / (lambda * gm**(lambda - 1))``.
Choosing ``lambda`` to make the response's own distribution look normal, ignoring the
factors, is a different question: in a design with large effects the marginal
response is multimodal, and that ``lambda`` says nothing about the model's errors.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize, stats

#: The range of powers searched for the maximum-likelihood ``lambda``.
_LAMBDA_RANGE = (-3.0, 3.0)
#: Grid step for the profile; the maximum is then refined between grid points.
_LAMBDA_STEP = 0.001

#: Conventional powers, and the name of the transform each one gives.
_NAMED_POWERS = {0.5: "square root", 0.0: "log transform", -1.0: "inverse"}


def _scaled_power(y: np.ndarray, lmbda: float, log_gm: float) -> np.ndarray:
    """Geometric-mean scaled Box-Cox transform, whose residual SS is comparable across ``lambda``."""
    log_y = np.log(y)
    if lmbda == 0.0:
        return np.exp(log_gm) * log_y
    return np.expm1(lmbda * log_y) / lmbda * np.exp((1.0 - lmbda) * log_gm)


def box_cox_profile(y: np.ndarray, exog: np.ndarray, alpha: float = 0.05) -> tuple[float, tuple[float, float]]:
    """Maximum-likelihood Box-Cox ``lambda`` for the model ``exog``, with its likelihood-ratio interval.

    Parameters
    ----------
    y : numpy.ndarray
        The response, all values positive.
    exog : numpy.ndarray
        The model matrix the response is regressed on.
    alpha : float
        One minus the confidence level of the interval.

    Returns
    -------
    tuple[float, tuple[float, float]]
        ``lambda`` and its ``1 - alpha`` interval, ``{lambda : 2 (L(lambda_hat) - L(lambda)) <= chi2(1)}``,
        clipped to the searched range of -3 to 3.
    """
    y = np.asarray(y, dtype=float)
    exog = np.asarray(exog, dtype=float)
    n = len(y)
    log_gm = float(np.mean(np.log(y)))
    residual_maker = np.identity(n) - exog @ np.linalg.pinv(exog)

    def sse(lmbda: float) -> float:
        resid = residual_maker @ _scaled_power(y, lmbda, log_gm)
        return max(float(resid @ resid), 1e-300)

    grid = np.round(np.arange(_LAMBDA_RANGE[0], _LAMBDA_RANGE[1] + _LAMBDA_STEP / 2, _LAMBDA_STEP), 6)
    log_lik = np.array([-n / 2 * np.log(sse(float(g))) for g in grid])
    best = int(np.argmax(log_lik))
    lo = float(grid[max(best - 1, 0)])
    hi = float(grid[min(best + 1, len(grid) - 1)])
    refined = optimize.minimize_scalar(sse, bounds=(lo, hi), method="bounded")
    lmbda = (
        float(refined.x) if refined.success and sse(float(refined.x)) <= sse(float(grid[best])) else float(grid[best])
    )
    max_log_lik = -n / 2 * np.log(sse(lmbda))

    cutoff = max_log_lik - stats.chi2.ppf(1 - alpha, df=1) / 2
    inside = np.flatnonzero(log_lik >= cutoff)
    ci_low = float(grid[inside[0]]) if inside.size else lmbda
    ci_high = float(grid[inside[-1]]) if inside.size else lmbda
    return lmbda, (min(ci_low, lmbda), max(ci_high, lmbda))


def box_cox_transform(y: np.ndarray, lmbda: float) -> np.ndarray:
    """Box-Cox transform ``(y**lambda - 1) / lambda`` (``log(y)`` at ``lambda = 0``), as scipy defines it."""
    y = np.asarray(y, dtype=float)
    if lmbda == 0.0:
        return np.log(y)
    return np.expm1(lmbda * np.log(y)) / lmbda


def _recommendation(lmbda: float, ci: tuple[float, float]) -> str:
    """Name the conventional transform the interval supports, preferring none at all."""
    low, high = ci
    if low <= 1.0 <= high:
        return "no transform (lambda = 1 is inside the confidence interval)"
    inside = [power for power in _NAMED_POWERS if low <= power <= high]
    if inside:
        return _NAMED_POWERS[min(inside, key=lambda power: abs(power - lmbda))]
    return f"power transform (lambda={lmbda:.3f})"


def _run_box_cox(
    design_df: pd.DataFrame,
    response_col: str,
    exog: np.ndarray | None = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Box-Cox diagnostic: the model-based ``lambda``, its confidence interval and a recommendation.

    Parameters
    ----------
    design_df : pandas.DataFrame
        Data holding the (untransformed) response.
    response_col : str
        Name of the response column.
    exog : numpy.ndarray or None
        The fitted model's matrix. ``None`` uses an intercept alone, which reduces to
        choosing ``lambda`` from the response's own distribution.
    alpha : float
        One minus the confidence level of the interval for ``lambda``.

    Returns
    -------
    dict[str, Any]
        ``{"box_cox": {...}}`` with ``lambda``, ``lambda_ci``, ``confidence_level``,
        ``transformed_values`` and ``recommendation``, or an ``error``.
    """
    y = np.asarray(design_df[response_col], dtype=float)

    if np.any(y <= 0):
        return {"box_cox": {"error": "Box-Cox requires all positive response values."}}

    model_matrix = np.ones((len(y), 1)) if exog is None else np.asarray(exog, dtype=float)
    lmbda, ci = box_cox_profile(y, model_matrix, alpha)

    return {
        "box_cox": {
            "lambda": lmbda,
            "lambda_ci": [ci[0], ci[1]],
            "confidence_level": 1.0 - alpha,
            "transformed_values": [float(v) for v in box_cox_transform(y, lmbda)],
            "recommendation": _recommendation(lmbda, ci),
        }
    }
