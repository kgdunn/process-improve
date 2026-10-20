# (c) Kevin Dunn, 2010-2026. MIT License.
"""Box-Cox transformation diagnostic (ENG-02).

The power ``lambda`` is chosen as Box and Cox (1964) and Montgomery (*Design and
Analysis of Experiments*, section 15.1.1) choose it: by maximising the profile
likelihood of the fitted model, which is the same as minimising the residual sum of
squares of the geometric-mean scaled response ``(y**lambda - 1) / (lambda * gm**(lambda - 1))``.
Choosing ``lambda`` to make the response's own distribution look normal, ignoring the
factors, is a different question: in a design with large effects the marginal
response is multimodal, and that ``lambda`` says nothing about the model's errors.

The interval for ``lambda`` is the set of powers whose residual sum of squares S(lambda)
stays below a critical value, and two conventions set that value:

- ``"likelihood"``: S_min exp(chi2_1 / n), the likelihood ratio's
  2 (L(lambda_hat) - L(lambda)) <= chi2(1), with n runs;
- ``"t"``: S_min (1 + t^2 / nu), with t on the model's nu residual degrees of freedom: the
  allowance for few degrees of freedom that Box, Hunter and Hunter (1978, section 7.8) and
  Montgomery (Equation 15.2) read their limits off with. It gives the wider interval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import optimize, stats

if TYPE_CHECKING:
    from collections.abc import Callable

#: The range of powers searched for the maximum-likelihood ``lambda``.
_LAMBDA_RANGE = (-3.0, 3.0)
#: Grid step for the profile; the maximum is then refined between grid points.
_LAMBDA_STEP = 0.001

#: Conventional powers, and the name of the transform each one gives.
_NAMED_POWERS = {0.5: "square root", 0.0: "log transform", -1.0: "inverse"}

#: The conventions for the interval of ``lambda``, as the module's text sets them out.
INTERVALS = ("likelihood", "t")


def _scaled_power(y: np.ndarray, lmbda: float, log_gm: float) -> np.ndarray:
    """Geometric-mean scaled Box-Cox transform, whose residual SS is comparable across ``lambda``."""
    log_y = np.log(y)
    if lmbda == 0.0:
        return np.exp(log_gm) * log_y
    return np.expm1(lmbda * log_y) / lmbda * np.exp((1.0 - lmbda) * log_gm)


@dataclass(frozen=True)
class _Profile:
    """SSE(lambda) of the scaled power over the searched range: its least value, where it is, and the grid's values."""

    lmbda: float
    s_min: float
    n_runs: int
    residual_df: int
    sse: Callable[[float], float]
    grid: np.ndarray
    values: np.ndarray


def _profile(y: np.ndarray, exog: np.ndarray) -> _Profile:
    """Return the profile of SSE(lambda) for the model ``exog``, its minimum refined between the grid's points."""
    y = np.asarray(y, dtype=float)
    exog = np.asarray(exog, dtype=float)
    log_gm = float(np.mean(np.log(y)))
    # An orthonormal basis of the model's column space (rank-deficient models too).
    u, singular, _ = np.linalg.svd(exog, full_matrices=False)
    basis = u[:, singular > singular.max(initial=0.0) * max(exog.shape) * np.finfo(float).eps]

    def sse(lmbda: float) -> float:
        z = _scaled_power(y, lmbda, log_gm)
        resid = z - basis @ (basis.T @ z)
        return max(float(resid @ resid), 1e-300)

    grid = np.round(np.arange(_LAMBDA_RANGE[0], _LAMBDA_RANGE[1] + _LAMBDA_STEP / 2, _LAMBDA_STEP), 6)
    values = np.array([sse(float(g)) for g in grid])
    best = int(np.argmin(values))
    lo = float(grid[max(best - 1, 0)])
    hi = float(grid[min(best + 1, len(grid) - 1)])
    refined = optimize.minimize_scalar(sse, bounds=(lo, hi), method="bounded")
    lmbda = float(refined.x) if refined.success and sse(float(refined.x)) <= values[best] else float(grid[best])
    return _Profile(lmbda, sse(lmbda), len(y), len(y) - basis.shape[1], sse, grid, values)


def _critical_sse(profile: _Profile, alpha: float, interval: str) -> float:
    """Return the residual sum of squares that bounds the ``1 - alpha`` interval, by the convention ``interval``."""
    if interval not in INTERVALS:
        raise ValueError(f"interval must be one of {INTERVALS}; got {interval!r}.")
    if interval == "likelihood":
        return profile.s_min * float(np.exp(stats.chi2.ppf(1 - alpha, df=1) / profile.n_runs))
    nu = profile.residual_df
    if nu < 1:
        raise ValueError("The t-based interval needs residual degrees of freedom; the model leaves none.")
    return profile.s_min * (1 + float(stats.t.ppf(1 - alpha / 2, nu)) ** 2 / nu)


def _limits(profile: _Profile, critical: float) -> tuple[float, float]:
    """Return the ends of ``{lambda : SSE(lambda) <= critical}``, each solved for between the grid points around it.

    An end the searched range cuts off stays at the range's end.
    """
    inside = np.flatnonzero(profile.values <= critical)
    if not inside.size:
        return profile.lmbda, profile.lmbda
    grid, first, last = profile.grid, int(inside[0]), int(inside[-1])

    def crossing(left: int) -> float:
        """Solve SSE(lambda) = critical between the grid's points ``left`` and ``left + 1``."""
        return float(optimize.brentq(lambda p: profile.sse(p) - critical, grid[left], grid[left + 1], xtol=1e-12))

    low = float(grid[0]) if first == 0 else crossing(first - 1)
    high = float(grid[-1]) if last == len(grid) - 1 else crossing(last)
    return min(low, profile.lmbda), max(high, profile.lmbda)


def box_cox_profile(
    y: np.ndarray, exog: np.ndarray, alpha: float = 0.05, interval: str = "likelihood"
) -> tuple[float, tuple[float, float]]:
    """Maximum-likelihood Box-Cox ``lambda`` for the model ``exog``, with its interval by the convention named.

    Parameters
    ----------
    y : numpy.ndarray
        The response, all values positive.
    exog : numpy.ndarray
        The model matrix the response is regressed on.
    alpha : float
        One minus the confidence level of the interval.
    interval : {"likelihood", "t"}
        The convention that sets the interval's critical sum of squares (see the module's text).

    Returns
    -------
    tuple[float, tuple[float, float]]
        ``lambda`` and its ``1 - alpha`` interval, the powers whose SSE(lambda) is at most the critical value,
        each end solved for, and clipped to the searched range of -3 to 3.

    Raises
    ------
    ValueError
        If ``interval`` is not one of :data:`INTERVALS`, or is ``"t"`` for a model that leaves no residual degrees of
        freedom.
    """
    profile = _profile(y, exog)
    return profile.lmbda, _limits(profile, _critical_sse(profile, alpha, interval))


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
        ``{"box_cox": {...}}`` with ``lambda``; ``lambda_ci``, the likelihood ratio's interval;
        ``confidence_level``; ``sse``, the least residual sum of squares of the scaled power, on
        ``residual_df`` degrees of freedom; ``intervals``, each convention's ``lambda_ci`` and its
        ``critical_sse``, keyed by name (``"t"`` is None when the model leaves no residual degrees
        of freedom); ``transformed_values`` and ``recommendation``. Or an ``error``.
    """
    y = np.asarray(design_df[response_col], dtype=float)

    if np.any(y <= 0):
        return {"box_cox": {"error": "Box-Cox requires all positive response values."}}

    model_matrix = np.ones((len(y), 1)) if exog is None else np.asarray(exog, dtype=float)
    profile = _profile(y, model_matrix)
    intervals: dict[str, Any] = {}
    for name in INTERVALS:
        try:
            critical = _critical_sse(profile, alpha, name)
        except ValueError:  # the t allowance, on a model that leaves no residual degrees of freedom
            intervals[name] = None
            continue
        intervals[name] = {"lambda_ci": list(_limits(profile, critical)), "critical_sse": critical}
    ci = intervals["likelihood"]["lambda_ci"]

    return {
        "box_cox": {
            "lambda": profile.lmbda,
            "lambda_ci": ci,
            "confidence_level": 1.0 - alpha,
            "sse": profile.s_min,
            "residual_df": profile.residual_df,
            "intervals": intervals,
            "transformed_values": [float(v) for v in box_cox_transform(y, profile.lmbda)],
            "recommendation": _recommendation(profile.lmbda, (ci[0], ci[1])),
        }
    }
