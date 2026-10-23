# (c) Kevin Dunn, 2010-2026. MIT License.
"""Residual diagnostics: normality, independence, homoscedasticity (ENG-02)."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.stats.stattools import durbin_watson

from process_improve.univariate.classical import autocorrelation

logger = logging.getLogger(__name__)

#: The run-order check when it is not run: no order known, or no residual to check.
_NOT_IN_RUN_ORDER: dict[str, Any] = {
    "known": False,
    "lag1_autocorrelation": None,
    "std_error": None,
    "durbin_watson": None,
    "serially_dependent": None,
}


def _in_run_order(residuals: np.ndarray, run_order: Sequence[float] | None) -> tuple[dict[str, Any], str | None]:
    """Return the residuals' lag-1 autocorrelation and Durbin-Watson statistic in run order, and a note.

    The run order is known when ``run_order`` gives each residual a distinct number; the
    residuals are sorted on it. Otherwise the check is not run, and the note says that the
    only order there is, the rows', is the one ``durbin_watson`` is worked in.

    Parameters
    ----------
    residuals : numpy.ndarray
        The residuals, in the rows' order.
    run_order : Sequence[float] or None
        Each residual's position in the run order, or ``None`` when it is not known.

    Returns
    -------
    tuple[dict[str, Any], str or None]
        The ``run_order`` entry of the diagnostics, and the note to report, if any.
    """
    order = None if run_order is None else pd.to_numeric(pd.Series(run_order), errors="coerce").to_numpy(float)
    if order is None or len(order) != len(residuals) or np.isnan(order).any() or len(np.unique(order)) < len(order):
        reason = (
            "No RunOrder column gives"
            if run_order is None
            else "The RunOrder column has missing, non-numeric or repeated values, so it does not give"
        )
        return dict(_NOT_IN_RUN_ORDER), (
            f"{reason} the order the runs were made in: durbin_watson is worked in the rows' order, and the "
            "residuals are not checked for serial dependence in run order."
        )
    in_order = residuals[np.argsort(order, kind="stable")]
    if np.ptp(in_order) == 0:  # residuals without spread have no autocorrelation (0/0)
        return {**_NOT_IN_RUN_ORDER, "known": True}, None
    found = autocorrelation(in_order, lags=1)
    r1, std_error = found["autocorrelation"][0], found["std_error"]
    # Box, Hunter and Hunter (1978) randomize the run order because successive runs of a process are
    # often serially correlated (r1 = -0.29 in their record of 210 yields, p. 63), and randomizing is
    # what lets the runs be analysed as if they were independent. For independent errors r1 has a
    # standard error of about 1/sqrt(N) (p. 587), so |r1| > 2/sqrt(N) is about the 5% level.
    dependent = abs(r1) > 2 * std_error
    check = {
        "known": True,
        "lag1_autocorrelation": r1,
        "std_error": std_error,
        "durbin_watson": float(durbin_watson(in_order)),
        "serially_dependent": bool(dependent),
    }
    note = (
        f"In run order the residuals have a lag-1 autocorrelation of {r1:.2f}, more than twice its standard error "
        f"1/sqrt({len(in_order)}) = {std_error:.2f}: successive runs look serially dependent, as from a drift or a "
        "shift in the process, and the tests, which take the runs as independent, may mislead. Plot the residuals "
        "against run order; if the runs were not made in a randomized order, a drift can also bias the effects."
    )
    return check, note if dependent else None


def _run_residual_diagnostics(
    ols_result: RegressionResultsWrapper, run_order: Sequence[float] | None = None
) -> dict[str, Any]:
    """Residual diagnostics: normality, independence, homoscedasticity.

    Each run's residual is also given studentized: ``studentized_residuals`` divides it
    by its own standard error, ``s * sqrt(1 - h)`` with ``h`` its leverage (internally
    studentized, as Design-Expert's "Student Residual"), and
    ``externally_studentized_residuals`` takes ``s`` from the fit without that run
    (R's ``rstudent``, Design-Expert's "Outlier t"), which is t-distributed on one
    degree of freedom fewer than the residual's.

    ``durbin_watson`` is worked on the residuals in the rows' order, which is the run
    order only when the rows were entered in the order the runs were made. The entry
    ``run_order`` checks independence in the order the argument of that name gives (a
    design's ``RunOrder`` column): ``known`` is whether that order was given, one distinct
    number per residual; ``lag1_autocorrelation`` is the residuals' lag-1 autocorrelation
    in run order, and ``std_error`` its standard error for independent errors, 1/sqrt(N);
    ``durbin_watson`` is the Durbin-Watson statistic in run order; and
    ``serially_dependent`` is whether the lag-1 autocorrelation is more than twice its
    standard error, when a ``note`` also says so. Without a usable run order the four are
    ``None``, and the ``note`` says that the only order is the rows', ``durbin_watson``'s.

    A saturated model (no residual degrees of freedom) fits every run exactly, so its
    residuals are rounding noise: the tests are then not run, their entries are
    ``None``, and a ``note`` says why.

    Parameters
    ----------
    ols_result : RegressionResultsWrapper
        The least-squares fit.
    run_order : Sequence[float] or None
        Each residual's position in the order the runs were made, in the order of the
        residuals; ``None`` when it is not known.

    Returns
    -------
    dict[str, Any]
        ``{"residual_diagnostics": {...}}``.
    """
    residuals = ols_result.resid.values
    fitted = ols_result.fittedvalues.values
    if ols_result.df_resid <= 0:
        return {
            "residual_diagnostics": {
                "shapiro_wilk": {"statistic": None, "p_value": None},
                "durbin_watson": None,
                "run_order": dict.fromkeys(_NOT_IN_RUN_ORDER),
                "breusch_pagan": {"statistic": None, "p_value": None},
                "cooks_distance": [None] * len(residuals),
                "studentized_residuals": [None] * len(residuals),
                "externally_studentized_residuals": [None] * len(residuals),
                "leverage": [float(h) for h in ols_result.get_influence().hat_matrix_diag],
                "residuals": [float(r) for r in residuals],
                "fitted_values": [float(f) for f in fitted],
                "note": (
                    "Saturated model - no residual degrees of freedom, so the residuals are rounding noise "
                    "and the residual tests are not run. Fit fewer terms, or use analysis_type='lenth_method'."
                ),
            }
        }

    # Shapiro-Wilk normality test
    n = len(residuals)
    if n >= 3:
        sw_stat, sw_p = stats.shapiro(residuals)
    else:
        sw_stat, sw_p = None, None

    # Durbin-Watson (independence), in the rows' order; the run order is checked below.
    dw = float(durbin_watson(residuals))
    in_run_order, note = _in_run_order(residuals, run_order)

    # Breusch-Pagan (homoscedasticity). The test can legitimately fail to compute
    # (singular exog, too few residuals); record None but log so the failure is
    # not silent, and do not swallow unexpected error types.
    try:
        from statsmodels.stats.diagnostic import het_breuschpagan  # noqa: PLC0415

        bp_stat, bp_p, _bp_f, _bp_fp = het_breuschpagan(residuals, ols_result.model.exog)
    except (ImportError, ValueError, ZeroDivisionError, np.linalg.LinAlgError) as exc:
        logger.warning("Breusch-Pagan test could not be computed: %s", exc)
        bp_stat, bp_p = None, None

    # Cook's distance
    influence = ols_result.get_influence()
    cooks_d = influence.cooks_distance[0]

    # Leverage
    leverage = influence.hat_matrix_diag

    # ``is not None``, not truthiness: a p-value that underflows to exactly
    # 0.0 is the MOST significant possible result, and the previous truthiness
    # test rendered it as "not available".
    return {
        "residual_diagnostics": {
            "shapiro_wilk": {
                "statistic": float(sw_stat) if sw_stat is not None else None,
                "p_value": float(sw_p) if sw_p is not None else None,
            },
            "durbin_watson": dw,
            "run_order": in_run_order,
            "breusch_pagan": {
                "statistic": float(bp_stat) if bp_stat is not None else None,
                "p_value": float(bp_p) if bp_p is not None else None,
            },
            "cooks_distance": [float(c) for c in cooks_d],
            "studentized_residuals": [float(r) for r in influence.resid_studentized_internal],
            "externally_studentized_residuals": [float(r) for r in influence.resid_studentized_external],
            "leverage": [float(h) for h in leverage],
            "residuals": [float(r) for r in residuals],
            "fitted_values": [float(f) for f in fitted],
            **({"note": note} if note else {}),
        }
    }
