# (c) Kevin Dunn, 2010-2026. MIT License.
"""Residual diagnostics: normality, independence, homoscedasticity (ENG-02)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy import stats
from statsmodels.regression.linear_model import RegressionResultsWrapper

logger = logging.getLogger(__name__)


def _run_residual_diagnostics(ols_result: RegressionResultsWrapper) -> dict[str, Any]:
    """Residual diagnostics: normality, independence, homoscedasticity."""
    residuals = ols_result.resid.values
    fitted = ols_result.fittedvalues.values

    # Shapiro-Wilk normality test
    n = len(residuals)
    if n >= 3:
        sw_stat, sw_p = stats.shapiro(residuals)
    else:
        sw_stat, sw_p = None, None

    # Durbin-Watson (independence)
    from statsmodels.stats.stattools import durbin_watson  # noqa: PLC0415

    dw = float(durbin_watson(residuals))

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

    return {
        "residual_diagnostics": {
            "shapiro_wilk": {
                "statistic": float(sw_stat) if sw_stat else None,
                "p_value": float(sw_p) if sw_p else None,
            },
            "durbin_watson": dw,
            "breusch_pagan": {
                "statistic": float(bp_stat) if bp_stat else None,
                "p_value": float(bp_p) if bp_p else None,
            },
            "cooks_distance": [float(c) for c in cooks_d],
            "leverage": [float(h) for h in leverage],
            "residuals": [float(r) for r in residuals],
            "fitted_values": [float(f) for f in fitted],
        }
    }
