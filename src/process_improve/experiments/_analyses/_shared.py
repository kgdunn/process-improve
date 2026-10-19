# (c) Kevin Dunn, 2010-2026. MIT License.
"""Always-on model-summary helpers used by :func:`analyze_experiment` (ENG-02)."""

from __future__ import annotations

import numpy as np
from statsmodels.regression.linear_model import OLSResults, RegressionResultsWrapper


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


def with_prior_sigma(ols_result: RegressionResultsWrapper, sigma: float | None) -> RegressionResultsWrapper:
    """Return ``ols_result`` with its standard errors from a prior ``sigma`` instead of the residual (#789 B6).

    The coefficients do not change. Their covariance becomes sigma² (X'X)⁻¹, and the tests
    and intervals read the normal distribution rather than the t, since sigma is not
    estimated from these runs. A saturated fit, which has no residual, gets standard errors
    too. ``ols_result`` itself is returned when ``sigma`` is None.
    """
    if sigma is None:
        return ols_result
    params, unscaled = np.asarray(ols_result.params), np.asarray(ols_result.normalized_cov_params)
    known = OLSResults(ols_result.model, params, unscaled, use_t=False)
    known.scale = float(sigma) ** 2  # set after construction, which clears the cache the scale is kept in
    known.prior_sigma = float(sigma)
    return RegressionResultsWrapper(known)


def has_error_variance(ols_result: RegressionResultsWrapper) -> bool:
    """Return whether the fit has an error variance to test its terms against: a prior sigma, or a residual."""
    return getattr(ols_result, "prior_sigma", None) is not None or int(ols_result.df_resid) > 0


#: The design column that names each run's block. Blocks enter a model as sum-coded
#: (deviation) contrast columns: with two blocks the contrast is a -1/+1 column, which
#: aliases exactly with the interaction it is confounded with, and setting every
#: contrast to zero predicts the average over the blocks.
BLOCK_COL = "Block"

#: The design column that names each run's whole plot in a split-plot design. Runs in one
#: whole plot share its hard-to-change factor settings, and so share that whole plot's error.
WHOLE_PLOT_COL = "WholePlot"


def is_block_term(label: str, block_names: tuple[str, ...] | list[str]) -> bool:
    """Whether a term, or an alias chain led by one, is a block contrast."""
    return any(label == name or label.startswith(f"{name} ") for name in block_names)
