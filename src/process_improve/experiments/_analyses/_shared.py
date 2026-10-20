# (c) Kevin Dunn, 2010-2026. MIT License.
"""Always-on model-summary helpers used by :func:`analyze_experiment` (ENG-02)."""

from __future__ import annotations

import numpy as np
from statsmodels.regression.linear_model import OLSResults, RegressionResultsWrapper


def _press(ols_result: RegressionResultsWrapper) -> float:
    """Return the prediction sum of squares: each run's residual when the fit leaves it out, squared and summed."""
    return float(np.sum(ols_result.get_influence().resid_press ** 2))


def _compute_pred_r_squared(ols_result: RegressionResultsWrapper) -> float:
    """Predicted R² from PRESS residuals, over the total sum of squares about the mean."""
    ss_total = float(np.sum((ols_result.model.endog - ols_result.model.endog.mean()) ** 2))
    if ss_total == 0:
        return 0.0
    return 1.0 - _press(ols_result) / ss_total


def _sums_of_squares(ols_result: RegressionResultsWrapper) -> dict[str, float | None]:
    """Return the fit's sums of squares, corrected and crude, and the coefficient of variation (#789 B1).

    Sources print the totals in two conventions: about the mean, and the crude sum of the
    squared responses, which adds the correction for the mean, n times the mean squared.
    ``ss_model`` is on the basis ``r_squared`` and the overall F test use: about the mean,
    or about zero for a model whose terms do not span a constant. ``ss_model_uncorrected``
    is the sum of the squared fitted values, the model's sum about zero in either case.
    """
    y = np.asarray(ols_result.model.endog, dtype=float)
    fitted = np.asarray(ols_result.fittedvalues, dtype=float)
    mean = float(y.mean())
    ss_mean = len(y) * mean**2
    ss_total = float(np.sum((y - mean) ** 2))
    deviation = float(np.sqrt(ols_result.mse_resid)) if ols_result.df_resid > 0 else float("nan")
    return {
        "ss_total": ss_total,
        "ss_mean": ss_mean,
        "ss_total_uncorrected": float(y @ y),
        "ss_model": float(ols_result.ess),
        "ss_model_uncorrected": float(fitted @ fitted),
        "ss_residual": float(ols_result.ssr),
        "coefficient_of_variation": 100.0 * deviation / mean if mean != 0 and np.isfinite(deviation) else None,
    }


def _without_blocks(ols_result: RegressionResultsWrapper, blocks: tuple[str, ...]) -> dict[str, float | None]:
    """Return R², adjusted R² and prediction R² over the total less the blocks' sum of squares (#789 B1).

    Design-Expert reports these for a blocked design: the blocks are fitted first, and
    their sum of squares is taken out of the total, so R² measures what the model explains
    of the variation within blocks. Empty when no block contrast is in the model.
    """
    names = [str(name) for name in ols_result.model.exog_names]
    columns = [names.index(name) for name in blocks if name in names]
    if not columns:
        return {}
    y = np.asarray(ols_result.model.endog, dtype=float)
    exog = np.asarray(ols_result.model.exog, dtype=float)
    blocks_only = np.column_stack([np.ones(len(y)), exog[:, columns]])
    coefficients, _, rank, _ = np.linalg.lstsq(blocks_only, y, rcond=None)
    within = float(np.sum((y - blocks_only @ coefficients) ** 2))  # the total less the blocks' sum of squares
    df_within, df_resid = len(y) - int(rank), float(ols_result.df_resid)
    rss = float(ols_result.ssr)
    defined = within > 0 and df_resid > 0  # else the blocks explain every difference, or nothing is left to test
    return {
        "ss_blocks": float(np.sum((y - y.mean()) ** 2)) - within,
        "r_squared_without_blocks": 1.0 - rss / within if defined else None,
        "r_squared_adj_without_blocks": 1.0 - (rss / df_resid) / (within / df_within) if defined else None,
        "r_squared_pred_without_blocks": 1.0 - _press(ols_result) / within if defined else None,
    }


def _no_intercept_note(ols_result: RegressionResultsWrapper) -> str | None:
    """Say, for a model with no intercept term, that its ANOVA rows do not add up to a model sum of squares."""
    if "Intercept" in ols_result.model.exog_names:
        return None
    return (
        "The model has no intercept term, so its rows do not add up to a model sum of squares: each is adjusted for "
        "the other terms. model_summary gives the model's sum of squares about zero, ss_model_uncorrected, beside "
        "ss_total_uncorrected, and on the basis of r_squared, ss_model."
    )


def average_fitted_variance(ols_result: RegressionResultsWrapper, variance: float | None = None) -> float:
    """Return the average variance of the fitted values over the runs, p variance/n (#797 R1).

    The leverages of the runs sum to p, the number of terms the model estimates (the rank of
    its matrix), so the average is p variance/n whatever the design (Box and Wetz).
    ``variance`` is the error variance: the residual mean square unless one is given.
    """
    error = float(ols_result.mse_resid) if variance is None else variance
    return int(ols_result.model.rank) * error / int(ols_result.nobs)


def _compute_adequate_precision(ols_result: RegressionResultsWrapper, variance: float | None = None) -> float:
    """Return adequate precision: the range of the fitted values over their average standard error.

    The average standard error is the square root of :func:`average_fitted_variance`, with
    the same ``variance``. Infinite when the fit leaves no error at all.
    """
    noise = np.sqrt(average_fitted_variance(ols_result, variance))
    if noise == 0:
        return float("inf")
    return float(np.ptp(np.asarray(ols_result.fittedvalues, dtype=float)) / noise)


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
