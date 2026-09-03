# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.
"""Score estimation for observations with missing values, on a fitted model.

The estimators here answer one question: given a fitted latent-variable model
and a new observation in which some features are missing (NaN), what are the
best estimates of that observation's scores? This is the "batch so far"
primitive of online batch monitoring (the future part of the trajectory is
missing by construction), and equally the sensor-failure case in continuous
monitoring.

Three estimators are provided, following Arteaga and Ferrer (2002) and the
batch-monitoring comparison of Garcia-Munoz, Kourti and MacGregor (2004):

- ``"tsr"`` (trimmed score regression, the default): regress the true scores
  on the trimmed scores computed from the observed columns. Statistically the
  strongest of the three in both papers' comparisons, and the inverted matrix
  is only (A x A).
- ``"scp"`` (single-component projection): estimate each component
  sequentially with deflation, using only the observed part of each loading
  vector. Simple and never ill-conditioned, but the weakest estimator, and
  errors propagate through the deflation.
- ``"pmp"`` (projection to the model plane): least-squares fit of the
  observed columns onto the corresponding loading rows. Can be
  ill-conditioned early in a batch, when few columns are observed.

For a fixed missingness pattern each estimator is a fixed linear operator on
the observed columns, exposed by :func:`operator_for_pattern` so callers that
revisit the same pattern (an online monitor at time k, or a mid-course
optimiser treating candidate columns as observed) can precompute the matrix
once. The conditioning of the estimator is a first-class output: nothing
degrades silently, and a documented ``ridge`` option is available for the
regression-based estimators.

References
----------
Arteaga, F. and Ferrer, A., "Dealing with missing data in MSPC: several
methods, different interpretations, some examples", Journal of Chemometrics,
16, 408-418, 2002. https://doi.org/10.1002/cem.750

Garcia-Munoz, S., Kourti, T. and MacGregor, J.F., "Model Predictive
Monitoring for Batch Processes", Industrial & Engineering Chemistry Research,
43, 5929-5941, 2004. https://doi.org/10.1021/ie034020w

Nelson, P.R.C., Taylor, P.A. and MacGregor, J.F., "Missing data methods in
PCA and PLS: score calculations with incomplete observations", Chemometrics
and Intelligent Laboratory Systems, 35, 45-65, 1996.
"""

from __future__ import annotations

import typing

import numpy as np
import pandas as pd
from sklearn.utils import Bunch

from .._linalg import safe_inverse

if typing.TYPE_CHECKING:
    from collections.abc import Hashable, Iterable

PROJECTION_METHODS = ("tsr", "scp", "pmp")
_EPS = np.sqrt(np.finfo(float).eps)


def coerce_observed_mask(observed: object, feature_names: Iterable[Hashable]) -> np.ndarray:
    """Normalise an ``observed`` argument to a boolean mask over the features.

    Accepts a boolean array-like of length ``n_features`` (True = observed),
    or a collection of feature labels to treat as observed; anything else, a
    wrong length, or an unknown label raises ``ValueError``.
    """
    names = feature_names if isinstance(feature_names, pd.Index) else pd.Index(list(feature_names))
    arr = np.asarray(observed)
    if arr.dtype == bool:
        if arr.shape != (len(names),):
            raise ValueError(f"A boolean observed mask must have length {len(names)}; got shape {arr.shape}.")
        return arr
    labels = list(observed)  # type: ignore[call-overload]
    unknown = [item for item in labels if item not in names]
    if unknown:
        raise ValueError(f"observed contains labels that are not model features: {unknown[:5]}.")
    return names.isin(labels)


def _validate_method(method: str) -> str:
    method = str(method).lower()
    if method not in PROJECTION_METHODS:
        raise ValueError(f"method must be one of {list(PROJECTION_METHODS)}; got {method!r}.")
    return method


def operator_for_pattern(  # noqa: PLR0913 - the operator inputs are irreducible
    x_loadings: np.ndarray,
    guide: np.ndarray,
    score_variances: np.ndarray,
    observed: np.ndarray,
    *,
    method: str = "tsr",
    ridge: float = 0.0,
) -> Bunch:
    """Build the linear score-estimation operator for one missingness pattern.

    For a fixed pattern of observed features, every estimator here is a fixed
    linear map from the observed (centred and scaled) values to the score
    estimates: ``t_hat = M @ z_observed``. This function builds that matrix
    once, so callers that reuse a pattern (an online monitor at time sample
    ``k``; a mid-course optimiser at a decision point) do not re-derive it per
    observation.

    Parameters
    ----------
    x_loadings : np.ndarray of shape (n_features, n_components)
        The model's X-block loadings ``P`` (the reconstruction basis:
        ``x_hat = t @ P.T``).
    guide : np.ndarray of shape (n_features, n_components)
        The matrix whose transpose maps complete data to scores:
        ``t = z @ guide``. For PCA this is ``P`` again; for PLS it is the
        direct weights ``R = W (P'W)^{-1}``.
    score_variances : np.ndarray of shape (n_components,)
        Variance of each training score (the model's ``explained_variance_``),
        used by the TSR regression.
    observed : np.ndarray of shape (n_features,), dtype bool
        Mask of the observed features for this pattern.
    method : {"tsr", "scp", "pmp"}, default="tsr"
        The estimator; see the module docstring.
    ridge : float, default=0.0
        Non-negative regularisation added to the diagonal of the matrix being
        inverted (``tsr`` and ``pmp`` only; ``scp`` inverts nothing). Use when
        ``condition_number`` reports near-singularity, at the cost of a small
        bias toward zero scores.

    Returns
    -------
    result : sklearn.utils.Bunch
        With keys ``matrix`` (np.ndarray of shape (n_components, n_observed),
        the operator ``M``), ``condition_number`` (float; for ``tsr`` and
        ``pmp`` the 2-norm condition number of the inverted matrix, for
        ``scp`` the largest per-component loading-norm inflation
        ``||p_a||^2 / ||p_a_observed||^2``; 1.0 means no loss), and
        ``method``.
    """
    method = _validate_method(method)
    if ridge < 0:
        raise ValueError(f"ridge must be non-negative; got {ridge}.")
    observed = np.asarray(observed, dtype=bool)
    if observed.shape != (x_loadings.shape[0],):
        raise ValueError(
            f"observed must be a boolean mask over the {x_loadings.shape[0]} features; got shape {observed.shape}."
        )
    n_observed = int(observed.sum())
    if n_observed == 0:
        raise ValueError("At least one feature must be observed to estimate scores; the observed mask is all-False.")

    p_obs = x_loadings[observed, :]
    g_obs = guide[observed, :]
    A = x_loadings.shape[1]

    if method == "pmp":
        gram = p_obs.T @ p_obs + ridge * np.eye(A)
        condition = float(np.linalg.cond(gram))
        matrix = safe_inverse(gram, what="(P_observed' @ P_observed)") @ p_obs.T
    elif method == "tsr":
        theta = np.diag(np.asarray(score_variances, dtype=float))
        inner = g_obs.T @ p_obs @ theta @ p_obs.T @ g_obs + ridge * np.eye(A)
        condition = float(np.linalg.cond(inner))
        matrix = theta @ p_obs.T @ g_obs @ safe_inverse(inner, what="(TSR inner matrix)") @ g_obs.T
    else:  # scp
        deflate = np.eye(n_observed)
        matrix = np.zeros((A, n_observed))
        worst = 1.0
        for a in range(A):
            p_a = p_obs[:, a]
            denom = float(p_a @ p_a)
            full = float(x_loadings[:, a] @ x_loadings[:, a])
            if denom > _EPS:
                worst = max(worst, full / denom)
                row = (p_a @ deflate) / denom
            else:
                worst = np.inf
                row = np.zeros(n_observed)
            matrix[a, :] = row
            deflate = deflate - np.outer(p_a, row)
        condition = worst

    return Bunch(matrix=matrix, condition_number=condition, method=method)


def project_rows(  # noqa: PLR0913 - mirrors operator_for_pattern
    x_loadings: np.ndarray,
    guide: np.ndarray,
    score_variances: np.ndarray,
    x_scaled: np.ndarray,
    *,
    method: str = "tsr",
    ridge: float = 0.0,
) -> Bunch:
    """Estimate scores, SPE and conditioning for rows that may contain NaN.

    Rows are grouped by their missingness pattern; each pattern's operator is
    built once with :func:`operator_for_pattern` and applied to all its rows.
    Rows with no missing values take the standard complete-data path
    (``t = z @ guide``), so on complete input the scores are bitwise equal to
    the model's usual ``transform``.

    Parameters
    ----------
    x_loadings, guide, score_variances
        As in :func:`operator_for_pattern`.
    x_scaled : np.ndarray of shape (n_rows, n_features)
        The observations in the model's centred and scaled space, NaN for
        missing entries.
    method : {"tsr", "scp", "pmp"}, default="tsr"
    ridge : float, default=0.0

    Returns
    -------
    result : sklearn.utils.Bunch
        With keys ``scores`` (n_rows x n_components), ``spe`` (n_rows; the
        square root of the residual sum of squares over the *observed*
        columns only), ``condition_number`` (n_rows; 1.0-adjacent for
        complete rows), and ``n_observed`` (n_rows, int).
    """
    method = _validate_method(method)
    n_rows, n_features = x_scaled.shape
    A = x_loadings.shape[1]
    scores = np.empty((n_rows, A))
    spe = np.empty(n_rows)
    condition = np.empty(n_rows)
    n_observed = np.empty(n_rows, dtype=int)

    observed_mask = ~np.isnan(x_scaled)
    complete = observed_mask.all(axis=1)
    if complete.any():
        rows = x_scaled[complete]
        scores[complete] = rows @ guide
        residuals = rows - scores[complete] @ x_loadings.T
        spe[complete] = np.sqrt(np.sum(residuals**2, axis=1))
        condition[complete] = 1.0
        n_observed[complete] = n_features

    incomplete_idx = np.flatnonzero(~complete)
    if incomplete_idx.size:
        patterns: dict[bytes, list[int]] = {}
        for i in incomplete_idx:
            patterns.setdefault(observed_mask[i].tobytes(), []).append(int(i))
        for key, row_ids in patterns.items():
            mask = np.frombuffer(key, dtype=bool)
            if not mask.any():
                bad = row_ids[0]
                raise ValueError(f"Row {bad} has no observed features (all-NaN); scores cannot be estimated for it.")
            op = operator_for_pattern(x_loadings, guide, score_variances, mask, method=method, ridge=ridge)
            z_obs = x_scaled[np.ix_(np.asarray(row_ids), np.flatnonzero(mask))]
            t_hat = z_obs @ op.matrix.T
            scores[row_ids] = t_hat
            residual_obs = z_obs - t_hat @ x_loadings[mask, :].T
            spe[row_ids] = np.sqrt(np.sum(residual_obs**2, axis=1))
            condition[row_ids] = op.condition_number
            n_observed[row_ids] = int(mask.sum())

    return Bunch(scores=scores, spe=spe, condition_number=condition, n_observed=n_observed)
