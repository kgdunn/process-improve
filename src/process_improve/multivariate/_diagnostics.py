# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.
"""Post-fit diagnostics and matrix-correlation helpers (ENG-01).

Functions that read a *fitted* PCA / PLS model and summarise it: variable
importance (VIP), squared cosine, observation contributions, the eigenvalue
summary, supplementary-variable projection, and the RV / modified-RV matrix
correlation coefficients. They consume model attributes only (via duck typing),
so they depend just on :mod:`process_improve.multivariate._common` and ``center``
from :mod:`process_improve.multivariate._preprocessing`, and annotate the model
parameter as ``BaseEstimator`` to avoid importing the estimator modules.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist
from sklearn.base import BaseEstimator
from sklearn.utils import Bunch

from ._common import DataMatrix, _align_to_fit_features, epsqrt
from ._preprocessing import center

# These diagnostics operate on a *fitted* PCA or PLS model via duck typing (they
# read attributes such as ``scores_`` / ``spe_`` after guarding with
# ``hasattr``). They are annotated against ``BaseEstimator`` rather than the
# concrete ``PCA`` / ``PLS`` classes so this leaf module does not import the
# estimator modules - which import these functions in turn - and so avoids a
# module-level import cycle.


def vip(model: BaseEstimator, n_components: int | None = None) -> pd.Series:
    r"""Calculate Variable Importance in Projection (VIP) scores.

    Works with fitted :class:`PCA` and :class:`PLS` models. For PCA the
    principal-component loadings ``loadings_`` are used as the weight matrix;
    for PLS the X-block weights ``x_weights_`` are used.

    The formula is:

    .. math::

        \\text{VIP}_j = \\sqrt{K \\cdot
            \\frac{\\sum_{a=1}^{A} r2_a \\cdot w_{ja}^2}{\\sum_{a=1}^{A} r2_a}}

    where :math:`K` is the number of features, :math:`A` the number of
    components, :math:`r2_a` the fraction of variance explained by component
    :math:`a`, and :math:`w_{ja}` the weight for feature :math:`j` in
    component :math:`a`.

    Parameters
    ----------
    model : PCA or PLS
        A fitted PCA or PLS model.
    n_components : int or None, default=None
        Number of components to include. ``None`` uses all fitted components.

    Returns
    -------
    pd.Series
        VIP scores indexed by feature names, named ``"VIP"``.

    Raises
    ------
    ValueError
        If the model is not fitted, if neither ``x_weights_`` nor
        ``loadings_`` is found, or if *n_components* is out of range.

    Examples
    --------
    >>> pls = PLS(n_components=3).fit(X_scaled, Y_scaled)
    >>> pls.vip()          # bound convenience method after fit()
    >>> vip(pls)           # or call the standalone function directly
    >>> pca = PCA(n_components=3).fit(X_scaled)
    >>> pca.vip(n_components=2)
    """
    if not hasattr(model, "r2_per_component_"):
        msg = "Model is not fitted. Call fit() before computing VIP."
        raise ValueError(msg)

    if hasattr(model, "x_weights_"):
        weights: pd.DataFrame = model.x_weights_
    elif hasattr(model, "loadings_"):
        weights = model.loadings_
    else:
        msg = "Model must have 'x_weights_' (PLS) or 'loadings_' (PCA) to compute VIP."
        raise ValueError(msg)

    r2: np.ndarray = model.r2_per_component_.values
    w: np.ndarray = weights.values  # (n_features, total_components)

    total_components = w.shape[1]
    if n_components is None:
        n_components = total_components
    elif not (1 <= n_components <= total_components):
        msg = f"n_components must be between 1 and {total_components}, got {n_components}."
        raise ValueError(msg)

    w = w[:, :n_components]
    r2 = r2[:n_components]

    n_features = w.shape[0]
    r2_row = r2.reshape(1, -1)  # (1, n_components)
    vip_values = np.sqrt(n_features * np.sum(r2_row * w**2, axis=1) / np.sum(r2))

    return pd.Series(vip_values, index=weights.index, name="VIP")


def _select_response(beta: pd.DataFrame, response: str | int | None) -> str:
    """Resolve a response selector to a single column label of ``beta``."""
    columns = list(beta.columns)
    if response is None:
        if len(columns) != 1:
            msg = f"This model has several responses ({columns}); pass response=<name> to pick one."
            raise ValueError(msg)
        return columns[0]
    if isinstance(response, (int, np.integer)) and response not in columns:
        if not (0 <= int(response) < len(columns)):
            msg = f"response index {response} is out of range for {len(columns)} responses."
            raise ValueError(msg)
        return columns[int(response)]
    if response not in columns:
        msg = f"response {response!r} is not one of the model responses {columns}."
        raise ValueError(msg)
    return response  # type: ignore[return-value]


def _target_projection_arrays(
    model: BaseEstimator, X: DataMatrix, response: str | int | None
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, str]:
    """Return ``(X_aligned, t_tp, p_tp, w_tp, response_label)`` for the TP component.

    The target-projection direction is the (unit-normalised) regression vector
    ``b`` for the chosen response, read from ``model.beta_coefficients_``; the
    scores are ``t = X w_tp`` and the loadings ``p = X^T t / (t^T t)``. ``X``
    must be preprocessed the same way as the training data (the same convention
    as :func:`t2_contributions` / :func:`spe_contributions`).
    """
    if not hasattr(model, "beta_coefficients_"):
        msg = "Model is not fitted, or is not a PLS model with 'beta_coefficients_'."
        raise ValueError(msg)
    beta = model.beta_coefficients_
    label = _select_response(beta, response)
    # The TP direction must live in the SAME space as the X the scores are
    # computed from. ``beta_coefficients_`` is deliberately rescaled to raw
    # X -> raw Y units in fit(); with scale=True (the default) that vector is
    # NOT a direction in the model's internal (scaled) space, and normalising
    # it does not repair the per-column unit mismatch. Use the scaled-space
    # regression vector (direct_weights_ @ y_loadings_.T) instead, and map X
    # into the same internal space via the model's own scaler when present.
    b_scaled = model.direct_weights_.to_numpy(dtype=float) @ model.y_loadings_.to_numpy(dtype=float).T
    b = b_scaled[:, list(beta.columns).index(label)]
    norm_b = float(np.sqrt(b @ b))
    if norm_b <= epsqrt:
        msg = f"The regression vector for response {label!r} is ~0; it is not predicted by X."
        raise ValueError(msg)
    w_tp = b / norm_b

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    X = _align_to_fit_features(X, beta.index)
    x_scaler = getattr(model, "_x_scaler", None)
    if x_scaler is not None:
        X = x_scaler.transform(X)
    X_values = X.to_numpy(dtype=float)
    t_tp = X_values @ w_tp
    ttt = float(t_tp @ t_tp)
    if ttt <= epsqrt:
        msg = "The target-projected scores have ~0 variance; cannot form the TP loading."
        raise ValueError(msg)
    p_tp = (X_values.T @ t_tp) / ttt
    return X, t_tp, p_tp, w_tp, label


def target_projection(model: BaseEstimator, X: DataMatrix, response: str | int | None = None) -> Bunch:
    r"""Target-projected (TP) component of a fitted PLS model for one response.

    Target projection (Kvalheim and Karstang, 1989) rotates the PLS solution so
    that a *single* latent component carries all of the predictive information
    for one response. The component points along the regression vector
    :math:`b` (the column of ``beta_coefficients_`` for that response):

    .. math::

        w_{\text{TP}} = \frac{b}{\lVert b \rVert}, \qquad
        t_{\text{TP}} = X\, w_{\text{TP}}, \qquad
        p_{\text{TP}} = \frac{X^\top t_{\text{TP}}}{t_{\text{TP}}^\top t_{\text{TP}}}.

    The TP component is the basis for the selectivity ratio
    (:func:`selectivity_ratio`).

    Parameters
    ----------
    model : PLS
        A fitted PLS model (must expose ``beta_coefficients_``).
    X : array-like of shape (n_samples, n_features)
        Preprocessed data, scaled the same way as the training data (for
        example with :class:`MCUVScaler`).
    response : str or int or None, default=None
        Which response (Y column) to project onto. ``None`` is allowed only for
        a single-response model; otherwise pass the response label (or its
        integer position).

    Returns
    -------
    sklearn.utils.Bunch
        With fields ``scores`` (pd.Series, the TP scores per sample),
        ``loadings`` (pd.Series, the TP loading per feature), ``weights``
        (pd.Series, the unit TP weight per feature) and ``response`` (the
        resolved response label).

    Raises
    ------
    ValueError
        If the model is not a fitted PLS, the response selector is invalid, or
        the regression vector / TP scores are degenerate (~0).

    References
    ----------
    Kvalheim, O. M. and Karstang, T. V. (1989). Interpretation of latent-variable
    regression models. *Chemometrics and Intelligent Laboratory Systems*, 7(1-2),
    39-51.

    Examples
    --------
    >>> pls = PLS(n_components=3).fit(X_scaled, y_scaled)
    >>> tp = pls.target_projection(X_scaled)        # bound convenience method
    >>> tp.scores.head()

    See Also
    --------
    selectivity_ratio : Per-variable explained/residual ratio on the TP component.
    """
    X_df, t_tp, p_tp, w_tp, label = _target_projection_arrays(model, X, response)
    return Bunch(
        scores=pd.Series(t_tp, index=X_df.index, name="TP score"),
        loadings=pd.Series(p_tp, index=X_df.columns, name="TP loading"),
        weights=pd.Series(w_tp, index=X_df.columns, name="TP weight"),
        response=label,
    )


def _selectivity_ratio_one(
    model: BaseEstimator, X: DataMatrix, response: str | int | None, conf_level: float
) -> pd.Series:
    """Compute the selectivity ratio per feature for a single response (public-API helper)."""
    X_df, t_tp, p_tp, _w_tp, label = _target_projection_arrays(model, X, response)
    X_values = X_df.to_numpy(dtype=float)
    ttt = float(t_tp @ t_tp)
    ss_explained = (p_tp**2) * ttt  # per-feature explained sum of squares on the TP component
    residuals = X_values - np.outer(t_tp, p_tp)
    ss_residual = (residuals**2).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        sr = ss_explained / ss_residual
    sr[~np.isfinite(sr)] = 0.0  # a feature with no residual (or no signal) gets 0 here

    series = pd.Series(sr, index=X_df.columns, name="selectivity_ratio")
    # F-based critical value (Rajalahti et al., 2009): SR_j above this is
    # "significant" at conf_level, with N-2 and N-3 degrees of freedom. The
    # sensory layer prefers a permutation test, so this is advisory metadata.
    n_samples = X_values.shape[0]
    if n_samples > 3:
        series.attrs["f_critical"] = float(f_dist.ppf(conf_level, dfn=n_samples - 2, dfd=n_samples - 3))
    else:
        series.attrs["f_critical"] = float("nan")
    series.attrs["conf_level"] = float(conf_level)
    series.attrs["response"] = label
    return series


def selectivity_ratio(
    model: BaseEstimator,
    X: DataMatrix,
    response: str | int | None = None,
    *,
    conf_level: float = 0.95,
) -> pd.Series | pd.DataFrame:
    r"""Compute the selectivity ratio of each feature on the target-projected component.

    The selectivity ratio (Rajalahti et al., 2009) ranks each feature by how
    much of its variance the *predictive* (target-projected) direction explains.
    On the TP component (:func:`target_projection`), for feature :math:`j`:

    .. math::

        \text{SR}_j = \frac{\text{SS}_{\text{explained},j}}
                           {\text{SS}_{\text{residual},j}}
                    = \frac{p_{\text{TP},j}^2\, (t_{\text{TP}}^\top t_{\text{TP}})}
                           {\sum_i (x_{ij} - t_{\text{TP},i}\, p_{\text{TP},j})^2}.

    A large SR means the feature is well aligned with the predictive direction.
    Unlike VIP, it is a true explained/residual variance ratio and can be
    compared against an F distribution. Note that two collinear features carry
    near-identical SR: the selectivity ratio ranks predictive relevance, it does
    not break ties between mutually collinear features.

    Parameters
    ----------
    model : PLS
        A fitted PLS model (must expose ``beta_coefficients_``).
    X : array-like of shape (n_samples, n_features)
        Preprocessed data, scaled the same way as the training data (for
        example with :class:`MCUVScaler`).
    response : str or int or None, default=None
        Which response to compute SR for. ``None`` returns a feature-by-response
        DataFrame when the model has several responses, or a Series for a
        single-response model.
    conf_level : float, default=0.95
        Confidence level for the advisory F-based critical value, attached to
        the result's ``.attrs["f_critical"]``.

    Returns
    -------
    pd.Series or pd.DataFrame
        Selectivity ratios indexed by feature. A Series for one response (with
        ``f_critical`` / ``conf_level`` / ``response`` in ``.attrs``), or a
        feature-by-response DataFrame when ``response`` is ``None`` and the
        model has several responses.

    Raises
    ------
    ValueError
        If the model is not a fitted PLS or the response selector is invalid.

    References
    ----------
    Rajalahti, T., Arneberg, R., Berven, F. S., Myhr, K.-M., Ulvik, R. J. and
    Kvalheim, O. M. (2009). Biomarker discovery in mass spectral profiles by
    means of selectivity ratio plot. *Chemometrics and Intelligent Laboratory
    Systems*, 95(1), 35-48.

    Examples
    --------
    >>> pls = PLS(n_components=3).fit(X_scaled, y_scaled)
    >>> pls.selectivity_ratio(X_scaled).sort_values(ascending=False).head()

    See Also
    --------
    target_projection : The target-projected component the ratio is built on.
    vip : Variable Importance in Projection, an alternative importance measure.
    """
    beta = getattr(model, "beta_coefficients_", None)
    if response is None and beta is not None and beta.shape[1] > 1:
        return pd.DataFrame({col: _selectivity_ratio_one(model, X, col, conf_level) for col in beta.columns})
    return _selectivity_ratio_one(model, X, response, conf_level)


def squared_cosine(model: BaseEstimator, n_components: int | None = None) -> pd.DataFrame:
    r"""Calculate the squared cosine (cos2): quality of representation of observations.

    Works with fitted :class:`PCA` and :class:`PLS` models. The squared cosine
    of observation :math:`i` on component :math:`a` is the squared score
    divided by that observation's total variation budget:

    .. math::

        \\cos^2_{ia} = \\frac{t_{ia}^2}
            {\\sum_{a=1}^{A} t_{ia}^2 + \\text{SPE}_i^2}

    where :math:`t_{ia}` is the score and :math:`\\text{SPE}_i` the residual
    (squared prediction error) of the observation. Across all components the
    cos2 values plus the residual fraction sum to 1. A value close to 1 means
    the observation is well represented on that component. For :class:`PCA`,
    whose loadings are orthonormal, the denominator equals the squared distance
    of the observation from the origin, matching the classical definition.

    cos2 complements the existing diagnostics: Hotelling's T² measures distance
    *within* the model plane, SPE measures distance *to* it, and cos2 reports
    how much of an observation's total variation a given component captures.

    Parameters
    ----------
    model : PCA or PLS
        A fitted PCA or PLS model.
    n_components : int or None, default=None
        Number of components to return. ``None`` returns all fitted components.

    Returns
    -------
    pd.DataFrame
        cos2 values of shape (n_samples, n_components), indexed by sample.

    Raises
    ------
    ValueError
        If the model is not fitted, or if *n_components* is out of range.

    Examples
    --------
    >>> pca = PCA(n_components=3).fit(X_scaled)
    >>> pca.squared_cosine()              # bound convenience method after fit()
    >>> squared_cosine(pca, n_components=2)  # or call the function directly
    """
    if not hasattr(model, "scores_") or not hasattr(model, "spe_"):
        msg = "Model is not fitted. Call fit() before computing the squared cosine."
        raise ValueError(msg)

    scores = model.scores_
    total_components = scores.shape[1]
    if n_components is None:
        n_components = total_components
    elif not (1 <= n_components <= total_components):
        msg = f"n_components must be between 1 and {total_components}, got {n_components}."
        raise ValueError(msg)

    score_ss = scores.to_numpy(dtype=float) ** 2
    residual_ss = model.spe_.to_numpy(dtype=float)[:, -1] ** 2
    total_ss = score_ss.sum(axis=1) + residual_ss

    with np.errstate(invalid="ignore", divide="ignore"):
        cos2 = score_ss[:, :n_components] / total_ss[:, None]
    cos2[~np.isfinite(cos2)] = 0.0

    return pd.DataFrame(cos2, index=scores.index, columns=scores.columns[:n_components])


def observation_contributions(model: BaseEstimator, n_components: int | None = None) -> pd.DataFrame:
    r"""Calculate the contribution of each observation to each component.

    Works with fitted :class:`PCA` and :class:`PLS` models. The contribution of
    observation :math:`i` to component :math:`a` is its squared score divided
    by the sum of squared scores of all observations on that component:

    .. math::

        \\text{contribution}_{ia} = \\frac{t_{ia}^2}{\\sum_{i=1}^{N} t_{ia}^2}

    Values lie between 0 and 1 and each column sums to 1, so a contribution
    well above the average :math:`1/N` flags an observation that strongly
    shapes that component.

    Note that this is *not* the same diagnostic as the ``score_contributions``
    method, despite the similar name. ``score_contributions`` is *per-variable*
    and signed: it decomposes one observation's position in score space back
    onto the original variables ("which **variables** explain why this
    observation sits where it does?"). ``observation_contributions`` is
    *per-observation* and non-negative: it reports each observation's share of
    a component's total inertia ("which **observations** most strongly shape
    this component?"). The two are orthogonal views of the same score matrix
    and are not interchangeable.

    Parameters
    ----------
    model : PCA or PLS
        A fitted PCA or PLS model.
    n_components : int or None, default=None
        Number of components to return. ``None`` returns all fitted components.

    Returns
    -------
    pd.DataFrame
        Contributions of shape (n_samples, n_components), indexed by sample.
        Each column sums to 1.

    Raises
    ------
    ValueError
        If the model is not fitted, or if *n_components* is out of range.

    Examples
    --------
    >>> pca = PCA(n_components=3).fit(X_scaled)
    >>> pca.observation_contributions()
    >>> observation_contributions(pca, n_components=2)

    See Also
    --------
    PCA.score_contributions : The per-variable counterpart - decomposes one
        observation's score-space position back onto the original variables.
    """
    if not hasattr(model, "scores_"):
        msg = "Model is not fitted. Call fit() before computing observation contributions."
        raise ValueError(msg)

    scores = model.scores_
    total_components = scores.shape[1]
    if n_components is None:
        n_components = total_components
    elif not (1 <= n_components <= total_components):
        msg = f"n_components must be between 1 and {total_components}, got {n_components}."
        raise ValueError(msg)

    score_ss = scores.to_numpy(dtype=float) ** 2
    column_ss = score_ss.sum(axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        contributions = score_ss[:, :n_components] / column_ss[:n_components]
    contributions[~np.isfinite(contributions)] = 0.0

    return pd.DataFrame(contributions, index=scores.index, columns=scores.columns[:n_components])


def _contribution_inputs(model: BaseEstimator, X: DataMatrix) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Align ``X`` to the fitted features and return ``(X_aligned, R, P)``.

    ``R`` generates the scores (``T = X @ R``) and ``P`` reconstructs the X-block
    (``X_hat = T @ P.T``). For PCA both are the loadings; for PLS ``R`` is the
    direct (rotated) weights ``direct_weights_`` and ``P`` is ``x_loadings_``.
    PLS is recognised by the presence of ``direct_weights_``.
    """
    if not hasattr(model, "scores_"):
        msg = "Model is not fitted. Call fit() before computing contributions."
        raise ValueError(msg)

    is_pls = hasattr(model, "direct_weights_")
    reconstruction = model.x_loadings_ if is_pls else model.loadings_
    directions = model.direct_weights_ if is_pls else model.loadings_
    P = np.asarray(reconstruction, dtype=float)
    R = np.asarray(directions, dtype=float)

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    X = _align_to_fit_features(X, reconstruction.index)
    return X, R, P


def t2_contributions(model: BaseEstimator, X: DataMatrix, components: list[int] | None = None) -> pd.DataFrame:
    r"""Per-variable contributions to Hotelling's :math:`T^2`.

    Works with fitted :class:`PCA` and :class:`PLS` models. Decomposes each
    observation's :math:`T^2` onto the original variables. The contribution of
    variable :math:`k` for observation :math:`i` is

    .. math::

        c^{T^2}_{ik} = x_{ik} \sum_{a} \frac{t_{ia}}{s_a^2}\, R_{ka},

    where :math:`t_{ia}` are the scores, :math:`s_a^2` is the score variance of
    component :math:`a` (``scaling_factor_for_scores_`` squared) and :math:`R` is
    the score-generating matrix (loadings for PCA, ``direct_weights_`` for PLS,
    so that :math:`T = XR`). Summed over the variables this telescopes to
    :math:`\sum_a t_{ia}^2 / s_a^2`, i.e. the observation's :math:`T^2`. The
    values are signed; a large magnitude flags a variable that drives the
    observation away from the model centre. This is the standard MSPC
    diagnostic (Westerhuis, Gurden and Smilde, 2000).

    Parameters
    ----------
    model : PCA or PLS
        A fitted PCA or PLS model.
    X : array-like of shape (n_samples, n_features)
        Preprocessed data, scaled the same way as the training data (for
        example with :class:`MCUVScaler`). Passing the training data reproduces
        the model's stored ``hotellings_t2_``.
    components : list of int, optional
        **1-based** component indices to decompose over, matching the model's
        column convention. ``None`` (default) uses all fitted components, so the
        row sums equal the cumulative :math:`T^2`.

    Returns
    -------
    pd.DataFrame
        Signed contributions of shape (n_samples, n_features). Each row sums to
        the observation's :math:`T^2` over the selected components.

    Examples
    --------
    >>> pca = PCA(n_components=3).fit(X_scaled)
    >>> contrib = pca.t2_contributions(X_scaled)
    >>> contrib.sum(axis=1)  # equals pca.hotellings_t2_.iloc[:, -1]

    See Also
    --------
    spe_contributions : The residual-space counterpart.
    PCA.score_contributions : Decomposes a single score-space movement.
    """
    X_df, R, _ = _contribution_inputs(model, X)
    X_values = X_df.to_numpy(dtype=float)
    A = R.shape[1]

    if components is None:
        idx = np.arange(A)
    else:
        idx = np.asarray(components, dtype=int) - 1
        if idx.size == 0 or idx.min() < 0 or idx.max() >= A:
            msg = f"components must be 1-based indices within 1..{A}, got {components}."
            raise ValueError(msg)

    s = np.asarray(model.scaling_factor_for_scores_, dtype=float)[idx]
    # ``s_a == 0`` for a degenerate component would give inf/NaN; clamp the
    # divisor so such a component contributes nothing rather than poisoning the
    # result (mirrors ``score_contributions(weighted=True)``).
    s2 = np.where(s**2 > epsqrt, s**2, 1.0)

    scores = X_values @ R[:, idx]  # (n, len(idx))
    contributions = X_values * ((scores / s2) @ R[:, idx].T)
    return pd.DataFrame(contributions, index=X_df.index, columns=X_df.columns)


def spe_contributions(model: BaseEstimator, X: DataMatrix) -> pd.DataFrame:
    r"""Per-variable squared-prediction-error (SPE / DModX) contributions.

    Works with fitted :class:`PCA` and :class:`PLS` models. Returns the signed
    residual of each variable after reconstructing the X-block from the full
    model:

    .. math::

        e_{ik} = x_{ik} - \hat{x}_{ik}, \qquad \hat{X} = T P^\top,

    where :math:`P` is the reconstruction loadings (``loadings_`` for PCA,
    ``x_loadings_`` for PLS). The squared residuals sum across variables to the
    observation's SPE; equivalently ``(spe_contributions(X) ** 2).sum(axis=1)``
    equals the stored ``spe_`` (final column) squared. The signs show whether a
    variable sits above or below its reconstruction, which is the standard SPE
    contribution plot used to diagnose why an observation has a high residual.

    Parameters
    ----------
    model : PCA or PLS
        A fitted PCA or PLS model.
    X : array-like of shape (n_samples, n_features)
        Preprocessed data, scaled the same way as the training data. Passing the
        training data reproduces the model's stored ``spe_``.

    Returns
    -------
    pd.DataFrame
        Signed per-variable residuals of shape (n_samples, n_features). The
        squared row sums equal the observation's SPE.

    Examples
    --------
    >>> pca = PCA(n_components=2).fit(X_scaled)
    >>> resid = pca.spe_contributions(X_scaled)
    >>> (resid ** 2).sum(axis=1)  # equals pca.spe_.iloc[:, -1] ** 2

    See Also
    --------
    t2_contributions : The :math:`T^2` (score-space) counterpart.
    """
    X_df, R, P = _contribution_inputs(model, X)
    X_values = X_df.to_numpy(dtype=float)
    scores = X_values @ R
    residuals = X_values - scores @ P.T
    return pd.DataFrame(residuals, index=X_df.index, columns=X_df.columns)


def _score_contribution_terms(model: BaseEstimator, X: DataMatrix, component: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Return ``(X_aligned, r_a)`` for the requested 1-based ``component``."""
    X_df, R, _ = _contribution_inputs(model, X)
    A = R.shape[1]
    if not (1 <= int(component) <= A):
        msg = f"component must be a 1-based index within 1..{A}, got {component}."
        raise ValueError(msg)
    return X_df, R[:, int(component) - 1]


_SCORE_CONTRIBUTIONS_USAGE = """\
score_contributions() takes the preprocessed data X, not a score vector.

A contribution is x_ik * R_ka, so the per-variable terms sum to the score being
decomposed. That needs the observation's data, which a score vector does not
carry. See Miller, Swanson and Heckler (1994).

    model.score_contributions(X)              # all observations, component 1
    model.score_contributions(X).iloc[i]      # just observation i

To compare two observations or two groups:

    model.group_contributions(X, group=[a], reference=[b])

For contributions to Hotelling's T-squared, which pools every component:

    model.t2_contributions(X)
"""


_SCORE_VECTOR_KEYWORDS = frozenset({"t_end", "components", "weighted"})


def _reject_score_vector_call(X: object, keywords: dict) -> None:
    """Raise a usage error if called with a score vector instead of ``X``."""
    looks_like_scores = isinstance(X, (pd.Series, np.ndarray, list, tuple)) and np.ndim(X) == 1
    if looks_like_scores or (set(keywords) & _SCORE_VECTOR_KEYWORDS):
        raise TypeError(_SCORE_CONTRIBUTIONS_USAGE)
    if keywords:
        unexpected = ", ".join(sorted(keywords))
        msg = f"score_contributions() got an unexpected keyword argument: {unexpected}."
        raise TypeError(msg)


def score_contributions(
    model: BaseEstimator,
    X: DataMatrix,
    component: int = 1,
    scaling: str = "none",
    **deprecated: object,
) -> pd.DataFrame:
    r"""Per-variable contributions to a single score, :math:`t_a`.

    Works with fitted :class:`PCA` and :class:`PLS` models. A score is a
    weighted sum of the (preprocessed) variables, so it splits exactly into one
    term per variable. The contribution of variable :math:`k` to the score of
    observation :math:`i` on component :math:`a` is

    .. math::

        c_{ik}^{(a)} = x_{ik}\, R_{ka}, \qquad
        \sum_{k=1}^{K} c_{ik}^{(a)} = t_{ia},

    where :math:`R` is the score-generating matrix (``loadings_`` for PCA,
    ``direct_weights_`` for PLS, so that :math:`T = XR`). This is the
    contribution of Miller, Swanson and Heckler (1994); the generalisation from
    PCA loadings to any latent-variable model's score-generating weights follows
    Westerhuis, Gurden and Smilde (2000).

    The distinction from a loading plot is the point of the diagnostic. A
    loading :math:`R_{ka}` describes the whole data set; a contribution
    :math:`x_{ik} R_{ka}` describes *one observation*, and a variable with a
    large loading contributes nothing when that observation sits at its mean.
    Ranking variables by loading can therefore point at a different cause than
    ranking them by contribution.

    Parameters
    ----------
    model : PCA or PLS
        A fitted PCA or PLS model.
    X : array-like of shape (n_samples, n_features)
        Preprocessed data, scaled the same way as the training data (for
        example with :class:`MCUVScaler`). Passing the training data reproduces
        the model's stored ``scores_``.
    component : int, default=1
        **1-based** component index whose score is decomposed, matching the
        model's column convention.
    scaling : {"none", "maximum", "within"}, default="none"
        Presentation scaling from Miller, Swanson and Heckler (1994). ``"none"``
        returns the raw contributions, which sum to the score. ``"maximum"``
        divides by the largest absolute contribution anywhere in ``X``, so a bar
        of :math:`\pm 1` marks the most extreme variable-observation pair in the
        data set. ``"within"`` divides each row by the sum of its absolute
        contributions, so each row is on a common footing. Both scalings leave
        the *pattern* of bars within a row unchanged; neither preserves the sum
        to the score.
    **deprecated
        Rejected. Captures ``t_end``, ``components`` and ``weighted`` so that a
        call passing a score vector rather than ``X`` raises a
        :class:`TypeError` explaining the correct usage. Passing a 1-D ``X``
        raises the same error.

    Returns
    -------
    pd.DataFrame
        Signed contributions of shape (n_samples, n_features). With the default
        ``scaling="none"``, each row sums to that observation's score on the
        selected component.

    Examples
    --------
    >>> pca = PCA(n_components=2).fit(X_scaled)
    >>> contrib = pca.score_contributions(X_scaled, component=1)
    >>> contrib.sum(axis=1)  # equals pca.scores_[1]
    >>> contrib.loc["33"].abs().sort_values()  # what makes observation 33 extreme

    References
    ----------
    Miller, P., Swanson, R.E. and Heckler, C.E. (1994). "Contribution plots: a
    missing link in multivariate quality control." Applied Mathematics and
    Computer Science, 8(4), 775-792.

    Westerhuis, J.A., Gurden, S.P. and Smilde, A.K. (2000). "Generalized
    contribution plots in multivariate statistical process monitoring."
    Chemometrics and Intelligent Laboratory Systems, 51(1), 95-114.

    See Also
    --------
    group_contributions : The same decomposition for a group of observations, or
        for the difference between two groups.
    t2_contributions : Decomposes Hotelling's :math:`T^2`, which pools all
        components rather than reading one at a time.
    spe_contributions : The residual-space counterpart.
    """
    _reject_score_vector_call(X, deprecated)
    X_df, r_a = _score_contribution_terms(model, X, component)
    contributions = X_df.to_numpy(dtype=float) * r_a

    if scaling == "maximum":
        largest = float(np.abs(contributions).max())
        contributions = contributions / (largest if largest > epsqrt else 1.0)
    elif scaling == "within":
        row_total = np.abs(contributions).sum(axis=1, keepdims=True)
        contributions = contributions / np.where(row_total > epsqrt, row_total, 1.0)
    elif scaling != "none":
        msg = f"scaling must be one of 'none', 'maximum' or 'within', got {scaling!r}."
        raise ValueError(msg)

    return pd.DataFrame(contributions, index=X_df.index, columns=X_df.columns)


def group_contributions(  # noqa: PLR0913 - group/reference are sugar for weights
    model: BaseEstimator,
    X: DataMatrix,
    group: Sequence | None = None,
    reference: Sequence | None = None,
    component: int = 1,
    weights: Sequence | None = None,
) -> pd.Series:
    r"""Per-variable contributions to a group's average score, or to a shift.

    The group form of :func:`score_contributions`. Combining the data rows
    before multiplying by the score-generating weights answers "what do these
    observations have in common?" rather than "why is this one observation
    unusual?", which is the question a cluster on a score plot, or a level shift
    part-way through a data set, actually poses.

    In general any linear combination of the rows may be used
    (Miller, Swanson and Heckler, 1994):

    .. math::

        c_k = \Bigl(\sum_i w_i x_{ik}\Bigr) R_{ka}, \qquad
        \sum_k c_k = \sum_i w_i t_{ia}.

    The common cases have their own arguments. With ``group`` alone the weights
    are :math:`1/n_G` over the group, comparing its mean against the model
    centre. With ``group`` and ``reference`` they are :math:`+1/n_G` and
    :math:`-1/n_H`, so the contributions sum to the difference in average score,
    which is the level-shift diagnostic of the paper's Figure 9. Pass ``weights``
    directly for anything else: the paper suggests the first-order orthogonal
    polynomial when a run of batches is drifting rather than stepping.

    Parameters
    ----------
    model : PCA or PLS
        A fitted PCA or PLS model.
    X : array-like of shape (n_samples, n_features)
        Preprocessed data, scaled the same way as the training data.
    group : sequence, optional
        Index labels of the observations of interest, or a boolean mask the
        same length as ``X``. Selection is by label, not by position; pass
        ``X.index[...]`` to select positionally. Required unless ``weights``
        is given.
    reference : sequence, optional
        Index labels selecting the observations to compare against. ``None``
        (default) compares the group against the model centre.
    component : int, default=1
        **1-based** component index whose score is decomposed.
    weights : sequence, optional
        One weight per row of ``X``, giving the linear combination directly.
        Mutually exclusive with ``group`` / ``reference``.

    Returns
    -------
    pd.Series
        Signed contributions, one per variable. Sums to the weighted combination
        of the scores: the group's average score, the difference in average
        score between the two groups, or :math:`\sum_i w_i t_{ia}`.

    Examples
    --------
    >>> pca = PCA(n_components=2).fit(X_scaled)
    >>> # Five batches that cluster together on the score plot:
    >>> pca.group_contributions(X_scaled, group=[31, 142, 147, 220, 221])
    >>> # What shifted at batch 74? (ten batches either side, by position)
    >>> pca.group_contributions(
    ...     X_scaled, group=X_scaled.index[64:74], reference=X_scaled.index[74:84]
    ... )
    >>> # A run of batches drifting rather than stepping: weight by a
    >>> # first-order orthogonal polynomial over the run.
    >>> slope = np.zeros(len(X_scaled))
    >>> slope[40:60] = np.arange(20) - 9.5
    >>> pca.group_contributions(X_scaled, weights=slope, component=3)

    See Also
    --------
    score_contributions : The single-observation form.
    """
    X_df, r_a = _score_contribution_terms(model, X, component)

    if weights is not None:
        if group is not None or reference is not None:
            msg = "Pass either weights, or group/reference, not both."
            raise ValueError(msg)
        w = np.asarray(weights, dtype=float)
        if w.ndim != 1 or w.size != len(X_df):
            msg = f"weights must have one entry per row of X ({len(X_df)}), got shape {w.shape}."
            raise ValueError(msg)
    else:
        if group is None:
            msg = "Pass group (optionally with reference), or weights."
            raise ValueError(msg)
        w = np.zeros(len(X_df), dtype=float)
        w += _mean_weights(X_df, group, "group")
        if reference is not None:
            w -= _mean_weights(X_df, reference, "reference")

    deviation = w @ X_df.to_numpy(dtype=float)
    return pd.Series(deviation * r_a, index=X_df.columns, name="group_contributions")


def _mean_weights(X_df: pd.DataFrame, selector: Sequence, name: str) -> np.ndarray:
    """Row weights that average the selected rows: ``1/n`` on each, 0 elsewhere."""
    selected = _select_rows(X_df, selector, name)
    if len(selected) == 0:
        msg = f"{name} selected no observations from X."
        raise ValueError(msg)
    w = np.zeros(len(X_df), dtype=float)
    w[X_df.index.get_indexer_for(selected.index)] = 1.0 / len(selected)
    return w


def _select_rows(X_df: pd.DataFrame, selector: Sequence, name: str) -> pd.DataFrame:
    """Select rows of ``X_df`` by index label or by boolean mask.

    Deliberately label-based only. Falling back to positional selection when a
    label happens to be missing would make the meaning depend on the data: on a
    frame indexed 1..54, ``[0, 1, 2]`` would be positions (0 is not a label)
    while ``[10, 11, 12]`` would be labels, silently selecting a different set
    of rows. Callers who want positions pass ``X.index[...]``.
    """
    values = list(selector)
    if values and all(isinstance(v, (bool, np.bool_)) for v in values):
        if len(values) != len(X_df):
            msg = f"{name} is a boolean mask of length {len(values)}, but X has {len(X_df)} rows."
            raise ValueError(msg)
        return X_df.loc[np.asarray(values, dtype=bool)]
    missing = [v for v in values if v not in X_df.index]
    if missing:
        msg = (
            f"{name} contains entries that are not index labels of X: {missing}. "
            "Selection is by index label or boolean mask; to select by position, "
            "pass X.index[...] instead."
        )
        raise ValueError(msg)
    return X_df.loc[values]


def eigenvalue_summary(model: BaseEstimator) -> pd.DataFrame:
    """Summarize the variance captured by each component as a tidy table.

    Works with fitted :class:`PCA` and :class:`PLS` models. Returns one row per
    component, collecting ``explained_variance_``, ``r2_per_component_`` and
    ``r2_cumulative_`` into a single table.

    Parameters
    ----------
    model : PCA or PLS
        A fitted PCA or PLS model.

    Returns
    -------
    pd.DataFrame
        Indexed by component, with columns ``eigenvalue`` (the variance of the
        component scores), ``percent_variance`` and ``cumulative_percent``. For
        PCA the percentages refer to variance in X; for PLS they refer to the
        variance in Y explained by each component.

    Raises
    ------
    ValueError
        If the model is not fitted.

    Examples
    --------
    >>> pca = PCA(n_components=3).fit(X_scaled)
    >>> pca.eigenvalue_summary()
    >>> eigenvalue_summary(pca)
    """
    if not hasattr(model, "r2_per_component_"):
        msg = "Model is not fitted. Call fit() before computing the eigenvalue summary."
        raise ValueError(msg)

    summary = pd.DataFrame(
        {
            "eigenvalue": np.asarray(model.explained_variance_, dtype=float),
            "percent_variance": model.r2_per_component_.to_numpy(dtype=float) * 100.0,
            "cumulative_percent": model.r2_cumulative_.to_numpy(dtype=float) * 100.0,
        },
        index=model.r2_per_component_.index,
    )
    summary.index.name = "component"
    return summary


def project_variables(model: BaseEstimator, supplementary_data: DataMatrix) -> pd.DataFrame:
    """Project supplementary (passive) variables onto a fitted model.

    Works with fitted :class:`PCA` and :class:`PLS` models. Supplementary
    variables are extra columns that did not take part in fitting the model but
    were measured on the *same observations*. Each supplementary variable is
    represented by its correlation with each component's scores, the standard
    representation for passive quantitative variables. This is the column-wise
    counterpart of ``transform``, which projects supplementary *rows* (new
    observations).

    Parameters
    ----------
    model : PCA or PLS
        A fitted PCA or PLS model.
    supplementary_data : array-like of shape (n_samples, n_supplementary)
        Passive variables measured on the same observations used to fit the
        model. Must have the same number of rows as the training data.

    Returns
    -------
    pd.DataFrame
        Correlations of shape (n_supplementary, n_components): the coordinate
        of each supplementary variable on each component.

    Raises
    ------
    ValueError
        If the model is not fitted, or if *supplementary_data* does not have
        the same number of rows as the training data.

    Examples
    --------
    >>> pca = PCA(n_components=3).fit(X_scaled)
    >>> pca.project_variables(passive_columns)
    >>> project_variables(pca, passive_columns)
    """
    if not hasattr(model, "scores_"):
        msg = "Model is not fitted. Call fit() before projecting variables."
        raise ValueError(msg)

    scores = model.scores_
    if not isinstance(supplementary_data, pd.DataFrame):
        supplementary_data = pd.DataFrame(supplementary_data)

    if supplementary_data.shape[0] != scores.shape[0]:
        msg = (
            f"Supplementary data must have {scores.shape[0]} rows (the number of "
            f"observations used to fit the model), got {supplementary_data.shape[0]}."
        )
        raise ValueError(msg)

    xs = supplementary_data.to_numpy(dtype=float)
    t = scores.to_numpy(dtype=float)
    xs_centered = xs - xs.mean(axis=0, keepdims=True)
    t_centered = t - t.mean(axis=0, keepdims=True)
    xs_norm = np.sqrt((xs_centered**2).sum(axis=0))
    t_norm = np.sqrt((t_centered**2).sum(axis=0))

    with np.errstate(invalid="ignore", divide="ignore"):
        correlations = (xs_centered.T @ t_centered) / np.outer(xs_norm, t_norm)
    correlations[~np.isfinite(correlations)] = 0.0

    return pd.DataFrame(correlations, index=supplementary_data.columns, columns=scores.columns)


def _column_centred_array(data: DataMatrix, name: str) -> np.ndarray:
    """Coerce *data* to a 2-D float array with every column mean-centred."""
    arr = data.to_numpy(dtype=float) if isinstance(data, pd.DataFrame) else np.asarray(data, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        msg = f"{name} must be 1- or 2-dimensional, got {arr.ndim} dimensions."
        raise ValueError(msg)
    return np.asarray(center(arr), dtype=float)


def _matrix_correlation(X: DataMatrix, Y: DataMatrix, *, modified: bool) -> float:
    """Shared core of the RV and modified-RV (RV2) coefficients."""
    x_centred = _column_centred_array(X, "X")
    y_centred = _column_centred_array(Y, "Y")
    if x_centred.shape[0] != y_centred.shape[0]:
        msg = f"X and Y must have the same number of rows; got {x_centred.shape[0]} and {y_centred.shape[0]}."
        raise ValueError(msg)

    s_x = x_centred @ x_centred.T
    s_y = y_centred @ y_centred.T
    if modified:
        np.fill_diagonal(s_x, 0.0)
        np.fill_diagonal(s_y, 0.0)

    denominator = np.sqrt(np.sum(s_x * s_x) * np.sum(s_y * s_y))
    if denominator == 0.0:
        return float("nan")
    return float(np.sum(s_x * s_y) / denominator)


def rv_coefficient(X: DataMatrix, Y: DataMatrix) -> float:
    """Compute the RV coefficient between two data blocks.

    The RV coefficient (Robert and Escoufier, 1976) measures how much common
    structure two matrices, measured on the *same observations*, share. It is a
    multivariate generalisation of the squared Pearson correlation: it compares
    the observation-by-observation configuration matrices :math:`XX^T` and
    :math:`YY^T` rather than individual variables.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features_x)
        First data block.
    Y : array-like of shape (n_samples, n_features_y)
        Second data block. Must have the same number of rows as *X*; the
        number of columns may differ.

    Returns
    -------
    float
        The RV coefficient in the range [0, 1]. A value of 1 means the two
        blocks describe the same configuration of observations up to a
        rotation and an overall scaling; 0 means no shared structure.
        ``nan`` is returned if either block has no variance.

    Notes
    -----
    Each column is mean-centred internally, since the RV coefficient is defined
    on centred data. The blocks are **not** scaled; scale the columns yourself
    (for example with :class:`MCUVScaler`) when the variables have different
    units.

    For high-dimensional data (many more variables than observations) the RV
    coefficient is biased upwards and tends towards 1 even for unrelated
    blocks. Use :func:`rv2_coefficient` in that regime.

    References
    ----------
    Robert, P. and Escoufier, Y. (1976). A unifying tool for linear
    multivariate statistical methods: the RV-coefficient. *Journal of the
    Royal Statistical Society, Series C*, 25(3), 257-265.

    See Also
    --------
    rv2_coefficient : Modified RV coefficient, unbiased for high-dimensional data.

    Examples
    --------
    >>> rv_coefficient(X, Y)
    >>> rv_coefficient(X, X)  # 1.0: a block is perfectly correlated with itself
    """
    return _matrix_correlation(X, Y, modified=False)


def rv2_coefficient(X: DataMatrix, Y: DataMatrix) -> float:
    """Compute the modified RV coefficient (RV2) between two data blocks.

    The modified RV coefficient (Smilde et al., 2009) is a variant of
    :func:`rv_coefficient` that removes the diagonals of the configuration
    matrices :math:`XX^T` and :math:`YY^T` before comparing them. This removes
    the upward bias that makes the ordinary RV coefficient tend towards 1 for
    high-dimensional data, so RV2 stays near 0 for genuinely unrelated blocks.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features_x)
        First data block.
    Y : array-like of shape (n_samples, n_features_y)
        Second data block. Must have the same number of rows as *X*; the
        number of columns may differ.

    Returns
    -------
    float
        The modified RV coefficient, in the range [-1, 1]. A value of 1 means
        the two blocks describe the same configuration of observations; values
        near 0 mean no shared structure, and small negative values can occur.
        ``nan`` is returned if either block has no variance.

    Notes
    -----
    Each column is mean-centred internally but the blocks are **not** scaled;
    scale the columns yourself (for example with :class:`MCUVScaler`) when the
    variables have different units.

    References
    ----------
    Smilde, A. K., Kiers, H. A. L., Bijlsma, S., Rubingh, C. M. and van Erk,
    M. J. (2009). Matrix correlations for high-dimensional data: the modified
    RV-coefficient. *Bioinformatics*, 25(3), 401-405.

    See Also
    --------
    rv_coefficient : The original RV coefficient.

    Examples
    --------
    >>> rv2_coefficient(X, Y)
    """
    return _matrix_correlation(X, Y, modified=True)
