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

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

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


def t2_contributions(
    model: BaseEstimator, X: DataMatrix, components: list[int] | None = None
) -> pd.DataFrame:
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
