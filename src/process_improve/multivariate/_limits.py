# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.
"""Confidence limits and ellipse geometry for the multivariate package (ENG-01).

Holds the statistical limits used to flag unusual observations in a fitted PCA /
PLS / TPLS / multiblock model: the Hotelling's T2 limit, the squared prediction
error (SPE) limit and its underlying calculation, the per-component score limit,
and the T2 confidence-ellipse coordinates. Depends only on
:mod:`process_improve.multivariate._common`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2, f, norm
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from ._common import epsqrt


def hotellings_t2_limit(conf_level: float = 0.95, n_components: int = 0, n_rows: int = 0) -> float:
    """Return the Hotelling's T2 value at the given level of confidence.

    Parameters
    ----------
    conf_level : float, optional
        Fractional confidence limit, less that 1.00; by default 0.95
    n_components : int
        Number of components in the fitted multivariate model.
    n_rows : int
        Number of rows (observations) used to fit the model. Must be > 0.

    Returns
    -------
    float
        The Hotelling's T2 limit at the given level of confidence. Returns
        ``inf`` when ``n_components == n_rows``.
    """
    if not 0.0 < conf_level < 1.0:
        raise ValueError(f"conf_level must lie in (0, 1); got {conf_level}.")
    if n_rows <= 0:
        raise ValueError(f"n_rows must be positive; got {n_rows}.")
    if n_components == n_rows:
        return float("inf")
    return (
        n_components
        * (n_rows - 1)
        * (n_rows + 1)
        / (n_rows * (n_rows - n_components))
        * f.isf((1 - conf_level), n_components, n_rows - n_components)
    )


def spe_limit(model: BaseEstimator, conf_level: float = 0.95) -> float:
    """Return the squared prediction error limit at the given level of confidence.

    Parameters
    ----------
    model : BaseEstimator
        A fitted multivariate model exposing a ``spe_`` attribute and an
        ``n_components`` attribute (e.g. a fitted PCA or PLS instance).
    conf_level : float, optional
        Fractional confidence limit, less that 1.00; by default 0.95

    Returns
    -------
    float
        The squared prediction error limit at the given level of confidence.
    """
    check_is_fitted(model, "spe_")

    return spe_calculation(
        spe_values=model.spe_.iloc[:, model.n_components - 1],
        conf_level=conf_level,
    )


def spe_calculation(spe_values: np.ndarray, conf_level: float = 0.95) -> float:
    """Return a limit for SPE (squared prediction error) at the given level of confidence.

    Parameters
    ----------
    spe_values : np.ndarray
        The SPE values from the last component in the multivariate model.
    conf_level : float, optional
        The confidence level, by default 0.95, i.e. the 95% confidence level.

    Returns
    -------
    float
        The SPE limit at the requested confidence level, returned on the
        sqrt-of-sum-of-squared-residuals scale (directly comparable to
        entries of ``model.spe_``, which are stored on the same scale).
        Values above this limit indicate observations whose correlation
        structure differs from the training data's.
    """
    if not 0.0 < conf_level < 1.0:
        raise ValueError(f"conf_level must lie in (0, 1); got {conf_level}.")

    # The limit is for the squares (i.e. the sum of the squared errors)
    # I.e. `spe_values` are square-rooted outside this function, so undo that.
    values = spe_values**2
    center_spe = float(values.mean())
    variance_spe = float(values.var(ddof=1))
    # A perfect-fit training set (A == K) or all-equal SPE values produces
    # ``variance_spe == 0`` or ``center_spe == 0``, which would divide by
    # zero below and yield NaN limits silently. Return the centre as the
    # limit -- there is no spread to bound, so anything above the centre
    # is by definition out of family. SEC-21 (#270) sub-item 3.
    if variance_spe <= epsqrt or center_spe <= epsqrt:
        return float(np.sqrt(center_spe))
    g = variance_spe / (2 * center_spe)
    h = (2 * (center_spe**2)) / variance_spe
    # Report square root again as SPE limit
    return np.sqrt(chi2.ppf(conf_level, h) * g)


def score_limit(model: BaseEstimator, conf_level: float = 0.95) -> np.ndarray:
    """Return two-sided confidence limits for each score component.

    The scores of component ``a`` have mean zero and are normally distributed,
    so the symmetric limit at the requested confidence level is
    ``z * std(score_a)``, with ``z`` the standard-normal quantile. A score
    outside ``[-limit, +limit]`` is unusual at that confidence level.

    Parameters
    ----------
    model : BaseEstimator
        A fitted PCA or PLS model exposing a ``scores_`` attribute.
    conf_level : float, optional
        Fractional confidence level in (0, 1); by default 0.95.

    Returns
    -------
    np.ndarray
        Array of length ``n_components`` with the positive score limit for
        each component.

    References
    ----------
    Score limits: the score ``t_a`` is normally distributed, so the limit is
    ``z_{(1 + conf_level) / 2} * s_a``. Equivalently ``(t_a / s_a) ** 2``
    follows an ``F(1, N - 1)`` distribution.
    """
    assert 0.0 < conf_level < 1.0, "conf_level must be a value between (0.0, 1.0)"
    check_is_fitted(model, "scores_")

    scores = np.asarray(model.scores_, dtype=float)
    std_per_component = scores.std(axis=0, ddof=1)
    z = norm.ppf(1 - (1 - conf_level) / 2)
    return z * std_per_component


def ellipse_coordinates(  # noqa: PLR0913
    score_horiz: int,
    score_vert: int,
    conf_level: float = 0.95,
    n_points: int = 100,
    n_components: int = 0,
    scaling_factor_for_scores: pd.Series | None = None,
    n_rows: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Get the (score_horiz, score_vert) coordinate pairs that form the T2 ellipse when
        plotting the score `score_horiz` on the horizontal axis and `score_vert` on the
        vertical axis.

        Scores are referred to by number, starting at 1 and ending with `model.A`


    Parameters
    ----------
    score_horiz : int
        1-based index of the score to plot on the horizontal axis. Must satisfy
        ``1 <= score_horiz <= n_components``.
    score_vert : int
        1-based index of the score to plot on the vertical axis. Must satisfy
        ``1 <= score_vert <= n_components``.
    conf_level : float
        The `conf_level` confidence value: e.g. 0.95 is for the 95% confidence limit.
    n_points : int, optional
        Number of points to use in the ellipse; by default 100.
    n_components : int
        Number of components `A` in the fitted model. Required to look up the
        Hotelling's T^2 limit and to bound `score_horiz`/`score_vert`.
    scaling_factor_for_scores : pd.Series
        Per-component standard deviations of the scores
        (``model.scaling_factor_for_scores_``). Used to scale the ellipse
        axes. Required: the signature defaults to ``None`` for backward
        compatibility, but passing ``None`` (or omitting the argument)
        raises :class:`AssertionError` inside the function.
    n_rows : int
        Number of rows `N` in the data used to fit the model. Required to compute the
        Hotelling's T^2 limit; must be strictly positive.

    Returns
    -------
    tuple of 2 elements; the first for the x-axis; the second for the y-axis.
        Returns `n_points` equispaced points that can be used to plot an ellipse.

    Background
    ----------

    Equation of ellipse in *canonical* form (http://en.wikipedia.org/wiki/Ellipse)

        (t_horiz/s_h)^2 + (t_vert/s_v)^2  =  hotellings_t2_limit_alpha
        s_horiz = stddev(T_horiz)
        s_vert  = stddev(T_vert)
        hotellings_t2_limit_alpha = T2 confidence limit at a given alpha value

    Equation of ellipse, *parametric* form (http://en.wikipedia.org/wiki/Ellipse):

        t_horiz = sqrt(hotellings_t2_limit_alpha)*s_h*cos(t)
        t_vert  = sqrt(hotellings_t2_limit_alpha)*s_v*sin(t)

        where t ranges between 0 and 2*pi.
    """
    if not 1 <= score_horiz <= n_components:
        raise ValueError(
            f"score_horiz must lie in [1, {n_components}]; got {score_horiz}."
        )
    if not 1 <= score_vert <= n_components:
        raise ValueError(
            f"score_vert must lie in [1, {n_components}]; got {score_vert}."
        )
    if not 0 < conf_level < 1:
        raise ValueError(f"conf_level must lie in (0, 1); got {conf_level}.")
    if n_rows <= 0:
        raise ValueError(f"n_rows must be positive; got {n_rows}.")
    assert scaling_factor_for_scores is not None  # required for the ellipse scaling
    s_h = scaling_factor_for_scores.iloc[score_horiz - 1]
    s_v = scaling_factor_for_scores.iloc[score_vert - 1]
    t2_limit_specific = np.sqrt(hotellings_t2_limit(conf_level, n_components=n_components, n_rows=n_rows))
    dt = 2 * np.pi / (n_points - 1)
    steps = np.linspace(0, n_points - 1, n_points)
    x = np.cos(steps * dt) * t2_limit_specific * s_h
    y = np.sin(steps * dt) * t2_limit_specific * s_v
    return x, y
