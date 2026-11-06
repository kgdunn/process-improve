# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.
"""Confidence limits and ellipse geometry for the multivariate package (ENG-01).

Holds the statistical limits used to flag unusual observations in a fitted PCA /
PLS / TPLS / multiblock model: the Hotelling's T2 limit, the squared prediction
error (SPE) limit and its underlying calculation, the per-component score limit,
and the T2 confidence-ellipse coordinates. Depends only on
:mod:`process_improve.multivariate._common`.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from scipy.stats import chi2, f
from scipy.stats import t as t_dist
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from ._common import epsqrt, user_stacklevel

#: The rows a Hotelling's T² limit can judge (#978): the rows the model was fitted to, or new rows.
JudgedRows = Literal["fitted", "new"]

#: What ``judged_rows=None`` resolves to. It becomes ``"fitted"`` in 3.0, as announced in 2.4.0 (#978).
_JUDGED_ROWS_BY_DEFAULT: JudgedRows = "new"


def _judged_rows(judged_rows: JudgedRows | None) -> JudgedRows:
    """Return the kind of row ``judged_rows`` names, with ``None`` resolved to the default."""
    if judged_rows is None:
        return _JUDGED_ROWS_BY_DEFAULT
    if judged_rows not in ("fitted", "new"):
        raise ValueError(f'judged_rows must be "fitted", "new" or None; got {judged_rows!r}.')
    return judged_rows


def _t2_limit(conf_level: float, n_components: int, n_rows: int, judged_rows: JudgedRows) -> float:
    """Return the T² limit for validated arguments; see :func:`hotellings_t2_limit`."""
    if n_components == 0:
        # With no components T2 is zero for every observation, and so is its limit: the
        # formulas below tend to 0 as A -> 0, but scipy cannot evaluate an F or a beta
        # distribution with a first shape of 0 and returned NaN.
        return 0.0
    if n_components == n_rows:
        return float("inf")  # pragma: no mutate - float('INF') is the same value
    if judged_rows == "new":
        return (
            n_components
            * (n_rows - 1)
            * (n_rows + 1)
            / (n_rows * (n_rows - n_components))
            * f.isf((1 - conf_level), n_components, n_rows - n_components)
        )
    # A fitted row's T² can never exceed (N - 1)^2 / N. With N = A + 1 every fitted row sits
    # exactly there, the limit of the beta law as its second shape (N - A - 1) / 2 tends to 0.
    bound = (n_rows - 1) ** 2 / n_rows
    if n_rows == n_components + 1:
        return bound
    return bound * beta_dist.isf(1 - conf_level, n_components / 2, (n_rows - n_components - 1) / 2)


def judged_rows_note(conf_level: float, n_components: int, n_rows: int, judged_rows: JudgedRows | None) -> str | None:
    """Return the text of the warning that ``judged_rows=None`` gives, or ``None`` when it gives none (#978).

    A call warns only when its number changes in 3.0: when ``judged_rows`` is ``None`` and the
    limit for fitted rows differs from the limit for new rows. Arguments are as validated by
    :func:`hotellings_t2_limit`.
    """
    if judged_rows is not None:
        return None
    new = _t2_limit(conf_level, n_components, n_rows, "new")
    fitted = _t2_limit(conf_level, n_components, n_rows, "fitted")
    if new == fitted:
        return None
    return (
        f"Hotelling's T² limit: judged_rows=None gives the F limit for new rows ({new:.4g} here) until 3.0, "
        f"and from 3.0 the beta limit for the rows the model was fitted to ({fitted:.4g} here), as announced "
        'in 2.4.0. Pass judged_rows="fitted" to judge the rows the model was fitted to (hotellings_t2_, a '
        'score plot, outlier detection), or judged_rows="new" to judge rows it was not fitted to; either '
        "silences this warning."
    )


def hotellings_t2_limit(
    conf_level: float = 0.95,
    n_components: int = 0,
    n_rows: int = 0,
    *,
    judged_rows: JudgedRows | None = None,
) -> float:
    """Return the Hotelling's T2 value at the given level of confidence.

    The rows a model was fitted to and new rows need different limits (#978). Both the centre and
    the spread of the scores are estimated from the ``N`` fitted rows. A fitted row took part in
    both, so it sits closer to the centre and cannot look extreme: its T² never exceeds
    ``(N - 1)^2 / N``. A new row took part in neither. For ``A`` components:

    - ``judged_rows="fitted"``: ``((N - 1)^2 / N) * Beta(A/2, (N - A - 1)/2)``, the law of a fitted
      row's T² (Tracy, Young and Mason, 1992).
    - ``judged_rows="new"``: ``A (N^2 - 1) / (N (N - A)) * F(A, N - A)``, the law of a new row's T².

    Both are exact when the ``A`` scores are a fixed set of normal variables whose centre and
    spread are estimated from the ``N`` rows. The loadings of a PCA or a PLS model are estimated
    from the same rows, so for its scores each is an approximation that improves as ``N`` grows.

    Parameters
    ----------
    conf_level : float, optional
        Fractional confidence limit, less that 1.00; by default 0.95
    n_components : int
        Number of components in the fitted multivariate model.
    n_rows : int
        Number of rows (observations) used to fit the model. Must be > 0.
    judged_rows : {"fitted", "new"} or None, optional
        Which rows the limit judges: ``"fitted"``, the rows the model was fitted to (the values
        in ``hotellings_t2_``, the points of a score plot); or ``"new"``, rows it was not fitted
        to (the T² from ``diagnose`` or ``project``, a monitored batch). ``None``, the default,
        gives the ``"new"`` limit until 3.0 and the ``"fitted"`` one from 3.0. Until then it warns
        with a :class:`FutureWarning` whenever the two differ; passing either value silences it.

    Returns
    -------
    float
        The Hotelling's T2 limit at the given level of confidence. Returns
        ``0.0`` when ``n_components == 0`` and ``inf`` when ``n_components == n_rows``. For
        fitted rows with ``n_rows == n_components + 1``, where every fitted row's T² is
        ``(N - 1)^2 / N``, it returns that value.

    Raises
    ------
    ValueError
        If ``conf_level`` is outside (0, 1), if ``n_rows`` is not positive,
        if ``n_components`` is negative, if ``n_components`` exceeds
        ``n_rows`` (which would otherwise return a silent NaN from a negative
        F denominator degrees of freedom), or if ``judged_rows`` is not one of
        its values.

    References
    ----------
    Tracy, N. D., Young, J. C. and Mason, R. L., "Multivariate control charts for individual
    observations", Journal of Quality Technology, 24(2), 88-95, 1992.
    """
    if not 0.0 < conf_level < 1.0:
        raise ValueError(f"conf_level must lie in (0, 1); got {conf_level}.")
    if n_rows <= 0:
        raise ValueError(f"n_rows must be positive; got {n_rows}.")
    if n_components < 0:
        raise ValueError(f"n_components must be non-negative; got {n_components}.")
    if n_components > n_rows:
        # Beyond equality the F denominator degrees of freedom go negative and
        # scipy returns NaN, so the caller silently receives a limit that no
        # value can ever exceed. A model cannot have more components than
        # observations, so this is a bad call, not a degenerate model.
        raise ValueError(
            f"n_components ({n_components}) cannot exceed n_rows ({n_rows}); "
            "a model cannot have more components than observations."
        )
    kind = _judged_rows(judged_rows)
    note = judged_rows_note(conf_level, n_components, n_rows, judged_rows)
    if note is not None:
        warnings.warn(note, FutureWarning, stacklevel=user_stacklevel())
    return _t2_limit(conf_level, n_components, n_rows, kind)


def spe_limit(model: BaseEstimator, conf_level: float = 0.95) -> float:
    """Return the squared prediction error limit at the given level of confidence.

    Parameters
    ----------
    model : BaseEstimator
        A fitted multivariate model exposing a ``spe_`` attribute and an
        ``n_components_`` attribute (e.g. a fitted PCA or PLS instance).
        The fitted ``n_components_`` (with the trailing underscore) is used,
        not the constructor's ``n_components`` parameter, so a model whose
        component count was clamped at ``min(n_samples, n_features)`` at fit
        time is read with the value it actually holds.
    conf_level : float, optional
        Fractional confidence limit, less that 1.00; by default 0.95

    Returns
    -------
    float
        The SPE limit at the requested confidence level, on the same square-root
        scale as entries of ``model.spe_`` (see :func:`spe_calculation` for the
        chi-square approximation underlying it). Compare an observation's
        ``spe_`` value directly against the returned limit.
    """
    check_is_fitted(model, "spe_")  # pragma: no mutate - None checks the same fitted state

    return spe_calculation(
        spe_values=model.spe_.iloc[:, model.n_components_ - 1],
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
        The SPE limit at the requested confidence level, on the square-root scale
        (directly comparable to entries of ``model.spe_``, which are stored on the same
        scale). Observations above it have a different correlation structure than those
        used to build the model.

    Notes
    -----
    The limit is the weighted chi-square approximation matched to the mean
    and the variance of the squared SPE values (Box, 1954), which Nomikos and
    MacGregor (1995) adopted for batch monitoring. It is not the
    Jackson-Mudholkar (1979) limit, which needs the eigenvalues of the
    residual covariance matrix. When the squared SPE values have no relative
    spread (a perfect-fit training set where ``A == K``, or an all-equal SPE
    column), the approximation degenerates. In that case
    the limit falls back to ``sqrt(center_spe)``: there is no spread to
    bound, so any value above the centre is by construction out of family.
    See SEC-21 (#270), sub-item 3.

    The degeneracy test is on the coefficient of variation of the squared
    SPE, which is dimensionless, so it is unaffected by the units the data
    happen to be in. An earlier version compared the variance of the squared
    SPE (an SPE^4 quantity) against an absolute tolerance, which fired on
    well-scaled data whose SPE values were merely small in magnitude.
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
    #
    # The degeneracy test must be RELATIVE. ``variance_spe`` is in SPE^4
    # units and ``center_spe`` in SPE^2, so comparing either against the
    # absolute ``epsqrt`` (the previous behaviour) made the test depend on
    # the units of the data: SPE values a thousand times smaller tripped the
    # guard despite having exactly the same relative spread, and the returned
    # "95% limit" was then the RMS SPE, which around 42 percent of the
    # training data exceeded. The coefficient of variation of the squared SPE
    # is dimensionless and unchanged by rescaling.
    if center_spe <= 0.0:
        # Every SPE is exactly zero: a perfect fit.
        return 0.0
    relative_spread = float(np.sqrt(variance_spe)) / center_spe
    if relative_spread <= epsqrt:  # pragma: no mutate - differs from < only at exact equality
        return float(np.sqrt(center_spe))
    g = variance_spe / (2 * center_spe)
    h = (2 * (center_spe**2)) / variance_spe
    # Report square root again as SPE limit
    return np.sqrt(chi2.ppf(conf_level, h) * g)


def score_limit(model: BaseEstimator, conf_level: float = 0.95) -> np.ndarray:
    """Return two-sided confidence limits for each score component.

    The scores of component ``a`` have mean zero, and their standard deviation
    is *estimated* from the same ``N`` observations, so the symmetric limit at
    the requested confidence level is ``t_{N - 1} * std(score_a)`` with
    ``t_{N - 1}`` the Student-t quantile on ``N - 1`` degrees of freedom. A
    score outside ``[-limit, +limit]`` is unusual at that confidence level.

    The Student-t quantile is used rather than the standard-normal ``z``
    because ``s_a`` is an estimate: ``z`` is only its large-``N`` limit and is
    too narrow otherwise (1.96 against 2.26 at ``N = 10``, a limit 15 percent
    too tight; the gap falls below 3 percent by ``N = 50``).

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

    Raises
    ------
    ValueError
        If ``conf_level`` is outside (0, 1).

    References
    ----------
    Score limits: the limit is ``t_{(1 + conf_level) / 2, N - 1} * s_a``.
    Equivalently ``(t_a / s_a) ** 2`` follows an ``F(1, N - 1)`` distribution,
    since ``sqrt(F(1, N - 1))`` is exactly the two-sided ``t_{N - 1}``
    quantile.
    """
    if not 0.0 < conf_level < 1.0:
        raise ValueError(f"conf_level must lie in (0, 1); got {conf_level}.")
    check_is_fitted(model, "scores_")  # pragma: no mutate - None checks the same fitted state

    scores = np.asarray(model.scores_, dtype=float)  # pragma: no mutate - the scores are already float
    std_per_component = scores.std(axis=0, ddof=1)
    critical_value = t_dist.ppf(1 - (1 - conf_level) / 2, scores.shape[0] - 1)
    return critical_value * std_per_component


def ellipse_coordinates(  # noqa: PLR0913 - judged_rows, the rows the ellipse judges, joins its public signature (#978)
    score_horiz: int,
    score_vert: int,
    conf_level: float = 0.95,
    n_points: int = 100,
    n_components: int = 0,
    scaling_factor_for_scores: pd.Series | None = None,
    n_rows: int = 0,
    *,
    judged_rows: JudgedRows | None = None,
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
        Number of components `A` in the fitted model. Used to bound
        `score_horiz`/`score_vert` to a valid component index. The Hotelling's
        T^2 limit for the ellipse itself is computed with 2 degrees of freedom
        regardless of `A`, since the ellipse is the joint confidence region for
        the two plotted scores (see Background below).
    scaling_factor_for_scores : pd.Series or None
        Per-component standard deviations of the scores (``model.scaling_factor_for_scores_``).
        Required, although the signature defaults to ``None``; omitting it raises
        :class:`TypeError`.
        Used to scale the ellipse axes.
    n_rows : int
        Number of rows `N` in the data used to fit the model. Required to compute the
        Hotelling's T^2 limit for 2 components; must be strictly positive.
    judged_rows : {"fitted", "new"} or None, optional
        Which rows the ellipse judges, as in :func:`hotellings_t2_limit`: ``"fitted"`` for the
        scores of the rows the model was fitted to, the points of a score plot, or ``"new"`` for
        rows it was not fitted to. ``None``, the default, draws the ``"new"`` ellipse until 3.0
        and the ``"fitted"`` one from 3.0, and warns with a :class:`FutureWarning` until then.

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
        raise ValueError(f"score_horiz must lie in [1, {n_components}]; got {score_horiz}.")
    if not 1 <= score_vert <= n_components:
        raise ValueError(f"score_vert must lie in [1, {n_components}]; got {score_vert}.")
    if not 0 < conf_level < 1:  # pragma: no mutate - hotellings_t2_limit repeats this check
        raise ValueError(f"conf_level must lie in (0, 1); got {conf_level}.")
    if n_rows <= 0:  # pragma: no mutate - hotellings_t2_limit refuses n_rows below 2 too
        raise ValueError(f"n_rows must be positive; got {n_rows}.")
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2 to draw an ellipse; got {n_points}.")
    if scaling_factor_for_scores is None:
        raise TypeError(
            "scaling_factor_for_scores is required: pass the per-component score standard deviations "
            f"(model.scaling_factor_for_scores_); got {scaling_factor_for_scores}."
        )
    s_h = scaling_factor_for_scores.iloc[score_horiz - 1]
    s_v = scaling_factor_for_scores.iloc[score_vert - 1]
    # The ellipse is the joint confidence region for the TWO plotted scores,
    # so the T2 limit is computed for 2 components: for new rows
    # 2 (N^2 - 1) / (N (N - 2)) * F(alpha; 2, N - 2), the convention used in
    # Wold's texts, and for fitted rows ((N - 1)^2 / N) * Beta(1, (N - 3) / 2).
    # Using the full model's A here (the previous behaviour)
    # draws a strictly larger ellipse (~42% too wide per axis at A=5, N=50;
    # ~2x the area), silently hiding genuine outliers.
    # ``n_components`` is still used above to bound the score indices.
    t2_limit_specific = np.sqrt(hotellings_t2_limit(conf_level, n_components=2, n_rows=n_rows, judged_rows=judged_rows))
    dt = 2 * np.pi / (n_points - 1)
    steps = np.linspace(0, n_points - 1, n_points)
    x = np.cos(steps * dt) * t2_limit_specific * s_h
    y = np.sin(steps * dt) * t2_limit_specific * s_v
    return x, y
