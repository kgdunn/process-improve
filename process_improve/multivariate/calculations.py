# (c) Kevin Dunn, 2010-2025. MIT License. Based on own private work over the years.
from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import numpy as np
import pandas as pd
from scipy.stats import chi2, f
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

DataMatrix: TypeAlias = np.ndarray | pd.DataFrame

epsqrt = np.sqrt(np.finfo(float).eps)


def center(X, func: Callable = np.mean, axis: int = 0, extra_output: bool = False) -> DataMatrix:  # noqa: ANN001
    """
    Perform centering of data, using a function, `func` (default: np.mean).
    The function, if supplied, but return a vector with as many columns as the matrix X.

    `axis` [optional; default=0] {integer or None}

    This specifies the axis along which the centering vector will be calculated if not provided.
    The function is applied along the `axis`: 0=down the columns; 1 = across the rows.

    *Missing values*: The sample mean is computed by taking the sum along the `axis`, skipping
    any missing data, and dividing by N = number of values which are present. Values which were
    missing before, are left as missing after.
    """
    vector = pd.DataFrame(X).apply(func, axis=axis).values
    if extra_output:
        return np.subtract(X, vector), vector
    else:
        return np.subtract(X, vector)


def scale(X: DataMatrix, func: Callable = np.std, axis: int = 0, extra_output: bool = False, **kwargs) -> DataMatrix:
    """
    Scales the data (does NOT do any centering); scales to unit variance by
    default.


    `func` [optional; default=np.std] {a function}
        The default (np.std) will use NumPy to calculate the sample standard
        deviation of the data, and use that as `scale`.

        TODO: provide a scaling vector.
        The sample standard deviation is computed along the required `axis`,
        skipping over any missing data, and dividing by N-1, where N = number
        of values which are present, i.e. not counting missing values.

    `axis` [optional; default=0] {integer}
        Transformations are applied on slices of data.  This specifies the
        axis along which the transformation will be applied.

    #`markers` [optional; default=None]
    #    A vector (or slice) used to store indices (the markers) where the
    ##    variance needs to be replaced with the `low_variance_replacement`
    #
    #`variance_tolerance` [optional; default=1E-7] {floating point}
    #    A slice is considered to have no variance when the actual variance of
    #    that slice is smaller than this value.
    #
    #`low_variance_replacement` [optional; default=0.0] {floating point}
    #    Used to replace values in the output where the `markers` indicates no
    #    or low variance.

    Usage
    =====

    X = ...  # data matrix
    X = scale(center(X))
    my_scale = np.mad
    X = scale(center(X), func=my_scale)

    """
    # options = {}
    # options["markers"] = None
    # options["variance_tolerance"] = epsqrt
    # options["low_variance_replacement"] = np.nan

    vector = pd.DataFrame(X).apply(func, axis=axis, **kwargs).values
    # if options["markers"] is None:
    # options["markers"] = vector < options["variance_tolerance"]
    # if options["markers"].any():
    #    options["scale"][options["markers"]] = options["low_variance_replacement"]

    vector = 1.0 / vector

    if extra_output:
        return np.multiply(X, vector), vector
    else:
        return np.multiply(X, vector)


def nan_to_zeros(in_array: np.ndarray) -> np.ndarray:
    """Convert NaN to zero and return a NaN map."""

    nan_map = np.isnan(in_array)
    in_array[nan_map] = 0.0
    return in_array


def regress_a_space_on_b_row(a_space: np.ndarray, b_row: np.ndarray, a_space_present_map: np.ndarray) -> np.ndarray:
    """
    Project each row of `a_space` onto row vector `b_row`, to return a regression coefficient for every row in A.

    NOTE: Neither of these two inputs may have missing values. It is assumed you have replaced missing values by zero,
          and have a map of where the missing values were (more correctly, where the non-missing values are is given
          by `a_space_present_map`).

    NOTE: No checks are done on the incoming data to ensure consistency. That is the caller's responsibility. This
          function is called thousands of times, so that overhead is not acceptable.

    The `a_space_present_map` has `False` entries where `a_space` originally had NaN values.
    The `b_row` may never have missing values, and no map is provided for it. These row vectors are latent variable
    vectors, and therefore never have missing values.

    a_space             = [n_rows x j_cols]
    b_row               = [1      x j_cols]    # in other words, a row vector of `j_cols` entries
    a_space_present_map = [n_rows x j_cols]

    Returns               [n_rows x 1] = a_space * b_row^T  / ( b_row * b_row^T)
                                         (n x j) * (j x 1)  /  (1 x j)* (j x 1)  = n x 1
    """
    denom = np.tile(b_row, (a_space.shape[0], 1))  # tiles, row-by-row the `b_row` row vector, to create `n_rows`
    denominator = np.sum((denom * a_space_present_map) ** 2, axis=1).astype("float")
    denominator[denominator == 0] = np.nan
    return np.array((np.sum(a_space * denom, axis=1)) / denominator).reshape(-1, 1)


def ssq(X: np.ndarray, axis: int | None = None) -> float | np.ndarray:
    """Calculate the sum of squares of a 2D matrix (not array! and not checked for either: code will simply fail),
    skipping over any NaN (missing) data.
    """
    N, K = X.shape
    if axis == 0:
        out_ax0 = np.zeros(K)
        for k in np.arange(K):
            out_ax0[k] += np.nansum(X[:, k] ** 2)

        return out_ax0

    if axis == 1:
        out_ax1 = np.zeros(N)
        for n in np.arange(N):
            out_ax1[n] += np.nansum(X[n, :] ** 2)

        return out_ax1

    out = 0.0
    if axis is None:
        out = np.nansum(X**2)

    return out


def terminate_check(t_a_guess: np.ndarray, t_a: np.ndarray, iterations: int, settings: dict) -> bool:
    """Terminate the PCA iterative algorithm when any one of these conditions is True.

    #. scores converge: the norm between two successive iterations
    #. maximum number of iterations is reached
    """
    score_tol = np.linalg.norm(t_a_guess - t_a, ord=None)
    converged = score_tol < settings["md_tol"]
    max_iter = iterations > settings["md_max_iter"]
    return bool(np.any([max_iter, converged]))


def quick_regress(Y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Regress vector `x` onto the columns in matrix ``Y`` one at a time.
    Return the vector of regression coefficients, one for each column in `Y`.
    There may be missing data in `Y`, but not in `x`.  The `x` vector
    *must* be a column vector.
    """
    Ny, K = Y.shape
    Nx = x.shape[0]
    if Ny == Nx:  # Case A: b' = (x'Y)/(x'x): (1xN)(NxK) = (1xK)
        b = np.zeros((K, 1))
        for k in np.arange(K):
            b[k] = np.sum(x.T * np.nan_to_num(Y[:, k]))
            temp = ~np.isnan(Y[:, k]) * x.T
            denom = np.dot(temp, temp.T)[0][0]
            if np.abs(denom) > epsqrt:
                b[k] /= denom
        return b

    elif Nx == K:  # Case B: b = (Yx)/(x'x): (NxK)(Kx1) = (Nx1)
        b = np.zeros((Ny, 1))
        for n in np.arange(Ny):
            b[n] = np.sum(x[:, 0] * np.nan_to_num(Y[n, :]))
            # TODO(KGD): check: this denom is usually(always?) equal to 1.0
            denom = ssq(~np.isnan(Y[n, :]) * x.T)
            if np.abs(denom) > epsqrt:
                b[n] /= denom
        return b

    else:
        raise ValueError("The dimensions of the input arrays are not compatible.")


def hotellings_t2_limit(conf_level: float = 0.95, n_components: int = 0, n_rows: int = 0) -> float:
    """Return the Hotelling's T2 value at the given level of confidence.

    Parameters
    ----------
    conf_level : float, optional
        Fractional confidence limit, less that 1.00; by default 0.95

    Returns
    -------
    float
        The Hotelling's T2 limit at the given level of confidence.
    """
    assert 0.0 < conf_level < 1.0
    assert n_rows > 0
    if n_components == n_rows:
        return float("inf")
    return (
        n_components
        * (n_rows - 1)
        * (n_rows + 1)
        / (n_rows * (n_rows - n_components))
        * float(f.isf((1 - conf_level), n_components, n_rows - n_components))
    )


def spe_limit(model: BaseEstimator, conf_level: float = 0.95) -> float:
    """Return the squared prediction error limit at the given level of confidence.

    Parameters
    ----------
    conf_level : float, optional
        Fractional confidence limit, less that 1.00; by default 0.95

    Returns
    -------
    float
        The squared prediction error limit at the given level of confidence.
    """
    check_is_fitted(model, "squared_prediction_error")

    return spe_calculation(
        spe_values=model.squared_prediction_error.iloc[:, model.A - 1],
        conf_level=conf_level,
    )


def spe_calculation(spe_values: np.ndarray, conf_level: float = 0.95) -> float:
    """Return a limit for SPE (squared prediction error) at the given level of confidence.

    Parameters
    ----------
    spe_values : pd.Series
        The SPE values from the last component in the multivariate model.
    conf_level : [float], optional
        The confidence level, by default 0.95, i.e. the 95% confidence level.

    Returns
    -------
    float
        The limit, above which we judge observations in the model to have a different correlation
        structure than those values which were used to build the model.
    """
    assert conf_level > 0.0, "conf_level must be a value between (0.0, 1.0)"
    assert conf_level < 1.0, "conf_level must be a value between (0.0, 1.0)"

    # The limit is for the squares (i.e. the sum of the squared errors)
    # I.e. `spe_values` are square-rooted outside this function, so undo that.
    values = spe_values**2
    center_spe = float(values.mean())
    variance_spe = float(values.var(ddof=1))
    g = variance_spe / (2 * center_spe)
    h = (2 * (center_spe**2)) / variance_spe
    # Report square root again as SPE limit
    return np.sqrt(chi2.ppf(conf_level, h) * g)


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
        [description]
    score_vert : int
        [description]
    conf_level : float
        The `conf_level` confidence value: e.g. 0.95 is for the 95% confidence limit.
    n_points : int, optional
        Number of points to use in the ellipse; by default 100.

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
    assert 1 <= score_horiz <= n_components
    assert 1 <= score_vert <= n_components
    assert 0 < conf_level < 1
    assert n_rows > 0
    s_h = scaling_factor_for_scores.iloc[score_horiz - 1]
    s_v = scaling_factor_for_scores.iloc[score_vert - 1]
    t2_limit_specific = np.sqrt(hotellings_t2_limit(conf_level, n_components=n_components, n_rows=n_rows))
    dt = 2 * np.pi / (n_points - 1)
    steps = np.linspace(0, n_points - 1, n_points)
    x = np.cos(steps * dt) * t2_limit_specific * s_h
    y = np.sin(steps * dt) * t2_limit_specific * s_v
    return x, y


def internal_pls_nipals_fit_one_pc(
    x_space: np.ndarray,
    y_space: np.ndarray,
    x_present_map: np.ndarray,
    y_present_map: np.ndarray,
) -> dict[str, np.ndarray]:
    """Fit a PLS model using the NIPALS algorithm."""
    max_iter: int = 500

    is_converged = False
    n_iter = 0
    u_i = y_space[:, [0]]
    while not is_converged:
        # Step 1. w_i = X'u / u'u. Regress the columns of X on u_i, and store the slope coeff in vectors w_i.
        w_i = regress_a_space_on_b_row(x_space.T, u_i.T, x_present_map.T)

        # Step 2. Normalize w to unit length.
        w_i = w_i / np.linalg.norm(w_i)

        # Step 3. t_i = Xw / w'w. Regress rows of X on w_i, and store slope coefficients in t_i.
        t_i = regress_a_space_on_b_row(x_space, w_i.T, x_present_map)

        # Step 4. q_i = Y't / t't. Regress columns of Y on t_i, and store slope coefficients in q_i.
        q_i = regress_a_space_on_b_row(y_space.T, t_i.T, y_present_map.T)

        # Step 5. u_new = Yq / q'q. Regress rows of Y on q_i, and store slope coefficients in u_new
        u_new = regress_a_space_on_b_row(y_space, q_i.T, y_present_map)

        if (abs(np.linalg.norm(u_i - u_new)) / np.linalg.norm(u_i)) < epsqrt:
            is_converged = True
        if n_iter > max_iter:
            is_converged = True

        n_iter += 1
        u_i = u_new

    # We have converged. Keep sign consistency. Fairly arbitrary rule, but ensures we report results consistently.
    if np.var(t_i[t_i < 0]) > np.var(t_i[t_i >= 0]):
        t_i = -1 * t_i
        u_new = -1 * u_new
        w_i = -1 * w_i
        q_i = -1 * q_i

    return dict(t_i=t_i, u_i=u_i, w_i=w_i, q_i=q_i)
