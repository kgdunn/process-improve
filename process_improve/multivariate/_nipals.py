# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.
"""Low-level NIPALS and least-squares math for the multivariate package (ENG-01).

These are the missing-data-aware numerical kernels shared by the PCA / PLS /
TPLS / multiblock fitters: NaN handling, row-wise projection, sum-of-squares,
the iterative-termination test, the shape-driven quick regression, and the
single-component PLS NIPALS inner loop. Depends only on
:mod:`process_improve.multivariate._common`.
"""

from __future__ import annotations

import logging

import numpy as np

from ._common import _nz, epsqrt

logger = logging.getLogger(__name__)


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
            out_ax0[k] = out_ax0[k] + np.nansum(X[:, k] ** 2)

        return out_ax0

    if axis == 1:
        out_ax1 = np.zeros(N)
        for n in np.arange(N):
            out_ax1[n] = out_ax1[n] + np.nansum(X[n, :] ** 2)

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
    # SEC-33 (#282): use ``>=`` so the loop runs exactly ``md_max_iter``
    # iterations rather than ``md_max_iter + 1``.
    max_iter = iterations >= settings["md_max_iter"]
    return bool(np.any([max_iter, converged]))


def quick_regress(Y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Quick least-squares regression with two shape-driven modes.

    The mode is selected from the shapes of ``Y`` (``Ny`` x ``K``) and ``x`` (``Nx`` x 1):

    * **Case A** (``Ny == Nx``): regress ``x`` onto each column of ``Y`` one at a
      time. Returns a ``(K, 1)`` vector of coefficients ``b_k = (x' y_k) / (x' x)``,
      one per column of ``Y``.
    * **Case B** (``Nx == K``): regress ``x`` onto each row of ``Y`` one at a
      time. Returns a ``(Ny, 1)`` vector of coefficients ``b_n = (y_n x) / (x' x)``,
      one per row of ``Y``.

    There may be missing data in ``Y``, but not in ``x``. The ``x`` vector
    *must* be a column vector. Raises ``ValueError`` if neither case matches.
    """
    Ny, K = Y.shape
    Nx = x.shape[0]
    if Ny == Nx:  # Case A: b' = (x'Y)/(x'x): (1xN)(NxK) = (1xK)
        b = np.zeros((K, 1))
        for k in np.arange(K):
            numer = np.sum(x.T * np.nan_to_num(Y[:, k]))
            temp = ~np.isnan(Y[:, k]) * x.T
            denom = np.dot(temp, temp.T)[0][0]
            # Coefficient is undefined when the effective ``x`` (after
            # NaN masking in Y) has no signal: ``x`` is all zero, or the
            # column ``Y[:, k]`` is all NaN. Return 0.0 (no contribution)
            # rather than the un-normalised numerator, which the previous
            # code returned silently. SEC-21 (#270) sub-item 7.
            b[k] = numer / denom if np.abs(denom) > epsqrt else 0.0
        return b

    elif Nx == K:  # Case B: b = (Yx)/(x'x): (NxK)(Kx1) = (Nx1)
        b = np.zeros((Ny, 1))
        for n in np.arange(Ny):
            numer = np.sum(x[:, 0] * np.nan_to_num(Y[n, :]))
            # TODO(KGD): check: this denom is usually(always?) equal to 1.0
            denom = ssq(~np.isnan(Y[n, :]) * x.T)
            # See sub-item 7 note above (mirror of Case A).
            b[n] = numer / denom if np.abs(denom) > epsqrt else 0.0
        return b

    else:
        raise ValueError("The dimensions of the input arrays are not compatible.")


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

        # Step 2. Normalize w to unit length. Floor the denominator so a
        # fully-deflated component doesn't divide by zero (SEC-21 #270 sub-item 1).
        w_i = w_i / _nz(np.linalg.norm(w_i))

        # Step 3. t_i = Xw / w'w. Regress rows of X on w_i, and store slope coefficients in t_i.
        t_i = regress_a_space_on_b_row(x_space, w_i.T, x_present_map)

        # Step 4. q_i = Y't / t't. Regress columns of Y on t_i, and store slope coefficients in q_i.
        q_i = regress_a_space_on_b_row(y_space.T, t_i.T, y_present_map.T)

        # Step 5. u_new = Yq / q'q. Regress rows of Y on q_i, and store slope coefficients in u_new
        u_new = regress_a_space_on_b_row(y_space, q_i.T, y_present_map)

        # Floor ``||u_i||`` so an all-zero starting vector (degenerate
        # Y column) doesn't produce NaN here. SEC-21 (#270) sub-item 1.
        if (abs(np.linalg.norm(u_i - u_new)) / _nz(np.linalg.norm(u_i))) < epsqrt:
            is_converged = True
        if n_iter > max_iter:
            is_converged = True

        n_iter += 1
        u_i = u_new

    logger.debug("PLS NIPALS inner loop converged in %d iterations (max_iter=%d)", n_iter, max_iter)

    # We have converged. Keep sign consistency. Fairly arbitrary rule, but ensures we report results consistently.
    # SEC-33 (#282): ``np.var`` on an empty slice returns NaN and emits
    # a RuntimeWarning; ``NaN > NaN`` is False so the comparison
    # silently skipped a sign-flip that may have been required. Guard
    # against the empty-slice case explicitly.
    neg = t_i[t_i < 0]
    nonneg = t_i[t_i >= 0]
    if neg.size > 0 and nonneg.size > 0 and np.var(neg) > np.var(nonneg):
        t_i = -t_i
        # ``u_i`` (the value actually returned) is the converged ``u_new`` and is
        # only consumed by callers for a sign-invariant convergence check, so it
        # is intentionally left un-flipped here. (The previous ``u_new = -u_new``
        # was a dead assignment: ``u_new`` is never read again.)
        w_i = -w_i
        q_i = -q_i

    return dict(t_i=t_i, u_i=u_i, w_i=w_i, q_i=q_i)
