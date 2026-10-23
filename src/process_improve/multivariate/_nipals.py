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
    """Return a copy of ``in_array`` with every NaN replaced by zero; ``in_array`` is left as it is.

    It used to write the zeros into the array it was given and hand that same array back,
    so a caller passing a view of its own data had that data overwritten (#513). The only
    caller in this package, ``TPLS.fit``, calls it once per block and keeps the result, so
    the copy costs one array per block and nothing inside the iteration. Take
    ``np.isnan(in_array)`` first if the positions of the missing values are needed too.
    """
    return np.where(np.isnan(in_array), 0.0, in_array)


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

    A row with nothing present has no regression, and its coefficient is 0. Its cells are
    all absent, so the 0 carries no weight in any later regression over that row, and the
    other rows are as they would be without it: the rule :func:`quick_regress` follows for
    an empty mask, and the score PCA and PLS give an all-missing row. It used to be NaN,
    which no later step could drop, so one empty row made every score of every later
    component NaN (#513). A row whose present cells all meet a zero in `b_row` is still NaN.
    """
    # TPLS hands in DataFrames. Without this, every product and sum below is a pandas
    # operation, about ten times slower, on a function called tens of thousands of times.
    a_space, b_row, a_space_present_map = np.asarray(a_space), np.asarray(b_row), np.asarray(a_space_present_map)
    denom = np.tile(b_row, (a_space.shape[0], 1))  # tiles, row-by-row the `b_row` row vector, to create `n_rows`
    denominator = np.sum((denom * a_space_present_map) ** 2, axis=1).astype("float")
    numerator = np.sum(a_space * denom, axis=1)
    empty = ~np.any(a_space_present_map, axis=1)
    # A row with cells present, but none where ``b_row`` is non-zero, is a degenerate
    # direction rather than missing data, and stays NaN so the caller can refuse it.
    denominator[denominator == 0] = np.nan
    denominator[empty] = 1.0
    return np.where(empty, 0.0, numerator / denominator).reshape(-1, 1)


def ssq(X: np.ndarray, axis: int | None = None) -> float | np.ndarray:
    """Calculate the sum of squares of a 2D matrix (not array! and not checked for either: code will simply fail),
    skipping over any NaN (missing) data.

    ``axis`` is 0 (one sum per column), 1 (one per row) or None (the total); any other
    value raises ValueError rather than returning 0.0.
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

    if axis is None:
        return np.nansum(X**2)
    raise ValueError(f"axis must be 0, 1 or None; got {axis}.")


def terminate_check(t_a_guess: np.ndarray, t_a: np.ndarray, iterations: int, settings: dict) -> bool:
    """Terminate the PCA / PLS NIPALS iterative algorithm when any one of these conditions is True.

    #. scores converge: the norm of the difference between two successive iterations, *relative to
       the norm of the current score vector*, falls below ``settings["md_tol"]``
    #. maximum number of iterations is reached

    The relative form makes the convergence decision invariant to a global rescaling of the input
    data (#504): the previous absolute comparison burned every ``md_max_iter`` iteration on
    large-magnitude data and "converged" instantly on tiny-magnitude data. ``md_tol`` is therefore
    a *relative* tolerance. The denominator is floored via :func:`_nz` so a fully-deflated,
    all-zero score vector cannot divide by zero (that case reports convergence, matching
    ``_tpls._has_converged``).
    """
    score_tol = float(np.linalg.norm(t_a_guess - t_a, ord=None) / _nz(float(np.linalg.norm(t_a, ord=None))))
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

    Degenerate denominators
    -----------------------
    The denominator is ``x'x`` restricted to the cells of ``Y`` that are present, so it
    is bounded above by ``ssq(x)`` and falls to zero only when the masked ``x`` carries
    no signal. The guard is therefore taken **relative to** ``ssq(x)``: a denominator
    below ``epsqrt * ssq(x)`` means the mask has thrown away all but a vanishing
    fraction of ``x``, whatever the units happen to be, and the coefficient is returned
    as ``0.0`` rather than as a ratio of noise.

    It used to be an absolute comparison against ``epsqrt`` (~1.5e-8), which is a
    statement about the *scale* of the data rather than its conditioning: on a
    well-conditioned block whose values are ~1e-5, the denominator is ~1e-9 and every
    coefficient was silently zeroed (#513). Scaling such a block by 1e5, which cannot
    change a ratio of the form ``(x'y)/(x'x)``, made the same call return the right
    answer.
    """
    # One reference scale for the whole call: the unmasked energy in ``x``. A non-finite
    # or zero value makes every comparison below False, so the degenerate branch is taken,
    # which is the right answer for an ``x`` that is all zero or has already overflowed.
    denom_floor = epsqrt * float(ssq(x))
    if Y.size and np.isfinite(Y).all():
        return _quick_regress_complete(Y, x, denom_floor)
    return _quick_regress_masked(Y, x, denom_floor)


def _quick_regress_masked(Y: np.ndarray, x: np.ndarray, denom_floor: float) -> np.ndarray:
    """:func:`quick_regress` one column (Case A) or row (Case B) at a time, masking missing cells of ``Y``."""
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
            # The threshold is relative to ssq(x); see "Degenerate denominators" above.
            b[k] = numer / denom if denom > denom_floor else 0.0
        return b

    elif Nx == K:  # Case B: b = (Yx)/(x'x): (NxK)(Kx1) = (Nx1)
        b = np.zeros((Ny, 1))
        for n in np.arange(Ny):
            numer = np.sum(x[:, 0] * np.nan_to_num(Y[n, :]))
            # This denominator is not always 1.0 and must not be dropped. It is the sum
            # of squares of `x` restricted to the non-missing cells of row `n`, so it is
            # 1.0 only when `x` is unit-norm AND that row of Y has no NaN. Two call sites
            # pass `c_a` (`_pls.py` and `_mbpls.py`), which NIPALS deliberately never
            # renormalises, and any missing cell in Y shrinks it further. Measured over
            # real fits: 1.0 for every complete-data PCA call, but only half of the
            # complete-data PLS calls and 46% of the calls with missing data.
            denom = ssq(~np.isnan(Y[n, :]) * x.T)
            # See sub-item 7 note above (mirror of Case A), relative threshold included.
            b[n] = numer / denom if denom > denom_floor else 0.0
        return b

    else:
        raise ValueError("The dimensions of the input arrays are not compatible.")


def _quick_regress_complete(Y: np.ndarray, x: np.ndarray, denom_floor: float) -> np.ndarray:
    """:func:`quick_regress` for a ``Y`` with no missing or infinite cell, without the Python loop.

    :func:`_quick_regress_masked` spends nearly all its time on per-row
    ``nan_to_num`` / ``isnan`` calls that change nothing when ``Y`` is complete, and NIPALS
    calls it thousands of times per fit. Here every coefficient comes from one vectorised
    product, and the result is **bit-for-bit** the loop's, not merely close to it:

    * each numerator is the same elementwise products summed along a C-contiguous row,
      which numpy reduces with the same pairwise summation as the loop's 1-D ``np.sum``;
    * with no cell masked, every coefficient shares one denominator, computed by exactly
      the call the loop makes for its first column (Case A) or row (Case B).

    ``tests/test_nipals_quick_regress.py`` checks the equality on random blocks.
    """
    Ny, K = Y.shape
    if Ny == x.shape[0]:  # Case A: one coefficient per column of Y
        numer = np.multiply(Y.T, x.T, order="C").sum(axis=1)
        temp = ~np.isnan(Y[:, 0]) * x.T
        denom = np.dot(temp, temp.T)[0][0]
    elif x.shape[0] == K:  # Case B: one coefficient per row of Y
        numer = np.multiply(Y, x[:, 0], order="C").sum(axis=1)
        denom = ssq(~np.isnan(Y[0, :]) * x.T)
    else:
        raise ValueError("The dimensions of the input arrays are not compatible.")
    if not denom > denom_floor:
        return np.zeros((numer.shape[0], 1))
    return (numer / denom).reshape(-1, 1)


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
        w_i = w_i / _nz(float(np.linalg.norm(w_i)))

        # Step 3. t_i = Xw / w'w. Regress rows of X on w_i, and store slope coefficients in t_i.
        t_i = regress_a_space_on_b_row(x_space, w_i.T, x_present_map)

        # Step 4. q_i = Y't / t't. Regress columns of Y on t_i, and store slope coefficients in q_i.
        q_i = regress_a_space_on_b_row(y_space.T, t_i.T, y_present_map.T)

        # Step 5. u_new = Yq / q'q. Regress rows of Y on q_i, and store slope coefficients in u_new.
        # A row with no Y value carries no information about u, so its empty regression gives 0,
        # and it has no weight in step 1, as a missing cell has none (#709).
        u_new = regress_a_space_on_b_row(y_space, q_i.T, y_present_map)

        # Floor ``||u_i||`` so an all-zero starting vector (degenerate
        # Y column) doesn't produce NaN here. SEC-21 (#270) sub-item 1.
        if (abs(np.linalg.norm(u_i - u_new)) / _nz(float(np.linalg.norm(u_i)))) < epsqrt:
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


def _kernel_pls(
    kernel_xx: np.ndarray, kernel_xy: np.ndarray, n_components: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Recompute PLS parameters from the association matrices ``X'X`` and ``X'Y``.

    Implements the Dayal-MacGregor (1997) kernel algorithm, which extracts the
    PLS weights, loadings and regression matrix directly from the kernels,
    without access to the original ``X`` / ``Y`` blocks. ``X'X`` is not deflated;
    each direction ``r`` is orthogonalised against the earlier loadings so that
    the score inner product is ``r' (X'X) r``.

    Parameters
    ----------
    kernel_xx : np.ndarray of shape (K, K)
        The X-space association matrix ``X'X`` (scaled data).
    kernel_xy : np.ndarray of shape (K, M)
        The cross association matrix ``X'Y`` (scaled data).
    n_components : int
        Number of latent variables ``A`` to extract.

    Returns
    -------
    weights : np.ndarray of shape (K, A)
        The X-space weights ``W`` (each column unit norm).
    loadings : np.ndarray of shape (K, A)
        The X-space loadings ``P``.
    direct_weights : np.ndarray of shape (K, A)
        The direct weights ``R = W (P'W)^-1`` such that scores ``T = X R``.
    y_loadings : np.ndarray of shape (M, A)
        The Y-space loadings ``C`` (also called ``Q``).
    score_ssq : np.ndarray of shape (A,)
        The score sums-of-squares ``r' (X'X) r`` per component; divide by
        ``N - 1`` to recover the score variances used for Hotelling's T2 scaling.
    """
    K = kernel_xx.shape[0]
    M = kernel_xy.shape[1]
    A = n_components
    weights = np.zeros((K, A))
    loadings = np.zeros((K, A))
    direct = np.zeros((K, A))
    y_loadings = np.zeros((M, A))
    score_ssq = np.zeros(A)

    xy = kernel_xy.copy()
    for a in range(A):
        if M == 1:
            w = xy[:, 0].copy()
        else:
            # Dominant eigenvector of X'Y (X'Y)' via the small M x M problem.
            eigvals, eigvecs = np.linalg.eigh(xy.T @ xy)
            q_dom = eigvecs[:, int(np.argmax(eigvals))]
            w = xy @ q_dom
        norm_w = float(np.linalg.norm(w))
        if norm_w < epsqrt:
            # No further usable covariance; leave the remaining columns at zero.
            break
        w = w / norm_w
        r = w.copy()
        for j in range(a):
            r = r - float(loadings[:, j] @ w) * direct[:, j]
        tt = float(r @ kernel_xx @ r)
        if tt < epsqrt:
            break
        p = (kernel_xx @ r) / tt
        c = (xy.T @ r) / tt  # (M,)
        weights[:, a] = w
        loadings[:, a] = p
        direct[:, a] = r
        y_loadings[:, a] = c
        score_ssq[a] = tt
        # Deflate X'Y only: X'Y <- X'Y - (X'X r) c' = X'Y - tt * p c'.
        xy = xy - tt * np.outer(p, c)
    return weights, loadings, direct, y_loadings, score_ssq


def _sign_align(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
    """Flip the sign of each column of ``current`` to best match ``previous``.

    PCA / PLS components are defined only up to a sign; the recomputed loadings
    can flip between updates, which makes score time-series jump. Aligning the
    sign to the previous iteration keeps the score traces continuous. Returns the
    per-column signs (``+1`` / ``-1``) actually applied, so the caller can flip
    the matching scores as well.

    Parameters
    ----------
    current : np.ndarray of shape (K, A)
        The freshly recomputed loadings / weights (modified in place).
    previous : np.ndarray of shape (K, A)
        The loadings / weights from the previous update.

    Returns
    -------
    np.ndarray of shape (A,)
        The signs applied to each column.
    """
    signs = np.ones(current.shape[1])
    for a in range(current.shape[1]):
        if float(current[:, a] @ previous[:, a]) < 0:
            signs[a] = -1.0
    return signs
