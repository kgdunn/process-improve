# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.
"""Robust building blocks for the resistant multivariate estimators (#191).

Partial Robust M-regression needs two primitives the package did not already
have: a multivariate location estimate that a handful of bad rows cannot drag
away, and a weight function that lets an observation's influence decay with its
distance instead of being switched off at a threshold.

Both are deliberately small and free of estimator state, so they can be reused
by any other resistant method added later.
"""

from __future__ import annotations

import numpy as np


def fair_weights(distances: np.ndarray, cutoff: float) -> np.ndarray:
    r"""Fair weight function, :math:`w(z) = (1 + |z| / c)^{-2}`.

    The weight falls off smoothly and never reaches zero, which is what makes
    the reweighting loop in Partial Robust M-regression converge: a hard cutoff
    (Huber's, or trimming) lets an observation cross the boundary and flip its
    weight between iterations, so the loop can cycle instead of settling.

    Parameters
    ----------
    distances : np.ndarray
        Standardised distances or residuals, of any shape. Signs are ignored.
    cutoff : float
        Tuning constant :math:`c`, strictly positive. An observation at
        :math:`|z| = c` gets weight 0.25. Serneels et al. recommend 4.

    Returns
    -------
    np.ndarray
        Weights in :math:`(0, 1]`, the same shape as ``distances``.

    Raises
    ------
    ValueError
        If ``cutoff`` is not strictly positive, or ``distances`` is not finite.

    Examples
    --------
    >>> fair_weights(np.array([0.0, 4.0]), cutoff=4.0)
    array([1.  , 0.25])
    """
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError(f"cutoff must be a positive finite number; got {cutoff!r}.")
    distances = np.asarray(distances, dtype=float)
    if not np.all(np.isfinite(distances)):
        raise ValueError("distances must be finite (no NaN / inf).")
    return 1.0 / (1.0 + np.abs(distances) / cutoff) ** 2


def l1_median(X: np.ndarray, *, tol: float = 1e-8, max_iter: int = 200) -> np.ndarray:
    """Spatial (L1) median: the point minimising the sum of Euclidean distances to the rows.

    This is the multivariate location estimate Partial Robust M-regression
    measures leverage against. Unlike the coordinate-wise median it is
    equivariant under rotation, which matters here because the scores it is
    applied to have no privileged axes; and unlike the mean it has a breakdown
    point of 0.5, so half the rows may be arbitrarily bad before it moves.

    Solved by the modified Weiszfeld iteration of Vardi and Zhang (2000). Plain
    Weiszfeld divides by the distance to each row and so fails the moment an
    iterate lands exactly on a data point; the modification bounds the step by
    how much weight sits at that point, which both fixes the division and keeps
    the iteration descending.

    Parameters
    ----------
    X : np.ndarray
        Data of shape (n_samples, n_features).
    tol : float, optional
        Convergence tolerance on the step length, default 1e-8. Also the radius
        within which a row counts as sitting on the current estimate.
    max_iter : int, optional
        Maximum iterations, default 200.

    Returns
    -------
    np.ndarray
        The L1 median, of shape (n_features,).

    Raises
    ------
    ValueError
        If ``X`` is not 2-D, is empty, or is not finite.

    References
    ----------
    Y. Vardi and C.-H. Zhang, "The multivariate L1-median and associated data
    depth", PNAS, 97 (2000), 1423-1426.

    Examples
    --------
    >>> l1_median(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    array([0.5, 0.5])
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-dimensional (n_samples, n_features); got {X.ndim} dimension(s).")
    if X.shape[0] == 0:
        raise ValueError("X must have at least one row.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must be finite (no NaN / inf).")

    # The coordinate-wise median is already robust, so the iteration starts
    # inside the bulk of the data rather than having to walk in from the mean.
    estimate = np.median(X, axis=0)

    for _ in range(max_iter):
        offsets = X - estimate
        distances = np.linalg.norm(offsets, axis=1)
        on_estimate = distances <= tol
        if on_estimate.all():
            # Every row coincides with the estimate: it is already the median.
            break

        away = ~on_estimate
        inverse = 1.0 / distances[away]
        # The Weiszfeld step: an inverse-distance weighted mean of the rows.
        weiszfeld = (X[away] * inverse[:, None]).sum(axis=0) / inverse.sum()

        n_on_estimate = int(on_estimate.sum())
        if n_on_estimate == 0:
            candidate = weiszfeld
        else:
            # Vardi-Zhang: rows sitting on the estimate exert a pull of their own
            # that the Weiszfeld step cannot see. Damping the step by how much
            # weight sits there keeps the iteration from overshooting past them.
            pull = float(np.linalg.norm((offsets[away] * inverse[:, None]).sum(axis=0)))
            if pull <= tol:
                break
            damping = min(1.0, n_on_estimate / pull)
            candidate = (1.0 - damping) * weiszfeld + damping * estimate

        step = float(np.linalg.norm(candidate - estimate))
        estimate = candidate
        if step <= tol:
            break

    return estimate
