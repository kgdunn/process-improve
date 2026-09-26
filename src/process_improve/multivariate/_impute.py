# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.
"""Fill missing cells by EM on a low-rank model of the whole matrix (#189).

PLS fitted to incomplete data needs the missing cells estimated, and the obvious
way to estimate them is from the PLS model itself: fit, rebuild each missing cell
from the loadings, refit. That was the first version of this feature, and it is
the wrong tool. PLS chooses its components for their covariance with Y, not to
reproduce X, so it rebuilds X only as well as its components happen to span it;
and alternating "fit PLS, rebuild from PLS" is not an EM algorithm, so nothing
makes it converge. On a synthetic fixture about one fit in seven never did,
cycling with imputed cells still moving by 0.6 standard deviations a round after
a thousand rounds.

The estimate here comes from a model of the *joint* distribution instead: a
principal-component model of ``[X Y]``, refitted every round from the completed
matrix. That is the model-building algorithm of Folch-Fortuny, Arteaga and Ferrer
for PCA, the same one ``PCA(algorithm="tsr")`` runs. It is a genuine EM, it
converged on every one of the fixtures where the PLS-based loop cycled, and it was
more accurate as well. PLS is then fitted once, to the completed data.

References
----------
A. Folch-Fortuny, F. Arteaga and A. Ferrer, "PCA model building with missing
data: new proposals and a comparative study", Chemometrics and Intelligent
Laboratory Systems, 146 (2015), 77-88.
"""

from __future__ import annotations

import numpy as np
from sklearn.utils import Bunch

#: The estimators this module offers, by their ``md_method`` names.
IMPUTATION_METHODS = ("tsr", "pmp")


def impute_low_rank(
    data: np.ndarray,
    n_components: int,
    *,
    method: str = "tsr",
    tol: float = 1e-8,
    max_iter: int = 1000,
) -> Bunch:
    """Fill the NaN cells of ``data`` by EM on an ``n_components`` principal-component model.

    Each round: standardise the currently completed matrix, fit its leading
    principal components, re-estimate every missing cell from the observed cells in
    its row, and stop once no missing cell moves by more than ``tol`` (measured in
    standard deviations, so the tolerance means the same thing whatever the units).
    The centre and spread are re-estimated every round rather than fixed from the
    observed cells, because they are parameters of the model like the components
    are, and the completed data estimates them better than the observed cells alone.

    A row is estimated with ``min(n_components, n_observed)`` components, so a row
    that observes fewer columns than there are components still has a well-posed
    estimate rather than a singular one; a row that observes nothing is placed at
    the mean.

    Parameters
    ----------
    data : np.ndarray of shape (n_rows, n_columns)
        The matrix to complete, NaN where missing.
    n_components : int
        Rank of the model the missing cells are estimated from.
    method : {"tsr", "pmp"}, default="tsr"
        ``"tsr"``, trimmed score regression, regresses the full scores on the
        scores the observed columns alone would give, using how the columns
        co-vary; it shrinks toward the mean when a row's observed cells say little.
        ``"pmp"``, projection to the model plane, fits the observed cells onto the
        plane by least squares and ignores that information, so it can place a
        sparsely observed row anywhere on the plane.
    tol : float, default=1e-8
        Convergence tolerance on the largest change in any missing cell, in
        standard deviations.
    max_iter : int, default=1000
        Maximum number of rounds.

    Returns
    -------
    result : sklearn.utils.Bunch
        With keys ``completed`` (np.ndarray, ``data`` with every NaN filled, in the
        original units), ``rounds`` (int), ``converged`` (bool) and ``shift``
        (float, the largest change on the last round, in standard deviations).

    Raises
    ------
    ValueError
        If ``method`` is not recognised, ``n_components`` is not positive, or a
        column has no observed values at all.
    """
    if method not in IMPUTATION_METHODS:
        raise ValueError(f"method must be one of {IMPUTATION_METHODS}; got {method!r}.")
    if n_components < 1:
        raise ValueError(f"n_components must be at least 1; got {n_components}.")
    data = np.asarray(data, dtype=float)
    missing = np.isnan(data)
    empty = np.all(missing, axis=0)
    if np.any(empty):
        raise ValueError(
            f"Columns at positions {np.flatnonzero(empty).tolist()} have no observed values, so there is "
            "nothing to estimate them from. Drop them before fitting."
        )

    completed = np.where(missing, np.nanmean(data, axis=0), data)
    if not missing.any():
        return Bunch(completed=completed, rounds=0, converged=True, shift=0.0)

    patterns = _group_by_pattern(missing)
    rounds, shift, converged = 0, 0.0, False
    while rounds < max_iter:
        rounds += 1
        centre, spread = completed.mean(axis=0), completed.std(axis=0, ddof=1)
        spread = np.where(spread > 0, spread, 1.0)
        standardised = (completed - centre) / spread
        components = _leading_components(standardised, n_components)
        covariance = standardised.T @ standardised / max(1, standardised.shape[0] - 1)

        updated = standardised.copy()
        for observed, rows in patterns:
            updated[np.ix_(rows, ~observed)] = _estimate_missing(
                standardised[np.ix_(rows, observed)], observed, components, covariance, method
            )

        shift = float(np.max(np.abs(updated[missing] - standardised[missing])))
        # Only the missing cells come back from the standardised space. The observed
        # ones are restored from ``data`` itself, because the round trip through
        # ``(x - c) / s * s + c`` is not exact: left alone, observed values would
        # drift by rounding a little every round, and a caller would get back data
        # that no longer matches what they passed in.
        completed = np.where(missing, updated * spread + centre, data)
        if shift <= tol:
            converged = True
            break

    return Bunch(completed=completed, rounds=rounds, converged=converged, shift=shift)


def _group_by_pattern(missing: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Group the incomplete rows by which columns they observe.

    Every row with the same pattern shares one estimation operator, so it is built
    once per pattern per round rather than once per row. A sensor that drops out
    produces many rows with one pattern; that case is common in process data.
    """
    groups: dict[bytes, list[int]] = {}
    for row in np.flatnonzero(missing.any(axis=1)):
        groups.setdefault((~missing[row]).tobytes(), []).append(int(row))
    return [(np.frombuffer(key, dtype=bool), np.asarray(rows)) for key, rows in groups.items()]


def _leading_components(standardised: np.ndarray, n_components: int) -> np.ndarray:
    """Return the leading right singular vectors as columns, shape (n_columns, n_components)."""
    _, _, vt = np.linalg.svd(standardised, full_matrices=False)
    return vt.T[:, : min(n_components, vt.shape[0])]


def _estimate_missing(
    observed_values: np.ndarray,
    observed: np.ndarray,
    components: np.ndarray,
    covariance: np.ndarray,
    method: str,
) -> np.ndarray:
    """Estimate the missing cells of rows that share one observation pattern.

    Parameters
    ----------
    observed_values : np.ndarray of shape (n_rows, n_observed)
        The standardised observed cells of those rows.
    observed : np.ndarray of shape (n_columns,), dtype bool
        Which columns the pattern observes.
    components : np.ndarray of shape (n_columns, n_components)
        The current principal components.
    covariance : np.ndarray of shape (n_columns, n_columns)
        The current covariance of the standardised completed matrix.
    method : {"tsr", "pmp"}

    Returns
    -------
    np.ndarray of shape (n_rows, n_missing)
        The estimates, in standardised units.
    """
    if not observed.any():
        # Nothing observed: the estimate is the mean, which is zero once standardised.
        return np.zeros((observed_values.shape[0], int((~observed).sum())))

    # Never more components than observed columns: that is what keeps a sparsely
    # observed row well-posed rather than singular.
    usable = min(components.shape[1], int(observed.sum()))
    trimmed = components[observed, :usable]
    if method == "tsr":
        # Trimmed score regression: z_missing = S21 L (L' S11 L)^+ L' z_observed.
        s11 = covariance[np.ix_(observed, observed)]
        s21 = covariance[np.ix_(~observed, observed)]
        operator = (s21 @ trimmed) @ np.linalg.pinv(trimmed.T @ s11 @ trimmed) @ trimmed.T
    else:
        # Projection to the model plane: least-squares scores from the observed cells.
        operator = components[~observed, :usable] @ np.linalg.pinv(trimmed)
    return observed_values @ operator.T
