"""(c) Kevin Dunn, 2010-2026. MIT License.

Small numerical-linear-algebra guards shared across the package.

``np.linalg.inv`` does not raise on an ill-conditioned (near-singular) matrix:
it returns overflow-driven garbage. These helpers detect singular and
ill-conditioned matrices up front so callers can either fail with a clear
message or fall back to a pseudo-inverse, instead of silently propagating
``inf``/``nan`` into a fitted model or a plot.
"""

from __future__ import annotations

import numpy as np

#: A matrix whose 2-norm condition number exceeds this is treated as singular.
#: ``1 / eps`` (~4.5e15) is the standard threshold beyond which a double-precision
#: solve has effectively no significant digits left.
DEFAULT_COND_LIMIT: float = 1.0 / np.finfo(float).eps


def is_singular(matrix: np.ndarray, *, cond_limit: float = DEFAULT_COND_LIMIT) -> bool:
    """Return True if *matrix* is non-square, non-finite, singular, or ill-conditioned.

    Parameters
    ----------
    matrix:
        The matrix to test. Coerced to a float array.
    cond_limit:
        Condition-number threshold; above this the matrix is considered
        numerically singular. Defaults to :data:`DEFAULT_COND_LIMIT`.
    """
    a = np.asarray(matrix, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        return True
    if not np.all(np.isfinite(a)):
        return True
    cond = np.linalg.cond(a)
    return (not np.isfinite(cond)) or cond > cond_limit


def safe_inverse(
    matrix: np.ndarray,
    *,
    what: str = "matrix",
    cond_limit: float = DEFAULT_COND_LIMIT,
) -> np.ndarray:
    """Invert *matrix*, raising a clear error if it is singular or ill-conditioned.

    For a well-conditioned matrix this returns exactly ``np.linalg.inv(matrix)``
    (no numerical change versus a bare ``inv``). For a singular or
    ill-conditioned matrix it raises :class:`numpy.linalg.LinAlgError` with an
    actionable message instead of returning ``inf``/``nan`` garbage.

    Parameters
    ----------
    matrix:
        The square matrix to invert.
    what:
        Human-readable description of the matrix, used in the error message.
    cond_limit:
        Condition-number threshold; see :func:`is_singular`.

    Raises
    ------
    numpy.linalg.LinAlgError
        If *matrix* is non-square, non-finite, singular, or ill-conditioned.
    """
    a = np.asarray(matrix, dtype=float)
    if is_singular(a, cond_limit=cond_limit):
        raise np.linalg.LinAlgError(
            f"{what} is singular or ill-conditioned (condition number exceeds "
            f"{cond_limit:.2e}); cannot invert reliably. Reduce the number of "
            "components/terms or remove collinear columns."
        )
    return np.linalg.inv(a)
