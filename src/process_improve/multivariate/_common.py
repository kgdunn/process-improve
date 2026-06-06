# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.
"""Shared primitives for the multivariate package (ENG-01).

This leaf module holds the small, dependency-free building blocks used across
the PCA / PLS / TPLS / multiblock implementations: the :data:`DataMatrix` type
alias, the :data:`epsqrt` tolerance, the NIPALS denominator floor helper
:func:`_nz`, and :class:`SpecificationWarning`. Nothing here imports any of the
sibling submodules, so it sits at the base of the dependency graph.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, TypeAlias

import numpy as np
import pandas as pd

# Names re-exported to the rest of the package. Declared explicitly so CodeQL
# does not flag the ``DataMatrix`` type alias (only ever referenced in lazy
# annotation strings under ``from __future__ import annotations``) as unused.
__all__ = ["DataMatrix", "NotEnoughVarianceError", "SpecificationWarning", "epsqrt"]

DataMatrix: TypeAlias = np.ndarray | pd.DataFrame

epsqrt = np.sqrt(np.finfo(float).eps)

#: Smallest positive float; used to floor NIPALS denominators away from zero.
_DENOM_FLOOR = float(np.finfo(float).tiny)


def _nz(denominator: float) -> float:
    """Floor a non-negative NIPALS denominator away from zero.

    Sum-of-squares and vector-norm denominators (``v @ v``, ``norm(v)``) are
    non-negative. When a score or loading vector collapses to (near) zero during
    NIPALS - a fully-deflated component, or a degenerate / perfectly collinear
    block - the denominator is ~0 and the division would yield ``inf``/``nan``
    that silently poisons the fitted model. Flooring to the smallest positive
    float leaves every well-conditioned value untouched (real denominators are
    far larger) while turning the degenerate ``0/0`` into a finite (~0)
    projection, since the numerator collapses with the same vector.
    """
    return max(_DENOM_FLOOR, denominator)


class SpecificationWarning(UserWarning):
    """Parent warning class."""


class NotEnoughVarianceError(RuntimeError):
    """Raised when more components are requested than the data's variance supports.

    During NIPALS extraction (PCA / PLS) the deflated data array can run out of
    variance before the requested number of components is reached, even when that
    count is within ``min(N, K)`` - for example with rank-deficient or perfectly
    collinear data. The model cannot compute any further components in that case.

    Subclasses :class:`RuntimeError` so existing ``except RuntimeError`` handlers
    keep working, while callers that want to react specifically (e.g. trimming the
    component count during cross-validation) can catch this narrower type.
    """


def _align_to_fit_features(X: pd.DataFrame, fit_feature_names: pd.Index) -> pd.DataFrame:
    """Validate and align new-data columns against the features seen during ``fit``.

    PCA / PLS only checked that data passed to ``transform`` / ``predict`` had the
    right *number* of columns. A correctly-shaped frame whose columns are renamed
    or reordered would otherwise be projected positionally (PCA, ``X.values @ P``)
    or silently label-aligned to all-``NaN`` (PLS, ``X @ direct_weights_``),
    producing wrong scores with no error raised (issue #195). This helper makes
    that consistency explicit, mirroring scikit-learn's ``feature_names_in_``
    handling:

    * If both the training data and ``X`` carry string feature names, the *set*
      of names must match (otherwise :class:`ValueError`); columns supplied in a
      different order are reordered to the training order.
    * If the training data had names but ``X`` does not (e.g. a bare ndarray was
      passed), the columns are taken to correspond positionally and are labelled
      with the training names, so downstream label-aligned arithmetic stays
      correct rather than collapsing to ``NaN``.
    * If the training data itself had no (string) feature names, there is nothing
      to validate and ``X`` is returned unchanged.

    The caller is expected to have already validated the column *count*.

    Parameters
    ----------
    X : pd.DataFrame
        New data passed to ``transform`` / ``predict`` (already coerced to a
        DataFrame and count-checked by the caller).
    fit_feature_names : pd.Index
        The ``X.columns`` captured during ``fit`` (``self._feature_names``).

    Returns
    -------
    pd.DataFrame
        ``X`` with columns aligned to the training feature order.

    Raises
    ------
    ValueError
        If both sides carry string names but the sets of names differ.
    """
    fit_names = list(fit_feature_names)
    if not (fit_names and all(isinstance(name, str) for name in fit_names)):
        # Training data had no string feature names (e.g. fitted from an ndarray);
        # only the column count is meaningful, which the caller already checked.
        return X

    new_names = list(X.columns)
    if not all(isinstance(name, str) for name in new_names):
        # New data carries default positional columns (e.g. came in as an
        # ndarray). Assume positional correspondence and label it with the
        # training names so label-aligned operations behave correctly.
        X = X.copy()
        X.columns = pd.Index(fit_names)
        return X

    if set(new_names) != set(fit_names):
        missing = [name for name in fit_names if name not in set(new_names)]
        unexpected = [name for name in new_names if name not in set(fit_names)]
        raise ValueError(
            "Feature names of the data passed to predict/transform do not match "
            "those seen during fit. "
            f"Missing columns: {missing}; unexpected columns: {unexpected}."
        )
    if new_names != fit_names:
        # Same names, different order: reorder to the training order so the
        # positional projection lines up.
        X = X[fit_names]
    return X


def _model_method(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a module-level ``fn(model, ...)`` as an introspectable instance method.

    ENG-05: estimators (PCA / PLS / ...) expose convenience methods such as
    ``score_plot``, ``spe_limit`` and ``vip`` that forward to the standalone
    functions with ``self`` supplied as the ``model`` argument. Defining them
    via this factory at class-body time - rather than binding
    ``functools.partial`` instances in ``fit`` - means ``help`` and
    ``inspect.signature`` report the underlying function (minus ``self``), the
    fitted model stays picklable, and subclasses can override cleanly.
    """

    @functools.wraps(fn)
    def method(self: object, *args, **kwargs) -> object:
        return fn(self, *args, **kwargs)

    return method
