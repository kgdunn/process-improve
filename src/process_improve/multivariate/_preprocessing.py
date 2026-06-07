# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.
"""Scaling and centering helpers for the multivariate package (ENG-01).

Holds :class:`MCUVScaler` (mean-center, unit-variance; the preferred scaler for
fitting PCA / PLS models) and the standalone :func:`center` / :func:`scale`
utilities. Depends only on :mod:`process_improve.multivariate._common`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ._common import DataMatrix


class MCUVScaler(BaseEstimator, TransformerMixin):
    """
    Create our own mean centering and scaling to unit variance (MCUV) class
    The default scaler in sklearn does not handle small datasets accurately, with ddof.
    """

    def __init__(self):
        pass

    def fit(self, X: DataMatrix, y=None) -> MCUVScaler:  # noqa: ANN001, ARG002
        """Get the centering and scaling object constants.

        ``y`` is accepted (and ignored) so the scaler plugs into
        ``sklearn.pipeline.Pipeline``: every Pipeline step's ``fit`` is
        called with both ``X`` and ``y``, even when (as for a transformer)
        the ``y`` is unused.
        """
        self.center_ = pd.DataFrame(X).mean()
        # this is the key difference with "preprocessing.StandardScaler"
        self.scale_ = pd.DataFrame(X).std(ddof=1)
        self.scale_[self.scale_ == 0] = 1.0  # columns with no variance are left as-is.
        return self

    def transform(self, X: DataMatrix, y=None) -> pd.DataFrame:  # noqa: ANN001, ARG002
        """Do work of the transformation. ``y`` is accepted and ignored (Pipeline interop)."""
        check_is_fitted(self, "center_")
        check_is_fitted(self, "scale_")

        X = pd.DataFrame(X).copy()
        return (X - self.center_) / self.scale_

    def inverse_transform(self, X: DataMatrix) -> pd.DataFrame:
        """Do the inverse transformation."""
        check_is_fitted(self, "center_")
        check_is_fitted(self, "scale_")

        X = pd.DataFrame(X).copy()
        return X * self.scale_ + self.center_


def center(
    X,  # noqa: ANN001
    func: Callable = np.mean,
    axis: int = 0,
    extra_output: bool = False,
) -> DataMatrix | tuple[DataMatrix, np.ndarray]:
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
    # pandas-stubs types apply()'s axis as a Literal, so a plain ``int`` axis does
    # not match any overload; the call is valid at runtime.
    vector = pd.DataFrame(X).apply(func, axis=axis).to_numpy()  # type: ignore[call-overload]  # pandas-stubs axis is Literal
    if extra_output:
        return np.subtract(X, vector), vector
    else:
        return np.subtract(X, vector)


def scale(
    X: DataMatrix,
    func: Callable = np.std,
    axis: int = 0,
    extra_output: bool = False,
    ddof: int = 0,
    **kwargs,
) -> DataMatrix | tuple[DataMatrix, np.ndarray]:
    """
    Scales the data (does NOT do any centering); scales to unit variance by
    default.


    `func` [optional; default=np.std] {a function}
        The default (np.std) uses NumPy to calculate the standard deviation of
        the data along the required `axis`, skipping over any missing data, and
        uses that as `scale`.

    `axis` [optional; default=0] {integer}
        Transformations are applied on slices of data.  This specifies the
        axis along which the transformation will be applied.

    `ddof` [optional; default=0] {integer}
        Delta degrees of freedom, forwarded to `np.std` when `func` is the
        default `np.std`. The standard deviation is computed by dividing by
        ``N - ddof``, where N is the number of values which are present. The
        default (``ddof=0``) divides by N (the population standard deviation);
        pass ``ddof=1`` for the sample standard deviation (dividing by N-1).

        Note: :class:`MCUVScaler` uses ``ddof=1`` and is the preferred scaler
        for fitting PCA / PLS models. Use ``scale(center(X), ddof=1)`` here to
        match it. The ``ddof`` argument is ignored when a custom `func` is
        supplied (forward your own keyword arguments via ``**kwargs`` instead).

    Constant (zero-variance) columns are left unchanged: a zero entry in the
    computed scaling vector is replaced by 1.0 before inversion, mirroring
    :class:`MCUVScaler`, so no ``inf`` / ``NaN`` is introduced.

    Usage
    =====

    X = ...  # data matrix
    X = scale(center(X))
    X = scale(center(X), ddof=1)  # sample standard deviation, matches MCUVScaler
    my_scale = np.mad
    X = scale(center(X), func=my_scale)

    """
    if func is np.std and "ddof" not in kwargs:
        kwargs["ddof"] = ddof
    # pandas-stubs types apply()'s axis as a Literal, so a plain ``int`` axis does
    # not match any overload; the call is valid at runtime.
    vector = pd.DataFrame(X).apply(func, axis=axis, **kwargs).to_numpy()  # type: ignore[call-overload]  # pandas-stubs axis is Literal
    # Zero-variance (constant) columns are left as-is, mirroring MCUVScaler, so
    # that ``1.0 / vector`` does not introduce inf/NaN.
    vector = np.where(vector == 0, 1.0, vector)
    vector = 1.0 / vector

    if extra_output:
        return np.multiply(X, vector), vector
    else:
        return np.multiply(X, vector)
