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

    def fit(self, X: DataMatrix) -> MCUVScaler:
        """Get the centering and scaling object constants."""
        self.center_ = pd.DataFrame(X).mean()
        # this is the key difference with "preprocessing.StandardScaler"
        self.scale_ = pd.DataFrame(X).std(ddof=1)
        self.scale_[self.scale_ == 0] = 1.0  # columns with no variance are left as-is.
        return self

    def transform(self, X: DataMatrix) -> pd.DataFrame:
        """Do work of the transformation."""
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
