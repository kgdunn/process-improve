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
from sklearn.utils.validation import _check_feature_names_in, check_is_fitted, validate_data

from ._common import DataMatrix


class MCUVScaler(TransformerMixin, BaseEstimator):
    """Mean-centre, unit-variance (MCUV) scaler.

    Unlike ``sklearn.preprocessing.StandardScaler`` this uses the sample
    standard deviation (``ddof=1``), the convention for chemometric data
    analysis where the population is the training set itself rather than a
    sampled super-population.

    The estimator follows the standard sklearn contract: ``n_features_in_``
    and ``feature_names_in_`` are populated by ``fit``; sparse / complex /
    object dtype / empty input are rejected with sklearn-style errors;
    NaN values pass through (the chemometric preprocessing pipeline expects
    to thread missing-data through to the downstream NIPALS estimator).
    """

    def __init__(self):
        pass

    def __sklearn_tags__(self):
        """Declare sklearn capability tags (sklearn 1.6+).

        ``allow_nan=True`` because :meth:`fit` and :meth:`transform` use
        ``np.nanmean`` / ``np.nanstd``: NaN cells flow through, get
        re-NaN'd by the centring/scaling arithmetic, and reach the
        downstream NIPALS estimator that knows how to handle them.
        """
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags

    def get_feature_names_out(self, input_features=None) -> np.ndarray:  # noqa: ANN001
        """Return the output column names of :meth:`transform`.

        :class:`MCUVScaler` is column-preserving (centring + scaling
        leave the X column layout unchanged), so the returned names
        mirror those captured during :meth:`fit` (or the
        ``input_features`` argument when no ``feature_names_in_`` was
        captured - the standard sklearn fallback for ndarray-fit
        estimators).

        Used by :meth:`set_output` (sklearn 1.2+) to label the
        :class:`~pandas.DataFrame` view of the output when
        ``set_output(transform="pandas")`` is on, and by Pipeline
        introspection.
        """
        return _check_feature_names_in(self, input_features)

    def fit(self, X: DataMatrix, y=None) -> MCUVScaler:  # noqa: ANN001, ARG002
        """Compute the column means and sample standard deviations.

        ``y`` is accepted (and ignored) so the scaler plugs into
        :class:`sklearn.pipeline.Pipeline`, which threads ``y`` through every
        step's ``fit`` even when (as for a transformer) it is unused.
        """
        # Convenience: accept a 1-D Series (a single-column y, common when
        # the scaler is used for the target side of a PLS fit). validate_data
        # itself requires 2-D input, so promote here before it sees X.
        if isinstance(X, pd.Series):
            X = X.to_frame()
        X_arr = validate_data(
            self,
            X,
            reset=True,
            accept_sparse=False,
            ensure_min_samples=2,
            ensure_min_features=1,
            dtype="numeric",
            ensure_all_finite="allow-nan",
        )
        feature_names = getattr(self, "feature_names_in_", None)
        index = pd.Index(feature_names) if feature_names is not None else pd.RangeIndex(X_arr.shape[1])

        # nanmean / nanstd so NaN cells pass through with the right
        # column-level statistics (the chemometric pipeline's missing-data
        # contract). std uses ddof=1: this is the difference from
        # sklearn.preprocessing.StandardScaler.
        center = np.nanmean(X_arr, axis=0)
        scale = np.nanstd(X_arr, axis=0, ddof=1)
        # Constant columns are left as-is (scale to 1.0) rather than
        # producing inf / nan when transform divides.
        scale = np.where(scale == 0, 1.0, scale)

        self.center_ = pd.Series(center, index=index)
        self.scale_ = pd.Series(scale, index=index)
        return self

    def transform(self, X: DataMatrix, y=None) -> pd.DataFrame:  # noqa: ANN001, ARG002
        """Mean-centre and unit-variance scale ``X``.

        ``y`` is accepted (and ignored) for :class:`Pipeline` compatibility.
        """
        check_is_fitted(self, ("center_", "scale_"))
        # Mirror fit()'s Series convenience for symmetric round-tripping.
        if isinstance(X, pd.Series):
            X = X.to_frame()
        # Preserve the row index for DataFrame input; ndarray input falls
        # back to a RangeIndex.
        index = X.index if isinstance(X, pd.DataFrame) else None
        X_arr = validate_data(
            self,
            X,
            reset=False,
            accept_sparse=False,
            dtype="numeric",
            ensure_all_finite="allow-nan",
        )
        out = (X_arr - self.center_.to_numpy()) / self.scale_.to_numpy()
        return pd.DataFrame(out, index=index, columns=self.center_.index)

    def inverse_transform(self, X: DataMatrix) -> pd.DataFrame:
        """Inverse the mean-centring and unit-variance scaling."""
        check_is_fitted(self, ("center_", "scale_"))
        index = X.index if isinstance(X, pd.DataFrame) else None
        # inverse_transform is intentionally NOT routed through validate_data:
        # callers (TransformedTargetRegressor included) pass ndarray output
        # from a downstream estimator that may have a different shape than
        # the fit-time X (typical: 1-D y_pred for a single-target regressor).
        # We coerce to 2-D, scale back, and return a DataFrame.
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        out = X_arr * self.scale_.to_numpy() + self.center_.to_numpy()
        return pd.DataFrame(out, index=index, columns=self.center_.index)


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
