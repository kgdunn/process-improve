# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.
"""Scaling and centering helpers for the multivariate package (ENG-01).

Holds :class:`MCUVScaler` (mean-center, unit-variance; the preferred scaler for
fitting PCA / PLS models) and the standalone :func:`center` / :func:`scale`
utilities. Depends only on :mod:`process_improve.multivariate._common`.
"""

from __future__ import annotations

import warnings
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
        with warnings.catch_warnings():
            # An all-NaN or single-observation column raises numpy
            # RuntimeWarnings here; both cases are handled explicitly below,
            # so the warnings are noise for the caller.
            warnings.simplefilter("ignore", RuntimeWarning)
            center = np.nanmean(X_arr, axis=0)
            scale = np.nanstd(X_arr, axis=0, ddof=1)
        # Constant columns are left as-is (scale to 1.0) rather than
        # producing inf / nan when transform divides. The guard must also
        # catch: a column with fewer than two observed values, whose
        # nanstd(ddof=1) is NaN (NaN == 0 is False, so the equality test
        # missed it and transform emitted an all-NaN column); and a
        # denormal-tiny standard deviation, whose reciprocal overflows.
        # An all-NaN column additionally has a NaN center; treat it as
        # constant-at-zero so transform passes the NaN cells through
        # unchanged instead of poisoning them further.
        tiny = float(np.finfo(float).tiny) ** 0.5
        scale = np.where(~np.isfinite(scale) | (scale <= tiny), 1.0, scale)
        center = np.where(np.isfinite(center), center, 0.0)

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
    The function, if supplied, must return a vector with as many columns as the matrix X.

    `axis` [optional; default=0] {integer}

    This specifies the axis along which the centering vector will be calculated if not provided.
    The function is applied along the `axis`: 0=down the columns; 1 = across the rows.

    *Missing values*: The sample mean is computed by taking the sum along the `axis`, skipping
    any missing data, and dividing by N = number of values which are present. Values which were
    missing before, are left as missing after.

    Returns
    -------
    centred : DataMatrix
        The centred data, returned when ``extra_output=False`` (the default).
    (centred, centre_vector) : tuple[DataMatrix, np.ndarray]
        When ``extra_output=True``, a tuple of the centred data and the
        centring vector.

    Notes
    -----
    **The extra output of** :func:`center` **and** :func:`scale` **are not the
    same kind of quantity.** :func:`center` returns the value that was
    *subtracted*, so replaying it means subtracting again. :func:`scale`
    returns the *multiplier* it applied, which is the reciprocal of `func`, so
    replaying that one means multiplying, not dividing. Getting the two the
    same way round is wrong by a factor of the variance::

        centred, subtrahend = center(X, extra_output=True)
        scaled, multiplier = scale(centred, extra_output=True)
        # replay on new rows:
        new_scaled = (new_X - subtrahend) * multiplier   # note: minus, then times

    They also disagree on degrees of freedom: :func:`scale` defaults to
    ``ddof=0`` while :class:`MCUVScaler` uses ``ddof=1``, a factor of
    ``sqrt(n / (n - 1))``. Prefer :class:`MCUVScaler` when preparing data for a
    PCA / PLS fit; it does both steps together, keeps the constants as fitted
    attributes, and has an :meth:`~MCUVScaler.inverse_transform`.

    See Also
    --------
    MCUVScaler : Mean-centre and unit-variance scale in one fitted estimator.
    scale : The scaling counterpart, whose extra output is a multiplier.
    """
    # pandas-stubs types apply()'s axis as a Literal, so a plain ``int`` axis does
    # not match any overload; the call is valid at runtime.
    vector = pd.DataFrame(X).apply(func, axis=axis).to_numpy()  # type: ignore[call-overload]  # pandas-stubs axis is Literal
    if axis == 1:
        # Row-wise centring: the statistic is one value per ROW, so it must
        # broadcast down the column axis. Without the reshape numpy broadcasts
        # the length-N vector across the columns instead: a ValueError for
        # N != K and a silently wrong answer for square matrices.
        vector = vector.reshape(-1, 1)
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
    from scipy.stats import median_abs_deviation as my_scale
    X = scale(center(X), func=my_scale)

    Returns
    -------
    scaled : DataMatrix
        The scaled data, returned when ``extra_output=False`` (the default).
    (scaled, scale_vector) : tuple[DataMatrix, np.ndarray]
        When ``extra_output=True``, a tuple of the scaled data and the
        per-column scaling vector (the reciprocal of `func` applied along
        `axis`, with zero entries replaced by 1.0 to leave constant columns
        unchanged) is returned instead.

    Notes
    -----
    **The extra output of** :func:`scale` **and** :func:`center` **are not the
    same kind of quantity.** This function returns the *multiplier* it applied
    (the reciprocal of `func`), whereas :func:`center` returns the value it
    *subtracted*. Replaying a scaling on new rows therefore means multiplying
    by ``scale_vector``; dividing by it is wrong by a factor of the variance.
    If dividing reads more naturally, invert it explicitly and name the
    variable for what it is::

        scaled, multiplier = scale(centred, extra_output=True)
        divisor = 1.0 / multiplier

    The two also disagree on degrees of freedom: this function defaults to
    ``ddof=0`` while :class:`MCUVScaler` uses ``ddof=1``, a factor of
    ``sqrt(n / (n - 1))``. Prefer :class:`MCUVScaler` when preparing data for a
    PCA / PLS fit.

    See Also
    --------
    MCUVScaler : Mean-centre and unit-variance scale in one fitted estimator.
    center : The centring counterpart, whose extra output is a subtrahend.

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
    if axis == 1:
        # Row-wise scaling: one value per ROW; see the reshape note in center().
        vector = vector.reshape(-1, 1)

    if extra_output:
        return np.multiply(X, vector), vector
    else:
        return np.multiply(X, vector)


#: A column whose ``|mean| / sd`` exceeds this is treated as un-centred. The
#: number is chosen from the damage it does: fitting without an intercept
#: displaces every prediction by roughly the block mean, which costs
#: approximately ``(mean / sd) ** 2`` of R², so 0.5 is the point where a quarter
#: of the variance has already been thrown away.
_UNCENTRED_MEAN_RATIO: float = 0.5

#: Tolerances for recognising a block the caller has already mean-centred and
#: unit-variance scaled. Loose enough to accept a ``ddof=0`` scaling (which is
#: off by ``sqrt(n / (n - 1))``, i.e. 2.6% at n=20) as "already scaled".
_PRESCALED_MEAN_ATOL: float = 0.05
_PRESCALED_SD_ATOL: float = 0.05


def _column_moments(X: DataMatrix) -> tuple[np.ndarray, np.ndarray]:
    """Return the NaN-skipping column means and sample (``ddof=1``) standard deviations."""
    values = np.asarray(pd.DataFrame(X), dtype=float)
    with warnings.catch_warnings():
        # All-NaN and single-observation columns raise numpy RuntimeWarnings;
        # both are filtered out by the callers below.
        warnings.simplefilter("ignore", RuntimeWarning)
        means = np.nanmean(values, axis=0)
        sds = np.nanstd(values, axis=0, ddof=1)
    return means, sds


def _uncentred_columns(X: DataMatrix, ratio: float = _UNCENTRED_MEAN_RATIO) -> list:
    """Return the labels of columns whose mean is large relative to their own spread.

    A column with no spread at all cannot have been centred unless its mean is
    also zero, so a constant non-zero column is always reported.

    Parameters
    ----------
    X : DataMatrix
        The block to inspect.
    ratio : float
        Report a column when ``|mean| / sd`` exceeds this. See
        :data:`_UNCENTRED_MEAN_RATIO`.

    Returns
    -------
    list
        Column labels, in column order. Empty when the block looks centred.
    """
    frame = pd.DataFrame(X)
    means, sds = _column_moments(frame)
    # A tiny spread is treated as no spread, mirroring MCUVScaler's constant-column
    # guard, so the division below cannot overflow.
    tiny = float(np.finfo(float).tiny) ** 0.5
    degenerate = ~np.isfinite(sds) | (sds <= tiny)
    with np.errstate(invalid="ignore"):
        flagged = np.where(
            degenerate,
            np.isfinite(means) & (np.abs(means) > tiny),
            np.abs(means) > ratio * np.where(degenerate, 1.0, sds),
        )
    return [label for label, flag in zip(frame.columns, flagged, strict=True) if bool(flag)]


def _looks_prescaled(X: DataMatrix) -> bool:
    """Return True when every non-constant column is already centred and unit-variance.

    Used to warn a caller who has done their own scaling and is about to have it
    re-done (and therefore erased) inside cross-validation folds. Constant
    columns carry no scaling evidence either way and are ignored, unless every
    column is constant, in which case there is nothing to judge and the answer
    is ``False``.
    """
    means, sds = _column_moments(X)
    usable = np.isfinite(means) & np.isfinite(sds) & (sds > 0)
    if not np.any(usable):
        return False
    return bool(
        np.all(np.abs(means[usable]) <= _PRESCALED_MEAN_ATOL)
        and np.all(np.abs(sds[usable] - 1.0) <= _PRESCALED_SD_ATOL)
    )
