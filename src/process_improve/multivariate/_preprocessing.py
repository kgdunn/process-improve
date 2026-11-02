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

from ._common import DataMatrix, SpecificationWarning, _reject_sparse


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

        Raises
        ------
        TypeError
            If ``X`` is a SciPy sparse matrix. Centring makes every zero non-zero, so
            there is no sparse path to take; see :func:`_reject_sparse` for the remedy
            the message names (#399).
        """
        _reject_sparse(X, "MCUVScaler")
        # Convenience: accept a 1-D Series (a single-column y, common when
        # the scaler is used for the target side of a PLS fit). validate_data
        # itself requires 2-D input, so promote here before it sees X.
        if isinstance(X, pd.Series):
            X = X.to_frame()
        # Label from the frame itself, as PCA and PLS do: sklearn sets
        # feature_names_in_ only for string labels, so integer labels (a
        # spectrum's wavelengths) would otherwise come out as 0..K-1. An array
        # has no labels, and pd.Series numbers its columns 0..K-1 from None.
        index: pd.Index | None = X.columns if isinstance(X, pd.DataFrame) else None
        # Dense (sparse was refused above), numeric, at least one column: validate_data's defaults.
        X_arr = validate_data(self, X, ensure_min_samples=2, ensure_all_finite="allow-nan")

        # nanmean / nanstd so NaN cells pass through with the right
        # column-level statistics (the chemometric pipeline's missing-data
        # contract). std uses ddof=1: this is the difference from
        # sklearn.preprocessing.StandardScaler.
        with warnings.catch_warnings():
            # An all-NaN or single-observation column raises numpy
            # RuntimeWarnings here; both cases are handled explicitly below,
            # so the warnings are noise for the caller. Nothing else here warns.
            warnings.simplefilter("ignore")
            center = np.nanmean(X_arr, axis=0)
            scale = np.nanstd(X_arr, axis=0, ddof=1)
            highest = np.nanmax(X_arr, axis=0)
            constant = highest == np.nanmin(X_arr, axis=0)
        # Constant columns are left as-is (scale to 1.0) rather than
        # producing inf / nan when transform divides. A column is constant when
        # its observed values are all equal: its nanstd is more often rounding
        # noise (1e-17) than an exact 0, and which one depends on the array's
        # memory layout, so testing the standard deviation turned two in three
        # constant columns into a column of +-sqrt((n-1)/n). Its centre is the
        # value itself, so it transforms to exactly 0. The guard must also
        # catch: a column with fewer than two observed values, whose
        # nanstd(ddof=1) is NaN (NaN == 0 is False, so the equality test
        # missed it and transform emitted an all-NaN column); and a
        # denormal-tiny standard deviation, whose reciprocal overflows.
        # An all-NaN column additionally has a NaN center; treat it as
        # constant-at-zero so transform passes the NaN cells through
        # unchanged instead of poisoning them further.
        tiny = float(np.finfo(float).tiny) ** 0.5
        scale = np.where(constant | ~np.isfinite(scale) | (scale <= tiny), 1.0, scale)
        center = np.where(constant, highest, center)
        center = np.where(np.isfinite(center), center, 0.0)

        self.center_ = pd.Series(center, index=index)
        self.scale_ = pd.Series(scale, index=index)
        return self

    def transform(self, X: DataMatrix, y=None) -> pd.DataFrame:  # noqa: ANN001, ARG002
        """Mean-centre and unit-variance scale ``X``.

        ``y`` is accepted (and ignored) for :class:`Pipeline` compatibility.
        """
        check_is_fitted(self)
        # Mirror fit()'s Series convenience for symmetric round-tripping.
        if isinstance(X, pd.Series):
            X = X.to_frame()
        # Preserve the row index for DataFrame input; ndarray input falls
        # back to a RangeIndex.
        index = X.index if isinstance(X, pd.DataFrame) else None
        X_arr = validate_data(  # pragma: no mutate - validate_data reads reset=None as reset=False
            self,
            X,
            reset=False,
            ensure_all_finite="allow-nan",
        )
        out = (X_arr - self.center_.to_numpy()) / self.scale_.to_numpy()
        return pd.DataFrame(out, index=index, columns=self.center_.index)

    def inverse_transform(self, X: DataMatrix) -> pd.DataFrame:
        """Inverse the mean-centring and unit-variance scaling."""
        check_is_fitted(self)
        index = X.index if isinstance(X, pd.DataFrame) else None
        # inverse_transform is intentionally NOT routed through validate_data:
        # callers (TransformedTargetRegressor included) pass ndarray output
        # from a downstream estimator that may have a different shape than
        # the fit-time X (typical: 1-D y_pred for a single-target regressor).
        # We coerce to 2-D, scale back, and return a DataFrame.
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, np.newaxis]
        out = X_arr * self.scale_.to_numpy() + self.center_.to_numpy()
        return pd.DataFrame(out, index=index, columns=self.center_.index)


def _back_to_units(*scalers: MCUVScaler | None) -> tuple[np.ndarray | float, np.ndarray | float]:
    """Return ``(centre, scale)`` with ``units = scaled * scale + centre``, undoing ``scalers`` in the order given.

    Each scaler maps ``u`` to ``(u - center_) / scale_``, so the one applied last is undone
    first; a missing scaler is the identity.
    """
    centre: np.ndarray | float = 0.0
    scale: np.ndarray | float = 1.0
    for scaler in scalers:
        if scaler is not None:
            centre = centre * scaler.scale_.to_numpy() + scaler.center_.to_numpy()
            scale = scale * scaler.scale_.to_numpy()
    return centre, scale


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

    *Missing values*: with the default ``func=np.mean``, any NaN along the reduction axis
    propagates into the centring vector, so an entire row or column of the returned data can end
    up NaN. To skip missing entries instead (summing along the `axis`, dividing by the number of
    values that are present, and leaving pre-existing NaNs as NaNs in the output), pass
    ``func=np.nanmean``.

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
        vector = vector[:, np.newaxis]
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
        The default (``np.std``) calculates the standard deviation of the data
        along the required `axis` and uses that as `scale`. Each column (each
        row for ``axis=1``) reaches `func` as a pandas Series, so ``np.std``
        computes pandas' standard deviation, which skips missing entries as
        ``np.nanstd`` does.

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

    Constant columns, whose observed values are all equal, are left unchanged
    (their multiplier is 1.0), mirroring :class:`MCUVScaler`, so no ``inf`` /
    ``NaN`` is introduced. The values are tested rather than the standard
    deviation, which for equal values is usually rounding noise, not 0.

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
        `axis`, and 1.0 for a constant column, which is left unchanged) is
        returned instead.

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
    if func is np.std:  # ``ddof`` is a parameter of its own, so it can never be in ``kwargs``
        kwargs["ddof"] = ddof
    # pandas-stubs types apply()'s axis as a Literal, so a plain ``int`` axis does
    # not match any overload; the call is valid at runtime.
    frame = pd.DataFrame(X)
    vector = frame.apply(func, axis=axis, **kwargs).to_numpy()  # type: ignore[call-overload]  # pandas-stubs axis is Literal
    # Constant columns (observed values all equal) are left as-is, mirroring
    # MCUVScaler. The values are tested, not the spread: the standard deviation of
    # equal values is more often rounding noise (1e-17) than an exact 0, and
    # 1.0 / vector then multiplied the column by about 1e16.
    constant = (frame.max(axis=axis) == frame.min(axis=axis)).to_numpy()  # type: ignore[call-overload]  # as above
    vector = np.where(constant | (vector == 0), 1.0, vector)
    vector = 1.0 / vector
    if axis == 1:
        # Row-wise scaling: one value per ROW; see the reshape note in center().
        vector = vector[:, np.newaxis]

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
        # both are filtered out by the callers below. Nothing else here warns.
        warnings.simplefilter("ignore")
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
    flagged = np.where(degenerate, np.isfinite(means) & (np.abs(means) > tiny), np.abs(means) > ratio * sds)
    return list(frame.columns[flagged])


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
    spread_error = np.abs(sds[usable] - 1.0)
    # Near 1, |sd - 1| is a whole number of steps of 2**-53, and 0.05 is not, so the two never meet.
    unit_variance = (spread_error <= _PRESCALED_SD_ATOL).all()  # pragma: no mutate - so < is the same as <=
    return bool(np.all(np.abs(means[usable]) <= _PRESCALED_MEAN_ATOL) and unit_variance)


def _warn_scaling_traps(X: DataMatrix, *, scale_inside_folds: bool, fold: str, metric: str) -> None:
    """Warn about the two ways scaling goes wrong in ``select_n_components``.

    Shared by :meth:`PCA.select_n_components` and :meth:`PLS.select_n_components`
    so the two selectors keep the same contract. With ``scale_inside_folds=False``
    the caller's full-data scaling leaks into every fold. With ``True`` and an
    ``X`` that is already centred and unit-variance scaled, the in-fold
    re-standardisation erases the scaling the caller chose, so a comparison
    between two scalings can never show which suits the data.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The block handed to the selector, before any in-fold scaling.
    scale_inside_folds : bool
        The flag as the caller passed it.
    fold : str
        What one fold is called in the message, e.g. ``"CV fold"`` or
        ``"element-fold"``.
    metric : str
        The reported error the message names, e.g. ``"RMSECV"`` or ``"PRESS"``.

    Notes
    -----
    ``stacklevel=3`` attributes the warning to the caller of the selector,
    not to the selector or to this helper.
    """
    if not scale_inside_folds:
        warnings.warn(
            f"scale_inside_folds=False leaks centring/scaling estimated on the full "
            f"dataset into every {fold}, making the reported {metric} optimistic. "
            "The default scale_inside_folds=True is preferred.",
            SpecificationWarning,
            stacklevel=3,  # pragma: no mutate - mutmut's trampoline adds frames, so no test can pin it there
        )
    elif _looks_prescaled(X):
        warnings.warn(
            "X is already centred and unit-variance scaled, so scale_inside_folds=True "
            f"is not protecting you from leakage: it re-standardises inside every {fold} "
            "and overwrites the scaling you chose. Two deliberately different scalings "
            "(say autoscale versus Pareto) collapse onto the same standardised model "
            f"this way, so a comparison of their {metric} reflects only the units the "
            "blocks arrived in, never which scaling suits the data. Pass the raw, "
            "unscaled X and let the folds scale it, or keep your own scaling and set "
            "scale_inside_folds=False (accepting the optimism that flag warns about).",
            SpecificationWarning,
            stacklevel=3,  # pragma: no mutate - mutmut's trampoline adds frames, so no test can pin it there
        )
