# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.
"""scikit-learn transformers over the batch alignment and feature-extraction functions.

:class:`ResampleAligner` and :class:`DTWAligner` put batches of unequal duration on one
common length, and :class:`BatchFeatureExtractor` summarises each batch as one row of
features. Each learns what it needs in ``fit`` and applies it to new batches in
``transform``, so an alignment and a batch model sit in one
:class:`~sklearn.pipeline.Pipeline`, and ``clone``, ``get_params`` and ``set_params`` work
on them. The functions they wrap, :func:`~process_improve.batch.resample_to_reference`,
:func:`~process_improve.batch.batch_dtw` and the extractors in
:mod:`process_improve.batch.features`, remain public.
"""

from __future__ import annotations

import typing

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .features import (
    f_agemax,
    f_agemin,
    f_count,
    f_iqr,
    f_last,
    f_max,
    f_mean,
    f_median,
    f_min,
    f_robust_mad,
    f_std,
    f_sum,
)
from .preprocessing import (
    _BATCH_DTW_SETTINGS,
    _RESAMPLE_SETTINGS,
    _check_long_enough_to_resample,
    _onto_time_axis,
    _refuse_missing_values,
    _resample_to_length,
    _resolve_columns_to_align,
    align_with_path,
    apply_scaling,
    batch_dtw,
    dtw_core,
    find_reference_batch,
)

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Sequence


def _columns_to_align(X: object, columns_to_align: Sequence | None, caller: str) -> list:
    """
    Resolve the columns to align, and refuse batches that lack one or hold a missing value in one.

    The resolution is the one :class:`~process_improve.batch.BatchScaler` uses: the numeric
    columns that vary within at least one batch, unless named. A missing value is refused as
    :func:`~process_improve.batch.resample_to_reference` refuses it: both aligners resample
    every batch, and a dynamic time warping distance with a missing value in it is ``NaN``.
    """
    batches = typing.cast("dict", X)
    named = None if columns_to_align is None else list(columns_to_align)
    columns = list(_resolve_columns_to_align(batches, named, caller))
    lacking = {batch_id: [c for c in columns if c not in batch] for batch_id, batch in batches.items()}
    lacking = {batch_id: tags for batch_id, tags in lacking.items() if tags}
    if lacking:
        raise ValueError(f"{caller} needs every batch to hold the tags it aligns; missing (batch: tags): {lacking}.")
    _refuse_missing_values(batches, columns, caller)
    return columns


def _reference_batch(X: dict, columns: list, reference_batch: Hashable | None, settings: dict | None) -> Hashable:
    """
    Return the batch named as the reference, or the most typical one.

    The most typical batch is the one :func:`~process_improve.batch.find_reference_batch`
    picks. Its Hotelling's T2 screen judges the batches its PCA was fitted to, so it is
    asked for the limit of those rows (``judged_rows="fitted"``), the default from 3.0,
    unless ``settings`` says otherwise.
    """
    if reference_batch is not None:
        if reference_batch not in X:
            raise KeyError(f"`reference_batch` was not found in the dict of batches; got {reference_batch!r}.")
        return reference_batch
    found = find_reference_batch(X, columns, settings={"judged_rows": "fitted", **(settings or {})})
    return found[0] if isinstance(found, list) else found


class ResampleAligner(TransformerMixin, BaseEstimator):
    """
    Align batches to one common length by resampling them (linear time warping).

    Every batch is interpolated onto evenly spaced samples, as many as the reference batch
    has, as :func:`~process_improve.batch.resample_to_reference` does. This is the simplest
    alignment: it assumes the difference in duration is spread evenly through the batch.
    :meth:`fit` learns the reference batch's length, so :meth:`transform` puts new batches on
    it without the reference batch among them.

    Parameters
    ----------
    columns_to_align : list or None, optional
        The tags to align and return. ``None`` (the default) takes the numeric columns that
        vary within at least one batch, as :class:`~process_improve.batch.BatchScaler` does.
    reference_batch : Hashable or None, optional
        Key of the batch whose length every batch takes. ``None`` (the default) takes the
        most typical batch, chosen by :func:`~process_improve.batch.find_reference_batch`
        with ``judged_rows="fitted"``.
    settings : dict or None, optional
        Passed to :func:`~process_improve.batch.resample_to_reference`, and to
        :func:`~process_improve.batch.find_reference_batch` when it chooses the reference.
        ``{"interpolate_kind": "cubic"}`` is the default; any ``kind`` of
        :func:`scipy.interpolate.interp1d` is accepted.

    Attributes
    ----------
    reference_batch_ : Hashable
        Key of the reference batch.
    n_timesteps_ : int
        Number of samples in every aligned batch: the length of the reference batch.
    columns_to_align_ : list
        The tags aligned, resolved during :meth:`fit`.
    n_features_in_ : int
        Number of tags aligned.

    See Also
    --------
    DTWAligner : nonlinear alignment by dynamic time warping.
    process_improve.batch.resample_to_reference : the function this wraps.

    Examples
    --------
    >>> from process_improve.batch import ResampleAligner, load_nylon
    >>> batches = load_nylon()
    >>> aligner = ResampleAligner(reference_batch=1).fit(batches)
    >>> aligned = aligner.transform(batches)
    >>> {frame.shape for frame in aligned.values()}
    {(114, 10)}
    """

    _parameter_constraints: typing.ClassVar = {
        "columns_to_align": ["array-like", None],
        "reference_batch": "no_validation",
        "settings": [dict, None],
    }

    def __init__(
        self,
        columns_to_align: list | None = None,
        reference_batch: Hashable | None = None,
        settings: dict | None = None,
    ) -> None:
        self.columns_to_align = columns_to_align
        self.reference_batch = reference_batch
        self.settings = settings

    def fit(self, X: dict[Hashable, pd.DataFrame], y: object = None) -> ResampleAligner:  # noqa: ARG002
        """
        Choose the reference batch and learn its length.

        Parameters
        ----------
        X : dict[Hashable, pd.DataFrame]
            Batch data, in the standard format: keyed by batch identifier, one row per sample.
        y : ignored
            Present for :class:`~sklearn.pipeline.Pipeline` compatibility.

        Returns
        -------
        self : ResampleAligner
        """
        self._validate_params()
        columns = _columns_to_align(X, self.columns_to_align, "ResampleAligner.fit")
        _check_long_enough_to_resample(X, "ResampleAligner.fit")
        reference = _reference_batch(X, columns, self.reference_batch, self.settings)
        self.reference_batch_ = reference
        self.n_timesteps_ = int(X[reference].shape[0])
        self.columns_to_align_ = columns
        self.n_features_in_ = len(columns)
        self._interpolate_kind = {**_RESAMPLE_SETTINGS, **(self.settings or {})}["interpolate_kind"]
        return self

    def transform(self, X: dict[Hashable, pd.DataFrame]) -> dict[Hashable, pd.DataFrame]:
        """
        Resample every batch onto the reference batch's number of samples.

        Parameters
        ----------
        X : dict[Hashable, pd.DataFrame]
            Batch data, in the standard format, holding at least the tags of :meth:`fit`.
            The reference batch need not be among them.

        Returns
        -------
        dict[Hashable, pd.DataFrame]
            The batches, in the order given, each with ``n_timesteps_`` rows and the
            ``columns_to_align_`` columns.
        """
        check_is_fitted(self, "n_timesteps_")
        columns = _columns_to_align(X, self.columns_to_align_, "ResampleAligner.transform")
        _check_long_enough_to_resample(X, "ResampleAligner.transform")
        selected = {batch_id: batch[columns] for batch_id, batch in X.items()}
        return _resample_to_length(selected, columns, self.n_timesteps_, self._interpolate_kind)


class DTWAligner(TransformerMixin, BaseEstimator):
    """
    Align batches by iterative, weighted dynamic time warping (DTW).

    :meth:`fit` runs :func:`~process_improve.batch.batch_dtw` (Kassidas, MacGregor and
    Taylor, 1998) on the training batches: it warps every batch nonlinearly onto a reference
    trajectory, and learns a weight per tag that favours the tags which align consistently.
    :meth:`transform` warps any batch, a training batch or a new one, onto the trajectory
    the last iteration aligned to, with the weights that iteration used, and resamples it
    onto the common time axis. A training batch therefore comes back as
    :func:`~process_improve.batch.batch_dtw` aligned it.

    Parameters
    ----------
    columns_to_align : list or None, optional
        The tags that drive the alignment, and that are aligned and returned. ``None`` (the
        default) takes the numeric columns that vary within at least one batch, as
        :class:`~process_improve.batch.BatchScaler` does.
    reference_batch : Hashable or None, optional
        Key of the batch the first iterations align to. ``None`` (the default) takes the
        most typical batch, chosen by :func:`~process_improve.batch.find_reference_batch`
        with ``judged_rows="fitted"``.
    settings : dict or None, optional
        Passed to :func:`~process_improve.batch.batch_dtw`, whose docstring lists them and
        their defaults, and to :func:`~process_improve.batch.find_reference_batch` when it
        chooses the reference. :meth:`transform` uses the same ``subsample``, ``band`` and
        time-axis settings. The default ``show_progress=True`` shows a progress bar of
        :meth:`fit` on stderr; pass ``{"show_progress": False}`` to hide it.

    Attributes
    ----------
    reference_batch_ : Hashable
        Key of the batch the first iterations aligned to.
    reference_trajectory_ : pd.DataFrame
        The trajectory every batch is aligned to, in the original units: the reference
        batch when the weights settled within three iterations, otherwise an average
        trajectory. One row per sample of the reference batch (after ``subsample``).
    weights_ : pd.Series
        The weight of each tag in the alignment, one per ``columns_to_align_``.
    scale_df_ : pd.DataFrame
        The ``Range`` and ``Minimum`` of each tag, from
        :func:`~process_improve.batch.determine_scaling`, that put the tags on one scale
        before their distances are compared.
    n_timesteps_ : int
        Number of samples in every aligned batch: the points of the common time axis.
    columns_to_align_ : list
        The tags aligned, resolved during :meth:`fit`.
    n_features_in_ : int
        Number of tags aligned.

    See Also
    --------
    ResampleAligner : linear alignment by resampling.
    process_improve.batch.batch_dtw : the function this wraps.

    References
    ----------
    Kassidas, A., MacGregor, J.F. and Taylor, P.A., "Synchronization of batch trajectories
    using dynamic time warping", AIChE Journal, 44, 864-875, 1998.

    Examples
    --------
    >>> from process_improve.batch import DTWAligner, load_nylon
    >>> batches = load_nylon()
    >>> training = {batch_id: batches[batch_id] for batch_id in (1, 2, 3, 4)}
    >>> aligner = DTWAligner(reference_batch=1, settings={"show_progress": False, "subsample": 3})
    >>> aligned = aligner.fit_transform(training)
    >>> {frame.shape for frame in aligned.values()}
    {(100, 10)}
    >>> aligner.transform({57: batches[57]})[57].shape
    (100, 10)
    """

    _parameter_constraints: typing.ClassVar = {
        "columns_to_align": ["array-like", None],
        "reference_batch": "no_validation",
        "settings": [dict, None],
    }

    def __init__(
        self,
        columns_to_align: list | None = None,
        reference_batch: Hashable | None = None,
        settings: dict | None = None,
    ) -> None:
        self.columns_to_align = columns_to_align
        self.reference_batch = reference_batch
        self.settings = settings

    def fit(self, X: dict[Hashable, pd.DataFrame], y: object = None) -> DTWAligner:  # noqa: ARG002
        """
        Learn the reference trajectory and the tag weights by iterative weighted DTW.

        Parameters
        ----------
        X : dict[Hashable, pd.DataFrame]
            Batch data, in the standard format: keyed by batch identifier, one row per sample.
        y : ignored
            Present for :class:`~sklearn.pipeline.Pipeline` compatibility.

        Returns
        -------
        self : DTWAligner
        """
        self._validate_params()
        columns = _columns_to_align(X, self.columns_to_align, "DTWAligner.fit")
        reference = _reference_batch(X, columns, self.reference_batch, self.settings)
        # Only the aligned tags go in: batch_dtw warps every column it is given, and a
        # column it cannot interpolate, a text label for instance, would fail there.
        selected = {batch_id: batch[columns] for batch_id, batch in X.items()}
        outputs = batch_dtw(typing.cast("dict", selected), columns, typing.cast("str", reference), self.settings)
        self.reference_batch_ = reference
        self.reference_trajectory_ = outputs["reference_trajectory"]
        self.weights_ = outputs["weight_history"].iloc[-1].rename("weight")
        self.scale_df_ = outputs["scale_df"]
        self._settings = {**_BATCH_DTW_SETTINGS, **(self.settings or {})}
        self.n_timesteps_ = len(next(iter(outputs["aligned_batch_dfdict"].values())))
        self.columns_to_align_ = columns
        self.n_features_in_ = len(columns)
        return self

    def transform(self, X: dict[Hashable, pd.DataFrame]) -> dict[Hashable, pd.DataFrame]:
        """
        Warp every batch onto the reference trajectory, and resample it onto the time axis.

        Parameters
        ----------
        X : dict[Hashable, pd.DataFrame]
            Batch data, in the standard format, holding at least the tags of :meth:`fit`.

        Returns
        -------
        dict[Hashable, pd.DataFrame]
            The batches, in the order given, each with ``n_timesteps_`` rows and the
            ``columns_to_align_`` columns, in the original units.
        """
        check_is_fitted(self, "weights_")
        columns = _columns_to_align(X, self.columns_to_align_, "DTWAligner.transform")
        step = int(self._settings["subsample"])
        reference = apply_scaling({"reference": self.reference_trajectory_}, self.scale_df_, columns)["reference"]
        scaled = apply_scaling(typing.cast("dict", X), self.scale_df_, columns)
        weight_matrix = np.diag(self.weights_.to_numpy())
        aligned = {}
        for batch_id, batch in X.items():
            result = dtw_core(scaled[batch_id].iloc[::step, :], reference, weight_matrix, band=self._settings["band"])
            synced = align_with_path(result.md_path, batch[columns].iloc[::step, :])
            aligned[batch_id] = _onto_time_axis(synced, self._settings)
        return aligned


#: The extractors that take only ``(data, tags, batch_col, phase_col)``, by the name ``features`` uses.
_FEATURES: dict[str, Callable[..., pd.DataFrame]] = {
    "mean": f_mean,
    "median": f_median,
    "std": f_std,
    "iqr": f_iqr,
    "robust_mad": f_robust_mad,
    "sum": f_sum,
    "min": f_min,
    "max": f_max,
    "agemin": f_agemin,
    "agemax": f_agemax,
    "last": f_last,
    "count": f_count,
}

#: The column that tells the batches apart once they are stacked. It holds each batch's
#: position in the dict, not its key, so a key of any type, a tuple among them, is safe.
_POSITION = "__batch_position__"


class BatchFeatureExtractor(TransformerMixin, BaseEstimator):
    """
    Summarise each batch as one row of features, for a model of the final quality.

    Each feature of :mod:`process_improve.batch.features` named in ``features`` is computed
    for every tag of every batch, and the results are joined into one table: one row per
    batch and one column per tag and feature, named as the extractor names it, such as
    ``"Tag01_mean"`` or ``"Tag01_endpoint"``. The table is the X block of a
    :class:`~process_improve.multivariate.PLS` model of the final quality. The batches need
    not be aligned: each feature summarises a batch whatever its duration.

    Parameters
    ----------
    features : sequence of str, optional
        The features to compute, from ``"mean"``, ``"median"``, ``"std"``, ``"iqr"``,
        ``"robust_mad"``, ``"sum"``, ``"min"``, ``"max"``, ``"agemin"``, ``"agemax"``,
        ``"last"`` and ``"count"``. The default is ``("mean", "std", "min", "max", "last")``.
        ``"agemin"`` and ``"agemax"`` report the label of each batch's own index where the
        tag is lowest or highest: a sample number, or a time stamp when the index holds time.
    tags : list or None, optional
        The tags to summarise. ``None`` (the default) takes every numeric column of the first
        batch. A batch identifier left in each frame, as
        :func:`~process_improve.batch.melted_to_dict` leaves it, is a numeric column too:
        name the ``tags`` to leave it out.

    Attributes
    ----------
    tags_ : list
        The tags summarised, resolved during :meth:`fit`.
    n_features_in_ : int
        Number of tags summarised.

    See Also
    --------
    process_improve.batch.f_mean : one of the extractors this wraps.

    Notes
    -----
    A missing value is skipped, as each extractor skips it: ``"mean"`` averages the samples
    that were measured, ``"last"`` is the last measured value, and ``"count"`` counts the
    measured samples.

    Examples
    --------
    >>> from process_improve.batch import BatchFeatureExtractor, load_nylon
    >>> batches = load_nylon()
    >>> table = BatchFeatureExtractor(features=("mean", "last"), tags=["Tag01", "Tag02"]).fit_transform(batches)
    >>> table.shape
    (57, 4)
    >>> list(table.columns)
    ['Tag01_mean', 'Tag02_mean', 'Tag01_endpoint', 'Tag02_endpoint']
    """

    _parameter_constraints: typing.ClassVar = {
        "features": [list, tuple],
        "tags": ["array-like", None],
    }

    def __init__(
        self,
        features: Sequence[str] = ("mean", "std", "min", "max", "last"),
        tags: list | None = None,
    ) -> None:
        self.features = features
        self.tags = tags

    def _checked_tags(self, X: object, tags: Sequence | None, caller: str) -> list:
        """Refuse what is not a dict of batches, and return the tags, each numeric and in every batch."""
        if not isinstance(X, dict):
            raise TypeError(
                f"{caller} expects a dict of per-batch DataFrames, keyed by batch identifier; got {type(X).__name__}."
            )
        if not X:
            raise ValueError(f"{caller} needs at least one batch; got an empty dict.")
        first = X[next(iter(X))]
        if tags is None:
            return list(first.select_dtypes(include="number").columns)
        tags = list(tags)
        lacking = {batch_id: [t for t in tags if t not in batch] for batch_id, batch in X.items()}
        lacking = {batch_id: missing for batch_id, missing in lacking.items() if missing}
        if lacking:
            raise ValueError(f"{caller} needs every batch to hold the tags; missing (batch: tags): {lacking}.")
        text = [t for t in tags if not pd.api.types.is_numeric_dtype(first[t])]
        if text:
            raise ValueError(f"{caller} can only summarise numeric tags; these are not: {text}.")
        return tags

    def fit(self, X: dict[Hashable, pd.DataFrame], y: object = None) -> BatchFeatureExtractor:  # noqa: ARG002
        """
        Resolve the tags, and check that each feature is known.

        Parameters
        ----------
        X : dict[Hashable, pd.DataFrame]
            Batch data, in the standard format: keyed by batch identifier, one row per sample.
        y : ignored
            Present for :class:`~sklearn.pipeline.Pipeline` compatibility.

        Returns
        -------
        self : BatchFeatureExtractor
        """
        self._validate_params()
        unknown = [name for name in self.features if name not in _FEATURES]
        if unknown or not self.features:
            raise ValueError(f"`features` must name at least one of {sorted(_FEATURES)}; got {list(self.features)!r}.")
        self.tags_ = self._checked_tags(X, self.tags, "BatchFeatureExtractor.fit")
        self.n_features_in_ = len(self.tags_)
        self._feature_names_out = np.asarray(self._extract(X).columns, dtype=object)
        return self

    def transform(self, X: dict[Hashable, pd.DataFrame]) -> pd.DataFrame:
        """
        Return one row of features per batch.

        Parameters
        ----------
        X : dict[Hashable, pd.DataFrame]
            Batch data, in the standard format, holding at least the tags of :meth:`fit`.

        Returns
        -------
        pd.DataFrame
            One row per batch, in the order given, indexed by batch identifier
            (``batch_id``), and one column per tag and feature.
        """
        check_is_fitted(self, "tags_")
        self._checked_tags(X, self.tags_, "BatchFeatureExtractor.transform")
        return self._extract(X)

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:  # noqa: ARG002
        """
        Return the column names of :meth:`transform`'s table.

        ``input_features`` is accepted, as :class:`~sklearn.pipeline.Pipeline` passes it,
        and ignored: the names follow from the fitted tags and the ``features``.
        """
        check_is_fitted(self, "tags_")
        return self._feature_names_out

    def _extract(self, X: dict[Hashable, pd.DataFrame]) -> pd.DataFrame:
        """Stack the batches, compute each feature per batch, and join the results side by side."""
        # Each batch keeps its own index, so "agemin" and "agemax" report a label of that batch.
        stacked = pd.concat(
            [batch[self.tags_].assign(**{_POSITION: position}) for position, batch in enumerate(X.values())]
        )
        blocks = []
        for name in self.features:
            block = _FEATURES[name](stacked, tags=list(self.tags_), batch_col=_POSITION)
            blocks.append(block.droplevel(1))  # the extractors also group by a phase; there is one
        table = pd.concat(blocks, axis=1).reindex(range(len(X)))
        table.index = pd.Index(list(X), name="batch_id", tupleize_cols=False)
        table.columns.name = None
        return table
