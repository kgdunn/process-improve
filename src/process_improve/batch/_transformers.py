# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.
"""sklearn-style transformer facades for the batch preprocessing pipeline.

These wrap the standalone alignment, scaling, and feature-extraction functions
in :mod:`process_improve.batch.preprocessing` and
:mod:`process_improve.batch.features` in the ``BaseEstimator`` /
``TransformerMixin`` API, so a batch workflow can be expressed as an sklearn
pipeline and the learned state (reference batch, weights, scaling ranges) is
carried on a fitted estimator. The free functions remain the implementation
layer and are unchanged.
"""

from __future__ import annotations

import typing

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

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
    align_with_path,
    apply_scaling,
    batch_dtw,
    determine_scaling,
    dtw_core,
    find_reference_batch,
    resample_to_reference,
    reverse_scaling,
)

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Hashable


def _str_keyed(batches: dict) -> dict:
    """Cast a batch dict to the ``dict[str, DataFrame]`` the wrapped functions annotate.

    The alignment / scaling free functions annotate their batch-key type as
    ``str``, but they operate on any hashable key (the bundled datasets use
    integer batch ids). This cast keeps mypy satisfied without changing the
    runtime behaviour.
    """
    return typing.cast("dict[str, pd.DataFrame]", batches)

# The feature extractors that share the uniform (data, tags, batch_col,
# phase_col) signature, so they can be dispatched by name.
_UNIFORM_FEATURES: dict[str, Callable] = {
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


class BatchScaler(TransformerMixin, BaseEstimator):
    """Range-scale batch trajectories, learning the ranges from a training set.

    Wraps :func:`process_improve.batch.determine_scaling` (fit) and
    :func:`process_improve.batch.apply_scaling` /
    :func:`process_improve.batch.reverse_scaling` (transform /
    inverse_transform). Each tag is scaled to a robust range so that no single
    high-variance tag dominates a downstream alignment or model.

    Parameters
    ----------
    columns_to_align : list, optional
        Tags to scale; defaults to every column of the first batch.
    robust : bool, default=True
        Use a robust range (98th minus 2nd percentile) rather than the raw
        (max minus min) range.

    Attributes (after fitting)
    --------------------------
    scale_df_ : pd.DataFrame
        The learned per-tag range and offset.
    """

    def __init__(self, columns_to_align: list | None = None, *, robust: bool = True) -> None:
        self.columns_to_align = columns_to_align
        self.robust = robust

    def fit(self, X: dict[Hashable, pd.DataFrame], y: object = None) -> BatchScaler:  # noqa: ARG002
        """Learn the per-tag scaling ranges from the batches ``X``."""
        self.scale_df_ = determine_scaling(
            batches=_str_keyed(X), columns_to_align=self.columns_to_align, settings={"robust": self.robust}
        )
        return self

    def transform(self, X: dict[Hashable, pd.DataFrame]) -> dict:
        """Scale the batches using the learned ranges."""
        return apply_scaling(_str_keyed(X), self.scale_df_, self.columns_to_align)

    def inverse_transform(self, X: dict[Hashable, pd.DataFrame]) -> dict:
        """Undo the scaling, returning the batches to their original units."""
        return reverse_scaling(_str_keyed(X), self.scale_df_, self.columns_to_align)


class ResampleAligner(TransformerMixin, BaseEstimator):
    """Align batches to a common length by resampling (linear time warping).

    Wraps :func:`process_improve.batch.resample_to_reference`: every batch is
    linearly resampled onto the sample grid of a reference batch, so all
    batches end up with the same number of samples. This is the simplest
    alignment and assumes the duration difference is spread evenly through the
    batch.

    Parameters
    ----------
    columns_to_align : list
        Tags to align.
    reference_batch : Hashable, optional
        Key of the batch whose length is the target. If omitted, the most
        central batch is chosen with
        :func:`process_improve.batch.find_reference_batch` at fit time.

    Attributes (after fitting)
    --------------------------
    reference_batch_ : Hashable
        The reference batch key actually used.
    """

    def __init__(self, columns_to_align: list, reference_batch: Hashable | None = None) -> None:
        self.columns_to_align = columns_to_align
        self.reference_batch = reference_batch

    reference_batch_: Hashable

    def fit(self, X: dict[Hashable, pd.DataFrame], y: object = None) -> ResampleAligner:  # noqa: ARG002
        """Choose (or record) the reference batch."""
        if self.reference_batch is None:
            reference = find_reference_batch(_str_keyed(X), columns_to_align=list(self.columns_to_align))
            self.reference_batch_ = reference[0] if isinstance(reference, list) else reference
        else:
            self.reference_batch_ = self.reference_batch
        self._reference_length = X[self.reference_batch_].shape[0]
        return self

    def transform(self, X: dict[Hashable, pd.DataFrame]) -> dict:
        """Resample every batch onto the reference length."""
        # resample_to_reference needs the reference batch present in the dict;
        # supply a target of the learned length via any batch of that length,
        # falling back to the stored reference when it is present.
        reference = self.reference_batch_ if self.reference_batch_ in X else next(iter(X))
        resampled = resample_to_reference(
            _str_keyed(X), list(self.columns_to_align), reference_batch=typing.cast("str", reference)
        )
        if resampled[reference].shape[0] != self._reference_length:
            # Force the learned length by resampling against a synthetic index.
            resampled = {k: _resample_frame(v, self._reference_length) for k, v in resampled.items()}
        return resampled


def _resample_frame(frame: pd.DataFrame, n_samples: int) -> pd.DataFrame:
    """Linearly resample every column of ``frame`` to ``n_samples`` rows."""
    source = np.linspace(0.0, 1.0, frame.shape[0])
    target = np.linspace(0.0, 1.0, n_samples)
    return pd.DataFrame(
        {column: np.interp(target, source, frame[column].to_numpy()) for column in frame.columns}
    )


class DTWAligner(TransformerMixin, BaseEstimator):
    """Align batches with iterative weighted dynamic time warping (DTW).

    Wraps :func:`process_improve.batch.batch_dtw` (Kassidas et al.): learns a
    reference trajectory and per-tag weights from the training batches, warping
    every batch nonlinearly onto the reference. New batches are aligned to the
    learned reference with the learned weights.

    Parameters
    ----------
    columns_to_align : list
        Tags to align.
    reference_batch : Hashable, optional
        Key of the initial reference batch. If omitted, it is chosen with
        :func:`process_improve.batch.find_reference_batch` at fit time.
    settings : dict, optional
        Extra settings forwarded to :func:`process_improve.batch.batch_dtw`.

    Attributes (after fitting)
    --------------------------
    aligned_batches_ : dict
        The training batches after alignment.
    reference_batch_ : Hashable
        The initial reference batch key.
    scale_df_ : pd.DataFrame
        The scaling ranges used internally by the DTW.
    weights_ : np.ndarray
        The final per-tag weight vector.
    """

    def __init__(
        self,
        columns_to_align: list,
        reference_batch: Hashable | None = None,
        settings: dict | None = None,
    ) -> None:
        self.columns_to_align = columns_to_align
        self.reference_batch = reference_batch
        self.settings = settings

    reference_batch_: Hashable

    def fit(self, X: dict[Hashable, pd.DataFrame], y: object = None) -> DTWAligner:  # noqa: ARG002
        """Run iterative weighted DTW and store the reference and weights."""
        columns = list(self.columns_to_align)
        if self.reference_batch is None:
            reference = find_reference_batch(_str_keyed(X), columns_to_align=columns, settings=self.settings)
            self.reference_batch_ = reference[0] if isinstance(reference, list) else reference
        else:
            self.reference_batch_ = self.reference_batch

        result = batch_dtw(
            _str_keyed(X),
            columns_to_align=columns,
            reference_batch=typing.cast("str", self.reference_batch_),
            settings=self.settings,
        )
        self.aligned_batches_ = result["aligned_batch_dfdict"]
        self.scale_df_ = result["scale_df"]
        self.weights_ = result["weight_history"].iloc[-1].to_numpy()
        self._reference = result["last_average_batch"]
        self._reference_scaled = apply_scaling({"_ref_": self._reference}, self.scale_df_, columns)["_ref_"]
        self._n_reference_rows = self.aligned_batches_[next(iter(self.aligned_batches_))].shape[0]
        return self

    def transform(self, X: dict[Hashable, pd.DataFrame]) -> dict:
        """Align batches to the learned reference.

        Training batches return their stored aligned version; any other batch
        is warped to the learned reference using the learned weights.
        """
        columns = list(self.columns_to_align)
        weight_matrix = np.diag(self.weights_)
        out = {}
        for batch_id, batch in X.items():
            if batch_id in self.aligned_batches_:
                out[batch_id] = self.aligned_batches_[batch_id]
                continue
            scaled = apply_scaling(_str_keyed({batch_id: batch}), self.scale_df_, columns)[batch_id]
            result = dtw_core(scaled[columns], self._reference_scaled[columns], weight_matrix)
            synced = align_with_path(result.md_path, batch)
            out[batch_id] = _resample_frame(
                synced[[c for c in synced.columns if c in columns]], self._n_reference_rows
            )
        return out


class BatchFeatureExtractor(TransformerMixin, BaseEstimator):
    """Extract per-batch landmark features into a batch-by-feature matrix.

    Wraps the feature extractors in :mod:`process_improve.batch.features`,
    turning a batch-data dictionary into a single ``(n_batches, n_features)``
    table (one row per batch, one column per tag-and-feature) suitable as the X
    block of a PLS-to-quality model.

    Parameters
    ----------
    features : sequence of str, optional
        Which features to compute; defaults to ``("mean", "std", "min",
        "max", "last")``. Each must be one of the uniform-signature extractors:
        mean, median, std, iqr, robust_mad, sum, min, max, agemin, agemax,
        last, count.
    tags : list of str, optional
        Which tags to extract features from; defaults to every tag.
    phase_col : str, optional
        Column identifying the phase within a batch, passed through to the
        extractors.

    Attributes (after fitting)
    --------------------------
    feature_names_out_ : list of str
        The generated ``"{tag}_{feature}"`` column names.
    """

    def __init__(
        self,
        features: typing.Sequence[str] = ("mean", "std", "min", "max", "last"),
        tags: list[str] | None = None,
        phase_col: str | None = None,
    ) -> None:
        self.features = features
        self.tags = tags
        self.phase_col = phase_col

    def _extract(self, X: dict[Hashable, pd.DataFrame]) -> pd.DataFrame:
        """Compute and horizontally concatenate the requested feature blocks."""
        unknown = [name for name in self.features if name not in _UNIFORM_FEATURES]
        if unknown:
            raise ValueError(
                f"Unknown feature(s) {unknown}; choose from {sorted(_UNIFORM_FEATURES)}."
            )
        # Melt directly (allowing unequal batch lengths, unlike dict_to_melted,
        # which requires aligned batches): landmark features summarise each
        # batch regardless of its duration.
        frames: list[pd.DataFrame] = []
        for batch_id, batch in X.items():
            frame = batch.copy()
            frame["batch_id"] = np.full(len(batch), batch_id, dtype=object)
            frames.append(frame)
        melted = pd.concat(frames, ignore_index=True)
        blocks = []
        for name in self.features:
            block = _UNIFORM_FEATURES[name](melted, tags=self.tags, batch_col="batch_id", phase_col=self.phase_col)
            blocks.append(block.reset_index(drop=True) if self.phase_col is None else block)
        combined = pd.concat(blocks, axis=1)
        combined.index = pd.Index(list(X.keys()), name="batch_id")
        return combined

    def fit(self, X: dict[Hashable, pd.DataFrame], y: object = None) -> BatchFeatureExtractor:  # noqa: ARG002
        """Record the generated feature-column names."""
        self.feature_names_out_ = list(self._extract(X).columns)
        return self

    def transform(self, X: dict[Hashable, pd.DataFrame]) -> pd.DataFrame:
        """Return the ``(n_batches, n_features)`` feature matrix for ``X``."""
        return self._extract(X)
