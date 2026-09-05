# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.
"""Shared helpers for projecting a partially observed batch (private).

A running batch has its unfolded ``[Z | X]`` row complete up to the current
time sample and missing after it. :class:`~process_improve.batch.BatchPCA`,
:class:`~process_improve.batch.BatchPLS` and the mid-course corrector in
:mod:`process_improve.batch.control` all build that NaN-padded row in the
model's scaled space and hand it to the missing-data projection in
:mod:`process_improve.multivariate._projection`. The pieces they share live
here so there is one copy of each.
"""

from __future__ import annotations

import typing

import numpy as np
import pandas as pd
from sklearn.utils import Bunch

if typing.TYPE_CHECKING:
    from collections.abc import Hashable, Sequence


class OnlineModel(typing.Protocol):
    """The fitted attributes a batch model must expose for the online helpers."""

    @property
    def feature_columns_(self) -> pd.Index: ...

    @property
    def center_(self) -> pd.Series: ...

    @property
    def scale_(self) -> pd.Series: ...

    @property
    def tag_names_(self) -> Sequence[Hashable]: ...

    @property
    def initial_condition_names_(self) -> Sequence[Hashable]: ...

    @property
    def n_initial_conditions_(self) -> int: ...

    @property
    def n_timesteps_(self) -> int: ...


def unfolded_layout(feature_columns: pd.Index) -> Bunch:
    """Describe the unfolded column layout of a fitted batch model.

    Parameters
    ----------
    feature_columns : pd.MultiIndex
        The model's ``feature_columns_``: 2-level ``(tag, sequence)`` labels,
        with initial-condition columns labelled ``(name, "")``.

    Returns
    -------
    layout : sklearn.utils.Bunch
        With keys ``is_z`` (bool array, True for initial-condition columns),
        ``sequence`` (int array, the time sample of each trajectory column and
        ``-1`` for initial conditions) and ``tags`` (object array of the tag or
        initial-condition name of each column).
    """
    if not isinstance(feature_columns, pd.MultiIndex):
        raise TypeError("The model's feature columns must carry the 2-level (tag, sequence) index.")
    sequence = feature_columns.get_level_values("sequence")
    is_z = np.array([s == "" for s in sequence])
    seq_num = np.array([-1 if z else int(s) for s, z in zip(sequence, is_z, strict=True)])
    return Bunch(is_z=is_z, sequence=seq_num, tags=np.asarray(feature_columns.get_level_values("tag")))


def coerce_single_initial_conditions(
    model: OnlineModel, initial_conditions: pd.Series | pd.DataFrame | None
) -> pd.DataFrame | None:
    """Normalise a single batch's initial conditions to a 1-row DataFrame.

    Accepts a Series (one value per initial condition) or a single-row
    DataFrame, and validates presence against how ``model`` was fitted.
    """
    if model.n_initial_conditions_ == 0:
        if initial_conditions is not None:
            raise ValueError("The model was fitted without initial conditions; do not pass any.")
        return None
    if initial_conditions is None:
        raise ValueError("The model was fitted with initial conditions; they are required here.")
    frame = initial_conditions.to_frame().T if isinstance(initial_conditions, pd.Series) else initial_conditions
    if frame.shape[0] != 1:
        raise ValueError("initial_conditions for a single batch must have exactly one row.")
    return frame.set_axis(["_online_"], axis=0)


def observed_series(
    model: OnlineModel,
    batch_so_far: pd.DataFrame,
    initial_conditions: pd.Series | pd.DataFrame | None,
    k: int,
) -> pd.Series:
    """Engineering-unit values of the cells observed after ``k`` time samples.

    Parameters
    ----------
    model : BatchPCA or BatchPLS
        A fitted batch model.
    batch_so_far : pd.DataFrame
        The batch's trajectories, at least ``k`` rows, the training tags as
        columns.
    initial_conditions : pd.Series or pd.DataFrame, optional
        The batch's Z values; required if (and only if) the model was fitted
        with initial conditions.
    k : int
        Number of leading time samples that have been observed.

    Returns
    -------
    pd.Series
        Indexed by the unfolded labels ``(name, "")`` and ``(tag, s)`` for
        ``s`` in ``0 .. k-1``, in engineering units.
    """
    if list(batch_so_far.columns) != list(model.tag_names_):
        raise ValueError(
            f"batch_so_far must carry exactly the training tags {model.tag_names_}; got {list(batch_so_far.columns)}."
        )
    if len(batch_so_far) < k:
        raise ValueError(f"batch_so_far has {len(batch_so_far)} rows; at least k = {k} are needed.")
    entries: dict[tuple[Hashable, Hashable], float] = {}
    if model.n_initial_conditions_:
        if initial_conditions is None:
            raise ValueError("The model was fitted with initial conditions; they are required here.")
        z_row = initial_conditions.iloc[0] if isinstance(initial_conditions, pd.DataFrame) else initial_conditions
        for name in model.initial_condition_names_:
            if name not in z_row.index:
                raise ValueError(f"initial_conditions is missing {name!r}.")
            entries[(name, "")] = float(z_row.get(name, np.nan))
    elif initial_conditions is not None:
        raise ValueError("The model was fitted without initial conditions; do not pass any.")
    values = batch_so_far.iloc[:k].to_numpy(dtype=float)
    for j, tag in enumerate(model.tag_names_):
        for s in range(k):
            entries[(tag, s)] = float(values[s, j])
    return pd.Series(entries)


def scaled_row(model: OnlineModel, observed: pd.Series) -> np.ndarray:
    """Place observed engineering-unit values into a NaN-padded row of the model's scaled space.

    Parameters
    ----------
    model : BatchPCA or BatchPLS
        A fitted batch model with public ``feature_columns_``, ``center_`` and
        ``scale_``.
    observed : pd.Series
        Values indexed by unfolded column labels (see :func:`observed_series`).

    Returns
    -------
    np.ndarray of shape (n_unfolded_features,)
        Scaled values where observed, NaN elsewhere.
    """
    features = pd.Index(model.feature_columns_)
    positions = features.get_indexer(observed.index)
    if (positions < 0).any():
        unknown = [label for label, pos in zip(observed.index, positions, strict=True) if pos < 0][:3]
        raise ValueError(f"observed carries labels that are not model columns, e.g. {unknown}.")
    center = model.center_.to_numpy(dtype=float)
    scale = model.scale_.to_numpy(dtype=float)
    row = np.full(len(features), np.nan)
    row[positions] = (observed.to_numpy(dtype=float) - center[positions]) / scale[positions]
    return row


def stack_online_patterns(scaled_full_row: np.ndarray, layout: Bunch, n_timesteps: int) -> np.ndarray:
    """Stack the per-sample missingness patterns of one complete scaled row.

    Row ``k - 1`` of the result is the row as it looks after ``k`` samples:
    initial conditions and the trajectory columns with ``sequence < k`` kept,
    every later trajectory column set to NaN.
    """
    stacked = np.tile(scaled_full_row, (n_timesteps, 1))
    for k in range(1, n_timesteps + 1):
        observed = layout.is_z | (layout.sequence < k)
        stacked[k - 1, ~observed] = np.nan
    return stacked


def residuals_of(stacked: np.ndarray, scores: np.ndarray, x_loadings: np.ndarray) -> np.ndarray:
    """Residual of each (possibly NaN-padded) row after reconstruction from its scores.

    NaN where the row was unobserved; ``stacked - scores @ x_loadings.T``.
    """
    return stacked - scores @ x_loadings.T


def instantaneous_spe(residuals: np.ndarray, layout: Bunch) -> np.ndarray:
    """SPE of the newest observed sample only, for each row of a pattern stack.

    Row ``k - 1`` holds the row after ``k`` samples, so its newest sample has
    ``sequence == k - 1``; the statistic is the length of the residual over
    that sample's tags, the per-interval SPE of Nomikos and MacGregor (1995).
    """
    n = residuals.shape[0]
    out = np.empty(n)
    for k in range(n):
        cells = layout.sequence == k
        out[k] = np.sqrt(np.nansum(residuals[k, cells] ** 2))
    return out


def forecast_frame(
    model: OnlineModel, scores: np.ndarray, batch: pd.DataFrame, upto_k: int, x_loadings: np.ndarray
) -> pd.DataFrame:
    """Trajectories of a batch with the unobserved remainder imputed from its score estimate.

    The imputation is ``tau @ P'`` mapped back to engineering units, the
    projection-to-the-model-plane forecast of Wold, Kettaneh-Wold, MacGregor
    and Dunn (2009), Eq. 4. Samples before ``upto_k`` are the batch's own
    values.
    """
    layout = unfolded_layout(model.feature_columns_)
    imputed = scores @ x_loadings.T * model.scale_.to_numpy(dtype=float) + model.center_.to_numpy(dtype=float)
    columns = {}
    for tag in model.tag_names_:
        cells = np.where((layout.tags == tag) & ~layout.is_z)[0]
        columns[tag] = imputed[cells[np.argsort(layout.sequence[cells])]]
    frame = pd.DataFrame(columns, index=pd.RangeIndex(int(model.n_timesteps_), name="sequence"))
    frame.iloc[:upto_k] = batch.iloc[:upto_k][model.tag_names_].to_numpy(dtype=float)
    return frame
