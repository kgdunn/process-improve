# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.
"""Online (real-time) monitoring of batches against a fitted BatchPCA model.

Implements the Nomikos-MacGregor online monitoring scheme: build a batchwise
PCA model from good (common-cause) batches, then track a new batch in real time
by projecting the partially-observed trajectories at each time sample (via
:meth:`process_improve.batch.BatchPCA.predict_online`) and comparing the
resulting Hotelling's T2 and SPE against time-varying control limits. The limits
are learned from the same projection applied to the good batches, so the
statistic at each sample is compared against the good-batch spread at that same
point in the batch evolution.

See Nomikos and MacGregor, "Multivariate SPC Charts for Monitoring Batch
Processes", Technometrics, 37, 41-59, 1995.
"""

from __future__ import annotations

import typing

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import Bunch
from sklearn.utils.validation import check_is_fitted

from ..multivariate._limits import spe_calculation

if typing.TYPE_CHECKING:
    import pandas as pd

    from ._batch_pca import BatchPCA


class BatchMonitor(BaseEstimator):
    """Time-varying (online) control limits for a fitted :class:`BatchPCA` model.

    Builds a Hotelling's T2 and SPE control limit at every time sample by
    passing each good batch through the model's projection-to-the-model-plane
    (PMP) estimate at that sample and summarising the good-batch spread. A new
    batch can then be tracked in real time: at each sample its projected T2 and
    SPE are compared against the limit for that sample, flagging abnormal
    behaviour while the batch is still running.

    Parameters
    ----------
    model : BatchPCA
        A fitted :class:`process_improve.batch.BatchPCA` model, ideally built
        from good (common-cause) batches only.
    conf_level : float, default=0.99
        Confidence level for the control limits.

    Attributes (after fitting)
    --------------------------
    spe_limit_over_time_ : np.ndarray of shape (n_timesteps,)
        SPE control limit at each time sample.
    t2_limit_over_time_ : np.ndarray of shape (n_timesteps,)
        Hotelling's T2 control limit at each time sample.
    spe_mean_over_time_, t2_mean_over_time_ : np.ndarray of shape (n_timesteps,)
        Mean good-batch SPE and T2 trace, for plotting the expected trajectory.
    n_timesteps_ : int
        Number of time samples per batch (from the model).

    Examples
    --------
    >>> from process_improve.batch import BatchPCA, BatchMonitor, load_nylon, resample_to_reference
    >>> batches = load_nylon()
    >>> tags = list(next(iter(batches.values())).columns)
    >>> aligned = resample_to_reference(batches, columns_to_align=tags, reference_batch=1)
    >>> good = {k: v for k, v in aligned.items() if k <= 40}
    >>> model = BatchPCA(n_components=3).fit(good)
    >>> monitor = BatchMonitor(model, conf_level=0.99).fit(good)
    >>> trace = monitor.monitor(aligned[49])
    >>> bool(trace.spe_alarm.any())  # doctest: +SKIP
    True

    References
    ----------
    Nomikos, P. and MacGregor, J.F., "Multivariate SPC Charts for Monitoring
    Batch Processes", Technometrics, 37, 41-59, 1995.
    """

    _parameter_constraints: typing.ClassVar = {
        "model": [BaseEstimator],
        "conf_level": [float],
    }

    def __init__(self, model: BatchPCA, *, conf_level: float = 0.99) -> None:
        self.model = model
        self.conf_level = conf_level

    def _trace_for_batch(
        self, batch: pd.DataFrame, initial_conditions: pd.Series | pd.DataFrame | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the (T2, SPE) trace over every time sample for one batch."""
        n_timesteps = self.model.n_timesteps_
        t2 = np.empty(n_timesteps)
        spe = np.empty(n_timesteps)
        for k in range(1, n_timesteps + 1):
            result = self.model.predict_online(batch, upto_k=k, initial_conditions=initial_conditions)
            t2[k - 1] = result.hotellings_t2
            spe[k - 1] = result.spe
        return t2, spe

    def fit(
        self,
        good_batches: dict[typing.Hashable, pd.DataFrame],
        y: object = None,  # noqa: ARG002
        *,
        initial_conditions: pd.DataFrame | None = None,
    ) -> BatchMonitor:
        """Learn the time-varying control limits from good batches.

        Parameters
        ----------
        good_batches : dict[Hashable, pd.DataFrame]
            Standard batch-data dictionary of aligned good (common-cause)
            batches, the same tags and length as the model's training data.
        y : ignored
            Present for sklearn Pipeline compatibility.
        initial_conditions : pd.DataFrame, optional
            The Z block for the good batches; required if (and only if) the
            model was fitted with one.

        Returns
        -------
        self : BatchMonitor
        """
        check_is_fitted(self.model, "loadings_")
        n_timesteps = self.model.n_timesteps_
        batch_ids = list(good_batches.keys())
        t2_matrix = np.empty((len(batch_ids), n_timesteps))
        spe_matrix = np.empty((len(batch_ids), n_timesteps))
        for row, batch_id in enumerate(batch_ids):
            z = None if initial_conditions is None else initial_conditions.loc[[batch_id]]
            t2_matrix[row], spe_matrix[row] = self._trace_for_batch(good_batches[batch_id], z)

        spe_limits = np.array(
            [spe_calculation(spe_matrix[:, k], conf_level=self.conf_level) for k in range(n_timesteps)]
        )
        # The Hotelling's T2 limit is the analytical F-based limit; it does not
        # vary over time, but is stored per-sample so the online chart has a
        # limit array of matching length.
        t2_limit = self.model.hotellings_t2_limit(conf_level=self.conf_level)

        self.spe_limit_over_time_ = spe_limits
        self.t2_limit_over_time_ = np.full(n_timesteps, t2_limit)
        self.spe_mean_over_time_ = spe_matrix.mean(axis=0)
        self.t2_mean_over_time_ = t2_matrix.mean(axis=0)
        self.n_timesteps_ = n_timesteps
        return self

    def monitor(
        self,
        batch: pd.DataFrame,
        upto_k: int | None = None,
        *,
        initial_conditions: pd.Series | pd.DataFrame | None = None,
    ) -> Bunch:
        """Track a batch in real time against the time-varying limits.

        Parameters
        ----------
        batch : pd.DataFrame
            A single aligned batch to monitor (the training tags as columns).
        upto_k : int, optional
            Track only up to this time sample (simulating a still-running
            batch). Defaults to the full batch length.
        initial_conditions : pd.Series or pd.DataFrame, optional
            The Z block for this batch; required if the model was fitted with
            one.

        Returns
        -------
        result : sklearn.utils.Bunch
            With keys ``time`` (1-based sample indices), ``hotellings_t2`` and
            ``spe`` (the batch's statistic traces), ``t2_limit`` and
            ``spe_limit`` (the limits over the same samples), and
            ``t2_alarm`` / ``spe_alarm`` (boolean arrays where the statistic
            exceeds its limit).
        """
        check_is_fitted(self, "spe_limit_over_time_")
        end = self.n_timesteps_ if upto_k is None else int(upto_k)
        if not 1 <= end <= self.n_timesteps_:
            raise ValueError(f"upto_k must lie in [1, {self.n_timesteps_}]; got {upto_k}.")

        t2, spe = self._trace_for_batch(batch, initial_conditions)
        t2, spe = t2[:end], spe[:end]
        t2_limit = self.t2_limit_over_time_[:end]
        spe_limit = self.spe_limit_over_time_[:end]
        return Bunch(
            time=np.arange(1, end + 1),
            hotellings_t2=t2,
            spe=spe,
            t2_limit=t2_limit,
            spe_limit=spe_limit,
            t2_alarm=t2 > t2_limit,
            spe_alarm=spe > spe_limit,
        )
