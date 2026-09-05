# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.
"""Online (real-time) monitoring of batches against a fitted batch model.

Implements the Nomikos-MacGregor online monitoring scheme: build a batchwise
model from good (common-cause) batches, then track a new batch in real time
by projecting the partially-observed trajectories at each time sample (via
:meth:`process_improve.batch.BatchPCA.predict_online_trace` or
:meth:`process_improve.batch.BatchPLS.predict_online_trace`) and comparing the
resulting Hotelling's T2 and SPE against per-sample control limits. The
limits are learned from the same projection applied to the reference batches,
so the statistic at each sample is compared against the reference-batch spread
at that same point in the batch evolution: the score estimates early in a batch
are noisier than the estimates near its end, and the T2 at each sample uses the
covariance of the reference batches' score estimates at that sample.

See Nomikos and MacGregor, "Multivariate SPC Charts for Monitoring Batch
Processes", Technometrics, 37, 41-59, 1995.
"""

from __future__ import annotations

import operator
import typing

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import Bunch
from sklearn.utils.validation import check_is_fitted

from .._linalg import safe_inverse
from ..multivariate._limits import hotellings_t2_limit, spe_calculation

if typing.TYPE_CHECKING:
    import pandas as pd

    from ._batch_pca import BatchPCA
    from ._batch_pls import BatchPLS

SPE_STATISTICS = ("cumulative", "instantaneous")


class BatchMonitor(BaseEstimator):
    """Per-sample (online) control limits for a fitted :class:`BatchPCA` or :class:`BatchPLS` model.

    Builds a Hotelling's T2 and an SPE control limit at every time sample by
    passing each reference batch through the model's missing-data score
    estimate at that sample and summarising the reference-batch spread. A new
    batch can then be tracked in real time: at each sample its projected T2
    and SPE are compared against the limit for that sample, flagging abnormal
    behaviour while the batch is still running.

    The T2 at sample ``k`` is ``t_k' S_k^-1 t_k`` with ``S_k`` the scatter of
    the reference batches' score estimates at that sample about zero, the
    centre of the training scores (so the reference batches' mean T2 is
    ``A (N - 1) / N`` at every sample), and its limit is the F-distribution
    limit for the number of reference batches and components. The reference
    batches are normally the batches the model was fitted on, as in Nomikos
    and MacGregor; a different reference set is centred on the model's
    training batches, not on itself. The SPE is either the length of the residual over every cell
    observed so far (``"cumulative"``) or over the newest sample only
    (``"instantaneous"``, the per-interval SPE of Nomikos and MacGregor,
    which reacts in the sample a fault begins); its limit at each sample is
    the moment-matched chi-squared limit on the reference batches' values.

    Parameters
    ----------
    model : BatchPCA or BatchPLS
        A fitted batch model, ideally built from good (common-cause) batches
        only.
    conf_level : float, default=0.99
        Confidence level for the control limits.
    method : {"tsr", "scp", "pmp"}, default="tsr"
        The missing-data score estimator used for every projection, passed to
        the model's ``predict_online_trace``. The limits and the monitored
        traces always use the same estimator, so the statistic at each sample
        is compared against the reference-batch spread computed the same way.
    spe_statistic : {"cumulative", "instantaneous"}, default="cumulative"
        Which SPE to chart and to build limits for (see above).
    ridge : float, default=0.0
        Regularisation for the ``"tsr"`` / ``"pmp"`` estimators, passed to
        ``predict_online_trace``.

    Attributes (after fitting)
    --------------------------
    spe_limit_over_time_ : np.ndarray of shape (n_timesteps,)
        The SPE limit at each sample.
    t2_limit_over_time_ : np.ndarray of shape (n_timesteps,)
        The T2 limit at each sample (the same value at every sample, since
        the T2 is standardised by the per-sample score covariance).
    spe_mean_over_time_, t2_mean_over_time_ : np.ndarray of shape (n_timesteps,)
        The mean reference-batch statistic at each sample.
    score_covariance_over_time_ : np.ndarray of shape (n_timesteps, n_components, n_components)
        The scatter matrix (about zero, divided by ``N - 1``) of the reference
        batches' score estimates at each sample.
    n_reference_batches_ : int
        Number of reference batches the limits were built from.
    n_timesteps_ : int
        Number of time samples in an aligned batch.
    """

    _parameter_constraints: typing.ClassVar = {
        "model": [BaseEstimator],
        "conf_level": [float],
        "method": [str],
        "spe_statistic": [str],
        "ridge": [float, int],
    }

    def __init__(
        self,
        model: BatchPCA | BatchPLS,
        *,
        conf_level: float = 0.99,
        method: str = "tsr",
        spe_statistic: str = "cumulative",
        ridge: float = 0.0,
    ) -> None:
        self.model = model
        self.conf_level = conf_level
        self.method = method
        self.spe_statistic = spe_statistic
        self.ridge = ridge

    def _trace_for_batch(self, batch: pd.DataFrame, initial_conditions: pd.Series | pd.DataFrame | None) -> Bunch:
        """Return the model's online trace (scores, T2, SPE, ...) over every time sample for one batch."""
        return self.model.predict_online_trace(
            batch, initial_conditions=initial_conditions, method=self.method, ridge=self.ridge
        )

    def _spe_from_trace(self, trace: Bunch) -> np.ndarray:
        """Return the SPE trace selected by ``spe_statistic``."""
        return np.asarray(trace.spe_instantaneous if self.spe_statistic == "instantaneous" else trace.spe, dtype=float)

    def _t2_from_scores(self, scores: np.ndarray) -> np.ndarray:
        """T2 at each sample from the score estimates, using that sample's reference covariance."""
        return np.einsum("ka,kab,kb->k", scores, self._score_precision_over_time, scores)

    def fit(
        self,
        good_batches: dict[typing.Hashable, pd.DataFrame],
        y: object = None,  # noqa: ARG002
        *,
        initial_conditions: pd.DataFrame | None = None,
    ) -> BatchMonitor:
        """Learn the per-sample control limits from reference batches.

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
        if self.spe_statistic not in SPE_STATISTICS:
            raise ValueError(f"spe_statistic must be one of {SPE_STATISTICS}; got {self.spe_statistic!r}.")
        check_is_fitted(self.model, "loadings_")
        n_timesteps = int(self.model.n_timesteps_)
        n_components = int(self.model.loadings_.shape[1])  # the fitted width, also when n_components=None
        batch_ids = list(good_batches.keys())
        n_reference = len(batch_ids)
        if n_reference <= n_components:
            raise ValueError(
                f"At least n_components + 1 = {n_components + 1} reference batches are needed to build "
                f"limits; got {n_reference}."
            )

        scores = np.empty((n_reference, n_timesteps, n_components))
        spe_matrix = np.empty((n_reference, n_timesteps))
        for row, batch_id in enumerate(batch_ids):
            z = None if initial_conditions is None else initial_conditions.loc[[batch_id]]
            trace = self._trace_for_batch(good_batches[batch_id], z)
            scores[row] = trace.scores.to_numpy(dtype=float)
            spe_matrix[row] = self._spe_from_trace(trace)

        # The score estimates early in a batch are shrunk and noisy compared
        # with those near its end, so T2 is standardised sample by sample. The
        # scatter is taken about zero, the centre of the training scores, so
        # the quadratic form and its normalisation agree.
        covariance = np.empty((n_timesteps, n_components, n_components))
        precision = np.empty_like(covariance)
        for k in range(n_timesteps):
            at_k = scores[:, k, :]
            covariance[k] = at_k.T @ at_k / (n_reference - 1)
            if np.linalg.matrix_rank(covariance[k]) < n_components:
                raise ValueError(
                    f"The reference batches' score estimates after {k + 1} sample(s) span fewer than "
                    f"{n_components} dimensions, so no T2 limit can be formed there: fewer cells than components "
                    "are observed. Use fewer components, add initial conditions, or pass a ridge."
                )
            precision[k] = safe_inverse(covariance[k], what=f"reference score scatter at sample {k + 1}")
        self._score_precision_over_time = precision
        t2_matrix = np.stack([self._t2_from_scores(scores[row]) for row in range(n_reference)])

        spe_limits = np.array(
            [spe_calculation(spe_matrix[:, k], conf_level=self.conf_level) for k in range(n_timesteps)]
        )
        t2_limit = hotellings_t2_limit(conf_level=self.conf_level, n_components=n_components, n_rows=n_reference)

        self.spe_limit_over_time_ = spe_limits
        self.t2_limit_over_time_ = np.full(n_timesteps, t2_limit)
        self.spe_mean_over_time_ = spe_matrix.mean(axis=0)
        self.t2_mean_over_time_ = t2_matrix.mean(axis=0)
        self.score_covariance_over_time_ = covariance
        self.n_reference_batches_ = n_reference
        self.n_timesteps_ = n_timesteps
        return self

    def monitor(
        self,
        batch: pd.DataFrame,
        upto_k: int | None = None,
        *,
        initial_conditions: pd.Series | pd.DataFrame | None = None,
    ) -> Bunch:
        """Track a batch in real time against the per-sample limits.

        This replays a complete, aligned batch and reports the statistics up
        to ``upto_k``, which is how limits are checked on historical batches.
        A batch that is genuinely still running, with only its first samples
        in hand, is scored with the model's ``predict_online`` and compared
        with ``spe_limit_over_time_`` and ``t2_limit_over_time_`` at that
        sample.

        Parameters
        ----------
        batch : pd.DataFrame
            A single complete, aligned batch to monitor (the training tags as
            columns).
        upto_k : int, optional
            Report only up to this time sample (simulating a still-running
            batch). Defaults to the full batch length.
        initial_conditions : pd.Series or pd.DataFrame, optional
            The Z block for this batch; required if the model was fitted with
            one.

        Returns
        -------
        result : sklearn.utils.Bunch
            With keys ``time`` (1-based number of samples observed),
            ``scores`` (DataFrame, the score estimates at each sample),
            ``hotellings_t2`` and ``spe`` (the batch's statistic traces),
            ``t2_limit`` and ``spe_limit`` (the limits over the same samples),
            and ``t2_alarm`` / ``spe_alarm`` (boolean arrays where the
            statistic exceeds its limit).
        """
        check_is_fitted(self, "spe_limit_over_time_")
        end = self.n_timesteps_ if upto_k is None else operator.index(upto_k)
        if not 1 <= end <= self.n_timesteps_:
            raise ValueError(f"upto_k must lie in [1, {self.n_timesteps_}]; got {upto_k}.")

        trace = self._trace_for_batch(batch, initial_conditions)
        scores = trace.scores.to_numpy(dtype=float)
        t2 = self._t2_from_scores(scores)[:end]
        spe = self._spe_from_trace(trace)[:end]
        t2_limit = self.t2_limit_over_time_[:end]
        spe_limit = self.spe_limit_over_time_[:end]
        return Bunch(
            time=np.arange(1, end + 1),
            scores=trace.scores.iloc[:end],
            hotellings_t2=t2,
            spe=spe,
            t2_limit=t2_limit,
            spe_limit=spe_limit,
            t2_alarm=t2 > t2_limit,
            spe_alarm=spe > spe_limit,
        )
