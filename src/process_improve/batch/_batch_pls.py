# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.
"""Batchwise-unfolded (multiway) PLS relating batch trajectories to final quality.

Mirrors :class:`process_improve.batch.BatchPCA`, but regresses the unfolded
``[Z | X]`` matrix onto a final-quality block ``Y`` (one row per batch) with the
existing :class:`process_improve.multivariate.PLS`. This is the batch
regression / prediction model: it captures how the initial conditions and the
time-varying trajectories drive the final product quality, and predicts the
quality of a completed batch from its data.

See Wold, Kettaneh-Wold, MacGregor and Dunn, "Batch Process Modeling and
MSPC", Comprehensive Chemometrics, Elsevier, 2009.
"""

from __future__ import annotations

import typing

import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import Bunch
from sklearn.utils.validation import check_is_fitted

from ..multivariate._pls import PLS
from .data_input import check_valid_batch_dict, dict_to_wide

if typing.TYPE_CHECKING:
    from collections.abc import Hashable


class BatchPLS(RegressorMixin, BaseEstimator):
    """Batchwise-unfolded PLS from batch trajectories (and Z) to final quality Y.

    Unfolds an aligned batch-data dictionary batchwise (one row per batch),
    optionally joins an initial-conditions (Z) block onto that row, and fits a
    :class:`process_improve.multivariate.PLS` model against a batch-indexed
    quality block ``Y``. The fitted model relates the initial conditions and
    the time-varying trajectory deviations to the final quality, and predicts
    the quality of a completed batch.

    The batches must be aligned before fitting (every batch the same number of
    samples; see :func:`process_improve.batch.resample_to_reference` and
    :func:`process_improve.batch.batch_dtw`) with no missing values.

    Parameters
    ----------
    n_components : int
        Number of PLS components.
    scale : bool, default=True
        Mean-centre and unit-variance scale each column before fitting (the
        centring removes the average trajectory). Passed to the underlying PLS.
    group_by_batch : bool, default=False
        Ordering of the unfolded column index, passed to
        :func:`process_improve.batch.dict_to_wide`.

    Attributes (after fitting)
    --------------------------
    x_weights_ : pd.DataFrame of shape (n_unfolded_features, n_components)
        X-block weights (w*), indexed by the 2-level unfolded column index so
        the trajectory part reshapes to a (tag, time) grid.
    loadings_ : pd.DataFrame
        Alias of ``x_weights_``, so
        :func:`process_improve.batch.time_varying_loading_plot` can plot the
        time-varying weights.
    beta_coefficients_ : pd.DataFrame
        Regression coefficients from the unfolded X to Y.
    r2_cumulative_ : pd.Series
        Cumulative R2 after each component.
    rmse_ : pd.Series or pd.DataFrame
        Root-mean-square error of the fit.
    scores_, spe_, hotellings_t2_ : pd.DataFrame
        Batch-level scores and diagnostics.
    n_batches_, n_tags_, n_timesteps_, n_initial_conditions_ : int
        Problem dimensions.
    batch_ids_, tag_names_, initial_condition_names_, time_index_ : list
        Labels for the batches, tags, initial conditions, and time samples.

    Examples
    --------
    >>> from process_improve.batch import BatchPLS, load_dryer, resample_to_reference
    >>> import pandas as pd
    >>> batches = load_dryer()
    >>> tags = [c for c in next(iter(batches.values())).columns if c != "ClockTime"]
    >>> aligned = resample_to_reference(batches, columns_to_align=tags, reference_batch=1)
    >>> quality = pd.DataFrame({"final": [float(b["DryerTemp"].iloc[-1]) for b in aligned.values()]},
    ...                        index=list(aligned.keys()))
    >>> model = BatchPLS(n_components=2).fit(aligned, quality)  # doctest: +SKIP
    >>> model.predict(aligned).y_hat.shape  # doctest: +SKIP
    (71, 1)

    See Also
    --------
    process_improve.batch.BatchPCA : the unsupervised (monitoring) counterpart.
    process_improve.multivariate.PLS : the underlying estimator.
    """

    _parameter_constraints: typing.ClassVar = {
        "n_components": [int, None],
        "scale": [bool],
        "group_by_batch": [bool],
    }

    def __init__(self, n_components: int, *, scale: bool = True, group_by_batch: bool = False) -> None:
        self.n_components = n_components
        self.scale = scale
        self.group_by_batch = group_by_batch

    def _unfold(
        self,
        batches: dict[Hashable, pd.DataFrame],
        initial_conditions: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """Unfold the batches batchwise and join the initial-conditions block.

        Trajectory columns keep their ``(tag, sequence)`` labels; initial-
        condition columns are labelled ``(name, "")`` (or ``("", name)`` when
        ``group_by_batch`` is set) since they carry no time axis.
        """
        check_valid_batch_dict(batches, no_nan=True)
        wide = dict_to_wide(batches, group_by_batch=self.group_by_batch)
        if initial_conditions is None:
            return wide
        if not isinstance(initial_conditions, pd.DataFrame):
            raise TypeError(
                "initial_conditions must be a pandas DataFrame indexed by batch identifier; "
                f"got {type(initial_conditions).__name__}."
            )
        if set(initial_conditions.index) != set(wide.index):
            raise ValueError("initial_conditions must have exactly one row per batch, indexed by batch id.")
        z_wide = initial_conditions.reindex(wide.index)
        if z_wide.select_dtypes(include="number").shape[1] != z_wide.shape[1]:
            raise ValueError("All initial_conditions columns must be numeric.")
        if z_wide.isna().to_numpy().sum() > 0:
            raise ValueError("No missing values allowed in initial_conditions.")
        tuples = [("", name) if self.group_by_batch else (name, "") for name in z_wide.columns]
        z_wide.columns = pd.MultiIndex.from_tuples(tuples, names=wide.columns.names)
        return pd.concat([z_wide, wide], axis=1)

    def fit(
        self,
        X: dict[Hashable, pd.DataFrame],
        Y: pd.DataFrame,
        *,
        initial_conditions: pd.DataFrame | None = None,
    ) -> BatchPLS:
        """Fit the batchwise-unfolded PLS model against the quality block ``Y``.

        Parameters
        ----------
        X : dict[Hashable, pd.DataFrame]
            Standard batch-data dictionary of aligned batches.
        Y : pd.DataFrame
            Final-quality block: one row per batch (indexed by the same batch
            identifiers as ``X``), one column per quality variable.
        initial_conditions : pd.DataFrame, optional
            The Z block: one row per batch, joined onto the unfolded row.

        Returns
        -------
        self : BatchPLS
        """
        wide = self._unfold(X, initial_conditions)
        if not isinstance(Y, pd.DataFrame):
            raise TypeError(f"Y must be a pandas DataFrame indexed by batch id; got {type(Y).__name__}.")
        if set(Y.index) != set(wide.index):
            raise ValueError("Y must have exactly one row per batch, indexed by the same batch ids as X.")
        y_aligned = Y.reindex(wide.index)

        self._pls = PLS(n_components=self.n_components, scale=self.scale).fit(wide, y_aligned)

        self.feature_columns_ = wide.columns
        # PLS flattens the feature index; re-attach the 2-level unfolded index
        # so the trajectory weights reshape to a (tag, time) grid.
        self.x_weights_ = pd.DataFrame(
            self._pls.x_weights_.to_numpy(), index=wide.columns, columns=self._pls.x_weights_.columns
        )
        self.loadings_ = self.x_weights_
        self.beta_coefficients_ = self._pls.beta_coefficients_
        self.r2_cumulative_ = self._pls.r2_cumulative_
        self.rmse_ = self._pls.rmse_
        self.scores_ = self._pls.scores_
        self.spe_ = self._pls.spe_
        self.hotellings_t2_ = self._pls.hotellings_t2_

        first_batch = X[next(iter(X.keys()))]
        self.batch_ids_ = list(wide.index)
        self.n_batches_ = len(self.batch_ids_)
        self.tag_names_ = list(first_batch.columns)
        self.n_tags_ = len(self.tag_names_)
        self.n_timesteps_ = int(first_batch.shape[0])
        self.time_index_ = list(range(self.n_timesteps_))
        self.target_names_ = list(Y.columns)
        if initial_conditions is None:
            self.initial_condition_names_ = []
            self.n_initial_conditions_ = 0
        else:
            self.initial_condition_names_ = list(initial_conditions.columns)
            self.n_initial_conditions_ = len(self.initial_condition_names_)
        self.n_samples_ = self._pls.n_samples_
        return self

    def _scaled_wide(
        self, batches: dict[Hashable, pd.DataFrame], initial_conditions: pd.DataFrame | None
    ) -> pd.DataFrame:
        """Unfold new batches and check they match the training column layout."""
        check_is_fitted(self, "x_weights_")
        wide = self._unfold(batches, initial_conditions)
        if list(wide.columns) != list(self.feature_columns_):
            raise ValueError(
                "The new batches do not unfold to the training column layout. Align new batches to the "
                "training length and pass the same tags and initial-condition columns."
            )
        return wide

    def predict(
        self,
        X: dict[Hashable, pd.DataFrame],
        *,
        initial_conditions: pd.DataFrame | None = None,
    ) -> Bunch:
        """Predict the final quality of completed batches.

        Parameters
        ----------
        X : dict[Hashable, pd.DataFrame]
            Standard batch-data dictionary of aligned batches with the same
            tags and length as the training data.
        initial_conditions : pd.DataFrame, optional
            The Z block for the new batches; required if the model was fitted
            with one.

        Returns
        -------
        result : sklearn.utils.Bunch
            With keys ``y_hat`` (predicted quality, one row per batch),
            ``scores``, ``hotellings_t2`` and ``spe`` (batch diagnostics).
        """
        wide = self._scaled_wide(X, initial_conditions)
        diagnostics = self._pls.diagnose(wide)
        order = list(X.keys())
        return Bunch(
            y_hat=self._pls.predict(wide).reindex(order),
            scores=diagnostics.scores.reindex(order),
            hotellings_t2=diagnostics.hotellings_t2.reindex(order),
            spe=diagnostics.spe.reindex(order),
        )

    def transform(
        self,
        X: dict[Hashable, pd.DataFrame],
        *,
        initial_conditions: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Return the batch-level PLS scores for ``X`` (in the input batch order)."""
        return self._pls.transform(self._scaled_wide(X, initial_conditions)).reindex(list(X.keys()))
