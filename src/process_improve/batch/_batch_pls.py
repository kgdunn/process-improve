# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.
"""Batchwise-unfolded (multiway) PLS relating batch trajectories to final quality.

Mirrors :class:`process_improve.batch.BatchPCA`, but regresses the unfolded
``[Z | X]`` matrix onto a final-quality block ``Y`` (one row per batch) with the
existing :class:`process_improve.multivariate.PLS`. This is the batch
regression / prediction model: it captures how the initial conditions and the
time-varying trajectories drive the final product quality, and predicts the
quality of a completed batch from its data.

The scaling of the unfolded ``[Z | X]`` block is owned by this class (a
:class:`process_improve.multivariate.MCUVScaler` whose ``center_`` and
``scale_`` are public fitted attributes), and the quality block has its own
scaler (``y_center_``, ``y_scale_``). Downstream users that need to move
between engineering units and the model's scaled space, such as the
mid-course-correction optimiser in :mod:`process_improve.batch.control`, read
these public attributes instead of reaching into the inner PLS estimator.

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
from ..multivariate._preprocessing import MCUVScaler
from .data_input import check_valid_batch_dict, dict_to_wide

if typing.TYPE_CHECKING:
    from collections.abc import Hashable


class BatchPLS(RegressorMixin, BaseEstimator):
    """Batchwise-unfolded PLS from batch trajectories (and Z) to final quality Y.

    Unfolds an aligned batch-data dictionary batchwise (one row per batch),
    optionally joins an initial-conditions (Z) block onto that row, centres and
    scales every column with its own :class:`~process_improve.multivariate.MCUVScaler`,
    and fits a :class:`process_improve.multivariate.PLS` model against a
    batch-indexed quality block ``Y``. The fitted model relates the initial
    conditions and the time-varying trajectory deviations to the final
    quality, and predicts the quality of a completed batch.

    The batches must be aligned before fitting (every batch the same number of
    samples; see :func:`process_improve.batch.resample_to_reference` and
    :func:`process_improve.batch.batch_dtw`) with no missing values.

    Parameters
    ----------
    n_components : int
        Number of PLS components.
    scale : bool, default=True
        Scale each unfolded ``[Z | X]`` column to unit variance after
        centring. Centring always happens (it removes the average
        trajectory); set this to False to keep the columns in their centred,
        unscaled units. The quality block ``Y`` is always centred and scaled
        to unit variance internally, and every reported prediction is mapped
        back to the original quality units.
    group_by_batch : bool, default=False
        Ordering of the unfolded column index, passed to
        :func:`process_improve.batch.dict_to_wide`.

    Attributes (after fitting)
    --------------------------
    x_weights_ : pd.DataFrame of shape (n_unfolded_features, n_components)
        X-block weights (w), indexed by the 2-level unfolded column index so
        the trajectory part reshapes to a (tag, time) grid.
    loadings_ : pd.DataFrame
        Alias of ``x_weights_``, so
        :func:`process_improve.batch.time_varying_loading_plot` can plot the
        time-varying weights.
    x_loadings_ : pd.DataFrame of shape (n_unfolded_features, n_components)
        X-block loadings (p), on the same 2-level index.
    direct_weights_ : pd.DataFrame of shape (n_unfolded_features, n_components)
        Direct weights ``R = W (P'W)^{-1}``, so scores are ``T = X_scaled R``.
    y_loadings_ : pd.DataFrame of shape (n_targets, n_components)
        Y-block loadings (c), mapping scores to the scaled quality space.
    explained_variance_ : np.ndarray of shape (n_components,)
        Variance of each training score.
    beta_coefficients_ : pd.DataFrame
        Regression coefficients from the unfolded X to Y, in the original
        (engineering) units of both blocks.
    r2_cumulative_ : pd.Series
        Cumulative R2 of the quality block after each component.
    rmse_ : pd.DataFrame of shape (n_targets, n_components)
        Root-mean-square error of the fit, on the original quality units.
    scores_, spe_, hotellings_t2_ : pd.DataFrame
        Batch-level scores and diagnostics; one row per batch.
    center_, scale_ : pd.Series of length n_unfolded_features
        The per-column centring and scaling of the ``[Z | X]`` block.
    y_center_, y_scale_ : pd.Series of length n_targets
        The centring and scaling of the quality block.
    n_batches_, n_tags_, n_timesteps_, n_initial_conditions_ : int
        Problem dimensions.
    batch_ids_, tag_names_, initial_condition_names_, target_names_, time_index_ : list
        Labels for the batches, tags, initial conditions, targets, and time
        samples.

    Examples
    --------
    >>> from process_improve.batch import BatchPLS, load_dryer, resample_to_reference
    >>> import pandas as pd
    >>> batches = load_dryer()
    >>> tags = [c for c in next(iter(batches.values())).columns if c != "ClockTime"]
    >>> trimmed = {k: v[tags] for k, v in batches.items()}
    >>> aligned = resample_to_reference(trimmed, columns_to_align=tags, reference_batch=1)
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
            missing = set(wide.index) - set(initial_conditions.index)
            extra = set(initial_conditions.index) - set(wide.index)
            raise ValueError(
                "initial_conditions must have exactly one row per batch. "
                f"Missing batch ids: {sorted(missing, key=str)}; unmatched extra ids: {sorted(extra, key=str)}."
            )
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
            missing = set(wide.index) - set(Y.index)
            extra = set(Y.index) - set(wide.index)
            raise ValueError(
                "Y must have exactly one row per batch, indexed by the same batch ids as X. "
                f"Missing batch ids: {sorted(missing, key=str)}; unmatched extra ids: {sorted(extra, key=str)}."
            )
        y_aligned = Y.reindex(wide.index)
        if y_aligned.isna().to_numpy().sum() > 0:
            raise ValueError("No missing values allowed in Y.")

        # This class owns the scaling of both blocks (the inner PLS is fitted
        # with scale=False), so the centring/scaling constants are public
        # fitted attributes rather than internals of the PLS estimator.
        x_scaler = MCUVScaler().fit(wide)
        if not self.scale:
            x_scaler.scale_ = pd.Series(1.0, index=x_scaler.scale_.index)
        y_scaler = MCUVScaler().fit(y_aligned)
        x_mcuv = pd.DataFrame(x_scaler.transform(wide).to_numpy(), index=wide.index, columns=wide.columns)
        y_mcuv = pd.DataFrame(
            y_scaler.transform(y_aligned).to_numpy(), index=y_aligned.index, columns=y_aligned.columns
        )

        self._x_scaler_own = x_scaler
        self._y_scaler_own = y_scaler
        self._pls = PLS(n_components=self.n_components, scale=False).fit(x_mcuv, y_mcuv)
        # Kept for per-decision-point reference distributions: the mid-course
        # corrector re-projects the training batches under each decision
        # point's missingness pattern to build time-varying SPE and T2 limits
        # (Garcia-Munoz, Kourti and MacGregor, 2004).
        self._x_scaled_training = x_mcuv
        # The scaled Y block is kept for the same reason: the corrector needs
        # the training batches' quality to measure the prediction error of the
        # model at each decision point (a time-varying prediction interval).
        self._y_scaled_training = y_mcuv
        self._expose_fitted_attributes(wide, x_scaler, y_scaler)

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

    def _expose_fitted_attributes(self, wide: pd.DataFrame, x_scaler: MCUVScaler, y_scaler: MCUVScaler) -> None:
        """Re-index the inner model's fitted attributes onto the unfolded layout."""
        self.feature_columns_ = wide.columns
        # PLS flattens the feature index; re-attach the 2-level unfolded index
        # so the trajectory weights reshape to a (tag, time) grid.
        self.x_weights_ = pd.DataFrame(
            self._pls.x_weights_.to_numpy(), index=wide.columns, columns=self._pls.x_weights_.columns
        )
        self.loadings_ = self.x_weights_
        self.x_loadings_ = pd.DataFrame(
            self._pls.x_loadings_.to_numpy(), index=wide.columns, columns=self._pls.x_loadings_.columns
        )
        self.direct_weights_ = pd.DataFrame(
            self._pls.direct_weights_.to_numpy(), index=wide.columns, columns=self._pls.direct_weights_.columns
        )
        # The inner model ran in the scaled spaces; report beta and RMSE on
        # the original (engineering) units: beta_orig = beta_scaled * sy / sx,
        # rmse_orig = rmse_scaled * sy.
        y_scale_row = y_scaler.scale_.to_numpy()[None, :]
        x_scale_col = x_scaler.scale_.to_numpy()[:, None]
        self.beta_coefficients_ = pd.DataFrame(
            self._pls.beta_coefficients_.to_numpy() * (y_scale_row / x_scale_col),
            index=wide.columns,
            columns=self._pls.beta_coefficients_.columns,
        )
        self.rmse_ = self._pls.rmse_.mul(y_scaler.scale_.to_numpy(), axis=0)
        self.y_loadings_ = self._pls.y_loadings_
        self.r2_cumulative_ = self._pls.r2_cumulative_
        self.explained_variance_ = self._pls.explained_variance_
        self.scores_ = self._pls.scores_
        self.spe_ = self._pls.spe_
        self.hotellings_t2_ = self._pls.hotellings_t2_
        self.scaling_factor_for_scores_ = self._pls.scaling_factor_for_scores_
        self.center_ = x_scaler.center_
        self.scale_ = x_scaler.scale_
        self.y_center_ = y_scaler.center_
        self.y_scale_ = y_scaler.scale_

    def _scaled_wide(
        self, batches: dict[Hashable, pd.DataFrame], initial_conditions: pd.DataFrame | None
    ) -> pd.DataFrame:
        """Unfold new batches, check the layout, and apply the training scaling."""
        check_is_fitted(self, "x_weights_")
        wide = self._unfold(batches, initial_conditions)
        if list(wide.columns) != list(self.feature_columns_):
            raise ValueError(
                "The new batches do not unfold to the training column layout. "
                f"Expected {len(self.feature_columns_)} unfolded columns "
                f"({self.n_tags_} tags x {self.n_timesteps_} samples"
                + (f" + {self.n_initial_conditions_} initial conditions" if self.n_initial_conditions_ else "")
                + f"); got {len(wide.columns)}. Align new batches to the training "
                "length and pass the same tags and initial-condition columns."
            )
        return pd.DataFrame(self._x_scaler_own.transform(wide).to_numpy(), index=wide.index, columns=wide.columns)

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
            With keys ``y_hat`` (predicted quality in the original units, one
            row per batch), ``scores``, ``hotellings_t2`` and ``spe`` (batch
            diagnostics).
        """
        wide = self._scaled_wide(X, initial_conditions)
        diagnostics = self._pls.diagnose(wide)
        y_hat = self._y_scaler_own.inverse_transform(self._pls.predict(wide))
        order = list(X.keys())
        return Bunch(
            y_hat=y_hat.reindex(order),
            scores=diagnostics.scores.reindex(order),
            hotellings_t2=diagnostics.hotellings_t2.reindex(order),
            spe=diagnostics.spe.reindex(order),
        )

    def prediction_interval(
        self,
        X: dict[Hashable, pd.DataFrame],
        *,
        conf_level: float = 0.95,
        initial_conditions: pd.DataFrame | None = None,
    ) -> Bunch:
        """Prediction interval for the final quality of completed batches.

        Forwards to :meth:`process_improve.multivariate.PLS.prediction_interval`
        on the scaled row and maps the bounds back to the original quality
        units.

        Parameters
        ----------
        X : dict[Hashable, pd.DataFrame]
            Standard batch-data dictionary of aligned batches.
        conf_level : float, default=0.95
            Confidence level for the interval, in (0.5, 1.0).
        initial_conditions : pd.DataFrame, optional
            The Z block for the new batches; required if the model was fitted
            with one.

        Returns
        -------
        result : sklearn.utils.Bunch
            With keys ``y_hat``, ``lower`` and ``upper`` (DataFrames on the
            original quality units) and ``conf_level``.
        """
        wide = self._scaled_wide(X, initial_conditions)
        scaled = self._pls.prediction_interval(wide, conf_level=conf_level)
        order = list(X.keys())
        return Bunch(
            y_hat=self._y_scaler_own.inverse_transform(scaled.y_hat).reindex(order),
            lower=self._y_scaler_own.inverse_transform(scaled.lower).reindex(order),
            upper=self._y_scaler_own.inverse_transform(scaled.upper).reindex(order),
            conf_level=scaled.conf_level,
        )

    def transform(
        self,
        X: dict[Hashable, pd.DataFrame],
        *,
        initial_conditions: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Return the batch-level PLS scores for ``X`` (in the input batch order)."""
        return self._pls.transform(self._scaled_wide(X, initial_conditions)).reindex(list(X.keys()))

    def projection_matrix(self, observed: object, *, method: str = "tsr", ridge: float = 0.0) -> Bunch:
        """Build the fixed operator mapping observed unfolded columns to score estimates.

        Forwards to :meth:`process_improve.multivariate.PLS.projection_matrix`
        on the inner model. The operator acts on the *scaled* space of the
        unfolded ``[Z | X]`` row; use the public ``center_`` and ``scale_``
        attributes to move engineering-unit values into that space. This is
        the primitive the mid-course corrector precomputes once per decision
        point: for a fixed pattern of observed columns, the score estimate is
        an affine function of any subset of those columns.

        Parameters
        ----------
        observed : array-like
            Boolean mask over ``feature_columns_`` (True = observed) or a
            list of unfolded column labels, e.g. ``[("temperature", 4), ...]``.
        method : {"tsr", "scp", "pmp"}, default="tsr"
        ridge : float, default=0.0

        Returns
        -------
        result : sklearn.utils.Bunch
            With keys ``matrix`` (DataFrame, n_components x n_observed),
            ``condition_number`` (float) and ``method``.
        """
        check_is_fitted(self, "x_weights_")
        return self._pls.projection_matrix(observed, method=method, ridge=ridge)
