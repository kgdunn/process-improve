# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.
"""Batchwise-unfolded (multiway) PCA for batch trajectory data.

Implements the batch modelling approach of Nomikos and MacGregor: align the
batches, unfold them batchwise (one row per batch), mean-centre and scale each
column (which removes the average trajectory), and fit an ordinary PCA on the
result. The fitted model summarizes the deviations of every batch from the
average trajectory, so batches can be compared, and new batches diagnosed,
in a low-dimensional score space with Hotelling's T2 and SPE limits.

Initial conditions (the Z block: one row of pre-batch measurements per batch)
can be appended to the unfolded row, so the model sees ``[Z | X-unfolded]``
with exactly one row per batch.

See Wold, Kettaneh-Wold, MacGregor and Dunn, "Batch Process Modeling and
MSPC", Comprehensive Chemometrics, Elsevier, 2009, for the methodology.
"""

from __future__ import annotations

import functools
import typing

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ..multivariate._diagnostics import spe_contributions as _spe_contributions
from ..multivariate._diagnostics import t2_contributions as _t2_contributions
from ..multivariate._limits import score_limit as _score_limit
from ..multivariate._limits import spe_limit as _spe_limit
from ..multivariate._pca import PCA
from ..multivariate._preprocessing import MCUVScaler
from ..multivariate.plots import explained_variance_plot as _explained_variance_plot
from ..multivariate.plots import loading_plot as _loading_plot
from ..multivariate.plots import score_plot as _score_plot
from ..multivariate.plots import spe_plot as _spe_plot
from ..multivariate.plots import t2_plot as _t2_plot
from .data_input import check_valid_batch_dict, dict_to_wide

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Hashable

    import numpy as np
    from sklearn.utils import Bunch


def _pca_method(fn: Callable[..., typing.Any]) -> Callable[..., typing.Any]:
    """Wrap a module-level ``fn(model, ...)`` as a method forwarding ``self._pca``.

    The convenience plots, limits, and contribution functions in
    :mod:`process_improve.multivariate` read only fitted attributes that the
    internal PCA model carries (``scores_``, ``loadings_``, ``spe_``,
    ``scaling_factor_for_scores_``, ``n_components``, ``n_samples_``), so the
    :class:`BatchPCA` methods forward to them with the wrapped estimator as
    the ``model`` argument. Mirrors ``_model_method`` (ENG-05), so ``help``
    and ``inspect.signature`` report the underlying function.
    """

    @functools.wraps(fn)
    def method(self: BatchPCA, *args, **kwargs) -> object:
        check_is_fitted(self, "loadings_")
        return fn(self._pca, *args, **kwargs)

    return method


class BatchPCA(TransformerMixin, BaseEstimator):
    """Batchwise-unfolded (multiway) PCA on aligned batch trajectory data.

    Each batch becomes one row of the model matrix: the trajectories are
    unfolded batchwise via :func:`process_improve.batch.dict_to_wide`, the
    optional initial-conditions block is joined on, every column is
    mean-centred and (optionally) scaled to unit variance with
    :class:`process_improve.multivariate.MCUVScaler`, and an ordinary
    :class:`process_improve.multivariate.PCA` is fitted to the result.
    Centring the unfolded columns removes the average trajectory, so the
    components model the batch-to-batch deviations.

    The batches must be aligned before fitting: every batch must have the
    same number of samples (see
    :func:`process_improve.batch.resample_to_reference` and
    :func:`process_improve.batch.batch_dtw`), and no missing values are
    allowed.

    Parameters
    ----------
    n_components : int
        Number of principal components to extract.
    scale : bool, default=True
        Scale each unfolded column to unit variance after centring. Centring
        always happens (it removes the average trajectory); set this to False
        to keep the columns in their centred, unscaled units.
    group_by_batch : bool, default=False
        Ordering of the unfolded column index, passed to
        :func:`process_improve.batch.dict_to_wide`: ``False`` groups all time
        samples of a tag together (``(tag, sequence)``); ``True`` groups all
        tags of a time sample together (``(sequence, tag)``).
    algorithm : str, default="auto"
        Fitting algorithm, passed to
        :class:`process_improve.multivariate.PCA`.

    Attributes (after fitting)
    --------------------------
    scores_ : pd.DataFrame of shape (n_batches, n_components)
        Batch-level scores; one row per batch, indexed by batch identifier.
    loadings_ : pd.DataFrame of shape (n_unfolded_features, n_components)
        Loadings, indexed by the 2-level unfolded column index, so the
        trajectory part reshapes to a (tag, time) grid. Initial-condition
        rows (if any) carry an empty string in the ``sequence`` level.
    spe_ : pd.DataFrame of shape (n_batches, n_components)
        Per-batch SPE after each component (residual scale).
    hotellings_t2_ : pd.DataFrame of shape (n_batches, n_components)
        Per-batch cumulative Hotelling's T2.
    explained_variance_ : np.ndarray of shape (n_components,)
        Variance explained by each component.
    r2_per_component_, r2_cumulative_ : pd.Series of length n_components
        Fractional and cumulative R2 of the unfolded matrix.
    n_batches_ : int
        Number of batches in the training set.
    n_tags_ : int
        Number of trajectory tags per batch.
    n_timesteps_ : int
        Number of (aligned) time samples per batch.
    n_initial_conditions_ : int
        Number of initial-condition (Z) columns; zero when none were given.
    batch_ids_ : list
        Batch identifiers, in model-row order.
    tag_names_ : list
        Trajectory tag names.
    initial_condition_names_ : list
        Initial-condition column names (empty when none were given).
    time_index_ : list
        The aligned sequence values (0, 1, ..., ``n_timesteps_`` - 1).
    center_, scale_ : pd.Series of length n_unfolded_features
        The per-column centring and scaling applied before the PCA fit.

    Examples
    --------
    >>> from process_improve.batch import BatchPCA, load_nylon, resample_to_reference
    >>> batches = load_nylon()
    >>> tags = list(next(iter(batches.values())).columns)
    >>> aligned = resample_to_reference(batches, columns_to_align=tags, reference_batch="1")
    >>> model = BatchPCA(n_components=3).fit(aligned)
    >>> model.scores_.shape
    (57, 3)

    See Also
    --------
    process_improve.multivariate.PCA : the underlying estimator.

    References
    ----------
    Nomikos, P. and MacGregor, J.F., "Monitoring of Batch Processes Using
    Multi-Way Principal Component Analysis", AIChE Journal, 40, 1361-1375,
    1994.

    Wold, S., Kettaneh-Wold, N., MacGregor, J.F. and Dunn, K.G., "Batch
    Process Modeling and MSPC", Comprehensive Chemometrics, Elsevier, 2009.
    """

    _parameter_constraints: typing.ClassVar = {
        "n_components": [int, None],
        "scale": [bool],
        "group_by_batch": [bool],
        "algorithm": [str],
    }

    def __init__(
        self,
        n_components: int,
        *,
        scale: bool = True,
        group_by_batch: bool = False,
        algorithm: str = "auto",
    ) -> None:
        self.n_components = n_components
        self.scale = scale
        self.group_by_batch = group_by_batch
        self.algorithm = algorithm

    def _unfold(
        self,
        batches: dict[Hashable, pd.DataFrame],
        initial_conditions: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """Unfold the batches batchwise and join the initial-conditions block.

        Returns the one-row-per-batch ``[Z | X-unfolded]`` matrix with a
        2-level column index throughout: trajectory columns keep their
        ``(tag, sequence)`` labels; initial-condition columns are labelled
        ``(name, "")`` since they carry no time axis.
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
        if self.group_by_batch:
            tuples = [("", name) for name in z_wide.columns]
        else:
            tuples = [(name, "") for name in z_wide.columns]
        z_wide.columns = pd.MultiIndex.from_tuples(tuples, names=wide.columns.names)
        return pd.concat([z_wide, wide], axis=1)

    def fit(
        self,
        X: dict[Hashable, pd.DataFrame],
        y: object = None,  # noqa: ARG002
        *,
        initial_conditions: pd.DataFrame | None = None,
    ) -> BatchPCA:
        """Fit the batchwise-unfolded PCA model.

        Parameters
        ----------
        X : dict[Hashable, pd.DataFrame]
            Standard batch-data dictionary of aligned batches: keys are batch
            identifiers, values are per-batch dataframes with identical
            all-numeric columns and the same number of rows. No missing
            values.
        y : ignored
            Present for sklearn Pipeline compatibility.
        initial_conditions : pd.DataFrame, optional
            The Z block: one row per batch (indexed by the same batch
            identifiers as ``X``), one column per pre-batch measurement.
            Joined onto the unfolded row before centring and scaling.

        Returns
        -------
        self : BatchPCA
        """
        wide = self._unfold(X, initial_conditions)

        scaler = MCUVScaler().fit(wide)
        if not self.scale:
            scaler.scale_ = pd.Series(1.0, index=scaler.scale_.index)
        mcuv = pd.DataFrame(scaler.transform(wide).to_numpy(), index=wide.index, columns=wide.columns)

        self._scaler = scaler
        self._pca = PCA(n_components=self.n_components, algorithm=self.algorithm).fit(mcuv)

        # Batch-shaped views over the fitted internal model. The internal PCA
        # was fitted on the wide frame directly, so its row index is already
        # the batch identifiers and its feature index is the 2-level unfolded
        # column index.
        self.feature_columns_ = wide.columns
        self.scores_ = self._pca.scores_
        self.loadings_ = self._pca.loadings_
        self.spe_ = self._pca.spe_
        self.hotellings_t2_ = self._pca.hotellings_t2_
        self.explained_variance_ = self._pca.explained_variance_
        self.r2_per_component_ = self._pca.r2_per_component_
        self.r2_cumulative_ = self._pca.r2_cumulative_
        self.scaling_factor_for_scores_ = self._pca.scaling_factor_for_scores_

        first_batch = X[next(iter(X.keys()))]
        self.batch_ids_ = list(wide.index)
        self.n_batches_ = len(self.batch_ids_)
        self.tag_names_ = list(first_batch.columns)
        self.n_tags_ = len(self.tag_names_)
        self.n_timesteps_ = int(first_batch.shape[0])
        self.time_index_ = list(range(self.n_timesteps_))
        if initial_conditions is None:
            self.initial_condition_names_ = []
            self.n_initial_conditions_ = 0
        else:
            self.initial_condition_names_ = list(initial_conditions.columns)
            self.n_initial_conditions_ = len(self.initial_condition_names_)
        self.center_ = scaler.center_
        self.scale_ = scaler.scale_
        self.n_samples_ = self._pca.n_samples_
        return self

    def _scaled_wide(
        self,
        batches: dict[Hashable, pd.DataFrame],
        initial_conditions: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """Unfold new batches and apply the training centring/scaling."""
        check_is_fitted(self, "loadings_")
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
        return pd.DataFrame(self._scaler.transform(wide).to_numpy(), index=wide.index, columns=wide.columns)

    def transform(
        self,
        X: dict[Hashable, pd.DataFrame],
        *,
        initial_conditions: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Project new (complete, aligned) batches onto the model.

        Parameters
        ----------
        X : dict[Hashable, pd.DataFrame]
            Standard batch-data dictionary of aligned batches with the same
            tags and number of samples as the training data.
        initial_conditions : pd.DataFrame, optional
            The Z block for the new batches; required if (and only if) the
            model was fitted with one.

        Returns
        -------
        pd.DataFrame of shape (n_new_batches, n_components)
            Batch-level scores, indexed by batch identifier.
        """
        return self._pca.transform(self._scaled_wide(X, initial_conditions))

    def fit_transform(
        self,
        X: dict[Hashable, pd.DataFrame],
        y: object = None,
        *,
        initial_conditions: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Fit the model and return the training batch scores."""
        self.fit(X, y, initial_conditions=initial_conditions)
        return self.scores_

    def diagnose(
        self,
        X: dict[Hashable, pd.DataFrame],
        *,
        initial_conditions: pd.DataFrame | None = None,
    ) -> Bunch:
        """Project new batches and compute their monitoring diagnostics.

        Parameters
        ----------
        X : dict[Hashable, pd.DataFrame]
            Standard batch-data dictionary of aligned batches with the same
            tags and number of samples as the training data.
        initial_conditions : pd.DataFrame, optional
            The Z block for the new batches; required if (and only if) the
            model was fitted with one.

        Returns
        -------
        result : sklearn.utils.Bunch
            With keys ``scores`` (DataFrame, one row per batch),
            ``hotellings_t2`` (DataFrame, cumulative per component), and
            ``spe`` (Series). Compare against :meth:`hotellings_t2_limit`
            and :meth:`spe_limit` to flag abnormal batches.
        """
        return self._pca.diagnose(self._scaled_wide(X, initial_conditions))

    def hotellings_t2_limit(self, conf_level: float = 0.95) -> float:
        """Hotelling's T2 limit at the given confidence level."""
        check_is_fitted(self, "loadings_")
        return self._pca.hotellings_t2_limit(conf_level=conf_level)

    def ellipse_coordinates(
        self,
        score_horiz: int,
        score_vert: int,
        conf_level: float = 0.95,
        n_points: int = 100,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Coordinates of the T2 confidence ellipse for a score plot."""
        check_is_fitted(self, "loadings_")
        return self._pca.ellipse_coordinates(
            score_horiz=score_horiz,
            score_vert=score_vert,
            conf_level=conf_level,
            n_points=n_points,
        )

    # Convenience methods forwarding to the standalone multivariate functions
    # with the internal (batchwise-unfolded) PCA as the model argument.
    score_plot = _pca_method(_score_plot)
    spe_plot = _pca_method(_spe_plot)
    t2_plot = _pca_method(_t2_plot)
    loading_plot = _pca_method(_loading_plot)
    explained_variance_plot = _pca_method(_explained_variance_plot)
    spe_limit = _pca_method(_spe_limit)
    score_limit = _pca_method(_score_limit)
    t2_contributions = _pca_method(_t2_contributions)
    spe_contributions = _pca_method(_spe_contributions)
