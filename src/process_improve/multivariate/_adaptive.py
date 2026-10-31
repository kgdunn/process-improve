# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.
"""Adaptive (recursive) PCA and PLS for on-line process monitoring.

This module revives the adaptive multivariate monitoring method the author
developed for an industrial fault-detection and soft-sensor system. A static
PCA / PLS model, built once from a stretch of common-cause data, slowly loses
relevance as a process drifts: sensors recalibrate, catalysts decay, equipment
fouls, feedstock and seasons change. The estimators here keep such a model
current by updating it recursively, one observation at a time, without ever
re-storing the historical data.

The state carried between observations is a small set of *association matrices*
(the kernels ``X'X`` and, for PLS, ``X'Y``) plus exponentially-weighted moving
average (EWMA) centering and scaling vectors. Each new in-control observation
nudges these quantities; the loadings / weights are then recomputed from the
kernels by a standard eigendecomposition (PCA) or the Dayal-MacGregor kernel
algorithm (PLS). Three design choices from the source method are reproduced:

1. **A forgetting factor** ``forgetting_factor`` (mu) sets how strongly a new
   observation is mixed into the kernel, trading detection speed against
   stability.
2. **An injection term** ``gamma`` re-adds a scaled portion of the *original*
   kernel to every update, in proportion to the new observation's information
   content. This gives perpetual excitation, guards the kernel against
   ill-conditioning under prolonged quiet operation, and pulls the adapting
   model gently back toward its starting subspace. ``gamma = 0`` recovers the
   textbook recursive update.
3. **A distance metric** ``distance_`` (after Krzanowski's between-group
   comparison of principal components) reports, in units of components, how much
   the current subspace still overlaps the original one. It ranges from the
   component count ``A`` (identical) down to ``0`` (orthogonal), is insensitive
   to sign flips, and its rate of change over time helps tune ``forgetting_factor``.

Adaptation is driven by two mechanisms, and the model exposes diagnostics for
each so they can be separated. The *preprocessing* channel is the moving
centering / scaling: ``center_shift_`` (how far the operating point has migrated
from the training data, in training standard-deviation units) and ``scale_shift_`` (the mean
fractional change of the scaling vector). The *kernel* channel is the moving
subspace / regression: ``distance_``, and for PLS ``beta_shift_``; the raw
per-step kernel movement is ``injection_ratio_`` (the gamma term's weight) and
``kernel_update_norm_``. For PLS, ``prediction_channels_`` and
``decompose_prediction`` split each prediction's departure from the frozen training
model into these two channels in the response's own units, and
``adaptation_plot`` renders the whole picture.

The estimators initialise themselves from the batch :class:`~process_improve.multivariate._pca.PCA`
/ :class:`~process_improve.multivariate._pls.PLS`, so the ``i = 0`` model, its
limits, and its sign conventions match the rest of the package exactly. They
then expose an :meth:`~_AdaptiveModel.update` (single observation) and
:meth:`~_AdaptiveModel.partial_fit` (a block) streaming interface, accumulating
the per-observation scores, Hotelling's T2, SPE and distance into history frames
so the existing plotting and limit helpers work unchanged.

References
----------
The kernel PLS recomputation follows Dayal and MacGregor, "Improved PLS
algorithms", *Journal of Chemometrics*, 11, 73-85, 1997. The subspace distance
metric follows Krzanowski, "Between-groups comparison of principal components",
*Journal of the American Statistical Association*, 74, 703-707, 1979. The
chi-squared moment-matched SPE limit follows Nomikos and MacGregor, "Multivariate
SPC charts for monitoring batch processes", *Technometrics*, 37, 41-59, 1995,
and is reused from :mod:`process_improve.multivariate._limits`.
"""

from __future__ import annotations

import typing
from collections import deque

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils import Bunch
from sklearn.utils.validation import check_is_fitted

from ._base import _LatentVariableModel
from ._common import DataMatrix, epsqrt
from ._diagnostics import _t2_score_sd
from ._limits import hotellings_t2_limit as _hotellings_t2_limit
from ._limits import spe_calculation
from ._nipals import _kernel_pls, _sign_align
from ._pca import _opposes_convention
from ._pls import PLS
from ._preprocessing import MCUVScaler

# -----------------------------------------------------------------------------
# Kernel-space model recomputation and subspace geometry helpers.
# -----------------------------------------------------------------------------


def _kernel_pca(kernel: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the top ``n_components`` loadings and eigenvalues of a kernel matrix.

    The symmetric association matrix ``kernel = X'X`` (built from mean-centered,
    scaled data) has eigendecomposition ``X'X = P L P'``; its eigenvectors are
    the PCA loadings and its eigenvalues are the score sums-of-squares, so
    ``explained_variance = L / (N - 1)`` for the ``N`` rows that formed it.

    Parameters
    ----------
    kernel : np.ndarray of shape (K, K)
        Symmetric positive-semidefinite association matrix ``X'X``.
    n_components : int
        Number of components ``A`` to keep.

    Returns
    -------
    loadings : np.ndarray of shape (K, n_components)
        The eigenvectors for the ``n_components`` largest eigenvalues, columns
        sign-fixed so the largest-magnitude entry of each is positive (the same
        convention as the SVD path in :meth:`PCA._fit_svd`).
    eigenvalues : np.ndarray of shape (n_components,)
        The corresponding eigenvalues, in descending order, clipped at zero.
    """
    # eigh returns ascending eigenvalues for a symmetric matrix; reverse to
    # descending and keep the leading A.
    eigenvalues, eigenvectors = np.linalg.eigh((kernel + kernel.T) / 2.0)
    order = np.argsort(eigenvalues)[::-1][:n_components]
    loadings = eigenvectors[:, order]
    values = np.clip(eigenvalues[order], a_min=0.0, a_max=None)
    return np.where(_opposes_convention(loadings), -loadings, loadings), values


def _subspace_distance(reference: np.ndarray, current: np.ndarray) -> float:
    """Krzanowski's subspace-overlap distance between two loading/weight matrices.

    Returns ``trace(M0' M M' M0)`` with ``M0 = reference`` and ``M = current``,
    the sum of squared cosines of the angles between the two subspaces. For
    orthonormal columns it ranges from ``A`` (the matrices span the same space)
    down to ``0`` (orthogonal). It is invariant to a sign flip of any column.

    Parameters
    ----------
    reference : np.ndarray of shape (K, A)
        The original loadings / weights ``M0``.
    current : np.ndarray of shape (K, A)
        The current loadings / weights ``M``.

    Returns
    -------
    float
        The subspace overlap, in units of components.
    """
    cross = reference.T @ current
    return float(np.trace(cross @ cross.T))


def _scores_and_spe(
    x: np.ndarray, direct: np.ndarray, guide: np.ndarray, loadings: np.ndarray
) -> tuple[np.ndarray, float]:
    """Return the scores and the SPE of one scaled row, ``NaN`` where a value is missing.

    A complete row's scores are ``x @ direct`` and its residual ``x - loadings @ scores``.
    A row with missing values is projected one component at a time (single-component
    projection; Nelson, Taylor and MacGregor, 1996): each score is the observed part of
    the row on the observed part of its ``guide`` column, over that part's squared length,
    and the row is deflated by the observed part of its loading. PCA guides by its
    loadings, PLS by its weights. The SPE is the length of what is left of the observed
    entries.

    Parameters
    ----------
    x : np.ndarray of shape (n_features,)
        The row, centred and scaled.
    direct, guide, loadings : np.ndarray of shape (n_features, n_components)
        The direct weights ``R`` (scores of a complete row), the columns each score is
        read along in a row with missing values, and the loadings ``P``.

    Returns
    -------
    scores : np.ndarray of shape (n_components,)
    spe : float
    """
    missing = np.isnan(x)
    if not missing.any():
        scores = x @ direct
        residual = x - loadings @ scores
        return scores, float(np.sqrt(residual @ residual))
    deflate = np.where(missing, 0.0, x)
    scores = np.zeros(guide.shape[1])
    for a in range(guide.shape[1]):
        along = np.where(missing, 0.0, guide[:, a])
        denom = float(along @ along)
        # A column with (almost) nothing observed gives no score.
        observed = denom > epsqrt  # pragma: no mutate - >= differs only at a denom of exactly epsqrt
        scores[a] = float(deflate @ along) / denom if observed else 0.0
        deflate = deflate - scores[a] * np.where(missing, 0.0, loadings[:, a])
    return scores, float(np.sqrt(deflate @ deflate))


def _as_rows(X: DataMatrix) -> tuple[pd.Index, np.ndarray]:
    """Return the row labels of ``X`` (0, 1, ... for an array) and its values as floats.

    ``to_numpy(dtype=float)`` reads a missing value of a nullable column as NaN and a
    flag as 0 or 1, in a frame of mixed column types as in any other.
    """
    frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
    return frame.index, frame.to_numpy(dtype=float)


def _ewma_step(
    centre: np.ndarray, scale: np.ndarray, value: np.ndarray, factor_centre: np.ndarray, factor_scale: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Move an EWMA centring and scaling vector one observation toward ``value``.

    Equations 1a and 2a of the method: the scale tracks the squared deviation from
    the centre before this step, then the centre moves toward the value. A variable
    with a zero factor is frozen, and one whose value is missing moves neither its
    centre nor its scale.

    Parameters
    ----------
    centre, scale : np.ndarray of shape (n_variables,)
        The current centring and scaling vectors.
    value : np.ndarray of shape (n_variables,)
        The new raw observation, ``NaN`` where missing.
    factor_centre, factor_scale : np.ndarray of shape (n_variables,)
        The EWMA factors of the centre (lambda) and of the scale (alpha).

    Returns
    -------
    centre, scale : np.ndarray of shape (n_variables,)
        The vectors after this step. A scale of exactly 0 (a scale factor of 1 and a
        value at the centre) is set to 1, as :class:`MCUVScaler` scales a column of
        equal values; a scale in small units is kept, in any units.
    """
    deviation = value - centre
    observed = ~np.isnan(deviation)
    spread = np.sqrt((1.0 - factor_scale) * scale**2 + factor_scale * deviation**2)
    spread = np.where(spread > 0.0, spread, 1.0)
    # (1 - lambda) m + lambda x, written as m + lambda (x - m): the same value, but exact
    # when x equals m, so an observation at the centre leaves it unchanged to the last bit.
    moved = centre + factor_centre * deviation
    return np.where(observed, moved, centre), np.where(observed, spread, scale)


# -----------------------------------------------------------------------------
# Shared base class.
# -----------------------------------------------------------------------------


class _AdaptiveModel(_LatentVariableModel):
    """Scaffolding shared by :class:`AdaptivePCA` and :class:`AdaptivePLS`.

    Holds the EWMA preprocessing update, the streaming history buffers, the
    rolling SPE-limit machinery, and the ``partial_fit`` loop. Subclasses provide
    the kernel state (``AdaptivePCA`` carries only ``X'X``; ``AdaptivePLS`` adds
    ``X'Y``) and the ``_recompute`` / ``_project`` steps specific to each method.
    """

    _RENAME_CONTEXT: typing.ClassVar[str] = "adaptive multivariate"

    if typing.TYPE_CHECKING:  # fitted state set in each subclass's fit()
        mx_: np.ndarray
        sx_: np.ndarray
        lambda_center_: np.ndarray
        alpha_scale_: np.ndarray
        n_components: int
        n_samples_: int
        adaptive_spe_limit: bool
        conf_level: float
        forgetting_factor: float
        gamma: float
        _component_names: list[int]
        _spe_buffer: deque[float]
        _min_spe_window: int
        _spe_limit_0: float
        _mx0: np.ndarray
        _sx0: np.ndarray
        _last_injection: float
        _last_kernel_norm: float
        XtX0_: np.ndarray
        XtX_: np.ndarray
        _norm_xx0: float

    # ---- helpers used by both subclasses -----------------------------------

    @staticmethod
    def _as_vector(value: float | np.ndarray, size: int, name: str) -> np.ndarray:
        """Broadcast a scalar or validate a per-variable vector of forgetting factors."""
        arr = np.broadcast_to(np.asarray(value, dtype=float), (size,)).copy()
        if not np.all((arr >= 0.0) & (arr <= 1.0)):  # a NaN is refused too
            raise ValueError(f"{name} must lie in [0, 1]; got values outside that range.")
        return arr

    def _init_history(self) -> None:
        """Allocate the empty per-observation history containers."""
        self._hist_scores: list[np.ndarray] = []
        self._hist_t2: list[float] = []
        self._hist_spe: list[float] = []
        self._hist_distance: list[float] = []
        self._hist_index: list[typing.Any] = []
        # Adaptation diagnostics (state-drift and kernel-update magnitudes).
        self._hist_center_shift: list[float] = []
        self._hist_scale_shift: list[float] = []
        self._hist_injection: list[float] = []
        self._hist_kernel_norm: list[float] = []

    def _start_limits(self, training_spe: np.ndarray) -> None:
        """Set the T2 limit, the training SPE limit, and the rolling window of squared SPEs it starts from."""
        A, N = self.n_components_, self.n_samples_
        self._t2_limit_0 = _hotellings_t2_limit(conf_level=self.conf_level, n_components=A, n_rows=N)
        self._spe_limit_0 = spe_calculation(training_spe, conf_level=self.conf_level)
        self._min_spe_window = max(20, A + 2)
        self._spe_buffer: deque[float] = deque((training_spe**2).tolist(), maxlen=int(self.spe_limit_window))

    def _monitor(self, x0: np.ndarray, t2: float, spe: float) -> tuple[float, bool, bool]:
        """Return the SPE limit, whether the row is in control, and whether it updates the model.

        A row with no observed value is out of control and updates nothing. An in-control
        row's squared SPE joins the rolling window, after its own limit was taken.
        """
        spe_limit = self._current_spe_limit()
        observed = not np.isnan(x0).all()
        below_t2 = t2 <= self._t2_limit_0  # pragma: no mutate - < differs only at a T2 exactly on its limit
        below_spe = spe <= spe_limit  # pragma: no mutate - < differs only at an SPE exactly on its limit
        in_control = below_t2 and below_spe and observed
        if in_control:
            self._spe_buffer.append(spe**2)
        return spe_limit, in_control, (in_control or self.update_when_out_of_control) and observed

    def _ewma_update_x(self, x0: np.ndarray) -> np.ndarray:
        """Advance the X-space EWMA centre and scale by one observation, and return it scaled by them.

        The monitoring statistics used the vectors before this step; the kernel takes the
        observation scaled with the moved ones, a missing value at the centre, 0.
        """
        self.mx_, self.sx_ = _ewma_step(self.mx_, self.sx_, x0, self.lambda_center_, self.alpha_scale_)
        return np.nan_to_num((x0 - self.mx_) / self.sx_)

    def _update_kernel_xx(self, x_scaled: np.ndarray) -> None:
        """Mix one scaled observation into ``X'X``, with the injection term, at the training trace.

        Records the injection weight and the relative size of the change to ``X'X`` for
        the drift diagnostics.
        """
        mu = self.forgetting_factor
        prev_xx = self.XtX_
        update_xx = mu * np.outer(x_scaled, x_scaled)
        # Nuclear norm (= trace for the PSD X'X) throughout, for consistency with the
        # re-scale below; the rank-1 update's trace equals its Frobenius norm.
        f_x = self.gamma * float(np.trace(update_xx)) / self._norm_xx0 if self._norm_xx0 > 0 else 0.0
        new_xx = (1.0 - mu) * prev_xx + update_xx + f_x * self.XtX0_
        norm_xx = float(np.trace(new_xx))
        if norm_xx > epsqrt:  # pragma: no mutate - >= differs only at a trace of exactly epsqrt
            new_xx = new_xx * (self._norm_xx0 / norm_xx)
        self._last_injection = f_x
        denom = float(np.linalg.norm(prev_xx))
        self._last_kernel_norm = float(np.linalg.norm(new_xx - prev_xx)) / denom if denom > 0 else 0.0
        self.XtX_ = new_xx

    def _record(self, scores: np.ndarray, t2: float, spe: float, updated: bool, label: object) -> float:
        """Append one observation's statistics and the model's drift to the history; return the distance.

        ``center_shift`` is how far the X-centering vector has migrated from the
        training data, in training standard-deviation units; ``scale_shift`` is the
        mean fractional change of the scaling vector. ``injection`` is the gamma
        term's weight ``f_x`` and ``kernel_norm`` the relative size of this step's
        change to ``X'X``; both are zero on observations that did not update.
        """
        distance = self._distance()
        self._hist_scores.append(scores)
        self._hist_t2.append(t2)
        self._hist_spe.append(spe)
        self._hist_distance.append(distance)
        self._hist_center_shift.append(float(np.linalg.norm((self.mx_ - self._mx0) / self._sx0)))
        self._hist_scale_shift.append(float(np.mean(np.abs(self.sx_ / self._sx0 - 1.0))))
        self._hist_injection.append(self._last_injection if updated else 0.0)
        self._hist_kernel_norm.append(self._last_kernel_norm if updated else 0.0)
        self._hist_index.append(label if label is not None else len(self._hist_index))
        return distance

    def _update_scaling_factor(self) -> None:
        """Recompute the per-score standard deviations used for Hotelling's T2."""
        self.scaling_factor_for_scores_ = pd.Series(
            np.sqrt(self.explained_variance_), index=self._component_names, name="Standard deviation per score"
        )

    def _hotellings_t2(self, scores: np.ndarray) -> float:
        """Return one row's Hotelling's T2, leaving out a component with no spread as PCA and PLS do."""
        return float(np.sum((scores / _t2_score_sd(self.scaling_factor_for_scores_.to_numpy())) ** 2))

    def _current_spe_limit(self) -> float:
        """Return the SPE limit: rolling-window adaptive, or the fixed training limit."""
        if self.adaptive_spe_limit and len(self._spe_buffer) >= self._min_spe_window:
            return spe_calculation(np.sqrt(np.asarray(self._spe_buffer)), conf_level=self.conf_level)
        return self._spe_limit_0

    def partial_fit(self, X: DataMatrix, Y: DataMatrix | None = None) -> _AdaptiveModel:
        """Stream a block of observations through :meth:`update`, in row order.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New observations, processed one row at a time.
        Y : array-like of shape (n_samples, n_targets), optional
            Matching responses (PLS only). A row whose response has a missing
            value updates the X-space model but not the regression part, which
            mirrors an infrequently-sampled laboratory measurement.

        Returns
        -------
        self : _AdaptiveModel
        """
        check_is_fitted(self, "mx_")
        labels, rows = _as_rows(X)
        y_rows = None if Y is None else _as_rows(Y)[1]  # by position: Y's labels are not read
        for pos, label in enumerate(labels):
            self.update(rows[pos], y_row=None if y_rows is None else y_rows[pos], label=label)
        return self

    def transform(self, X: DataMatrix) -> pd.DataFrame:
        """Project data onto the *current* model, returning the X scores (no update)."""
        check_is_fitted(self, "mx_")
        labels, rows = _as_rows(X)
        scores = np.reshape([self._project(row)[0] for row in rows], (len(labels), self.n_components))
        return pd.DataFrame(scores, index=labels, columns=self._component_names)

    # ---- history views (built lazily on access, so update() stays O(1)) ------

    @property
    def scores_(self) -> pd.DataFrame:
        """Per-observation scores accumulated by :meth:`update` (rows x components)."""
        data = np.array(self._hist_scores, dtype=float).reshape(-1, len(self._component_names))
        return pd.DataFrame(data, index=self._hist_index, columns=self._component_names)

    @property
    def spe_(self) -> pd.DataFrame:
        """Per-observation SPE history (single column, keyed by the component count)."""
        return pd.DataFrame({self.n_components: self._hist_spe}, index=self._hist_index)

    @property
    def hotellings_t2_(self) -> pd.DataFrame:
        """Per-observation Hotelling's T2 history (single column, keyed by the component count)."""
        return pd.DataFrame({self.n_components: self._hist_t2}, index=self._hist_index)

    @property
    def distance_(self) -> pd.Series:
        """Per-observation subspace overlap with the training model, in units of components."""
        return pd.Series(self._hist_distance, index=self._hist_index, name="Subspace overlap (components)")

    @property
    def center_shift_(self) -> pd.Series:
        """Per-observation X-centering migration from the training data, in training-SD units.

        The norm ``||(m_x,i - m_x,0) / s_x,0||``: how far the operating point the
        model is centered on has moved from where it was built, expressed in the
        training model's standard-deviation units. Part of the *preprocessing* channel
        of adaptation, complementary to :attr:`distance_` (the *kernel* channel).
        """
        return pd.Series(self._hist_center_shift, index=self._hist_index, name="Centre migration (training SD)")

    @property
    def scale_shift_(self) -> pd.Series:
        """Per-observation mean fractional change of the X-scaling vector from the training data.

        ``mean |s_x,i / s_x,0 - 1|`` across the tags: how much the per-variable
        spreads the model scales by have changed since the training data.
        """
        return pd.Series(self._hist_scale_shift, index=self._hist_index, name="Scale change (fraction)")

    @property
    def injection_ratio_(self) -> pd.Series:
        """Per-observation gamma-injection weight ``f_x`` applied to the training kernel.

        The scalar ``f_x = gamma * ||update|| / ||X'X_0||`` re-added to ``X'X`` each
        step; larger when the new observation carries more information relative to
        the training data. Zero on observations that did not update the model.
        """
        return pd.Series(self._hist_injection, index=self._hist_index, name="Injection weight f_x")

    @property
    def kernel_update_norm_(self) -> pd.Series:
        """Per-observation relative size of the change to ``X'X``: ``||dX'X|| / ||X'X||``.

        A direct measure of how much this step moved the X-space kernel. Zero on
        observations that did not update the model.
        """
        return pd.Series(self._hist_kernel_norm, index=self._hist_index, name="Kernel update (relative)")

    def _channels_to_plot(self) -> pd.DataFrame | None:
        """Return the prediction channels, and the regression change, the adaptation plot shows: none here."""
        return None

    def adaptation_plot(self, valid: np.ndarray | None = None) -> object:
        """Plot the adaptation as preprocessing vs kernel channels and state drift.

        Returns a Plotly figure. For a PLS model with a response the top panel
        splits each prediction's departure from the frozen training model into a
        centering/scaling channel and a kernel (subspace/regression) channel, with
        the total on a secondary axis; the lower panels show the preprocessing
        drift (centre migration, scale change) and the kernel drift (subspace
        rotation ``A - distance_``, and for PLS the regression change). For an
        AdaptivePCA model only the two state-drift panels are shown.

        Parameters
        ----------
        valid : array-like of bool, optional
            Mask of observations to show in the channel panel (others are blanked,
            for excluding held predictions on shutdown / transition rows).

        Returns
        -------
        plotly.graph_objects.Figure
        """
        import plotly.graph_objects as go  # noqa: PLC0415
        from plotly.subplots import make_subplots  # noqa: PLC0415

        x = list(self._hist_index)
        rot = self.n_components - np.asarray(self._hist_distance)
        channels = self._channels_to_plot()

        if channels is not None:
            fig = make_subplots(
                rows=3,  # in one column, the default
                shared_xaxes=True,
                vertical_spacing=0.07,
                specs=[[{"secondary_y": True}], [{}], [{}]],
                subplot_titles=["Prediction departure from static, by channel", "Preprocessing drift", "Kernel drift"],
            )
            prep = channels["preprocessing"].to_numpy()
            kern = channels["kernel"].to_numpy()
            if valid is not None:
                prep = np.where(valid, prep, np.nan)
                kern = np.where(valid, kern, np.nan)
            total = prep + kern
            # A history of one row or more; a frozen model's channels are all 0.
            span = float(np.nanmax(np.abs(np.concatenate([prep, kern])))) or 1.0
            fig.add_trace(
                go.Scatter(x=x, y=prep, name="Centering + scaling", line=dict(color="#1f4e79", width=1)), row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=x, y=kern, name="Kernel (subspace/regression)", line=dict(color="#c55a11", width=1)),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(x=x, y=total, name="Total (right axis)", line=dict(color="#222222", width=1)),
                row=1,
                col=1,
                secondary_y=True,
            )
            fig.update_layout(
                yaxis={"title_text": "Channel", "range": [-1.05 * span, 1.05 * span]},
                yaxis2={"title_text": "Total", "range": [-2.1 * span, 2.1 * span]},
            )
            r_pre, r_ker = 2, 3
        else:
            fig = make_subplots(
                rows=2,  # in one column, the default
                shared_xaxes=True,
                vertical_spacing=0.09,
                subplot_titles=["Preprocessing drift", "Kernel drift"],
            )
            r_pre, r_ker = 1, 2

        fig.add_trace(
            go.Scatter(
                x=x,
                y=self._hist_center_shift,
                name="Centre migration [training SD]",
                line=dict(color="#1f4e79", width=1),
            ),
            row=r_pre,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=self._hist_scale_shift,
                name="Scale change [fraction]",
                line=dict(color="#2e75b6", width=1, dash="dash"),
            ),
            row=r_pre,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=x, y=rot, name="Subspace rotation (A - distance)", line=dict(color="#c55a11", width=1)),
            row=r_ker,
            col=1,
        )
        if channels is not None:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=channels["regression"],
                    name="Regression change",
                    line=dict(color="#e0912f", width=1, dash="dash"),
                ),
                row=r_ker,
                col=1,
            )
        fig.update_layout(
            height=270 * (2 if channels is None else 3),
            margin=dict(l=70, r=70, t=40, b=40),
            legend=dict(font=dict(size=9)),
        )
        return fig


# -----------------------------------------------------------------------------
# Adaptive PCA.
# -----------------------------------------------------------------------------


class AdaptivePCA(_AdaptiveModel, TransformerMixin, BaseEstimator):
    """Recursive, adaptive Principal Component Analysis for on-line monitoring.

    Initialise the model on a block of common-cause data with :meth:`fit`, then feed
    new observations through :meth:`update` (one row) or :meth:`partial_fit` (a
    block). Each in-control observation updates the EWMA centering / scaling
    vectors and the ``X'X`` kernel; the loadings are recomputed from the kernel
    and the drift is tracked by :attr:`distance_`.

    Parameters
    ----------
    n_components : int
        Number of principal components ``A``.
    forgetting_factor : float, default=0.02
        The kernel forgetting factor ``mu`` in ``[0, 1]``. ``0`` freezes the
        kernel (no adaptation); larger values adapt faster but are less stable.
    gamma : float, default=0.1
        The injection weight in ``[0, 1]``. Re-adds ``gamma`` times the original
        kernel, scaled by the new observation's relative information content,
        into every update. ``0`` recovers the textbook recursive update.
    lambda_center : float or array-like, default=0.02
        EWMA factor(s) for the centering vector, scalar or one per variable. ``0``
        freezes a variable's centre (for tags that must not drift).
    alpha_scale : float or array-like, default=0.01
        EWMA factor(s) for the scaling vector, scalar or one per variable.
    update_when_out_of_control : bool, default=False
        If ``False`` (the recommended default) an observation flagged out of
        control does not update the model, so a fault is not absorbed as normal.
    adaptive_spe_limit : bool, default=True
        Recompute the SPE limit from a rolling window of recent in-control SPE
        values; otherwise keep the fixed limit from the training model.
    spe_limit_window : int, default=200
        Length of the rolling SPE window when ``adaptive_spe_limit`` is True.
    conf_level : float, default=0.95
        Confidence level for the T2 and SPE limits.

    Attributes
    ----------
    loadings_ : pd.DataFrame of shape (n_features, n_components)
        The current loadings ``P``.
    distance_ : pd.Series
        Per-update subspace overlap with the training model, in units of components
        (``n_components`` = identical, ``0`` = orthogonal).
    center_shift_, scale_shift_ : pd.Series
        Preprocessing-drift diagnostics: centre migration (training-SD units) and mean
        fractional scale change per observation.
    injection_ratio_, kernel_update_norm_ : pd.Series
        Kernel-update magnitudes per observation: the gamma-injection weight and
        the relative size of the change to ``X'X``.
    scores_, hotellings_t2_, spe_ : pd.DataFrame
        Per-observation history accumulated by :meth:`update`.
    """

    _ATTRIBUTE_RENAMES: typing.ClassVar[dict[str, str]] = {}

    def __init__(  # noqa: PLR0913
        self,
        n_components: int,
        *,
        forgetting_factor: float = 0.02,
        gamma: float = 0.1,
        lambda_center: float | np.ndarray = 0.02,
        alpha_scale: float | np.ndarray = 0.01,
        update_when_out_of_control: bool = False,
        adaptive_spe_limit: bool = True,
        spe_limit_window: int = 200,
        conf_level: float = 0.95,
    ):
        self.n_components = n_components
        self.forgetting_factor = forgetting_factor
        self.gamma = gamma
        self.lambda_center = lambda_center
        self.alpha_scale = alpha_scale
        self.update_when_out_of_control = update_when_out_of_control
        self.adaptive_spe_limit = adaptive_spe_limit
        self.spe_limit_window = spe_limit_window
        self.conf_level = conf_level

    def fit(self, X: DataMatrix, y: DataMatrix | None = None) -> AdaptivePCA:  # noqa: ARG002
        """Fit the adaptive model's initial state from a block of common-cause data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data used to establish the ``i = 0`` model, its centering /
            scaling vectors, kernel and limits.
        y : ignored

        Returns
        -------
        self : AdaptivePCA
        """
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
        self._feature_names = X_df.columns
        N, K = X_df.shape
        A = int(self.n_components)
        self.n_components = A
        self.n_components_ = A
        self.n_features_in_ = K
        self.n_samples_ = N
        self._component_names = list(range(1, A + 1))

        if not 0.0 <= self.forgetting_factor <= 1.0:
            raise ValueError(f"forgetting_factor must lie in [0, 1]; got {self.forgetting_factor}.")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError(f"gamma must lie in [0, 1]; got {self.gamma}.")
        self.lambda_center_ = self._as_vector(self.lambda_center, K, "lambda_center")
        self.alpha_scale_ = self._as_vector(self.alpha_scale, K, "alpha_scale")

        # Initialise preprocessing from MCUVScaler, then build the kernel and loadings
        # from the *scaled* data so every downstream projection (which scales the
        # incoming observation) lives in the same space. Initialising the loadings from
        # the same eigendecomposition that later updates use keeps the distance
        # metric at exactly A until the model actually adapts.
        self._scaler = MCUVScaler().fit(X_df)
        self.mx_ = self._scaler.center_.to_numpy().copy()
        self.sx_ = self._scaler.scale_.to_numpy().copy()
        self._mx0 = self.mx_.copy()  # frozen training centre / scale for the drift diagnostics
        self._sx0 = self.sx_.copy()

        # A missing cell is 0: at the centre, it adds nothing to X'X.
        Xs = np.nan_to_num(self._scaler.transform(X_df).to_numpy())
        self.XtX0_ = Xs.T @ Xs
        self.XtX_ = self.XtX0_.copy()
        # Nuclear norm of the kernel. For a symmetric PSD matrix this is the sum
        # of its eigenvalues, i.e. its trace, which for mean-centered unit-variance
        # data equals K * (N - 1): the total sum of squares. Holding this constant
        # in the injection ratio and the re-scale keeps the total variance fixed,
        # so the score scaling used for Hotelling's T2 does not drift as the
        # eigenvalue *shape* changes. The rank-1 update term has identical trace,
        # Frobenius and spectral norms, so only this denominator sees the choice.
        self._norm_xx0 = float(np.trace(self.XtX0_))

        loadings, eigenvalues = _kernel_pca(self.XtX0_, A)
        self.loadings0_ = loadings.copy()
        self._loadings = loadings.copy()
        self.explained_variance_ = eigenvalues / (N - 1)  # MCUVScaler refused fewer than 2 rows
        self._update_scaling_factor()

        # The limits, and the rolling SPE window, start from the training SPE, computed
        # with the training loadings on the scaled data.
        residual = Xs - Xs @ self._loadings @ self._loadings.T
        self._start_limits(np.sqrt(np.sum(residual**2, axis=1)))
        self._init_history()
        return self

    @property
    def loadings_(self) -> pd.DataFrame:
        """The current loadings ``P`` as a DataFrame (features x components)."""
        return pd.DataFrame(self._loadings, index=self._feature_names, columns=self._component_names)

    def _project(self, x0: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Project one raw observation onto the current model: its scores, Hotelling's T2 and SPE.

        Missing entries are handled by single-component projection along the loadings.
        """
        P = self._loadings
        t, spe = _scores_and_spe((x0 - self.mx_) / self.sx_, P, P, P)
        return t, self._hotellings_t2(t), spe

    def _distance(self) -> float:
        """Return the overlap of the current loadings with the training ones, in components."""
        return _subspace_distance(self.loadings0_, self._loadings)

    def _recompute(self, x_scaled: np.ndarray) -> None:
        """Update the kernel with one scaled observation and recompute the loadings."""
        self._update_kernel_xx(x_scaled)
        loadings, eigenvalues = _kernel_pca(self.XtX_, self.n_components)
        signs = _sign_align(loadings, self._loadings)
        loadings *= signs  # pragma: no mutate - the signs are +1 and -1, which / reads the same
        self._loadings = loadings
        self.explained_variance_ = eigenvalues / (self.n_samples_ - 1)
        self._update_scaling_factor()

    def update(
        self,
        x_row: np.ndarray,
        y_row: np.ndarray | None = None,  # noqa: ARG002
        *,
        label: object = None,
    ) -> Bunch:
        """Process one new observation and, if warranted, update the model.

        Parameters
        ----------
        x_row : array-like of shape (n_features,)
            The new raw observation.
        y_row : ignored
            Present for a uniform signature with :class:`AdaptivePLS`.
        label : hashable, optional
            Index label recorded in the history frames for this observation.

        Returns
        -------
        sklearn.utils.Bunch
            With fields ``scores``, ``hotellings_t2``, ``spe``, ``spe_limit``,
            ``hotellings_t2_limit``, ``in_control``, ``updated`` and ``distance``.
        """
        check_is_fitted(self, "mx_")
        x0 = np.asarray(x_row, dtype=float).ravel()
        t, t2, spe = self._project(x0)
        spe_limit, in_control, updated = self._monitor(x0, t2, spe)
        if updated:
            self._recompute(self._ewma_update_x(x0))
        distance = self._record(t, t2, spe, updated, label)
        return Bunch(
            scores=t,
            hotellings_t2=t2,
            spe=spe,
            spe_limit=spe_limit,
            hotellings_t2_limit=self._t2_limit_0,
            in_control=in_control,
            updated=updated,
            distance=distance,
        )


# -----------------------------------------------------------------------------
# Adaptive PLS.
# -----------------------------------------------------------------------------


class AdaptivePLS(_AdaptiveModel, RegressorMixin, TransformerMixin, BaseEstimator):
    """Recursive, adaptive Projection to Latent Structures for on-line monitoring.

    The PLS analogue of :class:`AdaptivePCA`, carrying both association matrices
    ``X'X`` and ``X'Y``. Initialise on a block of common-cause data with :meth:`fit`,
    then stream new observations through :meth:`update` / :meth:`partial_fit`. The
    weights, loadings and regression coefficients are recomputed from the kernels
    by the Dayal-MacGregor algorithm after each in-control update.

    A response row may be omitted (``y_row=None``): the X-space model
    (centering / scaling and ``X'X``) still adapts, but the ``X'Y`` kernel and the
    regression part are held until a response arrives. This matches a
    soft-sensor whose reference laboratory value is available only occasionally,
    while the process tags stream continuously.

    Parameters
    ----------
    n_components : int
        Number of latent variables ``A``.
    forgetting_factor : float, default=0.02
        The kernel forgetting factor ``mu`` in ``[0, 1]``.
    gamma : float, default=0.1
        The injection weight in ``[0, 1]`` (see :class:`AdaptivePCA`).
    lambda_center, alpha_scale : float or array-like
        EWMA factors for the X-block centering / scaling vectors.
    lambda_center_y, alpha_scale_y : float or array-like
        EWMA factors for the Y-block centering / scaling vectors.
    update_when_out_of_control, adaptive_spe_limit, spe_limit_window, conf_level
        As documented on :class:`AdaptivePCA`.

    Attributes
    ----------
    x_weights_, x_loadings_, direct_weights_ : pd.DataFrame
        The current PLS weights ``W``, loadings ``P`` and direct weights ``R``.
    beta_coefficients_ : pd.DataFrame
        The current regression coefficients mapping raw X to raw Y.
    distance_ : pd.Series
        Per-update subspace overlap (on the weights ``W``) with the training model.
    center_shift_, scale_shift_ : pd.Series
        Preprocessing-drift diagnostics: X-centre migration (training-SD units) and
        mean fractional scale change per observation.
    beta_shift_ : pd.Series
        Relative change of the scaled regression from the training data, per observation.
    injection_ratio_, kernel_update_norm_ : pd.Series
        Kernel-update magnitudes per observation (gamma-injection weight and
        relative change to ``X'X``).
    prediction_channels_ : pd.DataFrame
        Per-observation prediction split into ``static`` / ``preprocessing`` /
        ``kernel`` / ``adaptive`` channels (see :meth:`decompose_prediction`).
    scores_, hotellings_t2_, spe_, predictions_ : pd.DataFrame
        Per-observation history accumulated by :meth:`update`.
    """

    _ATTRIBUTE_RENAMES: typing.ClassVar[dict[str, str]] = {}

    if typing.TYPE_CHECKING:  # fitted state set in fit() / _recompute_from_kernels()
        _weights: np.ndarray
        _loadings: np.ndarray
        _direct: np.ndarray
        _y_loadings: np.ndarray
        _beta_scaled: np.ndarray
        my_: np.ndarray
        sy_: np.ndarray
        lambda_center_y_: np.ndarray
        alpha_scale_y_: np.ndarray
        n_targets_: int
        _feature_names: pd.Index
        _target_names: pd.Index
        _my0: np.ndarray
        _sy0: np.ndarray
        _beta0: np.ndarray
        _norm_beta0: float
        _hist_static: list[np.ndarray]
        _hist_prep: list[np.ndarray]
        _hist_kern: list[np.ndarray]
        _hist_beta_shift: list[float]

    def __init__(  # noqa: PLR0913
        self,
        n_components: int,
        *,
        forgetting_factor: float = 0.02,
        gamma: float = 0.1,
        lambda_center: float | np.ndarray = 0.02,
        alpha_scale: float | np.ndarray = 0.01,
        lambda_center_y: float | np.ndarray = 0.02,
        alpha_scale_y: float | np.ndarray = 0.01,
        update_when_out_of_control: bool = False,
        adaptive_spe_limit: bool = True,
        spe_limit_window: int = 200,
        conf_level: float = 0.95,
    ):
        self.n_components = n_components
        self.forgetting_factor = forgetting_factor
        self.gamma = gamma
        self.lambda_center = lambda_center
        self.alpha_scale = alpha_scale
        self.lambda_center_y = lambda_center_y
        self.alpha_scale_y = alpha_scale_y
        self.update_when_out_of_control = update_when_out_of_control
        self.adaptive_spe_limit = adaptive_spe_limit
        self.spe_limit_window = spe_limit_window
        self.conf_level = conf_level

    def fit(self, X: DataMatrix, Y: DataMatrix) -> AdaptivePLS:  # noqa: PLR0915
        """Fit the adaptive PLS model's initial state from a block of common-cause data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training predictors.
        Y : array-like of shape (n_samples, n_targets)
            Training responses.

        Returns
        -------
        self : AdaptivePLS
        """
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
        # np.asarray of a 1-D Series / array yields a single-column frame, so Y is
        # always 2-D here.
        Y_df = Y if isinstance(Y, pd.DataFrame) else pd.DataFrame(np.asarray(Y))
        N, K = X_df.shape
        M = Y_df.shape[1]
        A = int(self.n_components)
        self.n_components = A
        self.n_components_ = A
        self.n_features_in_ = K
        self.n_targets_ = M
        self.n_samples_ = N
        self._feature_names = X_df.columns
        self._target_names = Y_df.columns
        self._component_names = list(range(1, A + 1))

        if not 0.0 <= self.forgetting_factor <= 1.0:
            raise ValueError(f"forgetting_factor must lie in [0, 1]; got {self.forgetting_factor}.")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError(f"gamma must lie in [0, 1]; got {self.gamma}.")
        self.lambda_center_ = self._as_vector(self.lambda_center, K, "lambda_center")
        self.alpha_scale_ = self._as_vector(self.alpha_scale, K, "alpha_scale")
        self.lambda_center_y_ = self._as_vector(self.lambda_center_y, M, "lambda_center_y")
        self.alpha_scale_y_ = self._as_vector(self.alpha_scale_y, M, "alpha_scale_y")

        # Initialise preprocessing and the batch model, matching PLS(scale=True), its default.
        self._x_scaler = MCUVScaler().fit(X_df)
        self._y_scaler = MCUVScaler().fit(Y_df)
        self.mx_ = self._x_scaler.center_.to_numpy().copy()
        self.sx_ = self._x_scaler.scale_.to_numpy().copy()
        self.my_ = self._y_scaler.center_.to_numpy().copy()
        self.sy_ = self._y_scaler.scale_.to_numpy().copy()
        # It refuses a block with no X'Y covariance, so _norm_xy0 and _beta0 below are not 0.
        training_model = PLS(n_components=A).fit(X_df, Y_df)

        # A missing cell is 0: at the centre, it adds nothing to X'X or X'Y.
        Xs = np.nan_to_num(self._x_scaler.transform(X_df).to_numpy())
        Ys = np.nan_to_num(self._y_scaler.transform(Y_df).to_numpy())
        self.XtX0_ = Xs.T @ Xs
        self.XtY0_ = Xs.T @ Ys
        self.XtX_ = self.XtX0_.copy()
        self.XtY_ = self.XtY0_.copy()
        # Nuclear norm of each kernel, used in the injection ratio and the
        # re-scale. For the symmetric PSD X'X this is the trace (sum of
        # eigenvalues = K * (N - 1) for scaled data); for the K x M cross-kernel
        # X'Y it is the sum of singular values. Both reduce to the ordinary vector
        # norm when M = 1. Holding the X'X nuclear norm constant keeps the total
        # variance, and hence the T2 score scaling, from drifting with the
        # eigenvalue shape; the rank-1 update terms are norm-choice invariant.
        self._norm_xx0 = float(np.trace(self.XtX0_))
        self._norm_xy0 = float(np.linalg.norm(self.XtY0_, "nuc"))

        self._recompute_from_kernels(align_to=None)
        self.weights0_ = self._weights.copy()

        # Frozen training snapshot for the adaptation diagnostics (prediction-channel
        # split and the beta-drift metric).
        self._mx0 = self.mx_.copy()
        self._sx0 = self.sx_.copy()
        self._my0 = self.my_.copy()
        self._sy0 = self.sy_.copy()
        self._beta0 = self._beta_scaled.copy()
        self._norm_beta0 = float(np.linalg.norm(self._beta0))

        self._start_limits(training_model.spe_.iloc[:, A - 1].to_numpy())
        self._hist_pred: list[np.ndarray] = []
        self._hist_static = []
        self._hist_prep = []
        self._hist_kern = []
        self._hist_beta_shift = []
        self._init_history()
        return self

    def _channel_split(self, x0: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split the current model's prediction for one raw row into a static baseline and two channels.

        Returns ``(static, preprocessing, kernel)`` prediction vectors, each of length
        ``M``, whose sum is the adaptive prediction:

        - ``static``: the frozen training model's prediction (training centering/scaling
          and training regression);
        - ``preprocessing``: the extra correction from the *moved* centering /
          scaling vectors, holding the regression at its training value;
        - ``kernel``: the further correction from the *moved* subspace / regression
          (the kernels), on top of the updated preprocessing.

        The split is exact by a one-at-a-time swap and uses the current model
        state, so streamed on each observation it attributes the departure from
        the static model to the two mechanisms.
        """
        xs_cur = np.nan_to_num((x0 - self.mx_) / self.sx_)  # a missing value at the centre
        xs_static = np.nan_to_num((x0 - self._mx0) / self._sx0)
        y_static = self._my0 + self._sy0 * (self._beta0.T @ xs_static)
        y_prep = self.my_ + self.sy_ * (self._beta0.T @ xs_cur)
        y_full = self.my_ + self.sy_ * (self._beta_scaled.T @ xs_cur)
        return y_static, y_prep - y_static, y_full - y_prep

    def _recompute_from_kernels(self, align_to: np.ndarray | None) -> None:
        """Recompute W, P, R, C and beta from the current kernels; sign-align if asked."""
        W, P, R, C, score_ssq = _kernel_pls(self.XtX_, self.XtY_, self.n_components)
        if align_to is not None:
            signs = _sign_align(P, align_to)
            W, P, R, C = (part * signs for part in (W, P, R, C))  # pragma: no mutate - signs are +1, -1: / is *
        self._weights = W
        self._loadings = P
        self._direct = R
        self._y_loadings = C
        self._beta_scaled = R @ C.T  # (K, M) mapping scaled X to scaled Y
        self.explained_variance_ = score_ssq / (self.n_samples_ - 1)  # MCUVScaler refused fewer than 2 rows
        self._update_scaling_factor()

    # ---- current-parameter DataFrame views ---------------------------------

    @property
    def x_weights_(self) -> pd.DataFrame:
        """The current X-space weights ``W``."""
        return pd.DataFrame(self._weights, index=self._feature_names, columns=self._component_names)

    @property
    def x_loadings_(self) -> pd.DataFrame:
        """The current X-space loadings ``P``."""
        return pd.DataFrame(self._loadings, index=self._feature_names, columns=self._component_names)

    @property
    def direct_weights_(self) -> pd.DataFrame:
        """The current direct weights ``R`` such that ``T = X R``."""
        return pd.DataFrame(self._direct, index=self._feature_names, columns=self._component_names)

    @property
    def beta_coefficients_(self) -> pd.DataFrame:
        """The current regression coefficients mapping raw X to raw Y."""
        beta_raw = self._beta_scaled * (self.sy_[np.newaxis, :] / self.sx_[:, np.newaxis])
        return pd.DataFrame(beta_raw, index=self._feature_names, columns=self._target_names)

    def _project(self, x0: np.ndarray) -> tuple[np.ndarray, float, float, np.ndarray]:
        """Project one raw observation: its scores, Hotelling's T2, SPE and predicted response.

        Missing entries are handled by single-component projection along the weights.
        """
        t, spe = _scores_and_spe((x0 - self.mx_) / self.sx_, self._direct, self._weights, self._loadings)
        y_hat = (t @ self._y_loadings.T) * self.sy_ + self.my_
        return t, self._hotellings_t2(t), spe, y_hat

    def _distance(self) -> float:
        """Return the overlap of the current weights with the training ones, in components."""
        return _subspace_distance(self.weights0_, self._weights)

    def _recompute(self, x_scaled: np.ndarray, y_scaled: np.ndarray | None) -> None:
        """Update the kernels with one scaled observation and recompute the model."""
        self._update_kernel_xx(x_scaled)
        if y_scaled is not None:
            mu = self.forgetting_factor
            # The nuclear norm (the sum of singular values) throughout, as for X'X: the rank-1
            # update's is mu ||x|| ||y||, as are its Frobenius and spectral norms.
            f_y = self.gamma * mu * float(np.linalg.norm(x_scaled) * np.linalg.norm(y_scaled)) / self._norm_xy0
            self.XtY_ = (1.0 - mu) * self.XtY_ + mu * np.outer(x_scaled, y_scaled) + f_y * self.XtY0_
            norm_xy = float(np.linalg.norm(self.XtY_, "nuc"))
            if norm_xy > epsqrt:  # pragma: no mutate - >= differs only at a norm of exactly epsqrt
                self.XtY_ *= self._norm_xy0 / norm_xy
        self._recompute_from_kernels(align_to=self._loadings)

    def update(self, x_row: np.ndarray, y_row: np.ndarray | None = None, *, label: object = None) -> Bunch:
        """Process one new observation and, if warranted, update the model.

        Parameters
        ----------
        x_row : array-like of shape (n_features,)
            The new raw observation.
        y_row : array-like of shape (n_targets,), optional
            The matching response, if available. When omitted, the X-space model
            still adapts but the regression part is held.
        label : hashable, optional
            Index label recorded in the history frames.

        Returns
        -------
        sklearn.utils.Bunch
            With fields ``scores``, ``prediction``, ``hotellings_t2``, ``spe``,
            ``spe_limit``, ``hotellings_t2_limit``, ``in_control``, ``updated``
            and ``distance``.
        """
        check_is_fitted(self, "mx_")
        x0 = np.asarray(x_row, dtype=float).ravel()
        t, t2, spe, y_hat = self._project(x0)
        # Attribution of this (deployed) prediction to the two adaptation channels,
        # using the state before the update, as y_hat does.
        static_ch, prep_ch, kernel_ch = self._channel_split(x0)
        spe_limit, in_control, updated = self._monitor(x0, t2, spe)
        if updated:
            x_scaled = self._ewma_update_x(x0)
            y_scaled = None
            if y_row is not None:
                y0 = np.asarray(y_row, dtype=float).ravel()
                if not np.isnan(y0).any():
                    self.my_, self.sy_ = _ewma_step(self.my_, self.sy_, y0, self.lambda_center_y_, self.alpha_scale_y_)
                    y_scaled = (y0 - self.my_) / self.sy_  # with the moved vectors, as for X
            self._recompute(x_scaled, y_scaled)

        distance = self._record(t, t2, spe, updated, label)
        self._hist_pred.append(y_hat)
        self._hist_static.append(static_ch)
        self._hist_prep.append(prep_ch)
        self._hist_kern.append(kernel_ch)
        self._hist_beta_shift.append(float(np.linalg.norm(self._beta_scaled - self._beta0)) / self._norm_beta0)
        return Bunch(
            scores=t,
            prediction=y_hat,
            hotellings_t2=t2,
            spe=spe,
            spe_limit=spe_limit,
            hotellings_t2_limit=self._t2_limit_0,
            in_control=in_control,
            updated=updated,
            distance=distance,
        )

    @property
    def predictions_(self) -> pd.DataFrame:
        """Per-observation response predictions accumulated by :meth:`update`."""
        data = np.array(self._hist_pred, dtype=float).reshape(-1, self.n_targets_)
        return pd.DataFrame(data, index=self._hist_index, columns=self._target_names)

    def _channels_frame(
        self,
        static: typing.Any,  # noqa: ANN401
        prep: typing.Any,  # noqa: ANN401
        kern: typing.Any,  # noqa: ANN401
        index: typing.Any,  # noqa: ANN401
    ) -> pd.DataFrame:
        """Assemble a static / preprocessing / kernel / adaptive channel DataFrame.

        For a single response the columns are ``static``, ``preprocessing``,
        ``kernel`` and ``adaptive`` (their sum). For several responses the columns
        are a two-level ``(target, channel)`` index.
        """
        shape = (len(index), self.n_targets_)
        parts = {"static": np.reshape(static, shape), "preprocessing": np.reshape(prep, shape)}
        parts["kernel"] = np.reshape(kern, shape)
        parts["adaptive"] = parts["static"] + parts["preprocessing"] + parts["kernel"]
        if self.n_targets_ == 1:
            return pd.DataFrame({channel: values[:, 0] for channel, values in parts.items()}, index=index)
        data = {
            (name, channel): values[:, j]
            for j, name in enumerate(self._target_names)
            for channel, values in parts.items()
        }
        frame = pd.DataFrame(data, index=index)
        frame.columns = pd.MultiIndex.from_tuples(list(data), names=["target", "channel"])
        return frame

    def _channels_to_plot(self) -> pd.DataFrame | None:
        """Return the channels of a single response, and the regression change, once there is a history."""
        if self._hist_static and self.n_targets_ == 1:
            return self.prediction_channels_.assign(regression=self._hist_beta_shift)
        return None

    @property
    def beta_shift_(self) -> pd.Series:
        """Per-observation relative change of the scaled regression from the training data.

        ``||beta_i - beta_0|| / ||beta_0||`` on the scaled-space coefficients: the
        *kernel* channel's effect on the regression, complementary to the
        subspace-only :attr:`distance_`.
        """
        return pd.Series(self._hist_beta_shift, index=self._hist_index, name="Regression change (relative)")

    @property
    def prediction_channels_(self) -> pd.DataFrame:
        """Per-observation prediction split into static, preprocessing and kernel channels.

        Streamed by :meth:`update` (using each observation's deployed state), the
        columns attribute the prediction's departure from the frozen training model to
        the moving centering / scaling (``preprocessing``) and the moving subspace
        / regression (``kernel``); ``adaptive = static + preprocessing + kernel``.
        """
        return self._channels_frame(self._hist_static, self._hist_prep, self._hist_kern, self._hist_index)

    def decompose_prediction(self, X: DataMatrix) -> pd.DataFrame:
        """Split the *current* model's prediction for each row of ``X`` into channels.

        A snapshot version of :attr:`prediction_channels_` for arbitrary rows: it
        uses the model's current state, and returns the static (training) prediction
        plus the preprocessing and kernel corrections that carry it to the
        adaptive prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Rows to decompose (no model update is performed).

        Returns
        -------
        pd.DataFrame
            Columns ``static``, ``preprocessing``, ``kernel``, ``adaptive`` for a
            single response, or a ``(target, channel)`` column MultiIndex for
            several responses.
        """
        check_is_fitted(self, "mx_")
        labels, rows = _as_rows(X)
        split = np.reshape([self._channel_split(row) for row in rows], (len(labels), 3, self.n_targets_))
        return self._channels_frame(split[:, 0], split[:, 1], split[:, 2], labels)

    def predict(self, X: DataMatrix) -> pd.DataFrame:
        """Predict the response with the *current* model (no update)."""
        check_is_fitted(self, "mx_")
        labels, rows = _as_rows(X)
        predictions = np.reshape([self._project(row)[3] for row in rows], (len(labels), self.n_targets_))
        return pd.DataFrame(predictions, index=labels, columns=self._target_names)
