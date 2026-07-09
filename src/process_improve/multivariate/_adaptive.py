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

The estimators seed themselves from the batch :class:`~process_improve.multivariate._pca.PCA`
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
from ._limits import hotellings_t2_limit as _hotellings_t2_limit
from ._limits import spe_calculation
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
    for a in range(loadings.shape[1]):
        max_el_idx = int(np.argmax(np.abs(loadings[:, a])))
        if loadings[max_el_idx, a] < 0:
            loadings[:, a] *= -1.0
    return loadings, values


def _kernel_pls(
    kernel_xx: np.ndarray, kernel_xy: np.ndarray, n_components: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Recompute PLS parameters from the association matrices ``X'X`` and ``X'Y``.

    Implements the Dayal-MacGregor (1997) kernel algorithm, which extracts the
    PLS weights, loadings and regression matrix directly from the kernels,
    without access to the original ``X`` / ``Y`` blocks. ``X'X`` is not deflated;
    each direction ``r`` is orthogonalised against the earlier loadings so that
    the score inner product is ``r' (X'X) r``.

    Parameters
    ----------
    kernel_xx : np.ndarray of shape (K, K)
        The X-space association matrix ``X'X`` (scaled data).
    kernel_xy : np.ndarray of shape (K, M)
        The cross association matrix ``X'Y`` (scaled data).
    n_components : int
        Number of latent variables ``A`` to extract.

    Returns
    -------
    weights : np.ndarray of shape (K, A)
        The X-space weights ``W`` (each column unit norm).
    loadings : np.ndarray of shape (K, A)
        The X-space loadings ``P``.
    direct_weights : np.ndarray of shape (K, A)
        The direct weights ``R = W (P'W)^-1`` such that scores ``T = X R``.
    y_loadings : np.ndarray of shape (M, A)
        The Y-space loadings ``C`` (also called ``Q``).
    score_ssq : np.ndarray of shape (A,)
        The score sums-of-squares ``r' (X'X) r`` per component; divide by
        ``N - 1`` to recover the score variances used for Hotelling's T2 scaling.
    """
    K = kernel_xx.shape[0]
    M = kernel_xy.shape[1]
    A = n_components
    weights = np.zeros((K, A))
    loadings = np.zeros((K, A))
    direct = np.zeros((K, A))
    y_loadings = np.zeros((M, A))
    score_ssq = np.zeros(A)

    xy = kernel_xy.copy()
    for a in range(A):
        if M == 1:
            w = xy[:, 0].copy()
        else:
            # Dominant eigenvector of X'Y (X'Y)' via the small M x M problem.
            eigvals, eigvecs = np.linalg.eigh(xy.T @ xy)
            q_dom = eigvecs[:, int(np.argmax(eigvals))]
            w = xy @ q_dom
        norm_w = float(np.linalg.norm(w))
        if norm_w < epsqrt:
            # No further usable covariance; leave the remaining columns at zero.
            break
        w = w / norm_w
        r = w.copy()
        for j in range(a):
            r = r - float(loadings[:, j] @ w) * direct[:, j]
        tt = float(r @ kernel_xx @ r)
        if tt < epsqrt:
            break
        p = (kernel_xx @ r) / tt
        c = (xy.T @ r) / tt  # (M,)
        weights[:, a] = w
        loadings[:, a] = p
        direct[:, a] = r
        y_loadings[:, a] = c
        score_ssq[a] = tt
        # Deflate X'Y only: X'Y <- X'Y - (X'X r) c' = X'Y - tt * p c'.
        xy = xy - tt * np.outer(p, c)
    return weights, loadings, direct, y_loadings, score_ssq


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


def _sign_align(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
    """Flip the sign of each column of ``current`` to best match ``previous``.

    PCA / PLS components are defined only up to a sign; the recomputed loadings
    can flip between updates, which makes score time-series jump. Aligning the
    sign to the previous iteration keeps the score traces continuous. Returns the
    per-column signs (``+1`` / ``-1``) actually applied, so the caller can flip
    the matching scores as well.

    Parameters
    ----------
    current : np.ndarray of shape (K, A)
        The freshly recomputed loadings / weights (modified in place).
    previous : np.ndarray of shape (K, A)
        The loadings / weights from the previous update.

    Returns
    -------
    np.ndarray of shape (A,)
        The signs applied to each column.
    """
    signs = np.ones(current.shape[1])
    for a in range(current.shape[1]):
        if float(current[:, a] @ previous[:, a]) < 0:
            signs[a] = -1.0
    return signs


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

    # ---- helpers used by both subclasses -----------------------------------

    @staticmethod
    def _as_vector(value: float | np.ndarray, size: int, name: str) -> np.ndarray:
        """Broadcast a scalar or validate a per-variable vector of forgetting factors."""
        arr = np.broadcast_to(np.asarray(value, dtype=float), (size,)).astype(float).copy()
        if np.any(arr < 0) or np.any(arr > 1):
            raise ValueError(f"{name} must lie in [0, 1]; got values outside that range.")
        return arr

    def _init_history(self) -> None:
        """Allocate the empty per-observation history containers."""
        self._hist_scores: list[np.ndarray] = []
        self._hist_t2: list[float] = []
        self._hist_spe: list[float] = []
        self._hist_distance: list[float] = []
        self._hist_index: list[typing.Any] = []
        self._n_updates_ = 0

    def _ewma_update_x(self, x0: np.ndarray) -> None:
        """Advance the X-space EWMA centering and scaling vectors by one observation.

        Uses the pre-update centre for both steps (equations 1a / 2a of the
        method): the scaling tracks the squared deviation from the current
        centre, then the centre itself moves toward the new value. Frozen
        variables (a zero forgetting factor) are left untouched.
        """
        # Scaling first, referencing the not-yet-updated centre.
        self.sx_ = np.sqrt(
            (1.0 - self.alpha_scale_) * self.sx_**2 + self.alpha_scale_ * (x0 - self.mx_) ** 2
        )
        # Guard against a scale collapsing to zero on a frozen / constant tag.
        self.sx_ = np.where(self.sx_ < epsqrt, 1.0, self.sx_)
        self.mx_ = (1.0 - self.lambda_center_) * self.mx_ + self.lambda_center_ * x0

    def _current_spe_limit(self) -> float:
        """Return the SPE limit: rolling-window adaptive, or the fixed seed limit."""
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
            Matching responses (PLS only). Rows whose response is entirely
            missing update the X-space model but not the regression part, which
            mirrors an infrequently-sampled laboratory measurement.

        Returns
        -------
        self : _AdaptiveModel
        """
        check_is_fitted(self, "mx_")
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
        Y_df = None
        if Y is not None:
            Y_df = Y if isinstance(Y, pd.DataFrame) else pd.DataFrame(np.asarray(Y))
        for pos in range(X_df.shape[0]):
            idx = X_df.index[pos]
            y_row = None if Y_df is None else Y_df.iloc[pos].to_numpy(dtype=float)
            self.update(X_df.iloc[pos].to_numpy(dtype=float), y_row=y_row, label=idx)
        return self

    # ---- history views ------------------------------------------------------

    def _refresh_history_frames(self) -> None:
        """Rebuild the public ``scores_`` / ``spe_`` / ``hotellings_t2_`` / ``distance_`` frames."""
        cols = self._component_names
        index = self._hist_index
        self.scores_ = pd.DataFrame(np.array(self._hist_scores).reshape(-1, len(cols)), index=index, columns=cols)
        self.spe_ = pd.DataFrame({self.n_components: self._hist_spe}, index=index)
        self.hotellings_t2_ = pd.DataFrame({self.n_components: self._hist_t2}, index=index)
        self.distance_ = pd.Series(self._hist_distance, index=index, name="Subspace overlap (components)")


# -----------------------------------------------------------------------------
# Adaptive PCA.
# -----------------------------------------------------------------------------


class AdaptivePCA(_AdaptiveModel, TransformerMixin, BaseEstimator):
    """Recursive, adaptive Principal Component Analysis for on-line monitoring.

    Seed the model on a block of common-cause data with :meth:`fit`, then feed
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
        values; otherwise keep the fixed limit from the seed model.
    spe_limit_window : int, default=200
        Length of the rolling SPE window when ``adaptive_spe_limit`` is True.
    conf_level : float, default=0.95
        Confidence level for the T2 and SPE limits.

    Attributes
    ----------
    loadings_ : pd.DataFrame of shape (n_features, n_components)
        The current loadings ``P``.
    distance_ : pd.Series
        Per-update subspace overlap with the seed model, in units of components
        (``n_components`` = identical, ``0`` = orthogonal).
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
        """Seed the adaptive model from a block of common-cause data.

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
        self.n_features_in_ = K
        self.n_samples_ = N
        self._component_names = list(range(1, A + 1))

        if not 0.0 <= self.forgetting_factor <= 1.0:
            raise ValueError(f"forgetting_factor must lie in [0, 1]; got {self.forgetting_factor}.")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError(f"gamma must lie in [0, 1]; got {self.gamma}.")
        self.lambda_center_ = self._as_vector(self.lambda_center, K, "lambda_center")
        self.alpha_scale_ = self._as_vector(self.alpha_scale, K, "alpha_scale")

        # Seed preprocessing from MCUVScaler, then build the kernel and loadings
        # from the *scaled* data so every downstream projection (which scales the
        # incoming observation) lives in the same space. Seeding the loadings from
        # the same eigendecomposition that later updates use keeps the distance
        # metric at exactly A until the model actually adapts.
        self._scaler = MCUVScaler().fit(X_df)
        self.mx_ = self._scaler.center_.to_numpy(dtype=float).copy()
        self.sx_ = self._scaler.scale_.to_numpy(dtype=float).copy()

        Xs = np.nan_to_num(self._scaler.transform(X_df).to_numpy(dtype=float), nan=0.0)
        self.XtX0_ = Xs.T @ Xs
        self.XtX_ = self.XtX0_.copy()
        self._norm_xx0 = float(np.linalg.norm(self.XtX0_))

        loadings, eigenvalues = _kernel_pca(self.XtX0_, A)
        self.loadings0_ = loadings.copy()
        self._loadings = loadings.copy()
        self.explained_variance_ = eigenvalues / max(1, N - 1)
        self._update_scaling_factor()

        # Seed the limits, and the rolling SPE window, from the training SPE
        # computed with the seed loadings on the scaled data.
        self._t2_limit_0 = _hotellings_t2_limit(conf_level=self.conf_level, n_components=A, n_rows=N)
        scores = Xs @ self._loadings
        residual = Xs - scores @ self._loadings.T
        seed_spe = np.sqrt(np.sum(residual**2, axis=1))
        self._spe_limit_0 = spe_calculation(seed_spe, conf_level=self.conf_level)
        self._min_spe_window = max(20, A + 2)
        self._spe_buffer: deque[float] = deque(maxlen=int(self.spe_limit_window))
        self._spe_buffer.extend((seed_spe**2).tolist())

        self._init_history()
        self._refresh_history_frames()
        return self

    @property
    def loadings_(self) -> pd.DataFrame:
        """The current loadings ``P`` as a DataFrame (features x components)."""
        return pd.DataFrame(self._loadings, index=self._feature_names, columns=self._component_names)

    def _update_scaling_factor(self) -> None:
        """Recompute the per-score standard deviations used for Hotelling's T2."""
        var = self.explained_variance_
        self.scaling_factor_for_scores_ = pd.Series(
            np.sqrt(np.clip(var, a_min=epsqrt**2, a_max=None)),
            index=self._component_names,
            name="Standard deviation per score",
        )

    def _project(self, x0: np.ndarray) -> tuple[np.ndarray, float, float, np.ndarray]:
        """Project one raw observation onto the current model.

        Returns the scores, Hotelling's T2, SPE (on the residual scale) and the
        scaled observation. Missing entries are handled by single-component
        projection (Nelson, Taylor and MacGregor, 1996): scores use only the
        observed elements of each loading and the residual is taken over the
        observed entries only.
        """
        x = (x0 - self.mx_) / self.sx_
        P = self._loadings
        s = self.scaling_factor_for_scores_.to_numpy(dtype=float)
        missing = np.isnan(x)
        if not missing.any():
            t = x @ P
            residual = x - P @ t
            spe = float(np.sqrt(residual @ residual))
        else:
            x_obs = np.where(missing, 0.0, x)
            deflate = x_obs.copy()
            t = np.zeros(P.shape[1])
            for a in range(P.shape[1]):
                p_a = np.where(missing, 0.0, P[:, a])
                denom = float(p_a @ p_a)
                t[a] = float(deflate @ p_a) / denom if denom > epsqrt else 0.0
                deflate = deflate - t[a] * p_a
            spe = float(np.sqrt(deflate @ deflate))
        t2 = float(np.sum((t / s) ** 2))
        return t, t2, spe, x

    def _recompute(self, x_scaled: np.ndarray) -> None:
        """Update the kernel with one scaled observation and recompute the loadings."""
        mu = self.forgetting_factor
        update_xx = mu * np.outer(x_scaled, x_scaled)
        f_x = self.gamma * float(np.linalg.norm(update_xx)) / self._norm_xx0 if self._norm_xx0 > 0 else 0.0
        self.XtX_ = (1.0 - mu) * self.XtX_ + update_xx + f_x * self.XtX0_
        norm_xx = float(np.linalg.norm(self.XtX_))
        if norm_xx > epsqrt:
            self.XtX_ *= self._norm_xx0 / norm_xx

        loadings, eigenvalues = _kernel_pca(self.XtX_, self.n_components)
        signs = _sign_align(loadings, self._loadings)
        loadings *= signs
        self._loadings = loadings
        self.explained_variance_ = eigenvalues / max(1, self.n_samples_ - 1)
        self._update_scaling_factor()

    def update(
        self, x_row: np.ndarray, y_row: np.ndarray | None = None, *, label: object = None  # noqa: ARG002
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
        t, t2, spe, _x_scaled = self._project(x0)
        spe_limit = self._current_spe_limit()
        in_control = (t2 <= self._t2_limit_0) and (spe <= spe_limit) and not np.isnan(x0).all()

        updated = False
        if (in_control or self.update_when_out_of_control) and not np.isnan(x0).all():
            self._ewma_update_x(x0)
            # Re-scale the same observation with the freshly advanced vectors for
            # the kernel contribution, then recompute the loadings.
            x_for_kernel = np.nan_to_num((x0 - self.mx_) / self.sx_, nan=0.0)
            self._recompute(x_for_kernel)
            if in_control:
                self._spe_buffer.append(spe**2)
            self._n_updates_ += 1
            updated = True

        distance = _subspace_distance(self.loadings0_, self._loadings)
        self._hist_scores.append(t)
        self._hist_t2.append(t2)
        self._hist_spe.append(spe)
        self._hist_distance.append(distance)
        self._hist_index.append(label if label is not None else len(self._hist_index))
        self._refresh_history_frames()
        return Bunch(
            scores=t,
            hotellings_t2=t2,
            spe=spe,
            spe_limit=spe_limit,
            hotellings_t2_limit=self._t2_limit_0,
            in_control=bool(in_control),
            updated=updated,
            distance=distance,
        )

    def transform(self, X: DataMatrix) -> pd.DataFrame:
        """Project data onto the *current* model, returning the scores (no update)."""
        check_is_fitted(self, "mx_")
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
        out = np.array([self._project(row.to_numpy(dtype=float))[0] for _, row in X_df.iterrows()])
        return pd.DataFrame(out.reshape(-1, self.n_components), index=X_df.index, columns=self._component_names)


# -----------------------------------------------------------------------------
# Adaptive PLS.
# -----------------------------------------------------------------------------


class AdaptivePLS(_AdaptiveModel, RegressorMixin, TransformerMixin, BaseEstimator):
    """Recursive, adaptive Projection to Latent Structures for on-line monitoring.

    The PLS analogue of :class:`AdaptivePCA`, carrying both association matrices
    ``X'X`` and ``X'Y``. Seed on a block of common-cause data with :meth:`fit`,
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
        Per-update subspace overlap (on the weights ``W``) with the seed model.
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

    def fit(self, X: DataMatrix, Y: DataMatrix) -> AdaptivePLS:
        """Seed the adaptive PLS model from a block of common-cause data.

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

        # Seed preprocessing and the batch model, matching PLS(scale=True).
        self._x_scaler = MCUVScaler().fit(X_df)
        self._y_scaler = MCUVScaler().fit(Y_df)
        self.mx_ = self._x_scaler.center_.to_numpy(dtype=float).copy()
        self.sx_ = self._x_scaler.scale_.to_numpy(dtype=float).copy()
        self.my_ = self._y_scaler.center_.to_numpy(dtype=float).copy()
        self.sy_ = self._y_scaler.scale_.to_numpy(dtype=float).copy()
        seed = PLS(n_components=A, scale=True).fit(X_df, Y_df)

        Xs = np.nan_to_num(self._x_scaler.transform(X_df).to_numpy(dtype=float), nan=0.0)
        Ys = np.nan_to_num(self._y_scaler.transform(Y_df).to_numpy(dtype=float), nan=0.0)
        self.XtX0_ = Xs.T @ Xs
        self.XtY0_ = Xs.T @ Ys
        self.XtX_ = self.XtX0_.copy()
        self.XtY_ = self.XtY0_.copy()
        self._norm_xx0 = float(np.linalg.norm(self.XtX0_))
        self._norm_xy0 = float(np.linalg.norm(self.XtY0_))

        self._recompute_from_kernels(align_to=None)
        self.weights0_ = self._weights.copy()

        self._t2_limit_0 = _hotellings_t2_limit(conf_level=self.conf_level, n_components=A, n_rows=N)
        seed_spe = seed.spe_.iloc[:, A - 1].to_numpy(dtype=float)
        self._spe_limit_0 = spe_calculation(seed_spe, conf_level=self.conf_level)
        self._min_spe_window = max(20, A + 2)
        self._spe_buffer: deque[float] = deque(maxlen=int(self.spe_limit_window))
        self._spe_buffer.extend((seed_spe**2).tolist())

        self._hist_pred: list[np.ndarray] = []
        self._init_history()
        self._refresh_history_frames()
        return self

    def _recompute_from_kernels(self, align_to: np.ndarray | None) -> None:
        """Recompute W, P, R, C and beta from the current kernels; sign-align if asked."""
        W, P, R, C, score_ssq = _kernel_pls(self.XtX_, self.XtY_, self.n_components)
        if align_to is not None:
            signs = _sign_align(P, align_to)
            W *= signs
            P *= signs
            R *= signs
            C *= signs
        self._weights = W
        self._loadings = P
        self._direct = R
        self._y_loadings = C
        self._beta_scaled = R @ C.T  # (K, M) mapping scaled X to scaled Y
        self.explained_variance_ = score_ssq / max(1, self.n_samples_ - 1)
        self.scaling_factor_for_scores_ = pd.Series(
            np.sqrt(np.clip(self.explained_variance_, a_min=epsqrt**2, a_max=None)),
            index=self._component_names,
            name="Standard deviation per score",
        )

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

    def _project(self, x0: np.ndarray) -> tuple[np.ndarray, float, float, np.ndarray, np.ndarray]:
        """Project one raw observation; return scores, T2, SPE, raw prediction, scaled obs."""
        x = (x0 - self.mx_) / self.sx_
        P = self._loadings
        R = self._direct
        s = self.scaling_factor_for_scores_.to_numpy(dtype=float)
        missing = np.isnan(x)
        if not missing.any():
            t = x @ R
            residual = x - P @ t
            spe = float(np.sqrt(residual @ residual))
        else:
            x_obs = np.where(missing, 0.0, x)
            deflate = x_obs.copy()
            t = np.zeros(P.shape[1])
            for a in range(P.shape[1]):
                w_a = np.where(missing, 0.0, self._weights[:, a])
                denom = float(w_a @ w_a)
                t[a] = float(deflate @ w_a) / denom if denom > epsqrt else 0.0
                deflate = deflate - t[a] * np.where(missing, 0.0, P[:, a])
            spe = float(np.sqrt(deflate @ deflate))
        t2 = float(np.sum((t / s) ** 2))
        y_scaled_hat = t @ self._y_loadings.T
        y_hat = y_scaled_hat * self.sy_ + self.my_
        return t, t2, spe, y_hat, x

    def _recompute(self, x_scaled: np.ndarray, y_scaled: np.ndarray | None) -> None:
        """Update the kernels with one scaled observation and recompute the model."""
        mu = self.forgetting_factor
        update_xx = mu * np.outer(x_scaled, x_scaled)
        f_x = self.gamma * float(np.linalg.norm(update_xx)) / self._norm_xx0 if self._norm_xx0 > 0 else 0.0
        self.XtX_ = (1.0 - mu) * self.XtX_ + update_xx + f_x * self.XtX0_
        norm_xx = float(np.linalg.norm(self.XtX_))
        if norm_xx > epsqrt:
            self.XtX_ *= self._norm_xx0 / norm_xx

        if y_scaled is not None and self._norm_xy0 > 0:
            update_xy = mu * np.outer(x_scaled, y_scaled)
            f_y = self.gamma * float(np.linalg.norm(update_xy)) / self._norm_xy0
            self.XtY_ = (1.0 - mu) * self.XtY_ + update_xy + f_y * self.XtY0_
            norm_xy = float(np.linalg.norm(self.XtY_))
            if norm_xy > epsqrt:
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
        t, t2, spe, y_hat, _ = self._project(x0)
        spe_limit = self._current_spe_limit()
        all_missing = bool(np.isnan(x0).all())
        in_control = (t2 <= self._t2_limit_0) and (spe <= spe_limit) and not all_missing

        updated = False
        if (in_control or self.update_when_out_of_control) and not all_missing:
            self._ewma_update_x(x0)
            y_scaled = None
            if y_row is not None:
                y0 = np.asarray(y_row, dtype=float).ravel()
                if not np.isnan(y0).any():
                    self.sy_ = np.sqrt(
                        (1.0 - self.alpha_scale_y_) * self.sy_**2 + self.alpha_scale_y_ * (y0 - self.my_) ** 2
                    )
                    self.sy_ = np.where(self.sy_ < epsqrt, 1.0, self.sy_)
                    self.my_ = (1.0 - self.lambda_center_y_) * self.my_ + self.lambda_center_y_ * y0
                    y_scaled = (y0 - self.my_) / self.sy_
            x_for_kernel = np.nan_to_num((x0 - self.mx_) / self.sx_, nan=0.0)
            self._recompute(x_for_kernel, y_scaled)
            if in_control:
                self._spe_buffer.append(spe**2)
            self._n_updates_ += 1
            updated = True

        distance = _subspace_distance(self.weights0_, self._weights)
        self._hist_scores.append(t)
        self._hist_t2.append(t2)
        self._hist_spe.append(spe)
        self._hist_pred.append(y_hat)
        self._hist_distance.append(distance)
        self._hist_index.append(label if label is not None else len(self._hist_index))
        self._refresh_history_frames()
        return Bunch(
            scores=t,
            prediction=y_hat,
            hotellings_t2=t2,
            spe=spe,
            spe_limit=spe_limit,
            hotellings_t2_limit=self._t2_limit_0,
            in_control=bool(in_control),
            updated=updated,
            distance=distance,
        )

    def _refresh_history_frames(self) -> None:
        """Extend the base history frames with the Y-prediction history."""
        super()._refresh_history_frames()
        self.predictions_ = pd.DataFrame(
            np.array(self._hist_pred).reshape(-1, self.n_targets_),
            index=self._hist_index,
            columns=self._target_names,
        )

    def transform(self, X: DataMatrix) -> pd.DataFrame:
        """Project data onto the *current* model, returning the X scores (no update)."""
        check_is_fitted(self, "mx_")
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
        out = np.array([self._project(row.to_numpy(dtype=float))[0] for _, row in X_df.iterrows()])
        return pd.DataFrame(out.reshape(-1, self.n_components), index=X_df.index, columns=self._component_names)

    def predict(self, X: DataMatrix) -> pd.DataFrame:
        """Predict the response with the *current* model (no update)."""
        check_is_fitted(self, "mx_")
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
        out = np.array([self._project(row.to_numpy(dtype=float))[3] for _, row in X_df.iterrows()])
        return pd.DataFrame(out.reshape(-1, self.n_targets_), index=X_df.index, columns=self._target_names)
