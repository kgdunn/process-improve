# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.
"""Partial Robust M-regression: a PLS that a handful of bad rows cannot capture (#191).

Ordinary PLS minimises a sum of squares, so its breakdown point is zero: a single
sufficiently wild observation moves the fit without limit. PRM (Serneels, Croux,
Filzmoser and Van Espen, 2005) replaces that with an iteratively reweighted fit,
where each row's weight falls off with how badly the current model misses it and
with how far it sits from the middle of the score space.

The implementation is deliberately thin. PRM is *exactly* PLS on ``sqrt(w)``
rescaled rows, and :meth:`PLS.fit <process_improve.multivariate.methods.PLS.fit>`
already does that rescale for its ``sample_weight`` argument, so the loop below
calls the ordinary fit repeatedly rather than forking the NIPALS kernel. The one
thing it does override is the centring and scaling: see :meth:`PRM._make_scalers`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import DataMatrix, epsqrt
from ._pls import PLS
from ._preprocessing import MCUVScaler, _WeightedMCUVScaler
from ._robust import fair_weights, l1_median

#: Scale factor making the median absolute deviation consistent for the
#: standard deviation of a Gaussian.
_MAD_TO_SIGMA = 1.4826

#: A step counts as progress only if it beats the best step so far by this
#: factor; anything else is the loop marking time.
_STALL_RATIO = 0.99

#: How many non-improving steps to allow before giving up on convergence.
_STALL_PATIENCE = 10


def _robust_scale(values: np.ndarray) -> float:
    """Median absolute deviation of ``values``, scaled to estimate a Gaussian sigma.

    Floored away from zero: more than half the residuals being identical (a
    saturated sensor, a heavily quantised response) would otherwise give a scale
    of zero and turn every standardised residual into an infinity.
    """
    values = np.asarray(values, dtype=float).ravel()
    mad = float(np.median(np.abs(values - np.median(values))))
    return max(mad * _MAD_TO_SIGMA, epsqrt)


def _as_frame(data: DataMatrix) -> pd.DataFrame:
    """Coerce to a DataFrame, keeping a Series' name as its column label.

    PRM does its own numpy work on the blocks before handing them to ``PLS.fit``,
    so it needs the frame form up front rather than relying on the base class to
    build one.
    """
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, pd.Series):
        return data.to_frame()
    return pd.DataFrame(np.asarray(data, dtype=float))


class PRM(PLS):
    r"""Partial Robust M-regression: PLS with a bounded influence per observation.

    Each row carries a weight :math:`w_i = w_i^r \cdot w_i^x`, the product of a
    *residual* weight (how badly the current model predicts that row) and a
    *leverage* weight (how far its scores sit from the middle of the score
    cloud). Both come from the Fair function, so the two kinds of outlier that
    break a least-squares fit are handled by the same mechanism:

    - a **vertical outlier** has an ordinary position in X but a y that does not
      follow the relationship, and gets a small :math:`w^r`;
    - a **bad leverage point** sits far out in X as well, where least squares
      gives it the most influence of all, and gets a small :math:`w^x`.

    The weights are recomputed from the fit and the fit is recomputed from the
    weights until they settle.

    .. note::
       Centring and scaling are **weighted too**, which is not a refinement but
       the part that makes the method work. The column mean and standard
       deviation have a breakdown point of zero, so leaving them unweighted
       leaves the outliers setting the coordinate system that the weighted fit
       then runs in: measured on a fixture with 15% vertical outliers, weighting
       only the fit recovers none of the damage, and weighting the scaling as
       well recovers essentially all of it. See :meth:`_make_scalers`.

    Parameters
    ----------
    n_components : int
        Number of components to extract.
    cutoff : float, optional
        Tuning constant :math:`c` of the Fair weight function, default 4.0 as
        recommended by Serneels et al. Smaller is more aggressive: an
        observation at :math:`c` standardised units gets weight 0.25. This
        trades robustness against efficiency, and 4.0 keeps roughly 95% of the
        efficiency of ordinary PLS on clean Gaussian data.
    max_weight_iter : int, optional
        Maximum reweighting iterations, default 100. Each one is a full PLS fit.
    weight_tol : float, optional
        Convergence tolerance, default 1e-4, on the largest absolute change in
        any row's weight between iterations.
    scale, max_iter, tol, copy, missing_data_settings, warn_on_uncentred
        As for :class:`~process_improve.multivariate.methods.PLS`.

    Attributes
    ----------
    robust_weights_ : np.ndarray of shape (n_samples,)
        The converged row weights, in (0, 1]. Small entries are this model's
        statement about which rows it declined to be led by; see
        :meth:`outlier_summary`.
    n_weight_iter_ : int
        Reweighting iterations actually run.
    weights_converged_ : bool
        Whether the loop met ``weight_tol``. ``False`` is not necessarily a
        failure: read ``weight_shift_`` before treating it as one.
    weight_shift_ : float
        The largest change in any row's weight on the final iteration. This is
        what makes ``weights_converged_ = False`` actionable rather than a bare
        flag, because the reweighting can settle into a small limit cycle rather
        than a point; a shift of 0.03 says the weights are stable to 3%, which
        for most purposes is settled.

    All the fitted attributes of :class:`~process_improve.multivariate.methods.PLS`
    are also present, and mean the same thing, computed on the final weighted fit.

    References
    ----------
    S. Serneels, C. Croux, P. Filzmoser and P.J. Van Espen, "Partial Robust
    M-regression", Chemometrics and Intelligent Laboratory Systems, 79 (2005),
    55-64.

    Examples
    --------
    >>> model = PRM(n_components=2).fit(X, y)                  # doctest: +SKIP
    >>> model.outlier_summary(threshold=0.1)                   # doctest: +SKIP
    >>> model.predict(X_new)                                   # doctest: +SKIP
    """

    def __init__(  # noqa: PLR0913 - mirrors PLS's constructor plus three loop controls
        self,
        n_components: int,
        *,
        cutoff: float = 4.0,
        max_weight_iter: int = 100,
        weight_tol: float = 1e-4,
        scale: bool = True,
        max_iter: int = 1000,
        tol: float = epsqrt,
        copy: bool = True,
        missing_data_settings: dict | None = None,
        warn_on_uncentred: bool = True,
    ):
        # `PLS.__init__` rather than `super().__init__` for the same reason PLSDA
        # names it: it says which constructor runs instead of leaving it to be
        # read off the MRO, and CodeQL's py/missing-call-to-init can see it.
        PLS.__init__(
            self,
            n_components=n_components,
            scale=scale,
            max_iter=max_iter,
            tol=tol,
            copy=copy,
            missing_data_settings=missing_data_settings,
            warn_on_uncentred=warn_on_uncentred,
        )
        # Stored verbatim and read only in fit(), per the sklearn __init__ convention.
        self.cutoff = cutoff
        self.max_weight_iter = max_weight_iter
        self.weight_tol = weight_tol

    def _make_scalers(
        self,
        X: pd.DataFrame,
        Y: pd.DataFrame,
        sample_weight: np.ndarray | None,
    ) -> tuple[MCUVScaler, MCUVScaler]:
        """Centre and scale by *weighted* statistics rather than the plain mean and SD.

        This override is the reason PRM works at all. The base class fits its
        scalers on every row with non-zero weight, and PRM's weights are never
        exactly zero (the Fair function does not reach it), so without this the
        centring and scaling would be the ordinary, breakdown-point-zero ones.
        The weighted fit would then be running inside a coordinate system that
        the outliers had already chosen.

        With equal weights this returns exactly what the base class does, so the
        override changes nothing for a caller who is not reweighting.
        """
        return (
            _WeightedMCUVScaler().fit(X, sample_weight=sample_weight),
            _WeightedMCUVScaler().fit(Y, sample_weight=sample_weight),
        )

    def _starting_weights(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Weights for the first iteration, from the data alone.

        The loop needs a starting point that is already resistant, because an
        iteration started from the least-squares fit can be led straight to a
        bad local solution and stay there: reweighting descends, it does not
        search. So leverage is measured against the L1 median of X and the
        residual against the median of Y, neither of which the outliers control.
        """
        leverage = np.linalg.norm(X - l1_median(X), axis=1)
        median_leverage = max(float(np.median(leverage)), epsqrt)
        weights = fair_weights(leverage / median_leverage, self.cutoff)

        for column in Y.T:
            deviation = column - np.median(column)
            weights = weights * fair_weights(deviation / _robust_scale(deviation), self.cutoff)
        return weights

    def _updated_weights(self, X: pd.DataFrame, Y: pd.DataFrame) -> np.ndarray:
        """Recompute the row weights from the fit that is currently in place."""
        # Residuals in the scaled space, so that targets measured in different
        # units contribute comparably to a multi-target row's residual.
        # ``scale=False`` says the caller has already put Y into modelling units,
        # so there is nothing to divide by; the scaler is then absent, not broken.
        y_scale = 1.0 if self._y_scaler is None else np.asarray(self._y_scaler.scale_, dtype=float)
        residuals = (np.asarray(Y, dtype=float) - np.asarray(self.predict(X), dtype=float)) / y_scale
        # A single number per row: the paper is written for a single y, and the
        # row norm is the reading of it that keeps one bad target from being
        # averaged away by several good ones.
        per_row = np.linalg.norm(residuals, axis=1)
        residual_weights = fair_weights(per_row / _robust_scale(per_row), self.cutoff)

        scores = np.asarray(self.scores_, dtype=float)
        distances = np.linalg.norm(scores - l1_median(scores), axis=1)
        leverage_weights = fair_weights(distances / max(float(np.median(distances)), epsqrt), self.cutoff)

        return residual_weights * leverage_weights

    def fit(self, X: DataMatrix, Y: DataMatrix, sample_weight: np.ndarray | None = None) -> PRM:
        """Fit by alternating between a weighted PLS fit and a reweighting step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        Y : array-like of shape (n_samples, n_targets)
            Target values.
        sample_weight : array-like of shape (n_samples,), optional
            Prior row weights, multiplied into the robust weights at every
            iteration. Use this to express knowledge the data cannot carry, such
            as a run you already know was compromised; it is not needed to
            downweight outliers, which is what the model is for.

        Returns
        -------
        PRM
            ``self``, fitted.

        Raises
        ------
        ValueError
            If ``cutoff``, ``max_weight_iter`` or ``weight_tol`` is out of range,
            or if the data contain missing values (see Notes).

        Notes
        -----
        Missing data are refused rather than threaded through. The weights are
        built from residual and leverage distances, and a row with missing cells
        has a distance that is not comparable with a complete row's, so it would
        be downweighted for being incomplete rather than for being wrong. The
        underlying :class:`~process_improve.multivariate.methods.PLS` does handle
        missing data; use it, or impute first.
        """
        if not np.isfinite(self.cutoff) or self.cutoff <= 0:
            raise ValueError(f"cutoff must be a positive finite number; got {self.cutoff!r}.")
        if int(self.max_weight_iter) < 1:
            raise ValueError(f"max_weight_iter must be at least 1; got {self.max_weight_iter!r}.")
        if not np.isfinite(self.weight_tol) or self.weight_tol <= 0:
            raise ValueError(f"weight_tol must be a positive finite number; got {self.weight_tol!r}.")

        X_df, Y_df = _as_frame(X), _as_frame(Y)
        if np.any(np.isnan(X_df.to_numpy(dtype=float))) or np.any(np.isnan(Y_df.to_numpy(dtype=float))):
            raise ValueError(
                "PRM does not support missing data: a row with missing cells has residual and leverage "
                "distances that are not comparable with a complete row's, so it would be downweighted for "
                "being incomplete rather than for being an outlier. Use PLS, which threads missing data "
                "through NIPALS, or impute first."
            )

        prior = np.ones(X_df.shape[0]) if sample_weight is None else np.asarray(sample_weight, dtype=float).ravel()
        weights = self._starting_weights(X_df.to_numpy(dtype=float), Y_df.to_numpy(dtype=float))

        self.weights_converged_ = False
        self.n_weight_iter_ = 0
        self.weight_shift_ = np.inf
        best_shift, stalled = np.inf, 0
        for iteration in range(1, int(self.max_weight_iter) + 1):
            self.n_weight_iter_ = iteration
            PLS.fit(self, X_df, Y_df, sample_weight=prior * weights)
            updated = self._updated_weights(X_df, Y_df)
            shift = float(np.max(np.abs(updated - weights)))
            weights = updated
            self.weight_shift_ = shift
            if shift <= self.weight_tol:
                self.weights_converged_ = True
                break

            # The reweighting can settle into a small limit cycle instead of a
            # point: the leverage weight normalises by the median score distance
            # of the very scores it is changing, so suppressing an outlier
            # contracts the cloud, which raises everyone's standardised distance,
            # which changes the fit again. Once the step stops shrinking, every
            # further iteration costs a full PLS fit and buys nothing, so stop
            # and let ``weight_shift_`` say how stable it actually got.
            if shift < best_shift * _STALL_RATIO:
                best_shift, stalled = shift, 0
            else:
                stalled += 1
                if stalled >= _STALL_PATIENCE:
                    break

        # One last fit, so the model that is returned is the one the final
        # weights describe rather than the one they were computed from.
        PLS.fit(self, X_df, Y_df, sample_weight=prior * weights)
        self.robust_weights_ = weights
        return self

    def outlier_summary(self, threshold: float = 0.1) -> pd.DataFrame:
        """Rows the fit declined to be led by, weakest weight first.

        Parameters
        ----------
        threshold : float, optional
            Report rows whose final weight is below this, default 0.1. There is
            no distinguished value: the weights are continuous by design, so
            this is a reading aid and not a test.

        Returns
        -------
        pd.DataFrame
            Indexed as the training X was, with one ``weight`` column. Empty if
            nothing falls below ``threshold``.

        Raises
        ------
        AttributeError
            If the model is not fitted.
        """
        if not hasattr(self, "robust_weights_"):
            raise AttributeError("This PRM instance is not fitted yet; call 'fit' first.")
        flagged = pd.DataFrame({"weight": self.robust_weights_}, index=self._sample_index)
        return flagged[flagged["weight"] < threshold].sort_values("weight")
