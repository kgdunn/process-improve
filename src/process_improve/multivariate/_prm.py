# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.
"""Partial Robust M-regression: a PLS that a handful of bad rows cannot capture (#191).

Ordinary PLS minimises a sum of squares, so its breakdown point is zero: a single
sufficiently wild observation moves the fit without limit. PRM (Serneels, Croux,
Filzmoser and Van Espen, 2005) replaces that with an iteratively reweighted fit,
where each row's weight falls off with how badly the current model misses it and
with how far it sits from the middle of the score space.

The loop is the paper's, step for step, as the R package chemometrics (by one of
its authors) codes it, with one exception. chemometrics subtracts the scores' L1
median as ``T - mt``, and R recycles that vector down the columns cell by cell, so
with two or more components the median's entries land in the wrong cells; this
module subtracts it from each row, as the paper does. Each iteration is an
ordinary weighted fit: PRM is exactly PLS on ``sqrt(w)``-rescaled rows, which
:meth:`PLS.fit <process_improve.multivariate.methods.PLS.fit>` does for its
``sample_weight``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ._common import DataMatrix, epsqrt
from ._pls import PLS
from ._preprocessing import MCUVScaler
from ._robust import fair_weights, l1_median

#: Scale factor making the median absolute deviation consistent for the
#: standard deviation of a Gaussian, so a column's robust spread reads in its units.
_MAD_TO_SIGMA = 1.4826

#: The weight that stands in for an exact zero, as chemometrics does: a zero would
#: drop the row from the weighted fit, where the method only ever downweights.
_SMALLEST_WEIGHT = 1e-6


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


def _checked_prior(sample_weight: np.ndarray | None, n_samples: int) -> np.ndarray:
    """Resolve the caller's ``sample_weight`` into a usable prior-weight vector.

    Validated up front because the reweighting uses it as a mask, not only as a
    multiplier: a bad length would otherwise surface several iterations later as
    a confusing broadcasting error.

    Raises
    ------
    ValueError
        If the length does not match, or no row is left with positive weight.
    """
    if sample_weight is None:
        return np.ones(n_samples)
    prior = np.asarray(sample_weight, dtype=float).ravel()
    if prior.shape[0] != n_samples:
        raise ValueError(f"sample_weight has {prior.shape[0]} entries; expected {n_samples} to match X / Y.")
    if not np.any(prior > 0):
        raise ValueError("sample_weight must leave at least one row with a positive weight.")
    return prior


def _robust_scaler(block: pd.DataFrame, active: np.ndarray, *, scale: bool, l1: bool) -> MCUVScaler:
    """Return a scaler centred robustly, once, on the rows the prior keeps.

    The centre is the L1 median of the rows (``l1``, for X) or the median of each
    column (for Y). With ``scale``, each column is also divided by 1.4826 times its
    median absolute deviation, or by 1 where that is 0, and the L1 median is taken
    in those units, the space the leverage distances are then measured in.
    """
    rows = block.to_numpy(dtype=float)[active]
    spread = np.ones(rows.shape[1])
    if scale:
        mad = np.median(np.abs(rows - np.median(rows, axis=0)), axis=0) * _MAD_TO_SIGMA
        spread = np.where(mad > epsqrt, mad, 1.0)
    centre = (l1_median(rows / spread) if l1 else np.median(rows / spread, axis=0)) * spread
    # Fitted for the sklearn metadata of the block (feature names, counts); centre and spread replaced.
    scaler = MCUVScaler().fit(block)
    scaler.center_ = pd.Series(centre, index=scaler.center_.index)
    scaler.scale_ = pd.Series(spread, index=scaler.scale_.index)
    return scaler


def _residual_weights(residuals: np.ndarray, active: np.ndarray, cutoff: float) -> np.ndarray:
    """Fair weight of each row's residual, centred at its median and divided by its median absolute value.

    The paper's scale is the median absolute residual itself, not the 1.4826 times
    it that estimates a Gaussian standard deviation: the cutoff is tuned to it. With
    several targets each is standardised alone and a row's residual is the norm of
    the result, which for one target is the paper's ``|r| / median |r|``.
    """
    centred = residuals - np.median(residuals[active], axis=0)
    spread = np.maximum(np.median(np.abs(centred[active]), axis=0), epsqrt)
    return fair_weights(np.linalg.norm(centred / spread, axis=1), cutoff)


def _leverage_weights(points: np.ndarray, active: np.ndarray, cutoff: float) -> np.ndarray:
    """Fair weight of each row's distance from the L1 median of the points, over the median distance."""
    distances = np.linalg.norm(points - l1_median(points[active]), axis=1)
    return fair_weights(distances / max(float(np.median(distances[active])), epsqrt), cutoff)


class PRM(PLS):
    r"""Partial Robust M-regression: PLS with a bounded influence per observation.

    Each row carries a weight :math:`w_i = w_i^r \cdot w_i^t`, the product of a
    *residual* weight (how badly the current model predicts that row) and a
    *leverage* weight (how far its scores sit from the middle of the score
    cloud). Both come from the Fair function, so the two kinds of outlier that
    break a least-squares fit are handled by the same mechanism:

    - a **vertical outlier** has an ordinary position in X but a y that does not
      follow the relationship, and gets a small :math:`w^r`;
    - a **bad leverage point** sits far out in X as well, where least squares
      gives it the most influence of all, and gets a small :math:`w^t`.

    The steps are those of Serneels et al. (2005):

    1. X is centred at its L1 median and y at its median, once. Neither moves
       when an outlier does, so the outliers do not choose the coordinate system.
    2. The starting weights come from the data alone: each centred row's
       distance from the centre, over the median distance, and each centred y,
       over the median absolute y.
    3. A PLS model is fitted with the weights, and new weights come from it: the
       residual, centred at its median and over its median absolute value; and
       the distance of the row's scores from their L1 median, over the median
       distance, each component scaled to unit length in the weighted fit (as
       SIMPLS's scores are), so the first components do not decide it alone.
    4. Step 3 repeats until the norm of the regression of y on those unit-length
       scores changes by less than ``coefficient_tol``, relative.
    5. The intercept is the median of the residuals, :math:`y - Xb`.

    Parameters
    ----------
    n_components : int
        Number of components to extract.
    cutoff : float, optional
        Tuning constant :math:`c` of the Fair weight function, default 4.0 as
        recommended by Serneels et al. Smaller is more aggressive: an
        observation at :math:`c` standardised units gets weight 0.25.
    max_weight_iter : int, optional
        Maximum reweighting iterations, default 100. Each one is a full PLS fit.
    coefficient_tol : float, optional
        The relative change in the norm of the regression of y on the unit-length
        scores below which the reweighting stops, default 0.01 as in the paper.
    weight_tol : None
        Deprecated since 2.2.2 and removed in 3.0; it has no effect. PRM stops on
        ``coefficient_tol``, as the paper does, not on the change in the weights.
    scale : bool, optional
        True (the default) also divides each column of X and of Y, after the
        robust centring, by 1.4826 times its median absolute deviation (by 1
        where that is 0), computed once: columns in different units then count
        equally, as PLS's ``scale`` makes them. False is the paper's algorithm,
        which centres only.
    max_iter, tol, copy, missing_data_settings
        As for :class:`~process_improve.multivariate.methods.PLS`.
    warn_on_uncentred : bool, optional
        Accepted for :class:`~process_improve.multivariate.methods.PLS`'s
        signature; it has no effect, since PRM always centres.

    Attributes
    ----------
    robust_weights_ : np.ndarray of shape (n_samples,)
        The weight the returned model gives each row, in (0, 1]: its residual
        weight times its leverage weight, from the final fit, as the paper reports
        them. Small entries are this model's statement about which rows it
        declined to be led by; see :meth:`outlier_summary`.
    n_weight_iter_ : int
        Reweighting iterations actually run.
    weights_converged_ : bool
        Whether the loop met ``coefficient_tol`` within ``max_weight_iter``.
    weight_shift_ : float
        The largest change in any row's weight on the final iteration.

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

    def __init__(  # noqa: PLR0913 - mirrors PLS's constructor plus four loop controls
        self,
        n_components: int,
        *,
        cutoff: float = 4.0,
        max_weight_iter: int = 100,
        coefficient_tol: float = 0.01,
        weight_tol: float | None = None,
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
        self.coefficient_tol = coefficient_tol
        self.weight_tol = weight_tol

    def _check_settings(self) -> None:
        """Validate the loop controls before any fitting happens.

        Raises
        ------
        ValueError
            If ``cutoff``, ``max_weight_iter`` or ``coefficient_tol`` is out of range.
        """
        if not np.isfinite(self.cutoff) or self.cutoff <= 0:
            raise ValueError(f"cutoff must be a positive finite number; got {self.cutoff!r}.")
        if int(self.max_weight_iter) < 1:
            raise ValueError(f"max_weight_iter must be at least 1; got {self.max_weight_iter!r}.")
        if not np.isfinite(self.coefficient_tol) or self.coefficient_tol <= 0:
            raise ValueError(f"coefficient_tol must be a positive finite number; got {self.coefficient_tol!r}.")
        if self.weight_tol is not None:
            warnings.warn(
                "process_improve.multivariate.methods.PRM.weight_tol is deprecated since 2.2.2 and will be "
                "removed in 3.0; use coefficient_tol instead. weight_tol has no effect: PRM stops, as Serneels "
                "et al. (2005) do, when the norm of its regression on the unit-length scores changes by less "
                "than coefficient_tol.",
                DeprecationWarning,
                stacklevel=3,
            )

    def _centres_and_scales(self) -> bool:
        """Return True: PRM centres robustly even when it does not scale."""
        return True

    def _make_scalers(
        self,
        X: pd.DataFrame,  # noqa: ARG002 - the scalers were fitted once, before the loop
        Y: pd.DataFrame,  # noqa: ARG002
        sample_weight: np.ndarray | None,  # noqa: ARG002
    ) -> tuple[MCUVScaler, MCUVScaler]:
        """Return the robust scalers ``fit`` built before reweighting, the same at every iteration.

        Recomputing the centre with the weights would let the fit move its own
        origin; the paper centres once, at statistics no outlier controls.
        """
        return self._robust_scalers

    def _reweighted(self, Y: pd.DataFrame, fit_weights: np.ndarray, active: np.ndarray) -> tuple[np.ndarray, float]:
        """Return the weights the current fit gives each row, and the norm of its regression on unit-length scores."""
        residuals = Y.to_numpy(dtype=float) - self.predictions_.to_numpy(dtype=float)
        residual = _residual_weights(residuals, active, self.cutoff)
        # Each component scaled to unit length in the weighted fit, as SIMPLS's scores
        # are; NIPALS's first components are much the longest, and would decide alone.
        lengths = np.maximum(np.linalg.norm(np.sqrt(fit_weights)[:, None] * self._scores, axis=0), epsqrt)
        leverage = _leverage_weights(self._scores / lengths, active, self.cutoff)
        weights = residual * leverage
        weights[weights == 0] = _SMALLEST_WEIGHT
        # The regression of y on the unit-length scores is each y-loading times its score's length.
        return weights, float(np.linalg.norm(np.asarray(self.y_loadings_, dtype=float) * lengths))

    def fit(self, X: DataMatrix, Y: DataMatrix, sample_weight: np.ndarray | None = None) -> PRM:
        """Fit by alternating between a weighted PLS fit and a reweighting step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        Y : array-like of shape (n_samples, n_targets)
            Target values. The paper is written for one; with several, a row's
            residual is the norm of its residuals, each standardised alone.
        sample_weight : array-like of shape (n_samples,), optional
            Prior row weights, multiplied into the robust weights at every
            iteration. Use this to express knowledge the data cannot carry, such
            as a run you already know was compromised; it is not needed to
            downweight outliers, which is what the model is for. A row with a
            zero weight is left out of every median and centre, as if dropped.

        Returns
        -------
        PRM
            ``self``, fitted.

        Raises
        ------
        ValueError
            If ``cutoff``, ``max_weight_iter`` or ``coefficient_tol`` is out of
            range, or if the data contain missing values (see Notes).

        Notes
        -----
        Missing data are refused rather than threaded through. The weights are
        built from residual and leverage distances, and a row with missing cells
        has a distance that is not comparable with a complete row's, so it would
        be downweighted for being incomplete rather than for being wrong. The
        underlying :class:`~process_improve.multivariate.methods.PLS` does handle
        missing data; use it, or impute first.
        """
        self._check_settings()
        X_df, Y_df = _as_frame(X), _as_frame(Y)
        if np.any(np.isnan(X_df.to_numpy(dtype=float))) or np.any(np.isnan(Y_df.to_numpy(dtype=float))):
            raise ValueError(
                "PRM does not support missing data: a row with missing cells has residual and leverage "
                "distances that are not comparable with a complete row's, so it would be downweighted for "
                "being incomplete rather than for being an outlier. Use PLS, which threads missing data "
                "through NIPALS, or impute first."
            )

        prior = _checked_prior(sample_weight, X_df.shape[0])
        active = prior > 0
        self._robust_scalers = (
            _robust_scaler(X_df, active, scale=self.scale, l1=True),
            _robust_scaler(Y_df, active, scale=self.scale, l1=False),
        )
        x_centred = self._robust_scalers[0].transform(X_df).to_numpy(dtype=float)
        y_centred = self._robust_scalers[1].transform(Y_df).to_numpy(dtype=float)
        weights = _leverage_weights(x_centred, active, self.cutoff) * _residual_weights(y_centred, active, self.cutoff)

        self.weights_converged_ = False
        regression_norm = np.inf
        for iteration in range(1, int(self.max_weight_iter) + 1):
            self.n_weight_iter_ = iteration
            fit_weights = prior * weights
            PLS.fit(self, X_df, Y_df, sample_weight=fit_weights)
            updated, norm = self._reweighted(Y_df, fit_weights, active)
            self.weight_shift_ = float(np.max(np.abs(updated - weights)))
            change = abs(norm - regression_norm)  # infinite on the first iteration
            weights, regression_norm = updated, norm
            if change <= self.coefficient_tol * max(norm, epsqrt):
                self.weights_converged_ = True
                break

        # The returned model is the last fit, and the weights are the ones it gives each
        # row, as the paper has it; no fit after them, which would be another model.
        self.robust_weights_ = weights
        # The paper's intercept is the median residual of y - Xb, not the centre the
        # weighted fit ran about (the median of y): move the centre by the difference.
        shift = np.median(Y_df.to_numpy(dtype=float)[active] - self.predictions_.to_numpy(dtype=float)[active], axis=0)
        y_scaler = self._robust_scalers[1]  # the scaler PLS.fit set as self._y_scaler
        y_scaler.center_ = y_scaler.center_ + shift
        self.predictions_ = self.predictions_ + shift
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
