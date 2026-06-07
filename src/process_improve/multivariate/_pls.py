# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.
"""Projection to Latent Structures (PLS) regression estimator (ENG-01).

The sklearn-compatible :class:`PLS` regressor (NIPALS, missing-data aware), with
cross-validation, prediction intervals, diagnostics, confidence limits and
plotting bound as convenience methods after ``fit()``.
"""

from __future__ import annotations

import logging
import time
import typing
import warnings

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.metrics import r2_score
from sklearn.model_selection import BaseCrossValidator, KFold, RepeatedKFold, check_cv
from sklearn.utils import Bunch
from sklearn.utils.validation import check_is_fitted, validate_data
from tqdm import tqdm

from .._linalg import safe_inverse
from ..univariate.metrics import detect_outliers_esd
from ._base import _LatentVariableModel, _LazyFrame
from ._common import (
    Q2_MIN_INCREMENT,
    DataMatrix,
    NotEnoughVarianceError,
    SelectionRule,
    SpecificationWarning,
    _align_to_fit_features,
    _model_method,
    _select_n_components,
    epsqrt,
)
from ._nipals import quick_regress, ssq, terminate_check
from ._preprocessing import MCUVScaler
from .plots import (
    coefficient_plot as _coefficient_plot,
)
from .plots import (
    predictions_vs_observed_plot as _predictions_vs_observed_plot,
)

logger = logging.getLogger(__name__)


def _vandervoet_randomization(
    per_obs_sse: np.ndarray,
    *,
    total_rmsecv: np.ndarray,
    n_permutations: int = 999,
    alpha: float = 0.01,
    random_state: int | None = None,
) -> tuple[int, np.ndarray]:
    """Van der Voet (1994) randomization test for PLS component selection.

    Compares every candidate model against the reference (argmin-RMSECV)
    model under the null that the two have the same predictive ability.
    For each observation the paired difference of squared residuals
    ``D_i = sse[a, i] - sse[a*, i]`` is computed; under the null its sign
    is random, so the permutation distribution of ``T = sum_i D_i`` is
    obtained by flipping each ``D_i``'s sign with probability 1/2 over
    ``n_permutations`` draws. The *p*-value is the right-tail probability
    of seeing a sum as large as the observed one (``T_obs >= T_perm``);
    the recommendation is the smallest ``a`` whose ``p > alpha`` -
    statistically indistinguishable from the reference, but more
    parsimonious.

    Parameters
    ----------
    per_obs_sse : np.ndarray of shape (n_components, n_samples)
        Out-of-fold per-observation squared total residual at every
        component count, summed across Y columns. Rows that are NaN
        (observation never held out) are dropped.
    total_rmsecv : np.ndarray of shape (n_components,)
        Pooled total RMSECV per component count; used to pick the
        reference model ``a*`` = ``nanargmin(total_rmsecv) + 1``.
    n_permutations : int, default 999
        Number of sign-flip permutations.
    alpha : float, default 0.01
        Significance level. Smaller values are more parsimonious.
    random_state : int, optional
        Seed for reproducible permutations.

    Returns
    -------
    recommended : int
        Smallest 1-based component count with ``p > alpha``.
    p_values : np.ndarray of shape (n_components,)
        Right-tail *p*-value per candidate; the reference model gets
        ``1.0`` by construction (paired differences are all zero).

    References
    ----------
    Van der Voet, H. (1994). Comparing the predictive accuracy of
    models using a simple randomization test. *Chemom. Intell. Lab.
    Syst.*, 25(2), 313-323.
    """
    a_count = per_obs_sse.shape[0]
    a_ref = int(np.nanargmin(total_rmsecv))
    rng = np.random.default_rng(random_state)
    p_values = np.zeros(a_count)
    p_values[a_ref] = 1.0
    sse_ref = per_obs_sse[a_ref]
    for a in range(a_count):
        if a == a_ref:
            continue
        d = per_obs_sse[a] - sse_ref
        # Drop observations with NaN (a custom splitter may have left some
        # rows unheld), since the paired difference is undefined there.
        d = d[np.isfinite(d)]
        if d.size == 0:
            p_values[a] = 1.0
            continue
        t_obs = float(d.sum())
        signs = rng.choice([-1.0, 1.0], size=(n_permutations, d.size))
        t_perm = (signs * d).sum(axis=1)
        # Right-tail probability under the null. Add 1 to both numerator
        # and denominator (the "permutation test +1" correction) so the
        # p-value is strictly positive even at the extreme.
        p_values[a] = float((np.sum(t_perm >= t_obs) + 1) / (n_permutations + 1))

    recommended = a_ref + 1  # fall back to the reference if nothing qualifies
    for a in range(a_count):
        if p_values[a] > alpha:
            recommended = a + 1
            break
    return recommended, p_values


class PLS(_LatentVariableModel, RegressorMixin, TransformerMixin, BaseEstimator):
    """Projection to Latent Structures (PLS) regression with diagnostics.

    Implements PLS via the NIPALS algorithm with production diagnostics: SPE,
    Hotelling's T², score contributions, and outlier detection. The API mirrors
    :class:`PCA` so that ``model.scores_``, ``model.spe_``, and
    ``model.detect_outliers()`` work identically for both model types.

    Parameters
    ----------
    n_components : int
        Number of latent components to extract.
    scale : bool, default=True
        Whether to scale X and Y to unit variance (sklearn internal scaling).
        When using ``MCUVScaler`` externally, set ``scale=False`` to avoid
        double scaling.
    max_iter : int, default=1000
        Maximum number of iterations for the NIPALS algorithm.
    tol : float, default=sqrt(machine epsilon)
        Convergence tolerance for the NIPALS algorithm.
    copy : bool, default=True
        Whether to copy X and Y before fitting.
    missing_data_settings : dict or None, default=None
        Settings for missing data algorithms (NIPALS/TSR for PLS).
        Keys: ``md_method`` (``"tsr"``, ``"scp"``, ``"nipals"``),
        ``md_tol``, ``md_max_iter``.

    Attributes (after fitting)
    --------------------------
    scores_ : pd.DataFrame of shape (n_samples, n_components)
        X-block score matrix (T). This is the primary score matrix; equivalent
        to ``x_scores`` in older versions.
    y_scores_ : pd.DataFrame of shape (n_samples, n_components)
        Y-block score matrix (U).
    x_loadings_ : pd.DataFrame of shape (n_features, n_components)
        X-block loading matrix (P).
    y_loadings_ : pd.DataFrame of shape (n_targets, n_components)
        Y-block loading matrix (C).
    x_weights_ : pd.DataFrame of shape (n_features, n_components)
        X-block weight matrix (W).
    y_weights_ : pd.DataFrame of shape (n_targets, n_components)
        Y-block weight matrix.
    direct_weights_ : pd.DataFrame of shape (n_features, n_components)
        Direct (W*) weights: ``W (P'W)^{-1}``. Used for direct projection
        ``T = X @ W*``.
    beta_coefficients_ : pd.DataFrame of shape (n_features, n_targets)
        Regression coefficients linking X directly to Y.
    predictions_ : pd.DataFrame of shape (n_samples, n_targets)
        Y predictions from the training data.
    spe_ : pd.DataFrame of shape (n_samples, n_components)
        Squared Prediction Error (stored as sqrt of row sum-of-squares).
    hotellings_t2_ : pd.DataFrame of shape (n_samples, n_components)
        Cumulative Hotelling's T² statistic.
    r2_per_component_ : pd.Series of length n_components
        Fractional R² (on Y) explained by each component.
    r2_cumulative_ : pd.Series of length n_components
        Cumulative R² (on Y) after each component.
    r2_per_variable_ : pd.DataFrame of shape (n_features, n_components)
        Per-variable cumulative R² for X after each component.
    r2y_per_variable_ : pd.DataFrame of shape (n_targets, n_components)
        Per-variable R² for Y after each component.
    rmse_ : pd.DataFrame of shape (n_targets, n_components)
        Root mean squared error of Y predictions per component.
    explained_variance_ : np.ndarray of shape (n_components,)
        Variance explained by each component in X.
    scaling_factor_for_scores_ : pd.Series of length n_components
        Standard deviation per score (sqrt of explained variance).
    has_missing_data_ : bool
        Whether the training data contained missing values.
    fitting_info_ : dict
        Timing and iteration info from the fitting algorithm.

    See Also
    --------
    PCA : Principal Component Analysis.
    MCUVScaler : Mean-center unit-variance scaler.

    References
    ----------
    Abdi, "Partial least squares regression and projection on latent structure
    regression (PLS Regression)", 2010, DOI: 10.1002/wics.51

    Examples
    --------
    >>> import pandas as pd
    >>> from process_improve.multivariate.methods import PLS, MCUVScaler
    >>> X = pd.DataFrame({"A": [1, 2, 3, 4], "B": [4, 3, 2, 1]})
    >>> Y = pd.DataFrame({"y": [2.1, 3.9, 6.2, 7.8]})
    >>> pls = PLS(n_components=1)
    >>> pls = pls.fit(MCUVScaler().fit_transform(X), MCUVScaler().fit_transform(Y))
    >>> pls.scores_.shape
    (4, 1)
    """

    def __init__(  # noqa: PLR0913
        self,
        n_components: int,
        *,
        scale: bool = True,
        max_iter: int = 1000,
        tol: float = epsqrt,
        copy: bool = True,
        # Own extra inputs, for the case when there is missing data
        missing_data_settings: dict | None = None,
    ):
        self.n_components: int = n_components
        self.scale = scale
        self.max_iter = max_iter
        self.tol = tol
        self.copy = copy
        self.missing_data_settings = missing_data_settings

    def __sklearn_tags__(self):
        """Declare sklearn capability tags (sklearn 1.6+).

        - ``input_tags.allow_nan=True`` because the NIPALS fit threads
          missing data through; ``fit`` and ``predict`` both pass
          ``ensure_all_finite="allow-nan"`` to ``validate_data``.
        - ``target_tags.multi_output=True`` because the X / Y blocks
          can both be multi-column (multi-target Y is the default
          chemometric PLS case).
        """
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.target_tags.multi_output = True
        return tags

    # ENG-17: the 13 shared convenience methods, hotellings_t2_limit,
    # ellipse_coordinates and the rename __getattr__ are inherited from
    # _LatentVariableModel. PLS keeps only its two PLS-specific plot methods and
    # supplies its own rename map.
    predictions_vs_observed_plot = _model_method(_predictions_vs_observed_plot)
    coefficient_plot = _model_method(_coefficient_plot)

    _ATTRIBUTE_RENAMES: typing.ClassVar[dict[str, str]] = {
        "x_scores": "scores_",
        "y_scores": "y_scores_",
        "x_weights": "x_weights_",
        "y_weights": "y_weights_",
        "x_loadings": "x_loadings_",
        "y_loadings": "y_loadings_",
        "direct_weights": "direct_weights_",
        "beta_coefficients": "beta_coefficients_",
        "predictions": "predictions_",
        "squared_prediction_error": "spe_",
        "hotellings_t2": "hotellings_t2_",
        "R2": "r2_per_component_",
        "R2cum": "r2_cumulative_",
        "R2X_cum": "r2_per_variable_",
        "R2Y_cum": "r2y_per_variable_",
        "RMSE": "rmse_",
        "explained_variance": "explained_variance_",
        "scaling_factor_for_scores": "scaling_factor_for_scores_",
        "extra_info": "fitting_info_",
        "has_missing_data": "has_missing_data_",
        "N": "n_samples_",
        "K": "n_features_in_",
        "M": "n_targets_",
        "A": "n_components",
    }
    _RENAME_CONTEXT: typing.ClassVar[str] = "PLS"

    # Y-side fitted attributes: ndarrays while NIPALS fills them in, then wrapped
    # into the documented public DataFrames at the end of fit().
    y_scores_: np.ndarray | pd.DataFrame
    y_weights_: np.ndarray | pd.DataFrame
    y_loadings_: np.ndarray | pd.DataFrame
    # Fitted diagnostics: per-component arrays or scalar totals.
    fitting_info_: dict[str, np.ndarray | int | float]

    # ENG-18: public DataFrame views built lazily from the private ndarrays.
    scores_ = _LazyFrame("_scores", index="_sample_index", columns="_component_names")
    spe_ = _LazyFrame("_spe", index="_sample_index", columns="_component_names")
    x_loadings_ = _LazyFrame("_x_loadings", index="_feature_names", columns="_component_names")
    x_weights_ = _LazyFrame("_x_weights", index="_feature_names", columns="_component_names")

    def _fit_nipals(self, X: DataMatrix, Y: DataMatrix, A: int, settings: dict) -> None:  # noqa: PLR0915
        """Fit PLS via the NIPALS algorithm, handling missing data transparently.

        Parameters
        ----------
        X : DataMatrix
            Training X data (N x K).
        Y : DataMatrix
            Training Y data (N x M).
        A : int
            Number of components to extract.
        settings : dict
            Algorithm settings with keys ``md_method``, ``md_tol``, ``md_max_iter``.
        """
        N = self.n_samples_
        K = self.n_features_in_
        M = self.n_targets_

        md_method = settings.get("md_method", "nipals").lower()
        if md_method == "tsr":
            raise NotImplementedError("TSR for PLS not implemented yet")
        if md_method == "pmp":
            raise NotImplementedError("PMP for PLS not implemented yet")

        Xd = np.asarray(X, dtype=float).copy()
        Yd = np.asarray(Y, dtype=float).copy()

        self._scores = np.zeros((N, A))
        self.y_scores_ = np.zeros((N, A))
        self._x_weights = np.zeros((K, A))
        self.y_weights_ = np.zeros((M, A))
        self._x_loadings = np.zeros((K, A))
        self.y_loadings_ = np.zeros((M, A))

        self.fitting_info_ = {
            "timing": np.zeros(A) * np.nan,
            "iterations": np.zeros(A) * np.nan,
        }

        for a in range(A):
            start_time = time.time()
            itern = 0

            start_SSX_col = ssq(Xd, axis=0)
            start_SSY_col = ssq(Yd, axis=0)

            if np.sum(start_SSX_col) < epsqrt:
                emsg = (
                    "There is no variance left in the data array for X: cannot "
                    f"compute any more components beyond component {a}."
                )
                raise NotEnoughVarianceError(emsg)
            if np.sum(start_SSY_col) < epsqrt:
                emsg = (
                    "There is no variance left in the data array for Y: cannot "
                    f"compute any more components beyond component {a}."
                )
                raise NotEnoughVarianceError(emsg)

            # Seed u_a from the column of Y with the greatest sum-of-squares
            # (variance, for mean-centred data) rather than the arbitrary first
            # column (#195): NIPALS converges to the same component for any
            # non-degenerate seed, but the highest-variance column needs fewer
            # iterations and is more robust. The deterministic sign convention
            # applied below makes the fitted sign independent of this seed.
            # (Replace NaN with 0 for the missing-data path.)
            start_col = int(np.argmax(start_SSY_col))
            u_a_guess = Yd[:, [start_col]].copy()
            u_a_guess[np.isnan(u_a_guess)] = 0
            u_a = u_a_guess + 1.0

            while not terminate_check(u_a_guess, u_a, iterations=itern, settings=settings):
                u_a_guess = u_a.copy()

                # 1: w_a = X'u_a / (u_a'u_a)
                w_a = quick_regress(Xd, u_a)

                # 2: Normalize w_a to unit length
                w_a = w_a / np.sqrt(ssq(w_a))

                # 3: t_a = X w_a / (w_a'w_a)
                t_a = quick_regress(Xd, w_a)

                # 4: c_a = Y't_a / (t_a't_a)
                c_a = quick_regress(Yd, t_a)

                # 5: u_a = Y c_a / (c_a'c_a)
                u_a = quick_regress(Yd, c_a)

                itern += 1

            timing_arr = typing.cast("np.ndarray", self.fitting_info_["timing"])
            iterations_arr = typing.cast("np.ndarray", self.fitting_info_["iterations"])
            timing_arr[a] = time.time() - start_time
            iterations_arr[a] = itern
            logger.debug(
                "PLS NIPALS: component %d converged in %d iterations (md_tol=%g)",
                a + 1,
                itern,
                settings["md_tol"],
            )

            if itern > settings["md_max_iter"]:
                warnings.warn(
                    "PLS NIPALS: maximum number of iterations reached!",
                    SpecificationWarning,
                    stacklevel=2,
                )

            # 6: Compute loadings and deflate
            p_a = quick_regress(Xd, t_a)
            Xd = Xd - np.dot(t_a, p_a.T)
            Yd = Yd - np.dot(t_a, c_a.T)

            # Flip signs so largest-magnitude loading element is positive
            max_el_idx = np.argmax(np.abs(p_a))
            if np.sign(p_a[max_el_idx]) < 1:
                t_a *= -1.0
                u_a *= -1.0
                w_a *= -1.0
                p_a *= -1.0
                c_a *= -1.0

            self._scores[:, a] = t_a.flatten()
            self.y_scores_[:, a] = u_a.flatten()
            self._x_weights[:, a] = w_a.flatten()
            self._x_loadings[:, a] = p_a.flatten()
            self.y_loadings_[:, a] = c_a.flatten()
            # In PLS mode A (PLSRegression), y_weights == y_loadings
            self.y_weights_[:, a] = c_a.flatten()

    def fit(self, X: DataMatrix, Y: DataMatrix) -> PLS:  # noqa: PLR0915, C901
        """
        Fit a projection to latent structures (PLS) model to the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples (rows)
            and `n_features` is the number of features (columns).
        Y : array-like, shape (n_samples, n_targets)
            Training data, where `n_samples` is the number of samples (rows)
            and `n_targets` is the number of target outputs (columns).

        Returns
        -------
        PLS
            Model object.

        References
        ----------
        Abdi, "Partial least squares regression and projection on latent structure
        regression (PLS Regression)", 2010, DOI: 10.1002/wics.51
        """
        # Accept 1-D Y (the shape sklearn Pipelines pass for single-target
        # regression) by promoting to (N, 1) so the rest of fit can rely on
        # the 2-D shape.
        if hasattr(Y, "ndim") and Y.ndim == 1:
            Y = Y.to_frame() if isinstance(Y, pd.Series) else pd.DataFrame(np.asarray(Y).reshape(-1, 1))

        # Capture DataFrame metadata before validate_data converts X to ndarray
        # so the downstream DataFrame view keeps its row/column labels.
        sample_index: pd.Index | None = X.index if isinstance(X, pd.DataFrame) else None
        feature_columns: pd.Index | None = X.columns if isinstance(X, pd.DataFrame) else None
        X_arr = validate_data(
            self,
            X,
            reset=True,
            accept_sparse=False,
            ensure_min_samples=2,
            ensure_min_features=1,
            dtype="numeric",
            ensure_all_finite="allow-nan",
        )
        if feature_columns is None:
            feature_columns = pd.RangeIndex(X_arr.shape[1])  # type: ignore[assignment]
        if sample_index is None:
            sample_index = pd.RangeIndex(X_arr.shape[0])  # type: ignore[assignment]
        X = pd.DataFrame(X_arr, index=sample_index, columns=feature_columns)
        if not isinstance(Y, pd.DataFrame):
            Y = pd.DataFrame(Y, index=sample_index)

        self.n_samples_: int = X.shape[0]
        # n_features_in_ is set by validate_data; reassert for clarity.
        self.n_features_in_: int = X.shape[1]
        # Fitted flag, defaulted here (not in __init__) so __init__ sets only the
        # constructor parameters, per sklearn convention (ENG-07). _fit_nipals
        # flips it to True if missing data is detected.
        self.has_missing_data_ = False
        Ny: int = Y.shape[0]
        self.n_targets_: int = Y.shape[1]
        if Ny != self.n_samples_:
            raise ValueError(
                f"The X and Y arrays must have the same number of rows: X has {self.n_samples_} and Y has {Ny}."
            )

        N = self.n_samples_
        K = self.n_features_in_
        M = self.n_targets_

        # The remainder of fit() uses the pandas DataFrame API (.isna / .index /
        # .columns); the NIPALS core in _fit_nipals already np.asarray-copies the
        # numeric values. Narrow the static type accordingly (the DataFrame path
        # is unchanged at runtime).
        assert isinstance(X, pd.DataFrame)
        assert isinstance(Y, pd.DataFrame)

        # Check if number of components is supported against maximum requested
        min_dim = min(N, K)
        A = min_dim if self.n_components is None else int(self.n_components)
        if min_dim < A:
            warn = (
                "The requested number of components is more than can be "
                "computed from data. The maximum number of components is "
                f"the minimum of either the number of rows ({N}) or "
                f"the number of columns ({K})."
            )
            warnings.warn(warn, SpecificationWarning, stacklevel=2)
            A = self.n_components = min_dim

        if np.any(Y.isna()) or np.any(X.isna()):
            self.has_missing_data_ = True
            # Default to the NIPALS path because TSR / PMP for PLS are still
            # NotImplementedError in _fit_nipals; NIPALS handles per-cell NaN
            # directly via skipna sums inside the NIPALS iterations.
            default_mds = dict(md_method="nipals", md_tol=epsqrt, md_max_iter=self.max_iter)
            if isinstance(self.missing_data_settings, dict):
                default_mds.update(self.missing_data_settings)
            self.missing_data_settings = default_mds

        settings = self.missing_data_settings or {
            "md_method": "nipals",
            "md_tol": self.tol,
            "md_max_iter": self.max_iter,
        }
        self._fit_nipals(X, Y, A, settings)

        # --- Common post-fit path: wrap numpy arrays into pandas ---

        # R = W(P'W)^{-1} [KxA]; useful since T = XR
        direct_weights = self._x_weights @ safe_inverse(
            self._x_loadings.T @ self._x_weights, what="(x_loadings' @ x_weights)"
        )
        # beta = RC' [KxM]: direct link from k-th X variable to m-th Y variable
        beta_coefficients = direct_weights @ self.y_loadings_.T

        component_names = list(range(1, A + 1))
        # ENG-18: scores_ / x_weights_ / x_loadings_ / spe_ are stored as private
        # ndarrays (the source of truth); their public DataFrame views are built
        # lazily by the _LazyFrame descriptors from the metadata attrs below.
        self._sample_index = X.index
        self._feature_names = X.columns
        self._component_names = component_names
        self.y_scores_ = pd.DataFrame(self.y_scores_, index=Y.index, columns=component_names)
        self.y_weights_ = pd.DataFrame(self.y_weights_, index=Y.columns, columns=component_names)
        self.y_loadings_ = pd.DataFrame(self.y_loadings_, index=Y.columns, columns=component_names)
        self.predictions_ = pd.DataFrame(self._scores @ self.y_loadings_.values.T, index=Y.index, columns=Y.columns)
        self.direct_weights_ = pd.DataFrame(direct_weights, index=X.columns, columns=component_names)
        self.beta_coefficients_ = pd.DataFrame(beta_coefficients, index=X.columns, columns=Y.columns)
        # ``max(1, N-1)`` -- see SEC-21 (#270) sub-item 6.
        self.explained_variance_ = np.diag(self._scores.T @ self._scores) / max(1, N - 1)
        self.scaling_factor_for_scores_ = pd.Series(
            np.sqrt(self.explained_variance_),
            index=component_names,
            name="Standard deviation per score",
        )
        self.hotellings_t2_ = pd.DataFrame(
            np.zeros(shape=(N, A)),
            columns=component_names,
            index=X.index.copy(),
        )
        self._spe = np.zeros((N, A))
        self.r2_per_component_ = pd.Series(
            np.zeros(shape=(A)),
            index=component_names,
            name="Output R² per component",
        )
        self.r2_cumulative_ = pd.Series(
            np.zeros(shape=(A)),
            index=component_names,
            name="Output cumulative R²",
        )
        self.r2_per_variable_ = pd.DataFrame(
            np.zeros(shape=(K, A)),
            index=X.columns.copy(),
            columns=component_names,
        )
        self.r2y_per_variable_ = pd.DataFrame(
            np.zeros(shape=(M, A)),
            index=Y.columns.copy(),
            columns=component_names,
        )
        self.rmse_ = pd.DataFrame(
            np.zeros(shape=(M, A)),
            index=Y.columns.copy(),
            columns=component_names,
        )

        Xd = X.copy()
        Yd = Y.copy()
        prior_SSX_col = ssq(Xd.values, axis=0)
        prior_SSY_col = ssq(Yd.values, axis=0)
        base_variance_Y = np.sum(prior_SSY_col)
        for a in range(A):
            self.hotellings_t2_.iloc[:, a] = (
                self.hotellings_t2_.iloc[:, max(0, a - 1)]
                + (self.scores_.iloc[:, a] / self.scaling_factor_for_scores_.iloc[a]) ** 2
            )
            Xd = Xd - self.scores_.iloc[:, [a]] @ self.x_loadings_.iloc[:, [a]].T
            y_hat = self.scores_.iloc[:, 0 : (a + 1)] @ self.y_loadings_.iloc[:, 0 : (a + 1)].T
            row_SSX = ssq(Xd.values, axis=1)
            col_SSX = ssq(Xd.values, axis=0)
            row_SSY = ssq(y_hat.values, axis=1)
            col_SSY = ssq(y_hat.values, axis=0)
            self.r2_cumulative_.iloc[a] = np.sum(row_SSY) / base_variance_Y
            if a > 0:
                self.r2_per_component_.iloc[a] = self.r2_cumulative_.iloc[a] - self.r2_cumulative_.iloc[a - 1]
            else:
                self.r2_per_component_.iloc[a] = self.r2_cumulative_.iloc[a]

            self._spe[:, a] = np.sqrt(row_SSX)

            # Per-variable R^2 is undefined for a column with no variance to
            # explain; emit NaN there. SEC-21 (#270) sub-item 4.
            self.r2_per_variable_.iloc[:, a] = np.where(
                prior_SSX_col > 0, 1 - col_SSX / np.where(prior_SSX_col > 0, prior_SSX_col, 1.0), np.nan
            )
            self.r2y_per_variable_.iloc[:, a] = np.where(
                prior_SSY_col > 0, col_SSY / np.where(prior_SSY_col > 0, prior_SSY_col, 1.0), np.nan
            )
            residuals_y = Yd.to_numpy() - y_hat.to_numpy()
            self.rmse_.iloc[:, a] = np.sqrt(np.mean(residuals_y**2, axis=0))

        return self

    def transform(self, X: DataMatrix, Y: DataMatrix | None = None) -> pd.DataFrame:  # noqa: ARG002
        """Project X (and optionally Y) into the latent space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.
        Y : array-like of shape (n_samples, n_targets), optional
            Ignored. Present for API compatibility with sklearn pipelines.

        Returns
        -------
        X_scores : pd.DataFrame of shape (n_samples, n_components)
            Projected X data (scores).
        """
        check_is_fitted(self, "direct_weights_")
        # Realign reordered columns before validate_data (sklearn checks
        # feature-name *order*, _align_to_fit_features only needs set equality).
        if isinstance(X, pd.DataFrame):
            X = _align_to_fit_features(X, self._feature_names)
        sample_index: pd.Index | None = X.index if isinstance(X, pd.DataFrame) else None
        feature_columns: pd.Index | None = X.columns if isinstance(X, pd.DataFrame) else None
        X_arr = validate_data(
            self,
            X,
            reset=False,
            accept_sparse=False,
            dtype="numeric",
            ensure_all_finite="allow-nan",
        )
        if feature_columns is None:
            feature_columns = self._feature_names
        if sample_index is None:
            sample_index = pd.RangeIndex(X_arr.shape[0])  # type: ignore[assignment]
        X_df = pd.DataFrame(X_arr, index=sample_index, columns=feature_columns)
        return X_df @ self.direct_weights_

    def fit_transform(self, X: DataMatrix, Y: DataMatrix | None = None) -> pd.DataFrame:
        """Fit the model and return X scores.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        Y : array-like of shape (n_samples, n_targets)

        Returns
        -------
        X_scores : pd.DataFrame of shape (n_samples, n_components)
        """
        assert Y is not None, "PLS requires Y to be supplied to fit_transform."
        self.fit(X, Y)
        return self.scores_

    def score(self, X: DataMatrix, Y: DataMatrix, sample_weight: np.ndarray | None = None) -> float:
        """Return the R² score for the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        Y : array-like of shape (n_samples, n_targets)
            True target values.
        sample_weight : array-like of shape (n_samples,), optional

        Returns
        -------
        score : float
            R² of ``self.predict(X)`` w.r.t. *Y*.
        """
        y_pred = self.predict(X)
        return float(r2_score(Y, y_pred, sample_weight=sample_weight))

    def predict(self, X: DataMatrix) -> pd.DataFrame:
        """Predict Y for new observations.

        Returns just the predicted ``y_hat`` so the call satisfies the
        scikit-learn :class:`~sklearn.base.RegressorMixin` contract (and
        therefore composes inside :class:`~sklearn.pipeline.Pipeline`,
        :func:`~sklearn.model_selection.cross_val_score`, and
        :class:`~sklearn.model_selection.GridSearchCV`). For the rich
        diagnostic view (scores, Hotelling's T², SPE, plus ``y_hat``),
        see :meth:`diagnose`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_hat : pd.DataFrame of shape (n_samples, n_targets)
            Predicted target values, indexed by ``X``'s rows and labelled
            with the target column names captured during ``fit``.

        See Also
        --------
        diagnose : richer per-prediction diagnostics.

        Examples
        --------
        >>> y_pred = pls.predict(X_new)
        >>> diag = pls.diagnose(X_new)   # for scores / T² / SPE
        """
        return self.diagnose(X).y_hat

    def diagnose(self, X: DataMatrix) -> Bunch:
        """Project new data and compute predictions plus diagnostics.

        This is the rich view that :meth:`predict` used to return before
        1.35.0: alongside ``y_hat`` it reports the X scores, cumulative
        Hotelling's T², and SPE for every row of ``X`` so the user can
        flag out-of-model observations *and* read their predicted Y from
        one call.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        result : sklearn.utils.Bunch
            With keys ``scores``, ``hotellings_t2``, ``spe``, ``y_hat``.

        See Also
        --------
        predict : sklearn-compatible call returning just ``y_hat``.

        Examples
        --------
        >>> result = pls.diagnose(scaler_x.transform(X_new))
        >>> result.y_hat           # Predicted Y values
        >>> result.spe             # SPE for each new observation
        >>> result.hotellings_t2   # T² for each new observation
        """
        check_is_fitted(self, "scores_")
        # Realign reordered DataFrame columns before validate_data.
        if isinstance(X, pd.DataFrame):
            X = _align_to_fit_features(X, self._feature_names)
        sample_index: pd.Index | None = X.index if isinstance(X, pd.DataFrame) else None
        feature_columns: pd.Index | None = X.columns if isinstance(X, pd.DataFrame) else None
        X_arr = validate_data(
            self,
            X,
            reset=False,
            accept_sparse=False,
            dtype="numeric",
            ensure_all_finite="allow-nan",
        )
        if feature_columns is None:
            feature_columns = self._feature_names
        if sample_index is None:
            sample_index = pd.RangeIndex(X_arr.shape[0])  # type: ignore[assignment]
        X = pd.DataFrame(X_arr, index=sample_index, columns=feature_columns)

        scores = X @ self.direct_weights_

        # Hotelling's T² (cumulative over all components)
        t2_values = np.sum(np.power((scores / self.scaling_factor_for_scores_.to_numpy()), 2), axis=1)
        t2 = pd.Series(t2_values, index=X.index, name="Hotelling's T²")

        # SPE: residual after X reconstruction
        X_hat = scores @ self._x_loadings.T
        residuals = X - X_hat
        spe_values = pd.Series(np.sqrt(np.power(residuals, 2).sum(axis=1)), index=X.index, name="SPE")

        # Y predictions
        y_hat = scores @ self.y_loadings_.T

        return Bunch(scores=scores, hotellings_t2=t2, spe=spe_values, y_hat=y_hat)

    @classmethod
    def select_n_components(  # noqa: C901, PLR0912, PLR0913, PLR0915
        cls,
        X: DataMatrix,
        Y: DataMatrix,
        *,
        max_components: int | None = None,
        cv: int | BaseCrossValidator = 5,
        n_repeats: int | None = None,
        random_state: int | None = None,
        selection_rule: SelectionRule = "1se",
        scale_inside_folds: bool = True,
        min_q2_increase: float = Q2_MIN_INCREMENT,
        n_permutations: int = 999,
        alpha: float = 0.01,
        stability_threshold: float = 0.6,
        **pls_kwargs,
    ) -> Bunch:
        """Select the number of PLS components via cross-validation.

        Fits PLS models on cross-validation training folds and evaluates the
        out-of-fold prediction error for every component count
        ``1, 2, ..., max_components``. Reports per-fold and pooled RMSECV plus
        the validated cumulative R² curves, and recommends a component count
        from one of three rules (see ``selection_rule`` below).

        The defaults are the research-backed combination: the
        one-standard-error rule on top of repeated, shuffled K-fold CV, with
        :class:`MCUVScaler` re-fit inside every training fold so test data
        never leaks into the centring/scaling estimates.

        Unlike the calibration statistics stored on a fitted model
        (``rmse_``, ``r2_cumulative_``), the metrics returned here estimate
        performance on unseen data and are therefore suitable for choosing
        ``n_components``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training X. With the default ``scale_inside_folds=True`` the raw,
            unscaled X may be passed; scaling is fit inside every training
            fold.
        Y : array-like of shape (n_samples, n_targets)
            Training Y. Same treatment as ``X`` under ``scale_inside_folds``.
        max_components : int, optional
            Maximum number of components to evaluate. Default is the largest
            value supported by every cross-validation training fold,
            ``min(min_fold_size, n_features)``.
        cv : int or sklearn CV splitter, default 5
            If an integer, used as the ``n_splits`` of a shuffled
            :class:`~sklearn.model_selection.KFold` (or
            :class:`~sklearn.model_selection.RepeatedKFold` when
            ``n_repeats > 1``). Any sklearn splitter object (e.g.
            ``KFold(10, shuffle=True)`` or ``LeaveOneOut()``) is also accepted
            and is used as-is (``n_repeats`` is then ignored).
        n_repeats : int, optional
            Number of times the K-fold split is repeated with a fresh shuffle,
            used only when ``cv`` is an integer. Default ``10`` (giving a
            ``cv * 10`` per-fold sample for the 1-SE rule); pass ``1`` to
            disable repeats. Repeated K-fold's standard errors are slightly
            optimistic because test folds overlap across repeats; that is fine
            for the 1-SE *selection* rule but should not be reported as an
            unbiased generalisation variance.
        random_state : int, optional
            Seed forwarded to ``KFold`` / ``RepeatedKFold`` for reproducible
            shuffling. Ignored when ``cv`` is a pre-built splitter.
        selection_rule : {"1se", "min", "q2_increment", "randomization"}, default "1se"
            How the recommended component count is chosen.
            See :data:`~process_improve.multivariate._common.SelectionRule`
            for the rule semantics. ``"1se"`` is the default; ``"min"`` is the
            argmin RMSECV (the pre-1.28 default, prone to running to the
            maximum component count); ``"q2_increment"`` is the Wold's-R-style
            cumulative-Q² threshold; ``"randomization"`` is Van der Voet's
            (1994) permutation test (uses ``n_permutations`` and ``alpha``)
            that picks the smallest model whose predictive ability is
            statistically indistinguishable from the reference (argmin
            RMSECV) one.
        scale_inside_folds : bool, default True
            When True (the default), fit a fresh :class:`MCUVScaler` on each
            training fold's X and Y, apply it to the held-out rows, fit PLS in
            scaled space, then inverse-transform the predictions so RMSECV is
            reported on the original Y scale. This removes the centring /
            scaling leakage of the prior default. Set to False to keep the
            pre-1.28 behaviour, in which case ``X`` and ``Y`` should already
            be scaled; a :class:`SpecificationWarning` is emitted.
        min_q2_increase : float, default 0.01
            Threshold used only when ``selection_rule="q2_increment"``: the
            smallest increase in cumulative validated :math:`Q^2_Y` that
            justifies keeping an extra component.
        n_permutations : int, default 999
            Used only when ``selection_rule="randomization"``: number of
            sign-flip permutations driving the Van der Voet test.
        alpha : float, default 0.01
            Used only when ``selection_rule="randomization"``: significance
            level. The smallest component count whose Van der Voet *p*-value
            exceeds ``alpha`` is recommended. R's ``pls::selectNcomp`` uses
            the same default; smaller values pick more parsimonious models.
        stability_threshold : float, default 0.6
            For the per-repeat stability-selection diagnostic (``"1se"`` /
            ``"min"`` rules with ``n_repeats > 1`` only): the recommendation
            is judged ``selection_is_stable=True`` iff the modal vote share
            in ``selection_distribution`` is at least this fraction.
            Meinshausen & Bühlmann (2010, *JRSS-B*) suggest 0.6-0.9 for
            their variable-selection analogue; we default to the
            permissive end.
        **pls_kwargs
            Additional keyword arguments passed to the ``PLS()`` constructor
            (e.g. ``missing_data_settings``).

        Returns
        -------
        result : sklearn.utils.Bunch
            With keys:

            - ``n_components`` - recommended number of components (int).
            - ``rmsecv`` - pooled RMSECV per component count
              (pd.DataFrame, indexed ``1..A``; columns are the Y-variable
              names plus ``"total"``).
            - ``per_fold_rmsecv`` - per-fold total RMSECV (pd.DataFrame,
              indexed ``1..A``; one column per fold across all repeats).
              Drives the 1-SE rule.
            - ``se_rmsecv`` - standard error of the per-fold RMSECV per
              component count (pd.Series, indexed ``1..A``).
            - ``r2y_validated`` - validated cumulative :math:`R^2_Y`
              (pd.DataFrame, same shape as ``rmsecv``).
            - ``r2x_validated`` - validated cumulative :math:`R^2_X`
              (pd.DataFrame, indexed ``1..A``; columns are the X-variable
              names plus ``"total"``).
            - ``press`` - pooled Y prediction error sum of squares per
              component count (pd.Series, indexed ``1..A``).
            - ``cv_predictions`` - out-of-fold predictions of Y at the
              recommended component count, on the original Y scale
              (pd.DataFrame). For repeated K-fold, the *first* repeat's
              held-out predictions are reported so each row appears exactly
              once.
            - ``selection_rule`` - the rule used to pick ``n_components``.
            - ``randomization_pvalues`` - per-component Van der Voet
              right-tail *p*-values when ``selection_rule="randomization"``;
              ``None`` otherwise.
            - ``selection_distribution`` - per-repeat *vote share* over
              candidate component counts (pd.Series indexed ``1..A``).
              Populated only for ``selection_rule in {"1se", "min"}`` and
              ``n_repeats > 1``; ``None`` otherwise. A concentrated
              distribution signals a confident recommendation; a flat or
              multi-modal one flags it for review.
            - ``selection_mode`` - the most-voted component count, or
              ``None`` when ``selection_distribution`` is.
            - ``selection_is_stable`` - ``True`` iff the modal vote share
              meets ``stability_threshold``; ``None`` when no distribution
              was computed.

        Notes
        -----
        The pooled RMSECV in ``rmsecv["total"]`` is the square root of the
        total PRESS over all fold-test rows divided by ``(N_eff * M)`` where
        ``N_eff = N * n_repeats`` under repeated CV; the ``per_fold_rmsecv``
        column for fold *f* is the square root of fold-*f*'s sum-of-squared
        residuals over its own test rows.

        References
        ----------
        Breiman, Friedman, Olshen & Stone (1984), *CART*, sec.3.4.3 (1-SE rule).
        Hastie, Tibshirani & Friedman, *ESL*, sec.7.10. Kohavi (1995, IJCAI)
        recommends 10-fold stratified CV for model selection.

        Examples
        --------
        >>> from sklearn.model_selection import KFold
        >>> # Default: 1-SE on 10 x 5-fold repeated CV with in-fold scaling.
        >>> result = PLS.select_n_components(X, Y, max_components=6, random_state=0)
        >>> result.n_components, result.selection_rule
        >>> # Opt-in to the older argmin-RMSECV rule:
        >>> PLS.select_n_components(X, Y, max_components=6, selection_rule="min")
        >>> # Caller-supplied splitter (n_repeats is ignored here):
        >>> PLS.select_n_components(X, Y, cv=KFold(10, shuffle=True, random_state=0))
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if isinstance(Y, pd.Series):
            Y = Y.to_frame()
        elif not isinstance(Y, pd.DataFrame):
            Y = pd.DataFrame(Y)

        if not scale_inside_folds:
            warnings.warn(
                "scale_inside_folds=False leaks centring/scaling estimated on the "
                "full dataset into every CV fold, making the reported RMSECV "
                "optimistic. The default scale_inside_folds=True is preferred.",
                SpecificationWarning,
                stacklevel=2,
            )

        N, K = X.shape
        M = Y.shape[1]

        if isinstance(cv, int):
            if cv < 2:
                raise ValueError(f"cv must be >= 2 when given as an int; got {cv}.")
            repeats = 10 if n_repeats is None else int(n_repeats)
            if repeats < 1:
                raise ValueError(f"n_repeats must be >= 1; got {repeats}.")
            splitter: BaseCrossValidator
            if repeats == 1:
                splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
            else:
                splitter = RepeatedKFold(n_splits=cv, n_repeats=repeats, random_state=random_state)
            splits = list(splitter.split(X, Y))
            first_repeat_fold_count = cv
        else:
            splits = list(check_cv(cv).split(X, Y))
            first_repeat_fold_count = len(splits)
        if not splits:
            raise ValueError("The cross-validation splitter produced no folds.")
        n_folds_total = len(splits)
        min_train_size = min(len(train_idx) for train_idx, _ in splits)

        # Centring inside a fold removes one DoF, so a globally-centred matrix
        # restricted to (min_train_size) rows and re-centred has rank at most
        # min_train_size - 1. Cap A accordingly when in-fold scaling is on.
        upper = min(min_train_size - (1 if scale_inside_folds else 0), K)
        if max_components is None:
            max_components = upper
        A = min(int(max_components), upper)
        if A < 1:
            raise ValueError("No components can be evaluated; the data or folds are too small.")

        component_index = pd.Index(range(1, A + 1), name="n_components")

        press_y = np.zeros((A, M))
        press_x = np.zeros((A, K))
        # Per-fold total-RMSECV across every fold-fit (n_folds_total columns).
        # Drives the 1-SE rule's standard error.
        per_fold_rmse = np.full((A, n_folds_total), np.nan)
        # Out-of-fold predictions: with repeated K-fold each row appears in
        # multiple test folds, so populate only from the *first* repeat (which
        # covers every row exactly once) to preserve the existing semantic that
        # cv_predictions has one row per observation.
        oof = np.full((A, N, M), np.nan)

        x_columns = list(X.columns)
        y_columns = list(Y.columns)
        x_values = X.to_numpy()
        y_values = Y.to_numpy()
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_train_raw = X.iloc[train_idx]
            Y_train_raw = Y.iloc[train_idx]
            X_test_raw = X.iloc[test_idx]
            if scale_inside_folds:
                scaler_x = MCUVScaler().fit(X_train_raw)
                scaler_y = MCUVScaler().fit(Y_train_raw)
                X_train = scaler_x.transform(X_train_raw)
                Y_train = scaler_y.transform(Y_train_raw)
                X_test_scaled = scaler_x.transform(X_test_raw).to_numpy()
                y_centre = scaler_y.center_.to_numpy()
                y_scale = scaler_y.scale_.to_numpy()
                x_centre = scaler_x.center_.to_numpy()
                x_scale = scaler_x.scale_.to_numpy()
            else:
                X_train = X_train_raw
                Y_train = Y_train_raw
                X_test_scaled = X_test_raw.to_numpy()
                y_centre = np.zeros(M)
                y_scale = np.ones(M)
                x_centre = np.zeros(K)
                x_scale = np.ones(K)

            model = cls(n_components=A, **pls_kwargs).fit(X_train, Y_train)
            scores_test = X_test_scaled @ model.direct_weights_.to_numpy()
            # ``y_values`` and ``x_values`` are on the ORIGINAL (un-scaled)
            # input data: RMSECV / r2y_validated / r2x_validated are all
            # reported on the input scale.
            y_test = y_values[test_idx]
            x_test = x_values[test_idx]
            n_test = len(test_idx)
            y_loadings = typing.cast("pd.DataFrame", model.y_loadings_).to_numpy()  # shape (M, A)
            x_loadings = model.x_loadings_.to_numpy()  # shape (K, A)
            for a in range(1, A + 1):
                y_hat_scaled = scores_test[:, :a] @ y_loadings[:, :a].T
                x_hat_scaled = scores_test[:, :a] @ x_loadings[:, :a].T
                # inverse-transform: x_scale[None, :] broadcasts (n_test, K).
                y_hat = y_hat_scaled * y_scale + y_centre
                x_hat = x_hat_scaled * x_scale + x_centre
                residuals_y = y_test - y_hat
                press_y[a - 1] += np.nansum(residuals_y ** 2, axis=0)
                press_x[a - 1] += np.nansum((x_test - x_hat) ** 2, axis=0)
                per_fold_rmse[a - 1, fold_idx] = np.sqrt(
                    np.nansum(residuals_y ** 2) / max(1, n_test * M)
                )
                if fold_idx < first_repeat_fold_count:
                    oof[a - 1, test_idx, :] = y_hat

        # With repeated CV the total PRESS sums residuals across n_repeats
        # passes over every observation; divide by N_eff = N * n_repeats to
        # keep the RMSECV scale comparable to a single-pass CV.
        n_eff = N * max(1, n_folds_total // max(1, first_repeat_fold_count))

        tss_y = np.nansum((y_values - np.nanmean(y_values, axis=0)) ** 2, axis=0)
        tss_x = np.nansum((x_values - np.nanmean(x_values, axis=0)) ** 2, axis=0)

        rmsecv = pd.DataFrame(
            np.column_stack([np.sqrt(press_y / n_eff), np.sqrt(press_y.sum(axis=1) / (n_eff * M))]),
            index=component_index,
            columns=[*y_columns, "total"],
        )

        def _validated_r2(press: np.ndarray, tss: np.ndarray) -> np.ndarray:
            per_var = np.where(tss > 0, 1.0 - press / (tss * max(1, n_folds_total // first_repeat_fold_count)), np.nan)
            total = np.where(
                tss.sum() > 0,
                1.0 - press.sum(axis=1) / (tss.sum() * max(1, n_folds_total // first_repeat_fold_count)),
                np.nan,
            )
            return np.column_stack([per_var, total])

        r2y_validated = pd.DataFrame(
            _validated_r2(press_y, tss_y),
            index=component_index,
            columns=[*y_columns, "total"],
        )
        r2x_validated = pd.DataFrame(
            _validated_r2(press_x, tss_x),
            index=component_index,
            columns=[*x_columns, "total"],
        )
        press = pd.Series(press_y.sum(axis=1), index=component_index, name="PRESS")

        per_fold_rmsecv = pd.DataFrame(
            per_fold_rmse,
            index=component_index,
            columns=[f"fold_{i + 1}" for i in range(n_folds_total)],
        )
        # Standard error across folds (and repeats). ``ddof=1`` for the
        # sample SE; with a single fold (n_folds_total == 1) the SE is NaN
        # and the 1-SE rule degenerates to argmin gracefully.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            se_values = np.nanstd(per_fold_rmse, axis=1, ddof=1) / np.sqrt(
                np.maximum(1, np.sum(~np.isnan(per_fold_rmse), axis=1))
            )
        se_rmsecv = pd.Series(se_values, index=component_index, name="SE(RMSECV)")

        # If every CV fold produced NaN (e.g. zero-variance Y per fold) the
        # cross-validation never converged to anything we can judge. Raise so
        # the caller knows, rather than silently returning ``1``.
        # SEC-21 (#270) sub-item 8.
        total_rmsecv = rmsecv["total"].to_numpy()
        if np.all(np.isnan(total_rmsecv)):
            raise RuntimeError(
                "Cross-validation produced NaN total-RMSECV for every component count; "
                "no recommendation can be made. Likely cause: a per-fold zero-variance Y "
                "column, or every fold trivially degenerate."
            )
        # Per-observation squared total residual at every component count.
        # Drives Van der Voet's randomization test; cheap to compute and
        # otherwise diagnostic (rows of oof[a] that are NaN - typically
        # because some custom splitter never held them out - are skipped via
        # nansum).
        per_obs_sse = np.nansum((y_values[None, :, :] - oof) ** 2, axis=2)  # (A, N)

        randomization_pvalues: pd.Series | None = None
        if selection_rule == "randomization":
            recommended, p_values = _vandervoet_randomization(
                per_obs_sse,
                total_rmsecv=total_rmsecv,
                n_permutations=n_permutations,
                alpha=alpha,
                random_state=random_state,
            )
            randomization_pvalues = pd.Series(
                p_values, index=component_index, name="p-value (Van der Voet)"
            )
        else:
            recommended = _select_n_components(
                selection_rule,
                mean_error=total_rmsecv,
                se_error=se_values,
                q2_cumulative=r2y_validated["total"].to_numpy(),
                min_q2_increase=min_q2_increase,
            )
        cv_predictions = pd.DataFrame(oof[recommended - 1], index=Y.index, columns=Y.columns)

        # Stability selection: re-apply the chosen rule per repeat and
        # tabulate how often each component count wins. A multi-modal or
        # flat distribution flags the recommendation as low-confidence.
        # Only meaningful when the rule operates on the per-fold RMSE
        # curve (1se/min) and we ran more than one repeat; q2_increment
        # and randomization don't decompose per repeat without more
        # bookkeeping than they're worth at this stage.
        n_repeats_effective = max(1, n_folds_total // max(1, first_repeat_fold_count))
        selection_distribution: pd.Series | None = None
        selection_mode: int | None = None
        selection_is_stable: bool | None = None
        if (
            selection_rule in ("1se", "min")
            and n_repeats_effective > 1
            and not np.all(np.isnan(per_fold_rmse))
        ):
            votes: list[int] = []
            for r in range(n_repeats_effective):
                cols = slice(r * first_repeat_fold_count, (r + 1) * first_repeat_fold_count)
                fold_subset = per_fold_rmse[:, cols]
                if np.all(np.isnan(fold_subset)):
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    mean_r = np.nanmean(fold_subset, axis=1)
                    se_r = np.nanstd(fold_subset, axis=1, ddof=1) / np.sqrt(
                        np.maximum(1, np.sum(~np.isnan(fold_subset), axis=1))
                    )
                # The dispatcher needs q2_cumulative even for non-q2 rules;
                # pass a dummy zeros array since 1se/min don't read it.
                pick = _select_n_components(
                    selection_rule,
                    mean_error=mean_r,
                    se_error=se_r,
                    q2_cumulative=np.zeros_like(mean_r),
                    min_q2_increase=min_q2_increase,
                )
                votes.append(int(pick))
            if votes:
                counts = pd.Series(votes).value_counts().sort_index()
                dist = (counts / counts.sum()).reindex(component_index, fill_value=0.0)
                dist.name = "vote_share"
                dist.index.name = "n_components"
                selection_distribution = dist
                selection_mode = int(dist.idxmax())
                selection_is_stable = bool(dist.max() >= stability_threshold)

        return Bunch(
            n_components=recommended,
            rmsecv=rmsecv,
            per_fold_rmsecv=per_fold_rmsecv,
            se_rmsecv=se_rmsecv,
            r2y_validated=r2y_validated,
            r2x_validated=r2x_validated,
            press=press,
            cv_predictions=cv_predictions,
            selection_rule=selection_rule,
            randomization_pvalues=randomization_pvalues,
            selection_distribution=selection_distribution,
            selection_mode=selection_mode,
            selection_is_stable=selection_is_stable,
        )

    @classmethod
    def nested_cv(  # noqa: PLR0913, PLR0915
        cls,
        X: DataMatrix,
        Y: DataMatrix,
        *,
        max_components: int | None = None,
        outer_cv: int | BaseCrossValidator = 5,
        inner_cv: int = 5,
        n_inner_repeats: int = 10,
        selection_rule: SelectionRule = "1se",
        scale_inside_folds: bool = True,
        min_q2_increase: float = Q2_MIN_INCREMENT,
        n_permutations: int = 999,
        alpha: float = 0.01,
        random_state: int | None = None,
        **pls_kwargs,
    ) -> Bunch:
        """Nested cross-validation for an honest PLS performance estimate.

        Outer loop splits the data into outer-train / outer-test; the inner
        loop runs :meth:`select_n_components` on the outer-train (with the
        configured ``selection_rule`` over ``inner_cv * n_inner_repeats``
        folds) to pick the component count; a final PLS is fit on the
        outer-train at that count and used to predict the outer-test. The
        accumulated out-of-fold predictions give RMSEP that is *not*
        optimism-biased by the selection decision - the headline number to
        report when a clean test set is not available.

        Parameters
        ----------
        X, Y : array-like
            Training data. Treated as in :meth:`select_n_components` (raw
            if ``scale_inside_folds=True``, pre-scaled otherwise).
        max_components : int, optional
            Forwarded to the inner :meth:`select_n_components`.
        outer_cv : int or sklearn splitter, default 5
            Number of outer folds (or a custom splitter).
        inner_cv : int, default 5
            Number of inner folds passed to :meth:`select_n_components`.
        n_inner_repeats : int, default 10
            Number of inner-CV repeats per outer fold; the inner
            ``random_state`` is offset by the outer-fold index so each
            outer fold sees a fresh inner shuffle.
        selection_rule : str, default "1se"
            Selection rule applied inside the inner loop. See
            :data:`~process_improve.multivariate._common.SelectionRule`.
        scale_inside_folds : bool, default True
            Mirrors :meth:`select_n_components`. Also applied to the final
            outer-train fit, with the test-fold predictions inverse-
            transformed to the original Y scale before RMSEP accumulates.
        min_q2_increase, n_permutations, alpha
            Forwarded to the inner :meth:`select_n_components` per rule.
        random_state : int, optional
            Seed for the outer-fold shuffle and the inner CV. The inner
            seed is offset per outer fold so each outer split sees a fresh
            shuffled inner CV.
        **pls_kwargs
            Forwarded to :class:`PLS` for both the inner CV and the final
            outer-train fits.

        Returns
        -------
        result : sklearn.utils.Bunch
            With keys:

            - ``rmsep`` - honest held-out RMSEP per Y column plus a
              ``"total"`` entry (pd.Series).
            - ``q2y`` - validated :math:`Q^2_Y` per Y column plus
              ``"total"`` (pd.Series).
            - ``cv_predictions`` - out-of-fold predictions of Y at the
              per-outer-fold selected component counts (pd.DataFrame on
              the original Y scale).
            - ``selected_components_per_fold`` - list of inner
              recommendations, one per outer fold.
            - ``selected_components_distribution`` - vote share over
              candidate counts (pd.Series).

        Notes
        -----
        Runtime is roughly ``outer_cv * inner_cv * n_inner_repeats *
        max_components`` PLS fits. With the defaults that is 5 * 5 * 10 *
        max_components fits per call; for ``max_components=10`` and a
        moderate dataset that completes in seconds. Drop ``n_inner_repeats``
        if you need to bring it down further.

        Examples
        --------
        >>> from process_improve.multivariate import PLS
        >>> result = PLS.nested_cv(X, Y, max_components=8, random_state=0)
        >>> result.rmsep["total"]
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if isinstance(Y, pd.Series):
            Y = Y.to_frame()
        elif not isinstance(Y, pd.DataFrame):
            Y = pd.DataFrame(Y)

        N = X.shape[0]
        M = Y.shape[1]

        if isinstance(outer_cv, int):
            if outer_cv < 2:
                raise ValueError(f"outer_cv must be >= 2 when given as an int; got {outer_cv}.")
            outer_splitter: BaseCrossValidator = KFold(
                n_splits=outer_cv, shuffle=True, random_state=random_state
            )
        else:
            outer_splitter = outer_cv
        outer_splits = list(outer_splitter.split(X, Y))
        if not outer_splits:
            raise ValueError("The outer cross-validation splitter produced no folds.")

        y_columns = list(Y.columns)
        y_values = Y.to_numpy()
        oof_predictions = np.full((N, M), np.nan)
        selected_components_per_fold: list[int] = []

        for outer_idx, (train_idx, test_idx) in enumerate(outer_splits):
            X_outer_train = X.iloc[train_idx]
            Y_outer_train = Y.iloc[train_idx]
            X_outer_test = X.iloc[test_idx]

            inner_seed = None if random_state is None else int(random_state) + outer_idx
            inner_result = cls.select_n_components(
                X_outer_train,
                Y_outer_train,
                max_components=max_components,
                cv=inner_cv,
                n_repeats=n_inner_repeats,
                selection_rule=selection_rule,
                scale_inside_folds=scale_inside_folds,
                min_q2_increase=min_q2_increase,
                n_permutations=n_permutations,
                alpha=alpha,
                random_state=inner_seed,
                **pls_kwargs,
            )
            n_comp_inner = int(inner_result.n_components)
            selected_components_per_fold.append(n_comp_inner)

            # Final outer-train fit at the inner-selected component count.
            if scale_inside_folds:
                scaler_x = MCUVScaler().fit(X_outer_train)
                scaler_y = MCUVScaler().fit(Y_outer_train)
                X_train_s = scaler_x.transform(X_outer_train)
                Y_train_s = scaler_y.transform(Y_outer_train)
                X_test_s = scaler_x.transform(X_outer_test).to_numpy()
                y_centre = scaler_y.center_.to_numpy()
                y_scale = scaler_y.scale_.to_numpy()
            else:
                X_train_s = X_outer_train
                Y_train_s = Y_outer_train
                X_test_s = X_outer_test.to_numpy()
                y_centre = np.zeros(M)
                y_scale = np.ones(M)

            model = cls(n_components=n_comp_inner, **pls_kwargs).fit(X_train_s, Y_train_s)
            scores_test = X_test_s @ model.direct_weights_.to_numpy()
            y_loadings = typing.cast("pd.DataFrame", model.y_loadings_).to_numpy()
            y_hat_scaled = scores_test @ y_loadings.T
            oof_predictions[test_idx, :] = y_hat_scaled * y_scale + y_centre

        # RMSEP from the out-of-fold predictions (each row is predicted by
        # exactly one outer fold).
        residuals = y_values - oof_predictions
        mask = ~np.isnan(oof_predictions).any(axis=1)
        n_valid = int(mask.sum())
        if n_valid == 0:
            raise RuntimeError(
                "Nested CV produced no covered observations; check the outer splitter."
            )
        per_y_press = np.nansum(residuals[mask] ** 2, axis=0)
        rmsep_per_y = np.sqrt(per_y_press / n_valid)
        rmsep_total = np.sqrt(np.nansum(residuals[mask] ** 2) / (n_valid * M))
        rmsep = pd.Series(
            np.concatenate([rmsep_per_y, [rmsep_total]]),
            index=[*y_columns, "total"],
            name="RMSEP",
        )

        # Validated Q^2_Y per column and total, against the column mean.
        col_means = np.nanmean(y_values, axis=0)
        tss_per_y = np.nansum((y_values - col_means) ** 2, axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            q2y_per_y = np.where(tss_per_y > 0, 1.0 - per_y_press / tss_per_y, np.nan)
            q2y_total = (
                1.0 - residuals[mask].astype(float).__pow__(2).sum() / tss_per_y.sum()
                if tss_per_y.sum() > 0
                else np.nan
            )
        q2y = pd.Series(
            np.concatenate([q2y_per_y, [q2y_total]]),
            index=[*y_columns, "total"],
            name="Q2Y",
        )

        cv_predictions = pd.DataFrame(oof_predictions, index=Y.index, columns=Y.columns)
        counts = pd.Series(selected_components_per_fold).value_counts().sort_index()
        distribution = counts / counts.sum()
        distribution.name = "vote_share"
        distribution.index.name = "n_components"

        return Bunch(
            rmsep=rmsep,
            q2y=q2y,
            cv_predictions=cv_predictions,
            selected_components_per_fold=selected_components_per_fold,
            selected_components_distribution=distribution,
        )

    def score_contributions(
        self,
        t_start: np.ndarray | pd.Series,
        t_end: np.ndarray | pd.Series | None = None,
        components: list[int] | None = None,
        *,
        weighted: bool = False,
    ) -> pd.Series:
        """Contribution of each X variable to a score-space movement.

        Identical to ``PCA.score_contributions`` but uses the PLS X-loadings
        (P matrix) for back-projection.

        Parameters
        ----------
        t_start : array-like of shape (n_components,)
            Score vector of the observation of interest.
        t_end : array-like of shape (n_components,), optional
            Reference point in score space. Default is the model center (zeros).
        components : list of int, optional
            **1-based** component indices. Default: all components.
        weighted : bool, default False
            If True, scale by 1/sqrt(explained_variance) for T² contributions.

        Returns
        -------
        contributions : pd.Series of shape (n_features,)

        Examples
        --------
        >>> pls = PLS(n_components=3).fit(X_scaled, Y_scaled)
        >>> contrib = pls.score_contributions(pls.scores_.iloc[0].values)
        >>> contrib.abs().sort_values(ascending=False).head()  # top contributors

        See Also
        --------
        observation_contributions : The per-observation counterpart. Despite
            the similar name, the two answer different questions and are not
            interchangeable. ``score_contributions`` is *per-variable*: it
            decomposes a single observation's movement in score space back
            onto the original X variables, answering "which **variables**
            explain why this observation sits where it does?". It returns one
            signed value per variable. ``observation_contributions`` is
            *per-observation*: it reports each observation's share of a
            component's total inertia, answering "which **observations** most
            strongly shape this component?". It returns a non-negative
            sample-by-component table whose columns each sum to 1. The two are
            orthogonal views of the same score matrix - one decomposes across
            variables, the other across observations.
        """
        check_is_fitted(self, "x_loadings_")
        t_start = np.asarray(t_start, dtype=float)
        t_end = np.zeros(self.n_components) if t_end is None else np.asarray(t_end, dtype=float)

        idx = np.arange(self.n_components) if components is None else np.array(components) - 1

        dt = t_end[idx] - t_start[idx]

        if weighted:
            # ``explained_variance_[a] == 0`` for a degenerate component
            # would silently produce inf/NaN weighted contributions.
            # Clamp the divisor so weighting is a no-op on such components.
            # SEC-21 (#270) sub-item 5.
            ev = np.asarray(self.explained_variance_)[idx]
            dt = dt / np.sqrt(np.where(ev > epsqrt, ev, 1.0))

        P = self._x_loadings[:, idx].T
        contributions = dt @ P

        return pd.Series(contributions, index=self._feature_names, name="score_contributions")

    def detect_outliers(self, conf_level: float = 0.95) -> list[dict]:
        """Detect outlier observations using SPE and Hotelling's T² diagnostics.

        Same approach as ``PCA.detect_outliers``: combines statistical limits
        with the robust generalized ESD test.

        Parameters
        ----------
        conf_level : float, default 0.95
            Confidence level in [0.8, 0.999].

        Returns
        -------
        outliers : list of dict
            Sorted from most severe to least. Each dict contains
            ``observation``, ``outlier_types``, ``spe``, ``hotellings_t2``,
            ``spe_limit``, ``hotellings_t2_limit``, ``severity``.

        Examples
        --------
        >>> pls = PLS(n_components=3).fit(X_scaled, Y_scaled)
        >>> outliers = pls.detect_outliers(conf_level=0.95)
        >>> for o in outliers:
        ...     print(f"{o['observation']}: {o['outlier_types']}")
        """
        check_is_fitted(self, "spe_")
        if not (0.8 <= conf_level <= 0.999):
            raise ValueError(f"conf_level must be between 0.8 and 0.999, got {conf_level}.")

        N = self.n_samples_

        spe_values = self.spe_.iloc[:, -1]
        t2_values = self.hotellings_t2_.iloc[:, -1]

        spe_lim = self.spe_limit(conf_level=conf_level)
        t2_lim = self.hotellings_t2_limit(conf_level=conf_level)

        max_outliers = max(1, N // 5)
        alpha = 1 - conf_level

        spe_outlier_idx, _ = detect_outliers_esd(
            spe_values.to_numpy(), algorithm="esd", max_outliers_detected=max_outliers, alpha=alpha
        )
        t2_outlier_idx, _ = detect_outliers_esd(
            t2_values.to_numpy(), algorithm="esd", max_outliers_detected=max_outliers, alpha=alpha
        )

        spe_flagged = set(spe_outlier_idx)
        t2_flagged = set(t2_outlier_idx)

        for i in range(N):
            if spe_values.iloc[i] > spe_lim:
                spe_flagged.add(i)
            if t2_values.iloc[i] > t2_lim:
                t2_flagged.add(i)

        all_flagged = spe_flagged | t2_flagged
        results = []
        for i in all_flagged:
            types = []
            if i in spe_flagged:
                types.append("spe")
            if i in t2_flagged:
                types.append("hotellings_t2")

            spe_val = float(spe_values.iloc[i])
            t2_val = float(t2_values.iloc[i])
            severity = max(spe_val / spe_lim, t2_val / t2_lim)

            results.append(
                {
                    "observation": spe_values.index[i],
                    "outlier_types": types,
                    "spe": spe_val,
                    "hotellings_t2": t2_val,
                    "spe_limit": spe_lim,
                    "hotellings_t2_limit": t2_lim,
                    "severity": round(severity, 4),
                }
            )

        results.sort(key=lambda d: d["severity"], reverse=True)
        return results

    def cross_validate(  # noqa: PLR0912, PLR0913, PLR0915, C901
        self,
        X: DataMatrix,
        Y: DataMatrix,
        *,
        cv: int | str = "loo",
        n_bootstrap: int = 0,
        conf_level: float = 0.95,
        random_state: int | None = None,
        show_progress: bool = True,
    ) -> Bunch:
        """Cross-validate the PLS model and compute error bars for beta coefficients.

        Refits the model on data subsets (jackknife, K-fold, or bootstrap),
        collects ``beta_coefficients_`` from each refit, and computes
        confidence intervals. Also returns cross-validated predictions
        and prediction-error metrics (RMSE, Q²).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Predictor matrix (same data used for ``fit``).
        Y : array-like of shape (n_samples, n_targets)
            Response matrix (same data used for ``fit``).
        cv : int or ``"loo"``, default ``"loo"``
            Cross-validation strategy:

            * ``"loo"`` - leave-one-out (jackknife). Produces N resamples.
            * ``int`` - number of folds for K-fold CV.
        n_bootstrap : int, default 0
            If > 0, use bootstrap resampling instead of CV folds.
            The value specifies the number of bootstrap rounds.
            Overrides the ``cv`` parameter when set.
        conf_level : float, default 0.95
            Confidence level for the beta-coefficient intervals, in (0, 1).
        random_state : int or None, default None
            Random seed for reproducibility (K-fold shuffle and bootstrap).
        show_progress : bool, default True
            Whether to display a ``tqdm`` progress bar.

        Returns
        -------
        result : :class:`~sklearn.utils.Bunch`
            Dictionary-like object with the following keys:

            **Beta-coefficient uncertainty**

            beta_samples : np.ndarray of shape (n_resamples, n_features, n_targets)
                Raw beta coefficients from every resample.
            beta_mean : pd.DataFrame of shape (n_features, n_targets)
                Mean beta across resamples.
            beta_std : pd.DataFrame of shape (n_features, n_targets)
                Standard error of the beta coefficients.
            beta_ci_lower : pd.DataFrame of shape (n_features, n_targets)
                Lower bound of the confidence interval.
            beta_ci_upper : pd.DataFrame of shape (n_features, n_targets)
                Upper bound of the confidence interval.
            significant : pd.DataFrame of shape (n_features, n_targets)
                ``True`` where the confidence interval excludes zero.

            **Prediction metrics**

            y_hat_cv : pd.DataFrame of shape (n_samples, n_targets)
                Cross-validated predictions (out-of-fold). Only available for
                jackknife and K-fold; ``None`` for bootstrap.
            press : float
                Prediction Error Sum of Squares (sum over all Y elements).
                Only for jackknife / K-fold.
            rmse_cv : pd.Series of length n_targets
                Root-mean-square error per Y variable (cross-validated).
                Only for jackknife / K-fold.
            q_squared : pd.Series of length n_targets
                Cross-validated R² (Q²) per Y variable.
                Only for jackknife / K-fold.

            **Metadata**

            n_resamples : int
                Number of resamples performed.
            method : str
                ``"jackknife"``, ``"kfold"``, or ``"bootstrap"``.
            conf_level : float
                The confidence level used.

        Examples
        --------
        >>> from process_improve.multivariate import PLS, MCUVScaler
        >>> scaler_x = MCUVScaler().fit(X)
        >>> scaler_y = MCUVScaler().fit(Y)
        >>> X_s, Y_s = scaler_x.transform(X), scaler_y.transform(Y)
        >>> pls = PLS(n_components=2).fit(X_s, Y_s)
        >>> cv_results = pls.cross_validate(X_s, Y_s, cv="loo")
        >>> cv_results.beta_mean          # mean beta across LOO resamples
        >>> cv_results.significant        # which betas are significantly != 0
        >>> cv_results.q_squared          # cross-validated R²
        """
        check_is_fitted(self, "beta_coefficients_")

        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        Y = pd.DataFrame(Y) if not isinstance(Y, pd.DataFrame) else Y

        N, _K = X.shape
        M = Y.shape[1]

        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"X and Y must have the same number of rows, got {X.shape[0]} and {Y.shape[0]}.")
        if not (0.5 < conf_level < 1.0):
            raise ValueError(f"conf_level must be between 0.5 and 1.0, got {conf_level}.")

        # --- Determine resampling strategy ---
        use_bootstrap = n_bootstrap > 0
        if use_bootstrap:
            method = "bootstrap"
        elif cv == "loo":
            method = "jackknife"
        else:
            method = "kfold"

        # --- Collect beta coefficients (and out-of-fold predictions for CV) ---
        beta_collection: list[np.ndarray] = []
        y_hat_cv = np.full((N, M), np.nan) if not use_bootstrap else None
        rng = np.random.default_rng(random_state)

        if use_bootstrap:
            iterator = tqdm(range(n_bootstrap), desc="Bootstrap", disable=not show_progress)
            for _ in iterator:
                train_idx = rng.choice(N, size=N, replace=True)
                sub_model = clone(self).fit(X.iloc[train_idx], Y.iloc[train_idx])
                beta_collection.append(sub_model.beta_coefficients_.values)

        elif method == "jackknife":
            assert y_hat_cv is not None  # only None when use_bootstrap is True
            iterator = tqdm(range(N), desc="Jackknife (LOO)", disable=not show_progress)
            for i in iterator:
                train_idx = np.concatenate([np.arange(i), np.arange(i + 1, N)])
                sub_model = clone(self).fit(X.iloc[train_idx], Y.iloc[train_idx])
                beta_collection.append(sub_model.beta_coefficients_.values)
                pred = sub_model.predict(X.iloc[[i]])
                y_hat_cv[i, :] = pred.values.ravel()

        else:  # K-fold
            assert y_hat_cv is not None  # only None when use_bootstrap is True
            n_resamples = int(cv)
            kf = KFold(n_splits=n_resamples, shuffle=True, random_state=random_state)
            desc = f"{n_resamples}-Fold CV"
            for train_idx, test_idx in tqdm(kf.split(X), total=n_resamples, desc=desc, disable=not show_progress):
                sub_model = clone(self).fit(X.iloc[train_idx], Y.iloc[train_idx])
                beta_collection.append(sub_model.beta_coefficients_.values)
                pred = sub_model.predict(X.iloc[test_idx])
                y_hat_cv[test_idx, :] = pred.values

        beta_samples = np.array(beta_collection)  # (n_resamples, K, M)
        actual_n_resamples = beta_samples.shape[0]

        # --- Beta-coefficient statistics ---
        beta_mean_arr = beta_samples.mean(axis=0)

        if method == "jackknife":
            # Jackknife variance: var = (N-1)/N * sum_i (beta_i - beta_mean)^2
            jackknife_var = (N - 1) / N * np.sum((beta_samples - beta_mean_arr) ** 2, axis=0)
            beta_std_arr = np.sqrt(jackknife_var)
            # CI via t-distribution
            alpha = 1 - conf_level
            t_crit = t_dist.ppf(1 - alpha / 2, df=N - 1)
            beta_ci_lower_arr = beta_mean_arr - t_crit * beta_std_arr
            beta_ci_upper_arr = beta_mean_arr + t_crit * beta_std_arr
        elif method == "kfold":
            # For K-fold, use sample std of the K beta estimates
            beta_std_arr = beta_samples.std(axis=0, ddof=1)
            alpha = 1 - conf_level
            t_crit = t_dist.ppf(1 - alpha / 2, df=actual_n_resamples - 1)
            beta_ci_lower_arr = beta_mean_arr - t_crit * beta_std_arr
            beta_ci_upper_arr = beta_mean_arr + t_crit * beta_std_arr
        else:  # bootstrap percentile CI
            beta_std_arr = beta_samples.std(axis=0, ddof=1)
            alpha = 1 - conf_level
            beta_ci_lower_arr = np.percentile(beta_samples, 100 * alpha / 2, axis=0)
            beta_ci_upper_arr = np.percentile(beta_samples, 100 * (1 - alpha / 2), axis=0)

        # Significance: CI does not contain zero
        significant_arr = (beta_ci_lower_arr > 0) | (beta_ci_upper_arr < 0)

        x_cols = self.beta_coefficients_.index
        y_cols = self.beta_coefficients_.columns

        beta_mean_df = pd.DataFrame(beta_mean_arr, index=x_cols, columns=y_cols)
        beta_std_df = pd.DataFrame(beta_std_arr, index=x_cols, columns=y_cols)
        beta_ci_lower_df = pd.DataFrame(beta_ci_lower_arr, index=x_cols, columns=y_cols)
        beta_ci_upper_df = pd.DataFrame(beta_ci_upper_arr, index=x_cols, columns=y_cols)
        significant_df = pd.DataFrame(significant_arr, index=x_cols, columns=y_cols)

        # --- Cross-validated prediction metrics ---
        press_val = None
        rmse_cv_series = None
        q_squared_series = None
        y_hat_cv_df = None

        if y_hat_cv is not None:
            y_hat_cv_df = pd.DataFrame(y_hat_cv, index=Y.index, columns=Y.columns)
            residuals = Y.values - y_hat_cv
            press_val = float(np.nansum(residuals**2))
            ss_total = np.nansum((Y.values - Y.values.mean(axis=0)) ** 2, axis=0)
            ss_res = np.nansum(residuals**2, axis=0)

            rmse_vals = np.sqrt(np.nanmean(residuals**2, axis=0))
            rmse_cv_series = pd.Series(rmse_vals, index=Y.columns, name="RMSE_CV")

            q2_vals = 1.0 - ss_res / ss_total
            q_squared_series = pd.Series(q2_vals, index=Y.columns, name="Q_squared")

        return Bunch(
            beta_samples=beta_samples,
            beta_mean=beta_mean_df,
            beta_std=beta_std_df,
            beta_ci_lower=beta_ci_lower_df,
            beta_ci_upper=beta_ci_upper_df,
            significant=significant_df,
            y_hat_cv=y_hat_cv_df,
            press=press_val,
            rmse_cv=rmse_cv_series,
            q_squared=q_squared_series,
            n_resamples=actual_n_resamples,
            method=method,
            conf_level=conf_level,
        )

    def prediction_interval(
        self,
        X: DataMatrix,
        *,
        conf_level: float = 0.95,
        cv_result: Bunch | None = None,
    ) -> Bunch:
        """Prediction interval for the Y predictions of new observations.

        The interval combines the residual error variance with the leverage of
        each new observation in the latent-variable space. For a new
        observation the prediction-interval half-width on target ``m`` is

        ``t * s_E[m] * sqrt(1 + 1/N + T2_new / (N - 1))``

        where ``s_E`` is the residual error standard deviation, ``T2_new`` is
        the Hotelling's T² of the new observation, ``N`` is the number of
        calibration samples, and ``t`` is the Student-t quantile.

        Parameters
        ----------
        X : array-like of shape (n_new, n_features)
            New observations, pre-processed the same way as the training data.
        conf_level : float, default=0.95
            Confidence level for the interval, in (0.5, 1.0).
        cv_result : sklearn.utils.Bunch or None, default=None
            The result of :meth:`cross_validate`. When supplied, its
            cross-validated RMSE (``rmse_cv``) is used for the error variance,
            which is preferable to the optimistic calibration RMSE used
            otherwise.

        Returns
        -------
        sklearn.utils.Bunch
            With keys ``y_hat`` (point predictions), ``lower`` and ``upper``
            (prediction-interval bounds) - each a DataFrame of shape
            (n_new, n_targets) - and ``conf_level``.
        """
        check_is_fitted(self, "beta_coefficients_")
        if not (0.5 < conf_level < 1.0):
            raise ValueError(f"conf_level must be between 0.5 and 1.0, got {conf_level}.")

        diagnostics = self.diagnose(X)
        y_hat = diagnostics.y_hat
        t2_new = np.asarray(diagnostics.hotellings_t2, dtype=float)

        n_samples = self.n_samples_
        n_components = int(self.n_components)

        # Residual error std per Y variable: prefer the cross-validated RMSE
        # when a cross_validate() result is supplied (calibration RMSE is
        # optimistic for genuinely new observations).
        if cv_result is not None:
            error_std = np.asarray(cv_result.rmse_cv, dtype=float)
        else:
            error_std = np.asarray(self.rmse_.iloc[:, -1], dtype=float)

        # Leverage of a new observation in the latent space.
        leverage = 1.0 / n_samples + t2_new / (n_samples - 1)

        df = max(n_samples - n_components - 1, 1)
        t_crit = t_dist.ppf(1 - (1 - conf_level) / 2, df)
        half_width = t_crit * np.sqrt(1.0 + leverage)[:, None] * error_std[None, :]

        lower = pd.DataFrame(y_hat.values - half_width, index=y_hat.index, columns=y_hat.columns)
        upper = pd.DataFrame(y_hat.values + half_width, index=y_hat.index, columns=y_hat.columns)
        return Bunch(y_hat=y_hat, lower=lower, upper=upper, conf_level=conf_level)

