# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.
"""Principal Component Analysis (PCA) estimator (ENG-01).

The sklearn-compatible :class:`PCA` transformer, with NIPALS, SVD and TSR
fitting paths and full missing-data support. Diagnostics, confidence limits and
plotting are pulled in from the sibling submodules and bound as convenience
methods after ``fit()``.
"""

from __future__ import annotations

import logging
import time
import typing
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.decomposition import PCA as _SkPCA  # noqa: N811
from sklearn.model_selection import BaseCrossValidator, cross_val_score
from sklearn.utils import Bunch
from sklearn.utils.validation import check_is_fitted, validate_data

from ..univariate.metrics import detect_outliers_esd
from ._base import _LatentVariableModel, _LazyFrame
from ._common import (
    Q2_MIN_INCREMENT,
    DataMatrix,
    NotEnoughVarianceError,
    SelectionRule,
    SpecificationWarning,
    _align_to_fit_features,
    _select_n_components,
    epsqrt,
)
from ._nipals import quick_regress, ssq, terminate_check
from ._preprocessing import MCUVScaler

logger = logging.getLogger(__name__)


def _pca_ekf_press(  # noqa: PLR0913, PLR0915, PLR0912, C901
    X: np.ndarray,
    max_components: int,
    *,
    n_folds: int = 5,
    n_repeats: int = 1,
    n_iter: int = 50,
    tol: float = 1e-6,
    scale_inside_folds: bool = True,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Element-wise k-fold (ekf) PCA cross-validation.

    Partitions the elements of ``X`` into ``n_folds`` element-folds (each cell
    is held out exactly once across folds); for each fold and each candidate
    component count, the held-out cells are masked, initialised from the
    in-fold column means, and refined by iterating SVD reconstruction
    (Expectation-Maximisation style). The fitted model never sees the
    held-out true values, so the squared error of the prediction is an honest
    out-of-sample PRESS - the independence requirement of Bro, Kjeldahl,
    Smilde & Kiers (2008, *Anal. Bioanal. Chem.* 390:1241-1251) that the
    row-wise CV scheme violates.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Data matrix. With the default ``scale_inside_folds=True`` the raw
        unscaled matrix may be passed; with ``False`` the caller is expected
        to have mean-centred (and usually unit-variance scaled) it.
    max_components : int
        Maximum number of components to evaluate; PRESS is computed for
        ``1 .. max_components``.
    n_folds : int, default 5
        Number of element-folds. Bro 2008 uses 7 as a typical default; 5 is
        a faster choice that still gives a stable curve.
    n_repeats : int, default 1
        Number of times to repeat the ekf pass with a fresh random fold
        permutation. Each repeat covers every cell exactly once; ``n_repeats
        > 1`` averages over different element-fold partitions, narrowing
        the per-component PRESS standard error at extra runtime.
    n_iter : int, default 50
        Maximum number of EM iterations per fold and component count.
    tol : float, default 1e-6
        Relative change in the held-out cell predictions below which EM
        stops early.
    scale_inside_folds : bool, default True
        If True, fit per-column mean and unit-variance constants on each
        fold's in-fold cells and apply them to the whole matrix before
        running EM; predictions are inverse-transformed before PRESS is
        accumulated. This removes the centring/scaling leakage of the
        previous default (which used a single set of constants iteratively
        recomputed from the imputed matrix). If False, the scheme reverts
        to the prior behaviour (caller is responsible for scaling, and the
        in-loop column-mean is allowed to drift with the EM imputation).
    random_state : int, optional
        Seed for the element-fold permutation, for reproducibility across
        repeats.

    Returns
    -------
    press : np.ndarray of shape (max_components,)
        Per-cell PRESS per component count, averaged over ``n_repeats``
        passes so the scale is comparable to a single-pass run.
    per_fold_press : np.ndarray of shape (max_components, n_folds * n_repeats)
        Per-fold PRESS contributions across every fold of every repeat;
        drives the 1-SE rule's standard error.

    References
    ----------
    Bro, R., Kjeldahl, K., Smilde, A. K., & Kiers, H. A. L. (2008).
    Cross-validation of component models: a critical look at current
    methods. *Anal. Bioanal. Chem.*, 390(5), 1241-1251. PMID 18214448.

    Camacho, J., & Ferrer, A. (2012). Cross-validation in PCA models with
    the element-wise k-fold (ekf) algorithm: theoretical aspects.
    *J. Chemometrics*, 26(7), 361-373. DOI 10.1002/cem.2440.
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    rng = np.random.default_rng(random_state)

    total_folds = n_folds * n_repeats
    per_fold_press = np.zeros((max_components, total_folds))
    fold_counter = 0

    n_cells = n * p
    fold_size = n_cells // n_folds

    for _ in range(n_repeats):
        # Assign cells to folds via a balanced random permutation so every
        # fold has ~n*p/n_folds cells (not all the same column, not the
        # same row).
        perm = rng.permutation(n_cells)
        fold = np.empty(n_cells, dtype=np.int64)
        for k in range(n_folds):
            start = k * fold_size
            end = (k + 1) * fold_size if k < n_folds - 1 else n_cells
            fold[perm[start:end]] = k
        fold = fold.reshape(n, p)

        for k in range(n_folds):
            mask = fold == k  # (n, p) bool, True where the cell is held out
            if not mask.any():
                fold_counter += 1
                continue

            # Fit per-column centring/scaling on the in-fold cells. Only
            # consumed under scale_inside_folds=True; the False path uses
            # in_fold_vals.mean() for the initial imputation and recomputes
            # the column mean inside EM, so we skip the work here.
            col_centre = np.zeros(p)
            col_scale = np.ones(p)
            if scale_inside_folds:
                for j in range(p):
                    in_fold = X[~mask[:, j], j]
                    if in_fold.size > 1:
                        col_centre[j] = float(in_fold.mean())
                        sd = float(in_fold.std(ddof=1))
                        col_scale[j] = sd if sd > epsqrt else 1.0
                    elif in_fold.size == 1:
                        col_centre[j] = float(in_fold[0])

            # Initial imputation in original space: held-out cells take the
            # in-fold column mean (which is ``col_centre`` under
            # ``scale_inside_folds=True``, and the same value under False).
            Xtr = X.copy()
            for j in range(p):
                m_j = mask[:, j]
                if m_j.any():
                    if scale_inside_folds:
                        Xtr[m_j, j] = col_centre[j]
                    else:
                        in_fold_vals = X[~m_j, j]
                        Xtr[m_j, j] = in_fold_vals.mean() if in_fold_vals.size > 0 else 0.0

            for a in range(1, max_components + 1):
                Xa = Xtr.copy()
                prev_held = Xa[mask].copy()
                for _iteration in range(n_iter):
                    if scale_inside_folds:
                        # Centre and scale by the FIXED in-fold constants; the
                        # in-fold cells now sit on the analysis scale and the
                        # held-out cells move under EM.
                        Xs = (Xa - col_centre) / col_scale
                        _, S, Vt = np.linalg.svd(Xs, full_matrices=False)
                        rank = min(a, S.shape[0])
                        recon_s = (Xs @ Vt[:rank].T) @ Vt[:rank]
                        recon = recon_s * col_scale + col_centre
                    else:
                        # Prior behaviour: recompute the column mean each
                        # iteration. The mean drifts slightly as the held-out
                        # cells are updated.
                        col_mean = Xa.mean(axis=0)
                        Xc = Xa - col_mean
                        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
                        rank = min(a, S.shape[0])
                        recon_centred = (Xc @ Vt[:rank].T) @ Vt[:rank]
                        recon = recon_centred + col_mean
                    Xa[mask] = recon[mask]
                    delta = np.linalg.norm(Xa[mask] - prev_held)
                    scale = max(1.0, float(np.linalg.norm(prev_held)))
                    if delta < tol * scale:
                        break
                    prev_held = Xa[mask].copy()

                per_fold_press[a - 1, fold_counter] = float(
                    np.sum((X[mask] - Xa[mask]) ** 2)
                )
            fold_counter += 1

    # Average over repeats so PRESS stays on the per-cell scale.
    press = per_fold_press.sum(axis=1) / max(1, n_repeats)
    return press, per_fold_press


class PCA(_LatentVariableModel, TransformerMixin, BaseEstimator):
    """Principal Component Analysis with support for missing data.

    Parameters
    ----------
    n_components : int
        Number of principal components to extract.

    algorithm : str, default="auto"
        Algorithm to use for fitting the model.
        - ``"auto"``: Uses SVD when data is complete, NIPALS when data has missing values.
        - ``"svd"``: Singular Value Decomposition. Requires complete data.
        - ``"nipals"``: Non-linear Iterative Partial Least Squares. Handles missing data.
        - ``"tsr"``: Trimmed Score Regression. Handles missing data.

    missing_data_settings : dict or None, default=None
        Settings for iterative missing data algorithms (NIPALS, TSR).
        Keys: ``md_tol`` (convergence tolerance), ``md_max_iter`` (max iterations).

    Attributes (after fitting)
    --------------------------
    scores_ : pd.DataFrame of shape (n_samples, n_components)
        The score matrix (T).
    loadings_ : pd.DataFrame of shape (n_features, n_components)
        The loading matrix (P).
    r2_per_component_ : pd.Series of length n_components
        Fractional R² explained by each component.
    r2_cumulative_ : pd.Series of length n_components
        Cumulative R² after each component.
    r2_per_variable_ : pd.DataFrame of shape (n_features, n_components)
        Per-variable cumulative R² after each component.
    spe_ : pd.DataFrame of shape (n_samples, n_components)
        Squared Prediction Error (stored as sqrt of row sum-of-squares).
    hotellings_t2_ : pd.DataFrame of shape (n_samples, n_components)
        Cumulative Hotelling's T² statistic.
    explained_variance_ : np.ndarray of shape (n_components,)
        Variance explained by each component.
    scaling_factor_for_scores_ : pd.Series of length n_components
        Standard deviation per score (sqrt of explained variance).
    has_missing_data_ : bool
        Whether the training data contained missing values.
    fitting_info_ : dict
        Timing and iteration info from the fitting algorithm.
    """

    _valid_algorithms: typing.ClassVar[list[str]] = ["auto", "svd", "nipals", "tsr"]

    _parameter_constraints: typing.ClassVar = {
        "n_components": [int, None],
        "algorithm": [str],
        "missing_data_settings": [dict, None],
    }

    def __init__(
        self,
        n_components: int,
        *,
        algorithm: str = "auto",
        missing_data_settings: dict | None = None,
    ):
        self.n_components = n_components
        self.algorithm = algorithm
        self.missing_data_settings = missing_data_settings

    # ENG-17: the convenience methods (score_plot, vip, spe_limit, ...),
    # hotellings_t2_limit, ellipse_coordinates and the rename __getattr__ are
    # inherited from _LatentVariableModel. PCA supplies only its rename map.
    _ATTRIBUTE_RENAMES: typing.ClassVar[dict[str, str]] = {
        "x_scores": "scores_",
        "loadings": "loadings_",
        "x_loadings": "loadings_",
        "squared_prediction_error": "spe_",
        "R2": "r2_per_component_",
        "R2cum": "r2_cumulative_",
        "R2X_cum": "r2_per_variable_",
        "hotellings_t2": "hotellings_t2_",
        "scaling_factor_for_scores": "scaling_factor_for_scores_",
        "N": "n_samples_",
        "K": "n_features_in_",
        "A": "n_components",
        "extra_info": "fitting_info_",
    }
    _RENAME_CONTEXT: typing.ClassVar[str] = "PCA"

    # Fitted diagnostics: per-component arrays (NIPALS/TSR) or scalar totals (SVD).
    fitting_info_: dict[str, np.ndarray | int | float]

    # ENG-18: public DataFrame views built lazily from the private ndarrays.
    scores_ = _LazyFrame("_scores", index="_sample_index", columns="_component_names")
    loadings_ = _LazyFrame("_loadings", index="_feature_names", columns="_component_names")
    spe_ = _LazyFrame("_spe", index="_sample_index", columns="_component_names")

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: DataMatrix, y: DataMatrix | None = None) -> PCA:  # noqa: ARG002, PLR0912, PLR0915, C901
        """Fit a principal component analysis (PCA) model to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. May contain NaN values for missing data (the
            NIPALS / TSR algorithms thread them through; the SVD path
            rejects them).
        y : ignored

        Returns
        -------
        self : PCA
        """
        # Capture the original DataFrame's index/columns before validate_data
        # converts X to an ndarray, then rebuild the DataFrame view downstream
        # code expects. validate_data also sets n_features_in_ / feature_
        # names_in_ and runs the sklearn input rejections (sparse, complex,
        # empty, dtype-object) with the standard error messages.
        sample_index = X.index if isinstance(X, pd.DataFrame) else None
        feature_columns = X.columns if isinstance(X, pd.DataFrame) else None
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
            feature_columns = pd.RangeIndex(X_arr.shape[1])
        if sample_index is None:
            sample_index = pd.RangeIndex(X_arr.shape[0])
        X = pd.DataFrame(X_arr, index=sample_index, columns=feature_columns)

        N, K = X.shape
        self.n_samples_ = N
        # n_features_in_ already set by validate_data; reassert for clarity.
        self.n_features_in_ = K
        self._feature_names = X.columns
        self._sample_index = X.index

        # Clamp n_components
        min_dim = int(min(N, K))
        A = min_dim if self.n_components is None else int(self.n_components)
        if min_dim < A:
            warnings.warn(
                "The requested number of components is more than can be "
                "computed from data. The maximum number of components is "
                f"the minimum of either the number of rows ({N}) or "
                f"the number of columns ({K}).",
                SpecificationWarning,
                stacklevel=2,
            )
            A = min_dim
        self.n_components = A

        # Detect missing data and resolve algorithm
        self.has_missing_data_ = bool(np.any(X.isna()))
        algo = self.algorithm.lower()
        if algo not in self._valid_algorithms:
            raise ValueError(
                f"Algorithm '{self.algorithm}' is not recognized. Must be one of {self._valid_algorithms}."
            )

        if algo == "auto":
            algo = "nipals" if self.has_missing_data_ else "svd"
        if algo == "svd" and self.has_missing_data_:
            raise ValueError("SVD algorithm cannot handle missing data. Use 'nipals', 'tsr', or 'auto'.")
        self.algorithm_ = algo

        # Build settings for iterative algorithms
        settings = {"md_tol": epsqrt, "md_max_iter": 1000}
        if isinstance(self.missing_data_settings, dict):
            settings.update(self.missing_data_settings)
        settings["md_max_iter"] = int(settings["md_max_iter"])

        if algo in ("nipals", "tsr"):
            if not settings["md_tol"] < 10:
                raise ValueError("Tolerance should not be too large.")
            if not settings["md_tol"] > epsqrt**1.95:
                raise ValueError("Tolerance must exceed machine precision.")

        # Storage for numpy results (set by _fit_* methods)
        X_values = np.asarray(X.copy())

        # Dispatch
        if algo == "svd":
            self._fit_svd(X_values, N, K, A)
        elif algo == "nipals":
            self._fit_nipals(X_values, N, K, A, settings)
        elif algo == "tsr":
            self._fit_tsr(X_values, N, K, A, settings)

        # --- Common post-fit path ---
        # ENG-18: scores_ / loadings_ / spe_ are stored as private ndarrays (the
        # source of truth); the public DataFrame views are built lazily by the
        # _LazyFrame descriptors from these arrays plus the index/column metadata
        # (self._sample_index, self._feature_names, self._component_names).
        self._component_names = list(range(1, A + 1))
        component_names = self._component_names
        self._loadings = self._loadings_np
        self._scores = self._scores_np
        self._spe = self._spe_np

        self.r2_per_component_ = pd.Series(
            self._r2_np,
            index=component_names,
            name="R² per component",
        )
        self.r2_cumulative_ = pd.Series(
            self._r2cum_np,
            index=component_names,
            name="Cumulative R²",
        )
        self.r2_per_variable_ = pd.DataFrame(
            self._r2_per_var_np,
            index=self._feature_names,
            columns=component_names,
        )

        self.scaling_factor_for_scores_ = pd.Series(
            np.sqrt(self.explained_variance_),
            index=component_names,
            name="Standard deviation per score",
        )

        # Hotelling's T² (cumulative across components)
        self.hotellings_t2_ = pd.DataFrame(
            np.zeros((N, A)),
            columns=component_names,
            index=self._sample_index,
        )
        for a in range(A):
            self.hotellings_t2_.iloc[:, a] = (
                self.hotellings_t2_.iloc[:, max(0, a - 1)]
                + (self._scores[:, a] / self.scaling_factor_for_scores_.iloc[a]) ** 2
            )

        # Clean up temporary numpy staging names (the kept ndarrays above alias
        # the same arrays, so they survive these del statements).
        del self._loadings_np, self._scores_np, self._r2_np, self._r2cum_np, self._r2_per_var_np, self._spe_np

        return self

    def _fit_svd(self, X_values: np.ndarray, N: int, K: int, A: int) -> None:
        """Fit PCA using SVD decomposition (complete data only)."""
        U, S, Vt = np.linalg.svd(X_values, full_matrices=False)

        # Loadings are the first A right singular vectors (transposed to K x A)
        self._loadings_np = Vt[:A, :].T
        # Scores are U * S for the first A components
        self._scores_np = U[:, :A] * S[:A]

        # Sign convention: flip so largest magnitude element in each loading is positive
        # (Wold, Esbensen, Geladi, PCA, CILS, 1987, p 42)
        for a in range(A):
            max_el_idx = np.argmax(np.abs(self._loadings_np[:, a]))
            if self._loadings_np[max_el_idx, a] < 0:
                self._loadings_np[:, a] *= -1.0
                self._scores_np[:, a] *= -1.0

        # Explained variance. ``max(1, N-1)`` mirrors the MBPLS / MBPCA
        # paths and prevents a division-by-zero / negative-divisor when
        # the caller fits a model on a single row. SEC-21 (#270) sub-item 6.
        self.explained_variance_ = np.diag(self._scores_np.T @ self._scores_np) / max(1, N - 1)

        # Compute R2 and SPE via deflation
        self._r2_np = np.zeros(A)
        self._r2cum_np = np.zeros(A)
        self._r2_per_var_np = np.zeros((K, A))
        self._spe_np = np.zeros((N, A))

        Xd = X_values.copy()
        prior_ssx_col = ssq(Xd, axis=0)
        base_variance = np.sum(prior_ssx_col)

        for a in range(A):
            Xd = Xd - self._scores_np[:, [a]] @ self._loadings_np[:, [a]].T
            row_ssx = ssq(Xd, axis=1)
            col_ssx = ssq(Xd, axis=0)

            self._spe_np[:, a] = np.sqrt(row_ssx)
            # Per-variable R^2 is undefined for a column with no variance to
            # explain; emit NaN there instead of letting RuntimeWarning
            # ``invalid value encountered in divide`` poison the output.
            # SEC-21 (#270) sub-item 4.
            self._r2_per_var_np[:, a] = np.where(
                prior_ssx_col > 0, 1 - col_ssx / np.where(prior_ssx_col > 0, prior_ssx_col, 1.0), np.nan
            )
            self._r2cum_np[a] = 1 - np.sum(row_ssx) / base_variance if base_variance > 0 else np.nan
            self._r2_np[a] = self._r2cum_np[a] - self._r2cum_np[a - 1] if a > 0 else self._r2cum_np[a]

        self.fitting_info_ = {"timing": np.zeros(A) * np.nan, "iterations": np.zeros(A) * np.nan}

    def _fit_nipals(self, X_values: np.ndarray, N: int, K: int, A: int, settings: dict) -> None:
        """Fit PCA using the NIPALS algorithm (handles missing data)."""
        Xd = X_values.copy()
        base_variance = ssq(Xd)

        self._loadings_np = np.zeros((K, A))
        self._scores_np = np.zeros((N, A))
        self._r2_np = np.zeros(A)
        self._r2cum_np = np.zeros(A)
        self._r2_per_var_np = np.zeros((K, A))
        self._spe_np = np.zeros((N, A))
        self.fitting_info_ = {"timing": np.zeros(A) * np.nan, "iterations": np.zeros(A) * np.nan}

        for a in np.arange(A):
            start_time = time.time()
            itern = 0
            start_ss_col = ssq(Xd, axis=0)

            if np.sum(start_ss_col) < epsqrt:
                emsg = (
                    "There is no variance left in the data array: cannot "
                    f"compute any more components beyond component {a}."
                )
                raise NotEnoughVarianceError(emsg)

            # Seed the score from the column of X with the greatest
            # sum-of-squares (variance, for mean-centred data) rather than the
            # arbitrary first column (#195). NIPALS converges to the same
            # component for any non-degenerate seed, but the highest-variance
            # column is closest to the leading component, so it needs fewer
            # iterations and is far more robust when the first column happens to
            # be near-orthogonal to it. The deterministic sign convention applied
            # below makes the fitted sign independent of this seed.
            #
            # ``Xd[:, [start_col]]`` (fancy indexing) already returns a copy in
            # current numpy, so the in-place ``isnan -> 0`` does not poison Xd
            # today. The explicit ``.copy()`` here is defensive: it mirrors the
            # PLS path and protects against any future numpy change that flips
            # fancy indexing to a view-returning variant. SEC-21 (#270) sub-item 2.
            start_col = int(np.argmax(start_ss_col))
            t_a_guess = Xd[:, [start_col]].copy()
            t_a_guess[np.isnan(t_a_guess)] = 0
            t_a = t_a_guess + 1.0
            p_a = np.zeros((K, 1))
            while not (terminate_check(t_a_guess, t_a, iterations=itern, settings=settings)):
                t_a_guess = t_a.copy()

                # Regress X onto t_a to get loadings p_a
                p_a = quick_regress(Xd, t_a)
                p_a = p_a / np.sqrt(ssq(p_a))

                # Regress X onto p_a to get scores t_a
                t_a = quick_regress(Xd, p_a)

                itern += 1

            timing_arr = typing.cast("np.ndarray", self.fitting_info_["timing"])
            iterations_arr = typing.cast("np.ndarray", self.fitting_info_["iterations"])
            timing_arr[a] = time.time() - start_time
            iterations_arr[a] = itern
            logger.debug(
                "PCA NIPALS: component %d converged in %d iterations (md_tol=%g)",
                a + 1,
                itern,
                settings["md_tol"],
            )

            # Deflate
            Xd = Xd - np.dot(t_a, p_a.T)
            row_ssx = ssq(Xd, axis=1)
            col_ssx = ssq(Xd, axis=0)

            self._spe_np[:, a] = np.sqrt(row_ssx)
            # Per-variable R^2 is undefined for a column with no variance to
            # explain; emit NaN there. SEC-21 (#270) sub-item 4.
            self._r2_per_var_np[:, a] = np.where(
                start_ss_col > 0, 1 - col_ssx / np.where(start_ss_col > 0, start_ss_col, 1.0), np.nan
            )
            self._r2cum_np[a] = 1 - np.sum(row_ssx) / base_variance if base_variance > 0 else np.nan
            self._r2_np[a] = self._r2cum_np[a] - self._r2cum_np[a - 1] if a > 0 else self._r2cum_np[a]

            # Sign convention: largest magnitude element in loading is positive
            max_el_idx = np.argmax(np.abs(p_a))
            if np.sign(p_a[max_el_idx]) < 1:
                p_a *= -1.0
                t_a *= -1.0

            self._loadings_np[:, a] = p_a.flatten()
            self._scores_np[:, a] = t_a.flatten()

        # Explained variance. ``max(1, N-1)`` mirrors the MBPLS / MBPCA
        # paths and prevents a division-by-zero / negative-divisor when
        # the caller fits a model on a single row. SEC-21 (#270) sub-item 6.
        self.explained_variance_ = np.diag(self._scores_np.T @ self._scores_np) / max(1, N - 1)

    def _fit_tsr(self, X_values: np.ndarray, N: int, K: int, A: int, settings: dict) -> None:
        """Fit PCA using the Trimmed Score Regression algorithm (handles missing data).

        See papers by Abel Folch-Fortuny and also DOI: 10.1002/cem.750
        """
        start_time = time.time()
        delta = 1e100
        Xd = X_values.copy()
        X_original = X_values.copy()
        base_variance = ssq(Xd)

        mmap = np.isnan(Xd)
        Xd[mmap] = 0.0
        itern = 0
        while (itern < settings["md_max_iter"]) and (delta > settings["md_tol"]):
            itern += 1
            missing_X = Xd[mmap]
            mean_X = np.mean(Xd, axis=0)
            S = np.cov(Xd, rowvar=False, ddof=1)
            Xc = Xd - mean_X
            if N > K:
                _, _, V = np.linalg.svd(Xc, full_matrices=False)
            else:
                V, _, _ = np.linalg.svd(Xc.T, full_matrices=False)

            V = V.T[:, 0:A]
            for n in range(N):
                row_mis = mmap[n, :]
                row_obs = ~row_mis
                if np.any(row_mis):
                    L = V[row_obs, 0 : min(A, sum(row_obs))]
                    S11 = S[row_obs, :][:, row_obs]
                    S21 = S[row_mis, :][:, row_obs]
                    z2 = (S21 @ L) @ np.linalg.pinv(L.T @ S11 @ L) @ L.T
                    Xc[n, row_mis] = z2 @ Xc[n, row_obs]
            Xd = Xc + mean_X
            delta = np.mean((Xd[mmap] - missing_X) ** 2)

        # Final decomposition
        S = np.cov(Xd, rowvar=False, ddof=1)
        _, _, V = np.linalg.svd(S, full_matrices=False)

        self._loadings_np = (V[0:A, :]).T  # K x A
        self._scores_np = (Xd - np.mean(Xd, axis=0)) @ self._loadings_np

        # R2 and SPE
        self._r2_np = np.zeros(A)
        self._r2cum_np = np.zeros(A)
        self._r2_per_var_np = np.zeros((K, A))
        self._spe_np = np.zeros((N, A))

        for a in range(A):
            residuals = self._scores_np[:, : a + 1] @ self._loadings_np[:, : a + 1].T - X_original
            self._r2cum_np[a] = 1 - ssq(residuals, axis=None) / base_variance
            self._r2_np[a] = self._r2cum_np[a] - self._r2cum_np[a - 1] if a > 0 else self._r2cum_np[a]
            self._spe_np[:, a] = np.sqrt(ssq(residuals, axis=1))

        self.fitting_info_ = {"iterations": itern, "timing": time.time() - start_time}

        # Explained variance. ``max(1, N-1)`` mirrors the MBPLS / MBPCA
        # paths and prevents a division-by-zero / negative-divisor when
        # the caller fits a model on a single row. SEC-21 (#270) sub-item 6.
        self.explained_variance_ = np.diag(self._scores_np.T @ self._scores_np) / max(1, N - 1)

    def transform(self, X: DataMatrix) -> pd.DataFrame:
        """Project new data onto the fitted PCA model to obtain scores.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to project. Must have the same number of features as
            the training data.

        Returns
        -------
        scores : pd.DataFrame of shape (n_samples, n_components)
        """
        check_is_fitted(self, "loadings_")
        sample_index = X.index if isinstance(X, pd.DataFrame) else None
        feature_columns = X.columns if isinstance(X, pd.DataFrame) else None
        X_arr = validate_data(
            self,
            X,
            reset=False,
            accept_sparse=False,
            dtype="numeric",
            ensure_all_finite="allow-nan",
        )
        # _align_to_fit_features handles the renamed-columns / reordered-
        # columns paths that validate_data leaves alone (it only checks names
        # match if feature_names_in_ is present, doesn't reorder). Run it on
        # a DataFrame view so the existing logic still applies.
        if feature_columns is None:
            feature_columns = self._feature_names
        if sample_index is None:
            sample_index = pd.RangeIndex(X_arr.shape[0])
        X_df = pd.DataFrame(X_arr, index=sample_index, columns=feature_columns)
        X_df = _align_to_fit_features(X_df, self._feature_names)
        scores = X_df.values @ self._loadings
        return pd.DataFrame(scores, index=X_df.index, columns=self._component_names)

    def fit_transform(self, X: DataMatrix, y: DataMatrix | None = None) -> pd.DataFrame:  # noqa: ARG002
        """Fit the model and return the training scores."""
        self.fit(X)
        return self.scores_

    def predict(self, X: DataMatrix) -> Bunch:
        """Project new data and compute diagnostics (scores, Hotelling's T², SPE).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        result : sklearn.utils.Bunch
            With keys ``scores``, ``hotellings_t2``, ``spe``.
        """
        check_is_fitted(self, "loadings_")
        # predict() delegates to transform() which already runs validate_data;
        # call it once here so the rest of predict can work with the aligned
        # DataFrame view (and so the validate_data error pathways fire on
        # the predict() call directly, not on the recursive transform() one).
        scores = self.transform(X)
        # transform's output is indexed by the validated X's row index;
        # recover an aligned DataFrame for the diagnostics below.
        sample_index = X.index if isinstance(X, pd.DataFrame) else scores.index
        feature_columns = X.columns if isinstance(X, pd.DataFrame) else self._feature_names
        X = pd.DataFrame(np.asarray(X, dtype=float), index=sample_index, columns=feature_columns)
        X = _align_to_fit_features(X, self._feature_names)

        # Hotelling's T² (cumulative)
        component_names = self._component_names
        t2 = pd.DataFrame(np.zeros((X.shape[0], self.n_components)), columns=component_names, index=X.index)
        for a in range(self.n_components):
            t2.iloc[:, a] = (
                t2.iloc[:, max(0, a - 1)] + (scores.iloc[:, a] / self.scaling_factor_for_scores_.iloc[a]) ** 2
            )

        # SPE: residual after reconstruction
        X_hat = scores.values @ self._loadings.T
        residuals = X.values - X_hat
        spe_values = pd.Series(np.sqrt(np.sum(residuals**2, axis=1)), index=X.index, name="SPE")

        return Bunch(scores=scores, hotellings_t2=t2, spe=spe_values)

    def score(self, X: DataMatrix, y: DataMatrix | None = None) -> float:  # noqa: ARG002
        """Negative mean squared reconstruction error (higher is better).

        Follows the sklearn convention where higher scores indicate better
        model fit. This makes PCA compatible with ``cross_val_score``,
        ``GridSearchCV``, and other sklearn model-selection utilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data to score.
        y : ignored

        Returns
        -------
        score : float
            Negative mean squared reconstruction error.

        Examples
        --------
        >>> from sklearn.model_selection import cross_val_score
        >>> scores = cross_val_score(PCA(n_components=2), X_scaled, cv=5)
        >>> print(f"Mean CV score: {scores.mean():.4f}")
        """
        check_is_fitted(self, "loadings_")
        # transform() runs validate_data; build a DataFrame view here for
        # the residual computation that matches its shape.
        scores = self.transform(X)
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        X_hat = scores.values @ self._loadings.T
        residuals = X_arr - X_hat
        return -float(np.mean(residuals**2))

    @classmethod
    def minka_mle(cls, X: DataMatrix) -> int:
        """Minka (2000) automatic-dimensionality estimate for PCA.

        Closed-form Bayesian model selection on the PPCA evidence (Minka,
        T. P. 2000. *Automatic Choice of Dimensionality for PCA*. NIPS 13,
        pp. 598-604). Operates only on the covariance eigenvalues of ``X``
        and is therefore very cheap; in the simulations Minka reports it
        beats cross-validation. Use it alongside the ekf-CV recommendation
        from :meth:`select_n_components` as a fast cross-check.

        Internally ``X`` is mean-centred before estimation (a PPCA
        assumption); it is **not** unit-variance scaled, because dividing
        each column by its standard deviation compresses the noise
        eigenvalues to near-zero values the MLE misreads as additional
        latent signal. If your columns are on wildly different scales,
        pass the analysis-scale ``X`` produced by your own preprocessing
        (e.g. SNV for spectral data) and accept the centring this method
        applies.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data matrix.

        Returns
        -------
        n_components : int
            The MLE estimate of the effective dimensionality.

        References
        ----------
        Minka, T. P. (2000). Automatic Choice of Dimensionality for PCA.
        Advances in Neural Information Processing Systems, 13, 598-604.

        See Also
        --------
        parallel_analysis : Horn (1965) eigenvalue-vs-null retention.
        select_n_components : ekf cross-validation; pass
            ``return_consensus=True`` to report all three side by side.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X_arr = np.asarray(X, dtype=float)
        Xc = X_arr - X_arr.mean(axis=0)
        # ``svd_solver="full"`` is the only solver that supports
        # ``n_components="mle"`` in sklearn.
        sk = _SkPCA(n_components="mle", svd_solver="full")
        sk.fit(Xc)
        return int(sk.n_components_)

    @classmethod
    def parallel_analysis(
        cls,
        X: DataMatrix,
        *,
        n_simulations: int = 200,
        quantile: float = 0.95,
        scale: bool = True,
        random_state: int | None = None,
    ) -> Bunch:
        """Horn (1965) parallel analysis component-count estimate.

        Generates ``n_simulations`` random matrices of the same shape as
        ``X``, computes their eigenvalues, and retains every observed
        component whose eigenvalue exceeds the ``quantile`` of the null
        distribution at the same rank. Widely regarded in psychometrics
        as the best simple retention rule for PCA.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data matrix.
        n_simulations : int, default 200
            Number of random matrices drawn to build the null
            eigenvalue distribution.
        quantile : float, default 0.95
            Quantile of the null eigenvalues used as the retention
            threshold. Horn's original proposal was the mean (0.5);
            the more conservative 95th-percentile threshold is the
            modern recommendation.
        scale : bool, default True
            Mean-centre and unit-variance scale ``X`` before estimation
            (matches :meth:`minka_mle`).
        random_state : int, optional
            Seed for the null-matrix simulations.

        Returns
        -------
        result : sklearn.utils.Bunch
            With keys:

            - ``n_components`` - number of components retained (can be 0
              on pure noise).
            - ``observed_eigenvalues`` - eigenvalues of ``X`` after
              centring/scaling (np.ndarray of length ``min(n, p)``).
            - ``null_threshold`` - per-rank ``quantile`` of the null
              eigenvalue distribution (same length as
              ``observed_eigenvalues``).

        References
        ----------
        Horn, J. L. (1965). A rationale and test for the number of
        factors in factor analysis. *Psychometrika*, 30(2), 179-185.

        See Also
        --------
        minka_mle : closed-form PPCA evidence rule.
        select_n_components : ekf cross-validation; pass
            ``return_consensus=True`` to report all three side by side.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if scale:
            X = MCUVScaler().fit_transform(X)
        X_arr = np.asarray(X, dtype=float)
        n, p = X_arr.shape
        k = min(n, p)

        # Observed eigenvalues from the centred X. Centring removes one DoF
        # so the smallest singular value is at-or-near zero; that's expected.
        Xc = X_arr - X_arr.mean(axis=0)
        _, S, _ = np.linalg.svd(Xc, full_matrices=False)
        observed = (S**2) / max(1, n - 1)

        rng = np.random.default_rng(random_state)
        null_eigs = np.zeros((n_simulations, k))
        for i in range(n_simulations):
            R = rng.standard_normal((n, p))
            Rc = R - R.mean(axis=0)
            _, S_r, _ = np.linalg.svd(Rc, full_matrices=False)
            null_eigs[i, : S_r.shape[0]] = (S_r**2) / max(1, n - 1)

        null_threshold = np.quantile(null_eigs, quantile, axis=0)
        # Standard PA: retain consecutive components from the top while
        # observed > null. Components past the first failure are not
        # retained even if their observed eigenvalue happens to exceed
        # the null (a rare numerical coincidence on real data).
        n_retained = 0
        for obs, thr in zip(observed, null_threshold, strict=False):
            if obs > thr:
                n_retained += 1
            else:
                break

        return Bunch(
            n_components=int(n_retained),
            observed_eigenvalues=observed,
            null_threshold=null_threshold,
        )

    @classmethod
    def select_n_components(  # noqa: PLR0913, PLR0915, C901
        cls,
        X: DataMatrix,
        *,
        max_components: int | None = None,
        cv: int | BaseCrossValidator = 5,
        cv_scheme: typing.Literal["row_wise", "ekf"] = "ekf",
        n_repeats: int = 1,
        selection_rule: SelectionRule = "min",
        min_q2_increase: float = Q2_MIN_INCREMENT,
        scale_inside_folds: bool = True,
        n_iter: int = 50,
        tol: float = 1e-6,
        random_state: int | None = None,
        return_consensus: bool = False,
        threshold: float | None = None,
        **pca_kwargs,
    ) -> Bunch:
        """Select the number of PCA components via cross-validation.

        Evaluates every component count ``1, 2, ..., max_components`` and
        recommends one via the configured ``selection_rule``. The default
        ``cv_scheme="ekf"`` is the element-wise k-fold algorithm of Bro,
        Kjeldahl, Smilde & Kiers (2008, *Anal. Bioanal. Chem.* 390:1241-1251),
        which holds out individual cells of ``X`` and predicts them via
        EM-style imputation from a model that never sees their true values.
        This restores the prediction-independence requirement the legacy
        row-wise scheme violates, fixing the trivial-fit pathology where
        PRESS shrinks monotonically with components.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Should already be on the analysis scale (e.g.
            mean-centred and unit-variance via :class:`MCUVScaler`).
        max_components : int, optional
            Maximum number of components to evaluate. Default is
            ``min(n_samples - 1, n_features)``.
        cv : int or sklearn CV splitter, default 5
            For ``cv_scheme="ekf"``: the integer number of element-folds
            (splitter objects are ignored). For ``cv_scheme="row_wise"``:
            either an integer K (fed to ``KFold``) or any sklearn splitter.
        cv_scheme : {"ekf", "row_wise"}, default "ekf"
            ``"ekf"`` is the element-wise k-fold scheme of Bro et al. 2008
            with EM imputation; the research-recommended default. The legacy
            ``"row_wise"`` scheme is preserved for back-compat but emits a
            :class:`SpecificationWarning` because it over-selects (see the
            warning admonition below).
        n_repeats : int, default 1
            Repeat the ekf pass with a fresh random fold permutation this
            many times. Each repeat covers every cell exactly once;
            ``n_repeats > 1`` narrows the per-component PRESS standard
            error (helpful when the 1-SE rule sits on a borderline) at
            roughly linear extra runtime. Ignored under
            ``cv_scheme="row_wise"``.
        selection_rule : {"min", "1se", "q2_increment"}, default "min"
            How the recommended component count is chosen. ``"min"`` is the
            GlobalMin criterion Bro 2008 pairs with ekf - the component
            count with the lowest pooled PRESS. ``"1se"`` is the one-
            standard-error rule (needs ``per_fold_press``, available under
            both schemes). ``"q2_increment"`` is the Wold's-R-style
            cumulative-:math:`Q^2` threshold from PR #371; ``min_q2_increase``
            sets the threshold.
        min_q2_increase : float, default 0.01
            Threshold used only when ``selection_rule="q2_increment"``.
        scale_inside_folds : bool, default True
            With the default, mean-centring and unit-variance scaling
            constants are fit on each fold's in-fold cells and applied to
            the whole matrix before EM, removing the centring/scaling
            leakage of the prior implementation. Set to ``False`` to
            reproduce the previous behaviour (column mean recomputed each
            EM iteration from the imputed matrix, no scaling); this is
            useful only when ``X`` is already pre-scaled. Ignored under
            ``cv_scheme="row_wise"``.
        n_iter, tol : int and float, default 50 and 1e-6
            EM iteration cap and convergence tolerance for the ekf imputation
            step. Ignored under ``cv_scheme="row_wise"``.
        random_state : int, optional
            Seed for the ekf element-fold permutation.
        threshold : float, optional
            Deprecated. The original Wold PRESS-ratio cutoff. Passing it
            emits a :class:`DeprecationWarning`; the value is ignored. Use
            ``selection_rule="q2_increment"`` (and tune ``min_q2_increase``)
            for a comparable parsimony preference.
        **pca_kwargs
            Additional keyword arguments passed to the ``PCA()`` constructor
            under ``cv_scheme="row_wise"`` (e.g. ``algorithm="nipals"``).
            Ignored under ``cv_scheme="ekf"`` because ekf runs its own SVD
            loop.

        Returns
        -------
        result : sklearn.utils.Bunch
            With keys:

            - ``n_components`` - recommended number of components (int).
            - ``press`` - pooled PRESS per component count (pd.Series,
              indexed ``1..A_max``).
            - ``per_fold_press`` - per-fold PRESS contributions
              (pd.DataFrame, ``A_max`` rows x ``n_folds`` columns).
            - ``se_press`` - standard error of the per-fold PRESS curve
              (pd.Series, indexed ``1..A_max``). Drives the 1-SE rule.
            - ``press_ratio`` - ``PRESS_a / PRESS_{a-1}`` for inspection
              (pd.Series, indexed ``2..A_max``).
            - ``q2`` - cross-validated :math:`R^2_X` per component count
              (pd.Series, indexed ``1..A_max``). Computed as
              ``1 - press / (n_samples * n_features * mean_cell_ss)``,
              so it is directly comparable to ``r2_cumulative_`` and to
              PLS's ``r2y_validated``.
            - ``cv_scores`` - alias of ``per_fold_press`` under ekf, or
              per-fold negative MSE from ``cross_val_score`` under row-wise
              (preserved for back-compat).
            - ``cv_scheme`` - the scheme used (``"ekf"`` or ``"row_wise"``).
            - ``selection_rule`` - the rule used to pick ``n_components``.

        References
        ----------
        Bro, R., Kjeldahl, K., Smilde, A. K., & Kiers, H. A. L. (2008).
        Cross-validation of component models: a critical look at current
        methods. *Anal. Bioanal. Chem.*, 390(5), 1241-1251.

        Camacho, J., & Ferrer, A. (2012). Cross-validation in PCA models
        with the element-wise k-fold (ekf) algorithm: theoretical aspects.
        *J. Chemometrics*, 26(7), 361-373.

        .. warning::

           ``cv_scheme="row_wise"`` is preserved only for back-compat and
           emits a :class:`SpecificationWarning`. It suffers from the
           *trivial-fit* problem: holding out whole rows and projecting them
           back via :meth:`transform` lets the held-out row's own values
           reach its prediction, so PRESS shrinks monotonically with the
           component count and the recommendation tends to run to the
           maximum. Prefer the default ``"ekf"``.
        """
        if threshold is not None:
            warnings.warn(
                "The `threshold` (Wold PRESS-ratio) argument of "
                "PCA.select_n_components is deprecated and ignored; the "
                "recommendation now uses `selection_rule`. Pass "
                "`selection_rule='q2_increment'` (and tune `min_q2_increase`) "
                "for a comparable parsimony preference.",
                DeprecationWarning,
                stacklevel=2,
            )

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        N, K = X.shape
        if max_components is None:
            max_components = min(N - 1, K)
        max_components = min(int(max_components), N - 1, K)
        if max_components < 1:
            raise ValueError("No components can be evaluated; the data is too small.")

        component_index = pd.Index(range(1, max_components + 1), name="n_components")
        X_arr = np.asarray(X, dtype=float)

        if cv_scheme == "ekf":
            n_folds = cv if isinstance(cv, int) else 5
            if n_repeats < 1:
                raise ValueError(f"n_repeats must be >= 1; got {n_repeats}.")
            press_arr, per_fold_press_arr = _pca_ekf_press(
                X_arr,
                max_components,
                n_folds=n_folds,
                n_repeats=n_repeats,
                n_iter=n_iter,
                tol=tol,
                scale_inside_folds=scale_inside_folds,
                random_state=random_state,
            )
            press = pd.Series(press_arr, index=component_index, name="PRESS")
            per_fold_press = pd.DataFrame(
                per_fold_press_arr,
                index=component_index,
                columns=[
                    f"fold_{i + 1}" for i in range(per_fold_press_arr.shape[1])
                ],
            )
            cv_scores = per_fold_press
            # Q^2 normalisation: total sum-of-squares of X (the null-model
            # "predict the column mean" reference, which on mean-centred data
            # is the total variance). Bro 2008's q^2_x normalisation.
            null_model_ss = float(np.nansum(X_arr**2))
            q2 = 1.0 - press / null_model_ss if null_model_ss > epsqrt else press * np.nan
        elif cv_scheme == "row_wise":
            warnings.warn(
                "cv_scheme='row_wise' uses the legacy whole-row CV scheme that "
                "Bro et al. 2008 flagged as invalid: held-out row values flow "
                "back through transform() into their own prediction, so PRESS "
                "shrinks monotonically and the recommendation tends to run to "
                "the maximum component count. Prefer cv_scheme='ekf' (the new "
                "default).",
                SpecificationWarning,
                stacklevel=2,
            )
            press_values = {}
            all_cv_scores = {}
            for a in range(1, max_components + 1):
                scores_a = cross_val_score(cls(n_components=a, **pca_kwargs), X, cv=cv)
                all_cv_scores[a] = scores_a
                press_values[a] = -scores_a.mean()
            press = pd.Series(press_values, name="PRESS", index=component_index)
            cv_scores = pd.DataFrame(all_cv_scores).T
            cv_scores.index = component_index
            cv_scores.columns = [f"fold_{i + 1}" for i in range(cv_scores.shape[1])]
            # The row_wise per-fold values are negative MSE means, not raw
            # PRESS contributions, so don't pretend they are. Build a parallel
            # per_fold_press by undoing the negation.
            per_fold_press = -cv_scores
            null_model_ss = float(np.nanmean(X_arr**2))
            q2 = 1.0 - press / null_model_ss if null_model_ss > epsqrt else press * np.nan
        else:
            raise ValueError(
                f"Unknown cv_scheme {cv_scheme!r}; expected 'ekf' or 'row_wise'."
            )

        q2 = q2.rename("Q2")
        q2.index = component_index

        # PRESS ratio: still computable under either scheme, kept for inspection.
        ratio_values = {a: press[a] / press[a - 1] for a in range(2, max_components + 1)}
        press_ratio = pd.Series(ratio_values, name="PRESS ratio")
        press_ratio.index.name = "n_components"

        # Per-component standard error across folds.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            per_fold_arr = per_fold_press.to_numpy()
            n_folds_per_a = np.maximum(1, np.sum(~np.isnan(per_fold_arr), axis=1))
            se_values = np.nanstd(per_fold_arr, axis=1, ddof=1) / np.sqrt(n_folds_per_a)
        se_press = pd.Series(se_values, index=component_index, name="SE(PRESS)")

        press_arr_final = press.to_numpy()
        if np.all(np.isnan(press_arr_final)):
            raise RuntimeError(
                "Cross-validation produced NaN PRESS for every component count; "
                "no recommendation can be made."
            )
        recommended = _select_n_components(
            selection_rule,
            mean_error=press_arr_final,
            se_error=se_values,
            q2_cumulative=q2.to_numpy(),
            min_q2_increase=min_q2_increase,
        )

        consensus_fields: dict[str, object] = {}
        if return_consensus:
            # Two cheap cross-checks: Minka's PPCA MLE (mean-centred input,
            # no unit-variance scaling) and Horn's parallel analysis (which
            # accepts the same scaling convention as the rest of the method).
            minka_n = cls.minka_mle(X)
            pa_result = cls.parallel_analysis(
                X, scale=scale_inside_folds, random_state=random_state
            )
            counts = (int(recommended), int(minka_n), int(pa_result.n_components))
            consensus = "agree" if max(counts) - min(counts) <= 1 else "disagree"
            consensus_fields = {
                "minka_n_components": int(minka_n),
                "parallel_analysis_n_components": int(pa_result.n_components),
                "consensus": consensus,
                "consensus_counts": counts,
            }

        return Bunch(
            n_components=recommended,
            press=press,
            per_fold_press=per_fold_press,
            se_press=se_press,
            press_ratio=press_ratio,
            q2=q2,
            cv_scores=cv_scores,
            cv_scheme=cv_scheme,
            selection_rule=selection_rule,
            **consensus_fields,
        )

    def score_contributions(
        self,
        t_start: np.ndarray | pd.Series,
        t_end: np.ndarray | pd.Series | None = None,
        components: list[int] | None = None,
        *,
        weighted: bool = False,
    ) -> pd.Series:
        """Contribution of each variable to a score-space movement.

        Decomposes the difference (t_end - t_start) back into variable space
        via the loadings for the selected components.

        Parameters
        ----------
        t_start : array-like of shape (n_components,)
            Score vector of the observation of interest. Can be a row from
            ``self.scores_`` or from ``predict(X_new).scores``.
        t_end : array-like of shape (n_components,), optional
            Reference point in score space. Default is the model center
            (a vector of zeros), which is the most common choice.
        components : list of int, optional
            **1-based** component indices to decompose over, matching the
            model's column convention. Examples: ``[2, 3]`` for a PC2-vs-PC3
            score plot, or ``None`` (default) for all components - appropriate
            for Hotelling's T² contributions.
        weighted : bool, default False
            If True, scale the score difference by 1/sqrt(explained_variance)
            per component before back-projecting. This gives contributions to
            the T² statistic rather than to the Euclidean score distance.

        Returns
        -------
        contributions : pd.Series of shape (n_features,)
            One value per variable; sign indicates direction. Index contains
            the variable (feature) names.

        Examples
        --------
        >>> pca = PCA(n_components=3).fit(X_scaled)
        >>> # Why does observation 0 differ from the model center?
        >>> contrib = pca.score_contributions(pca.scores_.iloc[0].values)
        >>> # Which variables drive the difference between obs 0 and obs 5?
        >>> contrib = pca.score_contributions(
        ...     pca.scores_.iloc[0].values, pca.scores_.iloc[5].values
        ... )

        See Also
        --------
        observation_contributions : The per-observation counterpart. Despite
            the similar name, the two answer different questions and are not
            interchangeable. ``score_contributions`` is *per-variable*: it
            decomposes a single observation's movement in score space back
            onto the original variables, answering "which **variables**
            explain why this observation sits where it does?". It returns one
            signed value per variable. ``observation_contributions`` is
            *per-observation*: it reports each observation's share of a
            component's total inertia, answering "which **observations** most
            strongly shape this component?". It returns a non-negative
            sample-by-component table whose columns each sum to 1. The two are
            orthogonal views of the same score matrix - one decomposes across
            variables, the other across observations.
        """
        check_is_fitted(self, "loadings_")
        t_start = np.asarray(t_start, dtype=float)
        t_end = np.zeros(self.n_components) if t_end is None else np.asarray(t_end, dtype=float)

        idx = (
            np.arange(self.n_components) if components is None else np.array(components) - 1
        )  # convert 1-based to 0-based

        dt = t_end[idx] - t_start[idx]

        if weighted:
            # ``explained_variance_[a] == 0`` for a degenerate component
            # would silently produce inf/NaN weighted contributions.
            # Clamp the divisor so weighting is a no-op on such components.
            # SEC-21 (#270) sub-item 5.
            ev = np.asarray(self.explained_variance_)[idx]
            dt = dt / np.sqrt(np.where(ev > epsqrt, ev, 1.0))

        P = self._loadings[:, idx].T  # (len(idx), n_features)
        contributions = dt @ P  # (n_features,)

        return pd.Series(contributions, index=self._feature_names, name="score_contributions")

    def detect_outliers(self, conf_level: float = 0.95) -> list[dict]:
        """Detect outlier observations using SPE and Hotelling's T² diagnostics.

        Combines two approaches:

        1. **Statistical limits** - observations exceeding the SPE or T² limit
           at ``conf_level`` are flagged.
        2. **Robust ESD test** - the generalized ESD test (with robust median/MAD
           variant) identifies observations that are unusual *relative to the
           rest of the data*, even if they fall below the statistical limit.

        An observation can be flagged for one or both reasons.

        Parameters
        ----------
        conf_level : float, default 0.95
            Confidence level in [0.8, 0.999]. Controls both the statistical
            limits and the ESD test's significance level (alpha = 1 - conf_level).

        Returns
        -------
        outliers : list of dict
            Sorted from most severe to least. Each dict contains:

            - ``observation`` - index label of the observation
            - ``outlier_types`` - list of ``"spe"`` and/or ``"hotellings_t2"``
            - ``spe`` - SPE value for this observation
            - ``hotellings_t2`` - T² value for this observation
            - ``spe_limit`` - SPE limit at the given confidence level
            - ``hotellings_t2_limit`` - T² limit at the given confidence level
            - ``severity`` - max(spe/spe_limit, t2/t2_limit)

        Examples
        --------
        >>> pca = PCA(n_components=3).fit(X_scaled)
        >>> outliers = pca.detect_outliers(conf_level=0.95)
        >>> for o in outliers:
        ...     print(f"{o['observation']}: {o['outlier_types']} (severity={o['severity']})")
        """
        check_is_fitted(self, "spe_")
        if not (0.8 <= conf_level <= 0.999):
            raise ValueError(f"conf_level must be between 0.8 and 0.999, got {conf_level}.")

        N = self.n_samples_

        # Full-model SPE and cumulative T² (last column)
        spe_values = self.spe_.iloc[:, -1]
        t2_values = self.hotellings_t2_.iloc[:, -1]

        # Statistical limits
        spe_lim = self.spe_limit(conf_level=conf_level)
        t2_lim = self.hotellings_t2_limit(conf_level=conf_level)

        # Robust ESD outlier detection on each series
        max_outliers = max(1, N // 5)
        alpha = 1 - conf_level

        spe_outlier_idx, _ = detect_outliers_esd(
            spe_values.to_numpy(), algorithm="esd", max_outliers_detected=max_outliers, alpha=alpha
        )
        t2_outlier_idx, _ = detect_outliers_esd(
            t2_values.to_numpy(), algorithm="esd", max_outliers_detected=max_outliers, alpha=alpha
        )

        # Collect all flagged observations: ESD outliers + above-limit
        spe_flagged = set(spe_outlier_idx)
        t2_flagged = set(t2_outlier_idx)

        # Also flag any observation above the statistical limit
        for i in range(N):
            if spe_values.iloc[i] > spe_lim:
                spe_flagged.add(i)
            if t2_values.iloc[i] > t2_lim:
                t2_flagged.add(i)

        # Merge into result dicts
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
