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
from sklearn.model_selection import cross_val_score
from sklearn.utils import Bunch
from sklearn.utils.validation import check_is_fitted

from ..univariate.metrics import detect_outliers_esd
from ._base import _LatentVariableModel, _LazyFrame
from ._common import DataMatrix, SpecificationWarning, epsqrt
from ._nipals import quick_regress, ssq, terminate_check

logger = logging.getLogger(__name__)


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
            Training data. May contain NaN values for missing data.
        y : ignored

        Returns
        -------
        self : PCA
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        N, K = X.shape
        self.n_samples_ = N
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
            self._r2cum_np[a] = 1 - sum(row_ssx) / base_variance if base_variance > 0 else np.nan
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

            if sum(start_ss_col) < epsqrt:
                emsg = (
                    "There is no variance left in the data array: cannot "
                    f"compute any more components beyond component {a}."
                )
                raise RuntimeError(emsg)

            # Pick a column from X as the initial guess.
            # ``Xd[:, [0]]`` (fancy indexing) already returns a copy in
            # current numpy, so the in-place ``isnan -> 0`` does not
            # poison Xd today. The explicit ``.copy()`` here is
            # defensive: it mirrors the PLS path (~line 1527) and
            # protects against any future numpy change that flips
            # fancy indexing to a view-returning variant. SEC-21 (#270)
            # sub-item 2.
            t_a_guess = Xd[:, [0]].copy()
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

            self.fitting_info_["timing"][a] = time.time() - start_time
            self.fitting_info_["iterations"][a] = itern
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
            self._r2cum_np[a] = 1 - sum(row_ssx) / base_variance if base_variance > 0 else np.nan
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
            New data to project. Must have the same number of features as the training data.

        Returns
        -------
        scores : pd.DataFrame of shape (n_samples, n_components)
        """
        check_is_fitted(self, "loadings_")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"New data must have {self.n_features_in_} columns, got {X.shape[1]}."
            )
        scores = X.values @ self._loadings
        return pd.DataFrame(scores, index=X.index, columns=self._component_names)

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
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Prediction data must have {self.n_features_in_} columns, got {X.shape[1]}."
            )

        scores = self.transform(X)

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
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        scores = self.transform(X)
        X_hat = scores.values @ self._loadings.T
        residuals = X.values - X_hat
        return -float(np.mean(residuals**2))

    @classmethod
    def select_n_components(
        cls,
        X: DataMatrix,
        *,
        max_components: int | None = None,
        cv: int = 5,
        threshold: float = 0.95,
        **pca_kwargs,
    ) -> Bunch:
        """Select the number of components via PRESS cross-validation.

        Fits PCA models with 1, 2, ..., ``max_components`` components,
        evaluates each with K-fold cross-validation, and recommends the
        optimal number using Wold's criterion: stop adding components when
        PRESS_a / PRESS_{a-1} > ``threshold``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        max_components : int, optional
            Maximum number of components to evaluate. Default is
            ``min(n_samples - 1, n_features)``.
        cv : int or sklearn CV splitter, default 5
            Number of cross-validation folds, or an sklearn splitter object
            (e.g., ``KFold(n_splits=10, shuffle=True)``).
        threshold : float, default 0.95
            Wold's criterion threshold. If PRESS_a / PRESS_{a-1} exceeds
            this value, component ``a`` is deemed not significant. Lower
            values are more aggressive (fewer components).
        **pca_kwargs
            Additional keyword arguments passed to the ``PCA()`` constructor
            (e.g., ``algorithm="nipals"`` for data with missing values).

        Returns
        -------
        result : sklearn.utils.Bunch
            With keys:

            - ``n_components`` - recommended number of components (int)
            - ``press`` - PRESS per component count (pd.Series, indexed 1..A_max)
            - ``press_ratio`` - PRESS_a / PRESS_{a-1} (pd.Series, indexed 2..A_max)
            - ``cv_scores`` - per-fold scores (pd.DataFrame, A_max rows x cv cols)
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        N, K = X.shape
        if max_components is None:
            max_components = min(N - 1, K)
        max_components = min(max_components, N - 1, K)

        press_values = {}
        all_cv_scores = {}

        for a in range(1, max_components + 1):
            scores_a = cross_val_score(cls(n_components=a, **pca_kwargs), X, cv=cv)
            all_cv_scores[a] = scores_a
            press_values[a] = -scores_a.mean()  # undo negation from score()

        press = pd.Series(press_values, name="PRESS")
        press.index.name = "n_components"

        # Wold's criterion: ratio of consecutive PRESS values
        ratio_values = {a: press[a] / press[a - 1] for a in range(2, max_components + 1)}
        press_ratio = pd.Series(ratio_values, name="PRESS ratio")
        press_ratio.index.name = "n_components"

        # Recommend: last component where ratio <= threshold
        recommended = 1
        for a in range(2, max_components + 1):
            if press_ratio[a] <= threshold:
                recommended = a
            else:
                break

        cv_scores = pd.DataFrame(all_cv_scores).T
        cv_scores.index.name = "n_components"
        cv_scores.columns = [f"fold_{i + 1}" for i in range(cv_scores.shape[1])]

        return Bunch(
            n_components=recommended,
            press=press,
            press_ratio=press_ratio,
            cv_scores=cv_scores,
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
            spe_values.values, algorithm="esd", max_outliers_detected=max_outliers, alpha=alpha
        )
        t2_outlier_idx, _ = detect_outliers_esd(
            t2_values.values, algorithm="esd", max_outliers_detected=max_outliers, alpha=alpha
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
