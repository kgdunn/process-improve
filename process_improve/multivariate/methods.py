# (c) Kevin Dunn, 2010-2025. MIT License. Based on own private work over the years.
from __future__ import annotations

import time
import typing
import warnings
from collections.abc import Callable, KeysView
from functools import partial
from typing import TypeAlias

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from scipy.stats import chi2, f
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, _fit_context
from sklearn.cross_decomposition import PLSRegression as PLS_sklearn
from sklearn.decomposition import PCA as PCA_sklearn
from sklearn.metrics import r2_score
from sklearn.utils import Bunch
from sklearn.utils.validation import check_array, check_is_fitted

from .plots import loading_plot, score_plot, spe_plot, t2_plot

DataMatrix: TypeAlias = np.ndarray | pd.DataFrame

epsqrt = np.sqrt(np.finfo(float).eps)


class SpecificationWarning(UserWarning):
    """Parent warning class."""


class MCUVScaler(BaseEstimator, TransformerMixin):
    """
    Create our own mean centering and scaling to unit variance (MCUV) class
    The default scaler in sklearn does not handle small datasets accurately, with ddof.
    """

    def __init__(self):
        pass

    def fit(self, X: DataMatrix) -> MCUVScaler:
        """Get the centering and scaling object constants."""
        self.center_ = pd.DataFrame(X).mean()
        # this is the key difference with "preprocessing.StandardScaler"
        self.scale_ = pd.DataFrame(X).std(ddof=1)
        self.scale_[self.scale_ == 0] = 1.0  # columns with no variance are left as-is.
        return self

    def transform(self, X: DataMatrix) -> pd.DataFrame:
        """Do work of the transformation."""
        check_is_fitted(self, "center_")
        check_is_fitted(self, "scale_")

        X = pd.DataFrame(X).copy()
        return (X - self.center_) / self.scale_

    def inverse_transform(self, X: DataMatrix) -> pd.DataFrame:
        """Do the inverse transformation."""
        check_is_fitted(self, "center_")
        check_is_fitted(self, "scale_")

        X = pd.DataFrame(X).copy()
        return X * self.scale_ + self.center_


class PCA(PCA_sklearn):
    def __init__(  # noqa: PLR0913
        self,
        n_components: int,
        *,
        copy: bool = True,
        whiten: bool = False,
        svd_solver: str = "auto",
        tol: float = 0.0,
        iterated_power: str = "auto",
        random_state: int | None = None,
        # Own extra inputs, for the case when there is missing data
        missing_data_settings: dict | None = None,
    ):
        super().__init__(
            n_components=n_components,
            copy=copy,
            whiten=whiten,
            svd_solver=svd_solver,
            tol=tol,
            iterated_power=iterated_power,
            random_state=random_state,
        )
        self.n_components: int = n_components
        self.missing_data_settings = missing_data_settings
        self.has_missing_data = False

    def fit(self, X: DataMatrix, y: DataMatrix | None = None) -> PCA_sklearn:  # noqa: ARG002, PLR0915
        """
        Fit a principal component analysis (PCA) model to the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples (rows)
            and `n_features` is the number of features (columns).
        y : Not used; by default None

        Returns
        -------
        PCA
            Model object.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.N, self.K = X.shape

        # Check if number of components is supported against maximum requested
        min_dim = int(min(self.N, self.K))
        self.A: int = min_dim if self.n_components is None else int(self.n_components)

        if min_dim < self.A:
            warn = (
                "The requested number of components is more than can be "
                "computed from data. The maximum number of components is "
                f"the minimum of either the number of rows ({self.N}) or "
                f"the number of columns ({self.K})."
            )
            warnings.warn(warn, SpecificationWarning, stacklevel=2)
            self.A = self.n_components = min_dim

        if np.any(X.isna()):
            # If there are missing data, then the missing data settings apply. Defaults are:
            #
            #       md_method = "pmp"
            #       md_tol = np.sqrt(np.finfo(float).eps)
            #       md_max_iter = 1000

            default_mds = dict(md_method="tsr", md_tol=epsqrt, md_max_iter=1000)
            if isinstance(self.missing_data_settings, dict):
                default_mds.update(self.missing_data_settings)

            self.missing_data_settings = default_mds
            self = PCA_missing_values(
                n_components=self.n_components,
                missing_data_settings=self.missing_data_settings,
            )
            self.N, self.K = X.shape
            self.A = self.n_components
            self.fit(X)

        else:
            self = super().fit(X)
            self.x_loadings = self.components_.T
            self.extra_info = {}
            self.extra_info["timing"] = np.zeros((1, self.A)) * np.nan
            self.extra_info["iterations"] = np.zeros((1, self.A)) * np.nan

        # We have now fitted the model. Apply some convenience shortcuts for the user.
        self.A = self.n_components
        self.N = self.n_samples_
        self.K = self.n_features_in_

        self.loadings = pd.DataFrame(self.x_loadings.copy())
        self.loadings.index = X.columns

        component_names = [a + 1 for a in range(self.A)]
        self.loadings.columns = component_names

        self.scaling_factor_for_scores = pd.Series(
            np.sqrt(self.explained_variance_),
            index=component_names,
            name="Standard deviation per score",
        )
        self.hotellings_t2 = pd.DataFrame(
            np.zeros(shape=(self.N, self.A)),
            columns=component_names,
            index=X.index,
            # name="Hotelling's T^2 statistic, per component",
        )
        if self.has_missing_data:
            self.x_scores = pd.DataFrame(self.x_scores_, columns=component_names, index=X.index)
            self.squared_prediction_error = pd.DataFrame(
                self.squared_prediction_error_,
                columns=component_names,
                index=X.index.copy(),
            )
            self.R2 = pd.Series(
                self.R2_,
                index=component_names,
                name="Model's R^2, per component",
            )
            self.R2cum = pd.Series(
                self.R2cum_,
                index=component_names,
                name="Cumulative model's R^2, per component",
            )
            self.R2X_cum = pd.DataFrame(
                self.R2X_cum_,
                columns=component_names,
                index=X.columns,
                # name ="Per variable R^2, per component"
            )
        else:
            self.x_scores = pd.DataFrame(super().fit_transform(X), columns=component_names, index=X.index)
            self.squared_prediction_error = pd.DataFrame(
                np.zeros((self.N, self.A)),
                columns=component_names,
                index=X.index.copy(),
            )
            self.R2 = pd.Series(
                np.zeros(shape=(self.A,)),
                index=component_names,
                name="Model's R^2, per component",
            )
            self.R2cum = pd.Series(
                np.zeros(shape=(self.A,)),
                index=component_names,
                name="Cumulative model's R^2, per component",
            )
            self.R2X_cum = pd.DataFrame(
                np.zeros(shape=(self.K, self.A)),
                columns=component_names,
                index=X.columns,
                # name ="Per variable R^2, per component"
            )

        if not self.has_missing_data:
            xd = X.copy()
            prior_ssx_col = ssq(xd.values, axis=0)
            base_variance = np.sum(prior_ssx_col)
            for a in range(self.A):
                xd -= self.x_scores.iloc[:, [a]] @ self.loadings.iloc[:, [a]].T
                # These are the Residual Sums of Squares (RSS); i.e X-X_hat
                row_ssx = ssq(xd.values, axis=1)
                col_ssx = ssq(xd.values, axis=0)

                # Don't use a check correction factor. Define SPE simply as the sum of squares of
                # the errors, then take the square root, so it is interpreted like a standard error.
                # If the user wants to normalize it, then this is a clean base value to start from.
                self.squared_prediction_error.iloc[:, a] = np.sqrt(row_ssx)

                # TODO: some entries in prior_SSX_col can be zero and leads to nan's in R2X_cum
                self.R2X_cum.iloc[:, a] = 1 - col_ssx / prior_ssx_col

                # R2 and cumulative R2 value for the whole block
                self.R2cum.iloc[a] = 1 - sum(row_ssx) / base_variance
                if a > 0:
                    self.R2.iloc[a] = self.R2cum.iloc[a] - self.R2cum.iloc[a - 1]
                else:
                    self.R2.iloc[a] = self.R2cum.iloc[a]
        # end: has no missing data

        for a in range(self.A):
            self.hotellings_t2.iloc[:, a] = (
                self.hotellings_t2.iloc[:, max(0, a - 1)]
                + (self.x_scores.iloc[:, a] / self.scaling_factor_for_scores.iloc[a]) ** 2
            )

        # Replace `self.loadings` with self.x_loadings
        self.x_loadings = self.loadings
        self.ellipse_coordinates = partial(
            ellipse_coordinates,
            n_components=self.n_components,
            scaling_factor_for_scores=self.scaling_factor_for_scores,
            n_rows=self.N,
        )

        self.hotellings_t2_limit = partial(hotellings_t2_limit, n_components=self.n_components, n_rows=self.N)
        self.spe_plot = partial(spe_plot, model=self)
        self.t2_plot = partial(t2_plot, model=self)
        self.loading_plot = partial(loading_plot, model=self, loadings_type="p")
        self.score_plot = partial(score_plot, model=self)
        self.spe_limit = partial(spe_limit, model=self)

        return self

    def fit_transform(self, X: DataMatrix, y: DataMatrix | None = None) -> None:  # noqa: ARG002
        """Fit the PCA model and transform the data."""
        self.fit(X)
        pytest.fail("Still do the transform part")

    def predict(self, X: DataMatrix):
        """Use the PCA model on new data coming in matrix X."""

        class State:
            """Class to hold the prediction results together."""

            def __init__(self):
                self.N = None
                self.K = None
                self.x_scores = None
                self.hotellings_t2 = None
                self.squared_prediction_error = None

        state = State()
        state.N, state.K = X.shape
        assert self.K == state.K, "Prediction data must have same number of columns as training data."

        state.x_scores = X @ self.x_loadings

        # TODO: handle the missing data version here still
        for _ in range(self.A):
            pass
            # p = self.x_loadings.iloc[:, [a]]
            # temp = X @ self.x_loadings.iloc[:, [a]]
            # X_mcuv -= temp @ p.T
            # state.x_scores[:, [a]] = temp

        # Scores are calculated, now do the rest
        state.hotellings_t2 = np.sum(np.power((state.x_scores / self.scaling_factor_for_scores.values), 2), 1)
        # Calculate SPE-residuals (sum over rows of the errors)
        X_mcuv = X.copy()
        X_mcuv -= state.x_scores @ self.x_loadings.T
        state.squared_prediction_error = np.sqrt(np.power(X_mcuv, 2).sum(axis=1))
        return state


class PCA_missing_values(BaseEstimator, TransformerMixin):  # noqa: N801
    """
    Create a PCA class if there are 1 or more missing data values in the X input array.

    The default method to impute missing values is the TSR algorithm (`md_method="tsr"`).

    Missing data method options are:

    * 'pmp'         Projection to Model Plane
    * 'scp'         Single Component Projection
    * 'nipals'      Same as 'scp': non-linear iterative partial least squares.
    * 'tsr':        See papers by Abel Folch-Fortuny and also DOI: 10.1002/cem.750
    """

    valid_md_methods: typing.ClassVar[list[str]] = ["pmp", "scp", "nipals", "tsr"]

    def __init__(
        self,
        n_components: int,
        missing_data_settings: dict,
    ):
        self.n_components = n_components
        self.missing_data_settings = missing_data_settings
        self.has_missing_data = True
        self.missing_data_settings["md_max_iter"] = int(self.missing_data_settings["md_max_iter"])

        assert self.missing_data_settings["md_tol"] < 10, "Tolerance should not be too large"
        assert self.missing_data_settings["md_tol"] > epsqrt**1.95, "Tolerance must exceed machine precision"

        assert self.missing_data_settings["md_method"] in self.valid_md_methods, (
            f"Missing data method is not recognized. Must be one of {self.valid_md_methods}.",
        )

    def fit(self, X: DataMatrix, y: DataMatrix | None = None) -> PCA_missing_values:  # noqa: ARG002
        """Fit the PCA model with missing data."""
        # Force input to NumPy array:
        self.data = np.asarray(X.copy())

        # Other setups:
        self.n_components = self.A
        self.n_samples_ = self.N
        self.n_features_in_ = self.K

        self.x_loadings = np.zeros((self.K, self.A))
        self.x_scores_ = np.zeros((self.N, self.A))
        self.R2_ = np.zeros(shape=(self.A,))
        self.R2cum_ = np.zeros(shape=(self.A,))
        self.R2X_cum_ = np.zeros(shape=(self.K, self.A))
        self.squared_prediction_error_ = np.zeros((self.N, self.A))

        # Perform MD algorithm here
        if self.missing_data_settings["md_method"].lower() == "pmp":
            pytest.fail("The PMP method is not implemented yet")  # self._fit_pmp(X)
        elif self.missing_data_settings["md_method"].lower() in ["scp", "nipals"]:
            self._fit_nipals_pca(settings=self.missing_data_settings)
        elif self.missing_data_settings["md_method"].lower() in ["tsr"]:
            self._fit_tsr_pca(settings=self.missing_data_settings)

        # These fields must be set by the MD algorithm:
        required_fields = [
            "extra_info",
            "x_scores_",
            "x_loadings",
            "R2_",
            "R2cum_",
            "R2X_cum_",
            "squared_prediction_error_",
        ]
        assert all(getattr(self, attr, None) is not None for attr in required_fields)

        # Additional calculations, which can be done after the missing data method is complete.
        self.explained_variance_ = np.diag(self.x_scores_.T @ self.x_scores_) / (self.N - 1)
        return self

    def transform(self, X: DataMatrix) -> DataMatrix:
        """Transform the data."""
        check_is_fitted(self, "blah")

        return X.copy()

    def inverse_transform(self, X: DataMatrix) -> DataMatrix:
        """Inverse transform the data."""
        check_is_fitted(self, "blah")

        return X.copy()

    def _fit_nipals_pca(self, settings: dict) -> None:
        """Fit the PCA model using the NIPALS algorithm (internal method)."""
        # NIPALS algorithm
        K, A = self.K, self.A

        # Create direct links to the data
        Xd = np.asarray(self.data)
        base_variance = ssq(Xd)

        # Initialize storage:
        self.extra_info = {}
        self.extra_info["timing"] = np.zeros(A) * np.nan
        self.extra_info["iterations"] = np.zeros(A) * np.nan

        for a in np.arange(A):
            # Timers and housekeeping
            start_time = time.time()
            itern = 0
            start_SS_col = ssq(Xd, axis=0)

            if sum(start_SS_col) < epsqrt:
                emsg = (
                    "There is no variance left in the data array: cannot "
                    f"compute any more components beyond component {a}."
                )
                raise RuntimeError(emsg)

            # Initialize t_a with random numbers, or carefully select a column
            # from X. <-- Don't do this anymore.
            # Rather: Pick a column from X as the initial guess instead.
            t_a_guess = Xd[:, [0]]  # .reshape(self.N, 1)
            t_a_guess[np.isnan(t_a_guess)] = 0
            t_a = t_a_guess + 1.0
            p_a = np.zeros((K, 1))
            while not (terminate_check(t_a_guess, t_a, iterations=itern, settings=settings)):
                # 0: Richardson's acceleration, or any numerical acceleration
                #    method for PCA where there is slow convergence?

                # 0: starting point for convergence checking on next loop
                t_a_guess = t_a.copy()

                # 1: Regress the score, t_a, onto every column in X, compute the
                #    regression coefficient and store in p_a
                # p_a = X.T * t_a / (t_a.T * t_a)
                # p_a = (X.T)(t_a) / ((t_a.T)(t_a))
                # p_a = np.dot(X.T, t_a) / ssq(t_a)
                p_a = quick_regress(Xd, t_a)

                # 2: Normalize p_a to unit length
                p_a /= np.sqrt(ssq(p_a))

                # 3: Now regress each row in X on the p_a vector, and store the
                #    regression coefficient in t_a
                # t_a = X * p_a / (p_a.T * p_a)
                # t_a = (X)(p_a) / ((p_a.T)(p_a))
                # t_a = np.dot(X, p_a) / ssq(p_a)
                t_a = quick_regress(Xd, p_a)

                itern += 1

            self.extra_info["timing"][a] = time.time() - start_time
            self.extra_info["iterations"][a] = itern

            # Loop terminated!  Now deflate the X-matrix
            Xd -= np.dot(t_a, p_a.T)
            # These are the Residual Sums of Squares (RSS); i.e X-X_hat
            row_SSX = ssq(Xd, axis=1)
            col_SSX = ssq(Xd, axis=0)

            self.squared_prediction_error_[:, a] = np.sqrt(row_SSX)

            # TODO: some entries in start_SS_col can be zero and leads to nan's in R2X_cum
            self.R2X_cum_[:, a] = 1 - col_SSX / start_SS_col

            # R2 and cumulative R2 value for the whole block
            self.R2cum_[a] = 1 - sum(row_SSX) / base_variance
            if a > 0:
                self.R2_[a] = self.R2cum_[a] - self.R2cum_[a - 1]
            else:
                self.R2_[a] = self.R2cum_[a]

            # VIP value (only calculated for X-blocks); only last column is useful
            # self.VIP_a = np.zeros((self.K, self.A))
            # self.VIP = np.zeros(self.K)

            # Store results
            # -------------
            # Flip the signs of the column vectors in P so that the largest
            # magnitude element is positive
            # (Wold, Esbensen, Geladi, PCA,  CILS, 1987, p 42)
            # http://dx.doi.org/10.1016/0169-7439(87)80084-9
            max_el_idx = np.argmax(np.abs(p_a))
            if np.sign(p_a[max_el_idx]) < 1:
                p_a *= -1.0
                t_a *= -1.0

            # Store the loadings and scores
            self.x_loadings[:, a] = p_a.flatten()
            self.x_scores_[:, a] = t_a.flatten()

        # end looping on A components

    def _fit_tsr_pca(self, settings: dict) -> None:
        start_time = time.time()
        self.extra_info = dict(iterations=0, timing=0)
        delta = 1e100
        Xd = np.asarray(self.data.copy())
        base_variance = ssq(Xd)

        N, K, A = self.N, self.K, self.A
        mmap = np.isnan(Xd)
        # Could impute missing values per row, but leave it at zero, assuming the data are MCUV.
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

            V = V.T[:, 0:A]  # transpose first
            for n in range(N):
                # If there are missing values (mis) in the n-th row. Compared to observed values.
                row_mis = mmap[n, :]
                row_obs = ~row_mis
                if np.any(row_mis):
                    # Form the so-called key-matrix, L
                    L = V[row_obs, 0 : min(A, sum(row_obs))]
                    S11 = S[row_obs, :][:, row_obs]
                    S21 = S[row_mis, :][:, row_obs]
                    z2 = (S21 @ L) @ np.linalg.pinv(L.T @ S11 @ L) @ L.T
                    Xc[n, row_mis] = z2 @ Xc[n, row_obs]
            Xd = Xc + mean_X
            delta = np.mean((Xd[mmap] - missing_X) ** 2)

        # All done: return the results in `self`
        S = np.cov(Xd, rowvar=False, ddof=1)
        _, _, V = np.linalg.svd(S, full_matrices=False)
        self.extra_info["iterations"] = itern
        self.extra_info["timing"] = time.time() - start_time
        self.x_loadings = (V[0:A, :]).T  # transpose result to the right shape: K x A
        self.x_scores_ = (Xd - np.mean(Xd, axis=0)) @ self.x_loadings

        for a in range(A):
            residuals = self.x_scores_[:, : a + 1] @ self.x_loadings[:, : a + 1].T - self.data
            self.R2cum_[a] = 1 - ssq(residuals, axis=None) / base_variance
            if a > 0:
                self.R2_[a] = self.R2cum_[a] - self.R2cum_[a - 1]
            else:
                self.R2_[a] = self.R2cum_[a]

            # Per component, what is the SPE? We define SPE as a standard deviation like quantity.
            # So take the square root of the sum of squares of the residuals.
            self.squared_prediction_error_[:, a] = np.sqrt(ssq(residuals, axis=1))  # N x A matrix


class PLS(PLS_sklearn):
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
        super().__init__(
            n_components=n_components,
            scale=scale,
            max_iter=max_iter,
            tol=tol,
            copy=copy,
        )
        self.n_components: int = n_components
        self.missing_data_settings = missing_data_settings
        self.has_missing_data = False

    def fit(self, X: DataMatrix, Y: DataMatrix) -> PLS_sklearn:  # noqa: PLR0915
        """
        Fit a projection to latent structures (PLS) or Partial Least Square (PLS) model to the data.

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
        self.N: int = X.shape[0]
        self.K: int = X.shape[1]
        self.Ny: int = Y.shape[0]
        self.M: int = Y.shape[1]
        assert (
            self.Ny == self.N
        ), f"The X and Y arrays must have the same number of rows: X has {self.N} and Y has {self.Ny}."

        # Check if number of components is supported against maximum requested
        min_dim = min(self.N, self.K)
        self.A = min_dim if self.n_components is None else int(self.n_components)
        if min_dim < self.A:
            warn = (
                "The requested number of components is more than can be "
                "computed from data. The maximum number of components is "
                f"the minimum of either the number of rows ({self.N}) or "
                f"the number of columns ({self.K})."
            )
            warnings.warn(warn, SpecificationWarning, stacklevel=2)
            self.A = self.n_components = min_dim

        if np.any(Y.isna()) or np.any(X.isna()):
            # If there are missing data, then the missing data settings apply. Defaults are:
            #
            #       md_method = "tsr"
            #       md_tol = np.sqrt(np.finfo(float).eps)
            #       md_max_iter = self.max_iter

            default_mds = dict(md_method="tsr", md_tol=epsqrt, md_max_iter=self.max_iter)
            if isinstance(self.missing_data_settings, dict):
                default_mds.update(self.missing_data_settings)

            self.missing_data_settings = default_mds
            self = PLS_missing_values(
                n_components=self.n_components,
                missing_data_settings=self.missing_data_settings,
            )
            self.N, self.K = X.shape
            self.Ny, self.M = Y.shape
            self.A = self.n_components

            # Call the sub-function to do the PLS fit when missing data are present
            self.fit(X, Y)
        else:
            self = super().fit(X, Y)
            # x_scores #T: N x A
            # y_scores # U: N x A
            # x_weights # W: K x A
            # y_weights # identical values to y_loadings. So this is redundant then.
            # x_loadings  # P: K x A
            # y_loadings  # C: M x A

            self.extra_info = {}
            self.extra_info["timing"] = np.zeros(self.A) * np.nan
            self.extra_info["iterations"] = np.array(self.n_iter_)

        # We have now fitted the model. Apply some convenience shortcuts for the user.
        self.A = self.n_components

        # Further convenience calculations
        # "direct_weights" is matrix R = W(P'W)^{-1}  [R is KxA matrix]; useful since T = XR
        direct_weights = self.x_weights_ @ np.linalg.inv(self.x_loadings_.T @ self.x_weights_)
        # beta = RC' [KxA by AxM = KxM]: the KxM matrix gives the direct link from the k-th (input)
        # X variable, to the m-th (output) Y variable.
        beta_coefficients = direct_weights @ self.y_loadings_.T

        # Initialize storage:
        component_names = [a + 1 for a in range(self.A)]
        self.x_scores = pd.DataFrame(self.x_scores_, index=X.index, columns=component_names)
        self.y_scores = pd.DataFrame(self.y_scores_, index=Y.index, columns=component_names)
        self.x_weights = pd.DataFrame(self.x_weights_, index=X.columns, columns=component_names)
        self.y_weights = pd.DataFrame(self.y_weights_, index=Y.columns, columns=component_names)
        self.x_loadings = pd.DataFrame(self.x_loadings_, index=X.columns, columns=component_names)
        self.y_loadings = pd.DataFrame(self.y_loadings_, index=Y.columns, columns=component_names)
        self.predictions = pd.DataFrame(self.x_scores @ self.y_loadings.T, index=Y.index, columns=Y.columns)
        self.direct_weights = pd.DataFrame(direct_weights, index=X.columns, columns=component_names)
        self.beta_coefficients = pd.DataFrame(beta_coefficients, index=X.columns, columns=Y.columns)
        self.explained_variance = np.diag(self.x_scores.T @ self.x_scores) / (self.N - 1)
        self.scaling_factor_for_scores = pd.Series(
            np.sqrt(self.explained_variance),
            index=component_names,
            name="Standard deviation per score",
        )
        self.hotellings_t2 = pd.DataFrame(
            np.zeros(shape=(self.N, self.A)),
            columns=component_names,
            index=X.index.copy(),
        )
        self.hotellings_t2.index.name = "Hotelling's T^2 statistic, per component (col)"
        self.squared_prediction_error = pd.DataFrame(
            np.zeros((self.N, self.A)),
            columns=component_names,
            index=X.index.copy(),
        )
        self.R2 = pd.Series(
            np.zeros(shape=(self.A)),
            index=component_names,
            name="Output R^2, per component (col)",
        )
        self.R2cum = pd.Series(
            np.zeros(shape=(self.A)),
            index=component_names,
            name="Output cumulative R^2, per component (col)",
        )
        self.R2X_cum = pd.DataFrame(
            np.zeros(shape=(self.K, self.A)),
            index=X.columns.copy(),
            columns=component_names,
        )
        self.R2X_cum.index.name = "R^2 per variable (row), per component (col)"
        self.R2Y_cum = pd.DataFrame(
            np.zeros(shape=(self.M, self.A)),
            index=Y.columns.copy(),
            columns=component_names,
        )
        self.R2Y_cum.index.name = "R^2 per variable (row), per component (col)"
        self.RMSE = pd.DataFrame(
            np.zeros(shape=(self.M, self.A)),
            index=Y.columns.copy(),
            columns=component_names,
        )
        self.RMSE.index.name = "Root mean square error per variable (row), per component (col)"

        Xd = X.copy()
        Yd = Y.copy()
        prior_SSX_col = ssq(Xd.values, axis=0)
        prior_SSY_col = ssq(Yd.values, axis=0)
        base_variance_Y = np.sum(prior_SSY_col)
        for a in range(self.A):
            self.hotellings_t2.iloc[:, a] = (
                self.hotellings_t2.iloc[:, max(0, a - 1)]
                + (self.x_scores.iloc[:, a] / self.scaling_factor_for_scores.iloc[a]) ** 2
            )
            Xd -= self.x_scores.iloc[:, [a]] @ self.x_loadings.iloc[:, [a]].T
            y_hat = self.x_scores.iloc[:, 0 : (a + 1)] @ self.y_loadings.iloc[:, 0 : (a + 1)].T
            # These are the Residual Sums of Squares (RSS); i.e X-X_hat
            row_SSX = ssq(Xd.values, axis=1)
            col_SSX = ssq(Xd.values, axis=0)
            row_SSY = ssq(y_hat.values, axis=1)
            col_SSY = ssq(y_hat.values, axis=0)
            # R2 and cumulative R2 value for the whole block
            self.R2cum.iloc[a] = sum(row_SSY) / base_variance_Y
            if a > 0:
                self.R2.iloc[a] = self.R2cum.iloc[a] - self.R2cum.iloc[a - 1]
            else:
                self.R2.iloc[a] = self.R2cum.iloc[a]

            # Don't use a check correction factor. Define SPE simply as the sum of squares of
            # the errors, then take the square root, so it is interpreted like a standard error.
            # If the user wants to normalize it, then this is a clean base value to start from.
            self.squared_prediction_error.iloc[:, a] = np.sqrt(row_SSX)

            # TODO: some entries in prior_SSX_col can be zero and leads to nan's in R2X_cum
            self.R2X_cum.iloc[:, a] = 1 - col_SSX / prior_SSX_col
            self.R2Y_cum.iloc[:, a] = col_SSY / prior_SSY_col
            self.RMSE.iloc[:, a] = (Yd.values - y_hat).pow(2).mean().pow(0.5)

        self.ellipse_coordinates = partial(
            ellipse_coordinates,
            n_components=self.n_components,
            scaling_factor_for_scores=self.scaling_factor_for_scores,
            n_rows=self.N,
        )

        self.hotellings_t2_limit = partial(hotellings_t2_limit, n_components=self.n_components, n_rows=self.N)
        self.spe_plot = partial(spe_plot, model=self)
        self.t2_plot = partial(t2_plot, model=self)
        self.loading_plot = partial(loading_plot, model=self)
        self.score_plot = partial(score_plot, model=self)

        return self

    def predict(self, X: DataMatrix):
        """Use the PLS model on new data coming in matrix X."""

        class State:
            """Class to hold the prediction results together."""

            def __init__(self):
                self.N = None
                self.K = None
                self.x_scores = None
                self.hotellings_t2 = None
                self.squared_prediction_error = None
                self.y_hat = None

        state = State()
        state.N, state.K = X.shape
        assert self.K == state.K, "Prediction data must have same number of columns as training data."

        state.x_scores = X @ self.direct_weights

        # TODO: handle the missing data version here still
        for _ in range(self.A):
            pass
            # p = self.x_loadings.iloc[:, [a]]
            # w = self.x_weights.iloc[:, [a]]
            # temp = X @ self.direct_weights.iloc[:, [a]]
            # X_mcuv -= temp @ p.T
            # state.x_scores[:, [a]] = temp

        # Scores are calculated, now do the rest
        state.hotellings_t2 = np.sum(np.power((state.x_scores / self.scaling_factor_for_scores.values), 2), 1)
        # Calculate SPE-residuals (sum over rows of the errors)
        X_mcuv = X.copy()
        X_mcuv -= state.x_scores @ self.x_loadings.T
        state.squared_prediction_error = np.sqrt(np.power(X_mcuv, 2).sum(axis=1))
        # Predicted values from the model (user still has to un-preprocess these predictions!)
        state.y_hat = state.x_scores @ self.y_loadings.T

        return state

    def SPE_limit(self, conf_level: float = 0.95) -> float:  # noqa: N802
        """Calculate the SPE limit for the model."""
        check_is_fitted(self, "squared_prediction_error")

        return spe_calculation(
            spe_values=self.squared_prediction_error.iloc[:, self.A - 1],
            conf_level=conf_level,
        )


class PLS_missing_values(BaseEstimator, TransformerMixin):  # noqa: N801
    """
    Create our PLS class if there is even a single missing data value in the X input array.

    The default method to impute missing values is the TSR algorithm (`md_method="tsr"`).

    Missing data method options are:

    * 'pmp'         Projection to Model Plane
    * 'scp'         Single Component Projection
    * 'nipals'      Same as 'scp': non-linear iterative partial least squares.
    * 'tsr':        Trimmed score regression

    * Other options? See papers by Abel Folch-Fortuny and also:
    https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/abs/10.1002/cem.750

    """

    valid_md_methods: typing.ClassVar[list[str]] = ["pmp", "scp", "nipals", "tsr"]

    def __init__(
        self,
        n_components: int,
        missing_data_settings: dict,
    ):
        self.n_components = n_components
        self.missing_data_settings = missing_data_settings
        self.has_missing_data = True
        self.missing_data_settings["md_max_iter"] = int(self.missing_data_settings["md_max_iter"])

        assert self.missing_data_settings["md_tol"] < 10, "Tolerance should not be too large"
        assert self.missing_data_settings["md_tol"] > epsqrt**1.95, "Tolerance must exceed machine precision"
        assert self.missing_data_settings["md_method"] in self.valid_md_methods, (
            f"Missing data method is not recognized. Must be one of {self.valid_md_methods}.",
        )

    def fit(self, X: DataMatrix, Y: DataMatrix) -> DataMatrix:
        """
        Fits a PLS latent variable model between `X` and `Y` data arrays, accounting for missing
        values (nan's) in either or both arrays.

        1.  Höskuldsson, PLS regression methods, Journal of Chemometrics, 2(3), 211-228, 1998,
            http://dx.doi.org/10.1002/cem.1180020306
        """
        # Force input to NumPy array:
        self.Xd = np.asarray(X)
        self.Yd = np.asarray(Y)

        if np.any(np.sum(self.Yd, axis=1) == 0):
            raise Warning(
                "Cannot handle the case yet where the entire observation in Y-matrix is "
                "missing. Please remove those rows and refit model."
            )

        # Other setups:
        self.n_components = self.A
        self.n_samples_ = self.N
        self.n_features_in_ = self.K
        # self.M ?

        self.x_scores_ = np.zeros((self.N, self.A))  # T: N x A
        self.y_scores_ = np.zeros((self.N, self.A))  # U: N x A
        self.x_weights_ = np.zeros((self.K, self.A))  # W: K x A
        self.y_weights_ = None
        self.x_loadings_ = np.zeros((self.K, self.A))  # P: K x A
        self.y_loadings_ = np.zeros((self.M, self.A))  # C: M x A

        self.R2 = np.zeros(shape=(self.A,))
        self.R2cum = np.zeros(shape=(self.A,))
        self.R2X_cum = np.zeros(shape=(self.K, self.A))
        self.R2Y_cum = np.zeros(shape=(self.M, self.A))
        self.squared_prediction_error = np.zeros((self.N, self.A))

        # Perform MD algorithm here
        if self.missing_data_settings["md_method"].lower() == "pmp":
            pytest.fail("PMP for PLS not implemented yet")  # self._fit_pmp_pls(X)

        elif self.missing_data_settings["md_method"].lower() in ["scp", "nipals"]:
            self._fit_nipals_pls(settings=self.missing_data_settings)
        elif self.missing_data_settings["md_method"].lower() in ["tsr"]:
            pytest.fail("TSR for PLS not implemented yet")  # self._fit_tsr_pls(settings=self.missing_data_settings)

        # Additional calculations, which can be done after the missing data method is complete.
        # self.explained_variance_ = np.diag(self.x_scores.T @ self.x_scores) / (self.N - 1)
        return self

    def _fit_nipals_pls(self, settings: dict) -> None:
        """
        Fit the PLS model using the NIPALS algorithm.

        (Internal method)
        """
        # NIPALS algorithm
        A = self.A

        # Initialize storage:
        self.extra_info = {}
        self.extra_info["timing"] = np.zeros(A) * np.nan
        self.extra_info["iterations"] = np.zeros(A) * np.nan

        for a in np.arange(A):
            # Timers and housekeeping
            start_time = time.time()
            itern = 0

            start_SSX_col = ssq(self.Xd, axis=0)
            start_SSY_col = ssq(self.Yd, axis=0)

            if sum(start_SSX_col) < epsqrt:
                emsg = (
                    "There is no variance left in the data array for X: cannot "
                    f"compute any more components beyond component {a}."
                )
                raise RuntimeError(emsg)
            if sum(start_SSY_col) < epsqrt:
                emsg = (
                    "There is no variance left in the data array for Y: cannot "
                    f"compute any more components beyond component {a}."
                )
                raise RuntimeError(emsg)

            # Initialize t_a with random numbers, or carefully select a column from X or Y?
            # Find a column with the largest variance as t1_start; replace missing with zeros
            # All columns have the same variance if the data have been scaled to unit variance!
            u_a_guess = self.Yd[:, [0]]
            u_a_guess[np.isnan(u_a_guess)] = 0
            u_a = u_a_guess + 1.0

            while not (terminate_check(u_a_guess, u_a, iterations=itern, settings=settings)):
                # 0: starting point for convergence checking on next loop
                u_a_guess = u_a.copy()

                # 1: Regress the score, u_a, onto every column in X, compute the
                #    regression coefficient and store in w_a
                # w_a = X.T * u_a / (u_a.T * u_a)
                w_a = quick_regress(self.Xd, u_a)

                # 2: Normalize w_a to unit length
                w_a /= np.sqrt(ssq(w_a))

                # 3: Now regress each row in X on the w_a vector, and store the
                #    regression coefficient in t_a
                # t_a = X * w_a / (w_a.T * w_a)
                t_a = quick_regress(self.Xd, w_a)

                # 4: Now regress score, t_a, onto every column in Y, compute the
                #    regression coefficient and store in c_a
                # c_a = Y * t_a / (t_a.T * t_a)
                c_a = quick_regress(self.Yd, t_a)

                # 5: Now regress each row in Y on the c_a vector, and store the
                #    regression coefficient in u_a
                # u_a = Y * c_a / (c_a.T * c_a)
                #
                # TODO(KGD):  % Still handle case when entire row in Y is missing
                u_a = quick_regress(self.Yd, c_a)

                itern += 1

            self.extra_info["timing"][a] = time.time() - start_time
            self.extra_info["iterations"][a] = itern

            if itern > settings["md_max_iter"]:
                raise Warning("PLS missing data [SCP method]: maximum number of iterations reached!")

            # Loop terminated!
            # 6: Now deflate the X-matrix.  To do that we need to calculate loadings for the
            # X-space.  Regress columns of t_a onto each column in X and calculate loadings, p_a.
            # Use this p_a to deflate afterwards.
            p_a = quick_regress(self.Xd, t_a)  # Note the similarity with step 4!
            self.Xd -= np.dot(t_a, p_a.T)  # and that similarity helps understand
            self.Yd -= np.dot(t_a, c_a.T)  # the deflation process.

            ## VIP value (only calculated for X-blocks); only last column is useful
            # self.stats.VIP_a = np.zeros((self.K, self.A))
            # self.stats.VIP = np.zeros(self.K)

            # Store results
            # -------------
            # Flip the signs of the column vectors in P so that the largest
            # magnitude element is positive (Wold, Esbensen, Geladi, PCA,
            # CILS, 1987, p 42)
            max_el_idx = np.argmax(np.abs(p_a))
            if np.sign(p_a[max_el_idx]) < 1:
                t_a *= -1.0
                u_a *= -1.0
                w_a *= -1.0
                p_a *= -1.0
                c_a *= -1.0

            # Store the loadings and scores
            self.x_scores_[:, a] = t_a.flatten()
            self.y_scores_[:, a] = u_a.flatten()
            self.x_weights_[:, a] = w_a.flatten()
            self.x_loadings_[:, a] = p_a.flatten()
            self.y_loadings_[:, a] = c_a.flatten()
            # end looping on ``a``


def nan_to_zeros(in_array: np.ndarray) -> np.ndarray:
    """Convert NaN to zero and return a NaN map."""

    nan_map = np.isnan(in_array)
    in_array[nan_map] = 0.0
    return in_array


def regress_a_space_on_b_row(a_space: np.ndarray, b_row: np.ndarray, a_space_present_map: np.ndarray) -> np.ndarray:
    """
    Project each row of `a_space` onto row vector `b_row`, to return a regression coefficient for every row in A.

    NOTE: Neither of these two inputs may have missing values. It is assumed you have replaced missing values by zero,
          and have a map of where the missing values were (more correctly, where the non-missing values are is given
          by `a_space_present_map`).

    NOTE: No checks are done on the incoming data to ensure consistency. That is the caller's responsibility. This
          function is called thousands of times, so that overhead is not acceptable.

    The `a_space_present_map` has `False` entries where `a_space` originally had NaN values.
    The `b_row` may never have missing values, and no map is provided for it. These row vectors are latent variable
    vectors, and therefore never have missing values.

    a_space             = [n_rows x j_cols]
    b_row               = [1      x j_cols]    # in other words, a row vector of `j_cols` entries
    a_space_present_map = [n_rows x j_cols]

    Returns               [n_rows x 1] = a_space * b_row^T  / ( b_row * b_row^T)
                                         (n x j) * (j x 1)  /  (1 x j)* (j x 1)  = n x 1
    """
    denom = np.tile(b_row, (a_space.shape[0], 1))  # tiles, row-by-row the `b_row` row vector, to create `n_rows`
    denominator = np.sum((denom * a_space_present_map) ** 2, axis=1).astype("float")
    denominator[denominator == 0] = np.nan
    return np.array((np.sum(a_space * denom, axis=1)) / denominator).reshape(-1, 1)


def ssq(X: np.ndarray, axis: int | None = None) -> float | np.ndarray:
    """Calculate the sum of squares of a 2D matrix (not array! and not checked for either: code will simply fail),
    skipping over any NaN (missing) data.
    """
    N, K = X.shape
    if axis == 0:
        out_ax0 = np.zeros(K)
        for k in np.arange(K):
            out_ax0[k] += np.nansum(X[:, k] ** 2)

        return out_ax0

    if axis == 1:
        out_ax1 = np.zeros(N)
        for n in np.arange(N):
            out_ax1[n] += np.nansum(X[n, :] ** 2)

        return out_ax1

    out = 0.0
    if axis is None:
        out = np.nansum(X**2)

    return out


def terminate_check(t_a_guess: np.ndarray, t_a: np.ndarray, iterations: int, settings: dict) -> bool:
    """Terminate the PCA iterative algorithm when any one of these conditions is True.

    #. scores converge: the norm between two successive iterations
    #. maximum number of iterations is reached
    """
    score_tol = np.linalg.norm(t_a_guess - t_a, ord=None)
    converged = score_tol < settings["md_tol"]
    max_iter = iterations > settings["md_max_iter"]
    return bool(np.any([max_iter, converged]))


def quick_regress(Y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Regress vector `x` onto the columns in matrix ``Y`` one at a time.
    Return the vector of regression coefficients, one for each column in `Y`.
    There may be missing data in `Y`, but not in `x`.  The `x` vector
    *must* be a column vector.
    """
    Ny, K = Y.shape
    Nx = x.shape[0]
    if Ny == Nx:  # Case A: b' = (x'Y)/(x'x): (1xN)(NxK) = (1xK)
        b = np.zeros((K, 1))
        for k in np.arange(K):
            b[k] = np.sum(x.T * np.nan_to_num(Y[:, k]))
            temp = ~np.isnan(Y[:, k]) * x.T
            denom = np.dot(temp, temp.T)[0][0]
            if np.abs(denom) > epsqrt:
                b[k] /= denom
        return b

    elif Nx == K:  # Case B: b = (Yx)/(x'x): (NxK)(Kx1) = (Nx1)
        b = np.zeros((Ny, 1))
        for n in np.arange(Ny):
            b[n] = np.sum(x[:, 0] * np.nan_to_num(Y[n, :]))
            # TODO(KGD): check: this denom is usually(always?) equal to 1.0
            denom = ssq(~np.isnan(Y[n, :]) * x.T)
            if np.abs(denom) > epsqrt:
                b[n] /= denom
        return b

    else:
        raise ValueError("The dimensions of the input arrays are not compatible.")


def center(X, func: Callable = np.mean, axis: int = 0, extra_output: bool = False) -> DataMatrix:  # noqa: ANN001
    """
    Perform centering of data, using a function, `func` (default: np.mean).
    The function, if supplied, but return a vector with as many columns as the matrix X.

    `axis` [optional; default=0] {integer or None}

    This specifies the axis along which the centering vector will be calculated if not provided.
    The function is applied along the `axis`: 0=down the columns; 1 = across the rows.

    *Missing values*: The sample mean is computed by taking the sum along the `axis`, skipping
    any missing data, and dividing by N = number of values which are present. Values which were
    missing before, are left as missing after.
    """
    vector = pd.DataFrame(X).apply(func, axis=axis).values
    if extra_output:
        return np.subtract(X, vector), vector
    else:
        return np.subtract(X, vector)


def scale(X: DataMatrix, func: Callable = np.std, axis: int = 0, extra_output: bool = False, **kwargs) -> DataMatrix:
    """
    Scales the data (does NOT do any centering); scales to unit variance by
    default.


    `func` [optional; default=np.std] {a function}
        The default (np.std) will use NumPy to calculate the sample standard
        deviation of the data, and use that as `scale`.

        TODO: provide a scaling vector.
        The sample standard deviation is computed along the required `axis`,
        skipping over any missing data, and dividing by N-1, where N = number
        of values which are present, i.e. not counting missing values.

    `axis` [optional; default=0] {integer}
        Transformations are applied on slices of data.  This specifies the
        axis along which the transformation will be applied.

    #`markers` [optional; default=None]
    #    A vector (or slice) used to store indices (the markers) where the
    ##    variance needs to be replaced with the `low_variance_replacement`
    #
    #`variance_tolerance` [optional; default=1E-7] {floating point}
    #    A slice is considered to have no variance when the actual variance of
    #    that slice is smaller than this value.
    #
    #`low_variance_replacement` [optional; default=0.0] {floating point}
    #    Used to replace values in the output where the `markers` indicates no
    #    or low variance.

    Usage
    =====

    X = ...  # data matrix
    X = scale(center(X))
    my_scale = np.mad
    X = scale(center(X), func=my_scale)

    """
    # options = {}
    # options["markers"] = None
    # options["variance_tolerance"] = epsqrt
    # options["low_variance_replacement"] = np.nan

    vector = pd.DataFrame(X).apply(func, axis=axis, **kwargs).values
    # if options["markers"] is None:
    # options["markers"] = vector < options["variance_tolerance"]
    # if options["markers"].any():
    #    options["scale"][options["markers"]] = options["low_variance_replacement"]

    vector = 1.0 / vector

    if extra_output:
        return np.multiply(X, vector), vector
    else:
        return np.multiply(X, vector)


def hotellings_t2_limit(conf_level: float = 0.95, n_components: int = 0, n_rows: int = 0) -> float:
    """Return the Hotelling's T2 value at the given level of confidence.

    Parameters
    ----------
    conf_level : float, optional
        Fractional confidence limit, less that 1.00; by default 0.95

    Returns
    -------
    float
        The Hotelling's T2 limit at the given level of confidence.
    """
    assert 0.0 < conf_level < 1.0
    assert n_rows > 0
    if n_components == n_rows:
        return float("inf")
    return (
        n_components
        * (n_rows - 1)
        * (n_rows + 1)
        / (n_rows * (n_rows - n_components))
        * f.isf((1 - conf_level), n_components, n_rows - n_components)
    )


def spe_limit(model: BaseEstimator, conf_level: float = 0.95) -> float:
    """Return the squared prediction error limit at the given level of confidence.

    Parameters
    ----------
    conf_level : float, optional
        Fractional confidence limit, less that 1.00; by default 0.95

    Returns
    -------
    float
        The squared prediction error limit at the given level of confidence.
    """
    check_is_fitted(model, "squared_prediction_error")

    return spe_calculation(
        spe_values=model.squared_prediction_error.iloc[:, model.A - 1],
        conf_level=conf_level,
    )


def spe_calculation(spe_values: np.ndarray, conf_level: float = 0.95) -> float:
    """Return a limit for SPE (squared prediction error) at the given level of confidence.

    Parameters
    ----------
    spe_values : pd.Series
        The SPE values from the last component in the multivariate model.
    conf_level : [float], optional
        The confidence level, by default 0.95, i.e. the 95% confidence level.

    Returns
    -------
    float
        The limit, above which we judge observations in the model to have a different correlation
        structure than those values which were used to build the model.
    """
    assert conf_level > 0.0, "conf_level must be a value between (0.0, 1.0)"
    assert conf_level < 1.0, "conf_level must be a value between (0.0, 1.0)"

    # The limit is for the squares (i.e. the sum of the squared errors)
    # I.e. `spe_values` are square-rooted outside this function, so undo that.
    values = spe_values**2
    center_spe = float(values.mean())
    variance_spe = float(values.var(ddof=1))
    g = variance_spe / (2 * center_spe)
    h = (2 * (center_spe**2)) / variance_spe
    # Report square root again as SPE limit
    return np.sqrt(chi2.ppf(conf_level, h) * g)


def ellipse_coordinates(  # noqa: PLR0913
    score_horiz: int,
    score_vert: int,
    conf_level: float = 0.95,
    n_points: int = 100,
    n_components: int = 0,
    scaling_factor_for_scores: pd.Series | None = None,
    n_rows: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Get the (score_horiz, score_vert) coordinate pairs that form the T2 ellipse when
        plotting the score `score_horiz` on the horizontal axis and `score_vert` on the
        vertical axis.

        Scores are referred to by number, starting at 1 and ending with `model.A`


    Parameters
    ----------
    score_horiz : int
        [description]
    score_vert : int
        [description]
    conf_level : float
        The `conf_level` confidence value: e.g. 0.95 is for the 95% confidence limit.
    n_points : int, optional
        Number of points to use in the ellipse; by default 100.

    Returns
    -------
    tuple of 2 elements; the first for the x-axis; the second for the y-axis.
        Returns `n_points` equispaced points that can be used to plot an ellipse.

    Background
    ----------

    Equation of ellipse in *canonical* form (http://en.wikipedia.org/wiki/Ellipse)

        (t_horiz/s_h)^2 + (t_vert/s_v)^2  =  hotellings_t2_limit_alpha
        s_horiz = stddev(T_horiz)
        s_vert  = stddev(T_vert)
        hotellings_t2_limit_alpha = T2 confidence limit at a given alpha value

    Equation of ellipse, *parametric* form (http://en.wikipedia.org/wiki/Ellipse):

        t_horiz = sqrt(hotellings_t2_limit_alpha)*s_h*cos(t)
        t_vert  = sqrt(hotellings_t2_limit_alpha)*s_v*sin(t)

        where t ranges between 0 and 2*pi.
    """
    assert 1 <= score_horiz <= n_components
    assert 1 <= score_vert <= n_components
    assert 0 < conf_level < 1
    assert n_rows > 0
    s_h = scaling_factor_for_scores.iloc[score_horiz - 1]
    s_v = scaling_factor_for_scores.iloc[score_vert - 1]
    t2_limit_specific = np.sqrt(hotellings_t2_limit(conf_level, n_components=n_components, n_rows=n_rows))
    dt = 2 * np.pi / (n_points - 1)
    steps = np.linspace(0, n_points - 1, n_points)
    x = np.cos(steps * dt) * t2_limit_specific * s_h
    y = np.sin(steps * dt) * t2_limit_specific * s_v
    return x, y


def internal_pls_nipals_fit_one_pc(
    x_space: np.ndarray,
    y_space: np.ndarray,
    x_present_map: np.ndarray,
    y_present_map: np.ndarray,
) -> dict[str, np.ndarray]:
    """Fit a PLS model using the NIPALS algorithm."""
    max_iter: int = 500

    is_converged = False
    n_iter = 0
    u_i = y_space[:, [0]]
    while not is_converged:
        # Step 1. w_i = X'u / u'u. Regress the columns of X on u_i, and store the slope coeff in vectors w_i.
        w_i = regress_a_space_on_b_row(x_space.T, u_i.T, x_present_map.T)

        # Step 2. Normalize w to unit length.
        w_i = w_i / np.linalg.norm(w_i)

        # Step 3. t_i = Xw / w'w. Regress rows of X on w_i, and store slope coefficients in t_i.
        t_i = regress_a_space_on_b_row(x_space, w_i.T, x_present_map)

        # Step 4. q_i = Y't / t't. Regress columns of Y on t_i, and store slope coefficients in q_i.
        q_i = regress_a_space_on_b_row(y_space.T, t_i.T, y_present_map.T)

        # Step 5. u_new = Yq / q'q. Regress rows of Y on q_i, and store slope coefficients in u_new
        u_new = regress_a_space_on_b_row(y_space, q_i.T, y_present_map)

        if (abs(np.linalg.norm(u_i - u_new)) / np.linalg.norm(u_i)) < epsqrt:
            is_converged = True
        if n_iter > max_iter:
            is_converged = True

        n_iter += 1
        u_i = u_new

    # We have converged. Keep sign consistency. Fairly arbitrary rule, but ensures we report results consistently.
    if np.var(t_i[t_i < 0]) > np.var(t_i[t_i >= 0]):
        t_i = -t_i
        u_new = -u_new
        w_i = -w_i
        q_i = -q_i

    return dict(t_i=t_i, u_i=u_i, w_i=w_i, q_i=q_i)


# def _apply_pca(self, new=None):
#     """
#     Project new observations, ``new``, onto the existing latent variable
#     model.  Returns a ``Projection`` object, which contains the scores,
#     SPE, T2, and predictions.

#     If ``new`` is not provided it will return a ``Projection`` object
#     for the training data set.
#     """
#     # TODO: complete this code.
#     new = preprocess(LVM, new)
#     result = Projection()
#     result.scores = np.zeros([LVM.J, LVM.A])
#     result.SPE = np.zeros([LVM.J, 1])
#     result.T2 = np.zeros([LVM.J, 1])
#     result.Yhat = np.zeros([LVM.J, LVM.M])

#     # (1, 2, ... J) * every tag * for J time steps * for every LV
#     # result.c_scores = np.zeros([LVM.J, LVM.K, LVM.J, LVM.A]) * np.nan

#     K = LVM.K
#     x_project = new.copy()
#     for j in np.arange(LVM.J):
#         idx_beg, idx_end = K * j, K * (j + 1)
#         x = x_project.copy()
#         if LVM.opt.md_method == "scp":
#             for a in np.arange(LVM.A):

#                 if LVM.M:  # PLS
#                     r = LVM.W[0:idx_end, a]
#                 else:  # PCA
#                     r = LVM.P[0:idx_end, a]

#                 rtr = np.dot(r.T, r)
#                 p = LVM.P[0:idx_end, a]

#                 # t_{\text{new},j}(a) = \mathbf{P/W}'_*(:,a) \mathbf{x}_\text{pp}/DEN
#                 temp = (r * x[0, 0:idx_end]) / (rtr + 0.0)
#                 # result.c_scores[0:j+1, :, j, a] = temp.reshape(j+1, LVM.K)
#                 result.scores[j, a] = np.nansum(temp)

#                 # \mathbf{x}_\text{pp} -= t_{\text{new},j}(a) \mathbf{P}'_*(:,a)
#                 x[0, 0:idx_end] -= result.scores[j, a] * p

#             # The error_j is only the portion related to the current time step
#             error_j = x[0, idx_beg:idx_end]

#         result.SPE[j, 0] = np.nansum(error_j ** 2)
#         result.T2[j, 0] = np.sum((result.scores[j, :] / LVM.S[j, :]) ** 2)
#         Yhat_MCUV = np.dot(result.scores[j, :], LVM.C.T)
#         result.Yhat[j, :] = Yhat_MCUV / (LVM.PPY[1][1] + 0.0) + LVM.PPY[0][1]

#     return result


class DataFrameDict(dict):
    def __init__(self, datadict: dict[str, dict[str, pd.DataFrame]]):
        """
        Initialize a DataFrameDict to handle partitionable and static dataframes.

        datadict: Dictionary with 3 keys, one for each block: Z, F and Y.
                  Each block is itself a dictionary of dataframes: dict[str, dict[str, pd.DataFrame]]
        """

        self.partitionable_blocks: list[str] = ["Z", "F", "Y"]
        self.datadict: dict[str, dict[str, pd.DataFrame]] = {}
        for block in self.partitionable_blocks:
            self.datadict[block] = datadict.get(block, {})
        first_group = next(iter(self.datadict["F"].keys()))
        self.n_samples = self.datadict["F"][first_group].shape[0]
        self.shape = (self.n_samples, len(self.datadict))

        # Some basic checks: each dataframe inside each block has the same number of rows
        for block in set(self.partitionable_blocks) & set(self.datadict.keys()):
            for group, df in self.datadict[block].items():
                if not isinstance(df, pd.DataFrame):
                    raise TypeError(f"Expected a DataFrame for block {block}, group '{group}'; got instead{type(df)}.")
                if df.shape[0] != self.n_samples:
                    raise ValueError(
                        f"DataFrames in block {block} must have the same number of rows ({self.n_samples}). "
                        f"Group {group} has {df.shape[0]} rows."
                    )

    def keys(self) -> KeysView[str]:
        """Return the keys of the DataFrameDict."""
        return self.datadict.keys()

    def __setitem__(self, key: str, value: pd.DataFrame | dict) -> None:
        """Set a DataFrame for a specific key in the DataFrameDict."""
        if key not in self.partitionable_blocks:
            raise KeyError(f"Key {key} is not a valid partitionable block. Valid keys are: {self.partitionable_blocks}")

        if not isinstance(value, pd.DataFrame):
            raise TypeError(f"Expected a DataFrame for key {key}, got {type(value)}.")
        if value.shape[0] != self.n_samples:
            raise ValueError(
                f"DataFrames in block {key} must have the same number of rows ({self.n_samples}). "
                f"Provided DataFrame has {value.shape[0]} rows."
            )
        self.datadict[key] = value

    def __getitem__(self, lookup: int | list[int] | list[np.int64] | str) -> DataFrameDict | dict[str, pd.DataFrame]:
        """Return a new DataFrameDict with partitioned data."""

        if isinstance(lookup, str):
            return self.datadict[lookup]  # returns the `dict[str, pd.DataFrame]` version of the function

        datadict: dict[str, dict[str, pd.DataFrame]] = {}
        for block in self.partitionable_blocks:
            datadict[block] = {}
            for group, df in self.datadict[block].items():
                if isinstance(lookup, (int, np.integer)):
                    datadict[block][group] = df.iloc[[lookup]]
                if isinstance(lookup, list):
                    raise TypeError("datadict[block][group] = df.iloc[lookup] <-- test still")
                if isinstance(lookup, np.ndarray):
                    raise TypeError("datadict[block][group] = df.iloc[lookup.tolist()] <-- test still")
                if isinstance(lookup, tuple):
                    assert lookup[1] == Ellipsis
                    datadict[block][group] = df.iloc[[int(item) for item in lookup[0]]]
                else:
                    raise TypeError(f"Lookup must be an int, list of ints, or a string. Got {lookup}; {type(lookup)}")

        return DataFrameDict(datadict)

    def __len__(self):
        """Return the number of samples in the DataFrameDict."""
        return self.n_samples

    def __repr__(self):
        """Return a string representation of the DataFrameDict."""
        groups_in_block_f = list(self.datadict["F"].keys())
        groups_in_block_z = list(self.datadict["Z"].keys())
        groups_in_block_y = list(self.datadict["Y"].keys())
        output = f"DataFrameDict with {len(self)} samples and {len(self.datadict)} blocks: {list(self.datadict.keys())}"
        output += f"\n  F groups: {groups_in_block_f}"
        output += f"\n  Z groups: {groups_in_block_z}"
        output += f"\n  Y groups: {groups_in_block_y}"
        return output


class TPLS(RegressorMixin, BaseEstimator):
    """
    TPLS algorithm for T-shaped data structures (we also include standard pre-processing of the data inside this class).

    Source: Garcia-Munoz, https://doi.org/10.1016/j.chemolab.2014.02.006, Chem.Intell.Lab.Sys. v133, p 49 to 62, 2014.

    We change the notation from the original paper to avoid confusion with a generic "X" matrix, and match symbols
    that are more natural for our use.

    Paper           This code     Internal Numpy variable name (holds only NumPy values)
    =====           ========      ============================
    X^T             D             d_matrix (external);              Database of properties
                                  d_mats   (internal)               (we use a copy, d_mats, internally)
    X               D^T           --                                -- (we do not use the transposed version of D)
    R               F             f_mats                            Formula
    Z               Z             z_mats                            Conditions
    Y               Y             y_mats                            Quality indicators

    Notes
    1. Matrices in F, Z and Y must all have the same number of rows.
    2. Columns in F must be the same as the **rows** in D.
    3. Conditions in Z may be missing (turning it into an L-shaped data structure).

    Parameters
    ----------
    n_components : int
        A parameter used to specify the number of components.

    d_matrix : dict[str, dict[str, pd.DataFrame]]
        A dictionary containing the properties of each group of materials.
        The keys are the group names, and the values are dataframes with properties as columns and materials as rows.
        This "D" matrix is provided once when constructing the model, and reused for fitting, prediction and
        cross-validation.

    max_iterations: int, optional
        The maximum number of iterations for the TPLS algorithm. Default is 500.

    Data structures in input `X` (a dictionary with 4 keys, as listed below)
    ------------------------------------------------------------------------

    D. Database of dataframes, containing properties in the columns, while each row is a material.

        D = { "Group A": dataframe of properties of group A materials. (columns contain properties, rows are materials),
              "Group B": dataframe of properties of group B materials. (columns contain properties, rows are materials),
              ...
            }

    F. Formula matrices/ratio of materials, corresponding to the *rows* of D (or columns of D after transposing):

        F = { "GroupA": dataframe of formula for group A used in each blend (one formula per row, columns are materials)
              "GroupB": dataframe of formula for group B used in each blend (one formula per row, columns are materials)
              ...
            }

    Z. Process conditions. One row per formula/blend; one column per condition.

    Y. Product characteristics (quality space; key performance indicators). One row per formula/blend;
       one column per quality indicator.

    Attributes
    ----------
    n_samples : int
        The number of samples (rows) in the training data

    n_substances : int
        The number of substances (columns) in the training data, i.e. the number of materials in the F matrix.


    Example
    -------
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng()
    >>>
    >>> n_props_a, n_props_b = 6, 4            # Two groups of properties: A and B.
    >>> n_materials_a, n_materials_b = 12, 8   # Number of materials in each group.
    >>> n_formulas = 40                        # Number of formulas in matrix F.
    >>> n_outputs = 3
    >>> n_conditions = 2
    >>>
    >>> properties = {
    >>>     "Group A": pd.DataFrame(rng.standard_normal((n_materials_a, n_props_a))),
    >>>     "Group B": pd.DataFrame(rng.standard_normal((n_materials_b, n_props_b))),
    >>> }
    >>> formulas = {
    >>>     "Group A": pd.DataFrame(rng.standard_normal((n_formulas, n_materials_a))),
    >>>     "Group B": pd.DataFrame(rng.standard_normal((n_formulas, n_materials_b))),
    >>> }
    >>> process_conditions = {"Conditions": pd.DataFrame(rng.standard_normal((n_formulas, n_conditions)))}
    >>> quality_indicators = {"Quality":    pd.DataFrame(rng.standard_normal((n_formulas, n_outputs)))}
    >>> all_data = {"Z": process_conditions, "D": properties, "F": formulas, "Y": quality_indicators}
    >>> estimator = TPLS(n_components=4)
    >>> estimator.fit(all_data)
    """

    # This is a dictionary allowing to define the type of parameters.
    # It used to validate parameter within the `_fit_context` decorator.
    _parameter_constraints: typing.ClassVar = {
        "n_components": [int],
        "max_iterations": [int],
        "d_matrix": [dict, None],
    }

    def __init__(
        self,
        n_components: int,
        d_matrix: dict,
        max_iterations: int = 500,
    ):
        super().__init__()
        assert n_components > 0, "Number of components must be positive."
        self.n_components = n_components

        self.d_matrix = d_matrix  # This is required input dict containing the properties for each group.
        assert isinstance(self.d_matrix, dict), "d_matrix must be a dictionary of dataframes."

        assert all(isinstance(df, pd.DataFrame) for df in self.d_matrix.values()), "d_matrix must contain dataframes."

        self.max_iterations = max_iterations
        assert self.max_iterations > 0, "Maximum number of iterations must be positive."

        self.is_fitted_ = False
        self.n_substances = 0
        self.n_samples = 0
        self.tolerance_ = np.sqrt(np.finfo(float).eps)
        self.required_blocks_ = {"D", "F", "Y", "Z"}  # "Z" block is optional; an empty one is added if not provided
        # "required_inputs" used in the sense of inputs to this class; not in the sense of a "model input"
        self.required_inputs_ = {"F", "Y", "Z"}
        self.plot = Plot(self)

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: DataFrameDict, y: None = None) -> TPLS:  # noqa: ARG002, PLR0915
        """Fit the preprocessing parameters and also the latent variable model from the training data.

        Parameters
        ----------
        X : {dictionary of dataframes}, keys that must be present: "D", "F", "Z", and "Y"
            The training input samples. See documentation in the class definition for more information on each matrix.

        Returns
        -------
        self : object
            Returns self.
        """
        assert isinstance(X, DataFrameDict)
        self._input_data_checks(X)
        group_keys = [str(key) for key in self.d_matrix]

        # Storage for pre-processing and the raw matrices
        self.fitting_statistics: dict[str, list] = {"iterations": [], "convergance_tolerance": [], "milliseconds": []}
        self.preproc_: dict[str, dict[str, dict[str, pd.Series]]] = {key: {} for key in self.required_blocks_}
        self.sums_of_squares_: list[dict[str, dict[str, np.ndarray]]] = [{key: {} for key in self.required_blocks_}]
        # These are *fractional* R2 values, i.e. always less than or equal to 1.0.
        # As a list: entry 0 is zeros; entry 1 is after fitting the first component, and so on.
        # The keys are the blocks, and the values are dictionaries with group keys as keys.
        # The values are the R2 values for each column in the block.
        self.r2_frac: list[dict[str, dict[str, np.ndarray]]] = [{key: {} for key in self.required_blocks_}]
        self.feature_importance: dict[str, dict[str, pd.Series]] = {key: {} for key in self.required_blocks_}

        self.d_mats: dict[str, np.ndarray] = {key: self.d_matrix[key].values.copy() for key in group_keys}
        self.f_mats: dict[str, np.ndarray] = {key: X["F"][key].values.copy() for key in group_keys}
        self.z_mats: dict[str, np.ndarray] = {key: X["Z"][key].values.copy() for key in X["Z"]}
        self.y_mats: dict[str, np.ndarray] = {key: X["Y"][key].values.copy() for key in X["Y"]}

        # Empty model coefficients
        self.n_substances = sum(self.f_mats[key].shape[1] for key in group_keys)
        self.n_conditions = sum(self.z_mats[key].shape[1] for key in self.z_mats)
        self.n_outputs = sum(self.y_mats[key].shape[1] for key in self.y_mats)
        self.n_samples = self.f_mats[group_keys[0]].shape[0]

        # Learn the centering and scaling parameters
        for key in X["Y"]:
            self.preproc_["Y"][key] = {}
            self.preproc_["Y"][key]["center"], self.preproc_["Y"][key]["scale"] = (
                self._learn_center_and_scaling_parameters(X["Y"][key])
            )
        for key in X["Z"]:
            self.preproc_["Z"][key] = {}
            self.preproc_["Z"][key]["center"], self.preproc_["Z"][key]["scale"] = (
                self._learn_center_and_scaling_parameters(X["Z"][key])
            )
        for key, df_d in self.d_matrix.items():
            self.preproc_["D"][key] = {}
            self.preproc_["D"][key]["center"], self.preproc_["D"][key]["scale"] = (
                self._learn_center_and_scaling_parameters(df_d)
            )
            self.preproc_["D"][key]["block"] = pd.Series([np.sqrt(df_d.shape[1])])  # <-- sqrt(number of properties!)
            #
            # Also do the same for the formula matrix
            self.preproc_["F"][key] = {}
            self.preproc_["F"][key]["center"], self.preproc_["F"][key]["scale"] = (
                self._learn_center_and_scaling_parameters(X["F"][key])
            )

        # Then implement the preprocessing on the raw data
        self._preprocess_data()

        # Sum of square values for each column in each block (dicts) per component (elements in the list)
        # The first entry is after centering and scale (baseline variance) but before fitting any components.
        # The second entry is after fitting one component, and so on.
        # You can sum the sums-of-squares values for all columns to get the total variance for each block.
        self.sums_of_squares_ = [
            {
                "D": {key: np.nansum(self.d_mats[key] ** 2, axis=0) for key in group_keys},
                "F": {key: np.nansum(self.f_mats[key] ** 2, axis=0) for key in group_keys},
                "Z": {key: np.nansum(self.z_mats[key] ** 2, axis=0) for key in X["Z"]},
                "Y": {key: np.nansum(self.y_mats[key] ** 2, axis=0) for key in X["Y"]},
            }
        ]
        self.r2_frac = [
            {
                "D": {key: np.zeros(self.d_mats[key].shape[1]) for key in group_keys},
                "F": {key: np.zeros(self.f_mats[key].shape[1]) for key in group_keys},
                "Z": {key: np.zeros(self.z_mats[key].shape[1]) for key in self.z_mats},
                "Y": {key: np.zeros(self.y_mats[key].shape[1]) for key in self.y_mats},
            }
        ]

        # Then set missing data values to zeros (not because we are ignoring the values), but because we will use
        # the missing value maps to identify where the missing values are and therefore ignore them. But set to zero,
        # so these values have no influence on the calculations.
        self.d_mats = {key: nan_to_zeros(self.d_mats[key]) for key in group_keys}
        self.f_mats = {key: nan_to_zeros(self.f_mats[key]) for key in group_keys}
        self.z_mats = {key: nan_to_zeros(self.z_mats[key]) for key in X["Z"]}
        self.y_mats = {key: nan_to_zeros(self.y_mats[key]) for key in X["Y"]}

        # Storage for the model objects. Make a copy only of the Numpy values to use in the Estimator.
        self.observation_names = X["F"][group_keys[0]].index
        self.property_names = {key: self.d_matrix[key].columns.to_list() for key in group_keys}
        self.material_names = {key: self.d_matrix[key].index.to_list() for key in group_keys}
        self.condition_names = {key: X["Z"][key].columns.to_list() for key in X["Z"]}
        self.quality_names = {key: X["Y"][key].columns.to_list() for key in X["Y"]}

        # Create the missing value maps, except we store the opposite, i.e., not missing, since these are more useful.
        # We refer to these as `pmaps` in the code (present maps, as opposed to `mmap` or missing maps).
        self.not_na_d = {key: ~np.isnan(self.d_matrix[key].values) for key in self.d_mats}
        self.not_na_f = {key: ~np.isnan(X["F"][key].values) for key in self.f_mats}
        self.not_na_z = {key: ~np.isnan(X["Z"][key].values) for key in self.z_mats}
        self.not_na_y = {key: ~np.isnan(X["Y"][key].values) for key in self.y_mats}

        # Model parameters. Naming convention: x_i_j
        # x = block letter (P, W, R, T, etc)
        # i = block type: `scores` [for the observations (rows)] or `loadings` [for the variables (columns)]
        # j = block name [z, f, d, y, super]
        # ----------------
        self.t_scores_super: pd.DataFrame = pd.DataFrame(index=self.observation_names)
        self.r_loadings_f: dict[str, pd.DataFrame] = {
            key: pd.DataFrame(index=self.material_names[key]) for key in group_keys
        }
        self.w_loadings_z: dict[str, pd.DataFrame] = {
            key: pd.DataFrame(index=self.condition_names[key]) for key in self.z_mats
        }
        self.w_loadings_super = pd.DataFrame(index=["Z", "F"] if self.n_conditions > 0 else ["F"])
        # Capture the correlation of the properties in D; for the last component.
        self.s_loadings_d: dict[str, pd.DataFrame] = {
            key: pd.DataFrame(index=self.property_names[key]) for key in group_keys
        }
        # Captures the deflation of the properties in D; for the last component.
        self.v_loadings_d: dict[str, pd.DataFrame] = {
            key: pd.DataFrame(index=self.property_names[key]) for key in group_keys
        }
        self.p_loadings_f: dict[str, pd.DataFrame] = {
            key: pd.DataFrame(index=self.material_names[key]) for key in group_keys
        }
        self.p_loadings_z: dict[str, pd.DataFrame] = {
            key: pd.DataFrame(index=self.condition_names[key]) for key in self.z_mats
        }
        self.q_loadings_y: dict[str, pd.DataFrame] = {
            key: pd.DataFrame(index=self.quality_names[key]) for key in self.y_mats
        }

        # Model performance
        # -----------------
        # 1. Prediction matrices (hat matrices for Y-space) in pre-processed space
        self.hat_: dict[str, pd.DataFrame] = {
            key: pd.DataFrame(index=self.observation_names, columns=self.quality_names[key], dtype=float).fillna(0)
            for key in self.y_mats
        }
        # 2. Prediction matrix for the Y-space only, and then scaled back to the original space
        self.hat: dict[str, pd.DataFrame] = {
            key: pd.DataFrame(index=self.observation_names, columns=self.quality_names[key], dtype=float).fillna(0)
            for key in self.y_mats
        }
        # 3. Squared prediction error (SPE) for each observation, per component, per block
        self.spe: dict[str, dict[str, pd.DataFrame]] = {key: {} for key in self.required_blocks_}
        self.spe_limit: dict[str, dict[str, Callable]] = {key: {} for key in self.required_blocks_}

        # 4. Hotelling's T2 values for each observation, per component
        self.hotellings_t2: pd.DataFrame = pd.DataFrame()
        self.hotellings_t2_limit: Callable = hotellings_t2_limit
        self.scaling_factor_for_scores = pd.Series()
        self.ellipse_coordinates: Callable = ellipse_coordinates

        self._fit_iterative_regressions()
        self.is_fitted_ = True
        return self

    def predict(self, X: DataFrameDict) -> Bunch:  # noqa: C901
        """
        Model inference on new data.

        This will pre-process the new data and apply those subsequently to the latent variable model.

        Example
        -------

        # Training phase:
        estimator = TPLS(n_components=2).fit(training_data)

        # Testing/inference phase:
        new_data = {"Z": ..., "F": ...}  # you need at least the F block for a new prediction. "Z" is optional.
        predictions = estimator.predict(new_data_pp)

        Parameters
        ----------
        X : DataFrameDict
            The input samples.

        Returns
        -------
        y : dict
            Returns an array of prediction objects. More details to come here later. Please ask.
        """
        check_is_fitted(self)  # Check if fit had been called
        assert isinstance(X, DataFrameDict), "The input 'X' must be a DataFrameDict object."

        # TODO: Check consistency on the data: the columns names in the new data must match the columns names in the
        # training data.
        x_f: dict[str, pd.DataFrame] = {key: X["F"][key].copy() for key in X["F"]}
        x_z: dict[str, pd.DataFrame] = {key: X["Z"][key].copy() for key in X["Z"]}

        for key, df_f in x_f.items():
            x_f[key] = (df_f - self.preproc_["F"][key]["center"]) / self.preproc_["F"][key]["scale"]

        for key, df_z in x_z.items():
            x_z[key] = (df_z - self.preproc_["Z"][key]["center"]) / self.preproc_["Z"][key]["scale"]

        not_na_f = {key: ~np.isnan(X["F"][key].values) for key in X["F"]}
        not_na_z = {key: ~np.isnan(X["Z"][key].values) for key in X["Z"]}
        names_observations = X["F"][next(iter(X["F"]))].index
        num_obs = names_observations.shape[0]
        spe_f: dict[str, pd.DataFrame] = {
            key: pd.DataFrame(index=x_f[key].index, columns=range(1, self.n_components + 1)) for key in x_f
        }
        spe_z: dict[str, pd.DataFrame] = {
            key: pd.DataFrame(index=x_z[key].index, columns=range(1, self.n_components + 1)) for key in x_z
        }

        t_scores_super = pd.DataFrame(index=names_observations, columns=range(1, self.n_components + 1), dtype=float)
        # Hotelling's T2 values, after so many components. In other words, in column 3, it is the Hotelling's T2
        # computed with 3 components.
        hotellings_t2 = pd.DataFrame(index=names_observations, columns=range(1, self.n_components + 1), dtype=float)
        # Predictions are returned in un-scaled form, so they are in the same units as the training data.
        hat: dict[str, pd.DataFrame] = {
            key: pd.DataFrame(index=names_observations, columns=self.quality_names[key], dtype=float)
            for key in self.y_mats
        }

        for key, df_f in x_f.items():
            assert df_f.shape[0] == num_obs, "All formula blocks must have the same number of rows."
            assert set(df_f.columns) == set(
                self.material_names[key]
            ), f"Columns in block F, group [{key}] must match training data column names for each material"

        for key, df_z in x_z.items():
            assert df_z.shape[0] == num_obs, "All condition blocks must have the same number of rows."
            assert set(df_z.columns) == set(
                self.condition_names[key]
            ), f"Columns names in block Z, group [{key}] must match training data column names."

        for pc_a in range(self.n_components):
            # Regress the row of each new formula block on the r_loadings_f, to get the t-score for that pc_a component.
            # Add up the t-score as you go block by block.
            score_f_a = np.zeros(num_obs)
            denominators = np.zeros(num_obs)
            for key, df_x_f in x_f.items():
                b_row = np.array(self.r_loadings_f[key].iloc[:, pc_a].values)
                # Tile row-by-row to create `n_rows`, and maps missing entries to zero, so they have no effect
                denom = np.tile(b_row, (num_obs, 1)) * not_na_f[key]
                score_f_a += np.array(np.sum(df_x_f.values * denom, axis=1))  # numerator portion
                denominators += np.sum((denom * not_na_f[key]) ** 2, axis=1)

            denominators[denominators == 0] = np.nan  # Guard should not be needed; should never be zeros in here.
            score_f_a /= denominators

            # Repeat for the Z-space: regress the row of each new Z block on the w-loadings, to get the
            # t-score for that pc_a. It seems redundant to divide by w'w, since w is already normalized, but if there
            # are missing values, then that correction is needed, to avoid dividing by a larger value than is fair.
            if self.n_conditions > 0:
                score_z_a = np.zeros(num_obs)
                denominators = np.zeros(num_obs)
                for key, df_x_z in x_z.items():
                    b_row = np.array(self.w_loadings_z[key].iloc[:, pc_a].values)
                    denom = np.tile(b_row, (num_obs, 1)) * not_na_z[key]
                    score_z_a += np.array(np.sum(df_x_z.values * denom, axis=1))
                    denominators += np.sum((denom * not_na_z[key]) ** 2, axis=1)

                # Multiply the individual block scores by the super-weights, to get the super-scores.
                # After transposing below, rows are the observations, and columns are the blocks: [Z, F]
                super_score_a = np.vstack([score_z_a, score_f_a]).T @ np.asarray(
                    self.w_loadings_super.iloc[:, pc_a].values
                ).reshape(-1, 1)
            else:
                # The w_loadings_super are just "1" or "-1" in this case
                super_score_a = score_f_a.reshape(-1, 1) * self.w_loadings_super.iloc[:, pc_a].values

            # Deflate each block (key) in x_f matrices with the super_scores, to get values for the next iteration,
            # and to compute SPE.
            explained_f = {
                key: super_score_a @ np.asarray(self.p_loadings_f[key].iloc[:, pc_a].values).reshape(1, -1)
                for key in x_f
            }
            for key, df_x_f in x_f.items():
                x_f[key] -= explained_f[key]
                spe_f[key].iloc[:, pc_a] = np.sqrt(np.sum(np.square(df_x_f), axis=1))

            explained_z = {
                key: super_score_a @ np.asarray(self.p_loadings_z[key].iloc[:, pc_a].values).reshape(1, -1)
                for key in x_z
            }
            for key, df_x_z in x_z.items():
                x_z[key] -= explained_z[key]
                spe_z[key].iloc[:, pc_a] = np.sqrt(np.sum(np.square(df_x_z), axis=1))

            # Store values for the final output
            t_scores_super.iloc[:, pc_a] = super_score_a.flatten()
            hotellings_t2.iloc[:, pc_a] = np.sum(super_score_a**2, axis=1)

        # After the loop has repeated `self.n_components` times: calculate the predictions using the full set of super
        # scores and the q-loadings for the Y-space.
        for key in self.y_mats:
            hat[key].iloc[:, :] = (t_scores_super.values @ self.q_loadings_y[key].values.T) * self.preproc_["Y"][key][
                "scale"
            ].values[None, :] + self.preproc_["Y"][key]["center"].values[None, :]

        # Calculate the T2 values: for all the spaces
        hotellings_t2.iloc[:, :] = (
            # Last item in the statement here is not super_scores.values !! we want the result back as a DataFrame
            t_scores_super.values
            @ np.diag(np.power(1 / self.scaling_factor_for_scores.values, 2), 0)
            * t_scores_super
        ).cumsum(axis="columns")

        return Bunch(
            hat=hat,
            t_scores_super=t_scores_super,
            spe={"Z": spe_z, "F": spe_f},
            hotellings_t2=hotellings_t2,
        )

    def display_results(self, show_cumulative_stats: bool = True) -> str:
        """Display the results of the model fitting."""

        if not self.is_fitted_:
            raise RuntimeError("The model is not fitted yet. Please call `fit` first.")

        output = f"Hotelling's T2 limit [95% limit]: {self.hotellings_t2_limit():.4g}\n"
        output += f"                     [99% limit]: {self.hotellings_t2_limit(0.99):.4g}\n"
        # output += f"SPE limits: {self.spe_limit['Y'](self.spe['Y'])}\n"
        sep = "------ ---------- ---------- ---------- ----------  -------------\n"
        output += sep
        if show_cumulative_stats:
            header = "LV #   sum(R2: D) sum(R2: Z) sum(R2: F) sum(R2: Y) |    ms [iter]"
        else:
            header = "LV #        R2: D      R2: Z      R2: F      R2: Y |    ms [iter]"

        output += header + "\n" + sep
        r2_d_a_prior = np.mean([r2val.mean() for r2val in self.r2_frac[0]["D"].values()])
        r2_z_a_prior = (
            np.mean([r2val.mean() for r2val in self.r2_frac[0]["Z"].values()]) if self.n_conditions > 0 else 0
        )
        r2_f_a_prior = np.mean([r2val.mean() for r2val in self.r2_frac[0]["F"].values()])
        r2_y_a_prior = np.mean([r2val.mean() for r2val in self.r2_frac[0]["Y"].values()])
        for a in range(1, self.n_components + 1):
            r2_d_a = np.mean([np.nanmean(r2val) for r2val in self.r2_frac[a]["D"].values()])
            r2_z_a = (
                np.mean([np.nanmean(r2val) for r2val in self.r2_frac[a]["Z"].values()]) if self.n_conditions > 0 else 0
            )
            r2_f_a = np.mean([np.nanmean(r2val) for r2val in self.r2_frac[a]["F"].values()])
            r2_y_a = np.mean([np.nanmean(r2val) for r2val in self.r2_frac[a]["Y"].values()])
            if show_cumulative_stats:
                r2_d_a += r2_d_a_prior
                r2_z_a += r2_z_a_prior
                r2_f_a += r2_f_a_prior
                r2_y_a += r2_y_a_prior

            r2_d_a_prior = r2_d_a
            r2_z_a_prior = r2_z_a
            r2_f_a_prior = r2_f_a
            r2_y_a_prior = r2_y_a
            r2_z_a = f"{r2_z_a:>10.1f}" if self.n_conditions > 0 else "        -"

            # Calculate time per iteration for this component
            time_ms = self.fitting_statistics["milliseconds"][a - 1]
            iterations = self.fitting_statistics["iterations"][a - 1]
            time_iter = f"{time_ms:>5.0f} [{iterations:>3d}]"

            line = (
                f"LV {a:<2}  {r2_d_a * 100:>10.1f} {r2_z_a} {r2_f_a * 100:>10.1f} {r2_y_a * 100:>10.1f} "
                f"|{time_iter:>13}"
            )
            if self.fitting_statistics["iterations"][a - 1] >= self.max_iterations:
                line += "** (max iter reached)"
            output += line + "\n"

        output += sep
        ms_per_iter = round(
            sum(self.fitting_statistics["milliseconds"]) / sum(self.fitting_statistics["iterations"]), 2
        )
        output += f"Timing: {ms_per_iter} ms/iter; {sum(self.fitting_statistics['iterations'])} iterations required\n"
        output += f"Average tolerance: {np.mean(self.fitting_statistics['convergance_tolerance']):.4g}\n"

        return output

    def score(self, X: DataFrameDict, y: None = None, sample_weight: None | np.ndarray = None) -> float:  # noqa: ARG002
        """Return r2_score` on test data.

        See RegressorMixin.score for more details.

        Parameters
        ----------
        X : DataFrameDict
            Test samples.

        y : Not used. In the `X` input, there is a already a "Y" block. This will be the Y-data.

        sample_weight : Not used.

        Returns
        -------
        score : float
            :math:`R^2` of ``self.predict(X)``.
        """
        predictions = self.predict(X)
        y_pred = predictions.hat
        y_actual = X["Y"]
        r2_key = 0.0
        for _idx, key in enumerate(y_actual):
            r2_key += r2_score(y_true=y_actual[key], y_pred=y_pred[key], sample_weight=sample_weight)
            _ = np.corrcoef(y_actual[key].values.ravel(), y_pred[key].values.ravel())
        return r2_key / (_idx + 1)

    def help(self) -> str:
        """Help for the TPLS Estimator.

        Data organization
        -----------------

        Quick tips
        ----------
        Build model:                tpls = TPLS(n_components=2, d_matrix=d_matrix).fit(X)
        Get model's predictions:    tpls.hat            <-- the hat-matrix, i.e., the predictions
        Predict on new data:        tpls.predict(X_new)
        See model summary:          tpls.display_results()
        This help page:             tpls.help()

        Statistical values
        ------------------

        .t_scores_super             Super scores for the entire model                           [pd.DataFrame]
        .hotellings_t2              Hotelling's T2 values for each observation, per component   [pd.DataFrame]
        .spe                        Squared prediction error for each block                     [dict of pd.DataFrames]


        .hotellings_t2_limit()      Returns the Hotelling's T2 limit for the model              [float]
        .spe_limit[block]()         Return the SPE limit for the block, `.spe_limit["Y"]()`.    [float]
                Returns the SPE limit for the Y block.


        TODO:
        self.hotellings_t2: pd.DataFrame = pd.DataFrame()
        self.hotellings_t2_limit: Callable = hotellings_t2_limit
        self.scaling_factor_for_scores = pd.Series()
        self.ellipse_coordinates: Callable = ellipse_coordinates

        """

        # Return this function's docstring as the help text.
        # Dedent the self.__docs__ string and return that
        return self.help.__doc__.replace("        ", "").replace("\n\n", "\n").strip()

    def _input_data_checks(self, X: DataFrameDict) -> None:
        """Check the incoming data."""
        assert isinstance(X, DataFrameDict), "The input data must be a DataFrameDict."
        assert set(X.keys()) == self.required_inputs_, f"Expected keys: {self.required_inputs_}, got: {set(X.keys())}"
        group_keys = [str(key) for key in self.d_matrix]
        assert set(X["F"]) == set(group_keys), "The keys in F must match the keys in D."

        for key in X["Y"]:
            self._validate_df(X["Y"][key])
        for key in X["Z"]:
            self._validate_df(X["Z"][key])
        for key in self.d_matrix:
            self._validate_df(self.d_matrix[key])
            assert key in X["F"], f"Block/group name '{key}' in D must also be present in F."
            self._validate_df(X["F"][key])  # this also ensures the keys in F are the same as in D

    def _learn_center_and_scaling_parameters(self, y: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """
        Learn the centering and scaling parameters for the output space.

        Parameters
        ----------
        y : pd.DataFrame
            The output space.

        Returns
        -------
        centering : pd.Series
            The centering parameters.

        scaling : pd.Series
            The scaling parameters.
        """
        centering = y.mean(axis="index")
        scaling = y.std(ddof=1, axis="index") if y.shape[0] > 1 else pd.Series(1.0, index=y.columns)
        scaling[scaling < self.tolerance_] = 1.0  # columns with little/no variance are left as-is.
        return centering, scaling

    def _validate_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate a single dataframe using `check_array` from scikit-learn.

        Parameters
        ----------
        df : {pd.DataFrame}

        Returns
        -------
        y : {pd.DataFrame}
            Returns the input dataframe.
        """
        # Ensure all columns are dtype "float64" or "int64"
        if not all(good_cols := [isinstance(col, (np.dtypes.Float64DType, np.dtypes.IntDType)) for col in df.dtypes]):
            bad_columns = df.columns[[not item for item in good_cols]].to_list()
            raise ValueError(
                f"All columns in the DataFrame must be of type float64 or int64. Bad columns: {bad_columns}"
            )

        return check_array(
            df, accept_sparse=False, ensure_all_finite="allow-nan", ensure_2d=True, allow_nd=False, ensure_min_samples=1
        )

    def _has_converged(self, starting_vector: np.ndarray, revised_vector: np.ndarray, iterations: int) -> bool:
        """
        Terminate the iterative algorithm when any one of these conditions is True.

        #. scores converge: the norm between two successive iterations is smaller than a tolerance
        #. maximum number of iterations is reached
        """
        delta_gap = float(
            np.linalg.norm(starting_vector - revised_vector, ord=None) / np.linalg.norm(starting_vector, ord=None)
        )
        converged = delta_gap < self.tolerance_
        max_iter = iterations >= self.max_iterations
        return bool(np.any([max_iter, converged]))

    def _store_model_coefficients(  # noqa: PLR0913
        self,
        pc_a_column: int,  # one-based index for the component
        t_super_i: np.ndarray,
        r_i: dict[str, np.ndarray],
        w_i_z: dict[str, np.ndarray],
        w_super_i: np.ndarray,
        s_i: dict[str, np.ndarray],
    ) -> None:
        """Store the model coefficients for later use."""

        self.t_scores_super = self.t_scores_super.join(
            pd.DataFrame(t_super_i, index=self.observation_names, columns=[pc_a_column])
        )

        # These are loadings really, not scores, for each group in the F block.
        self.r_loadings_f = {
            key: self.r_loadings_f[key].join(
                pd.DataFrame(r_i[key], index=self.material_names[key], columns=[pc_a_column])
            )
            for key in r_i
        }

        # These are the loadings for the Z space
        self.w_loadings_z = {
            key: self.w_loadings_z[key].join(
                pd.DataFrame(w_i_z[key], index=self.condition_names[key], columns=[pc_a_column])
            )
            for key in w_i_z
        }

        self.w_loadings_super = self.w_loadings_super.join(
            pd.DataFrame(w_super_i, index=["Z", "F"] if self.n_conditions > 0 else ["F"], columns=[pc_a_column])
        )

        self.s_loadings_d = {
            key: self.s_loadings_d[key].join(
                pd.DataFrame(s_i[key], index=self.property_names[key], columns=[pc_a_column])
            )
            for key in s_i
        }

    def _calculate_and_store_deflation_matrices(
        self,
        pc_a: int,
        t_super_i: np.ndarray,
        q_super_i: np.ndarray,
        r_i: dict[str, np.ndarray],
    ) -> None:
        """
        Calculate and store the deflation matrices for the TPLS model.

        Deflate the matrices stored in the instance object.

        Returns the prediction matrices in a dictionary.
        """
        # Step 13: Deflate the Z matrix with a loadings vector, pz_b (_b is for block)
        pz_b = {
            key: regress_a_space_on_b_row(df_z.T, t_super_i.T, pmap_z.T)
            for key, df_z, pmap_z in zip(self.z_mats.keys(), self.z_mats.values(), self.not_na_z.values(), strict=True)
        }
        for key in self.z_mats:
            self.z_mats[key] -= (t_super_i @ pz_b[key].T) * self.not_na_z[key]
        self.p_loadings_z = {
            key: self.p_loadings_z[key].join(pd.DataFrame(pz_b[key], index=self.condition_names[key], columns=[pc_a]))
            for key in pz_b
        }

        # Step 13. p_i = F_i' t_i / t_i't_i. Regress the columns of F_i on t_i; store slope coeff in vectors p_i.
        # Note: the "t" vector is the t_i vector from the inner PLS model, marked as "Tt" in figure 4 of the paper.
        # It is the score column from the super score matrix regression onto Y.
        pf_i = {
            key: regress_a_space_on_b_row(df_f.T, t_super_i.T, pmap_f.T)
            for key, df_f, pmap_f in zip(self.f_mats.keys(), self.f_mats.values(), self.not_na_f.values(), strict=True)
        }
        self.p_loadings_f = {
            key: self.p_loadings_f[key].join(pd.DataFrame(pf_i[key], index=self.material_names[key], columns=[pc_a]))
            for key in pf_i
        }
        # Step 13: v_i = D_i' r_i / r_i'r_i. Regress the rows of D_i (properties) on r_i; store slopes in v_i.
        self.v_loadings_d = {
            key: self.v_loadings_d[key].join(
                pd.DataFrame(
                    regress_a_space_on_b_row(df_d.T, r_i[key].T, pmap_d.T),
                    index=self.property_names[key],
                    columns=[pc_a],
                )
            )
            for key, df_d, pmap_d in zip(self.d_mats.keys(), self.d_mats.values(), self.not_na_d.values(), strict=True)
        }
        # Step 14. Do the actual deflation.
        for key in self.d_mats:
            # Step to deflate F matrix
            self.f_mats[key] -= (t_super_i @ pf_i[key].T) * self.not_na_f[key]

            # Two sets of matrices to deflate: properties D and formulas F.
            self.d_mats[key] -= (r_i[key] @ self.v_loadings_d[key].iloc[:, [-1]].T) * self.not_na_d[key]

        # Deflate the Y-space as well
        self.q_loadings_y = {
            key: self.q_loadings_y[key].join(pd.DataFrame(q_super_i, index=self.quality_names[key], columns=[pc_a]))
            for key in self.y_mats
        }
        for key in self.y_mats:
            self.hat_[key] += t_super_i @ q_super_i.T
            self.y_mats[key] -= (t_super_i @ q_super_i.T) * self.not_na_y[key]

    def _update_performance_statistics(self) -> None:
        """Calculate and store the performance statistics of the model, such as SSQ, R2, etc."""
        # Calculate the sums of squares for each block, per column.
        # Note: the `ddof=0` is used to calculate the population variance, which is proportional to the SSQ.
        calc_ssq = {
            "D": {key: np.nansum(self.d_mats[key] ** 2, axis=0) for key in self.d_mats},
            "F": {key: np.nansum(self.f_mats[key] ** 2, axis=0) for key in self.f_mats},
            "Z": {key: np.nansum(self.z_mats[key] ** 2, axis=0) for key in self.z_mats},
            "Y": {key: np.nansum(self.y_mats[key] ** 2, axis=0) for key in self.y_mats},
        }
        self.sums_of_squares_.append(calc_ssq)

        # Calculate the incremental (not cumulative!) R2 values for each block, per column:
        # Cumulative R2 values can be found by summation. The R2 values are **always** fractional (between 0 and 1).
        ssq_prior_pc = self.sums_of_squares_[-2]
        ssq_start_0 = self.sums_of_squares_[0]
        calc_r2 = {
            "D": {key: (ssq_prior_pc["D"][key] - calc_ssq["D"][key]) / ssq_start_0["D"][key] for key in self.d_mats},
            "F": {key: (ssq_prior_pc["F"][key] - calc_ssq["F"][key]) / ssq_start_0["F"][key] for key in self.f_mats},
            "Z": {key: (ssq_prior_pc["Z"][key] - calc_ssq["Z"][key]) / ssq_start_0["Z"][key] for key in self.z_mats},
            "Y": {key: (ssq_prior_pc["Y"][key] - calc_ssq["Y"][key]) / ssq_start_0["Y"][key] for key in self.y_mats},
        }
        self.r2_frac.append(calc_r2)

        # VIP for each block, for given number of components we currently have. VIP are cumulative.
        # For the D-block and F-block: you want to be able to use the VIPs to compare across the entire block, without
        # regards to the banding in groups that might have to be done. So use the total R2 for that block, and do not
        # work per group.
        r2_d_a: list[float] = [
            float(np.mean([np.nanmean(r2val) for r2val in r2_frac["D"].values()])) for r2_frac in self.r2_frac
        ]
        r2_f_a: list[float] = [
            float(np.mean([np.nanmean(r2val) for r2val in r2_frac["F"].values()])) for r2_frac in self.r2_frac
        ]
        loadings_s = np.concatenate(list(self.s_loadings_d.values()))
        loadings_f = np.concatenate(list(self.p_loadings_f.values()))
        vip_d = self._calculate_vip(loadings_s, np.array(r2_d_a[1:]))
        vip_f = self._calculate_vip(loadings_f, np.array(r2_f_a[1:]))
        # Split the `vip_d` back into the original groups that it was merged from.
        # For example: if there are two groups, one with 17 columns, and the second with 3 columns, then there are
        # a total of 20 values in `vip_d`, and the first 17 values correspond to the first group, and the last 3 values.
        # Create a dictionary with the group names as keys, and the VIP values as values, split correctly:
        vip_split_d = {}
        start = 0
        for key in self.property_names:
            end = start + len(self.property_names[key])
            vip_split_d[key] = pd.Series(vip_d[start:end], index=self.property_names[key])
            start = end

        vip_split_f = {}
        start = 0
        for key in self.material_names:
            end = start + len(self.material_names[key])
            vip_split_f[key] = pd.Series(vip_f[start:end], index=self.material_names[key])
            start = end

        self.feature_importance["D"] = vip_split_d  # TODO: should it not be based on deflated matrices? S(V^TS)^{-1}
        self.feature_importance["F"] = vip_split_f  # TODO: should it not be based on deflated matrices? P(_^TP)^{-1}

    def _calculate_vip(self, loadings: np.ndarray, r2_vector: np.ndarray) -> np.ndarray:
        """Calculate the VIP values for the current component.

        The `loadings` has as many rows as there are feature varaibles, and A columns, where A = number of components.
        The `r2_vector` is a vector of fractional R^2 values for the current component, with `A` entries.
        The `r2_vector` values should be between 0 and 1; the fraction of variance explained by the component for that
        given `loadings` matrix.

        The VIP values are calculated as follows:
            VIP = sqrt(n * sum((r2_vector * (loadings ** 2)) / sum(r2_vector)))

        where n is the number of features (rows in the loadings matrix).
        """
        # VIP = sqrt(n * sum((r2_vector * (loadings ** 2)) / sum(r2_vector)))
        n = loadings.shape[0]
        r2_vector = r2_vector.reshape(1, -1)  # Ensure r2_vector is a row vector
        return np.sqrt(n * np.sum(r2_vector * (loadings**2), axis=1) / np.sum(r2_vector))

    def _calculate_model_statistics_and_limits(self) -> None:
        """Calculate and store the model limits.

        Limits calculated:
        1. Hotelling's T2 limits
        2. Squared prediction error limits

        Other calculations:
        1. The model's Y-space predictions are scaled back to the original space.
        """

        # Calculate the Hotelling's T2 values, and limits. Could do a ddof correction (n-1) for the variance matrix.
        variance_matrix = self.t_scores_super.T @ self.t_scores_super / self.t_scores_super.shape[0]
        self.hotellings_t2[:, :] = np.sum(
            (self.t_scores_super.values @ np.linalg.inv(variance_matrix)) * self.t_scores_super, axis=1
        )
        self.hotellings_t2_limit = partial(
            hotellings_t2_limit, n_components=self.n_components, n_rows=self.hotellings_t2.shape[0]
        )
        self.scaling_factor_for_scores = pd.Series(
            np.sqrt(np.diag(variance_matrix)),
            index=[a + 1 for a in range(self.n_components)],
            name="Standard deviation per score",
        )
        self.ellipse_coordinates = partial(
            ellipse_coordinates,
            n_components=self.n_components,
            scaling_factor_for_scores=self.scaling_factor_for_scores,
            n_rows=self.t_scores_super.shape[0],
        )

        # Squared prediction error limits. This is a measure of the prediction error = difference between the actual
        # and predicted values. Since the matrices are deflated by the predictive part of the model already, the
        # data in these matrices is already the prediction error. Calculate the **squared** portion, and store it.
        column_name = [f"SPE with A={self.n_components}"]
        self.spe["Y"] = {
            key: pd.DataFrame(
                np.sqrt(np.sum(np.square(self.y_mats[key]), axis=1, keepdims=True)),
                index=self.observation_names,
                columns=column_name,
            )
            for key in self.y_mats
        }
        self.spe_limit["Y"] = {key: partial(spe_calculation, self.spe["Y"][key].values) for key in self.y_mats}
        self.spe["Z"] = {
            key: pd.DataFrame(
                np.sqrt(np.sum(np.square(self.z_mats[key]), axis=1, keepdims=True)),
                index=self.observation_names,
                columns=column_name,
            )
            for key in self.z_mats
        }
        self.spe_limit["Z"] = {key: partial(spe_calculation, self.spe["Z"][key].values) for key in self.z_mats}

        # SPE for the D-space. There are two options: per property feature, or per material feature.
        self.spe["D"] = {key: self.d_mats[key].pow(2).sum(axis="columns").pow(0.5) for key in self.d_mats}
        self.spe_limit["D"] = {key: partial(spe_calculation, self.spe["D"][key].values) for key in self.d_mats}
        self.spe["F"] = {
            key: pd.DataFrame(
                np.sqrt(np.sum(np.square(self.f_mats[key]), axis=1, keepdims=True)),
                index=self.observation_names,
                columns=column_name,
            )
            for key in self.f_mats
        }
        self.spe_limit["F"] = {key: partial(spe_calculation, self.spe["F"][key].values) for key in self.f_mats}

        # Y-space predictions
        for key in self.y_mats:
            # The Y-space predictions are already in the pre-processed space, so we need to scale them back to the
            self.hat[key] = pd.DataFrame(self.hat_[key], index=self.observation_names, columns=self.quality_names[key])
            self.hat[key] = self.hat[key].multiply(self.preproc_["Y"][key]["scale"].values[None, :], axis=1)
            self.hat[key] += self.preproc_["Y"][key]["center"].values[None, :]

    def _preprocess_data(self) -> None:
        """Pre-process the training data."""

        for key in self.f_mats:
            self.f_mats[key] = (self.f_mats[key] - self.preproc_["F"][key]["center"].values[None, :]) / self.preproc_[
                "F"
            ][key]["scale"].values[None, :]
            self.d_mats[key] = (
                (self.d_mats[key] - self.preproc_["D"][key]["center"].values[None, :])
                / self.preproc_["D"][key]["scale"].values[None, :]
                / self.preproc_["D"][key]["block"][0]  # scalar!
            )
        for key in self.z_mats:
            self.z_mats[key] = (self.z_mats[key] - self.preproc_["Z"][key]["center"].values[None, :]) / self.preproc_[
                "Z"
            ][key]["scale"].values[None, :]

        for key in self.y_mats:
            self.y_mats[key] = (self.y_mats[key] - self.preproc_["Y"][key]["center"].values[None, :]) / self.preproc_[
                "Y"
            ][key]["scale"].values[None, :]

        # Test that all blocks and groups within a block have a mean of 0 and a standard deviation of 1.
        # Note the extra complexity for checking columns that have perfectly zero variance.
        for key in self.z_mats:
            assert pytest.approx(0) == np.nanmean(self.z_mats[key], axis=0)
            for item in np.nanvar(self.z_mats[key], axis=0, ddof=1):
                if item != 0:
                    assert pytest.approx(item) == 1

        # Check the F, D, and Y blocks as well.
        for key in self.f_mats:
            assert pytest.approx(0) == np.nanmean(self.f_mats[key], axis=0)
            for item in np.nanvar(self.f_mats[key], axis=0, ddof=1):
                if item != 0:
                    assert pytest.approx(item) == 1

        assert all(pytest.approx(np.nanmean(self.y_mats[key], axis=0)) == 0 for key in self.y_mats)
        assert all(
            pytest.approx(np.where((in_array := np.nanstd(self.y_mats[key], axis=0, ddof=1)) == 0, 1, in_array)) == 1
            for key in self.y_mats
        )

        assert all(pytest.approx(np.nanmean(self.d_mats[key], axis=0)) == 0 for key in self.d_mats)
        assert all(
            pytest.approx(
                np.where(
                    (
                        in_array := np.nanstd(self.d_mats[key], axis=0, ddof=1)
                        * self.preproc_["D"][key]["block"].values[0]
                    )
                    == 0,
                    1,
                    in_array,
                )
            )
            == 1
            for key in self.d_mats
        )

    def _fit_iterative_regressions(self) -> None:
        """Fit the model via iterative regressions and store the model coefficients in the class instance."""

        # Formula matrix: assemble all not-na maps from blocks in F: make a single matrix.
        pmap_f = np.concatenate(list(self.not_na_f.values()), axis=1)

        # Follow the steps in the paper on page 54
        for pc_a in range(self.n_components):
            n_iter = 0
            milliseconds_start = time.time()

            # Step 1: Select any column in Y as initial guess (they have all be scaled anyway, so it doesn't matter)
            u_super_i = next(iter(self.y_mats.values()))[:, [0]]
            u_prior = u_super_i + 1

            while not self._has_converged(starting_vector=u_prior, revised_vector=u_super_i, iterations=n_iter):
                n_iter += 1
                u_prior = u_super_i.copy()
                # Step 2. h_i = F_i' u / u'u. Regress the columns of F on u_i, and store the slope coeff in vectors h_i
                h_i = {
                    key: regress_a_space_on_b_row(df_f.T, u_super_i.T, pmap_f.T)
                    for key, df_f, pmap_f in zip(
                        self.f_mats.keys(), self.f_mats.values(), self.not_na_f.values(), strict=True
                    )
                }

                # Step 3. s_i = D_i' h_i / h_i'h_i. Regress the rows of D_i on h_i, and store slope coeff in vectors s_i
                s_i = {
                    key: regress_a_space_on_b_row(df_d.T, h_i[key].T, pmap_d.T)
                    for key, df_d, pmap_d in zip(
                        self.d_mats.keys(), self.d_mats.values(), self.not_na_d.values(), strict=True
                    )
                }
                # Step 4: combine the entries in s_i to form a joint `s` and normalize it to unit length.
                joint_s_normalized = np.linalg.norm(np.concatenate(list(s_i.values())))
                s_i = {key: s / joint_s_normalized for key, s in s_i.items()}

                # Step 5: r_i = D_i s_i / s_i's_i. Regress columns of D_i on s_i, and store slope coefficients in r_i.
                r_i = {
                    key: regress_a_space_on_b_row(df_d, s_i[key].T, self.not_na_d[key])
                    for key, df_d in zip(self.d_mats.keys(), self.d_mats.values(), strict=True)
                }

                # Step 6: Combine the entries in r_i to form a joint r (which is the name of the method in the paper).
                #         Horizontally concatenate all matrices in F_i to form a joint F matrix.
                #         Regress rows of the joint F matrix onto the joint r vector. Store coeff in block scores, t_f
                joint_r = np.concatenate(list(r_i.values()))
                joint_f = np.concatenate(list(self.f_mats.values()), axis=1)
                t_f = regress_a_space_on_b_row(joint_f, joint_r.T, pmap_f)

                # If there is a Condition matrix (non-empty Z block)
                if self.n_conditions > 0:
                    # Step 7: w_i = Z_i' u / u'u. Regress the columns of Z on u_i, and store the slope coefficients
                    #         in vectors w_i.
                    w_i_z = {
                        key: regress_a_space_on_b_row(df_z.T, u_super_i.T, self.not_na_z[key].T)
                        for key, df_z in zip(self.z_mats.keys(), self.z_mats.values(), strict=True)
                    }

                    # Step 8: Normalize joint w to unit length. See MB-PLS by Westerhuis et al. 1998. This is normal.
                    w_i_z = {key: w / np.linalg.norm(w) for key, w in w_i_z.items()}

                    # Step 9: regress rows of Z on w_i, and store slope coefficients in t_z. There is an error in the
                    #        paper here, but in figure 4 it is clear what should be happening.
                    t_zb = {
                        key: regress_a_space_on_b_row(df_z, w_i_z[key].T, self.not_na_z[key])
                        for key, df_z in zip(self.z_mats.keys(), self.z_mats.values(), strict=True)
                    }
                    t_z = np.concatenate(list(t_zb.values()), axis=1)

                else:
                    # Step 7: No Z block. Take an empty matrix across to the the superblock.
                    w_i_z = {}
                    t_z = np.zeros((t_f.shape[0], 0))  # empty matrix: in other words, no Z block

                # Step 10: Combine t_z and t_f to form a joint t matrix.
                t_combined = np.concatenate([t_z, t_f], axis=1)

                # Step 11: Build an inner PLS model: using the t_combined as the X matrix, and the Y (quality space)
                #          as the Y matrix.
                inner_pls = internal_pls_nipals_fit_one_pc(
                    x_space=t_combined,
                    y_space=np.array(next(iter(self.y_mats.values()))),
                    x_present_map=np.ones(t_combined.shape).astype(bool),
                    y_present_map=np.array(next(iter(self.not_na_y.values()))),
                )
                u_super_i = inner_pls["u_i"]  # only used for convergence check; not stored or used further
                t_super_i = inner_pls["t_i"]
                q_super_i = inner_pls["q_i"]
                w_super_i = inner_pls["w_i"]

            # After convergance. Step 12: Now store information.
            # =================
            delta_gap = float(np.linalg.norm(u_prior - u_super_i, ord=None) / np.linalg.norm(u_prior, ord=None))
            self.fitting_statistics["iterations"].append(n_iter)
            self.fitting_statistics["convergance_tolerance"].append(delta_gap)
            self.fitting_statistics["milliseconds"].append((time.time() - milliseconds_start) * 1000)

            # Store model coefficients
            self._store_model_coefficients(
                pc_a + 1, t_super_i=t_super_i, r_i=r_i, w_i_z=w_i_z, w_super_i=w_super_i, s_i=s_i
            )

            # Calculate and store the deflation vectors. See equation 7 on page 55.
            self._calculate_and_store_deflation_matrices(pc_a + 1, t_super_i=t_super_i, q_super_i=q_super_i, r_i=r_i)

            # Update performance statistics for this component
            self._update_performance_statistics()

        # Step 15: Calculate the final model limits (after all components have been fitted).
        self._calculate_model_statistics_and_limits()

    # def _calculate_r2_score(
    #     self, y_true: dict[str, pd.DataFrame], y_pred: dict[str, pd.DataFrame], sample_weight: np.ndarray|None = None
    # ) -> float:
    #     """Calculate R^2 score across all Y blocks."""
    #     total_ss_res = 0.0
    #     total_ss_tot = 1e-10

    # for key in y_true.keys():
    #     y_true_values = y_true[key].values
    #     y_pred_values = y_pred[key].values

    #     # Handle sample weights
    #     if sample_weight is not None:
    #         weights = sample_weight.reshape(-1, 1)
    #         # Residual sum of squares (weighted)
    #         ss_res = np.sum(weights * (y_true_values - y_pred_values) ** 2)
    #         # Total sum of squares (weighted)
    #         y_mean_weighted = np.average(y_true_values, weights=sample_weight.flatten(), axis=0)
    #         ss_tot = np.sum(weights * (y_true_values - y_mean_weighted) ** 2)
    #     else:
    #         # Residual sum of squares
    #         ss_res = np.sum((y_true_values - y_pred_values) ** 2)
    #         # Total sum of squares
    #         y_mean = np.mean(y_true_values, axis=0)
    #         ss_tot = np.sum((y_true_values - y_mean) ** 2)

    #     total_ss_res += ss_res
    #     total_ss_tot += ss_tot

    # # Calculate R² = 1 - (SS_res / SS_tot)
    # if total_ss_tot == 0:
    #     return 0.0 if total_ss_res == 0 else float("-inf")

    #    return 1.0 - (total_ss_res / total_ss_tot)


class Plot:
    """Create plots of estimators."""

    def __init__(self, parent: BaseEstimator) -> None:
        self._parent = parent

    def scores(self, pc_horiz: int = 1, pc_vert: int = 2, **kwargs) -> go.Figure:
        """Generate a score plot."""
        return score_plot(self, pc_horiz=pc_horiz, pc_vert=pc_vert, **kwargs)

    def loadings(self, pc_horiz: int = 1, pc_vert: int = 2, **kwargs) -> go.Figure:
        """Generate a loading plot."""
        return loading_plot(self, pc_horiz=pc_horiz, pc_vert=pc_vert, **kwargs)
