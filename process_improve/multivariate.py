# (c) Kevin Dunn, 2010-2021. MIT License. Based on own private work over the years.
import time
from functools import partial
from typing import Optional, Any
import warnings

import numpy as np
import pandas as pd
from scipy.stats import f, chi2
from scipy.sparse import issparse

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA as PCA_sklearn
from sklearn.cross_decomposition import PLSRegression
from sklearn.utils.validation import check_is_fitted

epsqrt = np.sqrt(np.finfo(float).eps)


class SpecificationWarning(UserWarning):
    """ Parent warning class. """

    pass


def T2_limit(conf_level: float = 0.95, n_components: int = 0, n_rows: int = 0) -> float:
    """Returns the Hotelling's T2 value at the given level of confidence.

    Parameters
    ----------
    conf_level : float, optional
        Fractional confidence limit, less that 1.00; by default 0.95

    Returns
    -------
    float
        The Hotelling's T2 limit at the given level of confidence.
    """
    assert conf_level > 0.0
    assert conf_level < 1.0
    assert n_rows > 0
    A, N = n_components, n_rows
    return A * (N - 1) * (N + 1) / (N * (N - A)) * f.isf((1 - conf_level), A, N - A)


def ellipse_coordinates(
    score_horiz: int,
    score_vert: int,
    T2_limit_conf_level: float = 0.95,
    n_points: int = 100,
    n_components: int = 0,
    scaling_factor_for_scores=None,
    n_rows: int = 0,
) -> tuple:
    """Get the (score_horiz, score_vert) coordinate pairs that form the T2 ellipse when
        plotting the score `score_horiz` on the horizontal axis and `score_vert` on the
        vertical axis.

        Scores are referred to by number, starting at 1 and ending with `model.components_`


    Parameters
    ----------
    score_horiz : int
        [description]
    score_vert : int
        [description]
    T2_limit_conf_level : float
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

        (t_horiz/s_h)^2 + (t_vert/s_v)^2  =  T2_limit_alpha
        s_horiz = stddev(T_horiz)
        s_vert  = stddev(T_vert)
        T2_limit_alpha = T2 confidence limit at a given alpha value

    Equation of ellipse, *parametric* form (http://en.wikipedia.org/wiki/Ellipse):

        t_horiz = sqrt(T2_limit_alpha)*s_h*cos(t)
        t_vert  = sqrt(T2_limit_alpha)*s_v*sin(t)

        where t ranges between 0 and 2*pi.
    """
    assert score_horiz >= 1
    assert score_vert >= 1
    assert score_horiz <= n_components
    assert score_vert <= n_components
    assert T2_limit_conf_level > 0
    assert T2_limit_conf_level < 1
    assert n_rows > 0
    s_h = scaling_factor_for_scores[score_horiz - 1]
    s_v = scaling_factor_for_scores[score_vert - 1]
    T2_limit_specific = np.sqrt(
        T2_limit(T2_limit_conf_level, n_components=n_components, n_rows=n_rows)
    )
    dt = 2 * np.pi / (n_points - 1)
    steps = np.linspace(0, n_points - 1, n_points)
    x = np.cos(steps * dt) * T2_limit_specific * s_h
    y = np.sin(steps * dt) * T2_limit_specific * s_v
    return x, y


class PCA(PCA_sklearn):
    def __init__(
        self,
        n_components=None,
        *,
        copy=True,
        whiten=False,
        svd_solver="auto",
        tol=0.0,
        iterated_power="auto",
        random_state=None,
        # Own extra inputs, for the case when there is missing data
        missing_data_settings: Optional[dict] = None,
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
        self.missing_data_settings = missing_data_settings
        self.has_missing_data = False

    def fit(self, X, y=None) -> PCA_sklearn:

        self.N, self.K = X.shape

        # Check if number of components is supported against maximum requested
        min_dim = min(self.N, self.K)
        self.A = min_dim if self.n_components is None else int(self.n_components)
        if self.A > min_dim:
            warn = (
                "The requested number of components is more than can be "
                "computed from data. The maximum number of components is "
                f"the minimum of either the number of rows ({self.N}) or "
                f"the number of columns ({self.K})."
            )
            warnings.warn(warn, SpecificationWarning)
            self.A = self.n_components = min_dim

        if np.any(X.isna()):
            # If there are missing data, then the missing data settings apply. Defaults are:
            #
            #       md_method = "pmp"
            #       md_tol = np.sqrt(np.finfo(float).eps)
            #       md_max_iter = (1000,)

            default_mds = dict(md_method="tsr", md_tol=epsqrt, md_max_iter=100)
            if isinstance(self.missing_data_settings, dict):
                default_mds.update(self.missing_data_settings)

            self.missing_data_settings = default_mds
            self = PCA_missing_values(
                n_components=self.n_components,
                random_state=self.random_state,
                missing_data_settings=self.missing_data_settings,
            )
            self.N, self.K = X.shape
            self.A = self.n_components
            self.fit(X)

        else:
            self = super().fit(X)
            self.components_ = self.components_.copy().T

        # We have now fitted the model. Apply some convenience shortcuts for the user.
        self.A = self.n_components
        self.N = self.n_samples_
        self.K = self.n_features_
        # Note: this one is transposed, to conform to standards

        self.loadings = pd.DataFrame(self.components_.copy())
        self.loadings.index = X.columns

        component_names = [f"PC {a+1}" for a in range(self.A)]
        self.loadings.columns = component_names

        self.scaling_factor_for_scores = pd.Series(
            np.sqrt(self.explained_variance_),
            index=component_names,
            name="Standard deviation per score",
        )
        self.Hotellings_T2 = pd.DataFrame(
            np.zeros(shape=(self.N, self.A)),
            columns=component_names,
            index=X.index,
            # name="Hotelling's T^2 statistic, per component",
        )
        if self.has_missing_data:
            self.t_scores = pd.DataFrame(self.t_scores, columns=component_names, index=X.index)
            self.squared_prediction_error = pd.DataFrame(
                self.squared_prediction_error,
                columns=component_names,
                index=X.index.copy(),
            )
            self.R2 = pd.Series(
                self.R2,
                index=component_names,
                name="Model's R^2, per component",
            )
            self.R2cum = pd.Series(
                self.R2cum,
                index=component_names,
                name="Cumulative model's R^2, per component",
            )
            self.R2k_cum = pd.DataFrame(
                self.R2k_cum,
                columns=component_names,
                index=X.columns,
                # name ="Per variable R^2, per component"
            )
        else:
            self.t_scores = pd.DataFrame(
                super().fit_transform(X), columns=component_names, index=X.index
            )
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
            self.R2k_cum = pd.DataFrame(
                np.zeros(shape=(self.K, self.A)),
                columns=component_names,
                index=X.columns,
                # name ="Per variable R^2, per component"
            )

        if not self.has_missing_data:
            Xd = X.copy()
            prior_SS_col = ssq(Xd.values, axis=0)
            base_variance = np.sum(prior_SS_col)
            for a in range(self.A):
                Xd -= self.t_scores.iloc[:, [a]] @ self.loadings.iloc[:, [a]].T
                # These are the Residual Sums of Squares (RSS); i.e X-X_hat
                row_SSX = ssq(Xd.values, axis=1)
                col_SSX = ssq(Xd.values, axis=0)

                # Don't use a check correction factor. Define SPE simply as the sum of squares of
                # the errors, then take the square root, so it is interpreted like a standard error.
                # If the user wants to normalize it, then this is a clean base value to start from.
                self.squared_prediction_error.iloc[:, a] = np.sqrt(row_SSX)

                # TODO: some entries in prior_SS_col can be zero and leads to nan entries in R2k_cum
                self.R2k_cum.iloc[:, a] = 1 - col_SSX / prior_SS_col

                # R2 and cumulative R2 value for the whole block
                self.R2cum[a] = 1 - sum(row_SSX) / base_variance
                if a > 0:
                    self.R2[a] = self.R2cum[a] - self.R2cum[a - 1]
                else:
                    self.R2[a] = self.R2cum[a]
        # end: has no missing data

        for a in range(self.A):
            self.Hotellings_T2.iloc[:, a] = (
                self.Hotellings_T2.iloc[:, max(0, a - 1)]
                + (self.t_scores.iloc[:, a] / self.scaling_factor_for_scores[a]) ** 2
            )

        self.ellipse_coordinates = partial(
            ellipse_coordinates,
            n_components=self.n_components,
            scaling_factor_for_scores=self.scaling_factor_for_scores,
            n_rows=self.N,
        )

        self.T2_limit = partial(T2_limit, n_components=self.n_components, n_rows=self.N)

        return self

    def fit_transform(self, X, y=None):
        self.fit(X)
        assert False, "Still do the transform part"

    def SPE_limit(self, conf_level=0.95) -> float:
        check_is_fitted(self, "squared_prediction_error")

        assert conf_level > 0.0
        assert conf_level < 1.0

        # The limit is for the squares (i.e. the sum of the squared errors)
        # I.e. self.squared_prediction_error was square-rooted outside this function, so undo that.
        values = self.squared_prediction_error.iloc[:, self.A - 1] ** 2

        center_spe = values.mean()
        variance_spe = values.var(ddof=1)

        g = variance_spe / (2 * center_spe)
        h = (2 * (center_spe ** 2)) / variance_spe
        # Then take the square root again, to return the limit for SPE
        return np.sqrt(chi2.ppf(conf_level, h) * g)


class PCA_missing_values(BaseEstimator, TransformerMixin):
    """
    Create our PCA class if there is even a single missing data value in the X input array.

    The default method to impute missing values is the PMP algorithm (`md_method="pmp"`).

    Missing data method options are:

    * 'pmp'         Projection to Model Plane
    * 'scp'         Single Component Projection
    * 'nipals'      Same as 'scp': non-linear iterative partial least squares.
    * 'tsr': TODO
    * Other options? See papers by Abel Folch-Fortuny and also:
    https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/abs/10.1002/cem.750

    """

    valid_md_methods = ["pmp", "scp", "nipals", "tsr"]

    def __init__(
        self,
        n_components=None,
        copy: bool = True,
        random_state=None,
        missing_data_settings=dict,
    ):
        self.n_components = n_components
        self.random_state = None
        self.missing_data_settings = missing_data_settings
        self.has_missing_data = True

        # TODO: various settings assertions here
        assert self.missing_data_settings["md_tol"] < 10, "Tolerance should not be too large"
        assert (
            self.missing_data_settings["md_tol"] > epsqrt ** 1.95
        ), "Tolerance must exceed machine precision"

        assert self.missing_data_settings["md_method"] in self.valid_md_methods, (
            f"Missing data method is not recognized. ",
            "Must be one of {valid_md_methods}.",
        )

        assert True

    def fit(self, X, y=None):

        # Force input to NumPy array:
        self.data = np.asarray(X)

        # Other setups:
        self.n_components = self.A
        self.n_samples_ = self.N
        self.n_features_ = self.K

        self.components_ = np.zeros((self.K, self.A))
        self.t_scores = np.zeros((self.N, self.A))
        self.R2 = np.zeros(shape=(self.A,))
        self.R2cum = np.zeros(shape=(self.A,))
        self.R2k_cum = np.zeros(shape=(self.K, self.A))
        self.squared_prediction_error = np.zeros((self.N, self.A))

        # Perform MD algorithm here
        if self.missing_data_settings["md_method"].lower() == "pmp":
            self._fit_pmp(X)
        elif self.missing_data_settings["md_method"].lower() in ["scp", "nipals"]:
            self._fit_nipals_pca(settings=self.missing_data_settings)
        elif self.missing_data_settings["md_method"].lower() in ["tsr"]:
            self._fit_tsr(settings=self.missing_data_settings)

        # Additional calculations, which can be done after the missing data method is complete.
        self.explained_variance_ = np.diag(self.t_scores.T @ self.t_scores) / (self.N - 1)
        return self

    def transform(self, X):
        check_is_fitted(self, "blah")

        X = X.copy()
        return X

    def inverse_transform(self, X):
        check_is_fitted(self, "blah")

        X = X.copy()
        return X

    def _fit_nipals_pca(self, settings):
        """
        Internal method to fit the PCA model using the NIPALS algorithm.
        """
        # NIPALS algorithm
        N, K, A = self.N, self.K, self.A

        # 2. Initialize, or build on the existing results:
        self.timing = np.zeros((A, 1))

        # 4. Create direct links to the data
        Xd = np.asarray(self.data)
        base_variance = ssq(Xd)

        # Initialize storage:
        self.extra_info = {}
        self.extra_info["timing"] = np.zeros((1, A)) * np.nan
        self.extra_info["iterations"] = np.zeros((1, A)) * np.nan

        for a in np.arange(A):

            # 0. Timers and housekeeping
            start_time = time.time()
            itern = 0

            # 1. Find a column with the largest variance as t1_start; replace missing with zeros
            col_max_variance = Xd.var(axis=0).argmax()
            score_start = Xd[:, col_max_variance].reshape(N, 1)
            score_start[np.isnan(score_start)] = 0
            start_SS_col = ssq(Xd, axis=0)

            if sum(start_SS_col) < settings["md_tol"]:
                emsg = (
                    "There is no variance left in the data array: cannot "
                    f"compute any more components beyond component {a}."
                )
                raise RuntimeError(emsg)

            # Initialize t_a with random numbers, or carefully select a column
            # from X. <-- Don't do this anymore.
            # Rather: Pick a column from X as the initial guess instead.
            t_a = score_start + 1.0
            p_a = np.zeros((K, 1))
            while not (terminate_check(score_start, t_a, iterations=itern, settings=settings)):

                # 0: Richardson's acceleration, or any numerical acceleration
                #    method for PCA where there is slow convergence?

                # 0: starting point for convergence checking on next loop
                score_start = t_a.copy()

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

            self.extra_info["timing"][0, a] = time.time() - start_time
            self.extra_info["iterations"][0, a] = itern

            # Loop terminated!  Now deflate the X-matrix
            Xd -= np.dot(t_a, p_a.T)
            # These are the Residual Sums of Squares (RSS); i.e X-X_hat
            row_SSX = ssq(Xd, axis=1)
            col_SSX = ssq(Xd, axis=0)

            self.squared_prediction_error[:, a] = np.sqrt(row_SSX)

            # TODO: some entries in start_SS_col can be zero and leads to nan entries in R2k_cum
            self.R2k_cum[:, a] = 1 - col_SSX / start_SS_col

            # R2 and cumulative R2 value for the whole block
            self.R2cum[a] = 1 - sum(row_SSX) / base_variance
            if a > 0:
                self.R2[a] = self.R2cum[a] - self.R2cum[a - 1]
            else:
                self.R2[a] = self.R2cum[a]

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
            self.components_[:, a] = p_a.flatten()
            self.t_scores[:, a] = t_a.flatten()

        # end looping on A components

    def _fit_tsr(self, settings):
        # TODO: these eventually move to the defaults/user specified
        settings["md_tol"] = epsqrt
        settings["md_max_iter"] = 500
        delta = 1e100

        Xd = np.asarray(self.data)
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
                # If there are missing values (mis) in the n-th row. Compared to obs-erved values.
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
        self.components_ = (V[0:A, :]).T  # transpose result to the right shape: K x A
        self.t_scores = (Xd - np.mean(Xd, axis=0)) @ self.components_
        self.data = Xd


class MCUVScaler(BaseEstimator, TransformerMixin):
    """
    Create our own mean centering and scaling to unit variance (MCUV) class
    The default scaler in sklearn does not handle small datasets accurately, with ddof.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.center_x_ = X.mean()
        # this is the key difference with "preprocessing.StandardScaler"
        self.scale_x_ = X.std(ddof=1)
        self.scale_x_[self.scale_x_ == 0] = 1.0  # columns with no variance are left as-is.
        return self

    def transform(self, X):
        check_is_fitted(self, "center_x_")
        check_is_fitted(self, "scale_x_")

        X = X.copy()
        return (X - self.center_x_) / self.scale_x_

    def inverse_transform(self, X):
        check_is_fitted(self, "center_x_")
        check_is_fitted(self, "scale_x_")

        X = X.copy()
        return X * self.scale_x_ + self.center_x_


class PLS:
    """
    Performs a project to latent structures (PLS) or Partial Least Square (PLS) on the data.

    References
    ----------

    Abdi, "Partial least squares regression and projection on latent structure
    regression (PLS Regression)", 2010, DOI: 10.1002/wics.51
    """

    def __init__(
        self,
        n_components=None,
        method="nipals",
        tol=np.sqrt(np.finfo(float).eps),
        max_iter=1000,
        conf=0.05,  # 95% confidence level
        md_method="pmp",
        random_state=None,
    ):
        """
        Currently only the NIPALS algorithm is supported. Missing data is NOT yet supported.
        """

        # Attributes which come later, during the .fit() function:
        self.X = None
        self.Y = None
        self.conf = conf
        self.N = self.K = None
        self.A = max(0, int(n_components)) if n_components is not None else None
        self.random_state = 13 if random_state is None else int(random_state)

        # Check the remaining inputs
        assert self.conf < 0.50, "Confidence level must be a small fraction, e.g. 0.05 for 95%"
        self.n_components = self.A
        self.tol = float(tol)
        if not 1e-16 < self.tol < 1:
            raise ValueError("Tolerance `tol`` must be between 1E-16 and 1.0")
        self.max_iter = int(max_iter)

        self.method = method.lower()
        if self.method not in ("svd", "nipals"):
            raise ValueError(f"Method '{method}' is not known.")

        self.md_method = md_method.lower()  # Missing data method
        if self.md_method not in ("pmp",):
            raise ValueError(f"Missing data method '{md_method}' is not known.")

        # Attributes and internal values initialized
        self.TSS = 0.0
        self.ESS = None
        self.scores = self.T = None
        self.scores_y = None
        self.loadings = self.P = None
        self.loadings_y = None
        self.weights_x = None
        self.Tsq = None  # Hotelling's T2
        self.Tsq_limit = None
        self.SPE = None
        self.SPE_limit = None
        self.SD_t = None
        self.coeff = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.R2Xcum = None
        self.R2Xk_cum = None
        self.R2Ycum = None
        self.R2Yk_cum = None
        self.timing = None
        self.iterations = None

    def fit(self, X, Y):
        """
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples (rows)
            and `n_features` is the number of features (columns).

        Y : array-like, shape (n_samples, n_targets)
            Training data, where `n_samples` is the number of samples (rows)
            and `n_targets` is the number of target outputs (columns).
        """
        # Check the data:
        self._index = self._columns = None
        if isinstance(X, pd.DataFrame):
            self._index = X.index
            self._columns = X.columns

        # Raise an error for sparse input.
        # This is more informative than the generic one raised by check_array.
        if issparse(X) or issparse(Y):
            raise TypeError("This PLS class does not support sparse input.")

        # Force to NumPy array:
        self.X = np.asarray(X)
        self.N, self.K = self.X.shape

        self.Y = np.asarray(Y)
        if len(self.Y.shape) == 1:
            self.Y = np.reshape(self.Y, (self.Y.shape[0], 1))

        self.Ny, self.Ky = self.Y.shape

        assert self.Ny == self.N, (
            f"The X and Y arrays must have the same number of rows: "
            f"X has {self.N} and Y has {self.Ny}."
        )

        # Check if number of components is supported. against maximum
        min_dim = min(self.N, self.K)
        self.A = min_dim if self.A is None else int(self.A)
        if self.A > min_dim:
            import warnings

            warn = (
                "The requested number of components is more than can be "
                "computed from data. The maximum number of components is "
                f"the minimum of either the number of rows ({self.N}) or "
                f"the number of columns ({self.K})."
            )
            warnings.warn(warn, SpecificationWarning)
            self.A = min_dim

        # TODO: If missing data: implement PMP here.
        # If no missing data
        self._fit_nipals()

        # TODO: Final calculations
        # self._compute_rsquare_and_ic()
        # if self._index is not None:
        # self._to_pandas()

    def _fit_nipals(self):
        """This wrapper around the Scikit-Learn PLS function."""
        plsmodel = PLSRegression(n_components=self.A, scale="True")
        plsmodel.fit(self.X, self.Y)
        t1_predict, y_scores = plsmodel.transform(self.X, self.Y)

        # Extract the model parameters

        self.TSS = np.sum(np.power(self.Y - np.mean(self.Y), 2))
        self.ESS = "TODO"

        self.x_mean_ = plsmodel.x_mean_
        self.x_std_ = plsmodel.x_std_
        self.y_mean_ = plsmodel.y_mean_
        self.y_std_ = plsmodel.y_std_

        self.scores = self.T = plsmodel.x_scores_
        self.scores_y = y_scores
        self.loadings = plsmodel.x_loadings_
        self.loadings_y = self.C = plsmodel.y_loadings_
        self.predictions_pp = self.scores @ self.loadings_y.T
        self.predictions = self.predictions_pp * plsmodel.y_std_ + plsmodel.y_mean_
        self.weights_x = plsmodel.x_weights_

        # Calculate Hotelling's T-squared
        self.SD_t = np.std(self.T, axis=0, ddof=1)
        self.Tsq = np.sum(self.T / self.SD_t, axis=1) ** 2
        # Calculate confidence level for T-squared from the ppf of the F distribution
        f_value = f.ppf(q=(1 - self.conf), dfn=self.A, dfd=self.N)
        self.Tsq_limit = f_value * self.A * (self.N - 1) / (self.N - self.A)

        # SPE, Q, DModX (different names for the same thing)
        # --------------------------------------------------
        # Predictions of X and Y spaces
        X_hat = self.scores @ self.loadings.T
        X_check = self.X.copy()
        X_check -= self.x_mean_
        X_check_mcuv = X_check / self.x_std_
        error_X = X_check_mcuv - X_hat

        # Calculate Q-residuals (sum over the rows of the error array)
        # Estimate the confidence level for the Q-residuals
        # See: https://nirpyresearch.com/outliers-detection-pls-regression-nir-spectroscopy-python/
        self.SPE = np.sum(error_X ** 2, axis=1)
        max_value = np.max(self.SPE) + 1
        while 1 - np.sum(self.SPE > max_value) / np.sum(self.SPE > 0) > (1 - self.conf):
            max_value -= 1

        self.SPE_limit = max_value

        self.coeff = "TODO"
        self.eigenvalues = "TODO"
        self.eigenvectors = "TODO"

        self.R2Xcum = "TODO"
        self.R2Xk_cum = "TODO"

        error_y_ssq = np.sum((self.predictions - self.Y) ** 2, axis=None)

        self.R2Ycum = 1 - error_y_ssq / self.TSS
        self.R2Yk_cum = "TODO"

        self.timing = "TODO"
        self.iterations = "TODO"

        self._sklean_model = plsmodel

    def predict(self, X):
        """
        Using the PLS model on new data coming in matrix X.
        """

        class State(object):
            """ Class object to hold the prediction results together."""

            pass

        state = State()
        state.N, state.K = X.shape

        assert self.K == state.K, "Prediction data must same number of columns as training data."
        X_mcuv = (X - self.x_mean_) / self.x_std_

        state.scores = np.zeros((state.N, self.A))
        for a in range(self.A):
            p = self.loadings[:, a].reshape(self.K, 1)
            w = self.weights_x[:, a].reshape(self.K, 1)
            temp = X_mcuv @ w
            X_mcuv -= temp @ p.T
            state.scores[:, a] = temp.ravel()

        # After using all self.A components, calculate SPE-residuals (sum over rows of the errors)
        state.SPE = np.power(X_mcuv, 2).sum(axis=1)
        state.Tsq = np.sum(np.power((state.scores / self.SD_t), 2), 1)
        y_hat = state.scores @ self.loadings_y.T

        # Un-preprocess and return the entire state object
        state.y_hat = y_hat * self.y_std_ + self.y_mean_
        return state


def ssq(X: np.ndarray, axis: Optional[int] = None) -> Any:
    """A function than calculates the sum of squares of a 2D matrix
    (not array! and not checked for either: code will simply fail),
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
        out = np.nansum(X ** 2)

    return out


def terminate_check(
    t_a_guess: np.ndarray, t_a: np.ndarray, iterations: int, settings: dict
) -> bool:
    """The PCA iterative algorithm is terminated when any one of these conditions is True:

    #. scores converge: the norm between two successive iterations
    #. maximum number of iterations is reached
    """
    score_tol = np.linalg.norm(t_a_guess - t_a, ord=None)
    converged = score_tol < settings["md_tol"]
    max_iter = iterations > settings["md_max_iter"]
    if np.any([max_iter, converged]):
        return True
    else:
        return False


def quick_regress(Y, x):
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

    elif K == Nx:  # Case B: b = (Yx)/(x'x): (NxK)(Kx1) = (Nx1)
        b = np.zeros((Ny, 1))
        for n in np.arange(Ny):
            b[n] = np.sum(x[:, 0] * np.nan_to_num(Y[n, :]))
            # TODO(KGD): check: this denom is usually(always?) equal to 1.0
            denom = ssq(~np.isnan(Y[n, :]) * x.T)
            if np.abs(denom) > epsqrt:
                b[n] /= denom
        return b


def center(X, func=np.mean, axis=0, extra_output=False):
    """
    Performs centering of data, using a function, `func` (default: np.mean).
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


def scale(X, func=np.std, axis=0, extra_output=False, **kwargs):
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

    `axis` [optional; default=0] {integer or None}
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
    # options["variance_tolerance"] = 1e-7
    # options["low_variance_replacement"] = np.NaN

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
