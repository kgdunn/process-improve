from typing import Optional, Any

import numpy as np
import pandas as pd
from scipy.stats import f, chi2
from scipy.sparse import issparse

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA as PCA_sklearn
from sklearn.cross_decomposition import PLSRegression
from sklearn.utils.validation import check_is_fitted

from .robust import Sn


class SpecificationWarning(UserWarning):
    """ Parent warning class. """

    pass


eps = np.sqrt(np.finfo(float).eps)


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
    ):
        super().__init__(
            n_components, copy, whiten, svd_solver, tol, iterated_power, random_state
        )

    def fit(self, X, y=None) -> PCA_sklearn:
        self = super().fit(X)

        # Reference points for convenience:
        self.A = self.n_components
        self.N = self.n_samples_
        self.K = self.n_features_
        # note: this one is transposed, to conform to standards
        self.loadings = self.components_.T

        component_names = [f"PC {a+1}" for a in range(self.A)]
        self.scaling_factor_for_scores = pd.Series(
            np.sqrt(self.explained_variance_),
            index=component_names,
            name="Standard deviation per score",
        )
        self.t_scores = pd.DataFrame(
            super().fit_transform(X), columns=component_names, index=X.index
        )
        self.Hotellings_T2 = pd.DataFrame(
            np.zeros(shape=(self.N, self.A)),
            columns=component_names,
            index=X.index,
            # name="Hotelling's T^2 statistic, per component",
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

        # error_X = X - self.t_scores @ self.loadings.T
        # self.squared_prediction_error = np.sum(error_X ** 2, axis=1)
        self.squared_prediction_error = pd.DataFrame(
            np.zeros((self.N, self.A)), columns=component_names, index=X.index.copy()
        )
        Xd = X.copy()
        prior_SS_col = ssq(Xd.values, axis=0)
        base_variance = np.sum(prior_SS_col)
        for a in range(self.A):
            self.Hotellings_T2.iloc[:, a] = (
                self.Hotellings_T2.iloc[:, max(0, a - 1)]
                + (self.t_scores.iloc[:, a] / self.scaling_factor_for_scores[a]) ** 2
            )

            Xd -= self.t_scores.iloc[:, [a]] @ self.loadings[:, [a]].T
            # These are the Residual Sums of Squares (RSS); i.e X-X_hat
            row_SSX = ssq(Xd.values, axis=1)
            col_SSX = ssq(Xd.values, axis=0)

            # TODO(KGD): check correction factor
            self.squared_prediction_error.iloc[:, a] = row_SSX / self.K

            # TODO: some entries in prior_SS_col can be zero and leads to nan entries in R2k_cum
            self.R2k_cum.iloc[:, a] = 1 - col_SSX / prior_SS_col

            # R2 and cumulative R2 value for the whole block
            self.R2cum[a] = 1 - sum(row_SSX) / base_variance
            if a > 0:
                self.R2[a] = self.R2cum[a] - self.R2cum[a - 1]
            else:
                self.R2[a] = self.R2cum[a]

        return self

    def T2_limit(self, conf_level=0.95) -> float:
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
        A, N = self.n_components, self.N
        return A * (N - 1) * (N + 1) / (N * (N - A)) * f.isf((1 - conf_level), A, N - A)

    def SPE_limit(self, conf_level=0.95, robust=True) -> float:
        check_is_fitted(self, "squared_prediction_error")

        assert conf_level > 0.0
        assert conf_level < 1.0

        values = self.squared_prediction_error.iloc[:, self.A - 1]
        if (self.N > 15) and robust:
            # The "15" is just a rough cut off, above which the robust estimators would
            # start to work well. Below which we can get doubtful results.
            center_spe = values.median()
            variance_spe = Sn(values) ** 2
        else:
            center_spe = values.mean()
            variance_spe = values.var()

        g = variance_spe / (2 * center_spe)
        h = (2 * center_spe ** 2) / variance_spe
        return chi2.ppf(conf_level, h) * g

    def ellipse_coordinates(
        self,
        score_horiz: int,
        score_vert: int,
        T2_limit_conf_level: float = 0.05,
        n_points: int = 100,
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
        assert score_horiz <= self.n_components
        assert score_vert <= self.n_components
        assert T2_limit_conf_level > 0
        assert T2_limit_conf_level < 1
        s_h = self.scaling_factor_for_scores[score_horiz - 1]
        s_v = self.scaling_factor_for_scores[score_vert - 1]
        T2_limit = self.T2_limit(T2_limit_conf_level)
        dt = 2 * np.pi / (n_points - 1)
        steps = np.linspace(0, n_points - 1, n_points)
        x = np.cos(steps * dt) * np.sqrt(T2_limit) * s_h
        y = np.sin(steps * dt) * np.sqrt(T2_limit) * s_v
        return x, y


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
        self.scale_x_[
            self.scale_x_ == 0
        ] = 1.0  # columns with no variance are left as-is.
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
        assert (
            self.conf < 0.50
        ), "Confidence level must be a small fraction, e.g. 0.05 for 95%"
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
        self.scores = self.factors = self.T = None
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

        self.scores = self.factors = self.T = plsmodel.x_scores_
        self.scores_y = y_scores
        self.loadings = self.P = plsmodel.x_loadings_
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

        assert (
            self.K == state.K
        ), "Prediction data must same number of columns as training data."
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
            data = X[:, k]
            for val in data.flat:
                if not np.isnan(val):
                    out_ax0[k] += val ** 2

        return out_ax0

    if axis == 1:
        out_ax1 = np.zeros(N)
        for n in np.arange(N):
            data = X[n, :]
            for val in data.flat:
                if not np.isnan(val):
                    out_ax1[n] += val ** 2

        return out_ax1

    out = 0.0
    if axis is None:
        for val in X.flat:
            if not np.isnan(val):
                out += val ** 2

    return out


def terminate_check(
    t_a_guess: np.ndarray,
    t_a: np.ndarray,
    model: PCA,
    iterations: int,
) -> bool:
    """The PCA iterative algorithm is terminated when any one of these
    conditions is True
    #. scores converge: the norm between two successive iterations
    #. a max number of iterations is reached
    """
    score_tol = np.linalg.norm(t_a_guess - t_a, ord=None)
    # print(score_tol)
    converged = score_tol < model.tol
    max_iter = iterations > model.max_iter
    if np.any([max_iter, converged]):
        return True  # algorithm has converged
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
            if np.abs(denom) > eps:
                b[k] /= denom
        return b

    elif K == Nx:  # Case B: b = (Yx)/(x'x): (NxK)(Kx1) = (Nx1)
        b = np.zeros((Ny, 1))
        for n in np.arange(Ny):
            b[n] = np.sum(x[:, 0] * np.nan_to_num(Y[n, :]))
            # TODO(KGD): check: this denom is usually(always?) equal to 1.0
            denom = ssq(~np.isnan(Y[n, :]) * x.T)
            if np.abs(denom) > eps:
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
