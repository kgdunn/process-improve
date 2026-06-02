from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils import Bunch

from ..univariate.metrics import t_value

__eps = np.finfo(np.float32).eps


def fit_robust_lm(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Fits a robust linear model between Numpy vectors `x` and `y`, with an
    intercept. Returns a length-2 array ``[intercept, slope]`` (the
    ``params`` attribute returned by ``statsmodels.RLM``); no extra checking
    on data consistency is done.

    See also: regression.repeated_median_slope
    """
    rlm_model = sm.RLM(y, np.vstack([np.ones(x.size), x.ravel()]).T, M=sm.robust.norms.HuberT())
    rlm_results = rlm_model.fit()
    return rlm_results.params


def repeated_median_slope(x: np.ndarray, y: np.ndarray, nowarn: bool = False) -> float:
    """
    Robust slope calculation via Siegel's repeated-median estimator.

    https://en.wikipedia.org/wiki/Repeated_median_regression

    An elegant (simple) method to compute the robust slope between a vector ``x`` and ``y``.
    For each point ``i`` the median of the pairwise slopes ``(y[j] - y[i]) / (x[j] - x[i])``
    over all ``j != i`` is computed; the returned slope is the median of those per-point medians.

    Parameters
    ----------
    x : np.ndarray or sequence
        Independent variable. Coerced to a 1-D numpy array. Must have at least 3 elements
        (unless ``nowarn=True``).
    y : np.ndarray or sequence
        Dependent variable. Must have the same length as ``x`` (unless ``nowarn=True``).
    nowarn : bool, optional
        If ``True``, skip the length and equal-length input assertions. Default ``False``.

    Returns
    -------
    float
        The repeated-median estimate of the slope. Returns ``np.nan`` if all inner medians
        are undefined (e.g. all ``x`` values are equal).

    Notes
    -----
    INVESTIGATE: algorithm speed-ups via these articles:
    https://link.springer.com/article/10.1007/PL00009190
    http://www.sciencedirect.com/science/article/pii/S0020019003003508
    """
    # Slope
    medians = []
    x = x.copy().ravel() if isinstance(x, np.ndarray) else pd.Series(x).values.ravel()
    y = y.copy().ravel() if isinstance(y, np.ndarray) else pd.Series(y).values.ravel()

    if not (nowarn):
        if len(x) <= 2:
            raise ValueError("More than two samples are required for this function.")
        if len(x) != len(y):
            raise ValueError("Vectors x and y must have the same length.")
        # SEC-19 (#268): O(N^2) kernel; cap N so a 100k-point payload
        # cannot lock up CPU for many minutes.
        from process_improve.config import settings  # noqa: PLC0415

        if len(x) > settings.max_regression_points:
            raise ValueError(
                f"repeated_median_slope: len(x)={len(x)} exceeds the SEC-19 "
                f"cap of {settings.max_regression_points}. This is an O(N^2) "
                "algorithm; increase settings.max_regression_points if intentional."
            )

    for i in np.arange(len(x)):
        inner_medians = []
        for j in np.arange(len(y)):
            den = x[j] - x[i]
            if j != i and den != 0:
                inner_medians.append((y[j] - y[i]) / den)

        medians.append(np.nanmedian(inner_medians))

    return np.nanmedian(medians)


def robust_regression(  # noqa: PLR0913, PLR0915
    x: np.ndarray | pd.DataFrame | pd.Series,
    y: np.ndarray | pd.DataFrame | pd.Series,
    fit_intercept: bool = True,
    na_rm: bool = True,
    conflevel: float = 0.95,
    nowarn: bool = False,
    pi_resolution: int = 50,
) -> dict:
    """
    Perform the Simple robust regression analysis between `x` and `y` variables.

    Parameters
    - x, y: Sequences of numerical values.
    - fit_intercept: If True, fits an intercept term. If False, forces regression through origin.
    - na_rm: If True, removes all observations with one or more missing values.
    - conflevel: Confidence level for confidence intervals, default is 0.95.
    - nowarn: If True, suppresses warnings. Users should ensure data validity beforehand.
    - pi_resolution: The resolution of prediction intervals, default is 50.

    Simple robust regression between an `x` and a `y` using the `repeated_median_slope` method
    to calculate the slope. The intercept is the median intercept, when using that slope and the
    provided `x` and `y` values, or forced to zero if fit_intercept=False.

    Returns a dictionary of outputs with these keys::

        N:                        the number of observations used to fit the model
        coefficients:             a length-1 list containing the regression slope
        intercept:                returned if fit_intercept==True, otherwise 0
        standard_errors:          a length-1 list containing the standard error of the slope
        standard_error_intercept: standard error for the intercept (np.nan if fit_intercept=False)
        R2:                       the R^2 value
        SE:                       the model's standard error
        x_ssq:                    the sum of squares of (x - mean(x))
        k:                        the number of model parameters (2 if fit_intercept else 1)
        fitted_values:            the N predicted values, one per row in y
        residuals:                the N residuals
        t_value:                  the t-values for the standard errors
        conf_intervals:           K rows x 2 columns (lower, upper) confidence intervals
        conf_interval_intercept:  (lower, upper) confidence interval for the intercept
        pi_range:                 prediction intervals above and below, over the range of data
        leverage:                 the hat-matrix diagonal (leverage) for each observation
        influence:                Cook-style influence values for each observation
    """

    out: dict[str, Any] = {
        "N": None,
        "coefficients": [
            np.nan,
        ],
        "intercept": np.nan,
        "standard_errors": [
            np.nan,
        ],
        "standard_error_intercept": np.nan,
        "R2": np.nan,
        "SE": np.nan,
        "fitted_values": np.nan,
        "residuals": np.nan,
        "t_value": np.nan,
        "conf_intervals": np.array([[np.nan, np.nan]]),
        "conf_interval_intercept": np.array([np.nan, np.nan]),
        "pi_range": np.nan,
        "leverage": np.nan,
        "k": 1,
        "influence": np.nan,
    }

    #  Data pre-processing: handle both Pandas and NumPy -> use Pandas internally for X and y
    x_ = pd.DataFrame(x, copy=True) if isinstance(x, np.ndarray) else pd.DataFrame(x.values, copy=True)

    y_ = pd.DataFrame(y.ravel(), copy=True) if isinstance(y, np.ndarray) else pd.DataFrame(y.values, copy=True)

    # Removing missing values:
    missing_idx = y_.isna().any(axis=1)
    if na_rm:
        missing_idx = y_.isna().any(axis=1) | x_.isna().any(axis=1)
        x_ = x_.loc[~missing_idx, :]
        y_ = y_.loc[~missing_idx]

    # CASE when there is no data, or only 2 data point (repeate median slope needs more than 2)
    if (y_.size <= 2) or (x_.size <= 2):
        return out

    x = x_.values.ravel()
    y = y_.values.ravel()

    # initialize statistical variables
    k_params = 2 if fit_intercept else 1
    dof_resid = len(x) - k_params

    # Calculate robust regression
    slope = repeated_median_slope(x, y, nowarn=nowarn)
    intercept = np.nanmedian(y - slope * x) if fit_intercept else 0.0
    mean_x, mean_y = np.mean(x), np.mean(y)

    out["N"] = len(x)
    out["intercept"] = intercept
    out["coefficients"] = [
        slope,
    ]
    out["fitted_values"] = intercept + slope * x
    out["residuals"] = y - out["fitted_values"]

    # For robust method, calculate this way, since no guarantee both RegSS or RSS are < TSS.
    # So ensure this way that the TSS = RegSS + RSS, and R2 is the ratio of RegSS/TSS
    # https://learnche.org/pid/least-squares-modelling/least-squares-model-analysis
    regression_ssq = np.sum(np.power(out["fitted_values"] - mean_y, 2))
    residual_ssq = np.sum(out["residuals"] * out["residuals"])
    total_ssq = regression_ssq + residual_ssq
    out["R2"] = regression_ssq / total_ssq
    out["SE"] = np.sqrt(residual_ssq / dof_resid)
    out["x_ssq"] = np.sum(np.power(x - mean_x, 2))
    out["k"] = k_params

    # t-critical value for confidence intervals
    c_t = t_value(1 - (1 - conflevel) / 2, dof_resid)

    # Prediction intervals
    pi_range = np.linspace(np.min(x), np.max(x), pi_resolution)
    pi_y_pred = out["intercept"] + out["coefficients"][0] * pi_range

    # Handle degenerate case where x has no variation
    if out["x_ssq"] < __eps:
        out["standard_error_intercept"] = SE_b0 = np.nan
        out["conf_interval_intercept"] = np.array([np.nan, np.nan])
        out["standard_errors"] = [
            np.nan,
        ]
        out["pi_range"] = np.vstack([pi_range, pi_y_pred, pi_y_pred]).T
        out["leverage"] = 1 / out["N"] + np.power(x - mean_x, 2) / out["x_ssq"]

    else:
        out["standard_errors"] = [
            out["SE"] * 1 / np.sqrt(out["x_ssq"]),
        ]
        if fit_intercept:
            out["standard_error_intercept"] = SE_b0 = out["SE"] * np.sqrt(1 / out["N"] + (mean_x) ** 2 / out["x_ssq"])
            out["conf_interval_intercept"] = np.array(
                [out["intercept"] - c_t * SE_b0, out["intercept"] + c_t * SE_b0],
            )
            var_y = (out["SE"] ** 2) * (1 + 1 / out["N"] + (pi_range - np.mean(x)) ** 2 / out["x_ssq"])
            out["leverage"] = 1 / out["N"] + np.power(x - mean_x, 2) / out["x_ssq"]
        else:  # 1 / out["N"] is the term for the uncertainty for the intercept
            out["standard_error_intercept"] = np.nan
            out["conf_interval_intercept"] = np.array([np.nan, np.nan])
            # The model without intercept centers around the origin
            x_ssq_origin = np.sum(x**2)
            var_y = (out["SE"] ** 2) * (1 + (pi_range**2) / x_ssq_origin)
            out["leverage"] = np.power(x, 2) / x_ssq_origin

        std_y = np.sqrt(var_y)
        lower = pi_y_pred - c_t * std_y
        upper = pi_y_pred + c_t * std_y
        out["pi_range"] = np.vstack([pi_range, lower, upper]).T

    out["conf_intervals"] = np.array(
        [
            [
                out["coefficients"][0] - c_t * out["standard_errors"][0],
                out["coefficients"][0] + c_t * out["standard_errors"][0],
            ],
        ]
    )

    if out["SE"] < __eps:
        out["influence"] = out["residuals"] * 0.0
    else:
        out["influence"] = (
            np.power(out["residuals"] / ((1 - out["leverage"]) * out["SE"]), 2) * out["leverage"] / out["k"]
        )

    return out


_RENAMED = {"simple_robust_regression": "robust_regression"}

def __getattr__(name: str) -> None:
    """Raise a helpful error when a renamed module attribute is accessed."""
    if name in _RENAMED:
        new = _RENAMED[name]
        raise AttributeError(
            f"{name!r} has been renamed to {new!r}. "
            f"Use: from process_improve.regression.methods import {new}"
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def multiple_linear_regression(  # noqa: PLR0913
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.DataFrame | pd.Series,
    fit_intercept: bool = True,
    na_rm: bool = True,
    conflevel: float = 0.95,
    pi_resolution: int = 50,
) -> dict:
    """
    Linear regression of the N rows and K columns of matrix `X` onto the single column 'y'.

    Backwards-compatible wrapper around :class:`OLS`. New code should use the
    :class:`OLS` estimator directly, which exposes the same statistics as
    sklearn-style attributes and prints an R-like ``summary(lm(...))``.

    Notes and limitations:
        * does not handle weighting
        * N >= K at least as many rows as columns in X

    Returns a dictionary of outputs. Keys always present::

        N:                        number of observations actually used to fit
        coefficients:             a vector of K coefficients, one for each column in X
        intercept:                returned if fit_intercept==True
        standard_errors:          a vector of K standard errors, one per column in X
        standard_error_intercept: standard error for the intercept
        R2:                       the R^2 value
        SE:                       the model's standard error
        fitted_values:            the N predicted values, one per row in y
        residuals:                the N residuals
        t_value:                  the t-values for the standard errors
        conf_intervals:           K rows x 2 columns (lower, upper) confidence intervals

    Keys present only for single-feature ``X`` (and only when ``fit_intercept``
    is True and there is enough non-degenerate data)::

        x_ssq:                    sum of squares of the centred predictor
        leverage:                 hat-matrix diagonal
        influence:                Cook-style influence values
        pi_range:                 prediction interval above and below, over the
                                  range of the predictor (``pi_resolution`` points)
    """
    model = OLS(
        fit_intercept=fit_intercept,
        na_rm=na_rm,
        conflevel=conflevel,
        pi_resolution=pi_resolution,
    )
    return model.fit(X, y).to_dict()


class OLS(RegressorMixin, BaseEstimator):
    """Ordinary Least Squares regression with statistical diagnostics.

    A scikit-learn-compatible estimator that fits an OLS model and exposes
    inferential statistics (standard errors, t-values, p-values, confidence
    intervals, F-statistic) and influence diagnostics (leverage, Cook's
    distance). Calling ``print(model)`` after fitting renders a summary
    similar to R's ``summary(lm(...))``.

    Parameters
    ----------
    fit_intercept : bool, default=True
        If True, fits an intercept term. If False, the regression is forced
        through the origin.
    na_rm : bool, default=True
        If True, drops rows with one or more missing values before fitting.
    conflevel : float, default=0.95
        Confidence level for confidence and prediction intervals.
    pi_resolution : int, default=50
        Number of grid points at which to compute prediction intervals over
        the range of x. Only used when X has a single column and an intercept
        is fitted.

    Attributes
    ----------
    coefficients_ : np.ndarray of shape (K,)
        Fitted slope coefficients (excludes the intercept).
    intercept_ : float
        Fitted intercept (np.nan if ``fit_intercept`` is False).
    standard_errors_ : np.ndarray of shape (K,)
        Standard errors of ``coefficients_``.
    standard_error_intercept_ : float
        Standard error of the intercept.
    t_values_ : np.ndarray of shape (K,)
        t-statistics for each coefficient.
    t_value_intercept_ : float
        t-statistic for the intercept.
    p_values_ : np.ndarray of shape (K,)
        Two-sided p-values for each coefficient.
    p_value_intercept_ : float
        p-value for the intercept.
    conf_intervals_ : np.ndarray of shape (K, 2)
        Lower and upper bounds of the coefficient confidence intervals.
    conf_interval_intercept_ : np.ndarray of shape (2,)
        Lower and upper bounds of the intercept confidence interval.
    r2_ : float
        Coefficient of determination.
    adj_r2_ : float
        Adjusted R-squared.
    se_ : float
        Residual standard error (sqrt of residual variance).
    df_resid_ : int
        Residual degrees of freedom.
    df_model_ : int
        Model degrees of freedom (number of slope coefficients).
    f_statistic_ : float
        F-statistic for the overall regression.
    f_pvalue_ : float
        p-value associated with the F-statistic.
    fitted_values_ : np.ndarray of shape (N,)
        In-sample predictions.
    residuals_ : np.ndarray of shape (N_original,)
        In-sample residuals (NaN at rows removed by ``na_rm``).
    leverage_ : np.ndarray of shape (N,)
        Hat-matrix diagonal (only computed for single-feature X).
    influence_ : np.ndarray of shape (N,)
        Cook's distance (only computed for single-feature X with intercept).
    pi_range_ : np.ndarray of shape (pi_resolution, 3) or float
        Columns are x-grid, lower bound, upper bound of the prediction
        interval. ``np.nan`` if not applicable.
    feature_names_in_ : list[str]
        Column names of the feature matrix.
    target_name_ : str
        Name of the target variable.
    n_samples_ : int
        Number of samples used in the fit (after ``na_rm``).
    n_features_in_ : int
        Number of input features.
    is_fitted_ : bool
        Whether ``fit()`` has been called successfully.

    Examples
    --------
    >>> import numpy as np
    >>> from process_improve.regression.methods import OLS
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((50, 2))
    >>> y = X @ [1.5, -2.0] + 0.5 + 0.1 * rng.standard_normal(50)
    >>> model = OLS().fit(X, y)
    >>> print(model)  # doctest: +SKIP
    Call:
    OLS(fit_intercept=True, na_rm=True, conflevel=0.95)
    ...

    See Also
    --------
    multiple_linear_regression : Backwards-compatible function returning a dict.
    robust_regression : Robust regression via repeated-median slope.
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        na_rm: bool = True,
        conflevel: float = 0.95,
        pi_resolution: int = 50,
    ) -> None:
        self.fit_intercept = fit_intercept
        self.na_rm = na_rm
        self.conflevel = conflevel
        self.pi_resolution = pi_resolution

    def fit(  # noqa: C901, PLR0912, PLR0915
        self,
        X: np.ndarray | pd.DataFrame | pd.Series,
        y: np.ndarray | pd.DataFrame | pd.Series,
    ) -> OLS:
        """Fit the OLS model.

        Parameters
        ----------
        X : array-like of shape (N, K)
            Feature matrix. Pandas and NumPy inputs are both accepted.
        y : array-like of shape (N,) or (N, 1)
            Target vector.

        Returns
        -------
        self : OLS
            Fitted estimator.
        """
        # Preserve original input metadata (length, names) before we drop NA rows.
        if isinstance(X, pd.DataFrame):
            X_full = X.copy()
            # Default integer column names (the typical case for numpy → DataFrame)
            # are remapped to x1, x2, ... so the printed formula and table match R's lm().
            if list(X_full.columns) == list(range(X_full.shape[1])):
                X_full.columns = [f"x{i + 1}" for i in range(X_full.shape[1])]
        elif isinstance(X, pd.Series):
            X_full = X.to_frame(name=str(X.name) if X.name is not None else "x1")
        else:
            X_arr = np.asarray(X)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)
            X_full = pd.DataFrame(X_arr, columns=[f"x{i + 1}" for i in range(X_arr.shape[1])])
        if isinstance(y, pd.Series):
            y_full = pd.DataFrame(y.values.ravel())
            target_name = y.name if y.name is not None else "y"
        elif isinstance(y, pd.DataFrame):
            y_full = pd.DataFrame(y.values.ravel())
            target_name = y.columns[0] if len(y.columns) else "y"
        else:
            y_full = pd.DataFrame(np.asarray(y).ravel())
            target_name = "y"

        # X and y are matched positionally (row i of X with row i of y). Reset X
        # to a clean RangeIndex so that a non-default index on the input X (for
        # example a date index, or a sliced-out subset of a larger DataFrame)
        # cannot misalign with the freshly built, RangeIndex-ed ``y_full``.
        # Without this, statsmodels raises "indices for endog and exog are not
        # aligned" once a design matrix is handed to ``sm.OLS`` below.
        X_full = X_full.reset_index(drop=True)

        n_original = y_full.shape[0]
        self.feature_names_in_ = [str(c) for c in X_full.columns]
        self.target_name_ = str(target_name)
        self.n_features_in_ = X_full.shape[1]

        # Initialize all fitted attributes to a "not enough data" state.
        self._init_empty_attributes(n_original)

        # Missing-value handling.
        missing_idx = y_full.isna().any(axis=1)
        if self.na_rm:
            missing_idx = y_full.isna().any(axis=1) | X_full.isna().any(axis=1)
        X_ = X_full.loc[~missing_idx, :]
        y_ = y_full.loc[~missing_idx]

        # Degenerate input: too few rows.
        if y_.size <= 1 or X_.size <= 1:
            self.is_fitted_ = False
            return self

        n_samples, n_features = X_.shape
        k_params = n_features + (1 if self.fit_intercept else 0)
        if n_samples < k_params:
            raise ValueError(
                "N >= K: You need at least as many rows as there are columns to fit a linear regression."
            )
        if n_samples != y_.size:
            raise ValueError(
                f"X and y must have the same number of rows: got {n_samples} and {y_.size}."
            )

        # Build the design matrix.
        design = X_.copy()
        if self.fit_intercept:
            design.insert(0, "__constant__", 1.0)

        results = sm.OLS(y_, design).fit()
        alpha = 1.0 - self.conflevel
        conf = np.asarray(results._results.conf_int(alpha), dtype=float)

        # se^2 * (X'X)^-1 for the design (intercept first, if fitted): needed to
        # form prediction intervals at arbitrary new x via prediction_interval().
        self._cov_params_ = np.asarray(results.cov_params(), dtype=float)

        # statsmodels returns Series (indexed by column name) when fed a DataFrame,
        # so always go through ``.values`` for positional access.
        params = np.asarray(results.params.values if hasattr(results.params, "values")
                             else results.params, dtype=float)
        bse = np.asarray(results.bse.values if hasattr(results.bse, "values")
                         else results.bse, dtype=float)
        tvalues = np.asarray(results.tvalues.values if hasattr(results.tvalues, "values")
                             else results.tvalues, dtype=float)
        pvalues = np.asarray(results.pvalues.values if hasattr(results.pvalues, "values")
                             else results.pvalues, dtype=float)

        # Populate attributes.
        self.n_samples_ = n_samples
        self.df_resid_ = int(results.df_resid)
        self.df_model_ = int(results.df_model)

        if self.fit_intercept:
            self.intercept_ = float(params[0])
            self.standard_error_intercept_ = float(bse[0])
            self.t_value_intercept_ = float(tvalues[0])
            self.p_value_intercept_ = float(pvalues[0])
            self.conf_interval_intercept_ = conf[0, :]
            self.coefficients_ = params[1:]
            self.standard_errors_ = bse[1:]
            self.t_values_ = tvalues[1:]
            self.p_values_ = pvalues[1:]
            self.conf_intervals_ = conf[1:, :]
        else:
            self.intercept_ = float("nan")
            self.standard_error_intercept_ = float("nan")
            self.t_value_intercept_ = float("nan")
            self.p_value_intercept_ = float("nan")
            self.conf_interval_intercept_ = np.array([np.nan, np.nan])
            self.coefficients_ = params
            self.standard_errors_ = bse
            self.t_values_ = tvalues
            self.p_values_ = pvalues
            self.conf_intervals_ = conf

        self.fitted_values_ = np.asarray(results.fittedvalues, dtype=float)
        self.r2_ = float(results.rsquared)
        self.adj_r2_ = float(results.rsquared_adj)
        self.se_ = float(np.sqrt(results.scale))
        self.f_statistic_ = float(results.fvalue) if results.fvalue is not None else float("nan")
        self.f_pvalue_ = float(results.f_pvalue) if results.f_pvalue is not None else float("nan")

        # Residuals are written into a vector of the original shape (NaN where rows were dropped).
        self.residuals_ = np.full(n_original, np.nan)
        self.residuals_[~missing_idx.to_numpy()] = results.resid

        # R^2 reported two ways for backwards compatibility with the dict API.
        mean_y = float(np.mean(y_.values))
        total_ssq = float(np.sum(np.power(y_.values - mean_y, 2)))
        regression_ssq = float(np.sum(np.power(self.fitted_values_ - mean_y, 2)))
        residual_ssq = float(np.nansum(self.residuals_ * self.residuals_))
        self.r2_regression_based_ = (
            regression_ssq / total_ssq if total_ssq > 0 else float("nan")
        )
        self.r2_residual_based_ = (
            1.0 - residual_ssq / total_ssq if total_ssq > 0 else float("nan")
        )

        # Single-feature diagnostics: leverage, influence, and a prediction interval grid.
        self.x_ssq_ = float("nan")
        self.leverage_ = np.array([np.nan])
        self.influence_ = np.array([np.nan])
        self.pi_range_ = float("nan")

        if n_features == 1:
            x_col = X_.iloc[:, 0].to_numpy()
            mean_x = float(np.mean(x_col))
            x_ssq = float(np.sum(np.power(x_col - mean_x, 2)))
            self.x_ssq_ = x_ssq

            if self.fit_intercept and x_ssq > 0:
                self.leverage_ = 1.0 / n_samples + np.power(x_col - mean_x, 2) / x_ssq

                eps = np.finfo(np.float32).eps
                if self.se_ < eps:
                    self.influence_ = np.zeros(n_samples)
                else:
                    self.influence_ = (
                        np.power(results.resid / ((1 - self.leverage_) * self.se_), 2)
                        * self.leverage_
                        / k_params
                    )
                    pi_range = np.linspace(np.min(x_col), np.max(x_col), self.pi_resolution)
                    pi_y_pred = self.intercept_ + self.coefficients_[0] * pi_range
                    var_y = (self.se_**2) * (
                        1 + 1.0 / n_samples + (pi_range - mean_x) ** 2 / x_ssq
                    )
                    std_y = np.sqrt(var_y)
                    c_t = t_value(1 - (1 - self.conflevel) / 2, n_samples - 2)
                    self.pi_range_ = np.vstack(
                        [pi_range, pi_y_pred - c_t * std_y, pi_y_pred + c_t * std_y]
                    ).T

        self._k_ = k_params
        self.is_fitted_ = True
        return self

    def predict(
        self,
        X: np.ndarray | pd.DataFrame | pd.Series,
    ) -> np.ndarray:
        """Predict target values for ``X``.

        Parameters
        ----------
        X : array-like of shape (N, K)

        Returns
        -------
        y_pred : np.ndarray of shape (N,)
        """
        if not getattr(self, "is_fitted_", False):
            raise NotFittedError("OLS must be fitted before calling predict().")
        if isinstance(X, pd.DataFrame):
            X_arr = X.to_numpy(dtype=float)
        elif isinstance(X, pd.Series):
            X_arr = X.to_numpy(dtype=float).reshape(-1, 1)
        else:
            X_arr = np.asarray(X, dtype=float)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)
        n_features = X_arr.shape[1]
        if n_features != self.n_features_in_:
            raise ValueError(
                f"X has {n_features} feature(s), but OLS was fitted with "
                f"{self.n_features_in_} feature(s)."
            )
        intercept = 0.0 if not self.fit_intercept else self.intercept_
        return intercept + X_arr @ self.coefficients_

    def prediction_interval(
        self,
        X: np.ndarray | pd.DataFrame | pd.Series | float,
        conflevel: float | None = None,
    ) -> Bunch:
        """Prediction interval for new observations at arbitrary ``X``.

        Unlike the ``pi_range_`` attribute - which is evaluated on a fixed grid
        spanning the training data - this method evaluates the prediction
        interval at any predictor value(s) supplied by the caller, including
        points outside the training range.

        Parameters
        ----------
        X : array-like of shape (M, K), (K,) or scalar
            New predictor value(s). A scalar or 1-D array is interpreted as a
            list of points when the model has a single feature, or as a single
            multi-feature point otherwise.
        conflevel : float or None, default=None
            Confidence level for the interval. Defaults to the model's own
            ``conflevel``.

        Returns
        -------
        sklearn.utils.Bunch
            A bunch with three length-M arrays: ``predicted`` (the point
            prediction), and ``lower`` / ``upper`` (the prediction-interval
            bounds).
        """
        assert getattr(self, "is_fitted_", False), "OLS must be fitted before calling prediction_interval()."
        cl = self.conflevel if conflevel is None else conflevel

        X_arr = (
            X.to_numpy(dtype=float)
            if isinstance(X, pd.Series | pd.DataFrame)
            else np.asarray(X, dtype=float)
        )

        if X_arr.ndim == 0:
            X_arr = X_arr.reshape(1, 1)
        elif X_arr.ndim == 1:
            # Single-feature model: a 1-D input is a list of scalar points.
            # Otherwise it is interpreted as one multi-feature point.
            X_arr = X_arr.reshape(-1, 1) if self.n_features_in_ == 1 else X_arr.reshape(1, -1)

        if X_arr.shape[1] != self.n_features_in_:
            msg = f"Expected {self.n_features_in_} feature(s) per row, got {X_arr.shape[1]}."
            raise ValueError(msg)

        # Design rows must match the fitted design: intercept column first.
        if self.fit_intercept:
            design = np.column_stack([np.ones(X_arr.shape[0]), X_arr])
            full_params = np.concatenate([[self.intercept_], self.coefficients_])
        else:
            design = X_arr
            full_params = self.coefficients_

        predicted = design @ full_params
        # Variance of the estimated mean response: row @ (se^2 (X'X)^-1) @ row.
        # The prediction interval adds se^2 for the new observation's own noise.
        var_mean = np.einsum("ij,jk,ik->i", design, self._cov_params_, design)
        std_pred = np.sqrt(self.se_**2 + var_mean)
        c_t = t_value(1 - (1 - cl) / 2, self.df_resid_)
        half_width = c_t * std_pred
        return Bunch(predicted=predicted, lower=predicted - half_width, upper=predicted + half_width)

    def summary(self) -> str:
        """Return an R-style ``summary(lm(...))`` string for the fitted model."""
        if not getattr(self, "is_fitted_", False):
            return "OLS model has not been fitted (or fit failed due to insufficient data)."
        return self._format_summary()

    def to_dict(self) -> dict:
        """Return the legacy dictionary representation used by ``multiple_linear_regression``."""
        out: dict[str, Any] = {
            "N": None,
            "coefficients": [np.nan],
            "intercept": np.nan,
            "standard_errors": [np.nan],
            "standard_error_intercept": np.nan,
            "R2": np.nan,
            "SE": np.nan,
            "fitted_values": np.nan,
            "residuals": np.nan,
            "t_value": np.nan,
            "conf_intervals": [np.nan, np.nan],
        }
        if not getattr(self, "is_fitted_", False):
            return out
        out["N"] = self.n_samples_
        out["intercept"] = self.intercept_
        out["coefficients"] = self.coefficients_
        out["standard_errors"] = self.standard_errors_
        out["standard_error_intercept"] = self.standard_error_intercept_
        out["t_value"] = self.t_values_
        out["conf_intervals"] = self.conf_intervals_
        out["conf_interval_intercept"] = self.conf_interval_intercept_
        out["R2"] = self.r2_
        out["SE"] = self.se_
        out["fitted_values"] = self.fitted_values_
        out["residuals"] = self.residuals_
        out["R2_regression_based"] = self.r2_regression_based_
        out["R2_residual_based"] = self.r2_residual_based_
        out["k"] = self._k_
        if not np.isnan(self.x_ssq_):
            out["x_ssq"] = self.x_ssq_
        if not (isinstance(self.leverage_, np.ndarray) and self.leverage_.size == 1
                and np.isnan(self.leverage_[0])):
            out["leverage"] = self.leverage_
        if not (isinstance(self.influence_, np.ndarray) and self.influence_.size == 1
                and np.isnan(self.influence_[0])):
            out["influence"] = self.influence_
        if isinstance(self.pi_range_, np.ndarray):
            out["pi_range"] = self.pi_range_
        return out

    def __repr__(self) -> str:
        """Return the R-style summary if fitted, otherwise the sklearn parameter repr."""
        if not getattr(self, "is_fitted_", False):
            return super().__repr__()
        return self._format_summary()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _init_empty_attributes(self, n_original: int) -> None:
        """Set all fitted attributes to NaN/empty so attribute access is safe even when fit fails."""
        self.is_fitted_ = False
        self.n_samples_ = 0
        self.df_resid_ = 0
        self.df_model_ = 0
        self.intercept_ = float("nan")
        self.standard_error_intercept_ = float("nan")
        self.t_value_intercept_ = float("nan")
        self.p_value_intercept_ = float("nan")
        self.conf_interval_intercept_ = np.array([np.nan, np.nan])
        self.coefficients_ = np.array([np.nan])
        self.standard_errors_ = np.array([np.nan])
        self.t_values_ = np.array([np.nan])
        self.p_values_ = np.array([np.nan])
        self.conf_intervals_ = np.array([[np.nan, np.nan]])
        self.fitted_values_ = np.full(n_original, np.nan)
        self.residuals_ = np.full(n_original, np.nan)
        self.r2_ = float("nan")
        self.adj_r2_ = float("nan")
        self.se_ = float("nan")
        self.f_statistic_ = float("nan")
        self.f_pvalue_ = float("nan")
        self.r2_regression_based_ = float("nan")
        self.r2_residual_based_ = float("nan")
        self.x_ssq_ = float("nan")
        self.leverage_ = np.array([np.nan])
        self.influence_ = np.array([np.nan])
        self.pi_range_ = float("nan")
        self._cov_params_ = np.array([[np.nan]])
        self._k_ = 0

    @staticmethod
    def _signif_code(p: float) -> str:
        if np.isnan(p):
            return " "
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "** "
        if p < 0.05:
            return "*  "
        if p < 0.1:
            return ".  "
        return "   "

    @staticmethod
    def _fmt_number(value: float, width: int = 10) -> str:
        if np.isnan(value):
            return f"{'NaN':>{width}}"
        if value == 0:
            return f"{0.0:>{width}.5f}"
        magnitude = abs(value)
        if magnitude < 1e-3 or magnitude >= 1e6:
            return f"{value:>{width}.4e}"
        return f"{value:>{width}.5f}"

    @staticmethod
    def _fmt_pvalue(p: float, width: int = 10) -> str:
        if np.isnan(p):
            return f"{'NaN':>{width}}"
        if p < 1e-4:
            return f"{p:>{width}.3e}"
        return f"{p:>{width}.5f}"

    def _format_summary(self) -> str:
        # Call line: show the parameters that affected the fit.
        call = (
            f"OLS(fit_intercept={self.fit_intercept}, na_rm={self.na_rm}, "
            f"conflevel={self.conflevel})"
        )
        target = self.target_name_ or "y"
        feature_str = " + ".join(self.feature_names_in_) if self.feature_names_in_ else "1"
        formula_line = (
            f"  formula: {target} ~ {feature_str}"
            if self.fit_intercept
            else f"  formula: {target} ~ {feature_str} + 0"
        )

        # Residual quantiles.
        resid = self.residuals_[~np.isnan(self.residuals_)]
        q = (
            np.quantile(resid, [0.0, 0.25, 0.5, 0.75, 1.0])
            if resid.size
            else np.array([np.nan] * 5)
        )
        resid_header = f"{'Min':>10}{'1Q':>11}{'Median':>11}{'3Q':>11}{'Max':>11}"
        resid_values = (
            f"{self._fmt_number(q[0]):>10}"
            f"{self._fmt_number(q[1]):>11}"
            f"{self._fmt_number(q[2]):>11}"
            f"{self._fmt_number(q[3]):>11}"
            f"{self._fmt_number(q[4]):>11}"
        )

        # Coefficient table.
        rows: list[tuple[str, float, float, float, float]] = []
        if self.fit_intercept:
            rows.append((
                "(Intercept)",
                self.intercept_,
                self.standard_error_intercept_,
                self.t_value_intercept_,
                self.p_value_intercept_,
            ))
        for name, est, se, tv, pv in zip(
            self.feature_names_in_,
            self.coefficients_,
            self.standard_errors_,
            self.t_values_,
            self.p_values_,
            strict=False,
        ):
            rows.append((str(name), float(est), float(se), float(tv), float(pv)))

        name_width = max((len(r[0]) for r in rows), default=11)
        name_width = max(name_width, 11)

        coef_header = (
            f"{'':<{name_width}}  {'Estimate':>10}  {'Std. Error':>10}  "
            f"{'t value':>8}  {'Pr(>|t|)':>10}"
        )
        coef_lines = []
        for name, est, se, tv, pv in rows:
            line = (
                f"{name:<{name_width}}  "
                f"{self._fmt_number(est):>10}  "
                f"{self._fmt_number(se):>10}  "
                f"{self._fmt_number(tv, width=8):>8}  "
                f"{self._fmt_pvalue(pv):>10} {self._signif_code(pv)}"
            )
            coef_lines.append(line)

        # F-statistic line.
        if np.isnan(self.f_statistic_) or np.isnan(self.f_pvalue_):
            f_line = "F-statistic: not available"
        else:
            f_line = (
                f"F-statistic: {self.f_statistic_:.4g} on "
                f"{self.df_model_} and {self.df_resid_} DF, "
                f"p-value: {self.f_pvalue_:.4g}"
            )

        return (
            "Call:\n"
            f"  {call}\n"
            f"{formula_line}\n"
            "\n"
            "Residuals:\n"
            f"{resid_header}\n"
            f"{resid_values}\n"
            "\n"
            "Coefficients:\n"
            f"{coef_header}\n"
            + "\n".join(coef_lines)
            + "\n"
            "---\n"
            "Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1\n"
            "\n"
            f"Residual standard error: {self.se_:.4g} on {self.df_resid_} degrees of freedom\n"
            f"Multiple R-squared:  {self.r2_:.4g},\tAdjusted R-squared:  {self.adj_r2_:.4g}\n"
            f"{f_line}"
        )
