"""
Class for ControlChart: robust control charts with a balance between CUSUM and Shewhart properties.
"""

import numpy as np
import pandas as pd

from ..univariate.metrics import median_abs_deviation
from ..regression.methods import repeated_median_slope


def rho(x, k=2.52):
    """
    Bi-weight rho function.

    Fixed constant of k=2.52 is from p 289 of the paper
    https://onlinelibrary.wiley.com/doi/abs/10.1002/for.1125
    """
    return k if np.abs(x) > k else k * (1 - np.power(1 - np.power(x / k, 2), 3))


def psi(x, k=2.0):
    """
    Pre-cleaning based on the Huber y-function can be interpreted as replacing unexpected
    high or low values by a more likely value.
    From p 288 of the paper
    https://onlinelibrary.wiley.com/doi/abs/10.1002/for.1125
    """
    return x if abs(x) < k else k * np.sign(x)


class ControlChart(object):
    """
    Creates control chart instance objects.
    """

    def __init__(self, style="robust", variant="HW"):
        """
        Create/initialize a control chart.


        Args: style (str, optional): Which style control chart to calculate. Defaults to "robust".
            Other choice is 'regular' (i.e. not-robust) calculations. User should then ensure that
            no outliers are present in the data.

            variant (str, optional): Many variants of control charts are available.

                The default is a Holt-Winters (HW) chart, with automatic determination of control
                chart parameters. This chart is a blend of infinite history (CUSUM) charts, and an
                instantaneous (no history taken into account) Shewhart chart. The exact blend is
                specified by parameters `ld_1` (lambda 1) and `ld_2` (lambda 2).

                Other variants are:

                'xbar.no.subgroup' [Shewhart chart, with no subgroups]. In other words, each
                observation is independently plotted on the control chart.

                'CUSUM' (CUmulative SUM) chart, which uses all the history of the chart.
        """
        self.style = style.strip()
        self.variant = variant.strip().lower()

        # Will be calculated by the self.fit_limits() function
        self.target = None
        self._given_target = None
        self._given_s = None
        self.s = None
        # index of elements which are found to be outside +/- 3S
        self.idx_outside_3S = []
        self.warm_up = {}

        columns = [
            "y",
            "psi_input",
            "rho_input",
            "y_star",
            "alpha_hat",
            "beta_hat",
            "sigma_hat",
            "error",
        ]
        self.df = pd.DataFrame(columns=columns, dtype=np.float64)

    def calculate_limits(self, y, target=None, s=None, **kwargs):
        """
        Finds, for a given vector `y`, the control chart target and limits.

        Only for the Holt-Winters method, and only when there are more than
            min(20, max(10, np.ceil(0.10 * N))))

        measurements, where N is the length of the input vector. In other words, the provided
        target and standard deviation are only used if more than 10 to 20 measurements.
        If `target` and `s` are numeric, then that target value and that standard deviation value
        are used, otherwise these values are estimated.
        """
        self._given_target = target
        self._given_s = s

        if s is not None:
            self.s = float(s)
            assert (
                s > 0.0
            ), "The given standard deviation cannot be zero, and must be positive."
            assert s < 1e300, "The given standard deviation cannot be this large."

        if target is not None:
            self.target = float(target)

        self.df["y"] = y.ravel()
        self.N = self.df.shape[0]

        # Between M = 10 and 20 samples required to warm-up (calculate summary statistics)
        self.warm_up["M"] = int(min(20, max(10, np.ceil(0.10 * self.N))))

        if (self.warm_up["M"] > self.N) and self.variant.strip().lower() == "hw":
            # TO CHECK: Completely handle the case with very few samples. Is everything filled in?
            # Also check case when some of these samples are NAN, you might have even fewer still.
            self.target = self._target_calculated_best = self.df["y"].median()
            self.s = self._tau = self.df["y"].std()
            self.df["y_star"] = self.df["y"].values
            self.df["alpha_hat"] = self.target
            self.df["beta_hat"] = 0
            self.df["sigma_hat"] = np.nan
            self.df["error"] = np.nan
            return

        # Check if there are enough training samples:
        if self.N < 2 * self.warm_up["M"]:
            self.train_samples = list(np.arange(0, self.N))
        else:
            self.train_samples = list(np.arange(self.warm_up["M"], self.N))

        for key, val in kwargs.items():
            setattr(self, key, val)

        if self.variant.strip().lower() == "hw":
            if not hasattr(self, "ld_1"):
                setattr(self, "ld_1", None)

            if not hasattr(self, "ld_2"):
                setattr(self, "ld_2", None)

            self._holt_winters_parameter_fit()

        if self.variant.strip().lower() == "xbar.no.subgroup":
            self._xbar_no_subgroup_fit()

        # After whichever fit is completed, check which are outside +/- 3S:
        idx_bool = (self.df["y"] - self.target).abs() > 3.0 * self.s
        self.idx_outside_3S = np.nonzero(idx_bool.ravel())[0].tolist()

    def _xbar_no_subgroup_fit(self):
        """
        The control chart is fit from the data samples, assuming each sample is its own subgroup.
        Variant = 'regulur' | 'robust' switchs how the average and standard deviation are
        calculated.

        Control chart limits assume the data are normally distributed and independent. In
        particular, this last assumption can have consequences if not actually met. Limits may be
        too wide, or too narrow.

        """
        if self.style == "regular":
            self.target = self.df["y"].mean()
            self.s = self.df["y"].std()
        elif self.style == "robust":
            self.target = self.df["y"].median()
            self.s = (self.df["y"] - self.target).abs().median() * 1.4826

    def _holt_winters_parameter_fit(self):
        """
        Recommended in the paper: not to fit the lambda_s value, but to use a grid search for the
        lambda_1 and lambda_2 values. This is done in a 5x5 grid in the code below.
        """
        self.ld_s = ld_s = 0.2
        rho_func = np.vectorize(rho)

        if self.ld_1 and self.ld_2:
            # User has provided their own lambda_1 and lambda_2 values.
            assert self.ld_1 >= 0.0, "Lambda_1 must be greater than or equal to zero."
            assert self.ld_2 >= 0.0, "Lambda_2 must be greater than or equal to zero."
            assert self.ld_s >= 0.0, "Lambda_s must be greater than or equal to zero."
            assert self.ld_1 <= 1.0, "Lambda_1 must be less than or equal to 1.0."
            assert self.ld_2 <= 1.0, "Lambda_2 must be less than or equal to 1.0."
            assert self.ld_s <= 1.0, "Lambda_s must be less than or equal to 1.0."
            self._holt_winters_warmup_fit(
                ld_1=self.ld_1, ld_2=self.ld_2, ld_s=self.ld_s
            )

        else:
            # User wants to find an value for ld_1 and ld_2 that best fits the data

            ld_1_index = np.linspace(0.1, 0.9, num=5, endpoint=True)
            ld_2_index = np.linspace(0.1, 0.9, num=5, endpoint=True)
            residuals, _ = np.meshgrid(ld_1_index, ld_2_index)

            for i, ld_1 in enumerate(ld_1_index):
                for j, ld_2 in enumerate(ld_2_index):
                    self._holt_winters_warmup_fit(ld_1=ld_1, ld_2=ld_2, ld_s=ld_s)

                    # Apply equation 16 from the paper to the residuals in the 'training' period,
                    # that is the samples after the warm-up period.
                    future_errors = self.df["error"][self.train_samples]
                    S_T_median_error = 1.48 * np.median(abs(future_errors))
                    residuals[i, j] = np.power(S_T_median_error, 2) * np.average(
                        rho_func(future_errors / S_T_median_error)
                    )
                    # print(f"{i}, {j}, {residuals[i, j]:.5f}")

            min_idx = np.argmin(residuals)
            best_ld_1 = ld_1_index[np.unravel_index(min_idx, residuals.shape)[0]]
            best_ld_2 = ld_2_index[np.unravel_index(min_idx, residuals.shape)[1]]
            # Store the parameters that were calculated, even if a `target` or `s` were provided.
            self.ld_1 = best_ld_1
            self.ld_2 = best_ld_2
            self._residuals_HW = residuals
            self._holt_winters_warmup_fit(ld_1=best_ld_1, ld_2=best_ld_2, ld_s=ld_s)

        # Common code for both branches of if-else above
        future_errors = self.df["error"][self.train_samples]
        S_T_median_error = 1.48 * future_errors.abs().median()  # must handle NaNs!
        resids = np.power(S_T_median_error, 2) * np.nanmean(
            rho_func(future_errors / S_T_median_error)
        )

        # Ensure no negative square root is taken
        self._tau = np.sqrt(max(0.0, resids))
        if self.target is None:
            # Estimate the target as the median of the y-star (cleaned) y-values
            self.target = self.df["y_star"].median()
        else:
            self._target_calculated_best = self.df["y_star"].median()

        if self.s is None:
            self.s = (
                self._tau
            )  # or an alternative: self.df["sigma_hat"] is approximately OK

        # The "delta" emphasizes that it is the deviation from the target.
        self._delta_UCL_3sigma = +3.0 * self._tau
        self._delta_LCL_3sigma = -3.0 * self._tau

    def _holt_winters_warmup_fit(self, ld_1=0.5, ld_2=0.8, ld_s=0.2):
        """
        See paper: https://onlinelibrary.wiley.com/doi/abs/10.1002/for.1125

        Calculates the Holt-Winters fitting and control chart parameters, for given values of the
        smoothing parameters lambda_1 (how must local history for the level is used, with values
        approaching 1.0 implying that less history is used), and lambda_2 (history for the trend
        that is used, with lambda_2 approaching 1.0 implying that historical data is less
        interesting), and lambda_s, a similar parameter for the moving variance of the sequence.

        lambda_1 = ld_1 = 0.5 (default): value must be between 0 <= ld_1 <= 1.0
        lambda_2 = ld_2 = 0.8 (default): value must be between 0 <= ld_2 <= 1.0
        lambda_s = ld_s = 0.2 (default): value must be between 0 <= ld_2 <= 1.0, based on values
                                         used in the paper, recommended on page 291.

        The ideal lambda values (ld_1, ld_2, ld_s) can be found from a grid search.
        """
        df = self.df
        y_warm_up = df["y"].iloc[0 : self.warm_up["M"]]
        self.warm_up["y_zero_robust"] = y_warm_up.median()

        if isinstance(self.target, float):
            self.warm_up["alpha_0"] = self.target
            self.warm_up["beta_0"] = 0.0
        else:
            # p 290 of the paper, https://onlinelibrary.wiley.com/doi/abs/10.1002/for.1125
            self.warm_up["beta_0"] = repeated_median_slope(
                np.arange(self.warm_up["M"]), y_warm_up
            )
            self.warm_up["alpha_0"] = np.nanmedian(
                y_warm_up - self.warm_up["beta_0"] * np.arange(self.warm_up["M"])
            )

        if isinstance(self.s, float):
            self.warm_up["sigma_0"] = self.s

        else:
            # p 290 of the paper, https://onlinelibrary.wiley.com/doi/abs/10.1002/for.1125
            warm_up_residuals = (
                y_warm_up - self.warm_up["alpha_0"] - self.warm_up["beta_0"]
            )

            # Some other method that does not rely on SciPy for 1 function.
            self.warm_up["sigma_0"] = median_abs_deviation(
                warm_up_residuals, nan_policy="omit"
            )
            self.warm_up["residuals"] = warm_up_residuals

        df.loc[0, "rho_input"] = (
            self.warm_up["y_zero_robust"]
            - self.warm_up["alpha_0"]
            - self.warm_up["beta_0"]
        ) / self.warm_up["sigma_0"]
        df.loc[0, "psi_input"] = df["rho_input"][0]
        df.loc[0, "y_star"] = self.warm_up["y_zero_robust"]
        df.loc[0, "alpha_hat"] = self.warm_up["alpha_0"]
        df.loc[0, "beta_hat"] = self.warm_up["beta_0"]
        df.loc[0, "sigma_hat"] = self.warm_up["sigma_0"]

        for i in range(1, self.N):
            # Cover the warm-up period, and the rest of the data set. We need that for the residual
            # calculation later anyway.

            # Error = observed - predicted. Predicted = one-step-ahead prediction
            error_i = df["y"][i] - (df["alpha_hat"][i - 1] + df["beta_hat"][i - 1])
            if np.isnan(error_i):
                # If there is an error, replace it with the median of the last 10 error estimates
                # or as many points as available.
                error_i = df["error"].iloc[max(i - 10, 0) : i].abs().median()
            rho_i = error_i / df["sigma_hat"][i - 1]
            prior_variance = np.power(df["sigma_hat"][i - 1], 2)
            sigma_i = np.sqrt(
                rho(rho_i) * ld_s * prior_variance + (1.0 - ld_s) * prior_variance
            )
            psi_i = error_i / sigma_i
            y_star_i = (
                psi(psi_i) * sigma_i + df["alpha_hat"][i - 1] + df["beta_hat"][i - 1]
            )
            alpha_i = ld_1 * y_star_i + (1 - ld_1) * (
                df["alpha_hat"][i - 1] + df["beta_hat"][i - 1]
            )
            beta_i = (
                ld_2 * (alpha_i - df["alpha_hat"][i - 1])
                + (1 - ld_2) * df["beta_hat"][i - 1]
            )
            df.loc[i] = [
                df["y"][i],
                psi_i,
                rho_i,
                y_star_i,
                alpha_i,
                beta_i,
                sigma_i,
                error_i,
            ]

            # Checks: for algorithm debugging:
            # future_errors = df["error"]
            # S_T_median_error = 1.48 * future_errors.abs().median()  # must handle NaNs!
            # resids = np.power(S_T_median_error, 2) * \
            #                      np.nanmean((future_errors / S_T_median_error).apply(rho))
            # print(np.sqrt(max(0.0, resids)))
