# (c) Kevin Dunn, 2010-2021. MIT License. Based on own private work over the years.

import numpy as np
import pandas as pd


def summary_stats(x, method="robust") -> dict:
    """
    Returns summary statistics of the numeric values in vector `x`.

    Arguments:

        x (Numpy vector or Pandas series): A vector of univariate values to summarize.

    Returns:

        dict: a summary of the univariate vector. The following outputs are the most interesting:

            "center":   a measure of the center  (average). If method is ``robust``, this is the
                        median.
            "spread":   a measure of the spread. If method is ``robust``, this is the Sn, a robust
                        spread estimate.
    """
    if isinstance(x, pd.Series):
        x = x.copy(deep=True).values
    elif isinstance(x, np.ndarray):
        x = x.ravel()
    else:
        assert False, "Expecting a Numpy vector or Pandas series."

    out = {}
    out["mean"] = np.nanmean(x)
    out["std_ddof0"] = np.nanstd(x, ddof=0)
    out["std_ddof1"] = np.nanstd(x, ddof=1)
    out["rsd_classical"] = out["std_ddof1"] / out["mean"]
    (
        out["percentile_05"],
        out["percentile_25"],
        out["median"],
        out["percentile_75"],
        out["percentile_95"],
    ) = np.nanpercentile(x, [5, 25, 50, 75, 95])
    out["iqr"] = out["percentile_75"] - out["percentile_25"]
    out["min"] = np.nanmin(x)
    out["max"] = np.nanmax(x)
    out["N_non_missing"] = np.sum(~np.isnan(x))

    if method.lower() == "robust":
        out["center"], out["spread"] = out["median"], Sn(x)
        if (
            ((out["max"] - out["min"]) > 0)
            and (out["spread"] == 0)
            and (out["N_non_missing"] > 0)
        ):
            # Don't fully trust the Sn() yet. It works strangely on quantized data when there is
            # little variation. Replace the RSD with the classically calculated version in this
            # very specific case. This example shows it: [99, 95, 95, 100, 100, 100, 100, 95, 100,
            # 100, 100, 100, 105, 105, 100, 95, 105, 100, 95, 100]
            out["center"], out["spread"] = out["mean"], out["std_ddof1"]

    else:
        out["center"], out["spread"] = out["mean"], out["std_ddof1"]

    out["rsd"] = out["spread"] / out["center"]
    return out


def Sn(x, constant=1.1926):
    """
    Computes a robust scale estimator. The Sn metric is an efficient alternative to MAD.

    Args:
        x (iterable): A vector of values

    Outputs:
        a scalar value, the Sn estimate of spread

    The `constant` gives values which are consistent with iid values from a Gaussian distribution
    and no outliers.

    Tested against once of the most reliable open-source packages, written by some of the
    most respected names in the area of robust methods: [1]_ and [2]_.

    Disadvantages of MAD:

    *   It does not have particularly high efficiency for data that is in fact normal (37%).
        In comparison, the median has 64% efficiency for normal data.

    *   The MAD statistic also has an implicit assumption of symmetry. That is, it measures the
        distance from a measure of central location (the median).

    References
    ----------

    .. [1] https://cran.r-project.org/web/packages/robustbase/
    .. [2] Rousseeuw, Peter J.; Croux, Christophe (December 1993),
        "Alternatives to the Median Absolute Deviation", Journal of the American Statistical
        Association, American Statistical Association, 88 (424): 1273–1283,
        https://dx.doi.org/10.2307/2291267

    """
    n = np.sum(~np.isnan(x))
    if n == 0:
        return np.float64(np.nan)
    elif n == 1:
        return np.float64(0.0)
    medians = []
    for _, value in enumerate(x):
        # In the paper by Rousseeuw and Croux: they seem to iterate over all data. But
        # in the R-source code, they iterate over the two loops for i/= j. That also makes
        # sense, because the difference with itself does not help determine the spread. The
        # spread is exactly the average (median) of the deviations.

        if np.isnan(value):
            continue

        differences = []
        for j, other in enumerate(x):
            if np.isnan(other):
                continue
            differences.append(abs(value - other))

        # Belongs to the outer loop
        medians.append(np.nanmedian(differences))

    if n <= 9:
        # Correction factors for n = 2 to 9:
        correction = [0.743, 1.851, 0.954, 1.351, 0.993, 1.198, 1.005, 1.131][n - 2]
    elif n % 2:
        correction = n / (n - 0.9)  # odd number of cases
    else:
        correction = 1.0

    return constant * np.nanmedian(medians) * correction
