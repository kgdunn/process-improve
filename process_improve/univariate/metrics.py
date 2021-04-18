import warnings
from collections import defaultdict
from typing import Tuple, List, DefaultDict, Any

import numpy as np
import pandas as pd
from scipy.stats import shapiro, t

__eps = np.finfo(np.float32).eps


def t_value(p, v):
    r"""
    Returns the value on the x-axis if you plot the cumulative t-distribution with a fractional
    area of `p` (p is therefore a fractional value between 0 and 1 on the y-axis) and `v` is the
    degrees of freedom.


    Examples
    --------

    Since the cumulative distribution passes symmetrically through the x-axis at 0.0 for any
    number of degrees of freedom

    >>> t_value(0.5, v)
    0.0

    Zero fractional area under the curve is always at :math:`-\infty`:

    >>> t_value(0.0, v)
    -Inf

    100% fractional area is always at :math:`+\infty`:

    >>> t_value(1.0, v)
    +Inf

    See also
    --------
    t_value_cdf: does the inverse of this function.
    """
    return t.ppf(p, df=v)


def t_value_cdf(z, v):
    r"""
    Returns the value on the y-axis if you plot the cumulative t-distribution with a fractional
    area of `p` (p is therefore a fractional value between 0 and 1 on the y-axis) and `v` is the
    degrees of freedom.


    Examples
    --------

    Since the cumulative distribution passes symmetrically through the x-axis at 0.0 for any
    number of degrees of freedom

    >>> t_value_cdf(z=0.0, v)
    0.5

    Zero fractional area under the curve is always at :math:`-\infty`:

    >>> t_value_cdf(np.ninf, v)
    0.0

    100% fractional area is always at :math:`+\infty`:

    >>> t_value(np.inf, v)
    1.0

    See also
    --------
    t_value: does the inverse of this function.
    """
    return t.cdf(z, df=v)


def normality_check(x):
    """
    The p-value of the hypothesis that the data are from a normal distribution.

    If the p-value is less than the chosen alpha level (e.g. 0.05 or 0.025), then there is
    evidence that the data tested are NOT normally distributed.

    On the other hand, if the p-value is greater than the chosen alpha level, then the null
    hypothesis that the data came from a normally distributed population can not be rejected.
    NOTE: it does not mean that the data are normally distributed, just that we have nothing better
    to say about it. See the `Shapiro-Wilk test`_.

    Implementation

    Uses the Shapiro Wilk test directly taken from `scipy.stats.shapiro`_.

    .. _Shapiro-Wilk test:
        https://en.wikipedia.org/wiki/Shapiro-Wilk_test

    .. _scipy.stats.shapiro:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html
    """
    output = shapiro(x)
    return output[1]


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
    for i, value in enumerate(x):
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


def _contains_nan(a, nan_policy="propagate"):
    """
    From scipy.stats.stats
    """
    policies = ["propagate", "raise", "omit"]
    if nan_policy not in policies:
        raise ValueError(
            "nan_policy must be one of {%s}" % ", ".join("'%s'" % s for s in policies)
        )
    try:
        # Calling np.sum to avoid creating a huge array into memory
        # e.g. np.isnan(a).any()
        with np.errstate(invalid="ignore"):
            contains_nan = np.isnan(np.sum(a))
    except TypeError:
        # This can happen when attempting to sum things which are not
        # numbers (e.g. as in the function `mode`). Try an alternative method:
        try:
            contains_nan = np.nan in set(a.ravel())
        except TypeError:  # pragma: no cover
            # Don't know what to do. Fall back to omitting nan values and
            # issue a warning.
            contains_nan = False
            nan_policy = "omit"
            warnings.warn(
                "The input array could not be properly checked for nan "
                "values. nan values will be ignored.",
                RuntimeWarning,
            )

    if contains_nan and nan_policy == "raise":
        raise ValueError("The input contains nan values")

    return (contains_nan, nan_policy)


def ttest_difference_calculate(sample_A, sample_B, conflevel=0.995) -> dict:
    """Core calculation for a test of differences between the average of A and the average of B.
    No checking of inputs.

    Parameters
    ----------
    sample_A : iterable
        Vector of n_a measurements.
    sample_B : iterable
        Vector of n_b measurements.
    conflevel : float
        Value between 0 and 1 (closer to 1.0), that gives the level of confidence required for
        the 2-sided test.

    Returns
    -------
    dict
        Outcomes from the statistical test.
    """
    axis = 0
    v1, v2 = sample_A.var(axis, ddof=1), sample_B.var(axis, ddof=1)
    n_A, n_B = sample_A.shape[axis], sample_B.shape[axis]
    m1, m2 = sample_A.mean(), sample_B.mean()
    df = n_A + n_B - 2.0
    ct = abs(t_value(p=(1 - conflevel) / 2.0, v=df))
    d = m2 - m1
    svar = ((n_A - 1) * v1 + (n_B - 1) * v2) / df

    with np.errstate(divide="ignore", invalid="ignore"):
        sd_z_variate = np.sqrt(svar * (np.divide(1.0, n_A) + np.divide(1.0, n_B)))
        z_variate = np.divide(d, sd_z_variate)

    confint_lo = d - ct * sd_z_variate
    confint_hi = d + ct * sd_z_variate

    return {
        "Group A number": int(n_A),
        "Group B number": int(n_B),
        "Group A average": sample_A.mean(),
        "Group B average": sample_B.mean(),
        "z value": z_variate,
        "ConfInt: Lo": confint_lo,
        "ConfInt: Hi": confint_hi,
        "p value": 2 * t_value_cdf(-np.abs(z_variate), df),
        "Degrees of freedom": df,
        "Pooled standard deviation": sd_z_variate,
    }


def ttest_difference(
    df: pd.DataFrame, grouper_column: str, values_column: str, conflevel=0.995
):
    """
    Calculates the t-test for differences between two or more groups and returns a confidence
    interval for the difference. The test is for UNPAIRED differences.

    The dataframe `df` contains a `grouper_column` with 2 or more unique values (e.g. 'A' and 'B').
    All unique values of the `grouper_column` are used, and t-tests are done between the values
    in the `values_column`.

    Args:
        df (pd.DataFrame): Dataframe of the values and grouping variable.
        grouper_column (str): Indicates which column will be grouped on.
        values_column (str): Which column contains the numeric values to calculate the test on.
        conflevel (float, optional): [description]. Defaults to 0.995.

    Output: Dataframe with columns containing the statistical outputs of the t-test, including:
        1. Group "A" name
        2. Group "B" name
        3. Group "A" mean
        4. Group "B" mean
        5. z-value for the difference between group "B" minus group "A"
        6. p-value for this z-value
        7. Confidence interval low value for difference between group "B" minus group "A"
        8. Confidence interval high value for difference between group "B" minus group "A"

    Example:
    df : has 3 levels in the grouper variable; ['Marco', 'Pete', 'Sam']

    Output will have 3 rows:
    Group A name  Group B name
    Marco         Pete
    Marco         Sam
    Pete          Sam
    """

    data_subset = df[[grouper_column, values_column]].copy()
    data_subset = data_subset.dropna()
    groups = df[grouper_column].unique()
    output = pd.DataFrame()
    groups = list(groups)
    while len(groups) > 0:
        groupA_name = groups.pop(0)
        for groupB_name in groups:

            sample_A = data_subset[data_subset[grouper_column].eq(groupA_name)][
                values_column
            ]
            sample_B = data_subset[data_subset[grouper_column].eq(groupB_name)][
                values_column
            ]
            sample_A = sample_A.astype(np.float64)
            sample_B = sample_B.astype(np.float64)
            basic_stats = ttest_difference_calculate(sample_A, sample_B, conflevel)
            basic_stats.update(
                {
                    "Group A name": groupA_name,
                    "Group B name": groupB_name,
                }
            )
            output = output.append(basic_stats, ignore_index=True)

    return output


def ttest_paired_difference_calculate(differences, conflevel=0.995) -> dict:
    """Core calculation for a test of differences.

    Parameters
    ----------
    sample_A : iterable
        Vector of n_a measurements.
    sample_B : iterable
        Vector of n_b measurements.
    conflevel : float
        Value between 0 and 1 (closer to 1.0), that gives the level of confidence required for
        the 2-sided test.

    Returns
    -------
    dict
        Outcomes from the statistical test.
    """
    diff_mean = differences.mean()
    diff_svar = differences.std(ddof=1)
    dof = differences.shape[0] - 1  # n-1 d

    # By the central limit theorem, the `differences` values should be normally distributed
    # with average (central value) given by mean of group A values, minus mean of group B values.
    # Scale factor is the standard deviations of `differences`: estimated, therefore t-distribution
    ct = abs(t_value(p=(1 - conflevel) / 2.0, v=dof))

    with np.errstate(divide="ignore", invalid="ignore"):
        sd_z_variate = diff_svar * np.sqrt(np.divide(1.0, dof + 1))
        z_variate = np.divide(diff_mean, sd_z_variate)

    confint_lo = diff_mean - ct * sd_z_variate
    confint_hi = diff_mean + ct * sd_z_variate

    return {
        "Differences mean": diff_mean,
        "z value": z_variate,
        "ConfInt: Lo": confint_lo,
        "ConfInt: Hi": confint_hi,
        "p value": 2 * t_value_cdf(-np.abs(z_variate), dof),
        "Degrees of freedom": dof,
        "Standard deviation": sd_z_variate,
    }


def ttest_paired_difference(
    df: pd.DataFrame, grouper_column: str, values_column: str, conflevel=0.995
):
    """
    Calculates the t-test for paired differences between two or more groups and returns a
    confidence interval for the difference. The test is for PAIRED differences.
    The differences is always defined as the A values minus the B values: after - before, or A - B.

    The dataframe `df` contains a `grouper_column` with 2 or more unique values (e.g. 'A' and 'B').
    All unique values of the `grouper_column` are used, and t-tests are done between the values
    in the `values_column`.

    When selecting the columns, the number of values per column must be the same.

    Args:
        df (pd.DataFrame): Dataframe of the values and grouping variable.
        grouper_column (str): Indicates which column will be grouped on.
        values_column (str): Which column contains the numeric values to calculate the test on.
        conflevel (float, optional): [description]. Defaults to 0.995.

    Output: Dataframe with columns containing the statistical outputs of the t-test, including:
        1. Group A name
        2. Group B name
        3. Group A mean
        4. Group B mean
        5. Differences mean:  The average difference between the groups
            (Note: this is not the same as the difference of the averages, given by 4 and 5 above!)
        CHECK5. z-value for the difference between group "B" minus group "A"
        CHECK6. p-value for this z-value
        CHECK7. Confidence interval low value for difference between group "B" minus group "A"
        CHECK8. Confidence interval high value for difference between group "B" minus group "A"

    """
    data_subset = df[[grouper_column, values_column]].copy()
    data_subset = data_subset.dropna()
    groups = df[grouper_column].unique()
    output = pd.DataFrame()
    groups = list(groups)
    while len(groups) > 0:
        groupA_name = groups.pop(0)
        for groupB_name in groups:

            sample_A = data_subset[data_subset[grouper_column].eq(groupA_name)][
                values_column
            ]
            sample_B = data_subset[data_subset[grouper_column].eq(groupB_name)][
                values_column
            ]
            sample_A = sample_A.astype(np.float64)
            sample_B = sample_B.astype(np.float64)
            assert sample_A.shape[0] == sample_B.shape[0]
            differences = (
                sample_A - sample_B.values
            )  # only the .values of one vector are needed!
            basic_stats = ttest_paired_difference_calculate(differences, conflevel)
            basic_stats.update(
                {
                    "Group A name": groupA_name,
                    "Group B name": groupB_name,
                    "Group A number": sample_A.shape[0],
                    "Group B number": sample_B.shape[0],
                    "Group A average": sample_A.mean(),
                    "Group B average": sample_B.mean(),
                }
            )
            output = output.append(basic_stats, ignore_index=True)

    return output


def confidence_interval(
    df: pd.DataFrame, column_name: str, conflevel=0.95, style="robust"
) -> tuple:
    """
    Calculates the confidence interval, returned as a tuple, for the `column_name` (str) in the
    dataframe `df`, for a given confidence level `conflevel` (default: 0.95).

    `style`: ['robust'; 'regular']: indicates which style of estimates to use for the center and
                                    spread. Default: 'robust'

    Missing values are ignored.
    """

    # TODO : http://www.rips-irsp.com/article/10.5334/irsp.82/

    data = df[column_name]
    n = data.count()

    if style.lower() == "robust":
        center = data.median()
        spread = median_abs_deviation(data.values, nan_policy="omit")
    else:
        center = data.mean()
        spread = data.std()

    c_t = t_value(1 - (1 - conflevel) / 2, n - 1)
    return (center - c_t * spread / np.sqrt(n), center + c_t * spread / np.sqrt(n))


def _mad_1d(x, center, nan_policy):
    """

    Taken from `scipy.stats.stats`

    Median absolute deviation for 1-d array x.
    This is a helper function for `median_abs_deviation`; it assumes its
    arguments have been validated already.  In particular,  x must be a
    1-d numpy array, center must be callable, and if nan_policy is not
    'propagate', it is assumed to be 'omit', because 'raise' is handled
    in `median_abs_deviation`.
    No warning is generated if x is empty or all nan.
    """
    isnan = np.isnan(x)
    if isnan.any():
        if nan_policy == "propagate":
            return np.nan
        x = x[~isnan]
    if x.size == 0:
        # MAD of an empty array is nan.
        return np.nan
    # Edge cases have been handled, so do the basic MAD calculation.
    med = center(x)
    mad = np.median(np.abs(x - med))
    return mad


def median_abs_deviation(
    x, axis=0, center=np.median, scale="normal", nan_policy="omit"
):
    r"""

    Taken from `scipy.stats.stats`: we want the same functionality, but with a slightly different
    default function signature:

    *   `scale='normal'` instead of `scale=1.0`.
    *   `nan_policy='omit'` instead of `nan_policy='propogate'`

    Compute the median absolute deviation of the data along the given axis.  The median absolute
    deviation (MAD_) computes the median over the absolute deviations from the median. It is a
    measure of dispersion similar to the standard deviation but more `robust to outliers`_.
    The MAD of an empty array is ``np.nan``.

    Parameters
    ----------

    x : array_like
        Input array or object that can be converted to an array.
    axis : int or None, optional
        Axis along which the range is computed. Default is 0. If None, compute
        the MAD over the entire array.
    center : callable, optional
        A function that will return the central value. The default is to use
        np.median. Any user defined function used will need to have the
        function signature ``func(arr, axis)``.
    scale : scalar or str, optional
        The numerical value of scale will be divided out of the final
        result. The default is 1.0. The string "normal" is also accepted,
        and results in `scale` being the inverse of the standard normal
        quantile function at 0.75, which is approximately 0.67449.
        Array-like scale is also allowed, as long as it broadcasts correctly
        to the output such that ``out / scale`` is a valid operation. The
        output dimensions depend on the input array, `x`, and the `axis`
        argument.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan.
        The following options are available (default is 'propagate'):
        * 'propagate': returns nan
        * 'raise': throws an error
        * 'omit': performs the calculations ignoring nan values

    Returns
    -------
    mad : scalar or ndarray
        If ``axis=None``, a scalar is returned. If the input contains
        integers or floats of smaller precision than ``np.float64``, then the
        output data-type is ``np.float64``. Otherwise, the output data-type is
        the same as that of the input.

    See Also
    --------
    numpy.std, numpy.var, numpy.median, scipy.stats.iqr, scipy.stats.tmean,
    scipy.stats.tstd, scipy.stats.tvar

    Notes
    -----
    The `center` argument only affects the calculation of the central value
    around which the MAD is calculated. That is, passing in ``center=np.mean``
    will calculate the MAD around the mean - it will not calculate the *mean*
    absolute deviation.

    The input array may contain `inf`, but if `center` returns `inf`, the
    corresponding MAD for that data will be `nan`.

    References
    ----------

    .. _MAD:
        "Median absolute deviation",
        https://en.wikipedia.org/wiki/Median_absolute_deviation

    .. _robust to outliers:
        "Robust measures of scale",
        https://en.wikipedia.org/wiki/Robust_measures_of_scale

    Examples
    --------

    When comparing the behavior of `median_abs_deviation` with ``np.std``,
    the latter is affected when we change a single value of an array to have an
    outlier value while the MAD hardly changes:

    >>> from scipy import stats
    >>> x = stats.norm.rvs(size=100, scale=1, random_state=123456)
    >>> x.std()
    0.9973906394005013
    >>> stats.median_abs_deviation(x)
    0.82832610097857
    >>> x[0] = 345.6
    >>> x.std()
    34.42304872314415
    >>> stats.median_abs_deviation(x)
    0.8323442311590675

    Axis handling example:

    >>> x = np.array([[10, 7, 4], [3, 2, 1]])
    >>> x
    array([[10,  7,  4],
           [ 3,  2,  1]])
    >>> stats.median_abs_deviation(x)
    array([3.5, 2.5, 1.5])
    >>> stats.median_abs_deviation(x, axis=None)
    2.0

    Scale normal example:

    >>> x = stats.norm.rvs(size=1000000, scale=2, random_state=123456)
    >>> stats.median_abs_deviation(x)
    1.3487398527041636
    >>> stats.median_abs_deviation(x, scale='normal')
    1.9996446978061115
    """
    if not callable(center):
        raise TypeError(
            "The argument 'center' must be callable. The given "
            f"value {repr(center)} is not callable."
        )

    # An error may be raised here, so fail-fast, before doing lengthy
    # computations, even though `scale` is not used until later
    if isinstance(scale, str):
        if scale.lower() == "normal":
            scale = 0.6744897501960817  # special.ndtri(0.75)
        else:
            raise ValueError(f"{scale} is not a valid scale value.")

    x = np.asarray(x)

    # Consistent with `np.var` and `np.std`.
    if not x.size:
        if axis is None:
            return np.nan
        nan_shape = tuple(item for i, item in enumerate(x.shape) if i != axis)
        if nan_shape == ():
            # Return nan, not array(nan)
            return np.nan
        return np.full(nan_shape, np.nan)  # pragma: no cover

    contains_nan, nan_policy = _contains_nan(x, nan_policy)

    if contains_nan:
        if axis is None:
            mad = _mad_1d(x.ravel(), center, nan_policy)
        else:
            mad = np.apply_along_axis(_mad_1d, axis, x, center, nan_policy)
    else:
        if axis is None:
            med = center(x, axis=None)
            mad = np.median(np.abs(x - med))
        else:
            # Wrap the call to center() in expand_dims() so it acts like
            # keepdims=True was used.
            med = np.expand_dims(center(x, axis=axis), axis)
            mad = np.median(np.abs(x - med), axis=axis)

    return mad / scale


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
        raise ValueError("Expecting a NumPy vector or Pandas series.")

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


def outlier_detection_multiple(
    x, algorithm="esd", max_outliers_detected=1, **kwargs
) -> Tuple[List[int], DefaultDict[Any, Any]]:
    """
    Returns a list of indexes of points in the vector `x` which are likely outliers.

    A second output (can be ignored) contains the details of the values used to make the decision.

    Arguments:

        x {list, sequence, NumPy vector/array} -- [A sequence, list or vector which can be
        unravelled.]

    Keyword Arguments:

        algorithm -- Two algorithms are possible to detect outliers: (default: "esd")

            'esd': Generalized ESD Test for Outliers.
            If `max_outliers_detected=1` this is essentially Grubb's test.
            For more details, please see:
            https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h3.htm

            'cc-robust': Build a robust control-chart for the sequence `x` and
            points should lie outside the +/- 3 sigma limits are considered
            outliers.
            Not Implemented Yet: left here as an idea for the future, but not confirmed yet.

        max_outliers_detected -- The maximum number of outliers that
            should be detected, as required by the algorithms.

        kwargs -- Algorithm dependent arguments. Defaults are shown here.

            'esd':

                'robust_variant' = True. Uses the median and MAD for the center
                and the standard deviation respectively.

                'alpha' = 0.05. The significance level of the testing.


    """
    algorithm = algorithm.strip().lower()
    x = pd.Series(x).reset_index(drop=True)
    max_outliers_detected = int(max_outliers_detected)

    if algorithm == "esd":
        """
        https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h3.htm

        Note: the two-sided test is implemented here. p = 1-alpha/(2*(n-i+1))
        """

        # 0. Default settings
        robust_variant = bool(kwargs.get("robust_variant", True))
        alpha = float(kwargs.get("alpha", 0.05))

        assert alpha <= 1.0
        assert max_outliers_detected <= len(x)

        # 1. Run K-S test first to check normality

        # 2. https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h3.htm
        x = x.copy(deep=True)
        extra_out = defaultdict(list)
        N = x.shape[0] - pd.isna(x).sum()

        for k in range(max_outliers_detected):

            i = k + 1
            extra_out["i"].append(i)

            if robust_variant:
                variation = median_abs_deviation(x)
                R = ((x - x.median()) / variation).abs()
            else:
                variation = x.std()
                R = ((x - x.mean()) / variation).abs()

            dof = N - i - 1
            p = 1 - alpha / (2 * (N - i + 1))
            t_s = t_value(p, dof)
            lambda_i = (N - i) * t_s / np.sqrt((dof + np.power(t_s, 2)) * (dof + 2))
            extra_out["lambda"].append(lambda_i)

            R_i_idx = R.idxmax()
            g = R.max()
            # Formula from the R-function for the Grubb's calculation
            t = np.sqrt((g ** 2 * N * (2 - N)) / (g ** 2 * N - (N - 1) ** 2))
            if t <= 0:
                p_value = 0
            else:
                p_value = min(N * (1 - t_value_cdf(t, N - 2)), 1)

            extra_out["R_i_idx"].append(R_i_idx)
            extra_out["R_i"].append(R.max())
            extra_out["p-value"].append(p_value)

            if variation > __eps:
                # The variation, if zero or small, will fail to drop this index
                x.drop(R_i_idx, inplace=True)

        try:
            cutoff_i = np.where(
                np.array(extra_out["R_i"]) - np.array(extra_out["lambda"]) >= 0,
            )[0][0]
        except IndexError:
            cutoff_i = -1

        # The outlier indices are the points from the start of
        # `extra_out['R_i_idx']`, up to, and including, the `cutoff_i` index
        # in the list.
        extra_out["cutoff"] = cutoff_i

        outlier_index = extra_out["R_i_idx"][0 : (cutoff_i + 1)]
        return outlier_index, extra_out
    else:
        return [], defaultdict(dict)


def within_between_standard_deviation(df, measured: str, repeat: str) -> dict:
    """
    Given a DataFrame `df` of raw data, and an indication of which column is the `measured` value
    column, and which is the `repeat` indicator, it will calculate the within and between replicate
    standard deviation.

    Example

    Two measurements on day 1 ``[101, 102]`` and two measurements on day 2 ``[94, 95]``. The
    between-day variation can already be expected to be much greater than the within-day variation.

    >>> df = pd.DataFrame(data={'Result': [101, 102, 94, 95], 'Repeat': [1, 1, 2, 2]})
        Result  Repeat
    0     101       1
    1     102       1
    2      94       2
    3      95       2
    >>> output = within_between_standard_deviation(df, measured="Result", repeat="Repeat")
    {'total_ms':       16.666667,
     'total_dof':      3,
     'within_ms':      0.5,
     'within_stddev':  0.70711,
     'within_dof':     2,
     'between_ms':     49.0,
     'between_stddev': 7.0,
     'between_dof':    1}

    Note

    * SSQ = sum of squares
    * DOF= degrees of freedom
    * MS = mean square = (sum of squares) / (degrees of freedom) = SSQ / DOF = variance
    """

    # Overall statistics:
    total_ms = max(0, df[measured].var())
    total_dof = max(0, df[measured].count() - 1)

    # Within a group: calculate the standard deviation (of variance) of each group. If you take
    # the average of those variances, pooled, you get an estimate of the within-group variance.
    within_ms = 0.0
    within_dof = 0
    for idx, group in df.groupby(repeat):
        dof_group_i = max(0, group[measured].count() - 1)
        within_dof += dof_group_i
        # handling missing data makes for messier code
        within_ms = np.nansum([within_ms, group[measured].var() * dof_group_i])

    if within_dof == 0:
        within_ms = 0
    else:
        within_ms /= within_dof

    # Between groups: the ANOVA relationship is such that SSQ(total) = SSQ(between) + SSQ(within).
    # Therefore the between-group statistics are found by differencing
    between_dof = max(0, total_dof - within_dof)
    if between_dof == 0:
        between_ms = 0
    else:
        between_ms = max(
            0.0, (total_ms * total_dof - within_ms * within_dof) / between_dof
        )

    return {
        "total_ms": total_ms,
        "total_dof": total_dof,
        "within_ms": within_ms,
        "within_stddev": np.sqrt(within_ms),
        "within_dof": within_dof,
        "between_ms": between_ms,
        "between_stddev": np.sqrt(between_ms),
        "between_dof": between_dof,
    }
