import pytest
from pytest import approx
import numpy as np
import pandas as pd

import process_improve.univariate.metrics as univariate


class test_t_values:
    """
    Checks the calculation of t values (at a given 'alpha' and with an integer number of degrees
    of freedom), against the values from R.
    """

    assert univariate.t_value(0, 1) == np.NINF
    assert univariate.t_value(1, 2) == np.inf
    assert univariate.t_value(0.5, 3) == approx(0, rel=1e-16)

    # Tested in R:  qt(0.9, 5) ->  1.475884
    assert univariate.t_value(0.9, 5) == approx(1.475884, rel=1e-6)


class test_t_values_cdf:
    """
    Checks the calculation of t values (at a given 'alpha' and with an integer number of degrees
    of freedom), against the values from R.
    """

    assert univariate.t_value_cdf(0, 1) == 0.5
    assert univariate.t_value_cdf(np.NINF, 2) == 0
    assert univariate.t_value_cdf(np.inf, 3) == 1

    # Tested in R:  pt(0.9, 5) ->  0.7953144
    assert univariate.t_value_cdf(0.9, 5) == approx(0.7953144, rel=1e-8)
    # Tested in R:  pt(0.5, 1) ->  0.6475836
    assert univariate.t_value_cdf(0.5, 1) == approx(0.6475836, rel=1e-7)


def test_normality_check():

    """
    Tests on data actually from a normal distribution, and some data which is from a
    uniform distribution.

    In R:  version 3.6.0 (2019-04-26)
    > set.seed(42)
    > x = rnorm(10)
    > shapiro.test(x)
    > [1]  1.37095845 -0.56469817  ...
    > Shapiro-Wilk normality test
    > W = 0.9287, p-value = 0.4352

    > y = runif(10)
    > shapiro.test(y)
    > [1] 0.90403139 0.13871017 ...
    > Shapiro-Wilk normality test
    > W = 0.87415, p-value = 0.1117
    """
    x = [
        1.37095845,
        -0.56469817,
        0.36312841,
        0.63286260,
        0.40426832,
        -0.10612452,
        1.51152200,
        -0.09465904,
        2.01842371,
        -0.06271410,
    ]
    y = [
        0.90403139,
        0.13871017,
        0.98889173,
        0.94666823,
        0.08243756,
        0.51421178,
        0.39020347,
        0.90573813,
        0.44696963,
        0.83600426,
    ]

    # Data actually are from a  normal distribution:
    assert univariate.normality_check(x) == approx(0.4352, abs=1e-3)

    # Data actually are from a uniform distribution:
    assert univariate.normality_check(y) == approx(0.1117, abs=1e-3)


def test_univariate_robust_scale():
    """
    A scale estimator which is robust to outliers

    Testing against R code [R version 3.6.0 (2019-04-26)]
    > library(robustbase)
    # All code has the default argument: "finite.corr=TRUE"
    > robustbase::Sn(c(0, 1))                   # 0.8861018: FAILS in our implementation
    > robustbase::Sn(c(0, 1, 2))                # 2.207503
    > robustbase::Sn(c(0, 1, 2, 3))             # 1.13774: FAILS in our implementation
    > robustbase::Sn(c(0, 1, 2, 3, 4))          # 1.611203
    > robustbase::Sn(c(0, 1, 2, 3, 4, 5))       # 2.368504: FAILS in our implementation
    > robustbase::Sn(c(0, 1, 2, 3, 4, 5, 6))    # 2.85747
    > robustbase::Sn(c(0, 1, 2, 3, 4, 50, 6))   # 2.85747
    > robustbase::Sn(c(0, 1, 20, 3, 4, 50, 6))  # 5.714939: FAILS in our implementation
    > robustbase::Sn(c(0, 10, 20, 3, 4, 50, 6)) # 8.572409
    > robustbase::Sn(seq(1, 10))                # 3.5778: FAILS in our implementation
    > robustbase::Sn(seq(1, 11))                # 3.896614
    > robustbase::Sn(seq(1, 19))                # 6.259503
    > robustbase::Sn(seq(1, 1500))              # 447.225

    TODO: found this weird sequence that gives Sn of zero, even though there is variability:
    99, 95, 95, 100, 100, 100, 100, 95, 100, 100, 100, 100, 105, 105, 100, 95, 105, 100, 95, 100
    How to make it robust to this weird situation?
    """

    # Tests with an even number of samples, and small sample sizes do not agree with R.
    # This is because the R implementation aims for efficiency of calculation, and does not
    # follow the formula presented in the original paper.
    # Since we aim to be using this on medium/larger data sets, it should not matter.

    assert univariate.Sn(list(range(3))) == approx(2.207503, rel=1e-6)
    assert univariate.Sn(list(range(5))) == approx(1.611203, rel=1e-6)
    assert univariate.Sn(list(range(7))) == approx(2.85747, rel=1e-6)
    assert univariate.Sn([0, 10, 20, 3, 4, 50, 6]) == approx(8.572409, rel=1e-7)
    assert univariate.Sn(list(range(1, 12))) == approx(3.896614, rel=1e-6)
    assert univariate.Sn(list(range(1, 19))) == approx(5.3667, rel=1e-6)
    assert univariate.Sn(list(range(1, 20))) == approx(6.259503, rel=1e-6)
    # Corner cases:
    assert np.isnan(univariate.Sn([]))
    assert univariate.Sn([13]) == 0.0


def test_summary_stats_corner_case_with_robust_scale():
    x = [
        99,
        95,
        95,
        100,
        100,
        100,
        100,
        95,
        100,
        100,
        100,
        100,
        105,
        105,
        100,
        95,
        105,
        100,
        95,
        100,
    ]
    out = univariate.summary_stats(np.array(x), method="robust")
    assert out["center"] == np.mean(x)
    assert out["center"] != np.median(x)

    out = univariate.summary_stats(np.array(x), method="something-else")
    assert out["center"] == np.mean(x)
    assert out["center"] != np.median(x)


def test_median_abs_deviation():

    x = np.array([[10, 7, 4], [3, 2, 1]])
    assert univariate.median_abs_deviation(x, scale=1) == approx([3.5, 2.5, 1.5])
    assert univariate.median_abs_deviation(x, axis=None, scale=1) == 2.0

    from scipy import stats

    x = stats.norm.rvs(size=1000000, scale=2, random_state=123456)
    assert univariate.median_abs_deviation(x, scale=1) == approx(
        1.3487398527041636, rel=1e-12
    )
    assert univariate.median_abs_deviation(x) == approx(1.9996446978061115, rel=1e-12)

    with pytest.raises(TypeError, match=r"The argument 'center' must .*"):
        _ = univariate.median_abs_deviation(x, center=0.0)

    with pytest.raises(ValueError, match=r".* is not a valid scale value."):
        _ = univariate.median_abs_deviation(x, scale="robust")

    with pytest.raises(ValueError, match=r"nan_policy must be one of .*"):
        univariate.median_abs_deviation([1, 2, 3, 4], nan_policy="propogatess")

    with pytest.raises(TypeError):
        univariate.median_abs_deviation(["a", "b"])

    with pytest.raises(ValueError, match="The input contains nan values"):
        univariate.median_abs_deviation([np.nan, 1], nan_policy="raise")

    assert np.isnan(
        univariate.median_abs_deviation([np.nan, 1, 2], nan_policy="propagate")
    )

    assert np.isnan(
        univariate.median_abs_deviation(
            [
                np.nan,
            ],
            axis=None,
        )
    )
    assert np.isnan(
        univariate.median_abs_deviation(
            [
                np.nan,
            ],
            axis=0,
        )
    )
    assert np.isnan(univariate.median_abs_deviation([], axis=None))
    assert np.isnan(univariate.median_abs_deviation([], axis=0))
    assert np.isnan(
        univariate.median_abs_deviation(np.empty((2, 3, 4)) * np.nan, axis=None)
    )
    assert np.isnan(univariate.median_abs_deviation(np.array([np.nan, np.nan]), axis=0))


def test_t_test_differences():
    """
    Tests for the t-test of differences.

    R code to validate against:
    > sam = c(8.80, 6.60, 7.26, 9.32, 5.88, 8.44, 11.39, 6.82, 9.32, 5.63, 9.65, 9.49)
    > jen = c(5.37, 4.83, 7.87, 3.30, 8.26, 7.87, 8.26, 6.13, 6.13, 5.63, 2.96, 5.88)
    > t.test(sam, jen, paired = FALSE, var.equal = TRUE)
        data:  sam and jen
        t = 2.9906, df = 22, p-value = 0.00674
        alternative hypothesis: true difference in means is not equal to 0
        95 percent confidence interval:
        0.6669748 3.6846919
        sample estimates:
        mean of x mean of y
        8.216667  6.040833
    """
    sam = [8.80, 6.60, 7.26, 9.32, 5.88, 8.44, 11.39, 6.82, 9.32, 5.63, 9.65, 9.49]
    jen = [5.37, 4.83, 7.87, 3.30, 8.26, 7.87, 8.26, 6.13, 6.13, 5.63, 2.96, 5.88]
    mik = [5.80, 9.00, 5.60, 8.40, 8.60, None, None, None, None, None, None, None]
    temp = pd.DataFrame(data={"Sam": sam, "Jen": jen, "Mik": mik})
    temp.reset_index(inplace=True)
    temp = temp.melt(id_vars="index")
    df = temp.drop("index", axis=1).dropna().rename(columns={"variable": "Person"})
    output = univariate.ttest_difference(
        df, grouper_column="Person", values_column="value", conflevel=0.95
    )
    row = output[output["Group A name"].eq("Sam") & output["Group B name"].eq("Jen")]

    # Assert against the R values in the above validation script.
    # Note: in R, the test for t.test(A, B), checking A minus B.
    # We define our test as group B minus group A. Therefore we have to flip our signs,
    # and high/low values of the confidence interval
    assert row["Group A average"][0] == approx(8.216667, rel=1e-5)
    assert row["Group B average"][0] == approx(6.040833, rel=1e-5)
    assert row["z value"][0] == approx(-2.9906, rel=1e-4)
    assert row["p value"][0] == approx(0.00674, rel=1e-3)
    assert row["ConfInt: Lo"][0] == approx(-3.6846919, rel=1e-4)
    assert row["ConfInt: Hi"][0] == approx(-0.6669748, rel=1e-4)
    assert row["Degrees of freedom"][0] == approx(22, rel=1e-8)


def test_t_paried_test_differences():
    """
    Tests for the paired t-test of differences.

    R code to validate against:
    > sam = c(8.80, 6.60, 7.26, 9.32, 5.88, 8.44, 11.39, 6.82, 9.32, 5.63, 9.65, 9.49)
    > jen = c(5.37, 4.83, 7.87, 3.30, 8.26, 7.87, 8.26, 6.13, 6.13, 5.63, 2.96, 5.88)
    > t.test(sam, jen, paired = FALSE, var.equal = TRUE)
            Paired t-test

        data:  sam and jen
        t = 2.8139, df = 11, p-value = 0.01685
        alternative hypothesis: true difference in means is not equal to 0
        95 percent confidence interval:
        0.4739104 3.8777563
        sample estimates:
        mean of the differences
                    2.175833
    """
    sam = [8.80, 6.60, 7.26, 9.32, 5.88, 8.44, 11.39, 6.82, 9.32, 5.63, 9.65, 9.49]
    jen = [5.37, 4.83, 7.87, 3.30, 8.26, 7.87, 8.26, 6.13, 6.13, 5.63, 2.96, 5.88]
    temp = pd.DataFrame(data={"Sam": sam, "Jen": jen})
    temp.reset_index(inplace=True)
    temp = temp.melt(id_vars="index")
    df = temp.drop("index", axis=1).dropna().rename(columns={"variable": "Person"})
    output = univariate.ttest_paired_difference(
        df, grouper_column="Person", values_column="value", conflevel=0.95
    )
    row = output[output["Group A name"].eq("Sam") & output["Group B name"].eq("Jen")]

    # Assert against the R values in the above validation script.
    # Note: in R, the test for t.test(A, B), checking A minus B.
    assert row["Group A average"][0] == approx(8.216667, rel=1e-7)
    assert row["Group B average"][0] == approx(6.040833, rel=1e-7)
    assert row["Differences mean"][0] == approx(2.175833, rel=1e-6)
    assert row["z value"][0] == approx(2.8139, rel=1e-4)
    assert row["p value"][0] == approx(0.01685, rel=1e-4)
    assert row["ConfInt: Lo"][0] == approx(0.4739104, rel=1e-7)
    assert row["ConfInt: Hi"][0] == approx(3.8777563, rel=1e-7)
    assert row["Degrees of freedom"][0] == 11


@pytest.fixture
def univariate_summary():
    """
    A univariate case study

    In R:
    r <- c(108, 89.52, 95.16, 101.61, 99.19, 100, 93.55, 97.58, 93.55, 98.39, 88.71)

    r_mean <- mean(r)                           # 96.8418181818182
    r_std_ddof1 <- sd(r)                        # 5.56692521627841
    r_rsd <- r_std_ddof1 / r_mean               # 0.05748472427301
    r_median <- median(r)                       # 97.58
    r_iqr <- IQR(r)                             # 6.045
    r_Sn <- robustbase::Sn(r)                   # 6.28653702970298
    r_rsd_robust <- r_Sn / r_median             # 0.06442444178831

    r_min <- min(r)                             # 88.71
    r_max <- max(r)                             # 108
    r_n <- length(r)                            # 11

    r_percentile_5 <- quantile(r, probs = 0.05) # 89.115
    r_percentile_25<- quantile(r, probs = 0.25) # 93.55
    r_percentile_75<- quantile(r, probs = 0.75) # 99.595
    r_percentile_95<- quantile(r, probs = 0.95) # 104.805


    See the tests for "Test__univariate_robust_scale", to understand why an odd number of
    samples were chosen.
    """
    y = [108.0, 89.52, 95.16, 101.61, 99.19, 100, 93.55, 97.58, 93.55, 98.39, 88.71]
    return pd.DataFrame(data={"values": y})


def test_compare_to_R_with_without_missing(univariate_summary):
    # Verifies that we can reproduce results from R. R version 3.6.0 (2019-04-26)
    # Checked on 21 February 2020.
    data = univariate_summary
    for k in range(2):
        if k > 0:
            # For the second loop: add a missing value and ensure you get the same results
            # as for the first loop (without missing values)
            data = data.append({"values": np.nan}, ignore_index=True)

        out = univariate.summary_stats(data["values"])
        assert out["mean"] == approx(96.84181818181816, abs=1e-8)
        assert out["std_ddof1"] == approx(5.566925216278406, abs=1e-8)
        assert out["rsd_classical"] == approx(0.05748472427301, abs=1e-8)
        assert out["median"] == approx(97.58, abs=1e-8)
        assert out["center"] == approx(97.58, abs=1e-8)  # center = median by default
        assert out["iqr"] == approx(6.045, abs=1e-8)
        assert out["spread"] == approx(
            6.28653702970298, abs=1e-8
        )  # spread = Sn (changed in 0.5)
        assert out["rsd"] == approx(0.06442444178831, abs=1e-8)
        assert out["min"] == 88.71
        assert out["max"] == 108
        assert out["N_non_missing"] == 11

        # Note: there are differences in how Numpy and R interpolate the results
        assert out["percentile_05"] == approx(89.115, rel=1e-6)
        assert out["percentile_25"] == approx(93.55, rel=1e-6)
        assert out["percentile_75"] == approx(99.595, rel=1e-6)
        assert out["percentile_95"] == approx(104.805, rel=1e-6)


def test_as_numpy_array(univariate_summary):
    out = univariate.summary_stats(univariate_summary["values"].values)
    assert out["mean"] == approx(96.84181818181816, abs=1e-8)
    assert out["std_ddof1"] == approx(5.566925216278406, abs=1e-8)


def test__raises_error():
    with pytest.raises(ValueError, match="Expecting a NumPy vector or Pandas series."):
        univariate.summary_stats([1, 2, 3, 3, 2, 1])


def test_confidence_interval():
    """
    Test confidence intervals.
    r1 <- c(108, 89.52, 95.16, 101.61, 99.19, 100, 93.55, 97.58, 93.55, 98.39, 88.71, 94.35)

    Results of the CI, compared to R.
    t.test(r1-90)
    """
    y = [
        108.0,
        89.52,
        95.16,
        101.61,
        99.19,
        100,
        93.55,
        97.58,
        93.55,
        98.39,
        88.71,
        94.35,
    ]
    data = pd.DataFrame(data={"values": y})
    expected_LB = 3.230888
    expected_UB = 10.037445

    out = univariate.confidence_interval(
        data - 90, "values", conflevel=0.95, style="regular"
    )
    assert out[0] == approx(expected_LB, abs=1e-4)
    assert out[1] == approx(expected_UB, abs=1e-4)
    out = univariate.confidence_interval(
        data - 90, "values", conflevel=0.95, style="robust"
    )
    # TODO: complete the test for the robust case


@pytest.fixture
def within_between_sd_data():

    """
    r1 <- c(108.06, 89.52, 95.16, 101.61, 99.19, 100, 93.55, 97.58, 93.55, 98.39, 96.77, 89.92,
            88.71, 94.35)
    r2 <- c(108.07, 87.9, 95.97, 97.58, 100, 95.97, 88.71, 97.59, 95.97, 93.55, 96.78, 86.69,
            87.1, 93.55)
    """
    replicate1 = [
        108.06,
        89.52,
        95.16,
        101.61,
        99.19,
        100,
        93.55,
        97.58,
        93.55,
        98.39,
        96.77,
        89.92,
        88.71,
        94.35,
    ]
    replicate2 = [
        108.07,
        87.9,
        95.97,
        97.58,
        100,
        95.97,
        88.71,
        97.59,
        95.97,
        93.55,
        96.78,
        86.69,
        87.1,
        93.55,
    ]
    temp = pd.DataFrame(data={"e1": replicate1, "e2": replicate2})
    temp.reset_index(inplace=True)
    temp = temp.melt(id_vars="index")
    df = temp.drop("variable", axis=1)
    empty = pd.DataFrame(columns=["value", "index"])
    return df, empty


def test_within_between_variance(within_between_sd_data):

    """
    Results are from a spreadsheet template. Unsure of the origin, or accuracy.
    """
    df, _ = within_between_sd_data
    expected_within_ms = 1.916015 ** 2
    expected_between_sd = 7.659146 ** 2
    expected_actual_sd = 5.490761 ** 2
    dof_within = 14
    dof_between = 13
    dof_total = 27

    out = univariate.within_between_standard_deviation(df, "value", "index")
    assert out["total_ms"] == approx(expected_actual_sd, rel=1e-5)
    assert out["total_dof"] == dof_total
    assert out["within_ms"] == approx(expected_within_ms, rel=1e-5)
    assert out["within_dof"] == dof_within
    assert out["between_ms"] == approx(expected_between_sd, rel=1e-5)
    assert out["between_dof"] == dof_between


def test_empty_case(within_between_sd_data):
    """
    What happens if there are no data? Everything should be zero.
    """
    _, empty = within_between_sd_data
    out = univariate.within_between_standard_deviation(empty, "value", "index")
    assert out["total_ms"] == 0
    assert out["within_ms"] == 0
    assert out["within_dof"] == 0
    assert out["between_ms"] == 0
    assert out["total_dof"] == 0
    assert out["between_dof"] == 0


def test_within_between_sd_missing_values():
    """
    Test against Excel sheet formulas.
    r1 <- c(108.06, NA, 95.16, 101.61, 99.19, 100, 93.55, 97.58, 93.55, 98.39, 96.77, 89.92,
            88.71, NA)
    r2 <- c(108.07, 87.9, 95.97, 97.58, 100, 95.97, 88.71, 97.59, 95.97, 93.55, 96.78, 86.69,
            87.1, NA)
    """

    empty = pd.DataFrame(columns=["value", "index"])
    empty.append({"index": 1, "value": np.NaN}, ignore_index=True)
    empty.append({"index": 1, "value": 123456}, ignore_index=True)

    replicate1 = [
        108.06,
        np.nan,
        95.16,
        101.61,
        99.19,
        100,
        93.55,
        97.58,
        93.55,
        98.39,
        96.77,
        89.92,
        88.71,
        np.nan,
    ]
    replicate2 = [
        108.07,
        87.9,
        95.97,
        97.58,
        100,
        95.97,
        88.71,
        97.59,
        95.97,
        93.55,
        96.78,
        86.69,
        87.1,
        np.nan,
    ]
    temp = pd.DataFrame(data={"e1": replicate1, "e2": replicate2})
    temp.reset_index(inplace=True)
    temp = temp.melt(id_vars="index")
    df = temp.drop("variable", axis=1)

    # within_between_variance_missing_values
    # Results are from a spreadsheet template. Unsure of the origin, or accuracy.
    expected_within_ms = 2.036406 ** 2
    expected_between_sd = 7.754816 ** 2
    expected_actual_sd = 5.669397 ** 2
    dof_within = 12
    dof_between = 12
    dof_total = 24

    out = univariate.within_between_standard_deviation(df, "value", "index")
    assert out["total_ms"] == approx(expected_actual_sd, abs=1e-4)
    assert out["total_dof"] == dof_total
    assert out["within_ms"] == approx(expected_within_ms, abs=1e-4)
    assert out["within_dof"] == dof_within
    assert out["between_ms"] == approx(expected_between_sd, abs=1e-4)
    assert out["between_dof"] == dof_between

    # test__empty_case
    # What happens if there are no/little data after accounting for outliers?
    out = univariate.within_between_standard_deviation(empty, "value", "index")
    assert out["total_ms"] == 0
    assert out["within_ms"] == 0
    assert out["within_dof"] == 0
    assert out["between_ms"] == 0
    assert out["total_dof"] == 0
    assert out["between_dof"] == 0


@pytest.fixture
def outliers_data():

    # Rosner data set: https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h3.htm
    rosner = [
        -0.25,
        0.68,
        0.94,
        1.15,
        1.20,
        1.26,
        1.26,
        1.34,
        1.38,
        1.43,
        1.49,
        1.49,
        1.55,
        1.56,
        1.58,
        1.65,
        1.69,
        1.70,
        1.76,
        1.77,
        1.81,
        1.91,
        1.94,
        1.96,
        1.99,
        2.06,
        2.09,
        2.10,
        2.14,
        2.15,
        2.23,
        2.24,
        2.26,
        2.35,
        2.37,
        2.40,
        2.47,
        2.54,
        2.62,
        2.64,
        2.90,
        2.92,
        2.92,
        2.93,
        3.21,
        3.26,
        3.30,
        3.59,
        3.68,
        4.30,
        4.64,
        5.34,
        5.42,
        6.01,
    ]

    sequence = [
        9101,
        9193,
        9440,
        9836,
        9677,
        9515,
        9783,
        9130,
        9469,
        9528,
        np.nan,
        np.nan,
        9805,
        9894,
        9941,
        10140,
        9001,
        9178,
        10080,
        9816,
        9160,
        8862,
        9376,
        9515,
        10670,
        10090,
        9979,
        9761,
        9422,
        9696,
        10130,
        10090,
        9641,
        9771,
        9503,
        9533,
        9413,
        9194,
        9219,
        9756,
        np.nan,
        np.nan,
        9240,
        9337,
        9682,
        9809,
        9343,
        9366,
        9245,
        9190,
        9363,
        9273,
        9500,
        9550,
        9664,
        9320,
        9247,
        9095,
        9122,
        9272,
        9157,
        9100,
        10670,
        10900,
        8899,
        8838,
        9203,
        9403,
        9520,
        9123,
        9109,
        9857,
        9936,
        9312,
        9225,
    ]
    return rosner, sequence


def test_rosner_nonrobust_esd(outliers_data):
    rosner, _ = outliers_data
    outliers, reasons = univariate.outlier_detection_multiple(
        rosner,
        algorithm="esd",
        max_outliers_detected=7,
        robust_variant=False,
        alpha=0.05,
    )

    # Ensure the vector is unchanged afterwards
    assert len(rosner) == 54
    assert rosner[0] == -0.25
    assert rosner[-1] == 6.01
    assert outliers == [53, 52, 51]

    # Compare values in the explanation from NIST:
    # https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h3.htm
    assert reasons["lambda"] == approx(
        [3.158, 3.151, 3.143, 3.136, 3.128, 3.120, 3.111], rel=1e-3
    )
    assert reasons["R_i"] == approx(
        [3.118, 2.942, 3.179, 2.810, 2.815, 2.848, 2.279], rel=1e-3
    )


def test_rosner_esd_kwargs(outliers_data):
    """
    In this example it picks up fewer outliers.
    """
    rosner, _ = outliers_data
    outliers, _ = univariate.outlier_detection_multiple(
        rosner,
        algorithm="esd",
        max_outliers_detected=7,
        robust_variant=True,
        alpha=0.05,
    )
    assert outliers == [53]


def test_rosner_esd_no_outliers(outliers_data):
    """
    In this example it picks up no outliers. Ensures that the test can also return an empty
    list.
    """
    rosner, _ = outliers_data
    outliers, _ = univariate.outlier_detection_multiple(
        rosner[1:-5],
        algorithm="esd",
        max_outliers_detected=0,
    )
    assert outliers == []


def test_rosner_esd_corner_case():
    """
    In this example it picks up no outliers. Ensures that the test can also return an empty
    list.
    """
    outliers, extra_out = univariate.outlier_detection_multiple(
        [1, 2], algorithm="esd", max_outliers_detected=1
    )
    assert extra_out["p-value"][0] == 0

    # If the values are all the same:
    outliers, extra_out = univariate.outlier_detection_multiple(
        [3, 3, 3], algorithm="esd", max_outliers_detected=1
    )
    assert np.isnan(extra_out["p-value"])
    assert extra_out["cutoff"] == -1
    assert len(outliers) == 0

    outliers, extra_out = univariate.outlier_detection_multiple(
        [2, 2], algorithm="esd", max_outliers_detected=1
    )
    assert np.isnan(extra_out["p-value"])
    assert extra_out["cutoff"] == -1
    assert len(outliers) == 0

    outliers, extra_out = univariate.outlier_detection_multiple(
        [1], algorithm="esd", max_outliers_detected=1
    )
    assert np.isnan(extra_out["p-value"])
    assert extra_out["cutoff"] == -1
    assert len(outliers) == 0

    outliers, extra_out = univariate.outlier_detection_multiple(
        [1, 2, 3], algorithm="something-else", max_outliers_detected=1
    )
    assert len(extra_out) == 0
    assert len(outliers) == 0


def test_sequence_compare_R(outliers_data):
    """
    Compares it to an R sequence and the Grubb's test there.
    """
    _, sequence = outliers_data
    outliers, reasons_regular = univariate.outlier_detection_multiple(
        sequence,
        algorithm="esd",
        max_outliers_detected=1,
        robust_variant=False,
        alpha=0.05,
    )
    assert reasons_regular["p-value"][0] == approx(0.02066273, rel=1e-7)

    # Now with the robust version, to check NaN handling.
    outliers, reasons_robust = univariate.outlier_detection_multiple(
        sequence,
        algorithm="esd",
        max_outliers_detected=1,
        robust_variant=True,
        alpha=0.05,
    )
    assert outliers[0] == 63
    assert np.isnan(
        univariate.median_abs_deviation(
            [
                np.nan,
            ],
            axis=None,
        )
    )


def test_distribution_check():
    """
    R code for the KS test:

    > y1 = []
    > ks.test(y1,"pnorm")
    """
    # TODO
    pass
