from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import process_improve.univariate.metrics as univariate


class TestTValues:
    """
    Checks the calculation of t values (at a given 'alpha' and with an integer number of degrees
    of freedom), against the values from R.
    """

    assert univariate.t_value(0, 1) == -np.inf
    assert univariate.t_value(1, 2) == np.inf
    assert univariate.t_value(0.5, 3) == pytest.approx(0, rel=1e-16)

    # Tested in R:  qt(0.9, 5) ->  1.475884
    assert univariate.t_value(0.9, 5) == pytest.approx(1.475884, rel=1e-6)


class TestTValuesCdf:
    """
    Checks the calculation of t values (at a given 'alpha' and with an integer number of degrees
    of freedom), against the values from R.
    """

    assert univariate.t_value_cdf(0, 1) == 0.5
    assert univariate.t_value_cdf(-np.inf, 2) == 0
    assert univariate.t_value_cdf(np.inf, 3) == 1

    # Tested in R:  pt(0.9, 5) ->  0.7953144
    assert univariate.t_value_cdf(0.9, 5) == pytest.approx(0.7953144, rel=1e-8)
    # Tested in R:  pt(0.5, 1) ->  0.6475836
    assert univariate.t_value_cdf(0.5, 1) == pytest.approx(0.6475836, rel=1e-7)


def test_normality_check() -> None:
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
    assert univariate.test_normality(x) == pytest.approx(0.4352, abs=1e-3)

    # Data actually are from a uniform distribution:
    assert univariate.test_normality(y) == pytest.approx(0.1117, abs=1e-3)


def test_univariate_robust_scale() -> None:
    """
    A scale estimator which is robust to outliers.

    Testing against R code [R version 3.6.0 (2019-04-26)]
    > library(robustbase)
    # All code has the default argument: "finite.corr=TRUE"
    > robustbase::Sn(c(0, 1))                   # 0.8861018
    > robustbase::Sn(c(0, 1, 2))                # 2.207503
    > robustbase::Sn(c(0, 1, 2, 3))             # 1.13774
    > robustbase::Sn(c(0, 1, 2, 3, 4))          # 1.611203
    > robustbase::Sn(c(0, 1, 2, 3, 4, 5))       # 2.368504
    > robustbase::Sn(c(0, 1, 2, 3, 4, 5, 6))    # 2.85747
    > robustbase::Sn(c(0, 1, 2, 3, 4, 50, 6))   # 2.85747
    > robustbase::Sn(c(0, 1, 20, 3, 4, 50, 6))  # 5.714939
    > robustbase::Sn(c(0, 10, 20, 3, 4, 50, 6)) # 8.572409
    > robustbase::Sn(seq(1, 10))                # 3.5778
    > robustbase::Sn(seq(1, 11))                # 3.896614
    > robustbase::Sn(seq(1, 19))                # 6.259503
    > robustbase::Sn(seq(1, 1500))              # 447.225

    TODO: found this weird sequence that gives Sn of zero, even though there is variability:
    99, 95, 95, 100, 100, 100, 100, 95, 100, 100, 100, 100, 105, 105, 100, 95, 105, 100, 95, 100
    How to make it robust to this weird situation?
    """

    # Every value above is reproduced, at both odd and even n. Sn uses the high
    # median inside and the low median outside (Rousseeuw-Croux); these are order
    # statistics that differ from the averaging median exactly when n is even, so
    # using np.median for both (the behaviour before this was fixed) understated
    # every even-n estimate: 0.443 against 0.886 at n = 2.

    # Odd n
    assert univariate.Sn(list(range(3))) == pytest.approx(2.207503, rel=1e-6)
    assert univariate.Sn(list(range(5))) == pytest.approx(1.611203, rel=1e-6)
    assert univariate.Sn(list(range(7))) == pytest.approx(2.85747, rel=1e-6)
    assert univariate.Sn([0, 1, 2, 3, 4, 50, 6]) == pytest.approx(2.85747, rel=1e-6)
    assert univariate.Sn([0, 1, 20, 3, 4, 50, 6]) == pytest.approx(5.714939, rel=1e-6)
    assert univariate.Sn([0, 10, 20, 3, 4, 50, 6]) == pytest.approx(8.572409, rel=1e-7)
    assert univariate.Sn(list(range(1, 12))) == pytest.approx(3.896614, rel=1e-6)
    assert univariate.Sn(list(range(1, 20))) == pytest.approx(6.259503, rel=1e-6)

    # Even n: these are the cases the previous implementation got wrong.
    assert univariate.Sn([0, 1]) == pytest.approx(0.8861018, rel=1e-6)
    assert univariate.Sn(list(range(4))) == pytest.approx(1.13774, rel=1e-6)
    assert univariate.Sn(list(range(6))) == pytest.approx(2.368504, rel=1e-6)
    assert univariate.Sn(list(range(1, 11))) == pytest.approx(3.5778, rel=1e-6)
    assert univariate.Sn(list(range(1, 1501))) == pytest.approx(447.225, rel=1e-6)
    # Corner cases:
    assert np.isnan(univariate.Sn([]))
    assert univariate.Sn([13]) == 0.0


def test_summary_stats_corner_case_with_robust_scale() -> None:
    """Test summary stats corner case where Sn is zero despite variability."""
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


def test_median_abs_deviation() -> None:
    """Test median absolute deviation against known values and scipy."""
    x = np.array([[10, 7, 4], [3, 2, 1]])
    assert univariate.median_absolute_deviation(x, scale=1) == pytest.approx([3.5, 2.5, 1.5])
    assert univariate.median_absolute_deviation(x.ravel(), scale=1) == 2.0

    x = stats.norm.rvs(size=1000000, scale=2, random_state=123456)
    assert univariate.median_absolute_deviation(x, scale=1) == pytest.approx(1.3487398527041636, rel=1e-12)
    assert univariate.median_absolute_deviation(x) == pytest.approx(1.9996446978061115, rel=1e-12)

    with pytest.raises(TypeError, match=r"The argument 'center' must .*"):
        _ = univariate.median_absolute_deviation(x, center=0.0)

    with pytest.raises(ValueError, match=r".* is not a valid scale value."):
        _ = univariate.median_absolute_deviation(x, scale="robust")

    with pytest.raises(ValueError, match=r"nan_policy must be one of .*"):
        univariate.median_absolute_deviation([1, 2, 3, 4], nan_policy="propogatess")

    with pytest.raises(TypeError):
        univariate.median_absolute_deviation(["a", "b"])

    with pytest.raises(ValueError, match="The input contains nan values"):
        univariate.median_absolute_deviation([np.nan, 1], nan_policy="raise")

    assert np.isnan(univariate.median_absolute_deviation([np.nan, 1, 2], nan_policy="propagate"))

    assert np.isnan(univariate.median_absolute_deviation([np.nan]))
    assert np.isnan(
        univariate.median_absolute_deviation(
            [
                np.nan,
            ],
            axis=0,
        )
    )
    assert np.isnan(univariate.median_absolute_deviation([]))
    assert np.isnan(univariate.median_absolute_deviation([], axis=0))
    assert np.isnan(univariate.median_absolute_deviation((np.empty((2, 3, 4)) * np.nan).ravel()))
    assert np.isnan(univariate.median_absolute_deviation(np.array([np.nan, np.nan]), axis=0))


def test_t_test_differences() -> None:
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
    temp = pd.DataFrame(data={"Sam": sam, "Jen": jen, "Mik": mik}).reset_index().melt(id_vars="index")
    df = temp.drop("index", axis=1).dropna().rename(columns={"variable": "Person"})
    output = univariate.ttest_independent_from_df(df, grouper_column="Person", values_column="value", conflevel=0.95)
    row = output[output["Group A name"].eq("Sam") & output["Group B name"].eq("Jen")]

    # Assert against the R values in the above validation script.
    # Note: in R, the test for t.test(A, B), checking A minus B.
    # We define our test as group B minus group A. Therefore we have to flip our signs,
    # and high/low values of the confidence interval
    assert row["Group A average"][0] == pytest.approx(8.216667, rel=1e-5)
    assert row["Group B average"][0] == pytest.approx(6.040833, rel=1e-5)
    assert row["z value"][0] == pytest.approx(-2.9906, rel=1e-4)
    assert row["p value"][0] == pytest.approx(0.00674, rel=1e-3)
    assert row["ConfInt: Lo"][0] == pytest.approx(-3.6846919, rel=1e-4)
    assert row["ConfInt: Hi"][0] == pytest.approx(-0.6669748, rel=1e-4)
    assert row["Degrees of freedom"][0] == pytest.approx(22, rel=1e-8)


def test_t_paried_test_differences() -> None:
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
    temp = pd.DataFrame(data={"Sam": sam, "Jen": jen}).reset_index().melt(id_vars="index")
    df = temp.drop("index", axis=1).dropna().rename(columns={"variable": "Person"})
    output = univariate.ttest_paired_from_df(df, grouper_column="Person", values_column="value", conflevel=0.95)
    row = output[output["Group A name"].eq("Sam") & output["Group B name"].eq("Jen")]

    # Assert against the R values in the above validation script.
    # Note: in R, the test for t.test(A, B), checking A minus B.
    assert row["Group A average"][0] == pytest.approx(8.216667, rel=1e-7)
    assert row["Group B average"][0] == pytest.approx(6.040833, rel=1e-7)
    assert row["Differences mean"][0] == pytest.approx(2.175833, rel=1e-6)
    assert row["z value"][0] == pytest.approx(2.8139, rel=1e-4)
    assert row["p value"][0] == pytest.approx(0.01685, rel=1e-4)
    assert row["ConfInt: Lo"][0] == pytest.approx(0.4739104, rel=1e-7)
    assert row["ConfInt: Hi"][0] == pytest.approx(3.8777563, rel=1e-7)
    assert row["Degrees of freedom"][0] == 11


@pytest.fixture
def univariate_summary() -> pd.DataFrame:
    """
    Provide a univariate case study.

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


def test_compare_to_r_with_without_missing(univariate_summary: pd.DataFrame) -> None:
    """Verify summary stats reproduce R results, with and without missing values."""
    # Verifies that we can reproduce results from R. R version 3.6.0 (2019-04-26)
    # Checked on 21 February 2020.
    data = univariate_summary
    for k in range(2):
        if k > 0:
            # For the second loop: add a missing value and ensure you get the same results
            # as for the first loop (without missing values)
            data = pd.concat([data, pd.DataFrame(data={"values": [np.nan]})])

        out = univariate.summary_stats(data["values"])
        assert out["mean"] == pytest.approx(96.84181818181816, abs=1e-8)
        assert out["std_ddof1"] == pytest.approx(5.566925216278406, abs=1e-8)
        assert out["rsd_classical"] == pytest.approx(0.05748472427301, abs=1e-8)
        assert out["median"] == pytest.approx(97.58, abs=1e-8)
        assert out["center"] == pytest.approx(97.58, abs=1e-8)  # center = median by default
        assert out["iqr"] == pytest.approx(6.045, abs=1e-8)
        assert out["spread"] == pytest.approx(6.28653702970298, abs=1e-8)  # spread = Sn (changed in 0.5)
        assert out["rsd"] == pytest.approx(0.06442444178831, abs=1e-8)
        assert out["min"] == 88.71
        assert out["max"] == 108
        assert out["N_non_missing"] == 11

        # Note: there are differences in how Numpy and R interpolate the results
        assert out["percentile_05"] == pytest.approx(89.115, rel=1e-6)
        assert out["percentile_25"] == pytest.approx(93.55, rel=1e-6)
        assert out["percentile_75"] == pytest.approx(99.595, rel=1e-6)
        assert out["percentile_95"] == pytest.approx(104.805, rel=1e-6)


def test_as_numpy_array(univariate_summary: pd.DataFrame) -> None:
    """Test summary stats when input is a NumPy array instead of Pandas Series."""
    out = univariate.summary_stats(univariate_summary["values"].values)
    assert out["mean"] == pytest.approx(96.84181818181816, abs=1e-8)
    assert out["std_ddof1"] == pytest.approx(5.566925216278406, abs=1e-8)


def test__raises_error() -> None:
    """Test that summary_stats raises TypeError for non-array input."""
    with pytest.raises(TypeError, match=r"Expecting a NumPy vector or Pandas series\."):
        univariate.summary_stats([1, 2, 3, 3, 2, 1])


def test_confidence_interval() -> None:
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

    out = univariate.confidence_interval(data - 90, "values", conflevel=0.95, style="regular")
    assert out[0] == pytest.approx(expected_LB, abs=1e-4)
    assert out[1] == pytest.approx(expected_UB, abs=1e-4)
    out = univariate.confidence_interval(data - 90, "values", conflevel=0.95, style="robust")
    # TODO: complete the test for the robust case


# ---------------------------------------------------------------------------
# SEC-24 (#273) -- degenerate-sample-size guards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [0, 1])
def test_confidence_interval_rejects_fewer_than_two_samples(n: int) -> None:
    """``confidence_interval`` requires n >= 2; SEC-24 (#273)."""
    data = pd.DataFrame({"x": [1.0] * n})
    with pytest.raises(ValueError, match="at least 2 non-missing values"):
        univariate.confidence_interval(data, "x", conflevel=0.95, style="regular")


def test_confidence_interval_rejects_all_nan_column() -> None:
    """SEC-24 (#273) -- all-NaN counts as zero non-missing observations."""
    data = pd.DataFrame({"x": [np.nan, np.nan, np.nan]})
    with pytest.raises(ValueError, match="at least 2 non-missing values"):
        univariate.confidence_interval(data, "x", conflevel=0.95, style="regular")


@pytest.mark.parametrize("differences", [[], [3.0]])
def test_ttest_paired_rejects_fewer_than_two_observations(differences: list[float]) -> None:
    """``ttest_paired`` needs at least 2 differences; SEC-24 (#273)."""
    diff_series = pd.Series(differences, dtype=float)
    with pytest.raises(ValueError, match="at least 2 paired observations"):
        univariate.ttest_paired(diff_series, conflevel=0.95)


@pytest.fixture
def within_between_sd_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Within-between standard deviation test data.

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
    temp = pd.DataFrame(data={"e1": replicate1, "e2": replicate2}).reset_index().melt(id_vars="index")
    df = temp.drop("variable", axis=1)
    empty = pd.DataFrame(columns=["value", "index"])
    return df, empty


def test_within_between_variance(within_between_sd_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """Results are from a spreadsheet template. Unsure of the origin, or accuracy."""
    df, _ = within_between_sd_data
    expected_within_ms = 1.916015**2
    expected_between_sd = 7.659146**2
    expected_actual_sd = 5.490761**2
    dof_within = 14
    dof_between = 13
    dof_total = 27

    out = univariate.variance_decomposition(df, "value", "index")
    assert out["total_ms"] == pytest.approx(expected_actual_sd, rel=1e-5)
    assert out["total_dof"] == dof_total
    assert out["within_ms"] == pytest.approx(expected_within_ms, rel=1e-5)
    assert out["within_dof"] == dof_within
    assert out["between_ms"] == pytest.approx(expected_between_sd, rel=1e-5)
    assert out["between_dof"] == dof_between


def test_empty_case(within_between_sd_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """What happens if there are no data? Everything should be zero."""
    _, empty = within_between_sd_data
    out = univariate.variance_decomposition(empty, "value", "index")
    assert out["total_ms"] == 0
    assert out["within_ms"] == 0
    assert out["within_dof"] == 0
    assert out["between_ms"] == 0
    assert out["total_dof"] == 0
    assert out["between_dof"] == 0


def test_within_between_sd_missing_values() -> None:
    """
    Test against Excel sheet formulas.

    r1 <- c(108.06, NA, 95.16, 101.61, 99.19, 100, 93.55, 97.58, 93.55, 98.39, 96.77, 89.92,
            88.71, NA)
    r2 <- c(108.07, 87.9, 95.97, 97.58, 100, 95.97, 88.71, 97.59, 95.97, 93.55, 96.78, 86.69,
            87.1, NA)
    """

    empty = pd.DataFrame(
        data=[{"index": 1, "value": np.nan}, {"index": 2, "value": 123456}],
        columns=["value", "index"],
    )

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
    df = (
        pd.DataFrame(data={"e1": replicate1, "e2": replicate2})
        .reset_index()
        .melt(id_vars="index")
        .drop("variable", axis=1)
    )

    # within_between_variance_missing_values
    # Results are from a spreadsheet template. Unsure of the origin, or accuracy.
    expected_within_ms = 2.036406**2
    expected_between_sd = 7.754816**2
    expected_actual_sd = 5.669397**2
    dof_within = 12
    dof_between = 12
    dof_total = 24

    out = univariate.variance_decomposition(df, "value", "index")
    assert out["total_ms"] == pytest.approx(expected_actual_sd, abs=1e-4)
    assert out["total_dof"] == dof_total
    assert out["within_ms"] == pytest.approx(expected_within_ms, abs=1e-4)
    assert out["within_dof"] == dof_within
    assert out["between_ms"] == pytest.approx(expected_between_sd, abs=1e-4)
    assert out["between_dof"] == dof_between

    # test__empty_case
    # What happens if there are no/little data after accounting for outliers?
    out = univariate.variance_decomposition(empty, "value", "index")
    assert out["total_ms"] == 0
    assert out["within_ms"] == 0
    assert out["within_dof"] == 0
    assert out["between_ms"] == 0
    assert out["total_dof"] == 0
    assert out["between_dof"] == 0


@pytest.fixture
def outliers_data_measurement() -> list[float]:
    """From an actual use-case."""
    return [
        10769.166,
        10043.447,
        10171.783,
        10751.362,
        10684.675,
        10772.250,
        10830.804,
    ]


def test_edge_case_outliers(outliers_data_measurement: list[float]) -> None:
    """Test an actual edge case that did not return p-values.

    Exercises the (opt-in) robust variant. Note how far the MAD-scaled
    statistic (R_1 = 7.16) sits above the classical critical value
    (lambda_1 = 2.02) - the mismatch that makes the robust variant
    anti-conservative and is the reason it is no longer the default.
    With the corrected largest-crossing rule both tested points are flagged.
    """
    max_outliers = len(outliers_data_measurement) - 5
    outliers, reasons = univariate.detect_outliers_esd(
        outliers_data_measurement,
        algorithm="esd",
        max_outliers_detected=max_outliers,
        alpha=0.05,
        robust_variant=True,
    )

    assert len(outliers) == 2
    assert reasons["lambda"][0] == pytest.approx(2.0199685076)
    assert reasons["R_i"][0] == pytest.approx(7.16003736)
    assert reasons["p-value"][0] == 0
    assert reasons["p-value"][1] == 0


@pytest.fixture
def outliers_data() -> tuple[list[float], list[float]]:
    """Rosner data set and a sequence for outlier detection tests."""
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


def test_rosner_nonrobust_esd(outliers_data: tuple[list[float], list[float]]) -> None:
    """Test Rosner non-robust ESD against NIST reference values."""
    rosner, _ = outliers_data
    outliers, reasons = univariate.detect_outliers_esd(
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
    assert reasons["lambda"] == pytest.approx([3.158, 3.151, 3.143, 3.136, 3.128, 3.120, 3.111], rel=1e-3)
    assert reasons["R_i"] == pytest.approx([3.118, 2.942, 3.179, 2.810, 2.815, 2.848, 2.279], rel=1e-3)


def test_rosner_esd_kwargs(outliers_data: tuple[list[float], list[float]]) -> None:
    """The (opt-in) robust variant over-detects on the Rosner data.

    NIST's answer for this data set is 3 outliers; the MAD-scaled statistic
    against classical critical values flags 4. This pins the documented
    anti-conservative behaviour of ``robust_variant=True`` (the reason it is
    no longer the default), with the corrected largest-crossing rule.
    """
    rosner, _ = outliers_data
    outliers, _ = univariate.detect_outliers_esd(
        rosner,
        algorithm="esd",
        max_outliers_detected=7,
        robust_variant=True,
        alpha=0.05,
    )
    assert outliers == [53, 52, 51, 50]


def test_rosner_esd_no_outliers(outliers_data: tuple[list[float], list[float]]) -> None:
    """
    In this example it picks up no outliers. Ensures that the test can also return an empty
    list.
    """
    rosner, _ = outliers_data
    outliers, _ = univariate.detect_outliers_esd(
        rosner[1:-5],
        algorithm="esd",
        max_outliers_detected=0,
    )
    assert outliers == []


def test_ttest_independent_correctly_named_fields() -> None:
    """The correctly named fields hold what their names say.

    Regression test: `z value` is a t-statistic (its p value comes from the t
    distribution) and `Pooled standard deviation` held the standard ERROR of
    the difference, not the pooled standard deviation. Both are kept as
    deprecated aliases alongside correctly named entries.
    """
    a = pd.Series([102.0, 98, 100, 101, 97])
    b = pd.Series([110.0, 112, 108, 109])
    out = univariate.ttest_independent(a, b)

    n_a, n_b = len(a), len(b)
    dof = n_a + n_b - 2
    svar = ((n_a - 1) * a.var(ddof=1) + (n_b - 1) * b.var(ddof=1)) / dof
    se_difference = np.sqrt(svar * (1 / n_a + 1 / n_b))

    # The new names are accurate.
    assert out["Std error of difference"] == pytest.approx(se_difference, rel=1e-12)
    assert out["Pooled std dev"] == pytest.approx(np.sqrt(svar), rel=1e-12)
    assert out["t value"] == pytest.approx((b.mean() - a.mean()) / se_difference, rel=1e-12)

    # The statistic really is a t-statistic: it matches scipy's t-test, and the
    # pooled standard deviation is a genuinely different number from the
    # standard error that the old key held.
    assert out["t value"] == pytest.approx(stats.ttest_ind(b, a).statistic, rel=1e-10)
    assert out["Pooled std dev"] != pytest.approx(out["Std error of difference"], rel=1e-3)

    # Deprecated aliases still resolve to the same values, for one more cycle.
    assert out["z value"] == out["t value"]
    assert out["Pooled standard deviation"] == out["Std error of difference"]


def test_ttest_from_df_skips_groups_with_no_usable_data() -> None:
    """An all-NaN group is dropped instead of yielding a silent NaN row.

    Regression test: the group list came from the un-dropna-ed frame, so a
    group whose values are all missing (or a NaN group label) survived and was
    compared against an empty sample, producing NaN statistics with no warning.
    """
    df = pd.DataFrame({"g": ["A"] * 3 + ["B"] * 3 + ["C"] * 3, "v": [1.0, 2, 3] + [np.nan] * 3 + [7.0, 8, 9]})
    out = univariate.ttest_independent_from_df(df, "g", "v")
    assert list(zip(out["Group A name"], out["Group B name"], strict=True)) == [("A", "C")]
    assert not out["p value"].isna().any()

    # A NaN group label is likewise not a group.
    df_nan_label = pd.DataFrame(
        {"g": ["A", "A", "A", np.nan, np.nan, np.nan, "C", "C", "C"], "v": [1.0, 2, 3, 4, 5, 6, 7, 8, 9]}
    )
    out2 = univariate.ttest_independent_from_df(df_nan_label, "g", "v")
    assert list(zip(out2["Group A name"], out2["Group B name"], strict=True)) == [("A", "C")]


def test_ttest_from_df_multiplicity_correction() -> None:
    """The pairwise family can be corrected for multiplicity on request."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "g": ["A"] * 6 + ["B"] * 6 + ["C"] * 6 + ["D"] * 6,
            "v": np.concatenate(
                [rng.normal(0, 1, 6), rng.normal(0.5, 1, 6), rng.normal(3, 1, 6), rng.normal(0.2, 1, 6)]
            ),
        }
    )

    # Default is unchanged: raw, uncorrected p-values and no extra columns.
    raw = univariate.ttest_independent_from_df(df, "g", "v")
    assert "p value (adjusted)" not in raw.columns

    holm = univariate.ttest_independent_from_df(df, "g", "v", correction="holm")
    assert len(holm) == 6  # 4 groups -> 4*3/2 pairwise tests
    # Holm adjustment never decreases a p-value, and the raw column is kept.
    assert np.all(holm["p value (adjusted)"].to_numpy() >= holm["p value"].to_numpy() - 1e-12)
    assert holm["p value"].to_numpy() == pytest.approx(raw["p value"].to_numpy(), rel=1e-12)
    assert holm["reject"].dtype == bool

    # Benjamini-Hochberg controls the FDR, so it is no more conservative than Holm.
    bh = univariate.ttest_independent_from_df(df, "g", "v", correction="bh")
    assert np.all(bh["p value (adjusted)"].to_numpy() <= holm["p value (adjusted)"].to_numpy() + 1e-12)

    with pytest.raises(ValueError, match="correction must be one of"):
        univariate.ttest_independent_from_df(df, "g", "v", correction="bogus")


def test_summary_stats_rsd_with_zero_centre() -> None:
    """A zero centre gives NaN rather than inf plus a RuntimeWarning.

    Regression test: both relative-spread divisions were unguarded, so a
    mean-centred variable produced inf and a RuntimeWarning through the public
    API and the agent-facing tool.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = univariate.summary_stats(np.array([-1.0, 0.0, 1.0]))
        assert np.isnan(out["rsd"])
        assert np.isnan(out["rsd_classical"])

        all_zero = univariate.summary_stats(np.array([0.0, 0.0, 0.0]))
        assert np.isnan(all_zero["rsd"])
        assert np.isnan(all_zero["rsd_classical"])

    # A non-zero centre is unaffected.
    normal = univariate.summary_stats(np.array([10.0, 11.0, 12.0]))
    assert normal["rsd_classical"] == pytest.approx(np.std([10.0, 11, 12], ddof=1) / 11.0, rel=1e-12)


def test_rosner_esd_corner_case() -> None:
    """Degenerate inputs return an empty list rather than a spurious outlier."""
    # All values identical: the spread is zero, so there is no candidate.
    outliers, extra_out = univariate.detect_outliers_esd([3, 3, 3], algorithm="esd", max_outliers_detected=1)
    assert np.isnan(extra_out["p-value"])
    assert extra_out["cutoff"] == -1
    assert len(outliers) == 0

    # A clean sample large enough to test: nothing is flagged.
    outliers, extra_out = univariate.detect_outliers_esd([1, 2, 3, 4], algorithm="esd", max_outliers_detected=1)
    assert extra_out["cutoff"] == -1
    assert len(outliers) == 0

    outliers, extra_out = univariate.detect_outliers_esd([1, 2, 3], algorithm="something-else", max_outliers_detected=1)
    assert len(extra_out) == 0
    assert len(outliers) == 0


def test_rosner_esd_rejects_samples_too_small_to_test() -> None:
    """Testing r outliers needs N >= r + 2, since the ESD dof is N - i - 1.

    Regression test: these previously ran with dof <= 0, so `t.ppf` returned
    NaN, every critical value was NaN, no crossing could ever be found, and the
    empty result read as "no outliers" rather than "not testable".
    """
    for sample in ([1, 2], [2, 2], [1]):
        with pytest.raises(ValueError, match="cannot exceed the sample size minus 2"):
            univariate.detect_outliers_esd(sample, algorithm="esd", max_outliers_detected=1)

    # NaN entries do not count towards the usable sample size.
    with pytest.raises(ValueError, match="cannot exceed the sample size minus 2"):
        univariate.detect_outliers_esd([1.0, 2.0, np.nan, np.nan], algorithm="esd", max_outliers_detected=2)

    # Asking for zero outliers is always allowed.
    outliers, _ = univariate.detect_outliers_esd([1, 2], algorithm="esd", max_outliers_detected=0)
    assert outliers == []


def test_rosner_esd_rejects_non_positive_alpha() -> None:
    """The alpha level must lie in (0, 1]; only the upper bound was checked before."""
    for bad_alpha in (0.0, -0.5):
        with pytest.raises(ValueError, match="alpha must lie in"):
            univariate.detect_outliers_esd([1, 2, 3, 4, 50], algorithm="esd", max_outliers_detected=1, alpha=bad_alpha)
    with pytest.raises(ValueError, match="alpha must lie in"):
        univariate.detect_outliers_esd([1, 2, 3, 4, 50], algorithm="esd", max_outliers_detected=1, alpha=1.5)


def test_rosner_esd_handles_infinite_values() -> None:
    """An infinity no longer raises KeyError from the -1 index sentinel.

    Regression test: a NaN p-value set `R_i_idx = -1`, which is not a label in
    the reset RangeIndex, so the subsequent drop raised KeyError whenever the
    spread was non-zero.
    """
    for sample in ([1.0, 2.0, 3.0, np.inf], [1.0, 2.0, np.inf, 4.0, 5.0]):
        outliers, extra_out = univariate.detect_outliers_esd(sample, algorithm="esd", max_outliers_detected=1)
        assert isinstance(outliers, list)
        assert -1 not in extra_out["R_i_idx"]


def test_rosner_esd_pvalue_uses_the_current_sample_size() -> None:
    """The Grubbs p-value describes the sample in hand, not the original one.

    Regression test: N was computed once and never refreshed, so from the
    second iteration onwards the reported p-value was computed for a larger
    sample than the statistic actually came from.
    """
    rng = np.random.default_rng(0)
    sample = [*rng.normal(size=20).tolist(), 12.0, 15.0]
    _, reasons = univariate.detect_outliers_esd(sample, algorithm="esd", max_outliers_detected=4, alpha=0.05)

    # Recompute iteration 2 by hand on the sample that remains after the first
    # point is dropped, and confirm the reported value matches it.
    remaining = pd.Series(sample).drop(pd.Series(sample).sub(np.mean(sample)).abs().idxmax()).reset_index(drop=True)
    n_2 = len(remaining)
    g_2 = ((remaining - remaining.mean()) / remaining.std()).abs().max()
    s_2 = g_2**2 * n_2 * (2 - n_2) / (g_2**2 * n_2 - (n_2 - 1) ** 2)
    expected = 0 if s_2 <= 0 else min(n_2 * (1 - univariate.t_value_cdf(np.sqrt(s_2), n_2 - 2)), 1)
    assert reasons["p-value"][1] == pytest.approx(expected, rel=1e-10)


def test_sequence_compare_r(outliers_data: tuple[list[float], list[float]]) -> None:
    """Compare it to an R sequence and the Grubb's test there."""
    _, sequence = outliers_data
    outliers, reasons_regular = univariate.detect_outliers_esd(
        sequence,
        algorithm="esd",
        max_outliers_detected=1,
        robust_variant=False,
        alpha=0.05,
    )
    assert reasons_regular["p-value"][0] == pytest.approx(0.02066273, rel=1e-7)

    # Now with the robust version, to check NaN handling.
    outliers, _reasons_robust = univariate.detect_outliers_esd(
        sequence,
        algorithm="esd",
        max_outliers_detected=1,
        robust_variant=True,
        alpha=0.05,
    )
    assert outliers[0] == 63
    assert np.isnan(
        univariate.median_absolute_deviation(
            [np.nan],
        )
    )


def test_distribution_check() -> None:
    """
    R code for the KS test.

    > y1 = []
    > ks.test(y1,"pnorm")
    """
    # TODO


def test_biweight_midvariance_robust_to_outliers() -> None:
    """The Mosteller-Tukey robust scale is barely affected by gross outliers."""
    rng = np.random.default_rng(0)
    clean = rng.normal(loc=10, scale=2, size=200)
    contaminated = np.concatenate([clean, [1000.0, -1000.0]])

    bw_clean = univariate.biweight_midvariance(clean)
    bw_contaminated = univariate.biweight_midvariance(contaminated)

    # The classical variance explodes with the outliers; the robust scale barely moves.
    assert np.var(contaminated, ddof=1) > 100 * np.var(clean, ddof=1)
    assert abs(bw_contaminated - bw_clean) / bw_clean < 0.1
    # For clean normal data with sigma=2 it tracks the true variance (~4).
    assert 2.0 < bw_clean < 8.0


def test_biweight_midvariance_edge_cases() -> None:
    """Constant and empty samples are handled gracefully."""
    assert univariate.biweight_midvariance([5.0, 5.0, 5.0]) == 0.0
    assert np.isnan(univariate.biweight_midvariance([]))
    assert np.isnan(univariate.biweight_midvariance([1.0, np.nan], nan_policy="propagate"))
    # The default "omit" policy drops NaNs and computes on the remainder.
    rng = np.random.default_rng(11)
    clean = rng.normal(loc=0.0, scale=1.0, size=100)
    with_nan = np.concatenate([clean, [np.nan, np.nan]])
    omitted = univariate.biweight_midvariance(with_nan)
    assert np.isfinite(omitted)
    assert omitted == pytest.approx(univariate.biweight_midvariance(clean), rel=1e-12)


def test_holm_bonferroni_matches_statsmodels() -> None:
    """holm_bonferroni reproduces statsmodels' Holm correction."""
    from statsmodels.stats.multitest import multipletests

    p = np.array([0.001, 0.04, 0.03, 0.2, 0.009])
    result = univariate.holm_bonferroni(p, alpha=0.05)
    reject_ref, p_adj_ref, _, _ = multipletests(p, alpha=0.05, method="holm")

    np.testing.assert_allclose(result.p_adjusted, p_adj_ref, rtol=1e-12)
    np.testing.assert_array_equal(result.reject, reject_ref)


def test_holm_bonferroni_empty_input() -> None:
    """An empty set of p-values yields empty results."""
    result = univariate.holm_bonferroni([])
    assert result.p_adjusted.size == 0
    assert result.reject.size == 0


def test_tietjen_moore_detects_planted_outliers() -> None:
    """The Tietjen-Moore test flags two planted gross outliers."""
    rng = np.random.default_rng(0)
    sample = rng.normal(loc=0.0, scale=1.0, size=40)
    sample[5] = 12.0
    sample[20] = -11.0

    result = univariate.tietjen_moore_test(sample, n_outliers=2, n_simulations=2000, random_state=1)
    assert result.reject is True
    assert set(result.outlier_indices.tolist()) == {5, 20}


def test_tietjen_moore_no_outliers_in_clean_data() -> None:
    """Clean normal data should not be flagged as containing outliers."""
    rng = np.random.default_rng(2)
    sample = rng.normal(size=50)
    result = univariate.tietjen_moore_test(sample, n_outliers=2, n_simulations=2000, random_state=3)
    assert result.reject is False

    with pytest.raises(ValueError, match="n_outliers"):
        univariate.tietjen_moore_test(sample, n_outliers=50)


def test_distribution_fit_normal_and_non_normal() -> None:
    """distribution_fit accepts genuinely normal data and rejects skewed data."""
    rng = np.random.default_rng(4)
    normal_sample = rng.normal(loc=5.0, scale=2.0, size=500)
    fit_normal = univariate.distribution_fit(normal_sample, distribution="norm")
    assert fit_normal.distribution == "norm"
    assert fit_normal.n == 500
    assert fit_normal.fits_well is True

    exponential_sample = rng.exponential(scale=3.0, size=500)
    fit_bad = univariate.distribution_fit(exponential_sample, distribution="norm")
    assert fit_bad.fits_well is False
