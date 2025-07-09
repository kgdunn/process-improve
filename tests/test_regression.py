import numpy as np
import pandas as pd
import pytest

from process_improve.regression.methods import (
    multiple_linear_regression,
    repeated_median_slope,
    simple_robust_regression,
)


@pytest.fixture
def repeated_median() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fixture for the repeated median slope calculation."""
    x = np.array([0, 1, 2, 3])
    y = np.array([5, 1, 6, 72])

    # TODO: add a test where the inputs are Pandas Series. Should handle this case also.
    # TODO: handle cases where there are nans in the vectors

    divzero_x = np.array([2, 2, 3, 4])
    divzero_y = np.array([0, 1, 2, 3])
    return x, y, divzero_x, divzero_y


def test__repeated_median(repeated_median: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
    """
    Test the repeated median slope calculation.

    (1-5)/1 ; (6-5)/2 ; (72-5)/3   => 0.5
    (1-5)/-1 ; (6-1)/1 ; (72-1)/3   => 5
    (5-6)/-2 ; (1-6)/1 ; (72-6)/2   => 5
    (5-72)/-3 ; (1-72)/-2 ; (6-72)/1   => something (23.67)
    -------
    overall median of (0.5, 5, 5 and something) is (5+5)/2 = 5
    """
    x, y, *_ = repeated_median
    assert repeated_median_slope(x, y) == 5.0


def test__repeated_median_catch_division_by_zero(
    repeated_median: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Ensure division by zero does not crash the algorithm."""
    *_, divzero_x, divzero_y = repeated_median
    assert repeated_median_slope(divzero_x, divzero_y) == 1.0


@pytest.fixture
def multiple_linear_regression_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Verify the linear regression calculation matches the output from R.

    > x = c(0.019847603, 0.039695205, 0.059542808, 0.07939041, 0.099238013,
            0.119085616, 0.138933218)
    > y = c(0.2, 0.195089996, 0.284090012, 0.37808001, 0.46638, 0.561559975,
            0.652559996)
    > summary(lm(y~x))
    Call:
    lm(formula = y ~ x)
    Residuals:
            1         2         3         4         5         6         7
    0.052417 -0.033668 -0.025843 -0.013029 -0.005904  0.008101  0.017925
    Coefficients:
                Estimate Std. Error t value Pr(>|t|)
    (Intercept)  0.06641    0.02710   2.451   0.0579 .
    x            4.08993    0.30530  13.396 4.15e-05 ***
    ---
    Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    Residual standard error: 0.03206 on 5 degrees of freedom
    Multiple R-squared:  0.9729, Adjusted R-squared:  0.9675
    F-statistic: 179.5 on 1 and 5 DF,  p-value: 4.148e-05
    > confint(lm(y~x))
                    2.5 %    97.5 %
    (Intercept) -0.003253309 0.1360676
    x            3.305124756 4.8747402
    > max(residuals(lm(y~x)))
    [1] 0.05241749
    > min(residuals(lm(y~x)))
    [1] -0.03366786

    >hatvalues(lm(y~x))
    0.4642857 0.2857143 0.1785714 0.1428571 0.1785714 0.2857143 0.4642857

    > cooks.distance(model)
    1           2           3           4           5           6           7
    2.161740342 0.308710432 0.085960163 0.016051677 0.004486278 0.017871649 0.252805729

    > predict(lm(y~x), data.frame(x=c(0.019847603,0.138933218)), interval="predict")
            fit        lwr       upr
    1 0.1475825 0.04784391 0.2473211
    2 0.6346346 0.53489603 0.7343732
    """
    X = np.array(
        [
            0.019847603,
            0.039695205,
            0.059542808,
            0.07939041,
            0.099238013,
            0.119085616,
            0.138933218,
        ]
    ).reshape(7, 1)
    y = np.array([0.2, 0.195089996, 0.284090012, 0.37808001, 0.46638, 0.561559975, 0.652559996])
    return X, y


def test_inconsistent_sizes(multiple_linear_regression_data: tuple[np.ndarray, np.ndarray]) -> None:
    """Verifies that inconsistencies are picked up."""
    # TODO: X = 5 x 2, y = 7 x 1
    # TODO: X = 5 x 4, y = 5 x 1:  n=5, k=5: should work
    # TODO: X = 5 x 5, y = 5 x 1:  n=5, k=5+1 (with intercept): should fail
    X, y = multiple_linear_regression_data


def test_regression_model_with_intercept(multiple_linear_regression_data: tuple[np.ndarray, np.ndarray]) -> None:
    """Can we reproduce the R output for single column X.

    * model output
    * confidence intervals
    """
    X, y = multiple_linear_regression_data
    out = multiple_linear_regression(X, y, fit_intercept=True, na_rm=False)
    assert out["N"] == 7
    assert isinstance(out["intercept"], float)
    assert len(out["coefficients"]) == 1
    assert out["conf_intervals"].shape == (1, 2)
    assert out["conf_interval_intercept"].shape == (2,)

    assert out["SE"] == pytest.approx(0.03206, abs=1e-5)
    assert out["R2"] == pytest.approx(0.9729, rel=1e-5)
    assert out["intercept"] == pytest.approx(0.06641, abs=1e-5)
    assert out["coefficients"][0] == pytest.approx(4.08993, rel=1e-6)
    assert out["conf_intervals"][0] == pytest.approx([3.30512, 4.87474], rel=1e-5)
    assert out["conf_interval_intercept"] == pytest.approx([-0.003253309, 0.1360676], rel=1e-6)
    assert np.max(out["residuals"]) == pytest.approx(0.05241749, rel=1e-7)
    assert np.min(out["residuals"]) == pytest.approx(-0.03366786, rel=1e-7)

    # Residuals:
    #        1         2         3         4         5         6         7
    # 0.052417 -0.033668 -0.025843 -0.013029 -0.005904  0.008101  0.017925
    assert out["residuals"][0] == pytest.approx(0.052417, abs=1e-6)
    assert out["residuals"][1] == pytest.approx(-0.033668, abs=1e-6)
    assert out["residuals"][5] == pytest.approx(0.008101, abs=1e-6)
    assert out["residuals"][6] == pytest.approx(0.017925, abs=1e-6)

    # Hat values:
    # 0.4642857 0.2857143 0.1785714 0.1428571 0.1785714 0.2857143 0.4642857
    assert out["leverage"][0] == pytest.approx(0.4642857, abs=1e-6)
    assert out["leverage"][1] == pytest.approx(0.2857143, abs=1e-6)
    assert out["leverage"][2] == pytest.approx(0.1785714, abs=1e-6)
    assert out["leverage"][3] == pytest.approx(0.1428571, abs=1e-6)

    # Influence (Cook's D):
    # 2.161740342 0.308710432 0.085960163 0.016051677 0.004486278 0.017871649 0.252805729
    assert out["influence"][0] == pytest.approx(2.161740342, abs=1e-6)
    assert out["influence"][1] == pytest.approx(0.308710432, abs=1e-6)
    assert out["influence"][2] == pytest.approx(0.085960163, abs=1e-6)
    assert out["influence"][6] == pytest.approx(0.252805729, abs=1e-6)

    # Prediction interval at the extremes
    assert out["pi_range"][0, 1] == pytest.approx(0.04784391, abs=1e-8)
    assert out["pi_range"][0, 2] == pytest.approx(0.2473211, abs=1e-8)
    assert out["pi_range"][-1, 1] == pytest.approx(0.53489603, abs=1e-8)
    assert out["pi_range"][-1, 2] == pytest.approx(0.7343732, abs=1e-7)


def test__regression_model_no_intercept(multiple_linear_regression_data: tuple[np.ndarray, np.ndarray]) -> None:
    """Can  == pytest.approx( reproduce the R output for single rel 1E-9.

    > summary(lm(y~x+0))
    Call:
    lm(formula = y ~ x + 0)
    Residuals:
        Min        1Q    Median        3Q       Max
    -0.008637 -0.005542  0.000253  0.003448  0.105543
    Coefficients:
    Estimate Std. Error t value Pr(>|t|)
    x   4.7591     0.1849   25.74 2.27e-07 ***
    ---
    Signif. codes:  0 �***� 0.001 �**� 0.01 �*� 0.05 �.� 0.1 � � 1
    Residual standard error: 0.04343 on 6 degrees of freedom
    Multiple R-squared:  0.991, Adjusted R-squared:  0.9895
    F-statistic: 662.4 on 1 and 6 DF,  p-value: 2.268e-07
    > confint(lm(y~x+0))
        2.5 %  97.5 %
    x 4.306636 5.21157
    """
    X, y = multiple_linear_regression_data
    out = multiple_linear_regression(X, y, fit_intercept=False)

    assert len(out["coefficients"]) == 1
    assert out["conf_intervals"].shape == (1, 2)
    assert np.isnan(out["intercept"])

    assert out["SE"] == pytest.approx(0.04343, abs=1e-5)
    assert out["R2"] == pytest.approx(0.991, abs=1e-4)
    assert out["coefficients"][0] == pytest.approx(4.7591, rel=1e-6)
    assert out["conf_intervals"][0] == pytest.approx([4.306636, 5.21157], rel=1e-6)
    assert np.max(out["residuals"]) == pytest.approx(0.105543, abs=1e-6)
    assert np.min(out["residuals"]) == pytest.approx(-0.008637, abs=1e-6)


def test__regression_model_missing_values() -> None:
    """Test the regression model with missing values."""
    X, y = np.array([1, 2, 3, 4, 5]), np.array([2, np.nan, 4, np.nan, 9])
    out = multiple_linear_regression(X, y, na_rm=True, fit_intercept=True)
    assert out["intercept"] == pytest.approx(-0.25)
    assert out["coefficients"][0] == pytest.approx(1.75)
    assert out["SE"] == pytest.approx(1.225, abs=1e-3)
    assert out["R2"] == pytest.approx(0.9423, abs=1e-4)
    assert len(out["residuals"]) == 5  # not 3!
    assert out["residuals"] == pytest.approx([0.5, np.nan, -1.0, np.nan, 0.5], rel=1e-6, nan_ok=True)
    assert np.isnan(out["residuals"][1])
    assert np.isnan(out["residuals"][3])


def test_input_pandas(multiple_linear_regression_data: tuple[np.ndarray, np.ndarray]) -> None:
    """
    Pandas and Numpy inputs should be acceptable. Replicate a test used above, but with
    Pandas as the inputs.
    """
    x_, y_ = multiple_linear_regression_data
    x = pd.DataFrame(x_)
    y = pd.DataFrame(y_)
    out = multiple_linear_regression(x, y)
    assert out["SE"] == pytest.approx(0.03206, abs=1e-5)


def test_input_transposed_vector(multiple_linear_regression_data: tuple[np.ndarray, np.ndarray]) -> None:
    """
    Make the vector a column vector first.
    Also use a mix of Pandas (y) and Numpy (x).
    """
    X, y_ = multiple_linear_regression_data
    x = X.copy().T
    y = pd.DataFrame(y_)

    # There is a difference with a transposed array
    with pytest.raises(AssertionError, match=r"N >= K: You need at least as many rows .*"):
        _ = multiple_linear_regression(x, y)


def test_input_one_data_point() -> None:
    """Cannot work: fit a line with 1 datapoint."""
    x = np.array([2])
    y = np.array([-5])
    out = multiple_linear_regression(x, y)
    assert out["N"] is None
    assert np.isnan(out["coefficients"])
    assert np.isnan(out["residuals"])


@pytest.fixture
def simple_robust_regression_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Check the simple robust regression model.

    Validated against a data set where the values were calculated by hand/Excel. Can we reproduce the manually
    calculated output?

    # Check against R as well (results from the robust output below all agree well.)
    ---------------------------
    R version 3.6.0 (2019-04-26) -- "Planting of a Tree"
    Copyright (C) 2019 The R Foundation for Statistical Computing
    Platform: x86_64-w64-mingw32/x64 (64-bit)

    > x <- c(0.0599125391, 0.0998562289, 0.1397959245, 0.1797268338, 0.2496405722)
    > y <- c(0.2788299918, 0.4663000107, 0.6585199833, 0.8372399807, 1.1684000492)
    > model <- lm(y ~  x)
    > summary(model)

    Call:
    lm(formula = y ~ x)

    Residuals:
             1          2          3          4          5
    -0.0010087 -0.0005354  0.0047065 -0.0035104  0.0003479

    Coefficients:
                Estimate Std. Error t value Pr(>|t|)
    (Intercept) -0.0006424  0.0037744   -0.17    0.876
    x            4.6815091  0.0236153  198.24 2.83e-07
    ---
    Residual standard error: 0.003459 on 3 degrees of freedom
    Multiple R-squared:  0.9999, Adjusted R-squared:  0.9999
    F-statistic: 3.93e+04 on 1 and 3 DF,  p-value: 2.83e-07

    > confint(model)
                    2.5 %     97.5 %
    (Intercept) -0.01265427 0.01136939
    x            4.60635459 4.75666351

    """
    X = np.array(
        [
            0.0599125391,
            0.0998562289,
            0.1397959245,
            0.1797268338,
            0.2496405722,
        ]
    ).reshape(5, 1)
    y = np.array(
        [
            0.2788299918,
            0.4663000107,
            0.6585199833,
            0.8372399807,
            1.1684000492,
        ]
    )
    return X, y


def test_regression_simple_robust(simple_robust_regression_data: tuple[np.ndarray, np.ndarray]) -> None:
    """Tests the simple robust regression model."""
    X, y = simple_robust_regression_data
    out = simple_robust_regression(X, y)

    assert isinstance(out["intercept"], float)
    assert len(out["coefficients"]) == 1
    assert out["conf_interval_intercept"].shape == (2,)
    assert out["conf_intervals"].shape == (1, 2)

    assert out["SE"] == pytest.approx(0.003554009, abs=1e-10)
    assert out["R2"] == pytest.approx(0.999919429560, rel=1e-6)
    assert out["intercept"] == pytest.approx(-0.00218, abs=1e-5)
    assert out["coefficients"][0] == pytest.approx(4.69038, rel=1e-6)
    assert out["conf_intervals"][0] == pytest.approx([4.61316875, 4.76759488], rel=1e-8)
    assert out["conf_interval_intercept"] == pytest.approx([-0.0145235437, 0.0101581595], rel=1e-8)
    assert out["residuals"][0:5] == pytest.approx([0.0, 0.000118863, 0.005006413, -0.0035648, -0.000326859], abs=1e-9)


def test_simple_robust_regression_corner_case() -> None:
    """Tests some corner cases."""
    # No variation in x-space
    x = np.array([4, 4, 4, 4, 4])
    y = np.array([1, 2, 3, 4, 5])
    out = simple_robust_regression(x, y)
    assert np.isnan(out["standard_error_intercept"])
    assert np.isnan(out["standard_errors"][0])
    assert np.isnan(out["conf_intervals"][0][0])
    assert np.isnan(out["conf_intervals"][0][1])


def test_simple_robust_regression_missing_values() -> None:
    """Test y length less than 2 because of nan values."""

    x = np.array([1, 2, 3, 4, 5])
    y = np.array([np.nan, np.nan, np.nan, np.nan, 1])
    out = simple_robust_regression(x, y)
    assert np.isnan(out["standard_error_intercept"])
    assert np.isnan(out["standard_errors"][0])
    assert np.isnan(out["conf_intervals"][0][0])
    assert np.isnan(out["conf_intervals"][0][1])


def test_simple_regression_no_error() -> None:
    """Tests cases where there is perfect fit."""
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([9, 8, 7, 6, 5])
    robust = simple_robust_regression(x, y)
    regular = multiple_linear_regression(x, y, fit_intercept=True)
    assert robust["influence"] == pytest.approx(regular["influence"])
    assert robust["influence"] == pytest.approx([0] * 5)
    assert robust["conf_intervals"][0] == pytest.approx(regular["conf_intervals"][0])
    assert robust["coefficients"] == pytest.approx(regular["coefficients"])


def test_simple_robust_regression_no_intercept(simple_robust_regression_data: tuple[np.ndarray, np.ndarray]) -> None:
    """Test simple robust regression with fit_intercept=False."""
    X, y = simple_robust_regression_data
    out = simple_robust_regression(X.ravel(), y, fit_intercept=False)

    # Basic structure checks
    assert out["intercept"] == 0.0  # Should be exactly 0.0, not None or np.nan
    assert len(out["coefficients"]) == 1
    assert out["conf_intervals"].shape == (1, 2)
    assert np.isnan(out["standard_error_intercept"])
    assert np.array_equal(out["conf_interval_intercept"], np.array([np.nan, np.nan]), equal_nan=True)
    assert out["k"] == 1  # Only one parameter (slope)

    # Fitted values should pass through origin
    assert out["fitted_values"][0] == out["coefficients"][0] * X.ravel()[0]

    # Check that residuals are calculated correctly
    expected_fitted = out["coefficients"][0] * X.ravel()
    assert out["fitted_values"] == pytest.approx(expected_fitted)
    assert out["residuals"] == pytest.approx(y - expected_fitted)

    # Leverage should be calculated using distance from origin
    x_vals = X.ravel()
    expected_leverage = np.power(x_vals, 2) / np.sum(x_vals**2)
    assert out["leverage"] == pytest.approx(expected_leverage)

    # Standard error should be calculated with df = n - 1
    n = len(x_vals)
    expected_df = n - 1
    residual_ssq = np.sum(out["residuals"] ** 2)
    expected_se = np.sqrt(residual_ssq / expected_df)
    assert out["SE"] == pytest.approx(expected_se)


def test_simple_robust_regression_compare_with_regular_no_intercept() -> None:
    """Compare simple robust regression with regular regression for no-intercept case."""
    # Use simple linear data where robust and regular should be similar
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])  # Perfect linear relationship through origin

    robust_out = simple_robust_regression(x, y, fit_intercept=False)
    regular_out = multiple_linear_regression(x, y, fit_intercept=False)

    # Both should have intercept = 0 (robust) or np.nan (regular)
    assert robust_out["intercept"] == 0.0
    assert np.isnan(regular_out["intercept"])

    # Coefficients should be very close for this linear data
    assert robust_out["coefficients"][0] == pytest.approx(regular_out["coefficients"][0], rel=1e-3)

    # Both should have k=1 (one parameter)
    assert robust_out["k"] == 1
    assert regular_out["k"] == 1

    # Standard error calculations should use same degrees of freedom
    assert len(x) - robust_out["k"] == len(x) - regular_out["k"]
