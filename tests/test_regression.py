import numpy as np
import pandas as pd
import pytest

from process_improve.regression.methods import (
    OLS,
    multiple_linear_regression,
    repeated_median_slope,
    robust_regression,
)
from process_improve.regression.tools import (
    get_regression_tool_specs,
)
from process_improve.regression.tools import (
    repeated_median as repeated_median_tool,
)
from process_improve.regression.tools import (
    robust_regression as robust_regression_tool,
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
    _X, _y = multiple_linear_regression_data


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
    with pytest.raises(ValueError, match=r"N >= K: You need at least as many rows .*"):
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
    out = robust_regression(X, y)

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
    out = robust_regression(x, y)
    assert np.isnan(out["standard_error_intercept"])
    assert np.isnan(out["standard_errors"][0])
    assert np.isnan(out["conf_intervals"][0][0])
    assert np.isnan(out["conf_intervals"][0][1])


def test_simple_robust_regression_missing_values() -> None:
    """Test y length less than 2 because of nan values."""

    x = np.array([1, 2, 3, 4, 5])
    y = np.array([np.nan, np.nan, np.nan, np.nan, 1])
    out = robust_regression(x, y)
    assert np.isnan(out["standard_error_intercept"])
    assert np.isnan(out["standard_errors"][0])
    assert np.isnan(out["conf_intervals"][0][0])
    assert np.isnan(out["conf_intervals"][0][1])


def test_simple_regression_no_error() -> None:
    """Tests cases where there is perfect fit."""
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([9, 8, 7, 6, 5])
    robust = robust_regression(x, y)
    regular = multiple_linear_regression(x, y, fit_intercept=True)
    assert robust["influence"] == pytest.approx(regular["influence"])
    assert robust["influence"] == pytest.approx([0] * 5)
    assert robust["conf_intervals"][0] == pytest.approx(regular["conf_intervals"][0])
    assert robust["coefficients"] == pytest.approx(regular["coefficients"])


def test_simple_robust_regression_no_intercept(simple_robust_regression_data: tuple[np.ndarray, np.ndarray]) -> None:
    """Test simple robust regression with fit_intercept=False."""
    X, y = simple_robust_regression_data
    out = robust_regression(X.ravel(), y, fit_intercept=False)

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

    robust_out = robust_regression(x, y, fit_intercept=False)
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


# -------------------------------------------------------------------------
# OLS class
# -------------------------------------------------------------------------


def test_ols_attributes_match_r(multiple_linear_regression_data: tuple[np.ndarray, np.ndarray]) -> None:
    """OLS().fit() should reproduce the R lm() reference values exactly."""
    X, y = multiple_linear_regression_data
    model = OLS().fit(X, y)

    assert model.is_fitted_
    assert model.n_samples_ == 7
    assert model.n_features_in_ == 1
    assert model.df_resid_ == 5
    assert model.df_model_ == 1
    assert model.intercept_ == pytest.approx(0.06641, abs=1e-5)
    assert model.coefficients_[0] == pytest.approx(4.08993, rel=1e-6)
    assert model.standard_error_intercept_ == pytest.approx(0.02710, abs=1e-5)
    assert model.standard_errors_[0] == pytest.approx(0.30530, abs=1e-5)
    assert model.t_value_intercept_ == pytest.approx(2.451, abs=1e-3)
    assert model.t_values_[0] == pytest.approx(13.396, abs=1e-3)
    assert model.p_value_intercept_ == pytest.approx(0.0579, abs=1e-4)
    assert model.p_values_[0] == pytest.approx(4.148e-05, rel=1e-3)
    assert model.r2_ == pytest.approx(0.9729, rel=1e-5)
    assert model.adj_r2_ == pytest.approx(0.9675, abs=1e-4)
    assert model.se_ == pytest.approx(0.03206, abs=1e-5)
    assert model.f_statistic_ == pytest.approx(179.5, rel=1e-3)
    assert model.f_pvalue_ == pytest.approx(4.148e-05, rel=1e-3)
    assert model.conf_intervals_[0] == pytest.approx([3.30512, 4.87474], rel=1e-5)
    assert model.conf_interval_intercept_ == pytest.approx([-0.003253309, 0.1360676], rel=1e-6)


def test_ols_summary_contains_r_style_blocks(
    multiple_linear_regression_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """The summary should look like R's summary(lm(...)) output."""
    X, y = multiple_linear_regression_data
    model = OLS().fit(X, y)
    summary = model.summary()

    assert "Call:" in summary
    assert "Residuals:" in summary
    assert "Coefficients:" in summary
    assert "(Intercept)" in summary
    assert "Estimate" in summary
    assert "Std. Error" in summary
    assert "t value" in summary
    assert "Pr(>|t|)" in summary
    assert "Signif. codes" in summary
    assert "Residual standard error:" in summary
    assert "Multiple R-squared:" in summary
    assert "Adjusted R-squared:" in summary
    assert "F-statistic:" in summary

    # Significance codes should appear for the highly-significant slope.
    assert "***" in summary
    # repr should match summary when fitted.
    assert repr(model) == summary
    # str(model) is also the summary.
    assert str(model) == summary


def test_ols_unfitted_repr_is_sklearn_style() -> None:
    """An unfit model should show the default sklearn-style class repr (not the summary)."""
    model = OLS()
    text = repr(model)
    assert text.startswith("OLS(")
    # No fitted-only sections.
    assert "Coefficients:" not in text
    assert "Residual standard error" not in text

    # Non-default parameters appear in the sklearn repr.
    text_nondefault = repr(OLS(fit_intercept=False, conflevel=0.99))
    assert "fit_intercept=False" in text_nondefault
    assert "conflevel=0.99" in text_nondefault


def test_ols_predict_matches_fitted_values(
    multiple_linear_regression_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """predict() on the training data should equal fitted_values_."""
    X, y = multiple_linear_regression_data
    model = OLS().fit(X, y)
    np.testing.assert_allclose(model.predict(X), model.fitted_values_, rtol=1e-12)

    # predict on a held-out grid follows intercept + slope * x.
    x_new = np.array([0.05, 0.10, 0.15]).reshape(-1, 1)
    expected = model.intercept_ + model.coefficients_[0] * x_new.ravel()
    np.testing.assert_allclose(model.predict(x_new), expected, rtol=1e-12)


def test_ols_predict_wrong_shape_raises(
    multiple_linear_regression_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """predict() must reject an X whose feature count differs from fit (SEC-23)."""
    X, y = multiple_linear_regression_data
    model = OLS().fit(X, y)  # fitted with a single feature
    assert model.n_features_in_ == 1

    with pytest.raises(ValueError, match=r"X has 2 feature\(s\), but OLS was fitted with 1"):
        model.predict(np.ones((3, 2)))

    # The correct shape still returns finite predictions.
    good = model.predict(np.array([[0.05], [0.10]]))
    assert good.shape == (2,)
    assert np.all(np.isfinite(good))


def test_ols_prediction_interval_matches_pi_range_grid(
    multiple_linear_regression_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """prediction_interval() on the pi_range_ grid reproduces the stored grid."""
    X, y = multiple_linear_regression_data
    model = OLS().fit(X, y)
    grid_x = model.pi_range_[:, 0]
    pi = model.prediction_interval(grid_x)
    np.testing.assert_allclose(pi.lower, model.pi_range_[:, 1], rtol=1e-9)
    np.testing.assert_allclose(pi.upper, model.pi_range_[:, 2], rtol=1e-9)


def test_ols_prediction_interval_at_arbitrary_x(
    multiple_linear_regression_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """prediction_interval() works at any x, including outside the training range."""
    X, y = multiple_linear_regression_data
    model = OLS().fit(X, y)
    x_new = np.array([-5.0, 0.0, 100.0])
    pi = model.prediction_interval(x_new)

    np.testing.assert_allclose(pi.predicted, model.predict(x_new.reshape(-1, 1)), rtol=1e-12)
    assert np.all(pi.lower < pi.predicted)
    assert np.all(pi.predicted < pi.upper)

    # A higher confidence level widens the interval.
    pi_99 = model.prediction_interval(x_new, conflevel=0.99)
    assert np.all((pi_99.upper - pi_99.lower) > (pi.upper - pi.lower))


def test_ols_prediction_interval_scalar_and_no_intercept(
    multiple_linear_regression_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Scalar input is treated as one point, and no-intercept models are supported."""
    X, y = multiple_linear_regression_data

    # Scalar input.
    pi_scalar = OLS().fit(X, y).prediction_interval(0.12)
    assert pi_scalar.predicted.shape == (1,)
    assert pi_scalar.lower[0] < pi_scalar.upper[0]

    # No-intercept model.
    model = OLS(fit_intercept=False).fit(X, y)
    pi = model.prediction_interval(np.array([0.05, 0.10, 0.15]))
    assert pi.predicted.shape == (3,)
    assert np.all(pi.lower < pi.upper)


def test_ols_prediction_interval_multifeature() -> None:
    """prediction_interval() supports multi-feature models and single-point input."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(40, 2)), columns=["a", "b"])
    y = pd.Series(X["a"] * 2.0 - X["b"] + 0.5 + rng.normal(scale=0.1, size=40), name="y")
    model = OLS().fit(X, y)

    pi = model.prediction_interval([[0.0, 0.0], [1.0, -1.0]])
    assert pi.predicted.shape == (2,)
    assert np.all(pi.lower < pi.upper)

    # A bare 1-D array is treated as one multi-feature point.
    pi_single = model.prediction_interval(np.array([0.5, 0.5]))
    assert pi_single.predicted.shape == (1,)

    with pytest.raises(ValueError, match="feature"):
        model.prediction_interval([[1.0, 2.0, 3.0]])


def test_ols_no_intercept(multiple_linear_regression_data: tuple[np.ndarray, np.ndarray]) -> None:
    """OLS with fit_intercept=False should match R's summary(lm(y~x+0))."""
    X, y = multiple_linear_regression_data
    model = OLS(fit_intercept=False).fit(X, y)

    assert np.isnan(model.intercept_)
    assert len(model.coefficients_) == 1
    assert model.coefficients_[0] == pytest.approx(4.7591, rel=1e-6)
    assert model.se_ == pytest.approx(0.04343, abs=1e-5)
    assert model.r2_ == pytest.approx(0.991, abs=1e-4)

    summary = model.summary()
    assert "(Intercept)" not in summary
    assert "+ 0" in summary  # formula shows no-intercept form


def test_ols_to_dict_matches_legacy_function(
    multiple_linear_regression_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """OLS.to_dict() must return the same dict as multiple_linear_regression()."""
    X, y = multiple_linear_regression_data
    expected = multiple_linear_regression(X, y, fit_intercept=True, na_rm=False)
    got = OLS(na_rm=False).fit(X, y).to_dict()

    assert got["N"] == expected["N"]
    assert got["intercept"] == pytest.approx(expected["intercept"])
    np.testing.assert_allclose(got["coefficients"], expected["coefficients"])
    np.testing.assert_allclose(got["standard_errors"], expected["standard_errors"])
    assert got["R2"] == pytest.approx(expected["R2"])
    assert got["SE"] == pytest.approx(expected["SE"])


def test_ols_pandas_dataframe_with_named_columns() -> None:
    """Feature names from a DataFrame should appear in the summary's formula and coefficient table."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.standard_normal((40, 2)), columns=["temperature", "pressure"])
    y = pd.Series(X["temperature"] * 2 - X["pressure"] + 0.5 + 0.1 * rng.standard_normal(40), name="yield")

    model = OLS().fit(X, y)
    summary = model.summary()

    assert "yield ~ temperature + pressure" in summary
    assert "temperature" in summary
    assert "pressure" in summary
    assert model.feature_names_in_ == ["temperature", "pressure"]
    assert model.target_name_ == "yield"


def test_ols_fit_with_non_default_pandas_index() -> None:
    """A DataFrame/Series whose index is not 0..N-1 must still fit.

    Regression test: ``y`` was rebuilt with a fresh RangeIndex while ``X`` kept
    its original index, so statsmodels rejected the design matrix with
    "The indices for endog and exog are not aligned".
    """
    rng = np.random.default_rng(7)
    n = 30
    # A non-default index: e.g. the rows survived a filter on a larger frame.
    odd_index = pd.Index(range(100, 100 + 2 * n, 2))
    X = pd.DataFrame(rng.standard_normal((n, 2)), columns=["a", "b"], index=odd_index)
    y = pd.Series(X["a"] * 1.5 - X["b"] + 0.3 + 0.05 * rng.standard_normal(n), index=odd_index, name="resp")

    model = OLS().fit(X, y)
    assert model.is_fitted_ is True

    # A DatetimeIndex is another common non-default index.
    date_index = pd.date_range("2024-01-01", periods=n, freq="D")
    X_dt = X.set_axis(date_index, axis=0)
    y_dt = y.set_axis(date_index, axis=0)
    model_dt = OLS().fit(X_dt, y_dt)
    assert model_dt.is_fitted_ is True

    # The index must not change the fitted result: same data, default index.
    model_plain = OLS().fit(X.reset_index(drop=True), y.reset_index(drop=True))
    np.testing.assert_allclose(model.coefficients_, model_plain.coefficients_, rtol=1e-12)
    np.testing.assert_allclose(model.intercept_, model_plain.intercept_, rtol=1e-12)

    # na_rm path also reindexes X internally; a non-default index must work there too.
    X_na = X.copy()
    X_na.iloc[3, 0] = np.nan
    model_na = OLS(na_rm=True).fit(X_na, y)
    assert model_na.is_fitted_ is True
    assert model_na.n_samples_ == n - 1


def test_ols_handles_insufficient_data() -> None:
    """A fit with 1 datapoint should mark the model as unfit but not raise."""
    model = OLS().fit(np.array([2.0]), np.array([5.0]))
    assert model.is_fitted_ is False
    # Summary returns a graceful message rather than crashing.
    assert "not been fitted" in model.summary()


def test_ols_predict_accepts_dataframe_series_and_1d_numpy() -> None:
    """predict() should accept pandas DataFrame / Series and 1-D numpy arrays."""
    rng = np.random.default_rng(11)
    X_df = pd.DataFrame(rng.standard_normal((30, 2)), columns=["a", "b"])
    y = X_df @ [1.0, -1.0] + 0.5 * rng.standard_normal(30)
    model = OLS().fit(X_df, y)

    # DataFrame input
    pred_df = model.predict(X_df)
    np.testing.assert_allclose(pred_df, model.fitted_values_, rtol=1e-12)

    # 1-D numpy input on a single-feature model
    X1d_train = rng.standard_normal(30)
    y1d = 2.0 * X1d_train + 0.1 * rng.standard_normal(30)
    m1 = OLS().fit(X1d_train, y1d)
    pred_1d = m1.predict(np.array([0.1, 0.5, -0.2]))
    expected = m1.intercept_ + m1.coefficients_[0] * np.array([0.1, 0.5, -0.2])
    np.testing.assert_allclose(pred_1d, expected, rtol=1e-12)

    # pd.Series input
    pred_series = m1.predict(pd.Series([0.1, 0.5, -0.2]))
    np.testing.assert_allclose(pred_series, expected, rtol=1e-12)


def test_ols_summary_with_nonsignificant_coefficient() -> None:
    """The summary should render rows for non-significant coefficients without crashing."""
    rng = np.random.default_rng(2)
    # Pure noise: neither slope is meaningfully different from zero.
    X = pd.DataFrame(rng.standard_normal((20, 2)), columns=["x1", "x2"])
    y = pd.Series(rng.standard_normal(20), name="noise")
    model = OLS().fit(X, y)
    summary = model.summary()

    assert "x1" in summary
    assert "x2" in summary
    # All rows should be present even when no coefficient hits the *** threshold.
    assert summary.count("\n") > 5


def test_ols_missing_values_preserve_residual_shape() -> None:
    """Residuals should preserve the original y shape with NaN at dropped rows."""
    X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.0, np.nan, 4.0, np.nan, 9.0])
    model = OLS(na_rm=True).fit(X, y)

    assert model.is_fitted_
    assert len(model.residuals_) == 5  # original length
    assert np.isnan(model.residuals_[1])
    assert np.isnan(model.residuals_[3])
    assert model.intercept_ == pytest.approx(-0.25)
    assert model.coefficients_[0] == pytest.approx(1.75)


# ---------------------------------------------------------------------------
# Agent-tool wrappers: process_improve.regression.tools
# ---------------------------------------------------------------------------


def test_robust_regression_tool_recovers_known_line() -> None:
    """The wrapper should recover the slope/intercept of an exact line."""
    rng = np.random.default_rng(11)
    x = list(np.arange(0.0, 10.0, 0.1))
    y = [2.0 * xi + 1.0 + 0.01 * float(rng.standard_normal()) for xi in x]
    result = robust_regression_tool(x=x, y=y)

    assert "error" not in result
    assert result["slope"] == pytest.approx(2.0, abs=0.05)
    assert result["intercept"] == pytest.approx(1.0, abs=0.1)
    assert result["n"] == len(x)
    assert result["r2"] > 0.99
    assert result["confidence_level"] == 0.95
    assert len(result["fitted_values"]) == len(x)
    assert len(result["residuals"]) == len(x)
    # Confidence interval and prediction intervals should be present and well-shaped.
    assert "slope_confidence_interval" in result
    assert "prediction_interval_x" in result
    assert "prediction_interval_lower" in result
    assert "prediction_interval_upper" in result
    assert (
        len(result["prediction_interval_x"])
        == len(result["prediction_interval_lower"])
        == len(result["prediction_interval_upper"])
    )


def test_robust_regression_tool_no_intercept() -> None:
    """fit_intercept=False forces the line through the origin."""
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    y = [3.0, 6.0, 9.0, 12.0, 15.0, 18.0]
    result = robust_regression_tool(x=x, y=y, fit_intercept=False, confidence_level=0.99)

    assert "error" not in result
    assert result["slope"] == pytest.approx(3.0, abs=1e-6)
    assert result["intercept"] == pytest.approx(0.0, abs=1e-6)
    assert result["confidence_level"] == 0.99


def test_robust_regression_tool_returns_error_on_mismatched_lengths() -> None:
    """Length-mismatched inputs surface as an error dict, not an exception."""
    result = robust_regression_tool(x=[1.0, 2.0, 3.0], y=[1.0, 2.0])
    assert "error" in result


def test_repeated_median_tool_matches_underlying_method() -> None:
    """The wrapper's slope must match the underlying repeated_median_slope."""
    rng = np.random.default_rng(13)
    x = np.linspace(0.0, 10.0, 50)
    y = 1.5 * x - 4.0 + rng.standard_normal(50) * 0.05
    result = repeated_median_tool(x=list(x), y=list(y))

    assert "error" not in result
    assert result["n"] == len(x)
    assert result["slope"] == pytest.approx(repeated_median_slope(x, y), rel=1e-9)


def test_repeated_median_tool_returns_error_on_bad_input() -> None:
    """Non-numeric input should surface as an error dict."""
    result = repeated_median_tool(x=["a", "b"], y=[1.0, 2.0])  # type: ignore[arg-type]
    assert "error" in result


def test_get_regression_tool_specs_lists_both_tools() -> None:
    """The module-level convenience returns both registered specs."""
    specs = get_regression_tool_specs()
    names = {spec.get("name") for spec in specs}
    assert {"robust_regression", "repeated_median"}.issubset(names)
