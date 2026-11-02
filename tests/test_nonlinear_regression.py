"""Least squares for a model nonlinear in its parameters (#789 B8).

Box, Hunter and Hunter's section 14.6 fits the biochemical oxygen demand of six bottles against the
incubation time with eta = b1 (1 - exp(-b2 x)), and prints the estimates, the residual sum of
squares, the contour that bounds the 90% joint confidence region, and the fitted values. These tests
hold the function to those numbers, read from the reference catalogue, to scipy's curve_fit, and to
ordinary least squares for a model that is linear in its parameters.
"""

from __future__ import annotations

import numpy as np
import pytest
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import curve_fit

from process_improve.experiments.reference_catalogue import reference_experiments
from process_improve.regression import nonlinear_regression


def demand_model(x: np.ndarray, b1: float, b2: float) -> np.ndarray:
    """Return b1 (1 - exp(-b2 x)), the oxygen demand after x days."""
    return b1 * (1 - np.exp(-b2 * x))


@pytest.fixture(scope="module")
def record() -> dict:
    [entry] = [e for e in reference_experiments(source="box-hunter-hunter-1978") if e["name"] == "bhh-bod-nonlinear"]
    return entry


@pytest.fixture(scope="module")
def data(record) -> tuple[np.ndarray, np.ndarray]:
    rows = np.array(record["runs"]["rows"], dtype=float)
    return rows[:, 1], rows[:, 2]


@pytest.fixture(scope="module")
def fit(data) -> dict:
    days, demand = data
    return nonlinear_regression(demand_model, days, demand, {"b1": 200.0, "b2": 0.5}, conf_level=0.90)


def within_standard_errors(found, reference, std_errors, share: float = 1e-4) -> None:
    """Assert that each value is within ``share`` of its standard error of the reference.

    Two least-squares searches stop at slightly different points on the flat floor of the sum of
    squares, a few times 1e-8 standard errors apart, and how far apart depends on the platform's
    arithmetic. On the scale of the standard errors that distance is negligible; a wrong estimate
    is not.
    """
    gap = np.abs(np.asarray(found, dtype=float) - np.asarray(reference, dtype=float))
    np.testing.assert_array_less(gap, share * np.asarray(std_errors, dtype=float))


def printed(record, what: str) -> list[float]:
    [entry] = [e for e in record["expected"] if e["what"] == what]
    return [float(value) for value in entry["value"].split()]


# --------------------------------------------------------------------------- Box, Hunter and Hunter 14.6
def test_the_estimates_and_the_residual_sum_of_squares_are_the_book_s(fit):
    """b1 = 213.8 and S_R = 1168.01 as printed. The minimum is at b2 = 0.54724; the book prints 0.5473.

    The valley is flat along b2: S is 1168.00888 at the minimum and 1168.00902 at (213.8, 0.5473).
    """
    assert round(fit["estimates"]["b1"], 1) == 213.8
    assert fit["estimates"]["b2"] == pytest.approx(0.5473, abs=1e-4)
    assert round(fit["sse"], 2) == 1168.01
    assert (fit["n"], fit["df_resid"]) == (6, 4)
    assert fit["s2"] == pytest.approx(fit["sse"] / 4)


def test_the_90_percent_region_is_bounded_by_the_book_s_contour(fit):
    """S = S_R (1 + p/(n - p) F): 3693.57 with the exact F(2, 4) = 4.3246, and 3690.91 with the table's 4.32."""
    assert fit["region_sse"] == pytest.approx(fit["sse"] * (1 + 2 / 4 * stats.f.ppf(0.90, 2, 4)))
    assert round(fit["region_sse"], 2) == 3693.57
    assert round(1168.01 * (1 + 2 / 4 * 4.32), 2) == 3690.91


def test_the_fitted_values_are_within_0_3_of_table_14_14(fit, record):
    """Table 14.14's fitted values are those of b1 = 213.67 and b2 = 0.5499, a point on the same valley.

    There S is 1168.19, so the table's residuals add to nearly the same sum of squares as the minimum's.
    """
    assert np.max(np.abs(np.array(fit["fitted"]) - printed(record, "fitted"))) < 0.3
    assert np.array(fit["fitted"]) + np.array(fit["residuals"]) == pytest.approx([109, 149, 149, 191, 213, 224])
    assert demand_model(np.array([1.0, 10.0]), 213.67475, 0.549865) == pytest.approx([90.38, 212.81], abs=0.01)


def test_the_inference_is_curve_fit_s(fit, data):
    days, demand = data
    estimates, covariance = curve_fit(demand_model, days, demand, p0=[200.0, 0.5], ftol=1e-15, xtol=1e-15, gtol=1e-15)
    errors = [fit["std_errors"]["b1"], fit["std_errors"]["b2"]]
    within_standard_errors([fit["estimates"]["b1"], fit["estimates"]["b2"]], estimates, errors)
    sse = float(np.sum((demand - demand_model(days, *estimates)) ** 2))
    assert fit["sse"] == pytest.approx(sse, rel=1e-10)
    assert errors == pytest.approx(np.sqrt(np.diag(covariance)), rel=1e-5)
    assert fit["correlation"]["b1"]["b2"] == pytest.approx(
        covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1]), rel=1e-5
    )
    assert fit["correlation"]["b1"]["b1"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- against linear least squares
def test_a_model_linear_in_its_parameters_gives_ordinary_least_squares():
    """For y = a + b x + c x^2 the linearization is exact, so every number is the OLS fit's."""
    rng = np.random.default_rng(3)
    x = np.linspace(0, 4, 15)
    y = 1.0 + 0.5 * x - 0.3 * x**2 + rng.normal(0, 0.2, x.size)
    fit = nonlinear_regression(lambda x, a, b, c: a + b * x + c * x**2, x, y, {"a": 0, "b": 0, "c": 0}, 0.95)
    ols = sm.OLS(y, np.column_stack([np.ones_like(x), x, x**2])).fit()
    names = ["a", "b", "c"]
    within_standard_errors([fit["estimates"][n] for n in names], ols.params, ols.bse)
    assert [fit["std_errors"][n] for n in names] == pytest.approx(ols.bse, rel=1e-6)
    assert [fit["t_values"][n] for n in names] == pytest.approx(ols.tvalues, abs=1e-4)  # t is in standard errors
    assert [fit["p_values"][n] for n in names] == pytest.approx(ols.pvalues, rel=1e-5)
    for name, (low, high) in zip(names, ols.conf_int(), strict=True):
        within_standard_errors(fit["intervals"][name], [low, high], [fit["std_errors"][name]] * 2)
    assert fit["sse"] == pytest.approx(ols.ssr, rel=1e-12)


def test_two_regressors_reach_the_model_as_a_two_column_array():
    """An Arrhenius rate, k = A exp(-E / T) c, with the temperature and the concentration as columns."""
    temperature, concentration = (g.ravel() for g in np.meshgrid([300.0, 320.0, 340.0, 360.0], [0.5, 1.0, 2.0]))
    x = np.column_stack([temperature, concentration])

    def rate(x: np.ndarray, log_a: float, energy: float) -> np.ndarray:
        return np.exp(log_a - energy / x[:, 0]) * x[:, 1]

    y = rate(x, 10.0, 4000.0)
    fit = nonlinear_regression(rate, x, y, {"log_a": 9.0, "energy": 3500.0})
    assert fit["estimates"] == pytest.approx({"log_a": 10.0, "energy": 4000.0}, rel=1e-8)
    assert fit["sse"] == pytest.approx(0.0, abs=1e-18)


# --------------------------------------------------------------------------- what is refused
@pytest.mark.parametrize(
    ("start", "y", "conf_level", "message"),
    [
        ({}, [1.0, 2.0, 3.0], 0.95, "start must map each parameter's name"),
        ([1.0, 2.0], [1.0, 2.0, 3.0], 0.95, "start must map each parameter's name"),
        ({"b1": np.nan, "b2": 0.5}, [1.0, 2.0, 3.0], 0.95, "Each starting value must be a finite number"),
        ({"b1": 1.0, "b2": 0.5}, [1.0, np.inf, 3.0], 0.95, "y must be a sequence of finite numbers"),
        ({"b1": 1.0, "b2": 0.5}, [1.0, 2.0], 0.95, "2 rows leave no residual degree of freedom for 2 parameters"),
        ({"b1": 1.0, "b2": 0.5}, [1.0, 2.0, 3.0], 1.0, "conf_level must be between 0 and 1"),
    ],
)
def test_arguments_that_cannot_be_fitted_are_refused(start, y, conf_level, message):
    x = np.arange(1.0, len(y) + 1)
    with pytest.raises(ValueError, match=message):
        nonlinear_regression(demand_model, x, y, start, conf_level)


def test_a_model_that_returns_the_wrong_number_of_values_is_refused(data):
    days, demand = data
    with pytest.raises(ValueError, match="must return a finite value for each of the 6 rows"):
        nonlinear_regression(lambda x, b1, *_: b1 * x[:3], days, demand, {"b1": 1.0, "b2": 0.5})


def test_a_model_that_overflows_at_the_start_is_refused(data):
    days, demand = data
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="6 of them not finite"):
        nonlinear_regression(lambda x, b1, b2: b1 * np.exp(b2 * x), days, demand, {"b1": 1.0, "b2": 800.0})


def test_parameters_that_cannot_be_told_apart_are_refused(data):
    """In b1 * b2 * x only the product is estimable."""
    days, demand = data
    with pytest.raises(ValueError, match="cannot be told apart"):
        nonlinear_regression(lambda x, b1, b2: b1 * b2 * x, days, demand, {"b1": 2.0, "b2": 3.0})
