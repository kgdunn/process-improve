"""Box and Tidwell's estimate of the power of a regressor (#801 F2).

Montgomery, Peck and Vining's windmill data (Introduction to Linear Regression Analysis, chapter 5;
from Joglekar, Schuenemeyer and LaRiccia, 1989) relate a windmill's direct-current output to the
wind speed. The book's first Box-Tidwell step gives a power of -0.92, and it fits the reciprocal of
the wind speed. These tests hold the function to that step, to an independent nonlinear
least-squares fit of the same model, and to responses whose power is known.
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy.optimize import curve_fit

from process_improve.regression import box_tidwell
from process_improve.regression.box_tidwell import SERIES, _iterate, _power_columns, _std_errors

# The package exports the function under the module's name, so the module is fetched by its path.
box_tidwell_module = importlib.import_module("process_improve.regression.box_tidwell")

# fmt: off
WIND = np.array([
    5.00, 6.00, 3.40, 2.70, 10.00, 9.70, 9.55, 3.05, 8.15, 6.20, 2.90, 6.35, 4.60,
    5.80, 7.40, 3.60, 7.85, 8.80, 7.00, 5.45, 9.10, 10.20, 4.10, 3.95, 2.45,
])
OUTPUT = np.array([
    1.582, 1.822, 1.057, 0.500, 2.236, 2.386, 2.294, 0.558, 2.166, 1.866, 0.653, 1.930, 1.562,
    1.737, 2.088, 1.137, 2.179, 2.112, 1.800, 1.501, 2.303, 2.310, 1.194, 1.144, 0.123,
])
# fmt: on


@pytest.fixture(scope="module")
def windmill() -> dict:
    return box_tidwell(pd.DataFrame({"wind": WIND}), OUTPUT)


def nonlinear_fit(x: np.ndarray, y: np.ndarray, start: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Fit y = b0 + b1 x^a by nonlinear least squares; return the estimates and their standard errors."""
    estimates, covariance = curve_fit(lambda x, b0, b1, a: b0 + b1 * x**a, x, y, p0=start)
    return estimates, np.sqrt(np.diag(covariance))


# --------------------------------------------------------------------------- the windmill data
def test_the_data_give_the_book_s_straight_line():
    """The book's first fit, y = 0.1309 + 0.2411 x with R^2 = 0.8745, checks the 25 pairs."""
    slope, intercept = np.polyfit(WIND, OUTPUT, 1)
    r_squared = np.corrcoef(WIND, OUTPUT)[0, 1] ** 2
    assert (round(intercept, 4), round(slope, 4), round(r_squared, 4)) == (0.1309, 0.2411, 0.8745)


def test_the_first_step_is_the_book_s(windmill):
    """The book regresses on x and x ln x: -2.4168 + 1.5344 x - 0.4626 x ln x, so alpha_1 = -0.92."""
    first = windmill["iterations"][1]["wind"]
    assert windmill["iterations"][0] == {"wind": 1.0}
    assert first == pytest.approx(1 - 0.4626 / 0.2411, abs=1e-3)
    assert round(first, 2) == -0.92


def test_the_power_is_the_nonlinear_least_squares_estimate(windmill):
    """The steps settle at -0.833, with a standard error of 0.159 on 25 - 3 = 22 degrees of freedom."""
    estimates, errors = nonlinear_fit(WIND, OUTPUT, [3.0, -7.0, -1.0])
    assert windmill["power"]["wind"] == pytest.approx(estimates[2], rel=1e-6)
    assert windmill["std_error"]["wind"] == pytest.approx(errors[2], rel=1e-5)
    assert (round(windmill["power"]["wind"], 3), round(windmill["std_error"]["wind"], 3)) == (-0.833, 0.159)
    assert (windmill["n"], windmill["df_resid"]) == (25, 22)
    assert windmill["sse"] < windmill["sse_linear"]
    residuals = OUTPUT - (estimates[0] + estimates[1] * WIND ** estimates[2])
    assert windmill["sse"] == pytest.approx(residuals @ residuals, rel=1e-8)


def test_the_book_s_reciprocal_is_about_one_standard_error_from_the_estimate(windmill):
    """The book fits 1/x; -1 is (-1 + 0.833) / 0.159 = -1.05 standard errors from the estimate."""
    z = (-1 - windmill["power"]["wind"]) / windmill["std_error"]["wind"]
    assert z == pytest.approx(-1.05, abs=0.005)


def test_the_linearity_test_is_the_t_value_of_the_constructed_variable(windmill):
    """At power 1 the constructed variable is x ln x; its t value is -9.13, so the line is rejected."""
    fit = sm.OLS(OUTPUT, sm.add_constant(np.column_stack([WIND, WIND * np.log(WIND)]))).fit()
    assert windmill["linearity_t"]["wind"] == pytest.approx(fit.tvalues[2], rel=1e-9)
    assert windmill["linearity_p_value"]["wind"] == pytest.approx(fit.pvalues[2], rel=1e-6)
    assert round(windmill["linearity_t"]["wind"], 2) == -9.13


# --------------------------------------------------------------------------- responses whose power is known
@pytest.mark.parametrize(("power", "response"), [(0.5, np.sqrt), (-1.0, np.reciprocal), (2.0, np.square)])
def test_a_response_straight_in_a_power_gives_that_power(power, response):
    x = np.linspace(1.0, 8.0, 12)
    found = box_tidwell(x, 4.0 - 1.5 * response(x))
    assert found["power"]["x"] == pytest.approx(power, abs=1e-8)
    assert found["sse"] == pytest.approx(0.0, abs=1e-18)
    assert found["std_error"]["x"] == pytest.approx(0.0, abs=1e-6)


def test_a_logarithm_is_the_power_0():
    """The last steps pass within SERIES of 0, where the columns come from their series."""
    x = np.linspace(1.0, 20.0, 15)
    found = box_tidwell(x, 1.0 + 2.0 * np.log(x))
    assert abs(found["power"]["x"]) < 1e-8
    assert any(0 < abs(step["x"]) < SERIES for step in found["iterations"])


def test_two_powers_are_estimated_together_and_a_column_can_enter_as_it_is():
    """On a grid, y = 1 + 2 / a + 3 sqrt(b) + noise, against nonlinear least squares with 5 and 4 parameters."""
    a, b = (grid.ravel() for grid in np.meshgrid(np.linspace(1, 5, 6), np.linspace(1, 9, 6)))
    y = 1 + 2 / a + 3 * np.sqrt(b) + np.random.default_rng(1).normal(0, 0.05, a.size)
    frame = pd.DataFrame({"a": a, "b": b})

    both = box_tidwell(frame, y)
    estimates, covariance = curve_fit(
        lambda ab, b0, b1, p1, b2, p2: b0 + b1 * ab[0] ** p1 + b2 * ab[1] ** p2,
        np.vstack([a, b]),
        y,
        p0=[1, 2, -1, 3, 0.5],
    )
    assert [both["power"]["a"], both["power"]["b"]] == pytest.approx(estimates[[2, 4]], rel=1e-6)
    assert [both["std_error"]["a"], both["std_error"]["b"]] == pytest.approx(
        np.sqrt(np.diag(covariance))[[2, 4]], rel=1e-4
    )
    assert both["df_resid"] == 36 - 5

    one = box_tidwell(frame, y, columns=["a"])
    estimates, covariance = curve_fit(
        lambda ab, b0, b1, p1, b2: b0 + b1 * ab[0] ** p1 + b2 * ab[1], np.vstack([a, b]), y, p0=[1, 2, -1, 3]
    )
    assert one["power"] == pytest.approx({"a": estimates[2]}, rel=1e-6)
    assert one["std_error"]["a"] == pytest.approx(np.sqrt(covariance[2, 2]), rel=1e-4)
    assert one["df_resid"] == 36 - 4


def test_an_array_s_columns_are_named_x1_x2_and_so_on():
    a, b = (grid.ravel() for grid in np.meshgrid(np.linspace(1, 5, 5), np.linspace(1, 9, 5)))
    found = box_tidwell(np.column_stack([a, b]), 2 + 1 / a + 0.5 * b**2)
    assert found["power"] == pytest.approx({"x1": -1.0, "x2": 2.0}, abs=1e-7)


# --------------------------------------------------------------------------- the columns and their derivatives
@pytest.mark.parametrize("power", [-1.3, 0.0, 2e-7, 3e-6, 0.5, 2.0])
def test_each_derivative_column_is_the_derivative_of_its_column(power):
    x = np.array([[0.5], [1.0], [2.0], [7.5]])
    h = 1e-5
    above, below = (_power_columns(x, np.array([power + d]))[0] for d in (h, -h))
    assert _power_columns(x, np.array([power]))[1] == pytest.approx((above - below) / (2 * h), abs=1e-8)


def test_the_series_and_the_closed_form_meet_at_series():
    x = np.array([[0.5], [1.0], [2.0], [7.5]])
    series, closed = (_power_columns(x, np.array([SERIES * f])) for f in (0.999999, 1.000001))
    assert series[0] == pytest.approx(closed[0], abs=1e-11)
    assert series[1] == pytest.approx(closed[1], abs=1e-9)


# --------------------------------------------------------------------------- what is refused
@pytest.mark.parametrize(
    ("x", "y", "columns", "message"),
    [
        ([1.0, 2.0, 0.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0], None, "must be positive"),
        ([1.0, 2.0, 1.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0, 5.0], None, "three or more different values"),
        ([1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0], None, "one value per row of X"),
        ([1.0, 2.0, np.nan, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0], None, "finite number"),
        ([1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 2.0, 2.0, 2.0, 2.0], None, "same value in every row"),
        ([1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0], ["z"], "columns must name"),
        ([1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0], ["x", "x"], "columns must name"),
        ([1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0], [], "columns must name"),
        ([1.0, 2.0, 3.0], [1.0, 2.0, 4.0], None, "no residual degree of freedom"),
    ],
)
def test_data_that_cannot_give_a_power_are_refused(x, y, columns, message):
    with pytest.raises(ValueError, match=message):
        box_tidwell(x, y, columns=columns)


@pytest.mark.parametrize(("max_iter", "tol"), [(0, 1e-8), (True, 1e-8), (2.5, 1e-8), (10, 0.0), (10, np.nan)])
def test_bad_settings_are_refused(max_iter, tol):
    with pytest.raises(ValueError, match=r"max_iter must be|tol must be"):
        box_tidwell(WIND, OUTPUT, max_iter=max_iter, tol=tol)


@pytest.mark.parametrize(
    ("y", "message"),
    [
        ([1.0, 0.5, 0.5, 1.0], "has no straight-line effect on y at a trial power"),
        ([1.0, 0.0, 0.0, 1.0], "has no straight-line effect on y at a trial power"),
        ([3.0, 1.0, 2.0, 2.5, 0.5, 2.0], "did not settle in max_iter=100 steps"),
    ],
)
def test_a_regressor_with_no_clear_straight_line_effect_has_no_power(y, message):
    """Symmetric responses have no slope to move a power by; on the last, the power runs off to -infinity.

    The slope of a symmetric response is zero only to rounding, so it is judged against the spread of
    y: the same refusal on every platform, whatever rounding leaves in it.
    """
    with pytest.raises(ValueError, match=message):
        box_tidwell(np.arange(1.0, len(y) + 1), y)


def rises_away_from_one(*args: np.ndarray) -> float:
    """Return a residual sum of squares that every step away from the power 1 raises (``args[-1]`` holds the powers)."""
    return 1.0 if args[-1][0] == 1 else 2.0


def test_a_step_that_never_lowers_the_sum_of_squares_is_refused(monkeypatch):
    """Halving stops at a step still larger than ``tol``: the powers cannot be estimated."""
    monkeypatch.setattr(box_tidwell_module, "_sse", rises_away_from_one)
    x, linear = WIND[:, None], np.ones((len(WIND), 1))
    with pytest.raises(ValueError, match=r"no step from \[1.0\] lowers the residual sum of squares"):
        _iterate(OUTPUT, linear, x, max_iter=10, tol=1e-300)


def test_a_step_halved_below_tol_leaves_the_powers_where_they_are(monkeypatch):
    """At a minimum to rounding, a step that only raises the sum of squares is halved to nothing."""
    monkeypatch.setattr(box_tidwell_module, "_sse", rises_away_from_one)
    powers, history = _iterate(OUTPUT, np.ones((len(WIND), 1)), WIND[:, None], max_iter=10, tol=1e-8)
    assert powers.tolist() == [1.0]
    assert len(history) == 1


def test_powers_whose_derivative_columns_vanish_have_no_standard_error():
    """With no slope, the derivative column is zero and the powers cannot be told from the intercept."""
    x = np.arange(1.0, 5.0)[:, None]
    with pytest.raises(ValueError, match="derivative columns are linearly dependent"):
        _std_errors(np.array([1.0, 0.5, 0.5, 1.0]), np.ones((4, 1)), x, np.array([1.0]), df_resid=1)


def test_powers_still_settling_are_refused_with_a_hint():
    with pytest.raises(ValueError, match="otherwise pass a larger max_iter"):
        box_tidwell(WIND, OUTPUT, max_iter=2)
