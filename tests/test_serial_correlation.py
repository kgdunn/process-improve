"""Serially correlated records: the variance of a difference of averages, and a first-order moving average (#802 G3).

Box, Hunter and Hunter's 210 batch yields (sections 3.1 and 18.2) are serially correlated, r1 = -0.29.
These tests hold the two functions to the book's numbers, read from the reference catalogue's record,
and to simulations of a process whose model is known.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from process_improve.experiments.reference_catalogue import reference_experiments
from process_improve.univariate import difference_variance, fit_ma1


@pytest.fixture(scope="module")
def yields() -> np.ndarray:
    """Return Box, Hunter and Hunter's 210 consecutive batch yields of the standard process."""
    [record] = [
        e for e in reference_experiments(source="box-hunter-hunter-1978") if e["name"] == "bhh-production-record-210"
    ]
    return np.array([row[1] for row in record["runs"]["rows"]], dtype=float)


def ma1_record(theta: float, n: int, seed: int) -> np.ndarray:
    """Simulate y_t = e_t - theta e_{t-1}, with e_t of variance 1."""
    e = np.random.default_rng(seed).normal(0.0, 1.0, n + 1)
    return e[1:] - theta * e[:-1]


# --------------------------------------------------------------------------- the variance of a difference
def test_box_hunter_and_hunter_s_difference_of_two_averages_of_ten(yields):
    """Equation 18.5: 2 s^2 / 10 = 1.66 if the yields were independent, and 1.66 (1 + 1.7 r1) = 0.84 as they are.

    The book prints 1.67 for 2 x 8.29 / 10, which is 1.658. The plant trial's difference of 1.30 then
    has z = 1.01 and 1.41 and one-sided significance levels 0.16 and 0.08.
    """
    s2 = yields.var(ddof=1)
    r1 = round(fit_ma1(yields)["r1"], 2)  # -0.29, as the book uses it: unrounded, -0.2885 gives 0.846
    independent = difference_variance(10, variance=s2)
    correlated = difference_variance(10, r1, s2)
    assert independent == pytest.approx(2 * s2 / 10)
    assert independent == pytest.approx(1.66, abs=0.005)
    assert correlated == pytest.approx(0.84, abs=0.005)
    assert 1.30 / math.sqrt(independent) == pytest.approx(1.01, abs=0.005)
    assert stats.norm.sf(1.30 / math.sqrt(independent)) == pytest.approx(0.16, abs=0.005)
    assert stats.norm.sf(1.30 / math.sqrt(correlated)) == pytest.approx(0.08, abs=0.005)


@pytest.mark.parametrize("rho1", [-0.4, -0.29, 0.0, 0.25])
def test_the_bracket_for_averages_of_ten_is_one_plus_1_7_rho1(rho1):
    assert difference_variance(10, rho1) / difference_variance(10) == pytest.approx(1 + 1.7 * rho1)
    assert difference_variance(10, rho1, adjacent=False) / difference_variance(10) == pytest.approx(1 + 1.8 * rho1)


@pytest.mark.parametrize("adjacent", [True, False])
def test_the_variance_of_a_difference_agrees_with_a_simulated_moving_average(adjacent):
    """With theta = 0.6, rho1 is -0.6 / 1.36 and each observation's variance 1.36."""
    theta, n = 0.6, 5
    y = ma1_record(theta, 1_000_000, seed=3)
    blocks = y[: len(y) // n * n].reshape(-1, n).mean(axis=1)
    differences = blocks[1:] - blocks[:-1] if adjacent else blocks[2::2] - blocks[:-2:2]
    expected = difference_variance(n, -theta / (1 + theta**2), 1 + theta**2, adjacent=adjacent)
    assert np.var(differences) == pytest.approx(expected, rel=0.02)


# --------------------------------------------------------------------------- a first-order moving average
def test_box_hunter_and_hunter_s_moving_average_by_moments(yields):
    """Section 18.2: matched to r1 = -0.29 and s^2 = 8.30, theta = 0.32 and sigma_e^2 = 7.5 (equation 18.16)."""
    found = fit_ma1(yields)
    assert found["n"] == 210
    assert found["method"] == "moments"
    assert found["r1"] == pytest.approx(-0.29, abs=0.005)
    assert found["theta"] == pytest.approx(0.32, abs=0.005)
    assert found["sigma2"] == pytest.approx(7.5, abs=0.05)
    assert found["mean"] == pytest.approx(84.1, abs=0.05)
    assert found["theta_std_error"] is None
    assert -found["theta"] / (1 + found["theta"] ** 2) == pytest.approx(found["r1"]), "the model's r1 is the record's"


def test_the_likelihood_uses_the_whole_record_and_gives_a_standard_error(yields):
    """On the 210 yields the likelihood's theta is 0.48, beside the moments' 0.32: the first uses every lag."""
    found = fit_ma1(yields, method="likelihood")
    assert found["method"] == "likelihood"
    assert found["theta"] == pytest.approx(0.48, abs=0.01)
    assert found["theta_std_error"] == pytest.approx(0.063, abs=0.005)
    assert found["sigma2"] == pytest.approx(7.07, abs=0.05)
    assert found["r1"] == pytest.approx(fit_ma1(yields)["r1"])


@pytest.mark.parametrize("method", ["moments", "likelihood"])
def test_both_methods_find_the_theta_a_long_simulated_record_was_made_with(method):
    found = fit_ma1(ma1_record(0.4, 5_000, seed=11) + 50.0, method=method)
    assert found["theta"] == pytest.approx(0.4, abs=0.03)
    assert found["sigma2"] == pytest.approx(1.0, abs=0.05)
    assert found["mean"] == pytest.approx(50.0, abs=0.05)


def test_a_record_with_no_lag_1_autocorrelation_has_a_theta_of_0():
    found = fit_ma1([0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0])  # each neighbour of a 1 or a -1 is a 0
    assert (found["r1"], found["theta"]) == (0.0, 0.0)
    assert found["sigma2"] == pytest.approx(4 / 7), "the record's own variance"


# --------------------------------------------------------------------------- what is refused
def test_a_lag_1_autocorrelation_no_moving_average_has_is_refused_by_moments():
    alternating = [1.0, -1.0] * 10
    with pytest.raises(ValueError, match=r"lag-1 autocorrelation is -0\.950; a first-order moving average"):
        fit_ma1(alternating)


@pytest.mark.parametrize(
    ("record", "method", "message"),
    [
        ([1.0, 2.0], "moments", "at least three finite numbers"),
        ([1.0, 2.0, math.nan], "moments", "at least three finite numbers"),
        ([3.0, 3.0, 3.0], "moments", "Every value of the record is the same"),
        ([1.0, 2.0, 3.0], "least squares", "method must be one of"),
    ],
)
def test_a_record_that_cannot_be_fitted_is_refused(record, method, message):
    with pytest.raises(ValueError, match=message):
        fit_ma1(record, method=method)


@pytest.mark.parametrize(
    ("n", "rho1", "variance", "message"),
    [
        (0, 0.0, 1.0, "a whole number of at least 1"),
        (2.5, 0.0, 1.0, "a whole number of at least 1"),
        (True, 0.0, 1.0, "a whole number of at least 1"),
        (10, 0.6, 1.0, "rho1 must be from -0.5 to 0.5"),
        (10, 0.0, -1.0, "The variance must be a finite number, 0 or more"),
    ],
)
def test_a_difference_of_averages_that_cannot_be_formed_is_refused(n, rho1, variance, message):
    with pytest.raises(ValueError, match=message):
        difference_variance(n, rho1, variance)
