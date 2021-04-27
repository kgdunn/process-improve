import pytest
import numpy as np
from process_improve.bivariate.methods import find_elbow_point


@pytest.fixture
def straight_line():
    slope = 2.5
    intercept = -13
    delta = 0.1
    x = np.arange(0, 5, delta)
    y0 = np.zeros_like(x)
    y1 = slope * x + intercept
    y2 = np.ones_like(x)
    return x, y0, y1, y2


def test__validate_nan_output(straight_line):
    """
    Is the elbow where we expect it? For clean, perfect data.
    """
    x, y0, y1, y2 = straight_line
    # Horizontal line
    elbow = find_elbow_point(x, y0)
    assert np.isnan(elbow)

    # Vertical line
    elbow = find_elbow_point(x, y2)
    assert np.isnan(elbow)

    # A line, but no elbow.
    elbow = find_elbow_point(x, y1)
    assert np.isnan(elbow)


@pytest.fixture
def elbow_with_synthetic_data():
    """
    Creates simulated data to test against.

    Create a line with slope of 2.0 between 0 and 5 on the x-axis.
    Then a line of slope of 3.0 between 5 and 10.
    The intersection is of course an elbow, at the break point of x=5.
    Add no noise: the function must still be able to find it.
    """
    delta = 0.1
    slope_2 = 2
    slope_3 = 3
    break_pt = 5
    intercept_2 = 0.0
    line_2 = np.arange(0, break_pt, delta) * slope_2 + intercept_2
    line_3 = (
        np.arange(break_pt, break_pt * 2, delta) * slope_3
        + (slope_2 - slope_3) * break_pt
    )
    x = np.arange(0, break_pt * 2, delta)
    y = np.concatenate((line_2, line_3))
    break_pt = break_pt

    return x, y, break_pt


def test__validate_with_synthetic_data(elbow_with_synthetic_data):
    """
    Is the elbow where we expect it? For clean, perfect data.
    """
    x, y, break_pt = elbow_with_synthetic_data
    expected_elbow = np.argmin(np.abs(x - break_pt))
    assert expected_elbow == find_elbow_point(x, y)


def test__validate_with_synthetic_data_plus_noise(elbow_with_synthetic_data):
    """
    Is the elbow where we expect it? For low noisy data. Use normally distributed
    noise, with standard deviation = 2% of the range of y.

    Elbow should be found within +/- 10 index positions from actual point
    """
    # Test with an odd number of points
    x, y, break_pt = elbow_with_synthetic_data
    np.random.seed(12)
    range = (y.max() - y.min()) * 0.02
    y = y + np.random.randn(y.shape[0]) * range
    expected_elbow = np.argmin(np.abs(x - break_pt))
    found_elbow = find_elbow_point(x, y)
    assert abs(found_elbow - expected_elbow) < 10

    # Test with an even number of points
    x, y, break_pt = elbow_with_synthetic_data
    range = (y.max() - y.min()) * 0.02
    y = y + np.random.randn(y.shape[0]) * range
    expected_elbow = np.argmin(np.abs(x - break_pt))
    found_elbow = find_elbow_point(x, y, max_iter=40)
    assert abs(found_elbow - expected_elbow) < 10


def test__corner_case():
    found_elbow = find_elbow_point(
        [np.nan] * 11,
        [np.nan] * 11,
    )
    assert found_elbow == -1
