import numpy as np
import pytest

from process_improve.bivariate.methods import find_elbow_point, find_line_intersection
from process_improve.bivariate.tools import get_bivariate_tool_specs
from process_improve.tool_safety import ToolInputInvalidError
from process_improve.tool_spec import execute_tool_call


@pytest.fixture
def straight_line() -> tuple:
    """Create straight line test data."""
    slope = 2.5
    intercept = -13
    delta = 0.1
    x = np.arange(0, 5, delta)
    y0 = np.zeros_like(x)
    y1 = slope * x + intercept
    y2 = np.ones_like(x)
    return x, y0, y1, y2


def test__validate_nan_output(straight_line: tuple) -> None:
    """Is the elbow where we expect it? For clean, perfect data."""
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
def elbow_with_synthetic_data() -> tuple:
    """Create simulated data to test against.

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
    line_3 = np.arange(break_pt, break_pt * 2, delta) * slope_3 + (slope_2 - slope_3) * break_pt
    x = np.arange(0, break_pt * 2, delta)
    y = np.concatenate((line_2, line_3))

    return x, y, break_pt


def test__validate_with_synthetic_data(elbow_with_synthetic_data: tuple) -> None:
    """Is the elbow where we expect it? For clean, perfect data."""
    x, y, break_pt = elbow_with_synthetic_data
    expected_elbow = np.argmin(np.abs(x - break_pt))
    assert expected_elbow == find_elbow_point(x, y)


def test__validate_with_synthetic_data_plus_noise(elbow_with_synthetic_data: tuple) -> None:
    """Is the elbow where we expect it? For low noisy data.

    Use normally distributed noise, with standard deviation = 2% of the range of y.

    Elbow should be found within +/- 10 index positions from actual point
    """
    # Test with an odd number of points
    x, y, break_pt = elbow_with_synthetic_data
    rng = np.random.default_rng(12)
    y_range = (y.max() - y.min()) * 0.02
    y = y + rng.standard_normal(y.shape[0]) * y_range
    expected_elbow = np.argmin(np.abs(x - break_pt))
    found_elbow = find_elbow_point(x, y)
    assert abs(found_elbow - expected_elbow) < 10

    # Test with an even number of points
    x, y, break_pt = elbow_with_synthetic_data
    y_range = (y.max() - y.min()) * 0.02
    y = y + rng.standard_normal(y.shape[0]) * y_range
    expected_elbow = np.argmin(np.abs(x - break_pt))
    found_elbow = find_elbow_point(x, y, max_iter=40)
    assert abs(found_elbow - expected_elbow) < 10


def test__robust_to_gross_outliers(elbow_with_synthetic_data: tuple) -> None:
    """A handful of gross outliers should not move the detected elbow far."""
    x, y, break_pt = elbow_with_synthetic_data
    y = y.copy()
    rng = np.random.default_rng(3)
    outlier_idx = rng.choice(len(y), size=5, replace=False)
    y[outlier_idx] += 50.0
    expected_elbow = np.argmin(np.abs(x - break_pt))
    found_elbow = find_elbow_point(x, y)
    assert abs(found_elbow - expected_elbow) < 10


def test__corner_case() -> None:
    """Test corner case with all NaN input."""
    found_elbow = find_elbow_point(
        [np.nan] * 11,
        [np.nan] * 11,
    )
    assert found_elbow == -1


def test__mismatched_lengths_raise() -> None:
    """Vectors of different lengths raise a clear ValueError."""
    with pytest.raises(ValueError, match="same length"):
        find_elbow_point(np.arange(12.0), np.arange(11.0))


def test__too_few_non_missing_values_raise() -> None:
    """Ten or fewer non-missing pairs raise, even if the raw vectors are longer."""
    x = np.arange(13.0)
    y = 2.0 * x
    y[[2, 5, 8]] = np.nan  # 10 non-missing pairs remain
    with pytest.raises(ValueError, match="more than 10 values"):
        find_elbow_point(x, y)


def test_find_line_intersection_parallel_lines() -> None:
    """Essentially parallel lines have no intersection: returns (nan, nan)."""
    x, y = find_line_intersection(m1=2.0, b1=0.0, m2=2.0, b2=5.0)
    assert np.isnan(x)
    assert np.isnan(y)


def test_find_line_intersection_known_point() -> None:
    """Two crossing lines intersect where algebra says they should."""
    # y = x and y = -x + 4 cross at (2, 2).
    x, y = find_line_intersection(m1=1.0, b1=0.0, m2=-1.0, b2=4.0)
    assert x == pytest.approx(2.0)
    assert y == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Agent-tool wrapper: process_improve.bivariate.tools
# ---------------------------------------------------------------------------


def test_find_elbow_tool_returns_index_on_clean_data(elbow_with_synthetic_data: tuple) -> None:
    """The tool wrapper returns an integer index plus the (x, y) at the elbow."""
    x, y, break_pt = elbow_with_synthetic_data
    result = execute_tool_call("find_elbow", {"x": list(x), "y": list(y)})

    assert "error" not in result
    assert result["n"] == len(x)
    expected_idx = int(np.argmin(np.abs(x - break_pt)))
    assert result["elbow_index"] == expected_idx
    assert result["elbow_x"] == x[expected_idx]
    assert result["elbow_y"] == y[expected_idx]


def test_find_elbow_tool_reports_error_on_too_few_points() -> None:
    """Inputs shorter than the pydantic min_length raise ToolInputInvalidError (ENG-04)."""
    with pytest.raises(ToolInputInvalidError):
        execute_tool_call("find_elbow", {"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})


def test_get_bivariate_tool_specs_lists_find_elbow() -> None:
    """The module-level convenience returns at least the find_elbow spec."""
    specs = get_bivariate_tool_specs()
    assert isinstance(specs, list)
    names = {spec.get("name") for spec in specs}
    assert "find_elbow" in names
