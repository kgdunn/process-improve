"""
Bivariate statistical tools:

* elbow detection in an (x,y) plot
* peaks: finding peaks, quantifying their height, width, center, area, left & right boundaries
* area under curve
"""

from typing import List, Union

import numpy as np

from ..regression.methods import fit_robust_lm


def find_elbow_point(x: np.ndarray, y: np.ndarray, max_iter=41) -> Union[int, float]:
    """
    Finds the elbow point when plotting numeric entries in `x` vs numeric values in list `y`.

    Returns the index into the vectors `x` and `y` [the vectors must have the same length], where
    the elbow point occurs.

    Using a robust linear fit, sorts the samples in X (independent variable)
    and takes sample 1:5 from the left, and samples (end-5):end and fits two
    linear regressions. Finds the angle between the two lines.
    Adds a point to each regression, so (1:6) and (end-6:end) and repeats.

    Finds the median angle, which is where it should stabilize.

    Will probably not work well on few data points. If so, try fitting a spline
    to the raw data and then repeat with the interpolated data.

    """
    start = 5
    # assert divmod(max_iter, 2)[1]  # must be odd number; to ensure we calculate the median later

    def calculate_line_length(
        x1: float, y1: float, x2: np.ndarray, y2: np.ndarray
    ) -> Union[float, np.ndarray]:
        """Returns the length of the line between 2 points (x1, y1) and (x2, y2), defined as:
        :math:`\\sqrt{(x2 - x1)^2 + (y2 - y1)^2}`
        """
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Ensure it is a Numpy array: pandas objects and lists are correctly handled.
    x = np.array(x.copy())
    y = np.array(y.copy())
    assert len(x) == len(y)

    # Stop if everything is missing:
    if np.isnan(np.nanmedian(x)) or np.isnan(np.nanmedian(y)):
        return -1

    # Eliminate missing values in x and y simultaneously.
    x, y = x[~(np.isnan(x) | np.isnan(y))], y[~(np.isnan(x) | np.isnan(y))]
    assert (
        len(x) > 10
    ), "Requires more than 10 values in the vectors (not including missing data)."
    idx_sort = x.argsort()
    x = x[idx_sort]
    y = y[idx_sort]
    N = x.size
    # Left and right anchor points: use median of the 5 points at start and end
    lft_x_avg = np.median(x[0:start])
    rgt_x_avg = np.median(x[-start:])

    int_x_list = []
    int_y_list = []
    lft_line_list = []  # type: List[float]
    rgt_line_list = []  # type: List[float]
    angle_list = []  # type: List[float]
    for i in np.floor(np.linspace(0, int(N / 2 - start) + 1, max_iter)):
        idx = int(start + i)
        lo_x, lo_y = x[0:idx], y[0:idx]
        hi_x, hi_y = x[-idx:], y[-idx:]
        lft = fit_robust_lm(lo_x, lo_y)
        rgt = fit_robust_lm(hi_x, hi_y)
        int_x, int_y = find_line_intersection(lft[1], lft[0], rgt[1], rgt[0])
        int_x_list.append(int_x)
        int_y_list.append(int_y)
        continue

        # The rest of this code is not reachable. Exploratory code: was used during development
        # only, and can be used to come up with alternative approaches of finding the elbow.
        if False:  # pragma: no cover
            # Left edge: (x=median(left 5 values), y = prediction from regression(x))
            lft_y = lft[0] + lft[1] * lft_x_avg

            # Right edge: (x=median(right 5 values), y = prediction from regression(x))
            rgt_y = rgt[0] + rgt[1] * rgt_x_avg

            lft_line = calculate_line_length(int_x, int_y, lft_x_avg, lft_y)
            rgt_line = calculate_line_length(int_x, int_y, rgt_x_avg, rgt_y)
            hypotenuse_line = calculate_line_length(lft_x_avg, lft_y, rgt_x_avg, rgt_y)
            lft_line_list.append(lft_line)
            rgt_line_list.append(rgt_line)

            angle = (
                np.arccos(
                    (lft_line ** 2 + rgt_line ** 2 - hypotenuse_line ** 2)
                    / (2 * lft_line * rgt_line)
                )
                * 180.0
                / np.pi
            )
            angle_list.append(angle)

    # Visualize the elbow point
    if False:
        import pandas as pd

        data = pd.DataFrame(data={"x": x, "y": y})
        ax = data.plot.scatter(x="x", y="y")

        intersections = pd.DataFrame(data={"x_int": int_x_list, "y_int": int_y_list})
        intersections.plot.scatter(x="x_int", y="y_int", ax=ax)
        pd.DataFrame(intersections.median()).T.plot.scatter(
            x="x_int", y="y_int", color="red", ax=ax
        )
        ax.grid(True)

    # Elbow point is taken as the average intersection point which is closest
    # to the raw data. Handle the case for even and odd number of data points
    if divmod(len(int_x_list), 2)[1] == 0:
        mid_idx = np.argmin((np.array(int_x_list) - np.nanmedian(int_x_list)) ** 2)
    else:
        mid_idx = np.where(int_x_list == np.nanmedian(int_x_list))
        if mid_idx[0].any():
            mid_idx = mid_idx[0][0]
        else:
            return np.nan

    mid_x = np.nanmedian(int_x_list)
    if np.isnan(mid_x):
        return np.nan

    # TODO: Could robustify it:
    # np.quantile(calculate_line_length(mid_x, mid_y, xraw, yraw), 0.05)
    mid_y = int_y_list[int(mid_idx)]
    return int(np.argmin(calculate_line_length(mid_x, mid_y, x, y)))


def find_line_intersection(m1: float, b1: float, m2: float, b2: float) -> tuple:
    """
    From Stackoverflow:
    stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines

    Returns a tuple: (x, y) where the two lines intersect, given slopes `m1` and
    `m2`, and intercepts `b1` and `b2`.
    """
    # These lines are essentially parallel!
    if np.abs(m1 - m2) < np.sqrt(np.finfo(float).eps):
        return np.nan, np.nan

    # y = mx + b
    # Derivation: Set both lines equal to find the intersection point in the x direction
    # m1 * x + b1 = m2 * x + b2
    # m1 * x - m2 * x = b2 - b1
    # x * (m1 - m2) = b2 - b1
    # Solving the above equation for x:
    x = (b2 - b1) / (m1 - m2)

    # Now solve it for y: use either line, because they are equal here:
    y = m1 * x + b1
    return x, y
