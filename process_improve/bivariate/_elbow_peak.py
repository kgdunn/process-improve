"""
Bivariate statistical tools.

* elbow detection in an (x,y) plot
* peaks: finding peaks, quantifying their height, width, center, area, left & right boundaries
* area under curve
"""

import numpy as np

from ..regression.methods import fit_robust_lm


def find_elbow_point(x: np.ndarray, y: np.ndarray, max_iter: int = 41) -> int | float:
    """
    Find the elbow point when plotting numeric entries in `x` vs numeric values in list `y`.

    Return the index into the vectors `x` and `y` [the vectors must have the same length], where
    the elbow point occurs. Returns -1 if every value in `x` or `y` is missing.

    Using a robust linear fit, sorts the samples in X (independent variable)
    and takes the first 5 samples from the left, and the last 5 from the right,
    then fits two linear regressions and computes the intersection of the two
    fitted lines. The window size is then grown over `max_iter` (default 41)
    evenly spaced steps, via `numpy.linspace`, up to roughly half the data,
    accumulating one intersection point per step.

    The elbow is taken as the data point whose (x, y) location is closest to
    the median of the accumulated intersection points; the median location is
    where the intersections should stabilise.

    Will probably not work well on few data points. If so, try fitting a spline
    to the raw data and then repeat with the interpolated data.

    """
    start = 5
    # assert divmod(max_iter, 2)[1]  # must be odd number; to ensure we calculate the median later

    def calculate_line_length(x1: float, y1: float, x2: np.ndarray, y2: np.ndarray) -> float | np.ndarray:
        r"""Return the length of the line between 2 points (x1, y1) and (x2, y2).

        Defined as :math:`\\sqrt{(x2 - x1)^2 + (y2 - y1)^2}`.
        """
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Ensure it is a Numpy array: pandas objects and lists are correctly handled.
    x = np.array(x.copy())
    y = np.array(y.copy())
    if len(x) != len(y):
        raise ValueError(
            f"x and y must have the same length; got len(x)={len(x)}, len(y)={len(y)}."
        )

    # Stop if everything is missing:
    if np.isnan(np.nanmedian(x)) or np.isnan(np.nanmedian(y)):
        return -1

    # Eliminate missing values in x and y simultaneously.
    x, y = x[~(np.isnan(x) | np.isnan(y))], y[~(np.isnan(x) | np.isnan(y))]
    if len(x) <= 10:
        raise ValueError(
            "Requires more than 10 values in the vectors (not including missing data)."
        )
    idx_sort = x.argsort()
    x = x[idx_sort]
    y = y[idx_sort]
    N = x.size

    int_x_list: list[float] = []
    int_y_list: list[float] = []
    for i in np.floor(np.linspace(0, int(N / 2 - start) + 1, max_iter)):
        idx = int(start + i)
        lo_x, lo_y = x[0:idx], y[0:idx]
        hi_x, hi_y = x[-idx:], y[-idx:]
        lft = fit_robust_lm(lo_x, lo_y)
        rgt = fit_robust_lm(hi_x, hi_y)
        int_x, int_y = find_line_intersection(lft[1], lft[0], rgt[1], rgt[0])
        int_x_list.append(int_x)
        int_y_list.append(int_y)

    # The elbow is the raw data point closest to the consensus intersection
    # point. Taking the median of every accumulated intersection - in both x
    # and y - keeps the estimate robust to the occasional spurious window fit
    # (e.g. near-parallel lines yielding a NaN or far-off intersection),
    # instead of relying on a single, arbitrarily selected intersection.
    mid_x = np.nanmedian(int_x_list)
    mid_y = np.nanmedian(int_y_list)
    if np.isnan(mid_x) or np.isnan(mid_y):
        return np.nan

    return int(np.argmin(calculate_line_length(mid_x, mid_y, x, y)))


def find_line_intersection(m1: float, b1: float, m2: float, b2: float) -> tuple:
    """
    Find the intersection point of two lines.

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



# ENG-23 (#305): explicit ``__all__`` so the thin re-exporter ``methods.py``
# can do ``from ._elbow_peak import *`` without triggering CodeQL's
# py/polluting-import warning.
__all__ = [
    "find_elbow_point",
    "find_line_intersection",
    "fit_robust_lm",
]
