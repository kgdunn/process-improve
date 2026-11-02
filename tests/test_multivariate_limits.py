"""Confidence limits in ``multivariate/_limits.py``, each checked against an independent derivation.

The expected values here do not reuse the module's own formulas: the F quantile comes from the
Beta distribution, the SPE limit from the moment-matched Gamma distribution, and one value from a
printed F table. Mutation testing (``[tool.mutmut]`` in ``pyproject.toml``) found the boundaries
and defaults pinned below untested.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from process_improve.multivariate._limits import ellipse_coordinates, hotellings_t2_limit, spe_calculation, spe_limit
from process_improve.multivariate.methods import PCA, MCUVScaler


def f_quantile(q: float, dfn: int, dfd: int) -> float:
    """F quantile through the Beta distribution: X ~ Beta(dfn/2, dfd/2) gives dfd X / (dfn (1 - X)) ~ F."""
    x = stats.beta.ppf(q, dfn / 2, dfd / 2)
    return dfd * x / (dfn * (1 - x))


def t2_limit_by_hand(q: float, n_components: int, n_rows: int) -> float:
    """Return A (N - 1)(N + 1) / (N (N - A)) F(q; A, N - A), the T2 limit for a new observation."""
    a, n = n_components, n_rows
    return a * (n - 1) * (n + 1) / (n * (n - a)) * f_quantile(q, a, n - a)


# --------------------------------------------------------------------------- Hotelling's T2 limit
@pytest.mark.parametrize(("conf_level", "n_components", "n_rows"), [(0.95, 1, 10), (0.95, 3, 40), (0.99, 5, 200)])
def test_the_t2_limit_matches_the_f_quantile_derived_from_the_beta_distribution(conf_level, n_components, n_rows):
    """The T2 limit agrees with a T2 limit built from a Beta quantile, not scipy's F."""
    expected = t2_limit_by_hand(conf_level, n_components, n_rows)
    assert hotellings_t2_limit(conf_level, n_components, n_rows) == pytest.approx(expected, rel=1e-10)


def test_the_t2_limit_matches_a_printed_f_table():
    """A = 2, N = 20 at 95%: F(2, 18) = 3.5546 from a table, so the limit is 2 * 19 * 21 / (20 * 18) * 3.5546."""
    assert hotellings_t2_limit(0.95, 2, 20) == pytest.approx(2 * 19 * 21 / (20 * 18) * 3.5546, rel=1e-4)


def test_the_t2_limit_defaults_to_95_percent():
    """Leaving out ``conf_level`` gives the 95% limit."""
    assert hotellings_t2_limit(n_components=3, n_rows=30) == hotellings_t2_limit(0.95, 3, 30)


def test_with_no_components_the_t2_limit_is_zero_not_nan():
    """With A = 0 every T2 value is 0, and so is the limit; scipy's F with 0 degrees of freedom gave NaN."""
    assert hotellings_t2_limit(0.95, 0, 10) == 0.0
    assert hotellings_t2_limit(0.95, 0, 1) == 0.0  # a single row is a valid, if tiny, model
    assert hotellings_t2_limit(n_rows=10) == 0.0  # n_components defaults to 0


def test_as_many_components_as_rows_gives_an_infinite_limit():
    """A == N leaves no denominator degrees of freedom: nothing can exceed the limit."""
    assert hotellings_t2_limit(0.95, 1, 1) == float("inf")
    assert hotellings_t2_limit(0.95, 7, 7) == float("inf")


@pytest.mark.parametrize("conf_level", [0.0, 1.0, -0.1, 1.5])
def test_the_t2_limit_refuses_a_confidence_level_outside_the_open_interval(conf_level):
    """``conf_level`` must lie strictly between 0 and 1."""
    with pytest.raises(ValueError, match=r"conf_level must lie in \(0, 1\)"):
        hotellings_t2_limit(conf_level, 2, 20)


def test_the_t2_limit_refuses_bad_counts():
    """No rows, negative components, and more components than rows are each refused by name."""
    with pytest.raises(ValueError, match="n_rows must be positive; got 0"):
        hotellings_t2_limit(0.95, 2)  # n_rows is required in practice
    with pytest.raises(ValueError, match="n_components must be non-negative; got -1"):
        hotellings_t2_limit(0.95, -1, 10)
    with pytest.raises(ValueError, match=r"n_components \(11\) cannot exceed n_rows \(10\); a model cannot have more"):
        hotellings_t2_limit(0.95, 11, 10)


# --------------------------------------------------------------------------- SPE limit
def moment_matched_limit(spe: np.ndarray, conf_level: float) -> float:
    """Square root of the quantile of the Gamma distribution with the mean and variance of SPE squared."""
    squares = spe**2
    mean, variance = squares.mean(), squares.var(ddof=1)
    return float(np.sqrt(stats.gamma.ppf(conf_level, a=mean**2 / variance, scale=variance / mean)))


@pytest.mark.parametrize("conf_level", [0.9, 0.95, 0.99])
def test_the_spe_limit_is_the_moment_matched_gamma_quantile(conf_level):
    """The scaled chi-square g * chi2(h) with matched moments is Gamma(m^2 / v, v / m) for SPE squared."""
    spe = np.sqrt(np.random.default_rng(7).chisquare(4, size=60))
    assert spe_calculation(spe, conf_level) == pytest.approx(moment_matched_limit(spe, conf_level), rel=1e-10)


def test_the_spe_limit_defaults_to_95_percent():
    """Leaving out ``conf_level`` gives the 95% limit, for the calculation and for a fitted model."""
    spe = np.sqrt(np.random.default_rng(3).chisquare(3, size=40))
    assert spe_calculation(spe) == spe_calculation(spe, 0.95)
    rng = np.random.default_rng(11)
    X = pd.DataFrame(rng.normal(size=(30, 5)) @ rng.normal(size=(5, 5)), columns=list("abcde"))
    model = PCA(n_components=2).fit(MCUVScaler().fit_transform(X))
    assert spe_limit(model) == spe_limit(model, 0.95)
    assert spe_limit(model) == pytest.approx(moment_matched_limit(model.spe_.iloc[:, 1].to_numpy(), 0.95), rel=1e-10)


def test_spe_values_with_no_spread_give_their_own_level_as_the_limit():
    """Equal SPE values have no spread to bound; all zero means a perfect fit."""
    assert spe_calculation(np.full(10, 2.0)) == pytest.approx(2.0)
    assert spe_calculation(np.zeros(10)) == 0.0


@pytest.mark.parametrize("conf_level", [0.0, 1.0])
def test_the_spe_limit_refuses_a_confidence_level_outside_the_open_interval(conf_level):
    """``conf_level`` must lie strictly between 0 and 1."""
    with pytest.raises(ValueError, match=r"conf_level must lie in \(0, 1\)"):
        spe_calculation(np.array([1.0, 2.0, 3.0]), conf_level)


# --------------------------------------------------------------------------- T2 ellipse
SCALES = pd.Series([3.0, 2.0, 0.5])


def test_the_ellipse_has_the_two_score_t2_radius_on_each_axis():
    """At angle 0 and pi / 2 the ellipse touches sqrt(T2(2, N)) times each score's standard deviation."""
    x, y = ellipse_coordinates(1, 2, 0.95, n_points=5, n_components=3, scaling_factor_for_scores=SCALES, n_rows=25)
    radius = np.sqrt(t2_limit_by_hand(0.95, 2, 25))
    assert x[0] == pytest.approx(radius * 3.0)
    assert y[1] == pytest.approx(radius * 2.0)  # 5 points: angles 0, pi/2, pi, 3pi/2, 2pi
    assert np.allclose((x / 3.0) ** 2 + (y / 2.0) ** 2, radius**2)


def test_the_ellipse_accepts_the_last_component_on_either_axis():
    """Score indices run from 1 to A inclusive."""
    x, y = ellipse_coordinates(3, 1, 0.95, n_points=3, n_components=3, scaling_factor_for_scores=SCALES, n_rows=25)
    radius = np.sqrt(t2_limit_by_hand(0.95, 2, 25))
    assert x[0] == pytest.approx(radius * 0.5)  # the third score's scale
    assert y[0] == pytest.approx(0.0)


def test_two_points_is_the_smallest_ellipse_drawn():
    """Two points are allowed: the start and the end, which coincide."""
    x, y = ellipse_coordinates(1, 2, 0.95, n_points=2, n_components=2, scaling_factor_for_scores=SCALES, n_rows=25)
    assert len(x) == len(y) == 2
    assert x[0] == pytest.approx(x[1])
    with pytest.raises(ValueError, match="n_points must be >= 2 to draw an ellipse; got 1"):
        ellipse_coordinates(1, 2, 0.95, n_points=1, n_components=2, scaling_factor_for_scores=SCALES, n_rows=25)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"score_horiz": 0}, r"score_horiz must lie in \[1, 3\]; got 0"),
        ({"score_horiz": 4}, r"score_horiz must lie in \[1, 3\]; got 4"),
        ({"score_vert": 0}, r"score_vert must lie in \[1, 3\]; got 0"),
        ({"score_vert": 4}, r"score_vert must lie in \[1, 3\]; got 4"),
        ({"conf_level": 0.0}, r"conf_level must lie in \(0, 1\); got 0.0"),
        ({"conf_level": 1.0}, r"conf_level must lie in \(0, 1\); got 1.0"),
        ({"n_rows": 0}, "n_rows must be positive; got 0"),
    ],
)
def test_the_ellipse_refuses_each_bad_argument_by_name(kwargs, message):
    """Each argument outside its range is refused with a message that names it."""
    arguments = {"score_horiz": 1, "score_vert": 2, "conf_level": 0.95, "n_components": 3, "n_rows": 25}
    with pytest.raises(ValueError, match=message):
        ellipse_coordinates(**{**arguments, **kwargs}, scaling_factor_for_scores=SCALES)


def test_the_ellipse_needs_the_component_count_and_the_row_count():
    """``n_components`` and ``n_rows`` default to 0, so leaving either out is refused by name."""
    with pytest.raises(ValueError, match=r"score_horiz must lie in \[1, 0\]"):
        ellipse_coordinates(1, 1, scaling_factor_for_scores=SCALES, n_rows=25)
    with pytest.raises(ValueError, match="n_rows must be positive; got 0"):
        ellipse_coordinates(1, 2, n_components=2, scaling_factor_for_scores=SCALES)
