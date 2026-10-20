# (c) Kevin Dunn, 2010-2026. MIT License.
r"""Classical tests and the percentage points of their distributions.

The percentage points take a cumulative probability ``p`` first, as :func:`t_value` does, so the
upper 5% point of :math:`F` on 3 and 16 degrees of freedom, :math:`F_{0.05,3,16}` in most tables,
is ``f_value(0.95, 3, 16)``.

The tests are the ones a first course in designed experiments uses before any model is fitted:
a mean against a stated value, by :math:`t` or with :math:`\sigma` known by :math:`z`; two groups by
Wilcoxon's rank sum; and two groups, or pairs, by a randomization test that enumerates every way
the labels could have fallen, which is what Fisher's argument for randomization rests on. Several
groups are compared by the Kruskal-Wallis test, and their variances by Bartlett's test; a variance
has its chi-square interval. The power of the one-way analysis of variance, and the smallest
number of runs a group needs to reach a stated power, are worked from the noncentral :math:`F`.
The autocorrelation of a record in time order says whether its observations can be treated as
independent.

Examples
--------
>>> from process_improve.univariate.classical import f_value, randomization_test
>>> round(f_value(0.95, 3, 16), 4)
3.2389
>>> found = randomization_test([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
>>> found["n_arrangements"], found["n_at_least"], round(found["p_value"], 3)
(20, 1, 0.05)
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import integrate, optimize, stats

from process_improve._random import check_random_state

ALTERNATIVES = ("two-sided", "greater", "less")
#: Beyond this many arrangements, a randomization test samples them instead of enumerating them all.
MOST_ARRANGEMENTS = 2_000_000
_TIES = 1e-9


def f_value(p: float, df1: float, df2: float) -> float:
    """Return the point of the F distribution on ``df1`` and ``df2`` degrees of freedom below which ``p`` lies."""
    return float(stats.f.ppf(p, df1, df2))


def chi2_value(p: float, df: float) -> float:
    """Return the point of the chi-squared distribution on ``df`` degrees of freedom below which ``p`` lies."""
    return float(stats.chi2.ppf(p, df))


def z_value(p: float) -> float:
    """Return the point of the standard normal distribution below which ``p`` lies."""
    return float(stats.norm.ppf(p))


def studentized_range_value(p: float, k: int, df: float) -> float:
    r"""Return the point of the studentized range of ``k`` means on ``df`` degrees of freedom below which ``p`` lies.

    Tukey's critical difference between two of ``k`` means of ``n`` observations each is this point,
    at ``p`` = 0.95 for 95%, times :math:`\sqrt{MS_E / n}`.
    """
    return float(stats.studentized_range.ppf(p, k, df))


def _dunnett_probability(d: float, k: int, df: float, correlation: float, two_sided: bool) -> float:
    """Return the probability that all ``k`` comparisons with the control lie within ``d`` (or below it)."""
    nodes, weights = np.polynomial.hermite_e.hermegauss(48)
    weights = weights / math.sqrt(2 * math.pi)
    root, rest = math.sqrt(correlation), math.sqrt(1 - correlation)

    def given_scale(s: float) -> float:
        upper = stats.norm.cdf((d * s - root * nodes) / rest)
        inside = upper - stats.norm.cdf((-d * s - root * nodes) / rest) if two_sided else upper
        return float(np.sum(weights * inside**k))

    if math.isinf(df):
        return given_scale(1.0)
    density = stats.chi(df, scale=1 / math.sqrt(df))  # s, the estimated standard deviation over the true one
    return float(integrate.quad(lambda s: given_scale(s) * density.pdf(s), 0, np.inf, limit=200)[0])


def dunnett_value(p: float, k: int, df: float, alternative: str = "two-sided", correlation: float = 0.5) -> float:
    r"""Return Dunnett's critical value: the point of the largest of ``k`` t statistics against one control.

    Each of ``k`` treatments is compared with one control on ``df`` error degrees of freedom; with
    equal numbers of runs the comparisons are correlated by one half (``correlation``). The value
    ``d`` has probability ``p`` that every comparison lies within :math:`\pm d` (two-sided) or below
    ``d`` (``alternative="greater"``) when no treatment differs from the control. The critical
    difference is ``d`` times the standard error of a difference, :math:`\sqrt{2 MS_E / n}`.

    Parameters
    ----------
    p : float
        The cumulative probability, 0.95 for a 5% test.
    k : int
        The number of treatments compared with the control.
    df : float
        The error degrees of freedom; ``numpy.inf`` for a known variance.
    alternative : {"two-sided", "greater"}
        Whether a treatment may differ from the control in either direction or only exceed it.
    correlation : float
        The correlation of two comparisons, :math:`n_0 / (n_0 + n)` for a control of :math:`n_0` runs
        and treatments of :math:`n`; 0.5 when they are equal.

    Returns
    -------
    float
        The critical value ``d``.
    """
    if alternative not in ("two-sided", "greater"):
        raise ValueError(f"alternative must be 'two-sided' or 'greater'; got {alternative!r}.")
    if not 0 < p < 1 or k < 1 or not 0 <= correlation < 1:
        raise ValueError("p must lie between 0 and 1, k must be at least 1, and correlation in [0, 1).")
    two_sided = alternative == "two-sided"
    return float(
        optimize.brentq(lambda d: _dunnett_probability(d, k, df, correlation, two_sided) - p, 1e-6, 50, xtol=1e-10)
    )


def _p_values(statistic: float, distribution: Any) -> dict[str, float]:  # noqa: ANN401 - a frozen scipy distribution
    """Return the two-sided and one-sided p-values of a statistic from a symmetric distribution."""
    upper = float(distribution.sf(statistic))
    lower = float(distribution.cdf(statistic))
    return {"p_value": 2 * min(upper, lower), "p_value_greater": upper, "p_value_less": lower}


def ttest_one_sample(x: Sequence[float], mu: float = 0.0, conflevel: float = 0.95) -> dict[str, float]:
    r"""Test a mean against a stated value by Student's t, with its confidence interval.

    Parameters
    ----------
    x : sequence of float
        The observations.
    mu : float
        The stated value of the mean.
    conflevel : float
        The confidence level of the interval.

    Returns
    -------
    dict[str, float]
        ``n``, ``mean``, ``std_dev``, ``std_error`` (:math:`s / \sqrt{n}`), ``t_value``, ``df``, the
        two-sided ``p_value`` and the one-sided ``p_value_greater`` and ``p_value_less``, and
        ``ci_low`` and ``ci_high``.
    """
    values = np.asarray(x, dtype=float)
    n = len(values)
    if n < 2:
        raise ValueError("A t-test needs at least two observations.")
    mean, std_dev = float(values.mean()), float(values.std(ddof=1))
    std_error = std_dev / math.sqrt(n)
    t_value = (mean - mu) / std_error
    half = float(stats.t.ppf(0.5 + conflevel / 2, n - 1)) * std_error
    return {
        "n": n,
        "mean": mean,
        "std_dev": std_dev,
        "std_error": std_error,
        "t_value": t_value,
        "df": n - 1,
        **_p_values(t_value, stats.t(n - 1)),
        "ci_low": mean - half,
        "ci_high": mean + half,
    }


def ztest(
    x: Sequence[float], y: Sequence[float] | None = None, sigma: float = 1.0, mu: float = 0.0
) -> dict[str, float]:
    r"""Test a mean, or the difference of two means, by z, with the standard deviation known.

    With ``y`` the difference is :math:`\bar y - \bar x`, with standard error
    :math:`\sigma \sqrt{1/n_x + 1/n_y}`; without it, :math:`\bar x` is tested, with standard error
    :math:`\sigma / \sqrt{n_x}`. Either is compared with ``mu``.

    Returns
    -------
    dict[str, float]
        ``difference`` (the mean, or the difference of means), ``std_error``, ``z_value``, the
        two-sided ``p_value`` and the one-sided ``p_value_greater`` and ``p_value_less``.
    """
    if not sigma > 0:
        raise ValueError(f"sigma must be positive; got {sigma!r}.")
    first = np.asarray(x, dtype=float)
    if y is None:
        difference, std_error = float(first.mean()), sigma / math.sqrt(len(first))
    else:
        second = np.asarray(y, dtype=float)
        difference = float(second.mean() - first.mean())
        std_error = sigma * math.sqrt(1 / len(first) + 1 / len(second))
    z = (difference - mu) / std_error
    return {"difference": difference, "std_error": std_error, "z_value": z, **_p_values(z, stats.norm())}


def _arrangements(n: int, size: int, limit: int, rng: np.random.Generator) -> tuple[np.ndarray, bool]:
    """Return every subset of ``size`` of ``range(n)``, one per row, or a random sample of ``limit`` of them."""
    total = math.comb(n, size)
    if total <= limit:
        combos = np.fromiter(
            itertools.chain.from_iterable(itertools.combinations(range(n), size)), dtype=np.int64, count=total * size
        ).reshape(total, size)
        return combos, True
    return np.argsort(rng.random((limit, n)), axis=1)[:, :size], False


def _arrangement_sums(pooled: np.ndarray, size: int, limit: int, rng: np.random.Generator) -> tuple[np.ndarray, bool]:
    """Return the sum of every subset of ``size`` of the pooled values, or of a random sample of subsets."""
    subsets, exact = _arrangements(len(pooled), size, limit, rng)
    return pooled[subsets].sum(axis=1), exact


def _sign_patterns(n: int, limit: int, rng: np.random.Generator) -> tuple[np.ndarray, bool]:
    """Return every pattern of n signs, one per row, or a random sample of ``limit`` of them."""
    if 2**n <= limit:
        return np.array(list(itertools.product((1.0, -1.0), repeat=n))).reshape(2**n, n), True
    return rng.choice((1.0, -1.0), size=(limit, n)), False


def _counts(found: np.ndarray, observed: float, scale: float) -> dict[str, Any]:
    """Count the arrangements above, at and below the observed statistic, and their p-values."""
    tolerance = _TIES * max(1.0, scale)
    greater = int(np.count_nonzero(found > observed + tolerance))
    equal = int(np.count_nonzero(np.abs(found - observed) <= tolerance))
    less = len(found) - greater - equal
    extreme = int(np.count_nonzero(np.abs(found) >= abs(observed) - tolerance))
    n = len(found)
    return {
        "n_arrangements": n,
        "n_greater": greater,
        "n_equal": equal,
        "n_less": less,
        "n_at_least": greater + equal,
        "p_value": (greater + equal) / n,
        "p_value_mid": (greater + equal / 2) / n,
        "p_value_less": (less + equal) / n,
        "p_value_two_sided": extreme / n,
    }


def randomization_test(
    x: Sequence[float],
    y: Sequence[float] | None = None,
    paired: bool = False,
    max_arrangements: int = MOST_ARRANGEMENTS,
    random_state: int | np.random.Generator | None = None,
) -> dict[str, Any]:
    r"""Compare two groups, or pairs, against every arrangement the randomization could have made.

    For two groups the statistic is :math:`\bar y - \bar x`, and the arrangements are every way of
    splitting the pooled observations into groups of the two sizes. For pairs (``paired=True``,
    ``y`` the second of each pair, or ``y=None`` with ``x`` the differences) the statistic is the
    mean difference, and the arrangements are every pattern of signs on the differences. Under the
    null hypothesis that the treatments do not differ, each arrangement was as likely as the one
    observed. When there are more than ``max_arrangements``, that many are drawn at random instead.
    To test a stated difference, give the differences less it, or ``y`` less it for two groups;
    :func:`randomization_interval` gives every difference this test does not reject.

    Returns
    -------
    dict[str, Any]
        ``statistic``, the observed difference; ``n_arrangements``; ``n_greater``, ``n_equal`` and
        ``n_less``, the arrangements whose statistic is above, equal to and below the observed one;
        ``n_at_least``, the first two together; the one-sided ``p_value`` (the share at least as large
        as the observed statistic, itself included), ``p_value_mid`` (the ties counted as half, as
        Box, Hunter and Hunter count them), ``p_value_less`` and ``p_value_two_sided`` (at least as
        far from zero); and ``exact``, false when the arrangements were sampled.
    """
    rng = check_random_state(random_state)
    first = np.asarray(x, dtype=float)
    if paired or y is None:
        differences = first if y is None else np.asarray(y, dtype=float) - first
        signs, exact = _sign_patterns(len(differences), max_arrangements, rng)
        found = (signs * np.abs(differences)).mean(axis=1)
        observed = float(differences.mean())
        scale = float(np.abs(differences).max(initial=0.0))
    else:
        second = np.asarray(y, dtype=float)
        pooled = np.concatenate([first, second])
        sums, exact = _arrangement_sums(pooled, len(second), max_arrangements, rng)
        found = sums / len(second) - (pooled.sum() - sums) / len(first)
        observed = float(second.mean() - first.mean())
        scale = float(np.abs(pooled).max(initial=0.0))
    return {"statistic": observed, **_counts(found, observed, scale), "exact": exact}


def randomization_interval(
    x: Sequence[float],
    y: Sequence[float] | None = None,
    paired: bool = False,
    confidence: float = 0.95,
    random_state: int | np.random.Generator | None = None,
) -> dict[str, Any]:
    r"""Return the confidence interval for a difference that inverts the randomization test.

    The test of a stated difference :math:`\delta` is :func:`randomization_test` of the
    observations shifted by it: the differences less :math:`\delta` for pairs, or ``y`` less
    :math:`\delta` for two groups. The interval holds every :math:`\delta` that neither one-sided
    test rejects at level :math:`(1 - \text{confidence}) / 2`, counting an arrangement whose
    statistic ties the observed one as at least as extreme, as ``p_value`` does.

    Each arrangement other than the observed one passes the observed statistic at one value of
    :math:`\delta`, its breakpoint. For pairs this is the mean of the differences whose signs it
    changes; for two groups, the mean of the ``y`` it moves less the mean of the ``x`` it moves.
    Of the :math:`M` arrangements, the number at least as large as the observed statistic is 1
    plus the number of breakpoints at or below :math:`\delta`. So the limits are the
    :math:`m`-th smallest and the :math:`m`-th largest breakpoint, with
    :math:`m = \lfloor M (1 - \text{confidence}) / 2 \rfloor` (Hartigan, 1969, for pairs), and the
    interval's own confidence, :math:`1 - 2m/M`, is the largest the arrangements allow that does
    not exceed the one asked for. When there are more than
    :data:`MOST_ARRANGEMENTS` arrangements, that many are drawn at random instead.

    Parameters
    ----------
    x : sequence of float
        The first group, or the first of each pair, or, with ``y=None``, the differences.
    y : sequence of float or None
        The second group, or the second of each pair.
    paired : bool
        Whether ``x`` and ``y`` are pairs.
    confidence : float
        The confidence asked for, between 0 and 1.
    random_state : int, numpy.random.Generator or None
        Seeds the sample of arrangements, when they are too many to enumerate.

    Returns
    -------
    dict[str, Any]
        ``statistic`` (the mean difference, or :math:`\bar y - \bar x`), ``ci_low`` and
        ``ci_high``, ``confidence`` (the interval's own, :math:`1 - 2m/M`), ``n_arrangements``
        and ``exact``, false when the arrangements were sampled.

    Raises
    ------
    ValueError
        If ``confidence`` is not between 0 and 1, a group is empty, or there are too few
        arrangements for any interval at that confidence: a level of :math:`(1 - c)/2` needs at
        least :math:`2/(1 - c)` of them.

    Examples
    --------
    Box, Hunter and Hunter's boys' shoes (Section 4.2): ten differences in wear, B less A, give
    1024 patterns of signs.

    >>> from process_improve.univariate.classical import randomization_interval
    >>> found = randomization_interval([0.8, 0.6, 0.3, -0.1, 1.1, -0.2, 0.3, 0.5, 0.5, 0.3])
    >>> round(found["ci_low"], 3), round(found["ci_high"], 3), round(found["confidence"], 4)
    (0.125, 0.7, 0.9512)
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be between 0 and 1; got {confidence!r}.")
    rng = check_random_state(random_state)
    first = np.asarray(x, dtype=float)
    second = None if y is None else np.asarray(y, dtype=float)
    if not all(len(v) and np.isfinite(v).all() for v in (first, second) if v is not None):
        raise ValueError("Each group, or the differences, needs a value, and every value must be a finite number.")
    if paired or second is None:
        differences = first if second is None else second - first
        signs, exact = _sign_patterns(len(differences), MOST_ARRANGEMENTS, rng)
        moved, observed, n_arrangements = signs < 0, float(differences.mean()), len(signs)
        breakpoints = _breakpoints(moved @ differences, moved.sum(axis=1))
    else:
        pooled = np.concatenate([first, second])
        subsets, exact = _arrangements(len(pooled), len(second), MOST_ARRANGEMENTS, rng)
        observed, n_arrangements = float(second.mean() - first.mean()), len(subsets)
        breakpoints = _breakpoints(second.sum() - pooled[subsets].sum(axis=1), (subsets < len(first)).sum(axis=1))
    m = math.floor(n_arrangements * (1 - confidence) / 2)
    if m < 1:
        most = 1 - 2 / n_arrangements
        msg = f"{n_arrangements} arrangements allow no {confidence:.1%} interval: at most {most:.1%}."
        raise ValueError(msg)
    return {
        "statistic": observed,
        "ci_low": float(breakpoints[m - 1]),
        "ci_high": float(breakpoints[len(breakpoints) - m]),
        "confidence": 1 - 2 * m / n_arrangements,
        "n_arrangements": n_arrangements,
        "exact": exact,
    }


def _breakpoints(moved_sums: np.ndarray, moved_counts: np.ndarray) -> np.ndarray:
    """Return, sorted, the difference at which each arrangement that moves something passes the observed one."""
    moves = moved_counts > 0
    return np.sort(moved_sums[moves] / moved_counts[moves])


def rank_sum_test(
    x: Sequence[float],
    y: Sequence[float],
    tie_correction: bool = True,
    max_arrangements: int = MOST_ARRANGEMENTS,
) -> dict[str, Any]:
    r"""Compare two groups by Wilcoxon's rank sum: the normal approximation and, for small groups, the exact level.

    The observations are ranked together, ties taking the average of their ranks. The rank sum of
    the first group, :math:`W_x`, has mean :math:`n_x (N + 1) / 2` and variance
    :math:`n_x n_y (N + 1) / 12` when the groups do not differ, less a correction for ties
    (``tie_correction``). The exact level enumerates every split of the ranks, when there are no
    more than ``max_arrangements`` of them.

    Returns
    -------
    dict[str, Any]
        ``rank_sum_x`` and ``rank_sum_y``; ``u_statistic``, :math:`W_x - n_x (n_x + 1) / 2`;
        ``z_value``, :math:`(W_x - E W_x) / \sqrt{\operatorname{Var} W_x}`, and its two-sided
        ``p_value`` and one-sided ``p_value_greater`` and ``p_value_less``; and, when the groups are
        small enough, ``exact_p_value_less`` and ``exact_p_value_greater``, the shares of splits
        whose rank sum for the first group is at most, or at least, the observed one.
    """
    first, second = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ranks = stats.rankdata(np.concatenate([first, second]))
    n_x, n_y = len(first), len(second)
    n = n_x + n_y
    w_x = float(ranks[:n_x].sum())
    variance = n_x * n_y * (n + 1) / 12
    if tie_correction:
        _, counts = np.unique(ranks, return_counts=True)
        variance *= 1 - float(np.sum(counts**3 - counts)) / (n**3 - n)
    z = (w_x - n_x * (n + 1) / 2) / math.sqrt(variance)
    result: dict[str, Any] = {
        "rank_sum_x": w_x,
        "rank_sum_y": float(ranks[n_x:].sum()),
        "u_statistic": w_x - n_x * (n_x + 1) / 2,
        "z_value": z,
        **_p_values(z, stats.norm()),
    }
    if math.comb(n, n_x) <= max_arrangements:
        sums, _ = _arrangement_sums(ranks, n_x, max_arrangements, np.random.default_rng(0))
        result["exact_p_value_less"] = float(np.mean(sums <= w_x + _TIES))
        result["exact_p_value_greater"] = float(np.mean(sums >= w_x - _TIES))
    return result


def _groups(groups: tuple[Sequence[float], ...], least: int) -> list[np.ndarray]:
    """Return the groups as float arrays; raise ``ValueError`` for fewer than two, or a group under ``least``."""
    arrays = [np.asarray(group, dtype=float) for group in groups]
    if len(arrays) < 2:
        raise ValueError(f"Two or more groups are needed; got {len(arrays)}.")
    if any(len(group) < least for group in arrays):
        raise ValueError(f"Each group needs at least {least} observation(s); got sizes {[len(g) for g in arrays]}.")
    return arrays


def bartlett_test(*groups: Sequence[float]) -> dict[str, Any]:
    r"""Test whether groups share one variance, by Bartlett's test.

    With :math:`a` groups of :math:`n_i` observations and variances :math:`S_i^2`, pooled to
    :math:`S_p^2` on :math:`N - a` degrees of freedom (Montgomery, section 3.4.3),

    .. math::

        q = (N - a) \log_{10} S_p^2 - \sum_i (n_i - 1) \log_{10} S_i^2, \qquad
        c = 1 + \frac{1}{3 (a - 1)} \left( \sum_i \frac{1}{n_i - 1} - \frac{1}{N - a} \right),

    and :math:`\chi^2_0 = \ln(10)\, q / c` on :math:`a - 1` degrees of freedom. The test is sensitive
    to a departure from normality, which can look like unequal variances.

    Returns
    -------
    dict[str, Any]
        ``statistic`` (:math:`\chi^2_0`), ``df``, ``p_value``, ``q``, ``c``, ``pooled_variance`` and
        the groups' ``variances``.

    Examples
    --------
    Montgomery's etch rates at four RF powers (Example 3.4):

    >>> found = bartlett_test(
    ...     [575, 542, 530, 539, 570], [565, 593, 590, 579, 610], [600, 651, 610, 637, 629], [725, 700, 715, 685, 710]
    ... )
    >>> round(found["q"], 2), round(found["c"], 2), round(found["statistic"], 2), round(found["p_value"], 3)
    (0.21, 1.1, 0.43, 0.933)
    """
    arrays = _groups(groups, 2)
    a, sizes = len(arrays), np.array([len(group) for group in arrays], dtype=float)
    variances = np.array([group.var(ddof=1) for group in arrays])
    dof = sizes - 1
    pooled = float(dof @ variances / dof.sum())
    q = float(dof.sum() * math.log10(pooled) - dof @ np.log10(variances))
    c = 1 + (float(np.sum(1 / dof)) - 1 / float(dof.sum())) / (3 * (a - 1))
    statistic = math.log(10) * q / c
    return {
        "statistic": statistic,
        "df": a - 1,
        "p_value": float(stats.chi2.sf(statistic, a - 1)),
        "q": q,
        "c": c,
        "pooled_variance": pooled,
        "variances": variances.tolist(),
    }


def kruskal_wallis_test(*groups: Sequence[float]) -> dict[str, Any]:
    r"""Compare groups by the Kruskal-Wallis test: the analysis of variance of their ranks.

    The observations are ranked together, ties taking the average of their ranks. With
    :math:`R_i` the rank sum of a group of :math:`n_i` (Montgomery, section 3.11.1),

    .. math::

        H = \frac{1}{S^2} \left( \sum_i \frac{R_i^2}{n_i} - \frac{N (N + 1)^2}{4} \right), \qquad
        S^2 = \frac{1}{N - 1} \left( \sum r^2 - \frac{N (N + 1)^2}{4} \right),

    the groups' sum of squares of the ranks over the ranks' variance, which allows for ties. It is
    referred to :math:`\chi^2` on :math:`a - 1` degrees of freedom.

    Returns
    -------
    dict[str, Any]
        ``statistic`` (:math:`H`), ``df``, ``p_value``, and the groups' ``rank_sums`` and
        ``mean_ranks``.
    """
    arrays = _groups(groups, 1)
    sizes = np.array([len(group) for group in arrays], dtype=float)
    ranks = stats.rankdata(np.concatenate(arrays))
    n = len(ranks)
    sums = np.array([part.sum() for part in np.split(ranks, np.cumsum(sizes[:-1]).astype(int))])
    correction = n * (n + 1) ** 2 / 4
    spread = (float(np.sum(ranks**2)) - correction) / (n - 1)
    if spread == 0:
        raise ValueError("Every observation is tied, so the ranks do not vary and H is undefined.")
    statistic = (float(np.sum(sums**2 / sizes)) - correction) / spread
    return {
        "statistic": statistic,
        "df": len(arrays) - 1,
        "p_value": float(stats.chi2.sf(statistic, len(arrays) - 1)),
        "rank_sums": sums.tolist(),
        "mean_ranks": (sums / sizes).tolist(),
    }


def variance_interval(variance: float, df: float, conflevel: float = 0.95) -> dict[str, float]:
    r"""Return the chi-square confidence interval of a variance, and of its standard deviation.

    For a variance :math:`s^2` on :math:`\nu` degrees of freedom the interval is
    :math:`\nu s^2 / \chi^2_{1 - \alpha/2, \nu}` to :math:`\nu s^2 / \chi^2_{\alpha/2, \nu}`, the
    :math:`\chi^2` points being the upper and lower percentage points. :math:`\nu` need not be a whole
    number: a variance whose degrees of freedom come from Satterthwaite's approximation is common.

    Returns
    -------
    dict[str, float]
        ``variance``, ``df``, ``ci_low`` and ``ci_high`` for the variance, and ``std_dev_ci_low`` and
        ``std_dev_ci_high``, their square roots.

    Examples
    --------
    >>> found = variance_interval(22.75 / 12, df=12)  # Montgomery's REML table: 22.75 / 23.3367
    >>> round(found["ci_low"], 6)
    0.974861
    """
    if not variance >= 0 or not math.isfinite(variance):
        raise ValueError(f"variance must be a finite number of at least 0; got {variance!r}.")
    if not df > 0:
        raise ValueError(f"df must be positive; got {df!r}.")
    if not 0 < conflevel < 1:
        raise ValueError(f"conflevel must be between 0 and 1; got {conflevel!r}.")
    alpha = 1 - conflevel
    low = df * variance / float(stats.chi2.ppf(1 - alpha / 2, df))
    high = df * variance / float(stats.chi2.ppf(alpha / 2, df))
    return {
        "variance": float(variance),
        "df": float(df),
        "ci_low": low,
        "ci_high": high,
        "std_dev_ci_low": math.sqrt(low),
        "std_dev_ci_high": math.sqrt(high),
    }


def least_favourable_means(difference: float, n_groups: int) -> list[float]:
    r"""Return the means of ``n_groups`` groups at which a ``difference`` between two of them is hardest to detect.

    Two groups sit ``difference`` apart and the others midway between them, so the departures from
    the average have the least sum of squares the difference allows, :math:`D^2/2`. Minitab works
    the power for a "maximum difference" at these means.

    Examples
    --------
    >>> least_favourable_means(75, 4)
    [-37.5, 0.0, 0.0, 37.5]
    """
    if n_groups < 2:
        raise ValueError(f"n_groups must be at least 2; got {n_groups!r}.")
    return [-difference / 2, *[0.0] * (n_groups - 2), difference / 2]


def one_way_power(n: int, sigma: float, means: Sequence[float], alpha: float = 0.05) -> dict[str, float]:
    r"""Return the power of the one-way analysis of variance's F test, with ``n`` runs in each group.

    The F statistic follows the noncentral :math:`F` on :math:`a - 1` and :math:`a (n - 1)` degrees
    of freedom, with noncentrality :math:`\lambda = n \sum \tau_i^2 / \sigma^2`, where
    :math:`\tau_i` are the ``means``' departures from their average. For the smallest difference
    between two groups worth detecting, pass :func:`least_favourable_means`. ``phi`` is
    :math:`\sqrt{\lambda / a}`, the parameter of the operating characteristic curves of
    Montgomery's Appendix Chart V.

    Returns
    -------
    dict[str, float]
        ``power``, ``noncentrality``, ``phi``, ``df_between``, ``df_within`` and ``critical_f``.

    Examples
    --------
    >>> found = one_way_power(5, 25, least_favourable_means(75, 4), alpha=0.01)  # Montgomery 3.7
    >>> round(found["power"], 6)
    0.804838
    """
    if n < 2:
        raise ValueError(f"n must be at least 2; got {n!r}.")
    if not sigma > 0:
        raise ValueError(f"sigma must be positive; got {sigma!r}.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be between 0 and 1; got {alpha!r}.")
    centred = np.asarray(means, dtype=float) - float(np.mean(means))
    a = len(centred)
    if a < 2:
        raise ValueError(f"Two or more means are needed; got {a}.")
    noncentrality = n * float(centred @ centred) / sigma**2
    df_between, df_within = a - 1, a * (n - 1)
    critical = float(stats.f.ppf(1 - alpha, df_between, df_within))
    return {
        "power": float(stats.ncf.sf(critical, df_between, df_within, noncentrality)),
        "noncentrality": noncentrality,
        "phi": math.sqrt(noncentrality / a),
        "df_between": df_between,
        "df_within": df_within,
        "critical_f": critical,
    }


def one_way_sample_size(
    power: float, sigma: float, means: Sequence[float], alpha: float = 0.05, max_n: int = 10_000
) -> dict[str, float]:
    r"""Return the smallest number of runs in each group whose one-way F test reaches ``power``.

    The groups are described as for :func:`one_way_power`; the search starts at two runs in each
    group and stops at ``max_n``.

    Returns
    -------
    dict[str, float]
        ``n`` (runs in each group), ``n_total``, the ``power`` reached, and ``power_below``, the
        power with one run fewer in each group (NaN at two).

    Raises
    ------
    ValueError
        If ``power`` is not between 0 and 1, or no number of runs up to ``max_n`` reaches it.

    Examples
    --------
    >>> found = one_way_sample_size(0.95, 0.25, least_favourable_means(0.5, 2))  # Montgomery 2.4: 8 of each
    >>> found["n"], round(found["power"], 4)
    (8, 0.9602)
    """
    if not 0 < power < 1:
        raise ValueError(f"power must be between 0 and 1; got {power!r}.")
    below = math.nan
    for n in range(2, max_n + 1):
        found = one_way_power(n, sigma, means, alpha)
        if found["power"] >= power:
            return {"n": n, "n_total": n * len(means), "power": found["power"], "power_below": below}
        below = found["power"]
    raise ValueError(f"No number of runs up to {max_n} in each group reaches a power of {power}.")


def autocorrelation(x: Sequence[float], lags: int | Sequence[int] = 1) -> dict[str, Any]:
    r"""Return the autocorrelations of a record in time order, with their approximate standard error.

    The lag-k autocorrelation is
    :math:`r_k = \sum_{t=1}^{n-k} (x_t - \bar x)(x_{t+k} - \bar x) / \sum_{t=1}^{n} (x_t - \bar x)^2`;
    when the observations are independent its standard error is about :math:`1/\sqrt{n}`.

    Returns
    -------
    dict[str, Any]
        ``autocorrelation``, one value per lag in ``lags`` (a list, in order), and ``std_error``.
    """
    values = np.asarray(x, dtype=float)
    centred = values - values.mean()
    wanted = [lags] if isinstance(lags, int) else list(lags)
    if any(k < 1 or k >= len(values) for k in wanted):
        raise ValueError(f"Each lag must be from 1 to {len(values) - 1}; got {wanted}.")
    total = float(centred @ centred)
    return {
        "autocorrelation": [float(centred[:-k] @ centred[k:]) / total for k in wanted],
        "std_error": 1 / math.sqrt(len(values)),
    }
