"""Classical tests and percentage points, checked against the tables and worked examples of the textbooks."""

from __future__ import annotations

import doctest
import itertools
import math

import numpy as np
import pytest
from scipy import stats

from process_improve.univariate import (
    autocorrelation,
    bartlett_test,
    chi2_value,
    classical,
    dunnett_value,
    f_value,
    kruskal_wallis_test,
    least_favourable_means,
    one_way_power,
    one_way_sample_size,
    randomization_test,
    rank_sum_test,
    studentized_range_value,
    ttest_one_sample,
    variance_interval,
    z_value,
    ztest,
)

# Box, Hunter and Hunter's plant trial (Table 3.1): methods A and B, ten runs each.
A = [89.7, 81.4, 84.5, 84.8, 87.3, 79.7, 85.1, 81.7, 83.7, 84.5]
B = [84.7, 86.1, 83.2, 91.9, 86.3, 79.3, 82.6, 89.1, 83.7, 88.5]


@pytest.mark.parametrize(
    ("found", "table"),
    [
        (lambda: f_value(0.95, 3, 16), 3.24),  # Montgomery, Appendix Table IV
        (lambda: f_value(0.99, 3, 16), 5.29),
        (lambda: chi2_value(0.95, 3), 7.81),  # Appendix Table III
        (lambda: chi2_value(0.975, 12), 23.34),
        (lambda: z_value(0.975), 1.96),
        (lambda: studentized_range_value(0.95, 4, 16), 4.05),  # Appendix Table VII
        (lambda: dunnett_value(0.95, 3, 16), 2.59),  # Appendix Table VIII, two-sided
        (lambda: dunnett_value(0.95, 3, 16, alternative="greater"), 2.23),  # one-sided
    ],
)
def test_percentage_points_match_the_printed_tables(found, table):
    assert found() == pytest.approx(table, abs=0.005)


def test_dunnett_with_one_comparison_is_student_s_t():
    assert dunnett_value(0.95, 1, 12) == pytest.approx(stats.t.ppf(0.975, 12), abs=1e-6)
    assert dunnett_value(0.95, 1, np.inf, alternative="greater") == pytest.approx(stats.norm.ppf(0.95), abs=1e-6)


def test_dunnett_agrees_with_the_multivariate_t_distribution():
    """Three comparisons with a control, correlated by one half: the levels by SciPy's multivariate t."""
    shape = np.full((3, 3), 0.5) + 0.5 * np.eye(3)
    joint = stats.multivariate_t(shape=shape, df=16, seed=1)
    upper = dunnett_value(0.95, 3, 16, alternative="greater")
    both = dunnett_value(0.95, 3, 16)
    assert joint.cdf(np.full(3, upper), maxpts=500_000) == pytest.approx(0.95, abs=5e-4)
    assert joint.cdf(np.full(3, both), lower_limit=np.full(3, -both), maxpts=500_000) == pytest.approx(0.95, abs=5e-4)


def test_dunnett_refuses_what_it_cannot_compute():
    with pytest.raises(ValueError, match="alternative"):
        dunnett_value(0.95, 3, 16, alternative="less")
    with pytest.raises(ValueError, match="p must lie"):
        dunnett_value(1.5, 3, 16)


def test_a_mean_against_a_stated_value():
    """B's ten yields against 84.12: t = 1.23 on 9 degrees of freedom, a one-sided level of 0.125."""
    found = ttest_one_sample(B, mu=84.12)
    assert (found["t_value"], found["df"], found["p_value_greater"]) == pytest.approx((1.230, 9, 0.125), abs=5e-4)
    assert found["p_value"] == pytest.approx(2 * found["p_value_greater"])
    assert found["ci_low"] < found["mean"] < found["ci_high"]
    with pytest.raises(ValueError, match="at least two"):
        ttest_one_sample([1.0], 0.0)


def test_a_difference_of_means_with_sigma_known():
    """With sigma = 2.88 from past records: a standard error of 1.29, z = 1.01, Pr(z > 1.01) = 0.156."""
    found = ztest(A, B, sigma=2.88)
    assert (found["std_error"], found["z_value"], found["p_value_greater"]) == pytest.approx(
        (1.29, 1.01, 0.156), abs=0.005
    )
    alone = ztest(B, sigma=2.88, mu=84.12)
    assert alone["z_value"] == pytest.approx((np.mean(B) - 84.12) / (2.88 / np.sqrt(10)))
    with pytest.raises(ValueError, match="sigma"):
        ztest(A, sigma=0)


def test_wilcoxon_rank_sums_and_their_normal_approximation():
    """The plant trial: rank sums 95.5 and 114.5, z = 0.718 without the tie correction, a level of 0.236."""
    found = rank_sum_test(B, A, tie_correction=False)
    assert (found["rank_sum_x"], found["rank_sum_y"]) == (114.5, 95.5)
    assert (found["z_value"], found["p_value_greater"]) == pytest.approx((0.718, 0.236), abs=5e-4)
    assert rank_sum_test(B, A)["z_value"] > found["z_value"]  # ties shrink the variance a little


def test_the_exact_rank_sum_level_enumerates_every_split():
    found = rank_sum_test([1.0, 2.0], [3.0, 4.0, 5.0])
    assert found["exact_p_value_less"] == pytest.approx(1 / 10)  # the lowest two ranks: one split of ten
    assert found["u_statistic"] == 0


def test_a_randomization_test_counts_every_split():
    """Every split of six values into three and three: only the observed one has the largest difference."""
    found = randomization_test([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    assert (found["n_arrangements"], found["n_at_least"], found["exact"]) == (20, 1, True)
    assert found["p_value"] == pytest.approx(1 / 20)
    assert found["p_value_two_sided"] == pytest.approx(2 / 20)


def test_a_paired_randomization_test_flips_the_signs_and_can_count_ties_as_half():
    differences = np.array([0.8, 0.6, 0.3, -0.1, 1.1, -0.2, 0.3, 0.5, 0.5, 0.3])  # BHH's boys' shoes
    found = randomization_test(differences)
    assert found["n_arrangements"] == 1024
    assert (found["n_greater"], found["n_equal"]) == (3, 4)
    assert found["p_value_mid"] == pytest.approx(5 / 1024)


def test_more_arrangements_than_allowed_are_sampled_reproducibly():
    x, y = np.arange(10.0), np.arange(10.0) + 3
    first = randomization_test(x, y, max_arrangements=500, random_state=1)
    again = randomization_test(x, y, max_arrangements=500, random_state=1)
    assert first["exact"] is False
    assert first["n_arrangements"] == 500
    assert first["n_at_least"] == again["n_at_least"]


def test_autocorrelation_of_a_record():
    x = np.array(list(itertools.chain.from_iterable([[1.0, -1.0]] * 10)))  # alternating: r1 near -1
    found = autocorrelation(x, [1, 2])
    assert found["autocorrelation"] == pytest.approx([-0.95, 0.9])
    assert found["std_error"] == pytest.approx(1 / np.sqrt(20))
    with pytest.raises(ValueError, match="lag"):
        autocorrelation(x, 20)


# Montgomery's plasma etching (Example 3.1): etch rate at four RF powers, five wafers each.
ETCH = [[575, 542, 530, 539, 570], [565, 593, 590, 579, 610], [600, 651, 610, 637, 629], [725, 700, 715, 685, 710]]


class TestBartlett:
    def test_montgomery_example_3_4(self) -> None:
        """Q = 0.21, c = 1.10, chi-square 0.43 on 3 df, P = 0.934 (scipy: 0.4335 and 0.9332)."""
        found = bartlett_test(*ETCH)
        assert (round(found["q"], 2), round(found["c"], 2)) == (0.21, 1.1)
        assert found["c"] == pytest.approx(1.1042, abs=1e-4)
        assert (found["statistic"], found["df"]) == (pytest.approx(0.4335, abs=1e-4), 3)
        assert found["p_value"] == pytest.approx(0.9332, abs=1e-4)
        assert found["pooled_variance"] == pytest.approx(333.7)

    def test_it_is_scipys_bartlett(self) -> None:
        rng = np.random.default_rng(3)
        groups = [rng.normal(scale=s, size=n) for s, n in ((1.0, 6), (2.0, 9), (1.5, 4))]
        found = bartlett_test(*groups)
        expected = stats.bartlett(*groups)
        assert (found["statistic"], found["p_value"]) == pytest.approx((expected.statistic, expected.pvalue))

    @pytest.mark.parametrize(
        ("groups", "message"),
        [(([1.0, 2.0],), "Two or more groups"), (([1.0, 2.0], [3.0]), "at least 2 observation")],
        ids=["one group", "a group of one"],
    )
    def test_refusals(self, groups: tuple, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            bartlett_test(*groups)


class TestKruskalWallis:
    def test_montgomery_section_3_11(self) -> None:
        """H = 16.91 and P = 7.38e-4: the ranks' treatment sum of squares, 591.30, over their variance, 34.97."""
        found = kruskal_wallis_test(*ETCH)
        assert (found["statistic"], found["df"]) == (pytest.approx(16.907, abs=1e-3), 3)
        assert found["p_value"] == pytest.approx(7.386e-4, rel=1e-3)
        assert sum(found["rank_sums"]) == 210

    def test_ties_are_allowed_for_as_scipy_does(self) -> None:
        groups = [[1.0, 2.0, 2.0, 3.0], [2.0, 3.0, 3.0, 4.0], [5.0, 5.0, 6.0, 1.0]]
        expected = stats.kruskal(*groups)
        found = kruskal_wallis_test(*groups)
        assert (found["statistic"], found["p_value"]) == pytest.approx((expected.statistic, expected.pvalue))

    def test_refusals(self) -> None:
        with pytest.raises(ValueError, match="Two or more groups"):
            kruskal_wallis_test([1.0, 2.0])
        with pytest.raises(ValueError, match="Every observation is tied"):
            kruskal_wallis_test([1.0, 1.0], [1.0])


class TestVarianceInterval:
    @pytest.mark.parametrize(
        ("sum_of_squares", "df", "low"),
        [(22.75, 12, 0.974861), (109.886, 15, 3.99755), (166.333, 15, 6.05104)],
        ids=["REML table, 12 df", "15 df", "15 df, second"],
    )
    def test_montgomerys_lower_limits(self, sum_of_squares: float, df: int, low: float) -> None:
        """The lower 95% limit is the sum of squares over the upper chi-square point: 23.3367 (12 df), 27.4884 (15)."""
        found = variance_interval(sum_of_squares / df, df)
        assert found["ci_low"] == pytest.approx(low, rel=1e-5)
        assert found["std_dev_ci_low"] == pytest.approx(math.sqrt(low), rel=1e-5)

    def test_fractional_degrees_of_freedom(self) -> None:
        found = variance_interval(2.0, 7.5, conflevel=0.90)
        assert found["ci_low"] == pytest.approx(7.5 * 2.0 / stats.chi2.ppf(0.95, 7.5))
        assert found["ci_high"] == pytest.approx(7.5 * 2.0 / stats.chi2.ppf(0.05, 7.5))

    @pytest.mark.parametrize(
        ("arguments", "message"),
        [((-1.0, 5), "variance must be"), ((1.0, 0), "df must be positive"), ((1.0, 5, 1.0), "conflevel must")],
    )
    def test_refusals(self, arguments: tuple, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            variance_interval(*arguments)


class TestOneWayPower:
    def test_montgomery_minitab_power_for_a_largest_difference(self) -> None:
        """Sigma 25, alpha 0.01, a difference of 75 among four powers: 0.804838 at five wafers, 0.915384 at six."""
        means = least_favourable_means(75, 4)
        assert one_way_power(5, 25, means, alpha=0.01)["power"] == pytest.approx(0.804838, abs=5e-7)
        assert one_way_power(6, 25, means, alpha=0.01)["power"] == pytest.approx(0.915384, abs=5e-7)

    def test_montgomery_operating_characteristic_example(self) -> None:
        """Means 575, 600, 650 and 675: phi = sqrt(2.5 n), 2.74 at n = 3; power 0.96 at 4 and above 0.99 at 5."""
        means = [575, 600, 650, 675]
        assert round(one_way_power(3, 25, means, alpha=0.01)["phi"], 2) == 2.74
        assert round(one_way_power(4, 25, means, alpha=0.01)["power"], 2) == 0.96
        assert one_way_power(5, 25, means, alpha=0.01)["power"] > 0.99

    def test_montgomery_portland_cement(self) -> None:
        """Two mortars, a difference of 0.5 and sigma 0.25: 0.929 at 7 each, 0.9602 at 8, so 8 of each (16 in all)."""
        means = least_favourable_means(0.5, 2)
        assert round(one_way_power(7, 0.25, means)["power"], 3) == 0.929
        found = one_way_sample_size(0.95, 0.25, means)
        assert (found["n"], found["n_total"], round(found["power"], 4)) == (8, 16, 0.9602)
        assert found["power_below"] == pytest.approx(0.9291, abs=1e-4)

    def test_montgomery_portland_cement_for_a_smaller_difference(self) -> None:
        """A difference of 0.25 at power 0.9 needs 23 of each: 0.8997 at 22, 0.9125 at 23."""
        found = one_way_sample_size(0.90, 0.25, least_favourable_means(0.25, 2))
        assert (found["n"], round(found["power"], 4), round(found["power_below"], 4)) == (23, 0.9125, 0.8997)

    def test_the_text_chooses_six_wafers(self) -> None:
        assert one_way_sample_size(0.90, 25, least_favourable_means(75, 4), alpha=0.01)["n"] == 6

    def test_least_favourable_means_have_the_least_spread(self) -> None:
        """Two groups D apart and the rest midway: the sum of squared departures is D^2/2."""
        means = np.array(least_favourable_means(10.0, 5))
        assert means.tolist() == [-5.0, 0.0, 0.0, 0.0, 5.0]
        assert float(np.sum((means - means.mean()) ** 2)) == 50.0

    @pytest.mark.parametrize(
        ("call", "message"),
        [
            (lambda: one_way_power(1, 1.0, [0.0, 1.0]), "n must be at least 2"),
            (lambda: one_way_power(3, 0.0, [0.0, 1.0]), "sigma must be positive"),
            (lambda: one_way_power(3, 1.0, [0.0, 1.0], alpha=1.5), "alpha must be"),
            (lambda: one_way_power(3, 1.0, [0.0]), "Two or more means"),
            (lambda: least_favourable_means(1.0, 1), "n_groups must be at least 2"),
            (lambda: one_way_sample_size(1.0, 1.0, [0.0, 1.0]), "power must be between"),
            (lambda: one_way_sample_size(0.99, 1.0, [0.0, 0.01], max_n=5), "No number of runs up to 5"),
        ],
    )
    def test_refusals(self, call, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            call()


def test_the_examples_in_the_docstrings_print_what_they_show() -> None:
    result = doctest.testmod(classical, report=False)
    assert result.attempted > 0
    assert result.failed == 0
