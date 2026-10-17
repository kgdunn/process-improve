"""Classical tests and percentage points, checked against the tables and worked examples of the textbooks."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy import stats

from process_improve.univariate import (
    autocorrelation,
    chi2_value,
    dunnett_value,
    f_value,
    randomization_test,
    rank_sum_test,
    studentized_range_value,
    ttest_one_sample,
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
