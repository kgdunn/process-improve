"""Results judged against values stated before the data (#789 B5 and B6).

Box, Hunter and Hunter print each of these, and the reference catalogue held them as manual checks:
the five cars of Exercise 2.12 (a variance about the known mean 25.0, and a new car's 28.2 against
it), the production record's nearly independent differences (Table 2.5, about zero), and the blood
coagulation times of Table 6.1 against the norm of 63.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from process_improve.experiments.reference_catalogue import reference_experiments
from process_improve.univariate import grand_mean_test, ttest_new_observation, ttest_one_sample, variance_about


def bhh(name: str) -> dict:
    [entry] = [e for e in reference_experiments(source="box-hunter-hunter-1978") if e["name"] == name]
    return entry


@pytest.fixture(scope="module")
def cars() -> list[float]:
    return [row[1] for row in bhh("bhh-ex-2-12-cars")["runs"]["rows"]]


@pytest.fixture(scope="module")
def coagulation() -> tuple[list[float], list[str]]:
    rows = bhh("bhh-blood-coagulation")["runs"]["rows"]
    return [row[3] for row in rows], [row[1] for row in rows]


# --------------------------------------------------------------------------- a variance about a known mean
def test_the_cars_variance_about_the_known_mean_is_the_book_s(cars):
    """Exercise 2.12: sum (y - 25.0)^2 / 5 = 2.96, on all five degrees of freedom."""
    found = variance_about(cars, 25.0)
    assert round(found["variance"], 2) == 2.96
    assert found["df"] == len(cars)
    assert found["std_dev"] == pytest.approx(np.sqrt(found["variance"]))


def test_the_nearly_independent_differences_about_zero_are_table_2_5_s():
    """Averages of 10 from the 210 batch yields, two at a time 21 apart: the 10 differences give 0.36 and 0.60."""
    yields = np.array([row[1] for row in bhh("bhh-production-record-210")["runs"]["rows"]])
    differences = [yields[s + 10 : s + 20].mean() - yields[s : s + 10].mean() for s in range(0, 190, 21)]
    found = variance_about(differences, 0.0)
    assert found["df"] == 10
    assert (round(found["variance"], 2), round(found["std_dev"], 2)) == (0.36, 0.60)


# --------------------------------------------------------------------------- one new observation
def test_a_new_car_against_the_known_mean(cars):
    """(28.2 - 25.0) / sqrt(2.96) = 1.86 on 5 degrees of freedom."""
    found = ttest_new_observation(28.2, 25.0, cars)
    assert (round(found["t_value"], 2), found["df"]) == (1.86, 5)
    assert found["p_value_greater"] == pytest.approx(stats.t.sf(found["t_value"], 5))
    assert found["p_value"] == pytest.approx(2 * found["p_value_greater"])


def test_a_new_car_with_s_about_the_cars_own_average(cars):
    """On 4 degrees of freedom, s^2 = 3.702 and t = 1.663; the book prints 1.67."""
    found = ttest_new_observation(28.2, 25.0, cars, mean_known=False)
    assert found["df"] == 4
    assert found["std_dev"] ** 2 == pytest.approx(np.var(cars, ddof=1))
    assert round(found["t_value"], 3) == 1.663


# --------------------------------------------------------------------------- the grand mean against a norm
def test_the_coagulation_sums_of_squares_about_the_norm_are_the_book_s(coagulation):
    """Table 6.2: 364 = 24 + 228 + 112, on 24 = 1 + 3 + 20 degrees of freedom."""
    found = grand_mean_test(*coagulation, value=63)
    assert {part: round(ss, 6) for part, ss in found["sum_of_squares"].items()} == {
        "total": 364,
        "mean": 24,
        "groups": 228,
        "residual": 112,
    }
    assert found["df"] == {"total": 24, "mean": 1, "groups": 3, "residual": 20}


def test_the_coagulation_grand_mean_against_63_is_the_book_s_t(coagulation):
    """T = (64 - 63) / sqrt(5.6 / 24) = 2.07 on 20 degrees of freedom; its square is the F of the mean's row."""
    found = grand_mean_test(*coagulation, value=63)
    assert found["grand_mean"] == pytest.approx(64.0)
    assert (round(found["t_value"], 2), found["df_resid"]) == (2.07, 20)
    sums, df = found["sum_of_squares"], found["df"]
    assert found["t_value"] ** 2 == pytest.approx(sums["mean"] / (sums["residual"] / df["residual"]))


def test_with_one_group_the_grand_mean_test_is_the_one_sample_t():
    rng = np.random.default_rng(7)
    y = rng.normal(10, 2, 12)
    found, one_sample = grand_mean_test(y, ["only"] * len(y), 9.0), ttest_one_sample(y, mu=9.0)
    assert (found["t_value"], found["df_resid"]) == pytest.approx((one_sample["t_value"], one_sample["df"]))
    assert found["p_value"] == pytest.approx(one_sample["p_value"])


# --------------------------------------------------------------------------- what is refused
@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: variance_about([], 0.0), "non-empty sequence of finite numbers"),
        (lambda: variance_about([1.0, np.nan], 0.0), "non-empty sequence of finite numbers"),
        (lambda: variance_about([1.0, 2.0], np.inf), "mean must be a finite number"),
        (lambda: ttest_new_observation(1.0, 0.0, [2.0], mean_known=False), "needs at least two of them"),
        (lambda: grand_mean_test([1.0, 2.0], ["a"], 0.0), "the same length"),
        (lambda: grand_mean_test([1.0, 2.0], ["a", "b"], 0.0), "leave no residual degree of freedom"),
    ],
)
def test_inputs_that_cannot_be_judged_are_refused(call, message):
    with pytest.raises(ValueError, match=message):
        call()
