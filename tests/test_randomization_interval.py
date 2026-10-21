"""The confidence interval that inverts the randomization test (#789 B6), on Box, Hunter and Hunter's examples."""

from __future__ import annotations

import doctest

import numpy as np
import pytest

from process_improve.univariate import classical, randomization_interval, randomization_test

# Box, Hunter and Hunter's boys' shoes (Section 4.2), B less A; and their tomato plants (Section 4.1), A and B.
SHOES = np.array([0.8, 0.6, 0.3, -0.1, 1.1, -0.2, 0.3, 0.5, 0.5, 0.3])
TOMATO_A = [29.9, 11.4, 25.3, 16.5, 21.1]
TOMATO_B = [26.6, 23.7, 28.5, 14.2, 17.9, 24.3]


def rejected(found: dict, level: float) -> bool:
    """Whether either one-sided randomization test rejects at ``level``."""
    return min(found["p_value"], found["p_value_less"]) <= level


def test_a_stated_difference_is_tested_on_the_differences_less_it():
    """BHH Section 5.2 tests the shoes at the t interval's limits, 0.14 and 0.68 (the book counts 25 and 19)."""
    at_low, at_high = randomization_test(SHOES - 0.14), randomization_test(SHOES - 0.68)
    assert (at_low["n_greater"], at_low["n_equal"]) == (28, 1)
    assert (at_high["n_less"], at_high["n_equal"]) == (26, 1)


def test_the_paired_interval_is_every_difference_the_test_does_not_reject():
    """1024 patterns of signs: the 25th smallest and largest of the 1023 subset means, at 95.1% (t: 0.133 to 0.687)."""
    found = randomization_interval(SHOES)
    assert (found["ci_low"], found["ci_high"]) == pytest.approx((0.125, 0.7))
    assert found["confidence"] == pytest.approx(1 - 50 / 1024)
    assert (found["n_arrangements"], found["exact"]) == (1024, True)
    level = (1 - 0.95) / 2
    for delta in np.arange(-0.3, 1.2, 0.0025):
        inside = found["ci_low"] - 1e-9 <= delta <= found["ci_high"] + 1e-9
        assert rejected(randomization_test(SHOES - delta), level) is not inside, delta


def test_the_two_group_interval_inverts_the_test_of_a_shift():
    """462 splits of eleven plants into five and six; the breakpoints are differences of means of the plants moved."""
    found = randomization_interval(TOMATO_A, TOMATO_B, confidence=0.9)
    assert found["statistic"] == pytest.approx(np.mean(TOMATO_B) - np.mean(TOMATO_A))
    assert found["confidence"] == pytest.approx(1 - 2 * 23 / 462)
    level = (1 - 0.9) / 2
    for delta in np.arange(-20.0, 25.0, 0.05):
        inside = found["ci_low"] - 1e-9 <= delta <= found["ci_high"] + 1e-9
        assert rejected(randomization_test(TOMATO_A, np.subtract(TOMATO_B, delta)), level) is not inside, delta


def test_pairs_given_as_two_columns_are_their_differences():
    first = np.array([13.2, 8.2, 10.9, 14.3, 10.7, 6.6, 9.5, 10.8, 8.8, 13.3])
    found = randomization_interval(first, first + SHOES, paired=True)
    assert (found["ci_low"], found["ci_high"]) == pytest.approx((0.125, 0.7))


def test_too_many_arrangements_are_sampled_reproducibly(monkeypatch):
    monkeypatch.setattr(classical, "MOST_ARRANGEMENTS", 400)
    first = randomization_interval(SHOES, random_state=7)
    assert (first["n_arrangements"], first["exact"]) == (400, False)
    assert first == randomization_interval(SHOES, random_state=np.random.default_rng(7))
    assert first["ci_low"] == pytest.approx(0.125, abs=0.1)
    assert first["ci_high"] == pytest.approx(0.7, abs=0.1)


@pytest.mark.parametrize(
    ("x", "y", "kwargs", "match"),
    [
        ([1.0, 2.0, 3.0, 4.0, 5.0], None, {}, r"32 arrangements allow no 95.0% interval: at most 93.8%"),
        (SHOES, None, {"confidence": 1.0}, "between 0 and 1"),
        ([], None, {}, "needs a value"),
        ([1.0, np.nan, 2.0], None, {}, "finite"),
        (TOMATO_A, [], {}, "needs a value"),
    ],
)
def test_an_interval_the_arrangements_cannot_give_is_refused(x, y, kwargs, match):
    with pytest.raises(ValueError, match=match):
        randomization_interval(x, y, **kwargs)


def test_the_examples_in_the_docstrings_print_what_they_show():
    """No CI step runs doctests, so an example whose output drifted would go unnoticed."""
    result = doctest.testmod(classical, report=False)
    assert result.attempted > 0
    assert result.failed == 0
