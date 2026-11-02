"""A reference distribution from past records, to judge a difference of averages (#789 B6).

Box, Hunter and Hunter's chapter 2 judges a plant trial's difference of 1.30 against the 191
differences of adjacent averages of 10 in the process's own 210-batch record, and against 10 nearly
independent ones (Table 2.5); its exercises do the same for cell growth, a patient's temperature and
five batches. Each record and printed level is read from the reference catalogue.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from process_improve.experiments.reference_catalogue import reference_experiments
from process_improve.univariate import reference_distribution


def entry(name: str) -> dict:
    [found] = [e for e in reference_experiments(source="box-hunter-hunter-1978") if e["name"] == name]
    return found


def past_record(name: str) -> list[float]:
    """Return the external reference that the entry's notes list after the colon."""
    return [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", entry(name)["notes"].split(":", 1)[1])]


def trial_difference(name: str) -> float:
    """Return the trial's later average less its earlier one, B less A."""
    rows = entry(name)["runs"]["rows"]
    later = [row[-1] for row in rows if row[1] == "B"]
    earlier = [row[-1] for row in rows if row[1] == "A"]
    return float(np.mean(later) - np.mean(earlier))


@pytest.fixture(scope="module")
def yields() -> list[float]:
    return [row[1] for row in entry("bhh-production-record-210")["runs"]["rows"]]


# --------------------------------------------------------------------------- the plant trial, Tables 2.2 to 2.5
def test_nine_of_the_191_differences_exceed_1_30(yields):
    found = reference_distribution(yields, 10, observed=1.30)
    assert (found["n_differences"], found["n_beyond"], found["n_tied"]) == (191, 9, 0)
    assert round(found["significance"], 3) == 0.047


def test_the_ten_nearly_independent_differences_of_table_2_5(yields):
    """Groups of 20 a gap of one apart: -0.43, 0.43, -0.17, ...; s^2 = 0.36 about zero, s = 0.60."""
    found = reference_distribution(yields, 10, observed=1.30)
    printed = [
        x
        for x in entry("bhh-production-record-210")["expected"]
        if str(x.get("term", "")).startswith("Table 2.5, obs.")
    ]
    assert len(printed) == 10
    assert found["nearly_independent"] == pytest.approx([float(x["value"]) for x in printed], abs=0.005)
    assert found["df"] == 10
    assert (round(found["s"] ** 2, 2), round(found["s"], 2)) == (0.36, 0.60)


def test_the_t_level_of_the_plant_trial(yields):
    """t0 = 1.30 / 0.60 = 2.17 with the rounded s (2.16 unrounded), and Pr(t > t0) about 0.028 on 10 df."""
    found = reference_distribution(yields, 10, observed=1.30)
    assert round(found["t"], 2) == 2.16
    assert round(found["t_significance"], 3) == 0.028


def test_the_differences_are_the_later_average_less_the_earlier(yields):
    found = reference_distribution(yields, 10)
    assert found["differences"][0] == pytest.approx(np.mean(yields[10:20]) - np.mean(yields[0:10]))
    assert "significance" not in found


# --------------------------------------------------------------------------- the exercises
def test_cell_growth_against_its_past_preparations():
    """0 of 17 differences of averages of 2 reach 9.5; t = 2.21 on 4 df gives 0.046 (printed 0.047)."""
    found = reference_distribution(past_record("bhh-ex-cell-growth"), 2, trial_difference("bhh-ex-cell-growth"))
    assert (found["n_differences"], found["n_beyond"], found["significance"]) == (17, 0, 0.0)
    assert (round(found["t"], 2), found["df"]) == (2.21, 4)
    assert found["t_significance"] == pytest.approx(0.047, abs=0.002)


def test_a_patient_s_temperature_against_the_same_patient_off_the_drug():
    """3 of 31 differences of averages of 3 reach 3.33, 0.097; t = 1.46 on 5 df gives 0.102 (printed 0.103)."""
    name = "bhh-ex-patient-temperature"
    found = reference_distribution(past_record(name), 3, trial_difference(name))
    assert (found["n_differences"], found["n_beyond"]) == (31, 3)
    assert round(found["significance"], 3) == 0.097
    assert (round(found["t"], 2), found["df"]) == (1.46, 5)
    assert found["t_significance"] == pytest.approx(0.103, abs=0.002)


def test_five_batches_against_the_production_record_counts_a_tie_as_half(yields):
    """B averages 1.60 below A: 16 of the 201 differences of averages of 5 lie below it and one equals it."""
    found = reference_distribution(yields, 5, trial_difference("bhh-ex-2-3-five-batches"), alternative="less")
    assert (found["n_differences"], found["n_beyond"], found["n_tied"]) == (201, 16, 1)
    assert round(found["significance"], 3) == 0.082


# --------------------------------------------------------------------------- what is refused
@pytest.mark.parametrize(
    ("history", "window", "options", "message"),
    [
        ([1.0, 2.0, np.nan, 4.0], 1, {}, "finite numbers"),
        ([1.0, 2.0, 3.0, 4.0], 0, {}, "window must be a whole number"),
        ([1.0, 2.0, 3.0, 4.0], True, {}, "window must be a whole number"),
        ([1.0, 2.0, 3.0, 4.0], 1, {"gap": -1}, "gap must be a whole number"),
        ([1.0, 2.0, 3.0], 2, {}, "too short for one difference"),
        ([1.0, 2.0, 3.0, 4.0], 1, {"alternative": "two-sided"}, "alternative must be one of"),
    ],
)
def test_a_reference_that_cannot_be_built_is_refused(history, window, options, message):
    with pytest.raises(ValueError, match=message):
        reference_distribution(history, window, **options)


def test_a_constant_history_gives_an_infinite_t():
    found = reference_distribution([5.0] * 8, 2, observed=1.0)
    assert found["s"] == 0.0
    assert found["t"] == np.inf
    assert found["t_significance"] == 0.0
