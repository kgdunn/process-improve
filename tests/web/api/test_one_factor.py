"""A study of one factor: two treatments compared, such as two suppliers, or a temperature at its low and its high.

The page makes a full factorial in one factor, its two levels each run several times in random order,
and takes one factor's runs as given. The design's notes say how many runs sit at each level, as an
alias sentence would mean nothing with one factor, and the design is analysed for its main effect.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from tests.web._support import FACTORS, held_of, run_sheet

SUPPLIER = {"name": "Supplier", "low": "A", "high": "B", "units": ""}


def comparison(factor: dict, **options: object) -> dict:
    """Ask for a full factorial in one factor, with the options given."""
    return {"design_type": "full_factorial", "factors": [factor], "response": "y", **options}


def test_a_full_factorial_in_one_factor_compares_its_two_levels(api):
    result = api.ok("api_make_design", comparison(SUPPLIER, n_replicates=4, n_center_points=0))
    sheet = run_sheet(result)
    assert sorted(sheet["Supplier"]) == ["A"] * 4 + ["B"] * 4
    assert result["model"] == result["spec"]["model"] == "main_effects", "one factor has no interaction"
    assert result["facts"]["notes"] == ["The runs compare Supplier at A and at B: 4 runs at each, in random order."]


def test_a_numeric_factor_s_center_runs_are_said_with_their_setting(api):
    result = api.ok("api_make_design", comparison(FACTORS[0], n_replicates=2, n_center_points=3))
    assert sorted(run_sheet(result)["T"]) == [150] * 2 + [175] * 3 + [200] * 2
    assert result["facts"]["notes"][0] == (
        "The runs compare T at 150 and at 200 degC: 2 runs at each, and 3 runs at its center, 175 degC, "
        "in random order."
    )


def test_one_factor_s_runs_as_given_keep_their_order_and_are_counted_at_each_level(api):
    runs = [["A"], ["B"], ["B"], ["A"], ["A"]]
    result = api.ok("api_make_design", {"design_type": "given", "factors": [SUPPLIER], "runs": runs})
    assert run_sheet(result)["Supplier"].tolist() == ["A", "B", "B", "A", "A"]
    assert result["model"] == "main_effects"
    assert result["facts"]["notes"][1] == "The runs compare Supplier at A and at B: 3 runs at A and 2 at B."


def test_one_factor_s_runs_as_given_at_several_settings_are_said_as_levels(api):
    runs = [[150], [160], [175], [190], [200]]
    result = api.ok("api_make_design", {"design_type": "given", "factors": [FACTORS[0]], "runs": runs})
    assert result["facts"]["notes"][1] == "The runs set T at 5 levels, from 150 to 200 degC."


@pytest.mark.parametrize(
    ("design_type", "counts"),
    [("plackett_burman", "2 to 11"), ("ccd", "2 to 7"), ("box_behnken", "3 to 7")],
)
def test_the_other_designs_still_say_the_factors_they_take(api, design_type, counts):
    error = api.error("api_make_design", {"design_type": design_type, "factors": FACTORS[:1]})
    assert error.endswith(f"takes {counts} factors on this page; you have 1.")


# --------------------------------------------------------------------------- the comparison of the two levels
#: Box, Hunter and Hunter's plant trial (their Table 3.1): ten batches by method A, then ten by method B, in that order.
METHOD = {"name": "method", "low": "A", "high": "B", "units": ""}
YIELDS_A = [89.7, 81.4, 84.5, 84.8, 87.3, 79.7, 85.1, 81.7, 83.7, 84.5]
YIELDS_B = [84.7, 86.1, 83.2, 91.9, 86.3, 79.3, 82.6, 89.1, 83.7, 88.5]


@pytest.fixture(scope="module")
def plant_trial(api) -> dict:
    """Return the plant trial's analysis, its runs as given, as the catalog opens them."""
    runs = [["A"]] * 10 + [["B"]] * 10
    design = api.ok("api_make_design", {"design_type": "given", "factors": [METHOD], "runs": runs})
    return api.ok("api_analyze_runs", {**held_of(design), "responses": YIELDS_A + YIELDS_B})


def test_each_level_is_summarised_from_its_own_runs(plant_trial):
    a, b = plant_trial["comparison"]["levels"]
    assert (a["level"], a["n"], b["level"], b["n"]) == ("A", 10, "B", 10)
    assert a["mean"] == pytest.approx(84.24)
    assert b["mean"] == pytest.approx(85.54)
    assert a["std_dev"] == pytest.approx(np.std(YIELDS_A, ddof=1))
    half = stats.t.ppf(0.975, 9) * a["std_dev"] / np.sqrt(10)
    assert (a["ci_low"], a["ci_high"]) == pytest.approx((84.24 - half, 84.24 + half))
    assert [r["run"] for r in b["runs"]] == list(range(11, 21))
    assert [r["value"] for r in b["runs"]] == YIELDS_B


def test_the_difference_is_the_high_level_s_average_less_the_low_s_tested_four_ways(plant_trial):
    difference = plant_trial["comparison"]["difference"]
    assert (difference["from"], difference["to"]) == ("A", "B")
    assert difference["estimate"] == pytest.approx(1.30)
    tests = {t["test"]: t for t in difference["tests"]}
    assert list(tests) == ["pooled", "welch", "randomization", "rank_sum"]
    for name, equal in (("pooled", True), ("welch", False)):
        expected = stats.ttest_ind(YIELDS_B, YIELDS_A, equal_var=equal)
        interval = expected.confidence_interval(0.95)
        assert tests[name]["p_value"] == pytest.approx(expected.pvalue)
        assert (tests[name]["ci_low"], tests[name]["ci_high"]) == pytest.approx((interval.low, interval.high))
    shuffled = tests["randomization"]
    assert (shuffled["exact"], shuffled["n_arrangements"]) == (True, 184756), "every split of 20 runs into 10 and 10"
    assert shuffled["p_value"] == pytest.approx(0.3923, abs=1e-4)
    assert shuffled["ci_low"] < 1.30 < shuffled["ci_high"]
    ranks = tests["rank_sum"]
    assert ranks["exact"] is True
    assert ranks["p_value"] == pytest.approx(0.4929, abs=1e-4), "every split of the ranks, ties at their average"


def test_the_model_s_coefficient_is_half_the_difference(plant_trial):
    """Coded -1 and +1, the levels are two units apart, so the coefficient of method is half B's average less A's."""
    coefficient = next(c for c in plant_trial["coefficients"] if c["term"] == "method")
    assert coefficient["coefficient"] == pytest.approx(plant_trial["comparison"]["difference"]["estimate"] / 2)
    pooled = plant_trial["comparison"]["difference"]["tests"][0]
    assert coefficient["p_value"] == pytest.approx(pooled["p_value"]), "the model's test is the pooled t-test"


def test_a_level_with_one_run_has_no_spread_and_no_difference_is_tested(api):
    design = api.ok("api_make_design", {"design_type": "given", "factors": [METHOD], "runs": [["A"], ["A"], ["B"]]})
    comparison = api.ok("api_analyze_runs", {**held_of(design), "responses": [1.0, 2.0, 4.0]})["comparison"]
    assert [level["std_dev"] for level in comparison["levels"]] == [pytest.approx(0.7071, abs=1e-4), None]
    assert comparison["difference"] is None


def test_a_numeric_factor_s_three_settings_are_each_a_level(api):
    design = api.ok("api_make_design", comparison(FACTORS[0], n_replicates=2, n_center_points=2))
    held = held_of(design)
    settings = run_sheet(design)["T"].tolist()
    found = api.ok("api_analyze_runs", {**held, "responses": [s / 10 for s in settings]})["comparison"]
    assert [level["level"] for level in found["levels"]] == [150, 175, 200]
    assert [level["mean"] for level in found["levels"]] == pytest.approx([15, 17.5, 20])
    assert found["difference"] is None, "a difference is between two levels"


def test_simulated_results_carry_the_true_difference(api):
    design = api.ok("api_make_design", comparison(SUPPLIER, n_replicates=4, n_center_points=0))
    held = held_of(design)
    simulated = api.ok("api_simulate", {**held, "scenario": "main_effects", "sigma": 1, "seed": 3})
    request = {**held, "responses": simulated["responses"], "truth": simulated["truth"]}
    found = api.ok("api_analyze_runs", request)
    true = next(c["true_value"] for c in found["coefficients"] if c["term"] == "Supplier")
    assert found["comparison"]["difference"]["true"] == pytest.approx(2 * true)


def test_a_study_of_two_factors_has_no_comparison(api, held, simulated):
    assert api.ok("api_analyze_runs", {**held, "responses": simulated["responses"]})["comparison"] is None
