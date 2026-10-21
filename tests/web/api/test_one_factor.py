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


# --------------------------------------------------------------------------- how many runs at each level
def planned(api, factor: dict = SUPPLIER, model: str = "main_effects", info: dict | None = None) -> dict:
    """Make and score the designs for a brief of one factor, as Design does."""
    request = {"factors": [factor], "model": model, "response": "y", "response_info": info, "random_state": 7}
    return api.ok("api_candidates", request)


def test_a_design_in_one_factor_is_sized_for_the_power_its_plan_asks(api, bootstrap):
    """Two noise standard deviations, the default change, take 6 runs at each level for 80% power; 5 give 79%."""
    found = planned(api)
    plan = found["levels"]
    power = {row["n"]: row["power"] for row in plan["rows"]}
    assert (plan["n"], plan["reached"], plan["power"]) == (6, True, bootstrap.LEVEL_POWER)
    assert power[5] < bootstrap.LEVEL_POWER <= power[6]
    assert [row["runs"] for row in plan["rows"]] == [2 * row["n"] for row in plan["rows"]]
    [made] = found["made"]
    assert made["design"]["spec"]["options"] | {"random_state": 0} == {
        "random_state": 0,
        "n_center_points": 0,
        "n_replicates": 6,
    }
    assert made["design"]["n_runs"] == 12


def test_the_plan_s_power_is_the_power_the_designs_are_compared_on(api):
    """The t-test of two levels is the model's test of the factor: its power is the comparison's main-effect power."""
    found = planned(api, info={"noise": 2, "smallest_change": 3})
    plan = found["levels"]
    assert plan["n"] == 9
    label = found["result"]["designs"][0]
    shown = found["result"]["values"][label]["power_main"]
    assert shown == pytest.approx(next(row["power"] for row in plan["rows"] if row["n"] == 9))


def test_a_change_too_small_for_any_design_the_page_makes_is_said(api, bootstrap):
    plan = planned(api, info={"noise": 1, "smallest_change": 0.3})["levels"]
    assert (plan["n"], plan["reached"]) == (bootstrap.MAX_LEVEL_RUNS, False)


def test_a_categorical_factor_is_scored_on_its_main_effect_whatever_the_aim(api):
    """A categorical factor cannot curve: its two levels fit the main effect, which the plan's power is for."""
    found = planned(api, model="main_quadratic")
    assert found["model"] == "main_effects"
    assert found["levels"]["n"] == 6
    assert found["result"]["values"][found["result"]["designs"][0]]["fits"] is True
    assert found["made"][0]["design"]["spec"]["aim"] == "main_quadratic", "the reader's aim is kept"


def test_a_numeric_factor_the_study_aims_to_see_curve_keeps_its_center_runs(api):
    found = planned(api, factor=FACTORS[0], model="main_quadratic")
    assert found["levels"] is None
    assert found["made"][0]["design"]["spec"]["options"]["n_center_points"] == 3


def test_one_factor_s_levels_take_more_replicates_than_a_design_of_more_factors(api, bootstrap):
    most = bootstrap.MAX_LEVEL_RUNS
    assert api.ok("api_make_design", comparison(SUPPLIER, n_replicates=most, n_center_points=0))["n_runs"] == 2 * most
    error = api.error("api_make_design", comparison(SUPPLIER, n_replicates=most + 1))
    assert error == f"Replicates must be a whole number from 1 to {most}."
    two = {"design_type": "full_factorial", "factors": FACTORS[:2], "n_replicates": bootstrap.MAX_REPLICATES + 1}
    assert (
        api.error("api_make_design", two) == f"Replicates must be a whole number from 1 to {bootstrap.MAX_REPLICATES}."
    )


def test_a_brief_of_more_factors_has_no_plan_of_levels(api):
    request = {"factors": FACTORS, "model": "main_effects", "response": "y", "random_state": 7}
    assert api.ok("api_compare_plan", request)["levels"] is None


def test_a_limit_on_one_factor_alone_is_refused_with_what_to_do_instead(api):
    """No design on the page keeps one factor inside a limit: its range says the same, and that the page takes."""
    request = comparison(FACTORS[0], n_replicates=3, constraints=[{"expression": "T <= 190"}])
    assert api.error("api_make_design", request) == (
        "A limit on T alone narrows its range: set its low and high to the settings you can run, and remove the limit."
    )
