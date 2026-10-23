"""Results read more than once at a setting: their spread, Taguchi's signal-to-noise ratios and the response table.

The analysis (``api_analyze_runs``) takes the runs at one setting as readings of it. Each setting read twice or more
gives its readings' mean, standard deviation and ln s, and, for a response with a goal, the signal-to-noise ratio of
that goal: larger is better to maximise, smaller is better to minimise, nominal is best for a target. The response
table averages each over the settings at each level of a factor. Schmidt and Launsby's signal-to-noise example, a
2^(3-1) read twice at each setting, prints the table for all three ratios, and the page reproduces it.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from tests.web._support import FACTORS, held_of, run_sheet

#: Schmidt and Launsby's 2^(3-1), C = AB in levels 1 and 2, and the two readings at each of its settings.
SETTINGS = [[1, 1, 2], [2, 1, 1], [1, 2, 1], [2, 2, 2]]
READINGS = [[20, 30], [9, 11], [26, 24], [15, 5]]
LEVELS_1_2 = [{"name": n, "low": 1, "high": 2, "units": ""} for n in "ABC"]


def analysed(api, goal: str | None = None, readings: list[list] = READINGS, settings: list[list] = SETTINGS) -> dict:
    """Analyse each setting run once for each of its readings, in that order, with the response's goal."""
    info = {"goal": goal, "target": 20} if goal else {}
    runs = [s for s, values in zip(settings, readings, strict=True) for _ in values]
    design = api.ok(
        "api_make_design",
        {"design_type": "given", "factors": LEVELS_1_2, "runs": runs, "response": "y", "response_info": info},
    )
    results = [None if v is None else float(v) for values in readings for v in values]
    return api.ok("api_analyze_runs", {**held_of(design), "responses": results})


def averages(spread: dict, summary: str) -> dict[str, list[float]]:
    """Return each factor's averages of a summary, at its levels in their order."""
    return {f["name"]: [level[summary] for level in f["levels"]] for f in spread["factors"]}


# --------------------------------------------------------------------------- Schmidt and Launsby
@pytest.mark.parametrize(
    ("goal", "kind", "printed"),
    [
        ("maximize", "larger", {"A": [27.69, 18.20], "B": [23.65, 22.24], "C": [23.90, 21.98]}),
        ("minimize", "smaller", {"A": [-28.05, -20.51], "B": [-24.09, -24.47], "C": [-24.00, -24.55]}),
        ("target", "nominal", {"A": [17.96, 10.00], "B": [13.98, 13.98], "C": [20.97, 6.99]}),
    ],
)
def test_each_goal_s_ratio_averages_to_schmidt_and_launsby_s_table(api, goal, kind, printed):
    spread = analysed(api, goal)["spread"]
    assert spread["kind"] == kind
    found = averages(spread, "sn")
    for factor, values in printed.items():
        np.testing.assert_allclose(found[factor], values, atol=0.006, err_msg=factor)
        delta = next(f["delta"]["sn"] for f in spread["factors"] if f["name"] == factor)
        assert delta == pytest.approx(abs(values[1] - values[0]), abs=0.011)


def test_ln_s_tells_the_location_effect_from_the_dispersion_effect(api):
    """A moves the mean and not the spread, C the spread and not the mean; nominal is best mixes the two."""
    spread = analysed(api, "target")["spread"]
    assert averages(spread, "mean") == {"A": [25.0, 10.0], "B": [17.5, 17.5], "C": [17.5, 17.5]}
    ln_s = averages(spread, "ln_std_dev")
    assert ln_s["A"][0] == pytest.approx(ln_s["A"][1])
    np.testing.assert_allclose(ln_s["C"], [np.log(np.sqrt(2)), np.log(np.sqrt(50))])
    best = {f["name"]: f["best"] for f in spread["factors"]}
    assert best["C"] == {"ln_std_dev": 1.0, "sn": 1.0}, "level 1 of C has the smaller spread and the higher ratio"
    assert best["A"] == {"sn": 1.0}, "the ratio moves with A's mean, the spread does not"
    assert best["B"] == {}, "levels that tie have no best"
    assert spread["adjustment"] == "A", "A moves the mean to the target without moving the spread"


def test_each_setting_sums_up_its_own_readings(api):
    spread = analysed(api, "target")["spread"]
    by_runs = {tuple(s["runs"]): s for s in spread["settings"]}
    assert sorted(by_runs) == [(1, 2), (3, 4), (5, 6), (7, 8)]
    first = by_runs[(1, 2)]
    assert first["settings"] == {"A": 1.0, "B": 1.0, "C": 2.0}
    assert first["n"] == 2
    assert first["mean"] == 25.0
    assert first["std_dev"] == pytest.approx(np.sqrt(50))
    assert first["ln_std_dev"] == pytest.approx(np.log(np.sqrt(50)))
    assert first["sn"] == pytest.approx(10 * np.log10(625 / 50))
    assert spread["left_out"] == []


def test_the_factors_are_ranked_by_their_delta(api):
    spread = analysed(api, "target")["spread"]
    ranks = {f["name"]: f["rank"] for f in spread["factors"]}
    assert ranks == {
        "A": {"mean": 1, "ln_std_dev": 2, "sn": 2},
        "B": {"mean": 2, "ln_std_dev": 2, "sn": 3},
        "C": {"mean": 2, "ln_std_dev": 1, "sn": 1},
    }, "1 for the largest delta; factors that tie share a rank"


# --------------------------------------------------------------------------- what the goal reads
@pytest.mark.parametrize("goal", ["learn", None])
def test_without_a_goal_to_reach_there_is_no_ratio(api, goal):
    spread = analysed(api, goal)["spread"]
    assert spread["kind"] is None
    assert all(s["sn"] is None for s in spread["settings"])
    assert all("sn" not in level for f in spread["factors"] for level in f["levels"])
    assert spread["adjustment"] is None
    assert {f["name"]: f["best"] for f in spread["factors"]}["C"] == {"ln_std_dev": 1.0}


@pytest.mark.parametrize("goal", ["maximize", "minimize"])
def test_only_a_target_has_an_adjustment_factor(api, goal):
    assert analysed(api, goal)["spread"]["adjustment"] is None


def test_no_factor_adjusts_the_mean_that_moves_the_spread_as_much(api):
    """C doubles the mean and the spread together: it moves both by the largest delta of each, so it adjusts neither."""
    readings = [[18, 22] if c == 2 else [9, 11] for _, _, c in SETTINGS]
    assert analysed(api, "target", readings)["spread"]["adjustment"] is None


# --------------------------------------------------------------------------- what is left out
def test_runs_read_once_have_no_spread(api):
    assert analysed(api, "target", [[25], [10], [25], [10]])["spread"] is None


def test_a_setting_repeated_alone_is_no_table(api):
    """A centre run read three times, beside corners read once, is one setting with a spread: no table."""
    runs = [*SETTINGS, [1.5, 1.5, 1.5], [1.5, 1.5, 1.5], [1.5, 1.5, 1.5]]
    design = api.ok("api_make_design", {"design_type": "given", "factors": LEVELS_1_2, "runs": runs, "response": "y"})
    analysis = api.ok("api_analyze_runs", {**held_of(design), "responses": [25.0, 10, 25, 10, 17, 18, 19]})
    assert analysis["spread"] is None


def test_a_study_of_one_factor_has_its_levels_compared_instead(api):
    """Each level of one factor is a setting: the comparison of its levels already gives their means and spreads."""
    factor = [{"name": "A", "low": 1, "high": 2, "units": ""}]
    runs = [[1], [1], [1], [2], [2], [2]]
    design = api.ok("api_make_design", {"design_type": "given", "factors": factor, "runs": runs, "response": "y"})
    analysis = api.ok("api_analyze_runs", {**held_of(design), "responses": [20.0, 30, 25, 9, 11, 10]})
    assert analysis["comparison"]["difference"] is not None
    assert analysis["spread"] is None


def test_a_setting_read_once_is_left_out_and_named_by_its_run(api):
    readings = [[20, 30], [9, 11], [26, 24], [15, None]]
    spread = analysed(api, "target", readings)["spread"]
    assert spread["left_out"] == [7]
    assert len(spread["settings"]) == 3
    assert averages(spread, "mean")["A"] == [25.0, 10.0], "level 2 of A is the one setting read twice"
    assert [level["n"] for level in spread["factors"][0]["levels"]] == [2, 1]


def test_readings_that_do_not_vary_give_no_ln_s_and_are_left_out_of_its_average(api):
    readings = [[20, 30], [10, 10], [26, 24], [15, 5]]
    spread = analysed(api, "target", readings)["spread"]
    flat = next(s for s in spread["settings"] if s["runs"] == [3, 4])
    assert flat["std_dev"] == 0.0
    assert flat["ln_std_dev"] is None
    assert flat["sn"] is None, "nominal is best of readings that do not vary is infinite"
    assert averages(spread, "ln_std_dev")["A"][1] == pytest.approx(np.log(np.sqrt(50))), "the other setting alone"
    json.dumps(spread, allow_nan=False)


def test_larger_is_better_of_a_reading_of_zero_is_no_ratio(api):
    readings = [[20, 30], [0, 11], [26, 24], [15, 5]]
    flat = next(s for s in analysed(api, "maximize", readings)["spread"]["settings"] if s["runs"] == [3, 4])
    assert flat["sn"] is None
    assert flat["mean"] == 5.5


# --------------------------------------------------------------------------- other designs
def test_a_replicated_taguchi_array_at_three_levels_has_three_levels_a_factor(api):
    design = api.ok(
        "api_make_design",
        {
            "design_type": "taguchi",
            "factors": [*FACTORS, {"name": "Supplier", "low": "A", "high": "B", "units": ""}],
            "n_levels": 3,
            "n_replicates": 2,
            "response": "Yield",
            "response_info": {"goal": "maximize"},
        },
    )
    sheet = run_sheet(design)
    rng = np.random.default_rng(7)
    results = (60 + 0.05 * sheet["T"] + rng.normal(0, 1, len(sheet))).round(2).tolist()
    spread = api.ok("api_analyze_runs", {**held_of(design), "responses": results})["spread"]
    assert len(spread["settings"]) == 18
    assert all(s["n"] == 2 for s in spread["settings"])
    levels = {f["name"]: [level["level"] for level in f["levels"]] for f in spread["factors"]}
    assert levels["T"] == [150.0, 175.0, 200.0]
    assert levels["Supplier"] == ["A", "B"]
    assert [level["n"] for level in spread["factors"][0]["levels"]] == [6, 6, 6]
