"""A limit that holds at some levels of a categorical factor (#813).

A tester's case: raw material A can be dosed up to 1.5 %, raw material B up to 2 %. The library reads such a
limit as ``not (RM == 'A' and conc > 1.5)``; these tests are of what the page makes of it, from the design to
the settings Learn finds.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.web._support import held_of, run_sheet

FACTORS = [
    {"name": "RM", "low": "A", "high": "B", "units": ""},
    {"name": "conc", "low": 0, "high": 2, "units": "%"},
    {"name": "temp", "low": 20, "high": 60, "units": "degC"},
]
UP_TO = "not (RM == 'A' and conc > 1.5)"
#: Ranges that do not overlap: A from 0 to 1.5 %, B from 2 to 3 %.
WIDE = [FACTORS[0], {**FACTORS[1], "high": 3}, FACTORS[2]]
APART = [UP_TO, "not (RM == 'B' and conc < 2)"]


def made(api, limits: list[str], factors: list[dict] = FACTORS) -> dict:
    constraints = [{"expression": text} for text in limits]
    return api.ok("api_make_design", {"design_type": "d_optimal", "factors": factors, "constraints": constraints})


def results(design: dict) -> list[float]:
    """Return results that rise with the concentration, a little higher with raw material A."""
    sheet = run_sheet(design)
    rng = np.random.default_rng(1)
    y = 50 + 6 * sheet["conc"] + 0.1 * sheet["temp"] + 2 * (sheet["RM"] == "A") + rng.normal(0, 0.3, len(sheet))
    return y.round(3).tolist()


def told(api, limits: list[str], factors: list[dict] = FACTORS, **more: object) -> dict:
    constraints = [{"expression": text} for text in limits]
    return api.ok("api_limits", {"factors": factors, "constraints": constraints, **more})


# --------------------------------------------------------------------------- the design
def test_a_design_keeps_each_level_inside_its_own_limit(api):
    design = made(api, [UP_TO])
    sheet = run_sheet(design)
    assert sheet.loc[sheet["RM"] == "A", "conc"].max() == pytest.approx(1.5), "A up to its limit"
    assert sheet.loc[sheet["RM"] == "B", "conc"].max() == pytest.approx(2.0), "B over the whole range"
    assert design["spec"]["constraints"] == [{"expression": UP_TO, "type": "linear"}]
    assert "Every run is inside your limits on the settings." in design["facts"]["notes"]


def test_ranges_that_do_not_overlap_are_two_limits(api):
    sheet = run_sheet(made(api, APART, WIDE))
    low, high = (sheet.loc[sheet["RM"] == level, "conc"] for level in "AB")
    assert (low.min(), low.max()) == pytest.approx((0.0, 1.5))
    assert (high.min(), high.max()) == pytest.approx((2.0, 3.0))


def test_a_design_that_cannot_keep_limits_still_says_so(api):
    request = {"design_type": "full_factorial", "factors": FACTORS, "constraints": [{"expression": UP_TO}]}
    assert "cannot keep its runs inside limits on the settings" in api.error("api_make_design", request)


# --------------------------------------------------------------------------- the limits panel
def test_the_panel_is_told_at_the_first_levels_the_limits_allow_then_at_the_reader_s(api):
    first = told(api, APART, WIDE)
    assert first["levels"]["factors"] == ["RM"]
    assert first["levels"]["chosen"] == {"RM": "A"}
    assert first["levels"]["combinations"] == [
        {"levels": {"RM": "A"}, "limits": ["conc <= 1.5"]},
        {"levels": {"RM": "B"}, "limits": ["conc >= 2"]},
    ]
    assert first["kept"] == pytest.approx(0.5), "0 to 1.5 of a range of 0 to 3"
    here, elsewhere = first["constraints"]
    assert (here["holds"], here["when"], here["removed"]) == ("conc <= 1.5", {"RM": "A"}, pytest.approx(0.5))
    assert (elsewhere["holds"], elsewhere["removed"]) == (None, 0.0), "a limit for B rules out nothing at A"
    second = told(api, APART, WIDE, levels={"RM": "B"})
    assert second["levels"]["chosen"] == {"RM": "B"}
    assert second["kept"] == pytest.approx(1 / 3), "2 to 3 of a range of 0 to 3"


def test_a_study_without_such_a_limit_has_no_levels_to_choose(api):
    assert told(api, ["conc + temp/40 <= 3"])["levels"] is None


def test_levels_the_limits_do_not_allow_fall_back_to_the_first(api):
    assert told(api, [UP_TO], levels={"RM": "Z"})["levels"]["chosen"] == {"RM": "A"}


# --------------------------------------------------------------------------- Learn
@pytest.fixture
def study(api) -> dict:
    design = made(api, [UP_TO])
    held = held_of(design)
    held["spec"]["response_info"] = {"goal": "maximize"}
    return {**held, "responses": results(design), "model": "main_effects"}


def test_the_best_settings_are_inside_the_limit_of_the_level_they_choose(api, study):
    """A gives 2 more, but only up to 1.5 %: B at 2 % gives 3 more than that, so B is the best."""
    best = api.ok("api_best_settings", study)
    found = {s["name"]: s.get("value", s.get("level")) for s in best["settings"]}
    assert (found["RM"], found["conc"]) == ("B", pytest.approx(2.0))
    assert best["limits"] == [], "a limit for A is not met at B"


def test_with_b_made_worse_the_best_settings_stop_at_a_s_limit(api, study):
    design = {key: study[key] for key in ("spec", "columns", "rows")}
    sheet = run_sheet(design)
    worse = [y - 10 * (level == "B") for y, level in zip(study["responses"], sheet["RM"], strict=True)]
    best = api.ok("api_best_settings", {**study, "responses": worse})
    found = {s["name"]: s.get("value", s.get("level")) for s in best["settings"]}
    assert (found["RM"], found["conc"]) == ("A", pytest.approx(1.5))
    assert [on["expression"] for on in best["limits"]] == [UP_TO], "the settings are on that limit"


def test_settings_break_a_limit_only_at_its_level(api, study):
    design = {"spec": study["spec"]}
    at_a = api.ok("api_outside_limits", {**design, "settings": {"RM": "A", "conc": 1.8, "temp": 40}})
    at_b = api.ok("api_outside_limits", {**design, "settings": {"RM": "B", "conc": 1.8, "temp": 40}})
    assert [b["expression"] for b in at_a["broken"]] == [UP_TO]
    assert at_b["broken"] == []


def test_a_target_is_met_with_settings_the_limit_allows(api, study):
    found = api.ok("api_target_settings", {**study, "targets": {"y": 64}})
    at = {s["name"]: s.get("value", s.get("level")) for s in found["settings"]}
    assert at["RM"] in "AB"
    assert at["conc"] <= (1.5 if at["RM"] == "A" else 2.0) + 1e-9


def test_a_specification_is_judged_at_each_level_in_its_own_region(api, study):
    judged = api.ok(
        "api_specification", {**study, "windows": {"y": [61, None]}, "at": {"RM": "A", "conc": 1, "temp": 40}}
    )
    rows = {row["levels"][0]["value"]: row for row in judged["levels"]}
    assert set(rows) == {"A", "B"}
    assert rows["A"]["here"]
    assert rows["A"]["share"] == pytest.approx(judged["share"]), "the reply's own share is that of the level it is at"


def test_the_charts_are_drawn_and_the_next_runs_keep_the_limit(api, study):
    api.ok("api_plots", study)
    api.ok("api_next_runs", study)


def test_the_trade_offs_are_drawn_in_the_region_at_the_balance_s_levels(api):
    responses = [{"name": "y", "goal": "maximize"}, {"name": "cost", "goal": "minimize"}]
    constraints = [{"expression": UP_TO}]
    design = api.ok(
        "api_make_design",
        {"design_type": "d_optimal", "factors": FACTORS, "responses": responses, "constraints": constraints},
    )
    sheet = run_sheet(design)
    cost = (10 + 3 * sheet["conc"]).round(3).tolist()
    request = {**held_of(design), "responses": results(design), "more_responses": {"cost": cost}}
    balance = api.ok("api_balance", {**request, "model": "main_effects"})
    api.ok("api_trade_offs", {**request, "model": "main_effects", "at": balance["coded"]})


# --------------------------------------------------------------------------- runs as given
def test_runs_as_given_are_judged_at_their_own_levels(api):
    runs = [["A", 1.0, 20], ["A", 1.8, 60], ["B", 1.8, 20]]
    request = {"design_type": "given", "factors": FACTORS, "runs": runs, "constraints": [{"expression": UP_TO}]}
    notes = " ".join(api.ok("api_make_design", request)["facts"]["notes"])
    assert "Run 2" in notes, "raw material A at 1.8 % is past its limit"
    assert "Run 3" not in notes, "raw material B at 1.8 % is not"
