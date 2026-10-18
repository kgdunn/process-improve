"""Several responses: a study that measures a yield, a cost and a purity in the same runs.

The brief lists the responses in a table, as it lists the factors, and the first one sizes the
design. The run sheet has a column for each, practice results are simulated for each, and Learn
has a tab per response above the analysis, each with its own model. A study of one response, saved
before the page measured several, opens as it was.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tests.web._support import FACTORS
from tests.web.e2e._page import (
    SAVED,
    calls,
    candidates_done,
    generate,
    headers,
    last_reply,
    name_response,
    practise,
    reveal,
    run_sheet,
    saved_studies,
    set_factors,
    settled,
    status,
)
from tests.web.e2e.test_layout import overflowing
from tests.web.e2e.test_workbook import upload

PHONE = {"viewport": {"width": 390, "height": 844}, "is_mobile": True}

#: The later responses, as a reader types them: a name, units, a goal and a target. Practice results put the purity
#: near 62, the scenario's own centre, so its target is one the settings can reach.
LATER = [("Cost", "$/kg", "minimize", None), ("Purity", "%", "target", 63)]

#: Each response's results on the run sheet, as the reader sees them: its column and the value in its box.
RESULTS = (
    "() => [...document.querySelectorAll('#design-table input[data-response]')]"
    ".map((i) => [i.dataset.response, i.value])"
)


def add_response(page: Any, label: str, units: str = "", goal: str = "learn", target: object = None) -> None:  # noqa: ANN401
    """Add a response to the brief, as a reader would: a new row, its name, units, goal and target."""
    reveal(page, "#add-response")
    page.click("#add-response")
    row = page.locator("#responses tbody tr").last
    row.locator("[name=label]").fill(label)
    row.locator("[name=units]").fill(units)
    row.locator("[name=goal]").select_option(goal)
    if target is not None:
        row.locator("[name=target]").fill(str(target))
    row.locator("[name=label]").press("Tab")  # a name is taken when the reader leaves its box


def three_responses(page: Any) -> None:  # noqa: ANN401
    """Describe the yield, the cost and the purity, type the worked example's factors, and make a Box-Behnken design."""
    name_response(page, "Yield", "%", goal="maximize")
    for label, units, goal, target in LATER:
        add_response(page, label, units, goal, target)
    set_factors(page, FACTORS)
    candidates_done(page)
    generate(page, design_type="box_behnken")


def tabs(page: Any) -> list[tuple[str, str]]:  # noqa: ANN401
    """Read Learn's tabs, one per response: each one's name, and whether it is the one chosen."""
    return page.evaluate(
        "() => [...document.querySelectorAll('#response-picker [role=tab]')]"
        ".map((t) => [t.textContent, t.getAttribute('aria-selected')])"
    )


def pick(page: Any, label: str) -> None:  # noqa: ANN401
    """Choose a response's tab in Learn, as a reader would, and wait for its analysis."""
    before = page.evaluate("window.__pi.count('api_analyze_runs')")
    reveal(page, "#response-picker")
    page.get_by_role("tab", name=label).click()
    page.wait_for_function(f"window.__pi.count('api_analyze_runs') > {before}")
    settled(page)


def equation(page: Any) -> str:  # noqa: ANN401
    return page.text_content("#analysis .equation")


# --------------------------------------------------------------------------- the brief
def test_the_brief_lists_the_responses_in_a_table_and_says_the_first_sizes_the_design(fresh_page):
    page = fresh_page
    assert page.locator("#responses tbody tr").count() == 1
    assert page.is_hidden("#first-sizes"), "one response: no first to speak of"
    assert page.get_by_role("button", name="Remove response").count() == 0, "the first response stays"
    name_response(page, "Yield", "%", goal="maximize")
    for label, units, goal, target in LATER:
        add_response(page, label, units, goal, target)
    assert page.is_visible("#first-sizes")
    assert (
        page.text_content("#first-sizes")
        == "The first response sizes the design, so the page asks what you know of it."
    )
    assert page.text_content("#response-known summary") == "What you know of Yield already"
    rows = page.locator("#responses tbody tr")
    assert rows.count() == 3
    assert rows.nth(1).locator("[name=target]").is_disabled(), "a target only for a goal of hitting one"
    assert rows.nth(2).locator("[name=target]").input_value() == "63"
    assert page.text_content("#study-title") == "Yield, Cost and Purity"


def test_a_study_measures_at_most_five_responses(fresh_page):
    page = fresh_page
    name_response(page, "Yield")
    for label in ("Cost", "Purity", "Colour", "Haze"):
        add_response(page, label)
    assert page.is_disabled("#add-response")
    assert page.text_content("#responses-full") == "A study measures at most 5 responses."
    page.get_by_role("button", name="Remove response").last.click()
    assert page.is_enabled("#add-response")


def test_two_responses_of_one_name_are_said_and_hold_the_run_sheet_back(fresh_page):
    page = fresh_page
    name_response(page, "Yield")
    add_response(page, "yield")
    set_factors(page, FACTORS)
    assert page.text_content("#response-problems") == "yield: two responses cannot share a name."
    reveal(page, "#generate")
    assert page.is_disabled("#generate")
    page.locator("#responses tbody tr").last.locator("[name=label]").fill("Cost")
    candidates_done(page)
    assert page.is_hidden("#response-problems")
    assert page.is_enabled("#generate")


def test_a_brief_of_several_responses_is_kept_as_a_draft(fresh_page):
    page = fresh_page
    name_response(page, "Yield", "%", goal="maximize")
    add_response(page, "Cost", "$/kg", "minimize")
    page.wait_for_timeout(500)  # the draft is saved a moment after the last change
    page.reload()
    page.wait_for_function("window.__pi.count('api_catalogue') === 1 && window.__pi.idle()")
    rows = page.locator("#responses tbody tr")
    assert rows.count() == 2
    assert rows.nth(1).locator("[name=label]").input_value() == "Cost"
    assert rows.nth(1).locator("[name=goal]").input_value() == "minimize"


# --------------------------------------------------------------------------- the run sheet
def test_three_responses_are_made_simulated_and_analysed_response_by_response(fresh_page):
    page = fresh_page
    three_responses(page)
    assert headers(page)[-3:] == ["Yield", "Cost", "Purity"], "a column for each response"
    made = calls(page, "api_make_design")[-1]["payload"]
    assert [r["name"] for r in made["responses"]] == ["Yield", "Cost", "Purity"]
    assert page.text_content("#run-source") == (
        "No results yet: type them into the Yield, Cost and Purity columns as the runs are made, or paste them from a "
        "spreadsheet."
    )
    practise(page)
    assert page.locator("#design-table input.sim").count() == 3 * 15, "every result simulated, in italics"
    simulated = last_reply(page, "api_simulate")["result"]
    assert set(simulated["more"]) == {"Cost", "Purity"}
    assert page.text_content("#badge-run") == "15 of 15"
    assert page.locator("#truth .equation").count() == 3, "each response's true process"

    reveal(page, "#response-picker")
    assert tabs(page) == [["Yield", "true"], ["Cost", "false"], ["Purity", "false"]]
    assert equation(page).startswith("Yield = ")
    pick(page, "Cost")
    assert tabs(page)[1] == ["Cost", "true"]
    assert equation(page).startswith("Cost = ")
    assert calls(page, "api_analyze_runs")[-1]["payload"]["response"] == "Cost"
    assert calls(page, "api_best_settings")[-1]["payload"]["response"] == "Cost"
    assert page.text_content("#best .lead").startswith("These settings give the lowest Cost the model predicts: ")
    contour = page.locator("#charts .js-plotly-plot").nth(1)
    assert "Cost ($/kg)" in contour.evaluate("(d) => d.data[0].colorbar.title.text")


def test_the_response_tabs_answer_the_arrow_keys(fresh_page):
    page = fresh_page
    three_responses(page)
    practise(page)
    reveal(page, "#response-picker")
    page.get_by_role("tab", name="Yield").focus()
    before = page.evaluate("window.__pi.count('api_analyze_runs')")
    page.keyboard.press("ArrowLeft")
    page.wait_for_function(f"window.__pi.count('api_analyze_runs') > {before}")
    settled(page)
    assert tabs(page)[2] == ["Purity", "true"], "from the first tab, left goes round to the last"
    assert page.evaluate("document.activeElement.textContent") == "Purity"
    assert equation(page).startswith("Purity = ")
    assert page.text_content("#best .lead").startswith("These settings give Purity on its target, 63%")


def test_each_response_keeps_its_own_model(fresh_page):
    page = fresh_page
    three_responses(page)
    practise(page)
    pick(page, "Cost")
    page.select_option("#model", "main_effects")
    settled(page)
    assert calls(page, "api_analyze_runs")[-1]["payload"]["model"] == "main_effects"
    pick(page, "Yield")
    assert page.input_value("#model") != "main_effects", "the yield keeps the model it had"
    pick(page, "Cost")
    assert page.input_value("#model") == "main_effects"
    saved = saved_studies(page, lambda s: bool(s) and s[0].get("more_models", {}).get("Cost") == "main_effects")
    assert saved[0]["more_models"]["Cost"] == "main_effects"


def test_a_run_with_some_of_its_results_is_fitted_for_the_responses_it_has(fresh_page):
    page = fresh_page
    three_responses(page)
    practise(page)
    reveal(page, "#design-table")
    page.fill('#design-table input[data-response="Cost"][data-index="2"]', "")
    page.wait_for_timeout(800)  # the analysis waits for the typing to rest
    settled(page)
    assert page.text_content("#badge-run") == "14 of 15", "a run is complete when every response has its result"
    assert page.text_content("#run-source").startswith(
        '14 of 15 runs\' results simulated from "Reactor yield", each response'
    )
    reveal(page, "#analysis")
    lines = page.locator("#analysis ul.fit li").all_text_contents()
    run = run_sheet(page)[2][1]
    assert lines[0] == "14 completed runs"
    assert lines[1] == (
        f"Run {run} has some of its results, not all. Each response is fitted to the runs that have its result: "
        "Yield to 15."
    )
    pick(page, "Cost")
    lines = page.locator("#analysis ul.fit li").all_text_contents()
    assert lines[1].endswith("Cost to 14.")
    assert lines[2] == f"Still to run: {run}"


def test_a_response_added_to_a_run_sheet_takes_a_column_and_one_removed_leaves_with_its_results(fresh_page):
    page = fresh_page
    three_responses(page)
    practise(page)
    add_response(page, "Haze", "NTU", "minimize")
    page.wait_for_function(
        "[...document.querySelectorAll('#design-table thead th')].some((t) => t.textContent.startsWith('Haze'))"
    )
    settled(page)
    assert headers(page)[-4:] == ["Yield", "Cost", "Purity", "Haze"]
    assert page.locator('#design-table input.sim[data-response="Haze"]').count() == 15, "practice: simulated too"
    reveal(page, "#response-picker")
    assert [t[0] for t in tabs(page)] == ["Yield", "Cost", "Purity", "Haze"]

    reveal(page, "#responses")
    page.locator("#responses tbody tr").nth(1).get_by_role("button", name="Remove response").click()
    settled(page)
    assert headers(page)[-3:] == ["Yield", "Purity", "Haze"]
    assert status(page)[0] == "Cost is off the run sheet, with its 15 results." or "Cost" not in headers(page)
    reveal(page, "#response-picker")
    assert [t[0] for t in tabs(page)] == ["Yield", "Purity", "Haze"]


def test_a_study_of_three_responses_is_saved_and_opens_again(fresh_page):
    page = fresh_page
    three_responses(page)
    practise(page)
    saved = saved_studies(page, lambda s: bool(s) and bool(s[0].get("more_responses", {}).get("Purity", [None])[0]))
    record = saved[0]
    assert [r["name"] for r in record["design"]["spec"]["responses"]] == ["Yield", "Cost", "Purity"]
    assert len(record["responses"]) == 15, "the first response's results, where a study of one keeps them"
    page.reload()
    page.wait_for_function("window.__pi.idle() && !document.querySelector('#analysis').hidden", timeout=30_000)
    settled(page)
    assert page.locator("#responses tbody tr").count() == 3
    reveal(page, "#response-picker")
    assert [t[0] for t in tabs(page)] == ["Yield", "Cost", "Purity"]
    assert page.locator("#design-table input.sim").count() == 45


def test_a_workbook_of_three_responses_comes_back_with_them(fresh_page, tmp_path):
    page = fresh_page
    three_responses(page)
    practise(page)
    pick(page, "Cost")
    page.select_option("#model", "interactions")
    settled(page)
    results = page.evaluate(RESULTS)
    reveal(page, "#download")
    with page.expect_download() as download:
        page.click("#download")
    path = tmp_path / "three.xlsx"
    download.value.save_as(path)
    sent = calls(page, "api_workbook")[-1]["payload"]
    assert sent["models"] == {"Cost": "interactions"}, "the cost's model, kept in the workbook"
    upload(page, path)  # in place of the study on the page, as a workbook sent back would be
    page.wait_for_function("window.__pi.count('api_read_workbook') === 1 && window.__pi.idle()", timeout=30_000)
    settled(page)
    assert page.evaluate(RESULTS) == results
    reveal(page, "#response-picker")
    assert [t[0] for t in tabs(page)] == ["Yield", "Cost", "Purity"]
    pick(page, "Cost")
    assert page.input_value("#model") == "interactions"


# --------------------------------------------------------------------------- a study made before
def test_a_study_of_one_response_saved_before_opens_as_it_was(context, site, api):
    """A study saved before the page measured several responses: its record, word for word, opens unchanged."""
    design = api.ok(
        "api_make_design",
        {
            "design_type": "box_behnken",
            "factors": FACTORS,
            "response": "Yield",
            "response_info": {"label": "Yield", "units": "%", "goal": "maximize"},
        },
    )
    assert "responses" not in design["spec"]
    held = {"spec": design["spec"], "columns": design["columns"], "rows": design["rows"]}
    simulated = api.ok("api_simulate", {**held, "scenario": "reactor", "sigma": 1.0, "seed": 1})
    record = {
        "v": 1,
        "title": "Yield (%), maximise: Box-Behnken in T, P and F",
        "design": {**held, "model": design["model"], "n_runs": 15, "facts": design["facts"], "outside_range": []},
        "responses": simulated["responses"],
        "sources": ["simulated"] * 15,
        "simulation": {k: v for k, v in simulated.items() if k != "responses"},
        "simulate": {"scenario": "reactor", "sigma": 1.0, "seed": 1},
        "model": "",
        "id": "saved-before",
        "created": 1,
        "updated": 2,
    }
    page = context.new_page()
    page.goto(site)
    page.wait_for_function("window.__pi.count('api_catalogue') === 1 && window.__pi.idle()")
    page.evaluate(f"async () => (await import('./store.js')).saveCampaign({json.dumps(record)})")
    page.reload()
    page.wait_for_function("window.__pi.idle() && !document.querySelector('#analysis').hidden", timeout=30_000)
    settled(page)
    assert page.locator("#responses tbody tr").count() == 1
    assert page.input_value("#response") == "Yield"
    assert page.input_value("#response-goal") == "maximize"
    assert page.is_hidden("#response-picker"), "one response: no tabs"
    assert equation(page).startswith("Yield = ")
    assert "response" not in calls(page, "api_analyze_runs")[-1]["payload"], "asked as a study of one response asks"
    stored = page.evaluate(SAVED)
    assert "more_responses" not in stored[0], "and saved again as one"


# --------------------------------------------------------------------------- a phone
@pytest.mark.parametrize("context_options", [PHONE])
def test_three_responses_fit_a_phone_without_sideways_scrolling(fresh_page):
    page = fresh_page
    three_responses(page)
    practise(page)
    for phase in ("plan", "run", "learn"):
        page.click(f"#go-{phase}")
        # The charts drawn while Learn was hidden are fitted to it as it is shown, which takes Plotly a moment.
        page.wait_for_function("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
        assert overflowing(page) == [], phase
    reveal(page, "#response-picker")
    box = page.locator("#response-picker").bounding_box()
    assert box["x"] + box["width"] <= 390
    pick(page, "Purity")  # a response chosen on a phone: its charts are drawn in the width there is
    page.wait_for_function("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
    assert overflowing(page) == []
