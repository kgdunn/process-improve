"""What if, in Learn: the reader sets each factor, and every response's model predicts there.

The panel opens at the best settings, or at the balanced settings once two responses or more have a
goal, and predicts there what they show. Moving a setting changes every response at once, in the
page's own JavaScript, without a call to Python; settings past a limit on the settings name it.
Settings kept under a name go with the study, through this browser and a link, and are listed
below the panel with each response there; they can become confirmation runs.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from tests.web._support import FACTORS
from tests.web.e2e._page import (
    candidates_done,
    generate,
    last_reply,
    practise,
    reveal,
    run_sheet,
    saved_studies,
    set_factors,
    settled,
    type_limit,
)
from tests.web.e2e.test_best_settings import num, table_rows
from tests.web.e2e.test_layout import WIDE_FONT, overflowing
from tests.web.e2e.test_responses import PHONE, three_responses


def rows(page: Any) -> list[list[str]]:  # noqa: ANN401
    """Read the what-if table, a row per response, as the reader sees it."""
    return table_rows(page, "#what-if-out")


def predictions(page: Any) -> int:  # noqa: ANN401
    """Count the calls the page made for the fits it predicts from."""
    return page.evaluate("window.__pi.count('api_predictors')")


def set_number(page: Any, factor: str, value: float) -> None:  # noqa: ANN401
    """Type a setting in a factor's number box and leave it, as a reader would."""
    box = page.locator(f'#what-if-settings [data-factor="{factor}"] input[name="number"]')
    box.fill(str(value))
    box.press("Tab")


def predicted_by_hand(fit: dict, factors: list[dict], settings: dict) -> float:
    """Work out a fit's prediction at numeric settings from the numbers Python sent: a term is its columns' product."""
    middle = {f["name"]: ((f["high"] + f["low"]) / 2, (f["high"] - f["low"]) / 2) for f in factors}
    coded = {name: (settings[name] - centre) / half for name, (centre, half) in middle.items()}
    terms = zip(fit["terms"], fit["coefficients"], strict=True)
    return sum(c * math.prod(coded[p] for p in parts) for parts, c in terms)


@pytest.fixture
def three(fresh_page: Any) -> Any:  # noqa: ANN401 - a Playwright Page
    """Return a study of a yield, a cost and a purity, each with a goal, its results simulated and analysed."""
    three_responses(fresh_page)
    practise(fresh_page)
    return fresh_page


def test_the_worked_example_opens_at_its_best_settings_and_follows_each_move(page, first_paint):
    reveal(page, "#what-if")
    page.wait_for_selector("#what-if:not([hidden])")
    best = first_paint["best"]
    assert page.text_content("#h-what-if") == "What if"
    assert rows(page) == [rows(page)[0]], "one row: the worked example measures one response"
    assert rows(page)[0][:2] == ["Yield", f"{num(best['predicted'])}%"], "the prediction the best settings show"
    shown = {s["name"]: num(s["value"]) for s in best["settings"]}
    boxes = page.evaluate(
        "() => Object.fromEntries([...document.querySelectorAll('#what-if-settings [data-factor]')]"
        ".map((row) => [row.dataset.factor, row.querySelector('input[name=number]').value]))"
    )
    assert boxes == shown
    assert page.is_hidden("#what-if-back"), "at the best settings there is nowhere to come back to"

    before = predictions(page)
    set_number(page, "T", 160)
    moved = {s["name"]: s["value"] for s in best["settings"]} | {"T": 160}
    [fit] = first_paint["predictors"]["responses"]
    expected = predicted_by_hand(fit, first_paint["predictors"]["factors"], moved)
    assert rows(page)[0][1] == f"{num(expected)}%"
    assert predictions(page) == before, "the page predicts in its own JavaScript, without asking Python"
    assert page.text_content("#what-if-back") == "Back to the best settings"

    page.click("#what-if-back")
    assert rows(page)[0][1] == f"{num(best['predicted'])}%"
    assert page.is_hidden("#what-if-back")
    assert "new_points=at" in page.text_content("#py-what-if code"), "the code that makes the same predictions"


def test_a_number_past_its_range_is_held_at_the_end_of_it(page, first_paint):
    reveal(page, "#what-if")
    page.wait_for_selector("#what-if:not([hidden])")
    set_number(page, "T", 260)
    box = page.locator('#what-if-settings [data-factor="T"] input[name="number"]')
    assert box.input_value() == "200", "the models hold inside the range they were fitted over"


def test_with_three_responses_the_panel_opens_at_the_balanced_settings(three):
    page = three
    reveal(page, "#what-if")
    page.wait_for_selector("#what-if:not([hidden])")
    balanced = last_reply(page, "api_balance")["result"]
    table = rows(page)
    assert [row[0] for row in table] == ["Yield", "Cost", "Purity"]
    units = {"Yield": "%", "Cost": " $/kg", "Purity": "%"}
    assert [row[1] for row in table] == [f"{num(r['predicted'])}{units[r['name']]}" for r in balanced["responses"]]
    set_number(page, "P", 2.5)
    assert page.text_content("#what-if-back") == "Back to the balanced settings"


def test_settings_past_a_limit_name_it(fresh_page):
    page = fresh_page
    set_factors(page, FACTORS)
    candidates_done(page)
    type_limit(page, "T + 25 * P <= 215")
    generate(page, design_type="d_optimal")
    practise(page)
    reveal(page, "#what-if")
    page.wait_for_selector("#what-if:not([hidden])")
    before = page.evaluate("window.__pi.count('api_outside_limits')")
    set_number(page, "T", 200)
    set_number(page, "P", 3)
    page.wait_for_function(f"window.__pi.count('api_outside_limits') > {before}")
    settled(page)
    limit = last_reply(page, "api_outside_limits")["result"]["broken"][0]["expression"]
    warnings = page.locator("#what-if-out ul.warnings li").all_text_contents()
    assert f"These settings break your limit 1, {limit}." in warnings


@pytest.mark.parametrize("context_options", [PHONE])
def test_on_a_phone_the_panel_fits(page):
    page.add_style_tag(content=WIDE_FONT)
    reveal(page, "#what-if")
    page.wait_for_selector("#what-if:not([hidden])")
    keep(page, "Settings with a name long enough to need more than one line on a phone")
    page.wait_for_selector("#what-if-kept:not([hidden])")
    assert overflowing(page) == []


# --------------------------------------------------------------------------- settings kept with the study
#: The box the reader names the panel's settings in, to keep them.
KEEP = "#what-if-keep-name"


def keep(page: Any, name: str = "") -> None:  # noqa: ANN401
    """Keep the panel's settings under ``name`` (none typed: the page names them), from the keyboard."""
    page.fill(KEEP, name)
    page.press(KEEP, "Enter")


def kept(page: Any) -> list[dict]:  # noqa: ANN401
    """Read the settings kept with the study as the list shows them: each one's line, table, sentence and buttons."""
    return page.evaluate(
        """() => [...document.querySelectorAll('#what-if-kept-list > li')].map((li) => ({
            said: li.querySelector('p').textContent,
            rows: [...li.querySelectorAll('tbody tr')].map((tr) => [...tr.cells].map((c) => c.textContent)),
            hint: li.querySelector('p.hint')?.textContent ?? null,
            buttons: [...li.querySelectorAll('.actions button')].map((b) => b.getAttribute('aria-label')),
        }))"""
    )


def names(page: Any) -> list[str]:  # noqa: ANN401
    """Name the settings the list holds, in its order."""
    return [one["said"].split(": ")[0] for one in kept(page)]


def in_learn(page: Any) -> Any:  # noqa: ANN401
    """Show Learn with the what-if panel, once the analysis is done."""
    reveal(page, "#what-if")
    page.wait_for_selector("#what-if:not([hidden])")
    return page


@pytest.fixture
def studied(blank_page: Any) -> Any:  # noqa: ANN401 - a Playwright Page
    """Return a study of the three factors, its results simulated and analysed, shown in Learn at the what-if panel."""
    generate(blank_page)
    practise(blank_page)
    return in_learn(blank_page)


def test_settings_kept_under_a_name_are_listed_with_each_response_there(studied):
    page = studied
    assert page.is_hidden("#what-if-kept"), "nothing is kept yet"
    set_number(page, "T", 160)
    there = rows(page)
    keep(page, "Cooler")
    page.wait_for_selector("#what-if-kept:not([hidden])")
    [cooler] = kept(page)
    assert cooler["said"].startswith("Cooler: T = 160 degC, P = ")
    assert cooler["rows"] == there, "each response at the kept settings, as the panel showed it"
    assert cooler["buttons"] == [
        "Add 3 confirmation runs at the settings kept as Cooler",
        "Delete the settings kept as Cooler",
    ]
    assert page.input_value(KEEP) == "", "ready to name the next"

    set_number(page, "T", 190)
    assert kept(page)[0]["rows"] == there, "the panel moves on; the kept settings stay where they were kept"
    page.click("#what-if-keep button[type=submit]")
    assert names(page) == ["Cooler", "Settings 2"], "a name the page gives, when none is typed"
    keep(page, "Cooler")
    assert page.text_content("#toast") == "Settings are kept as Cooler already: give these another name."
    assert names(page) == ["Cooler", "Settings 2"]
    assert "Cooler" in page.text_content("#py-what-if code"), "the code predicts at the kept settings too"

    page.click('[aria-label="Delete the settings kept as Settings 2"]')
    assert names(page) == ["Cooler"]
    assert page.evaluate("document.activeElement.id") == "what-if-keep-name", "the keyboard stays in the panel"
    [study] = saved_studies(page, lambda saved: len(saved) == 1 and len(saved[0].get("what_if", [])) == 1)
    assert study["v"] == 5, "a study with kept settings is saved in the format that holds them"
    assert study["what_if"][0]["name"] == "Cooler"
    assert study["what_if"][0]["settings"]["T"] == 160


def test_kept_settings_go_out_by_link_and_come_back_with_the_study(open_context, context, studied):
    page = studied
    set_number(page, "T", 160)
    keep(page, "Cooler, more pressure")
    page.wait_for_selector("#what-if-kept:not([hidden])")
    listed = kept(page)
    reveal(page, "#share")
    page.click("#share")
    page.wait_for_selector("#share-out", state="visible")

    elsewhere = open_context().new_page()
    elsewhere.goto(page.input_value("#share-url"))
    elsewhere.wait_for_selector("#campaign-banner", state="visible")
    elsewhere.wait_for_function("window.__pi.count('api_catalogue') === 1 && window.__pi.idle()")
    settled(elsewhere)
    in_learn(elsewhere)
    elsewhere.wait_for_selector("#what-if-kept:not([hidden])")
    assert kept(elsewhere) == listed, "the settings, and each response there worked out again from the results"

    again = context.new_page()
    again.goto(page.url.split("#")[0])
    again.wait_for_selector("#campaign-banner", state="visible")
    again.wait_for_function("window.__pi.count('api_catalogue') === 1 && window.__pi.idle()")
    settled(again)
    in_learn(again)
    again.wait_for_selector("#what-if-kept:not([hidden])")
    assert kept(again) == listed, "picked up where the reader left off"


def test_kept_settings_become_confirmation_runs(studied):
    page = studied
    set_number(page, "T", 160)
    keep(page, "Cooler")
    page.wait_for_selector("#what-if-kept:not([hidden])")
    before = len(run_sheet(page))
    settings = page.evaluate(
        "() => Object.fromEntries([...document.querySelectorAll('#what-if-settings [data-factor]')]"
        ".map((row) => [row.dataset.factor, row.querySelector('input[name=number]').value]))"
    )
    page.click('[aria-label="Add 3 confirmation runs at the settings kept as Cooler"]')
    settled(page)
    sheet = run_sheet(page)
    assert len(sheet) == before + 3
    reveal(page, "#design-table")
    columns = page.eval_on_selector_all("#design-table thead th", "ths => ths.map((th) => th.textContent)")
    at = {name: next(i for i, c in enumerate(columns) if c.startswith(name)) for name in settings}
    confirming = next(i for i, c in enumerate(columns) if c.startswith("Confirmation run"))
    for run in sheet[before:]:
        assert run[confirming] == "yes"
        assert {name: num(float(run[i])) for name, i in at.items()} == {k: num(float(v)) for k, v in settings.items()}
    in_learn(page)
    page.wait_for_selector("#what-if-kept:not([hidden])")
    assert kept(page)[0]["buttons"] == ["Delete the settings kept as Cooler"], "runs there already"


def test_kept_settings_past_a_limit_say_so_instead_of_predicting(fresh_page):
    page = fresh_page
    set_factors(page, FACTORS)
    candidates_done(page)
    type_limit(page, "T + 25 * P <= 215")
    generate(page, design_type="d_optimal")
    practise(page)
    in_learn(page)
    set_number(page, "T", 200)
    set_number(page, "P", 3)
    keep(page, "Too far")
    page.wait_for_function("document.querySelector('#what-if-kept-list p.hint') !== null")
    [too_far] = kept(page)
    limit = last_reply(page, "api_outside_limits")["result"]["broken"][0]["expression"]
    assert too_far["hint"] == f"These settings break your limit 1, {limit}, so the models do not predict there."
    assert too_far["rows"] == []
    assert too_far["buttons"] == ["Delete the settings kept as Too far"], "no confirmation runs where none can be made"


def test_kept_settings_outside_the_ranges_of_a_study_edited_since_say_so(open_context, studied):
    page = studied
    keep(page, "Edited")
    [study] = saved_studies(page, lambda saved: len(saved) == 1 and saved[0].get("what_if"))
    link = page.evaluate(
        """async (study) => {
            const { packCampaign, shareable } = await import('./campaign.js');
            study.what_if[0].settings.T = 260;
            return `${location.origin}${location.pathname}#c=${await packCampaign(shareable(study, true))}`;
        }""",
        study,
    )
    elsewhere = open_context().new_page()
    elsewhere.goto(link)
    elsewhere.wait_for_selector("#campaign-banner", state="visible")
    elsewhere.wait_for_function("window.__pi.count('api_catalogue') === 1 && window.__pi.idle()")
    settled(elsewhere)
    in_learn(elsewhere)
    elsewhere.wait_for_selector("#what-if-kept:not([hidden])")
    [edited] = kept(elsewhere)
    assert edited["hint"] == "T = 260 degC is outside its range, 150 to 200 degC, so the models do not predict there."
    assert edited["rows"] == []
