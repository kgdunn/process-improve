"""A limit that holds at one level of a categorical factor, in the browser (#813).

A tester's case: raw material A can be dosed up to 1.5 %, raw material B up to 2 %.
"""

from __future__ import annotations

from typing import Any

from tests.web.e2e._page import (
    candidates_done,
    choose,
    generate,
    limits_done,
    practise,
    reveal,
    run_sheet,
    set_factors,
    settled,
)

FACTORS = [
    {"name": "RM", "low": "A", "high": "B", "units": ""},
    {"name": "conc", "low": 0, "high": 2, "units": "%"},
    {"name": "temp", "low": 20, "high": 60, "units": "degC"},
]
UP_TO = "not (RM == 'A' and conc > 1.5)"


def add_for_one_level(page: Any, level: str, side: str, value: str) -> None:  # noqa: ANN401 - a Playwright Page
    """Build a limit for one level from the five choices, as a reader does."""
    reveal(page, "#limits")
    if page.locator("#limits").get_attribute("open") is None:
        page.click("#limits > summary")
    page.click("#add-level-limit")
    page.select_option("#level-limit-level", level)
    page.select_option("#level-limit-numeric", "conc")
    page.select_option("#level-limit-side", side)
    page.fill("#level-limit-value", value)
    page.click("#level-limit-add")
    limits_done(page)


def test_a_limit_for_one_level_is_built_from_five_choices_and_said_in_words(blank_page):
    page = blank_page
    set_factors(page, FACTORS)
    add_for_one_level(page, "A", "most", "1.5")
    row = page.locator("#limit-list .limit").first
    assert row.locator("[name=expression]").input_value() == UP_TO
    assert (
        row.locator(".limit-effect")
        .text_content()
        .startswith("When RM is A: conc <= 1.5. Rules out 25% of your ranges")
    )
    assert page.is_hidden("#level-limit"), "the choices close once the limit is in the list"
    assert not row.locator(".limit__bounds").is_visible(), "no bound box: it would write the limit for every level"
    assert page.locator("#limits-at option").all_text_contents() == ["RM = A", "RM = B"]
    assert page.text_content("#limits-summary").startswith("At RM = A: The limits leave 75% of your ranges to run.")


def test_the_panel_is_shown_for_the_level_the_reader_chooses(blank_page):
    page = blank_page
    set_factors(page, FACTORS)
    add_for_one_level(page, "A", "most", "1.5")
    page.select_option("#limits-at", label="RM = B")
    page.wait_for_function(
        "() => document.querySelector('#limit-list .limit-effect').textContent.includes('nothing at the levels shown')"
    )
    said = page.locator("#limit-list .limit-effect").first.text_content()
    assert said == "When RM is A: conc <= 1.5. It rules out nothing at the levels shown."


def test_the_button_is_offered_only_with_a_factor_that_has_levels(blank_page):
    page = blank_page
    reveal(page, "#limits")
    assert page.is_hidden("#add-level-limit"), "the worked example's factors all have ranges"


def test_the_run_sheet_keeps_each_level_inside_its_own_limit(blank_page):
    page = blank_page
    set_factors(page, FACTORS)
    add_for_one_level(page, "A", "most", "1.5")
    candidates_done(page)
    choose(page, design_type="d_optimal")
    generate(page)
    reveal(page, "#design-table")
    sheet = run_sheet(page)
    at = {level: [float(run[3]) for run in sheet if run[2] == level] for level in "AB"}
    assert at["A"], "raw material A is run"
    assert at["B"], "and so is B"
    assert max(at["A"]) <= 1.5 + 1e-9
    assert max(at["B"]) > 1.5, "B is run over its whole range"


def what_if_at(page: Any, level: str, conc: float) -> list[str]:  # noqa: ANN401 - a Playwright Page
    """Set the raw material and its concentration in What if, and return the warnings the panel then gives."""
    before = page.evaluate("window.__pi.count('api_outside_limits')")
    page.select_option('#what-if-settings [data-factor="RM"] select[name="level"]', level)
    box = page.locator('#what-if-settings [data-factor="conc"] input[name="number"]')
    box.fill(str(conc))
    box.press("Tab")
    page.wait_for_function(f"window.__pi.count('api_outside_limits') > {before}")
    settled(page)
    return page.locator("#what-if-out ul.warnings li").all_text_contents()


def test_what_if_says_a_limit_is_broken_only_at_the_level_it_is_for(blank_page):
    """The sliders stay free; the panel names the limit when the settings are past it, at that level alone."""
    page = blank_page
    set_factors(page, FACTORS)
    add_for_one_level(page, "A", "most", "1.5")
    candidates_done(page)
    choose(page, design_type="d_optimal")
    generate(page)
    practise(page)
    reveal(page, "#what-if")
    page.wait_for_selector("#what-if:not([hidden])")
    broken = "These settings break your limit 1: when RM is A, conc <= 1.5."
    assert broken in what_if_at(page, "A", 1.8), "raw material A at 1.8 % is past its limit"
    assert broken not in what_if_at(page, "B", 1.8), "raw material B at 1.8 % is not"
    assert broken not in what_if_at(page, "A", 1.2)
