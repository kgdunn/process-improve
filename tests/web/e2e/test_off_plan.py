"""A run made at settings that differ from the plan: the setting is typed over on the run sheet, and the fit uses it.

A numeric factor's setting on the run sheet is a button that turns into a box. What is typed is the setting the
run was really made at: the model is fitted there, the sheet says which runs left the plan, and one press puts the
planned settings back. The planned setting is kept with the study.
"""

from __future__ import annotations

import pytest

from tests.web.e2e._page import generate, practise, reveal, saved_studies, settled

SETTING = '#design-table button.setting[data-index="0"][data-factor="Temperature"]'
#: The same setting in a study of the factors T, P and F.
OWN = '#design-table button.setting[data-index="0"][data-factor="T"]'


def type_setting(page, value: str, key: str = "Enter") -> None:
    page.click(SETTING)
    box = page.locator("#design-table input.setting")
    assert box.evaluate("(one) => one === document.activeElement"), "the box is where the button was, ready to type"
    box.fill(value)
    box.press(key)
    settled(page)


def test_a_setting_typed_over_is_what_the_model_is_fitted_on_and_the_sheet_says_so(page):
    page.click("#go-run")
    reveal(page, "#design-table")
    planned = page.text_content(SETTING)
    assert page.text_content("#off-plan-says").startswith("A run made at another setting than planned?")
    assert page.is_hidden("#off-plan-back")
    page.click("#go-learn")
    before = page.text_content("#answer-expect")

    page.click("#go-run")
    type_setting(page, "163")
    assert page.text_content(SETTING) == "163"
    assert "off" in page.get_attribute(SETTING, "class").split()
    assert page.get_attribute(SETTING, "title") == f"Planned: {planned}"
    assert f"planned {planned}" in page.get_attribute(SETTING, "aria-label")
    assert page.evaluate("document.activeElement.matches('button.setting')"), "the focus is back on the setting"
    run = page.locator(SETTING).evaluate("(setting) => setting.closest('tr').dataset.run")
    assert page.text_content("#off-plan-says").startswith(f"Run {run} was made at settings that differ from the plan.")
    page.click("#go-learn")
    # The analysis follows a moment after the last change: the model is fitted on the setting really made.
    page.wait_for_function("(before) => document.querySelector('#answer-expect').textContent !== before", arg=before)
    assert page.locator("#status[data-state=error]").count() == 0

    page.click("#go-run")
    page.click("#off-plan-back")
    settled(page)
    assert page.text_content(SETTING) == planned
    assert "off" not in page.get_attribute(SETTING, "class").split()
    assert page.is_hidden("#off-plan-back")
    page.click("#go-learn")
    # Back on the plan, the answer is the plan's.
    page.wait_for_function("(before) => document.querySelector('#answer-expect').textContent === before", arg=before)


@pytest.mark.parametrize(("typed", "key"), [("163", "Escape"), ("", "Enter")], ids=["Escape", "a blank box"])
def test_escape_or_a_blank_box_leaves_the_planned_setting(page, typed, key):
    page.click("#go-run")
    reveal(page, "#design-table")
    planned = page.text_content(SETTING)
    type_setting(page, typed, key)
    assert page.text_content(SETTING) == planned
    assert page.is_hidden("#off-plan-back")


def test_the_results_boxes_are_still_the_sheet_s_only_boxes_until_a_setting_is_pressed(page):
    page.click("#go-run")
    reveal(page, "#design-table")
    rows = page.locator("#design-table tr[data-run]").count()
    assert page.locator("#design-table tr[data-run] input").count() == rows, "one box a run: its result"
    assert page.locator("#design-table button.setting").count() == 3 * rows, "three numeric factors a run"


def test_the_study_keeps_the_setting_made_and_the_planned_one(blank_page):
    page = blank_page
    generate(page)
    practise(page)
    reveal(page, "#design-table")
    planned = page.text_content(OWN)
    page.click(OWN)
    box = page.locator("#design-table input.setting")
    box.fill("163")
    box.press("Enter")
    # The save is put off a moment: wait for the study this browser keeps to hold the planned setting.
    saved_studies(page, lambda kept: len(kept) == 1 and bool(kept[0].get("planned")))
    page.reload()
    settled(page)
    page.click("#go-run")
    reveal(page, "#design-table")
    assert page.text_content(OWN) == "163"
    assert page.get_attribute(OWN, "title") == f"Planned: {planned}"
    assert page.is_visible("#off-plan-back")
