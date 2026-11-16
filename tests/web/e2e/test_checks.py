"""Under Learn's answer, the checks: whether to rely on it, each with its evidence a press away.

The fit, the runs one by one, the repeated runs, and the reach of the runs. Each is a heading with its verdict that
opens to a sentence and its statistic. Python has already judged each (the runs to look at again, a lack of fit,
settings beyond the runs): a check to look at is open, so its caution is read without a press.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from tests.web.e2e._page import open_question, settled

CHECKS = "#checks .learn-check"


def checks(page) -> dict[str, dict]:
    """Return each check as the page shows it: its state, its heading, and whether it is open."""
    return page.locator(CHECKS).evaluate_all(
        """(boxes) => Object.fromEntries(boxes.map((box) => [box.dataset.check, { state: box.dataset.state,
             heading: box.querySelector('summary').textContent, open: box.open }]))"""
    )


def test_the_worked_example_has_four_checks_and_the_one_to_look_at_is_open(page):
    shown = checks(page)
    assert list(shown) == ["fit", "runs", "repeats", "reach"]
    assert shown["fit"]["heading"].startswith("Passed: The model explains ")
    assert shown["reach"] == {"state": "look", "heading": "To look at: Beyond the runs", "open": True}
    assert [name for name, check in shown.items() if check["open"]] == ["reach"], "the rest are a press away"
    caution = "These settings lie beyond the runs, so the prediction leans on the model there: confirm it with a run."
    assert page.locator(f"{CHECKS}[data-check=reach] p").first.text_content() == caution
    assert caution in page.text_content("#best-out"), "the panel's own words"
    box, answer, analysis = (page.locator(part).bounding_box() for part in ("#checks", "#answer", "#analysis"))
    assert answer["y"] + answer["height"] <= box["y"] < analysis["y"], "under the answer, before the tables"


def test_a_check_opens_by_keyboard_to_its_sentence_and_its_statistic(page):
    fit = page.locator(f"{CHECKS}[data-check=fit]")
    assert fit.locator("p").first.is_hidden()
    fit.locator("summary").focus()
    page.keyboard.press("Enter")
    said = fit.locator("p").all_text_contents()
    assert said[0].startswith("The model explains ")
    assert said[0] in page.locator("#findings").text_content(), "the sentence Learn's findings give"
    assert said[1].startswith("R² = ")
    page.keyboard.press("Enter")
    assert fit.locator("p").first.is_hidden(), "and closes"


def test_a_check_leads_to_its_evidence_on_the_page(page):
    runs = page.locator(f"{CHECKS}[data-check=runs]")
    runs.locator("summary").click()
    runs.get_by_role("button", name="See each run as the model sees it").click()
    page.wait_for_selector("#diagnostics.shown-part")
    assert page.locator("#diagnostics").evaluate("(box) => box.open")


def test_a_check_the_reader_opened_stays_open_through_a_new_fit(page):
    page.locator(f"{CHECKS}[data-check=repeats] summary").click()
    page.select_option("#model", "main_effects")
    settled(page)
    page.wait_for_function("document.querySelector('#analysis .equation').textContent.split('+').length < 6")
    assert checks(page)["repeats"]["open"]


def test_with_no_goal_there_is_no_check_of_the_runs_reach(page):
    open_question(page)
    page.select_option("#response-goal", "learn")
    settled(page)
    page.wait_for_function("document.querySelector('#answer-expect').hidden")
    assert list(checks(page)) == ["fit", "runs", "repeats"]


def test_the_report_holds_each_check_open(page):
    with page.expect_popup() as popup:
        page.click("#learn-report-view")
    report = popup.value
    report.wait_for_url("blob:**")
    text = report.text_content("body")
    assert "To look at: Beyond the runs" in text
    assert "Passed: No run stands apart" in text
    assert "No run's result is far from what the model expects of it." in text, "a check's sentence, with no press"


@pytest.mark.parametrize("context_options", [{"viewport": {"width": 390, "height": 844}, "is_mobile": True}])
def test_on_a_phone_the_checks_are_one_under_another(page):
    boxes = [box.bounding_box() for box in page.locator(CHECKS).all()]
    assert all(a["y"] + a["height"] <= b["y"] for a, b in pairwise(boxes))
    assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
