"""Learn opens with the answer: the settings to run and what the model expects there, before the evidence.

The answer is worded from the best settings the page already finds, so its numbers are the panel's own. With no
goal to reach there are no settings to state, and the answer is what matters most among the factors.
"""

from __future__ import annotations

import pytest

from tests.web.e2e._page import generate, open_question, reveal, settled

RUN = "Run at Temperature = 200 degC, Pressure = 1 bar and Flow = "
EXPECT = "The model predicts the highest Yield there: "


def test_learn_opens_with_where_to_run_and_what_to_expect_in_the_panel_s_own_numbers(page):
    assert page.get_attribute("#workspace", "data-phase") == "learn"
    run, expect = page.text_content("#answer-run"), page.text_content("#answer-expect")
    assert run.startswith(RUN)
    assert expect.startswith(EXPECT)

    # Each number is the panel's, to the digit: every setting of its table, its prediction, and its interval.
    for name, setting in (
        page.locator("#best-out table")
        .first.locator("tbody tr")
        .evaluate_all("(rows) => rows.map((row) => [row.cells[0].textContent, row.cells[1].textContent])")
    ):
        assert f"{name} = {setting}" in run
    predicted = page.text_content("#best-out .lead").rsplit(": ", 1)[1].rstrip(".")
    assert f"{EXPECT}{predicted} (" in expect
    label, interval = page.locator("#best-out .known-table tbody tr").first.evaluate(
        "(row) => [row.cells[0].textContent, row.cells[1].textContent]"
    )
    assert label == "95% confidence interval for the mean"
    assert expect.endswith(f"({label}: {interval}).")


@pytest.mark.parametrize(
    ("context_options", "within"),
    [
        ({"viewport": {"width": 1440, "height": 900}}, 300),
        ({"viewport": {"width": 390, "height": 844}, "is_mobile": True}, 844),
    ],
    ids=["a laptop", "a phone"],
)
def test_the_answer_is_at_the_top_of_learn(page, within):
    """Under 300 px from the top on a laptop; on a phone, whole within the first screen."""
    box = page.locator("#answer").bounding_box()
    edge = box["y"] if within == 300 else box["y"] + box["height"]
    assert edge < within
    assert box["y"] < page.locator("#analysis").bounding_box()["y"], "before the evidence"
    assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")


def test_the_answer_leads_to_the_panel_of_the_best_settings(page):
    page.click("#answer-more")
    assert page.evaluate("document.activeElement.id") == "best"
    top = page.locator("#best").bounding_box()["y"]
    assert 0 <= top < 400, "the best settings, under the bar"


def test_with_no_goal_to_reach_the_answer_is_what_matters_most(page):
    open_question(page)
    page.select_option("#response-goal", "learn")
    settled(page)
    page.wait_for_function("document.querySelector('#answer-expect').hidden")
    said = page.text_content("#answer-run")
    assert " matters most: " in said
    assert said == page.locator("#findings li").first.text_content(), "the first thing the results say"
    assert page.is_hidden("#answer-more"), "there are no best settings to lead to"


def test_no_answer_before_there_are_results_and_one_once_they_are_in(blank_page):
    page = blank_page
    generate(page)
    page.click("#go-learn")
    assert page.is_hidden("#answer")
    reveal(page, "#panel-simulate")
    page.click("#tab-simulate")
    page.click("#simulate")
    settled(page)
    page.click("#go-learn")
    page.wait_for_selector("#answer:not([hidden])")
    assert page.text_content("#answer-run"), "what matters most: the response has no goal yet"


def test_the_report_opens_learn_with_the_same_answer(page):
    run, expect = page.text_content("#answer-run"), page.text_content("#answer-expect")
    with page.expect_popup() as popup:
        page.click("#learn-report-view")
    report = popup.value
    report.wait_for_url("blob:**")
    text = report.text_content("body")
    assert run in text
    assert expect in text
    assert text.index(run) < text.index("Coefficients"), "before the tables"
    assert "How sure this is" not in text, "the way to the panel is a button, which paper has no use for"
