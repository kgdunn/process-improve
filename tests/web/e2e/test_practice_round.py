"""A practice round: before a run is made, one press makes the run sheet, simulates its results and opens Learn.

The reader sees what the plan can and cannot tell. Learn says it is a practice round, and one press clears the simulated
results and leads to the run sheet, to make the runs for real.
"""

from __future__ import annotations

from tests.web.e2e._page import reveal, settled


def results(page) -> list[str]:
    return page.locator("#design-table input[data-response]").evaluate_all("(all) => all.map((one) => one.value)")


def test_a_practice_round_goes_from_the_plan_to_learn_in_one_press_and_says_what_it_is(blank_page):
    page = blank_page
    reveal(page, "#practice-round")
    assert page.text_content("#practice-round") == "Do a practice round"
    assert page.is_hidden("#practice-note")
    page.click("#practice-round")
    page.wait_for_selector("#answer:not([hidden])")
    settled(page)
    assert page.get_attribute("#workspace", "data-phase") == "learn"
    note = page.locator("#practice-note")
    assert note.is_visible()
    assert note.text_content().split(":")[0].strip() == "A practice round"
    assert "No run has been made." in note.text_content()
    assert note.bounding_box()["y"] < page.locator("#answer").bounding_box()["y"], "over what it is a practice round of"
    page.click("#go-run")
    assert all(results(page)), "every run has a simulated result"
    assert page.locator("#design-table input.sim").count() == len(results(page))


def test_the_practice_round_is_cleared_in_one_press_and_the_run_sheet_waits_for_real_results(blank_page):
    page = blank_page
    reveal(page, "#practice-round")
    page.click("#practice-round")
    page.wait_for_selector("#answer:not([hidden])")
    settled(page)
    page.click("#practice-over")
    settled(page)
    assert page.get_attribute("#workspace", "data-phase") == "run"
    assert page.get_attribute("#tab-real", "aria-selected") == "true", "on the panel for real results"
    assert results(page), "the run sheet is still there"
    assert not any(results(page)), "the simulated results are gone"
    page.click("#go-learn")
    assert page.is_hidden("#practice-note")
    assert page.is_hidden("#answer")
    assert page.text_content("#analyse-empty").startswith("No results yet")


def test_a_result_typed_over_ends_the_practice_round_s_note(blank_page):
    page = blank_page
    reveal(page, "#practice-round")
    page.click("#practice-round")
    page.wait_for_selector("#answer:not([hidden])")
    settled(page)
    page.click("#go-run")
    reveal(page, "#design-table")
    page.locator("#design-table input[data-response]").first.fill("61.5")
    settled(page)
    page.click("#go-learn")
    assert page.is_hidden("#practice-note"), "one result is the reader's own: it is no longer only a practice round"
