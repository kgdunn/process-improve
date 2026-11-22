"""A stage named in a sentence is a way to it, and Run says at its top when every result is in."""

from __future__ import annotations

from tests.web.e2e._page import generate, practise, settled, study_menu


def test_the_notes_of_an_empty_run_and_an_empty_learn_lead_to_the_stage_they_name(blank_page):
    page = blank_page
    page.click("#go-run")
    assert page.text_content("#run-empty") == "No run sheet yet: make it in Design."
    page.click("#run-empty button")
    assert page.get_attribute("#workspace", "data-phase") == "plan"
    generate(page)
    page.click("#go-learn")
    assert page.text_content("#analyse-empty") == "No results yet: enter them in Run, or practice with simulated ones."
    page.click("#analyse-empty button")
    assert page.get_attribute("#workspace", "data-phase") == "run"


def test_run_says_at_its_top_when_every_result_is_in_and_leads_to_learn(blank_page):
    page = blank_page
    generate(page)
    assert page.is_hidden("#run-done"), "no result yet"
    practise(page)
    note = page.locator("#run-done")
    assert note.is_visible()
    assert note.text_content().split()[:5] == ["Every", "run", "has", "its", "result."]
    assert note.bounding_box()["y"] < page.locator("#design-table").bounding_box()["y"], "over the run sheet"
    note.locator("button").click()
    assert page.get_attribute("#workspace", "data-phase") == "learn"
    # One result taken away, and Run no longer says every run has one.
    page.click("#go-run")
    page.locator("#design-table input[data-response]").first.fill("")
    settled(page)
    assert page.is_hidden("#run-done")
    study_menu(page, "#new-study")
    assert page.is_hidden("#run-done")
