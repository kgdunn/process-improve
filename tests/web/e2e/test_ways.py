"""A stage named in a sentence is a way to it, and Run says at its top when every result is in."""

from __future__ import annotations

import re

from tests.web.e2e._page import generate, practise, reveal, settled, status, study_menu
from tests.web.e2e.test_report import download_report

#: A stage named in a sentence, as the page shows it: the dot before its name, and the dot of its stage in the bar.
DOTS = """(link) => [
  getComputedStyle(link, '::before').backgroundColor,
  getComputedStyle(document.querySelector(`#go-${link.dataset.go} .step`)).backgroundColor,
]"""

#: The mark a sentence names a stage by in the page's code: a reader never sees one.
MARK = re.compile(r"\[(Design|Run|Learn|Next)\]")


def stage_links(page, selector):
    """Return the stages a part of the page names, as its links say them, each checked for its stage's dot."""
    links = page.locator(f"{selector} button.stage-link")
    for i in range(links.count()):
        dot, in_the_bar = links.nth(i).evaluate(DOTS)
        assert dot == in_the_bar, "the dot before the name has the color of its stage"
        assert links.nth(i).get_attribute("data-go") is not None
    return links.all_text_contents()


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


def test_a_stage_named_in_a_sentence_has_its_dot_and_is_a_way_to_it(blank_page):
    """A stage named in a sentence wears the dot it has in the bar, and a press on its name goes there.

    Some sentences named a stage as a link, with no dot, and some as plain words: "once Learn has analyzed them".
    """
    page = blank_page
    reveal(page, "#response-curve")
    page.click("#response-curve summary")
    said = " ".join(page.text_content("#response-curve p").split())
    assert said.startswith("You can paste one curve per run under the Run sheet. In the Learn step it will predict")
    assert stage_links(page, "#response-curve p") == ["Run", "Learn"]
    page.click("#response-curve p button.stage-link >> nth=1")
    assert page.get_attribute("#workspace", "data-phase") == "learn"
    assert stage_links(page, "#analyse-empty") == ["Run"]
    page.click("#go-next")
    assert page.text_content("#next-empty") == "What the results call for is here once Learn has analyzed them."
    assert stage_links(page, "#next-empty") == ["Learn"]
    page.click("#go-run")
    assert stage_links(page, "#run-empty") == ["Design"]
    page.click("#run-empty button.stage-link")
    assert page.get_attribute("#workspace", "data-phase") == "plan"
    assert not MARK.search(page.inner_text("body")), "a stage's mark in the code is never on the page"


def test_a_stage_named_in_a_message_or_in_learn_is_a_link_and_a_report_keeps_its_name(page):
    """The status line and Learn's own sentences name a stage as the page's notes do, and a report prints the name."""
    page.evaluate("window.__pi.fake('api_simulate', {ok: false, error: 'refused'})")
    reveal(page, "#best")
    page.click("#add-confirmation")
    settled(page)
    assert status(page)[0] == "3 confirmation runs added at the end of the run sheet: enter their results in Run."
    assert stage_links(page, "#status .status__text") == ["Run"]
    reveal(page, "#confirm-out")
    waiting = page.locator("#confirm-out .verdict p").first
    assert "Enter their results in Run." in waiting.text_content()
    assert stage_links(page, "#confirm-out .verdict") == ["Run"]
    assert not MARK.search(page.inner_text("body"))
    _, report = download_report(page)
    assert "Enter their results in Run." in report, "a report has no link to press: the stage's name is its text"
    assert not MARK.search(report)
    assert "stage-link" not in report
