"""The status line comes and goes in the study's bar, and the page under the bar holds still.

The line says what Python is doing. It takes a row of the bar, which is in the page's flow: without a correction
the page under the bar moves down by that row as the line comes, and up as it goes, five seconds after a success.
A reader's mouse then rests on another word than the one it was put on. The page is scrolled by the same amount
instead, so what is under the bar stays where it is in the window, and a word's definition, open or about to
open, is left alone.

Each test simulates a run sheet's results with Python's answer held, so the line is sure to come: an answer that
takes less than 400 ms brings no line at all.
"""

from __future__ import annotations

import pytest

from tests.web.e2e._page import generate, reveal, settled

WORD = '#aim-card .term[data-term="main-effect"]'
TIP = ".term-tip:popover-open"

#: Where the page is: how far it is scrolled, how tall the bar is, whether the line shows, and where two things under
#: the bar are in the window: a word of the brief, which is pinned beside Run and Learn, and the run sheet, which is not.
WHERE = f"""() => ({{ y: scrollY, bar: document.querySelector('#studybar').offsetHeight,
  line: !document.querySelector('#status').hidden, pinned: getComputedStyle(document.querySelector('#brief')).position,
  word: document.querySelector('{WORD}').getBoundingClientRect().top,
  sheet: document.querySelector('#design-table').getBoundingClientRect().top }})"""


@pytest.fixture
def sheet_page(blank_page):
    """Return a study with its run sheet made and the way to simulate its results shown, the status line gone."""
    generate(blank_page)
    reveal(blank_page, "#panel-simulate")
    blank_page.click("#tab-simulate")
    gone(blank_page)
    return blank_page


def gone(page) -> dict:
    """Wait until the status line has gone, and return where the page is."""
    page.locator("#status").wait_for(state="hidden", timeout=15_000)
    return page.evaluate(WHERE)


def come(page) -> dict:
    """Ask for simulated results and hold Python's answer until the line has come to say so; return where the page is."""
    page.evaluate("window.__pi.hold('api_simulate')")
    page.evaluate("document.querySelector('#simulate').click()")  # where the page is: a pointer would scroll to it
    page.locator("#status").wait_for(state="visible")
    return page.evaluate(WHERE)


def answered(page) -> dict:
    """Let Python answer: the results are analysed and the line says so, for five seconds. Return where the page is."""
    page.evaluate("window.__pi.release('api_simulate')")
    settled(page)
    where = page.evaluate(WHERE)
    assert where["line"], "the line says the analysis is complete"
    return where


def test_the_page_under_the_bar_holds_still_as_the_line_comes_and_as_it_goes(sheet_page):
    page = sheet_page
    page.evaluate("scrollTo(0, 500)")
    before = page.evaluate(WHERE)
    assert before["y"] == 500, "the page is long enough to be there"

    assert before["pinned"] == "sticky", "beside Run the brief is pinned under the bar"

    with_line = come(page)
    row = with_line["bar"] - before["bar"]
    assert row > 20, "the line takes a row of the bar"
    assert with_line["sheet"] == pytest.approx(before["sheet"], abs=1), "the run sheet is where it was"
    assert with_line["word"] == pytest.approx(before["word"], abs=1), "and so is the brief's word"
    assert with_line["y"] == before["y"] + row, "because the page was scrolled by the row"

    shown = answered(page)
    after = gone(page)
    assert after["bar"] == shown["bar"] - row
    assert after["sheet"] == pytest.approx(shown["sheet"], abs=1)
    assert after["word"] == pytest.approx(shown["word"], abs=1)
    assert after["y"] == shown["y"] - row


def test_a_definition_that_is_open_stays_open_as_the_line_goes(sheet_page):
    page = sheet_page
    come(page)
    answered(page)
    page.hover(WORD)
    page.locator(TIP).wait_for()
    gone(page)
    page.wait_for_timeout(400)  # longer than a definition takes to close once its word has left the mouse
    assert page.locator(TIP).count() == 1, "its word did not move, so it was not closed"
    assert page.locator(TIP).text_content().startswith("How much the response changes")


def test_a_mouse_put_on_a_word_just_before_the_line_goes_still_opens_its_definition(sheet_page):
    page = sheet_page
    come(page)
    answered(page)
    page.locator(WORD).scroll_into_view_if_needed()
    page.wait_for_function("document.querySelector('#status').classList.contains('fading')", timeout=15_000)
    page.wait_for_timeout(450)  # the line goes 600 ms after it starts to fade: inside the 300 ms the mouse rests
    page.hover(WORD)
    page.locator(TIP).wait_for(timeout=5_000)
    assert not page.evaluate(WHERE)["line"], "the line went while the mouse rested"


def test_a_scroll_that_moves_a_word_still_closes_its_definition(sheet_page):
    page = sheet_page
    page.hover(WORD)
    page.locator(TIP).wait_for()
    page.evaluate("scrollBy(0, -60); document.querySelector('#brief').scrollTop -= 60")  # whichever holds the word
    page.locator(".term-tip").wait_for(state="hidden")


def test_at_the_top_of_the_page_the_line_goes_with_nothing_to_scroll_back_over(sheet_page):
    """A reader who went back to the top while the line showed: the page moves there, and no error comes of it."""
    page = sheet_page
    come(page)
    answered(page)
    page.evaluate("scrollTo(0, 0)")
    after = gone(page)
    assert after["y"] == 0
