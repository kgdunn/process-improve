"""The status line comes and goes under the study's bar, and neither the bar nor the page moves.

The line says what Python is doing. It hangs under the bar, over room the page leaves for it, and is no part of
the bar's height. So the bar is one height with the line and without it, the page is not scrolled, and what is
under a reader's mouse stays there: a word's definition, open or about to open, is left alone.

Each test simulates a run sheet's results with Python's answer held, so the line is sure to come: an answer that
takes less than 400 ms brings no line at all.
"""

from __future__ import annotations

import pytest

from tests.web.e2e._page import generate, reveal, settled

WORD = '#panel-simulate .term[data-term="noise"]'
TIP = ".term-tip:popover-open"

#: Where the page is: how far it is scrolled, how tall the bar is, whether the line shows, and where two things under
#: the bar are in the window: a word with a definition, and the run sheet.
WHERE = f"""() => ({{ y: scrollY, bar: document.querySelector('#studybar').offsetHeight,
  line: !document.querySelector('#status').hidden,
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
    """Ask for simulated results, hold Python's answer until the line has come to say so; return where the page is."""
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

    # The line comes, and later it goes: each time the page is as it was just before. (Between the two the results
    # are analysed, and the brief says more.)
    with_line = come(page)
    shown = answered(page)
    after = gone(page)
    for was, now in ((before, with_line), (shown, after)):
        assert now["bar"] == before["bar"], "the line is no part of the bar's height"
        assert now["y"] == was["y"], "the page is not scrolled"
        assert now["sheet"] == was["sheet"], "the run sheet is where it was"
        assert now["word"] == was["word"], "and so is a word under it"


def test_a_definition_that_is_open_stays_open_as_the_line_goes(sheet_page):
    page = sheet_page
    come(page)
    answered(page)
    page.hover(WORD)
    page.locator(TIP).wait_for()
    gone(page)
    page.wait_for_timeout(400)  # longer than a definition takes to close once its word has left the mouse
    assert page.locator(TIP).count() == 1, "its word did not move, so it was not closed"
    assert page.locator(TIP).text_content().startswith("How much the response varies")


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
    page.evaluate("scrollBy(0, -60)")
    page.locator(".term-tip").wait_for(state="hidden")


#: At the top of the page: the bar's foot, the line's box, and the top of what the page puts first under the bar.
TOP = """() => { const box = (s) => document.querySelector(s).getBoundingClientRect();
  return { bar: box('#studybar').bottom, line: [box('#status').top, box('#status').bottom], first: box('.ways').top,
           sheet: box('#design-table').top }; }"""


@pytest.mark.parametrize(
    "context_options",
    [{"viewport": {"width": 1440, "height": 900}}, {"viewport": {"width": 390, "height": 844}, "is_mobile": True}],
    ids=["a laptop", "a phone"],
)
def test_at_the_top_of_the_page_the_line_has_room_of_its_own_and_the_page_holds_still(sheet_page):
    """The line comes between the bar and what the page puts first: it lies on neither, and nothing moves for it."""
    page = sheet_page
    page.evaluate("scrollTo(0, 0)")
    before = page.evaluate(TOP)
    come(page)
    with_line = page.evaluate(TOP)
    top, foot = with_line["line"]
    assert with_line["bar"] <= top, "the line is under the bar"
    assert foot <= with_line["first"], "and above what the page puts first"
    answered(page)
    shown = page.evaluate(TOP)
    gone(page)
    after = page.evaluate(TOP)
    for was, now in ((before, with_line), (shown, after)):
        assert (now["bar"], now["first"], now["sheet"]) == (was["bar"], was["first"], was["sheet"])
    assert page.evaluate("scrollY") == 0
