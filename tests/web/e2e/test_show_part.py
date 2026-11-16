"""A link that asks for a part of the page: ``?show=<id>`` beside the example it opens.

The list of what Factorial does ends a point with "Show me": a link that opens the app on an example of the
feature, at the part of the page that has it. The page shows the step the part is in, opens the folds around it,
brings it under the study's bar and outlines it for a moment. A part that an analysis has still to make is waited
for, unless the reader has begun to do something else. The address then loses the word, so a reload does not jump.
"""

from __future__ import annotations

import pytest

from tests.web.e2e._page import first_paint_done, settled

#: The part a link named, as the page shows it: how far under the study's bar, in which step, and whether marked.
SHOWN = """(id) => { const part = document.getElementById(id);
  const bar = document.querySelector('#studybar').getBoundingClientRect();
  return { under: part.getBoundingClientRect().top - bar.bottom,
    phase: document.querySelector('#workspace').dataset.phase,
    marked: part.classList.contains('shown-part'), focused: document.activeElement === part,
    seen: part.checkVisibility(), address: location.search,
    end: scrollY + innerHeight >= document.documentElement.scrollHeight - 2 }; }"""

#: Whether any part of the page is marked as the one a link asked for.
MARKED = "() => document.querySelectorAll('.shown-part').length"


def opened(context, site: str, query: str):
    """Open the page at an address, and return it once the worked example it asks for is drawn."""
    page = context.new_page()
    page.goto(f"{site}?{query}")
    first_paint_done(page)
    return page


@pytest.mark.parametrize(
    ("part", "step"),
    [
        ("best", "learn"),
        ("what-if", "learn"),
        ("next", "next"),
        ("charts", "learn"),
        ("step-run", "run"),
        ("step-plan", "plan"),
    ],
)
def test_a_link_opens_the_worked_example_at_the_part_it_names(context, site, part, step):
    page = opened(context, site, f"example=worked&show={part}")
    page.wait_for_selector(f"#{part}.shown-part")
    shown = page.evaluate(SHOWN, part)
    assert shown["phase"] == step, "the step the part is in is the one shown"
    assert shown["seen"]
    page.wait_for_timeout(400)  # the page puts the part under the bar again once the step's charts have their room
    shown = page.evaluate(SHOWN, part)
    assert 0 <= shown["under"] < 40 or shown["end"], "the part is just under the study's bar, or the page is at its end"
    assert shown["focused"], "a screen reader goes on from the part"
    assert shown["address"] == "?example=worked", "the address keeps the example and loses the part"


def test_the_word_for_where_a_link_was_posted_leaves_the_address_and_changes_nothing_else(context, site):
    """``?ref=linkedin`` is for the site's usage report: a reload, or the address copied on, is not counted again."""
    page = opened(context, site, "example=worked&ref=linkedin&show=best")
    page.wait_for_selector("#best.shown-part")
    assert page.evaluate("location.search") == "?example=worked", "the example stays; the part and the place go"
    alone = context.new_page()
    alone.goto(f"{site}?ref=linkedin")
    alone.wait_for_function("window.__pi.count('api_catalogue') === 1 && window.__pi.idle()")
    assert alone.url == site, "a first visit by such a link opens the page as any other does"


def test_a_part_in_a_fold_has_its_fold_opened_and_its_outline_goes_after_a_moment(context, site):
    page = opened(context, site, "example=worked&show=py-analyse")
    page.wait_for_selector("#py-analyse.shown-part")
    assert page.evaluate("() => document.querySelector('#py-analyse').open"), "the code of the analysis, unfolded"
    assert page.is_visible("#py-analyse pre")
    page.wait_for_function("() => !document.querySelectorAll('.shown-part').length", timeout=10_000)


def test_a_book_example_is_shown_at_its_part_once_it_has_been_analysed(context, site):
    page = context.new_page()
    page.goto(f"{site}?book=side-product&show=charts")
    page.wait_for_selector("#charts.shown-part", timeout=30_000)
    shown = page.evaluate(SHOWN, "charts")
    assert (shown["phase"], shown["seen"]) == ("learn", True)
    assert shown["address"] == "", "a book example's address is cleared, as without the part"


def test_a_part_that_is_not_there_yet_is_waited_for_unless_the_reader_has_begun_something_else(context, site):
    # A new study has no best settings: the part is hidden until an analysis shows it.
    waits = context.new_page()
    waits.goto(f"{site}?show=best")
    waits.wait_for_function("window.__pi.count('api_catalogue') === 1 && window.__pi.idle()")
    settled(waits)
    assert waits.evaluate(MARKED) == 0
    waits.evaluate("() => { document.querySelector('#best').hidden = false; }")
    waits.wait_for_selector("#best.shown-part")
    assert waits.evaluate(SHOWN, "best")["phase"] == "learn"

    typing = context.new_page()
    typing.goto(f"{site}?show=best")
    typing.wait_for_function("window.__pi.count('api_catalogue') === 1 && window.__pi.idle()")
    settled(typing)
    typing.keyboard.type("Yield")  # the reader has begun the study's brief
    typing.evaluate("() => { document.querySelector('#best').hidden = false; }")
    typing.wait_for_timeout(700)  # the page looks for the part every 200 ms
    assert typing.evaluate(MARKED) == 0, "the reader is left where they are"
    assert typing.get_attribute("#workspace", "data-phase") == "plan"


@pytest.mark.parametrize("name", ["no-such-part", "Not%20a%20name", "best%22%3E%3Cb", ""])
def test_a_name_that_is_no_part_of_the_page_is_passed_over(context, site, name):
    page = opened(context, site, f"example=worked&show={name}")
    settled(page)
    assert page.evaluate(MARKED) == 0
    assert page.evaluate("() => location.search") == "?example=worked"
    assert page.get_attribute("#workspace", "data-phase") == "learn", "the worked example opens as it does"
