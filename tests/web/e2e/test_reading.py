"""How much each screen asks a reader to read, held to a budget; and the definitions a "?" keeps to hand.

A screen says what only it can say: a warning, a fact about this study, a result. Its headings,
labels, tables and badges say the rest, and what a term means is a "?" away. The words of running
text on each screen (sentences, hints, captions and notes; not headings, labels, buttons, table
cells or charts) are counted as a reader sees them, in a desktop window and on a phone, and held to
a budget. A sentence added to a screen has to fit in what the screen has, or its budget is raised
in the same change, where a reviewer sees it.
"""

from __future__ import annotations

import pytest

from tests.web.e2e._page import generate, practise, reveal

#: The words of running text a reader sees inside ``root`` (the page when null): text in a paragraph, a
#: hint, a caption, a list item or a banner, shown, and not inside a heading, label, button, table cell,
#: equation, chart or code. The study bar's title and status line are the page's own, not text to read.
PROSE = r"""
(root) => {
  const prose = ".hint, p, figcaption, li, .banner";
  const not = "script, style, svg, .js-plotly-plot, pre, select, td, th, .equation, h1, h2, h3, h4, legend, " +
    "summary, button, nav";
  const walker = document.createTreeWalker(root ? document.querySelector(root) : document.body, NodeFilter.SHOW_TEXT);
  let words = 0;
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    const el = node.parentElement;
    if (!el.closest(prose) || el.closest(not)) continue;
    if (!el.checkVisibility({ checkOpacity: true, checkVisibilityCSS: true })) continue;
    words += node.textContent.split(/\s+/).filter((w) => /[\p{L}\p{N}]/u.test(w)).length;
  }
  return words;
}
"""

#: The most words of running text each screen may show, in a desktop window or on a phone: about a tenth
#: more than it shows now. On a desktop the brief stays beside Run and Learn, so a screen there reads a
#: little more; on a phone the comparison says that its table scrolls sideways.
BUDGET = {
    "a first visit": 85,
    "the designs for a brief": 140,
    "a new run sheet": 125,
    "practising": 150,
    "the worked example's analysis": 210,
    "the comparison in detail": 175,
}

WINDOWS = pytest.mark.parametrize(
    "context_options",
    [{"viewport": {"width": 1280, "height": 900}}, {"viewport": {"width": 390, "height": 844}, "is_mobile": True}],
    ids=["desktop", "phone"],
)


def within_budget(page, screen: str, root: str | None = None) -> None:
    words = page.evaluate(PROSE, root)
    assert words <= BUDGET[screen], (
        f"{screen} shows {words} words of running text, over its budget of {BUDGET[screen]}: say less, or let a "
        "heading, a label or a ? say it; else raise the budget here, in the same change."
    )


@WINDOWS
def test_a_first_visit(fresh_page):
    within_budget(fresh_page, "a first visit")


@WINDOWS
def test_the_designs_for_a_brief(blank_page):
    within_budget(blank_page, "the designs for a brief")


@WINDOWS
def test_a_new_run_sheet(blank_page):
    generate(blank_page)
    within_budget(blank_page, "a new run sheet")


@WINDOWS
def test_practising(blank_page):
    page = blank_page
    generate(page)
    practise(page)
    reveal(page, "#run-source")
    within_budget(page, "practising")


@WINDOWS
def test_the_worked_example_s_analysis(page):
    within_budget(page, "the worked example's analysis")


@WINDOWS
def test_the_comparison_in_detail(page):
    reveal(page, "#compare")
    page.click("#compare")
    page.wait_for_function("document.querySelectorAll('#compare-charts .js-plotly-plot').length > 1")
    within_budget(page, "the comparison in detail", "#compare-dialog")


# --------------------------------------------------------------------------- definitions on request
@pytest.mark.parametrize("context_options", [{"viewport": {"width": 1280, "height": 600}}])
def test_a_definition_opens_under_its_question_mark_and_closes_on_escape_or_a_scroll(fresh_page):
    page = fresh_page
    page.click("#response-known summary")
    button = page.locator('[popovertarget="tip-noise"]')
    note = page.locator("#tip-noise")
    assert note.is_hidden(), "a definition waits to be asked for"
    button.click()
    assert note.is_visible()
    at, shown = button.bounding_box(), note.bounding_box()
    below, above = shown["y"] >= at["y"] + at["height"], shown["y"] + shown["height"] <= at["y"]
    assert below or above, "beside its ?, never over it: above it, near the foot of the window"
    assert shown["x"] >= 0
    assert shown["x"] + shown["width"] <= 1280
    assert shown["y"] >= 0
    assert shown["y"] + shown["height"] <= 600, "inside the window"
    assert button.get_attribute("data-open") == "", "the ? shows that its definition is open"
    page.keyboard.press("Escape")
    assert note.is_hidden()
    assert button.get_attribute("data-open") is None
    button.click()
    page.mouse.wheel(0, 300)
    note.wait_for(state="hidden")


@pytest.mark.parametrize("context_options", [{"viewport": {"width": 390, "height": 844}, "is_mobile": True}])
def test_a_definition_stays_inside_a_phone_s_screen(blank_page):
    page = blank_page
    tip = page.locator("#candidates thead th[colspan] button.tip")
    tip.scroll_into_view_if_needed()
    tip.click()
    note = page.locator(".tip-text:popover-open")
    shown = note.bounding_box()
    assert shown["x"] >= 0
    assert shown["x"] + shown["width"] <= 390
    assert note.text_content().startswith("The chance that the analysis finds a change of two noise standard")


def test_the_table_of_designs_defines_its_spare_runs_with_the_model_s_coefficients(blank_page):
    page = blank_page
    page.click("#candidates thead th[rowspan] button.tip")
    assert page.text_content(".tip-text:popover-open").startswith("The runs left once the model's 7 coefficients")
