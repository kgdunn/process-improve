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

from tests.web.e2e._page import candidates_done, corners_done, generate, mark_corners, practise, reveal, type_limit

#: The words of running text a reader sees inside ``root`` (the page when null): text in a paragraph, a
#: hint, a caption, a list item or a banner, shown, and not inside a heading, label, button, table cell,
#: equation, chart or code. A dotted word in a sentence is still read, though it is a button. The study
#: bar's title and status line are the page's own, not text to read.
PROSE = r"""
(root) => {
  const prose = ".hint, p, figcaption, li, .banner";
  const not = "script, style, svg, .js-plotly-plot, pre, select, td, th, .equation, h1, h2, h3, h4, legend, " +
    "summary, button:not(.term), nav";
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
#: little more; on a phone the comparison says that its table scrolls sideways. The footer's seven words
#: on Feedback, the one report that reaches the server, are on every screen.
BUDGET = {
    "a first visit": 92,  # with the footer's words on Feedback
    "the designs for a brief": 155,  # with the sentence on why Design picked the design it holds
    "a brief with a limit": 240,
    "a limit proposed from a corner": 270,
    "a new run sheet": 136,  # with the footer's words on Feedback
    "practising": 150,
    "the worked example's analysis": 270,  # with the best settings for its goal, which add some 55 words
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
def test_a_brief_with_a_limit(blank_page):
    """The limits panel open on a limit: what it rules out, the picture of what is left, and the corners to mark."""
    type_limit(blank_page, "T + 25 * P <= 250")
    candidates_done(blank_page)
    within_budget(blank_page, "a brief with a limit")


@WINDOWS
def test_a_limit_proposed_from_a_corner(blank_page):
    type_limit(blank_page, "T + 25 * P <= 250")
    corners_done(blank_page)
    mark_corners(blank_page, 3)
    within_budget(blank_page, "a limit proposed from a corner")


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
PHONE = {"viewport": {"width": 390, "height": 844}, "is_mobile": True, "has_touch": True}
TIP = ".term-tip:popover-open"
HINT = "Press and hold a dotted word to see what it means."


def press_and_hold(page, selector: str, ms: int = 700) -> None:
    """Rest a finger on the middle of an element, as a reader on a phone does, then lift it."""
    word = page.locator(selector)
    word.scroll_into_view_if_needed()
    box = word.bounding_box()
    point = {"x": box["x"] + box["width"] / 2, "y": box["y"] + box["height"] / 2}
    touch = page.context.new_cdp_session(page)
    touch.send("Input.dispatchTouchEvent", {"type": "touchStart", "touchPoints": [point]})
    page.wait_for_timeout(ms)
    touch.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})
    touch.detach()


def inside_the_window(page, box: dict) -> bool:
    size = page.viewport_size
    return (
        box["x"] >= 0
        and box["y"] >= 0
        and box["x"] + box["width"] <= size["width"]
        and box["y"] + box["height"] <= size["height"]
    )


@pytest.mark.parametrize("context_options", [PHONE])
def test_a_press_and_hold_opens_a_word_s_definition_above_it_and_leaves_its_choice_alone(fresh_page):
    page = fresh_page
    word = '#aim-card .term[data-term="interaction"]'
    press_and_hold(page, word)
    tip = page.locator(TIP)
    assert tip.text_content() == "Two factors interact when the effect of one depends on the level of the other."
    at, shown = page.locator(word).bounding_box(), tip.bounding_box()
    assert shown["y"] + shown["height"] <= at["y"], "above the word, where the finger does not cover it"
    assert inside_the_window(page, shown)
    assert page.is_checked('input[name="aim"][value="main_quadratic"]'), "holding a choice's word does not choose it"
    page.touchscreen.tap(195, 10)
    assert page.locator(TIP).count() == 0, "a tap elsewhere closes it"


@pytest.mark.parametrize("context_options", [PHONE])
def test_a_tap_on_a_word_does_what_its_control_does_and_opens_nothing(fresh_page):
    page = fresh_page
    word = page.locator('#aim-card .term[data-term="interaction"]')
    word.scroll_into_view_if_needed()
    box = word.bounding_box()
    page.touchscreen.tap(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    assert page.is_checked('input[name="aim"][value="interactions"]')
    page.wait_for_timeout(600)
    assert page.locator(TIP).count() == 0


@pytest.mark.parametrize("context_options", [PHONE])
def test_a_word_that_stands_alone_opens_with_a_tap_and_stays_inside_the_screen(blank_page):
    page = blank_page
    power = page.locator("#candidates thead th[colspan] .term")
    power.scroll_into_view_if_needed()
    box = power.bounding_box()
    page.touchscreen.tap(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    tip = page.locator(TIP)
    assert tip.text_content().startswith("The chance that the analysis finds a change of two noise standard deviations")
    assert inside_the_window(page, tip.bounding_box())
    page.mouse.wheel(0, 200)
    page.locator(".term-tip").wait_for(state="hidden")


def test_a_mouse_resting_on_a_word_opens_its_definition_which_it_can_move_onto(blank_page):
    page = blank_page
    page.hover("#candidates thead th[rowspan] .term")
    tip = page.locator(TIP)
    assert tip.text_content().startswith("The runs left once the model's 7 coefficients are fitted")
    box = tip.bounding_box()
    page.mouse.move(box["x"] + 10, box["y"] + box["height"] / 2)
    page.wait_for_timeout(400)
    assert tip.is_visible(), "the definition stays while the mouse is on it"
    page.keyboard.press("Escape")
    assert page.locator(TIP).count() == 0


def test_the_keyboard_opens_the_definition_of_the_box_it_reaches_and_a_screen_reader_hears_it(fresh_page):
    page = fresh_page
    page.click("#response-known summary")
    page.focus("#response-typical")
    assert page.locator(TIP).count() == 0, "a click or a script moving the focus opens nothing"
    page.keyboard.press("Tab")
    tip = page.locator(TIP)
    noise = "How much the response varies when one run is repeated at the same settings"
    assert tip.text_content().startswith(noise)
    assert tip.evaluate("(t) => getComputedStyle(t).pointerEvents") == "none", "it never stands in the way of a click"
    described = page.evaluate(
        "() => document.getElementById(document.querySelector('#response-noise').getAttribute('aria-describedby'))"
        ".textContent"
    )
    assert described.startswith(noise)
    page.keyboard.press("Escape")
    assert page.locator(TIP).count() == 0


def test_every_dotted_word_has_a_definition_its_control_describes(page):
    """Each word marked in the worked example, in Design, Run, Learn and the comparison, opens a definition."""
    undescribed = """() => [...document.querySelectorAll('.term')].filter((word) => {
        const control = word.matches('button') ? word : word.closest('button, a') ?? word.closest('label')?.control;
        const ids = (control?.getAttribute('aria-describedby') ?? '').split(' ').filter(Boolean);
        return !ids.some((id) => document.getElementById(id)?.textContent.trim());
      }).map((word) => word.textContent)"""
    for phase in ("plan", "run", "learn"):
        page.click(f"#go-{phase}")
        assert page.evaluate(undescribed) == [], phase
    assert page.locator(".term").count() > 40, "the words the worked example marks"
    page.click("#go-plan")
    page.click("#compare")
    page.wait_for_function("document.querySelectorAll('#compare-charts .js-plotly-plot').length > 1")
    assert page.evaluate(undescribed) == []


@pytest.mark.parametrize("context_options", [PHONE])
def test_a_phone_is_told_once_that_a_dotted_word_opens_its_definition(context, site):
    page = context.new_page()
    page.goto(site)
    page.wait_for_function(f"document.querySelector('#toast').textContent === {HINT!r}")
    page.reload()
    page.wait_for_function("window.__pi.count('api_catalogue') === 1 && window.__pi.idle()")
    page.wait_for_timeout(2_000)
    assert page.text_content("#toast") != HINT, "once, on a first visit"
