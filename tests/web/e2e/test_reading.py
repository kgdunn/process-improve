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
#: little more; on a phone the comparison says that its table scrolls sideways. The footer's one sentence,
#: on what Factorial is and that nothing runs on a server, is on every screen; its links are lists, not text.
BUDGET = {
    # The welcome has the first screen: what Factorial does in a sentence, a study's four steps in a line each and
    # three promises, some 90 words. It goes once the reader starts, so no other screen here has them.
    "a first visit": 150,  # with the footer's sentence
    "the designs for a brief": 139,  # with the sentence on why Design picked the design it holds
    "a brief with a limit": 224,
    "a limit proposed from a corner": 254,
    "a new run sheet": 120,  # with the footer's sentence
    "practising": 134,
    # With the best settings for its goal, which add some 55 words, and the what-if panel, some 23: its hint, and its
    # warning that the best settings lie beyond the runs. With "What the results say", some 70: the sentences a
    # reader would else have to work out of the tables below them. The settings that give a target and those that
    # meet a specification add 4 words, the label of the run to match: their headings and their boxes' labels say
    # what to type, so neither has a hint. The what-if panel's hint says what its pictures are, some 13 words more.
    # A prediction as 20 runs has a sentence of 10 words: it says what the drawing shows, to a reader who does not
    # see the drawing too, so it is not cut. What if as one table moved that sentence into the drawing's own label and
    # added the best run so far, some 12 words, the name of the What-if row, 6, and the label of its presets, 5.
    "the worked example's analysis": 440,
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
MORE = f"{TIP} a.term-more"  # under a definition, the way to its word's term in the concept map
HINT = "Press and hold a dotted word to see what it means."
# The concept map's own box is at the top of the window, not a screen or more below the page's title.
AT_THE_MAP = "(({ top }) => top >= 0 && top <= 16)(document.querySelector('.map-box').getBoundingClientRect())"


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


def said(tip) -> str:
    """Return what a pop-up says of its word: its definition, without the link under it."""
    return tip.evaluate("(tip) => tip.firstChild.textContent")


def at_term(the_map, name: str) -> None:
    """Wait until the concept map shows the term ``name``, with the map at the top of the window."""
    the_map.wait_for_function("(name) => document.querySelector('#card h2')?.textContent === name", arg=name)
    the_map.wait_for_function(AT_THE_MAP)


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
    assert said(tip) == "Two factors interact when the effect of one depends on the level of the other."
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


def test_without_a_network_a_definition_leads_nowhere_the_copy_cannot_open(fresh_page, context):
    """Offline, with no concept map in the copy, a definition has no "See it in context" (#1263).

    The link would open "This page needs a network". The definition itself is shown as before.
    """
    page = fresh_page
    context.set_offline(True)
    try:
        press_and_hold(page, '#aim-card .term[data-term="interaction"]')
        assert page.locator(TIP).is_visible()
        assert page.locator(MORE).count() == 0
    finally:
        context.set_offline(False)


@pytest.mark.parametrize("context_options", [PHONE])
def test_a_definition_leads_to_its_word_s_term_in_the_concept_map_in_a_tab_of_its_own(fresh_page, context):
    page = fresh_page
    press_and_hold(page, '#aim-card .term[data-term="interaction"]')
    link = page.locator(MORE)
    assert link.text_content() == "See it in context"
    assert link.get_attribute("aria-label") == "See it in context: two-factor interactions"
    assert page.locator(TIP).get_attribute("role") == "group", "a tooltip holds nothing a reader can go into"
    assert inside_the_window(page, page.locator(TIP).bounding_box())
    with context.expect_page() as opened:
        link.tap()  # at once: not taken for the end of the press that opened the definition
    the_map = opened.value
    at_term(the_map, "Interaction")
    assert the_map.url.endswith("map/#interaction")
    assert the_map.is_visible("#card >> text=Read it in the book"), "and from there, the book"
    assert page.is_checked('input[name="aim"][value="main_quadratic"]'), "the study is as it was"


def test_a_second_word_s_term_opens_in_the_tab_of_the_first_and_a_word_the_map_lacks_has_no_link(blank_page, context):
    page = blank_page
    page.hover('#aim-card .term[data-term="interaction"]')
    with context.expect_page() as opened:
        page.click(MORE)
    the_map = opened.value
    at_term(the_map, "Interaction")
    the_map.evaluate("scrollTo(0, document.body.scrollHeight)")  # the reader read on, down the page

    page.bring_to_front()
    page.mouse.move(5, 5)  # off the definition, which closes: here it lay over the next word
    page.locator(".term-tip").wait_for(state="hidden")
    page.hover('#aim-card .term[data-term="main-effect"]')
    page.wait_for_function("document.querySelector('.term-tip a')?.hash === '#main-effect'")
    page.click(MORE)
    at_term(the_map, "Main effect")
    assert len(context.pages) == 2, "one tab for the map, however many words open it"

    # A design's name in the list of designs leads to the design's term.
    page.mouse.move(5, 5)
    page.locator(".term-tip").wait_for(state="hidden")
    design = page.locator("#candidates tbody label .term[data-concept]").first
    term = design.get_attribute("data-concept")
    design.hover()
    page.wait_for_function(
        "(hash) => document.querySelector('.term-tip a')?.hash === hash", arg=f"#{term.replace('_', '-')}"
    )
    # A word the map has no term for says what it means, and no more.
    page.click("#response-known summary")
    page.hover('.term[data-term="typical"]')
    page.wait_for_function("document.querySelector('.term-tip').textContent.startsWith('Where the response sits now')")
    assert page.locator(MORE).count() == 0
    assert page.locator(TIP).get_attribute("role") == "tooltip"


def test_tab_goes_from_a_box_into_its_definition_s_link_and_on_from_there(fresh_page, context):
    page = fresh_page
    focused = "document.activeElement.id || document.activeElement.className"
    page.click("#response-known summary")
    page.focus("#response-typical")
    page.keyboard.press("Tab")  # to the noise, whose word has a term in the map
    assert page.locator(MORE).count() == 1
    page.keyboard.press("Tab")
    assert page.evaluate(focused) == "term-more", "into the definition, which stays while the keyboard is on its link"
    assert said(page.locator(TIP)).startswith("How much the response varies when one run is repeated")
    page.keyboard.press("Shift+Tab")
    assert page.evaluate(focused) == "response-noise", "and back to the box it came from"
    assert page.locator(MORE).count() == 1, "whose definition is open again"

    page.keyboard.press("Tab")
    with context.expect_page() as opened:
        page.keyboard.press("Enter")
    at_term(opened.value, "Noise standard deviation")
    page.bring_to_front()
    page.keyboard.press("Tab")
    assert page.evaluate(focused) == "response-change", "on from the link: the box after the one it belongs to"
    assert said(page.locator(TIP)).startswith("The smallest difference in the response")

    page.keyboard.press("Tab")
    assert page.evaluate(focused) == "term-more"
    page.keyboard.press("Escape")
    assert page.locator(TIP).count() == 0
    assert page.evaluate(focused) == "response-change", (
        "Escape closes the definition and leaves the keyboard on its box"
    )


def test_a_definition_that_follows_the_keyboard_opens_clear_of_the_box_it_describes(fresh_page):
    """The word is the label over its box: a pop-up under the word alone lay over the box being typed in."""
    page = fresh_page
    page.click("#response-known summary")
    page.focus("#response-typical")
    page.keyboard.press("Tab")
    tip = page.locator(TIP).bounding_box()
    box = page.locator("#response-noise").bounding_box()
    word = page.locator("label:has(#response-noise) .term").bounding_box()
    assert tip["y"] >= box["y"] + box["height"] or tip["y"] + tip["height"] <= word["y"], "under both, or over both"
    assert inside_the_window(page, tip)


def test_the_words_over_the_list_of_designs_lead_to_their_terms_too(blank_page):
    page = blank_page
    for word, term in (("Spare runs", "#degrees-of-freedom"), ("Power to find", "#statistical-power")):
        page.mouse.move(5, 5)
        page.locator(".term-tip").wait_for(state="hidden")
        page.locator("#candidates thead .term", has_text=word).first.hover()
        page.wait_for_function("(hash) => document.querySelector('.term-tip a')?.hash === hash", arg=term)


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
