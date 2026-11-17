"""The page on a phone and on a desktop: it reads in both themes, scrolls only downwards, and lays the study out.

On a desktop Design has the brief (the response, the factors, the aim) in a column of its own
beside the designs; on a phone the brief opens Design. Over Run and Learn the question is one
line that opens to the same form, so both take the width. The detailed
comparison is wider than a phone: it scrolls sideways inside its own box, and says so, rather
than pushing the page sideways.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from tests.web._support import FACTORS
from tests.web.e2e._page import (
    WIDE_FONT,
    candidates_done,
    corners_done,
    mark_corners,
    overflowing,
    reveal,
    set_factors,
    settled,
    type_limit,
)

SIDEWAYS = "#comparison p.hint >> text=Scroll the table sideways"
PHASES = ("plan", "run", "learn")


def background(page) -> tuple[int, ...]:
    colour = page.evaluate("getComputedStyle(document.body).backgroundColor")
    return tuple(int(v) for v in colour.removeprefix("rgb(").removesuffix(")").split(", "))


def box(page, selector: str) -> dict:
    return page.locator(selector).bounding_box()


@pytest.mark.parametrize(
    "context_options", [{"viewport": {"width": 390, "height": 844}, "color_scheme": "dark", "is_mobile": True}]
)
def test_a_phone_in_dark_mode_scrolls_only_downwards_in_every_phase(page):
    page.add_style_tag(content=WIDE_FONT)
    assert max(background(page)) < 64, "a dark background"
    for phase in PHASES:
        page.click(f"#go-{phase}")
        assert overflowing(page) == [], phase
        assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth"), phase
    reveal(page, "#compare")
    page.click("#compare")
    page.wait_for_selector(SIDEWAYS, state="visible")


@pytest.mark.parametrize("context_options", [{"viewport": {"width": 390, "height": 844}, "is_mobile": True}])
def test_on_a_phone_the_brief_opens_plan_and_the_other_phases_take_the_width(page):
    page.click("#go-plan")
    assert page.is_visible("#response-card")
    assert box(page, "#response-card")["y"] < box(page, "#step-plan")["y"], "the brief first, then the designs"
    page.click("#go-run")
    assert page.is_hidden("#response-card")
    assert box(page, "#step-run")["width"] > 340


FITS_ACROSS = "document.documentElement.scrollWidth <= document.documentElement.clientWidth"
LINE = "#question > summary"
CARDS = ["#response-card", "#factors-card", "#aim-card"]


@pytest.mark.parametrize("context_options", [{"viewport": {"width": 1280, "height": 900}, "color_scheme": "light"}])
def test_a_desktop_shows_the_brief_beside_design_and_the_question_as_one_line_over_run_and_learn(page):
    assert min(background(page)) > 192, "a light background"
    page.click("#go-plan")
    brief, pane = box(page, "#brief"), box(page, "[data-pane='plan'].pane")
    assert brief["x"] + brief["width"] <= pane["x"], "on Design the question is the form, beside the designs"
    assert page.is_hidden(LINE)
    for phase in ("run", "learn"):
        page.click(f"#go-{phase}")
        assert page.is_hidden("#brief"), phase
        line, pane, workspace = box(page, LINE), box(page, f"[data-pane='{phase}'].pane"), box(page, "#workspace")
        assert line["height"] <= 48, f"one line over {phase}"
        assert line["y"] + line["height"] <= pane["y"], phase
        assert pane["width"] == workspace["width"], f"{phase} has the page's width"
        assert page.evaluate(FITS_ACROSS), phase


@pytest.mark.parametrize("context_options", [{"viewport": {"width": 1280, "height": 900}}])
def test_the_question_s_line_says_the_study_and_opens_to_the_form(page):
    page.click("#go-learn")
    assert page.text_content("#question-says") == (
        "Yield (%), maximise · Temperature 150 to 200 degC, Pressure 1 to 3 bar and Flow 10 to 20 L/min"
        " · to learn where each factor's effect curves"
    )
    assert page.is_visible(f"{LINE} [data-when='closed']")
    assert page.is_hidden(f"{LINE} [data-when='open']")

    page.focus(LINE)
    page.keyboard.press("Enter")  # the line is a summary: the keyboard opens it as a press does
    cards = [box(page, card) for card in CARDS]
    assert [card["y"] for card in cards] == [cards[0]["y"]] * 3, "the three cards share a row"
    assert all(a["x"] + a["width"] <= b["x"] for a, b in pairwise(cards)), "side by side"
    assert max(card["y"] + card["height"] for card in cards) <= box(page, "#step-analyse")["y"], "over Learn"
    assert page.is_visible(f"{LINE} [data-when='open']"), "the line now offers to close it"
    assert page.evaluate(FITS_ACROSS)

    # The form is the one Design has: a change made here is the study's, and the line and the bar follow it.
    page.select_option("#response-goal", "minimize")
    assert page.text_content("#study-title") == "Yield (%), minimise"
    assert page.text_content("#question-says").startswith("Yield (%), minimise · Temperature 150 to 200 degC")

    page.click(LINE)
    assert page.is_hidden("#brief"), "closed again"


@pytest.mark.parametrize("context_options", [{"viewport": {"width": 1280, "height": 900}}])
def test_the_question_is_closed_on_the_way_in_from_design_and_kept_between_run_and_learn(page):
    page.click("#go-run")
    assert page.is_hidden("#brief")
    page.click(LINE)
    page.click("#go-learn")
    assert page.is_visible("#brief"), "left open as the reader has it, from Run to Learn"
    page.click("#go-plan")
    assert page.is_visible("#brief")
    assert page.is_hidden(LINE)
    page.click("#go-run")
    assert page.is_hidden("#brief"), "one line again on the way in from Design"


@pytest.mark.parametrize("context_options", [{"viewport": {"width": 390, "height": 844}, "is_mobile": True}])
def test_on_a_phone_the_question_is_one_line_cut_short_and_opens_to_cards_one_under_another(page):
    page.click("#go-run")
    assert box(page, LINE)["height"] <= 48
    assert page.evaluate("((e) => e.scrollWidth > e.clientWidth)(document.querySelector('#question-says'))"), (
        "cut short"
    )
    assert overflowing(page) == [], "and not pushed past the screen"
    assert page.evaluate(FITS_ACROSS)
    page.click(f"{LINE} .question__change")
    cards = [box(page, card) for card in CARDS]
    assert all(a["y"] + a["height"] <= b["y"] for a, b in pairwise(cards)), "one under another"
    assert page.evaluate(FITS_ACROSS)


@pytest.mark.parametrize("context_options", [{"viewport": {"width": 1280, "height": 900}, "color_scheme": "light"}])
def test_a_desktop_shows_every_design_of_the_detailed_comparison(page):
    reveal(page, "#compare")
    page.click("#compare")
    page.wait_for_selector("#comparison table", state="visible")
    assert page.is_hidden(SIDEWAYS)
    page.click("#compare-close")
    assert page.is_hidden("#compare-dialog")


@pytest.mark.parametrize("context_options", [{"viewport": {"width": 1280, "height": 900}}])
def test_the_study_bar_stays_in_view_and_names_the_study(page):
    page.click("#go-learn")
    page.mouse.wheel(0, 2000)
    page.wait_for_function("window.scrollY > 500")
    assert box(page, "#studybar")["y"] == 0, "kept at the top as the page scrolls"
    assert page.text_content("#study-title") == "Yield (%), maximise"
    assert page.text_content("#study-about") == "· Box-Behnken, 15 runs"
    assert page.text_content("#badge-run") == "15 of 15"


#: What the bar holds, each as its box when it is drawn: the mark, the name, the study's name, the stages, feedback
#: and the colour mode.
BAR = """() => Object.fromEntries(Object.entries({ mark: '.brand__home .mark', name: '.brand__home h1',
    study: '#study-title', stages: '.phases', feedback: '#feedback', mode: '#mode' }).map(([part, at]) => {
  const box = document.querySelector(at).getBoundingClientRect();
  return [part, { top: box.top, bottom: box.bottom, left: box.left, right: box.right, drawn: box.width > 2 }]; }))"""


@pytest.mark.parametrize("context_options", [{"viewport": {"width": 1440, "height": 900}}])
def test_on_a_laptop_the_bar_is_one_row_no_taller_than_56_px(fresh_page):
    """The mark, the name, the study, the stages, feedback and the colour mode are one row; the work starts under it."""
    bar = box(fresh_page, "#studybar")
    assert bar["y"] == 0, "the bar opens the page"
    assert bar["height"] <= 56
    parts = fresh_page.evaluate(BAR)
    assert all(part["drawn"] for part in parts.values()), parts
    order = sorted(parts, key=lambda part: parts[part]["left"])
    assert order == ["mark", "name", "study", "stages", "feedback", "mode"]
    assert all(part["top"] >= 0 and part["bottom"] <= bar["height"] for part in parts.values()), "each within the bar"
    # What leads away from the study is beside the stages on a screen this wide.
    links = fresh_page.locator(".studybar__link")
    assert links.all_text_contents() == ["Design catalog", "How it works"]
    assert [link.get_attribute("href") for link in links.all()] == ["catalog/", "about.html"]
    assert all(link.is_visible() for link in links.all())


@pytest.mark.parametrize(
    "context_options",
    [{"viewport": {"width": width, "height": 1000}} for width in (744, 820, 1024, 1099)],
    ids=["744", "820", "1024", "1099"],
)
def test_on_a_tablet_the_bar_is_two_rows_and_the_study_s_name_is_read_whole(page):
    """A study with results: four stages and their badges. In one row they left the study's name no room at all."""
    parts = page.evaluate(BAR)
    assert all(part["drawn"] for part in parts.values()), parts
    assert parts["study"]["bottom"] <= parts["stages"]["top"], "the study's name is over the stages"
    title = page.locator("#study-title")
    assert title.evaluate("(one) => one.scrollWidth <= one.clientWidth"), "none of the name is cut off"
    caret = page.locator(".study__caret").bounding_box()
    assert caret["y"] + caret["height"] <= parts["stages"]["top"], "the menu's mark is not over a stage"
    assert box(page, "#studybar")["height"] <= 100
    assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
    # In a wider font too: the name keeps its width, and the feedback button gives up its words first.
    page.add_style_tag(content="body, button { font-family: 'DejaVu Sans', sans-serif !important; font-size: 18px; }")
    assert title.evaluate("(one) => one.scrollWidth <= one.clientWidth"), "nor in a wider font"
    assert page.evaluate(BAR)["study"]["bottom"] <= page.evaluate(BAR)["stages"]["top"], "and still over the stages"


@pytest.mark.parametrize("context_options", [{"viewport": {"width": 390, "height": 844}, "is_mobile": True}])
def test_on_a_phone_the_bar_is_two_rows_the_study_s_name_over_the_stages(fresh_page):
    """The name "Factorial" is read out and not drawn there, so the study's name has the first row."""
    parts = fresh_page.evaluate(BAR)
    assert not parts["name"]["drawn"]
    assert fresh_page.get_attribute(".brand__home", "aria-label") == "Factorial: the home page", "the mark says so"
    assert parts["study"]["bottom"] <= parts["stages"]["top"], "the study's name is over the stages"
    assert parts["mode"]["bottom"] <= parts["stages"]["top"]
    assert box(fresh_page, "#studybar")["height"] <= 100
    assert fresh_page.locator(".studybar__link:visible").count() == 0, "the links away are in the menu and the footer"
    assert fresh_page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")


@pytest.mark.parametrize("context_options", [{"viewport": {"width": 390, "height": 844}, "is_mobile": True}])
def test_on_a_phone_the_limits_their_picture_and_the_corners_fit(fresh_page):
    """The table of corners scrolls in its own box; the panels of the picture and the slice view take the width."""
    page = fresh_page
    page.add_style_tag(content=WIDE_FONT)
    set_factors(page, [*FACTORS, {"name": "Stir", "low": 100, "high": 400, "units": "rpm"}])
    candidates_done(page)
    type_limit(page, "T + 25 * P + 5 * F <= 350")
    corners_done(page)
    mark_corners(page, 1)
    page.get_by_role("button", name="Adjust it factor by factor").click()
    assert overflowing(page) == []
    page.get_by_role("button", name="Look inside T and P").click()
    page.wait_for_timeout(400)
    settled(page)
    assert overflowing(page) == []
    assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
