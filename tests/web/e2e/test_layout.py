"""The page on a phone and on a desktop: it reads in both themes, scrolls only downwards, and lays the study out.

On a desktop the brief (the response, the factors, the aim) has a column of its own beside the
phase on show; on a phone the brief opens Design, and Run and Learn take the width. The detailed
comparison is wider than a phone: it scrolls sideways inside its own box, and says so, rather
than pushing the page sideways.
"""

from __future__ import annotations

import pytest

from tests.web._support import FACTORS
from tests.web.e2e._page import candidates_done, corners_done, mark_corners, reveal, set_factors, settled, type_limit

SIDEWAYS = "#comparison p.hint >> text=Scroll the table sideways"
PHASES = ("plan", "run", "learn")


def background(page) -> tuple[int, ...]:
    colour = page.evaluate("getComputedStyle(document.body).backgroundColor")
    return tuple(int(v) for v in colour.removeprefix("rgb(").removesuffix(")").split(", "))


#: A sans-serif font as wide as a reader's default may be: the page must fit a phone in it too.
WIDE_FONT = "* { font-family: 'DejaVu Sans', sans-serif !important; }"


def overflowing(page) -> list[str]:
    """Name the elements wider than the window that do not sit in a box of their own that scrolls."""
    return page.evaluate(
        """() => [...document.querySelectorAll('body *')]
             .filter((e) => e.getBoundingClientRect().right > document.documentElement.clientWidth + 1)
             .filter((e) => getComputedStyle(e).display !== 'none')
             .filter((e) => !e.parentElement.closest('.scroll, pre, .js-plotly-plot, dialog'))
             .map((e) => e.tagName.toLowerCase() + (e.id ? `#${e.id}` : ''))"""
    )


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


@pytest.mark.parametrize("context_options", [{"viewport": {"width": 1280, "height": 900}, "color_scheme": "light"}])
def test_a_desktop_shows_the_brief_beside_every_phase(page):
    assert min(background(page)) > 192, "a light background"
    for phase in PHASES:
        page.click(f"#go-{phase}")
        brief, pane = box(page, "#brief"), box(page, f"[data-pane='{phase}'].pane")
        assert brief["x"] + brief["width"] <= pane["x"], phase
        assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth"), phase


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
