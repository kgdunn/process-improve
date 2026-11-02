"""The colour-mode switch in the header.

The switch cycles the system's mode, light and dark, as the book's does: the first press always
changes the page. The choice is kept in this browser, put on the page before it is drawn, and the
charts are drawn again in its colours.
"""

from __future__ import annotations

import pytest

from tests.web.e2e._page import first_paint_done

LIGHT_BG, DARK_BG = "rgb(250, 248, 243)", "rgb(20, 21, 23)"


def background(page) -> str:
    return page.evaluate("getComputedStyle(document.body).backgroundColor")


def mode(page) -> tuple[str, str | None, str]:
    """Return the mode, the theme it forces (None for the system's), and the switch's label."""
    return page.evaluate(
        """() => [document.documentElement.dataset.mode, document.documentElement.dataset.theme ?? null,
                  document.querySelector('#mode').getAttribute('aria-label')]"""
    )


def shown_icon(page) -> list[str]:
    return page.eval_on_selector_all(
        "#mode svg", "icons => icons.filter((i) => getComputedStyle(i).display !== 'none').map((i) => i.dataset.for)"
    )


def chart_text_colour(page) -> str:
    """Return the colour of the first chart's axis text: the page's --muted when it was drawn."""
    return page.evaluate("getComputedStyle(document.querySelector('#charts .js-plotly-plot .xtick text')).fill")


@pytest.mark.parametrize("context_options", [{"viewport": {"width": 1280, "height": 900}, "color_scheme": "light"}])
def test_the_switch_cycles_the_system_s_mode_dark_and_light_as_the_book_s_does(page):
    assert mode(page) == ["auto", None, "Colour mode: the system's"]
    assert shown_icon(page) == ["auto"]
    assert page.get_attribute("#mode", "title") == "light/dark"
    assert background(page) == LIGHT_BG
    light_axes = chart_text_colour(page)

    page.click("#mode")
    assert mode(page) == ["dark", "dark", "Colour mode: dark"], "the first press changes the page"
    assert shown_icon(page) == ["dark"]
    assert background(page) == DARK_BG
    page.wait_for_function(
        f"getComputedStyle(document.querySelector('#charts .js-plotly-plot .xtick text')).fill !== '{light_axes}'"
    )

    page.click("#mode")
    assert mode(page) == ["light", "light", "Colour mode: light"]
    assert background(page) == LIGHT_BG
    page.click("#mode")
    assert mode(page) == ["auto", None, "Colour mode: the system's"]


@pytest.mark.parametrize("context_options", [{"viewport": {"width": 1280, "height": 900}, "color_scheme": "light"}])
def test_the_mode_chosen_is_kept_and_drawn_from_the_first_paint(context, page, site):
    page.click("#mode")
    again = context.new_page()
    painted = []
    again.on("domcontentloaded", lambda p: painted.append(p.evaluate("document.documentElement.dataset.theme")))
    again.goto(f"{site}?example=worked")
    first_paint_done(again)
    assert painted == ["dark"], "set by the head of the page, before app.js runs"
    assert mode(again)[:2] == ["dark", "dark"]
    assert background(again) == DARK_BG
    assert again.evaluate("[...document.querySelectorAll('meta[name=theme-color]')].map((m) => m.media)") == [
        "not all",
        "all",
    ], "the browser's bar follows the mode, not the system"


@pytest.mark.parametrize("context_options", [{"viewport": {"width": 1280, "height": 900}, "color_scheme": "dark"}])
def test_on_a_dark_system_the_first_press_shows_light(page):
    assert background(page) == DARK_BG
    page.click("#mode")
    assert mode(page)[:2] == ["light", "light"]
    assert background(page) == LIGHT_BG
