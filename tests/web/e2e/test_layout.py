"""The page on a phone and on a desktop: it reads in both themes and scrolls only downwards.

The comparison table is wider than a phone: it scrolls sideways inside its own box, and says
so, rather than pushing the page sideways.
"""

from __future__ import annotations

import pytest

SIDEWAYS = "#comparison p.hint >> text=Scroll the table sideways"


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
             .filter((e) => !e.parentElement.closest('.scroll, pre, .js-plotly-plot'))
             .map((e) => e.tagName.toLowerCase() + (e.id ? `#${e.id}` : ''))"""
    )


@pytest.mark.parametrize(
    "context_options", [{"viewport": {"width": 390, "height": 844}, "color_scheme": "dark", "is_mobile": True}]
)
def test_a_phone_in_dark_mode_scrolls_only_downwards(page):
    page.add_style_tag(content=WIDE_FONT)
    assert overflowing(page) == []
    assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
    assert max(background(page)) < 64, "a dark background"
    page.wait_for_selector(SIDEWAYS, state="visible")


@pytest.mark.parametrize("context_options", [{"viewport": {"width": 1280, "height": 900}, "color_scheme": "light"}])
def test_a_desktop_in_light_mode_shows_every_design_of_the_comparison(page):
    assert min(background(page)) > 192, "a light background"
    assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
    assert page.is_hidden(SIDEWAYS)
