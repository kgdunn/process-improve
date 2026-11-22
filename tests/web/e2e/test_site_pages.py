"""Every page of the site beside the app has the same top line and the same footer (web/site.js).

The color-mode switch was on the app and the concept map only, each page ended in a footer of its own or in none,
and the page of the textbooks' answers was linked from one door on the catalog. Each page now has the switch at the
end of its top line and one footer, with the same links in the same order.
"""

from __future__ import annotations

import pytest

PAGES = [
    "technical-details-how-factorial-works.html",
    "features.html",
    "validation.html",
    "catalog/",
    "catalog/reproduced/",
    "map/",
]
LINKS = [
    "Factorial, the app",
    "What Factorial does",
    "Catalog of experimental designs",
    "Validation & technical accuracy",
    "Reproduced from the books",
    "Experimental design concept map",
    "See behind the scenes: how Factorial works",
    "process-improve, the Python package",
    "Process Improvement using Data, the book",
]
PHONE = {"viewport": {"width": 390, "height": 844}, "is_mobile": True}


def opened(context, site: str, where: str):
    page = context.new_page()
    page.goto(f"{site}{where}")
    page.wait_for_selector(".site-foot a")
    return page


@pytest.mark.parametrize("where", PAGES)
def test_each_page_ends_in_the_same_footer_with_the_same_links_in_the_same_order(reproduced_context, site, where):
    page = opened(reproduced_context, site, where)
    links = page.locator(".site-foot a")
    assert links.all_text_contents() == LINKS
    hrefs = links.evaluate_all("(all) => all.map((one) => one.href)")
    assert hrefs[:7] == [
        f"{site}{path}"
        for path in (
            "",
            "features.html",
            "catalog/",
            "validation.html",
            "catalog/reproduced/",
            "map/",
            "technical-details-how-factorial-works.html",
        )
    ]
    assert page.locator(".site-foot").count() == 1
    assert page.locator("footer").count() == 1, "one footer, not the page's own and the shared one"
    current = page.locator('.site-foot a[aria-current="page"]')
    assert current.evaluate_all("(all) => all.map((one) => one.href)") == [f"{site}{where}"], (
        "the page on show is marked"
    )


@pytest.mark.parametrize("where", PAGES)
def test_each_page_has_the_colour_mode_switch_at_the_end_of_its_top_line(reproduced_context, site, where):
    page = opened(reproduced_context, site, where)
    switch = page.locator("#mode")
    assert switch.count() == 1
    back = page.locator(".about__back a").bounding_box()
    box = switch.bounding_box()
    assert abs((box["y"] + box["height"] / 2) - (back["y"] + back["height"] / 2)) < 24, "on the line of the way back"
    assert box["x"] > back["x"] + back["width"], "at its end"
    before = page.evaluate("getComputedStyle(document.body).backgroundColor")
    switch.click()
    assert page.evaluate("getComputedStyle(document.body).backgroundColor") != before, (
        "the first press changes the page"
    )
    assert page.evaluate("localStorage.getItem('factorial-mode')") in ("dark", "light")
    assert switch.get_attribute("aria-label").startswith("Color mode: ")


def test_the_mode_chosen_on_one_page_is_the_mode_of_the_next(reproduced_context, site):
    page = opened(reproduced_context, site, "technical-details-how-factorial-works.html")
    page.click("#mode")
    chosen = page.evaluate("document.documentElement.dataset.theme")
    page.click('.site-foot a:has-text("Validation & technical accuracy")')
    page.wait_for_selector(".site-foot a")
    assert page.url == f"{site}validation.html"
    assert page.evaluate("document.documentElement.dataset.theme") == chosen


def test_the_app_links_to_validation_and_to_the_textbooks_answers_in_its_footer(page, site):
    assert page.locator('footer.foot a[href="validation.html"]').text_content() == "Validation & technical accuracy"
    assert page.locator('footer.foot a[href="catalog/reproduced/"]').text_content() == "Reproduced from the books"
    footer = page.locator("footer.foot nav a")
    assert footer.all_text_contents() == LINKS, "the same links as every other page's footer, in its order"


@pytest.mark.parametrize("context_options", [PHONE])
@pytest.mark.parametrize("where", PAGES)
def test_on_a_phone_the_top_line_and_the_footer_fit_the_screen(reproduced_context, site, where):
    page = opened(reproduced_context, site, where)
    assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
    assert page.is_visible("#mode")
