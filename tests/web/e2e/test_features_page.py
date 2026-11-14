"""What Factorial does (features.html), and the footer that leads to it.

The page shows ``web/FEATURES.md``, the document the app is described from, read in the browser: one text, so the
page cannot say what the document does not. The footer is the way to it: what Factorial is in a sentence, a button
to the page, and its links in groups, each under a line that names it.
"""

from __future__ import annotations

import pytest

from tests.web._support import WEB
from tests.web.e2e._page import overflowing

#: The list is on the page: the document has been read and drawn.
SHOWN = "document.querySelector('#features').getAttribute('aria-busy') === 'false'"

#: Whether a part of the list, named by its address, is at the top of the window.
AT_TOP = "(id) => Math.abs(document.getElementById(id).getBoundingClientRect().top) < 80"

PHONE = {"viewport": {"width": 390, "height": 844}, "is_mobile": True}


def features(context, site, part: str = ""):
    """Open the page, at one of its parts when ``part`` names it; return it once the list is shown."""
    page = context.new_page()
    page.goto(f"{site}features.html{part}")
    page.wait_for_function(SHOWN)
    return page


# --------------------------------------------------------------------------- the footer
def test_the_footer_says_what_factorial_is_and_names_each_group_of_its_links(page):
    footer = page.locator("footer")
    said = " ".join(footer.locator(".foot__lead p").text_content().split())
    assert said.startswith("Factorial is the browser app of process-improve, the Python package of the book")
    assert said.endswith("Everything here runs in your browser tab, nothing on our server.")
    groups = footer.locator("nav ul").evaluate_all(
        """(lists) => lists.map((list) => [document.getElementById(list.getAttribute('aria-labelledby')).textContent,
             [...list.querySelectorAll('a, button')].map((to) => to.textContent)])"""
    )
    assert groups == [
        ["The app", ["How Factorial works", "The design catalog", "How its ideas link"]],
        ["Behind it", ["process-improve, the Python package", "Process Improvement using Data, the book"]],
        ["Something wrong or missing?", ["Send feedback"]],
    ]
    assert footer.locator("nav").get_attribute("aria-label") == "More about Factorial"
    assert page.is_hidden("#offline"), "no copy is kept here, so the footer has no word on one"


def test_the_footer_s_button_opens_the_list_of_what_factorial_does(page):
    button = page.locator("footer a.button.primary")
    assert button.text_content() == "See what Factorial does"
    button.click()
    page.wait_for_url("**/features.html")
    page.wait_for_function(SHOWN)
    assert page.text_content("h1") == "What Factorial does"
    parts = page.locator("#features h2").all_text_contents()
    assert len(parts) == 13
    assert [parts[0], parts[-1]] == ["1. In one minute", "13. Not in the app"]
    assert page.locator("#features table").count() == 3, "the designs, the charts and the limits"
    assert page.locator("#features .error").count() == 0


@pytest.mark.parametrize("context_options", [PHONE])
def test_on_a_phone_the_footer_and_the_list_fit_the_screen(page, context, site):
    page.locator("footer").scroll_into_view_if_needed()
    assert overflowing(page) == []
    heads = page.locator("footer nav .foot__head").evaluate_all("(heads) => heads.map((head) => head.offsetLeft)")
    assert len(set(heads)) <= 2, "the groups stand under one another, two abreast at most"
    listed = features(context, site)
    assert overflowing(listed) == [], "a table too wide for the screen scrolls in a box of its own"
    assert listed.locator("#features .scroll table").count() == 3


# --------------------------------------------------------------------------- the page
def test_the_page_shows_the_document_from_its_contents_on(context, site):
    page = features(context, site)
    document = (WEB / "FEATURES.md").read_text(encoding="utf-8")
    designs = page.locator("#features table").first.locator("tbody tr td:first-child").all_text_contents()
    assert len(designs) == 16
    assert all(f"| {design} |" in document for design in designs), "each row is a row of the document"
    said = page.text_content("#features")
    assert said.lstrip().startswith("In one minute"), "the contents come first"
    assert "CONTRIBUTING" not in said, "how the document is kept is for those who keep it"
    assert "**" not in said
    assert "](" not in said


def test_the_contents_lead_to_each_part_and_an_address_opens_at_its_part(context, site):
    page = features(context, site)
    page.click("#features ol >> text=Limits")
    page.wait_for_function(AT_TOP, arg="12-limits")
    assert page.url.endswith("features.html#12-limits")
    # The part an address names is not on the page when the browser first looks for it: the page goes to it once shown.
    other = features(context, site, "#6-figures")
    other.wait_for_function(AT_TOP, arg="6-figures")


def test_the_page_s_links_go_to_the_app_and_its_other_pages(context, site):
    page = features(context, site)
    inside = page.locator("#features a:not([href^='#']):not([href^='http'])").evaluate_all(
        "(links) => links.map((link) => link.getAttribute('href'))"
    )
    assert inside == ["about.html", "catalog/", "catalog/reproduced/", "map/", "mva/"]
    outside = page.locator("#features a[href^='http']")
    assert outside.count() >= 5
    assert outside.evaluate_all("(links) => links.every((link) => link.target === '_blank' && link.rel === 'noopener')")
    page.click("text=Start a study")
    page.wait_for_url(site)


def test_a_list_that_cannot_be_read_says_so(context, site, console):
    console.allow(r"net::ERR_FAILED")  # the browser reports the request the lost network refuses
    page = context.new_page()
    page.route("**/FEATURES.md", lambda route: route.abort())
    page.goto(f"{site}features.html")
    page.wait_for_function(SHOWN)
    assert page.text_content("#features .error").startswith("The list could not be read.")
    assert page.locator("main a[href$='web/FEATURES.md']").count() == 1, "the document, in the repository"
