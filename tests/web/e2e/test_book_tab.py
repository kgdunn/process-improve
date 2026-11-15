"""Links into the book open in one tab, which a later link finds again and goes to (``campaign.tabOf``).

A browser finds a tab by its name only from the tab that opened it. A link that cuts that tie (``rel="noopener"``)
therefore opens a tab of its own every time, however the tab is named: that is what the book's links did. They now
keep the tie, and the book's tab can reach the app's in return, which is accepted as the book is the app's own
site. A link to any other site still opens a new tab with no tie. The sites are answered by the test, so no test
here needs a network.
"""

from __future__ import annotations

import pytest

from tests.web.e2e._page import settled

#: Whether a tab can reach the tab that opened it.
TIED = "() => window.opener !== null"


@pytest.fixture
def sites(context):
    """Answer the book, and any other site a link leads to, with a page of the test's own."""
    for site in ("https://learnche.org/**", "https://en.wikipedia.org/**"):
        context.route(site, lambda route: route.fulfill(content_type="text/html", body="<title>A site</title>"))
    return context


def the_map(context, site: str):
    page = context.new_page()
    page.goto(f"{site}map/")
    page.wait_for_function("() => window.conceptMap && document.querySelector('#card h2')")
    return page


def go_to(page, term: str) -> str:
    """Select a term of the map by its address, and return where its card's link into the book leads."""
    page.evaluate("(term) => { location.hash = term; }", term)
    page.wait_for_function("(term) => window.conceptMap.cy.nodes('.sel').map((n) => n.id())[0] === term", arg=term)
    page.wait_for_selector("#card .map-book a")
    return page.get_attribute("#card .map-book a", "href")


def test_the_map_s_links_into_the_book_open_one_tab_and_go_to_it_again(sites, site):
    page = the_map(sites, site)
    first = go_to(page, "resolution")
    with sites.expect_page() as opened:
        page.click("#card .map-book a")
    book = opened.value
    book.wait_for_url(first)
    assert len(sites.pages) == 2, "the map, and the book in a tab of its own"

    page.bring_to_front()
    second = go_to(page, "blocking")
    assert second != first
    page.click("#card .map-book a")
    book.wait_for_url(second)
    assert len(sites.pages) == 2, "the same tab, not one more"
    assert book.evaluate(TIED), "found again by the tie to the tab that opened it"
    assert page.url.startswith(site), "and the map stays where it is"


def test_the_app_s_links_into_the_book_use_the_same_tab_as_the_map_s(sites, site):
    page = the_map(sites, site)
    go_to(page, "resolution")
    with sites.expect_page() as opened:
        page.click("#card .map-book a")
    book = opened.value
    book.wait_for_load_state()
    # The same tab goes on to the app, as a reader who follows the map's way back does.
    page.bring_to_front()
    page.goto(f"{site}?example=worked")
    page.wait_for_selector("footer a[target='pid-book']")
    settled(page)
    link = page.locator("footer a[target='pid-book']")
    assert link.get_attribute("rel") == "opener"
    link.click()
    book.wait_for_url(link.get_attribute("href"))
    assert len(sites.pages) == 2, "the book's tab, which the map opened, is the one the app's link goes to"


def test_a_link_to_any_other_site_opens_a_new_tab_each_time_that_cannot_reach_the_app(sites, site):
    page = the_map(sites, site)
    page.click("#details")
    other = page.locator("a[href^='https://en.wikipedia.org/']").first
    # As a reader who opened the part of the page that holds the link.
    other.evaluate(
        "(a) => { for (let d = a.closest('details'); d; d = d.parentElement.closest('details')) d.open = true; }"
    )
    other.scroll_into_view_if_needed()
    tabs = []
    for _ in range(2):
        page.bring_to_front()
        with sites.expect_page() as opened:
            other.click()
        tabs.append(opened.value)
        tabs[-1].wait_for_load_state()
    assert len(sites.pages) == 3, "a tab for each press"
    assert not any(tab.evaluate(TIED) for tab in tabs), "and neither can reach back into the app's tab"
