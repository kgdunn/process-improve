"""Every page of the site beside the app has the same top line and the same footer (web/site.js).

The color-mode switch was on the app and the concept map only, each page ended in a footer of its own or in none,
and the page of the textbooks' answers was linked from one door on the catalog. Each page now has the switch at the
end of its top line and one footer, with the same links in the same order.

The footer had six links whose names ran to two lines on a phone, two of them on whether Factorial can be trusted.
It names each page in one to three words, and "Is it accurate?" is the one way to validation: the textbooks'
answers are reached from that page, and lead back to it.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

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
    "Open the app",
    "User guide",
    "Design catalog",
    "Is it accurate?",
    "Learn experimental design",
    "How it works",
    "Python package",
]
#: The page the footer does not name: it is reached from validation, so no link of the footer is the page on show.
REACHED_FROM_VALIDATION = "catalog/reproduced/"
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
    assert hrefs[:6] == [
        f"{site}{path}"
        for path in (
            "",
            "features.html",
            "catalog/",
            "validation.html",
            "map/",
            "technical-details-how-factorial-works.html",
        )
    ]
    assert page.locator(".site-foot").count() == 1
    assert page.locator("footer").count() == 1, "one footer, not the page's own and the shared one"
    current = page.locator('.site-foot a[aria-current="page"]')
    shown = [] if where == REACHED_FROM_VALIDATION else [f"{site}{where}"]
    assert current.evaluate_all("(all) => all.map((one) => one.href)") == shown, "the page on show is marked"


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
    page.click('.site-foot a:has-text("Is it accurate?")')
    page.wait_for_selector(".site-foot a")
    assert page.url == f"{site}validation.html"
    assert page.evaluate("document.documentElement.dataset.theme") == chosen


def test_the_app_links_to_validation_in_its_footer_under_the_question_a_reader_asks(page, site):
    assert page.locator('footer.foot a[href="validation.html"]').text_content() == "Is it accurate?"
    assert page.locator('footer.foot a[href="catalog/reproduced/"]').count() == 0, "reached from validation"
    footer = page.locator("footer.foot nav a")
    assert footer.all_text_contents() == LINKS, "the same links as every other page's footer, in its order"


def test_is_it_accurate_opens_validation_and_the_textbooks_answers_lead_back_to_it(reproduced_context, site):
    page = opened(reproduced_context, site, "features.html")
    page.click('.site-foot a:has-text("Is it accurate?")')
    page.wait_for_url("**/validation.html")
    assert page.text_content("h1") == "Is Factorial accurate?", "the heading is the question the link asked"
    page.click("text=Browse the books")
    page.wait_for_url("**/catalog/reproduced/")
    back = page.locator(".about__back a")
    assert back.text_content().strip() == 'Back to "Is it accurate?"'
    back.click()
    page.wait_for_url("**/validation.html")


@pytest.mark.parametrize("context_options", [PHONE])
def test_on_a_phone_no_link_of_the_footer_takes_a_second_line(reproduced_context, site):
    """Six long names took nine lines of a phone's screen, and a reader could not tell them apart at a glance."""
    page = opened(reproduced_context, site, "validation.html")
    heights = page.locator(".site-foot a").evaluate_all("(all) => all.map((one) => one.getClientRects().length)")
    assert heights == [1] * len(LINKS), "each name is on one line"


@pytest.mark.parametrize("context_options", [PHONE])
@pytest.mark.parametrize("where", PAGES)
def test_on_a_phone_the_top_line_and_the_footer_fit_the_screen(reproduced_context, site, where):
    page = opened(reproduced_context, site, where)
    assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
    assert page.is_visible("#mode")


#: The dot's color, and the page's red (--error) as the browser resolves it.
DOT_AND_RED = """(dot) => {
  const probe = document.createElement('i');
  probe.style.color = 'var(--error)';
  dot.closest('footer').append(probe);
  const red = getComputedStyle(probe).color;
  probe.remove();
  return [getComputedStyle(dot).fill, red];
}"""


def counts(recent: int, days: int = 90, *, hours_old: float = 0.25) -> str:
    """Return usage.json as the server writes it: ``days`` finished days of readers, and ``recent`` in the last
    24 hours, written ``hours_old`` hours ago.
    """
    updated = datetime.now(UTC) - timedelta(hours=hours_old)
    end = updated.date() - timedelta(days=1)
    series = [[(end - timedelta(days=days - 1 - i)).isoformat(), 40 + i % 7] for i in range(days)]
    stamp = updated.strftime("%Y-%m-%dT%H:%M:%SZ")
    return json.dumps({"updated": stamp, "days": series, "last_24_hours": recent})


@pytest.mark.parametrize("where", ["", "features.html", "map/"])
def test_the_footer_draws_how_many_people_used_factorial_each_day(reproduced_context, site, where):
    """Under "Behind it": a line of the last 90 days, then a red dot on the last 24 hours, and that count beside it."""
    reproduced_context.route(
        f"{site}usage.json", lambda route: route.fulfill(body=counts(30), content_type="application/json")
    )
    page = reproduced_context.new_page()
    page.goto(f"{site}{where}")
    usage = page.locator("footer .foot__usage")
    usage.wait_for(state="visible")
    link = usage.locator("a.usage")
    assert link.inner_text().split() == ["30", "users", "last", "24", "hours"]
    assert link.get_attribute("href").endswith("technical-details-how-factorial-works.html#h-servers")
    labelled = "(item) => document.getElementById(item.closest('ul').getAttribute('aria-labelledby')).textContent"
    assert usage.evaluate(labelled) == "Behind it"
    points = link.locator("polyline").get_attribute("points").split()
    assert len(points) == 91, "90 finished days, then the last 24 hours"
    dot, red = link.locator("circle").evaluate(DOT_AND_RED)
    assert dot == red, "the dot is the page's red, in light and in dark"
    cx, last_x = link.locator("circle").get_attribute("cx"), points[-1].split(",")[0]
    assert float(cx) == float(last_x), "the dot is on the last 24 hours"


NO_COUNTS = [
    (None, "the build's own file, of no counts: a copy of Factorial elsewhere draws nothing"),
    (counts(12, hours_old=25), "a file a day old is stale"),
    (json.dumps({"updated": "2026-11-25T10:00:00Z", "days": [["2026-11-25", 9]]}), "an older server's file"),
    ("not json", "a file that is no JSON"),
]


# Each case is named by what it says, not by its body: the stale file's body holds the clock's second, so two
# xdist workers that collect a second apart would name it differently and the run would stop.
@pytest.mark.parametrize(("body", "said"), NO_COUNTS, ids=[said for _, said in NO_COUNTS])
def test_the_footer_leaves_the_line_out_when_there_are_no_counts_to_draw(reproduced_context, site, body, said):
    if body is not None:
        reproduced_context.route(
            f"{site}usage.json", lambda route: route.fulfill(body=body, content_type="application/json")
        )
    page = opened(reproduced_context, site, "features.html")
    page.wait_for_load_state("networkidle")
    assert page.locator(".site-foot .foot__usage").is_hidden(), said
    assert page.locator(".site-foot a").all_text_contents() == LINKS
