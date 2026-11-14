"""What Factorial does (features.html), and the footer that leads to it.

The page shows ``web/FEATURES.md``, the document the app is described from, read in the browser: one text, so the
page cannot say what the document does not. It shows it for a reader of the app: the headline points as cards, each
part folded under a line that says what it holds, each point in plain words with its details a press away, a box
that finds a word, and the app's own reading level. The footer is the way to it: what Factorial is in a sentence, a
button to the page, and its links in groups, each under a line that names it.
"""

from __future__ import annotations

import re

import pytest

from tests.web._support import WEB
from tests.web.e2e._page import overflowing

#: The list is on the page: the document has been read and drawn.
SHOWN = "document.querySelector('#features').getAttribute('aria-busy') === 'false'"

#: Whether a part of the list, named by its address, is at the top of the window.
AT_TOP = "(id) => Math.abs(document.getElementById(id).getBoundingClientRect().top) < 80"

#: The list as a reader sees it: the parts open, the parts and points shown, and the lines of details in view.
SEEN = """() => { const all = (s) => [...document.querySelectorAll(s)];
  const drawn = (e) => e.checkVisibility();   // not hidden, and in no fold that is closed
  return { open: all('.feat-part[open]').map((p) => p.querySelector('h3').id),
    parts: all('.feat-part').filter(drawn).map((p) => p.querySelector('h3').id),
    steps: all('h2.feat-step').filter(drawn).map((h) => h.textContent),
    cards: all('.feat-cards li').filter(drawn).length,
    points: all('.feat-body li, .feat-body tbody tr').filter(drawn).length,
    asks: all('details.more > summary').filter(drawn).length,
    details: all('details.more > p').filter(drawn).length,
    found: document.querySelector('#found').textContent, says: document.querySelector('#level-says').textContent,
    level: document.documentElement.dataset.level }; }"""

#: Where the app, the concept map and this page keep the reading level (levels.js).
LEVEL_KEY = "factorial-map-level"

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
    seen = page.evaluate(SEEN)
    assert seen["steps"] == ["In one minute", "Design", "Run", "Learn", "Around the study", "The fine print"]
    assert seen["cards"] == 9, "the headline points, as cards"
    assert len(seen["parts"]) >= 12
    assert seen["open"] == [], "each part is folded, under a line that says what it holds"
    assert page.locator(".feat-part .feat-says").count() == len(seen["parts"])
    assert seen["points"] == 0
    assert page.locator("#features .error").count() == 0


@pytest.mark.parametrize("context_options", [PHONE])
def test_on_a_phone_the_footer_and_the_list_fit_the_screen(page, context, site):
    page.locator("footer").scroll_into_view_if_needed()
    assert overflowing(page) == []
    heads = page.locator("footer nav .foot__head").evaluate_all("(heads) => heads.map((head) => head.offsetLeft)")
    assert len(set(heads)) <= 2, "the groups stand under one another, two abreast at most"
    listed = features(context, site)
    assert overflowing(listed) == []
    listed.evaluate("() => document.querySelectorAll('details').forEach((fold) => { fold.open = true; })")
    assert overflowing(listed) == [], "with every part and every line of details open too"
    tables = listed.locator("#features .scroll table")
    assert tables.count() == 2, "the limits and the library's designs, each in a box that scrolls if it must"
    level = listed.locator("#level label").evaluate_all("(labels) => labels.map((l) => l.getBoundingClientRect().top)")
    assert len(set(level)) == 1, "the three levels on one line"


# --------------------------------------------------------------------------- the page
def test_the_page_shows_the_document_from_its_headline_points_on(context, site):
    page = features(context, site)
    document = (WEB / "FEATURES.md").read_text(encoding="utf-8")
    designs = page.locator(".feat-part").first.locator(".feat-body ul").first.locator("li > strong").all_text_contents()
    assert len(designs) == 16
    assert all(f"- **{design}** For " in document for design in designs), "each is a point of the document"
    said = page.text_content("#features")
    assert said.lstrip().startswith("In one minute"), "the headline points come first"
    assert "CONTRIBUTING" not in said, "how the document is kept is for those who keep it"
    assert "**" not in said
    assert "](" not in said
    assert "Details:" not in said, "a line of details is under its point, behind the word Details"


def test_a_part_opens_on_a_press_and_from_the_keyboard(context, site):
    page = features(context, site)
    page.click("#run-it")
    seen = page.evaluate(SEEN)
    assert seen["open"] == ["run-it"]
    assert seen["points"] >= 7, "its points are shown"
    page.click("#run-it")
    assert page.evaluate(SEEN)["open"] == []
    page.focus(".feat-part >> nth=0 >> summary")
    page.keyboard.press("Enter")
    assert page.evaluate(SEEN)["open"] == ["choose-a-design"]
    assert page.locator(".feat-part[open] h4, .feat-part[open] p").first.is_visible()


def test_an_address_opens_at_its_part_and_one_from_when_the_parts_were_numbered_still_does(context, site):
    page = features(context, site, "#limits")
    # The part an address names is not on the page when the browser first looks for it: the page goes to it once shown.
    page.wait_for_function(AT_TOP, arg="limits")
    assert page.evaluate(SEEN)["open"] == ["limits"], "opened, as its heading is inside the fold"
    assert page.locator("#features table tbody tr").first.is_visible()
    old = features(context, site, "#9-enterprise-privacy-security-and-operation")
    old.wait_for_function(AT_TOP, arg="your-data-and-your-trust")
    assert old.evaluate(SEEN)["open"] == ["your-data-and-your-trust"]
    # A link of the list to another of its parts opens that part.
    old.click("#learn-the-method")
    old.click("#features a[href='#more-at-factorial']")
    old.wait_for_function(AT_TOP, arg="more-at-factorial")
    assert "more-at-factorial" in old.evaluate(SEEN)["open"]


def test_the_reading_level_sets_the_details_and_is_the_app_s_own(context, site):
    page = features(context, site)
    page.evaluate(f"localStorage.removeItem('{LEVEL_KEY}')")  # as on a first visit: the other tests run at the highest
    page.reload()
    page.wait_for_function(SHOWN)
    page.click("#your-data-and-your-trust")
    first = page.evaluate(SEEN)
    assert first["level"] == "2", "a first visit is at Intermediate, as the app's is"
    assert page.is_checked("#level input[value='2']")
    assert first["asks"] > 8, "each point's details are a press away"
    assert first["details"] == 0, "and none is open"
    assert first["says"] == "Each point in plain words. Press Details under a point for the rest."
    page.click(".feat-part[open] details.more >> nth=0 >> summary")
    assert page.evaluate(SEEN)["details"] == 1
    said = page.text_content(".feat-part[open] details.more[open] p")
    assert "no cookies and no tracking script" in said, "what only some readers ask is there for those who do"

    page.click("#level >> text=Beginner")
    plain = page.evaluate(SEEN)
    assert (plain["level"], plain["asks"], plain["details"]) == ("1", 0, 0), "plain words alone"
    assert plain["points"] == first["points"], "and no point is left out"
    assert "stay in your browser" in page.text_content(".feat-part[open] .feat-body")
    assert page.evaluate(f"() => localStorage.getItem('{LEVEL_KEY}')") == "1", "kept where the app keeps its level"

    page.click("#level >> text=Adventurous")
    full = page.evaluate(SEEN)
    assert full["level"] == "3"
    assert full["details"] == full["asks"] == first["asks"], "every line of details is open"
    page.reload()
    page.wait_for_function(SHOWN)
    assert page.is_checked("#level input[value='3']"), "a reload keeps the level"
    app = context.new_page()
    app.goto(site)
    assert app.evaluate("() => document.documentElement.dataset.level") == "3", "and the app opens at it"


def test_the_box_finds_a_word_and_shows_what_mentions_it_alone(context, site):
    page = features(context, site)
    page.click("#level >> text=Beginner")  # the level at which a line of details is not shown
    before = page.evaluate(SEEN)
    page.fill("#find", "box-cox")
    found = page.evaluate(SEEN)
    assert found["found"] == '1 point mentions "box-cox".'
    assert found["parts"] == found["open"] == ["understand-the-results"], "its part alone, opened"
    assert found["steps"] == ["Learn"]
    assert (found["cards"], found["points"]) == (0, 1)
    assert found["details"] == 1, "the word is in the point's details, so they are shown, at any level"
    assert "Box-Cox" in page.text_content(".feat-part[open] details.more[open] p")
    assert page.locator(".feat-part[open] h4").evaluate_all("(h) => h.filter((e) => e.checkVisibility()).length") == 1

    page.fill("#find", "limits")  # the name of a part: the part is shown whole, with what else mentions the word
    named = page.evaluate(SEEN)
    assert "limits" in named["open"]
    assert page.locator("#features table tbody tr").evaluate_all("(rows) => rows.every((r) => r.checkVisibility())")
    assert re.fullmatch(r'\d+ points mention "limits"\.', named["found"])

    page.fill("#find", "zebra")
    none = page.evaluate(SEEN)
    assert none["found"] == 'Nothing here mentions "zebra".'
    assert (none["parts"], none["cards"], none["steps"]) == ([], 0, [])

    page.fill("#find", "")
    assert page.evaluate(SEEN) == before, "with the box empty the list is as it was, its parts folded again"
    page.click("#run-it")  # a part the reader opened stays open through a search
    page.fill("#find", "noise")
    page.fill("#find", "")
    assert page.evaluate(SEEN)["open"] == ["run-it"]


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
    assert page.is_hidden("#find"), "with no list there is nothing to find a word in"
    assert page.is_hidden("#level")
    assert page.locator("main a[href$='web/FEATURES.md']").count() == 1, "the document, in the repository"
