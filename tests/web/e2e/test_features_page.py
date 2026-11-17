"""What Factorial does (features.html), and the footer that leads to it.

The page shows ``web/FEATURES.md``, the document the app is described from, read in the browser: one text, so the
page cannot say what the document does not. It shows it for a reader of the app: the headline points as cards, each
part folded under a line that says what it holds, each point in plain words with its details a press away, a box
that finds a word, and a switch that opens every point's details. The footer is the way to it: what Factorial is in a
sentence, a button to the page, and its links in groups, each under a line that names it.
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
    found: document.querySelector('#found').textContent, says: document.querySelector('#details-says').textContent,
    switched: document.querySelector('#details').checked }; }"""

#: Where the page keeps whether every point's details are open: a key of its own, apart from the app's.
DETAILS_KEY = "factorial-features-details"

#: Where the concept map keeps its reading level, which this page leaves alone.
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
    assert seen["steps"] == ["In one minute", "Design", "Run", "Learn", "Next", "Around the study", "The fine print"]
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
    switch = listed.locator(".feat-details").bounding_box()
    assert switch["height"] >= 44, "the switch is a row a finger can press"


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


def test_a_switch_opens_every_point_s_details_and_the_choice_is_the_page_s_own(context, site):
    page = features(context, site)
    page.evaluate(f"localStorage.removeItem('{DETAILS_KEY}')")  # as on a first visit: other tests have them open
    page.reload()
    page.wait_for_function(SHOWN)
    page.click("#your-data-and-your-trust")
    first = page.evaluate(SEEN)
    assert first["switched"] is False, "a first visit has the details closed"
    assert first["asks"] > 8, "each point's details are a press away"
    assert first["details"] == 0, "and none is open"
    assert first["says"] == "Each point in plain words. Press Details under a point for the rest."
    page.click(".feat-part[open] details.more >> nth=0 >> summary")
    assert page.evaluate(SEEN)["details"] == 1
    said = page.text_content(".feat-part[open] details.more[open] p")
    assert "no cookies and no tracking script" in said, "what only some readers ask is there for those who do"

    page.check("#details")
    full = page.evaluate(SEEN)
    assert full["details"] == full["asks"] == first["asks"], "every line of details is open"
    assert full["points"] == first["points"], "and the points are the same"
    assert full["says"] == "Each point in plain words, with its details open."
    assert page.evaluate(f"() => localStorage.getItem('{DETAILS_KEY}')") == "open"
    page.reload()
    page.wait_for_function(SHOWN)
    assert page.is_checked("#details"), "a reload keeps the choice"
    page.click("#your-data-and-your-trust")
    assert page.evaluate(SEEN)["details"] == first["asks"]

    page.uncheck("#details")
    closed = page.evaluate(SEEN)
    assert (closed["asks"], closed["details"]) == (first["asks"], 0), "closed again, and still a press away"
    assert page.evaluate(f"() => localStorage.getItem('{DETAILS_KEY}')") is None
    assert page.evaluate(f"() => localStorage.getItem('{LEVEL_KEY}')") is None, "the map's level is not the page's"
    assert page.evaluate("() => document.documentElement.dataset.level") is None, "and the page is drawn at no level"


def test_the_box_finds_a_word_and_shows_what_mentions_it_alone(context, site):
    page = features(context, site)
    page.uncheck("#details")  # closed, so that only the search opens a line of details
    before = page.evaluate(SEEN)
    page.fill("#find", "box-cox")
    found = page.evaluate(SEEN)
    assert found["found"] == '1 point mentions "box-cox".'
    assert found["parts"] == found["open"] == ["understand-the-results"], "its part alone, opened"
    assert found["steps"] == ["Learn"]
    assert (found["cards"], found["points"]) == (0, 1)
    assert found["details"] == 1, "the word is in the point's details, so they are opened to show it"
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
    inside = page.locator("#features a:not([href^='#']):not([href^='http']):not(.show-me)").evaluate_all(
        "(links) => links.map((link) => link.getAttribute('href'))"
    )
    pages = [href for href in inside if not href.startswith("./?request=")]
    assert pages == ["about.html", "catalog/", "catalog/reproduced/", "map/", "mva/"]
    assert len(inside) - len(pages) >= 5, "and a request for each design the app does not make yet"
    outside = page.locator("#features a[href^='http']")
    assert outside.count() >= 5
    tabs = outside.evaluate_all("(links) => links.map((link) => [new URL(link.href).hostname, link.target, link.rel])")
    assert {(target, rel) for host, target, rel in tabs if host == "learnche.org"} == {("pid-book", "opener")}
    assert {(target, rel) for host, target, rel in tabs if host != "learnche.org"} == {("_blank", "noopener")}
    page.click("text=Start a study")
    page.wait_for_url(site)


def test_request_it_opens_the_app_s_feedback_with_the_request_for_that_design_written(context, site):
    page = features(context, site)
    link = page.locator("#features a[href='./?request=supersaturated']")
    assert link.count() == 1
    assert (link.text_content(), link.get_attribute("aria-label")) == ("Request it", "Request it: supersaturated")
    link.evaluate("(a) => { a.closest('details.feat-part').open = true; }")  # as a reader who opened its part
    link.click()
    page.wait_for_selector("#feedback-dialog[open]", timeout=30_000)
    asked = "Please add the supersaturated design, which the process-improve library makes, to Factorial."
    assert page.input_value("#feedback-message") == asked
    assert page.url.startswith(site), "in this copy of the app, whatever address the document names"
    assert "request=" not in page.url


#: Each "Show me" of the document: the example it opens and the part of the app's page it asks for.
SHOW_ME = re.findall(
    r"\[Show me\]\(https://factori\.al/\?((?:book|example)=[a-z0-9-]+)&show=([a-z0-9-]+)\)",
    (WEB / "FEATURES.md").read_text(encoding="utf-8"),
)


@pytest.mark.parametrize(("example", "part"), SHOW_ME, ids=[f"{example}, {part}" for example, part in SHOW_ME])
def test_show_me_opens_the_app_on_an_example_of_the_point_at_the_part_that_has_it(context, site, example, part):
    page = features(context, site)
    link = page.locator(f"#features a.show-me[href='./?{example}&show={part}']")
    assert link.count() == 1
    assert link.text_content() == "Show me"
    assert link.get_attribute("aria-label").startswith("Show me: "), "a screen reader is told which point it shows"
    link.evaluate("(a) => { a.closest('details.feat-part').open = true; }")  # as a reader who opened its part
    link.click()
    page.wait_for_selector(f"#{part}.shown-part", timeout=30_000)
    assert page.locator(f"#{part}").evaluate("(e) => e.checkVisibility()")
    assert page.url.startswith(site), "in this copy of the app, whatever address the document names"
    assert "show=" not in page.url


# --------------------------------------------------------------------------- the charts' pictures
#: The charts' cards as a reader sees them: each picture's size as loaded and as drawn, and whether it is turned over.
PICTURES = """() => [...document.querySelectorAll('.feat-pictures > li')].map((card) => {
  const picture = card.querySelector('img');
  return { title: card.querySelector('strong').textContent, file: picture?.getAttribute('src'),
    loaded: picture ? [picture.naturalWidth, picture.naturalHeight] : null, alt: picture?.getAttribute('alt'),
    wide: picture ? picture.getBoundingClientRect().width / card.getBoundingClientRect().width : 0,
    filter: picture ? getComputedStyle(picture).filter : null }; })"""

#: Every picture on the page has loaded, or has gone.
LOADED = (
    "[...document.querySelectorAll('.feat-pictures img')].every((picture) => picture.complete && picture.naturalWidth)"
)


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_each_chart_has_its_picture_in_the_page_s_colours(context, site, scheme):
    page = context.new_page()
    page.emulate_media(color_scheme=scheme)
    fetched: list[str] = []
    page.on("request", lambda request: fetched.append(request.url))
    page.goto(f"{site}features.html")
    page.wait_for_function(SHOWN)
    assert not [url for url in fetched if ".webp" in url], "none is fetched while its fold is closed"
    page.goto(f"{site}features.html#the-17-charts")
    page.wait_for_function(SHOWN)
    page.evaluate("document.querySelectorAll('.feat-pictures img').forEach((picture) => (picture.loading = 'eager'))")
    page.wait_for_function(LOADED)
    cards = page.evaluate(PICTURES)
    size = page.evaluate("async () => (await import('./features.js')).PICTURE_SIZE")
    assert len(cards) == 17
    assert cards[0]["title"] == "Coefficients."
    assert cards[0]["file"] == "./chart-pictures/coefficients.webp"
    for card in cards:
        assert card["loaded"] == size, card["title"]
        assert card["alt"] == "", "the words beside it say what it shows"
        assert card["wide"] > 0.85, "it fills its card"
        assert (card["filter"] != "none") == (scheme == "dark"), "turned over in the dark, as the app draws the chart"
    assert (
        page.locator(".feat-pictures").evaluate(
            "(list) => getComputedStyle(list).gridTemplateColumns.split(' ').length"
        )
        == 3
    )
    assert not overflowing(page)


@pytest.mark.parametrize("context_options", [PHONE])
def test_on_a_phone_the_charts_are_two_to_a_row_and_fit_the_screen(context, site):
    page = features(context, site, "#the-17-charts")
    columns = page.locator(".feat-pictures").evaluate(
        "(list) => getComputedStyle(list).gridTemplateColumns.split(' ').length"
    )
    assert columns == 2
    assert not overflowing(page)


def test_a_picture_that_cannot_be_had_leaves_its_card_to_its_words(context, site, console):
    console.allow(r"net::ERR_FAILED")  # the browser reports the picture it could not fetch
    context.route("**/chart-pictures/contour.webp", lambda route: route.abort())
    page = features(context, site, "#the-17-charts")
    page.evaluate("document.querySelectorAll('.feat-pictures img').forEach((picture) => (picture.loading = 'eager'))")
    page.wait_for_function("document.querySelectorAll('.feat-pictures img').length === 16")
    page.wait_for_function(LOADED)
    card = page.locator(".feat-pictures > li", has_text="Fitted response surface.").first
    assert card.locator("img").count() == 0
    assert card.text_content().startswith("Fitted response surface. A map of the response")


def test_the_box_finds_a_chart_and_shows_its_card_alone(context, site):
    page = features(context, site)
    page.fill("#find", "half-normal")
    shown = page.locator(".feat-pictures > li").evaluate_all(
        "(cards) => cards.filter((c) => c.checkVisibility()).map((c) => c.querySelector('strong').textContent)"
    )
    assert shown == ["Half-normal plot of the effects."]
    page.fill("#find", "no such word at all")
    assert page.locator(".feat-pictures").evaluate("(list) => list.checkVisibility()") is False, (
        "the set of cards goes with its points"
    )


def test_a_list_that_cannot_be_read_says_so(context, site, console):
    console.allow(r"net::ERR_FAILED")  # the browser reports the request the lost network refuses
    page = context.new_page()
    page.route("**/FEATURES.md", lambda route: route.abort())
    page.goto(f"{site}features.html")
    page.wait_for_function(SHOWN)
    assert page.text_content("#features .error").startswith("The list could not be read.")
    assert page.is_hidden("#find"), "with no list there is nothing to find a word in"
    assert page.is_hidden("#details")
    assert page.locator("main a[href$='web/FEATURES.md']").count() == 1, "the document, in the repository"
