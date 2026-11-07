"""The concept map (map/index.html): the ideas of designed experiments, and how the book links them.

Its graph, ``web/map/graph.json``, is built and checked against the book in pid-book's ``knowledge-graph/``. These
tests drive the page as a reader would, at a phone's width where that matters: what a tap opens, what zooming shows,
and what the page remembers. ``window.conceptMap`` gives them the map's ``cy`` and its data.
"""

from __future__ import annotations

import json
import re
from itertools import pairwise
from typing import Any

import pytest

from tests.web._support import WEB

GRAPH = json.loads((WEB / "map" / "graph.json").read_text(encoding="utf-8"))
PHONE = {"viewport": {"width": 390, "height": 844}, "is_mobile": True}
DETAIL_GAP, LABEL_GAP, TERM_GAP = 19, 45, 62  # as map.js: terms appear, then sub-topic terms are named

# Every term's name drawn on the map, and every topic's and sub-topic's, as rendered boxes; and how many overlap.
STATE = """() => {
  const cy = window.conceptMap.cy;
  const box = (n) => n.renderedBoundingBox({ includeNodes: false, includeEdges: false, includeLabels: true });
  const hit = (a, b) => a.x1 < b.x2 && b.x1 < a.x2 && a.y1 < b.y2 && b.y1 < a.y2;
  const labels = cy.nodes(".term.lab"), boxes = labels.map(box);
  const names = cy.nodes(".topic, .subtopic, .subname").filter((n) => n.visible() && n.style("label")).map(box);
  let overlaps = 0;
  boxes.forEach((a, i) => {
    boxes.slice(i + 1).forEach((b) => { if (hit(a, b)) overlaps++; });
    names.forEach((g) => { if (hit(a, g)) overlaps++; });
  });
  return { zoom: cy.zoom(), pan: { ...cy.pan() }, labels: labels.length, overlaps,
    font: labels.length ? parseFloat(labels[0].renderedStyle("font-size")) : 0,
    terms: cy.nodes(".term").filter((n) => n.visible()).length,
    links: cy.edges(".link").filter((e) => e.visible()).length };
}"""


# Resolves once the map has stopped moving and redrawn: a fixed sleep raced the zoom under load. Two frames first, as
# an animation started by a click joins cy.animated() on the next tick; two after, for the refresh a move queues.
SETTLED = """() => new Promise((done) => {
  const cy = window.conceptMap.cy, frame = (f) => requestAnimationFrame(f);
  const wait = () => (cy.animated() ? frame(wait) : frame(() => frame(done)));
  frame(() => frame(wait));
})"""


def _settled(page: Any) -> None:  # noqa: ANN401 - a Playwright Page
    page.evaluate(SETTLED)


def _go(page: Any, term_id: str) -> None:  # noqa: ANN401 - a Playwright Page
    """Select a term by its address, as a shared link does, and wait for the map to reach it."""
    name = next(n["name"] for n in GRAPH["nodes"] if n["id"] == term_id)
    page.evaluate("(h) => { location.hash = h; }", term_id.replace("_", "-"))
    page.wait_for_function("(name) => document.querySelector('#card h2').textContent === name", arg=name)
    _settled(page)


def _open(context: Any, site: str) -> Any:  # noqa: ANN401 - Playwright objects
    page = context.new_page()
    page.goto(f"{site}map/")
    page.wait_for_function("() => window.conceptMap && document.querySelector('#card h2')")
    return page


def _on_map(page: Any, point: dict[str, float]) -> tuple[float, float]:  # noqa: ANN401 - a Playwright Page
    """Return a point drawn on the map in the page's coordinates."""
    box = page.locator("#cy").bounding_box()
    return box["x"] + point["x"], box["y"] + point["y"]


def _rendered(page: Any, node_id: str) -> dict[str, float]:  # noqa: ANN401 - a Playwright Page
    return page.evaluate("(i) => window.conceptMap.cy.getElementById(i).renderedPosition()", node_id)


def _open_topic(page: Any, topic: dict[str, Any]) -> None:  # noqa: ANN401 - a Playwright Page
    """Tap the topic's circle, from the overview, as a reader would."""
    page.click("#home")
    _settled(page)
    page.mouse.click(*_on_map(page, _rendered(page, topic["id"])))
    page.wait_for_function("(name) => document.querySelector('#card h2').textContent === name", arg=topic["name"])
    _settled(page)


BIGGEST = max(GRAPH["clusters"], key=lambda c: c["size"])


@pytest.mark.parametrize("context_options", [PHONE, {"viewport": {"width": 1280, "height": 900}}])
def test_it_opens_on_the_topics_with_their_terms_hidden(context, site):
    page = _open(context, site)
    biggest = page.evaluate("(i) => window.conceptMap.cy.getElementById(i).data('olabel')", BIGGEST["id"])
    assert biggest == f"{BIGGEST['short']}\n[{BIGGEST['size']}]", "a topic's name, and its number of terms below it"
    sub = BIGGEST["subs"][0]
    named = page.evaluate("(i) => window.conceptMap.cy.getElementById(i).data('label')", f"{sub['id']}-name")
    assert named == f"{sub['name']}\n[{sub['size']}]", "and a sub-topic's"
    assert page.text_content("#card h2") == f"{len(GRAPH['clusters'])} topics"
    assert page.evaluate(STATE)["terms"] == 0
    assert page.locator("#links .map-chip").count() == len(GRAPH["clusters"])
    examples = page.get_attribute('#legend button[data-fam="context"]', "aria-pressed")
    assert examples == "false", "examples, questions and sources start hidden"


@pytest.mark.parametrize("context_options", [PHONE])
def test_a_tap_on_a_topic_opens_it_and_draws_its_sub_topics_named_in_their_middle(context, site):
    page = _open(context, site)
    _open_topic(page, BIGGEST)
    inner = page.evaluate(
        """(i) => {
          const cy = window.conceptMap.cy;
          const subs = window.conceptMap.data.clusters.find((c) => c.id === i).subs;
          return { subs: subs.length, drawn: subs.filter((s) => cy.getElementById(s.id).visible()).length,
            gap: 62 * cy.zoom(), named: cy.nodes(".term.lab").filter((n) => n.data("topic") === i).length };
        }""",
        BIGGEST["id"],
    )
    assert inner["subs"] > 1
    assert inner["drawn"] == inner["subs"], "opening a topic comes close enough for its sub-topics, even on a phone"
    assert inner["gap"] >= DETAIL_GAP
    if inner["gap"] < LABEL_GAP:
        assert inner["named"] == 0, "while sub-topics are named mid-circle, they stand for their terms"


@pytest.mark.parametrize("context_options", [PHONE])
def test_a_tap_inside_a_sub_topic_clear_of_its_terms_opens_it(context, site):
    page = _open(context, site)
    _open_topic(page, BIGGEST)
    spot = page.evaluate(
        """(i) => {
          const cy = window.conceptMap.cy, sub = window.conceptMap.data.clusters.find((c) => c.id === i).subs[0];
          const node = cy.getElementById(sub.id), c = node.renderedPosition(), r = node.renderedWidth() / 2;
          const dots = cy.nodes(".term").filter((n) => n.visible()).map((n) => n.renderedPosition());
          for (let f = 0; f <= 0.8; f += 0.1)
            for (let a = 0; a < 2 * Math.PI; a += Math.PI / 8) {
              const x = c.x + f * r * Math.cos(a), y = c.y + f * r * Math.sin(a);
              const inside = x > 5 && y > 5 && x < cy.width() - 5 && y < cy.height() - 5;
              if (inside && dots.every((d) => Math.hypot(d.x - x, d.y - y) > 16)) return { x, y, name: sub.name };
            }
          return null;
        }""",
        BIGGEST["id"],
    )
    assert spot, "a clear spot to tap"
    page.mouse.click(*_on_map(page, spot))
    page.wait_for_function("(name) => document.querySelector('#card h2').textContent === name", arg=spot["name"])
    assert page.text_content("#card .map-type").startswith("Sub-topic of ")


@pytest.mark.parametrize("context_options", [PHONE, {"viewport": {"width": 1280, "height": 900}}])
def test_zooming_in_names_more_terms_never_overlapping_and_a_little_larger(context, site):
    page = _open(context, site)
    _open_topic(page, BIGGEST)
    fonts = []
    for _ in range(3):
        state = page.evaluate(STATE)
        assert state["overlaps"] == 0, f"{state['overlaps']} overlaps among {state['labels']} names"
        fonts.append(state["font"])
        page.click("#zin")
        _settled(page)
    named = [f for f in fonts if f]
    assert named, "terms are named once zoomed in"
    assert all(b > a or b >= 19 for a, b in pairwise(named)), fonts


def test_a_tap_on_a_term_shows_its_definition_its_book_line_and_lights_its_links(context, site):
    page = _open(context, site)
    _go(page, "full_factorial")
    page.mouse.click(*_on_map(page, _rendered(page, "full_factorial")))
    _settled(page)
    term = next(n for n in GRAPH["nodes"] if n["id"] == "full_factorial")
    assert page.text_content("#card h2") == term["name"]
    assert page.text_content("#card .map-quote") == term["quote"]
    assert ".rst" not in page.text_content("#card"), "the book's line, not its source file"
    assert page.get_attribute("#card .map-book a", "href").startswith("https://learnche.org/pid/")
    lit = page.evaluate(
        "() => { const cy = window.conceptMap.cy; "
        "return [cy.edges('.link.hl').length, cy.getElementById('full_factorial').connectedEdges('.link').length]; }"
    )
    assert lit[0] == lit[1] > 0


def test_every_link_reads_as_a_sentence_from_both_ends(context, site):
    """Each verb in the graph has a reading from its source and from its target: no raw verb reaches a card."""
    verbs = {e["type"] for e in GRAPH["edges"]}
    covering, seen = [], set()
    for e in GRAPH["edges"]:  # a few terms whose cards, between them, show every verb from both ends
        if e["type"] not in seen:
            covering += [e["source"], e["target"]]
            seen.add(e["type"])
    assert seen == verbs
    page = _open(context, site)
    for term in dict.fromkeys(covering):
        page.evaluate("(h) => { location.hash = h; }", term.replace("_", "-"))
        page.wait_for_function("(t) => window.location.hash.slice(1) === t.replaceAll('_', '-')", arg=term)
        headings = page.locator("#links .map-rel h3").all_text_contents()
        raw = [h for h in headings if "_" in h or "(from)" in h]
        assert not raw, f"{term}: {raw}"


@pytest.mark.parametrize("context_options", [PHONE])
def test_a_drag_that_starts_on_a_topic_pans_the_map(context, site):
    page = _open(context, site)
    before = page.evaluate(STATE)["pan"]
    x, y = _on_map(page, _rendered(page, BIGGEST["id"]))
    page.mouse.move(x, y)
    page.mouse.down()
    page.mouse.move(x + 50, y + 30, steps=6)
    page.mouse.up()
    after = page.evaluate(STATE)["pan"]
    assert abs(after["x"] - before["x"]) > 20, "the topic circles cover most of the map: a drag on one must pan it"


def test_term_names_can_be_switched_off_and_the_choice_is_remembered(context, site):
    page = _open(context, site)
    _go(page, "resolution")
    page.click(".map-names")
    _settled(page)
    assert page.evaluate(STATE)["labels"] == 1, "only the selected term keeps its name"
    page.reload()
    page.wait_for_function("() => window.conceptMap && document.querySelector('#card h2')")
    assert not page.is_checked("#labels")


def test_a_kind_of_term_can_be_hidden_and_the_choice_is_remembered(context, site):
    page = _open(context, site)
    _open_topic(page, BIGGEST)
    shown = page.evaluate(STATE)
    page.click('#legend button[data-fam="concept"]')
    _settled(page)
    fewer = page.evaluate(STATE)
    concepts = page.evaluate(
        "() => window.conceptMap.cy.nodes('.term').filter((n) => n.data('fam') === 'concept' && n.visible()).length"
    )
    assert concepts == 0
    assert fewer["links"] < shown["links"], "a hidden term's links go with it"
    _go(page, "confounding")
    assert page.evaluate("() => window.conceptMap.cy.getElementById('confounding').visible()"), "the selected stays"
    page.reload()
    page.wait_for_function("() => window.conceptMap && document.querySelector('#card h2')")
    assert page.get_attribute('#legend button[data-fam="concept"]', "aria-pressed") == "false"


def test_how_the_map_was_made_is_read_from_its_data(context, site):
    page = _open(context, site)
    page.click("#howlink")
    how = page.locator("#how")
    text = how.text_content()
    assert "undefined" not in text
    assert "NaN" not in text
    told = page.evaluate(
        "() => [...document.querySelectorAll('#how > p, #how > ul')].map((e) => e.textContent).join(' ')"
    )
    assert not re.search(r"\d", told), "an overview in words: the exact numbers wait behind 'Show the numbers'"
    for details in how.locator("details").all():
        details.locator("summary").click()
    numbers = how.text_content()
    assert f"{len(GRAPH['edges'])} links" in numbers
    assert how.locator(".map-kind svg").count() == 5, "each kind of term drawn in its legend shape"
    guide = how.locator("details.map-guide")
    assert guide.locator("summary").text_content() == "Build one on your own texts: a step-by-step guide"
    assert guide.locator("ol > li").count() == 7
    folder = guide.locator("a", has_text="knowledge-graph folder").get_attribute("href")
    assert folder == "https://github.com/kgdunn/pid-book/tree/main/knowledge-graph", "the newest code to copy"
    assert folder.endswith("/knowledge-graph")
    assert (
        guide.locator("a", has_text="README").get_attribute("href") == f"{folder.replace('/tree/', '/blob/')}/README.md"
    )
    assert f"{GRAPH['method']['single_median']:.2f}" in numbers
    for topic in (c for c in GRAPH["clusters"] if c.get("moved")):
        assert f"{topic['agreement']:.2f}" in numbers


def test_the_catalog_and_the_start_page_lead_to_it(context, site):
    page = context.new_page()
    page.goto(f"{site}catalog/")
    page.click("text=How the ideas link")
    page.wait_for_url("**/map/")
    page.wait_for_function("() => window.conceptMap && document.querySelector('#card h2')")
    page.click("text=Back to Factorial")
    page.wait_for_selector("footer a[href='map/']")
    page.click("footer a[href='map/']")
    page.wait_for_url("**/map/")


@pytest.mark.parametrize("context_options", [PHONE, {"viewport": {"width": 1280, "height": 900}}])
def test_the_appearance_switches_sit_by_the_search_and_zoom_stays_below_the_map(context, site):
    page = _open(context, site)
    search, names, verbs = (page.locator(s).bounding_box() for s in (".map-search input", ".map-names", ".map-verbs"))
    assert names["y"] == pytest.approx(verbs["y"], abs=2), "the two switches share a row"
    assert names["x"] + names["width"] <= verbs["x"], "term names, then link labels"
    if page.viewport_size["width"] > 600:
        assert names["x"] >= search["x"] + search["width"], "in the space to the right of the search"
        assert names["y"] < search["y"] + search["height"], "on the search's row"
    else:
        assert names["y"] >= search["y"] + search["height"], "on a phone, on the row below the search"
    assert page.is_checked("#labels"), "term names start on"
    assert not page.is_checked("#verbs"), "link labels start off"
    assert page.locator(".map-controls button").evaluate_all("(bs) => bs.map((b) => b.id)") == ["zin", "zout", "home"]


@pytest.mark.parametrize("context_options", [PHONE, {"viewport": {"width": 1280, "height": 900}}])
def test_link_labels_read_each_lit_link_s_verb_and_are_remembered(context, site):
    page = _open(context, site)
    # Each label also as solid as the faded lines it sits on are not, and set apart from the term names: italic, accent.
    labelled = """() => { const cy = window.conceptMap.cy, ink = cy.edges('.link.hl')[0].style('line-color');
      return cy.edges('.link').filter((e) => e.visible() && e.style('label'))
        .map((e) => [e.hasClass('hl'), e.style('label'), e.data('verb'), e.style('target-arrow-shape'),
                     e.style('opacity') === '1' && e.style('color') === ink
                       && e.style('font-style') === 'italic']); }"""
    _go(page, "full_factorial")
    assert page.evaluate(labelled) == []
    page.click(".map-verbs")
    _settled(page)
    shown = page.evaluate(labelled)
    lit, held = page.evaluate(
        "() => { const lit = window.conceptMap.cy.edges('.link.hl').filter((e) => e.visible()); "
        "return [lit.length, lit.filter((e) => e.hasClass('mute')).length]; }"
    )
    assert shown, "the selected term's links are labelled"
    assert len(shown) + held == lit, "each drawn link is labelled, or held back where its label would land on a term"
    assert all(hl and label == verb and verb and arrow == "triangle" for hl, label, verb, arrow, _ in shown)
    assert all(solid for *_, solid in shown), "the labels are solid, italic and in the accent colour"
    page.click("#zin")
    _settled(page)
    assert len(page.evaluate(labelled)) >= len(shown), "zooming in makes room for more of them"
    page.reload()
    page.wait_for_function("() => window.conceptMap && document.querySelector('#card h2')")
    assert page.is_checked("#verbs"), "the choice is remembered"


# A lit link's line width and arrowhead length on screen, beside a term's dot and an ordinary link: Cytoscape draws
# the triangle 0.3 x max((13.37 x width)^0.9, 29) x arrow-scale drawing units long.
ARROW = """() => { const cy = window.conceptMap.cy, z = cy.zoom();
  const lit = cy.edges('.link.hl')[0], plain = cy.edges('.link').not('.hl')[0], dot = cy.nodes('.term.nb')[0];
  const width = parseFloat(lit.style('width'));
  const head = 0.3 * Math.max(Math.pow(13.37 * width, 0.9), 29) * parseFloat(lit.style('arrow-scale')) * z;
  return { head, dot: dot.renderedWidth(), line: width * z, plain: parseFloat(plain.style('width')) * z }; }"""


@pytest.mark.parametrize("context_options", [PHONE, {"viewport": {"width": 1280, "height": 900}}])
def test_a_lit_link_s_arrowhead_matches_a_term_and_its_line_is_not_bold(context, site):
    page = _open(context, site)
    _go(page, "full_factorial")
    for deepest in (False, True):
        if deepest:
            page.evaluate("() => { const cy = window.conceptMap.cy; cy.zoom(cy.maxZoom()); }")
            _settled(page)
        drawn = page.evaluate(ARROW)
        assert drawn["head"] == pytest.approx(drawn["dot"], rel=0.05), f"arrowhead against a term's dot: {drawn}"
        assert drawn["line"] <= 1.3 * drawn["plain"], f"a lit line against an ordinary one: {drawn}"


@pytest.mark.parametrize("context_options", [PHONE, {"viewport": {"width": 1280, "height": 900}}])
def test_the_deepest_zoom_sets_the_closest_terms_far_apart(context, site):
    page = _open(context, site)
    for _ in range(12):
        page.click("#zin")
        _settled(page)
    zoom, deepest = page.evaluate("() => [window.conceptMap.cy.zoom(), window.conceptMap.cy.maxZoom()]")
    assert zoom == pytest.approx(deepest), "the + button reaches the limit"
    assert TERM_GAP * deepest >= 360, f"the closest terms are {TERM_GAP * deepest:.0f} px apart at the deepest zoom"


@pytest.mark.parametrize("context_options", [PHONE])
def test_it_reads_on_a_phone_without_scrolling_sideways(context, site):
    page = _open(context, site)
    page.click("#howlink")
    for details in page.locator("#how details").all():
        details.locator("summary").click()
    wider = page.evaluate(
        """() => [...document.querySelectorAll('body *')]
             .filter((e) => e.getBoundingClientRect().right > document.documentElement.clientWidth + 1)
             .filter((e) => !e.closest('.map-tablewrap'))
             .map((e) => e.tagName.toLowerCase() + (e.className ? '.' + e.className : ''))"""
    )
    assert wider == []
    lines = page.evaluate(
        # In DejaVu Sans, wider than any phone's own font, so two lines here are two lines on any phone.
        """() => { const lede = document.querySelector('.lede'); lede.style.fontFamily = '"DejaVu Sans", sans-serif';
                   const r = document.createRange(); r.selectNodeContents(lede);
                   return new Set([...r.getClientRects()].map((b) => Math.round(b.top))).size; }"""
    )
    assert lines <= 2, f"the lede takes {lines} lines on a phone"


@pytest.mark.parametrize("context_options", [{**PHONE, "has_touch": True}])
def _drag(page: Any, finger: Any, rise: float, fingers: int = 1) -> None:  # noqa: ANN401 - Playwright objects
    """Drag one finger, or two side by side, up the map by ``rise`` pixels, from low in the map, as a reader does."""
    box = page.locator("#cy").bounding_box()
    xs = [box["x"] + box["width"] * (k + 1) / (fingers + 1) for k in range(fingers)]
    y = box["y"] + 0.8 * box["height"]
    touch = lambda kind, dy: finger.send(  # noqa: E731 - one call, three times
        "Input.dispatchTouchEvent",
        {"type": kind, "touchPoints": [{"x": x, "y": y - dy, "id": k} for k, x in enumerate(xs)]},
    )
    touch("touchStart", 0)
    for step in range(1, int(rise / 15) + 1):
        touch("touchMove", 15 * step)
    finger.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})
    page.wait_for_timeout(600)  # a scroll the browser makes from a drag goes on a moment after the finger lifts
    _settled(page)


@pytest.mark.parametrize("context_options", [{**PHONE, "has_touch": True}])
def test_on_a_touch_screen_one_finger_scrolls_the_page_and_two_move_the_map(context, site):
    """On a phone the map fills the screen: a finger that only moved the map would leave the reader stuck in it."""
    page = _open(context, site)
    finger = context.new_cdp_session(page)
    view = "() => ({ scrolled: scrollY, pan: window.conceptMap.cy.pan(), zoom: window.conceptMap.cy.zoom() })"
    before = page.evaluate(view)
    _drag(page, finger, 300)
    after = page.evaluate(view)
    assert after["scrolled"] > before["scrolled"] + 100, "one finger scrolls the page, by the browser's own scrolling"
    assert (after["pan"], after["zoom"]) == (before["pan"], before["zoom"]), "and leaves the map where it was"
    assert page.text_content("#card h2") == f"{len(GRAPH['clusters'])} topics", "with no tap where the finger lifted"
    assert page.locator(".map-hint").text_content() == "Two fingers move the map"

    page.set_viewport_size({"width": 390, "height": 900})  # the browser's toolbar hides as the page scrolls
    page.wait_for_timeout(400)
    _settled(page)
    kept = page.evaluate(view)
    assert (kept["pan"], kept["zoom"]) == (after["pan"], after["zoom"]), "a taller window keeps the reader's view"

    page.evaluate("() => window.scrollTo(0, 0)")
    _drag(page, finger, 150, fingers=2)
    moved = page.evaluate(view)
    assert moved["pan"]["y"] < kept["pan"]["y"] - 50, "two fingers move the map"
    assert moved["scrolled"] == 0, "and leave the page where it was"

    box = page.locator("#cy").bounding_box()
    x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    for kind, apart in (("touchStart", 30), *(("touchMove", 30 + 10 * k) for k in range(1, 9))):
        finger.send(
            "Input.dispatchTouchEvent",
            {"type": kind, "touchPoints": [{"x": x - apart, "y": y, "id": 0}, {"x": x + apart, "y": y, "id": 1}]},
        )
    finger.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})
    _settled(page)
    assert page.evaluate(view)["zoom"] > moved["zoom"] * 1.5, "and spreading them zooms in"


def test_the_colour_mode_switches_here_and_is_kept_with_the_app_s(context, site):
    """The map has the start page's switch: light, dark or the system's, in turn, kept for every page of the app."""
    page = _open(context, site)
    theme = "() => document.documentElement.dataset.theme ?? 'system'"
    seen = []
    for _ in range(3):
        page.click("#mode")
        seen.append(page.evaluate(theme))
    assert sorted(seen) == ["dark", "light", "system"], "each press moves to the next mode"
    page.click("#mode")
    chosen = page.evaluate(theme)
    page.goto(site)
    page.wait_for_function("() => document.documentElement.dataset.mode")
    assert page.evaluate(theme) == chosen, "the start page opens in the mode chosen on the map"
