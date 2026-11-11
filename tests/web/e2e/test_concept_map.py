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


def _open(context: Any, site: str, level: int | None = 3) -> Any:  # noqa: ANN401 - Playwright objects
    """Open the map, at the whole graph unless ``level`` says otherwise; ``None`` opens it as a first visit does."""
    page = context.new_page()
    if level:  # as a reader who chose it before; a level chosen in the test still wins on a reload
        store = "localStorage.getItem('factorial-map-level') || localStorage.setItem('factorial-map-level', '%s')"
        page.add_init_script(f"try {{ {store % level}; }} catch {{}}")
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
    drawn = set(_drawn(3, {n["id"] for n in GRAPH["nodes"] if n["type"] in CONTEXT}))
    size = sum(1 for n in GRAPH["nodes"] if n["cluster"] == BIGGEST["id"] and n["id"] in drawn)
    assert size < BIGGEST["size"], "the hidden examples are not counted"
    biggest = page.evaluate("(i) => window.conceptMap.cy.getElementById(i).data('olabel')", BIGGEST["id"])
    assert biggest == f"{BIGGEST['short']}\n[{size}]", "a topic's name, and its number of terms drawn below it"
    sub = BIGGEST["subs"][0]
    size = sum(1 for n in GRAPH["nodes"] if n["sub"] == sub["id"] and n["id"] in drawn)
    named = page.evaluate("(i) => window.conceptMap.cy.getElementById(i).data('label')", f"{sub['id']}-name")
    assert named == f"{sub['name']}\n[{size}]", "and a sub-topic's"
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
    howto = "https://github.com/kgdunn/pid-book/blob/main/knowledge-graph/HOWTO.md"
    assert page.get_attribute("#howto", "href") == howto, "above the map: one tap to the steps for a reader's own texts"
    assert how.locator("a", has_text="The how-to").get_attribute("href") == howto
    assert not how.locator("ol").count(), "the how-to holds the steps, so the page does not repeat them"
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


SMALL_PHONE = {"viewport": {"width": 360, "height": 780}, "is_mobile": True}


@pytest.mark.parametrize("context_options", [SMALL_PHONE, PHONE, {"viewport": {"width": 1280, "height": 900}}])
def test_above_the_map_one_row_of_switches_and_one_row_of_kinds(context, site):
    page = _open(context, site)
    page.locator("#level").fill("2")  # Intermediate, the longest name
    # The kinds in DejaVu Sans, wider than any phone's own font: one line here is one line on any phone.
    page.evaluate("""() => { document.querySelector('#legend').style.fontFamily = '"DejaVu Sans", sans-serif'; }""")
    rows = page.evaluate("""() => {
      const top = (s) => [...document.querySelectorAll(s)].map((e) => Math.round(e.getBoundingClientRect().top));
      const box = (s) => document.querySelector(s).getBoundingClientRect();
      return { switches: top('.map-view > label'), kinds: top('#legend button'), search: box('.map-search input'),
        legend: box('#legend'), view: box('.map-view'), map: box('#cy'), zoom: box('.map-controls') }; }""")
    assert len(set(rows["switches"])) == 1, "term names, link labels and the level on one line"
    assert len(set(rows["kinds"])) == 1, "the kinds of term on one line"
    assert rows["view"]["bottom"] <= rows["legend"]["top"], "the switches, then the kinds"
    assert rows["legend"]["bottom"] <= rows["map"]["top"], "both above the map"
    assert rows["zoom"]["top"] >= rows["map"]["bottom"], "zoom and fit stay below the map"
    if page.viewport_size["width"] > 600:
        assert rows["view"]["left"] >= rows["search"]["right"], "beside the search"
    else:
        assert rows["view"]["top"] >= rows["search"]["bottom"], "on a phone, on the row below the search"
    assert page.is_checked("#labels"), "term names start on"
    assert not page.is_checked("#verbs"), "link labels start off"
    assert page.locator(".map-controls button").evaluate_all("(bs) => bs.map((b) => b.id)") == ["zin", "zout", "home"]


@pytest.mark.parametrize("context_options", [PHONE, {"viewport": {"width": 1280, "height": 900}}])
def test_link_labels_read_each_lit_link_s_verb_and_are_remembered(context, site):
    page = _open(context, site)
    # Each label also as solid as the faded lines it sits on are not, and set apart from the term names: italic, accent.
    labelled = """() => { const cy = window.conceptMap.cy, ink = cy.edges('.link.hl')[0].style('line-color');
      return cy.edges('.link').filter((e) => e.visible() && e.style('source-label'))
        .map((e) => [e.hasClass('hl'), e.style('source-label'), e.data('verb'), e.style('target-arrow-shape'),
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
def _map_in_view(page: Any) -> None:  # noqa: ANN401 - a Playwright Page
    """Scroll the whole map onto the screen, as a reader does before touching it."""
    page.evaluate("() => document.querySelector('.map-box').scrollIntoView({ block: 'center' })")
    _settled(page)


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
    _map_in_view(page)  # the filters above it push its lower part below a phone's first screen
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

    _map_in_view(page)
    still = page.evaluate("() => scrollY")
    _drag(page, finger, 150, fingers=2)
    moved = page.evaluate(view)
    assert moved["pan"]["y"] < kept["pan"]["y"] - 50, "two fingers move the map"
    assert moved["scrolled"] == still, "and leave the page where it was"

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


# Each drawn link's label: where along its line it sits (0 at the source term, 1 at the target), or None if held back.
SPOTS = """() => { const cy = window.conceptMap.cy, z = cy.zoom();
  return cy.edges('.link').filter((e) => e.visible() && !e.hasClass('dim')).map((e) => {
    if (e.hasClass('mute')) return null;
    const s = e.source().renderedPosition(), t = e.target().renderedPosition();
    return (e.data('at') * z + e.source().renderedWidth() / 2) / Math.hypot(t.x - s.x, t.y - s.y); }); }"""
# The longest of the selected term's drawn links with its middle in view: its sentence, and that middle on the page.
LONGEST = """() => { const cy = window.conceptMap.cy, box = cy.container().getBoundingClientRect();
  const name = (n) => n.data('label');
  const mid = (x) => { const s = x.source().renderedPosition(), t = x.target().renderedPosition();
    return { x: (s.x + t.x) / 2, y: (s.y + t.y) / 2, len: Math.hypot(t.x - s.x, t.y - s.y) }; };
  const inView = (m) => m.x > 20 && m.x < cy.width() - 20 && m.y > 20 && m.y < cy.height() - 20;
  const e = cy.edges('.link.hl').filter((x) => x.visible() && inView(mid(x))).max((x) => mid(x).len).ele;
  const m = e.renderedMidpoint();   // on the curve: a link bows, so the straight line's middle is off it
  const said = `${name(e.source())} ${e.data('verb')} ${name(e.target())}`;
  return { said, x: box.left + m.x, y: box.top + m.y }; }"""


@pytest.mark.parametrize("context_options", [PHONE, {"viewport": {"width": 1280, "height": 900}}])
def test_a_link_label_with_its_middle_taken_slides_along_its_line(context, site):
    page = _open(context, site)
    _go(page, "full_factorial")
    page.click(".map-verbs")
    page.evaluate(
        "() => { const cy = window.conceptMap.cy; "
        "cy.zoom({ level: cy.zoom() * 0.7, renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 } }); }"
    )
    _settled(page)
    spots = [k for k in page.evaluate(SPOTS) if k is not None]
    assert spots, "some links are labelled"
    assert all(0.15 < k < 0.85 for k in spots), f"each label sits on its line, clear of both terms: {spots}"
    assert any(abs(k - 0.5) > 0.05 for k in spots), "a label whose middle is taken moves along its line"


@pytest.mark.parametrize("context_options", [{"viewport": {"width": 1280, "height": 900}}])
def test_the_mouse_on_a_link_reads_it_as_a_sentence_below_the_map(context, site):
    page = _open(context, site)
    _go(page, "full_factorial")
    told = page.text_content("#striptext")
    link = page.evaluate(LONGEST)
    page.mouse.move(link["x"], link["y"])
    page.wait_for_function("(said) => document.querySelector('#striptext').textContent === said", arg=link["said"])
    assert page.locator("#striptext i").text_content() in link["said"], "the verb set apart, as on the map"
    box = page.locator("#cy").bounding_box()
    page.mouse.move(box["x"] + 4, box["y"] + 4)
    page.wait_for_function("(told) => document.querySelector('#striptext').textContent === told", arg=told)


@pytest.mark.parametrize("context_options", [{**PHONE, "has_touch": True}])
def test_a_tap_on_a_link_of_the_selected_term_reads_it_and_keeps_the_term(context, site):
    page = _open(context, site)
    _go(page, "full_factorial")
    _map_in_view(page)
    link = page.evaluate(LONGEST)
    finger = context.new_cdp_session(page)
    point = [{"x": link["x"], "y": link["y"], "id": 0}]
    finger.send("Input.dispatchTouchEvent", {"type": "touchStart", "touchPoints": point})
    finger.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})
    page.wait_for_function("(said) => document.querySelector('#striptext').textContent === said", arg=link["said"])
    assert page.text_content("#card h2") == "Full factorial", "the selected term stays selected"


# Where a term sits in the drawing, and the map's view; and a neighbour of the selected term well inside the map.
WHERE = """(i) => { const cy = window.conceptMap.cy;
  return { at: { ...cy.getElementById(i).position() }, pan: { ...cy.pan() }, zoom: cy.zoom(), scrolled: scrollY }; }"""
NEIGHBOUR = """() => { const cy = window.conceptMap.cy;
  const inside = (p) => p.x > 60 && p.x < cy.width() - 60 && p.y > 60 && p.y < cy.height() - 60;
  return cy.nodes('.term.nb').filter((n) => n.visible() && inside(n.renderedPosition()))[0].id(); }"""


@pytest.mark.parametrize("context_options", [{"viewport": {"width": 1280, "height": 900}}])
def test_a_term_dragged_with_the_mouse_moves_and_the_map_stays(context, site):
    """To uncover a link's label that a term sits on, the reader moves the term."""
    page = _open(context, site)
    _go(page, "full_factorial")
    term = page.evaluate(NEIGHBOUR)
    before = page.evaluate(WHERE, term)
    x, y = _on_map(page, _rendered(page, term))
    page.mouse.move(x, y)
    page.mouse.down()
    page.mouse.move(x + 60, y + 40, steps=8)
    page.mouse.up()
    _settled(page)
    after = page.evaluate(WHERE, term)
    assert after["at"]["x"] - before["at"]["x"] == pytest.approx(60 / before["zoom"], rel=0.1)
    assert after["at"]["y"] - before["at"]["y"] == pytest.approx(40 / before["zoom"], rel=0.1)
    assert (after["pan"], after["zoom"]) == (before["pan"], before["zoom"]), "the map itself stays where it was"
    assert page.text_content("#card h2") == "Full factorial", "a drag is not a tap"


@pytest.mark.parametrize("context_options", [{**PHONE, "has_touch": True}])
def test_on_a_touch_screen_a_term_held_then_dragged_moves_and_a_quick_swipe_still_scrolls(context, site):
    page = _open(context, site)
    _go(page, "full_factorial")
    finger = context.new_cdp_session(page)
    term = page.evaluate(NEIGHBOUR)

    def touch(kind: str, dx: float, dy: float) -> None:
        x, y = start
        finger.send("Input.dispatchTouchEvent", {"type": kind, "touchPoints": [{"x": x + dx, "y": y + dy, "id": 0}]})

    def lift() -> None:
        finger.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})
        page.wait_for_timeout(600)  # a scroll the browser makes from a drag goes on a moment after the finger lifts
        _settled(page)

    before = page.evaluate(WHERE, term)
    start = _on_map(page, _rendered(page, term))
    touch("touchStart", 0, 0)
    for k in range(1, 11):
        touch("touchMove", 0, -15 * k)
    lift()
    swiped = page.evaluate(WHERE, term)
    assert swiped["scrolled"] > before["scrolled"] + 50, "a finger that moves at once scrolls the page"
    assert swiped["at"] == before["at"], "and leaves the term where it was"

    page.evaluate("() => window.scrollTo(0, 0)")
    start = _on_map(page, _rendered(page, term))
    touch("touchStart", 0, 0)
    page.wait_for_timeout(700)
    for k in range(1, 7):
        touch("touchMove", 10 * k, 0)
    assert page.locator(".map-hint").text_content() == "Drag to move this term"
    lift()
    held = page.evaluate(WHERE, term)
    assert held["at"]["x"] - before["at"]["x"] == pytest.approx(60 / before["zoom"], rel=0.15), "held, it follows"
    assert held["scrolled"] == 0, "and the page stays where it is"
    assert page.text_content("#card h2") == "Full factorial", "a move is not a tap"


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


# The terms a reader sees at a reading level: every term at that level or below, of the kinds not hidden, that keeps
# a link to another such term.
SHOWN = """() => window.conceptMap.cy.nodes('.term').filter((n) => !n.hasClass('off')).map((n) => n.id()).sort()"""
CONTEXT = {"Example", "Question", "Reference"}  # hidden by default, as the first test checks
LEVELS = ["Beginner", "Intermediate", "Advanced"]


def _drawn(level: int, hidden: set[str]) -> list[str]:
    """List the terms drawn at ``level`` with the terms in ``hidden`` hidden by their kind, nothing selected."""
    kept = {n["id"] for n in GRAPH["nodes"] if n["level"] <= level and n["id"] not in hidden}
    ends = [(e["source"], e["target"]) for e in GRAPH["edges"]]
    linked = {end for pair in ends if set(pair) <= kept for end in pair}
    return sorted(kept & linked)


# What the reader must never see: a drawn term with no drawn link, or a topic or sub-topic whose count is not the
# number of terms it draws.
FAULTS = """() => {
  const cy = window.conceptMap.cy, on = cy.nodes('.term').filter((n) => !n.hasClass('off'));
  const drawn = (e) => !e.source().hasClass('off') && !e.target().hasClass('off');
  const alone = on.filter((n) => n.connectedEdges('.link').filter(drawn).empty()).map((n) => n.id());
  const count = (id) => on.filter((n) => n.data('topic') === id || n.data('sub') === id).length;
  const off = (key, id) => (n) => !n.data(key).endsWith(`[${count(id(n))}]`);
  const miscounted = [
    ...cy.nodes('.topic').filter(off('olabel', (n) => n.id())).map((n) => n.id()),
    ...cy.nodes('.subname').filter(off('label', (n) => n.id().slice(0, -5))).map((n) => n.id()),
  ];
  return { alone, miscounted, drawn: on.length };
}"""


@pytest.mark.parametrize("context_options", [PHONE, {"viewport": {"width": 1280, "height": 900}}])
def test_the_level_slider_shows_each_level_with_every_level_below_it(context, site):
    page = _open(context, site, level=None)
    assert page.input_value("#level") == "1", "a first visit opens on Beginner"
    for level, name in reversed(list(enumerate(LEVELS, 1))):
        page.locator("#level").fill(str(level))
        assert page.text_content("#level-name") == name
        expected = _drawn(level, {n["id"] for n in GRAPH["nodes"] if n["type"] in CONTEXT})
        assert page.evaluate(SHOWN) == expected, f"{name}: exactly the terms at that level or below"
        size = sum(1 for n in GRAPH["nodes"] if n["cluster"] == BIGGEST["id"] and n["id"] in expected)
        label = page.evaluate("(i) => window.conceptMap.cy.getElementById(i).data('olabel')", BIGGEST["id"])
        assert label == f"{BIGGEST['short']}\n[{size}]", "a topic counts the terms shown at the level"
    sizes = [len([n for n in GRAPH["nodes"] if n["level"] <= level]) for level in (1, 2, 3)]
    assert sizes[0] < sizes[1] < sizes[2] == len(GRAPH["nodes"]), "each level adds terms to the one below"


def test_the_level_is_remembered_and_search_stays_within_it(context, site):
    page = _open(context, site, level=None)
    page.locator("#level").fill("1")
    advanced = next(n for n in GRAPH["nodes"] if n["level"] == 3 and n["type"] not in CONTEXT)
    page.fill("#q", advanced["name"])
    offered = "(i) => document.querySelectorAll(`#results button[data-id='${i}']`).length"
    assert page.evaluate(offered, advanced["id"]) == 0, "a term above the level is not offered"
    beginner = next(n for n in GRAPH["nodes"] if n["level"] == 1 and n["type"] not in CONTEXT)
    page.fill("#q", beginner["name"])
    assert page.evaluate(offered, beginner["id"]) == 1
    page.reload()
    page.wait_for_function("() => window.conceptMap && document.querySelector('#card h2')")
    assert page.input_value("#level") == "1"
    assert page.text_content("#level-name") == "Beginner"


def test_a_term_is_found_under_a_name_another_text_uses_and_its_card_does_not_show_that_name(context, site):
    page = _open(context, site)
    term = next(n for n in GRAPH["nodes"] if n.get("found_as") and n["type"] not in CONTEXT)
    hidden = term["found_as"][0]
    shown = " ".join([term["name"], *term["aliases"], term["definition"]]).lower()
    assert hidden.lower() not in shown, "only found_as holds it"
    page.fill("#q", hidden)
    page.click(f"#results button[data-id='{term['id']}']")
    page.wait_for_function("(name) => document.querySelector('#card h2')?.textContent === name", arg=term["name"])
    assert hidden.lower() not in page.text_content("#card").lower(), "searched and matched, never shown"


@pytest.mark.parametrize(
    "context_options", [{"viewport": {"width": 320, "height": 700}, "is_mobile": True}, SMALL_PHONE, PHONE]
)
def test_on_a_phone_the_level_fits_inside_the_page(context, site):
    page = _open(context, site, level=None)
    page.locator("#level").fill("2")  # Intermediate, the longest name
    row = page.evaluate("""() => ({
      right: document.querySelector('#level-name').getBoundingClientRect().right,
      width: document.documentElement.clientWidth, scroll: document.documentElement.scrollWidth })""")
    assert row["right"] <= row["width"] - 16 + 0.5, "the level's name ends inside the page's side margin"
    assert row["scroll"] == row["width"], "and the page does not scroll sideways"


# Kinds of term to hide, by their legend key: the default, the one that drew lone terms and empty sub-topics on
# 2026-11-10, and all but one kind.
HIDDEN_KINDS = [{"context"}, {"method", "concept", "context"}, {"design", "method", "concept", "context"}]


@pytest.mark.parametrize("kinds", HIDDEN_KINDS, ids=lambda k: "+".join(sorted(k)))
def test_no_term_is_drawn_alone_and_every_count_is_what_is_drawn(context, site, kinds):
    page = _open(context, site)
    for key in ("design", "method", "concept", "guide", "context"):
        button = f'#legend button[data-fam="{key}"]'
        if (page.get_attribute(button, "aria-pressed") == "true") == (key in kinds):
            page.click(button)
    kind_of = page.evaluate(
        "() => Object.fromEntries(window.conceptMap.cy.nodes('.term').map((n) => [n.id(), n.data('fam')]))"
    )
    hidden = {i for i, k in kind_of.items() if k in kinds}
    for level in (3, 2, 1):
        page.locator("#level").fill(str(level))
        faults = page.evaluate(FAULTS)
        assert faults["alone"] == [], f"level {level}: every drawn term keeps a drawn link"
        assert faults["miscounted"] == [], f"level {level}: each topic counts the terms it draws"
        assert page.evaluate(SHOWN) == _drawn(level, hidden)
    _go(page, "confounding")  # a concept: drawn while selected, with its count, even when concepts are hidden
    assert page.evaluate("() => window.conceptMap.cy.getElementById('confounding').visible()")
    assert page.evaluate(FAULTS)["miscounted"] == []


def test_a_topic_with_no_term_at_the_level_is_dimmed(context, site):
    page = _open(context, site, level=1)
    empty = page.evaluate("() => window.conceptMap.cy.nodes('.topic.empty').map((n) => n.id()).sort()")
    drawn = set(_drawn(1, {n["id"] for n in GRAPH["nodes"] if n["type"] in CONTEXT}))
    expected = sorted(
        c["id"]
        for c in GRAPH["clusters"]
        if not any(n["cluster"] == c["id"] for n in GRAPH["nodes"] if n["id"] in drawn)
    )
    assert empty == expected, "exactly the topics with no term drawn"
    assert empty, "the beginner level leaves at least one topic empty, or this test checks nothing"
    page.locator("#level").fill("3")
    assert page.evaluate("() => window.conceptMap.cy.nodes('.topic.empty').length") == 0, (
        "every topic has terms at the top"
    )


@pytest.mark.parametrize("context_options", [{"viewport": {"width": 1860, "height": 1000}}])
def test_on_a_wide_screen_the_map_takes_the_width_and_a_tap_brings_it_up(context, site):
    page = _open(context, site)
    widths = page.evaluate("""() => ['.map-box', '#card', '.map-find'].map((s) =>
      Math.round(document.querySelector(s).getBoundingClientRect().width))""")
    assert widths[0] == 1600, "the map stretches to its 1600 px"
    assert widths[1] == widths[2] == 980, "the text keeps its reading width"
    assert page.evaluate("scrollY") == 0
    page.mouse.click(*_on_map(page, _rendered(page, BIGGEST["id"])))
    page.wait_for_function("(name) => document.querySelector('#card h2').textContent === name", arg=BIGGEST["name"])
    page.wait_for_function("() => Math.abs(document.querySelector('.map-find').getBoundingClientRect().top - 8) < 2")
    card = page.evaluate("() => document.querySelector('#card').getBoundingClientRect().top")
    assert card < 1000, "the card the tap opened is on the screen"


def test_every_link_bows_a_little(context, site):
    page = _open(context, site)
    _go(page, "full_factorial")
    bows = page.evaluate("""() => window.conceptMap.cy.edges('.link.hl').filter((e) => e.visible()).map((e) => {
      const s = e.source().position(), t = e.target().position(), m = e.midpoint();
      const len = Math.hypot(t.x - s.x, t.y - s.y);
      return Math.hypot(m.x - (s.x + t.x) / 2, m.y - (s.y + t.y) / 2) / len; })""")
    assert bows, "the selected term has links"
    assert all(0.03 < b < 0.12 for b in bows), f"a mild arc, a few percent of the link's length: {bows}"


# The topics as drawn: how much of the map's width they fill, any two circles that meet, and how far each has moved
# from where the drawing puts it.
SPREAD = """(graph) => { const cy = window.conceptMap.cy, t = cy.nodes('.topic');
  const meet = [];
  t.forEach((a, i) => t.forEach((b, j) => { if (j <= i) return; const pa = a.position(), pb = b.position();
    if (Math.hypot(pa.x - pb.x, pa.y - pb.y) < (a.width() + b.width()) / 2 - 1) meet.push(a.id() + '+' + b.id()); }));
  const moved = Math.max(...graph.map((c) => Math.abs(cy.getElementById(c.id).position().x - c.x)));
  return { fill: t.renderedBoundingBox().w / cy.width(), meet, moved }; }"""


@pytest.mark.parametrize("context_options", [{"viewport": {"width": 1846, "height": 1000}}])
def test_on_a_wide_screen_the_topics_spread_across_it_without_meeting(context, site):
    page = _open(context, site)
    drawn = page.evaluate(SPREAD, GRAPH["clusters"])
    assert drawn["fill"] > 0.75, "the topics fill most of the map's width"
    assert drawn["meet"] == [], "no two circles meet"
    _go(page, "full_factorial")
    term = page.evaluate("() => window.conceptMap.cy.getElementById('full_factorial').position()")
    topic = next(
        c
        for c in GRAPH["clusters"]
        if c["id"] == next(n for n in GRAPH["nodes"] if n["id"] == "full_factorial")["cluster"]
    )
    centre = page.evaluate("(i) => window.conceptMap.cy.getElementById(i).position()", topic["id"])
    node = next(n for n in GRAPH["nodes"] if n["id"] == "full_factorial")
    assert term["x"] - centre["x"] == pytest.approx(node["x"] - topic["x"]), "a term keeps its place in its topic"


@pytest.mark.parametrize("context_options", [PHONE])
def test_on_a_phone_held_upright_the_drawing_is_as_the_book_lays_it_out(context, site):
    page = _open(context, site)
    assert page.evaluate(SPREAD, GRAPH["clusters"])["moved"] == pytest.approx(0)


def test_a_term_s_book_link_goes_to_the_sentence_it_quotes(context, site):
    """The link is the page with the quote, its section when the quote is from there, and a text fragment of the
    quote's words that read the same on the page; a word with markup, or that Sphinx rewrites, is left out.
    """
    page = _open(context, site)
    url = lambda term: page.evaluate(  # noqa: E731 - one call, three terms
        "(t) => window.conceptMap.bookUrl({ book_file: 'a/b.rst', book_label: 'DOE_x', evidence: 'a/b.rst:3', ...t })",
        term,
    )
    plain = url({"quote": "because one or more columns are exactly identical, also known as"})
    assert plain == (
        "https://learnche.org/pid/a/b#doe-x:~:text=because%20one%20or%20more,identical%2C%20also%20known%20as"
    ), "a long quote: its first and last four words, the comma encoded"
    marked = url(
        {"quote": "we can run a :math:`2^{k-p}` fractional factorial with all the factors' levels -- in short"}
    )
    assert marked.endswith(":~:text=fractional%20factorial%20with%20all%20the"), (
        "the longest run free of markup and rewrites"
    )
    elsewhere = url({"quote": "Each row is one experimental run.", "evidence": "c/d.rst:147"})
    assert elsewhere == "https://learnche.org/pid/c/d#:~:text=Each%20row%20is%20one%20experimental%20run.", (
        "a quote from another page links to that page, without the section's anchor"
    )
    assert url({"quote": "a :math:`b`"}) == "https://learnche.org/pid/a/b#doe-x", "no fragment: the section alone"
