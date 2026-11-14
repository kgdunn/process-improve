"""The concept map (map/index.html): the ideas of designed experiments, and how the book links them.

Its graph, ``web/map/graph.json``, is built and checked against the book in pid-book's ``knowledge-graph/``. These
tests drive the page as a reader would, at a phone's width where that matters: what a tap opens, what zooming shows,
and what the page remembers. ``window.conceptMap`` gives them the map's ``cy`` and its data.
"""

from __future__ import annotations

import json
import math
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
    page.wait_for_function("() => !document.querySelector('.tex:not(.done)')")  # its formulas drawn, or given up on
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


# One link put in the middle of the map, its two terms `span` apart: a share of the map's shorter side, or, above 1, of
# its longer side, which puts both terms out of view.
VIEW_LINK = """([id, span]) => { const cy = window.conceptMap.cy, e = cy.getElementById(id);
  const s = e.source().position(), t = e.target().position(), side = span > 1 ? Math.max : Math.min;
  const zoom = (span * side(cy.width(), cy.height())) / Math.hypot(t.x - s.x, t.y - s.y);
  const middle = { x: (s.x + t.x) / 2, y: (s.y + t.y) / 2 };
  cy.viewport({ zoom, pan: { x: cy.width() / 2 - zoom * middle.x, y: cy.height() / 2 - zoom * middle.y } });
  return { zoom, least: cy.minZoom(), most: cy.maxZoom() }; }"""
# Whether that link is drawn, and what is wrong in the rest: links drawn with a term well out of view, and links not
# drawn though both their terms are well in view and neither is switched off. "Well" is by 20 px, more than the map's
# own margin. Of the selected term's links, which all stay: how many there are to draw, are drawn, and leave the view.
LINKS_IN_VIEW = """(id) => { const cy = window.conceptMap.cy, W = cy.width(), H = cy.height();
  const past = (n, by) => { const p = n.renderedPosition();
    return p.x < -by || p.x > W + by || p.y < -by || p.y > H + by; };
  const ends = (e) => [e.source(), e.target()], on = (e) => !ends(e).some((n) => n.hasClass('off'));
  const links = cy.edges('.link').filter((e) => !e.hasClass('hl') && on(e)), lit = cy.edges('.link.hl').filter(on);
  return { drawn: cy.getElementById(id).visible(), lit: lit.length, lit_drawn: lit.filter((e) => e.visible()).length,
    lit_leaving: lit.filter((e) => ends(e).some((n) => past(n, 20))).length,
    astray: links.filter((e) => e.visible() && ends(e).some((n) => past(n, 20))).length,
    held: links.filter((e) => !e.visible() && !ends(e).some((n) => past(n, -20))).length }; }"""


def test_a_link_is_drawn_only_while_both_its_terms_are_in_view(context, site):
    page = _open(context, site)
    link = page.evaluate("""() => { const cy = window.conceptMap.cy, far = (e) => Math.hypot(
        e.target().position('x') - e.source().position('x'), e.target().position('y') - e.source().position('y'));
      return cy.edges('.link').sort((a, b) => Math.abs(far(a) - 350) - Math.abs(far(b) - 350))[0].id(); }""")

    def look(span: float) -> dict[str, Any]:
        view = page.evaluate(VIEW_LINK, [link, span])
        assert view["least"] < view["zoom"] < view["most"], "the map can be zoomed this far"
        assert view["zoom"] * TERM_GAP > DETAIL_GAP, "and its terms and links are drawn at this zoom"
        _settled(page)
        return page.evaluate(LINKS_IN_VIEW, link)

    both = look(0.4)
    assert both["drawn"], "both its terms are in view"
    assert (both["astray"], both["held"]) == (0, 0)
    through = look(1.2)
    assert not through["drawn"], "zoomed in between its terms, it only crosses the view, so it is gone"
    assert (through["astray"], through["held"]) == (0, 0), "and so is every other link that leaves the view"
    assert look(0.4)["drawn"], "zooming out brings it back"
    # A selected term keeps its links, wherever they lead.
    term = next(e for i, e in enumerate(GRAPH["edges"]) if f"l{i}" == link)["source"]
    _go(page, term)
    lit = look(1.2)
    assert lit["lit"] == lit["lit_drawn"] > 0
    assert lit["lit_leaving"], "one of them leaves the view here"


# A part of the card as the graph writes it: each formula drawn there put back as its LaTeX, in the book's own role.
AS_WRITTEN = """(selector) => { const copy = document.querySelector(selector).cloneNode(true);
  const latex = (f) => f.querySelector('annotation').textContent;
  for (const f of copy.querySelectorAll('.tex')) f.replaceWith(':math:`' + latex(f) + '`');
  return copy.textContent; }"""


def test_a_tap_on_a_term_shows_its_definition_its_book_line_and_lights_its_links(context, site):
    page = _open(context, site)
    _go(page, "full_factorial")
    page.mouse.click(*_on_map(page, _rendered(page, "full_factorial")))
    _settled(page)
    term = next(n for n in GRAPH["nodes"] if n["id"] == "full_factorial")
    assert page.text_content("#card h2") == term["name"]
    assert page.evaluate(AS_WRITTEN, "#card .map-quote") == term["quote"]
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
    assert not page.locator("#howlink, #howto").count(), "said once, below the map: no links to it above the map"
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
LONG_LEVELS = ("2", "3")  # Intermediate and Adventurous: the row of switches must hold either name


@pytest.mark.parametrize("context_options", [SMALL_PHONE, PHONE, {"viewport": {"width": 1280, "height": 900}}])
def test_above_the_map_one_row_of_switches_and_one_row_of_kinds(context, site):
    page = _open(context, site)
    # The kinds in DejaVu Sans, wider than any phone's own font: one line here is one line on any phone.
    page.evaluate("""() => { document.querySelector('#legend').style.fontFamily = '"DejaVu Sans", sans-serif'; }""")
    for level in LONG_LEVELS:
        page.locator("#level").fill(level)
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


# Every link label on the map as Cytoscape drew it, not as the map meant it: the label's middle is the middle of its
# own box. For each: its type size; how far its middle is from its own line (`off`) and from the nearer of its link's
# two terms (`near`), in type sizes; whether it stands above its line, as the text reads, and whether that line is
# steep there; whether it is inside the map; and whether its link leaves the map. Then the pairs
# of labels whose text overlaps, as slanted boxes, and the labels that a line in a strong colour runs through: a line
# of the selected term, or, with the other textbooks shown, one that another textbook states.
DRAWN = """() => { const cy = window.conceptMap.cy, z = cy.zoom(), pan = cy.pan();
  const wide = cy.width(), high = cy.height();
  const own = { includeNodes: false, includeEdges: false, includeLabels: true, includeMainLabels: false,
                includeSourceLabels: true, includeTargetLabels: false, includeOverlays: false };
  const at = (p) => ({ x: p.x * z + pan.x, y: p.y * z + pan.y });
  const curve = (e) => { const s = at(e.sourceEndpoint()), t = at(e.targetEndpoint()), bow = e.controlPoints();
    const c = bow ? at(bow[0]) : { x: (s.x + t.x) / 2, y: (s.y + t.y) / 2 };
    const n = Math.ceil(Math.max(200, Math.hypot(t.x - s.x, t.y - s.y) / 3));
    return Array.from({ length: n + 1 }, (_, i) => { const u = i / n, v = 1 - u;
      return { x: v * v * s.x + 2 * u * v * c.x + u * u * t.x, y: v * v * s.y + 2 * u * v * c.y + u * u * t.y }; }); };
  const far = (p, q) => Math.hypot(p.x - q.x, p.y - q.y), ruler = document.createElement('canvas').getContext('2d');
  const outside = (n) => { const p = n.renderedPosition(); return p.x < 0 || p.x > wide || p.y < 0 || p.y > high; };
  const lit = (e) => e.visible() && !e.hasClass('dim');
  const labels = cy.edges('.link').filter((e) => lit(e) && !e.hasClass('mute') && e.style('source-label')).map((e) => {
    const b = e.renderedBoundingBox(own), m = { x: (b.x1 + b.x2) / 2, y: (b.y1 + b.y2) / 2 }, line = curve(e);
    const size = parseFloat(e.style('font-size')) * z;
    const i = line.reduce((best, p, j) => (far(p, m) < far(line[best], m) ? j : best), 0);
    const a = line[Math.max(0, i - 1)], c = line[Math.min(line.length - 1, i + 1)], d = far(a, c);
    ruler.font = `italic 600 ${size}px ${e.style('font-family')}`;
    const ux = ((c.x - a.x) / d) * (c.x < a.x ? -1 : 1), uy = ((c.y - a.y) / d) * (c.x < a.x ? -1 : 1);   // as it reads
    return { id: e.id(), verb: e.data('verb'), ...m, size, ux, uy, steep: Math.abs(uy) > 2 * Math.abs(ux),
      hw: ruler.measureText(e.data('verb')).width / 2, off: far(line[i], m) / size,
      above: (m.x - line[i].x) * uy - (m.y - line[i].y) * ux > 0,
      near: Math.min(...e.connectedNodes().map((n) => far(n.renderedPosition(), m) - n.renderedWidth() / 2)) / size,
      inside: m.x > 0 && m.x < wide && m.y > 0 && m.y < high, leaves: outside(e.source()) || outside(e.target()) }; });
  // Two slanted boxes are apart when, along a side of one of them, a gap lies between them.
  const apart = (p, q) => [[p.ux, p.uy], [-p.uy, p.ux], [q.ux, q.uy], [-q.uy, q.ux]].some(([x, y]) => {
    const reach = (r) => Math.abs(r.ux * x + r.uy * y) * r.hw + Math.abs(r.ux * y - r.uy * x) * r.size * 0.55;
    return Math.abs((q.x - p.x) * x + (q.y - p.y) * y) > reach(p) + reach(q) - 1; });
  const overlaps = labels.flatMap((p, i) => labels.slice(i + 1).filter((q) => !apart(p, q))
    .map((q) => `${p.verb} / ${q.verb}`));
  const lines = cy.edges('.link').filter((e) => lit(e) && e.hasClass('hl')).map(curve);
  const within = (l, p) => { const x = p.x - l.x, y = p.y - l.y;
    return Math.abs(x * l.ux + y * l.uy) < l.hw - 1 && Math.abs(y * l.ux - x * l.uy) < l.size * 0.4; };
  const through = labels.filter((l) => lines.some((line) => line.some((p) => within(l, p)))).map((l) => l.verb);
  return { labels, overlaps, through }; }"""
CLOSER = """(k) => { const cy = window.conceptMap.cy;
  cy.zoom({ level: cy.zoom() * k, renderedPosition: cy.nodes('.sel')[0].renderedPosition() }); }"""


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
    closer = page.evaluate(DRAWN)["labels"]
    assert closer, "zoomed in, links are still labelled"
    assert all(label["inside"] for label in closer), "each on the map, though its link may now leave it"
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
    page.locator("#how").scroll_into_view_if_needed()
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


# The selected term's drawn links, longest first; and one link's middle brought to the middle of the map, with or
# without a term that is on none of those links dragged onto it, as a reader can drag one.
LONG_FIRST = """() => { const span = (e, k) => e.target().position(k) - e.source().position(k);
  const len = (e) => Math.hypot(span(e, 'x'), span(e, 'y'));
  return window.conceptMap.cy.edges('.link.hl').filter((e) => e.visible()).sort((a, b) => len(b) - len(a))
    .map((e) => e.id()); }"""
AT_MIDDLE = """([id, take]) => { const cy = window.conceptMap.cy, e = cy.getElementById(id), z = cy.zoom();
  const s = e.source().position(), t = e.target().position(), [c] = e.controlPoints();
  const m = { x: (s.x + 2 * c.x + t.x) / 4, y: (s.y + 2 * c.y + t.y) / 4 };   // of the curve, as map.js has it
  if (take) cy.nodes('.term').filter((n) => n.visible() && !n.hasClass('sel') && !n.hasClass('nb'))[0].position(m);
  else cy.pan({ x: cy.width() / 2 - m.x * z, y: cy.height() / 2 - m.y * z }); }"""
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


def _label(page: Any, link: str, *, take: bool) -> dict[str, float] | None:  # noqa: ANN401 - a Playwright Page
    """Bring a link's middle to the middle of the map, or drag a term onto it; then its label as drawn, if it is."""
    page.evaluate(AT_MIDDLE, [link, take])
    _settled(page)
    return next((label for label in page.evaluate(DRAWN)["labels"] if label["id"] == link), None)


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
    labels = page.evaluate(DRAWN)["labels"]
    assert labels, "some links are labelled"
    assert all(label["inside"] for label in labels), "each label is on the map"
    assert all(label["near"] > 0.5 for label in labels), f"and clear of both its terms: {labels}"
    # The test takes a middle itself: whether the drawing takes one changes with every drawing. A label with no room
    # left on its line is held back, so it tries the longest links first, which have the most room.
    here = (page.viewport_size["width"] / 2, page.evaluate("window.conceptMap.cy.height() / 2"))
    before = moved = None
    for link in page.evaluate(LONG_FIRST):
        before = _label(page, link, take=False)
        if before is None or math.dist((before["x"], before["y"]), here) > 2 * before["size"]:
            continue  # not labelled, or not beside its line's middle: that is taken already
        moved = _label(page, link, take=True)
        if moved is not None:
            break
    assert moved is not None, "a label still drawn after a term took its middle"
    slid = math.dist((moved["x"], moved["y"]), (before["x"], before["y"]))
    assert slid > before["size"], "a label whose middle is taken moves along its line"
    assert moved["off"] < 2, "still beside its own line"
    assert moved["near"] > 0.5, "and clear of both its terms"
    assert moved["inside"]


def _read_clearly(page: Any, where: str) -> list[dict[str, Any]]:  # noqa: ANN401 - a Playwright Page
    """Check the labels drawn now are on the map, apart from each other and off the strong lines; return them."""
    drawn = page.evaluate(DRAWN)
    assert all(label["inside"] for label in drawn["labels"]), f"{where}: a label is off the map"
    assert drawn["overlaps"] == [], f"{where}: two labels overlap"
    assert drawn["through"] == [], f"{where}: a line in a strong colour runs through a label"
    return drawn["labels"]


@pytest.mark.parametrize("others", [False, True], ids=["this book alone", "compared with the other textbooks"])
@pytest.mark.parametrize("context_options", [PHONE, {"viewport": {"width": 1280, "height": 900}}])
def test_a_link_label_is_beside_its_line_on_the_map_and_clear_of_labels_and_strong_lines(context, site, others):
    page = _open(context, site)
    page.click(".map-verbs")
    if others:
        page.click("#books .switch")
    labels, unselected = [], []
    view = "() => ({ zoom: window.conceptMap.cy.zoom(), pan: { ...window.conceptMap.cy.pan() } })"
    look = "(view) => { window.conceptMap.cy.stop(); window.conceptMap.cy.viewport(view); }"
    # Terms with many links, which run every way; the last has two long links that run side by side.
    for term in ("full_factorial", "blocking", "resolution"):
        _go(page, term)
        for closer in (1, 1.6, 1.6, 1.6):  # as selected, then closer in: its lines leave the map, close together
            page.evaluate(CLOSER, closer)
            _settled(page)
            labels += _read_clearly(page, f"{term}, closer by {closer}")
            # The same place with no term selected: every link is drawn, or none while the books are compared.
            here = page.evaluate(view)
            page.evaluate("() => { location.hash = ''; }")
            page.wait_for_function("() => !window.conceptMap.cy.nodes('.sel').length")
            _settled(page)
            page.evaluate(look, here)
            _settled(page)
            unselected += _read_clearly(page, f"round {term}, closer by {closer}, no term selected")
            _go(page, term)
            page.evaluate(look, here)
            _settled(page)
    assert len(labels) > 20, "the views do draw labels"
    assert bool(unselected) != others, "with no term selected too, unless the books are compared: no link is drawn then"
    # On its line, the line ran through the gaps between the label's words, in the label's own colour.
    assert all(label["off"] > 0.8 for label in labels + unselected), "each label is off its own line"
    assert all(label["above"] for label in labels + unselected), "and above it, as the text reads"
    assert any(label["steep"] for label in labels), "on a steep line"
    assert any(not label["steep"] for label in labels), "as on a flat one"
    # A selected term keeps its links to terms off the map: each is labelled on the part of its line that can be seen.
    assert any(label["leaves"] for label in labels), "a link that leaves the map carries its label on the map"


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
LEVELS = ["Beginner", "Intermediate", "Adventurous"]


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


def test_a_term_is_found_under_a_name_another_text_uses_and_its_card_lists_that_name_apart(context, site):
    page = _open(context, site)
    term = next(n for n in GRAPH["nodes"] if n.get("found_as") and n["type"] not in CONTEXT)
    hidden = term["found_as"][0]
    shown = " ".join([term["name"], *term["aliases"], term["definition"]]).lower()
    assert hidden.lower() not in shown, "only found_as holds it"
    page.fill("#q", hidden)
    page.click(f"#results button[data-id='{term['id']}']")
    page.wait_for_function("(name) => document.querySelector('#card h2')?.textContent === name", arg=term["name"])
    elsewhere = page.text_content("#card .map-elsewhere")
    assert elsewhere == "In other texts: " + ", ".join(term["found_as"]), "on a line of its own"
    also = page.locator("#card .map-aka:not(.map-elsewhere):not(.map-backed)")  # not the other texts' lines
    assert also.count() == bool(term["aliases"]), "the line of the book's own other names is unchanged"
    assert not also.count() or hidden.lower() not in also.text_content().lower()
    page.fill("#q", term["name"])
    page.click(f"#results button[data-id='{term['id']}']")
    assert page.text_content("#card .map-elsewhere") == elsewhere, "the same card when the leading name was searched"


@pytest.mark.parametrize(
    "context_options", [{"viewport": {"width": 320, "height": 700}, "is_mobile": True}, SMALL_PHONE, PHONE]
)
def test_on_a_phone_the_level_fits_inside_the_page(context, site):
    page = _open(context, site, level=None)
    for level in LONG_LEVELS:
        page.locator("#level").fill(level)
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
      const len = Math.hypot(t.x - s.x, t.y - s.y);   // across the line only: unequal terms shift the middle along it
      return Math.abs((t.x - s.x) * (m.y - s.y) - (t.y - s.y) * (m.x - s.x)) / len / len; })""")
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


# ---------- the other textbooks compared with this one
# What each other textbook also has, worked out as map.js does: a link by its own list; a term by its own list or by
# any link it is on. The page names a textbook by a letter. While the books are compared no link is drawn but the
# selected term's own, and each topic and each term carries a ring: one sector per book, clockwise from 12 o'clock,
# this book first.
BOOK_NAME = {"reviewed_text_1": "Book B", "reviewed_text_2": "Book M"}
RING_INKS = ["--accent", "--map-book2", "--map-book3"]  # this book's, then each other textbook's
RING_GAP = 9  # degrees left open between two sectors, as map.js


def _backing(graph: dict[str, Any]) -> tuple[list[str], dict[int, list[str]], dict[str, list[str]]]:
    books = list(graph.get("sources", {}))
    links = {i: [b for b in books if b in e.get("also", [])] for i, e in enumerate(graph["edges"])}
    terms = {}
    for n in graph["nodes"]:
        on = [links[i] for i, e in enumerate(graph["edges"]) if n["id"] in (e["source"], e["target"])]
        terms[n["id"]] = [b for b in books if b in n.get("also", []) or any(b in mine for mine in on)]
    return books, links, terms


def _in_books(mine: list[str]) -> str:
    names = [BOOK_NAME[b] for b in mine]
    return f"{', '.join(names[:-1])} and {names[-1]}" if len(names) > 1 else names[0]


def _others(graph: dict[str, Any], count: int) -> dict[str, Any]:
    """Return the graph with other textbooks of the test's own: one states every second link, the next every third."""
    more = json.loads(json.dumps(graph))
    more["sources"] = dict.fromkeys(list(BOOK_NAME)[:count], "A textbook for this test")
    for n in more["nodes"]:
        n["also"] = []
    for i, e in enumerate(more["edges"]):
        e["also"] = [b for b, every in zip(more["sources"], (2, 3), strict=False) if i % every == 0]
    return more


TWO_OTHERS = _others(GRAPH, 2)
GRAPHS = {"as copied": GRAPH, "one other": _others(GRAPH, 1), "two others": TWO_OTHERS}
# Each colour of the page asked for by name, as [red, green, blue].
COLOURS = """(names) => names.map((name) => { const i = document.createElement('i'); i.style.color = `var(${name})`;
  document.body.append(i); const c = getComputedStyle(i).color.match(/[\\d.]+/g).slice(0, 3).map(Number); i.remove();
  return c; })"""
# The colour drawn on the map at each point, once each is nearer the colour wanted there than any other a ring is drawn
# in, or the map has had 90 frames to draw it: after a move Cytoscape draws from a coarser copy first. "Nearer", as a
# thin rim's edge is blended with what is beside it.
INKS = """([points, indices, palette]) => new Promise((done) => {
  const wanted = indices.map((i) => palette[i]);
  const canvas = document.querySelector('#cy canvas[data-id="layer2-node"]'), k = canvas.width / canvas.clientWidth;
  const read = () => points.map((p) => [...canvas.getContext('2d')
    .getImageData(Math.round(p.x * k), Math.round(p.y * k), 1, 1).data].slice(0, 3));
  const far = (a, b) => Math.hypot(...a.map((v, i) => v - b[i]));
  const nearest = (c) => palette.reduce((best, ink) => (far(c, ink) < far(c, best) ? ink : best));
  let frames = 0;
  const look = () => { const seen = read();
    if (seen.every((c, i) => nearest(c) === wanted[i]) || ++frames > 90) done(seen.map(nearest));
    else requestAnimationFrame(look); };
  look(); })"""
# Where each topic is drawn, how wide, and the box of its name: a small topic's name reaches over its rim.
TOPICS_DRAWN = """() => Object.fromEntries(window.conceptMap.cy.nodes('.topic').map((n) => [n.id(), {
  ...n.renderedPosition(), w: n.renderedWidth(),
  name: n.renderedBoundingBox({ includeNodes: false, includeEdges: false, includeLabels: true }) }]))"""
# Each term drawn whole on the map with its ring: where, how wide, and whether no name is drawn over its ring.
TERMS_DRAWN = """() => { const cy = window.conceptMap.cy, wide = cy.width(), high = cy.height();
  const names = cy.nodes('.topic, .subtopic, .subname, .term.lab').filter((n) => n.visible() && n.style('label'))
    .map((n) => n.renderedBoundingBox({ includeNodes: false, includeEdges: false, includeLabels: true }));
  return cy.nodes('.term').filter((n) => n.visible()).map((n) => {
    const p = n.renderedPosition(), w = n.renderedWidth(), r = 1.1 * w;   // as far as its ring reaches
    return { id: n.id(), x: p.x, y: p.y, w, whole: p.x > r && p.x < wide - r && p.y > r && p.y < high - r,
      clear: !names.some((b) => p.x + r > b.x1 && p.x - r < b.x2 && p.y + r > b.y1 && p.y - r < b.y2) }; }); }"""
BETWEEN = "() => window.conceptMap.cy.edges('.between').filter((e) => e.visible()).length"
SWITCH = "#books .switch"


def _open_with(context: Any, site: str, which: str) -> tuple[dict[str, Any], Any]:  # noqa: ANN401 - Playwright objects
    """Open the map on the copied graph, or on one with other textbooks of the test's own."""
    graph = GRAPHS[which]
    if graph is not GRAPH:
        context.route("**/map/graph.json", lambda route: route.fulfill(json=graph))
    return graph, _open(context, site)


def _on_ring(at: dict[str, float], radius: float, sector: int, sectors: int, along: float) -> dict[str, float]:
    """Return the point of a ring that is ``along`` the way round one of its sectors, clockwise from 12 o'clock."""
    start, end = sector * 360 / sectors + RING_GAP / 2, (sector + 1) * 360 / sectors - RING_GAP / 2
    angle = math.radians(start + (end - start) * along)
    return {"x": at["x"] + radius * math.sin(angle), "y": at["y"] - radius * math.cos(angle)}


def _not_as_wanted(page: Any, places: list[tuple[str, dict[str, float], int]], palette: list[list[int]]) -> list[str]:  # noqa: ANN401
    """Return each place, of (what it is, where, which colour of the palette is wanted there), drawn in another."""
    seen = page.evaluate(INKS, [[where for _, where, _ in places], [wanted for _, _, wanted in places], palette])
    return [
        f"{what}: drawn {got}, wanted {palette[wanted]}"
        for (what, _, wanted), got in zip(places, seen, strict=True)
        if got != palette[wanted]
    ]


@pytest.mark.parametrize("context_options", [{"viewport": {"width": 1280, "height": 1100}}])  # a map tall enough to
@pytest.mark.parametrize("which", list(GRAPHS))  # hold most names inside their rims
def test_compared_with_other_books_a_topic_s_ring_is_filled_by_the_share_of_its_terms_each_has(context, site, which):
    graph, page = _open_with(context, site, which)
    books, _, terms = _backing(graph)
    assert books, "the graph names the other textbooks read against it"
    assert page.is_visible("#books")
    assert not page.is_checked("#others"), "the map opens as the book alone"
    assert page.evaluate(BETWEEN) > 0
    page.click(SWITCH)
    _settled(page)
    assert page.evaluate(BETWEEN) == 0, "the lines between the topics go: the rings alone say who has what"
    shown = set(page.evaluate(SHOWN))
    palette = page.evaluate(
        COLOURS, [*RING_INKS[: len(books) + 1], "--line"]
    )  # each book's ink, then the empty track's
    drawn, places, partly, under = page.evaluate(TOPICS_DRAWN), [], 0, 0
    for topic in graph["clusters"]:
        mine = [n["id"] for n in graph["nodes"] if n["cluster"] == topic["id"] and n["id"] in shown]
        shares = [1.0] + [sum(b in terms[i] for i in mine) / len(mine) for b in books]  # this book has all of them
        partly += sum(0.1 < share < 0.9 for share in shares)
        at = drawn[topic["id"]]
        for sector, share in enumerate(shares):
            for along in (0.04, 0.2, 0.4, 0.6, 0.8, 0.96):
                if abs(along - share) < 0.05:
                    continue  # too near where the colour ends to say which is drawn there
                # The middle of the rim: 9 px wide on a large topic, 7.5% of the topic's width on a small one.
                where = _on_ring(at, at["w"] / 2 - min(4.5, 0.0375 * at["w"]), sector, len(shares), along)
                name = at["name"]  # its box is some 6 px wider each side than its letters: their halo, and padding
                if name["x1"] + 6 < where["x"] < name["x2"] - 6 and name["y1"] < where["y"] < name["y2"]:
                    under += 1  # the topic's name is drawn there
                    continue
                ink = sector if along < share else len(shares)
                places.append((f"{topic['short']}, sector {sector}, {along:.0%} of the way round it", where, ink))
    assert partly, "some topic is covered in part by another textbook, or this test sees no sector end"
    assert under < len(places) / 5, f"few places of a rim are under a name: {under} against {len(places)} in the clear"
    assert not _not_as_wanted(page, places, palette)


@pytest.mark.parametrize("which", list(GRAPHS))
def test_zoomed_in_each_term_carries_a_ring_with_a_sector_for_each_book_this_book_first(context, site, which):
    graph, page = _open_with(context, site, which)
    books, _, terms = _backing(graph)
    page.click(SWITCH)
    page.click(".map-names")  # no term's name is drawn, so none lies over a ring
    palette = page.evaluate(
        COLOURS, [*RING_INKS[: len(books) + 1], "--line"]
    )  # each book's ink, then the empty track's
    places, sampled = [], {}
    for topic in sorted(graph["clusters"], key=lambda c: -c["size"])[:4]:
        # Close enough that a ring is several pixels wide.
        page.evaluate(
            """(id) => { const cy = window.conceptMap.cy, at = cy.getElementById(id).position(); cy.stop();
              cy.viewport({ zoom: 3, pan: { x: cy.width() / 2 - 3 * at.x, y: cy.height() / 2 - 3 * at.y } }); }""",
            topic["id"],
        )
        _settled(page)
        assert page.evaluate(STATE)["links"] == 0, "no link is drawn"
        for term in page.evaluate(TERMS_DRAWN):
            if not (term["whole"] and term["clear"]):
                continue
            has = [True] + [b in terms[term["id"]] for b in books]
            sampled[term["id"]] = has
            # Its ring is outside the term: the ring's middle is 0.94 of the term's width from the term's centre.
            middles = [_on_ring(term, 0.94 * term["w"], k, len(has), 0.5) for k in range(len(has))]
            places += [(f"{term['id']}, sector {k}", middles[k], k if h else len(has)) for k, h in enumerate(has)]
        assert not _not_as_wanted(page, places, palette)
        places = []
    assert len(sampled) > 20, "the views do draw terms"
    for k in range(1, len(books) + 1):
        assert {has[k] for has in sampled.values()} == {True, False}, "terms a textbook has, and terms it has not"


HOW_DRAWN = """() => { const cy = window.conceptMap.cy;
  const shade = (value) => { const i = document.createElement('i'); i.style.color = value;
    document.body.append(i); const c = getComputedStyle(i).color.replace(/\\s/g, ''); i.remove(); return c; };
  const colour = (name) => shade(`var(${name})`);
  const fam = (n) => n.classes().find((c) => c.startsWith('fam-')).slice(4);
  const term = cy.nodes('.term').filter((n) => n.visible() && !n.hasClass('sel') && !n.hasClass('held'))[0];
  const seen = cy.edges('.link').filter((e) => e.visible());
  const icon = document.querySelector('#legend button[aria-pressed="true"]');
  return { accent: colour('--accent'), muted: colour('--muted'),
    links: seen.length, lit: seen.filter((e) => e.hasClass('hl')).length,
    line: shade(cy.edges('.link.hl')[0].style('line-color')),
    term: { fill: shade(term.style('background-color')), kind: colour(`--map-${fam(term)}`) },
    icon: { fill: shade(icon.querySelector('path').getAttribute('fill')),
      kind: colour(`--map-${icon.dataset.fam}`) } }; }"""


def _a_term_of_both(graph: dict[str, Any], links: dict[int, list[str]]) -> tuple[str, int]:
    """Return a term with a link another textbook states and a link of this book alone, and the first of those links."""
    for n in graph["nodes"]:
        mine = [i for i, e in enumerate(graph["edges"]) if n["id"] in (e["source"], e["target"])]
        if any(links[i] for i in mine) and not all(links[i] for i in mine):
            return n["id"], next(i for i in mine if links[i])
    raise AssertionError("no term has both kinds of link")


@pytest.mark.parametrize("which", ["as copied", "two others"])
def test_compared_with_other_books_only_the_selected_term_s_links_are_drawn_in_a_neutral(context, site, which):
    graph, page = _open_with(context, site, which)
    _, links, terms = _backing(graph)
    # A term with a link another textbook states: its card names the textbooks whether or not they are compared.
    term, link = _a_term_of_both(graph, links)
    _go(page, term)
    assert page.text_content("#card .map-backed") == f"Also in {_in_books(terms[term])}"
    off = page.evaluate(HOW_DRAWN)
    assert off["links"] > off["lit"] > 0, "until the books are compared every link is drawn, the term's own lit"
    assert off["line"] == off["accent"], "in the accent colour"
    assert off["term"]["fill"] == off["term"]["kind"], "and a term has the colour of its kind"
    assert off["icon"]["fill"] == off["icon"]["kind"], "as its shape has above the map"
    read = """(id) => { window.conceptMap.cy.getElementById(id).emit('mouseover');
      return document.querySelector('#striptext').textContent; }"""
    assert "also in" not in page.evaluate(read, f"l{link}")

    page.click(SWITCH)
    _settled(page)
    on = page.evaluate(HOW_DRAWN)
    assert on["links"] == on["lit"] == off["lit"], "then the selected term's links alone"
    assert on["line"] == on["muted"], "in a neutral: colour now says which book has a term"
    assert on["term"]["fill"] == on["muted"] != on["term"]["kind"], "the kinds keep their shapes, not their colours"
    assert on["icon"]["fill"] == on["muted"] != on["icon"]["kind"], "above the map too"
    said = page.evaluate(read, f"l{link}")
    assert said.endswith(f"(also in {_in_books(links[link])})"), "a link read out says who else states it"
    page.reload()
    page.wait_for_function("() => window.conceptMap && window.conceptMap.cy.nodes('.sel').length")
    _settled(page)
    assert page.is_checked("#others"), "a reload keeps the choice"
    assert page.evaluate(HOW_DRAWN)["links"] == on["lit"]
    page.evaluate("() => { location.hash = ''; }")
    page.wait_for_function("() => !window.conceptMap.cy.nodes('.sel').length")
    _settled(page)
    assert page.evaluate(STATE)["links"] == 0, "with no term selected, no link"


KEY = """() => { const key = document.querySelector('#bookkey');
  return { sectors: [...key.querySelectorAll('svg path')].map((p) => getComputedStyle(p).stroke.replace(/\\s/g, '')),
    swatches: [...key.querySelectorAll('i')].map((i) => getComputedStyle(i).backgroundColor.replace(/\\s/g, '')),
    faint: Number(getComputedStyle(key).opacity) < 1 }; }"""


@pytest.mark.parametrize("which", list(GRAPHS))
def test_the_key_names_the_books_in_the_order_of_the_sectors_and_a_topic_s_card_counts_what_each_has(
    context, site, which
):
    graph, page = _open_with(context, site, which)
    books, _, terms = _backing(graph)
    names = ["This book"] + [BOOK_NAME[b] for b in books]
    assert [t.strip() for t in page.locator("#bookkey span").all_text_contents()] == names
    assert page.get_attribute("#bookkey", "aria-label") == f"Each ring, clockwise from the top: {', '.join(names)}"
    inks = [f"rgb({','.join(str(v) for v in ink)})" for ink in page.evaluate(COLOURS, RING_INKS[: len(names)])]
    key = page.evaluate(KEY)
    assert key["sectors"] == key["swatches"] == inks, "a sector for each book, in its colour, beside its name"
    assert key["faint"], "the key waits for the switch"
    page.click(SWITCH)
    assert not page.evaluate(KEY)["faint"]
    # A topic's card says in words what its ring draws, whether or not the books are compared.
    _open_topic(page, BIGGEST)
    shown = set(page.evaluate(SHOWN))
    mine = [n["id"] for n in graph["nodes"] if n["cluster"] == BIGGEST["id"] and n["id"] in shown]
    counts = [f"{BOOK_NAME[b]} has {sum(b in terms[i] for i in mine)}" for b in books]
    said = f"{counts[0]} of these terms{''.join(f', {c}' for c in counts[1:])}."
    assert page.text_content("#card .map-backed") == said


@pytest.mark.parametrize("context_options", [PHONE, {"viewport": {"width": 1280, "height": 900}}])
def test_zooming_while_books_are_compared_loads_no_image(context, site):
    """A ring's image is the same at every zoom: one loaded as the reader zooms would be missing for a frame."""
    page = _open(context, site)
    page.click(SWITCH)
    _settled(page)
    images = """() => window.conceptMap.cy.nodes('.topic, .term').map((n) => n.style('background-image')).join(' ')"""
    opened = page.evaluate(images)
    assert opened.count("data:image/svg+xml") > len(GRAPH["nodes"]), "every topic and term has its ring"
    zoom = """(k) => { const cy = window.conceptMap.cy;
      cy.zoom({ level: cy.zoom() * k, renderedPosition: { x: 200, y: 200 } }); }"""
    for _ in range(5):  # from the topics alone to the terms close up
        page.evaluate(zoom, 1.9)
        _settled(page)
        assert page.evaluate(images) == opened
    assert page.evaluate(STATE)["terms"], "the terms are drawn by now"


# Each name drawn that lies on another term's ring, and how many names are above their terms.
NAMES_ON_RINGS = """() => { const cy = window.conceptMap.cy;
  const terms = cy.nodes('.term').filter((n) => n.visible());
  const named = terms.filter((n) => n.hasClass('lab') && !n.hasClass('sel') && !n.hasClass('nb'));
  const box = (n) => n.renderedBoundingBox({ includeNodes: false, includeEdges: false, includeLabels: true });
  const on = named.filter((n) => { const b = box(n);
    return terms.some((t) => { const p = t.renderedPosition(), r = 1.075 * t.renderedWidth();   // its ring's outer edge
      return t !== n && p.x + r > b.x1 + 2 && p.x - r < b.x2 - 2 && p.y + r > b.y1 + 2 && p.y - r < b.y2 - 2; }); });
  return { on: on.map((n) => n.data('label')), named: named.length,
    above: named.filter((n) => box(n).y2 < n.renderedPosition().y).length }; }"""


@pytest.mark.parametrize("context_options", [PHONE, {"viewport": {"width": 1280, "height": 900}}])
def test_while_books_are_compared_a_term_s_name_keeps_clear_of_the_rings_above_its_term_if_need_be(context, site):
    page = _open(context, site)
    page.click(SWITCH)
    _open_topic(page, BIGGEST)
    named = above = 0
    for _ in range(4):
        page.click("#zin")
        _settled(page)
        state, names = page.evaluate(STATE), page.evaluate(NAMES_ON_RINGS)
        assert state["overlaps"] == 0, "no name is on another"
        assert names["on"] == [], "nor on another term's ring"
        named, above = named + names["named"], above + names["above"]
    assert named > 10, "the views do name terms"
    assert above, "a name goes above its term where the place below it is taken"
    page.click(SWITCH)
    _settled(page)
    assert page.evaluate(NAMES_ON_RINGS)["above"] == 0, "with no rings, every name is below its term"


@pytest.mark.parametrize(
    "context_options", [{"viewport": {"width": 320, "height": 700}, "is_mobile": True}, SMALL_PHONE, PHONE]
)
def test_on_a_phone_the_kinds_are_large_and_the_key_to_the_rings_is_one_row_under_its_switch(context, site):
    context.route("**/map/graph.json", lambda route: route.fulfill(json=TWO_OTHERS))  # two textbooks: the longest row
    page = _open(context, site)
    page.click(SWITCH)
    # The names in DejaVu Sans, wider than any phone's own font: one line here is one line on any phone.
    page.evaluate("""() => { document.querySelector('#books').style.fontFamily = '"DejaVu Sans", sans-serif'; }""")
    rows = page.evaluate("""() => { const box = (s) => document.querySelector(s).getBoundingClientRect();
      const page = document.documentElement;
      const middle = (e) => { const b = e.getBoundingClientRect(); return (b.top + b.bottom) / 2; };
      return { icon: box('#legend svg').width, legend: box('#legend'), books: box('#books'), key: box('#bookkey'),
        switch: box('#books .switch'), words: box('#books .switch__text').width,
        middles: [...document.querySelectorAll('#bookkey svg, #bookkey span')].map(middle),
        map: box('#cy'), width: page.clientWidth, scroll: page.scrollWidth }; }""")
    assert rows["icon"] >= 22, "each kind's shape is large enough to tell apart and to tap"
    assert rows["legend"]["bottom"] <= rows["books"]["top"], "the kinds, then the other textbooks"
    assert rows["books"]["bottom"] <= rows["map"]["top"], "both above the map"
    assert rows["words"] > 60, "the switch says what it does"
    assert rows["switch"]["bottom"] <= rows["key"]["top"] + 1, "the key is under its switch"
    assert rows["key"]["right"] <= rows["width"] - 16 + 0.5, "the row ends inside the page's side margin"
    assert rows["scroll"] == rows["width"], "and the page does not scroll sideways"
    assert max(rows["middles"]) - min(rows["middles"]) < 3, "the ring and the three names on one line"


# At one width: whether the page is wider than the window, what reaches past it, and how many lines the kinds take.
ACROSS = """() => { const page = document.documentElement, wide = page.clientWidth;
  const box = (e) => e.getBoundingClientRect();
  const shown = (e) => getComputedStyle(e).display !== 'none' && box(e).width > 0;
  const past = [...document.querySelectorAll('main *')]   // not the map's own drawing, which its box cuts off
    .filter((e) => !e.closest('#cy') && shown(e) && box(e).right > wide + 1);
  const lines = new Set([...document.querySelectorAll('#legend button')].map((b) => Math.round(box(b).top)));
  return { scrolls: page.scrollWidth > wide, past: [...new Set(past.map((e) => e.closest('[id]')?.id ?? e.tagName))],
           lines: lines.size, stacked: getComputedStyle(document.querySelector('#legend button')).flexDirection }; }"""


@pytest.mark.parametrize("others", [False, True], ids=["this book alone", "with the other textbooks"])
def test_at_no_width_does_the_map_s_page_scroll_sideways(context, site, others):
    context.route("**/map/graph.json", lambda route: route.fulfill(json=TWO_OTHERS))  # two textbooks: the longest row
    page = _open(context, site)
    if others:
        page.click("#books .switch")
    forms = []
    for width in range(320, 1281, 8):  # a phone on its side and a window beside another are between the two layouts
        page.set_viewport_size({"width": width, "height": 800})
        across = page.evaluate(ACROSS)
        assert not across["scrolls"], f"at {width} px the page is wider than the window: {across['past']}"
        assert across["past"] == [], f"at {width} px: {across}"
        if width >= 360:
            assert across["lines"] == 1, f"at {width} px the five kinds are on one line"
        forms += [] if forms and forms[-1] == across["stacked"] else [across["stacked"]]
    assert forms == ["column", "row"], "each shape above its name while that is what fits, then beside it"


# --------------------------------------------------------------------------- formulas
# A term of the test's own: a formula in its definition, between dollar signs as the graph writes one, and another
# in the line it quotes, in the book's own role.
TRACE, INVERSE = r"\operatorname{trace}\left((\mathbf{X}^T\mathbf{X})^{-1}\right)", r"\mathbf{M}^{-1}"


def _with_formulas(graph: dict[str, Any]) -> dict[str, Any]:
    graph = json.loads(json.dumps(graph))
    term = next(n for n in graph["nodes"] if n["id"] == "a_optimal")
    term["definition"] = f"Minimize ${TRACE}$: the average variance of the parameters."
    term["quote"] = f"the sum of the coefficient variances (the diagonal entries of :math:`{INVERSE}`)"
    return graph


WITH_FORMULAS = _with_formulas(GRAPH)
# Each formula on the card and in the strip under the map: where it is, the LaTeX its MathML carries, whether the
# drawn copy is kept from a screen reader, and whether it can be seen.
FORMULAS = """() => [...document.querySelectorAll('#card .tex, #striptext .tex')].map((f) => ({
  where: f.closest('blockquote') ? 'quote' : f.closest('#card') ? 'definition' : 'strip',
  latex: f.querySelector('math annotation[encoding="application/x-tex"]')?.textContent ?? null,
  drawn: f.querySelector('.katex-html')?.getAttribute('aria-hidden') ?? null,
  seen: getComputedStyle(f).visibility === 'visible' && f.getBoundingClientRect().width > 0 }))"""


def _katex_asked(page: Any) -> list[str]:  # noqa: ANN401 - a Playwright Page
    """Record what the page asks KaTeX's folder for, and where it asks."""
    asked: list[str] = []
    page.on("request", lambda request: asked.append(request.url) if "katex" in request.url.lower() else None)
    return asked


@pytest.mark.parametrize("context_options", [PHONE, {"viewport": {"width": 1280, "height": 900}}])
def test_a_formula_on_a_card_is_drawn_and_a_screen_reader_is_given_it_as_math(context, site):
    context.route("**/map/graph.json", lambda route: route.fulfill(json=WITH_FORMULAS))
    page = context.new_page()
    asked = _katex_asked(page)
    page.goto(f"{site}map/")
    page.wait_for_function("() => window.conceptMap && document.querySelector('#card h2')")
    _go(page, "blocking")
    assert [url.rsplit("/", 1)[1] for url in asked] == ["katex.min.css"], "no formula yet, so no script and no font"
    _go(page, "a_optimal")
    drawn = {"drawn": "true", "seen": True}
    assert page.evaluate(FORMULAS) == [
        {"where": "strip", "latex": TRACE, **drawn},
        {"where": "definition", "latex": TRACE, **drawn},
        {"where": "quote", "latex": INVERSE, **drawn},
    ]
    said = page.text_content("#card")
    assert "$" not in said, "the marks around a formula are not shown"
    assert ":math:" not in said
    # What a screen reader is given: the words, and the formula once, as math; not the copy drawn for the eye too.
    tree = page.locator("#card p").first.aria_snapshot()
    assert tree.count("- math") == 1, tree
    assert "- text: Minimize" in tree, tree
    assert all(url.startswith(f"{site}vendor/katex/") for url in asked), "KaTeX and its fonts come from this site"
    names = [url.rsplit("/", 1)[1] for url in asked]
    assert "katex.min.js" in names
    assert any(name.endswith(".woff2") for name in names)
    assert not [name for name in names if name.endswith((".woff", ".ttf"))], "only the fonts that are here"
    # The link to the book looks for the quote's words, and a formula is none of the page's words.
    assert page.get_attribute("#card .map-book a", "href").endswith(
        ":~:text=the%20sum%20of%20the,(the%20diagonal%20entries%20of"
    )


def test_a_formula_is_shown_as_it_is_written_when_katex_cannot_be_fetched(context, site, console):
    console.allow(r"katex|404")  # the browser reports the script it could not load
    context.route("**/map/graph.json", lambda route: route.fulfill(json=WITH_FORMULAS))
    context.route("**/vendor/katex/katex.min.js", lambda route: route.fulfill(status=404, body=""))
    page = _open(context, site)
    _go(page, "a_optimal")
    assert page.evaluate(FORMULAS) == [
        {"where": where, "latex": None, "drawn": None, "seen": True} for where in ("strip", "definition", "quote")
    ]
    assert f"Minimize {TRACE}: the average variance of the parameters." in page.text_content("#card")


# --------------------------------------------------------------------------- the book's markup
# A quote is a line of the book's source. This one holds each mark such a line can: an index entry with its target,
# a reference with its title and one without, a literal that holds asterisks, strong, emphasis, words in single
# backquotes, both forms of the double dash, a formula, asterisks that are no marks, and text a browser could
# take for a tag.
TYPED = (
    r"The **second phase** adds *sequential runs* -- the :index:`bias error <pair: bias error; experiments>` "
    r":math:`\beta_2` of :ref:`blocking <DOE-blocking>` and :ref:`DOE-label`, with ``a * b`` and "
    r"`simplex-lattice design` always measurable--in some way: 2 * 3 * 4, a*b*c and <b>no tag</b>"
)
PRINTED = (
    "The second phase adds sequential runs \u2013 the bias error  of blocking and DOE-label, with a * b and "
    "simplex-lattice design always measurable\u2013in some way: 2 * 3 * 4, a*b*c and <b>no tag</b>"
)
# The quote on the card: its words with each formula taken out, and the words set as strong, emphasis and literal.
QUOTE = """() => { const quote = document.querySelector('#card .map-quote p'), copy = quote.cloneNode(true);
  copy.querySelectorAll('.tex').forEach((f) => f.remove());
  const words = (tag) => [...quote.querySelectorAll(tag)].map((e) => e.textContent);
  return { text: copy.textContent, strong: words('strong'), em: words('em'), code: words('code'),
           upright: [...quote.querySelectorAll('em, code')].every((e) => getComputedStyle(e).fontStyle === 'normal'),
           italic: getComputedStyle(quote).fontStyle, formulas: quote.querySelectorAll('.tex .katex').length,
           tags: [...new Set([...copy.querySelectorAll('*')].map((e) => e.tagName.toLowerCase()))].sort() }; }"""
MARKS = re.compile(r"[*`]|:[a-z]+:|--")


def test_a_quote_is_shown_as_the_book_prints_it_without_its_marks(context, site):
    graph = json.loads(json.dumps(GRAPH))
    next(n for n in graph["nodes"] if n["id"] == "blocking")["quote"] = TYPED
    context.route("**/map/graph.json", lambda route: route.fulfill(json=graph))
    page = _open(context, site)
    _go(page, "blocking")
    shown = page.evaluate(QUOTE)
    assert shown["text"] == PRINTED
    assert shown["strong"] == ["second phase"]
    assert shown["em"] == ["sequential runs"]
    assert shown["code"] == ["a * b"], "a literal keeps what it holds"
    assert shown["tags"] == ["code", "em", "strong"], "and nothing typed in the line is read as a tag"
    assert shown["formulas"] == 1
    assert shown["italic"] == "italic", "the quoted line is in italics"
    assert shown["upright"], "so its emphasis and its literal stand upright"
    # The link to the book still looks for words that are on the page: none with a mark on it.
    href = page.get_attribute("#card .map-book a", "href")
    assert not MARKS.search(href.split(":~:text=")[1].replace("%2D", "-")), href


def test_no_quote_of_the_graph_shows_a_mark_of_the_book_s_source(context, site):
    page = _open(context, site)
    typed = [n for n in GRAPH["nodes"] if MARKS.search(n["quote"])]
    assert len(typed) >= 10, "the graph quotes lines with emphasis, index entries, formulas and double dashes"
    for term in typed:
        _go(page, term["id"])
        shown = page.evaluate(QUOTE)
        assert not MARKS.search(shown["text"]), f"{term['id']}: {shown['text']}"
        assert shown["formulas"] == term["quote"].count(":math:"), term["id"]
