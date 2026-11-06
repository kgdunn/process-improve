"""The concept map (map/index.html): the ideas of designed experiments, and how the book links them.

Its graph, ``web/map/graph.json``, is built and checked against the book in pid-book's ``knowledge-graph/``. These
tests drive the page as a reader would, at a phone's width where that matters: what a tap opens, what zooming shows,
and what the page remembers. ``window.conceptMap`` gives them the map's ``cy`` and its data.
"""

from __future__ import annotations

import json
from itertools import pairwise
from typing import Any

import pytest

from tests.web._support import WEB

GRAPH = json.loads((WEB / "map" / "graph.json").read_text(encoding="utf-8"))
PHONE = {"viewport": {"width": 390, "height": 844}, "is_mobile": True}
DETAIL_GAP, LABEL_GAP, TERM_GAP = 22, 45, 62  # as map.js: terms appear, then sub-topic terms are named

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
    page.wait_for_timeout(500)
    page.mouse.click(*_on_map(page, _rendered(page, topic["id"])))
    page.wait_for_function("(name) => document.querySelector('#card h2').textContent === name", arg=topic["name"])
    page.wait_for_timeout(500)


BIGGEST = max(GRAPH["clusters"], key=lambda c: c["size"])


def test_it_opens_on_the_topics_with_their_terms_hidden(context, site):
    page = _open(context, site)
    assert page.text_content("#card h2") == f"{len(GRAPH['clusters'])} topics"
    assert page.evaluate(STATE)["terms"] == 0
    assert page.locator("#links .map-chip").count() == len(GRAPH["clusters"])
    assert page.text_content("#map-stats").endswith(" topics.")


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
        page.wait_for_timeout(500)
    named = [f for f in fonts if f]
    assert named, "terms are named once zoomed in"
    assert all(b > a or b >= 19 for a, b in pairwise(named)), fonts


def test_a_tap_on_a_term_shows_its_definition_its_book_line_and_lights_its_links(context, site):
    page = _open(context, site)
    page.evaluate("() => { location.hash = 'full-factorial'; }")
    page.wait_for_function("() => document.querySelector('#card h2').textContent === 'Full factorial'")
    page.wait_for_timeout(600)
    page.mouse.click(*_on_map(page, _rendered(page, "full_factorial")))
    page.wait_for_timeout(400)
    term = next(n for n in GRAPH["nodes"] if n["id"] == "full_factorial")
    assert page.text_content("#card h2") == term["name"]
    assert term["evidence"] in page.text_content("#card .map-quote cite")
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
    page.evaluate("() => { location.hash = 'resolution'; }")
    page.wait_for_function("() => document.querySelector('#card h2').textContent === 'Resolution'")
    page.wait_for_timeout(600)
    page.click("#labels")
    page.wait_for_timeout(300)
    assert page.evaluate(STATE)["labels"] == 1, "only the selected term keeps its name"
    page.reload()
    page.wait_for_function("() => window.conceptMap && document.querySelector('#card h2')")
    assert page.get_attribute("#labels", "aria-pressed") == "false"


def test_a_kind_of_term_can_be_hidden_and_the_choice_is_remembered(context, site):
    page = _open(context, site)
    _open_topic(page, BIGGEST)
    shown = page.evaluate(STATE)
    page.click('#legend button[data-fam="concept"]')
    page.wait_for_timeout(300)
    fewer = page.evaluate(STATE)
    concepts = page.evaluate(
        "() => window.conceptMap.cy.nodes('.term').filter((n) => n.data('fam') === 'concept' && n.visible()).length"
    )
    assert concepts == 0
    assert fewer["links"] < shown["links"], "a hidden term's links go with it"
    page.evaluate("() => { location.hash = 'confounding'; }")
    page.wait_for_timeout(600)
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
    for details in how.locator("details").all():
        details.locator("summary").click()
    numbers = how.text_content()
    assert f"{len(GRAPH['nodes'])} terms" in numbers
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
    page.wait_for_selector("footer >> text=concept map")
    page.click("footer >> text=concept map")
    page.wait_for_url("**/map/")


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
