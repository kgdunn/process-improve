"""The concept map's files, checked without a browser: the vendored Cytoscape, and the graph the page draws.

``web/map/graph.json`` is built and checked against the book in pid-book's ``knowledge-graph/``; these tests catch a
copy that arrives incomplete. How the page behaves is tested in ``tests/web/e2e/test_concept_map.py``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct

from process_improve.experiments.design_catalog import FAMILIES
from tests.web._support import APP_SOURCES, WEB, source

GRAPH = json.loads((WEB / "map" / "graph.json").read_text(encoding="utf-8"))
APART = 0.15  # in OKLab: closer than this, two colours start to read as one


def test_the_map_s_cytoscape_is_the_published_release_with_its_licence():
    """``web/vendor/cytoscape.mjs`` is npm's file unchanged after a header naming its release and its digest."""
    vendored = (WEB / "vendor" / "cytoscape.mjs").read_bytes()
    header, _, published = vendored.partition(b" */\n")
    words = header.decode("ascii")
    release = re.search(r"cytoscape\.js (\d+\.\d+\.\d+), https://github\.com/cytoscape/cytoscape\.js", words)
    assert release
    assert f"/cytoscape-{release[1]}.tgz" in words
    digest = re.search(r"SHA-256 of dist/cytoscape\.esm\.min\.js: ([0-9a-f]{64})", words)[1]
    assert digest == hashlib.sha256(published).hexdigest()
    assert "The MIT License (MIT)" in words
    assert "Copyright (c) 2016-2023, The Cytoscape Consortium." in words
    assert 'import cytoscape from "./vendor/cytoscape.mjs";' in source("map.js"), "the map page imports it"
    assert "cytoscape" not in source("app.js"), "never the start page"


def test_the_graph_names_its_source():
    """The copy says where in pid-book it was built, and from which commit, so it can be traced and updated."""
    assert GRAPH["source"]["repository"] == "kgdunn/pid-book"
    assert GRAPH["source"]["path"] == "knowledge-graph/doe_kg_map.json"
    assert re.fullmatch(r"[0-9a-f]{40}", GRAPH["source"]["commit"])


def test_every_term_cites_the_book_and_has_a_place_on_the_map():
    ids = [n["id"] for n in GRAPH["nodes"]]
    assert len(ids) == len(set(ids)), "no term twice"
    topics = {c["id"] for c in GRAPH["clusters"]}
    subtopics = {s["id"] for c in GRAPH["clusters"] for s in c["subs"]}
    for n in GRAPH["nodes"]:
        assert re.fullmatch(r"[\w/-]+\.rst:\d+", n["evidence"]), n["id"]
        assert n["quote"].strip(), n["id"]
        assert n["book_file"].endswith(".rst"), n["id"]
        assert n["definition"].strip(), n["id"]
        assert n["cluster"] in topics, n["id"]
        assert n["sub"] is None or n["sub"] in subtopics, n["id"]
        assert isinstance(n["x"], float | int), n["id"]
        assert isinstance(n["y"], float | int), n["id"]


def test_every_link_joins_two_terms_and_every_term_has_one():
    ids = {n["id"] for n in GRAPH["nodes"]}
    ends = set()
    for e in GRAPH["edges"]:
        assert e["source"] in ids, e
        assert e["target"] in ids, e
        assert e["source"] != e["target"], e
        assert re.fullmatch(r"[A-Z_]+", e["type"]), e
        ends |= {e["source"], e["target"]}
    assert ends == ids, "a term with no link would float alone on the map"


def test_the_explanation_has_every_number_it_reads():
    """The page's "How this map was made" reads these; a copy without them would show gaps."""
    method = GRAPH["method"]
    for key in (
        "runs",
        "threshold",
        "sub_min_size",
        "sub_min_agreement",
        "light",
        "single_runs",
        "single_median",
        "single_min",
        "consensus_other",
        "code",
    ):
        assert key in method, key
    for c in GRAPH["clusters"]:
        assert {"agreement", "moved", "subs", "short", "name", "size", "r", "x", "y"} <= set(c), c["id"]


def test_the_page_never_says_the_book_teaches():
    """The map shows or explains what the book holds; "teach" and "taught" are kept off the page and its terms."""
    words = source("map.js") + source("map/index.html") + " ".join(n["definition"] for n in GRAPH["nodes"])
    assert not re.search(r"\bteach|\btaught", words, re.IGNORECASE)
    assert GRAPH["method"]["code"].startswith("https://github.com/kgdunn/pid-book/blob/"), (
        "the clustering code it links"
    )


def test_a_link_to_the_map_shows_the_map():
    """A chat or a blog shows og:image beside a link: for the map, the map's own card, drawn by map_card.py."""
    page = source("map/index.html")
    assert '<meta property="og:image" content="https://factori.al/map/og-card.png">' in page
    assert '<meta name="twitter:card" content="summary_large_image">' in page
    card = (WEB / "map" / "og-card.png").read_bytes()
    assert card.startswith(b"\x89PNG\r\n\x1a\n")
    width, height = struct.unpack(">II", card[16:24])
    assert (width, height) == (1200, 630)
    assert f'<meta property="og:image:width" content="{width}">' in page
    assert f'<meta property="og:image:height" content="{height}">' in page
    assert "map/og-card.png" in APP_SOURCES, "the build publishes it"


def test_each_catalog_family_the_map_has_links_to_its_term():
    """The catalog's design details link to the map by term id: a family must exist, and so must its term."""
    table = re.search(r"const ON_THE_MAP = \{(.*?)\};", source("catalog.js"), re.DOTALL)
    assert table
    pairs = dict(re.findall(r"(\w+): \"(\w+)\"", table.group(1)))
    terms = {n["id"] for n in GRAPH["nodes"]}
    assert pairs, "the catalog links to the map"
    for family, term in pairs.items():
        assert family in FAMILIES, family
        assert term in terms, term


def _oklab(colour: str) -> tuple[float, float, float]:
    """One ``#rrggbb`` colour in OKLab, where equal distances look equally far apart."""
    red, green, blue = (int(colour[i : i + 2], 16) / 255 for i in (1, 3, 5))
    red, green, blue = (c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in (red, green, blue))
    long = (0.4122214708 * red + 0.5363325363 * green + 0.0514459929 * blue) ** (1 / 3)
    medium = (0.2119034982 * red + 0.6806995451 * green + 0.1073969566 * blue) ** (1 / 3)
    short = (0.0883024619 * red + 0.2817188376 * green + 0.6299787005 * blue) ** (1 / 3)
    return (
        0.2104542553 * long + 0.7936177850 * medium - 0.0040720468 * short,
        1.9779984951 * long - 2.4285922050 * medium + 0.4505937099 * short,
        0.0259040371 * long + 0.7827717662 * medium - 0.8086757660 * short,
    )


def test_another_textbook_s_colour_is_no_other_colour_of_the_map():
    """In each theme a book's colour stands apart from a selected term's links, the plain links and the kinds' fills.

    Book B's blue was once a third of ``APART`` from the accent, so its links could not be told from this book's.
    """
    css = source("style.css")
    names = ("map-book2", "map-book3", "accent", "muted", "fg", "map-method", "map-guide", "map-context")
    themes = zip(*(re.findall(rf"--{name}: (#[0-9a-f]{{6}});", css) for name in names), strict=True)
    for second, third, *rest in themes:  # the light theme, then the dark one twice: the device's choice, the reader's
        for book, others in ((second, [third, *rest]), (third, rest)):
            for other in others:
                assert math.dist(_oklab(book), _oklab(other)) >= APART, f"{book} is too close to {other}"
