"""The concept map's files, checked without a browser: the vendored Cytoscape, and the graph the page draws.

``web/map/graph.json`` is built and checked against the book in pid-book's ``knowledge-graph/``; these tests catch a
copy that arrives incomplete. How the page behaves is tested in ``tests/web/e2e/test_concept_map.py``.
"""

from __future__ import annotations

import hashlib
import json
import re

from tests.web._support import WEB, source

GRAPH = json.loads((WEB / "map" / "graph.json").read_text(encoding="utf-8"))


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
