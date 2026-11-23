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

from process_improve.experiments.compare import MEASURES
from process_improve.experiments.design_catalog import FAMILIES
from tests.web._support import APP_SOURCES, WEB, javascript, source

GRAPH = json.loads((WEB / "map" / "graph.json").read_text(encoding="utf-8"))
APART = 0.15  # in OKLab: closer than this, two colors start to read as one

#: The page's words the map has no term for, and none near: their definitions end without "See it in context".
NO_CONTEXT = {
    *("typical", "true-process", "true-value", "model-terms"),  # the page's own
    *("tukey", "dunnett", "bartlett", "kruskal-wallis", "rank-sum", "signed-rank"),  # tests of a study of one factor
    *("r2", "r2-adjusted", "r2-predicted"),
    *("signal-to-noise", "response-table", "adjustment-factor"),  # Taguchi's
    "cox-direction",
}


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


def test_the_map_s_katex_is_the_published_release_with_its_licence_and_every_font_its_stylesheet_names():
    """``web/vendor/katex/`` is npm's files unchanged: a header names the release and each file's digest."""
    katex = WEB / "vendor" / "katex"
    header, _, script = (katex / "katex.min.js").read_bytes().partition(b" */\n")
    words = header.decode("ascii")
    release = re.search(r"KaTeX (\d+\.\d+\.\d+), https://github\.com/KaTeX/KaTeX", words)
    assert release
    assert f"/katex-{release[1]}.tgz" in words
    assert re.search(r"SHA-256 of dist/katex\.min\.js: ([0-9a-f]{64})", words)[1] == hashlib.sha256(script).hexdigest()
    assert "The MIT License (MIT)" in words
    assert "Copyright (c) 2013-2020 Khan Academy and other contributors" in words

    header, _, sheet = (katex / "katex.min.css").read_bytes().partition(b" */\n")
    words = header.decode("ascii")
    assert f"KaTeX {release[1]}'s stylesheet" in words
    assert f"/katex-{release[1]}.tgz" in words, "the same release as the script"
    assert re.search(r"SHA-256 of dist/katex\.min\.css: ([0-9a-f]{64})", words)[1] == hashlib.sha256(sheet).hexdigest()
    listed = dict(re.findall(r"([0-9a-f]{64})  (fonts/\S+)", words))
    here = {f"fonts/{p.name}": hashlib.sha256(p.read_bytes()).hexdigest() for p in (katex / "fonts").iterdir()}
    assert here == {name: digest for digest, name in listed.items()}, "each font is the published one, and none else"
    named = set(re.findall(r"url\((fonts/[^)]+\.woff2)\)", sheet.decode("utf-8")))
    assert named == set(here), "every font the stylesheet names for a browser of today is here"
    assert {f"vendor/katex/{name}" for name in ("katex.min.js", "katex.min.css", *here)} <= set(APP_SOURCES)

    assert 'import("./vendor/katex/katex.min.js")' in source("map.js"), "fetched when a formula is shown, not before"
    assert '<link rel="stylesheet" href="../vendor/katex/katex.min.css">' in source("map/index.html")
    assert "katex" not in source("app.js"), "never the start page"
    assert "katex" not in source("index.html")


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


def test_the_page_names_three_ways_to_learn_over_the_map_and_the_book_s_doi_as_text():
    """The chapter, the course and the course's videos are links over the map; the book's DOI is text, in no link."""
    page = " ".join(source("map/index.html").split())
    ways = page[page.index('<ul class="learn">') : page.index("</ul>")]
    links = re.findall(r'<li><a href="([^"]+)"[^>]*>([^<]+)</a>', ways)
    assert [name for _, name in links] == ["Read the chapter", "Take the course", "Watch the videos"]
    chapter, course, videos = (address for address, _ in links)
    assert chapter == "https://learnche.org/pid/design-analysis-experiments/index", "the chapter, not the whole book"
    assert course == "https://yint.org/experiment", "the address kept pointing at the course as it is offered now"
    assert "list=PLHUnYbefLmeOPRuT1sukKmRyOVd4WSxJE" in videos, "the first video of the chapter, in the book's playlist"
    assert "index=" not in videos, "a place in the playlist moves when the playlist is reordered; the video does not"
    assert "The same videos as the course" in ways, "so a reader does not take the two for two courses"
    assert "no quizzes" in ways
    doi = re.search(r'<span class="learn__doi">Book DOI: (10\.5281/zenodo\.\d+)</span>', ways)
    assert doi, "the DOI of the book, to read and to copy"
    assert "doi.org" not in page, "and not a link"
    assert page.index("</ul>") < page.index('<h2 class="learn__map">Concept map</h2>') < page.index('id="box"')


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
    """One ``#rrggbb`` color in OKLab, where equal distances look equally far apart."""
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


def test_another_textbook_s_colour_is_no_other_colour_of_the_links():
    """In each theme a book's color stands apart from the accent and from the other colors a link can take.

    While the other textbooks are shown a link is this book's alone (the muted neutral), one other book's, or every
    book's (the text color). Book B's blue was once a third of ``APART`` from the accent, which a selected term's
    links then took.
    """
    css = source("style.css")
    names = ("map-book2", "map-book3", "accent", "muted", "fg")
    themes = zip(*(re.findall(rf"--{name}: (#[0-9a-f]{{6}});", css) for name in names), strict=True)
    for second, third, *rest in themes:  # the light theme, then the dark one twice: the device's choice, the reader's
        for book, others in ((second, [third, *rest]), (third, rest)):
            for other in others:
                assert math.dist(_oklab(book), _oklab(other)) >= APART, f"{book} is too close to {other}"


def test_each_word_s_context_is_a_term_of_the_map_and_the_words_with_none_are_named(node):
    """``glossary.CONTEXT`` gives a word the term "See it in context" opens: the map must have it.

    A word new to the glossary is given its term there, or named in ``NO_CONTEXT``: either way, on purpose.
    """
    glossary, context = javascript(node, source("glossary.js"), [("GLOSSARY", []), ("CONTEXT", [])])
    terms = {term["id"] for term in GRAPH["nodes"]}
    assert not [word for word in context if word not in glossary], "each word is the glossary's"
    assert not {word: term for word, term in context.items() if term not in terms}, "each term is the map's"
    assert set(glossary) - set(context) == NO_CONTEXT


def test_each_design_s_context_is_a_term_of_the_map_and_a_design_the_page_makes(node, bootstrap):
    """``glossary.DESIGN_CONTEXT`` gives a design's name the term "See it in context" opens: the map must have it."""
    (context,) = javascript(node, source("glossary.js"), [("DESIGN_CONTEXT", [])])
    assert set(context) <= {*bootstrap.DESIGNS, *bootstrap.MIXTURES}, "each design is one the page makes"
    assert set(context.values()) <= {term["id"] for term in GRAPH["nodes"]}, "each term is the map's"
    assert {*bootstrap.DESIGNS, *bootstrap.MIXTURES} - set(context) == {"taguchi", "maximin_lhs", "sobol"}, (
        "the designs the map has no term of its own for"
    )


def test_each_measure_s_context_is_a_term_of_the_map_and_a_measure_the_comparison_gives(node):
    """``glossary.MEASURE_CONTEXT`` gives a measure of the comparison in detail its term in the map (#1263)."""
    (context,) = javascript(node, source("glossary.js"), [("MEASURE_CONTEXT", [])])
    keys = {measure.key for measure in MEASURES}
    assert set(context) <= keys, "each key is one of the library's measures"
    assert set(context.values()) <= {term["id"] for term in GRAPH["nodes"]}, "each term is the map's"
    assert keys - set(context) == {"runs", "fits"}, "the measures the map has no term of its own for"


def test_a_word_defined_where_the_page_draws_it_names_a_term_of_the_map():
    """``term(..., { concept })`` in ``app.js`` gives such a word its term, as ``CONTEXT`` does a glossary word."""
    named = set(re.findall(r'concept: "([a-z_]+)"', source("app.js")))
    assert named == {"degrees_of_freedom", "statistical_power"}, "Spare runs, and the power of a design"
    assert named <= {term["id"] for term in GRAPH["nodes"]}
