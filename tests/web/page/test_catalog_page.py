"""The catalog page (``web/catalog/index.html``): what it reads, links to and copies must hold of what the build writes.

``catalog-logic.js`` names the fields it reads from ``catalog/index.json``; the build writes them.
The page is served from a folder of its own, so its address is factori.al/catalog/. It loads only
files the build publishes, at addresses relative to itself or to ``catalog.js``, so it works on
factori.al and on the copy under GitHub Pages. The Python it copies for a design makes that design.
"""

from __future__ import annotations

import contextlib
import io
import json
import posixpath
import re
from collections import Counter

import numpy as np
import pytest

from process_improve.experiments import design_catalog
from process_improve.experiments.reference_catalogue import (
    reference_designs,
    reference_experiments,
    reference_sources,
)
from tests.web._support import javascript, source

LOGIC = "catalog-logic.js"
#: The catalog page, in a folder of its own: factori.al/catalog/ serves it.
PAGE = "catalog/index.html"


def test_the_page_reads_the_fields_and_format_the_build_writes(node, build_script):
    fields, version = javascript(node, source(LOGIC), [("INDEX_FIELDS", []), ("CATALOG_VERSION", [])])
    assert fields == [*design_catalog.INDEX_FIELDS, *build_script.CATALOG_FIELDS]
    assert version == build_script.CATALOG_VERSION


def test_every_family_the_map_colours_is_one_of_the_catalog_s(node):
    (colours,) = javascript(node, source(LOGIC), [("FAMILY_COLOURS", [])])
    assert set(colours) <= set(design_catalog.FAMILIES)


def test_every_file_the_catalog_loads_is_published(build_script):
    """What the page and catalog.js load is copied by the build or written by it, at a relative address.

    The page's own references are read from its folder, ``catalog/``; catalog.js takes every address
    it fetches from its own, at the top of the app, so it needs no ``../``.
    """
    refs = set(re.findall(r'(?:href|src)="([^"]+)"', source(PAGE)))
    refs = {ref for ref in refs if "://" not in ref and not ref.startswith("#")}
    loaded = {posixpath.normpath(posixpath.join("catalog", ref)) for ref in refs}
    imported = set(re.findall(r'from "\./([^"]+)"', source("catalog.js")))
    fetched = set(re.findall(r'fetch\(app\(`?"?([\w./-]+)', source("catalog.js")))
    assert {"style.css", "catalog.js", "."} <= loaded, "the page's stylesheet, its script, and the app"
    assert {"campaign.js", "charts.js", "catalog-logic.js", "catalog-views.js"} <= imported
    assert set(re.findall(r'from "\./([^"]+)"', source("catalog-views.js"))) == {"catalog-logic.js"}
    assert fetched == {"catalog/index.json", "wheels/manifest.json", "catalog/detail/", "catalog/errata.json"}
    assert source("catalog.js").count("fetch(") == len(fetched), "every fetch goes through app()"
    assert not any(ref.startswith("/") for ref in refs | fetched), "an absolute address breaks on GitHub Pages"
    assert (loaded - {"."}) | imported <= set(build_script.APP_FILES)
    assert "app.js" not in imported, "importing app.js would start the app's Python worker"


def test_the_app_links_to_the_catalog_and_the_catalog_back():
    assert 'href="catalog/"' in source("index.html")
    assert '<a href="../">' in source(PAGE)


def test_the_catalog_names_its_own_address_for_a_link_card():
    page = source(PAGE)
    assert '<link rel="canonical" href="https://factori.al/catalog/">' in page
    assert '<meta property="og:url" content="https://factori.al/catalog/">' in page
    assert "feed.xml" not in page, "the build writes the feed of new sources, and the page does not link to it"


def _run(code: str) -> dict:
    """Run the Python the page copies, as a reader would paste it; return the names it defines."""
    names: dict = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(code, names)  # noqa: S102 - the page's own code, which the reader runs
    return names


@pytest.mark.parametrize("name", ["lib-ccd-k3-face-centered", "lib-fractional-factorial-k6-res4", "lib-dsd-k5"])
def test_the_python_copied_for_a_generated_design_makes_that_design(node, name):
    spec = next(s for s in design_catalog.GENERATED if design_catalog._spec_id(s) == name)
    record = design_catalog._generated_record(spec)
    detail = {"spec": record["spec"], "factors": record["factors"]}
    (code,) = javascript(node, source(LOGIC), [("pythonCode", [{"id": name}, detail])])
    made = _run(code)["design"]
    names = [f["name"] for f in record["factors"]]
    runs = Counter(tuple(round(v, 6) for v in row) for row in made.design[names].to_numpy().tolist())
    assert runs == Counter(tuple(row) for row in record["coded_runs"])


def _published_code(node, name: str, source_key: str, listed: str) -> str:
    """Return the Python the page copies for a published design, from its record as the catalog's build writes it."""
    entries = reference_designs(source_key) if listed == "designs" else reference_experiments(source_key)
    entry = next(e for e in entries if e["name"] == name)
    record = design_catalog._published_record(entry, reference_sources()[source_key])
    detail = {key: record[key] for key in ("factors", "coded_runs", "response_values")} | {"listed": listed}
    (code,) = javascript(node, source(LOGIC), [("pythonCode", [{"id": name, "source": source_key}, detail])])
    return code


def test_the_python_copied_for_a_published_design_reads_its_catalogue_entry(node):
    code = _published_code(node, "bhh-pilot-plant-2-3", "box-hunter-hunter-1978", "experiments")
    names = _run(code)
    assert names["entry"]["name"] == "bhh-pilot-plant-2-3"
    assert names["table"]["y"].iloc[0] == 60, "the runs as the source prints them, results and all"
    assert list(names["factors"]["name"]) == ["T", "C", "K"]
    assert "next(" not in code, "one step a line"
    assert max(map(len, code.splitlines())) <= 80, "no line wider than the box shows"


@pytest.mark.parametrize(
    ("name", "source_key", "runs"),
    [
        ("nunez-ares-2023-design-2-oa", "nunez-ares-et-al-2023", 48),
        ("bhh-blocking-2k-factorials", "box-hunter-hunter-1978", 0),
    ],
)
def test_the_python_copied_for_a_catalogue_design_runs_with_or_without_printed_runs(node, name, source_key, runs):
    """A design the source prints as a table gives its runs; one it only describes prints what the source says."""
    names = _run(_published_code(node, name, source_key, "designs"))
    assert names["entry"]["name"] == name
    assert len(names["table"]) == runs if runs else "table" not in names


#: Applications chosen in the page's filter, each compared with what ``search_catalog`` keeps for them.
CHOSEN_TOPICS = (["food-beverages"], ["semiconductors", "chemical-processes"], ["general"], ["pharmaceuticals"])


@pytest.mark.slow
@pytest.mark.dataset
def test_every_search_the_page_suggests_finds_designs_and_its_filters_agree_with_python(node, monkeypatch):
    """The searches beside the box, run over the whole catalog as the page runs them, each find some designs.

    The page's application filter keeps the designs ``search_catalog`` keeps, and a word of an
    application's name finds its designs on the page and in Python alike.
    """
    tries = re.findall(r'<button type="button" class="link try">([^<]+)</button>', source(PAGE))
    placeholder = re.search(r'placeholder="([^"]+)"', source(PAGE)).group(1).replace("&lt;", "<")
    assert len(tries) >= 3
    records = design_catalog.catalog_records()
    designs = [
        {key: r[key] for key in design_catalog.INDEX_FIELDS} | {"open": False, "regenerable": None} for r in records
    ]
    catalog = {
        "designs": designs,
        "families": design_catalog.catalog_families(),
        "sources": design_catalog.catalog_sources(),
        "topics": design_catalog.catalog_topics(),
    }
    base = javascript(node, source(LOGIC), [("DEFAULT_STATE", [])])[0]
    read = json.loads(json.dumps(catalog))
    words = {"semiconductors": "semiconductors", "food": "food-beverages"}
    asked = [
        *({**base, "query": query} for query in [*tries, placeholder]),
        *({**base, "topics": topics} for topics in CHOSEN_TOPICS),
        *({**base, "query": word} for word in words),
    ]
    found = [
        {d["id"] for d in result} for result in javascript(node, source(LOGIC), [("results", [read, a]) for a in asked])
    ]
    searched, filtered, by_word = found[: len(tries) + 1], found[len(tries) + 1 : -len(words)], found[-len(words) :]
    for query, designs_found in zip([*tries, placeholder], searched, strict=True):
        assert designs_found, f"the search {query!r} finds nothing"

    monkeypatch.setattr(design_catalog, "catalog_records", lambda: records)  # search_catalog reads the same records
    design_catalog._searchable.cache_clear()
    try:
        for topics, on_page in zip(CHOSEN_TOPICS, filtered, strict=True):
            assert on_page, f"the application {topics} keeps nothing"
            assert on_page == {d["id"] for d in design_catalog.search_catalog(topics=topics)}, topics
        for (word, topic), on_page in zip(words.items(), by_word, strict=True):
            listed = {d["id"] for d in design_catalog.search_catalog(topics=[topic])}
            assert listed <= on_page, f"the word {word!r} misses designs of {topic} on the page"
            assert listed <= {d["id"] for d in design_catalog.search_catalog(text=word)}, word
    finally:
        design_catalog._searchable.cache_clear()


@pytest.mark.parametrize(
    "limit",
    ["A + B <= 1", "-0.5 <= A - 2*C <= 0.5", "A^2 + B^2 <= 1", "(A + C) / 2 >= -0.25", "A*B < 0.3"],
)
def test_a_limit_opened_in_factorial_rules_out_the_runs_the_catalog_marks(bootstrap, node, limit):
    """The catalog writes a limit on the coded settings in a design's own units; Factorial must read it alike.

    The page's Python reads the text the catalog writes, as it reads a limit typed in the brief, and its
    inequalities on the design's coded runs rule out exactly the runs the catalog marks.
    """
    factors = [
        {"name": "T", "low": 150, "high": 200},
        {"name": "P", "low": 1, "high": 3},
        {"name": "F", "low": -2, "high": 2},
    ]
    runs = [[a, b, c] for a in (-1, 0, 1) for b in (-1, 0.5, 1) for c in (-1, 0, 1.68)]
    logic = source(LOGIC)
    (parsed,) = javascript(node, logic, [("parseLimit", [limit])])
    text, outcome = javascript(
        node, logic, [("limitInUnits", [parsed, factors]), ("limitOutcome", [parsed, runs, ["numeric"] * 3])]
    )
    held = bootstrap._factors({"factors": factors})
    canonical, _ = bootstrap._read_limit(text, held)
    region = bootstrap._region(held, [canonical])
    points = np.array(runs, dtype=float)[:, [["T", "P", "F"].index(n) for n in region.names]]
    meets = np.all([g(points) <= bootstrap.LIMIT_TOLERANCE for g in region.inequalities], axis=0)
    assert np.flatnonzero(~meets).tolist() == outcome["outside"], text


def test_the_catalog_links_to_the_form_for_proposing_a_design():
    """The link names an issue form the repository holds, which asks for the source, the runs and where they are."""
    import yaml

    from tests.web._support import WEB

    form = re.search(r"issues/new\?template=([\w.]+)", source(PAGE))
    assert form, "the catalog links to the form"
    held = yaml.safe_load((WEB.parent / ".github" / "ISSUE_TEMPLATE" / form[1]).read_text(encoding="utf-8"))
    required = {item["id"] for item in held["body"] if item.get("validations", {}).get("required")}
    assert {"source", "where", "factors", "runs"} <= required
