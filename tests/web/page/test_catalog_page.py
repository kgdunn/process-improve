"""The catalog page (``web/catalog.html``): what it reads, links to and copies must hold of what the build writes.

``catalog-logic.js`` names the fields it reads from ``catalog/index.json``; the build writes them.
The page loads only files the build publishes, at addresses relative to itself, so it works on
factori.al and on the copy under GitHub Pages. The Python it copies for a design makes that design.
"""

from __future__ import annotations

import contextlib
import io
import json
import re
from collections import Counter

import pytest

from process_improve.experiments import design_catalog
from tests.web._support import javascript, source

LOGIC = "catalog-logic.js"


def test_the_page_reads_the_fields_and_format_the_build_writes(node, build_script):
    fields, version = javascript(node, source(LOGIC), [("INDEX_FIELDS", []), ("CATALOG_VERSION", [])])
    assert fields == [*design_catalog.INDEX_FIELDS, *build_script.CATALOG_FIELDS]
    assert version == build_script.CATALOG_VERSION


def test_every_family_the_map_colours_is_one_of_the_catalog_s(node):
    (colours,) = javascript(node, source(LOGIC), [("FAMILY_COLOURS", [])])
    assert set(colours) <= set(design_catalog.FAMILIES)


def test_every_file_the_catalog_loads_is_published(build_script):
    """What catalog.html and catalog.js load is copied by the build or written by it, at a relative address."""
    loaded = set(re.findall(r'(?:href|src)="([^"]+)"', source("catalog.html")))
    loaded = {ref for ref in loaded if "://" not in ref and not ref.startswith("#")}
    imported = set(re.findall(r'from "\./([^"]+)"', source("catalog.js")))
    fetched = set(re.findall(r'fetch\(`?"?([\w./-]+)', source("catalog.js")))
    assert {"style.css", "catalog.js", "./"} <= loaded
    assert {"campaign.js", "charts.js", "catalog-logic.js"} <= imported
    assert fetched == {"catalog/index.json", "wheels/manifest.json", "catalog/detail/"}
    assert not any(ref.startswith("/") for ref in loaded | fetched), "an absolute address breaks on GitHub Pages"
    assert (loaded - {"./"}) | imported <= set(build_script.APP_FILES)
    assert "app.js" not in imported, "importing app.js would start the app's Python worker"


def test_the_app_links_to_the_catalog_and_the_catalog_back():
    assert 'href="catalog.html"' in source("index.html")
    assert '<a href="./">' in source("catalog.html")


def test_the_catalog_names_its_own_address_for_a_link_card():
    page = source("catalog.html")
    assert '<link rel="canonical" href="https://factori.al/catalog.html">' in page
    assert '<meta property="og:url" content="https://factori.al/catalog.html">' in page


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


def test_the_python_copied_for_a_published_design_reads_its_catalogue_entry(node):
    design = {"id": "bhh-pilot-plant-2-3", "source": "box-hunter-hunter-1978"}
    (code,) = javascript(node, source(LOGIC), [("pythonCode", [design, {"listed": "experiments"}])])
    entry = _run(code)["entry"]
    assert entry["name"] == "bhh-pilot-plant-2-3"
    assert entry["runs"]["rows"][0][-1] == 60


@pytest.mark.slow
@pytest.mark.dataset
def test_every_search_the_page_suggests_finds_designs(node, build_script):
    """The searches beside the box, run over the whole catalog as the page runs them, each find some designs."""
    tries = re.findall(r'<button type="button" class="link try">([^<]+)</button>', source("catalog.html"))
    placeholder = re.search(r'placeholder="([^"]+)"', source("catalog.html")).group(1).replace("&lt;", "<")
    assert len(tries) >= 3
    records = design_catalog.catalog_records()
    designs = [
        {key: r[key] for key in design_catalog.INDEX_FIELDS} | {"open": False, "regenerable": None} for r in records
    ]
    catalog = {
        "designs": designs,
        "families": design_catalog.catalog_families(),
        "sources": design_catalog.catalog_sources(),
    }
    base = javascript(node, source(LOGIC), [("DEFAULT_STATE", [])])[0]
    found = javascript(
        node,
        source(LOGIC),
        [("results", [json.loads(json.dumps(catalog)), {**base, "query": query}]) for query in [*tries, placeholder]],
    )
    for query, designs_found in zip([*tries, placeholder], found, strict=True):
        assert designs_found, f"the search {query!r} finds nothing"
