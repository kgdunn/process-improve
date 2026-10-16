"""The design catalog's files: ``write_catalog`` describes each design for ``catalog.html``, and opens some in the page.

``catalog/index.json`` holds what the catalog page reads of every design at once; a detail file per
source holds what it shows on opening one, with the campaign that opens it in Factorial where the page
makes that kind of design. A published design opens only when the page's design has the source's runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from process_improve.experiments import design_catalog
from process_improve.experiments.reference_catalogue import reference_experiments, reference_sources
from tests.web._support import coded, javascript, no_bytecode, run_sheet, source

#: Published entries and generated designs that show each way a design is opened, or not.
PUBLISHED = ("bhh-pilot-plant-2-3", "bhh-plant-trial-two-methods")
GENERATED = ("lib-ccd-k3-face-centered", "lib-omars-k4")


@pytest.fixture(scope="module")
def few() -> list[dict]:
    """Describe a few designs as ``catalog_records`` does: two published entries and two generated designs."""
    sources = reference_sources()
    entries = [e for e in reference_experiments("box-hunter-hunter-1978") if e["name"] in PUBLISHED]
    published = [
        {
            **design_catalog._published_record(e, sources[e["source"]]),
            "checks": design_catalog.printed_checks(e),
            "listed": "experiments",
        }
        for e in entries
    ]
    generated = [
        {**design_catalog._generated_record(spec), "checks": None, "listed": None}
        for spec in design_catalog.GENERATED
        if design_catalog._spec_id(spec) in GENERATED
    ]
    return [*published, *generated]


@pytest.fixture(scope="module")
def catalog(build_script, few, tmp_path_factory) -> Path:
    """Write the few designs' catalog as the build does, into a folder of their own."""
    out = tmp_path_factory.mktemp("app")
    with pytest.MonkeyPatch.context() as patch, no_bytecode():
        patch.setattr(design_catalog, "catalog_records", lambda: few)
        build_script.write_catalog(out)
    return out / "catalog"


@pytest.fixture(scope="module")
def index(catalog) -> dict:
    return json.loads((catalog / "index.json").read_text())


@pytest.fixture(scope="module")
def details(catalog) -> dict:
    """Every design's detail, read from the detail file the index names for it."""
    held = {}
    for path in (catalog / "detail").glob("*.json"):
        held |= json.loads(path.read_text())["designs"]
    return held


def _design(index: dict, name: str) -> dict:
    return next(d for d in index["designs"] if d["id"] == name)


def test_the_index_holds_every_field_the_page_reads(build_script, index):
    assert index["version"] == build_script.CATALOG_VERSION
    assert index["fields"] == [*design_catalog.INDEX_FIELDS, *build_script.CATALOG_FIELDS]
    assert sorted(d["id"] for d in index["designs"]) == sorted([*PUBLISHED, *GENERATED])
    assert all(list(d) == index["fields"] for d in index["designs"])
    assert set(index["families"]) == set(design_catalog.FAMILIES)
    assert index["sources"]["box-hunter-hunter-1978"]["year"] == 1978


def test_each_design_s_detail_is_in_the_file_the_index_names(catalog, index):
    for design in index["designs"]:
        held = json.loads((catalog / "detail" / f"{design['detail']}.json").read_text())["designs"]
        assert design["id"] in held


def test_the_page_reads_every_campaign_the_build_wrote(node, details):
    """Each campaign passes the page's own check, ``checkCampaign``, as a link to it is checked on opening."""
    campaigns = {name: d["campaign"] for name, d in details.items() if d["campaign"]}
    assert set(campaigns) == {"bhh-pilot-plant-2-3", "lib-ccd-k3-face-centered"}
    checked = javascript(node, source("campaign.js"), [("checkCampaign", [c]) for c in campaigns.values()])
    for (name, campaign), read in zip(campaigns.items(), checked, strict=True):
        assert read["design"]["rows"] == campaign["design"]["rows"], name
        assert read["responses"] == campaign["responses"], name


def test_a_published_design_the_library_regenerates_opens_with_its_results(index, details):
    """The page's 2^3 is Box, Hunter and Hunter's pilot plant: their runs, in their order, with their yields."""
    design, detail = _design(index, "bhh-pilot-plant-2-3"), details["bhh-pilot-plant-2-3"]
    assert (design["open"], design["regenerable"]) == (True, True)
    campaign = detail["campaign"]
    assert campaign["responses"] == [60, 72, 54, 68, 52, 83, 45, 80]
    assert campaign["sources"] == ["entered"] * 8
    runs = coded(run_sheet(campaign["design"]), campaign["design"]["spec"]["factors"])
    assert runs.to_numpy().tolist() == detail["coded_runs"]


def test_a_design_the_page_does_not_make_says_why_it_does_not_open(index, details):
    for name in ("bhh-plant-trial-two-methods", "lib-omars-k4"):
        design = _design(index, name)
        assert (design["open"], design["regenerable"]) == (False, None), name
        assert details[name]["campaign"] is None
        assert details[name]["closed"] == "the page does not make this kind of design"


def test_a_generated_design_opens_in_the_page_s_randomised_run_order(index, details):
    design, detail = _design(index, "lib-ccd-k3-face-centered"), details["lib-ccd-k3-face-centered"]
    assert (design["open"], design["regenerable"]) == (True, None), "a generated design has no source to regenerate"
    campaign = detail["campaign"]
    sheet = run_sheet(campaign["design"])
    assert sorted(sheet["Run order"]) == list(range(1, 18))
    assert sorted(sheet["Std order"]) == list(range(1, 18))
    assert campaign["responses"] == [None] * 17
    runs = coded(sheet, campaign["design"]["spec"]["factors"]).round(6).to_numpy().tolist()
    assert sorted(runs) == sorted(detail["coded_runs"]), "the design drawn is the design opened"


def test_a_published_design_with_other_runs_than_the_page_s_does_not_open(build_script, few):
    """The pilot plant with its first run moved off the cube: the page's 2^3 lacks that run."""
    record = next(r for r in few if r["id"] == "bhh-pilot-plant-2-3")
    moved = {**record, "coded_runs": [[0.5, -1.0, -1.0], *record["coded_runs"][1:]]}
    with no_bytecode():
        campaign, reason, tried = build_script.published_campaign(moved)
    assert campaign is None
    assert tried, "the page made the design and compared its runs"
    assert reason.startswith("the page's design differs from the source's: its run 1")


def test_a_blocked_design_is_not_opened(build_script, few):
    record = next(r for r in few if r["id"] == "bhh-pilot-plant-2-3")
    with no_bytecode():
        campaign, reason, tried = build_script.published_campaign({**record, "blocked": True})
    assert (campaign, tried) == (None, False)
    assert reason == "the page opens designs without blocks or whole plots"


@pytest.mark.slow
@pytest.mark.dataset
def test_the_whole_catalog_opens_every_generated_design_the_page_makes(build_script, tmp_path):
    with no_bytecode():
        build_script.write_catalog(tmp_path)
    designs = json.loads((tmp_path / "catalog" / "index.json").read_text())["designs"]
    makes = set(build_script.load_bootstrap().DESIGNS)
    generated = [d for d in designs if d["collection"] == "generated"]
    assert all(d["open"] for d in generated if d["design_type"] in makes)
    published = [d for d in designs if d["collection"] == "published"]
    assert sum(d["open"] for d in published) >= 150, "most published factorials and response surfaces open"
    assert all(d["open"] == bool(d["regenerable"]) for d in published)
