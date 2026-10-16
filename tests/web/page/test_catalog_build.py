"""The design catalog's files: ``write_catalog`` describes each design for the catalog page, and opens some in the app.

``catalog/index.json`` holds what the catalog page reads of every design at once; a detail file per
source holds what it shows on opening one, with the campaign that opens it in Factorial where the page
makes that kind of design. A published design opens only when the page's design has the source's runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from process_improve.experiments import design_catalog
from tests.web._support import CATALOG_SAMPLE, catalog_sample, coded, javascript, no_bytecode, run_sheet, source

PUBLISHED, GENERATED = CATALOG_SAMPLE[:2], CATALOG_SAMPLE[2:]


@pytest.fixture(scope="module")
def few() -> list[dict]:
    """Describe a few designs as ``catalog_records`` does: two published entries and two generated designs."""
    return catalog_sample()


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
    assert set(campaigns) == {
        "bhh-pilot-plant-2-3",
        "lib-ccd-k3-face-centered",
        "lib-full-factorial-k3",
        "lib-omars-k4",
    }
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


def test_a_design_the_page_does_not_make_opens_with_its_runs_as_given(index, details):
    """The page makes no OMARS design: Factorial opens the library's runs as they are, in their order."""
    design, detail = _design(index, "lib-omars-k4"), details["lib-omars-k4"]
    assert (design["open"], design["regenerable"]) == (True, None)
    assert (detail["opens_as"], detail["unmade"], detail["closed"]) == (
        "given",
        "the page does not make this kind of design",
        "",
    )
    campaign = detail["campaign"]
    assert campaign["design"]["spec"]["design_type"] == "given"
    runs = coded(run_sheet(campaign["design"]), campaign["design"]["spec"]["factors"])
    assert runs.to_numpy().tolist() == detail["coded_runs"]


def test_a_design_the_page_cannot_take_says_why_it_does_not_open(index, details):
    design, detail = _design(index, "bhh-plant-trial-two-methods"), details["bhh-plant-trial-two-methods"]
    assert (design["open"], detail["campaign"], detail["opens_as"]) == (False, None, None)
    assert detail["closed"] == "the page opens designs of 2 to 12 factors"


def test_a_design_with_runs_the_page_makes_otherwise_opens_as_printed(build_script, few):
    """The pilot plant with its first run moved: the page's 2^3 lacks that run, so the source's runs open as given."""
    record = next(r for r in few if r["id"] == "bhh-pilot-plant-2-3")
    moved = {**record, "coded_runs": [[0.5, -1.0, -1.0], *record["coded_runs"][1:]]}
    with no_bytecode():
        campaign, reason = build_script.given_campaign(moved)
    assert reason == ""
    assert campaign["design"]["spec"]["design_type"] == "given"
    assert campaign["responses"] == [60, 72, 54, 68, 52, 83, 45, 80]
    assert campaign["design"]["rows"][0][2:5] == [175.0, 20.0, "A"], "the moved run, in the factors' own units"


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"blocked": True}, "the page opens designs without blocks or whole plots"),
        ({"coded_runs": None}, "the source prints no complete table of runs"),
        ({"n_mixture": 3}, "the page has no model for a mixture's proportions"),
        ({"coded_runs": [[0.0, -1.0, 0.0]]}, "a categorical factor has more than the two levels the page takes"),
    ],
    ids=["blocked", "no-runs", "mixture", "a-third-level"],
)
def test_runs_the_page_cannot_take_are_not_given(build_script, few, change, reason):
    record = next(r for r in few if r["id"] == "bhh-pilot-plant-2-3")
    with no_bytecode():
        assert build_script.given_campaign({**record, **change}) == (None, reason)


@pytest.mark.parametrize("blank", ["None", " NA "])
def test_a_label_a_spreadsheet_reads_as_blank_is_not_opened(build_script, few, blank):
    """Montgomery's Problem 8.30 has stems at None and All: the page refuses the label, so neither path tries it."""
    record = next(r for r in few if r["id"] == "bhh-pilot-plant-2-3")
    factors = [*record["factors"][:2], {**record["factors"][2], "low": blank, "high": "All"}]
    with no_bytecode():
        assert build_script.given_campaign({**record, "factors": factors}) == (
            None,
            "a spreadsheet reads a label such as None as a blank cell",
        )
        campaign, reason, tried = build_script.published_campaign({**record, "factors": factors})
    assert (campaign, tried) == (None, False), "the page is not asked for a design it refuses"
    assert reason == "a factor's name, type or range is not one the page accepts"


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
    bootstrap = build_script.load_bootstrap()
    makes = set(bootstrap.DESIGNS)
    generated = [d for d in designs if d["collection"] == "generated"]
    # An orthogonal array is made whole, a factor to each column: the page takes the arrays of up to 12 columns.
    most = bootstrap.GIVEN_INFO["max_factors"]
    assert all(d["open"] for d in generated if d["design_type"] in makes and d["n_factors"] <= most)
    detail = json.loads((tmp_path / "catalog" / "detail" / "generated.json").read_text())["designs"]
    made = {name for name, d in detail.items() if name.startswith("lib-taguchi-") and d["opens_as"] == "made"}
    assert made == {"lib-taguchi-l4-2-3", "lib-taguchi-l8-2-7", "lib-taguchi-l9-3-4", "lib-taguchi-l12-2-11"}
    published = [d for d in designs if d["collection"] == "published"]
    assert sum(d["open"] for d in published) >= 300, "a published design with a complete table of runs opens"
    assert all(d["open"] for d in published if d["regenerable"])


def _taguchi_record(array: str) -> dict:
    """Describe the library's orthogonal array ``array``, whole, as ``catalog_records`` does."""
    spec = next(s for s in design_catalog.GENERATED if s.get("array") == array)
    return {**design_catalog._generated_record(spec), "checks": None, "listed": None}


@pytest.mark.parametrize(
    ("array", "options"), [("L8(2^7)", {}), ("L9(3^4)", {"n_levels": 3})], ids=["two levels", "three levels"]
)
def test_an_orthogonal_array_the_page_makes_opens_as_made(build_script, array, options):
    """The page makes the array for as many factors as it has columns, at the levels its columns have."""
    record = _taguchi_record(array)
    with no_bytecode():
        campaign, reason = build_script.generated_campaign(record)
    assert reason == ""
    spec = campaign["design"]["spec"]
    assert (spec["design_type"], spec["options"]) == ("taguchi", {"random_state": 42, **options})
    runs = coded(run_sheet(campaign["design"]), spec["factors"]).round(6).to_numpy().tolist()
    assert sorted(runs) == sorted(record["coded_runs"]), "the array drawn is the array opened"


@pytest.mark.parametrize(
    ("array", "reason"),
    [
        ("L16(4^5)", "the page runs every numeric factor of a Taguchi array at 2 levels, or every one at 3"),
        ("L16(2^15)", "the page makes this design for 2 to 12 factors"),
    ],
    ids=["four levels", "fifteen columns"],
)
def test_an_orthogonal_array_the_page_does_not_make_says_why(build_script, array, reason):
    with no_bytecode():
        assert build_script.generated_campaign(_taguchi_record(array)) == (None, reason)


def test_a_published_taguchi_array_at_three_levels_opens_as_made(build_script):
    """Schmidt and Launsby's L9 is the library's: the page makes it with its numeric factors at three levels."""
    from process_improve.experiments.reference_catalogue import reference_designs, reference_sources

    source = reference_sources()["schmidt-launsby-2000"]
    entry = next(e for e in reference_designs("schmidt-launsby-2000") if e["name"] == "schmidt-taguchi-l9")
    record = {**design_catalog._published_record(entry, source), "checks": None, "listed": "designs"}
    with no_bytecode():
        campaign, reason, tried = build_script.published_campaign(record)
    assert (reason, tried) == ("", True)
    assert campaign["design"]["spec"]["options"]["n_levels"] == 3
    runs = coded(run_sheet(campaign["design"]), campaign["design"]["spec"]["factors"])
    assert runs.round(6).to_numpy().tolist() == record["coded_runs"], "the source's runs, in its order"


def test_the_index_names_the_measures_designs_are_compared_on(index):
    from process_improve.experiments.compare import MEASURES

    assert [m["key"] for m in index["measures"]] == [m.key for m in MEASURES]
    assert {"label", "direction", "group", "description"} <= set(index["measures"][0])


def test_a_design_of_numeric_factors_carries_its_measures_for_each_model(index, details):
    ccd = details["lib-ccd-k3-face-centered"]["compare"]
    assert _design(index, "lib-ccd-k3-face-centered")["compared"] is True
    assert set(ccd) == {"main_effects", "interactions", "quadratic"}
    assert ccd["quadratic"]["fits"] is True
    assert len(ccd["quadratic"]["fds"]) == 21
    cube = details["lib-full-factorial-k3"]["compare"]
    assert (cube["quadratic"]["fits"], cube["quadratic"]["fds"]) == (False, None)
    two_methods = _design(index, "bhh-plant-trial-two-methods")
    assert (two_methods["compared"], details[two_methods["id"]]["compare"]) == (False, None)


def test_the_errata_and_the_feed_are_written_beside_the_index(catalog):
    import xml.etree.ElementTree as ET

    errata = json.loads((catalog / "errata.json").read_text())
    assert errata["version"] == 1
    assert any(e["id"] == "bhh-plant-trial-two-methods" for e in errata["errata"])
    feed = ET.fromstring((catalog / "feed.xml").read_bytes())  # noqa: S314 - the feed the build just wrote
    entries = feed.findall("{http://www.w3.org/2005/Atom}entry")
    assert [e.find("{http://www.w3.org/2005/Atom}id").text for e in entries] == [
        "https://factori.al/catalog/#src=box-hunter-hunter-1978"
    ]
