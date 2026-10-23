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
from tests.web._support import (
    BOOK_DESIGN,
    CATALOG_SAMPLE,
    catalog_sample,
    coded,
    javascript,
    no_bytecode,
    run_sheet,
    source,
)

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


def test_the_index_names_each_application_and_each_design_lists_its_own(index):
    """The page's Application filter reads each topic's name and description from the index."""
    assert index["topics"] == design_catalog.catalog_topics()
    assert index["topics"]["general"]["label"] == design_catalog.GENERAL_LABEL
    assert _design(index, "bhh-pilot-plant-2-3")["topics"] == ["chemical-processes"]
    assert all(_design(index, name)["topics"] == [] for name in GENERATED), "a generated design has no application"


@pytest.fixture(scope="module")
def book_catalog(build_script, tmp_path_factory) -> Path:
    """Write the catalog of one design the book prints, as the build does."""
    out = tmp_path_factory.mktemp("book")
    book = catalog_sample((BOOK_DESIGN,))
    with pytest.MonkeyPatch.context() as patch, no_bytecode():
        patch.setattr(design_catalog, "catalog_records", lambda: book)
        build_script.write_catalog(out)
    return out / "catalog"


def test_a_book_design_s_place_is_its_section_and_named_figures_with_the_section_s_address(book_catalog):
    index = json.loads((book_catalog / "index.json").read_text())
    detail = json.loads((book_catalog / "detail" / "pid-book.json").read_text())["designs"][BOOK_DESIGN]
    assert detail["where_url"] == (
        "https://learnche.org/pid/design-analysis-experiments/omars-worked-study#doe-omars-worked-study"
    )
    assert detail["where"]["section"] == "A worked OMARS study: four factors on a fed-batch bioreactor"
    assert "The thirty batches in run order" in detail["where"]["figures"]
    assert set(detail["where"]) == {"section", "tables", "figures"}, "the source file, lines and commit stay out"
    shown = json.dumps(detail["where"]) + _design(index, BOOK_DESIGN)["where"]
    assert ".png" not in shown
    assert "../" not in shown
    assert _design(index, BOOK_DESIGN)["topics"] == ["bioprocessing"]


def test_a_design_printed_in_a_book_has_no_address_for_its_place(details):
    assert details["bhh-pilot-plant-2-3"]["where_url"] is None
    assert details["bhh-pilot-plant-2-3"]["where"]["pages"] == "307-324, 342-344"
    assert details["bhh-pilot-plant-2-3"]["where"]["figures"] == ["Figure 10.1", "Figure 10.2", "Figure 10.4"]
    assert details["lib-ccd-k3-face-centered"]["where_url"] is None


def test_each_design_s_detail_is_in_the_file_the_index_names(catalog, index):
    for design in index["designs"]:
        held = json.loads((catalog / "detail" / f"{design['detail']}.json").read_text())["designs"]
        assert design["id"] in held


def test_the_page_reads_every_campaign_the_build_wrote(node, details):
    """Each campaign passes the page's own check, ``checkCampaign``, as a link to it is checked on opening."""
    campaigns = {name: d["campaign"] for name, d in details.items() if d["campaign"]}
    assert set(campaigns) == {
        "bhh-pilot-plant-2-3",
        "bhh-plant-trial-two-methods",
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


def test_a_comparison_of_two_treatments_opens_with_its_runs_as_given(index, details):
    """Box, Hunter and Hunter's plant trial: ten batches by method A, then ten by B, in that order, with yields."""
    design, detail = _design(index, "bhh-plant-trial-two-methods"), details["bhh-plant-trial-two-methods"]
    assert (design["open"], detail["opens_as"], detail["closed"]) == (True, "given", "")
    campaign = detail["campaign"]
    assert run_sheet(campaign["design"])["method"].tolist() == ["A"] * 10 + ["B"] * 10
    assert campaign["responses"][:3] == [89.7, 81.4, 84.5]
    assert campaign["design"]["facts"]["notes"][1] == "The runs compare method at A and at B: 10 runs at each."


def test_a_design_the_page_cannot_take_says_why_it_does_not_open(build_script, few):
    """The plant trial said to be run in blocks, with no run's block printed: the page cannot place the runs."""
    record = next(r for r in few if r["id"] == "bhh-plant-trial-two-methods")
    with no_bytecode():
        campaign, reason = build_script.given_campaign({**record, "blocked": True})
    assert (campaign, reason) == (None, "the source runs the design in blocks, but does not print each run's block")


def test_a_design_opens_with_each_run_s_block_as_its_source_prints_it(build_script, few):
    """The pilot plant's eight runs in two days: the run sheet has each run's block, and the day is no factor."""
    record = next(r for r in few if r["id"] == "bhh-pilot-plant-2-3")
    with no_bytecode():
        campaign, reason = build_script.given_campaign({**record, "blocked": True, "blocks": ["2", "1"] * 4})
    assert reason == ""
    sheet = run_sheet(campaign["design"])
    assert sheet["Block"].tolist() == [2, 1] * 4, "numbered blocks keep their numbers"
    assert campaign["design"]["spec"]["options"]["n_blocks"] == 2
    assert [f["name"] for f in campaign["design"]["spec"]["factors"]] == [f["name"] for f in record["factors"]]


def test_runs_as_given_take_more_blocks_than_a_design_the_page_makes(build_script, few):
    """The pilot plant's eight runs as four pairs: blocks of two, as a paired comparison's, open; 33 blocks do not."""
    record = next(r for r in few if r["id"] == "bhh-pilot-plant-2-3")
    bootstrap = build_script.load_bootstrap()
    assert bootstrap.MAX_GIVEN_BLOCKS > bootstrap.MAX_BLOCKS
    many = {**record, "blocked": True, "blocks": list(range(1, bootstrap.MAX_GIVEN_BLOCKS + 2))}
    with no_bytecode():
        assert build_script._not_given(many) == f"the page takes runs in at most {bootstrap.MAX_GIVEN_BLOCKS} blocks"
        campaign, reason = build_script.given_campaign({**record, "blocked": True, "blocks": [1, 1, 2, 2, 3, 3, 4, 4]})
    assert reason == ""
    assert campaign["design"]["spec"]["options"]["n_blocks"] == 4


@pytest.mark.parametrize(
    ("blocks", "numbers"),
    [
        ([3, 1, "2", 1.0], [3, 1, 2, 1]),
        (["Tuesday", "Wednesday", "Tuesday", "Friday"], [1, 2, 1, 3]),
        (["II", "I", "II"], [1, 2, 1]),
    ],
    ids=["numbers", "days", "named"],
)
def test_blocks_are_numbered_by_their_numbers_else_as_first_printed(build_script, blocks, numbers):
    with no_bytecode():
        assert build_script._block_numbers(blocks) == numbers


@pytest.mark.dataset
def test_a_block_listed_as_a_factor_opens_as_the_page_s_blocks(build_script):
    """Montgomery's 3^2 in three blocks lists the block as a factor: it opens as a 3^2, in its three blocks."""
    (record,) = catalog_sample(("montgomery-3-2-in-three-blocks",))
    assert len(record["factors"]) == 2, "the block is no factor"
    with no_bytecode():
        campaign, reason = build_script.given_campaign(record)
    assert reason == ""
    sheet = run_sheet(campaign["design"])
    assert sorted(set(sheet["Block"])) == [1, 2, 3]
    assert sheet.groupby("Block").size().tolist() == [3, 3, 3]


@pytest.mark.dataset
def test_an_entry_that_is_no_design_says_so_and_a_fraction_named_by_its_generators_opens(build_script):
    """Montgomery's Table 7.9 and Problem 2.8 hold no design; Table X's 2^(5-2) and BHH's cast films are fractions."""
    records = {
        r["id"]: r
        for r in catalog_sample(
            (
                "montgomery-blocking-arrangements",
                "montgomery-problem-2-8",
                "montgomery-table-x-2-5-2",
                "bhh-cast-films-2-5-2",
            )
        )
    }
    with no_bytecode():
        facts = build_script.given_campaign(records["montgomery-blocking-arrangements"])
        sample = build_script.given_campaign(records["montgomery-problem-2-8"])
        table_x, reason, _ = build_script.published_campaign(records["montgomery-table-x-2-5-2"])
        films, why = build_script.given_campaign(records["bhh-cast-films-2-5-2"])
    assert facts == (None, "the source prints facts about designs here, not the runs of one")
    assert sample == (None, "nothing is varied: the runs measure one sample, with no factor to set")
    assert (reason, table_x["design"]["n_runs"]) == ("", 8), "the page makes D = AB, E = AC itself"
    assert why == ""
    cast = records["bhh-cast-films-2-5-2"]
    assert cast["runs_from"] == ["D=BC", "E=ABC"]
    sheet = run_sheet(films["design"])
    assert sheet[list("ABCDE")].to_numpy().tolist() == cast["coded_runs"], "the library's runs, in standard order"


def test_a_range_written_high_first_opens_low_to_high_with_the_source_s_settings(build_script, few):
    """The pilot plant's temperature written 180 to 160, as a source codes its larger setting -1: the runs keep it."""
    record = next(r for r in few if r["id"] == "bhh-pilot-plant-2-3")
    temperature = {**record["factors"][0], "low": 180, "high": 160}
    with no_bytecode():
        campaign, reason = build_script.given_campaign({**record, "factors": [temperature, *record["factors"][1:]]})
    assert reason == ""
    page = campaign["design"]["spec"]["factors"][0]
    assert (page["low"], page["high"]) == (160, 180)
    settings = run_sheet(campaign["design"])[temperature["name"]].tolist()
    assert settings == [180 if row[0] < 0 else 160 for row in record["coded_runs"]], "-1 is the source's 180"


def test_a_range_written_high_first_keeps_each_result_at_its_setting_in_the_design_the_page_makes(build_script, few):
    """The same temperature on the 2^3 the page makes, which codes 180 as +1: each yield stays at its temperature."""
    record = next(r for r in few if r["id"] == "bhh-pilot-plant-2-3")
    temperature = {**record["factors"][0], "low": 180, "high": 160}
    with no_bytecode():
        campaign, reason, _ = build_script.published_campaign(
            {**record, "factors": [temperature, *record["factors"][1:]]}
        )
    assert reason == "", "the page makes the design"
    settings = run_sheet(campaign["design"])[temperature["name"]].tolist()
    printed = record["response_values"][build_script._response_of(record)]
    expected = [(180 if row[0] < 0 else 160, y) for row, y in zip(record["coded_runs"], printed, strict=True)]
    assert sorted(zip(settings, campaign["responses"], strict=True)) == sorted(expected), "each yield at its 180 or 160"


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
        ({"blocked": True}, "the source runs the design in blocks, but does not print each run's block"),
        (
            {"blocked": True, "blocks": [1, 2, 3, None, 1, 2, 3, 1]},
            "the source runs the design in blocks, but does not print each run's block",
        ),
        (
            {"blocked": True, "blocks": [1, 2, 3, 4, 5, 6, 7, 8]},
            "each run is in a block of its own, so the blocks leave nothing to compare",
        ),
        ({"split_plot": True}, "the page opens designs without whole plots"),
        ({"coded_runs": None}, "the source prints no complete table of runs"),
        ({"n_mixture": 3}, "the page makes mixtures of its own, but takes no mixture's runs as given"),
        ({"coded_runs": [[0.0, -1.0, 0.0]]}, "a run sets a categorical factor between two of its levels"),
    ],
    ids=["blocked", "a-block-not-printed", "a-block-each", "whole-plots", "no-runs", "mixture", "a-third-level"],
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


def test_a_blocked_design_is_not_made_by_the_page(build_script, few):
    """The page makes its blocked designs with blocks of its own, so the source's runs open as given instead."""
    record = next(r for r in few if r["id"] == "bhh-pilot-plant-2-3")
    with no_bytecode():
        campaign, reason, tried = build_script.published_campaign({**record, "blocked": True})
    assert (campaign, tried) == (None, False)
    assert reason == "the page makes its designs in blocks of its own"


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


# --------------------------------------------------------------------------- categorical factors of several levels
@pytest.fixture(scope="module")
def several() -> dict[str, dict]:
    """Designs with a categorical factor of three levels or more, labels that read as numbers among them."""
    names = (
        "bhh-blood-coagulation",
        "davies-vulcanised-rubber-5x3x4",
        "montgomery-syrup-loss",
        "bhh-yarn-latin-squares",
    )
    return {r["id"]: r for r in catalog_sample(names)}


def test_four_diets_open_with_their_runs_as_given_each_at_its_label(build_script, several):
    """Box, Hunter and Hunter's blood coagulation times: one factor, diets A to D, in study format 3."""
    with no_bytecode():
        campaign, reason = build_script.given_campaign(several["bhh-blood-coagulation"])
    assert reason == ""
    [diet] = campaign["design"]["spec"]["factors"]
    assert (diet["type"], diet["levels"]) == ("categorical", ["A", "B", "C", "D"])
    assert {row[2] for row in campaign["design"]["rows"]} == {"A", "B", "C", "D"}
    assert campaign["v"] == build_script.LEVELS_FORMAT


def test_a_factorial_of_five_three_and_four_numbered_levels_is_made_by_the_page(build_script, several):
    """Davies's vulcanised rubber: levels 1 to 5, 1 to 3 and 1 to 4 are labels, and the page makes the 60 runs."""
    with no_bytecode():
        campaign, reason, tried = build_script.published_campaign(several["davies-vulcanised-rubber-5x3x4"])
    assert (reason, tried) == ("", True)
    levels = [f["levels"] for f in campaign["design"]["spec"]["factors"]]
    assert levels == [["1", "2", "3", "4", "5"], ["1", "2", "3"], ["1", "2", "3", "4"]]
    assert campaign["design"]["n_runs"] == 60


def test_numbered_nozzles_are_labels_not_a_range(build_script, several):
    """Montgomery's syrup loss: nozzles 1, 2 and 3, once read as a numeric range from 1 to 2."""
    record = several["montgomery-syrup-loss"]
    with no_bytecode():
        campaign, reason, _ = build_script.published_campaign(record)
        if campaign is None:
            campaign, reason = build_script.given_campaign(record)
    assert reason == ""
    nozzle = next(f for f in campaign["design"]["spec"]["factors"] if f["name"] == "nozzle")
    assert (nozzle["type"], nozzle["levels"]) == ("categorical", ["1", "2", "3"])


def test_a_factor_of_more_levels_than_the_page_takes_says_so(build_script, several):
    with no_bytecode():
        assert build_script.given_campaign(several["bhh-yarn-latin-squares"]) == (
            None,
            "a categorical factor has more than the 8 levels the page takes",
        )


def test_the_page_keys_a_term_s_parts_as_the_build_wrote_them(build_script, node):
    """Learn finds a printed number's row by the parts of its term: logic.js termPart is the build's term_part."""
    texts = ["A", "A²", "I(A ** 2)", "C(diet, Sum)", "C(diet)", "température²", "I(x_1 ** 2)", "diet[A]"]
    assert javascript(node, source("logic.js"), [("termPart", [t]) for t in texts]) == list(
        map(build_script.term_part, texts)
    )


@pytest.mark.dataset
def test_a_factor_its_source_compares_as_levels_opens_with_the_source_s_comparisons(build_script):
    """Montgomery's plasma etching: four RF powers, which Learn compares with Tukey's, as the book does."""
    from process_improve.experiments.reference_catalogue import reference_experiments

    entry = next(e for e in reference_experiments("montgomery-2013") if e["name"] == "montgomery-plasma-etching")
    (record,) = catalog_sample(("montgomery-plasma-etching",))
    with no_bytecode():
        campaign, reason = build_script.given_campaign(record)
        numbers = build_script.printed_numbers(record, entry, campaign["design"]["spec"])
    assert reason == ""
    assert campaign["design"]["spec"]["factors"][0]["levels"] == ["160", "180", "200", "220"]
    pairs = {(n["read"]["a"], n["read"]["b"]) for n in numbers if n["read"]["table"] == "comparison"}
    assert ("160", "180") in pairs, "the book's 160.0 less 180.0, in the page's labels"
    assert all(n["model"] == [["power"]] for n in numbers), "no line or curve in power: the page fits none here"
