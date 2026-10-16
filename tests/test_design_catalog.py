"""The design catalog: published and generated designs, each described by the facets a reader filters on."""

from __future__ import annotations

import dataclasses
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from process_improve.experiments import compare_designs, design_catalog, generate_design
from process_improve.experiments.compare import MEASURES
from process_improve.experiments.design_catalog import (
    CATALOG_URL,
    FAMILIES,
    GENERATED,
    GLYPH_ROWS,
    INDEX_FIELDS,
    CatalogQuery,
    catalog_comparisons,
    catalog_errata,
    catalog_families,
    catalog_feed,
    catalog_records,
    catalog_sources,
    design_facets,
    glyph,
    normalise_family,
    printed_checks,
    published_runs,
    search_catalog,
)
from process_improve.experiments.design_measures import estimable_models
from process_improve.experiments.factor import Factor
from process_improve.experiments.reference_catalogue import (
    reference_designs,
    reference_experiments,
    reproduce_reference_catalogue,
)


@pytest.fixture(scope="module")
def records() -> list[dict]:
    """Build the whole catalog once: every published entry and every generated design."""
    return catalog_records()


# --------------------------------------------------------------------------- families
@pytest.mark.parametrize(
    ("text", "family"),
    [
        ("2^(5-2) fractional factorial", "fractional_factorial"),
        ("2^(7-4) fraction and its fold-over", "fractional_factorial"),
        ("2^3 full factorial in two blocks", "full_factorial"),
        ("3^2 factorial", "full_factorial"),
        ("2^2 factorial with a centre point (EVOP)", "full_factorial"),
        ("CCD", "ccd"),
        ("rotatable central composite design", "ccd"),
        ("CCF, k = 3, 6 centre runs", "ccd"),
        ("small composite: 2^(4-1) resolution III cube, 8 star points, 4 centre runs", "ccd"),
        ("Box-Behnken, k = 3, 3 centre runs", "box_behnken"),
        ("DSD-augmented definitive screening design (DSD-augment method)", "dsd"),
        ("mixed-level OMARS design, foldover", "omars"),
        ("Taguchi L18", "orthogonal_array"),
        ("Plackett-Burman", "plackett_burman"),
        ("4-by-4 Graeco-Latin Square", "latin_square"),
        ("randomized complete block design", "block_design"),
        ("D-optimal, chosen from the 20 candidates of Table 5.1", "optimal"),
        ("{3, 2} simplex lattice, pure blends twice and binary blends three times", "mixture"),
        ("strip plot: implant 2^3 crossed with anneal 2^3, three sizes of experimental unit", "split_plot"),
        ("two-group comparison", "comparison"),
        ("regression on one variable", "comparison"),
        ("other", "other"),
    ],
)
def test_a_design_type_in_words_names_its_family(text, family):
    assert normalise_family(text) == family


@pytest.mark.slow
@pytest.mark.dataset
def test_every_published_entry_has_a_family_and_few_have_none(records):
    published = [r for r in records if r["collection"] == "published"]
    assert len(published) == len(reference_experiments()) + len(reference_designs())
    assert {r["family"] for r in records} <= set(FAMILIES)
    unknown = [r["design_text"] for r in published if r["family"] == "other"]
    assert len(unknown) <= 21, f"a rule is needed for: {sorted(set(unknown))}"


def test_every_family_with_a_year_names_the_work_that_published_it():
    families = catalog_families()
    assert set(families) == set(FAMILIES)
    for key, family in families.items():
        assert family["label"], key
        assert (family["year"] is None) == (family["reference"] == ""), key
        if family["year"] is not None:
            assert f"({family['year']})" in family["reference"], key


# --------------------------------------------------------------------------- what a design estimates
def _coded(design_type: str, k: int, **options) -> np.ndarray:
    factors = [Factor(name=chr(ord("A") + i), low=-1, high=1) for i in range(k)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # a DSD leaves no error degrees of freedom: not what these tests are about
        result = generate_design(factors, design_type=design_type, **options)
    return result.design[[f.name for f in factors]].to_numpy(dtype=float)


def test_a_resolution_three_fraction_estimates_main_effects_only():
    coded = _coded("fractional_factorial", 7, resolution=3, n_center_points=0)
    assert len(coded) == 8
    assert estimable_models(coded, ["numeric"] * 7) == {
        "main_effects": True,
        "interactions": False,
        "quadratic": False,
    }


def test_a_central_composite_design_estimates_the_full_quadratic():
    coded = _coded("ccd", 3, alpha="face_centered", n_center_points=3)
    assert estimable_models(coded, ["numeric"] * 3)["quadratic"] is True


def test_a_factor_held_at_one_level_adds_nothing_to_estimate():
    runs = np.array([[-1.0, 1.0], [1.0, 1.0], [-1.0, 1.0], [1.0, 1.0]])
    assert estimable_models(runs, ["numeric", "categorical"])["main_effects"] is True


def test_the_glyph_spells_each_setting_as_a_letter():
    assert glyph(np.array([[-2.0, -1.0, 0.0], [1.0, 2.0, np.nan]])) == "afkpu."


def test_a_large_design_is_drawn_from_evenly_spaced_runs():
    coded = _coded("full_factorial", 8, n_center_points=0)
    assert len(coded) == 256
    picture = glyph(coded)
    assert len(picture) == GLYPH_ROWS * 8
    assert set(picture) == {"f", "p"}


def test_the_facets_of_a_definitive_screening_design():
    coded = _coded("dsd", 6)
    facets = design_facets(coded, ["numeric"] * 6)
    assert facets["n_runs"] == 13
    assert facets["n_centre"] == 1
    assert facets["levels"] == [3] * 6
    assert facets["models"]["main_effects"] is True
    assert facets["runs_per_parameter"] == round(13 / 7, 2)
    assert len(facets["glyph"]) == 13 * 6


# --------------------------------------------------------------------------- published entries
def test_printed_settings_are_coded_from_the_factor_s_range():
    entry = {
        "factors": [
            {"name": "T", "low": 160, "high": 180, "type": "numeric"},
            {"name": "K", "low": "A", "high": "B", "type": "categorical"},
        ],
        "runs": {"columns": ["No.", "T", "K", "y"], "rows": [[1, 160, "A", 60], [2, 170, "B", 72], [3, None, "Z", 1]]},
    }
    coded = published_runs(entry)
    assert coded is not None
    np.testing.assert_array_equal(coded[:2], [[-1.0, -1.0], [0.0, 1.0]])
    assert np.isnan(coded[2]).all(), "a blank and an unknown label are no settings"


def test_where_a_source_prints_an_entry_is_said_in_a_few_words():
    assert design_catalog._where(None) == ""
    assert design_catalog._where("Table 3, p. 12") == "Table 3, p. 12"
    assert design_catalog._where({"pages": "12", "tables": ["Table 2"], "figures": ["Figure 1"]}) == (
        "p. 12; Table 2; Figure 1"
    )
    assert design_catalog._where({"pages": "12-14"}) == "pp. 12-14"
    assert design_catalog._where({"section": "4.1"}) == "section 4.1"


def test_an_entry_without_runs_or_factors_has_no_coded_runs():
    assert published_runs({"factors": [], "runs": None}) is None
    assert published_runs({"factors": [{"name": "x"}], "runs": {"columns": ["y"], "rows": [[1]]}}) is None


@pytest.mark.slow
@pytest.mark.dataset
def test_the_pilot_plant_experiment_is_described_as_box_hunter_and_hunter_print_it(records):
    record = next(r for r in records if r["id"] == "bhh-pilot-plant-2-3")
    assert record["family"] == "full_factorial"
    assert record["source"] == "box-hunter-hunter-1978"
    assert record["year"] == 1978
    assert record["added"] == "2026-10-14"
    assert record["kind"] == "experiment"
    assert (record["n_runs"], record["n_factors"], record["n_numeric"], record["n_categorical"]) == (8, 3, 2, 1)
    assert record["models"] == {"main_effects": True, "interactions": True, "quadratic": False}
    assert record["where"].startswith("pp. 307-324")
    assert record["response_values"]["y"] == [60, 72, 54, 68, 52, 83, 45, 80]


# --------------------------------------------------------------------------- generated designs
def test_every_generated_design_has_a_name_of_its_own():
    assert len(GENERATED) > 100
    families = {spec["design_type"] for spec in GENERATED}
    assert {"full_factorial", "fractional_factorial", "dsd", "omars", "ccd", "box_behnken", "taguchi"} <= families


@pytest.mark.slow
def test_generated_records_are_the_designs_the_library_makes(records):
    generated = [r for r in records if r["collection"] == "generated"]
    assert 100 < len(generated) <= len(GENERATED), "options that make one design keep it once"
    ids = [r["id"] for r in records]
    assert len(set(ids)) == len(ids), "each design has an id of its own"
    dsd = next(r for r in generated if r["id"] == "lib-dsd-k6")
    assert (dsd["n_runs"], dsd["n_centre"], dsd["family"]) == (13, 1, "dsd")
    ccd = next(r for r in generated if r["id"] == "lib-ccd-k2-rotatable")
    assert ccd["levels"] == [5, 5], "the axial runs add two levels to the cube's two and the centre"
    assert all(r["source"] is None and r["year"] is None for r in generated)


@pytest.mark.slow
def test_generated_records_carry_the_measures_of_their_runs(records):
    by_id = {r["id"]: r for r in records}
    dsd = by_id["lib-dsd-k6"]
    assert (dsd["quadratic_projection"], dsd["projectivity"]) == (3, None)
    assert (dsd["run_loss_model"], dsd["survives_run_loss"]) == ("main_effects", True)
    pb = by_id["lib-plackett-burman-k11"]
    assert (pb["survives_run_loss"], pb["critical_runs"], pb["projectivity"]) == (False, 12, 3)
    assert len(pb["leverage"]) == 12


@pytest.mark.slow
@pytest.mark.dataset
def test_one_design_printed_in_several_places_is_one_class(records):
    """Box, Hunter and Hunter's pilot plant is a 2^3 factorial: the class of the one the library makes."""
    by_id = {r["id"]: r for r in records}
    assert by_id["bhh-pilot-plant-2-3"]["design_class"] == by_id["lib-full-factorial-k3"]["design_class"]
    assert by_id["lib-full-factorial-k3"]["design_class"] != by_id["lib-fractional-factorial-k4-res4"]["design_class"]
    assert by_id["bhh-plant-trial-two-methods"]["design_class"] is not None
    assert all(r["design_class"] is None for r in records if not r["glyph"])


@pytest.mark.slow
@pytest.mark.dataset
def test_every_record_carries_every_field_the_page_reads(records):
    for record in records:
        missing = set(INDEX_FIELDS) - set(record)
        assert not missing, f"{record['id']} lacks {missing}"
        assert record["n_runs"] is None or record["n_runs"] >= 1, record["id"]
        if record["glyph"]:
            assert len(record["glyph"]) == min(record["n_runs"], GLYPH_ROWS) * record["n_factors"], record["id"]


# --------------------------------------------------------------------------- checks and sources
def test_an_entry_s_printed_numbers_are_counted_by_what_their_checks_say():
    entry = {
        "expected": [
            {"check": {"kind": "coefficient"}},
            {"check": [{"kind": "anova"}, {"kind": "anova", "disagreement": "the book rounds twice"}]},
            {"check": {"manual": "printed in words"}},
        ]
    }
    assert printed_checks(entry) == {"printed": 4, "checked": 3, "agree": 2, "explained": 1}


@pytest.mark.slow
@pytest.mark.dataset
def test_the_counts_are_what_the_reproduce_report_finds():
    """The counts read from the catalogue are the report's, since the tests make every check agree or explain it."""
    report = reproduce_reference_catalogue("ullmann-doe-2012")
    published = {r["id"]: r for r in catalog_records("published") if r["source"] == "ullmann-doe-2012"}
    for name, rows in report.groupby("experiment"):
        checked = rows[rows["checked"].astype(bool)]
        assert published[name]["checks"] == {
            "printed": len(rows),
            "checked": len(checked),
            "agree": int(checked["agrees"].astype(bool).sum()),
            "explained": int(checked["disagreement"].notna().sum()),
        }, name


def test_generated_records_count_no_checks():
    assert all(r["checks"] is None for r in catalog_records("generated"))


def test_an_unknown_collection_is_refused():
    with pytest.raises(ValueError, match="Unknown collection 'all'"):
        catalog_records("all")  # type: ignore[arg-type]


def test_each_source_carries_its_citation_and_terms():
    sources = catalog_sources()
    assert sources["box-hunter-hunter-1978"]["year"] == 1978
    assert sources["jones-nachtsheim-2013"]["doi"] == "10.1080/00224065.2013.11917921"
    assert all(source["cite"] and source["rights"] for source in sources.values())
    assert sources["box-hunter-hunter-1978"]["short"] == "Box, Hunter and Hunter"
    assert sources["davies-1956"]["short"] == "Davies"
    assert sources["schmidt-launsby-2000"]["short"] == "Schmidt and Launsby"
    assert all(0 < len(source["short"]) <= 50 for source in sources.values())


# --------------------------------------------------------------------------- searching the catalog
@pytest.mark.slow
@pytest.mark.dataset
def test_a_search_keeps_the_designs_that_pass_every_filter():
    found = search_catalog(min_factors=6, max_factors=6, max_runs=17, min_quadratic_projection=3)
    assert {d["family"] for d in found} == {"dsd", "omars"}
    assert [d["n_runs"] for d in found] == sorted(d["n_runs"] for d in found), "the fewest runs first"
    assert all(d["url"] == f"{CATALOG_URL}#d={d['id']}" for d in found)
    published = search_catalog(text="pilot plant", collection="published", has_results=True)
    assert "bhh-pilot-plant-2-3" in {d["id"] for d in published}
    pilot = next(d for d in published if d["id"] == "bhh-pilot-plant-2-3")
    assert pilot["source_short"] == "Box, Hunter and Hunter (1978)"
    assert pilot["same_design"] >= 1, "the 2^3 factorial is printed in many places"


@pytest.mark.slow
@pytest.mark.dataset
def test_a_year_keeps_published_designs_and_the_newest_come_first():
    found = search_catalog(year_from=2019, sort="newest")
    assert found
    assert all(d["collection"] == "published" and d["year"] >= 2019 for d in found)
    assert [d["year"] for d in found] == sorted((d["year"] for d in found), reverse=True)


def test_a_search_refuses_a_family_or_model_the_catalog_does_not_have():
    with pytest.raises(ValueError, match=r"Unknown families \['nope'\]"):
        search_catalog(families=["nope"])
    with pytest.raises(ValueError, match="Unknown model 'cubic'"):
        search_catalog(model="cubic")


def test_a_search_takes_a_query_whose_fields_named_ones_replace_and_refuses_a_misspelt_one():
    query = CatalogQuery(families=("dsd",), sort="factors")
    assert dataclasses.replace(query, max_runs=13).families == ("dsd",)
    with pytest.raises(TypeError, match="min_factor"):
        search_catalog(query, min_factor=3)


def test_the_search_tool_is_registered_with_a_strict_schema():
    from process_improve.tool_spec import discover_tools, get_tool_specs

    discover_tools()
    spec = {s["name"]: s for s in get_tool_specs()}["search_design_catalog"]
    assert spec["category"] == "experiments"
    assert spec["input_schema"]["additionalProperties"] is False
    assert "https://factori.al/catalog/" in spec["description"]


@pytest.mark.slow
@pytest.mark.dataset
def test_the_search_tool_returns_designs_with_their_pages():
    from process_improve.tool_spec import execute_tool_call

    found = execute_tool_call("search_design_catalog", {"families": ["box_behnken"], "max_factors": 3, "limit": 2})
    assert found["count"] >= found["shown"] == len(found["designs"]) == 2
    assert all(d["family"] == "box_behnken" and d["url"].startswith(CATALOG_URL) for d in found["designs"])
    assert execute_tool_call("search_design_catalog", {"families": ["nope"]})["error"].startswith("Unknown families")


# --------------------------------------------------------------------------- comparing designs
def _generated(*ids: str) -> list[dict]:
    """Return the generated records with these ids, made one by one."""
    return [design_catalog._generated_record(s) for s in GENERATED if design_catalog._spec_id(s) in ids]


def test_each_design_is_scored_as_compare_designs_scores_it_alongside_others():
    ccd, bbd = _generated("lib-ccd-k3-face-centered", "lib-box-behnken-k3")
    scored = catalog_comparisons([ccd, bbd])
    direct = compare_designs({r["id"]: np.array(r["coded_runs"], dtype=float) for r in (ccd, bbd)}, "quadratic").values
    for record in (ccd, bbd):
        values = scored[record["id"]]["quadratic"]
        label = f"{record['id']} [n={record['n_runs']}]"
        assert values["fits"] is True
        assert values["i"] == pytest.approx(direct[label]["i"], rel=1e-3)
        assert values["max_r"] == pytest.approx(direct[label]["max_r"], rel=1e-3)
        assert len(values["fds"]) == 21
        assert set(values) == {m.key for m in MEASURES} | {"rank", "fds"}


def test_a_model_a_design_cannot_fit_is_scored_as_such_without_a_curve():
    cube = _generated("lib-full-factorial-k3")
    values = catalog_comparisons(cube)["lib-full-factorial-k3"]
    assert values["interactions"]["fits"] is True
    assert values["quadratic"]["fits"] is False
    assert values["quadratic"]["fds"] is None
    assert values["quadratic"]["g"] is None


def test_designs_with_other_factors_or_missing_settings_are_not_scored():
    mixture = _generated("lib-mixture-k3-linear")
    gap = {**_generated("lib-full-factorial-k3")[0], "id": "gap"}
    gap["coded_runs"] = [[None, *row[1:]] if i == 0 else row for i, row in enumerate(gap["coded_runs"])]
    assert catalog_comparisons([*mixture, gap]) == {}


# --------------------------------------------------------------------------- errata, the feed, next steps
@pytest.mark.dataset
def test_every_number_known_not_to_reproduce_is_an_erratum_with_its_reason():
    errata = catalog_errata()
    explained = sum(printed_checks(e)["explained"] for e in [*reference_experiments(), *reference_designs()])
    assert len(errata) == explained
    pilot = next(e for e in errata if e["id"] == "bhh-plant-trial-two-methods" and e["what"] == "std_error")
    assert (pilot["printed"], pilot["where"], pilot["rounding"]) == ("1.16", "p. 73", True)
    assert "already rounded" in pilot["reason"]


@pytest.mark.parametrize(
    ("where", "said"),
    [
        ({"page": 73, "quote": "..."}, "p. 73"),
        ({"pdf_page": 4}, "PDF p. 4"),
        ({"quiz": "Week 3", "question": "q2"}, "quiz Week 3, question q2"),
        ({"file": "doe.rst", "line": 120}, "doe.rst, line 120"),
        ({"file": "doe.rst"}, "doe.rst"),
        ({"url": "https://example.org"}, ""),
        ("Table 4", "Table 4"),
        (None, ""),
    ],
)
def test_where_an_erratum_is_printed_is_said_in_a_few_words(where, said):
    assert design_catalog._errata_where(where) == said


def test_the_feed_is_atom_with_an_entry_per_source_newest_first():
    records = [
        {
            "collection": "published",
            "source": "davies-1956",
            "added": "2026-10-14",
            "checks": {"printed": 5, "checked": 4},
        },
        {"collection": "published", "source": "davies-1956", "added": "2026-10-14", "checks": None},
        {
            "collection": "published",
            "source": "goos-2025",
            "added": "2026-10-13",
            "checks": {"printed": 2, "checked": 2},
        },
        {"collection": "published", "source": "pid-book", "added": None, "checks": None},
        {"collection": "generated", "source": None, "added": None, "checks": None},
    ]
    feed = ET.fromstring(catalog_feed(records).encode())  # noqa: S314 - the feed the library just wrote
    atom = "{http://www.w3.org/2005/Atom}"
    assert feed.tag == f"{atom}feed"
    for required in ("title", "id", "updated", "author"):
        assert feed.find(f"{atom}{required}") is not None
    entries = feed.findall(f"{atom}entry")
    assert [e.find(f"{atom}id").text for e in entries] == [
        f"{CATALOG_URL}#src=davies-1956",
        f"{CATALOG_URL}#src=goos-2025",
    ]
    assert feed.find(f"{atom}updated").text == "2026-10-14T00:00:00Z"
    assert entries[0].find(f"{atom}title").text.endswith(": 2 designs and experiments")
    assert "5 printed numbers, 4 of them recomputed" in entries[0].find(f"{atom}summary").text


def test_a_family_says_which_families_it_can_lead_to():
    families = catalog_families()
    assert {"family": "ccd", "how": "Add center + axial points"} in families["full_factorial"]["next"]
    assert families["latin_square"]["next"] == []
    assert all(step["family"] in FAMILIES for f in families.values() for step in f["next"])
