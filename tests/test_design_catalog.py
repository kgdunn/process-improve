"""The design catalog: published and generated designs, each described by the facets a reader filters on."""

from __future__ import annotations

import dataclasses
import re
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
    catalog_topics,
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
    reference_sources,
    reference_topics,
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
        ("optimal design, two categorical factors", "optimal"),
        ("minimum-run resolution IV design", "optimal"),
        ("two-level resolution V design, irregular", "fractional_factorial"),
        ("conference matrix", "dsd"),
        ("{3, 2} simplex lattice, pure blends twice and binary blends three times", "mixture"),
        ("strip plot: implant 2^3 crossed with anneal 2^3, three sizes of experimental unit", "split_plot"),
        ("two-group comparison", "comparison"),
        ("regression on one variable", "comparison"),
        ("single-factor random effects design", "comparison"),
        ("steps along a path of steepest ascent", "comparison"),
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
    # pid-book's 19 "other", two projections, and three response surface designs no family holds:
    # an equiradial hexagon, a hybrid design and a sphere-packing design
    assert len(unknown) <= 24, f"a rule is needed for: {sorted(set(unknown))}"


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


def test_a_level_printed_as_a_number_is_coded_as_the_level_listed_as_text():
    """Davies's lots, listed as "1" to "3" and printed as 1 to 3: each is coded at its place among the levels."""
    lots = {"name": "Lot", "type": "categorical", "levels": ["1", "2", "3"]}
    entry = {"factors": [lots], "runs": {"columns": ["Lot", "y"], "rows": [[1, 5.0], [2.0, 6.0], ["3", 7.0]]}}
    np.testing.assert_array_equal(published_runs(entry), [[-1.0], [0.0], [1.0]])


def test_two_levels_listed_high_first_are_coded_low_first_as_the_source_codes_them():
    """Montgomery's mortars, listed modified then unmodified, with unmodified the low: it is coded -1."""
    mortar = {"name": "mortar", "type": "categorical", "levels": ["modified", "unmodified"]}
    mortar |= {"low": "unmodified", "high": "modified"}
    entry = {
        "factors": [mortar],
        "runs": {"columns": ["mortar", "y"], "rows": [["unmodified", 16.9], ["modified", 16.6]]},
    }
    np.testing.assert_array_equal(published_runs(entry), [[-1.0], [1.0]])


def test_a_table_of_coded_runs_that_prints_a_categorical_factor_s_levels_codes_them():
    """Goos and Jones's suppliers 1, 2 and 3 beside coded settings: the suppliers are levels, the settings codes."""
    factors = [
        {"name": "x", "low": 0, "high": 10, "type": "numeric"},
        {"name": "supplier", "type": "categorical", "levels": [1, 2, 3], "low": 1, "high": 3},
    ]
    rows = [[-1, 1, 5.0], [1, 2, 6.0], [0, 3, 7.0]]
    entry = {"factors": factors, "runs": {"columns": ["x", "supplier", "y"], "rows": rows, "coded": True}}
    np.testing.assert_array_equal(published_runs(entry), [[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
    codes = {**entry, "runs": {**entry["runs"], "rows": [[-1, -1, 5.0], [1, 0, 6.0], [0, 1, 7.0]]}}
    np.testing.assert_array_equal(published_runs(codes), [[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]], "codes stay codes")


_SOURCE = {"key": "a-source", "year": 2000}


def test_a_block_column_is_each_run_s_block_and_no_factor_of_the_design():
    """A 3^2 in three blocks that lists its block as a factor: the record keeps the block apart, run by run."""
    entry = {
        "name": "three-blocks",
        "design": {"type": "3^2 factorial in three blocks"},
        "factors": [
            {"name": "A", "low": 0, "high": 2, "type": "numeric"},
            {"name": "block", "type": "categorical", "levels": ["1", "2", "3"]},
        ],
        "runs": {"columns": ["A", "block", "y"], "rows": [[0, 1, 5.0], [1, 2, 6.0], [2, 3, 7.0], [0, 2, 4.0]]},
    }
    record = design_catalog._published_record(entry, _SOURCE)
    assert [f["name"] for f in record["factors"]] == ["A"]
    assert record["coded_runs"] == [[-1.0], [0.0], [1.0], [-1.0]]
    assert (record["blocked"], record["blocks"], record["n_blocks"]) == (True, [1, 2, 3, 2], 3)


def test_an_entry_names_its_block_column_when_its_source_calls_the_blocks_something_else():
    """Four pressures in each of three batches: the entry says the batch is the block, and the record keeps it apart."""
    pressure = {"name": "pressure", "type": "categorical", "levels": ["8500", "8700"]}
    batch = {"name": "batch", "type": "categorical", "levels": ["1", "2", "3"]}
    rows = [[p, b, 90.0 + b] for b in (1, 2, 3) for p in ("8500", "8700")]
    entry = {
        "name": "grafts",
        "design": {"type": "randomized complete block design", "blocks": "batch"},
        "factors": [pressure, batch],
        "runs": {"columns": ["pressure", "batch", "y"], "rows": rows},
    }
    record = design_catalog._published_record(entry, _SOURCE)
    assert [f["name"] for f in record["factors"]] == ["pressure"]
    assert (record["blocks"], record["n_blocks"]) == ([1, 1, 2, 2, 3, 3], 3)
    unnamed = design_catalog._published_record(
        {**entry, "design": {"type": "randomized complete block design"}}, _SOURCE
    )
    assert ([f["name"] for f in unnamed["factors"]], unnamed["blocks"]) == (["pressure", "batch"], None)
    assert unnamed["blocked"], "the text still says the design is run in blocks"


@pytest.mark.dataset
def test_every_block_column_an_entry_names_is_in_its_run_table_and_no_factor_of_its_record():
    entries = [*reference_experiments(), *reference_designs()]
    named = [e for e in entries if e["design"].get("blocks") or e["design"].get("blockings")]
    assert len(named) >= 35
    sources = reference_sources()
    for entry in named:
        for arranged in design_catalog._arrangements(entry):
            blocks = arranged["design"]["blocks"]
            columns = [blocks] if isinstance(blocks, str) else blocks
            assert set(columns) <= set(entry["runs"]["columns"]), arranged["name"]
            record = design_catalog._published_record(arranged, sources[entry["source"]])
            assert not set(columns) & {f["name"] for f in record["factors"]}, arranged["name"]
            assert len(record["blocks"]) == len(entry["runs"]["rows"]), arranged["name"]


def test_a_block_of_several_columns_is_each_combination_of_their_values():
    """Two replicates of two shifts each: a block is a shift of a replicate, and neither column is a factor."""
    entry = {
        "name": "shifts",
        "design": {"type": "2^1 in two replicates of two blocks", "blocks": ["rep", "shift"]},
        "factors": [
            {"name": "A", "low": -1, "high": 1, "type": "numeric"},
            {"name": "rep", "type": "categorical", "levels": ["I", "II"]},
        ],
        "runs": {
            "columns": ["rep", "shift", "A", "y"],
            "rows": [["I", 1, -1, 1.0], ["I", 2, 1, 2.0], ["II", 1, -1, 3.0], ["II", 2, 1, 4.0]],
        },
    }
    record = design_catalog._published_record(entry, _SOURCE)
    assert [f["name"] for f in record["factors"]] == ["A"]
    assert (record["blocks"], record["n_blocks"]) == (["I / 1", "I / 2", "II / 1", "II / 2"], 4)


def test_an_entry_can_say_it_is_run_in_no_blocks_whatever_its_text_says():
    """Montgomery's smelting algorithms, analysed as one factor: only a later problem calls the runs blocks."""
    entry = {
        "name": "smelting",
        "design": {
            "type": "single-factor design",
            "details": "a later problem takes the runs as blocks",
            "blocks": False,
        },
        "factors": [{"name": "algorithm", "type": "categorical", "levels": ["1", "2"]}],
        "runs": {"columns": ["algorithm", "y"], "rows": [["1", 4.9], ["2", 4.8]]},
    }
    record = design_catalog._published_record(entry, _SOURCE)
    assert (record["blocked"], record["blocks"]) == (False, None)
    assert design_catalog._published_record({**entry, "design": {"type": "in blocks"}}, _SOURCE)["blocked"]


def test_each_way_a_source_blocks_the_runs_is_a_record_of_its_own():
    """Davies's penicillin 2^5: in two blocks (weeks) in one example, in four (sets) in the next."""
    rows = [[a, w, s, 1.0] for a, w, s in zip([-1, 1] * 4, ["left", "right"] * 4, [1, 2, 3, 4] * 2, strict=True)]
    entry = {
        "name": "penicillin",
        "title": "Penicillin",
        "design": {"type": "2^1 in blocks", "blockings": ["Week", "Set"]},
        "factors": [{"name": "A", "low": -1, "high": 1, "type": "numeric"}, {"name": "Week"}, {"name": "Set"}],
        "runs": {"columns": ["A", "Week", "Set", "y"], "rows": rows},
    }
    weeks, sets = design_catalog._arrangements(entry)
    assert (weeks["name"], sets["name"], sets["title"]) == (
        "penicillin",
        "penicillin-in-4-blocks",
        "Penicillin, in 4 blocks",
    )
    for arranged, n in ((weeks, 2), (sets, 4)):
        record = design_catalog._published_record(arranged, _SOURCE)
        assert ([f["name"] for f in record["factors"]], record["n_blocks"]) == (["A"], n)
    assert design_catalog._published_record(entry, _SOURCE)["n_blocks"] == 2, (
        "the first arrangement, unless one is named"
    )


@pytest.mark.dataset
def test_the_catalog_lists_each_arrangement_once_under_its_own_name(records):
    ids = [r["id"] for r in records]
    assert len(ids) == len(set(ids))
    assert {"davies-penicillin-2-5", "davies-penicillin-2-5-in-4-blocks"} <= set(ids)


@pytest.mark.parametrize(
    ("text", "blocked"),
    [
        ("2^3 full factorial in two blocks", True),
        ("G being the heap of soil (a blocking factor confounded at the highest level)", True),
        ("One centre point, no replicates, no blocks.", False),
        ("Array 'nonfoldover' in the code block.", False),
        ("alpha = 1.633 is the orthogonal-blocking alpha (see expected)", False),
        ("defining relation I = ABC (stated in generators-defining-relationships-blocking.rst line 77)", False),
    ],
)
def test_only_words_that_say_a_design_is_run_in_blocks_mark_it_blocked(text, blocked):
    assert design_catalog._blocks({}, text, None) == (blocked, None)


def test_where_a_source_prints_an_entry_is_said_in_a_few_words():
    assert design_catalog._where(None) == ""
    assert design_catalog._where("Table 3, p. 12") == "Table 3, p. 12"
    assert design_catalog._where({"pages": "12", "tables": ["Table 2"], "figures": ["Figure 1"]}) == (
        "p. 12; Table 2; Figure 1"
    )
    assert design_catalog._where({"pages": "12-14"}) == "pp. 12-14"
    assert design_catalog._where({"section": "4.1"}) == "section 4.1"
    assert design_catalog._where({"section": "4.1", "figures": ["DOE-fig-Cube-plot (cube.png)"]}) == "Cube plot"


def test_the_few_words_name_three_tables_and_figures_and_count_the_rest():
    where = {"pages": "12-30", "tables": ["Table 1", "Table 2"], "figures": ["Figure 1", "Figure 2", "Figure 3"]}
    assert design_catalog._where(where) == "pp. 12-30; Table 1; Table 2; Figure 1; and 2 more tables and figures"
    assert design_catalog._where({**where, "figures": ["Figure 1", "Figure 2"]}).endswith("and 1 more table or figure")


@pytest.mark.parametrize(
    ("note", "name"),
    [
        ("Factor ranges (list-table at line 46, no caption): Factor / Low / Centre / High", "Factor ranges"),
        ("Coded runs in standard order (lines 29-39)", "Coded runs in standard order"),
        ("Grid table, lines 1075-1093 (Experiment, D, R, P, Taste)", "Table of Experiment, D, R, P and Taste"),
        (
            "Solution part 5: csv-table of the four no-profile days, lines 107-114",
            "Solution part 5: table of the four no-profile days",
        ),
        (
            "Recipe / True titer, g/L / Gain over current (list-table at line 538, no caption)",
            "A table: Recipe / True titer, g/L / Gain over current",
        ),
        ("untitled table (Lot of material, D, E), lines 139-151", "Table of Lot of material, D and E"),
        (":math:`R^2_Y` by component", "R^2_Y by component"),
        ("the same table, highly-fractionated-designs.rst lines 41-61 (Example 1)", "The same table of Example 1"),
        ("No table printed; the solution names experiments 5, 2, 3 and 8 of the table at lines 1075-1093", ""),
        ("Table 10.1", "Table 10.1"),
    ],
    ids=[
        "caption-then-headings",
        "caption-with-lines",
        "kind-and-columns",
        "csv-table",
        "headings-only",
        "untitled",
        "math-role",
        "a-source-file",
        "no-table",
        "a-printed-book-s-table",
    ],
)
def test_a_table_is_named_by_its_caption_or_its_columns(note, name):
    assert design_catalog.table_name(note) == name


@pytest.mark.dataset
def test_no_table_is_named_by_a_line_note_a_file_or_markup_and_other_sources_keep_theirs():
    """The book's tables lose their maintainers' notes; a printed book's "Table 10.1" is kept as printed."""
    entries = [*reference_experiments(), *reference_designs()]
    for entry in entries:
        for note in (entry.get("where") or {}).get("tables", []):
            name = design_catalog.table_name(note)
            if entry["source"] != "pid-book":
                assert name == note.strip().removesuffix("."), note
                continue
            assert not re.search(r"list-table|csv-table|grid table|\blines? \d|:math:|no caption|\.rst\b", name), note
            assert name or note.lower().startswith("no table printed"), note


def _entry(name: str) -> dict:
    """Return one entry of the reference catalogue, an experiment or a design, by its name."""
    return next(e for e in [*reference_experiments(), *reference_designs()] if e["name"] == name)


def _book_figures() -> list[str]:
    """Return every figure the book's entries record, as their notes for the catalogue's maintainers hold it."""
    return [f for e in reference_experiments("pid-book") + reference_designs("pid-book") for f in e["where"]["figures"]]


#: The estimability frontier's caption, as the book prints it.
FRONTIER = (
    "The estimability frontier N = k^2 + k + 1 for a foldover, against the parameter count of the full "
    "second-order model and the size of a definitive screening design"
)
#: The alt text of two of the book's figures that have no caption.
HISTOGRAM = "Histogram of the average B-minus-A difference over all randomization permutations"
CONTOURS = "Contour plot illustrating the one-factor-at-a-time search path"


@pytest.mark.parametrize(
    ("note", "name"),
    [
        (f"{FRONTIER} (../figures/doe/omars-estimability-frontier.png)", FRONTIER),
        (
            f"../figures/doe/single-experiment-randomization.png (no caption; alt '{HISTOGRAM}'), lines 244-248",
            HISTOGRAM,
        ),
        (f"DOE-fig-COST: ../figures/doe/COST-contours.png (no caption; alt '{CONTOURS}'), lines 31-36", CONTOURS),
        ("DOE-fig-Cube-plot (factorial-two-levels-two-variables-no-analysis.png)", "Cube plot"),
        ("factorial-two-level-surface-example-cropped.png (true response surface)", "True response surface"),
        ("../figures/doe/pareto-plot-pid.png (Pareto plot, lines 152-156)", "Pareto plot"),
        ("../figures/doe/central-composite-question.png (line 395): diagram of the design", "Diagram of the design"),
        (
            "Individual desirability for purity. ... (multi-response-desirability-functions.png, lines 283-290)",
            ("Individual desirability for purity"),
        ),
        ("../figures/doe/colour-pls-inversion.png", "Colour pls inversion"),
        ("Figure 1(a)", "Figure 1(a)"),
        ("Fig. 11-9", "Fig. 11-9"),
    ],
    ids=[
        "caption-then-path",
        "path-then-alt",
        "label-path-alt",
        "label-then-file",
        "file-then-description",
        "description-with-lines",
        "caption-after-colon",
        "caption-with-ellipsis",
        "bare-path",
        "a-book-s-figure-number",
        "an-abbreviated-figure-number",
    ],
)
def test_a_figure_is_named_by_its_caption_alt_text_description_or_label(note, name):
    assert design_catalog.figure_name(note) == name


@pytest.mark.dataset
def test_no_figure_of_the_book_is_named_by_a_path_a_line_note_or_a_label():
    """A note's provenance (its file, the lines that print it, its label's prefix) stays in the catalogue's data."""
    notes = _book_figures()
    assert sum(".png" in note for note in notes) >= 98
    for note in notes:
        name = design_catalog.figure_name(note)
        assert name, note
        assert ".png" not in name, note
        assert "../" not in name, note
        assert not re.search(r"\blines? \d", name), note
        assert not name.startswith(("DOE-", "DOE_")), note


@pytest.mark.dataset
def test_a_book_entry_s_place_names_its_figures_and_keeps_its_section():
    where = _entry("omars-fed-batch-bioreactor")["where"]
    shown = design_catalog.shown_where(where)
    assert set(shown) == {"section", "tables", "figures"}, "the source file, its lines and commit stay out"
    assert shown["section"] == where["section"]
    assert shown["figures"][2] == "The thirty batches in run order"
    assert shown["tables"][0] == "Factor ranges", "the caption, without the note of its lines or its headings"
    assert "../" not in design_catalog._where(where)
    assert "list-table" not in design_catalog._where(where)
    assert design_catalog.shown_where("Table 4") == "Table 4"


@pytest.mark.dataset
@pytest.mark.parametrize(
    ("name", "url"),
    [
        (
            "omars-fed-batch-bioreactor",
            "https://learnche.org/pid/design-analysis-experiments/omars-worked-study#doe-omars-worked-study",
        ),
        (
            "rsm-central-composite",
            "https://learnche.org/pid/design-analysis-experiments/response-surface-methods#doe-central-composite-designs",
        ),
        ("bhh-pilot-plant-2-3", None),
    ],
    ids=["a-book-section", "a-label-with-underscores", "a-printed-book"],
)
def test_the_place_a_web_source_prints_an_entry_has_an_address(name, url):
    assert design_catalog.where_url(_entry(name)["where"]) == url


@pytest.mark.dataset
def test_a_handbook_entry_s_address_is_its_page():
    nist = reference_experiments("nist-ehandbook-ch5")[0]
    assert design_catalog.where_url(nist["where"]) == nist["where"]["url"]
    assert nist["where"]["url"].startswith("https://www.itl.nist.gov/")


@pytest.mark.parametrize(
    ("label", "anchor"),
    [
        ("DOE-omars-estimability-frontier", "doe-omars-estimability-frontier"),
        ("APPS_mixed_level_profile_case_study", "apps-mixed-level-profile-case-study"),
        ("DOE-fig--Odd  label__", "doe-fig-odd-label"),
    ],
)
def test_a_section_s_anchor_is_its_label_as_sphinx_writes_an_id(label, anchor):
    assert design_catalog._anchor(label) == anchor


@pytest.mark.dataset
def test_a_record_carries_its_applications_and_the_address_of_its_place():
    sources = reference_sources()
    book = design_catalog._published_record(_entry("biologic-drug-yield"), sources["pid-book"])
    assert book["topics"] == ["bioprocessing", "pharmaceuticals"]
    assert book["where_url"].startswith("https://learnche.org/pid/")
    pilot = design_catalog._published_record(_entry("bhh-pilot-plant-2-3"), sources["box-hunter-hunter-1978"])
    assert (pilot["topics"], pilot["where_url"]) == (["chemical-processes"], None)
    generated = _generated("lib-full-factorial-k3")[0]
    assert (generated["topics"], generated["where_url"]) == ([], None)
    assert "topics" in INDEX_FIELDS


def test_every_application_has_a_name_and_says_what_it_covers():
    topics = catalog_topics()
    assert set(topics) == set(reference_topics())
    assert all(t["label"] and t["description"] for t in topics.values())
    assert topics["general"]["label"] == design_catalog.GENERAL_LABEL
    assert topics["semiconductors"]["label"] == "Semiconductors and electronic assembly"


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


def test_a_categorical_factor_of_more_labels_leaves_no_projectivity():
    """Which two of twenty parts sit at -1 and +1 is arbitrary, so their combinations count for nothing."""
    parts = np.repeat(np.linspace(-1, 1, 20), 6)
    operators = np.tile(np.repeat([-1.0, 0.0, 1.0], 2), 20)
    gauge = design_facets(np.column_stack([parts, operators]), ["categorical", "categorical"])
    assert gauge["projectivity"] is None
    cube = np.array([[a, b] for a in (-1.0, 1.0) for b in (-1.0, 1.0)])
    assert design_facets(cube, ["categorical", "numeric"])["projectivity"] == 2, "two labels are two levels"


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


@pytest.mark.slow
@pytest.mark.dataset
def test_an_application_keeps_the_published_designs_that_list_it():
    food = search_catalog(topics=["food-beverages"])
    assert food
    assert all("food-beverages" in d["topics"] and d["collection"] == "published" for d in food)
    either = {d["id"] for d in search_catalog(topics=("food-beverages", "semiconductors"))}
    assert either == {d["id"] for d in food} | {d["id"] for d in search_catalog(topics=["semiconductors"])}
    assert "biologic-drug-yield" in {d["id"] for d in search_catalog(topics=["pharmaceuticals"])}, "its second topic"
    general = search_catalog(topics=["general"], collection="generated")
    assert general == [], "a generated design has no application"


@pytest.mark.slow
@pytest.mark.dataset
def test_a_word_of_an_application_s_name_finds_its_designs_and_each_carries_its_place():
    listed = {d["id"] for d in search_catalog(topics=["semiconductors"])}
    assert listed <= {d["id"] for d in search_catalog(text="semiconductors")}
    book = next(d for d in search_catalog(text="omars-fed-batch-bioreactor"))
    assert book["where_url"].endswith("#doe-omars-worked-study")
    assert "../" not in book["where"]


def test_a_search_refuses_a_family_or_model_the_catalog_does_not_have():
    with pytest.raises(ValueError, match=r"Unknown families \['nope'\]"):
        search_catalog(families=["nope"])
    with pytest.raises(ValueError, match="Unknown model 'cubic'"):
        search_catalog(model="cubic")
    with pytest.raises(ValueError, match=r"Unknown applications \['cooking'\]: choose from agriculture"):
        search_catalog(topics=["cooking"])


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
    named = re.findall(r"'([a-z-]+)'", spec["input_schema"]["properties"]["topics"]["description"])
    assert named, "the schema names applications an agent can choose"
    assert set(named) <= set(catalog_topics()), "each one the catalog has"


@pytest.mark.slow
@pytest.mark.dataset
def test_the_search_tool_returns_designs_with_their_pages():
    from process_improve.tool_spec import execute_tool_call

    found = execute_tool_call("search_design_catalog", {"families": ["box_behnken"], "max_factors": 3, "limit": 2})
    assert found["count"] >= found["shown"] == len(found["designs"]) == 2
    assert all(d["family"] == "box_behnken" and d["url"].startswith(CATALOG_URL) for d in found["designs"])
    assert execute_tool_call("search_design_catalog", {"families": ["nope"]})["error"].startswith("Unknown families")
    food = execute_tool_call("search_design_catalog", {"topics": ["food-beverages"], "limit": 50})
    assert food["count"] == len(search_catalog(topics=["food-beverages"])) > 0, "the tool filters as search_catalog"
    assert all("food-beverages" in d["topics"] for d in food["designs"])


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


def test_a_family_the_knowledge_base_describes_says_what_it_is_in_plain_words():
    families = catalog_families()
    assert families["full_factorial"]["description"].startswith("Tests every combination of factor levels.")
    assert "\n" not in families["dsd"]["description"]
    assert families["latin_square"]["description"] == ""


def test_a_family_says_which_families_it_can_lead_to():
    families = catalog_families()
    assert {"family": "ccd", "how": "Add center + axial points"} in families["full_factorial"]["next"]
    assert families["latin_square"]["next"] == []
    assert all(step["family"] in FAMILIES for f in families.values() for step in f["next"])
