"""The design catalog: published and generated designs, each described by the facets a reader filters on."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import generate_design
from process_improve.experiments.design_catalog import (
    FAMILIES,
    GENERATED,
    GLYPH_ROWS,
    INDEX_FIELDS,
    catalog_families,
    catalog_records,
    catalog_sources,
    design_facets,
    estimable_models,
    glyph,
    normalise_family,
    published_runs,
)
from process_improve.experiments.factor import Factor
from process_improve.experiments.reference_catalogue import reference_designs, reference_experiments


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


def test_a_two_level_factorial_estimates_interactions_but_no_squares():
    square = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]], dtype=float)
    assert estimable_models(square, ["numeric"] * 2) == {"main_effects": True, "interactions": True, "quadratic": False}


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


def test_a_categorical_factor_enters_with_a_column_per_extra_level():
    """Three suppliers crossed with two temperatures: six runs, main effects and the interaction in full."""
    runs = np.array([[s, t] for s in (-1.0, 0.0, 1.0) for t in (-1.0, 1.0)])
    models = estimable_models(runs, ["categorical", "numeric"])
    assert models == {"main_effects": True, "interactions": True, "quadratic": False}
    assert estimable_models(runs[:, :1], ["categorical"])["quadratic"] is None, "no numeric factor to square"


def test_a_mixture_or_a_missing_setting_leaves_the_models_unstated():
    simplex = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    assert set(estimable_models(simplex, ["mixture"] * 3).values()) == {None}
    gap = np.array([[-1, 1], [1, np.nan], [-1, -1], [1, 1]])
    assert set(estimable_models(gap, ["numeric"] * 2).values()) == {None}


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
    assert len(generated) == len(GENERATED)
    ids = [r["id"] for r in records]
    assert len(set(ids)) == len(ids), "each design has an id of its own"
    dsd = next(r for r in generated if r["id"] == "lib-dsd-k6")
    assert (dsd["n_runs"], dsd["n_centre"], dsd["family"]) == (13, 1, "dsd")
    ccd = next(r for r in generated if r["id"] == "lib-ccd-k2-rotatable")
    assert ccd["levels"] == [5, 5], "the axial runs add two levels to the cube's two and the centre"
    assert all(r["source"] is None and r["year"] is None for r in generated)


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
def test_a_report_s_checks_are_counted_entry_by_entry():
    report = pd.DataFrame(
        {
            "experiment": ["bhh-pilot-plant-2-3"] * 4,
            "checked": [True, True, True, False],
            "agrees": [True, True, False, None],
            "disagreement": [None, None, "the book rounds twice", None],
        }
    )
    published = catalog_records("published", report=report)
    record = next(r for r in published if r["id"] == "bhh-pilot-plant-2-3")
    assert record["checks"] == {"printed": 4, "checked": 3, "agree": 2, "explained": 1}
    assert next(r for r in published if r["id"] != record["id"])["checks"] is None


def test_without_a_report_no_record_counts_checks():
    assert all(r["checks"] is None for r in catalog_records("generated"))


def test_an_unknown_collection_is_refused():
    with pytest.raises(ValueError, match="Unknown collection 'all'"):
        catalog_records("all")  # type: ignore[arg-type]


def test_each_source_carries_its_citation_and_terms():
    sources = catalog_sources()
    assert sources["box-hunter-hunter-1978"]["year"] == 1978
    assert sources["jones-nachtsheim-2013"]["doi"] == "10.1080/00224065.2013.11917921"
    assert all(source["cite"] and source["rights"] for source in sources.values())
