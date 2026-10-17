"""Renaming a run sheet's factors: ``api_relabel``, which the page calls when a factor's name or units change.

A factor's name and units are labels, so a rename keeps the runs, their order and their results.
Everything Python wrote with the old names is written again with the new ones: the run sheet's
columns, the design's facts and notes, the factors outside their ranges, and the process that
practice results were simulated from. The property checked here is that renaming a design gives
exactly what making it under the new names would give, for each kind of note a design carries.
"""

from __future__ import annotations

import pytest

from tests.web._support import ALL_SCENARIOS, FACTORS, factors_named, held_of

#: The worked example's factors under longer names, as a reader would write them.
WORDS = {"T": "Temperature", "P": "Pressure", "F": "Feed_rate"}

#: Designs whose facts between them carry every note that names factors, each with the names it
#: is given instead: the request without its factors, the factors, and each factor's new name.
CASES = {
    "a resolution III fraction": (
        {"design_type": "fractional_factorial", "resolution": 3},
        factors_named(*"ABCDE"),
        dict(zip("ABCDE", ["Temp", "Time", "Conc", "Speed", "Cat"], strict=True)),
    ),
    "a resolution IV fraction": (
        {"design_type": "fractional_factorial", "resolution": 4},
        factors_named(*"ABCD"),
        {"A": "Temp", "C": "Conc"},
    ),
    "a blocked full factorial": (
        {"design_type": "full_factorial", "n_center_points": 2, "n_blocks": 2},
        FACTORS,
        WORDS,
    ),
    "a blocked fraction, two of its factors swapped": (
        {"design_type": "fractional_factorial", "n_blocks": 2},
        factors_named(*"ABCD"),
        {"A": "B", "B": "A"},
    ),
    "a Plackett-Burman design": ({"design_type": "plackett_burman"}, factors_named(*"ABCDE"), {"E": "Speed"}),
    "a replicated Box-Behnken design": ({"design_type": "box_behnken", "n_replicates": 2}, FACTORS, WORDS),
    "a Box-Behnken design in blocks that confound no effect": (
        {"design_type": "box_behnken", "n_blocks": 2},
        FACTORS,
        WORDS,
    ),
    "a rotatable CCD, outside its ranges": ({"design_type": "ccd", "alpha": "rotatable"}, FACTORS, WORDS),
    "a categorical factor": (
        {"design_type": "full_factorial"},
        [*FACTORS[:2], {"name": "Supplier", "low": "old", "high": "new", "units": ""}],
        {"Supplier": "Vendor", "T": "Temperature"},
    ),
    "a DSD": ({"design_type": "dsd", "n_center_points": 3}, factors_named(*"ABCDE"), {"B": "Speed"}),
    "a Taguchi L8": ({"design_type": "taguchi"}, factors_named(*"ABCDEG"), {"A": "Temp", "G": "Speed"}),
    "a Taguchi L18 at three levels, with a categorical factor": (
        {"design_type": "taguchi", "n_levels": 3},
        [*FACTORS, {"name": "Supplier", "low": "old", "high": "new", "units": ""}],
        {"Supplier": "Vendor", "T": "Temperature"},
    ),
    "a D-optimal design under its budget": (
        {"design_type": "d_optimal", "model_type": "quadratic", "budget": 6},
        FACTORS,
        WORDS,
    ),
}


def named(factors: list[dict], names: dict[str, str]) -> list[dict]:
    """Return the factors with the new names given, in the same places, with the same levels."""
    return [{**f, "name": names.get(f["name"], f["name"])} for f in factors]


def sent(design: dict) -> dict:
    """Return what the page sends of a design to rename it: what it holds, with the facts it shows."""
    return {**held_of(design), "facts": design["facts"], "outside_range": design["outside_range"]}


@pytest.mark.parametrize("case", CASES.values(), ids=CASES)
def test_renaming_a_design_gives_what_making_it_under_the_new_names_gives(api, case):
    request, factors, names = case
    before = api.ok("api_make_design", {**request, "factors": factors})
    after = api.ok("api_make_design", {**request, "factors": named(factors, names)})
    assert after["rows"] == before["rows"], "the library makes a design from its factors' places, not their names"

    renamed = api.ok("api_relabel", {"design": sent(before), "factors": after["spec"]["factors"]})
    assert renamed["columns"] == after["columns"]
    assert renamed["facts"] == after["facts"]
    assert renamed["outside_range"] == after["outside_range"]
    assert renamed["spec"] == after["spec"]


#: Each part of a sentence in a design's notes that names factors, and the case above whose notes hold it.
NAMING = {
    "Resolution III, generators ": "a resolution III fraction",
    "Main effects are aliased with two-factor interactions: ": "a resolution III fraction",
    "Main effects are clear of two-factor interactions, which are aliased": "a resolution IV fraction",
    " is confounded with the blocks": "a blocked full factorial",
    " are confounded with the blocks": "a blocked fraction, two of its factors swapped",
}


@pytest.mark.parametrize(("said", "case"), NAMING.items(), ids=NAMING.values())
def test_the_cases_carry_every_note_that_names_a_factor(api, said, case):
    request, factors, _ = CASES[case]
    notes = api.ok("api_make_design", {**request, "factors": factors})["facts"]["notes"]
    assert any(said in note for note in notes), notes


def test_the_factors_outside_their_ranges_are_renamed(api):
    request, factors, names = CASES["a rotatable CCD, outside its ranges"]
    made = api.ok("api_make_design", {**request, "factors": factors})
    assert made["outside_range"] == ["T", "P", "F"]
    renamed = api.ok("api_relabel", {"design": sent(made), "factors": named(factors, names)})
    assert renamed["outside_range"] == ["Temperature", "Pressure", "Feed_rate"]


@pytest.mark.parametrize("scenario", ALL_SCENARIOS)
def test_the_process_the_results_came_from_is_renamed_with_them(api, scenario):
    before = api.ok("api_make_design", {"design_type": "ccd", "factors": FACTORS})
    after = api.ok("api_make_design", {"design_type": "ccd", "factors": named(FACTORS, WORDS)})
    old, new = (api.ok("api_simulate", {**held_of(d), "scenario": scenario, "seed": 3}) for d in (before, after))

    renamed = api.ok("api_relabel", {"design": sent(before), "factors": after["spec"]["factors"], "simulation": old})
    assert renamed["simulation"] == new, "the same responses, from the same process, under the new names"


def test_a_renamed_run_sheet_is_analysed_under_its_new_names(api, design, simulated):
    after = api.ok("api_make_design", {**api.ok("api_catalogue")["demo"]["design"], "factors": named(FACTORS, WORDS)})
    renamed = api.ok(
        "api_relabel", {"design": sent(design), "factors": after["spec"]["factors"], "simulation": simulated}
    )
    held = {"spec": renamed["spec"], "columns": renamed["columns"], "rows": design["rows"]}
    analysed = {"responses": simulated["responses"], "truth": renamed["simulation"]["truth"]}

    got = api.ok("api_analyze_runs", {**held, **analysed})
    assert got == api.ok("api_analyze_runs", {**held_of(after), **analysed})
    assert [c["label"] for c in got["coefficients"]][:4] == ["Intercept", "Temperature", "Pressure", "Feed_rate"]


def test_a_run_sheet_read_from_a_workbook_has_no_notes_and_is_renamed_all_the_same(api, design, simulated):
    """A workbook brings back the runs and the design's specification, not its notes: there are none to rename."""
    book = api.ok("api_workbook", {**held_of(design), "responses": simulated["responses"]})
    read = api.ok("api_read_workbook", book)
    held = {key: read[key] for key in ("spec", "columns", "rows")}
    renamed = api.ok("api_relabel", {"design": held, "factors": named(FACTORS, WORDS)})
    assert "facts" not in renamed
    assert renamed["columns"] == [WORDS.get(c, c) for c in read["columns"]]

    request = {"spec": renamed["spec"], "columns": renamed["columns"], "rows": read["rows"]}
    analysed = api.ok("api_analyze_runs", {**request, "responses": read["responses"]})
    assert [c["label"] for c in analysed["coefficients"]][1:4] == ["Temperature", "Pressure", "Feed_rate"]


def test_the_terms_a_workbook_was_analysed_with_are_renamed(api, design):
    """A workbook keeps the terms it was last analysed with in its specification; a rename carries them."""
    terms = [["T"], ["P"], ["T", "P"], ["T", "T"]]
    held = {**sent(design), "spec": {**design["spec"], "analysis_model": {"terms": terms}}}
    renamed = api.ok("api_relabel", {"design": held, "factors": named(FACTORS, WORDS)})
    assert renamed["spec"]["analysis_model"] == {
        "terms": [["Temperature"], ["Pressure"], ["Temperature", "Pressure"], ["Temperature", "Temperature"]]
    }


def test_units_change_as_names_do(api, design):
    factors = [{**f, "units": units} for f, units in zip(FACTORS, ["K", "kPa", "L/min"], strict=True)]
    renamed = api.ok("api_relabel", {"design": sent(design), "factors": factors})
    assert [f["units"] for f in renamed["spec"]["factors"]] == ["K", "kPa", "L/min"]
    assert renamed["columns"] == design["columns"]
    assert renamed["facts"] == design["facts"]


def test_a_rename_that_changes_nothing_gives_the_design_back(api, design):
    renamed = api.ok("api_relabel", {"design": sent(design), "factors": FACTORS})
    assert {key: renamed[key] for key in ("spec", "columns", "facts", "outside_range")} == {
        key: design[key] for key in ("spec", "columns", "facts", "outside_range")
    }


REFUSALS = {
    "a level changed": ([{**FACTORS[0], "high": 210}, *FACTORS[1:]], "full_factorial", "Only the factors' names"),
    "a factor fewer": (FACTORS[:2], "full_factorial", "Only the factors' names"),
    "two factors one name": (named(FACTORS, {"P": "T"}), "full_factorial", "Factor names must be unique"),
    "not a variable name": (named(FACTORS, {"T": "2T"}), "full_factorial", "look like variable names"),
    "the results column's name": (named(FACTORS, {"T": "y"}), "full_factorial", "A factor cannot be named y"),
    "a blocked run sheet's Block": (named(FACTORS, {"T": "Block"}), "blocked", "A factor cannot be named Block"),
}


@pytest.mark.parametrize("case", REFUSALS.values(), ids=REFUSALS)
def test_a_rename_that_would_change_the_run_sheet_is_refused(api, case):
    factors, design_type, message = case
    request = {"design_type": "full_factorial", "factors": FACTORS, "response": "y"}
    if design_type == "blocked":
        request["n_blocks"] = 2
    made = api.ok("api_make_design", request)
    assert message in api.error("api_relabel", {"design": sent(made), "factors": factors})


def test_renaming_a_design_with_limits_renames_inside_them(api):
    """Swapping two names swaps them in every limit, as making the design under the new names writes the limits."""
    request = {"design_type": "d_optimal", "budget": 12}
    limits = [{"expression": "T + 25*P <= 250"}, {"expression": "T*F <= 3600", "note": "heat"}]
    swapped = [{"expression": "F + 25*P <= 250"}, {"expression": "F*T <= 3600", "note": "heat"}]
    before = api.ok("api_make_design", {**request, "factors": FACTORS, "constraints": limits})
    after = api.ok(
        "api_make_design", {**request, "factors": named(FACTORS, {"T": "F", "F": "T"}), "constraints": swapped}
    )
    assert after["rows"] == before["rows"], "the same limits, by the factors' places"
    renamed = api.ok("api_relabel", {"design": sent(before), "factors": after["spec"]["factors"]})
    assert renamed["spec"] == after["spec"]
    assert renamed["facts"] == after["facts"]
    assert [c["expression"] for c in renamed["spec"]["constraints"]] == ["F + 25 * P <= 250", "F * T <= 3600"]
