"""The designs chosen by name from the design list: a Sobol space-filling design, A-optimal and OMARS.

The Sobol design is made for no model, as the Latin hypercube is. An A-optimal design is made for a model
and inside limits on the settings, as the other optimal designs are. An OMARS design has the three levels and the
clear main effects of a definitive screening design, in a number of runs the reader chooses above its own.
None of the three is in the comparison of designs for a model: the first fits no model's comparison, and the
others are searched for only when chosen.
"""

from __future__ import annotations

from tests.web._support import FACTORS, factors_named, held_of


def made(api, **spec: object) -> dict:
    """Make a design for the worked example's three factors, unless the specification names others."""
    return api.ok("api_make_design", {"factors": FACTORS, **spec})


def test_a_sobol_design_spreads_its_runs_and_says_how(api):
    said = "The 16 runs follow a Sobol sequence, which fills the ranges evenly"
    design = made(api, design_type="sobol", budget=16)
    assert design["n_runs"] == 16
    assert design["facts"]["notes"][0].startswith(said)
    at = design["columns"].index("T")
    settings = [row[at] for row in design["rows"]]
    assert 150 <= min(settings) < max(settings) <= 200
    assert len(set(settings)) > 8, "its settings are many, where a design for a model has a few"
    assert design["outside_range"] == []


def test_a_sobol_design_opens_on_ten_runs_a_factor_and_learn_fits_the_aim(api):
    design = made(api, design_type="sobol", aim="interactions")
    assert (design["n_runs"], design["model"]) == (30, "interactions")
    held = held_of(design)
    responses = api.ok("api_simulate", {**held, "seed": 1})["responses"]
    assert len(api.ok("api_analyze_runs", {**held, "responses": responses})["coefficients"]) == 7


def test_an_a_optimal_design_is_made_for_its_model_and_inside_the_limits(api):
    design = made(
        api, design_type="a_optimal", model_type="interactions", constraints=[{"expression": "T + 25 * P <= 250"}]
    )
    assert (design["n_runs"], design["model"]) == (11, "interactions"), "seven coefficients and the spare runs"
    assert design["facts"]["notes"][-1] == "Every run is inside your limits on the settings."
    at = {name: design["columns"].index(name) for name in ("T", "P")}
    assert all(row[at["T"]] + 25 * row[at["P"]] <= 250 + 1e-9 for row in design["rows"])


def test_an_omars_design_has_three_levels_and_the_runs_chosen(api):
    """A definitive screening design in four factors has 9 runs: an OMARS design of 15 has six more."""
    design = made(api, design_type="omars", factors=factors_named(*"ABCD"), budget=16)
    assert design["n_runs"] == 15, "the largest odd number of runs within the 16 asked for"
    factor, at = design["spec"]["factors"][0], design["columns"].index("A")
    middle, half = (factor["low"] + factor["high"]) / 2, (factor["high"] - factor["low"]) / 2
    levels = {round((row[at] - middle) / half, 6) for row in design["rows"]}
    assert levels == {-1.0, 0.0, 1.0}
    assert design["facts"]["aliases"] == [], "its main effects are clear of each other"


def test_an_omars_design_opens_four_runs_above_a_definitive_screening_design(api):
    screening = made(api, design_type="dsd", n_center_points=0)
    assert made(api, design_type="omars")["n_runs"] == screening["n_runs"] + 4


def test_the_comparison_of_designs_makes_none_of_the_designs_chosen_by_name(api, bootstrap):
    compared = {d["design_type"] for d in api.ok("api_candidates", {"factors": FACTORS, "model": "quadratic"})["made"]}
    assert not compared & {*bootstrap.SPACE_FILLING, *bootstrap.CHOSEN_BY_NAME}
    assert {"d_optimal", "i_optimal"} <= compared
