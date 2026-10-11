"""Generating a design: ``api_make_design``, which step 1 of the page calls.

The run sheet lists the runs in the order they are to be run; the facts say what the design is
and what it aliases; the options each design takes reach the library; and a design the factor
count or the options rule out is refused, saying why.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from process_improve.experiments import generate_design
from tests.web._support import ALL_DESIGNS, FACTORS, coded, factors_named, run_sheet

SIX = factors_named("A", "B", "C", "D", "E", "G")


def test_every_design_the_page_offers_is_tested_here(bootstrap):
    """The tests run over ``ALL_DESIGNS``: a design added to the page must be added there too."""
    assert list(bootstrap.DESIGNS) == list(ALL_DESIGNS)


@pytest.mark.parametrize("design_type", ALL_DESIGNS)
def test_every_offered_design_generates(api, design_type):
    result = api.ok("api_make_design", {"design_type": design_type, "factors": FACTORS, "budget": 14})
    assert result["n_runs"] == len(result["rows"])
    assert result["columns"][:2] == ["Std order", "Run order"]
    run_order = [row[1] for row in result["rows"]]
    assert run_order == sorted(run_order), "rows are listed in the order they are to be run"


@pytest.mark.parametrize("design_type", ALL_DESIGNS)
def test_run_sheet_lists_the_runs_in_their_randomised_order(api, bootstrap, design_type):
    """Row i of the sheet is the i-th run the library randomised, and "Std order" is where it sits unrandomised.

    ``DesignResult.design_actual`` is already in run order, and ``run_order[i]`` is the
    standard-order position of its i-th row. Sorting the sheet on that position instead
    lists the runs unrandomised, centre points last, under a "Run order" of 1..n.
    """
    spec = {"design_type": design_type, "factors": FACTORS, "budget": 14}
    sheet = run_sheet(api.ok("api_make_design", spec))
    options = bootstrap._design_options(spec, design_type)
    design = generate_design(bootstrap._factors(spec), design_type=design_type, **options)
    names = [f["name"] for f in FACTORS]
    np.testing.assert_allclose(sheet[names].to_numpy(float), design.design_actual[names].to_numpy(float))
    assert sheet["Run order"].tolist() == list(range(1, design.n_runs + 1))
    assert sheet["Std order"].tolist() == design.run_order


def test_std_order_rebuilds_the_standard_order_factorial(api):
    """Sorted on "Std order", a 2^3 factorial reads corner by corner, last factor fastest, then its centre runs."""
    spec = {"design_type": "full_factorial", "factors": FACTORS, "n_center_points": 3}
    sheet = run_sheet(api.ok("api_make_design", spec))
    standard = sheet.sort_values("Std order")
    corners = np.array(list(itertools.product([-1, 1], repeat=3)))
    np.testing.assert_allclose(coded(standard, FACTORS), np.vstack([corners, np.zeros((3, 3))]))
    assert not np.allclose(coded(sheet, FACTORS), coded(standard, FACTORS)), "the runs are made in a randomised order"


def test_std_order_rebuilds_the_definitive_screening_foldover(api):
    """Sorted on "Std order", a DSD reads as its conference matrix C, then -C, then the centre run."""
    spec = {"design_type": "dsd", "factors": FACTORS, "n_center_points": 1}
    standard = coded(run_sheet(api.ok("api_make_design", spec)).sort_values("Std order"), FACTORS).to_numpy()
    half = (len(standard) - 1) // 2
    np.testing.assert_allclose(standard[:half], -standard[half : 2 * half])
    np.testing.assert_allclose(standard[-1], 0)


def test_the_response_column_comes_last_and_empty(api):
    spec = {"design_type": "full_factorial", "factors": FACTORS, "response": "Yield"}
    result = api.ok("api_make_design", spec)
    assert result["columns"] == ["Std order", "Run order", "T", "P", "F", "Yield"]
    assert [row[-1] for row in result["rows"]] == [None] * result["n_runs"], "blank until the runs are done"


def test_the_seed_sets_the_run_order_and_nothing_else(api):
    spec = {"design_type": "box_behnken", "factors": FACTORS}
    first, again = (run_sheet(api.ok("api_make_design", spec)) for _ in range(2))
    other = run_sheet(api.ok("api_make_design", {**spec, "random_state": 7}))
    assert first.equals(again), "the page's default seed gives the same sheet every time"
    assert first["Std order"].tolist() != other["Std order"].tolist()
    names = [f["name"] for f in FACTORS]
    assert first.sort_values("Std order")[names].to_numpy().tolist() == (
        other.sort_values("Std order")[names].to_numpy().tolist()
    ), "the same runs, in another order"


def test_the_stored_specification_is_what_the_page_sends_back(api):
    """The specification travels with the runs: in every later call, and in the workbook."""
    result = api.ok("api_make_design", {"design_type": "ccd", "factors": FACTORS, "alpha": "rotatable"})
    spec = result["spec"]
    assert spec["version"] == 1
    assert spec["design_type"] == "ccd"
    assert spec["response"] == "y"
    assert spec["model"] == result["model"] == "quadratic"
    assert spec["factors"] == [{**f, "low": float(f["low"]), "high": float(f["high"])} for f in FACTORS]
    assert spec["options"] == {"random_state": 42, "n_center_points": 3, "alpha": "rotatable"}


@pytest.mark.parametrize("design_type", ["d_optimal", "i_optimal"])
def test_optimal_design_leaves_degrees_of_freedom_for_error_by_default(api, design_type):
    """With no budget given, an optimal design has four runs more than its model has coefficients.

    The page used to send 16 runs whatever the model: six factors and the quadratic
    model have 28 coefficients, so the library raised the budget to exactly 28 and the
    fit had nothing left to estimate the error with.
    """
    spec = {"design_type": design_type, "factors": SIX[:5], "model_type": "quadratic"}
    assert api.ok("api_make_design", spec)["n_runs"] == 21 + 4


def test_a_raised_budget_is_reported(api):
    spec = {"design_type": "d_optimal", "factors": SIX, "model_type": "quadratic", "budget": 16}
    result = api.ok("api_make_design", spec)
    assert result["n_runs"] == 28
    assert any("16 runs" in note and "28 coefficients" in note for note in result["facts"]["notes"])


@pytest.mark.parametrize(("design_type", "extra"), [("box_behnken", {}), ("ccd", {"alpha": "rotatable"})])
def test_designs_singular_without_a_center_point_need_one(api, design_type, extra):
    """Without a center run, every other run of these designs lies on one sphere, so the squares are aliased."""
    spec = {"design_type": design_type, "factors": FACTORS, "n_center_points": 0, **extra}
    assert "at least one center point" in api.error("api_make_design", spec)


def test_face_centred_ccd_may_skip_center_points(api):
    spec = {"design_type": "ccd", "factors": FACTORS, "n_center_points": 0, "alpha": "face_centered"}
    assert api.ok("api_make_design", spec)["n_runs"] == 8 + 6


@pytest.mark.parametrize(
    ("k", "resolution", "n_runs", "achieved"),
    [(3, 3, 4, 3), (4, 4, 8, 4), (5, 4, 16, 5), (6, 3, 8, 3), (6, 4, 16, 4), (6, 5, 32, 6), (7, 4, 16, 4)],
)
def test_fractional_factorial_takes_the_resolution_asked_for(api, k, resolution, n_runs, achieved):
    spec = {
        "design_type": "fractional_factorial",
        "factors": SIX[:k] if k <= 6 else factors_named(*"ABCDEFG"),
        "resolution": resolution,
        "n_center_points": 0,
    }
    result = api.ok("api_make_design", spec)
    assert result["n_runs"] == n_runs
    assert result["facts"]["resolution"] == achieved


def test_fractional_factorial_defaults_to_resolution_iv_not_the_half_fraction(api):
    """Six factors: 16 runs at resolution IV, where the half fraction the page used to send has 32."""
    spec = {"design_type": "fractional_factorial", "factors": SIX, "n_center_points": 0}
    assert api.ok("api_make_design", spec)["n_runs"] == 16


def test_fractional_factorial_refuses_a_resolution_above_the_factor_count(api):
    spec = {"design_type": "fractional_factorial", "factors": FACTORS, "resolution": 4}
    assert "resolution III" in api.error("api_make_design", spec)


@pytest.mark.parametrize(("k", "model"), [(4, "main_effects"), (5, "interactions")])
def test_fraction_is_analysed_with_the_model_its_resolution_supports(api, k, model):
    """Resolution IV keeps main effects clear; resolution V keeps two-factor interactions clear too."""
    spec = {"design_type": "fractional_factorial", "factors": SIX[:k], "resolution": 4}
    assert api.ok("api_make_design", spec)["model"] == model


def test_ccd_uses_a_half_fraction_cube_from_six_factors(api):
    spec = {"design_type": "ccd", "factors": SIX, "n_center_points": 3, "alpha": "face_centered"}
    assert api.ok("api_make_design", spec)["n_runs"] == 32 + 12 + 3


@pytest.mark.parametrize(
    ("design_type", "k", "word"),
    [("dsd", 2, "3 to"), ("box_behnken", 2, "3 to"), ("fractional_factorial", 2, "3 to"), ("i_optimal", 7, "to 6")],
)
def test_designs_outside_their_factor_range_are_refused(api, design_type, k, word):
    spec = {"design_type": design_type, "factors": factors_named(*"ABCDEFG"[:k])}
    assert word in api.error("api_make_design", spec)


def test_facts_name_main_effects_aliased_with_interactions(api):
    """A 2^(3-1) design: each main effect shares its column with the other two factors' interaction."""
    facts = api.ok("api_make_design", {"design_type": "fractional_factorial", "factors": FACTORS})["facts"]
    assert sorted(sorted(group) for group in facts["aliases"]) == [["F", "T·P"], ["P", "T·F"], ["P·F", "T"]]
    assert any("Main effects are aliased with two-factor interactions" in note for note in facts["notes"])


def test_facts_name_interactions_aliased_with_each_other(api):
    names = ("Temp", "Pres", "Flow", "Cat")
    spec = {"design_type": "fractional_factorial", "factors": factors_named(*names), "resolution": 4}
    facts = api.ok("api_make_design", spec)["facts"]
    assert sorted(sorted(group) for group in facts["aliases"]) == [
        ["Flow·Cat", "Temp·Pres"],
        ["Pres·Cat", "Temp·Flow"],
        ["Pres·Flow", "Temp·Cat"],
    ]
    assert facts["generators"] == ["Cat = Temp·Pres·Flow"]


@pytest.mark.parametrize("design_type", ["full_factorial", "dsd", "box_behnken", "ccd"])
def test_facts_say_when_nothing_is_fully_aliased(api, design_type):
    facts = api.ok("api_make_design", {"design_type": design_type, "factors": FACTORS})["facts"]
    assert facts["aliases"] == []
    assert any("No main effect or two-factor interaction is fully aliased" in note for note in facts["notes"])


def test_plackett_burman_center_points_can_be_chosen(api):
    spec = {"design_type": "plackett_burman", "factors": FACTORS, "n_center_points": 0}
    result = api.ok("api_make_design", spec)
    assert result["n_runs"] == 4
    assert result["facts"]["center_runs"] == 0


def test_face_centred_ccd_stays_inside_the_ranges(api):
    face = api.ok("api_make_design", {"design_type": "ccd", "factors": FACTORS, "alpha": "face_centered"})
    rotatable = api.ok("api_make_design", {"design_type": "ccd", "factors": FACTORS, "alpha": "rotatable"})
    assert face["outside_range"] == []
    assert rotatable["outside_range"] == ["T", "P", "F"]
