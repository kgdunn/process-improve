"""Generating a design: ``api_make_design``, which Design calls to make the run sheet.

The run sheet lists the runs in the order they are to be run; the facts say what the design is
and what it aliases; the options each design takes reach the library; and a design the factor
count or the options rule out is refused, saying why.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from process_improve.experiments import generate_design
from tests.web._support import ALL_DESIGNS, FACTORS, coded, factors_named, held_of, run_sheet

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
    made = design.design_actual[names].to_numpy(float)
    if design_type in bootstrap.SPACE_FILLING:  # rounded as the reader can set them: by at most a twentieth of a step
        steps = np.array([f["high"] - f["low"] for f in FACTORS]) / options["budget"]
        assert (np.abs(sheet[names].to_numpy(float) - made) <= steps / 20).all()
    else:
        np.testing.assert_allclose(sheet[names].to_numpy(float), made)
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


def test_a_budget_below_the_model_is_refused(api):
    """The page used to enlarge the design and say so; like the library (#957), it now refuses the budget."""
    spec = {"design_type": "d_optimal", "factors": SIX, "model_type": "quadratic", "budget": 16}
    assert api.error("api_make_design", spec) == (
        "The quadratic model in 6 factors has 28 coefficients, so it needs at least 28 runs."
    )


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


# --------------------------------------------------------------------------- replicates
def test_replicates_repeat_every_run_but_the_center_runs(api):
    spec = {"design_type": "full_factorial", "factors": FACTORS, "n_center_points": 3, "n_replicates": 2}
    result = api.ok("api_make_design", spec)
    sheet = coded(run_sheet(result), FACTORS)
    assert result["n_runs"] == 2 * 8 + 3
    corners = sheet[(sheet.abs() > 0.5).all(axis=1)]
    assert corners.value_counts().tolist() == [2] * 8, "each corner twice"
    assert result["spec"]["options"]["n_replicates"] == 2
    assert any("appears 2 times" in note for note in result["facts"]["notes"])


@pytest.mark.parametrize("design_type", ALL_DESIGNS)
def test_every_design_takes_replicates(api, design_type):
    spec = {"design_type": design_type, "factors": FACTORS, "budget": 14}
    once = api.ok("api_make_design", spec)
    twice = api.ok("api_make_design", {**spec, "n_replicates": 2})
    assert twice["n_runs"] > once["n_runs"]
    assert twice["spec"]["options"]["n_replicates"] == 2


def test_a_design_made_once_does_not_name_replicates(api):
    """One replicate is the design as it is, so the stored options, and the Python shown for them, leave it out."""
    result = api.ok("api_make_design", {"design_type": "full_factorial", "factors": FACTORS, "n_replicates": 1})
    assert "n_replicates" not in result["spec"]["options"]


def test_replicates_give_the_pure_error_for_a_lack_of_fit_test(api):
    """An unreplicated 2^2 factorial has no pure error; replicated, it tests whether the main effects suffice."""
    spec = {"design_type": "full_factorial", "factors": FACTORS[:2], "n_center_points": 0, "n_replicates": 3}
    design = api.ok("api_make_design", spec)
    runs = coded(run_sheet(design), FACTORS[:2])
    noise = np.random.default_rng(0).normal(0, 0.2, len(runs))
    y = (10 + 2 * runs["T"] - runs["P"] + 1.5 * runs["T"] * runs["P"]).to_numpy() + noise
    held = {"spec": design["spec"], "columns": design["columns"], "rows": design["rows"]}
    analysis = api.ok("api_analyze_runs", {**held, "responses": y.tolist(), "model": "main_effects"})
    assert analysis["lack_of_fit"]["df_pure_error"] == 8
    assert analysis["lack_of_fit"]["p_value"] < 0.001, "the interaction left out of the model shows as lack of fit"


def test_an_optimal_design_s_runs_are_counted_before_it_is_replicated(api):
    """The budget is the runs of one replicate, so it is checked against the model before replicating."""
    spec = {"design_type": "d_optimal", "factors": FACTORS, "model_type": "interactions", "budget": 10}
    assert api.ok("api_make_design", {**spec, "n_replicates": 2})["n_runs"] == 20
    assert "needs at least 7 runs" in api.error("api_make_design", {**spec, "budget": 5, "n_replicates": 2})


# --------------------------------------------------------------------------- a CCD's axial distance as a number
def test_a_ccd_takes_an_axial_distance_typed_as_a_number(api):
    spec = {"design_type": "ccd", "factors": FACTORS[:2], "n_center_points": 3, "alpha": 1.4}
    result = api.ok("api_make_design", spec)
    levels = sorted(set(np.round(coded(run_sheet(result), FACTORS[:2]).to_numpy().ravel(), 9)))
    assert levels == [-1.4, -1.0, 0.0, 1.0, 1.4]
    assert result["spec"]["options"]["alpha"] == 1.4
    assert result["outside_range"] == ["T", "P"], "an axial distance above 1 leaves the ranges"


def test_an_axial_distance_inside_the_cube_keeps_every_run_in_the_ranges(api):
    spec = {"design_type": "ccd", "factors": FACTORS[:2], "alpha": 0.5}
    assert api.ok("api_make_design", spec)["outside_range"] == []


@pytest.mark.parametrize(("k", "alpha"), [(2, 2**0.5), (3, 3**0.5)])
def test_an_axial_distance_at_the_corners_radius_needs_a_center_point(api, k, alpha):
    """At the square root of k every run but the center runs lies on one sphere, as in a rotatable CCD in 2 or 4."""
    spec = {"design_type": "ccd", "factors": FACTORS[:k], "n_center_points": 0, "alpha": alpha}
    assert "at least one center point" in api.error("api_make_design", spec)
    assert api.ok("api_make_design", {**spec, "alpha": 1.4})["n_runs"] == 2**k + 2 * k


# --------------------------------------------------------------------------- blocks
def test_a_blocked_design_lists_each_run_s_block_one_block_after_another(api):
    spec = {"design_type": "full_factorial", "factors": FACTORS, "n_center_points": 2, "n_blocks": 2}
    result = api.ok("api_make_design", spec)
    sheet = run_sheet(result)
    assert result["columns"][:3] == ["Std order", "Run order", "Block"]
    assert sheet["Block"].tolist() == sorted(sheet["Block"]), "the runs are listed block by block"
    assert sheet["Block"].value_counts().tolist() == [5, 5], "four corners and a center run in each"
    x = coded(sheet, FACTORS)
    corners = x[(x.abs() > 0.5).all(axis=1)]
    tpf = corners.prod(axis=1).groupby(sheet["Block"]).nunique()
    assert tpf.tolist() == [1, 1], "the three-factor interaction takes one value in each block"
    assert result["spec"]["options"]["n_blocks"] == 2
    note = next(n for n in result["facts"]["notes"] if n.startswith("The runs are made in 2 blocks of 5 runs each"))
    assert note.endswith("T·P·F is confounded with the blocks, so the analysis leaves it out.")


@pytest.mark.parametrize("design_type", ALL_DESIGNS)
def test_every_design_can_be_made_in_blocks(api, design_type):
    # A fraction or a Taguchi array in three factors has four runs, and any split of four runs in two confounds a
    # main effect.
    factors = SIX[:4] if design_type in ("fractional_factorial", "taguchi") else FACTORS
    result = api.ok("api_make_design", {"design_type": design_type, "factors": factors, "budget": 14, "n_blocks": 2})
    assert sorted(set(run_sheet(result)["Block"])) == [1, 2]


def test_a_split_that_would_confound_a_main_effect_is_refused_in_the_library_s_words(api):
    spec = {"design_type": "fractional_factorial", "factors": FACTORS, "n_blocks": 2}
    assert "confounds a main effect with blocks" in api.error("api_make_design", spec)


def test_a_design_in_one_block_has_no_block_column(api):
    result = api.ok("api_make_design", {"design_type": "full_factorial", "factors": FACTORS, "n_blocks": 1})
    assert "Block" not in result["columns"]
    assert "n_blocks" not in result["spec"]["options"]


def test_a_fraction_s_facts_name_the_aliases_confounded_with_the_blocks(api):
    """In a 2^(4-1) of resolution IV, A·B = C·D: both go with the blocks the design confounds A·B with."""
    spec = {"design_type": "fractional_factorial", "factors": factors_named(*"ABCD"), "n_blocks": 2}
    notes = api.ok("api_make_design", {**spec, "n_center_points": 0})["facts"]["notes"]
    said = "A·B and C·D are confounded with the blocks, so the analysis leaves them out."
    assert any(n.endswith(said) for n in notes)


# --------------------------------------------------------------------------- categorical factors
SUPPLIER = {"name": "Supplier", "low": "A", "high": "B", "units": ""}


@pytest.mark.parametrize("design_type", ["full_factorial", "fractional_factorial", "plackett_burman"])
def test_a_two_level_design_takes_a_factor_whose_levels_are_labels(api, design_type):
    factors = [*FACTORS, SUPPLIER]
    result = api.ok("api_make_design", {"design_type": design_type, "factors": factors, "n_center_points": 2})
    sheet = run_sheet(result)
    assert set(sheet["Supplier"]) == {"A", "B"}, "the run sheet lists the labels themselves"
    assert result["spec"]["factors"][3] == SUPPLIER
    assert result["outside_range"] == []
    numeric = coded(sheet, FACTORS)
    centre = (numeric.abs() < 1e-9).all(axis=1)
    assert centre.sum() == 2, "a center run is at the center of the numeric factors"
    assert set(sheet.loc[centre, "Supplier"]) <= {"A", "B"}, "and at one of the labels"


def test_a_label_is_coded_minus_one_for_the_low_and_plus_one_for_the_high(api):
    design = api.ok("api_make_design", {"design_type": "full_factorial", "factors": [FACTORS[0], SUPPLIER]})
    sheet = run_sheet(design)
    supplier_b = (sheet["Supplier"] == "B").to_numpy()
    responses = np.where(supplier_b, 14.0, 10.0) + coded(sheet, FACTORS[:1])["T"].to_numpy()
    analysis = api.ok("api_analyze_runs", {**held_of(design), "responses": responses.tolist()})
    found = {c["term"]: c["coefficient"] for c in analysis["coefficients"]}
    assert found["Supplier"] == pytest.approx(2.0), "half the difference from A to B, as for a numeric factor"
    assert found["T"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- limits on the settings
#: Limits in the worked example's factors, a straight one and a curved one, and each written as Python.
LIMITS = {
    "T + 25 * P <= 250": lambda t, p, _f: t + 25 * p <= 250 + 1e-9,
    "T * F <= 3600": lambda t, _p, f: t * f <= 3600 + 1e-9,
}


def _with_limits(design_type: str, **extra: object) -> dict:
    return {"design_type": design_type, "factors": FACTORS, "constraints": [{"expression": t} for t in LIMITS], **extra}


@pytest.mark.parametrize("design_type", ["d_optimal", "i_optimal"])
@pytest.mark.parametrize("extra", [{}, {"n_blocks": 2}, {"n_replicates": 2}, {"model_type": "main_effects"}])
def test_an_optimal_design_keeps_every_run_inside_the_limits(api, design_type, extra):
    result = api.ok("api_make_design", _with_limits(design_type, **extra))
    runs = run_sheet(result)[["T", "P", "F"]].itertuples(index=False)
    assert all(holds(*run) for run in runs for holds in LIMITS.values())
    assert result["facts"]["notes"][-1] == "Every run is inside your limits on the settings."


def test_the_limits_change_where_the_runs_go(api):
    """Without them, the D-optimal design runs the corner the heat limit cuts off."""
    free = run_sheet(api.ok("api_make_design", {"design_type": "d_optimal", "factors": FACTORS}))
    assert not all(LIMITS["T + 25 * P <= 250"](*run) for run in free[["T", "P", "F"]].itertuples(index=False))


def test_a_design_stores_its_limits_beside_its_options(api):
    """The options hold one design's settings, which a comparison reuses; the limits are the study's."""
    spec = api.ok("api_make_design", _with_limits("d_optimal"))["spec"]
    assert spec["constraints"] == [
        {"expression": "T + 25 * P <= 250", "type": "linear"},
        {"expression": "T * F <= 3600", "type": "nonlinear"},
    ]
    assert "constraints" not in spec["options"]


def test_a_design_without_limits_is_as_it_was(api, design):
    assert "constraints" not in design["spec"]
    assert "Every run is inside your limits on the settings." not in design["facts"]["notes"]


@pytest.mark.parametrize("design_type", ["full_factorial", "d_optimal"])
def test_limits_with_a_categorical_factor_name_the_conflict_whatever_the_design(api, design_type):
    """A full factorial would send the reader to a D-optimal design, and back: the refusal names the real conflict."""
    factors = [*FACTORS[:2], SUPPLIER]
    request = {"design_type": design_type, "factors": factors, "constraints": [{"expression": "T <= 190"}]}
    assert api.error("api_make_design", request).startswith("Limits on the settings work with numeric factors only")
