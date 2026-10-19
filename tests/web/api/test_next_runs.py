"""Runs added to a study: ``api_next_runs`` lists the ways Learn offers, ``api_add_runs`` adds one's runs.

A study learns in steps. A screening fraction whose main effects are aliased with two-factor interactions
is folded over; a two-level design that needs curvature gains axial runs and becomes a central composite
design; a design with no run at the center gains some; runs chosen for the model the study aims to learn
fill what no other way can; a design that leaves too few runs for the noise is made again. The runs made
so far keep their places and their cells, so the results the page holds still belong to them, and the
runs added follow as a block of their own, except center runs, whose shift from the others is the
curvature they test.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from tests.web._support import FACTORS, coded, factors_named, held_of, run_sheet

#: Seven factors from -1 to +1, which a resolution III fraction screens in 8 runs.
SEVEN = factors_named(*"ABCDEFG")


def offers(api, design: dict) -> dict[str, dict]:
    """Return the ways Learn offers to add runs to a design the page made, by their kind, in the order listed."""
    return {o["kind"]: o for o in api.ok("api_next_runs", held_of(design))["offers"]}


def added(api, design: dict, kind: str) -> dict:
    """Add one way's runs to a design the page made, and return the design the study then holds."""
    return api.ok("api_add_runs", {**held_of(design), "kind": kind})


def make(api, **spec: object) -> dict:
    """Make a design as Design does, its response named y."""
    return api.ok("api_make_design", {"response": "y", **spec})


def mains_clear(design: dict) -> bool:
    """Say whether the design's runs keep every main effect uncorrelated with every two-factor interaction."""
    x = coded(run_sheet(design), design["spec"]["factors"]).to_numpy(dtype=float)
    products = np.column_stack([x[:, i] * x[:, j] for i, j in itertools.combinations(range(x.shape[1]), 2)])
    centred = [m - m.mean(axis=0) for m in (x, products)]
    return bool(np.allclose(centred[0].T @ centred[1], 0, atol=1e-9))


# --------------------------------------------------------------------------- what is offered
def test_a_box_behnken_design_is_offered_no_runs(api, held):
    """The worked example fits its full quadratic model with runs to spare: no way of adding runs applies."""
    assert api.ok("api_next_runs", held) == {"offers": [], "notes": []}


def test_a_resolution_three_fraction_is_offered_its_fold_over_first(api):
    fraction = make(api, design_type="fractional_factorial", factors=SEVEN, resolution=3, n_center_points=0)
    offered = offers(api, fraction)
    assert next(iter(offered)) == "foldover"
    fold = offered["foldover"]
    assert fold["runs"] == 8
    assert fold["blocked"] is True
    assert fold["label"] == "Fold the design over: 8 runs"
    assert "separates every main effect from the two-factor interactions" in fold["why"]


def test_a_factorial_that_fits_the_interactions_is_offered_axial_runs(api):
    cube = make(api, design_type="full_factorial", factors=FACTORS, n_center_points=3, aim="quadratic")
    axial = offers(api, cube)["axial"]
    assert axial["label"] == "Add 6 axial runs and 2 center runs"  # the library tops the center runs up to five
    assert axial["runs"] == 8
    assert axial["model"] == "quadratic"
    assert "face-centred central composite design" in axial["why"]


def test_center_runs_are_offered_to_a_two_level_design_without_any(api):
    cube = make(api, design_type="full_factorial", factors=FACTORS, n_center_points=0)
    center = offers(api, cube)["center"]
    assert center == {
        "kind": "center",
        "label": "Add 3 center runs",
        "why": (
            "No run is at the center of the ranges. Runs there show whether the response curves between the low and "
            "high settings, and their repeats estimate the noise."
        ),
        "model": "interactions",
        "blocked": False,
        "runs": 3,
    }


def test_runs_for_the_aim_are_offered_when_no_other_way_fits_it(api):
    """A resolution III fraction aiming at the interactions: a fold-over keeps them aliased, so runs are chosen."""
    fraction = make(
        api, design_type="fractional_factorial", factors=SEVEN, resolution=3, n_center_points=0, aim="interactions"
    )
    optimal = offers(api, fraction)["optimal"]
    # 29 coefficients, 8 of them spanned by the fraction, and 4 runs to spare for the noise.
    assert optimal["runs"] == 29 - 8 + 4
    assert optimal["model"] == "interactions"
    assert optimal["label"] == "Add 25 runs for the main effects and two-factor interactions model"


def test_a_saturated_design_is_offered_its_runs_again(api):
    fraction = make(api, design_type="fractional_factorial", factors=SEVEN, resolution=3, n_center_points=0)
    replicate = offers(api, fraction)["replicate"]
    assert replicate["runs"] == 8
    assert replicate["why"].startswith("The main effects model leaves 0 runs to estimate the noise.")


def test_a_categorical_factor_rules_out_runs_between_its_levels(api):
    """A supplier has no center and no axial setting; a fold-over swaps its two labels, which it can."""
    factors = [*FACTORS[:2], {"name": "S", "low": "A", "high": "B", "units": ""}]
    pb = make(api, design_type="plackett_burman", factors=factors, n_center_points=0)
    offered = offers(api, pb)
    assert "axial" not in offered
    assert "center" not in offered
    assert "optimal" not in offered
    assert "foldover" in offered


def test_a_way_that_breaks_a_limit_is_noted_and_not_offered(api):
    half = make(
        api,
        design_type="given",
        factors=factors_named("A", "B", "C"),
        runs=[[-1, -1, 1], [1, -1, -1], [-1, 1, -1], [1, 1, 1]],
        constraints=[{"expression": "A + B + C >= -2"}],
    )
    reply = api.ok("api_next_runs", held_of(half))
    assert "foldover" not in {o["kind"] for o in reply["offers"]}
    assert "center" in {o["kind"] for o in reply["offers"]}  # the center meets the limit
    assert reply["notes"] == [
        "The runs of a fold-over would fall outside your limits on the settings, so Learn does not offer them."
    ]


def test_a_mixture_is_offered_nothing(api, bootstrap):
    blends = [{"name": n, "low": None, "high": None, "units": ""} for n in "ABC"]
    lattice = make(api, design_type="simplex_lattice", factors=blends)
    assert api.ok("api_next_runs", held_of(lattice)) == {"offers": [], "notes": [bootstrap.MIXTURE_EXTENSIONS]}


# --------------------------------------------------------------------------- the runs added
def test_the_fold_over_clears_every_main_effect_and_blocks_its_runs(api):
    fraction = make(api, design_type="fractional_factorial", factors=SEVEN, resolution=3, n_center_points=0)
    assert not mains_clear(fraction)
    folded = added(api, fraction, "foldover")
    assert folded["added"] == 8
    assert folded["n_runs"] == 16
    assert mains_clear(folded)
    sheet = run_sheet(folded)
    assert folded["columns"][:3] == ["Std order", "Run order", "Block"]
    assert sheet["Block"].tolist() == [1] * 8 + [2] * 8
    assert sheet["Run order"].tolist() == list(range(1, 17))
    assert sorted(sheet["Std order"][8:]) == list(range(9, 17))
    # Every run added is a run made before, every setting switched to its other end.
    names = [f["name"] for f in SEVEN]
    before = {tuple(-v for v in run) for run in sheet[names][:8].to_numpy().tolist()}
    assert {tuple(run) for run in sheet[names][8:].to_numpy().tolist()} == before
    assert folded["spec"]["design_type"] == "given"
    assert folded["spec"]["options"] == {"grown": True, "random_state": 42, "n_blocks": 2}
    notes = folded["facts"]["notes"]
    assert notes[0] == "Runs 9 to 16 fold the design over: each is a run made before, every setting switched."
    assert notes[1].startswith("Main effects are clear of two-factor interactions")
    assert notes[2].startswith("The runs are made in 2 blocks of 8 runs each")


def test_the_runs_before_keep_their_places_and_their_cells(api):
    """The results the page holds are by place: the runs before come back as they went, a Block cell apart."""
    cube = make(api, design_type="full_factorial", factors=FACTORS, n_center_points=3, aim="quadratic")
    sent = held_of(cube)
    columns = [*sent["columns"], "Confirmation run"]
    at = {"T": 200, "P": 3, "F": 15}
    confirmation = [{"Run order": 12, "Confirmation run": "yes", **at}.get(c) for c in columns]
    rows = [*([*row, None] for row in sent["rows"]), confirmation, confirmation]
    grown = api.ok("api_add_runs", {**sent, "columns": columns, "rows": rows, "kind": "axial"})
    block = grown["columns"].index("Block")
    assert [[*row[:block], *row[block + 1 :]] for row in grown["rows"][: len(rows)]] == rows
    assert [row[block] for row in grown["rows"][: len(rows)]] == [1] * 11 + [None, None]
    sheet = run_sheet(grown)
    assert sheet["Run order"][len(rows) :].tolist() == list(range(13, 21))  # after the confirmation runs
    assert sheet["Confirmation run"][len(rows) :].isna().all()


def test_axial_runs_make_a_face_centred_central_composite_design(api):
    cube = make(api, design_type="full_factorial", factors=FACTORS, n_center_points=3, aim="quadratic")
    grown = added(api, cube, "axial")
    assert grown["model"] == "quadratic"
    new = coded(run_sheet(grown)[11:], FACTORS).to_numpy()
    axial = new[np.abs(new).sum(axis=1) > 0]
    assert sorted(map(tuple, axial)) == sorted(
        tuple(s if j == i else 0.0 for j in range(3)) for i in range(3) for s in (-1.0, 1.0)
    )
    assert (np.abs(new).sum(axis=1) == 0).sum() == 2
    assert grown["outside_range"] == []  # on the faces: inside the ranges
    assert grown["facts"]["center_runs"] == 5


def test_a_grown_study_is_analysed_with_its_blocks(api):
    cube = make(api, design_type="full_factorial", factors=FACTORS, n_center_points=3, aim="quadratic")
    grown = held_of(added(api, cube, "axial"))
    simulated = api.ok("api_simulate", {**grown, "scenario": "reactor", "seed": 3})
    analysis = api.ok("api_analyze_runs", {**grown, "responses": simulated["responses"]})
    terms = {c["term"] for c in analysis["coefficients"]}
    assert {"I(T ** 2)", "I(P ** 2)", "I(F ** 2)", "T:P"} <= terms
    assert len([c for c in analysis["coefficients"] if c["block"]]) == 1  # the shift between the two blocks


def test_center_runs_are_no_block_of_their_own(api):
    cube = make(api, design_type="full_factorial", factors=FACTORS, n_center_points=0)
    grown = added(api, cube, "center")
    assert "Block" not in grown["columns"]
    assert grown["spec"]["options"] == {"grown": True, "random_state": 42}
    sheet = run_sheet(grown)
    assert sheet[["T", "P", "F"]][8:].to_numpy().tolist() == [[175.0, 2.0, 15.0]] * 3
    assert grown["facts"]["notes"][0] == "Runs 9 to 11 are at the center of the ranges, to test for curvature."


def test_runs_chosen_for_the_aim_let_its_model_be_fitted(api, bootstrap):
    fraction = make(
        api, design_type="fractional_factorial", factors=SEVEN, resolution=3, n_center_points=0, aim="interactions"
    )
    grown = added(api, fraction, "optimal")
    assert grown["added"] == 25
    assert grown["model"] == "interactions"
    x = coded(run_sheet(grown), SEVEN).to_numpy(dtype=float)
    assert bootstrap.estimable_models(x, ["numeric"] * 7)["interactions"]


def test_the_runs_made_again_are_the_runs_before(api):
    fraction = make(api, design_type="fractional_factorial", factors=SEVEN, resolution=3, n_center_points=0)
    grown = added(api, fraction, "replicate")
    sheet = run_sheet(grown)
    names = [f["name"] for f in SEVEN]
    assert sorted(map(tuple, sheet[names][8:].to_numpy().tolist())) == sorted(
        map(tuple, sheet[names][:8].to_numpy().tolist())
    )
    assert sheet["Block"].tolist() == [1] * 8 + [2] * 8


def test_the_results_before_are_simulated_again_as_they_were(api):
    """A practice study's runs added are simulated from the same process and seed, the results before unchanged."""
    cube = make(api, design_type="full_factorial", factors=FACTORS, n_center_points=3, aim="quadratic")
    before = api.ok("api_simulate", {**held_of(cube), "scenario": "reactor", "seed": 4})["responses"]
    grown = held_of(added(api, cube, "axial"))
    after = api.ok("api_simulate", {**grown, "scenario": "reactor", "seed": 4})["responses"]
    assert after[: len(before)] == before


def test_a_grown_study_made_again_keeps_its_blocks(api):
    """Runs as given come in blocks: a grown study keeps them when the reader makes its runs again."""
    runs = [[-1, -1], [1, -1], [-1, 1], [1, 1], [0, 0], [0, 0]]
    made = make(api, design_type="given", factors=factors_named("A", "B"), runs=runs, blocks=[1, 1, 1, 1, 2, 2])
    assert made["columns"][:3] == ["Std order", "Run order", "Block"]
    assert made["spec"]["options"] == {"n_blocks": 2}
    assert made["facts"]["notes"][-1].startswith("The runs are made in 2 blocks of 4 and 2 runs")


@pytest.mark.parametrize("blocks", [None, [1, 1, 1, 1]])
def test_runs_as_given_in_one_block_are_in_none(api, blocks):
    runs = [[-1, -1], [1, -1], [-1, 1], [1, 1]]
    made = make(api, design_type="given", factors=factors_named("A", "B"), runs=runs, blocks=blocks)
    assert "Block" not in made["columns"]
    assert made["spec"]["options"] == {}


# --------------------------------------------------------------------------- the study's seed
def test_the_runs_added_are_drawn_with_the_studys_seed_plus_the_runs_before(api):
    """Each set of runs added has an order of its own, drawn from the study's seed, which the study keeps."""
    cube = make(api, design_type="full_factorial", factors=FACTORS, n_center_points=0, random_state=1792419802)
    grown = added(api, cube, "center")
    assert grown["spec"]["options"] == {"grown": True, "random_state": 1792419802}
    std = run_sheet(grown)["Std order"][8:].to_numpy() - 9
    assert std.tolist() == np.random.default_rng(1792419802 + 8).permutation(3).tolist()
    again = added(api, grown, "axial")
    assert again["spec"]["options"]["random_state"] == 1792419802
    std = run_sheet(again)["Std order"][11:].to_numpy() - 12
    assert std.tolist() == np.random.default_rng(1792419802 + 11).permutation(len(std)).tolist()


def test_a_study_saved_without_a_seed_draws_the_runs_added_with_the_pages_seed(api):
    """A study saved before each study had a seed keeps the one seed its report names, and gains none."""
    runs = [[-1, -1], [1, -1], [-1, 1], [1, 1]]
    given = make(api, design_type="given", factors=factors_named("A", "B"), runs=runs)
    grown = added(api, given, "center")
    assert grown["spec"]["options"] == {"grown": True}
    std = run_sheet(grown)["Std order"][4:].to_numpy() - 5
    assert std.tolist() == np.random.default_rng(42).permutation(len(std)).tolist()


def test_runs_as_given_keep_the_studys_seed(api):
    runs = [[-1, -1], [1, -1], [-1, 1], [1, 1]]
    made = make(api, design_type="given", factors=factors_named("A", "B"), runs=runs, random_state=1792419802)
    assert made["spec"]["options"] == {"random_state": 1792419802}
