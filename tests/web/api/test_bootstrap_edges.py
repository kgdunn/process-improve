"""Edge cases of ``web/bootstrap.py`` that no other test of the page reaches, each driven as the page drives it.

Each test asks one entry point for what a reader (or a workbook) could send at an edge: a label a
spreadsheet read as TRUE, a mixture without a column, a study with no numeric factor, a confirmation
run of a fit with no error left, limits that rule out the centre or the optimum. Where the answer is a
number, it is checked against one worked out here by hand or with a simpler formula.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from process_improve.experiments import TrueProcess
from tests.web._support import FACTORS, coded, edit_workbook, factors_named, fill_workbook, held_of, run_sheet
from tests.web.api.test_balance import TWO, results_of
from tests.web.api.test_best_settings import CONFIRMATION, best_of, confirmed, setting
from tests.web.api.test_errors import SUPPLIED
from tests.web.api.test_levels import CATALYST
from tests.web.api.test_mixture import components
from tests.web.api.test_next_runs import CURVED, curved_study, results
from tests.web.api.test_responses import three

#: Two labels, each of two levels: four combinations of levels, more than the three the search takes in full.
S1 = {"name": "S1", "low": "a", "high": "b", "units": ""}
S2 = {"name": "S2", "low": "c", "high": "d", "units": ""}


def sign(row: list, columns: list[str], factor: dict) -> int:
    """Return a two-level label's coded setting in one row of a run sheet: -1 at its low label, +1 at its high."""
    return 1 if row[columns.index(factor["name"])] == factor["high"] else -1


def lattice(api) -> dict:
    """Make the {3, 2} simplex lattice in A, B and C: six blends and the centroid, for the quadratic blending model."""
    return api.ok(
        "api_make_design", {"design_type": "simplex_lattice", "lattice_degree": 2, "factors": components(*"ABC")}
    )


# --------------------------------------------------------------------------- refusals
def test_a_label_a_spreadsheet_read_as_true_is_no_level(api):
    """A cell typed TRUE comes back from a spreadsheet as a boolean, which is no label of the factor."""
    error = api.error("api_analyze_runs", {**SUPPLIED, "rows": [*SUPPLIED["rows"][:3], [4, 4, 200, True, None]]})
    assert error == "Factor S: each run must be at A or B, not True."


@pytest.mark.parametrize("blank", [None, "", "  "])
def test_a_blank_label_is_named_by_its_run_not_as_nan(api, blank):
    """A categorical cell left blank is named by the run the reader sees, its run order, not by its row (#1006)."""
    rows = [[1, 3, 150, "A", None], [2, 1, 200, "A", None], [3, 4, 150, "B", None], [4, 2, 200, blank, None]]
    error = api.error("api_analyze_runs", {**SUPPLIED, "rows": rows})
    assert error == "Factor S: run 2 has no setting; each run must be at A or B."


def test_a_mixture_workbook_without_a_component_is_refused(api):
    """A component's column deleted from the Runs sheet in Excel is named in the refusal."""
    design = lattice(api)
    filled = fill_workbook(design["xlsx_base64"], lambda runs: 10 + 5 * runs["A"])
    book = edit_workbook(filled, lambda sheets: sheets["Runs"].pop("B"))
    assert api.error("api_analyze", {"xlsx_base64": book}) == "The Runs sheet is missing column(s): B."


def test_a_mixture_needs_a_completed_run_per_coefficient(api):
    """The quadratic blending model of three components has 3 + 3 = 6 coefficients; 5 runs have their results."""
    held = held_of(lattice(api))
    error = api.error("api_analyze_runs", {**held, "responses": [1.0, 2.0, 3.0, 4.0, 5.0, None, None]})
    assert error == (
        "The quadratic blending model has 6 coefficients, so it needs at least 6 completed runs, and 5 are complete. "
        "Still to run: 6, 7."
    )


def test_a_mixture_sent_back_with_limits_on_its_settings_is_refused(api):
    """A mixture's components take bounds, not limits: a specification sent back with limits is refused."""
    held = held_of(lattice(api))
    spec = {**held["spec"], "constraints": [{"expression": "A <= 0.5"}]}
    error = api.error("api_analyze_runs", {**held, "spec": spec, "responses": [1.0] * 7})
    assert error == (
        "A mixture takes no limits on the settings on this page: give each component its lower and upper "
        "proportions instead, or remove the limits."
    )


def test_a_design_whose_options_are_not_a_mapping_is_not_one_the_page_made(api):
    """The page stores a design's options as a mapping; a list in their place is refused as data."""
    held = held_of(lattice(api))
    spec = {**held["spec"], "options": ["n_blocks", 2]}
    error = api.error("api_analyze_runs", {**held, "spec": spec, "responses": [1.0] * 7})
    assert error == "The page sent a design this page did not make."


def test_a_mixture_s_results_cannot_be_simulated_from_an_unknown_process(api):
    """A mixture's scenario must be one of its own; any other name is refused by name."""
    held = held_of(lattice(api))
    assert api.error("api_simulate", {**held, "scenario": "nonsense"}) == "Unknown process 'nonsense'."


def test_one_factor_cannot_be_folded_over(api):
    """A single factor has no two-factor interaction for its main effect to be aliased with."""
    design = api.ok(
        "api_make_design", {"design_type": "full_factorial", "factors": factors_named("A"), "n_center_points": 0}
    )
    error = api.error("api_add_runs", {**held_of(design), "kind": "foldover"})
    assert error == (
        "This study's runs cannot be extended that way: a fold-over takes a two-level design whose main effects "
        "are aliased with two-factor interactions."
    )


def test_a_factor_of_three_levels_gets_no_next_step_and_is_told_why(api):
    """The next step takes two-level labels only: the library's refusal is shown as why there is no step."""
    design = api.ok(
        "api_make_design",
        {
            "design_type": "full_factorial",
            "factors": [CATALYST, FACTORS[0]],
            "n_center_points": 0,
            "n_replicates": 2,
            "response_info": {"goal": "maximize"},
        },
    )
    reply = api.ok("api_next_runs", {**held_of(design), "responses": [50.0 + i for i in range(design["n_runs"])]})
    assert reply["recommendation"] is None
    assert reply["waiting"] == (
        "The results cannot say what to run next: Categorical factor 'Catalyst' has 3 levels; recommend_next_step "
        "takes categorical factors at two levels."
    )


# --------------------------------------------------------------------------- designs and their analysis
def test_extreme_vertices_beyond_three_runs_a_term_are_a_d_optimal_choice(api):
    """Four components each from 0.05 to 0.40 leave 12 vertices: 13 runs with the centroid, over 3 per linear term."""
    low, high = 0.05, 0.40
    vertices = set()
    for free in range(4):  # a vertex has three components at a bound; the fourth makes the blend sum to 1
        for bounds in itertools.product((low, high), repeat=3):
            rest = 1 - sum(bounds)
            if low - 1e-12 <= rest <= high + 1e-12:
                vertices.add(tuple(round(v, 9) for v in (*bounds[:free], rest, *bounds[free:])))
    assert len(vertices) == 12
    factors = components(*"ABCD", **dict.fromkeys("ABCD", (low, high)))
    design = api.ok(
        "api_make_design", {"design_type": "extreme_vertices", "mixture_model": "scheffe_linear", "factors": factors}
    )
    assert design["facts"]["notes"][1] == (
        "Your bounds leave 12 extreme vertices, more than the model needs: the runs are a D-optimal choice among those "
        "vertices, the midpoints of the edges between them and their centroid."
    )
    assert design["n_runs"] == 4 + 5, "the linear blending model's 4 terms and 5 runs to spare"


def test_a_transformed_response_s_levels_are_compared_as_measured(api):
    """The model is fitted to ln(Strength); the comparison of the two suppliers is in MPa, as the results were read."""
    supplier = {"name": "Supplier", "low": "A", "high": "B", "units": ""}
    design = api.ok(
        "api_make_design",
        {
            "design_type": "full_factorial",
            "factors": [supplier],
            "n_replicates": 4,
            "n_center_points": 0,
            "response_info": {"label": "Strength", "units": "MPa"},
        },
    )
    held = held_of(design)
    held["spec"]["analysis_transforms"] = {"y": "log"}
    strength = [(20.0 if row[2] == "B" else 10.0) + 0.1 * i for i, row in enumerate(held["rows"])]
    analysis = api.ok("api_analyze_runs", {**held, "responses": strength})
    assert analysis["notes"] == [
        (
            "The levels are compared on the results as measured, in MPa; the model is fitted to ln(Strength), so "
            "its coefficients and tests are on that scale."
        )
    ]
    at = {level: [y for row, y in zip(held["rows"], strength, strict=True) if row[2] == level] for level in "AB"}
    assert analysis["comparison"]["difference"]["estimate"] == pytest.approx(np.mean(at["B"]) - np.mean(at["A"]))


def test_a_mixture_fitted_with_no_run_to_spare_has_no_residual_plot(api):
    """Six blends with results fit the quadratic blending model's six coefficients exactly: no residuals to draw."""
    design = lattice(api)
    responses = [float(i + 1) for i in range(7)]
    centroid = next(i for i, row in enumerate(design["rows"]) if all(abs(v - 1 / 3) < 1e-6 for v in row[2:5]))
    responses[centroid] = None
    drawn = api.ok("api_plots", {**held_of(design), "responses": responses})
    assert drawn["figures"] == {}
    assert "ternary" in drawn, "the fitted response over the triangle is still drawn"


def test_a_rename_keeps_a_model_by_name_and_renames_each_response_s_terms(api):
    """A model kept by name has no factor names to change; each later response's chosen terms are renamed."""
    held = held_of(three(api))
    models = {"Cost": "interactions", "Purity": {"terms": [["T"], ["P", "P"]]}}
    spec = {**held["spec"], "analysis_model": "quadratic", "analysis_models": models}
    words = {"T": "Temperature", "P": "Pressure", "F": "Feed_rate"}
    factors = [{**f, "name": words[f["name"]]} for f in FACTORS]
    renamed = api.ok("api_relabel", {"design": {**held, "spec": spec}, "factors": factors})["spec"]
    assert renamed["analysis_model"] == "quadratic"
    assert renamed["analysis_models"] == {
        "Cost": "interactions",
        "Purity": {"terms": [["Temperature"], ["Pressure", "Pressure"]]},
    }


def test_designs_that_cannot_fit_the_aim_score_no_power(api):
    """The quadratic model in a catalyst of three levels, T and P has 12 columns.

    The intercept, the catalyst's 2 contrasts, T, P, their squares, T:P and the catalyst's 2 interactions with each
    of T and P. The Taguchi array's 9 runs are too few; the full factorial's two-level T and P, with centre runs,
    give T squared and P squared the same column.
    """
    found = api.ok(
        "api_candidates",
        {"factors": [CATALYST, *FACTORS[:2]], "model": "quadratic", "response": "y", "random_state": 7},
    )
    assert len(found["result"]["terms"]) == 1 + 2 + 2 + 2 + 1 + 2 * 2
    assert {m["design_type"] for m in found["made"]} == {"full_factorial", "taguchi"}
    for values in found["result"]["values"].values():
        assert (values["fits"], values["residual_df"]) == (False, 0)
        assert values["power_main"] is values["power_interaction"] is values["power_quadratic"] is None


def test_eighteen_pairs_are_too_many_to_count_every_pattern_of_signs(api):
    """2^18 = 262144 patterns are more than the 200000 counted: they are sampled, and the interval is not given."""
    material = {"name": "material", "low": "A", "high": "B", "units": ""}
    n = 18
    runs = [[level] for _ in range(n) for level in ("A", "B")]
    blocks = [pair for pair in range(1, n + 1) for _ in (0, 1)]
    design = api.ok("api_make_design", {"design_type": "given", "factors": [material], "runs": runs, "blocks": blocks})
    rng = np.random.default_rng(4)
    low = 10 + rng.normal(0, 1, n)
    differences = rng.normal(0.3, 1, n)
    wear = [float(v) for pair in zip(low, low + differences, strict=True) for v in pair]
    found = api.ok("api_analyze_runs", {**held_of(design), "responses": wear})["comparison"]["difference"]
    flips = next(t for t in found["tests"] if t["test"] == "sign_flip")
    assert (flips["exact"], flips["n_arrangements"]) == (False, 200_000)
    assert (flips["ci_low"], flips["ci_high"], flips["confidence"]) == (None, None, None)
    sums = np.zeros(1)
    for d in np.abs(differences):  # the signed sum under every pattern of signs, counted here
        sums = np.concatenate([sums + d, sums - d])
    exact = float(np.mean(np.abs(sums) >= abs(differences.sum()) - 1e-9))
    assert flips["p_value"] == pytest.approx(exact, abs=0.006), "the sample of 200000 is within 5 standard errors"


# --------------------------------------------------------------------------- the best settings
@pytest.mark.parametrize(("goal", "target"), [("minimize", None), ("target", 51.0)])
def test_each_level_of_a_categorical_factor_is_tried_for_every_goal(api, goal, target):
    """50 + T + 2 P + 3 S - 2.5 T S is least, 41.5, at T and P low with S at a; 51 is reached at either level."""
    factors = [*factors_named("T", "P"), S1 | {"name": "S"}]
    held = held_of(
        api.ok(
            "api_make_design",
            {"design_type": "full_factorial", "factors": factors, "n_center_points": 0, "n_replicates": 2},
        )
    )
    columns = held["columns"]
    responses = [
        50 + row[columns.index("T")] + 2 * row[columns.index("P")] + 3 * s - 2.5 * row[columns.index("T")] * s
        for row in held["rows"]
        for s in [sign(row, columns, factors[2])]
    ]
    best = best_of(api, held, responses, goal, target, model="interactions")
    if goal == "minimize":
        assert best["predicted"] == pytest.approx(41.5)
        assert best["confirm_at"] == {"T": -1.0, "P": -1.0, "S": "a"}
    else:
        assert best["predicted"] == pytest.approx(51.0, abs=1e-6)
        assert best["reached"] is True


def test_nine_numeric_factors_and_four_combinations_of_levels_are_searched_from_sampled_corners(api):
    """2^9 corners are more than the 256 judged: a sample of them is. A main effects model is best at a corner."""
    names = [f"X{i}" for i in range(9)]
    effects = {name: (i + 1) * (-1) ** i for i, name in enumerate(names)}
    held = held_of(
        api.ok(
            "api_make_design",
            {
                "design_type": "plackett_burman",
                "factors": [*factors_named(*names), S1, S2],
                "n_center_points": 0,
                "n_replicates": 2,
            },
        )
    )
    columns = held["columns"]
    responses = [
        100
        + sum(c * row[columns.index(n)] for n, c in effects.items())
        + 2 * sign(row, columns, S1)
        - 3 * sign(row, columns, S2)
        for row in held["rows"]
    ]
    best = best_of(api, held, responses, model="main_effects")
    assert best["predicted"] == pytest.approx(100 + sum(abs(c) for c in effects.values()) + 2 + 3)
    assert best["confirm_at"] == {**{n: float(np.sign(c)) for n, c in effects.items()}, "S1": "b", "S2": "c"}


def given_inside(api, limit: str, minimum: float) -> tuple[dict, list[list]]:
    """Make runs as given in A and B on a 3 x 3 grid, those inside ``A + B >= minimum``, at every pair of S1 and S2."""
    runs = [
        [a, b, s1, s2]
        for a, b in itertools.product((-1, 0, 1), repeat=2)
        for s1, s2 in itertools.product("ab", "cd")
        if a + b >= minimum
    ]
    request = {"design_type": "given", "factors": [*factors_named("A", "B"), S1, S2], "runs": runs}
    return held_of(api.ok("api_make_design", {**request, "constraints": [{"expression": limit}]})), runs


def test_the_combinations_of_levels_are_judged_inside_the_limits(api):
    """Least 20 + 2 A + B + 3 S1 - S2 with A + B >= 0.5: on the limit at A = -0.5, B = 1, S1 at a and S2 at d: 16."""
    held, runs = given_inside(api, "A + B >= 0.5", 0.5)
    responses = [20 + 2 * a + b + 3 * (1 if s1 == "b" else -1) - (1 if s2 == "d" else -1) for a, b, s1, s2 in runs]
    best = best_of(api, held, responses, "minimize", model="main_effects")
    assert best["predicted"] == pytest.approx(16.0)
    assert best["confirm_at"] == {"A": -0.5, "B": 1.0, "S1": "a", "S2": "d"}


def test_a_model_of_labels_alone_holds_the_numeric_factors_at_a_central_setting_the_limits_allow(api):
    """The centre breaks A + B >= 0.5, so A and B are held near the centroid of the triangle left, (0.5, 0.5).

    The fit of S1 and S2 alone has the runs' average for its intercept: 20 plus the average of 2 A + B over the
    runs, 1, 2 and 3 at (0, 1), (1, 0) and (1, 1), which is 2. Its best, S1 at b and S2 at c, adds 3 + 1.
    """
    held, runs = given_inside(api, "A + B >= 0.5", 0.5)
    responses = [20 + 2 * a + b + 3 * (1 if s1 == "b" else -1) - (1 if s2 == "d" else -1) for a, b, s1, s2 in runs]
    best = best_of(api, held, responses, model={"terms": [["S1"], ["S2"]]})
    assert best["predicted"] == pytest.approx(22 + 3 + 1)
    a, b = setting(best, "A"), setting(best, "B")
    assert (a["in_model"], b["in_model"]) == (False, False)
    assert a["value"] == b["value"] == pytest.approx(0.5, abs=0.02), "the region is symmetric in A and B"
    assert a["value"] + b["value"] >= 0.5


def limited(api, limit: str) -> dict:
    """Make an 8-run D-optimal design in A and B, each from -1 to +1, inside one limit."""
    request = {"design_type": "d_optimal", "factors": factors_named("A", "B"), "model_type": "main_effects"}
    return held_of(api.ok("api_make_design", {**request, "budget": 8, "constraints": [{"expression": limit}]}))


def test_a_factor_the_model_leaves_out_stays_inside_the_limits_when_its_centre_breaks_them(api):
    """Best at A = 1, where A + B >= 1.2 needs B >= 0.2: B is not moved to its centre, 0."""
    held = limited(api, "A + B >= 1.2")
    responses = [10 + 3 * row[held["columns"].index("A")] for row in held["rows"]]
    best = best_of(api, held, responses, model={"terms": [["A"]]})
    assert best["predicted"] == pytest.approx(13.0)
    assert best["coded"]["A"] == 1.0
    assert setting(best, "B")["in_model"] is False
    assert 0.2 - 1e-6 <= best["coded"]["B"] <= 1.0


def test_a_factor_with_no_effect_at_all_is_a_ridge(api):
    """A response of 10 + 2 A exactly is best at A = 1, and B, with a coefficient of 0 and no curvature, is a ridge."""
    held = held_of(
        api.ok(
            "api_make_design",
            {
                "design_type": "full_factorial",
                "factors": factors_named("A", "B"),
                "n_center_points": 1,
                "n_replicates": 2,
            },
        )
    )
    responses = [10 + 2 * row[held["columns"].index("A")] for row in held["rows"]]
    best = best_of(api, held, responses, model="main_effects")
    assert best["predicted"] == pytest.approx(12.0)
    assert best["stationary"] == {"factors": ["B"], "inside": False, "kind": "ridge"}


def test_a_maximum_in_one_factor_and_none_in_the_other_is_a_ridge(api):
    """A response of 50 - 3 (A - 0.3)^2 exactly is curved along A, flat along B, so settings along B predict alike."""
    held = held_of(api.ok("api_make_design", {"design_type": "ccd", "factors": factors_named("A", "B")}))
    responses = [50 - 3 * (row[held["columns"].index("A")] - 0.3) ** 2 for row in held["rows"]]
    best = best_of(api, held, responses, model="quadratic")
    assert best["predicted"] == pytest.approx(50.0)
    assert best["coded"]["A"] == pytest.approx(0.3, abs=1e-6)
    assert best["stationary"] == {"factors": ["A", "B"], "inside": True, "kind": "ridge"}


def test_a_confirmation_run_of_a_fit_with_no_error_left_has_no_interval(api):
    """A 2^2 fitted with its interaction on ln(y) passes through every run: at A = B = 1 it predicts 5, exactly.

    With no residual degrees of freedom there is no interval, on either scale; the sensitivity is put back in y's
    units: up along A at B = 1 the slope of ln(y) is (ln 5 - ln 2) / 2, so y moves 5 (2.5^0.05 - 1).
    """
    held = held_of(
        api.ok(
            "api_make_design",
            {"design_type": "full_factorial", "factors": factors_named("A", "B"), "n_center_points": 0},
        )
    )
    columns = held["columns"]
    corners = {(-1.0, -1.0): 1.0, (1.0, -1.0): 3.0, (-1.0, 1.0): 2.0, (1.0, 1.0): 5.0}
    y = [corners[row[columns.index("A")], row[columns.index("B")]] for row in held["rows"]]
    held["spec"]["analysis_transforms"] = {"y": "log"}
    sheet, responses = confirmed(held, y, {"A": 1.0, "B": 1.0}, [4.8])
    best = best_of(api, sheet, responses, model="interactions")
    assert best["predicted"] == pytest.approx(5.0)
    assert (best["confidence"], best["prediction"]) == (None, None)
    [group] = best["confirmation"]
    assert (group["mean"], group["predicted"]) == pytest.approx((4.8, 5.0))
    assert (group["interval"], group["inside"]) == (None, None)
    moved = next(s for s in best["sensitivity"] if s["name"] == "A")
    assert moved["up"] == pytest.approx(5 * (2.5**0.05 - 1))


# --------------------------------------------------------------------------- the balance of several responses
def test_a_response_whose_results_are_all_alike_gets_a_range_a_tenth_of_their_size_each_way(api):
    """Every cost is 20: its default range is 20 - 2 to 20 + 2, and every setting scores d = (22 - 20) / 4 = 0.5."""
    held = held_of(
        api.ok(
            "api_make_design",
            {"design_type": "ccd", "factors": FACTORS[:2], "responses": TWO, "alpha": "face_centered"},
        )
    )
    sent = results_of(held)
    sent["more_responses"]["Cost"] = [20.0] * len(held["rows"])
    reply = api.ok("api_balance", {**held, **sent, "models": {"Yield": "quadratic", "Cost": "quadratic"}})
    cost = next(r for r in reply["responses"] if r["name"] == "Cost")
    assert (cost["default_low"], cost["default_high"]) == pytest.approx((18.0, 22.0))
    assert (cost["low"], cost["high"]) == pytest.approx((18.0, 22.0))
    assert cost["d"] == pytest.approx(0.5)
    best_yield = max(sent["responses"])  # the yield's default range ends at its best result, where d reaches 1
    assert next(r for r in reply["responses"] if r["name"] == "Yield")["predicted"] >= best_yield - 1e-6
    assert reply["composite"] == pytest.approx(math.sqrt(1.0 * 0.5))


def labels_only(api, transforms: dict | None = None) -> dict:
    """Balance a yield to maximise and a cost to minimise over S1 and S2 alone, each run twice.

    Yield = 50 + 5 S1 + 2 S2 and Cost = 20 + S1 - 3 S2 exactly, on the labels' codes.
    """
    request = {"design_type": "full_factorial", "factors": [S1, S2], "responses": TWO, "n_replicates": 2}
    held = held_of(api.ok("api_make_design", {**request, "n_center_points": 0}))
    if transforms:
        held["spec"]["analysis_transforms"] = transforms
    columns = held["columns"]
    codes = [(sign(row, columns, S1), sign(row, columns, S2)) for row in held["rows"]]
    return api.ok(
        "api_balance",
        {
            **held,
            "responses": [50 + 5 * s1 + 2 * s2 for s1, s2 in codes],
            "more_responses": {"Cost": [20 + s1 - 3 * s2 for s1, s2 in codes]},
            "models": {"Yield": "main_effects", "Cost": "main_effects"},
        },
    )


def test_labels_alone_are_balanced_level_by_level(api):
    """D at each pair of levels, from the results' ranges 43 to 57 and 16 to 24: 0, 0.53, 0 and sqrt(0.75) at b, d."""
    reply = labels_only(api)
    d_yield = {(s1, s2): (50 + 5 * s1 + 2 * s2 - 43) / 14 for s1, s2 in itertools.product((-1, 1), repeat=2)}
    d_cost = {(s1, s2): (24 - (20 + s1 - 3 * s2)) / 8 for s1, s2 in itertools.product((-1, 1), repeat=2)}
    by_hand = {at: math.sqrt(d_yield[at] * d_cost[at]) for at in d_yield}
    assert max(by_hand, key=by_hand.get) == (1, 1)
    assert reply["composite"] == pytest.approx(by_hand[1, 1]) == pytest.approx(math.sqrt(0.75))
    assert reply["confirm_at"] == {"S1": "b", "S2": "d"}
    assert reply["overlay"] is None, "an overlay is over two numeric factors"


def test_a_transformed_response_balanced_over_labels_is_put_back_without_an_overlay(api):
    """ln(Yield) fitted with main effects to the four cells' ln 43, ln 47, ln 53 and ln 57 predicts at b, d.

    Its prediction, (-ln 43 + ln 47 + ln 53 + 3 ln 57) / 4, is put back as (47 * 53 * 57^3 / 43)^(1/4).
    """
    reply = labels_only(api, {"Yield": "log"})
    assert reply["overlay"] is None
    assert reply["confirm_at"] == {"S1": "b", "S2": "d"}
    yield_ = next(r for r in reply["responses"] if r["name"] == "Yield")
    assert yield_["transform"] == "log"
    assert yield_["predicted"] == pytest.approx((47 * 53 * 57**3 / 43) ** 0.25)


# --------------------------------------------------------------------------- what the results call for next
@pytest.fixture(scope="module")
def grown(api) -> tuple[dict, list[float], dict]:
    """Return a 2^3 grown into a central composite design, in two blocks, its results, and where it calls to confirm."""
    cube, y = curved_study(api)
    study = api.ok("api_add_runs", {**held_of(cube), "responses": y, "kind": "recommended"})
    y = y + results(study, CURVED, 3, start=len(y))
    called = api.ok("api_next_runs", {**held_of(study), "responses": y})["recommendation"]
    assert called["action"] == "confirm"
    return study, y, called["settings"]


def with_confirmations(study: dict, y: list[float], at: dict, *groups: list[float]) -> dict:
    """Add confirmation runs at ``at`` after a study's runs, as ``best.js`` adds them, a group at a time."""
    columns = [*study["columns"], CONFIRMATION]
    rows = [[*row, None] for row in study["rows"]]
    last = max(row[columns.index("Run order")] for row in rows)
    responses = list(y)
    for result in itertools.chain(*groups):
        last += 1
        cells = {CONFIRMATION: "yes", "Run order": last, **at}
        rows.append([cells.get(c) for c in columns])
        responses.append(result)
    return {"spec": study["spec"], "columns": columns, "rows": rows, "responses": responses}


#: Three confirmation runs that miss the prediction, about 60, by far; and three that meet it.
MISSED, MET = [40.0, 41.0, 42.0], [60.0, 60.5, 59.8]


def test_confirmation_settings_outside_the_limits_are_given_unrounded_and_not_offered(api, grown):
    """The optimum, near A = 0.17 and B = -0.21, breaks A + B <= -0.5: no rounding brings it inside, so none is kept."""
    study, y, settings = grown
    spec = {**study["spec"], "constraints": [{"expression": "A + B <= -0.5"}]}
    called = api.ok("api_next_runs", {**held_of(study), "spec": spec, "responses": y})["recommendation"]
    assert called["action"] == "confirm"
    assert (called["does"], called["label"], called["outside_limits"]) == (None, None, True)
    assert called["settings"]["A"] + called["settings"]["B"] > -0.5
    for name in "ABC":  # the rounded settings without the limit are the same settings to three decimals
        assert called["settings"][name] == pytest.approx(settings[name], abs=5e-4)
    assert called["settings"]["A"] != round(called["settings"]["A"], 3), "unrounded"


def without_run_order(request: dict) -> dict:
    """Return a run sheet request without its Run order column."""
    at = request["columns"].index("Run order")
    return {
        **request,
        "columns": [c for c in request["columns"] if c != "Run order"],
        "rows": [[v for i, v in enumerate(row) if i != at] for row in request["rows"]],
    }


def test_without_a_run_order_the_latest_confirmation_runs_are_the_last_on_the_sheet(api, grown):
    """With no Run order column, the last three confirmation runs on the sheet are the ones judged."""
    study, y, settings = grown
    met_last = without_run_order(with_confirmations(study, y, settings, MISSED, MET))
    assert api.ok("api_next_runs", met_last)["recommendation"]["action"] == "stop"
    missed_last = without_run_order(with_confirmations(study, y, settings, MET, MISSED))
    assert api.ok("api_next_runs", missed_last)["recommendation"]["action"] == "investigate"


def test_confirmation_runs_that_missed_before_are_fitted_as_runs_of_the_latest_block(api, grown):
    """Three missed confirmation runs, then three that meet the prediction: the first three join block 2.

    The step is the one for the same three runs entered as runs of block 2, before the three confirmation runs.
    """
    study, y, settings = grown
    twice = api.ok("api_next_runs", with_confirmations(study, y, settings, MISSED, MET))["recommendation"]
    once = with_confirmations(study, y, settings, MET)
    columns = once["columns"]
    first_confirmation = len(study["rows"])
    last = max(row[columns.index("Run order")] for row in study["rows"])
    as_runs = [
        [{"Block": 2, "Run order": last + 1 + i, **settings}.get(c) for c in columns] for i in range(len(MISSED))
    ]
    rows = [*once["rows"][:first_confirmation], *as_runs, *once["rows"][first_confirmation:]]
    responses = [*y, *MISSED, *MET]
    in_block = api.ok("api_next_runs", {**once, "rows": rows, "responses": responses})["recommendation"]
    assert twice == in_block
    assert twice["action"] == "stop"


def test_a_new_study_further_along_keeps_a_categorical_factor_s_levels(api):
    """The path moves A by half its range and B by its slope's share of A's, to two figures; S keeps a and b."""
    factors = [*factors_named("A", "B"), S1 | {"name": "S"}]
    cube = api.ok(
        "api_make_design",
        {
            "design_type": "full_factorial",
            "factors": factors,
            "n_center_points": 0,
            "response": "y",
            "response_info": {"goal": "maximize"},
        },
    )
    truth = TrueProcess({"Intercept": 50, "A": 5, "B": 4, "S": 1}, sigma=0.5)
    y = truth.simulate(coded(run_sheet(cube), cube["spec"]["factors"]), random_state=1).round(3).tolist()
    called = api.ok("api_next_runs", {**held_of(cube), "responses": y})["recommendation"]
    assert (called["action"], called["does"]) == ("move_region", "study")
    fitted = {
        c["term"]: c["coefficient"]
        for c in api.ok("api_analyze_runs", {**held_of(cube), "responses": y})["coefficients"]
    }
    shift = float(f"{fitted['B'] / fitted['A']:.2g}")
    assert called["ranges"] == [
        {"name": "A", "low": 0.0, "high": 2.0, "units": ""},
        {"name": "B", "low": pytest.approx(-1 + shift), "high": pytest.approx(1 + shift), "units": ""},
        {"name": "S", "low": "a", "high": "b", "units": ""},
    ]


# --------------------------------------------------------------------------- runs added to a study
def square_inside(api) -> dict:
    """Make the 2^2 in A and B as given, inside the limit A + B <= 2, which its corner (1, 1) is on."""
    request = {"design_type": "given", "factors": factors_named("A", "B"), "runs": [[-1, -1], [1, -1], [-1, 1], [1, 1]]}
    return held_of(api.ok("api_make_design", {**request, "constraints": [{"expression": "A + B <= 2"}]}))


def test_runs_made_again_inside_the_limits_say_so(api):
    """Every run before and every run added keeps to the limit, and the notes say so."""
    extended = api.ok("api_add_runs", {**square_inside(api), "kind": "replicate"})
    assert extended["added"] == 4
    assert extended["facts"]["notes"][0] == "Runs 5 to 8 make every run of the design again."
    assert extended["facts"]["notes"][-1] == "Every run is inside your limits on the settings."


def test_a_run_sheet_without_a_std_order_numbers_the_runs_added_from_its_run_order(api):
    """A sheet with no Std order column gains none; the runs added follow its Run order, 5 to 8."""
    held = square_inside(api)
    at = held["columns"].index("Std order")
    sheet = {
        **held,
        "columns": [c for c in held["columns"] if c != "Std order"],
        "rows": [[v for i, v in enumerate(row) if i != at] for row in held["rows"]],
    }
    extended = api.ok("api_add_runs", {**sheet, "kind": "replicate"})
    assert extended["columns"] == ["Run order", "Block", "A", "B", "y"]
    assert [row[0] for row in extended["rows"]] == list(range(1, 9))
    assert extended["facts"]["notes"][0] == "Runs 5 to 8 make every run of the design again."


def test_runs_made_again_keep_a_categorical_factor_s_labels(api):
    """Each run made again sets the label at its level, as the run it repeats does."""
    design = api.ok(
        "api_make_design",
        {"design_type": "full_factorial", "factors": [*factors_named("A"), S1 | {"name": "S"}], "n_center_points": 0},
    )
    sheet = run_sheet(api.ok("api_add_runs", {**held_of(design), "kind": "replicate"}))
    before, after = sheet.iloc[:4], sheet.iloc[4:]
    assert sorted(zip(after["A"], after["S"], strict=True)) == sorted(zip(before["A"], before["S"], strict=True))
    assert set(after["S"]) == {"a", "b"}, "each run made again is at one of the labels, not at a code"
