"""The settings that give a target: ``api_target_settings`` and ``api_solve_for``, which Learn calls for its targets.

The reader types what a response should be, or several, and each response's model is run backwards to settings at
which it predicts that. These tests hold the settings found to surfaces known here: the targets are met where the
surfaces can meet them, the answer is the one nearest the center of the ranges, the limits on the settings and the
levels of a categorical factor are kept, and one factor is solved for from the others where the surface crosses the
target.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.web._support import FACTORS, coded, held_of, run_sheet
from tests.web.api.test_balance import SURFACES, TWO, design_of, results_of


def asked(held: dict, **more) -> dict:
    """Return a request about the known surfaces, every response fitted with the quadratic model."""
    names = [r["name"] for r in held["spec"]["responses"]]
    return {**held, **results_of(held), "models": dict.fromkeys(names, "quadratic"), **more}


def found_for(api, held: dict, targets: dict, **more) -> dict:
    """Return the settings the page finds for some targets on the known surfaces."""
    return api.ok("api_target_settings", asked(held, targets=targets, **more))


def at(found: dict) -> tuple[float, float]:
    """Return the settings found for T and P, coded from the ranges the tests' factors have."""
    by = {s["name"]: s["value"] for s in found["settings"]}
    return (by["T"] - 175) / 25, by["P"] - 2


def test_one_target_is_met_at_the_settings_nearest_the_center_of_the_ranges(api):
    found = found_for(api, design_of(api, TWO), {"Cost": 20.0})  # the cost is 20 at the center, and along a line
    assert found["reached"]
    t, p = at(found)
    assert SURFACES["Cost"](t, p) == pytest.approx(20.0, abs=1e-6), "the model is the surface, met exactly"
    assert max(abs(t), abs(p)) < 0.05, "of the settings on that line, the ones at the center"
    (cost,) = found["responses"]
    assert (cost["name"], cost["target"], cost["given"]) == ("Cost", 20.0, True)
    assert cost["predicted"] == pytest.approx(20.0, abs=1e-6)
    assert 0 < found["share"] < 0.2, "a thin band of the settings tried gives it"
    assert (found["tried"], found["near"]) == (20_000, 0.02)
    assert found["search"]["drawn"] == 20_000
    assert found["search"]["seed"] == 1
    assert not found["extrapolating"]


def test_two_targets_are_met_together_where_the_surfaces_cross(api):
    found = found_for(api, design_of(api, TWO), {"Yield": 61.0, "Cost": 21.0})
    assert found["reached"]
    t, p = at(found)
    assert SURFACES["Yield"](t, p) == pytest.approx(61.0, abs=1e-5)
    assert SURFACES["Cost"](t, p) == pytest.approx(21.0, abs=1e-5)
    assert [r["given"] for r in found["responses"]] == [True, True]
    assert found["confirm_at"].keys() == {"T", "P"}


def test_a_target_no_setting_gives_is_answered_with_the_settings_that_come_closest(api):
    found = found_for(api, design_of(api, TWO), {"Yield": 100.0})  # the surface peaks at 61.83, at t = 2/3, p = 1/2
    assert not found["reached"]
    assert found["share"] == 0
    assert at(found) == pytest.approx((2 / 3, 0.5), abs=1e-3)
    (yields,) = found["responses"]
    assert yields["predicted"] == pytest.approx(60 + 4 / 3 + 0.5, abs=1e-4)
    assert not yields["given"]


def test_the_settings_found_keep_to_the_limits_on_the_settings(api):
    limit = {"expression": "T + 25*P <= 215"}
    held = design_of(api, TWO, design_type="d_optimal", constraints=[limit], budget=12, model="quadratic")
    found = found_for(api, held, {"Cost": 18.0})  # met along a line that the limit cuts
    by = {s["name"]: s["value"] for s in found["settings"]}
    assert by["T"] + 25 * by["P"] <= 215 + 1e-6
    assert found["reached"]
    assert found["overlay"]["runnable"] is not None, "the chart greys what the limit rules out"
    # A target only settings past the limit give: the answer stays inside, and says the target is not reached.
    past = found_for(api, held, {"Cost": 22.0})  # t + p <= -0.4 inside the limit, and 22 needs 3 t + p near 2
    by = {s["name"]: s["value"] for s in past["settings"]}
    assert by["T"] + 25 * by["P"] <= 215 + 1e-6
    assert not past["reached"]


def test_a_categorical_factor_is_at_the_level_that_gives_the_target(api):
    factors = [*FACTORS[:2], {"name": "S", "low": "A", "high": "B", "units": ""}]
    spec = {"design_type": "full_factorial", "factors": factors, "responses": TWO[:1], "n_center_points": 3}
    held = held_of(api.ok("api_make_design", spec))
    x = coded(run_sheet(held), factors)
    yields = (60 + 4 * x["T"] - 2 * x["P"] + 3 * x["S"]).tolist()  # 51 to 63 with A, 57 to 69 with B
    request = {**held, "responses": yields, "models": {"Yield": "main_effects"}}
    high = api.ok("api_target_settings", {**request, "targets": {"Yield": 66.0}})
    assert high["reached"]
    assert {s["name"]: s["value"] for s in high["settings"]}["S"] == "B", "only B reaches 66"
    assert high["search"]["held"] == {"S": 1.0}
    low = api.ok("api_target_settings", {**request, "targets": {"Yield": 53.0}})
    assert {s["name"]: s["value"] for s in low["settings"]}["S"] == "A", "only A reaches 53"
    assert [h for h in low["overlay"]["held"] if h["name"] == "S"] == [{"name": "S", "value": "A"}]


def test_the_chart_holds_each_response_s_line_of_settings_and_the_runs(api):
    held = design_of(api, TWO)
    found = found_for(api, held, {"Yield": 61.0, "Cost": 21.0}, pair=["P", "T"])
    overlay = found["overlay"]
    assert (overlay["x"], overlay["y"]) == ("P", "T"), "the two factors the reader chose, in their order"
    assert overlay["acceptable"] is None, "a line each, and nothing shaded"
    assert [(s["name"], s["levels"]) for s in overlay["responses"]] == [("Yield", [61.0]), ("Cost", [21.0])]
    t, p = at(found)
    assert overlay["at"] == pytest.approx([p, t], abs=1e-9)
    grid = np.asarray(overlay["grid"])
    across, up = np.meshgrid(grid, grid)  # P across, T up
    np.testing.assert_allclose(overlay["responses"][0]["z"], SURFACES["Yield"](up, across), rtol=1e-5)
    # A face-centered design in two factors has nine places: its corners, its faces and its center.
    assert sorted(map(tuple, overlay["runs"])) == sorted((float(a), float(b)) for a in (-1, 0, 1) for b in (-1, 0, 1))
    assert found_for(api, held, {"Yield": 61.0})["overlay"]["x"] == "T", (
        "with no pair chosen, the factors that matter most"
    )


def test_a_transformed_response_s_target_is_typed_and_answered_in_its_own_units(api):
    held = design_of(api, TWO)
    held = {**held, "spec": {**held["spec"], "analysis_transforms": {"Yield": "log"}}}
    found = found_for(api, held, {"Yield": 61.0})
    (yields,) = found["responses"]
    assert yields["target"] == pytest.approx(61.0)
    assert yields["predicted"] == pytest.approx(61.0, rel=1e-4)
    assert yields["transform"] == "log"
    assert found["search"]["targets"]["Yield"] == pytest.approx(np.log(61.0)), "the search is on the model's scale"
    assert found["overlay"]["responses"][0]["levels"] == pytest.approx([61.0])


def solved(api, held: dict, response: str, target: float, **more) -> dict:
    """Return what T must be for a response to meet a target on the known surfaces, with P at 2 bar."""
    request = {"response": response, "target": target, "factor": "T", "settings": {"T": 160, "P": 2}}
    return api.ok("api_solve_for", asked(held, **{**request, **more}))


def test_one_factor_is_solved_for_where_the_surface_crosses_the_target(api):
    held = design_of(api, TWO)
    cost = solved(api, held, "Cost", 21.5)  # 20 + 3 t + t^2 / 2 = 21.5 at t = sqrt(12) - 3
    assert (cost["response"], cost["factor"], cost["target"]) == ("Cost", "T", 21.5)
    (one,) = cost["solutions"]
    assert one["value"] == pytest.approx(175 + 25 * (np.sqrt(12) - 3), abs=1e-3)
    # The yield along T, 60 + 4 t - 3 t^2, is 61 twice: at t = 1/3 and at the high end of the range.
    twice = [s["value"] for s in solved(api, held, "Yield", 61.0)["solutions"]]
    assert twice == pytest.approx([175 + 25 / 3, 200.0], abs=1e-3)
    # With P at its low end the yield is 4 lower along T, and 61 is out of reach: its peak, 57.33, is the nearest.
    none = solved(api, held, "Yield", 61.0, settings={"P": 1})
    assert none["solutions"] == []
    assert none["nearest"]["value"] == pytest.approx(175 + 25 * 2 / 3, abs=0.1)
    assert none["nearest"]["predicted"] == pytest.approx(60 + 4 / 3 - 4, abs=1e-3)
    # Without noise the fit leaves no doubt, and the interval closes on the solution.
    assert one["interval"] == pytest.approx([one["value"]] * 2, abs=0.07)


def test_a_solution_s_interval_is_where_the_prediction_s_interval_still_holds_the_target(api):
    held = design_of(api, TWO, n_center_points=4)
    noisy = np.random.default_rng(3).normal(0, 0.3, len(held["rows"]))
    results = {**results_of(held)}
    results["more_responses"] = {"Cost": (np.asarray(results["more_responses"]["Cost"]) + noisy).tolist()}
    request = {**asked(held), **results, "response": "Cost", "target": 21.5, "factor": "T", "settings": {"P": 2}}
    (one,) = api.ok("api_solve_for", request)["solutions"]
    low, high = one["interval"]
    assert 150 < low < one["value"] < high < 200
    assert high - low < 25, "the factor is known to a part of its range"
    # A target the interval of the prediction holds to the end of the range has no end there.
    (edge,) = api.ok("api_solve_for", {**request, "target": 17.7})["solutions"]
    assert edge["interval"][0] is None
    assert edge["value"] < edge["interval"][1] < 175


def test_settings_that_give_every_target_are_not_left_for_settings_that_miss_one(api):
    """The step that brings the predictions nearer their targets, taken together, may not push one out of its reach."""
    three = [*TWO, {"name": "Purity", "label": "Purity", "units": "%", "goal": "learn"}]
    held = design_of(api, three)
    x = coded(run_sheet(held), FACTORS[:2])
    results = {
        "responses": x["T"].tolist(),
        "more_responses": {"Cost": x["P"].tolist(), "Purity": (x["T"] + x["P"]).tolist()},
    }
    request = {**held, **results, "models": dict.fromkeys(("Yield", "Cost", "Purity"), "main_effects")}
    # Each of the first two within 0.04 of 0, and their sum within 0.08 of 0.13: a sliver of the settings.
    found = api.ok("api_target_settings", {**request, "targets": {"Yield": 0.0, "Cost": 0.0, "Purity": 0.13}})
    assert 0 < found["share"] < 0.001
    assert found["reached"], "a setting tried gives all three, and the answer is one that does"
    assert [r["given"] for r in found["responses"]] == [True, True, True]


def test_a_factor_in_no_response_s_model_is_at_the_center_of_its_range(api):
    held = held_of(api.ok("api_make_design", {"design_type": "box_behnken", "factors": FACTORS, "responses": TWO[:1]}))
    x = coded(run_sheet(held), FACTORS)
    yields = (60 + 4 * x["T"] + 2 * x["P"]).tolist()
    request = {**held, "responses": yields, "models": {"Yield": {"terms": [["T"], ["P"]]}}}
    found = api.ok("api_target_settings", {**request, "targets": {"Yield": 61.0}})
    flow = next(s for s in found["settings"] if s["name"] == "F")
    assert (flow["value"], flow["in_model"]) == (15.0, False)
    assert found["reached"]


#: A temperature with a range, and a supplier, A or B: one factor to move, and no two to draw a chart over.
WITH_SUPPLIER = [FACTORS[0], {"name": "S", "low": "A", "high": "B", "units": ""}]


def supplied(api, **more) -> tuple[dict, list[float]]:
    """Make a factorial in a temperature and a supplier, and its yields, 60 + 4 t + 3 s + t s; return both."""
    spec = {
        "design_type": "full_factorial",
        "factors": WITH_SUPPLIER,
        "responses": TWO[:1],
        "n_center_points": 0,
        **more,
    }
    held = held_of(api.ok("api_make_design", spec))
    x = coded(run_sheet(held), WITH_SUPPLIER)
    return held, (60 + 4 * x["T"] + 3 * x["S"] + x["T"] * x["S"]).tolist()


def test_a_study_with_one_factor_with_a_range_has_its_settings_and_no_chart(api):
    held, yields = supplied(api, n_replicates=2)
    request = {**held, "responses": yields, "models": {"Yield": "interactions"}}
    found = api.ok("api_target_settings", {**request, "targets": {"Yield": 62.0}})
    assert found["overlay"] is None, "a chart is over two factors with ranges"
    assert found["reached"]
    by = {s["name"]: s["value"] for s in found["settings"]}
    # With B the yield is 63 + 5 t, on 62 at t = -0.2; with A it is 57 + 3 t, which stays below 62.
    assert by["S"] == "B"
    assert by["T"] == pytest.approx(175 - 0.2 * 25, abs=1e-3)
    # A transformed response is answered in its own units there too, with no chart to put back.
    logged = {**request, "spec": {**held["spec"], "analysis_transforms": {"Yield": "log"}}}
    found = api.ok("api_target_settings", {**logged, "targets": {"Yield": 62.0}})
    assert found["overlay"] is None
    assert found["responses"][0]["predicted"] == pytest.approx(62.0, rel=1e-3)


def test_a_factor_is_solved_for_at_a_categorical_factor_s_level_with_no_interval_from_a_fit_with_no_run_to_spare(api):
    held, yields = supplied(api)  # four runs and four terms: nothing is left to estimate the noise from
    request = {**held, "responses": yields, "models": {"Yield": "interactions"}}
    asked_for = {"response": "Yield", "target": 62.0, "factor": "T"}
    (with_b,) = api.ok("api_solve_for", {**request, **asked_for, "settings": {"S": "B"}})["solutions"]
    assert with_b["value"] == pytest.approx(170.0, abs=1e-3)
    assert with_b["interval"] is None
    with_a = api.ok("api_solve_for", {**request, **asked_for, "settings": {"S": "A"}})
    assert with_a["solutions"] == [], "57 + 3 t stays below 62"
    assert with_a["nearest"] == pytest.approx({"value": 200.0, "predicted": 60.0})


def test_a_factor_that_does_not_change_the_response_is_not_solved_for(api):
    """A factor its model leaves out, or whose effect cancels at the other settings: every setting predicts alike."""
    held = design_of(api, TWO[:1])
    x = coded(run_sheet(held), FACTORS[:2])
    refused = (
        "{} does not change Yield at these settings, by its model: no setting of it gives the target more than another."
    )
    left_out = {**held, "responses": (60 + 2 * x["P"]).tolist(), "models": {"Yield": {"terms": [["P"]]}}}
    asked_for = {"response": "Yield", "target": 60.5, "factor": "T", "settings": {"P": 2}}
    assert api.error("api_solve_for", {**left_out, **asked_for}) == refused.format("T")
    # 60 + 2 p - 2 p t: with t at +1 (200 degC) the pressure's effect is gone.
    cancelled = {**held, "responses": (60 + 2 * x["P"] - 2 * x["P"] * x["T"]).tolist()}
    cancelled["models"] = {"Yield": {"terms": [["P"], ["T", "P"]]}}
    asked_for = {"response": "Yield", "target": 60.0, "factor": "P", "settings": {"T": 200}}
    assert api.error("api_solve_for", {**cancelled, **asked_for}) == refused.format("P")
