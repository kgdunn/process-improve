"""The best settings for the response's goal: ``api_best_settings``, which Learn calls once the model is fitted.

The best settings are found on the fitted model, inside the factors' ranges and any limits on the
settings, with the library's ``optimize_responses``. These tests hold them to what can be checked
another way: a dense grid over the same model, the intervals statsmodels gives at the same point,
the leverage of the runs, and the model's own predictions beside the best settings. Confirmation
runs, made at the best settings to test the model, never enter its fit.
"""

from __future__ import annotations

import itertools
import json
import math
import re

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from scipy import stats

from process_improve.experiments import DesignRegion, Factor, analyze_experiment
from tests.web._support import FACTORS, coded, factors_named, held_of, run_sheet

#: Where the confirmation runs are marked on the run sheet, and how: as ``web/app.js`` adds them.
CONFIRMATION = "Confirmation run"


# --------------------------------------------------------------------------- helpers
def design_of(api, **spec: object) -> dict:
    """Make a design, as the page holds it."""
    return held_of(api.ok("api_make_design", spec))


def responses_of(held: dict, surface, noise: float = 0.01, seed: int = 0) -> list[float]:
    """Return results for each run from a known surface over the coded settings, plus a little noise."""
    x = coded(run_sheet(held), held["spec"]["factors"])
    return (surface(x) + np.random.default_rng(seed).normal(0.0, noise, len(x))).tolist()


def aimed(held: dict, goal: str, target: float | None = None) -> dict:
    """Give the run sheet's response a goal, as the brief does."""
    info = {"label": "Yield", "units": "%", "goal": goal, **({"target": target} if target is not None else {})}
    return {**held, "spec": {**held["spec"], "response_info": info}}


def best_of(api, held: dict, responses: list, goal: str = "maximize", target: float | None = None, **more) -> dict:
    """Return the best settings the page finds for a goal."""
    return api.ok("api_best_settings", {**aimed(held, goal, target), "responses": responses, **more})


def predict(coefficients: list[dict], x: dict[str, np.ndarray | float]) -> np.ndarray | float:
    """Evaluate a fitted model at coded settings, in the average block, as an independent check of the page's own."""
    total: np.ndarray | float = 0.0
    for c in coefficients:
        term = c["term"]
        if re.fullmatch(r"Block\d+_*", term):
            continue
        if term == "Intercept":
            value: np.ndarray | float = 1.0
        elif square := re.fullmatch(r"I\((\w+) \*\* 2\)", term):
            value = np.asarray(x[square[1]]) ** 2
        else:
            value = math.prod(np.asarray(x[p]) for p in term.split(":"))
        total = total + c["coefficient"] * value
    return total


def grid(names: list[str], points: int = 41) -> dict[str, np.ndarray]:
    """Return a dense grid over the coded ranges of the named factors, flattened."""
    axes = np.meshgrid(*[np.linspace(-1, 1, points)] * len(names), indexing="ij")
    return {name: axis.ravel() for name, axis in zip(names, axes, strict=True)}


def setting(best: dict, name: str) -> dict:
    return next(s for s in best["settings"] if s["name"] == name)


def confirmed(held: dict, responses: list, at: dict, results: list) -> tuple[dict, list]:
    """Add confirmation runs at ``at`` to a run sheet, as ``web/app.js`` adds them, with their results."""
    columns = [*held["columns"], CONFIRMATION]
    rows = [[*row, None] for row in held["rows"]]
    order = len(rows)
    for i, _ in enumerate(results):
        cells = {"Std order": None, "Run order": order + i + 1, CONFIRMATION: "yes", **at}
        rows.append([cells.get(c) for c in columns])
    return {**held, "columns": columns, "rows": rows}, [*responses, *results]


# --------------------------------------------------------------------------- the three goals, on known surfaces
@pytest.fixture(scope="module")
def square(api) -> dict:
    """Make a face-centred CCD in two factors, each from -1 to +1, so a setting is its own coded value."""
    return design_of(api, design_type="ccd", factors=factors_named("A", "B"))


def test_a_maximum_inside_the_ranges_is_found_and_named_a_stationary_point(api, square):
    responses = responses_of(square, lambda x: 50 - 3 * (x["A"] - 0.3) ** 2 - 2 * (x["B"] + 0.4) ** 2)
    best = best_of(api, square, responses)
    assert best["coded"] == pytest.approx({"A": 0.3, "B": -0.4}, abs=0.02)
    assert best["predicted"] == pytest.approx(50, abs=0.05)
    assert best["stationary"] == {"factors": ["A", "B"], "inside": True, "kind": "maximum"}
    assert [s["end"] for s in best["settings"]] == [None, None]
    assert best["ascent"] is None or not any(s["beyond"] for s in best["settings"]), "nowhere to go beyond the ranges"
    assert best["limits"] == []


def test_a_maximum_on_an_edge_says_the_best_may_lie_beyond_the_range(api, square):
    responses = responses_of(square, lambda x: 50 + 4 * x["A"] - 2 * (x["B"] - 0.25) ** 2)
    best = best_of(api, square, responses)
    a, b = setting(best, "A"), setting(best, "B")
    assert (a["value"], a["end"], a["beyond"]) == (1.0, "high", True)
    assert (b["end"], b["beyond"]) == (None, False)
    assert b["coded"] == pytest.approx(0.25, abs=0.02)
    assert best["stationary"] == {"factors": ["B"], "inside": False, "kind": "maximum"}, "B peaks inside its range"
    assert best["ascent"] == {"A": 1.0}, "the response climbs fastest along A, and A only"


def test_a_minimum_on_an_edge_points_the_other_way(api, square):
    responses = responses_of(square, lambda x: 20 + 3 * (x["A"] - 0.2) ** 2 + 2 * x["B"])
    best = best_of(api, square, responses, "minimize")
    a, b = setting(best, "A"), setting(best, "B")
    assert a["coded"] == pytest.approx(0.2, abs=0.02)
    assert (b["value"], b["end"], b["beyond"]) == (-1.0, "low", True), "lower B would lower the response further"
    assert best["predicted"] == pytest.approx(18, abs=0.05)
    assert best["ascent"] == {"B": -1.0}, "the way the response falls fastest, as a minimum wants"


def test_a_target_in_reach_is_hit(api, square):
    responses = responses_of(square, lambda x: 50 + 4 * x["A"] + 2 * x["B"])
    best = best_of(api, square, responses, "target", 53.0)
    assert best["reached"] is True
    assert best["predicted"] == pytest.approx(53.0, abs=1e-3)
    assert (best["stationary"], best["ascent"], best["limits"]) == (None, None, []), "many settings give a target"
    assert not any(s["beyond"] for s in best["settings"])


def test_a_target_out_of_reach_is_approached_as_closely_as_the_ranges_allow(api, square):
    responses = responses_of(square, lambda x: 50 + 4 * x["A"] + 2 * x["B"])
    best = best_of(api, square, responses, "target", 70.0)
    highest = best_of(api, square, responses, "maximize")
    assert best["reached"] is False
    assert best["coded"] == pytest.approx(highest["coded"], abs=1e-6), "the closest to 70 is the highest"
    assert best["predicted"] == pytest.approx(highest["predicted"])
    assert [s["beyond"] for s in best["settings"]] == [True, True], "the target lies beyond both ranges"
    below = best_of(api, square, responses, "target", 30.0)
    assert below["coded"] == pytest.approx({"A": -1.0, "B": -1.0})


# --------------------------------------------------------------------------- against a dense grid
@pytest.mark.parametrize(("goal", "target"), [("maximize", None), ("minimize", None), ("target", 60.0)])
def test_the_best_settings_agree_with_a_dense_grid_over_the_fitted_model(api, held, simulated, goal, target):
    """The worked example's fitted surface, searched on a grid of 41 settings a side: nothing on it does better."""
    analysis = api.ok("api_analyze_runs", {**held, "responses": simulated["responses"]})
    best = best_of(api, held, simulated["responses"], goal, target)
    points = grid(["T", "P", "F"])
    values = predict(analysis["coefficients"], points)
    at_best = predict(analysis["coefficients"], best["coded"])
    assert best["predicted"] == pytest.approx(at_best, abs=1e-9), "the prediction is the fitted model's"
    if goal == "maximize":
        assert at_best >= values.max() - 1e-9
        found = int(np.argmax(values))
    elif goal == "minimize":
        assert at_best <= values.min() + 1e-9
        found = int(np.argmin(values))
    else:
        assert abs(at_best - target) <= np.abs(values - target).min() + 1e-6
        return  # many settings give the target: no single grid point to land near
    step = 2 / 40
    assert [best["coded"][n] for n in "TPF"] == pytest.approx([points[n][found] for n in "TPF"], abs=step)


def test_the_worked_example_is_best_at_two_corners_and_a_curve(api, held, simulated):
    """Yield climbs with T and falls with P to the ends of their ranges; along F it curves and peaks inside."""
    best = best_of(api, held, simulated["responses"])
    assert [(s["end"], s["beyond"]) for s in best["settings"]] == [("high", True), ("low", True), (None, False)]
    assert setting(best, "T")["value"] == 200
    assert setting(best, "P")["value"] == 1
    assert 15 < setting(best, "F")["value"] < 20
    assert best["stationary"] == {"factors": ["F"], "inside": False, "kind": "maximum"}
    assert best["ascent"]["T"] == 1.0
    assert best["ascent"]["P"] < 0
    assert "F" not in best["ascent"], "F is at its best already"


# --------------------------------------------------------------------------- the intervals and the leverage
def test_the_intervals_are_those_statsmodels_gives_at_the_best_settings(api, held, simulated):
    analysis = api.ok("api_analyze_runs", {**held, "responses": simulated["responses"]})
    best = best_of(api, held, simulated["responses"])
    data = coded(run_sheet(held), held["spec"]["factors"]).assign(y=simulated["responses"])
    terms = [c["term"] for c in analysis["coefficients"] if c["term"] != "Intercept"]
    fit = smf.ols(f"y ~ {' + '.join(terms)}", data=data).fit()
    frame = fit.get_prediction(pd.DataFrame([best["coded"]])).summary_frame(alpha=0.05).iloc[0]
    assert best["predicted"] == pytest.approx(frame["mean"], rel=1e-12)
    assert best["confidence"] == pytest.approx([frame["mean_ci_lower"], frame["mean_ci_upper"]], rel=1e-10)
    assert best["prediction"] == pytest.approx([frame["obs_ci_lower"], frame["obs_ci_upper"]], rel=1e-10)
    assert best["level"] == 0.95
    assert best["max_leverage"] == pytest.approx(fit.get_influence().hat_matrix_diag.max(), rel=1e-10)
    assert best["leverage"] == pytest.approx(frame["mean_se"] ** 2 / fit.mse_resid, rel=1e-10), (
        "the leverage is the variance of the predicted mean, over the noise variance"
    )


def test_in_a_blocked_design_the_intervals_are_for_the_average_block(api):
    """The equation is for the average block, and so are the prediction and its intervals, as the library gives them."""
    held = design_of(api, design_type="box_behnken", factors=FACTORS, n_blocks=2)
    responses = responses_of(
        held, lambda x: 60 + 3 * x["T"] - 2 * x["P"] ** 2 - x["F"] ** 2 + 0.5 * x["T"] * x["P"], noise=0.5, seed=4
    )
    best = best_of(api, held, responses)
    runs = run_sheet(held)
    frame = coded(runs, held["spec"]["factors"]).assign(Block=runs["Block"].to_numpy())
    library = analyze_experiment(
        frame,
        pd.Series(responses, name="y"),
        model="quadratic",
        analysis_type="prediction",
        new_points=pd.DataFrame([best["coded"]]),
    )["predictions"][0]
    assert best["predicted"] == pytest.approx(library["predicted"], rel=1e-10)
    assert best["confidence"] == pytest.approx([library["ci_low"], library["ci_high"]], rel=1e-10)
    assert best["prediction"] == pytest.approx([library["pi_low"], library["pi_high"]], rel=1e-10)


def test_a_corner_no_run_reaches_is_an_extrapolation(api, held, simulated):
    """A Box-Behnken design has no run at a corner of the ranges, where the worked example is best."""
    best = best_of(api, held, simulated["responses"])
    assert best["leverage"] > best["max_leverage"]
    assert best["extrapolating"] is True


def test_settings_among_the_runs_are_no_extrapolation(api, square):
    responses = responses_of(square, lambda x: 50 - 3 * (x["A"] - 0.3) ** 2 - 2 * (x["B"] + 0.4) ** 2)
    best = best_of(api, square, responses)
    assert best["leverage"] < best["max_leverage"]
    assert best["extrapolating"] is False


def test_with_no_runs_left_for_the_error_there_is_no_interval(api):
    held = design_of(api, design_type="full_factorial", factors=factors_named("A", "B"), n_center_points=0)
    best = best_of(api, held, [1.0, 3.0, 2.0, 5.0], model="interactions")
    assert (best["confidence"], best["prediction"]) == (None, None)
    assert best["predicted"] == pytest.approx(5.0), "a saturated fit passes through every run"


# --------------------------------------------------------------------------- how tightly to hold each factor
def test_the_sensitivity_is_the_change_in_the_prediction_for_a_twentieth_of_each_range(api, held, simulated):
    analysis = api.ok("api_analyze_runs", {**held, "responses": simulated["responses"]})
    best = best_of(api, held, simulated["responses"])
    at = predict(analysis["coefficients"], best["coded"])
    for item, factor in zip(best["sensitivity"], FACTORS, strict=True):
        assert item["name"] == factor["name"]
        assert item["step"] == pytest.approx(0.05 * (factor["high"] - factor["low"]))
        for side, sign in (("down", -1), ("up", 1)):
            moved = {**best["coded"], factor["name"]: best["coded"][factor["name"]] + sign * 0.1}
            assert item[side] == pytest.approx(predict(analysis["coefficients"], moved) - at, abs=1e-9), side
    t, p, f = best["sensitivity"]
    assert (t["down_runnable"], t["up_runnable"]) == (True, False), "T is at its high end: up leaves the range"
    assert (p["down_runnable"], p["up_runnable"]) == (False, True), "P is at its low end"
    assert (f["down_runnable"], f["up_runnable"]) == (True, True), "F is inside its range"
    assert abs(f["down"]) < abs(t["down"]), "at its peak F matters less than T"


# --------------------------------------------------------------------------- categorical factors and limits
def test_each_level_of_a_categorical_factor_is_searched(api):
    """Supplier B lifts the response and turns T's effect round: the best is at B, with T at its low end."""
    factors = [*factors_named("T", "P"), {"name": "Supplier", "low": "A", "high": "B", "units": ""}]
    held = design_of(api, design_type="full_factorial", factors=factors, n_center_points=0, n_replicates=2)
    responses = responses_of(
        held, lambda x: 50 + x["T"] + 2 * x["P"] + 3 * x["Supplier"] - 2.5 * x["T"] * x["Supplier"], noise=0.05
    )
    best = best_of(api, held, responses, model="interactions")
    supplier = setting(best, "Supplier")
    assert (supplier["kind"], supplier["value"], supplier["coded"]) == ("categorical", "B", 1.0)
    assert best["coded"]["T"] == -1.0
    assert best["coded"]["P"] == 1.0
    analysis = api.ok("api_analyze_runs", {**held, "responses": responses, "model": "interactions"})
    corners = [dict(zip(["T", "P", "Supplier"], c, strict=True)) for c in itertools.product((-1, 1), repeat=3)]
    assert best["predicted"] == pytest.approx(max(predict(analysis["coefficients"], c) for c in corners), abs=1e-9)
    assert best["search"]["held"] == {"Supplier": 1.0}, "what the code shown holds it at"
    assert best["confirm_at"]["Supplier"] == "B"


def test_many_categorical_factors_are_judged_first_and_the_best_searched(api):
    """Six two-level labels make 64 combinations: each is judged at a sample, and the best three searched."""
    labels = [{"name": f"S{i}", "low": "a", "high": "b", "units": ""} for i in range(6)]
    held = design_of(
        api, design_type="plackett_burman", factors=[*factors_named("T"), *labels], n_center_points=0, n_replicates=2
    )
    signs = [1, -1, 1, 1, -1, -1]
    responses = responses_of(held, lambda x: 10 + 2 * x["T"] + sum(s * x[f"S{i}"] for i, s in enumerate(signs)))
    best = best_of(api, held, responses, model="main_effects")
    assert [setting(best, f"S{i}")["value"] for i in range(6)] == ["b" if s > 0 else "a" for s in signs]
    assert best["coded"]["T"] == 1.0
    assert best["predicted"] == pytest.approx(10 + 2 + 6, abs=0.1)


def test_limits_on_the_settings_are_kept_and_named_when_the_best_is_on_one(api):
    limits = [{"expression": "T + 25*P <= 215"}]
    held = design_of(api, design_type="d_optimal", factors=FACTORS, constraints=limits)
    responses = responses_of(held, lambda x: 50 + 3 * x["T"] + 2 * x["P"] - x["F"] ** 2)
    best = best_of(api, held, responses)
    t, p = setting(best, "T")["value"], setting(best, "P")["value"]
    assert t + 25 * p == pytest.approx(215, abs=1e-6), "on the limit: more of T or P would break it"
    region = DesignRegion([Factor(**f) for f in FACTORS], [limits[0]["expression"]])
    assert region.feasible(np.array([[best["coded"][n] for n in "TPF"]]))[0]
    assert best["limits"] == [{"index": 0, "expression": "T + 25 * P <= 215", "beyond": True}]
    assert best["stationary"] is None, "on a limit, the surface is flat along the limit, not along the factors"
    at = best["confirm_at"]
    assert region.feasible(np.array([[(at["T"] - 175) / 25, at["P"] - 2, (at["F"] - 15) / 5]]))[0], "rounded inside"
    analysis = api.ok("api_analyze_runs", {**held, "responses": responses})
    points = grid(["T", "P", "F"], 31)
    inside = region.feasible(np.column_stack([points[n] for n in "TPF"]))
    assert best["predicted"] >= predict(analysis["coefficients"], points)[inside].max() - 1e-9


@pytest.mark.parametrize(
    "limits",
    [
        ["T + 25*P <= 215"],
        ["T * F <= 3000"],
        ["T + 25 * P + 5 * F <= 290"],
        ["T + 25*P <= 250", "T * F <= 3600"],
    ],
    ids=["straight", "curved", "centre ruled out", "straight and curved"],
)
@pytest.mark.parametrize("goal", ["maximize", "minimize"])
def test_the_best_settings_and_the_confirmation_runs_keep_to_every_limit(api, limits, goal):
    """To the optimiser's own tolerance, a millionth in each limit's units: the best on a limit is on it."""
    held = design_of(api, design_type="d_optimal", factors=FACTORS, constraints=[{"expression": e} for e in limits])
    responses = api.ok("api_simulate", {**held, "scenario": "random", "seed": 3})["responses"]
    best = best_of(api, held, responses, goal)
    region = DesignRegion([Factor(**f) for f in FACTORS], limits)
    at = best["confirm_at"]
    settings = [[best["coded"][n] for n in "TPF"], [(at["T"] - 175) / 25, at["P"] - 2, (at["F"] - 15) / 5]]
    assert region.feasible(np.array(settings), tol=1e-6).all()


# --------------------------------------------------------------------------- what the page says beside them
def test_a_factor_the_model_leaves_out_is_held_at_the_centre(api, held, simulated):
    model = {"terms": [["T"], ["P"], ["T", "P"]]}
    best = best_of(api, held, simulated["responses"], model=model)
    f = setting(best, "F")
    assert (f["in_model"], f["value"], f["end"]) == (False, 15.0, None)
    moved = next(s for s in best["sensitivity"] if s["name"] == "F")
    assert (moved["in_model"], moved["down"], moved["up"]) == (False, 0.0, 0.0)


def test_when_no_term_is_clearly_real_the_page_says_so(api, held):
    noise = np.random.default_rng(3).normal(0, 1, len(held["rows"]))
    best = best_of(api, held, (50 + noise).tolist())
    assert best["unclear"] is True


def test_when_every_coefficient_is_zero_no_setting_is_better(api, held):
    best = best_of(api, held, [50.0] * len(held["rows"]))
    assert best["flat"] is True
    assert "settings" not in best


def test_a_goal_of_learning_asks_for_a_goal(api, held, simulated):
    assert best_of(api, held, simulated["responses"], "learn") == {"goal": "learn"}


def test_a_target_not_yet_typed_asks_for_it(api, held, simulated):
    assert best_of(api, held, simulated["responses"], "target") == {"goal": "target", "target": None}


def test_the_search_is_what_the_code_shown_reproduces(api, held, simulated):
    best = best_of(api, held, simulated["responses"])
    search = best["search"]
    assert search["held"] == {}
    assert search["goal"]["goal"] == "maximize"
    assert search["goal"]["low"] < best["predicted"] < search["goal"]["high"], "the desirability is a straight line"


def test_confirmation_settings_are_rounded_for_the_plant(api, held, simulated):
    best = best_of(api, held, simulated["responses"])
    at = best["confirm_at"]
    assert (at["T"], at["P"]) == (200, 1)
    assert at["F"] == round(setting(best, "F")["value"], 2), "a range of 10 L/min: hundredths"


# --------------------------------------------------------------------------- confirmation runs
def test_confirmation_runs_never_enter_the_fit(api, held, simulated):
    best = best_of(api, held, simulated["responses"])
    sheet, responses = confirmed(held, simulated["responses"], best["confirm_at"], [10.0, 500.0, -80.0])
    without = api.ok("api_analyze_runs", {**held, "responses": simulated["responses"]})
    with_them = api.ok("api_analyze_runs", {**sheet, "responses": responses})
    assert with_them["coefficients"] == without["coefficients"]
    assert with_them["n_completed"] == without["n_completed"]
    again = api.ok("api_best_settings", {**aimed(sheet, "maximize"), "responses": responses})
    assert again["coded"] == best["coded"], "nor the search"


def test_confirmation_runs_still_to_make_are_not_runs_still_to_do(api, held, simulated):
    best = best_of(api, held, simulated["responses"])
    sheet, responses = confirmed(held, simulated["responses"], best["confirm_at"], [None, None, None])
    assert api.ok("api_analyze_runs", {**sheet, "responses": responses})["pending_runs"] == []
    group = api.ok("api_best_settings", {**aimed(sheet, "maximize"), "responses": responses})["confirmation"][0]
    assert (group["runs"], group["done"], group["mean"], group["inside"]) == (3, 0, None, None)
    assert group["interval"][0] < group["predicted"] < group["interval"][1], "what to expect of their average"


def test_their_average_is_judged_against_the_prediction_interval_for_an_average(api, held, simulated):
    best = best_of(api, held, simulated["responses"])
    sheet, responses = confirmed(held, simulated["responses"], best["confirm_at"], [67.0, 68.5, None])
    group = api.ok("api_best_settings", {**aimed(sheet, "maximize"), "responses": responses})["confirmation"][0]
    assert group["rows"] == [15, 16, 17]
    assert (group["runs"], group["done"], group["mean"]) == (3, 2, 67.75)
    data = coded(run_sheet(held), held["spec"]["factors"]).assign(y=simulated["responses"])
    analysis = api.ok("api_analyze_runs", {**held, "responses": simulated["responses"]})
    terms = [c["term"] for c in analysis["coefficients"] if c["term"] != "Intercept"]
    fit = smf.ols(f"y ~ {' + '.join(terms)}", data=data).fit()
    at = {
        f["name"]: (group["at"][f["name"]] - (f["low"] + f["high"]) / 2) / ((f["high"] - f["low"]) / 2) for f in FACTORS
    }
    frame = fit.get_prediction(pd.DataFrame([at])).summary_frame(alpha=0.05).iloc[0]
    mean = frame["mean"]
    se = math.sqrt(frame["mean_se"] ** 2 + fit.mse_resid / 2)  # the average of two new runs varies half as much as one
    t = stats.t.ppf(0.975, fit.df_resid)
    assert group["predicted"] == pytest.approx(mean, rel=1e-12)
    assert group["interval"] == pytest.approx([mean - t * se, mean + t * se], rel=1e-10)
    assert group["inside"] is True
    far = confirmed(held, simulated["responses"], best["confirm_at"], [40.0, 41.0, 42.0])
    judged = api.ok("api_best_settings", {**aimed(far[0], "maximize"), "responses": far[1]})["confirmation"][0]
    assert judged["inside"] is False, "the process differs from the model there"


def test_confirmation_runs_at_two_settings_are_judged_apart(api, held, simulated):
    first, responses = confirmed(held, simulated["responses"], {"T": 200, "P": 1, "F": 18}, [68.0])
    rows = [*first["rows"], [None, 17, 175, 2, 15, None, "yes"]]
    groups = api.ok(
        "api_best_settings", {**aimed({**first, "rows": rows}, "maximize"), "responses": [*responses, 62.0]}
    )["confirmation"]
    assert [(g["rows"], g["at"]) for g in groups] == [
        ([15], {"T": 200, "P": 1, "F": 18}),
        ([16], {"T": 175, "P": 2, "F": 15}),
    ]


def test_simulating_again_keeps_the_design_s_results(api, held, simulated):
    """Practice results for added confirmation runs leave every result before them as it was: the same draws."""
    best = best_of(api, held, simulated["responses"])
    sheet, _ = confirmed(held, simulated["responses"], best["confirm_at"], [None] * 3)
    again = api.ok("api_simulate", {**sheet, "scenario": "reactor", "sigma": 1.0, "seed": 1})
    assert again["responses"][:15] == simulated["responses"]
    assert len(again["responses"]) == 18


def test_a_workbook_keeps_the_confirmation_runs_marked(api, held, simulated):
    best = best_of(api, held, simulated["responses"])
    sheet, responses = confirmed(held, simulated["responses"], best["confirm_at"], [67.0, 68.0, 69.0])
    book = api.ok("api_workbook", {**sheet, "responses": responses})
    read = api.ok("api_read_workbook", book)
    assert read["columns"][-1] == CONFIRMATION
    assert [row[-1] for row in read["rows"]] == [None] * 15 + ["yes"] * 3
    analysed = api.ok("api_analyze", book)
    assert analysed["n_completed"] == 15, "the confirmation runs are left out of the fit"


def test_renaming_a_factor_keeps_the_design_s_facts_beside_confirmation_runs(api, design, simulated):
    held = held_of(design)
    sheet, _ = confirmed(held, simulated["responses"], {"T": 200, "P": 1, "F": 18}, [None] * 3)
    sent = {**sheet, "facts": design["facts"], "outside_range": design["outside_range"]}
    renamed = api.ok("api_relabel", {"design": sent, "factors": [{**FACTORS[0], "name": "Temp"}, *FACTORS[1:]]})
    assert renamed["facts"]["aliases"] == design["facts"]["aliases"]
    assert renamed["facts"]["notes"] == design["facts"]["notes"]
    assert renamed["columns"][-1] == CONFIRMATION


# --------------------------------------------------------------------------- the response surface through them
def test_the_response_surface_is_a_slice_through_the_best_settings(api, held, simulated):
    pytest.importorskip("plotly", reason="the page installs plotly for its first chart")
    best = best_of(api, held, simulated["responses"])
    plots = api.ok("api_plots", {**held, "responses": simulated["responses"], "best": best["coded"]})
    assert plots["held_best"] is True
    assert plots["held"] == [{"name": "F", "value": pytest.approx(setting(best, "F")["value"])}]
    contour = json.loads(plots["figures"]["contour"])["data"][0]
    analysis = api.ok("api_analyze_runs", {**held, "responses": simulated["responses"]})
    corner = predict(analysis["coefficients"], {"T": 1.0, "P": -1.0, "F": best["coded"]["F"]})
    x, y = np.asarray(contour["x"]), np.asarray(contour["y"])
    assert contour["z"][int(np.argmin(np.abs(y + 1)))][int(np.argmin(np.abs(x - 1)))] == pytest.approx(corner)
    plain = api.ok("api_plots", {**held, "responses": simulated["responses"]})
    assert (plain["held_best"], plain["held"]) == (False, [])
