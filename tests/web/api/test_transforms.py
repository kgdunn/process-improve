"""Analysing a transformed response: ``analysis_transforms`` in the specification, as the page and a workbook keep it.

The model is fitted to the response's log, square root or -1/y (an increasing form of 1/y, as Box-Cox's is),
and what the replies say in the response's units is put back in them: a prediction becomes the response's median.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import analyze_experiment
from tests.web._support import FACTORS, held_of


def transformed(held: dict, transform: str | None, response: str | None = None) -> dict:
    """Return what the page holds with its response, or the one named, analysed with ``transform``."""
    spec = held["spec"]
    return {**held, "spec": {**spec, "analysis_transforms": {response or spec["response"]: transform}}}


def coded(held: dict) -> pd.DataFrame:
    """Return the runs' settings coded to -1 and +1 from the factors' ranges, as the page fits them."""
    rows = pd.DataFrame(held["rows"], columns=held["columns"])
    return pd.DataFrame(
        {
            f["name"]: (rows[f["name"]] - (f["low"] + f["high"]) / 2) / ((f["high"] - f["low"]) / 2)
            for f in held["spec"]["factors"]
        }
    )


def lognormal(held: dict) -> list[float]:
    """Results whose log is linear in the coded settings, with a little noise: a log transform fits them."""
    x = coded(held).to_numpy().T
    noise = np.random.default_rng(7).normal(0, 0.02, x.shape[1])
    return list(np.exp(3 + 0.6 * x[0] - 0.4 * x[1] + 0.2 * x[2] + noise))


# --------------------------------------------------------------------------- the analysis
def test_a_log_analysis_is_the_library_s_fit_of_the_log(api, held):
    y = lognormal(held)
    plain = api.ok("api_analyze_runs", {**held, "responses": y, "model": "main_effects"})
    logged = api.ok("api_analyze_runs", {**transformed(held, "log"), "responses": y, "model": "main_effects"})
    assert (plain["transform"], logged["transform"]) == (None, "log")
    expected = analyze_experiment(
        coded(held), pd.Series(np.log(y), name="y"), model="main_effects", analysis_type="coefficients", coding="actual"
    )["coefficients"]
    assert {c["term"]: round(c["coefficient"], 9) for c in logged["coefficients"]} == {
        c["term"]: round(c["coefficient"], 9) for c in expected
    }
    assert logged["equation"].startswith("ln(")


def test_on_the_log_scale_a_main_effect_is_a_ratio_from_low_to_high(api, held):
    y = lognormal(held)
    logged = api.ok("api_analyze_runs", {**transformed(held, "log"), "responses": y, "model": "main_effects"})
    first = held["spec"]["factors"][0]["name"]
    c = next(c for c in logged["coefficients"] if c["term"] == first)
    assert c["ratio"] == pytest.approx(np.exp(2 * c["coefficient"]))
    assert c["ratio"] == pytest.approx(np.exp(1.2), rel=0.05), "the process's ln y rises 0.6 a coded unit"
    assert "ratio" not in next(c for c in logged["coefficients"] if c["term"] == "Intercept")


def test_the_inverse_fits_minus_one_over_y_and_tests_as_one_over_y_does(api, held):
    """-1/y keeps the results' order, so a goal means the same on its scale; its tests are 1/y's."""
    y = lognormal(held)
    inverse = api.ok("api_analyze_runs", {**transformed(held, "inverse"), "responses": y, "model": "main_effects"})
    library = analyze_experiment(
        coded(held),
        pd.Series(y, name="y"),
        model="main_effects",
        transform="inverse",
        coding="actual",
        analysis_type=["coefficients", "anova"],
    )
    ours = {c["term"]: c["coefficient"] for c in inverse["coefficients"]}
    theirs = {c["term"]: c["coefficient"] for c in library["coefficients"]}
    assert ours == pytest.approx({term: -b for term, b in theirs.items()})
    assert {a["source"]: a["F"] for a in inverse["anova"] if a.get("F") is not None} == pytest.approx(
        {a["source"]: a["F"] for a in library["anova_table"] if a.get("F") is not None}
    )


def test_box_cox_points_to_the_log_for_results_whose_log_is_linear(api, held):
    y = lognormal(held)
    plain = api.ok("api_analyze_runs", {**held, "responses": y, "model": "main_effects"})
    hint = plain["box_cox"]
    assert hint["suggests"] == "log"
    assert hint["ci"][0] <= 0 <= hint["ci"][1] < 1
    assert hint["current"] is None
    logged = api.ok("api_analyze_runs", {**transformed(held, "log"), "responses": y, "model": "main_effects"})
    assert (logged["box_cox"]["suggests"], logged["box_cox"]["current"]) == ("log", "log"), "judged on the results"


def test_box_cox_suggests_nothing_when_the_results_need_no_transform(api, held, simulated):
    plain = api.ok("api_analyze_runs", {**held, "responses": simulated["responses"]})
    assert plain["box_cox"] is None or plain["box_cox"]["suggests"] is None


def test_a_simulated_process_s_true_values_are_not_set_beside_a_transformed_fit(api, held, simulated):
    request = {**transformed(held, "sqrt"), "responses": simulated["responses"], "truth": simulated["truth"]}
    analysis = api.ok("api_analyze_runs", request)
    assert all("true_value" not in c for c in analysis["coefficients"])


def test_a_result_the_transform_cannot_take_is_refused_naming_its_run(api, held):
    y = lognormal(held)
    y[3] = 0.0
    message = api.error("api_analyze_runs", {**transformed(held, "log"), "responses": y})
    order = pd.DataFrame(held["rows"], columns=held["columns"])["Run order"].iloc[3]
    assert message == (
        f"The log of {held['spec']['response']} needs every result above 0, and run {order} has 0. "
        "Analyse it as it is, or with another transform."
    )
    assert api.ok("api_analyze_runs", {**transformed(held, "sqrt"), "responses": y}), "a square root takes 0"


def test_a_transform_the_page_does_not_offer_is_refused(api, held):
    message = api.error("api_analyze_runs", {**transformed(held, "box_cox"), "responses": lognormal(held)})
    assert message == "A response is analysed as it is, or as its log, square root or inverse."


def test_a_transform_of_a_response_since_renamed_is_dropped(api, held):
    spec = {**held["spec"], "analysis_transforms": {"no_longer_here": "log"}}
    analysis = api.ok("api_analyze_runs", {**held, "spec": spec, "responses": lognormal(held)})
    assert analysis["transform"] is None


def test_a_workbook_keeps_the_transform_and_is_analysed_with_it(api, held):
    y = lognormal(held)
    sent = transformed(held, "log")
    written = api.ok("api_workbook", {**sent, "responses": y})
    read = api.ok("api_read_workbook", {"xlsx_base64": written["xlsx_base64"]})
    assert read["spec"]["analysis_transforms"] == {held["spec"]["response"]: "log"}
    analysis = api.ok("api_analyze_runs", {**held_of(read), "responses": read["responses"]})
    assert analysis["transform"] == "log"
    assert api.ok("api_analyze", {"xlsx_base64": written["xlsx_base64"]})["transform"] == "log"


# --------------------------------------------------------------------------- the best settings
def aimed(held: dict, transform: str, **info: object) -> dict:
    """Return what the page holds with its response transformed and given a goal."""
    sent = transformed(held, transform)
    return {**sent, "spec": {**sent["spec"], "response_info": {**sent["spec"].get("response_info", {}), **info}}}


def test_a_target_is_met_in_the_response_s_units_and_its_interval_is_wider_above(api, held):
    """The search is on ln y, with the target's log; the reply says the prediction and its interval in y."""
    request = {**aimed(held, "log", goal="target", target=25.0), "responses": lognormal(held), "model": "main_effects"}
    best = api.ok("api_best_settings", request)
    assert (best["target"], best["reached"]) == (25.0, True)
    assert best["predicted"] == pytest.approx(25.0, rel=1e-3)
    low, high = best["confidence"]
    assert low < 25.0 < high
    assert high - 25.0 > 25.0 - low, "a log scale's interval is wider above the median than below"


def test_the_best_of_a_logged_response_is_its_median_at_the_best_corner(api, held):
    request = {**aimed(held, "log", goal="maximize"), "responses": lognormal(held), "model": "main_effects"}
    best = api.ok("api_best_settings", request)
    assert best["predicted"] == pytest.approx(np.exp(3 + 0.6 + 0.4 + 0.2), rel=0.05)
    first = held["spec"]["factors"][0]["name"]  # its ln y rises 0.6 a coded unit
    step = next(s for s in best["sensitivity"] if s["name"] == first)
    assert step["down"] < 0 < step["up"]
    assert step["up"] > -step["down"], "on a log scale a step up raises the median more than one down lowers it"


# --------------------------------------------------------------------------- the pictures, the what-if, the balance
def test_the_surface_of_a_logged_response_is_drawn_in_its_units_and_its_residuals_on_its_scale(api, held):
    reply = api.ok("api_plots", {**transformed(held, "log"), "responses": lognormal(held), "model": "main_effects"})
    contour = json.loads(reply["figures"]["contour"])
    z = [v for t in contour["data"] if t.get("type") == "contour" and t.get("z") for row in t["z"] for v in row]
    assert min(v for v in z if v is not None) > np.exp(2), "the median, put back, not its log, which is about 3"
    residuals = json.loads(reply["figures"]["residuals"])
    assert residuals["layout"]["xaxis"]["title"]["text"] == "Fitted ln(Yield)"
    assert residuals["layout"]["yaxis"]["title"]["text"] == "Residual", "a log has no units"


def test_the_what_if_models_say_which_response_is_on_which_scale(api, held):
    y = lognormal(held)
    reply = api.ok("api_predictors", {**transformed(held, "log"), "responses": y, "models": {"Yield": "main_effects"}})
    (fit,) = reply["responses"]
    assert fit["transform"] == "log"
    logged = api.ok("api_analyze_runs", {**transformed(held, "log"), "responses": y, "model": "main_effects"})
    assert fit["coefficients"][0] == pytest.approx(logged["coefficients"][0]["coefficient"]), "fitted to ln y"


#: A yield to maximise and a cost to minimise, over a face-centred design in T and P.
TWO = [
    {"name": "Yield", "label": "Yield", "units": "%", "goal": "maximize"},
    {"name": "Cost", "label": "Cost", "units": "$/kg", "goal": "minimize"},
]


def test_a_balance_works_a_logged_response_s_d_on_its_scale_and_says_it_in_its_units(api):
    held = held_of(
        api.ok(
            "api_make_design",
            {"design_type": "ccd", "factors": FACTORS[:2], "responses": TWO, "alpha": "face_centered"},
        )
    )
    x = coded(held)
    yield_ = (60 + 4 * x["T"] + 2 * x["P"] - 3 * x["T"] ** 2 - 2 * x["P"] ** 2).tolist()
    cost = np.exp(3 + 0.3 * x["T"] + 0.1 * x["P"]).tolist()
    request = {
        **held,
        "responses": yield_,
        "more_responses": {"Cost": cost},
        "models": {"Yield": "quadratic", "Cost": "quadratic"},
        "goals": {},
    }
    request["spec"] = {**held["spec"], "analysis_transforms": {"Cost": "log"}}
    reply = api.ok("api_balance", request)
    by = {r["name"]: r for r in reply["responses"]}
    assert by["Cost"]["transform"] == "log"
    assert "transform" not in by["Yield"]
    assert (by["Cost"]["default_low"], by["Cost"]["default_high"]) == pytest.approx((min(cost), max(cost)))
    assert min(cost) <= by["Cost"]["predicted"] <= max(cost), "in $/kg, not in ln($/kg)"
    assert 0 <= by["Cost"]["d"] <= 1
    narrower = {"Cost": {"low": 22.0, "high": 30.0}}  # the reader's range, in $/kg
    reply = api.ok("api_balance", {**request, "goals": narrower})
    cost_goal = next(g for g in reply["search"]["goals"] if g["response"] == "Cost")
    assert (cost_goal["low"], cost_goal["high"]) == pytest.approx((np.log(22.0), np.log(30.0))), "searched on ln"
    assert next(r for r in reply["responses"] if r["name"] == "Cost")["low"] == pytest.approx(22.0)


def test_the_next_step_for_a_transformed_response_says_its_numbers_are_on_the_model_s_scale(api, held, simulated):
    called = api.ok("api_next_runs", {**transformed(held, "sqrt"), "responses": simulated["responses"]})
    assert (
        called["recommendation"]["reasons"][0]
        == "The model is fitted to √Yield, so the numbers below are on that scale."
    )
