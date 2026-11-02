"""Several responses: a study that measures a yield, a cost and a purity in the same runs.

The brief lists up to five responses. The specification keeps the first one as it always has
(``response``, described by ``response_info``), so a study of one response is stored exactly as
before, and lists them all in ``responses`` when there are more. The run sheet has a column for each;
the workbook, the simulation and every analysis follow. Each response is analysed on its own, fitted
to the runs that have its result, and the page names the response it asks about (``response``).
"""

from __future__ import annotations

import base64
import io
import json

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import TrueProcess
from tests.web._support import FACTORS, coded, held_of, run_sheet, sheets_of

#: Three responses, as the brief describes them: the first with what the reader knows of it.
RESPONSES = [
    {"name": "Yield", "label": "Yield", "units": "%", "goal": "maximize", "typical": 62.0, "noise": 1.0},
    {"name": "Cost", "label": "Cost", "units": "$/kg", "goal": "minimize"},
    {"name": "Purity", "label": "Purity", "units": "%", "goal": "target", "target": 99.0},
]

#: A specification as the page stored one before it measured several responses, word for word.
OLD_SPEC = {
    "version": 1,
    "design_type": "full_factorial",
    "factors": [
        {"name": "T", "low": 150, "high": 200, "units": "degC"},
        {"name": "P", "low": 1, "high": 3, "units": "bar"},
    ],
    "response": "Yield",
    "model": "interactions",
    "options": {"random_state": 42, "n_center_points": 3},
    "response_info": {"label": "Yield", "units": "%", "goal": "maximize"},
    "aim": "interactions",
}


def three(api, **change: object) -> dict:
    """Make the worked example's Box-Behnken design, measuring the three responses."""
    request = {"design_type": "box_behnken", "factors": FACTORS, "response": "Yield", "responses": RESPONSES}
    return api.ok("api_make_design", {**request, **change})


def surfaces(held: dict) -> dict[str, list[float]]:
    """Return each response's results from a known surface over the coded settings, with a little noise."""
    x = coded(run_sheet(held), held["spec"]["factors"])
    t, p, f = x["T"], x["P"], x["F"]
    rng = np.random.default_rng(3)
    known = {
        "Yield": 60 + 4 * t - 3 * p + 2 * t * p - 5 * t**2,
        "Cost": 20 + 3 * t + 1.5 * f + 0.5 * t * f,
        "Purity": 98 + 0.8 * p - 0.6 * f - 0.7 * p**2,
    }
    return {name: (v + rng.normal(0, 0.1, len(x))).round(3).tolist() for name, v in known.items()}


def results_of(held: dict) -> dict:
    """Return the results the page holds for the three responses: the first in ``responses``, the rest beside it."""
    values = surfaces(held)
    return {"responses": values["Yield"], "more_responses": {"Cost": values["Cost"], "Purity": values["Purity"]}}


# --------------------------------------------------------------------------- the specification
def test_a_design_measures_each_response_in_a_column_of_its_own(api):
    design = three(api)
    assert design["columns"][-3:] == ["Yield", "Cost", "Purity"], "a column per response, in the brief's order"
    assert all(row[-3:] == [None, None, None] for row in design["rows"])
    spec = design["spec"]
    assert spec["response"] == "Yield", "the first response is the study's response, as it always was"
    assert spec["response_info"] == {k: v for k, v in RESPONSES[0].items() if k != "name"}
    assert spec["responses"] == RESPONSES


def test_a_study_of_one_response_is_stored_as_before(api):
    """A brief of one response, sent as a list or as before, makes the same specification as before: no list."""
    one = {k: v for k, v in RESPONSES[0].items() if k != "name"}
    before = api.ok(
        "api_make_design", {"design_type": "box_behnken", "factors": FACTORS, "response": "Yield", "response_info": one}
    )
    listed = api.ok("api_make_design", {"design_type": "box_behnken", "factors": FACTORS, "responses": [RESPONSES[0]]})
    assert "responses" not in before["spec"]
    assert listed["spec"] == before["spec"]
    assert listed["columns"] == before["columns"]


def test_a_specification_made_before_opens_unchanged(api):
    """A study saved before the page measured several responses: its specification is read as it was written."""
    rows = [[i, i, t, p, None] for i, (t, p) in enumerate([(150, 1), (200, 1), (150, 3), (200, 3)], 1)]
    held = {"spec": OLD_SPEC, "columns": ["Std order", "Run order", "T", "P", "Yield"], "rows": rows}
    workbook = api.ok("api_workbook", {**held, "responses": [10.0, 12.0, 11.0, 15.0]})
    read = api.ok("api_read_workbook", {"xlsx_base64": workbook["xlsx_base64"]})
    assert read["spec"] == OLD_SPEC
    assert "more_responses" not in read
    assert "more_sources" not in read
    assert read["responses"] == [10.0, 12.0, 11.0, 15.0]
    analysis = api.ok("api_analyze_runs", {**held, "responses": read["responses"]})
    assert analysis["spec"] == OLD_SPEC
    assert analysis["n_completed"] == 4


def test_a_blank_response_takes_the_name_y_as_before(api):
    design = api.ok("api_make_design", {"design_type": "box_behnken", "factors": FACTORS})
    assert design["spec"]["response"] == "y"
    assert design["columns"][-1] == "y"


# --------------------------------------------------------------------------- the workbook
def test_a_workbook_of_three_responses_comes_back_as_it_went(api):
    design = three(api)
    held = held_of(design)
    results = results_of(held)
    n = design["n_runs"]
    sources = ["simulated"] * n
    more_sources = {"Cost": ["simulated"] * (n - 1) + ["entered"], "Purity": ["simulated"] * n}
    models = {"Cost": "interactions", "Purity": {"terms": [["P"], ["F"], ["P", "P"]]}}
    written = api.ok(
        "api_workbook",
        {**held, **results, "sources": sources, "more_sources": more_sources, "model": "quadratic", "models": models},
    )
    runs = sheets_of(written["xlsx_base64"])["Runs"]
    for name in ("Yield", "Cost", "Purity"):
        expected = results["responses"] if name == "Yield" else results["more_responses"][name]
        np.testing.assert_allclose(runs[name], expected)
    marks = runs["Result source"].fillna("").tolist()
    assert marks[:-1] == ["simulated"] * (n - 1)
    assert marks[-1] == "", "a run with a result typed in is not marked as simulated"

    read = api.ok("api_read_workbook", {"xlsx_base64": written["xlsx_base64"]})
    assert read["columns"] == design["columns"]
    assert read["responses"] == results["responses"]
    assert read["more_responses"] == results["more_responses"]
    assert read["sources"][:-1] == ["simulated"] * (n - 1)
    assert read["more_sources"]["Cost"][-1] == "entered"
    assert read["spec"]["responses"] == RESPONSES
    assert read["spec"]["analysis_model"] == "quadratic"
    assert read["spec"]["analysis_models"] == models


def test_a_workbook_is_analysed_response_by_response(api):
    design = three(api)
    held = held_of(design)
    results = results_of(held)
    written = api.ok("api_workbook", {**held, **results, "models": {"Cost": "interactions"}})
    for name, model in (("Yield", "quadratic"), ("Cost", "interactions")):
        from_workbook = api.ok("api_analyze", {"xlsx_base64": written["xlsx_base64"], "response": name})
        on_the_page = api.ok("api_analyze_runs", {**held, **results, "response": name, "model": model})
        assert from_workbook["formula"] == on_the_page["formula"], "each response's own model, as the workbook keeps it"
        assert from_workbook["coefficients"] == on_the_page["coefficients"]


def test_a_workbook_without_a_response_s_column_is_refused(api):
    design = three(api)
    written = api.ok("api_workbook", {**held_of(design), **results_of(held_of(design))})
    sheets = sheets_of(written["xlsx_base64"])
    runs = sheets["Runs"].drop(columns=["Cost"])
    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as xl:
        runs.to_excel(xl, sheet_name="Runs", index=False)
        sheets["_pi_design"].to_excel(xl, sheet_name="_pi_design", index=False)
    error = api.error("api_read_workbook", {"xlsx_base64": base64.b64encode(buf.getvalue()).decode()})
    assert error == "The Runs sheet has no 'Cost' column for the results."


# --------------------------------------------------------------------------- the simulation
def test_the_first_response_is_simulated_as_a_study_of_one_response_is(api):
    """Measuring more responses does not change the first one's practice results."""
    request = {"scenario": "reactor", "sigma": 1.0, "seed": 4}
    alone = api.ok("api_make_design", {"design_type": "box_behnken", "factors": FACTORS, "responses": RESPONSES[:1]})
    together = three(api)
    one = api.ok("api_simulate", {**held_of(alone), **request})
    several = api.ok("api_simulate", {**held_of(together), **request})
    assert {k: v for k, v in several.items() if k != "more"} == one
    assert set(several["more"]) == {"Cost", "Purity"}


def test_each_later_response_follows_the_scenario_moved_to_other_factors(api, bootstrap):
    """Response j acts through factor (i + j) mod k where the scenario acts through factor i, and odd j reflects."""
    held = held_of(three(api))
    simulated = api.ok("api_simulate", {**held, "scenario": "reactor", "sigma": 1.0, "seed": 4})
    reactor = bootstrap.SCENARIOS["reactor"]["terms"]
    # Cost, in place 1: T's effect moves to P, P's to F, F's to T, and every factor is reflected.
    cost = {t["term"]: t["coefficient"] for t in simulated["more"]["Cost"]["truth"]}
    assert cost == {"Intercept": 62.0, "P": -4.0, "F": 3.1, "T": -1.75, "P:F": 1.5, "I(T ** 2)": -2.25}
    # Purity, in place 2: T's effect moves to F, P's to T, F's to P, and nothing is reflected.
    purity = {t["term"]: t["coefficient"] for t in simulated["more"]["Purity"]["truth"]}
    assert purity == {"Intercept": 62.0, "F": 4.0, "T": -3.1, "P": 1.75, "T:F": 1.5, "I(P ** 2)": -2.25}
    assert reactor[(0,)] == 4.0, "the scenario itself is left as it was"


def test_each_response_s_noise_is_drawn_with_its_own_seed(api, bootstrap):
    held = held_of(three(api))
    seed = 4
    simulated = api.ok("api_simulate", {**held, "scenario": "optimum", "sigma": 1.0, "seed": seed})
    x = coded(run_sheet(held), held["spec"]["factors"])
    for place, name in ((1, "Cost"), (2, "Purity")):
        one = simulated["more"][name]
        assert one["seed"] == seed + bootstrap.SEED_STEP * place
        process = TrueProcess({t["term"]: t["coefficient"] for t in one["truth"]}, sigma=one["sigma"])
        expected = process.simulate(x, random_state=one["seed"]).round(2)
        np.testing.assert_allclose(one["responses"], expected)
        assert one["equation"].startswith(f"{name} = ")
    assert simulated["more"]["Cost"]["responses"] != simulated["responses"], "noise of its own, and a process too"


def test_a_later_response_s_numbers_put_its_practice_results_on_its_scale(api):
    """Without its typical value and noise a later response keeps the scenario's centre and a noise of 1."""
    described = [RESPONSES[0], {**RESPONSES[1], "typical": 12.0, "noise": 0.5}, RESPONSES[2]]
    held = held_of(three(api, responses=described))
    simulated = api.ok("api_simulate", {**held, "scenario": "reactor", "sigma": 1.0, "seed": 1})
    cost = {t["term"]: t["coefficient"] for t in simulated["more"]["Cost"]["truth"]}
    assert cost["Intercept"] == 12.0
    assert cost["P"] == -2.0, "its effects in its own noise standard deviations"
    assert simulated["more"]["Cost"]["sigma"] == 0.5
    assert simulated["more"]["Purity"]["sigma"] == 1.0


def test_two_responses_of_two_factors_trade_off(api):
    """With two factors the second response's peak is the first's moved and reflected, so their best differ."""
    factors = FACTORS[:2]
    plain = [{"name": "Yield"}, {"name": "Cost"}]  # no typical value or noise: the scenario's own centre and scale
    spec = {"design_type": "ccd", "factors": factors, "responses": plain, "alpha": "face_centered"}
    held = held_of(api.ok("api_make_design", spec))
    simulated = api.ok("api_simulate", {**held, "scenario": "optimum", "sigma": 1.0, "seed": 1})
    first = {t["term"]: t["coefficient"] for t in simulated["truth"]}
    second = {t["term"]: t["coefficient"] for t in simulated["more"]["Cost"]["truth"]}
    assert first == {"Intercept": 80.0, "T": 1.2, "P": -0.8, "T:P": 1.0, "I(T ** 2)": -3.0, "I(P ** 2)": -2.5}
    assert second == {"Intercept": 80.0, "P": -1.2, "T": 0.8, "T:P": 1.0, "I(P ** 2)": -3.0, "I(T ** 2)": -2.5}


# --------------------------------------------------------------------------- the analysis
def test_each_response_is_analysed_as_a_study_of_that_response_alone(api):
    design = three(api)
    held = held_of(design)
    results = results_of(held)
    for name in ("Cost", "Purity"):
        together = api.ok("api_analyze_runs", {**held, **results, "response": name, "model": "quadratic"})
        info = next(r for r in RESPONSES if r["name"] == name)
        alone_spec = {
            **design["spec"],
            "response": name,
            "response_info": {k: v for k, v in info.items() if k != "name"},
        }
        alone_spec.pop("responses")
        alone = api.ok(
            "api_analyze_runs",
            {
                "spec": alone_spec,
                "columns": design["columns"],
                "rows": design["rows"],
                "responses": results["more_responses"][name],
                "model": "quadratic",
            },
        )
        assert together["coefficients"] == alone["coefficients"]
        assert together["equation"] == alone["equation"]
        assert together["equation"].startswith(f"{name} = ")


def test_a_run_with_some_of_its_results_counts_for_the_responses_it_has(api):
    design = three(api)
    held = held_of(design)
    results = results_of(held)
    results["more_responses"]["Cost"][0] = None  # run 1's cost is still to come
    yield_ = api.ok("api_analyze_runs", {**held, **results, "response": "Yield"})
    cost = api.ok("api_analyze_runs", {**held, **results, "response": "Cost"})
    assert yield_["n_completed"] == design["n_runs"]
    assert yield_["pending_runs"] == []
    assert cost["n_completed"] == design["n_runs"] - 1
    assert cost["pending_runs"] == [1]


def test_each_response_has_the_best_settings_for_its_own_goal(api):
    held = held_of(three(api))
    results = results_of(held)
    yield_ = api.ok("api_best_settings", {**held, **results, "model": "quadratic"})
    cost = api.ok("api_best_settings", {**held, **results, "response": "Cost", "model": "interactions"})
    purity = api.ok("api_best_settings", {**held, **results, "response": "Purity", "model": "quadratic"})
    assert (yield_["goal"], cost["goal"], purity["goal"]) == ("maximize", "minimize", "target")
    assert cost["predicted"] < min(results["more_responses"]["Cost"]), "the lowest cost, below every run's"
    assert purity["target"] == 99.0
    assert yield_["coded"] != cost["coded"]


def test_each_response_has_its_charts(api):
    held = held_of(three(api))
    results = results_of(held)
    plots = api.ok("api_plots", {**held, **results, "response": "Cost", "model": "interactions"})
    contour = json.loads(plots["figures"]["contour"])
    title = next(t["colorbar"]["title"]["text"] for t in contour["data"] if t.get("type") == "contour")
    assert title == "Cost ($/kg)"
    assert plots["contour_factors"] == ["T", "F"], "the factors with the largest effects on the cost"


def test_a_response_the_study_does_not_measure_is_refused(api):
    held = held_of(three(api))
    error = api.error("api_analyze_runs", {**held, **results_of(held), "response": "Colour"})
    assert error == "The study measures no response named 'Colour'."


# --------------------------------------------------------------------------- renaming the factors
def test_renaming_a_factor_renames_every_response_s_process(api):
    design = three(api)
    held = held_of(design)
    simulated = api.ok("api_simulate", {**held, "scenario": "reactor", "sigma": 1.0, "seed": 1})
    renamed = [{**FACTORS[0], "name": "Temp"}, *FACTORS[1:]]
    reply = api.ok("api_relabel", {"design": held, "factors": renamed, "simulation": simulated})
    cost = reply["simulation"]["more"]["Cost"]
    assert {t["term"] for t in cost["truth"]} == {"Intercept", "P", "F", "Temp", "P:F", "I(Temp ** 2)"}
    assert "Temp" in cost["equation"]
    assert cost["equation"].startswith("Cost = ")


@pytest.mark.parametrize("name", ["Cost", "Purity"])
def test_a_factor_cannot_take_a_response_s_name(api, name):
    held = held_of(three(api))
    renamed = [{**FACTORS[0], "name": name}, *FACTORS[1:]]
    error = api.error("api_relabel", {"design": held, "factors": renamed})
    assert error == f"A factor cannot be named {name}: the run sheet has a {name} column."
