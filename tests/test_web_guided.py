"""The guided page's entry points: simulate results, analyse the runs the page holds, draw them.

``web/bootstrap.py`` is driven through its JSON entry points, as the page drives it.
"""

from __future__ import annotations

import base64
import importlib.util
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import TrueProcess

pytest.importorskip("openpyxl")

WEB = Path(__file__).resolve().parent.parent / "web"


@pytest.fixture(scope="module")
def app():
    spec = importlib.util.spec_from_file_location("web_bootstrap_guided", WEB / "bootstrap.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reply(app, fn: str, payload: dict) -> dict:
    return json.loads(getattr(app, fn)(json.dumps(payload)))


def call(app, fn: str, payload: dict) -> dict:
    answer = reply(app, fn, payload)
    assert answer["ok"], answer.get("error")
    return answer["result"]


@pytest.fixture(scope="module")
def demo(app) -> dict:
    return call(app, "api_catalogue", {})["demo"]


@pytest.fixture(scope="module")
def held(app, demo) -> dict:
    """Return what the page holds after generating the worked example's design."""
    design = call(app, "api_make_design", demo["design"])
    return {"spec": design["spec"], "columns": design["columns"], "rows": design["rows"]}


@pytest.fixture(scope="module")
def simulated(app, demo, held) -> dict:
    return call(
        app, "api_simulate", {**held, "scenario": demo["scenario"], "sigma": demo["sigma"], "seed": demo["seed"]}
    )


def coded_runs(held: dict) -> pd.DataFrame:
    runs = pd.DataFrame(held["rows"], columns=held["columns"])
    return pd.DataFrame(
        {
            f["name"]: (runs[f["name"]] - (f["high"] + f["low"]) / 2) / ((f["high"] - f["low"]) / 2)
            for f in held["spec"]["factors"]
        }
    )


# --------------------------------------------------------------------------- simulation
def test_the_worked_example_is_a_box_behnken_design_with_the_reactor_process(demo, held, simulated):
    assert demo["design"]["design_type"] == "box_behnken"
    assert len(held["rows"]) == 15
    assert simulated["equation"] == "y = 62 + 4·T - 3.1·P + 1.75·F + 1.5·T·P - 2.25·F²"
    assert len(simulated["responses"]) == 15


def test_simulated_responses_are_the_true_process_plus_seeded_noise(held, simulated):
    truth = TrueProcess({t["term"]: t["coefficient"] for t in simulated["truth"]}, sigma=simulated["sigma"])
    expected = truth.simulate(coded_runs(held), random_state=simulated["seed"])
    np.testing.assert_allclose(simulated["responses"], expected, atol=0.005, err_msg="rounded to two decimals")


def test_the_same_seed_gives_the_same_results_and_new_noise_does_not(app, held):
    first, again, other = (call(app, "api_simulate", {**held, "seed": seed}) for seed in (4, 4, 5))
    assert first["responses"] == again["responses"]
    assert first["responses"] != other["responses"]
    assert first["truth"] == other["truth"], "new noise, same process"


def test_a_random_process_changes_with_the_seed(app, held):
    first, other = (call(app, "api_simulate", {**held, "scenario": "random", "seed": s}) for s in (1, 2))
    assert first["truth"] != other["truth"]
    active = [t for t in first["truth"] if t["term"] != "Intercept"]
    assert 3 <= len(active) <= 4, "two or three main effects and one second-order term"


def test_a_scenario_drops_the_terms_of_factors_the_reader_does_not_have(app):
    spec = {
        "design_type": "full_factorial",
        "factors": [{"name": "A", "low": 0, "high": 1}, {"name": "B", "low": 0, "high": 1}],
        "n_center_points": 2,
    }
    design = call(app, "api_make_design", spec)
    held2 = {"spec": design["spec"], "columns": design["columns"], "rows": design["rows"]}
    result = call(app, "api_simulate", {**held2, "scenario": "reactor"})
    assert result["equation"] == "y = 62 + 4·A - 3.1·B + 1.5·A·B"
    assert result["inactive"] == []


def test_the_true_process_uses_the_factor_names_and_lists_the_factors_it_leaves_out(app):
    """Renamed and added factors reach the simulation: the reactor process acts on the first three only."""
    factors = [
        {"name": "Temp", "low": 150, "high": 200},
        {"name": "Pb", "low": 1, "high": 3},
        {"name": "Fill", "low": 10, "high": 20},
        {"name": "Salt", "low": -1, "high": 1},
    ]
    design = call(app, "api_make_design", {"design_type": "dsd", "factors": factors, "n_center_points": 3})
    held4 = {"spec": design["spec"], "columns": design["columns"], "rows": design["rows"]}
    result = call(app, "api_simulate", {**held4, "scenario": "reactor", "seed": 1})
    assert result["equation"] == "y = 62 + 4·Temp - 3.1·Pb + 1.75·Fill + 1.5·Temp·Pb - 2.25·Fill²"
    assert result["inactive"] == ["Salt"]


@pytest.mark.parametrize("scenario", ["reactor", "main_effects", "interaction", "optimum", "nothing", "random"])
def test_every_scenario_simulates(app, held, scenario):
    result = call(app, "api_simulate", {**held, "scenario": scenario, "sigma": 0.5, "seed": 2})
    assert result["sigma"] == 0.5
    assert all(isinstance(v, float) for v in result["responses"])


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"scenario": "unknown"}, "Unknown process"),
        ({"sigma": -1}, "noise standard deviation must be a number above 0"),
        ({"sigma": 0}, "noise standard deviation must be a number above 0"),  # not read as "not given"
        ({"sigma": "abc"}, "noise standard deviation must be a number above 0"),
        ({"spec": {"design_type": "rocket"}}, "did not make"),
        ({"columns": ["Run order"]}, "missing the factor column"),
    ],
)
def test_a_simulation_the_page_did_not_set_up_is_refused(app, held, change, message):
    answer = reply(app, "api_simulate", {**held, **change})
    assert not answer["ok"]
    assert message in answer["error"]


# --------------------------------------------------------------------------- analysis of the runs the page holds
def test_the_worked_example_recovers_the_true_process(app, held, simulated):
    analysis = call(app, "api_analyze_runs", {**held, "responses": simulated["responses"]})
    truth = {t["term"]: t["coefficient"] for t in simulated["truth"]}
    for c in analysis["coefficients"]:
        assert c["ci_low"] <= truth.get(c["term"], 0.0) <= c["ci_high"], c["term"]
    assert analysis["model"] == "quadratic"


def test_true_values_are_what_each_coefficient_estimates(app, held, simulated):
    """In a Box-Behnken design every term is estimated on its own, so the true values are the process's."""
    request = {**held, "responses": simulated["responses"], "truth": simulated["truth"]}
    analysis = call(app, "api_analyze_runs", request)
    truth = {t["term"]: t["coefficient"] for t in simulated["truth"]}
    assert {c["term"]: c["true_value"] for c in analysis["coefficients"]} == {
        c["term"]: truth.get(c["term"], 0.0) for c in analysis["coefficients"]
    }, "exact, with no rounding error left on the zeros"


def test_an_aliased_estimate_is_set_beside_the_sum_it_estimates(app):
    """In a 2^(3-1) design F is aliased with T:P, so its estimate is centred on F + T:P."""
    spec = {
        "design_type": "fractional_factorial",
        "resolution": 3,
        "n_center_points": 0,
        "factors": [{"name": n, "low": -1, "high": 1} for n in "TPF"],
    }
    design = call(app, "api_make_design", spec)
    held2 = {"spec": design["spec"], "columns": design["columns"], "rows": design["rows"]}
    sim = call(app, "api_simulate", {**held2, "scenario": "reactor", "sigma": 0.01})
    analysis = call(
        app,
        "api_analyze_runs",
        {**held2, "responses": sim["responses"], "model": "interactions", "truth": sim["truth"]},
    )
    true = {c["label"]: c["true_value"] for c in analysis["coefficients"]}
    assert true["F + T·P"] == pytest.approx(1.75 + 1.5)
    assert true["Intercept"] == pytest.approx(62 - 2.25), "F squared is 1 at every run of a two-level design"


def test_analysing_the_page_s_runs_matches_analysing_its_workbook(app, held, simulated):
    responses = simulated["responses"]
    in_page = call(app, "api_analyze_runs", {**held, "responses": responses})
    book = call(app, "api_workbook", {**held, "responses": responses})
    from_book = call(app, "api_analyze", {"xlsx_base64": book["xlsx_base64"]})
    assert in_page["equation"] == from_book["equation"]
    assert in_page["summary"] == from_book["summary"]


def test_runs_still_to_do_are_left_out_and_listed(app, held, simulated):
    responses = [*simulated["responses"][:-2], None, None]
    analysis = call(app, "api_analyze_runs", {**held, "responses": responses})
    assert analysis["n_completed"] == 13
    assert analysis["pending_runs"] == [14, 15]


def test_a_mismatched_response_list_is_refused(app, held):
    answer = reply(app, "api_analyze_runs", {**held, "responses": [1.0, 2.0]})
    assert not answer["ok"]
    assert "2 responses were sent for 15 runs" in answer["error"]


@pytest.mark.parametrize(
    "spec_change",
    [
        {"response": "__import__('os')"},
        {"model": "y ~ T"},
        {"factors": [{"name": "T; import os", "low": 0, "high": 1}, {"name": "P", "low": 0, "high": 1}]},
    ],
)
def test_a_specification_is_data_never_code(app, held, spec_change):
    answer = reply(app, "api_analyze_runs", {**held, "spec": {**held["spec"], **spec_change}, "responses": [1.0] * 15})
    assert not answer["ok"]


# --------------------------------------------------------------------------- the workbook
def test_the_workbook_marks_simulated_results(app, held, simulated):
    sources = ["simulated"] * 14 + ["entered"]
    book = call(
        app, "api_workbook", {**held, "responses": simulated["responses"], "sources": sources, "simulated": simulated}
    )
    sheets = pd.read_excel(io.BytesIO(base64.b64decode(book["xlsx_base64"])), sheet_name=None)
    assert sheets["Runs"]["Source"].fillna("").tolist() == ["simulated"] * 14 + [""]
    stored = json.loads(sheets["_pi_design"]["spec"].iloc[0])
    assert stored["simulated"] == {"scenario": "reactor", "sigma": 1.0, "seed": 1}
    np.testing.assert_allclose(sheets["Runs"]["y"], simulated["responses"])


# --------------------------------------------------------------------------- the charts
def test_the_charts_are_the_library_s_plots(app, held, simulated):
    pytest.importorskip("plotly")
    result = call(app, "api_plots", {**held, "responses": simulated["responses"], "truth": simulated["truth"]})
    assert set(result["figures"]) == {"contour", "true_contour", "residuals"}
    assert result["contour_factors"] == ["T", "P"], "the two largest main effects"
    contour = json.loads(result["figures"]["contour"])
    assert contour["data"][0]["type"] == "contour"
    assert result["plotly_js"]


def test_a_saturated_two_level_design_is_drawn_with_lenth_s_half_normal_plot(app):
    pytest.importorskip("plotly")
    spec = {
        "design_type": "plackett_burman",
        "factors": [{"name": n, "low": -1, "high": 1} for n in "ABCDEFG"],
        "n_center_points": 0,
    }
    design = call(app, "api_make_design", spec)
    held2 = {"spec": design["spec"], "columns": design["columns"], "rows": design["rows"]}
    responses = call(app, "api_simulate", {**held2, "scenario": "main_effects", "seed": 1})["responses"]
    result = call(app, "api_plots", {**held2, "responses": responses})
    assert "half_normal" in result["figures"]
    assert "residuals" not in result["figures"], "a saturated fit leaves no residuals to plot"


def test_a_true_process_naming_other_factors_is_refused(app, held, simulated):
    pytest.importorskip("plotly")
    truth = [{"term": "Z", "coefficient": 1.0}]
    answer = reply(app, "api_plots", {**held, "responses": simulated["responses"], "truth": truth})
    assert not answer["ok"]
    assert "['Z']" in answer["error"]


# --------------------------------------------------------------------------- the page and its build
def test_the_page_lists_the_scenarios_bootstrap_simulates(app):
    """The process list is in the HTML, so it reads right before Python loads; it says what SCENARIOS says."""
    from tests.test_web_app import _DesignOptions

    parser = _DesignOptions("scenario")
    parser.feed((WEB / "index.html").read_text(encoding="utf-8"))
    in_page = {o["value"]: o for o in parser.options}
    assert list(in_page) == list(app.SCENARIOS)
    for key, info in app.SCENARIOS.items():
        assert in_page[key]["label"] == info["label"], key
        assert in_page[key]["data-hint"] == info["hint"], key
    assert [o["value"] for o in parser.options if "selected" in o] == [app.DEMO["scenario"]]


def test_the_page_s_default_factors_are_the_worked_example_s(app):
    source = (WEB / "app.js").read_text(encoding="utf-8")
    block = re.search(r"const DEFAULT_FACTORS = \[(.*?)\];", source, re.DOTALL).group(1)
    in_page = [
        {"name": n, "low": float(lo), "high": float(hi), "units": u}
        for n, lo, hi, u in re.findall(r'name: "(\w+)", low: ([\d.]+), high: ([\d.]+), units: "([^"]*)"', block)
    ]
    demo = [{**f, "low": float(f["low"]), "high": float(f["high"])} for f in app.DEMO["design"]["factors"]]
    assert in_page == demo


def test_every_module_the_page_imports_is_published():
    build = _load("build_web_app", WEB.parent / "scripts" / "build_web_app.py")
    imported = set(re.findall(r'from "\./([\w.-]+)"', (WEB / "app.js").read_text(encoding="utf-8")))
    assert imported, "app.js imports charts.js"
    assert imported <= set(build.APP_FILES)


@pytest.mark.slow
def test_the_built_worked_example_carries_what_the_page_draws():
    pytest.importorskip("plotly")
    build = _load("build_web_app", WEB.parent / "scripts" / "build_web_app.py")
    example = json.loads(json.dumps(build.worked_example()))  # as the page reads it
    assert example["version"]
    assert set(example["inputs"]) >= {"design", "scenario", "sigma", "seed", "model"}
    assert set(example["design"]) >= {"spec", "columns", "rows", "facts", "model", "n_runs"}
    assert "xlsx_base64" not in example["design"], "the page writes the workbook on demand"
    assert set(example["simulation"]) >= {"responses", "truth", "equation", "inactive", "label", "sigma", "seed"}
    assert all("true_value" in c for c in example["analysis"]["coefficients"])
    assert set(example["analysis"]) >= {"equation", "summary", "notes", "anova", "pending_runs", "n_completed"}
    assert set(example["plots"]["figures"]) == {"contour", "true_contour", "residuals"}
    assert example["plots"]["plotly_js"]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
