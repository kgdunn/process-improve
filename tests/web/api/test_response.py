"""The response as the reader describes it: its name, units and goal, and the numbers they know of it.

The page asks first what the experiment measures. A design keeps that description in its
specification (``response_info``), beside the column its results go in (``response``), and so
does every reply that sends the specification back, a workbook, a saved study and a link. The
description puts practice results on the reader's scale, sets the change the designs are
compared at, and names the response on the equations and the charts.
"""

from __future__ import annotations

import json

import pytest

from tests.web._support import FACTORS, held_of

DESCRIBED = {
    "label": "Purity",
    "units": "%",
    "goal": "target",
    "target": 99.5,
    "typical": 97.0,
    "noise": 0.4,
    "smallest_change": 1.2,
}


def _design(api, **change: object) -> dict:
    return api.ok("api_make_design", {"design_type": "box_behnken", "factors": FACTORS, "response": "Purity", **change})


def test_a_design_keeps_the_reader_s_description_of_its_response(api):
    design = _design(api, response_info=DESCRIBED, aim="quadratic")
    assert design["spec"]["response_info"] == DESCRIBED
    assert design["spec"]["aim"] == "quadratic"
    assert design["columns"][-1] == "Purity", "the results go in the column the page named"


def test_an_empty_or_unknown_part_of_the_description_is_left_out(api):
    """Blank boxes are not kept, and a goal or aim this page does not know is dropped for a later page to add."""
    design = _design(api, response_info={"label": "  Purity ", "units": "", "goal": "soar"}, aim="cubic")
    assert design["spec"]["response_info"] == {"label": "Purity"}
    assert "aim" not in design["spec"]


def test_a_design_made_without_a_description_has_none(api):
    """Designs, workbooks and links made before the page asked for one keep working as they were."""
    design = _design(api)
    assert "response_info" not in design["spec"]
    assert "aim" not in design["spec"]
    blank = _design(api, response_info={"label": "", "units": "   "})
    assert "response_info" not in blank["spec"]


def test_the_description_travels_back_with_the_runs(api):
    design = _design(api, response_info=DESCRIBED, aim="quadratic")
    held = held_of(design)
    responses = [97.0 + 0.1 * i for i in range(design["n_runs"])]
    analysis = api.ok("api_analyze_runs", {**held, "responses": responses})
    assert analysis["spec"]["response_info"] == DESCRIBED
    assert analysis["spec"]["aim"] == "quadratic"
    book = api.ok("api_workbook", {**held, "responses": responses})
    back = api.ok("api_read_workbook", book)
    assert back["spec"]["response_info"] == DESCRIBED, "a workbook keeps the description too"


def test_a_description_sent_back_changed_is_checked_again(api):
    design = _design(api)
    held = held_of(design)
    spec = {**held["spec"], "response_info": {"label": "Purity", "noise": -1}}
    message = api.error("api_simulate", {**held, "spec": spec})
    assert message == "The run-to-run noise must be a number above 0."


def test_the_equation_names_the_response_as_the_reader_does(api):
    design = _design(api, response="Purity_pct", response_info={"label": "Purity of the product"})
    simulated = api.ok("api_simulate", {**held_of(design), "seed": 1})
    assert simulated["equation"].startswith("Purity of the product = ")
    analysis = api.ok("api_analyze_runs", {**held_of(design), "responses": simulated["responses"]})
    assert analysis["equation"].startswith("Purity of the product = ")
    unnamed = _design(api, response="Purity_pct")
    assert api.ok("api_simulate", held_of(unnamed))["equation"].startswith("Purity_pct = ")


def test_practice_results_are_on_the_reader_s_scale(api, bootstrap):
    """The process is centred on the typical value, and its effects are its sizes in noise units times the noise."""
    plain = api.ok("api_simulate", {**held_of(_design(api)), "scenario": "reactor", "seed": 3})
    scaled = api.ok(
        "api_simulate",
        {**held_of(_design(api, response_info={"typical": 97.0, "noise": 0.4})), "scenario": "reactor", "seed": 3},
    )
    reactor = bootstrap.SCENARIOS["reactor"]["terms"]
    plain_truth = {t["term"]: t["coefficient"] for t in plain["truth"]}
    scaled_truth = {t["term"]: t["coefficient"] for t in scaled["truth"]}
    assert plain_truth["Intercept"] == reactor[()]
    assert scaled_truth["Intercept"] == 97.0
    for term, value in plain_truth.items():
        if term != "Intercept":
            assert scaled_truth[term] == pytest.approx(0.4 * value)


def test_the_worked_example_s_description_leaves_its_results_as_the_reactor_gives_them(api, demo, held, simulated):
    """Typical 62 and noise 1 are the reactor's own centre and noise: the example's numbers do not move."""
    info = demo["design"]["response_info"]
    bare = {**held, "spec": {k: v for k, v in held["spec"].items() if k != "response_info"}}
    plain = api.ok("api_simulate", {**bare, "scenario": demo["scenario"], "sigma": demo["sigma"], "seed": demo["seed"]})
    assert simulated["responses"] == plain["responses"]
    assert simulated["truth"] == plain["truth"]
    assert info["smallest_change"] / (2 * info["noise"]) == 1.0, "power at the comparison's default effect size"


@pytest.mark.parametrize(
    ("described", "effect_size"),
    [({}, 1.0), ({"noise": 1.5, "smallest_change": 6.0}, 2.0), ({"noise": 1.5}, 1.0)],
    ids=["no numbers", "a change of 6 over a noise of 1.5", "noise without a change"],
)
def test_power_is_for_the_smallest_change_worth_finding(api, described, effect_size):
    """A change from low to high of 6, with noise 1.5, is a coefficient of 3, or 2 noise standard deviations."""
    design = _design(api)
    item = {"label": "Box-Behnken", **held_of(design)}
    compared = api.ok("api_compare", {"model": "main_effects", "designs": [item], **described})
    assert compared["effect_size"] == effect_size


def test_the_charts_are_in_the_reader_s_words_and_units(api, held, simulated):
    pytest.importorskip("plotly", reason="the page installs plotly for its first chart")
    result = api.ok("api_plots", {**held, "responses": simulated["responses"], "truth": simulated["truth"]})
    contour = json.loads(result["figures"]["contour"])
    assert contour["layout"]["xaxis"]["title"]["text"] == "T (degC)"
    assert contour["layout"]["xaxis"]["ticktext"] == ["150", "175", "200"]
    assert contour["layout"]["xaxis"]["tickvals"] == [-1, 0, 1]
    assert contour["layout"]["yaxis"]["title"]["text"] == "P (bar)"
    assert contour["data"][0]["colorbar"]["title"]["text"] == "Yield (%)"
    residuals = json.loads(result["figures"]["residuals"])
    assert residuals["layout"]["xaxis"]["title"]["text"] == "Fitted Yield (%)"
    assert residuals["layout"]["yaxis"]["title"]["text"] == "Residual (%)"


def test_a_categorical_axis_is_marked_with_its_labels(api):
    pytest.importorskip("plotly", reason="the page installs plotly for its first chart")
    factors = [FACTORS[0], {"name": "S", "low": "old", "high": "new", "units": ""}]
    design = api.ok("api_make_design", {"design_type": "full_factorial", "factors": factors, "n_center_points": 0})
    responses = [10.0, 14.0, 11.0, 17.0][: design["n_runs"]] * (design["n_runs"] // 4)
    result = api.ok("api_plots", {**held_of(design), "responses": responses, "model": "main_effects"})
    contour = json.loads(result["figures"]["contour"])
    axes = {contour["layout"][a]["title"]["text"]: contour["layout"][a] for a in ("xaxis", "yaxis")}
    assert axes["S"]["ticktext"] == ["old", "new"]
    assert axes["T (degC)"]["ticktext"] == ["150", "175", "200"]
