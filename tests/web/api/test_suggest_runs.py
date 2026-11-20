"""Runs from a model of every run: ``api_suggest_runs``, and ``api_add_runs`` with the kind ``suggested`` (#1374).

A Gaussian process of every run with a result proposes where the most is expected to gain over the best so far. The
worked example's goal is to make the yield as large as it can be; a target is a goal too.
"""

from __future__ import annotations

import pytest

from tests.web._support import FACTORS, held_of


def suggested(api, held: dict, simulated: dict, **more: object) -> dict:
    """Return the runs suggested for the worked example's results."""
    return api.ok("api_suggest_runs", {**held, "responses": simulated["responses"], **more})


def inside(settings: dict) -> bool:
    """Say whether a run's settings are inside the worked example's ranges."""
    return all(f["low"] <= settings[f["name"]] <= f["high"] for f in FACTORS)


def test_the_runs_suggested_are_inside_the_ranges_each_with_why(api, held, simulated):
    found = suggested(api, held, simulated, n_runs=4)
    assert held["spec"]["response_info"]["goal"] == "maximize"
    assert len(found["runs"]) == 4
    for run in found["runs"]:
        assert inside(run["settings"])
        assert run["expected_improvement"] >= 0
        assert run["why"] in ("it looks better than the best so far", "little is known there")
    assert found["runs"][0]["settings"]["T"] == round(found["runs"][0]["settings"]["T"], 1), "as a plant sets it"
    assert found == suggested(api, held, simulated, n_runs=4), "the same results suggest the same runs"


def test_the_runs_suggested_are_added_as_they_were_shown(api, held, simulated):
    found = suggested(api, held, simulated, n_runs=2)
    grown = api.ok("api_add_runs", {**held, "responses": simulated["responses"], "kind": "suggested", "n_runs": 2})
    assert grown["added"] == 2
    names = [f["name"] for f in FACTORS]
    added = [dict(zip(grown["columns"], row, strict=True)) for row in grown["rows"][-2:]]
    assert sorted(tuple(run[n] for n in names) for run in added) == sorted(
        tuple(run["settings"][n] for n in names) for run in found["runs"]
    )
    assert grown["spec"]["design_type"] == "given"


def test_a_target_is_a_goal_and_its_best_is_nearest_it(api, held, simulated):
    """A goal of a target: the best so far is the prediction at a run nearest it, and a run looks better nearer it."""
    results = [r for r in simulated["responses"] if r is not None]
    target = sorted(results)[len(results) // 2]  # a result in the middle: neither the largest nor the smallest
    info = {**held["spec"]["response_info"], "goal": "target", "target": target}
    found = suggested(api, {**held, "spec": {**held["spec"], "response_info": info}}, simulated, n_runs=2)
    largest = suggested(api, held, simulated, n_runs=1)["best"]
    assert abs(found["best"] - target) < simulated["sigma"] < abs(largest - target)
    for run in found["runs"]:
        nearer = abs(run["predicted"] - target) < abs(found["best"] - target)
        assert run["why"] == ("it looks better than the best so far" if nearer else "little is known there")


@pytest.mark.parametrize(
    ("change", "refusal"),
    [
        ({"goal": "learn"}, "for a goal of making the response as large or as small"),
        ({"goal": "target"}, "or of bringing it to a target"),
        ({"n_runs": 9}, "Ask for 1 to 8 runs."),
        ({"n_runs": True}, "Ask for 1 to 8 runs."),
    ],
)
def test_what_a_model_of_every_run_cannot_suggest_for_is_refused(api, held, simulated, change, refusal):
    spec = held["spec"]
    if "goal" in change:
        spec = {**spec, "response_info": {**spec["response_info"], "goal": change.pop("goal")}}
    request = {**held, "spec": spec, "responses": simulated["responses"], **change}
    assert refusal in api.error("api_suggest_runs", request)


def test_too_few_results_are_refused(api, held, simulated):
    few = [*simulated["responses"][:4], *([None] * (len(simulated["responses"]) - 4))]
    assert "needs at least 6 runs with a result here; this study has 4" in api.error(
        "api_suggest_runs", {**held, "responses": few}
    )


def test_a_categorical_factor_is_refused(api):
    factors = [*FACTORS[:2], {"name": "Supplier", "low": "A", "high": "B", "units": ""}]
    made = api.ok("api_make_design", {"design_type": "full_factorial", "factors": factors})
    spec = {**made["spec"], "response_info": {**(made["spec"].get("response_info") or {}), "goal": "maximize"}}
    request = {**held_of(made), "spec": spec, "responses": [float(i) for i in range(len(made["rows"]))]}
    assert "numeric factors only" in api.error("api_suggest_runs", request)


def test_a_run_rounding_would_take_past_a_limit_keeps_its_setting_unrounded(
    api, bootstrap, held, simulated, monkeypatch
):
    """Rounded as a plant sets it, a run on a limit's edge can land outside it: such a run keeps its setting."""
    asked = {"factors": held["spec"]["factors"], "constraints": [{"expression": "T + 25*P <= 260"}]}
    spec = {**held["spec"], "constraints": api.ok("api_limits", asked)["constraints"]}
    monkeypatch.setattr(bootstrap, "_plant_setting", lambda factor, _value: 10 * factor["high"])  # far past the limit
    found = suggested(api, {**held, "spec": spec}, simulated, n_runs=1)
    settings = found["runs"][0]["settings"]
    assert settings["T"] + 25 * settings["P"] <= 260 + 1e-6
    assert settings["T"] != 10 * 200, "the setting the model chose, not the rounded one"
