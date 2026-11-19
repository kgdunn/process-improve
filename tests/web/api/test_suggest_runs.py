"""Runs from a model of every run: ``api_suggest_runs``, and ``api_add_runs`` with the kind ``suggested`` (#1374).

A Gaussian process of every run with a result proposes where the most is expected to gain over the best so far. The
worked example's goal is to make the yield as large as it can be.
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


@pytest.mark.parametrize(
    ("change", "refusal"),
    [
        ({"goal": "learn"}, "for a goal of making the response as large or as small"),
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
