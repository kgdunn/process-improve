"""Simulating the results: ``api_simulate``, which step 2 of the page calls.

A known process, written over the reader's own factors in coded units, plus noise drawn from
the seed: so the analysis can be set beside the truth, and "New noise" draws the same process
again with new noise.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from process_improve.experiments import TrueProcess
from tests.web._support import ALL_SCENARIOS, FACTORS, coded, held_of, run_sheet


def test_every_process_the_page_offers_is_tested_here(bootstrap):
    """The tests run over ``ALL_SCENARIOS``: a process added to the page must be added there too."""
    assert list(bootstrap.SCENARIOS) == list(ALL_SCENARIOS)


def test_the_worked_example_is_a_box_behnken_design_with_the_reactor_process(demo, held, simulated):
    assert demo["design"]["design_type"] == "box_behnken"
    assert len(held["rows"]) == 15
    assert simulated["equation"] == "Yield = 62 + 4·T - 3.1·P + 1.75·F + 1.5·T·P - 2.25·F²"
    assert len(simulated["responses"]) == 15


def test_simulated_responses_are_the_true_process_plus_seeded_noise(held, simulated):
    truth = TrueProcess({t["term"]: t["coefficient"] for t in simulated["truth"]}, sigma=simulated["sigma"])
    expected = truth.simulate(coded(run_sheet(held), held["spec"]["factors"]), random_state=simulated["seed"])
    np.testing.assert_allclose(simulated["responses"], expected, atol=0.005, err_msg="rounded to two decimals")


def test_the_same_seed_gives_the_same_results_and_new_noise_does_not(api, held):
    first, again, other = (api.ok("api_simulate", {**held, "seed": seed}) for seed in (4, 4, 5))
    assert first["responses"] == again["responses"]
    assert first["responses"] != other["responses"]
    assert first["truth"] == other["truth"], "new noise, same process"


def test_without_a_choice_the_reactor_process_is_simulated_with_its_own_noise(api, held, bootstrap):
    result = api.ok("api_simulate", held)
    assert result["scenario"] == "reactor"
    assert result["label"] == bootstrap.SCENARIOS["reactor"]["label"]
    assert result["sigma"] == bootstrap.SCENARIOS["reactor"]["sigma"]
    assert result["seed"] == 1


@pytest.mark.parametrize(("sigma", "decimals"), [(0.01, 4), (0.5, 3), (1.0, 2), (30.0, 1), (500.0, 0)])
def test_results_are_rounded_to_the_precision_of_the_noise(api, held, sigma, decimals):
    """Two significant decimals below the noise's standard deviation, as a measurement would be."""
    responses = api.ok("api_simulate", {**held, "sigma": sigma, "seed": 3})["responses"]
    assert all(round(v, decimals) == v for v in responses)
    if decimals:
        assert any(round(v, decimals - 1) != v for v in responses), "and no coarser"


def test_a_random_process_changes_with_the_seed(api, held):
    first, other = (api.ok("api_simulate", {**held, "scenario": "random", "seed": s}) for s in (1, 2))
    assert first["truth"] != other["truth"]
    active = [t for t in first["truth"] if t["term"] != "Intercept"]
    assert 3 <= len(active) <= 4, "two or three main effects and one second-order term"


def test_a_scenario_drops_the_terms_of_factors_the_reader_does_not_have(api):
    spec = {
        "design_type": "full_factorial",
        "factors": [{"name": "A", "low": 0, "high": 1}, {"name": "B", "low": 0, "high": 1}],
        "n_center_points": 2,
    }
    held = held_of(api.ok("api_make_design", spec))
    result = api.ok("api_simulate", {**held, "scenario": "reactor"})
    assert result["equation"] == "y = 62 + 4·A - 3.1·B + 1.5·A·B"
    assert result["inactive"] == []


def test_the_true_process_uses_the_factor_names_and_lists_the_factors_it_leaves_out(api):
    """Renamed and added factors reach the simulation: the reactor process acts on the first three only."""
    factors = [
        {"name": "Temp", "low": 150, "high": 200},
        {"name": "Pb", "low": 1, "high": 3},
        {"name": "Fill", "low": 10, "high": 20},
        {"name": "Salt", "low": -1, "high": 1},
    ]
    design = api.ok("api_make_design", {"design_type": "dsd", "factors": factors, "n_center_points": 3})
    result = api.ok("api_simulate", {**held_of(design), "scenario": "reactor", "seed": 1})
    assert result["equation"] == "y = 62 + 4·Temp - 3.1·Pb + 1.75·Fill + 1.5·Temp·Pb - 2.25·Fill²"
    assert result["inactive"] == ["Salt"]


@pytest.mark.parametrize("scenario", ALL_SCENARIOS)
def test_every_scenario_simulates(api, held, scenario):
    result = api.ok("api_simulate", {**held, "scenario": scenario, "sigma": 0.5, "seed": 2})
    assert result["sigma"] == 0.5
    assert all(isinstance(v, float) for v in result["responses"])
    assert all(math.isfinite(t["coefficient"]) for t in result["truth"])


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
def test_a_simulation_the_page_did_not_set_up_is_refused(api, held, change, message):
    assert message in api.error("api_simulate", {**held, **change})


def test_a_categorical_factor_has_no_square_in_the_process_simulated(api):
    """The reactor's curvature is in its third factor; a category there is +1 at both levels, so it has none."""
    factors = [*FACTORS[:2], {"name": "Supplier", "low": "A", "high": "B", "units": ""}]
    design = api.ok("api_make_design", {"design_type": "full_factorial", "factors": factors})
    simulated = api.ok("api_simulate", {**held_of(design), "scenario": "reactor", "seed": 1})
    terms = [t["term"] for t in simulated["truth"]]
    assert "Supplier" in terms
    assert "I(Supplier ** 2)" not in terms
