"""Alternative near-best settings from the desirability search (#785, the library part).

Beside the best setting, ``optimize_responses(..., n_alternatives=k)`` reports up to k other local
optima of its multistart search: each with a composite desirability of at least 95% of the best's,
and at least 0.5 apart, in coded units, from the best and from each other.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import analyze_experiment, optimize_responses
from process_improve.experiments.reference_catalogue import reference_experiments


def model(**coefficients: float) -> dict:
    terms = {"Intercept": 0.0, **{k.replace("_sq", " ** 2)").replace("I_", "I("): v for k, v in coefficients.items()}}
    return {
        "response_name": "y",
        "factor_names": ["A", "B"],
        "coefficients": [{"term": t, "coefficient": c} for t, c in terms.items()],
    }


def coded(point: dict) -> np.ndarray:
    return np.array([point["optimal_coded"]["A"], point["optimal_coded"]["B"]])


def test_a_saddle_s_two_best_corners_are_the_best_and_its_alternative():
    """The response A^2 - B^2 is greatest, 1, at (1, 0) and at (-1, 0)."""
    saddle = model(I_A_sq=1.0, I_B_sq=-1.0)
    goal = [{"response": "y", "goal": "maximize", "low": 0, "high": 1}]
    found = optimize_responses([saddle], goals=goal, method="desirability", n_alternatives=2)["desirability"]
    [other] = found["alternatives"]  # only one other optimum exists
    assert other["composite_desirability"] == pytest.approx(1.0, abs=1e-9)
    corners = sorted([coded(found).round(6).tolist(), coded(other).round(6).tolist()])
    assert corners == [[-1.0, 0.0], [1.0, 0.0]]


def test_alternatives_on_a_ring_of_optima_are_apart_and_nearly_as_good():
    """The response A^2 + B^2 on target 0.5: every point of the ring A^2 + B^2 = 0.5 is best."""
    ring = model(I_A_sq=1.0, I_B_sq=1.0)
    goal = [{"response": "y", "goal": "target", "low": 0, "target": 0.5, "high": 1}]
    found = optimize_responses([ring], goals=goal, method="desirability", n_alternatives=3)["desirability"]
    points = [coded(found), *map(coded, found["alternatives"])]
    assert len(points) == 4
    for a, b in itertools.combinations(points, 2):
        assert np.linalg.norm(a - b) >= 0.5
    for alternative in found["alternatives"]:
        assert alternative["composite_desirability"] >= 0.95 * found["composite_desirability"]
        assert alternative["predicted_responses"]["y"] == pytest.approx(0.5, abs=1e-4)


@pytest.mark.dataset
def test_alternatives_keep_to_an_in_range_limit():
    """Montgomery's yield CCD: every alternative holds the molecular weight within 3200 to 3400."""
    [entry] = [e for e in reference_experiments("montgomery-2013") if e["name"] == "montgomery-yield-ccd"]
    runs = pd.DataFrame(entry["runs"]["rows"], columns=entry["runs"]["columns"])
    quadratic = "time * temp + I(time ** 2) + I(temp ** 2)"
    fits = [
        analyze_experiment(runs[["time", "temp"]], runs[r], model=m, analysis_type="coefficients", coding="coded")
        for r, m in (("yield", quadratic), ("viscosity", quadratic), ("mn", "time + temp"))
    ]
    goals = [
        {"response": "yield", "goal": "maximize", "low": 70, "high": 80},
        {"response": "viscosity", "goal": "target", "low": 62, "target": 65, "high": 68},
        {"response": "mn", "goal": "in_range", "low": 3200, "high": 3400},
    ]
    found = optimize_responses(fits, goals=goals, method="desirability", n_alternatives=3)["desirability"]
    for point in [found, *found["alternatives"]]:
        assert 3200 - 1e-6 <= point["predicted_responses"]["mn"] <= 3400 + 1e-6
        assert point["composite_desirability"] >= 0.95 * found["composite_desirability"]


def test_without_alternatives_the_search_is_as_it_was():
    saddle = model(I_A_sq=1.0, I_B_sq=-1.0)
    goal = [{"response": "y", "goal": "maximize", "low": 0, "high": 1}]
    found = optimize_responses([saddle], goals=goal, method="desirability")["desirability"]
    assert "alternatives" not in found
    assert coded(found).round(6).tolist() == [1.0, 0.0]


@pytest.mark.parametrize("n_alternatives", [-1, 1.5, True])
def test_a_count_of_alternatives_that_is_not_a_whole_number_is_refused(n_alternatives):
    goal = [{"response": "y", "goal": "maximize", "low": 0, "high": 1}]
    with pytest.raises(ValueError, match="n_alternatives must be a whole number"):
        optimize_responses([model(A=1.0)], goals=goal, method="desirability", n_alternatives=n_alternatives)
