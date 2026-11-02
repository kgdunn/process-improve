"""The "in_range" goal: hold a response within limits while the others are optimised (#789 B4).

Montgomery (Section 11.3.4) maximises the yield of a CCD and puts its viscosity on target 65,
with the molecular weight held between 3200 and 3400 by Design-Expert's "is in range" goal. His
printed composites (0.822 and 0.792) are the geometric means of the yield's and the viscosity's
desirabilities alone, so the in-range goal is left out of the composite. The problem as he states
it has a better optimum than both printed solutions: D = 0.929 near 86.2 min and 170.2 F.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import analyze_experiment, optimize_responses
from process_improve.experiments._desirability import check_goal, individual_desirability, overall_desirability
from process_improve.experiments.reference_catalogue import reference_experiments
from process_improve.experiments.visualization.plots.registry import create_plot

QUADRATIC = "time * temp + I(time ** 2) + I(temp ** 2)"
GOALS = [
    {"response": "yield", "goal": "maximize", "low": 70, "high": 80},
    {"response": "viscosity", "goal": "target", "low": 62, "target": 65, "high": 68},
    {"response": "mn", "goal": "in_range", "low": 3200, "high": 3400},
]


@pytest.fixture(scope="module")
def yield_ccd() -> list[dict]:
    """Montgomery's three models, fitted in coded units: two quadratics and the molecular weight's plane."""
    [entry] = [e for e in reference_experiments("montgomery-2013") if e["name"] == "montgomery-yield-ccd"]
    runs = pd.DataFrame(entry["runs"]["rows"], columns=entry["runs"]["columns"])
    models = []
    for response, model in (("yield", QUADRATIC), ("viscosity", QUADRATIC), ("mn", "time + temp")):
        fit = analyze_experiment(
            runs[["time", "temp"]], runs[response], model=model, analysis_type="coefficients", coding="coded"
        )
        models.append(
            {"response_name": response, "coefficients": fit["coefficients"], "factor_names": ["time", "temp"]}
        )
    return models


def actual(coded: dict[str, float]) -> tuple[float, float]:
    """Return a coded setting in minutes and degrees: 85 +/- 5 min and 175 +/- 5 F."""
    return 85 + 5 * coded["time"], 175 + 5 * coded["temp"]


# --------------------------------------------------------------------------- the source
@pytest.mark.dataset
def test_montgomery_s_yield_ccd_with_the_molecular_weight_held_in_range(yield_ccd):
    found = optimize_responses(yield_ccd, goals=GOALS, method="desirability")["desirability"]
    time, temp = actual(found["optimal_coded"])
    assert round(found["composite_desirability"], 3) == 0.929
    assert (round(time, 1), round(temp, 1)) == (86.1, 170.2)
    assert found["predicted_responses"]["viscosity"] == pytest.approx(65.0, abs=1e-4)
    assert 3200 <= found["predicted_responses"]["mn"] <= 3400
    d = found["individual_desirability"]
    assert d["mn"] == 1.0
    # The composite is the geometric mean of the two scored responses; the in-range one is left out.
    assert found["composite_desirability"] == pytest.approx(np.sqrt(d["yield"] * d["viscosity"]))


@pytest.mark.dataset
def test_the_limit_on_the_molecular_weight_is_what_holds_the_optimum_back(yield_ccd):
    """Without it, D = 1 at a setting whose molecular weight is near 3590, outside the range."""
    free = optimize_responses(yield_ccd[:2], goals=GOALS[:2], method="desirability")["desirability"]
    assert free["composite_desirability"] == pytest.approx(1.0, abs=1e-6)
    mn = {c["term"]: c["coefficient"] for c in yield_ccd[2]["coefficients"]}
    at_free = mn["Intercept"] + mn["time"] * free["optimal_coded"]["time"] + mn["temp"] * free["optimal_coded"]["temp"]
    assert at_free > 3400


@pytest.mark.dataset
def test_the_printed_solutions_score_below_the_optimum_found(yield_ccd):
    """Design-Expert's two solutions, 86.5 min and 170.5 F, and 82 min and 178.8 F: D = 0.826 and 0.786."""
    found = optimize_responses(yield_ccd, goals=GOALS, method="desirability")["desirability"]
    terms = [{c["term"]: c["coefficient"] for c in m["coefficients"]} for m in yield_ccd]
    for (time, temp), printed in (((86.5, 170.5), 0.826), ((82.0, 178.8), 0.786)):
        x1, x2 = (time - 85) / 5, (temp - 175) / 5
        values = {"time": x1, "temp": x2, "time:temp": x1 * x2, "I(time ** 2)": x1**2, "I(temp ** 2)": x2**2}
        predicted = [t["Intercept"] + sum(c * values[k] for k, c in t.items() if k != "Intercept") for t in terms]
        d = [individual_desirability(y, g) for y, g in zip(predicted, GOALS, strict=True)]
        assert round(overall_desirability(d, GOALS), 3) == printed
        assert overall_desirability(d, GOALS) < found["composite_desirability"]


# --------------------------------------------------------------------------- the goal itself
def test_an_in_range_goal_scores_1_inside_its_limits_and_0_outside():
    goal = {"goal": "in_range", "low": 3200, "high": 3400}
    assert [individual_desirability(y, goal) for y in (3199.9, 3200, 3300, 3400, 3400.1)] == [0, 1, 1, 1, 0]


def test_the_composite_leaves_out_an_in_range_goal_that_is_met_and_is_0_when_one_is_not():
    goals = [{"goal": "maximize", "low": 0, "high": 1}, {"goal": "in_range", "low": 0, "high": 1}]
    assert overall_desirability([0.25, 1.0], goals) == 0.25
    assert overall_desirability([0.25, 0.0], goals) == 0.0
    # The importance of an in-range goal is not used.
    assert overall_desirability([0.25, 1.0], goals, importances=[1.0, 5.0]) == 0.25


def test_the_desirability_contour_is_0_where_an_in_range_response_is_outside_its_limits():
    """The response 80 + 5A is maximised from 75 to 85, and the response B is held between -0.5 and 0.5."""
    responses = [
        {
            "coefficients": [{"term": "Intercept", "coefficient": 80.0}, {"term": "A", "coefficient": 5.0}],
            "goal": "maximize",
            "low": 75.0,
            "high": 85.0,
        },
        {
            "coefficients": [{"term": "Intercept", "coefficient": 0.0}, {"term": "B", "coefficient": 1.0}],
            "goal": "in_range",
            "low": -0.5,
            "high": 0.5,
        },
    ]
    plot = create_plot(
        "desirability_contour",
        analysis_results={"coefficients": responses[0]["coefficients"], "optimization": {"responses": responses}},
        factors_to_plot=["A", "B"],
    )
    style = plot.to_spec().panels[0].layers[0].style
    a, b = np.meshgrid(style["x_grid"], style["y_grid"])
    expected = np.where(np.abs(b) <= 0.5, (1 + a) / 2, 0.0)
    np.testing.assert_allclose(style["z_matrix"], expected, atol=1e-12)


# --------------------------------------------------------------------------- what is refused
@pytest.mark.parametrize(
    ("goal", "message"),
    [
        ({"goal": "in_range", "low": 1, "high": 2, "weight": 2}, "takes no weight"),
        ({"goal": "in_range", "low": 1}, "needs both 'low' and 'high'"),
        ({"goal": "within", "low": 1, "high": 2}, "'target' or 'in_range'"),
    ],
)
def test_malformed_goals_are_refused(goal, message):
    with pytest.raises(ValueError, match=message):
        check_goal(goal)


@pytest.mark.dataset
def test_goals_that_are_all_in_range_are_refused(yield_ccd):
    goals = [{**g, "goal": "in_range"} for g in GOALS]
    for goal in goals:
        goal.pop("target", None)
    with pytest.raises(ValueError, match="Every goal is 'in_range'"):
        optimize_responses(yield_ccd, goals=goals, method="desirability")


@pytest.mark.dataset
def test_limits_no_setting_can_meet_are_refused(yield_ccd):
    goals = [*GOALS[:2], {"response": "mn", "goal": "in_range", "low": 5000, "high": 6000}]
    with pytest.raises(ValueError, match="holds every 'in_range' response within its limits"):
        optimize_responses(yield_ccd, goals=goals, method="desirability")


@pytest.mark.dataset
def test_the_pareto_front_says_where_an_in_range_goal_belongs(yield_ccd):
    with pytest.raises(ValueError, match="use method='desirability' with an 'in_range' goal"):
        optimize_responses(yield_ccd, goals=GOALS, method="pareto_front")
