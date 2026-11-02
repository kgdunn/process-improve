"""Predictions and optimisation from a split-plot fit, by generalized least squares (#789 B9).

Goos and Jones's battery cell (Section 11.3) is a two-way split-plot design, analysed by REML with
its rows and columns as crossed random effects. Its least OCV is at x1 high, x4 low, x5 and x6
high, predicted 7.42; the ordinary least-squares fit of the same model predicts 7.75 there. Their
pastry dough (Section 7.3), run on seven days taken as random blocks, has both responses on target
at 40.6 kg/h, 22.3% moisture and 300 rpm.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import analyze_experiment, optimize_responses
from process_improve.experiments.reference_catalogue import reference_experiments

BATTERY_MODEL = "x1 + x4 + x5 + x6 + x1:x5 + x1:x6 + x4:x6"
JMP = {"df_method": "kenward_roger", "information": "observed"}
BEST_CORNER = {"x1": 1.0, "x4": -1.0, "x5": 1.0, "x6": 1.0}


def catalogue(name: str) -> pd.DataFrame:
    [entry] = [e for e in reference_experiments("goos-jones-2011") if e["name"] == name]
    return pd.DataFrame(entry["runs"]["rows"], columns=entry["runs"]["columns"])


@pytest.fixture(scope="module")
def battery() -> pd.DataFrame:
    return catalogue("goos-jones-battery-cell")


def battery_fit(runs: pd.DataFrame, points: list[dict[str, float]] | None = None, **options) -> dict:
    """Return the split-plot fit of the battery cell's OCV, with predictions at ``points``."""
    new_points = None if points is None else pd.DataFrame(points)
    return analyze_experiment(
        runs[["row", "column", "x1", "x4", "x5", "x6"]],
        runs["ocv"],
        model=BATTERY_MODEL,
        analysis_type="split_plot",
        whole_plot=["row", "column"],
        new_points=new_points,
        **(JMP | options),
    )["split_plot"]


@pytest.fixture(scope="module")
def pastry() -> pd.DataFrame:
    """Return the pastry dough's runs, coded as the book codes them, with the days as labels."""
    runs = catalogue("goos-jones-pastry-dough")
    return runs.assign(
        flow=(runs["flow"] - 37.5) / 7.5, moisture=(runs["moisture"] - 21) / 3, speed=(runs["speed"] - 350) / 50
    )


PASTRY_MODELS = {
    "y1": "flow + moisture + speed + flow:moisture + I(moisture ** 2) + I(speed ** 2)",
    "y2": "moisture + speed + I(moisture ** 2)",
}


def pastry_fit(runs: pd.DataFrame, response: str, points: list[dict[str, float]] | None = None) -> dict:
    return analyze_experiment(
        runs[["day", "flow", "moisture", "speed"]],
        runs[response],
        model=PASTRY_MODELS[response],
        analysis_type="split_plot",
        whole_plot="day",
        new_points=None if points is None else pd.DataFrame(points),
        **JMP,
    )["split_plot"]


# --------------------------------------------------------------------------- the sources
@pytest.mark.dataset
def test_the_battery_cell_s_least_ocv_is_predicted_from_the_gls_fit(battery):
    [at_corner] = battery_fit(battery, [BEST_CORNER])["predictions"]
    assert round(at_corner["predicted"], 2) == 7.42
    ols = analyze_experiment(
        battery[["x1", "x4", "x5", "x6"]],
        battery["ocv"],
        model=BATTERY_MODEL,
        analysis_type="prediction",
        new_points=pd.DataFrame([BEST_CORNER]),
    )
    assert round(ols["predictions"][0]["predicted"], 2) == 7.75


@pytest.mark.dataset
def test_the_optimizer_finds_the_best_corner_from_the_split_plot_fit(battery):
    fit = battery_fit(battery)
    assert (fit["response_name"], fit["coding"]) == ("ocv", "coded")
    goal = {"response": "ocv", "goal": "minimize", "low": 0, "high": 60}
    found = optimize_responses([fit], goals=[goal], method="desirability")["desirability"]
    assert {k: round(v, 6) for k, v in found["optimal_coded"].items()} == BEST_CORNER
    assert round(found["predicted_responses"]["ocv"], 2) == 7.42


@pytest.mark.dataset
def test_the_pastry_dough_is_on_target_at_the_settings_the_book_shows(pastry):
    """At 40.6 kg/h, 22.3% and 300 rpm the fits give 12.03 and 3.99: the book's targets, 12 and 4, to its precision.

    The catalogue's note gives 12.019 and 3.994 from Table 7.9's coefficients as printed, which are rounded.
    """
    point = [{"flow": (40.6 - 37.5) / 7.5, "moisture": (22.3 - 21) / 3, "speed": -1.0}]
    longitudinal = pastry_fit(pastry, "y1", point)["predictions"][0]["predicted"]
    cross_sectional = pastry_fit(pastry, "y2", point)["predictions"][0]["predicted"]
    assert (round(longitudinal, 2), round(cross_sectional, 2)) == (12.03, 3.99)
    assert (round(longitudinal), round(cross_sectional)) == (12, 4)


@pytest.mark.dataset
def test_both_pastry_targets_are_met_together_from_the_split_plot_fits(pastry):
    fits = [pastry_fit(pastry, "y1"), pastry_fit(pastry, "y2")]
    goals = [
        {"response": "y1", "goal": "target", "low": 10, "target": 12, "high": 14},
        {"response": "y2", "goal": "target", "low": 3, "target": 4, "high": 5},
    ]
    found = optimize_responses(fits, goals=goals, method="desirability")["desirability"]
    assert found["composite_desirability"] == pytest.approx(1.0, abs=1e-6)
    assert found["predicted_responses"] == pytest.approx({"y1": 12.0, "y2": 4.0}, abs=1e-5)


# --------------------------------------------------------------------------- what the predictions are
@pytest.mark.dataset
@pytest.mark.parametrize("df_method", ["kenward_roger", "satterthwaite"])
def test_the_prediction_at_the_centre_is_the_intercept_s_row(battery, df_method):
    fit = battery_fit(battery, [dict.fromkeys(BEST_CORNER, 0.0)], df_method=df_method)
    [centre] = fit["predictions"]
    [intercept] = [row for row in fit["coefficients"] if row["term"] == "Intercept"]
    for key in ("std_error", "df", "ci_low", "ci_high"):
        assert centre[key] == pytest.approx(intercept[key], rel=1e-10)
    assert centre["predicted"] == pytest.approx(intercept["coefficient"], rel=1e-12)


@pytest.mark.dataset
def test_a_new_run_s_variance_adds_every_variance_component(battery, pastry):
    """A new run falls in a new whole plot: a new row and column, or a new day."""
    for fit in (
        battery_fit(battery, [BEST_CORNER]),
        pastry_fit(pastry, "y1", [{"flow": 0, "moisture": 0, "speed": 0}]),
    ):
        [at] = fit["predictions"]
        added = at["pi_std_error"] ** 2 - at["std_error"] ** 2
        variances = [value for name, value in fit["variance_components"].items() if name != "eta"]
        assert added == pytest.approx(sum(variances), rel=1e-10)
        assert at["pi_low"] < at["ci_low"] < at["predicted"] < at["ci_high"] < at["pi_high"]


@pytest.mark.dataset
def test_coefficients_in_actual_units_are_labelled_and_refused_by_the_optimizer():
    """A split-plot fit is made on the runs as given, so actual units give actual-unit coefficients."""
    runs = catalogue("goos-jones-pastry-dough")
    fit = analyze_experiment(
        runs[["day", "flow", "moisture", "speed"]], runs["y1"], model=PASTRY_MODELS["y1"], analysis_type="split_plot",
        whole_plot="day",
    )["split_plot"]  # fmt: skip
    assert fit["coding"] == "actual"
    assert fit["factor_coding"]["flow"] == {"low": 30.0, "high": 45.0}
    goal = {"response": "y1", "goal": "target", "low": 10, "target": 12, "high": 14}
    with pytest.raises(ValueError, match="code its factors to -1 and \\+1"):
        optimize_responses([fit], goals=[goal], method="desirability")


def test_without_new_points_there_are_no_predictions():
    rng = np.random.default_rng(3)
    runs = pd.DataFrame({"WholePlot": np.repeat(range(6), 4), "A": np.repeat([-1, 1] * 3, 4), "B": [-1, 1] * 12})
    runs["y"] = runs["A"] + runs["B"] + rng.normal(size=24) + np.repeat(rng.normal(size=6), 4)
    fit = analyze_experiment(runs, response_column="y", model="A + B", analysis_type="split_plot")["split_plot"]
    assert "predictions" not in fit
    assert fit["factor_names"] == ["A", "B"]
