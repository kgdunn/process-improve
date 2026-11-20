"""The settings that meet a specification: ``api_specification``, which Learn calls for the windows typed.

The reader types a window for a response, or for several, and the page says over what share of the ranges every
prediction is inside its window, how surely one new run meets every window, and draws both over two factors.
These tests hold the region to surfaces known here, the chances to the predictive distribution worked out
independently, and the settings judged to the limits and to the level a categorical factor is held at.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from tests.web._support import FACTORS, coded, held_of, run_sheet
from tests.web.api.test_balance import SURFACES, TWO, design_of

#: The settings every test's chart is drawn through: the center of the ranges.
CENTER = {"T": 175, "P": 2}


def measured(held: dict) -> list[str]:
    """Return the columns of the responses a study measures, the first first."""
    return [r["name"] for r in held["spec"].get("responses") or [{"name": held["spec"]["response"]}]]


def results_of(held: dict) -> dict:
    """Return each response's results from its known surface, without noise, as the page sends them."""
    x = coded(run_sheet(held), FACTORS[:2])
    values = {n: SURFACES[n](x["T"], x["P"]).tolist() for n in measured(held)}
    first, *later = measured(held)
    return {"responses": values[first], "more_responses": {n: values[n] for n in later}}


def judged(api, held: dict, windows: dict, results: dict | None = None, **more) -> dict:
    """Return what the page says of some windows on the known surfaces, each fitted with the quadratic model."""
    sent = {**held, **(results or results_of(held)), "models": dict.fromkeys(measured(held), "quadratic")}
    return api.ok("api_specification", {**sent, "windows": windows, "at": CENTER, **more})


def drawn() -> tuple[np.ndarray, np.ndarray]:
    """Return the coded settings of T and P the page tries, drawn here as the page draws them."""
    settings = np.random.default_rng(1).uniform(-1.0, 1.0, size=(20_000, 2))
    return settings[:, 0], settings[:, 1]


def noisy(held: dict, sd: float = 0.3, seed: int = 3) -> dict:
    """Return the known surfaces' results with noise on the yield, as the page sends them."""
    results = results_of(held)
    noise = np.random.default_rng(seed).normal(0.0, sd, len(held["rows"]))
    return {**results, "responses": (np.asarray(results["responses"]) + noise).tolist()}


def test_the_share_and_the_region_are_those_of_the_known_surface(api):
    """The cost's model is its surface, so the settings inside the window are known exactly."""
    found = judged(api, design_of(api, TWO), {"Cost": [19.31, 21.69]})
    t, p = drawn()
    inside = (SURFACES["Cost"](t, p) >= 19.31) & (SURFACES["Cost"](t, p) <= 21.69)
    assert found["share"] == pytest.approx(inside.mean(), abs=1e-12)
    assert found["tried"] == 20_000
    overlay = found["overlay"]
    grid = np.asarray(overlay["grid"])
    across, up = np.meshgrid(grid, grid)
    surface = SURFACES["Cost"](across, up)
    assert np.array_equal(np.asarray(overlay["acceptable"]), ((surface > 19.31) & (surface < 21.69)).astype(int))
    assert overlay["responses"][0]["levels"] == [19.31, 21.69]
    assert found["responses"] == [{"name": "Cost", "label": "Cost", "units": "$/kg", "low": 19.31, "high": 21.69}]
    assert found["search"] == {
        "formulas": {"Cost": "quadratic"},
        "windows": {"Cost": [19.31, 21.69]},
        "at": {"T": 0.0, "P": 0.0},
        "correlation": None,
        "correlation_df": None,
        "drawn": 20_000,
        "seed": 1,
    }


def test_a_window_open_at_one_end_is_bounded_on_the_other_alone(api):
    held = design_of(api, TWO)
    found = judged(api, held, {"Yield": [59.0, None], "Cost": [None, 21.0]})
    t, p = drawn()
    inside = (SURFACES["Yield"](t, p) >= 59.0) & (SURFACES["Cost"](t, p) <= 21.0)
    assert found["share"] == pytest.approx(inside.mean(), abs=1e-12)
    assert [s["levels"] for s in found["overlay"]["responses"]] == [[59.0], [21.0]]
    assert [(r["low"], r["high"]) for r in found["responses"]] == [(59.0, None), (None, 21.0)]


def test_the_chance_is_the_share_of_a_new_run_s_distribution_inside_the_window(api):
    """Worked out here from a fit of the same model: Student's t about the prediction, by an observation's error."""
    import statsmodels.formula.api as smf

    held = design_of(api, TWO[:1], n_center_points=4)
    results = noisy(held)
    found = judged(api, held, {"Yield": [58.5, 61.5]}, results)
    runs = coded(run_sheet(held), FACTORS[:2]).assign(y=results["responses"])
    fit = smf.ols("y ~ T + P + T:P + I(T ** 2) + I(P ** 2)", data=runs).fit()
    grid = np.asarray(found["overlay"]["grid"])
    across, up = np.meshgrid(grid, grid)
    predicted = fit.get_prediction(pd.DataFrame({"T": across.ravel(), "P": up.ravel()}))
    scatter = np.sqrt(predicted.var_pred_mean + fit.mse_resid)
    chance = stats.t.cdf((61.5 - predicted.predicted_mean) / scatter, fit.df_resid) - stats.t.cdf(
        (58.5 - predicted.predicted_mean) / scatter, fit.df_resid
    )
    np.testing.assert_allclose(np.asarray(found["overlay"]["probability"]).ravel(), chance, atol=2e-6)
    assert found["overlay"]["chances"] == [0.5, 0.9]
    # The settings tried at which the chance is each of those or more, and the one at which it is highest.
    assert 0 < found["likely"]["0.9"] < found["likely"]["0.5"] < found["share"]
    surest = found["surest"]
    assert surest["probability"] == pytest.approx(chance.max(), abs=0.01)
    assert 58.5 < surest["responses"]["Yield"] < 61.5
    assert surest["confirm_at"].keys() == {"T", "P"}
    assert surest["settings"]["T"] == pytest.approx(175 + 25 * surest["coded"]["T"])


def test_more_runs_widen_the_settings_that_meet_the_window_nine_times_in_ten(api):
    """Every run made twice, with the same noise: the model is surer, and the 90% region grows inside the window's."""
    held = design_of(api, TWO[:1], n_center_points=4)
    once = noisy(held)
    rows = [
        *held["rows"],
        *([row[0] + len(held["rows"]), row[1] + len(held["rows"]), *row[2:]] for row in held["rows"]),
    ]
    twice = {**held, "rows": rows}
    doubled = {**once, "responses": [*once["responses"], *once["responses"]]}
    few = judged(api, held, {"Yield": [58.5, 61.5]}, once)
    many = judged(api, twice, {"Yield": [58.5, 61.5]}, doubled)
    assert few["share"] == pytest.approx(many["share"]), "the same coefficients, so the same region"
    assert few["likely"]["0.9"] < many["likely"]["0.9"] < many["share"]


def together(held: dict) -> dict:
    """Return the known surfaces' results with a noise the yield and the cost share on each run."""
    clean = results_of(held)
    shared = np.random.default_rng(11).normal(0.0, 0.4, len(held["rows"]))
    own = np.random.default_rng(12).normal(0.0, 0.1, len(held["rows"]))
    cost = np.asarray(clean["more_responses"]["Cost"]) + shared + own
    return {"responses": (np.asarray(clean["responses"]) + shared).tolist(), "more_responses": {"Cost": cost.tolist()}}


def test_two_responses_errors_are_taken_as_correlated_as_their_fits_residuals_are(api, bootstrap, monkeypatch):
    """The yield and the cost miss their surfaces together: a high yield with a low cost is then less likely."""
    held = design_of(api, TWO, n_center_points=6)  # 14 runs, and 6 terms in each quadratic model: 8 runs to spare
    results, windows = together(held), {"Yield": [59.0, None], "Cost": [None, 21.0]}
    found = judged(api, held, windows, results)
    x = coded(run_sheet(held), FACTORS[:2])
    matrix = np.column_stack([np.ones(len(x)), x["T"], x["P"], x["T"] * x["P"], x["T"] ** 2, x["P"] ** 2])
    residuals = [
        np.asarray(y) - matrix @ np.linalg.lstsq(matrix, np.asarray(y), rcond=None)[0]
        for y in (results["responses"], results["more_responses"]["Cost"])
    ]
    expected = np.corrcoef(residuals)
    assert expected[0, 1] > 0.8
    np.testing.assert_allclose(found["search"]["correlation"], expected, atol=1e-9)
    monkeypatch.setattr(bootstrap, "CORRELATED_FROM", 100)  # the same study, its errors taken as independent
    apart = judged(api, held, windows, results)
    assert apart["search"]["correlation"] is None
    assert apart["share"] == found["share"], "where the predictions lie is the same"
    assert found["surest"]["probability"] < apart["surest"]["probability"]
    assert found["likely"]["0.5"] < apart["likely"]["0.5"]


def test_too_few_runs_to_spare_leave_the_responses_errors_independent(api):
    held = design_of(api, TWO, n_center_points=2)  # 10 runs and 6 terms: 4 to spare, one too few
    found = judged(api, held, {"Yield": [59.0, None], "Cost": [None, 21.0]}, together(held))
    assert found["search"]["correlation"] is None
    assert found["likely"] is not None, "the chance is still given, as the product of each response's"


def test_a_fit_with_no_run_left_for_the_noise_gives_the_region_and_no_chance(api):
    """Four runs and a model of four terms: the surface is found, and nothing is left to say how sure it is."""
    spec = {"design_type": "full_factorial", "factors": FACTORS[:2], "responses": TWO[:1], "n_center_points": 0}
    held = held_of(api.ok("api_make_design", spec))
    sent = {**held, **results_of(held), "models": {"Yield": "interactions"}}
    found = api.ok("api_specification", {**sent, "windows": {"Yield": [52, 56]}, "at": CENTER})
    assert found["likely"] is None
    assert found["surest"] is None
    assert found["overlay"]["probability"] is None
    assert 0 < found["share"] < 1


def test_the_settings_judged_keep_to_the_limits_on_the_settings(api):
    limit = {"expression": "T + 25*P <= 215"}
    held = design_of(api, TWO, design_type="d_optimal", constraints=[limit], budget=12, model="quadratic")
    found = judged(api, held, {"Cost": [None, 19.0]}, noisy(held, sd=0.2) | {"more_responses": noisy_cost(held)})
    t, p = drawn()
    allowed = t + p <= -0.4 + 1e-12  # the limit, in coded settings
    assert found["tried"] == int(allowed.sum())
    assert found["overlay"]["runnable"] is not None
    at = found["surest"]["settings"]
    assert at["T"] + 25 * at["P"] <= 215 + 1e-6


def noisy_cost(held: dict) -> dict:
    """Return the cost's results with noise, by its column, as the page sends a later response's."""
    cost = np.asarray(results_of(held)["more_responses"]["Cost"])
    return {"Cost": (cost + np.random.default_rng(5).normal(0.0, 0.2, len(cost))).tolist()}


def test_a_categorical_factor_is_held_at_the_level_the_reader_has_it_at(api):
    factors = [*FACTORS[:2], {"name": "S", "low": "A", "high": "B", "units": ""}]
    spec = {"design_type": "full_factorial", "factors": factors, "responses": TWO[:1], "n_center_points": 3}
    held = held_of(api.ok("api_make_design", spec))
    x = coded(run_sheet(held), factors)
    yields = (60 + 4 * x["T"] - 2 * x["P"] + 3 * x["S"]).tolist()  # 51 to 63 with A, 57 to 69 with B
    request = {**held, "responses": yields, "models": {"Yield": "main_effects"}, "windows": {"Yield": [64.0, None]}}
    with_b = api.ok("api_specification", {**request, "at": {**CENTER, "S": "B"}})
    with_a = api.ok("api_specification", {**request, "at": {**CENTER, "S": "A"}})
    assert with_a["share"] == 0, "no setting reaches 64 with A"
    t, p = drawn()
    assert with_b["share"] == pytest.approx((63 + 4 * t - 2 * p >= 64).mean(), abs=1e-12)
    assert with_b["held"] == [{"name": "S", "value": "B"}]
    assert with_b["overlay"]["held"] == [{"name": "S", "value": "B"}]
    assert with_b["search"]["at"] == {"T": 0.0, "P": 0.0, "S": 1.0}, "as the columns the model is fitted on"
    # Beside it, what the window gives at each level: a level is chosen, not moved to.
    for reply, here in ((with_a, "A"), (with_b, "B")):
        rows = {row["levels"][0]["value"]: row for row in reply["levels"]}
        assert [row["levels"] for row in reply["levels"]] == [[{"name": "S", "value": level}] for level in "AB"]
        assert {level: row["here"] for level, row in rows.items()} == {"A": here == "A", "B": here == "B"}
        assert rows["A"]["share"] == 0
        assert rows["B"]["share"] == with_b["share"], "the share where What if has the level is the reply's own"
        assert rows["B"]["surest"] == with_b["surest"]["probability"]
        assert rows["A"]["surest"] < rows["B"]["surest"]


def test_a_study_of_many_levels_is_judged_at_each_combination_up_to_a_limit_and_says_how_many_there_are(api, bootstrap):
    factors = [*FACTORS[:2], {"name": "S", "levels": list("ABCDE")}, {"name": "R", "levels": list("WXYZ")}]
    spec = {"design_type": "d_optimal", "factors": factors, "responses": TWO[:1], "budget": 40, "random_state": 1}
    held = held_of(api.ok("api_make_design", spec))
    yields = np.random.default_rng(1).normal(60, 2, len(held["rows"])).tolist()
    request = {**held, "responses": yields, "models": {"Yield": "main_effects"}, "windows": {"Yield": [58.0, None]}}
    reply = api.ok("api_specification", {**request, "at": {**CENTER, "S": "A", "R": "W"}})
    assert reply["levels_in_all"] == 20
    assert len(reply["levels"]) == 20, "more than the first 12, which was the limit"
    assert sum(row["here"] for row in reply["levels"]) == 1
    assert bootstrap.SPECIFICATION_LEVELS >= 20


def test_a_study_with_no_categorical_factor_has_no_levels_to_judge_at(api):
    found = judged(api, design_of(api, TWO[:1]), {"Yield": [60.0, None]})
    assert (found["levels"], found["levels_in_all"]) == ([], 0)


def test_a_transformed_response_s_window_is_typed_and_drawn_in_its_own_units(api):
    held = design_of(api, TWO[:1])
    held = {**held, "spec": {**held["spec"], "analysis_transforms": {"Yield": "log"}}}
    found = judged(api, held, {"Yield": [59.0, 61.0]})
    assert found["responses"][0]["low"] == pytest.approx(59.0)
    assert found["responses"][0]["high"] == pytest.approx(61.0)
    assert found["search"]["windows"]["Yield"] == pytest.approx([np.log(59.0), np.log(61.0)])
    assert found["overlay"]["responses"][0]["levels"] == pytest.approx([59.0, 61.0])
    assert 0 < found["share"] < 1


def test_the_chart_is_drawn_over_the_two_factors_chosen_through_the_reader_s_settings(api):
    held = held_of(api.ok("api_make_design", {"design_type": "box_behnken", "factors": FACTORS, "responses": TWO[:1]}))
    x = coded(run_sheet(held), FACTORS)
    yields = (60 + 4 * x["T"] + 2 * x["P"] - 3 * x["F"]).tolist()
    sent = {**held, "responses": yields, "models": {"Yield": "main_effects"}, "windows": {"Yield": [60.0, 63.0]}}
    at = {"T": 175, "P": 2.5, "F": 12}
    found = api.ok("api_specification", {**sent, "at": at, "pair": ["F", "T"]})
    overlay = found["overlay"]
    assert (overlay["x"], overlay["y"]) == ("F", "T")
    assert overlay["held"] == [{"name": "P", "value": 2.5}]
    assert overlay["at"] == pytest.approx([-0.6, 0.0])
    grid = np.asarray(overlay["grid"])
    across, up = np.meshgrid(grid, grid)  # F across, T up, with P at +0.5
    np.testing.assert_allclose(overlay["responses"][0]["z"], 60 + 4 * up + 1 - 3 * across, rtol=1e-5)
    assert len(overlay["runs"]) == 9, "a Box-Behnken design's runs lie at nine places over any two factors"
    assert api.ok("api_specification", {**sent, "at": at})["overlay"]["x"] == "T", "the factors that matter most"


def test_a_study_with_one_factor_with_a_range_has_its_share_and_no_chart(api):
    factors = [FACTORS[0], {"name": "S", "low": "A", "high": "B", "units": ""}]
    spec = {"design_type": "full_factorial", "factors": factors, "responses": TWO[:1], "n_replicates": 2}
    held = held_of(api.ok("api_make_design", {**spec, "n_center_points": 0}))
    x = coded(run_sheet(held), factors)
    request = {**held, "responses": (60 + 4 * x["T"] + 3 * x["S"]).tolist(), "models": {"Yield": "main_effects"}}
    found = api.ok("api_specification", {**request, "windows": {"Yield": [62.0, None]}, "at": {"T": 175, "S": "B"}})
    assert found["overlay"] is None, "a chart is over two factors with ranges"
    t = np.random.default_rng(1).uniform(-1.0, 1.0, size=(20_000, 1))[:, 0]
    assert found["share"] == pytest.approx((63 + 4 * t >= 62).mean(), abs=1e-12), "with B, from t = -0.25 up"
    assert found["held"] == [{"name": "S", "value": "B"}]
