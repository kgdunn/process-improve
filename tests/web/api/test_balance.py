"""The settings that balance the responses: ``api_balance``, which Learn calls once two responses or more have a goal.

Each response with a goal scores a Derringer-Suich desirability d over its acceptable range, and the
overall desirability D is their geometric mean, each weighted by its importance. The page finds the
settings with the best D with the library's ``optimize_responses``. These tests hold the answer to
what can be checked another way: a dense grid of D over the same models, computed here from the
desirability's own definition; the results' range, which sets the default ranges; and the way a
range, an importance or a shape the reader sets moves the answer.
"""

from __future__ import annotations

import numpy as np
import pytest

from process_improve.experiments import DesignRegion, Factor
from tests.web._support import FACTORS, coded, held_of, run_sheet

#: A yield to maximise and a cost to minimise, as the brief describes them.
TWO = [
    {"name": "Yield", "label": "Yield", "units": "%", "goal": "maximize"},
    {"name": "Cost", "label": "Cost", "units": "$/kg", "goal": "minimize"},
]

#: Known surfaces over the coded T and P: the yield peaks inside the ranges, the cost rises with T and P.
SURFACES = {
    "Yield": lambda t, p: 60 + 4 * t + 2 * p - 3 * t**2 - 2 * p**2,
    "Cost": lambda t, p: 20 + 3 * t + 1 * p + 0.5 * t**2,
    "Purity": lambda t, p: 98 - 1.5 * t + 0.8 * p - 0.6 * p**2,
}


# --------------------------------------------------------------------------- helpers
def design_of(api, responses: list[dict], **more) -> dict:
    """Make a face-centred CCD in T and P measuring these responses, as the page holds it."""
    spec = {"design_type": "ccd", "factors": FACTORS[:2], "responses": responses, "alpha": "face_centered", **more}
    return held_of(api.ok("api_make_design", spec))


def results_of(held: dict) -> dict:
    """Return each response's results from its known surface, without noise, as the page sends them."""
    x = coded(run_sheet(held), held["spec"]["factors"])
    names = [r["name"] for r in held["spec"]["responses"]]
    values = {n: SURFACES[n](x["T"], x["P"]).tolist() for n in names}
    return {"responses": values[names[0]], "more_responses": {n: values[n] for n in names[1:]}}


def balanced(api, held: dict, goals: dict | None = None, **more) -> dict:
    """Return the balance the page finds, every response fitted with the quadratic model."""
    names = [r["name"] for r in held["spec"]["responses"]]
    models = dict.fromkeys(names, "quadratic")
    return api.ok("api_balance", {**held, **results_of(held), "models": models, "goals": goals or {}, **more})


def desirability(y: np.ndarray, goal: dict) -> np.ndarray:
    """Derringer and Suich's d, written out here from its definition, as an independent check."""
    low, high, w = goal["low"], goal["high"], goal["weight"]
    if goal["goal"] == "maximize":
        return np.clip((y - low) / (high - low), 0, 1) ** w
    if goal["goal"] == "minimize":
        return np.clip((high - y) / (high - low), 0, 1) ** w
    t = goal["target"]
    rising = np.clip((y - low) / (t - low), 0, 1) ** w
    falling = np.clip((high - y) / (high - t), 0, 1) ** w
    return np.where(y <= t, rising, falling)


def overall(goals: list[dict], t: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Return D, the importance-weighted geometric mean of the d's, at coded settings of T and P."""
    weights = np.array([g["importance"] for g in goals], dtype=float)
    ds = np.array([desirability(SURFACES[g["response"]](t, p), g) for g in goals])
    with np.errstate(divide="ignore"):
        return np.exp((weights[:, None] * np.log(ds)).sum(axis=0) / weights.sum())


def grid_best(goals: list[dict], inside=None, points: int = 401) -> tuple[float, float, float]:
    """Return the best D on a dense grid over the coded ranges, and where it is."""
    t, p = (axis.ravel() for axis in np.meshgrid(np.linspace(-1, 1, points), np.linspace(-1, 1, points)))
    d = overall(goals, t, p)
    if inside is not None:
        d = np.where(inside(t, p), d, -1.0)
    i = int(np.argmax(d))
    return float(d[i]), float(t[i]), float(p[i])


# --------------------------------------------------------------------------- when there is a balance to find
def test_one_response_with_a_goal_has_nothing_to_balance(api):
    learn = [TWO[0], {**TWO[1], "goal": "learn"}, {"name": "Purity", "goal": "target"}]  # a target not yet typed
    held = design_of(api, learn)
    x = coded(run_sheet(held), held["spec"]["factors"])
    values = SURFACES["Yield"](x["T"], x["P"]).tolist()
    reply = api.ok("api_balance", {**held, "responses": values})
    assert reply == {"balanced": False, "aimed": ["Yield"]}


def test_the_balance_is_the_best_overall_desirability_on_a_grid(api):
    held = design_of(api, TWO)
    reply = balanced(api, held)
    assert reply["balanced"]
    goals = reply["search"]["goals"]
    best, *_ = grid_best(goals)
    assert reply["composite"] == pytest.approx(best, abs=2e-4), "no setting on a fine grid does better"
    t, p = reply["coded"]["T"], reply["coded"]["P"]
    assert reply["composite"] == pytest.approx(float(overall(goals, np.array([t]), np.array([p]))[0]), abs=1e-9)
    for r in reply["responses"]:
        assert r["predicted"] == pytest.approx(SURFACES[r["name"]](t, p), abs=1e-6), "an exact fit predicts the surface"
        assert r["d"] == pytest.approx(
            float(desirability(np.array(r["predicted"]), {**r, "goal": r["goal"]})), abs=1e-9
        )


def test_three_responses_one_with_a_target_are_balanced_together(api):
    three = [*TWO, {"name": "Purity", "label": "Purity", "units": "%", "goal": "target", "target": 98.5}]
    held = design_of(api, three)
    reply = balanced(api, held)
    assert [r["name"] for r in reply["responses"]] == ["Yield", "Cost", "Purity"]
    best, *_ = grid_best(reply["search"]["goals"])
    assert reply["composite"] == pytest.approx(best, abs=2e-4)


def test_the_default_ranges_are_the_results_own_and_a_target_s_reaches_the_farthest(api):
    three = [*TWO, {"name": "Purity", "label": "Purity", "units": "%", "goal": "target", "target": 98.5}]
    held = design_of(api, three)
    sent = results_of(held)
    reply = balanced(api, held)
    by_name = {r["name"]: r for r in reply["responses"]}
    yields, costs, purity = sent["responses"], sent["more_responses"]["Cost"], sent["more_responses"]["Purity"]
    assert (by_name["Yield"]["low"], by_name["Yield"]["high"]) == (min(yields), max(yields))
    assert (by_name["Cost"]["low"], by_name["Cost"]["high"]) == (min(costs), max(costs))
    half = max(98.5 - min(purity), max(purity) - 98.5)
    assert by_name["Purity"]["low"] == pytest.approx(98.5 - half)
    assert by_name["Purity"]["high"] == pytest.approx(98.5 + half)
    for r in reply["responses"]:
        assert (r["importance"], r["shape"], r["weight"]) == (3, "even", 1.0)
        assert (r["default_low"], r["default_high"]) == (r["low"], r["high"])


# --------------------------------------------------------------------------- what the reader sets
def test_a_response_counted_as_more_important_is_balanced_nearer_its_own_best(api):
    held = design_of(api, TWO)
    even = balanced(api, held)
    yield_first = balanced(api, held, {"Yield": {"importance": 5}, "Cost": {"importance": 1}})
    cost_first = balanced(api, held, {"Yield": {"importance": 1}, "Cost": {"importance": 5}})
    d = lambda reply, name: next(r["d"] for r in reply["responses"] if r["name"] == name)  # noqa: E731
    assert d(yield_first, "Yield") > d(even, "Yield") > d(cost_first, "Yield")
    assert d(cost_first, "Cost") > d(even, "Cost") > d(yield_first, "Cost")
    for reply in (yield_first, cost_first):
        best, *_ = grid_best(reply["search"]["goals"])
        assert reply["composite"] == pytest.approx(best, abs=2e-4)


def test_a_narrower_acceptable_range_pulls_the_balance_towards_it(api):
    held = design_of(api, TWO)
    even = balanced(api, held)
    cost = next(r for r in even["responses"] if r["name"] == "Cost")
    strict = balanced(api, held, {"Cost": {"low": cost["low"], "high": cost["low"] + 2.0}})
    cost_of = lambda reply: next(r["predicted"] for r in reply["responses"] if r["name"] == "Cost")  # noqa: E731
    assert cost_of(strict) < cost_of(even), "a cost acceptable only near its lowest: the balance costs less"
    assert next(r for r in strict["responses"] if r["name"] == "Cost")["high"] == cost["low"] + 2.0
    best, *_ = grid_best(strict["search"]["goals"])
    assert strict["composite"] == pytest.approx(best, abs=2e-4)


def test_a_shape_that_counts_the_good_end_more_lowers_d_inside_the_range(api):
    held = design_of(api, TWO)
    even = balanced(api, held)
    more = balanced(api, held, {"Yield": {"shape": "much_more"}})
    yield_more = next(r for r in more["responses"] if r["name"] == "Yield")
    assert yield_more["weight"] == 4.0
    assert more["composite"] < even["composite"], "the same results count for less unless they are near the best"
    best, *_ = grid_best(more["search"]["goals"])
    assert more["composite"] == pytest.approx(best, abs=2e-4)


def test_the_response_that_holds_d_down_most_is_named(api):
    held = design_of(api, TWO)
    reply = balanced(api, held, {"Cost": {"importance": 5}})
    weights = {r["name"]: r["importance"] for r in reply["responses"]}
    total = sum(weights.values())
    shares = {r["name"]: r["d"] ** (weights[r["name"]] / total) for r in reply["responses"]}
    assert reply["limiting"] == min(shares, key=shares.get)


def test_ranges_no_setting_meets_give_d_of_zero_and_the_closest_settings(api):
    held = design_of(api, TWO)
    reply = api.reply(
        "api_balance",
        {**held, **results_of(held), "goals": {"Yield": {"low": 90, "high": 95}}, "models": {"Yield": "quadratic"}},
    )
    assert reply["ok"]
    assert reply["result"]["composite"] == 0
    assert any("desirability above 0" in w for w in reply["warnings"])


# --------------------------------------------------------------------------- the limits and the categorical factors
def test_the_balance_keeps_to_the_limits_on_the_settings(api):
    limit = {"expression": "T + 25*P <= 215"}  # T 175 and P 1.6 at the centre of their ranges' corner
    held = design_of(api, TWO, design_type="d_optimal", constraints=[limit], model_type="quadratic", budget=12)
    reply = balanced(api, held)
    region = DesignRegion(
        [Factor(name=f["name"], low=f["low"], high=f["high"]) for f in FACTORS[:2]], [limit["expression"]]
    )
    at = np.array([[reply["coded"]["T"], reply["coded"]["P"]]])
    assert region.feasible(at, tol=1e-6)[0]
    inside = lambda t, p: region.feasible(np.column_stack([t, p]))  # noqa: E731
    best, *_ = grid_best(reply["search"]["goals"], inside, points=201)
    assert reply["composite"] >= best - 2e-4, "no runnable setting on a grid does better"
    assert reply["overlay"]["runnable"] is not None, "the overlay greys what cannot be run"


def test_a_categorical_factor_is_held_at_the_level_that_balances_best(api):
    factors = [FACTORS[0], {"name": "S", "low": "A", "high": "B", "units": ""}]
    spec = {"design_type": "full_factorial", "factors": factors, "responses": TWO, "n_center_points": 3}
    held = held_of(api.ok("api_make_design", spec))
    x = coded(run_sheet(held), factors)
    t, s = x["T"], x["S"]
    yields = (60 + 4 * t + 3 * s).tolist()
    costs = (20 + 2 * t + 0.5 * s).tolist()
    reply = api.ok(
        "api_balance",
        {**held, "responses": yields, "more_responses": {"Cost": costs}, "models": {"Cost": "interactions"}},
    )
    assert reply["coded"]["S"] == 1.0, "supplier B gives far more yield for a little more cost"
    assert next(s for s in reply["settings"] if s["name"] == "S")["value"] == "B"
    assert reply["search"]["held"] == {"S": 1.0}
    assert reply["overlay"] is None, "one numeric factor: no pair to draw"


# --------------------------------------------------------------------------- what the reply says
def test_each_response_has_its_intervals_and_the_leverage_is_the_furthest_model_s(api):
    held = design_of(api, [*TWO, {"name": "Purity", "goal": "target", "target": 98.5}])
    sent = results_of(held)
    noisy = {**sent, "responses": [v + 0.1 * (-1) ** i for i, v in enumerate(sent["responses"])]}
    reply = api.ok("api_balance", {**held, **noisy, "models": {"Yield": "quadratic", "Cost": "main_effects"}})
    for r in reply["responses"]:
        low, high = r["prediction"]
        assert low < r["predicted"] < high
        assert r["confidence"][0] > low, "a single run varies more than the mean"
    ratios = {r["name"]: r["leverage"] / r["max_leverage"] for r in reply["responses"]}
    assert reply["leverage_of"] == max(ratios, key=ratios.get)
    assert reply["search"]["formulas"]["Cost"] == "main_effects"


def test_the_overlay_shades_where_every_response_is_acceptable(api):
    held = design_of(api, TWO)
    reply = balanced(api, held)
    overlay = reply["overlay"]
    assert {overlay["x"], overlay["y"]} == {"T", "P"}
    grid = np.array(overlay["grid"])
    t, p = np.meshgrid(grid, grid) if overlay["x"] == "T" else np.meshgrid(grid, grid)[::-1]
    goals = {g["response"]: g for g in reply["search"]["goals"]}
    surfaces = {s["name"]: s for s in overlay["responses"]}
    np.testing.assert_allclose(surfaces["Yield"]["z"], SURFACES["Yield"](t, p), rtol=1e-5)
    yields, costs = SURFACES["Yield"](t, p), SURFACES["Cost"](t, p)
    acceptable = (yields > goals["Yield"]["low"]) & (costs < goals["Cost"]["high"])
    # A setting exactly on a boundary, as a run whose result is the least, may read either way to the fit's rounding.
    on_edge = np.isclose(yields, goals["Yield"]["low"], atol=1e-6) | np.isclose(costs, goals["Cost"]["high"], atol=1e-6)
    shaded = np.array(overlay["acceptable"], dtype=bool)
    assert np.array_equal(shaded[~on_edge], acceptable[~on_edge])
    assert shaded.any(), "some settings are acceptable for both"
    assert not shaded.all(), "and some are not"
    assert surfaces["Yield"]["levels"] == [goals["Yield"]["low"]], "a yield to maximise is acceptable above its low end"
    assert surfaces["Cost"]["levels"] == [goals["Cost"]["high"]], "a cost to minimise, below its high end"
    assert overlay["at"] == [reply["coded"][overlay["x"]], reply["coded"][overlay["y"]]]
    assert overlay["ticks"]["T"]["ticktext"] == ["150", "175", "200"]
    assert overlay["runnable"] is None


def test_confirmation_runs_are_judged_response_by_response_and_left_out_of_the_fit(api):
    held = design_of(api, TWO)
    before = balanced(api, held)
    at = before["confirm_at"]
    columns = [*held["columns"], "Confirmation run"]
    extra = [[{"Run order": 99 + i, "Confirmation run": "yes", **at}.get(c) for c in columns] for i in range(3)]
    rows = [*[[*row, None] for row in held["rows"]], *extra]
    sent = results_of(held)
    yields = [r["predicted"] for r in before["responses"] if r["name"] == "Yield"] * 3
    costs = [100.0, 100.0, 100.0]  # far above any cost the model predicts: the process differs there
    confirmed = {
        **held,
        "columns": columns,
        "rows": rows,
        "responses": [*sent["responses"], *yields],
        "more_responses": {"Cost": [*sent["more_responses"]["Cost"], *costs]},
        "models": {"Yield": "quadratic", "Cost": "quadratic"},
    }
    noisy = {**confirmed, "responses": [v + 0.05 * (-1) ** i for i, v in enumerate(confirmed["responses"])]}
    reply = api.ok("api_balance", noisy)
    (group,) = reply["confirmation"]
    assert group["runs"] == 3
    verdicts = {r["name"]: r for r in group["responses"]}
    assert verdicts["Yield"]["inside"] is True
    assert verdicts["Cost"]["inside"] is False
    assert verdicts["Cost"]["mean"] == 100.0
    assert reply["coded"]["T"] == pytest.approx(before["coded"]["T"], abs=0.05), (
        "the fit leaves the confirmation runs out"
    )


# --------------------------------------------------------------------------- refusals
@pytest.mark.parametrize(
    ("goals", "message"),
    [
        (
            {"Yield": {"low": 60, "high": 50}},
            "The acceptable range of Yield must run from a lower value to a higher one.",
        ),
        (
            {"Cost": {"low": 20, "high": "abc"}},
            "The acceptable range of Cost must run from a lower value to a higher one.",
        ),
        ({"Yield": {"importance": 0}}, "The importance of Yield must be a whole number from 1 to 5."),
        ({"Yield": {"importance": 2.5}}, "The importance of Yield must be a whole number from 1 to 5."),
        ({"Yield": {"importance": True}}, "The importance of Yield must be a whole number from 1 to 5."),
        ({"Cost": {"shape": "steep"}}, "The shape of Cost's desirability must be one the page offers."),
        ({"Cost": "narrow"}, "The page sent goals for the responses this page did not write."),
    ],
)
def test_a_goal_the_balance_cannot_take_is_refused_saying_why(api, goals, message):
    held = design_of(api, TWO)
    assert api.error("api_balance", {**held, **results_of(held), "goals": goals}) == message


def test_a_target_outside_its_acceptable_range_is_refused(api):
    held = design_of(api, [*TWO, {"name": "Purity", "label": "Purity", "goal": "target", "target": 98.5}])
    error = api.error("api_balance", {**held, **results_of(held), "goals": {"Purity": {"low": 90, "high": 95}}})
    assert error == "The target of Purity, 98.5, must lie inside its acceptable range."


def test_a_response_without_results_is_refused(api):
    held = design_of(api, TWO)
    sent = results_of(held)
    sent["more_responses"]["Cost"] = [None] * len(sent["responses"])
    error = api.error("api_balance", {**held, **sent})
    assert error == "Enter results for Cost first: the balance is found on each response's model."
