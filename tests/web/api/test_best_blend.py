"""A mixture's best blend: ``api_best_settings`` for a study whose factors are the components of a blend.

The search keeps to the blends: each component between its bounds, and the proportions summing to
1. These tests hold the best blend to what can be checked another way: a dense lattice of blends
over the same Scheffé model, the intervals ``analyze_experiment`` gives at the same blend, and the
model's own predictions along each component's Cox direction.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import analyze_experiment
from tests.web._support import held_of, run_sheet
from tests.web.api.test_balance import desirability

#: Where the confirmation runs are marked on the run sheet, and how: as ``web/app.js`` adds them.
CONFIRMATION = "Confirmation run"

#: A blend of three components with a synergy between A and B and an antagonism between B and C: its highest
#: response is on the edge without C, at A = 2/3, and its lowest on the edge without A, at B = 7/24.
SURFACE = {"A": 30.0, "B": 25.0, "C": 20.0, "A:B": 15.0, "B:C": -12.0}

#: Three components that blend better than any of them alone, and alike: the centroid gives the most.
CENTRED = {"A": 10.0, "B": 10.0, "C": 10.0, "A:B": 30.0, "A:C": 30.0, "B:C": 30.0}

#: Bounds that cut the triangle down to a hexagon, as extreme vertices need.
BOUNDS = {"A": (0.2, 0.7), "B": (0.1, 0.6), "C": (0.0, 0.5)}


# --------------------------------------------------------------------------- helpers
def components(names: str = "ABC", bounds: dict[str, tuple[float, float]] | None = None) -> list[dict]:
    """Return a mixture's components as the brief sends them: blank bounds, 0 and 1, unless ``bounds`` gives some."""
    bounds = bounds or {}
    return [{"name": n, "low": bounds.get(n, (None,))[0], "high": bounds.get(n, (None, None))[1]} for n in names]


def lattice(api, q: int = 3, **change: object) -> dict:
    """Make a simplex lattice of degree 2, its centroid added, in ``q`` components, as the page holds it."""
    request = {"design_type": "simplex_lattice", "factors": components("ABCDEF"[:q]), "lattice_degree": 2, **change}
    return held_of(api.ok("api_make_design", request))


def vertices(api, **change: object) -> dict:
    """Make extreme vertices for the bounded components, for the quadratic Scheffé model."""
    request = {"design_type": "extreme_vertices", "factors": components(bounds=BOUNDS), **change}
    return held_of(api.ok("api_make_design", {"mixture_model": "scheffe_quadratic", **request}))


def blends_of(held: dict) -> pd.DataFrame:
    """Return the run sheet's blends, a column per component."""
    return run_sheet(held)[[f["name"] for f in held["spec"]["factors"]]].astype(float)


def blended(blends: pd.DataFrame, terms: dict[str, float]) -> np.ndarray:
    """Return a Scheffé model's response at each blend."""
    return sum(value * blends[term.split(":")].prod(axis=1) for term, value in terms.items()).to_numpy()


def results(held: dict, terms: dict[str, float], noise: float = 0.05, seed: int = 0) -> list[float]:
    """Return each run's result from a known blend, plus a little noise."""
    y = blended(blends_of(held), terms)
    return (y + np.random.default_rng(seed).normal(0.0, noise, len(y))).tolist()


def aimed(held: dict, goal: str, target: float | None = None) -> dict:
    """Give the run sheet's response a goal, as the brief does."""
    info = {"label": "Hardness", "units": "HV", "goal": goal, **({"target": target} if target is not None else {})}
    return {**held, "spec": {**held["spec"], "response_info": info}}


def best_of(api, held: dict, responses: list, goal: str = "maximize", target: float | None = None) -> dict:
    """Return the best blend the page finds for a goal."""
    return api.ok("api_best_settings", {**aimed(held, goal, target), "responses": responses})


def fitted_terms(api, held: dict, responses: list) -> dict[str, float]:
    """Return the Scheffé model the page fits, as its analysis shows it."""
    analysis = api.ok("api_analyze_runs", {**held, "responses": responses})
    return {c["term"]: c["coefficient"] for c in analysis["coefficients"]}


def simplex_grid(q: int, steps: int) -> pd.DataFrame:
    """Return every blend of ``q`` components whose proportions are whole multiples of ``1 / steps``."""
    names = list("ABCDEF"[:q])
    points = [p for p in itertools.product(range(steps + 1), repeat=q - 1) if sum(p) <= steps]
    return pd.DataFrame([[*p, steps - sum(p)] for p in points], columns=names) / steps


def blend_of(best: dict) -> np.ndarray:
    """Return the best blend's proportions, in the components' order."""
    return np.array([s["value"] for s in best["settings"]])


# --------------------------------------------------------------------------- the search
@pytest.mark.parametrize(("goal", "sign"), [("maximize", 1.0), ("minimize", -1.0)])
def test_the_best_blend_is_the_best_of_a_dense_lattice_of_blends(api, goal, sign):
    held = lattice(api)
    responses = results(held, SURFACE)
    best = best_of(api, held, responses, goal)
    terms = fitted_terms(api, held, responses)
    grid = simplex_grid(3, 300)
    predicted = blended(grid, terms)
    top = int(np.argmax(sign * predicted))
    assert sign * best["predicted"] >= sign * predicted[top] - 1e-9, "no blend of the lattice does better"
    np.testing.assert_allclose(blend_of(best), grid.iloc[top], atol=2 / 300)
    assert best["predicted"] == pytest.approx(blended(pd.DataFrame([best["coded"]]), terms)[0], abs=1e-9)
    assert best["coded"] == {s["name"]: s["value"] for s in best["settings"]}, "the blend, as the model is fitted"


def test_the_best_blend_on_an_edge_is_at_the_lower_bound_of_the_component_it_leaves_out(api):
    held = lattice(api)
    best = best_of(api, held, results(held, SURFACE))
    settings = {s["name"]: s for s in best["settings"]}
    assert [s["kind"] for s in best["settings"]] == ["component"] * 3
    assert settings["C"]["value"] == 0.0
    assert settings["C"]["end"] == "low"
    assert not settings["C"]["beyond"], "no proportion is below 0"
    assert settings["A"]["value"] == pytest.approx(2 / 3, abs=0.01), "the synergy's peak along the edge"
    assert settings["A"]["end"] is None
    assert settings["B"]["end"] is None
    assert best["ascent"] is None, "no way to move on from a blend"
    assert best["stationary"] is None
    assert best["limits"] == []


def test_a_best_blend_inside_every_bound_is_found_where_the_blending_peaks(api):
    held = lattice(api)
    best = best_of(api, held, results(held, CENTRED, noise=0.01))
    np.testing.assert_allclose(blend_of(best), [1 / 3] * 3, atol=0.01)
    assert all(s["end"] is None for s in best["settings"])


@pytest.mark.parametrize("q", [4, 5])
def test_the_best_blend_of_more_components_sums_to_1_within_its_bounds(api, q):
    held = lattice(api, q)
    terms = {n: 20.0 + 3 * i for i, n in enumerate("ABCDEF"[:q])} | {"A:B": 25.0, "C:D": -20.0}
    responses = results(held, terms)
    best = best_of(api, held, responses)
    x = blend_of(best)
    assert x.sum() == pytest.approx(1.0, abs=1e-9)
    assert (x >= -1e-12).all()
    fitted = fitted_terms(api, held, responses)
    grid = simplex_grid(q, 40)
    assert best["predicted"] >= blended(grid, fitted).max() - 1e-9


def test_the_best_blend_keeps_to_the_bounds_the_reader_sets(api):
    held = vertices(api)
    terms = {"A": 40.0, "B": 20.0, "C": 10.0}  # more A is better, then more B: A stops at its upper bound
    responses = results(held, terms)
    best = best_of(api, held, responses)
    x = blend_of(best)
    assert x.sum() == pytest.approx(1.0, abs=1e-9)
    for s, (low, high) in zip(best["settings"], BOUNDS.values(), strict=True):
        assert low - 1e-12 <= s["value"] <= high + 1e-12, s["name"]
        assert (s["low"], s["high"]) == (low, high)
    settings = {s["name"]: s for s in best["settings"]}
    assert settings["A"]["value"] == pytest.approx(0.7, abs=1e-9)
    assert settings["A"]["end"] == "high"
    assert settings["A"]["beyond"], "more A, for less B, predicts a higher Hardness: the bound stands in the way"
    assert settings["C"]["end"] == "low"
    assert not settings["C"]["beyond"], "C's bound is 0: no proportion passes it"
    assert settings["B"]["end"] is None
    grid = simplex_grid(3, 200)
    inside = np.all([(grid[n] >= lo) & (grid[n] <= hi) for n, (lo, hi) in BOUNDS.items()], axis=0)
    assert best["predicted"] >= blended(grid[inside], fitted_terms(api, held, responses)).max() - 1e-9


def test_a_target_the_blends_reach_is_reached(api):
    held = lattice(api)
    best = best_of(api, held, results(held, SURFACE), "target", 27.0)
    assert best["reached"] is True
    assert best["predicted"] == pytest.approx(27.0, abs=1e-3)
    assert not any(s["beyond"] for s in best["settings"])


def test_a_target_beyond_every_blend_comes_as_close_as_a_blend_can(api):
    held = lattice(api)
    best = best_of(api, held, results(held, SURFACE), "target", 40.0)
    assert best["reached"] is False
    highest = best_of(api, held, results(held, SURFACE))
    assert best["predicted"] == pytest.approx(highest["predicted"], abs=1e-6), "the highest blend is the nearest"


def test_a_model_no_blend_changes_finds_no_best_blend(api):
    held = lattice(api)
    best = best_of(api, held, [5.0] * len(held["rows"]))
    assert best == {"goal": "maximize", "target": None, "flat": True, "confirmation": []}


def test_a_goal_of_learning_asks_for_no_blend(api):
    held = lattice(api)
    assert best_of(api, held, results(held, SURFACE), "learn") == {"goal": "learn"}


# --------------------------------------------------------------------------- what is known at the best blend
def test_the_intervals_and_leverage_are_the_scheffe_fit_s_at_the_blend(api):
    held = lattice(api, n_replicates=2)
    responses = results(held, SURFACE, noise=0.5, seed=3)
    best = best_of(api, held, responses)
    blends = blends_of(held)
    y = pd.Series(responses, name="y")
    at = pd.DataFrame([best["coded"]])
    shown = analyze_experiment(
        blends, y, model="scheffe_quadratic", analysis_type="prediction", coding="actual", new_points=at
    )["predictions"][0]
    assert best["predicted"] == pytest.approx(shown["predicted"], abs=1e-9)
    assert best["confidence"] == pytest.approx([shown["ci_low"], shown["ci_high"]], abs=1e-9)
    assert best["prediction"] == pytest.approx([shown["pi_low"], shown["pi_high"]], abs=1e-9)
    x = np.column_stack([blends[t.split(":")].prod(axis=1) for t in ["A", "B", "C", "A:B", "A:C", "B:C"]])
    row = np.array([at[t.split(":")].prod(axis=1).iloc[0] for t in ["A", "B", "C", "A:B", "A:C", "B:C"]])
    inverse = np.linalg.inv(x.T @ x)
    assert best["leverage"] == pytest.approx(float(row @ inverse @ row), rel=1e-9)
    assert best["max_leverage"] == pytest.approx(float(np.einsum("ij,jk,ik->i", x, inverse, x).max()), rel=1e-9)
    assert best["extrapolating"] is False, "on the edge the lattice's runs lie along"


def test_the_sensitivity_moves_each_component_along_its_cox_direction(api):
    held = lattice(api)
    responses = results(held, SURFACE)
    best = best_of(api, held, responses)
    terms = fitted_terms(api, held, responses)
    x = blend_of(best)
    for i, s in enumerate(best["sensitivity"]):
        assert s["step"] == pytest.approx(0.05), "a twentieth of the component's range, 0 to 1"
        for sign, change, runnable in ((-1, s["down"], s["down_runnable"]), (1, s["up"], s["up_runnable"])):
            moved = x * (1 - x[i] - sign * s["step"]) / (1 - x[i])  # the others keep their proportions to each other
            moved[i] = x[i] + sign * s["step"]
            assert moved.sum() == pytest.approx(1.0)
            expected = blended(pd.DataFrame([moved], columns=list("ABC")), terms)[0] - best["predicted"]
            assert change == pytest.approx(expected, abs=1e-9), (s["name"], sign)
            assert runnable == bool((moved >= -1e-12).all() and (moved <= 1 + 1e-12).all()), (s["name"], sign)
    by_name = {s["name"]: s for s in best["sensitivity"]}
    assert not by_name["C"]["down_runnable"], "C is at 0 already"
    assert by_name["C"]["up_runnable"]


def test_a_pure_component_s_blend_shares_its_change_among_the_others(bootstrap):
    moved = bootstrap._cox_moved(np.array([1.0, 0.0, 0.0]), 0, -0.1)
    np.testing.assert_allclose(moved, [0.9, 0.05, 0.05])
    moved = bootstrap._cox_moved(np.array([0.5, 0.3, 0.2]), 0, 0.1)
    np.testing.assert_allclose(moved, [0.6, 0.24, 0.16])


# --------------------------------------------------------------------------- the confirmation runs
def test_the_blend_is_weighed_out_to_a_thousandth_and_still_sums_to_1(api):
    held = lattice(api)
    best = best_of(api, held, results(held, CENTRED, noise=0.01))
    at = best["confirm_at"]
    thousandths = {name: round(value * 1000) for name, value in at.items()}
    assert sum(thousandths.values()) == 1000, "the proportions sum to 1, as decimals"
    for name, value in at.items():
        assert value == thousandths[name] / 1000, name
    assert best["rounding"]["decimals"] == 3
    adjusted = best["rounding"]["adjusted"]
    assert adjusted in at, "a third each rounds to 0.333 three times, 0.999 in all: one component takes the rest"
    assert at[adjusted] == 0.334


def test_rounding_keeps_each_component_within_its_bounds(bootstrap):
    spec = {"factors": [{"name": n, "low": lo, "high": hi, "units": ""} for n, (lo, hi) in BOUNDS.items()]}
    at, rounding = bootstrap._weighed(spec, np.array([0.7, 0.2996, 0.0004]))
    assert at == {"A": 0.7, "B": 0.3, "C": 0.0}
    assert rounding == {"decimals": 3, "adjusted": None}
    at, rounding = bootstrap._weighed(spec, np.array([0.7, 0.29955, 0.00045]))  # 0.7 + 0.300 + 0.000: A is full
    assert sum(round(v * 1000) for v in at.values()) == 1000
    assert at["A"] <= 0.7
    assert rounding["adjusted"] in {"B", "C", None}


def test_confirmation_runs_at_the_blend_are_judged_against_the_scheffe_model(api):
    held = lattice(api)
    responses = results(held, SURFACE)
    best = best_of(api, held, responses)
    at = best["confirm_at"]
    n = len(held["rows"])
    columns = [*held["columns"], CONFIRMATION]
    extra = [[{"Run order": n + 1 + i, CONFIRMATION: "yes", **at}.get(c) for c in columns] for i in range(3)]
    sheet = {**held, "columns": columns, "rows": [*[[*row, None] for row in held["rows"]], *extra]}
    predicted = blended(pd.DataFrame([at]), fitted_terms(api, held, responses))[0]
    again = best_of(api, sheet, [*responses, predicted + 0.01, predicted - 0.02, None])
    assert again["predicted"] == pytest.approx(best["predicted"], abs=1e-9), "the confirmation runs are not fitted"
    [group] = again["confirmation"]
    assert group["rows"] == [n, n + 1, n + 2]
    assert group["at"] == at
    assert (group["runs"], group["done"]) == (3, 2)
    assert group["predicted"] == pytest.approx(predicted, abs=1e-9), "predicted at the blend as weighed out"
    assert group["mean"] == pytest.approx(predicted - 0.005)
    assert group["inside"] is True


# --------------------------------------------------------------------------- the charts drawn through the best blend
@pytest.fixture
def plotly():
    return pytest.importorskip("plotly", reason="the page installs plotly for its first chart")


def test_the_triangle_is_drawn_as_before_and_the_page_stars_the_best_blend(api, plotly):
    held = lattice(api)
    responses = results(held, SURFACE)
    best = best_of(api, held, responses)
    plain = api.ok("api_plots", {**held, "responses": responses})
    starred = api.ok("api_plots", {**held, "responses": responses, "best": best["coded"]})
    assert (plain["held_best"], starred["held_best"]) == (False, True)
    assert starred["ternary"] == plain["ternary"], "the triangle shows every blend: the star is all that is added"


def test_four_components_are_traced_through_the_best_blend(api, plotly):
    held = lattice(api, 4)
    terms = {"A": 20.0, "B": 23.0, "C": 26.0, "D": 29.0, "A:B": 25.0, "C:D": -20.0, "A:D": 30.0}
    responses = results(held, terms)
    best = best_of(api, held, responses)
    plain = api.ok("api_plots", {**held, "responses": responses})["cox"]
    cox = api.ok("api_plots", {**held, "responses": responses, "best": best["coded"]})["cox"]
    assert "through" not in plain, "the reference blend, the centroid of the design's blends"
    assert cox["through"] == "best"
    assert cox["reference"] == best["coded"]
    x = blend_of(best)
    fitted = fitted_terms(api, held, responses)
    for i, trace in enumerate(cox["traces"]):
        at = trace["change"].index(0.0)
        assert trace["fitted"][at] == pytest.approx(best["predicted"], rel=1e-5), "the traces cross at the best blend"
        assert max(trace["fitted"]) <= best["predicted"] * (1 + 1e-5), "no blend along a trace does better"
        for change, value in zip(trace["change"], trace["fitted"], strict=True):
            moved = x * (1 - x[i] - change) / (1 - x[i])
            moved[i] = x[i] + change
            assert (moved >= -1e-6).all(), (trace["component"], change)
            assert value == pytest.approx(blended(pd.DataFrame([moved], columns=list("ABCD")), fitted)[0], rel=1e-5)


def test_a_blend_that_is_no_blend_of_the_components_is_refused(api, plotly):
    held = lattice(api)
    responses = results(held, SURFACE)
    for best in ({"A": 0.5, "B": 0.6, "C": 0.0}, {"A": 1.2, "B": -0.2, "C": 0.0}, {"A": 1.0, "B": 0.0}, ["A"]):
        sent = {**held, "responses": responses, "best": best}
        assert api.error("api_plots", sent) == "The page sent a best blend this page did not write.", best


# --------------------------------------------------------------------------- the balance of several responses
#: A cost to minimise beside the hardness of SURFACE: B is the cheapest component, and A with C costs less together.
COST = {"A": 20.0, "B": 10.0, "C": 15.0, "A:B": 8.0, "A:C": -6.0}

#: The two responses, as the brief describes them.
TWO = [{"name": "Hardness", "goal": "maximize", "units": "HV"}, {"name": "Cost", "goal": "minimize"}]


def two_blended(api, q: int = 3) -> tuple[dict, dict]:
    """Make a simplex lattice measuring a hardness and a cost; return it and the request with both results."""
    held = lattice(api, q, responses=TWO)
    hardness = {t: v for t, v in SURFACE.items() if all(n in "ABCDEF"[:q] for n in t.split(":"))}
    return held, {
        **held,
        "responses": results(held, hardness | ({"D": 22.0} if q > 3 else {})),
        "more_responses": {"Cost": results(held, COST | ({"D": 12.0} if q > 3 else {}), seed=1)},
    }


def terms_of(api, sent: dict, response: str) -> dict[str, float]:
    """Return the Scheffé model the page fits to one response."""
    analysis = api.ok("api_analyze_runs", {**sent, "response": response})
    return {c["term"]: c["coefficient"] for c in analysis["coefficients"]}


def overall_d(goals: list[dict], predictions: dict[str, np.ndarray]) -> np.ndarray:
    """Return D, the importance-weighted geometric mean of the d's, from the desirability's own definition."""
    weights = np.array([g["importance"] for g in goals], dtype=float)
    ds = np.array([desirability(predictions[g["response"]], g) for g in goals])
    with np.errstate(divide="ignore"):
        return np.exp((weights[:, None] * np.log(ds)).sum(axis=0) / weights.sum())


def test_the_balanced_blend_has_the_best_overall_desirability_of_a_dense_lattice(api):
    _, sent = two_blended(api)
    b = api.ok("api_balance", sent)
    assert b["balanced"] is True
    goals = b["search"]["goals"]
    fitted = {name: terms_of(api, sent, name) for name in ("Hardness", "Cost")}
    grid = simplex_grid(3, 300)
    d = overall_d(goals, {name: blended(grid, terms) for name, terms in fitted.items()})
    assert b["composite"] >= d.max() - 1e-9, "no blend of the lattice balances them better"
    np.testing.assert_allclose(blend_of(b), grid.iloc[int(np.argmax(d))], atol=0.02)
    at = pd.DataFrame([b["coded"]])
    for r in b["responses"]:
        assert r["predicted"] == pytest.approx(blended(at, fitted[r["name"]])[0], abs=1e-9)
    assert b["composite"] == pytest.approx(
        overall_d(goals, {r["name"]: np.array([r["predicted"]]) for r in b["responses"]})[0]
    )
    assert [s["kind"] for s in b["settings"]] == ["component"] * 3
    assert not any(s["beyond"] for s in b["settings"]), "no way to move on is said of a balance"
    assert b["limits"] == []
    assert b["search"]["formulas"] == {"Hardness": "scheffe_quadratic", "Cost": "scheffe_quadratic"}
    assert sum(round(v * 1000) for v in b["confirm_at"].values()) == 1000
    assert b["rounding"]["decimals"] == 3


def test_the_overlay_shades_the_blends_where_every_response_is_acceptable(api):
    _, sent = two_blended(api)
    b = api.ok("api_balance", sent)
    overlay = b["overlay"]
    assert (overlay["kind"], overlay["components"]) == ("ternary", ["A", "B", "C"])
    grid = pd.DataFrame({"A": overlay["a"], "B": overlay["b"], "C": overlay["c"]})
    np.testing.assert_allclose(grid.sum(axis=1), 1.0)
    goals = {g["response"]: g for g in b["search"]["goals"]}
    acceptable = []
    for surface in overlay["responses"]:
        terms = terms_of(api, sent, surface["name"])
        z = blended(grid, terms)
        np.testing.assert_allclose(surface["z"], z, rtol=1e-5)
        goal = goals[surface["name"]]
        acceptable.append(z > goal["low"] if goal["goal"] == "maximize" else z < goal["high"])
        [level] = surface["levels"]
        assert level == (goal["low"] if goal["goal"] == "maximize" else goal["high"])
        [line] = surface["lines"]
        ends = pd.DataFrame(
            {name: [v for v in line[axis] if v is not None] for axis, name in zip("abc", "ABC", strict=True)}
        )
        if z.min() < level < z.max():
            assert len(ends) >= 2, "the boundary crosses the triangle"
            np.testing.assert_allclose(blended(ends, terms), level, atol=0.01 * (z.max() - z.min()))
        else:
            assert ends.empty, "a boundary the blends do not reach is not drawn"
        assert len(line["a"]) == len(line["b"]) == len(line["c"])
    np.testing.assert_array_equal(overlay["acceptable"], np.all(acceptable, axis=0).astype(int))
    assert overlay["at"] == pytest.approx(blend_of(b).tolist())


def test_four_components_are_balanced_without_an_overlay(api):
    _, sent = two_blended(api, 4)
    b = api.ok("api_balance", sent)
    assert b["balanced"] is True
    assert b["overlay"] is None, "four components make a solid, which no triangle holds"
    assert sum(s["value"] for s in b["settings"]) == pytest.approx(1.0)


def test_confirmation_runs_at_the_balanced_blend_are_judged_response_by_response(api):
    held, sent = two_blended(api)
    b = api.ok("api_balance", sent)
    at = b["confirm_at"]
    n = len(held["rows"])
    columns = [*held["columns"], CONFIRMATION]
    extra = [[{"Run order": n + 1 + i, CONFIRMATION: "yes", **at}.get(c) for c in columns] for i in range(3)]
    fitted = {name: terms_of(api, sent, name) for name in ("Hardness", "Cost")}
    expected = {name: blended(pd.DataFrame([at]), terms)[0] for name, terms in fitted.items()}
    again = api.ok(
        "api_balance",
        {
            **sent,
            "columns": columns,
            "rows": [*[[*row, None] for row in held["rows"]], *extra],
            "responses": [*sent["responses"], expected["Hardness"], expected["Hardness"], None],
            "more_responses": {"Cost": [*sent["more_responses"]["Cost"], expected["Cost"], None, None]},
        },
    )
    assert again["composite"] == pytest.approx(b["composite"]), "the confirmation runs are not fitted"
    [group] = again["confirmation"]
    assert group["at"] == at
    verdicts = {v["name"]: v for v in group["responses"]}
    assert (verdicts["Hardness"]["done"], verdicts["Cost"]["done"]) == (2, 1)
    for name, value in expected.items():
        assert verdicts[name]["predicted"] == pytest.approx(value, abs=1e-9), name
        assert verdicts[name]["inside"] is True, name
