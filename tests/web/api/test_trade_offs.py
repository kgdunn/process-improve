"""What one response costs another: ``api_trade_offs``, which Learn calls once a balance is on the page.

The page tries settings on the models of the balance and draws each as a point: its settings, each response's
prediction there and the overall desirability, with a mark where a model predicts further from its runs than any
run lies. These tests hold the candidates to surfaces known here, to the limits on the settings and to the balance
itself, and the settings nearly as good as the balanced ones to a pair of responses made to have two bests.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.web._support import FACTORS, coded, held_of, run_sheet
from tests.web.api.test_balance import SURFACES, TWO, balanced, design_of, overall, results_of


def tried(api, held: dict, goals: dict | None = None, **more) -> tuple[dict, dict]:
    """Return a balance of the known surfaces and the trade-offs the page asks for at it."""
    balance = balanced(api, held, goals, **more)
    names = [r["name"] for r in held["spec"]["responses"]]
    request = {**held, **results_of(held), "models": dict.fromkeys(names, "quadratic"), "goals": goals or {}, **more}
    return balance, api.ok("api_trade_offs", {**request, "at": balance["coded"]})


def coded_settings(candidates: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return the candidates' settings of T and P, coded from the ranges the tests' factors have."""
    t = (np.asarray(candidates["settings"]["T"]) - 175) / 25
    p = np.asarray(candidates["settings"]["P"]) - 2
    return t, p


def test_each_candidate_holds_its_settings_and_what_each_surface_gives_there(api):
    held = design_of(api, TWO)
    balance, found = tried(api, held)
    candidates = found["candidates"]
    assert found["asked"] == {"drawn": 2000, "seed": 1}
    assert len(candidates["d"]) == 2000, "no limit rules a setting out"
    t, p = coded_settings(candidates)
    assert np.abs(t).max() <= 1
    assert np.abs(p).max() <= 1
    assert t.std() > 0.5, "drawn evenly from the whole range, not gathered at the balance"
    for name in ("Yield", "Cost"):  # the surfaces are quadratic, so the models fitted to them are the surfaces
        np.testing.assert_allclose(candidates["responses"][name], SURFACES[name](t, p), rtol=1e-4)
    goals = [{**r, "response": r["name"]} for r in balance["responses"]]
    np.testing.assert_allclose(candidates["d"], overall(goals, t, p), atol=1e-4)
    assert max(candidates["d"]) <= balance["composite"] + 1e-6, "no setting tried betters the balanced settings"
    assert set(candidates["far"]) == {0}, "the design's runs reach every corner: no setting lies beyond them"
    assert found["held"] == []


def test_the_candidates_keep_to_the_limits_on_the_settings(api):
    limit = {"expression": "T + 25*P <= 215"}
    held = design_of(api, TWO, design_type="d_optimal", constraints=[limit], budget=12, model="quadratic")
    _, found = tried(api, held)
    settings = found["candidates"]["settings"]
    t, p = np.asarray(settings["T"]), np.asarray(settings["P"])
    assert 0 < len(t) < 2000, "the settings the limit rules out are left out"
    assert (t + 25 * p <= 215 + 1e-6).all()


def test_a_setting_beyond_the_runs_is_marked_as_a_guess(api):
    """Runs as given that stay in half of T's range: the settings tried in the other half lean past every run."""
    runs = [[t, p] for t in (150, 160, 170) for p in (1, 2, 3)]
    spec = {"design_type": "given", "factors": FACTORS[:2], "responses": TWO, "runs": runs}
    held = held_of(api.ok("api_make_design", spec))
    results = {
        "responses": [60 + 0.1 * t + p for t, p in runs],
        "more_responses": {"Cost": [20 + 0.05 * t - p for t, p in runs]},
    }
    request = {**held, **results, "models": {"Yield": "main_effects", "Cost": "main_effects"}}
    balance = api.ok("api_balance", request)
    candidates = api.ok("api_trade_offs", {**request, "at": balance["coded"]})["candidates"]
    far = np.asarray(candidates["far"], dtype=bool)
    t = np.asarray(candidates["settings"]["T"])
    assert far.any()
    assert not far.all()
    assert t[far].mean() > t[~far].mean() + 10, "the guesses are at the high temperatures no run reached"


def test_a_categorical_factor_is_held_at_the_balanced_settings_level(api):
    factors = [*FACTORS[:2], {"name": "S", "low": "A", "high": "B", "units": ""}]
    made = api.ok("api_make_design", {"design_type": "full_factorial", "factors": factors, "responses": TWO})
    held = held_of(made)
    at = [dict(zip(held["columns"], row, strict=True)) for row in held["rows"]]
    supplier = [3.0 if r["S"] == "B" else 0.0 for r in at]  # supplier B gives more yield, at no cost
    results = {
        "responses": [60 + (r["T"] - 175) / 25 + s for r, s in zip(at, supplier, strict=True)],
        "more_responses": {"Cost": [20 + r["P"] for r in at]},
    }
    request = {**held, **results, "models": {"Yield": "main_effects", "Cost": "main_effects"}}
    balance = api.ok("api_balance", request)
    found = api.ok("api_trade_offs", {**request, "at": balance["coded"]})
    assert found["held"] == [{"name": "S", "value": "B"}]
    assert set(found["candidates"]["settings"]) == {"T", "P"}, "the factors with ranges are the ones tried"
    t = (np.asarray(found["candidates"]["settings"]["T"]) - 175) / 25
    np.testing.assert_allclose(found["candidates"]["responses"]["Yield"], 63 + t, rtol=1e-4)


def test_other_settings_nearly_as_good_are_listed_far_from_the_balanced_ones(api):
    """A yield that is high where T and P are both high or both low, and a cost that hardly moves: two bests."""
    made = api.ok("api_make_design", {"design_type": "full_factorial", "factors": FACTORS[:2], "responses": TWO})
    held = held_of(made)
    at = [dict(zip(held["columns"], row, strict=True)) for row in held["rows"]]
    t, p = [(r["T"] - 175) / 25 for r in at], [r["P"] - 2 for r in at]
    results = {
        "responses": [60 + 5 * a * b for a, b in zip(t, p, strict=True)],
        "more_responses": {"Cost": [20 + 0.2 * a + 0.2 * b for a, b in zip(t, p, strict=True)]},
    }
    request = {**held, **results, "models": dict.fromkeys(("Yield", "Cost"), "interactions")}
    request["goals"] = {"Cost": {"high": 30.0}}
    balance = api.ok("api_balance", request)
    assert balance["confirm_at"] == {"T": 150.0, "P": 1.0}, "the cheaper of the two corners"
    (other,) = api.ok("api_trade_offs", {**request, "at": balance["coded"]})["alternatives"]
    assert other["settings"] == pytest.approx({"T": 200.0, "P": 3.0}, abs=1e-3), "the corner across the range"
    assert other["responses"] == pytest.approx({"Yield": 65.0, "Cost": 20.4}, abs=1e-3)
    assert 0.95 * balance["composite"] <= other["composite"] < balance["composite"]
    assert other["d"]["Yield"] == pytest.approx(1.0, abs=1e-6)


def test_a_balance_with_one_best_lists_no_other_settings(api):
    _, found = tried(api, design_of(api, TWO))
    assert found["alternatives"] == []


def test_settings_that_differ_only_in_a_factor_no_model_holds_are_not_other_settings(api):
    """A search stops anywhere along a factor that changes nothing: the same settings, not settings nearly as good."""
    held = held_of(api.ok("api_make_design", {"design_type": "box_behnken", "factors": FACTORS, "responses": TWO}))
    x = coded(run_sheet(held), FACTORS)
    results = {
        "responses": (60 + 4 * x["T"] + 2 * x["P"] - 3 * x["T"] ** 2).tolist(),
        "more_responses": {"Cost": (20 + 3 * x["T"] + x["P"]).tolist()},
    }
    without_f = {"terms": [["T"], ["P"], ["T", "T"]]}
    request = {**held, **results, "models": {"Yield": without_f, "Cost": without_f}}
    balance = api.ok("api_balance", request)
    assert not next(s for s in balance["settings"] if s["name"] == "F")["in_model"]
    assert api.ok("api_trade_offs", {**request, "at": balance["coded"]})["alternatives"] == []
