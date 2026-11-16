"""A mixture's target and its specification: ``api_target_settings`` and ``api_specification`` for a blend.

A study whose factors are the components of a blend is asked the same two questions as one of factors with
ranges: which blend gives a target, and which blends meet a specification, and how surely. The blends tried
are drawn evenly from the region of blends, each component between its bounds and the proportions summing to
1. These tests hold the answers to what can be checked another way: a dense lattice of blends over the same
Scheffé model, and the predictive distribution worked out here by least squares.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from tests.web._support import held_of
from tests.web.api.test_best_blend import BOUNDS, SURFACE, blended, blends_of, components, lattice, results, vertices

#: A second response of the same blends: a cost that rises with A and falls with C, with no blending term.
COST = {"A": 12.0, "B": 8.0, "C": 5.0}


def dense(step: int = 200, bounds: dict[str, tuple[float, float]] | None = None) -> pd.DataFrame:
    """Return every blend of three components on a lattice of ``step`` steps a side, inside ``bounds``."""
    places = [(i, j, step - i - j) for i, j in itertools.product(range(step + 1), repeat=2) if i + j <= step]
    blends = pd.DataFrame(np.array(places, dtype=float) / step, columns=list("ABC"))
    for name, (low, high) in (bounds or {}).items():
        blends = blends[(blends[name] >= low) & (blends[name] <= high)]
    return blends.reset_index(drop=True)


def asked(held: dict, noise: float = 0.2, **more) -> dict:
    """Return a request about a blend whose response follows the known surface, with a little noise."""
    return {**held, "responses": results(held, SURFACE, noise=noise), **more}


def two_responses(api) -> dict:
    """Return extreme vertices measuring a hardness and a cost, each from its known surface, their noise shared."""
    responses = [{"name": "Hardness", "goal": "maximize"}, {"name": "Cost", "goal": "minimize"}]
    request = {"design_type": "extreme_vertices", "factors": components(bounds=BOUNDS), "responses": responses}
    held = held_of(api.ok("api_make_design", {"mixture_model": "scheffe_quadratic", **request}))
    blends = blends_of(held)
    shared = np.random.default_rng(3).normal(0.0, 0.3, len(blends))
    hardness, cost = blended(blends, SURFACE) + shared, blended(blends, COST) + shared
    return {**held, "responses": hardness.tolist(), "more_responses": {"Cost": cost.tolist()}}


# --------------------------------------------------------------------------- the blend that gives a target
def test_a_target_is_met_by_a_blend_near_the_middle_of_the_triangle(api):
    held = lattice(api)
    found = api.ok("api_target_settings", asked(held, noise=0.0, targets={"y": 27.0}))
    assert found["reached"]
    blend = np.array([s["value"] for s in found["settings"]])
    assert [s["kind"] for s in found["settings"]] == ["component"] * 3
    assert blend.sum() == pytest.approx(1.0, abs=1e-9)
    assert ((blend >= 0) & (blend <= 1)).all()
    surface = lambda blends: blended(blends, SURFACE)  # noqa: E731 - the model is the surface, fitted without noise
    assert surface(pd.DataFrame([blend], columns=list("ABC")))[0] == pytest.approx(27.0, abs=1e-4)
    assert found["responses"][0]["predicted"] == pytest.approx(27.0, abs=1e-4)
    # Of the blends that give the target, it is the one nearest the middle: no blend of a dense lattice within
    # reach of the target lies nearer the centroid by more than the lattice's own step.
    every = dense()
    reach = found["near"] * np.ptp(surface(blends_of(held)))
    giving = every[np.abs(surface(every) - 27.0) <= reach]
    nearest = np.sqrt(((giving - 1 / 3) ** 2).sum(axis=1)).min()
    assert np.sqrt(((blend - 1 / 3) ** 2).sum()) <= nearest + 0.02
    assert found["share"] == pytest.approx(len(giving) / len(every), abs=0.01), "the blends are drawn evenly"
    assert found["tried"] == 20_000
    assert sum(found["confirm_at"].values()) == pytest.approx(1.0, abs=1e-12), "weighed out, the blend still sums to 1"
    assert found["rounding"]["decimals"] == 3


def test_a_target_out_of_reach_gives_the_blend_that_comes_closest(api):
    held = lattice(api)
    found = api.ok("api_target_settings", asked(held, noise=0.0, targets={"y": 40.0}))
    assert not found["reached"]
    highest = blended(dense(), SURFACE).max()  # 32.5, on the edge without C
    assert found["responses"][0]["predicted"] == pytest.approx(highest, abs=0.05)
    assert not found["responses"][0]["given"]


def test_a_target_keeps_to_the_bounds_on_the_components(api):
    held = vertices(api)
    found = api.ok("api_target_settings", asked(held, targets={"y": 29.5}))
    assert found["reached"]
    for setting in found["settings"]:
        low, high = BOUNDS[setting["name"]]
        assert low - 1e-9 <= setting["value"] <= high + 1e-9, setting["name"]
        assert (setting["low"], setting["high"]) == (low, high)


def test_two_targets_are_met_by_one_blend(api):
    request = two_responses(api)
    found = api.ok("api_target_settings", {**request, "targets": {"Hardness": 28.5, "Cost": 9.0}})
    assert found["reached"]
    assert [(r["name"], r["given"]) for r in found["responses"]] == [("Hardness", True), ("Cost", True)]
    assert [r["predicted"] for r in found["responses"]] == pytest.approx([28.5, 9.0], abs=1e-3)
    assert found["leverage_of"] in ("Hardness", "Cost")


def test_the_target_s_chart_is_drawn_over_the_triangle_with_the_line_of_the_target(api):
    held = lattice(api)
    found = api.ok("api_target_settings", asked(held, targets={"y": 27.0}))
    overlay = found["overlay"]
    assert (overlay["kind"], overlay["components"]) == ("ternary", ["A", "B", "C"])
    assert overlay["acceptable"] is None, "a response's reach is thinner than the grid: nothing is shaded"
    (surface,) = overlay["responses"]
    assert (surface["name"], surface["levels"]) == ("y", [27.0])
    (line,) = surface["lines"]
    on_the_line = pd.DataFrame({name: [v for v in line[name.lower()] if v is not None] for name in "ABC"})
    fitted = found["responses"][0]["predicted"] - 27.0  # the model is a little off the surface: the noise
    assert np.abs(blended(on_the_line, SURFACE) - 27.0).max() < 0.6 + abs(fitted)
    assert overlay["at"] == pytest.approx([s["value"] for s in found["settings"]])
    assert sorted(map(tuple, overlay["runs"])) == sorted(map(tuple, blends_of(held).drop_duplicates().to_numpy()))


def test_four_components_have_a_blend_for_a_target_and_no_triangle(api):
    held = lattice(api, q=4)
    blends = blends_of(held)
    y = 20 + 10 * blends["A"] - 5 * blends["D"] + 8 * blends["A"] * blends["B"]
    found = api.ok("api_target_settings", {**held, "responses": y.tolist(), "targets": {"y": 24.0}})
    assert found["reached"]
    assert found["overlay"] is None
    assert sum(s["value"] for s in found["settings"]) == pytest.approx(1.0, abs=1e-9)
    judged = api.ok("api_specification", {**held, "responses": y.tolist(), "windows": {"y": [24.0, None]}})
    assert judged["overlay"] is None
    assert 0 < judged["share"] < 1


# --------------------------------------------------------------------------- the blends that meet a specification
def test_the_share_of_the_blends_that_meets_a_window_is_that_of_the_known_surface(api):
    held = lattice(api)
    judged = api.ok("api_specification", asked(held, noise=0.0, windows={"y": [28.0, None]}))
    every = dense()
    assert judged["share"] == pytest.approx((blended(every, SURFACE) >= 28.0).mean(), abs=0.01)
    assert judged["tried"] == 20_000
    assert (judged["held"], judged["levels"]) == ([], []), "a mixture has no level to hold"
    assert judged["responses"] == [{"name": "y", "label": "y", "units": "", "low": 28.0, "high": None}]
    assert judged["search"]["at"] == {}, "and no what-if panel to judge through"


def test_the_share_keeps_to_the_bounds_on_the_components(api):
    held = vertices(api)
    judged = api.ok("api_specification", asked(held, windows={"y": [29.0, None]}))
    inside = dense(bounds=BOUNDS)
    assert judged["share"] == pytest.approx((blended(inside, SURFACE) >= 29.0).mean(), abs=0.03)
    blend = judged["surest"]["settings"]
    assert all(BOUNDS[name][0] - 1e-9 <= value <= BOUNDS[name][1] + 1e-9 for name, value in blend.items())


def test_the_chance_at_the_surest_blend_is_the_share_of_a_new_run_s_distribution_inside_the_window(api):
    held = vertices(api)
    request = asked(held, noise=0.4, windows={"y": [29.0, 31.5]})
    judged = api.ok("api_specification", request)
    surest = judged["surest"]
    # The quadratic Scheffé model by least squares, and a new run about its prediction at the surest blend.
    terms = ["A", "B", "C", "A:B", "A:C", "B:C"]
    columns = lambda blends: np.column_stack([blends[t.split(":")].prod(axis=1) for t in terms])  # noqa: E731
    x, y = columns(blends_of(held)), np.asarray(request["responses"])
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    df = len(y) - len(terms)
    mse = float(((y - x @ beta) ** 2).sum() / df)
    row = columns(pd.DataFrame([surest["settings"]]))[0]
    predicted = float(row @ beta)
    scatter = np.sqrt(mse * (1 + row @ np.linalg.inv(x.T @ x) @ row))
    chance = stats.t.cdf((31.5 - predicted) / scatter, df) - stats.t.cdf((29.0 - predicted) / scatter, df)
    assert surest["responses"]["y"] == pytest.approx(predicted, abs=1e-9)
    assert surest["probability"] == pytest.approx(chance, abs=1e-9)
    assert sum(surest["confirm_at"].values()) == pytest.approx(1.0, abs=1e-12)
    assert 0 < judged["likely"]["0.9"] <= judged["likely"]["0.5"] <= judged["share"] < 1


def test_the_specification_s_chart_shades_the_blends_that_meet_it_and_draws_each_chance(api):
    held = vertices(api)
    judged = api.ok("api_specification", asked(held, noise=0.4, windows={"y": [29.0, 31.5]}))
    overlay = judged["overlay"]
    assert overlay["kind"] == "ternary"
    grid = pd.DataFrame({name: overlay[name.lower()] for name in "ABC"})
    (surface,) = overlay["responses"]
    z = np.asarray(surface["z"])
    assert overlay["acceptable"] == ((z >= 29.0) & (z <= 31.5)).astype(int).tolist()
    assert surface["levels"] == [29.0, 31.5]
    assert len(surface["lines"]) == 2, "a line at each end of the window"
    assert [line["chance"] for line in overlay["chances"]] == [0.5, 0.9]
    assert all(any(v is not None for v in line["a"]) for line in overlay["chances"]), "both chances are reached"
    assert overlay["at"] == pytest.approx(list(judged["surest"]["settings"].values()))
    assert len(overlay["runs"]) == len(blends_of(held).drop_duplicates())
    assert len(grid) == len(z)


def test_a_fit_with_no_run_left_for_the_noise_gives_the_share_and_no_chance(api):
    held = held_of(api.ok("api_make_design", {"design_type": "simplex_lattice", "factors": components()}))
    request = {**held, "responses": blended(blends_of(held), SURFACE).tolist(), "windows": {"y": [28.0, None]}}
    if len(held["rows"]) > 6:  # the centroid, or a replicate, leaves a run for the noise: fit the model that takes it
        request["models"] = {"y": "scheffe_special_cubic"}
    judged = api.ok("api_specification", request)
    assert 0 < judged["share"] < 1
    assert (judged["likely"], judged["surest"]) == (None, None)
    assert judged["overlay"]["at"] is None, "no blend is surest, so none is marked"
    assert judged["overlay"]["chances"] is None


def test_two_windows_are_joined_by_the_residuals_of_their_fits(api, bootstrap, monkeypatch):
    request = {**two_responses(api), "windows": {"Hardness": [28.0, None], "Cost": [None, 9.5]}}
    judged = api.ok("api_specification", request)
    correlation = judged["search"]["correlation"]
    assert correlation is not None, "the runs leave enough past the six terms of each model"
    assert correlation[0][1] > 0.9, "the two responses share their noise"
    monkeypatch.setattr(bootstrap, "CORRELATED_FROM", 100)
    apart = api.ok("api_specification", request)
    assert apart["search"]["correlation"] is None
    assert apart["share"] == judged["share"]
    assert judged["surest"]["probability"] != pytest.approx(apart["surest"]["probability"], abs=1e-6)


def test_a_window_upside_down_is_refused_for_a_blend_as_for_factors(api):
    held = lattice(api)
    error = api.error("api_specification", asked(held, windows={"y": [30.0, 28.0]}))
    assert error == "The window of y must run from a lower value to a higher one."
