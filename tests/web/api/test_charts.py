"""The charts: ``api_plots``, which draws step 3's analysis with the library's own plots, as Plotly figures.

The page draws them with Plotly.js at the version ``plotly_js`` names, beside the tables that
hold their numbers.
"""

from __future__ import annotations

import json

import pytest

from tests.web._support import factors_named, held_of

pytest.importorskip("plotly", reason="the page installs plotly for its first chart")


def test_the_charts_are_the_library_s_plots(api, held, simulated):
    result = api.ok("api_plots", {**held, "responses": simulated["responses"], "truth": simulated["truth"]})
    assert set(result["figures"]) == {"contour", "true_contour", "residuals"}
    assert result["contour_factors"] == ["T", "P"], "the two largest main effects"
    contour = json.loads(result["figures"]["contour"])
    assert contour["data"][0]["type"] == "contour"
    assert result["plotly_js"]


def test_the_true_surface_is_drawn_over_the_same_grid(api, held, simulated):
    """The page gives both surfaces one colour scale, which needs a grid of values in each."""
    result = api.ok("api_plots", {**held, "responses": simulated["responses"], "truth": simulated["truth"]})
    fitted, true = (json.loads(result["figures"][key])["data"][0] for key in ("contour", "true_contour"))
    assert len(fitted["z"]) == len(true["z"])
    assert {len(row) for row in fitted["z"]} == {len(row) for row in true["z"]}


def test_runs_still_to_do_are_left_out_of_the_charts(api, held, simulated):
    responses = [*simulated["responses"][:-2], None, None]
    result = api.ok("api_plots", {**held, "responses": responses})
    residuals = json.loads(result["figures"]["residuals"])
    assert sum(len(trace["x"]) for trace in residuals["data"] if trace.get("mode") == "markers") == 13


def test_a_saturated_two_level_design_is_drawn_with_lenth_s_half_normal_plot(api):
    spec = {"design_type": "plackett_burman", "factors": factors_named(*"ABCDEFG"), "n_center_points": 0}
    held = held_of(api.ok("api_make_design", spec))
    responses = api.ok("api_simulate", {**held, "scenario": "main_effects", "seed": 1})["responses"]
    result = api.ok("api_plots", {**held, "responses": responses})
    assert "half_normal" in result["figures"]
    assert "residuals" not in result["figures"], "a saturated fit leaves no residuals to plot"


def test_a_true_process_naming_other_factors_is_refused(api, held, simulated):
    truth = [{"term": "Z", "coefficient": 1.0}]
    assert "['Z']" in api.error("api_plots", {**held, "responses": simulated["responses"], "truth": truth})
