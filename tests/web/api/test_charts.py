"""The charts: ``api_plots``, which draws the analysis in Learn with the library's own plots, as Plotly figures.

The page draws them with Plotly.js at the version ``plotly_js`` names, beside the tables that
hold their numbers.
"""

from __future__ import annotations

import json

import pytest

from process_improve.experiments import DesignRegion, Factor
from tests.web._support import FACTORS, factors_named, held_of, run_sheet

pytest.importorskip("plotly", reason="the page installs plotly for its first chart")


def test_the_charts_are_the_library_s_plots(api, held, simulated):
    result = api.ok("api_plots", {**held, "responses": simulated["responses"], "truth": simulated["truth"]})
    assert set(result["figures"]) == {"contour", "true_contour", "residuals", *MORE}
    assert result["contour_factors"] == ["T", "P"], "the two largest main effects"
    assert not result["contour_chosen"]
    assert result["axes"] == ["T", "P", "F"]
    contour = json.loads(result["figures"]["contour"])
    assert contour["data"][0]["type"] == "contour"
    assert result["plotly_js"]


#: The page's further charts, as it draws them for the worked example: logic.MORE_CHARTS.
MORE = {
    "perturbation",
    "main_effects",
    "interaction",
    "surface_3d",
    "cube",
    "normal_probability",
    "residuals_order",
    "prediction_variance",
}


def test_the_reader_chooses_the_two_factors_of_the_surface(api, held, simulated):
    result = api.ok("api_plots", {**held, "responses": simulated["responses"], "contour_factors": ["F", "T"]})
    assert result["contour_factors"] == ["F", "T"]
    assert result["contour_chosen"]
    for key in ("contour", "surface_3d", "prediction_variance"):
        assert "F (L/min)" in result["figures"][key], f"{key} is drawn over the factors chosen"
    assert "F = 10" in result["figures"]["interaction"], "one line for each setting of the first factor, in its units"


def test_the_further_charts_are_in_the_reader_s_words_and_units(api, held, simulated):
    figures = {
        k: json.loads(v)
        for k, v in api.ok("api_plots", {**held, "responses": simulated["responses"]})["figures"].items()
    }
    assert figures["surface_3d"]["layout"]["scene"]["xaxis"]["ticktext"] == ["150", "175", "200"]
    assert figures["perturbation"]["layout"]["yaxis"]["title"]["text"] == "Predicted Yield (%)"
    assert figures["main_effects"]["data"][0]["x"] == ["Low", "Middle", "High"]
    assert figures["interaction"]["data"][0]["x"] == ["1", "2", "3"], "the settings of P, in bar"
    corners = next(t for t in figures["cube"]["data"] if t["name"] == "Vertices")
    assert corners["mode"] == "markers+text"
    assert len(corners["text"]) == 8
    edges = next(t for t in figures["cube"]["data"] if t["name"] == "Edges")
    assert (edges["mode"], edges["line"]["width"]) == ("lines", 3), "the library draws the cube: the page adds nothing"
    assert len(figures["residuals_order"]["data"][-1]["x"]) == 15, "one residual a run"


def test_a_transformed_prediction_is_drawn_in_the_response_s_units(api, held, simulated):
    """The model is fitted to ln(Yield): each prediction drawn is put back in percent, as the surface is."""
    spec = {**held["spec"], "analysis_transforms": {held["spec"]["response"]: "log"}}
    plain = api.ok("api_plots", {**held, "responses": simulated["responses"]})["figures"]
    logged = api.ok("api_plots", {**held, "spec": spec, "responses": simulated["responses"]})["figures"]
    assert "cube" not in logged, "the library prints the corners on the model's scale"
    for key, read in (
        ("perturbation", lambda f: f["data"][0]["y"][0]),
        ("surface_3d", lambda f: f["data"][0]["z"][0][0]),
    ):
        assert read(json.loads(logged[key])) == pytest.approx(read(json.loads(plain[key])), rel=0.05)


def test_averages_are_drawn_only_for_factors_at_three_settings_or_fewer(api):
    """A space-filling design runs each factor at a setting of its own in every run: there is nothing to average."""
    held = held_of(api.ok("api_make_design", {"design_type": "maximin_lhs", "factors": factors_named("A", "B", "C")}))
    responses = api.ok("api_simulate", {**held, "seed": 1})["responses"]
    figures = api.ok("api_plots", {**held, "responses": responses})["figures"]
    assert not {"main_effects", "interaction"} & set(figures)
    assert {"perturbation", "surface_3d"} <= set(figures)


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


def test_a_blocked_design_is_charted(api):
    pytest.importorskip("plotly", reason="the page installs plotly for its first chart")
    spec = {"design_type": "box_behnken", "factors": factors_named("A", "B", "C"), "n_blocks": 2}
    design = api.ok("api_make_design", spec)
    responses = [float((i * 7) % 11) for i in range(design["n_runs"])]
    plots = api.ok("api_plots", {**held_of(design), "responses": responses})
    assert {"contour", "residuals", "surface_3d"} <= set(plots["figures"])
    assert "prediction_variance" not in plots["figures"], "a block has no setting to hold"


# --------------------------------------------------------------------------- limits on the settings
def _limited(api, limit: str) -> tuple[dict, list[float]]:
    """Make a D-optimal design inside a limit, and results that rise with T and P, so they are the factors plotted."""
    design = api.ok(
        "api_make_design", {"design_type": "d_optimal", "factors": FACTORS, "constraints": [{"expression": limit}]}
    )
    sheet = run_sheet(design)
    responses = (50 + 3 * (sheet["T"] - 175) / 25 + 2 * (sheet["P"] - 2) + 0.1 * (sheet["F"] - 15) / 5).tolist()
    return design, responses


def test_a_design_with_limits_shades_the_settings_that_cannot_be_run(api):
    design, responses = _limited(api, "T + 25 * P <= 250")
    result = api.ok("api_plots", {**held_of(design), "responses": responses})
    contour = json.loads(result["figures"]["contour"])
    shading = next(trace for trace in contour["data"] if trace.get("name") == "Cannot be run")
    assert contour["data"][0]["type"] == "contour", "the response stays the first trace, which the page restyles"
    assert contour["data"].index(shading) > 0
    assert shading["contours"]["type"] == "constraint"
    assert "title" not in shading.get("colorbar", {}), "the shading has no scale to title"
    assert contour["data"][0]["colorbar"]["title"]["text"] == "y"
    assert (result["contour_factors"], result["shaded"], result["held"]) == (["T", "P"], True, [])


def test_with_the_centre_ruled_out_the_others_are_held_at_a_setting_that_can_be_run(api):
    """F is not plotted: it is held where the region's settings centre, and the reply says where, in L/min."""
    limit = "T + 25 * P + 5 * F <= 290"
    design, responses = _limited(api, limit)
    result = api.ok("api_plots", {**held_of(design), "responses": responses})
    region = DesignRegion([Factor(name=f["name"], low=f["low"], high=f["high"]) for f in FACTORS], [limit])
    assert result["held"] == [{"name": "F", "value": pytest.approx(region.centroid()["F"])}]
    assert result["held"][0]["value"] < 15, "the limit leaves more settings at low F"


def test_a_design_without_limits_is_drawn_as_before(api, held, simulated):
    result = api.ok("api_plots", {**held, "responses": simulated["responses"]})
    contour = json.loads(result["figures"]["contour"])
    assert all(trace.get("name") != "Cannot be run" for trace in contour["data"])
    assert (result["shaded"], result["held"]) == (False, [])
