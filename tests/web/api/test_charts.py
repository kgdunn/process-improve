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


def test_the_charts_read_on_a_light_page_and_a_dark_one_and_without_their_colours(api, held, simulated):
    figures = {
        k: json.loads(v)
        for k, v in api.ok("api_plots", {**held, "responses": simulated["responses"]})["figures"].items()
    }
    surface, runs = figures["contour"]["data"]
    assert runs["marker"]["color"] == "#ffffff", (
        "a run has a white face and a dark edge, on every colour of the surface"
    )
    assert runs["marker"]["line"]["color"] == "#1a1a1a"
    assert runs["opacity"] == 1
    assert "marker" not in surface
    assert figures["contour"]["layout"]["legend"]["y"] > 1, "the legend above the plot, clear of the axis title"
    for key in ("perturbation", "main_effects", "interaction"):
        dashes = [trace["line"]["dash"] for trace in figures[key]["data"]]
        assert dashes == ["solid", "dash", "dot"], f"{key}: each line has a pattern of its own"
        assert figures[key]["layout"]["legend"]["y"] > 1
    assert {trace["mode"] for trace in figures["perturbation"]["data"]} == {"lines"}, "a curve: no marks along it"
    for key in ("main_effects", "interaction"):
        assert {trace["mode"] for trace in figures[key]["data"]} == {"lines+markers"}, f"{key}: a mark at each average"


def test_the_variance_of_a_prediction_is_drawn_square_as_the_surface_it_is_over(api, held, simulated):
    figures = api.ok("api_plots", {**held, "responses": simulated["responses"]})["figures"]
    for key in ("contour", "prediction_variance"):
        axis = json.loads(figures[key])["layout"]["yaxis"]
        assert (axis["scaleanchor"], axis["scaleratio"]) == ("x", 1), f"{key}: one coded step is as long on each axis"
        assert axis["ticktext"] == ["1", "2", "3"]


RESIDUAL_FIGURES = ("residuals", "normal_probability", "residuals_order")


def residual_points(figure: str) -> dict:
    """Return the trace of a residual figure that draws a point for each run."""
    return next(trace for trace in json.loads(figure)["data"] if trace.get("mode") == "markers")


def test_each_residual_names_its_run_where_the_pointer_rests_on_it(api, held, simulated):
    """A residual out of line is a run to look at again: each figure of the residuals says which run a point is."""
    request = {**held, "responses": simulated["responses"]}
    residual = {row["run"]: row["residual"] for row in api.ok("api_analyze_runs", request)["diagnostics"]}
    figures = api.ok("api_plots", request)["figures"]
    for key in RESIDUAL_FIGURES:
        points = residual_points(figures[key])
        assert sorted(points["customdata"]) == sorted(residual), f"{key}: every run, once"
        for run, drawn in zip(points["customdata"], points["y"], strict=True):
            assert drawn == pytest.approx(residual[run]), f"{key}: run {run}"
        assert points["hovertemplate"] == "Run %{customdata}<br>Residual %{y:.3g}<extra></extra>"
        assert points["meta"] == "runs", "the page marks a run on every chart whose trace says it draws the runs"
    by_size = residual_points(figures["normal_probability"])["customdata"]
    assert by_size != sorted(by_size), "the normal plot orders the residuals by size, not by run"


def test_each_run_on_the_surface_is_named(api, held, simulated):
    """The library draws the runs at one setting together, not in the run sheet's order: each point is still its run.

    The glue repeats that order to name the runs (``_runs_on_surface``). Each point's words hold its run's result,
    so a change to the library's order fails here.
    """
    responses = simulated["responses"]
    request = {**held, "responses": responses, "truth": simulated["truth"]}
    figures = api.ok("api_plots", request)["figures"]
    for key in ("contour", "true_contour"):
        runs = residual_points(figures[key])
        assert runs["meta"] == "runs"
        assert sorted(runs["customdata"]) == list(range(1, len(responses) + 1)), f"{key}: every run, once"
        assert runs["customdata"] != sorted(runs["customdata"]), "three runs share the center, and are drawn together"
        for run, words in zip(runs["customdata"], runs["text"], strict=True):
            assert words.startswith(f"Run {run}<br>"), key
            result = float(words.split("Yield: ")[1].split("<br>")[0])
            assert result == pytest.approx(responses[run - 1], abs=5e-3), f"{key}: run {run}'s own result"


def test_the_residuals_in_run_order_are_drawn_at_each_run_s_own_number(api, held, simulated):
    """With runs still to do, a residual's place along the axis is its run's number, not its place among the done."""
    responses = [None, None, *simulated["responses"][2:]]
    points = residual_points(api.ok("api_plots", {**held, "responses": responses})["figures"]["residuals_order"])
    assert points["x"] == list(range(3, len(responses) + 1))
    assert points["customdata"] == points["x"]


def test_a_chart_of_lines_reads_every_line_at_one_setting_together(api, held, simulated):
    figures = api.ok("api_plots", {**held, "responses": simulated["responses"]})["figures"]
    for key in ("perturbation", "main_effects", "interaction"):
        figure = json.loads(figures[key])
        assert figure["layout"]["hovermode"] == "x unified", key
        assert {trace["hovertemplate"] for trace in figure["data"]} == {"%{y:.4g}"}, key
    assert "hovermode" not in json.loads(figures["residuals"])["layout"]


def test_a_study_of_one_factor_has_no_surface_and_its_other_charts_are_drawn(api):
    """A surface is drawn over two factors: with one, the page draws its levels, and the library the rest."""
    design = {"design_type": "full_factorial", "factors": FACTORS[:1], "n_replicates": 3, "n_center_points": 2}
    held = held_of(api.ok("api_make_design", design))
    responses = api.ok("api_simulate", {**held, "scenario": "main_effects", "seed": 1})["responses"]
    result = api.ok("api_plots", {**held, "responses": responses})
    assert result["contour_factors"] == ["T"]
    surfaces = {"contour", "surface_3d", "interaction", "prediction_variance", "cube"}
    assert surfaces.isdisjoint(result["figures"]), "each takes two factors, or three"
    assert {"residuals", "perturbation", "main_effects", "normal_probability"} <= set(result["figures"])
    assert set(result["plot_options"]) == set(result["figures"]), "each figure drawn has the code that draws it"
    assert result["plot_options"]["perturbation"] == {"plot_type": "perturbation", "factors_to_plot": ["T"]}


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
    assert {"contour", "residuals", "surface_3d", "prediction_variance"} <= set(plots["figures"])
    variance = json.loads(plots["figures"]["prediction_variance"])
    assert variance["data"][0]["type"] == "contour", "drawn for the factors' terms: a block has no setting to hold"


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
