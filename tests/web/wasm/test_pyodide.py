"""``web/bootstrap.py`` under Pyodide, asked what native Python is asked: the answers must agree.

This is where a dependency with no WebAssembly build, an import that reaches for one, or a
numerical difference under WebAssembly shows, rather than on a reader's screen. Every design
the page offers, at three and at six factors, is generated under Pyodide and analysed from
its own workbook, beside the same calls answered natively; the numbers agree to rounding. So do
the requests the page's options make: blocks, replicates, labels for a factor's levels, a model
of chosen terms and an axial distance typed as a number.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import pytest

from tests.web._build import manifest, wheel
from tests.web._support import (
    ALL_DESIGNS,
    FACTORS,
    WEB,
    coded,
    factors_named,
    fill_workbook,
    generate_all,
    held_of,
    run_sheet,
    workbook_of,
)

#: What the page loads into Pyodide, and the Python that release runs.
RUNTIME = json.loads((WEB / "runtime.json").read_text(encoding="utf-8"))

#: Each design type at three and at six factors, or at the most the page makes it for: the page's defaults otherwise.
MOST_FACTORS = {"omars": 5}
CASES = [(design_type, k) for design_type in ALL_DESIGNS for k in (3, MOST_FACTORS.get(design_type, 6))]

#: A factor whose two levels are labels.
SUPPLIER = {"name": "Supplier", "low": "A", "high": "B", "units": ""}


def numbers(rows: list[list]) -> np.ndarray:
    """Return a run sheet's settings: every column but the empty results column."""
    return np.array([row[:-1] for row in rows], dtype=float)


def known(runs) -> np.ndarray:
    """Return a response with a main effect for every factor, one interaction and a little noise."""
    settings = runs.iloc[:, 2:-1].to_numpy(dtype=float)
    noise = np.random.default_rng(len(runs)).normal(0, 0.05, len(runs))
    return 10 + settings @ np.arange(1, settings.shape[1] + 1) + 0.5 * settings[:, 0] * settings[:, 1] + noise


def test_python_under_pyodide_runs_the_app_s_own_wheel(pyodide, web_app, tmp_path):
    version = re.match(r"process_improve-([^-]+)-", wheel(web_app, tmp_path).name).group(1)
    assert pyodide.module.version == version


def test_the_app_s_own_python_is_loaded_from_the_bytecode_the_build_made(pyodide, web_app, tmp_path):
    """The build compiled the wheel and ``bootstrap.py`` for the Python Pyodide runs, so the start compiles neither.

    A build that could not compile them (uv could not provide that Python) names no bytecode in its manifest, and
    fails here: the app would work, and start two to three seconds slower at every visit.
    """
    compiled_for = manifest(web_app, tmp_path).get("bytecode")
    assert compiled_for == RUNTIME["python"], "the build made no bytecode for the Python that web/runtime.json names"
    assert pyodide.module.python.startswith(f"{compiled_for}."), "Pyodide runs another Python than runtime.json names"
    assert pyodide.module.bytecode == {"wheel": True, "glue": True}


def test_a_design_installs_nothing_its_analysis_what_a_model_is_fitted_with_and_a_workbook_openpyxl(pyodide):
    """Start-up leaves all three out: each comes when a call first lacks it, and the call is made again."""
    before = pyodide.module.installed()
    design = pyodide.ok("api_make_design", {"design_type": "full_factorial", "factors": factors_named(*"ABC")})
    assert pyodide.ok("api_candidates", {"factors": factors_named(*"ABC"), "model": "interactions"})["made"]
    assert pyodide.module.installed() == before, "a design, and the comparison of designs, with what Python starts with"
    request = {**held_of(design), "responses": known(run_sheet(design)).tolist()}
    assert pyodide.ok("api_analyze_runs", request)["coefficients"], "made again once what it lacked was installed"
    pyodide.ok("api_best_settings", {**request, "goal": "maximize"})
    fitted = pyodide.module.installed()
    assert {"statsmodels", "sklearn"} & set(fitted), "the first fit asked for what a model is fitted with"
    assert "openpyxl" in before or "openpyxl" not in fitted, "and for nothing a workbook needs"
    assert workbook_of(pyodide, design), "the call that lacked openpyxl is made again once it is installed"
    assert "openpyxl" in pyodide.module.installed()


def test_the_worker_s_warm_up_names_statsmodels_among_what_the_page_runs_on(pyodide):
    """The catalogue may come before statsmodels is in; the warm-up, which the worker makes once it is, names it."""
    warmed = pyodide.ok("warm")
    assert set(warmed["packages"]) == {"numpy", "pandas", "scipy", "statsmodels"}
    assert warmed["pyodide"] == RUNTIME["pyodide"]
    assert pyodide.ok("api_catalogue")["runtime"] == warmed, "and from then on the catalogue says the same"


@pytest.mark.parametrize(("design_type", "k"), CASES)
def test_each_design_and_its_analysis_agree_with_native_python(pyodide, api, design_type, k):
    spec = {"design_type": design_type, "factors": factors_named(*"ABCDEF"[:k])}
    under, native = pyodide.ok("api_make_design", spec), api.ok("api_make_design", spec)
    assert (under["columns"], under["n_runs"], under["model"]) == (native["columns"], native["n_runs"], native["model"])
    np.testing.assert_allclose(numbers(under["rows"]), numbers(native["rows"]), atol=1e-9)
    assert under["facts"] == native["facts"]

    workbook = {"xlsx_base64": fill_workbook(workbook_of(pyodide, under), known)}
    fitted, expected = pyodide.ok("api_analyze", workbook), api.ok("api_analyze", workbook)
    assert [c["term"] for c in fitted["coefficients"]] == [c["term"] for c in expected["coefficients"]]
    for got, want in zip(fitted["coefficients"], expected["coefficients"], strict=True):
        assert got["coefficient"] == pytest.approx(want["coefficient"], rel=1e-6, abs=1e-9), got["term"]
    assert fitted["summary"]["r_squared"] == pytest.approx(expected["summary"]["r_squared"], rel=1e-9)


@pytest.mark.parametrize("design_type", ["plackett_burman", "dsd"])
@pytest.mark.parametrize("k", [13, 24, 40])
def test_a_screening_design_of_many_factors_and_its_analysis_agree_with_native_python(pyodide, api, design_type, k):
    """Past 12 factors, as the page makes them: the design, its runs as given, and their main effects."""
    spec = {"design_type": design_type, "factors": factors_named(*(f"X{i + 1}" for i in range(k)))}
    under, native = pyodide.ok("api_make_design", spec), api.ok("api_make_design", spec)
    assert (under["columns"], under["n_runs"], under["model"]) == (native["columns"], native["n_runs"], native["model"])
    np.testing.assert_allclose(numbers(under["rows"]), numbers(native["rows"]), atol=1e-9)
    assert under["facts"] == native["facts"]
    request = {**held_of(under), "responses": known(run_sheet(under)).tolist()}
    fitted, expected = pyodide.ok("api_analyze_runs", request), api.ok("api_analyze_runs", request)
    assert fitted["model"] == expected["model"] == "main_effects"
    for got, want in zip(fitted["coefficients"], expected["coefficients"], strict=True):
        assert got["coefficient"] == pytest.approx(want["coefficient"], rel=1e-6, abs=1e-9), got["term"]
    given = {**spec, "design_type": "given", "runs": numbers(under["rows"])[:, 2:].tolist()}
    assert pyodide.ok("api_make_design", given)["facts"] == api.ok("api_make_design", given)["facts"]


def test_blocks_labels_replicates_and_chosen_terms_agree_with_native_python(pyodide, api):
    """Two replicates in two blocks, a factor whose levels are labels, and a three-factor interaction to find."""
    factors = [*FACTORS[:2], SUPPLIER]
    spec = {"design_type": "full_factorial", "factors": factors, "n_center_points": 2, "n_blocks": 2, "n_replicates": 2}
    under, native = pyodide.ok("api_make_design", spec), api.ok("api_make_design", spec)
    pd.testing.assert_frame_equal(run_sheet(under), run_sheet(native))
    assert under["facts"] == native["facts"]

    sheet = run_sheet(under)
    x = coded(sheet, factors)
    shift = np.where(sheet["Block"] == 2, 4.0, 0.0)
    noise = np.random.default_rng(0).normal(0, 0.2, len(sheet))
    y = 50 + 3 * x["T"] - 2 * x["P"] + x["Supplier"] + 2 * x["T"] * x["P"] * x["Supplier"] + shift + noise
    chosen = {"terms": [["T"], ["P"], ["Supplier"], ["T", "P", "Supplier"]]}
    book = pyodide.ok("api_workbook", {**held_of(under), "responses": y.tolist(), "model": chosen})
    fitted, expected = pyodide.ok("api_analyze", book), api.ok("api_analyze", book)
    assert (fitted["formula"], fitted["notes"]) == (expected["formula"], expected["notes"])
    assert [c["label"] for c in fitted["coefficients"]] == [c["label"] for c in expected["coefficients"]]
    for got, want in zip(fitted["coefficients"], expected["coefficients"], strict=True):
        assert got["coefficient"] == pytest.approx(want["coefficient"], rel=1e-6, abs=1e-9), got["label"]
    assert [(a["source"], a["df"]) for a in fitted["anova"]] == [(a["source"], a["df"]) for a in expected["anova"]]
    assert fitted["summary"]["r_squared"] == pytest.approx(expected["summary"]["r_squared"], rel=1e-9)


def test_the_simpler_model_proposed_for_blocks_and_levels_is_native_python_s(pyodide, api):
    """A blocked study, and a factor of three levels: the search keeps the blocks in and the factor whole.

    The results are far from any tie between two models, so the last bits of Pyodide's numerics pick no other.
    """
    blocked = {"design_type": "box_behnken", "factors": FACTORS, "n_blocks": 3}
    catalyst = {"name": "Catalyst", "type": "categorical", "levels": ["X", "Y", "Z"]}
    levels = {"design_type": "full_factorial", "factors": [*FACTORS[:2], catalyst], "n_center_points": 0}
    between, by_level = {1: 5.0, 2: -5.0, 3: 0.0}, {"X": 0.0, "Y": 6.0, "Z": -6.0}
    processes = {
        "blocked": (blocked, lambda t, run: 60 + 4 * t - 3 * t**2 + between[int(run["Block"])]),
        "levels": (levels, lambda t, run: 50 + 3 * t + by_level[run["Catalyst"]]),
    }
    for name, (spec, process) in processes.items():
        design = pyodide.ok("api_make_design", spec)
        runs = [dict(zip(design["columns"], row, strict=True)) for row in design["rows"]]
        noise = np.random.default_rng(3).normal(0, 0.1, len(runs))
        y = [float(process((run["T"] - 175) / 25, run) + e) for run, e in zip(runs, noise, strict=True)]
        request = {**held_of(design), "responses": y}
        under, native = pyodide.ok("api_simpler_model", request), api.ok("api_simpler_model", request)
        assert (under["terms"], under["kept"], under["dropped"]) == (native["terms"], native["kept"], native["dropped"])
        assert under["adjusted_r_squared"] == pytest.approx(native["adjusted_r_squared"], rel=1e-9), name
        assert under["kept"] == {"blocked": ["T", "T²"], "levels": ["T", "Catalyst"]}[name]


def test_the_correlation_map_of_a_design_agrees_with_native_python(pyodide, api):
    """A Plackett-Burman design in seven factors: the same pairs, in the same order, though all are of one size."""
    spec = {"design_type": "plackett_burman", "factors": factors_named(*"ABCDEFG")}
    under, native = (python.ok("api_make_design", spec) for python in (pyodide, api))
    got, want = pyodide.ok("api_correlations", held_of(under)), api.ok("api_correlations", held_of(native))
    assert (got["terms"], got["n_pairs"]) == (want["terms"], want["n_pairs"])
    assert [(p["a"], p["b"], p["both"]) for p in got["pairs"]] == [(p["a"], p["b"], p["both"]) for p in want["pairs"]]
    np.testing.assert_allclose(
        np.array(got["matrix"], dtype=float), np.array(want["matrix"], dtype=float), atol=1e-9, equal_nan=True
    )


def test_an_axial_distance_typed_as_a_number_places_the_runs_as_native_python_does(pyodide, api):
    spec = {"design_type": "ccd", "factors": FACTORS[:2], "n_center_points": 3, "alpha": 1.4}
    under, native = (python.ok("api_make_design", spec) for python in (pyodide, api))
    pd.testing.assert_frame_equal(run_sheet(under), run_sheet(native))
    assert under["facts"] == native["facts"]


def test_results_simulated_under_pyodide_are_native_python_s(pyodide, api, held):
    under, native = (python.ok("api_simulate", {**held, "seed": 3}) for python in (pyodide, api))
    assert under["truth"] == native["truth"]
    assert under["responses"] == pytest.approx(native["responses"], abs=0.011), "rounded to two decimals"


def test_what_the_results_call_for_under_pyodide_is_native_python_s(pyodide, api, held, simulated):
    """The next step's search, its tests and its simulated critical values run under WebAssembly as natively."""
    request = {**held, "responses": simulated["responses"]}
    under, native = (python.ok("api_next_runs", request)["recommendation"] for python in (pyodide, api))
    assert (under["action"], under["does"], under["active"]) == (native["action"], native["does"], native["active"])
    assert under["ranges"] == native["ranges"]
    added = {**request, "kind": "recommended"}  # the worked example's results call for a new study, not runs added
    assert pyodide.error("api_add_runs", added) == api.error("api_add_runs", added)


def test_the_runs_the_results_call_for_under_pyodide_are_native_python_s(pyodide, api):
    cube = api.ok("api_make_design", {"design_type": "full_factorial", "factors": FACTORS, "n_center_points": 3})
    settings = coded(run_sheet(cube), FACTORS).to_numpy(dtype=float)
    peaked = 60 - 4 * np.abs(settings).sum(axis=1) + 0.1 * np.arange(len(settings))  # curved: axial runs called for
    curved = {**held_of(cube), "responses": peaked.tolist(), "kind": "recommended"}
    curved["spec"] = {**curved["spec"], "response_info": {"goal": "maximize"}}
    under, native = (python.ok("api_add_runs", curved) for python in (pyodide, api))
    assert under["added"] == native["added"] == 8
    assert under["rows"] == native["rows"]


def test_a_workbook_written_under_pyodide_reads_back_the_same_either_way(pyodide, api, held, simulated):
    request = {**held, "responses": simulated["responses"], "sources": ["simulated"] * 15, "simulated": simulated}
    book = pyodide.ok("api_workbook", request)
    assert pyodide.ok("api_read_workbook", book) == api.ok("api_read_workbook", book)


def test_a_run_sheet_renamed_under_pyodide_is_native_python_s(pyodide, api, design, simulated):
    """A fraction's notes name its factors, so the renamed notes are read off the runs again: both ways alike."""
    fraction = api.ok("api_make_design", {"design_type": "fractional_factorial", "factors": factors_named(*"ABCD")})
    for made, simulation in ((design, simulated), (fraction, None)):
        factors = [{**f, "name": f"{f['name']}_new"} for f in made["spec"]["factors"]]
        sent = {**held_of(made), "facts": made["facts"], "outside_range": made["outside_range"]}
        request = {"design": sent, "factors": factors, "simulation": simulation}
        assert pyodide.ok("api_relabel", request) == api.ok("api_relabel", request)


def test_refusals_and_warnings_come_back_as_they_do_natively(pyodide, api):
    bad = {"design_type": "box_behnken", "factors": [*FACTORS[:2], {**FACTORS[2], "name": "a b"}]}
    assert pyodide.error("api_make_design", bad) == api.error("api_make_design", bad)
    warned = {"design_type": "dsd", "factors": factors_named(*"ABCD"), "n_center_points": 1}
    assert pyodide.reply("api_make_design", warned)["warnings"] == api.reply("api_make_design", warned)["warnings"]


def test_the_comparison_under_pyodide_scores_as_native_python_does(pyodide, api):
    _, designs = generate_all(api, FACTORS, "main_effects")
    under, native = (
        python.ok("api_compare", {"model": "main_effects", "designs": designs}) for python in (pyodide, api)
    )
    assert under["designs"] == native["designs"]
    for label in native["designs"]:
        for key, value in native["values"][label].items():
            if isinstance(value, float):
                assert under["values"][label][key] == pytest.approx(value, rel=1e-6, abs=1e-9), (label, key)


# --------------------------------------------------------------------------- limits on the settings
#: Limits on the worked example's settings: a straight one, with a note, and a curved one.
LIMITS = [{"expression": "T + 25*P <= 250", "note": "the product chars"}, {"expression": "T * F <= 3600"}]


def agree(under: object, native: object, at: str = "the reply") -> None:
    """Assert two replies agree: the same keys, lists and texts, and the same numbers to rounding."""
    if isinstance(native, dict):
        assert isinstance(under, dict), at
        assert under.keys() == native.keys(), at
        for key, value in native.items():
            agree(under[key], value, f"{at}.{key}")
    elif isinstance(native, list):
        assert isinstance(under, list), at
        assert len(under) == len(native), at
        for i, (u, n) in enumerate(zip(under, native, strict=True)):
            agree(u, n, f"{at}[{i}]")
    elif isinstance(native, float):
        assert under == pytest.approx(native, rel=1e-6, abs=1e-9), at
    else:
        assert under == native, at


def test_the_limits_are_read_and_pictured_as_native_python_reads_them(pyodide, api):
    """The share kept (a fixed sample), the narrowed ranges (linprog), the corners and the picture, a slice traced."""
    request = {"factors": FACTORS, "constraints": LIMITS, "slice": {"x": "T", "y": "P", "at": {"F": 18}}}
    agree(pyodide.ok("api_limits", request), api.ok("api_limits", request))


#: Straight limits whose shares are known exactly. The heat limit cuts an eighth off the T-P square and
#: ``T + 10*F <= 325`` half the T-F square; the two share T and keep 11/24 of the box together. The band
#: ``190 <= T + 25*P <= 260`` keeps 91% of the T-P square.
EXACT = {
    "chained": [{"expression": "T + 25*P <= 250"}, {"expression": "T + 10*F <= 325"}],
    "band": [{"expression": "190 <= T + 25*P <= 260"}],
}


def test_a_straight_limit_s_share_is_worked_out_exactly_under_pyodide(pyodide, api):
    """One limit by its closed form, a band between two parallel ones, and limits sharing T by Qhull's volume.

    A share drawn from a sample would miss each of these values by far more than 1e-9.
    """
    under = {}
    for name, limits in EXACT.items():
        request = {"factors": FACTORS, "constraints": limits}
        under[name] = pyodide.ok("api_limits", request)
        agree(under[name], api.ok("api_limits", request), name)
    assert [c["removed"] for c in under["chained"]["constraints"]] == [1 / 8, 1 / 2]
    assert under["chained"]["kept"] == pytest.approx(11 / 24, abs=1e-9)
    assert under["band"]["kept"] == 0.91


@pytest.mark.parametrize("marked", [[7], [6, 7], [3, 6, 7], [0, 7], [4, 5, 6, 7, 8], [8]])
def test_the_limit_proposed_for_corners_marked_is_native_python_s(pyodide, api, marked):
    """The cut between the corners marked and the others (a least-distance plane), and its round numbers."""
    request = {
        "factors": FACTORS,
        "constraints": [],
        "marked": marked,
        "cuts": {"marked": [7], "corner": 7, "values": [190, 2.5, None]},
    }
    agree(pyodide.ok("api_limits", request), api.ok("api_limits", request))


@pytest.mark.parametrize("design_type", ["d_optimal", "i_optimal"])
def test_a_design_inside_the_limits_is_native_python_s(pyodide, api, design_type):
    request = {"design_type": design_type, "factors": FACTORS, "constraints": LIMITS}
    under, native = (python.ok("api_make_design", request) for python in (pyodide, api))
    np.testing.assert_allclose(numbers(under["rows"]), numbers(native["rows"]), rtol=1e-9)
    assert under["spec"]["constraints"] == native["spec"]["constraints"]


@pytest.mark.parametrize("limits", [LIMITS[:1], LIMITS], ids=["straight", "curved"])
def test_the_comparison_inside_the_limits_scores_as_native_python_does(pyodide, api, limits):
    """With a curved limit, G is climbed by SLSQP from its best points: under Pyodide too."""
    _, designs = generate_all(api, FACTORS, "main_quadratic", limits)
    under, native = (
        python.ok("api_compare", {"model": "main_quadratic", "designs": designs}) for python in (pyodide, api)
    )
    agree(under["values"], native["values"])
    agree(under["fds"], native["fds"])
