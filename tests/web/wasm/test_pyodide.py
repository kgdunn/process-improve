"""``web/bootstrap.py`` under Pyodide, asked what native Python is asked: the answers must agree.

This is where a dependency with no WebAssembly build, an import that reaches for one, or a
numerical difference under WebAssembly shows, rather than on a reader's screen. Every design
the page offers, at three and at six factors, is generated under Pyodide and analysed from
its own workbook, beside the same calls answered natively; the numbers agree to rounding. So do
the requests the page's options make: blocks, replicates, labels for a factor's levels, a model
of chosen terms and an axial distance typed as a number.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from tests.web._build import wheel
from tests.web._support import (
    ALL_DESIGNS,
    FACTORS,
    coded,
    factors_named,
    fill_workbook,
    generate_all,
    held_of,
    run_sheet,
)

#: Each design type at three and at six factors: the page's defaults otherwise.
CASES = [(design_type, k) for design_type in ALL_DESIGNS for k in (3, 6)]

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


@pytest.mark.parametrize(("design_type", "k"), CASES)
def test_each_design_and_its_analysis_agree_with_native_python(pyodide, api, design_type, k):
    spec = {"design_type": design_type, "factors": factors_named(*"ABCDEF"[:k])}
    under, native = pyodide.ok("api_make_design", spec), api.ok("api_make_design", spec)
    assert (under["columns"], under["n_runs"], under["model"]) == (native["columns"], native["n_runs"], native["model"])
    np.testing.assert_allclose(numbers(under["rows"]), numbers(native["rows"]), atol=1e-9)
    assert under["facts"] == native["facts"]

    workbook = {"xlsx_base64": fill_workbook(under["xlsx_base64"], known)}
    fitted, expected = pyodide.ok("api_analyze", workbook), api.ok("api_analyze", workbook)
    assert [c["term"] for c in fitted["coefficients"]] == [c["term"] for c in expected["coefficients"]]
    for got, want in zip(fitted["coefficients"], expected["coefficients"], strict=True):
        assert got["coefficient"] == pytest.approx(want["coefficient"], rel=1e-6, abs=1e-9), got["term"]
    assert fitted["summary"]["r_squared"] == pytest.approx(expected["summary"]["r_squared"], rel=1e-9)


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


def test_an_axial_distance_typed_as_a_number_places_the_runs_as_native_python_does(pyodide, api):
    spec = {"design_type": "ccd", "factors": FACTORS[:2], "n_center_points": 3, "alpha": 1.4}
    under, native = (python.ok("api_make_design", spec) for python in (pyodide, api))
    pd.testing.assert_frame_equal(run_sheet(under), run_sheet(native))
    assert under["facts"] == native["facts"]


def test_results_simulated_under_pyodide_are_native_python_s(pyodide, api, held):
    under, native = (python.ok("api_simulate", {**held, "seed": 3}) for python in (pyodide, api))
    assert under["truth"] == native["truth"]
    assert under["responses"] == pytest.approx(native["responses"], abs=0.011), "rounded to two decimals"


def test_a_workbook_written_under_pyodide_reads_back_the_same_either_way(pyodide, api, held, simulated):
    request = {**held, "responses": simulated["responses"], "sources": ["simulated"] * 15, "simulated": simulated}
    book = pyodide.ok("api_workbook", request)
    assert pyodide.ok("api_read_workbook", book) == api.ok("api_read_workbook", book)


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
