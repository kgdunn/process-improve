"""Curves: ``api_functional``, a study's curves fitted as numbers the page predicts whole curves from.

Each run's response is a curve. The page sends the runs' curves; Python reduces them by functional
PCA and models each kept score against the factors, coded as Learn codes them. The reply holds the
fitted curve and each bootstrap replicate as ``mean + row @ map``, so the page predicts a curve and
its bands at any settings without asking again. These tests hold those maps to the library's own
prediction at the same coded settings, and the target match to the settings that made the target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import analyze_functional_experiment
from tests.web._support import coded, edit_workbook, held_of, javascript, run_sheet, sheets_of, source

TIME = np.linspace(0, 10, 11)
CURVE = {"label": "Colour", "units": "AU", "argument": {"label": "Time", "units": "min", "values": TIME.tolist()}}


def truth(settings: dict) -> np.ndarray:
    """Return the curve at settings in the factors' units: T sets how high it rises, P how it drifts late."""
    t, p = (settings["T"] - 175) / 25, settings["P"] - 2
    level = 2 + 0.8 * t - 0.3 * t**2 + {"A": 0.0, "B": 0.4, "C": -0.2}[settings.get("Catalyst", "A")]
    return level * (1 - np.exp(-TIME / 3)) + 0.3 * p * TIME / 10


def with_curves(held: dict, seed: int = 1) -> dict:
    """Return a request for the worked example's runs, each with its curve read with a little noise."""
    rng = np.random.default_rng(seed)
    sheet = run_sheet(held)
    curves = [(truth(row) + rng.normal(0, 0.02, TIME.size)).round(4).tolist() for row in sheet.to_dict("records")]
    return {**held, "spec": {**held["spec"], "curve": CURVE}, "curves": curves}


def row_of(reply: dict, columns: dict) -> np.ndarray:
    """Return a model row as the page builds it: each term the product of the columns it names."""
    return np.array([np.prod([columns[name] for name in term]) for term in reply["terms"]])


def test_the_maps_predict_the_curve_and_its_spread_as_the_library_does(api, held):
    request = with_curves(held)
    reply = api.ok("api_functional", request)
    sheet = run_sheet(held)
    design = coded(sheet, held["spec"]["factors"])
    fit = analyze_functional_experiment(design, pd.DataFrame(request["curves"], columns=TIME), model=reply["formula"])
    point = {"T": 0.4, "P": -0.6, "F": 0.2}  # coded settings, -1 to +1
    bands = fit.predict(pd.DataFrame([point]), n_bootstrap=200, random_state=0)
    row = row_of(reply, point)
    assert reply["points"] == pytest.approx(TIME.tolist())
    assert np.array(reply["mean"]) + row @ np.array(reply["map"]) == pytest.approx(bands["curve"].iloc[0], rel=1e-5)
    replicates = np.array([np.array(r["mean"]) + row @ np.array(r["map"]) for r in reply["replicates"]])
    assert replicates.std(axis=0, ddof=1) == pytest.approx(bands["standard_error"].iloc[0], rel=1e-3)
    assert reply["runs"] == len(sheet)
    assert reply["components"][0]["kept"]
    assert reply["t"] > reply["widening"] > 1


def test_a_run_without_its_curve_yet_and_a_confirmation_run_are_left_out(api, held):
    request = with_curves(held)
    request["curves"][0] = None
    request["columns"] = [*request["columns"], "Confirmation run"]
    request["rows"] = [[*row, "yes" if i == 1 else ""] for i, row in enumerate(request["rows"])]
    assert api.ok("api_functional", request)["runs"] == len(request["rows"]) - 2


def test_a_curve_read_at_many_points_is_drawn_at_fewer_spanning_the_same_range(api, held):
    request = with_curves(held)
    many = np.linspace(0, 10, 200)
    request["spec"]["curve"] = {**CURVE, "argument": {**CURVE["argument"], "values": many.tolist()}}
    request["curves"] = [np.interp(many, TIME, curve).tolist() for curve in request["curves"]]
    reply = api.ok("api_functional", request)
    assert len(reply["points"]) == 61
    assert (reply["points"][0], reply["points"][-1]) == (0.0, 10.0)
    assert np.array(reply["replicates"][0]["map"]).shape == (len(reply["terms"]), 61)


def test_the_target_curve_is_matched_by_settings_close_to_those_that_made_it(api, held):
    request = with_curves(held)
    made = {"T": 185.0, "P": 2.5, "F": 15.0}
    match = api.ok("api_functional", {**request, "target": truth(made).tolist()})["match"]
    assert match["settings"]["T"] == pytest.approx(made["T"], abs=2.5)
    assert match["settings"]["P"] == pytest.approx(made["P"], abs=0.1)
    assert match["distance"] < 0.05


def test_a_categorical_factor_is_held_at_its_level_while_the_rest_are_searched(api):
    factors = [
        {"name": "T", "low": 150, "high": 200, "units": "degC"},
        {"name": "P", "low": 1, "high": 3, "units": "bar"},
        {"name": "Catalyst", "type": "categorical", "levels": ["A", "B", "C"], "units": ""},
    ]
    held = held_of(api.ok("api_make_design", {"design_type": "full_factorial", "factors": factors}))
    request = with_curves(held)
    made = {"T": 160.0, "P": 1.5, "Catalyst": "B"}
    target = truth(made).tolist()
    match = api.ok("api_functional", {**request, "model": "main_effects", "target": target, "held": {"Catalyst": "B"}})
    assert match["match"]["settings"]["Catalyst"] == "B"
    assert match["match"]["settings"]["T"] == pytest.approx(made["T"], abs=6)
    assert [name for term in match["terms"] for name in term if name.startswith("Catalyst")]


def test_the_curves_go_out_in_a_workbook_and_come_back_to_their_runs_however_its_rows_are_sorted(api, held):
    request = with_curves(held)
    request["curves"][2] = None  # a run still to do
    request["curves"][3][4] = None  # a point a run missed
    xlsx = api.ok("api_workbook", request)["xlsx_base64"]
    sheets = sheets_of(xlsx)
    assert list(sheets["Curves"].columns) == ["Std order", "Run order", *TIME.tolist()]
    assert api.ok("api_read_workbook", {"xlsx_base64": xlsx})["curves"] == request["curves"]

    def reverse(sheets: dict) -> None:
        sheets["Curves"] = sheets["Curves"].iloc[::-1]  # sorted in Excel, from the last run to the first

    back = api.ok("api_read_workbook", {"xlsx_base64": edit_workbook(xlsx, reverse)})
    assert back["curves"] == request["curves"]

    def drop(sheets: dict) -> None:
        del sheets["Curves"]  # a workbook from before the study read curves

    without = api.ok("api_read_workbook", {"xlsx_base64": edit_workbook(xlsx, drop)})
    assert without["curves"] == [None] * len(request["curves"])


def test_the_page_draws_the_curve_and_both_its_bands_as_the_library_predicts_them(api, held, node):
    request = with_curves(held)
    reply = api.ok("api_functional", request)
    module = source("best.js")
    (columns,) = javascript(node, module, [("columnsAt", [reply["factors"], {"T": 185.0, "P": 1.4, "F": 12.0}])])
    (drawn,) = javascript(node, module, [("curveAt", [reply, columns])])
    design = coded(run_sheet(held), held["spec"]["factors"])
    fit = analyze_functional_experiment(design, pd.DataFrame(request["curves"], columns=TIME), model=reply["formula"])
    point = pd.DataFrame([{name: columns[name] for name in ("T", "P", "F")}])
    bands = fit.predict(point, n_bootstrap=200, random_state=0)
    assert drawn["curve"] == pytest.approx(bands["curve"].iloc[0].tolist(), rel=1e-5)
    assert drawn["pointwise"][0] == pytest.approx(bands["pointwise_low"].iloc[0].tolist(), rel=1e-4)
    assert drawn["simultaneous"][1] == pytest.approx(bands["simultaneous_high"].iloc[0].tolist(), rel=1e-4)
    assert drawn["critical"] == pytest.approx(bands["critical_value"].iloc[0], rel=1e-4)
