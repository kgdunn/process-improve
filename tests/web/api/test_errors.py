"""What the page tells a reader whose request it refuses.

Every refusal is a sentence the page shows as it is, so it must say what is wrong in the
reader's terms: which factor, which value, what to do instead. ``REFUSALS`` asks for each one
the way a reader (or a workbook) could, and the last test reads ``web/bootstrap.py`` for every
message it can refuse with, so a new refusal cannot ship without a case here.
"""

from __future__ import annotations

import ast
import base64
import io
import json
import re
import zipfile
from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest

from tests.web._support import FACTORS, WEB, edit_workbook, factors_named, fill_workbook, quadratic, workbook_of

if TYPE_CHECKING:
    from collections.abc import Callable

#: The worked example, as the page holds it, for the cases that send runs back.
Example = dict[str, Any]


def _make(**change: object) -> Callable[[Example], dict]:
    """Build a design request: the worked example's Box-Behnken design, changed."""
    return lambda _: {"design_type": "box_behnken", "factors": FACTORS, **change}


def _runs(**change: object) -> Callable[[Example], dict]:
    """Build a request about the runs the page holds, with the simulated results, changed."""
    return lambda ex: {**ex["held"], "responses": ex["simulated"]["responses"], **change}


def _zipped(size: int) -> str:
    """Return a zip file of one part of ``size`` zero bytes, in base64: small as sent, whatever it unpacks to."""
    held = io.BytesIO()
    with zipfile.ZipFile(held, "w", zipfile.ZIP_DEFLATED) as packed:
        packed.writestr("xl/sharedStrings.xml", b"\0" * size)
    return base64.b64encode(held.getvalue()).decode()


def _spec(**change: object) -> Callable[[Example], dict]:
    """Build a request whose design specification, sent back by the page, has been changed."""
    return lambda ex: {**ex["held"], "spec": {**ex["held"]["spec"], **change}}


def _workbook(edit: Callable[[dict], object], **request: object) -> Callable[[Example], dict]:
    """Build an upload of the worked example's workbook, filled in and then changed as a reader could in Excel."""

    def payload(ex: Example) -> dict:
        filled = fill_workbook(ex["workbook"], quadratic)
        return {"xlsx_base64": edit_workbook(filled, edit), **request}

    return payload


def _stored(**change: object) -> Callable[[dict], None]:
    """Change the specification a workbook stores on its hidden sheet."""

    def edit(sheets: dict) -> None:
        stored = json.loads(sheets["_pi_design"]["spec"].iloc[0])
        sheets["_pi_design"]["spec"] = [json.dumps({**stored, **change})]

    return edit


def _blank(rows: int) -> Callable[[dict], None]:
    """Clear the results of the first ``rows`` runs, as runs still to do."""

    def edit(sheets: dict) -> None:
        sheets["Runs"].loc[: rows - 1, "Yield"] = None

    return edit


def _renamed(factors: list[dict]) -> Callable[[Example], dict]:
    """Build a request to rename the factors of the worked example's run sheet, giving them ``factors``."""
    return lambda ex: {"design": ex["held"], "factors": factors}


def _foreign_file(_: Example) -> dict:
    return {"xlsx_base64": base64.b64encode(b"not a workbook").decode()}


def _many(k: int) -> list[dict]:
    """Return ``k`` factors, X1 to Xk, each from 0 to 1."""
    return [{"name": f"X{i + 1}", "low": 0, "high": 1, "units": ""} for i in range(k)]


def _screening(k: int, **request: object) -> dict:
    """Return 16 two-level runs in ``k`` factors as the page holds a screening design, a result each, and more."""
    spec = {"version": 1, "design_type": "plackett_burman", "factors": _many(k), "response": "y"}
    rows = [[run, run, *((run >> j % 4) & 1 for j in range(k)), None] for run in range(16)]
    columns = ["Std order", "Run order", *(f["name"] for f in _many(k)), "y"]
    return {"spec": spec, "columns": columns, "rows": rows, "responses": [float(r % 5) for r in range(16)], **request}


#: A 2^2 factorial in a temperature and a supplier, A or B, as the page holds its runs.
SUPPLIED = {
    "spec": {
        "design_type": "full_factorial",
        "factors": [FACTORS[0], {"name": "S", "low": "A", "high": "B", "units": ""}],
        "response": "y",
        "model": "interactions",
        "options": {},
    },
    "columns": ["Std order", "Run order", "T", "S", "y"],
    "rows": [[1, 1, 150, "A", None], [2, 2, 200, "A", None], [3, 3, 150, "B", None], [4, 4, 200, "B", None]],
    "responses": [10.0, 12.5, 11.0, 14.0],
}


#: A half fraction in A, B and C (I = ABC), as the page holds runs as given, inside a limit its fold-over breaks: the
#: other half holds the run with every factor low, which the limit rules out.
HALF_FRACTION = {
    "spec": {
        "design_type": "given",
        "factors": factors_named("A", "B", "C"),
        "response": "y",
        "model": "main_effects",
        "options": {},
        "constraints": [{"expression": "A + B + C >= -2"}],
    },
    "columns": ["Std order", "Run order", "A", "B", "C", "y"],
    "rows": [[1, 1, -1, -1, 1, None], [2, 2, 1, -1, -1, None], [3, 3, -1, 1, -1, None], [4, 4, 1, 1, 1, None]],
}


def _supplied(**change: object) -> Callable[[Example], dict]:
    """Build a request about the runs of a design with a categorical factor, changed."""
    return lambda _: {**SUPPLIED, **change}


#: A curve each run gives, read at three times.
CURVED = {"label": "Colour", "argument": {"label": "Time", "units": "min", "values": [0, 1, 2]}}


def _curved(held: str = "held", curve: object = CURVED, **change: object) -> Callable[[Example], dict]:
    """Build a request for the curves of the runs of ``held`` (a key of the example), each run with one, changed."""

    def payload(ex: Example) -> dict:
        runs = ex.get(held, SUPPLIED)
        curves = [[0.0, 1.0 + 0.1 * i, 1.5 + 0.02 * i * i] for i in range(len(runs["rows"]))]
        sent = {key: runs[key] for key in ("spec", "columns", "rows")}
        return {**sent, "spec": {**sent["spec"], "curve": curve}, "curves": curves, **change}

    return payload


def _curve_sheet(make: Callable[[pd.DataFrame], pd.DataFrame]) -> Callable[[dict], None]:
    """Describe a curve read at three times in the stored specification, and give the workbook a Curves sheet."""

    def edit(sheets: dict) -> None:
        _stored(curve=CURVED)(sheets)
        sheets["Curves"] = make(sheets["Runs"])

    return edit


def _typed(*limits: str, factors: list[dict] = FACTORS, **change: object) -> Callable[[Example], dict]:
    """Build the request the limits panel sends as the reader types: the brief's factors, and these limits."""
    return lambda _: {"factors": factors, "constraints": [{"expression": text} for text in limits], **change}


def _limited(design_type: str, *limits: str) -> Callable[[Example], dict]:
    """Build a design request, in the worked example's factors, with limits on the settings."""
    sent = [{"expression": text} for text in limits]
    return lambda _: {"design_type": design_type, "factors": FACTORS, "constraints": sent}


def _balanced(cost: dict | None = None, costs: list | None = None, **change: object) -> Callable[[Example], dict]:
    """Build a balance request: the worked example's yield, and a cost measured in the same runs (``cost``)."""

    def payload(ex: Example) -> dict:
        spec = ex["held"]["spec"]
        described = cost or {"name": "Cost", "label": "Cost", "goal": "minimize"}
        responses = [{"name": "Yield", **spec["response_info"]}, described]
        sent = costs if costs is not None else [20 + 0.1 * i for i in range(15)]
        return {
            **ex["held"],
            "spec": {**spec, "responses": responses},
            "responses": ex["simulated"]["responses"],
            "more_responses": {"Cost": sent},
            **change,
        }

    return payload


#: A temperature, a pressure and a supplier, A or B.
WITH_SUPPLIER = [*FACTORS[:2], {"name": "S", "low": "A", "high": "B", "units": ""}]

#: Three components of a blend, each from 0 to 1, as the brief sends them with their bounds left blank.
COMPONENTS = [{"name": n, "low": None, "high": None, "units": ""} for n in "ABC"]


def _blend(*factors: dict, **change: object) -> Callable[[Example], dict]:
    """Build a request for a simplex lattice in three components, the components changed by ``factors``, by name."""
    named = {f["name"]: f for f in factors}
    sent = [{**c, **named.pop(c["name"], {})} for c in COMPONENTS] + list(named.values())
    return lambda _: {"design_type": "simplex_lattice", "factors": sent, **change}


def _mixed(**change: object) -> Callable[[Example], dict]:
    """Build a request about the runs of a simplex lattice in three components, with simulated results, changed."""
    return lambda ex: {**ex["mixture"], **change}


#: The functions a limit can use, as a refusal lists them.
FUNCTIONS = "abs, sqrt, exp, log and log10"


#: Each refusal: the entry point, the request as a function of the worked example, and the message.
REFUSALS: dict[str, tuple[str, Callable[[Example], dict], str]] = {
    # The factors, as typed into the brief.
    "no factors": ("api_make_design", _make(factors=[]), "Enter at least one factor."),
    "one factor, for a design of three or more": (
        "api_make_design",
        _make(factors=FACTORS[:1]),
        "The Box-Behnken design takes 3 to 7 factors on this page; you have 1.",
    ),
    "repeated name": (
        "api_make_design",
        _make(factors=[FACTORS[0], {**FACTORS[1], "name": "T"}]),
        "Factor names must be unique.",
    ),
    "name with a space": (
        "api_make_design",
        _make(factors=[FACTORS[0], {**FACTORS[1], "name": "a b"}]),
        (
            "Factor names must look like variable names (letters, digits, _) and cannot be Python keywords: "
            "try 'a_b' for 'a b'."
        ),
    ),
    "names with a percent sign and a keyword": (
        "api_make_design",
        _make(factors=[{**FACTORS[0], "name": "Water %"}, {**FACTORS[1], "name": "lambda"}, FACTORS[2]]),
        (
            "Factor names must look like variable names (letters, digits, _) and cannot be Python keywords: "
            "try 'Water_pct' for 'Water %', 'lambda_' for 'lambda'."
        ),
    ),
    "low above high": (
        "api_make_design",
        _make(factors=[FACTORS[0], {**FACTORS[1], "low": 5}]),
        "Factor P: the low value must be below the high value.",
    ),
    "blank low": (  # a blank box, which the page sends as null
        "api_make_design",
        _make(factors=[FACTORS[0], {**FACTORS[1], "low": None}]),
        "Factor P: enter a number for both the low and the high value, or a label for both, as for a supplier A and B.",
    ),
    "text for high": (
        "api_make_design",
        _make(factors=[FACTORS[0], {**FACTORS[1], "high": "abc"}]),
        "Factor P: enter a number for both the low and the high value, or a label for both, as for a supplier A and B.",
    ),
    "infinite high": (
        "api_make_design",
        _make(factors=[FACTORS[0], {**FACTORS[1], "high": "inf"}]),
        "Factor P: enter a number for both the low and the high value, or a label for both, as for a supplier A and B.",
    ),
    "two labels the same": (
        "api_make_design",
        _make(factors=[FACTORS[0], {"name": "S", "low": "A", "high": "A", "units": ""}]),
        "Factor S: A is listed twice; its levels must differ.",
    ),
    "levels that read as the same number": (
        "api_make_design",
        _make(factors=[FACTORS[0], {"name": "S", "type": "categorical", "levels": ["1", "1.0"]}]),
        "Factor S: 1.0 and 1 read as the same number; its levels must differ.",
    ),
    "levels that are no list": (
        "api_make_design",
        _make(factors=[FACTORS[0], {"name": "S", "type": "categorical", "levels": "A, B"}]),
        "Factor S: list its levels, each a label.",
    ),
    "more levels than the page takes": (
        "api_make_design",
        _make(factors=[FACTORS[0], {"name": "S", "type": "categorical", "levels": list("ABCDEFGHJ")}]),
        "Factor S: give it 2 to 8 levels; it has 9.",
    ),
    "a blank level": (
        "api_make_design",
        _make(factors=[FACTORS[0], {"name": "S", "type": "categorical", "levels": ["A", " "]}]),
        "Factor S: a level is blank; give each level a label.",
    ),
    "a level too long": (
        "api_make_design",
        _make(factors=[FACTORS[0], {"name": "S", "type": "categorical", "levels": ["A", "x" * 65]}]),
        "Factor S: the level 'xxxxxxxxxxxxxxxxxxxx'... is longer than 64 characters.",
    ),
    "a name taken by another factor's levels": (
        "api_make_design",
        _make(
            factors=[
                {"name": "S", "type": "categorical", "levels": ["A", "B", "C"]},
                {"name": "S_L1", "low": 1, "high": 2, "units": ""},
            ]
        ),
        "Factor S_L1: rename it, as the page names a column of another factor's levels so.",
    ),
    "three levels in a two-level design": (
        "api_make_design",
        _make(
            design_type="fractional_factorial",
            factors=[*FACTORS, {"name": "S", "type": "categorical", "levels": ["A", "B", "C"]}],
        ),
        (
            "The Fractional factorial (2-level) design runs each categorical factor at two levels, and S has more: "
            "choose a full factorial or a Taguchi design."
        ),
    ),
    "a label a spreadsheet reads as blank": (
        "api_make_design",
        _make(factors=[FACTORS[0], {"name": "S", "low": "NA", "high": "EU", "units": ""}]),
        "Factor S: a spreadsheet reads 'NA' back as a blank cell; choose another label.",
    ),
    "a categorical factor in a design that needs numbers": (
        "api_make_design",
        _make(factors=[*FACTORS[:2], {"name": "S", "low": "A", "high": "B", "units": ""}]),
        (
            "The Box-Behnken design needs numeric factors, and S is categorical: choose a full factorial, "
            "fractional factorial, Plackett-Burman or Taguchi design."
        ),
    ),
    "a run at neither label": (
        "api_analyze_runs",
        _supplied(rows=[*SUPPLIED["rows"][:3], [4, 4, 200, "C", None]]),
        "Factor S: each run must be at A or B, not 'C'.",
    ),
    "a run with no label": (
        "api_analyze_runs",
        _supplied(rows=[*SUPPLIED["rows"][:3], [4, 4, 200, None, None]]),
        "Factor S: run 4 has no setting; each run must be at A or B.",
    ),
    "the square of a categorical factor": (
        "api_analyze_runs",
        _supplied(model={"terms": [["T"], ["S", "S"]]}),
        "S is categorical, so it has no square: it is +1 at both its levels.",
    ),
    "an interaction of a factor of three levels and another without either": (
        "api_analyze_runs",
        _supplied(
            spec={
                **SUPPLIED["spec"],
                "factors": [
                    {"name": "M", "type": "categorical", "levels": ["X", "Y", "Z"], "units": ""},
                    SUPPLIED["spec"]["factors"][1],
                ],
            },
            columns=["Std order", "Run order", "M", "S", "y"],
            rows=[[1, 1, "X", "A", None], [2, 2, "Y", "A", None], [3, 3, "X", "B", None], [4, 4, "Y", "B", None]],
            model={"terms": [["M", "S"]]},
        ),
        "The term M·S needs M or S in the model too: add one of them, or take it out.",
    ),
    # The request itself, which only a script could get wrong.
    "a request that is not an object": (
        "api_make_design",
        lambda _: [1],
        "The page sent a request that is not a JSON object.",
    ),
    # The design's settings.
    "unknown design": ("api_make_design", _make(design_type="rocket"), "Unknown design type 'rocket'."),
    # What if: each response's fit, and the limits the reader's settings break.
    "models that are not models": (
        "api_predictors",
        _runs(models=["quadratic"]),
        "The page sent models for the responses this page did not write.",
    ),
    "settings that are not settings": (
        "api_outside_limits",
        lambda ex: {"spec": ex["held"]["spec"], "settings": [175, 2, 15]},
        "The page sent settings this page did not write.",
    ),
    "a setting past its range": (
        "api_outside_limits",
        lambda ex: {"spec": ex["held"]["spec"], "settings": {"T": 210, "P": 2, "F": 15}},
        "Set T to a number from 150 to 200.",
    ),
    # The settings that balance the responses, and the goals the reader sets for them.
    "goals that are not goals": (
        "api_balance",
        _balanced(goals="narrow"),
        "The page sent goals for the responses this page did not write.",
    ),
    "an acceptable range upside down": (
        "api_balance",
        _balanced(goals={"Yield": {"low": 70, "high": 60}}),
        "The acceptable range of Yield must run from a lower value to a higher one.",
    ),
    "a target outside its acceptable range": (
        "api_balance",
        _balanced(
            {"name": "Cost", "label": "Cost", "goal": "target", "target": 21}, goals={"Cost": {"low": 0, "high": 10}}
        ),
        "The target of Cost, 21, must lie inside its acceptable range.",
    ),
    "an importance of six": (
        "api_balance",
        _balanced(goals={"Yield": {"importance": 6}}),
        "The importance of Yield must be a whole number from 1 to 5.",
    ),
    "a shape the page does not offer": (
        "api_balance",
        _balanced(goals={"Cost": {"shape": "steep"}}),
        "The shape of Cost's desirability must be one the page offers.",
    ),
    "a response to balance without results": (
        "api_balance",
        _balanced(costs=[None] * 15),
        "Enter results for Cost first: the balance is found on each response's model.",
    ),
    # Several responses, as the brief lists them.
    "a list of responses that is not one": (
        "api_make_design",
        _make(responses="Yield, Cost"),
        "The page sent a list of responses this page did not write.",
    ),
    "a first response that is not the study's": (
        "api_make_design",
        _make(response="Yield", responses=[{"name": "Cost"}, {"name": "Yield"}]),
        "The page sent a list of responses this page did not write.",
    ),
    "six responses": (
        "api_make_design",
        _make(responses=[{"name": f"y{i}"} for i in range(1, 7)]),
        "A study measures at most 5 responses; this one has 6.",
    ),
    "two responses of one name": (
        "api_make_design",
        _make(responses=[{"name": "Cost"}, {"name": "Cost"}]),
        "Two responses cannot share the name Cost.",
    ),
    "a response named as a factor, after the first": (
        "api_make_design",
        _make(responses=[{"name": "Yield"}, {"name": "P"}]),
        "The response name must be a variable name different from every factor.",
    ),
    "a response the study does not measure": (
        "api_analyze_runs",
        _runs(response="Colour"),
        "The study measures no response named 'Colour'.",
    ),
    "results for a response the study does not measure": (
        "api_analyze_runs",
        _runs(more_responses={"Colour": [1.0] * 15}),
        "The page sent results for a response the study does not measure.",
    ),
    "a later response's results, one short": (
        "api_simulate",
        lambda ex: {
            **ex["held"],
            "spec": {**ex["held"]["spec"], "responses": [{"name": "Yield"}, {"name": "Cost"}]},
            "more_responses": {"Cost": [1.0] * 14},
        },
        "14 responses were sent for 15 runs.",
    ),
    "a later response's model that is not one": (
        "api_workbook",
        lambda ex: {
            **ex["held"],
            "spec": {**ex["held"]["spec"], "responses": [{"name": "Yield"}, {"name": "Cost"}]},
            "models": {"Colour": "quadratic"},
        },
        "The page sent a design this page did not make.",
    ),
    "response named as a factor": (
        "api_make_design",
        _make(response="T"),
        "The response name must be a variable name different from every factor.",
    ),
    "too few factors for the design": (
        "api_make_design",
        lambda _: {"design_type": "dsd", "factors": factors_named("A", "B")},
        "The Definitive screening (DSD) design takes 3 to 40 factors on this page; you have 2.",
    ),
    "more factors than a Taguchi array on this page takes": (
        "api_make_design",
        lambda _: {"design_type": "taguchi", "factors": factors_named(*(f"X{i}" for i in range(41)))},
        "The Taguchi orthogonal array design takes 2 to 40 factors on this page; you have 41.",
    ),
    "a Taguchi array at levels the page does not offer": (
        "api_make_design",
        _make(design_type="taguchi", n_levels=4),
        "A Taguchi array runs each numeric factor at 2 levels or at 3 on this page.",
    ),
    "unknown axial distance": (
        "api_make_design",
        _make(design_type="ccd", alpha="spherical"),
        "Unknown axial distance 'spherical'.",
    ),
    "blank axial distance": (  # the number box beside "A distance you choose", left blank
        "api_make_design",
        _make(design_type="ccd", alpha=None),
        "Enter the axial distance in coded units: a number above 0, such as 1.4.",
    ),
    "axial distance not above 0": (
        "api_make_design",
        _make(design_type="ccd", alpha=-1),
        "The axial distance must be above 0 in coded units, not -1.",
    ),
    "blocks out of range": (
        "api_make_design",
        _make(n_blocks=9),
        "Blocks must be a whole number from 1 to 8.",
    ),
    "response named Block": (
        "api_make_design",
        _make(response="Block"),
        "The response cannot be named Block: a blocked design's run sheet has a Block column.",
    ),
    # The response's description, from the response card.
    "a description of the response that is not one": (
        "api_make_design",
        _make(response_info="Yield, in %"),
        "The page sent a description of the response this page did not write.",
    ),
    "a response's units that are not text": (
        "api_make_design",
        _make(response_info={"label": "Yield", "units": 5}),
        "The page sent a description of the response this page did not write.",
    ),
    "a typical value that is not a number": (
        "api_make_design",
        _make(response_info={"typical": "about sixty"}),
        "The response's typical value must be a number.",
    ),
    "a target that is not a number": (
        "api_make_design",
        _make(response_info={"goal": "target", "target": "high"}),
        "The response's target must be a number.",
    ),
    "no run-to-run noise": (
        "api_make_design",
        _make(response_info={"noise": 0}),
        "The run-to-run noise must be a number above 0.",
    ),
    "a smallest change below 0": (
        "api_compare",
        lambda _: {"model": "main_effects", "designs": [], "noise": 1.5, "smallest_change": -2},
        "The smallest change worth finding must be a number above 0.",
    ),
    "a blocked workbook without its blocks": (
        "api_analyze",
        _workbook(_stored(options={"random_state": 42, "n_center_points": 3, "n_blocks": 2})),
        "The Runs sheet is missing column(s): Block.",
    ),
    "replicates out of range": (
        "api_make_design",
        _make(n_replicates=0),
        "Replicates must be a whole number from 1 to 10.",
    ),
    "replicates not a whole number": (
        "api_make_design",
        _make(n_replicates=1.5),
        "Replicates must be a whole number from 1 to 10.",
    ),
    "a space-filling design with too few runs to fit the main effects": (
        "api_make_design",
        _make(design_type="maximin_lhs", budget=3),
        "A space-filling design in 3 factors has 4 to 200 runs.",
    ),
    "a space-filling design with runs not a whole number": (
        "api_make_design",
        _make(design_type="maximin_lhs", budget=12.5),
        "A space-filling design in 3 factors has 4 to 200 runs.",
    ),
    "an optimal design with fewer runs than its model has coefficients": (
        "api_make_design",
        _make(design_type="d_optimal", model_type="quadratic", budget=6),
        "The quadratic model in 3 factors has 10 coefficients, so it needs at least 10 runs.",
    ),
    "unknown model to design for": (
        "api_make_design",
        _make(design_type="d_optimal", model_type="cubic"),
        "Unknown model 'cubic'.",
    ),
    "negative center points": (
        "api_make_design",
        _make(n_center_points=-1),
        "The number of center points cannot be negative.",
    ),
    "no center point": (
        "api_make_design",
        _make(n_center_points=0),
        (
            "This design needs at least one center point: without one every other run lies on one sphere around "
            "the center, so the squared terms cannot be told apart from the intercept."
        ),
    ),
    "resolution above the factor count": (
        "api_make_design",
        _make(design_type="fractional_factorial", resolution=4),
        "A fractional factorial in 3 factors can have resolution III on this page, not IV.",
    ),
    "resolution above V": (
        "api_make_design",
        lambda _: {"design_type": "fractional_factorial", "factors": factors_named(*"ABCDEF"), "resolution": 6},
        "A fractional factorial in 6 factors can have resolution III to V on this page, not VI.",
    ),
    "resolution V past the factors whose interactions the page models": (
        "api_make_design",
        lambda _: {"design_type": "fractional_factorial", "factors": _many(13), "resolution": 5},
        "A fractional factorial in 13 factors can have resolution III to IV on this page, not V.",
    ),
    "a fraction past 24 factors": (
        "api_make_design",
        lambda _: {"design_type": "fractional_factorial", "factors": _many(25)},
        "The Fractional factorial (2-level) design takes 3 to 24 factors on this page; you have 25.",
    ),
    "a screening design past 40 factors": (
        "api_make_design",
        lambda _: {"design_type": "plackett_burman", "factors": _many(41)},
        "The Plackett-Burman screening design takes 2 to 40 factors on this page; you have 41.",
    ),
    "the interactions of more than 12 factors": (
        "api_analyze_runs",
        lambda _: _screening(13, model="interactions"),
        (
            "Past 12 factors, the page fits the main effects, or the terms you choose: 13 factors have 78 two-factor "
            "interactions."
        ),
    ),
    "a comparison past 12 factors aimed at more than main effects": (
        "api_candidates",
        lambda _: {"factors": _many(13), "model": "main_quadratic"},
        (
            "Past 12 factors, a study aims to find which factors matter, and the designs are compared for their main "
            "effects: 13 factors have 78 two-factor interactions."
        ),
    ),
    # The runs the page sends back.
    "a design the page did not make": (
        "api_simulate",
        _spec(design_type="rocket"),
        "The page sent a design this page did not make.",
    ),
    "response that is not a name": (
        "api_analyze_runs",
        lambda ex: {**_spec(response="__import__('os')")(ex), "responses": ex["simulated"]["responses"]},
        "The response name must be a variable name (letters, digits, _): try 'import_os'.",
    ),
    "unknown model to analyse with": (
        "api_analyze_runs",
        _runs(model="y ~ T"),
        "Model must be one of main_effects, interactions, quadratic.",
    ),
    "no terms chosen": (
        "api_analyze_runs",
        _runs(model={"terms": []}),
        "Choose at least one term for the model.",
    ),
    "a chosen term of another factor": (
        "api_analyze_runs",
        _runs(model={"terms": [["T"], ["Z"]]}),
        "A model term can name only the design's factors, T, P and F, not ['Z'].",
    ),
    "a chosen term that is no term": (
        "api_analyze_runs",
        _runs(model={"terms": [["T", "P", "T"]]}),
        "A model term is a factor, an interaction of two or three different factors, or a factor squared.",
    ),
    "a workbook analysed with a formula": (  # a workbook keeps its analysis model, which is checked as data
        "api_read_workbook",
        _workbook(_stored(analysis_model="y ~ __import__('os').getcwd()")),
        "Model must be one of main_effects, interactions, quadratic.",
    ),
    "kept settings that are not settings": (
        "api_workbook",
        lambda ex: {**ex["held"], "what_if": [{"name": "Hot", "settings": {"T": [200]}}]},
        "The page sent kept settings this page did not write.",
    ),
    "a workbook keeping two settings under one name": (  # edited by hand: its kept settings are checked as data
        "api_read_workbook",
        _workbook(_stored(what_if=[{"name": "Hot", "settings": {"T": 200}}, {"name": "Hot", "settings": {"T": 150}}])),
        "The page sent kept settings this page did not write.",
    ),
    "a transform the page does not offer": (
        "api_analyze_runs",
        lambda ex: {
            **_runs()(ex),
            "spec": {**ex["held"]["spec"], "analysis_transforms": {ex["held"]["spec"]["response"]: "box_cox"}},
        },
        "A response is analysed as it is, or as its log, square root or inverse.",
    ),
    "a workbook analysed with a transform it does not offer": (  # a workbook keeps its transforms, checked as data
        "api_read_workbook",
        _workbook(_stored(analysis_transforms={"y": "__import__('os')"})),
        "A response is analysed as it is, or as its log, square root or inverse.",
    ),
    "a result at or below 0 for its log": (
        "api_analyze_runs",
        lambda ex: {
            **_runs(responses=[-1.0] * len(ex["held"]["rows"]))(ex),
            "spec": {**ex["held"]["spec"], "analysis_transforms": {ex["held"]["spec"]["response"]: "log"}},
        },
        (
            "The log of Yield needs every result above 0, and run 1 has -1. "
            "Analyse it as it is, or with another transform."
        ),
    ),
    "an acceptable range below 0 for a logged response": (
        "api_balance",
        lambda ex: {
            **_balanced(goals={"Cost": {"low": 0, "high": 30}})(ex),
            "spec": {**_balanced()(ex)["spec"], "analysis_transforms": {"Cost": "log"}},
        },
        "The log of Cost needs an acceptable range above 0.",
    ),
    "a target at 0 for its log": (
        "api_best_settings",
        lambda ex: {
            **_runs()(ex),
            "spec": {
                **ex["held"]["spec"],
                "response_info": {**ex["held"]["spec"].get("response_info", {}), "goal": "target", "target": 0},
                "analysis_transforms": {ex["held"]["spec"]["response"]: "log"},
            },
        },
        "The log of Yield needs a target above 0.",
    ),
    "a mixture's response transformed": (
        "api_analyze_runs",
        lambda ex: {
            **_mixed()(ex),
            "spec": {**ex["mixture"]["spec"], "analysis_transforms": {ex["mixture"]["spec"]["response"]: "log"}},
        },
        "A mixture's responses are analysed as they are on this page.",
    ),
    "no term the design can estimate": (
        "api_analyze_runs",
        _runs(model={"terms": [["T", "P", "F"]]}),  # zero in every run of a Box-Behnken design
        "None of the model's terms can be estimated from this design.",
    ),
    "unknown model to chart": (
        "api_plots",
        _runs(model="y ~ T"),
        "Model must be one of main_effects, interactions, quadratic.",
    ),
    "missing factor column": (
        "api_analyze_runs",
        _runs(columns=["Std order", "Run order", "P", "F", "y"]),
        "The run sheet is missing the factor column(s) ['T'].",
    ),
    "no runs": ("api_workbook", _runs(rows=[]), "No runs."),
    "responses for other runs": (
        "api_analyze_runs",
        _runs(responses=[1.0, 2.0]),
        "2 responses were sent for 15 runs.",
    ),
    "sources for other runs": (
        "api_workbook",
        _runs(sources=["simulated", "entered"]),
        "2 sources were sent for 15 runs.",
    ),
    "unknown process": ("api_simulate", _runs(scenario="rocket"), "Unknown process 'rocket'."),
    # The best settings, and the confirmation runs made at them.
    "every run a confirmation run": (
        "api_analyze_runs",
        lambda ex: {
            **_runs()(ex),
            "columns": [*ex["held"]["columns"], "Confirmation run"],
            "rows": [[*row, "yes"] for row in ex["held"]["rows"]],
        },
        "Every run on the run sheet is marked as a confirmation run, and the model is fitted to the design's runs.",
    ),
    "best settings before any result": (
        "api_best_settings",
        _runs(responses=[None] * 15),
        "Enter results first: the best settings are found on the model fitted to them.",
    ),
    "best blend before any result": (
        "api_best_settings",
        lambda ex: {
            **_mixed()(ex),
            "spec": {**ex["mixture"]["spec"], "response_info": {"goal": "maximize"}},
            "responses": [None] * len(ex["mixture"]["rows"]),
        },
        "Enter results first: the best blend is found on the model fitted to them.",
    ),
    "a blend's charts drawn through a best blend the page did not write": (
        "api_plots",
        _mixed(best={"A": 0.5, "B": 0.6, "C": 0.0}),
        "The page sent a best blend this page did not write.",
    ),
    "a blocked design's model to simplify": (
        "api_simpler_model",
        lambda ex: {
            **ex["held"],
            "spec": {**ex["held"]["spec"], "options": {**ex["held"]["spec"].get("options", {}), "n_blocks": 3}},
            "responses": ex["simulated"]["responses"],
        },
        (
            "The page simplifies a model of numeric and two-level factors, in runs that are not blocked: "
            "choose this study's terms yourself, under Terms you choose."
        ),
    ),
    "a model that uses every run, to simplify": (  # ten coefficients, and ten runs with a result
        "api_simpler_model",
        lambda ex: {**ex["held"], "responses": [*ex["simulated"]["responses"][:10], *[None] * 5]},
        (
            "The model uses every run, so none is left to compare a simpler model with: "
            "judge the effects by Lenth's method, in the table of effects."
        ),
    ),
    "an OMARS design of too few runs": (
        "api_make_design",
        lambda _: {"design_type": "omars", "factors": FACTORS, "budget": 9},
        "An OMARS design in 3 factors has 11 to 41 runs on this page.",
    ),
    "no table to read": (
        "api_read_table",
        lambda _: {"text": "  "},
        "There is no table to read: paste one, or choose a .csv or .xlsx file.",
    ),
    "a table of headings alone": (
        "api_read_table",
        lambda _: {"text": "A,B,y\n1,2,3\n"},
        "The table needs a line of headings and at least two runs below it.",
    ),
    "a table too large to read": (
        "api_read_table",
        lambda _: {"text": "A,B\n" + "1,2\n" * 600_000},
        "The table is larger than this page reads: 2 MB at most.",
    ),
    "a file too large to read": (
        "api_read_table",
        lambda _: {"xlsx_base64": "A" * 2_800_000},
        "The file is larger than this page reads: 2 MB at most.",
    ),
    "a file that is no workbook": (
        "api_read_table",
        lambda _: {"xlsx_base64": "bm90IGEgd29ya2Jvb2s="},
        "The file could not be read as an Excel workbook (.xlsx).",
    ),
    "a quote never closed": (
        "api_read_table",
        lambda _: {"text": 'A,B,y\n1,"2,3\n' + "4,5,6\n" * 30_000},
        "The table could not be split into columns: look for a quote that is opened and never closed.",
    ),
    "a workbook that unpacks to far more than it weighs": (
        "api_read_table",
        lambda _: {"xlsx_base64": _zipped(60_000_000)},
        "The workbook unpacks to more than this page reads: 50 MB.",
    ),
    "a table of too many columns": (
        "api_read_table",
        lambda _: {"text": ",".join(f"c{i}" for i in range(61)) + "\n" + ("1," * 61 + "\n") * 2 + ("2," * 61 + "\n")},
        "The table has 61 columns; this page reads 60 at most.",
    ),
    "a table of too many runs": (
        "api_read_table",
        lambda _: {"text": "A,B\n" + "".join(f"{i},{i % 3}\n" for i in range(2001))},
        "The table has 2001 runs; a study holds 2000 at most.",
    ),
    "two columns of one heading": (
        "api_read_table",
        lambda _: {"text": "A,A,y\n1,2,3\n2,3,4\n3,4,5\n"},
        "Two columns share the heading A: give each column its own.",
    ),
    "a table the page did not read": (
        "api_import_study",
        lambda _: {"columns": [{"role": "factor"}], "rows": [[1, 2], [3, 4]]},
        "The page sent a table this page did not read.",
    ),
    "a table with no response chosen": (
        "api_import_study",
        lambda _: {"columns": [{"role": "factor", "name": "A"}, {"role": "ignore"}], "rows": [[1, 2], [2, 4], [1, 5]]},
        "A study needs at least one factor and one response: choose which columns they are.",
    ),
    "a table whose runs lack a factor's setting": (
        "api_import_study",
        lambda _: {
            "columns": [{"role": "factor", "name": "A"}, {"role": "response", "name": "y"}],
            "rows": [[1, 2], [None, 4], [None, 5]],
        },
        "Fewer than two runs have a setting for every factor: leave out a factor with blanks.",
    ),
    "a factor at one setting": (
        "api_import_study",
        lambda _: {
            "columns": [{"role": "factor", "name": "A"}, {"role": "response", "name": "y"}],
            "rows": [[1, 2], [1, 4], [1, 5]],
        },
        "A has one setting in every run, so these runs say nothing of it: leave it out.",
    ),
    "a factor of too many labels": (
        "api_import_study",
        lambda _: {
            "columns": [{"role": "factor", "name": "Lot"}, {"role": "response", "name": "y"}],
            "rows": [[f"lot {i}", i] for i in range(9)],
        },
        (
            "Lot has 9 different labels, and a categorical factor has 2 to 8 levels: "
            "leave it out, or make its column numbers."
        ),
    ),
    "a response of text": (
        "api_import_study",
        lambda _: {
            "columns": [{"role": "factor", "name": "A"}, {"role": "response", "name": "y"}],
            "rows": [[1, 2], [2, "high"], [1, 5]],
        },
        "The response y holds text in some runs: a response is a number.",
    ),
    "a surface over one factor twice": (
        "api_plots",
        _runs(contour_factors=["T", "T"]),
        "A response surface is drawn over two different factors of ['T', 'P', 'F'].",
    ),
    "a surface held at best settings the page did not write": (
        "api_plots",
        _runs(best={"T": 1.0, "P": -1.0, "F": 3.0}),
        "The page sent best settings this page did not write.",
    ),
    "no noise": ("api_simulate", _runs(sigma=0), "The noise standard deviation must be a number above 0."),
    "no design for these factors": (  # 41 factors, one more than any design on the page takes
        "api_candidates",
        lambda _: {
            "factors": [
                *({"name": f"X{i}", "low": -1, "high": 1} for i in range(40)),
                {"name": "S", "low": "A", "high": "B"},
            ],
            "model": "main_effects",
        },
        "None of the designs this page offers can be made for these factors.",
    ),
    "a true process over other factors": (
        "api_analyze_runs",
        _runs(truth=[{"term": "Z", "coefficient": 1.0}]),
        "The true process names factors the design does not have: ['Z'].",
    ),
    "too few completed runs": (
        "api_analyze",
        _workbook(_blank(8), model="quadratic"),
        (
            "The quadratic model has 10 coefficients, so it needs at least 10 completed runs, and 7 are complete. "
            "Still to run: 1, 2, 3, 4, 5, 6, 7, 8."
        ),
    ),
    "a design too small for the model": (
        "api_analyze_runs",
        lambda ex: {**ex["dsd"], "responses": [float(i) for i in range(len(ex["dsd"]["rows"]))], "model": "quadratic"},
        (
            "This design cannot fit the quadratic model: only 8 of its 10 coefficients can be estimated from these "
            "9 runs. Choose a smaller model."
        ),
    ),
    # Workbooks.
    "not a workbook": ("api_read_workbook", _foreign_file, "That file could not be read as an .xlsx workbook."),
    "a workbook from elsewhere": (
        "api_read_workbook",
        _workbook(lambda sheets: sheets.pop("_pi_design")),
        "This workbook was not made by this page: it needs the 'Runs' sheet and its design.",
    ),
    "a workbook from another version": (
        "api_read_workbook",
        _workbook(_stored(version=0)),
        "This workbook was made by a different version of the page.",
    ),
    "a workbook without its results column": (
        "api_read_workbook",
        _workbook(lambda sheets: sheets["Runs"].pop("Yield")),
        "The Runs sheet has no 'Yield' column for the results.",
    ),
    "an uploaded workbook without a factor": (
        "api_analyze",
        _workbook(lambda sheets: sheets["Runs"].pop("T")),
        "The Runs sheet is missing column(s): T.",
    ),
    # Renaming the factors of a run sheet.
    "a rename that moves a level": (
        "api_relabel",
        _renamed([{**FACTORS[0], "name": "Temp", "high": 210}, *FACTORS[1:]]),
        "Only the factors' names and units can change without making the run sheet again.",
    ),
    "a factor renamed as the results column": (
        "api_relabel",
        _renamed([{**FACTORS[0], "name": "Yield"}, *FACTORS[1:]]),
        "A factor cannot be named Yield: the run sheet has a Yield column.",
    ),
    # Comparisons.
    "unknown model to compare on": (
        "api_compare_plan",
        lambda _: {"factors": FACTORS, "model": "cubic"},
        "Model must be one of main_effects, interactions, main_quadratic, quadratic.",
    ),
    "unknown model for the comparison": (
        "api_compare",
        lambda _: {"model": "cubic", "designs": []},
        "Model must be one of main_effects, interactions, main_quadratic, quadratic.",
    ),
    "nothing to compare": (
        "api_compare",
        lambda _: {"model": "main_effects", "designs": []},
        "Give at least one design to compare.",
    ),
    # Limits on the settings, as the reader types them: the panel shows each limit's problem beside it.
    "an empty limit": (
        "api_limits",
        _typed(""),
        "Write each limit as an inequality of your factors, such as 2*T + P <= 10.",
    ),
    "a product without its *": (
        "api_limits",
        _typed("2T + P <= 300"),
        "The limit '2T + P <= 300' cannot be read: write 2*T for 2 times T.",
    ),
    "an equals sign for a limit": (
        "api_limits",
        _typed("T + P = 200"),
        "The limit 'T + P = 200' cannot be read: write <= for at most, and >= for at least.",
    ),
    "an unclosed bracket": (
        "api_limits",
        _typed("T + (P <= 3"),
        "The limit 'T + (P <= 3' cannot be read: '(' was never closed.",
    ),
    "an unfinished limit": (
        "api_limits",
        _typed("T + P <"),
        "The limit 'T + P <' cannot be read: check it for a missing number, factor or bracket.",
    ),
    "a decimal comma": (
        "api_limits",
        _typed("0,5*T + P <= 100"),
        "The limit '0,5*T + P <= 100' cannot be read: write a decimal number with a point, as in 0.5, not 0,5.",
    ),
    "a number before a bracket": (
        "api_limits",
        _typed("2(T + P) <= 300"),
        "The limit '2(T + P) <= 300' cannot be read: write 2*(T + P) for 2 times (T + P).",
    ),
    "two limits in one": (
        "api_limits",
        _typed("T <= 190, P <= 2.5"),
        "The limit 'T <= 190, P <= 2.5' holds 2 comparisons: type each as a limit of its own.",
    ),
    "a limit with or": (
        "api_limits",
        _typed("T <= 160 or P <= 1.5"),
        "The limit 'T <= 160 or P <= 1.5' joins comparisons with or, which a limit cannot: each limit must hold.",
    ),
    "a sum with no comparison": (
        "api_limits",
        _typed("T + 25*P"),
        "The limit 'T + 25*P' is not an inequality: write it as one, such as 2*T + P <= 10.",
    ),
    "a sum with no comparison, in a brief of one numeric factor": (
        "api_limits",
        _typed("T + 25", factors=[FACTORS[0], WITH_SUPPLIER[2]]),
        "The limit 'T + 25' is not an inequality: write it as one, such as 2*T <= 10.",
    ),
    "an equality": (
        "api_limits",
        _typed("T == 175"),
        "The limit 'T == 175' must compare with <, <=, > or >=, not with ==.",
    ),
    "a function a limit cannot use": (
        "api_limits",
        _typed("sin(T) <= 0.5"),
        f"The limit 'sin(T) <= 0.5' uses sin, which a limit cannot: the functions it can use are {FUNCTIONS}.",
    ),
    "an operator a limit cannot use": (
        "api_limits",
        _typed("T % 7 <= 3"),
        (
            "The limit 'T % 7 <= 3' cannot use 'T % 7': a limit is written with numbers, your factors, + - * / ** "
            f"and the functions {FUNCTIONS}."
        ),
    ),
    "a limit on a categorical factor": (
        "api_limits",
        _typed("T + S <= 3", factors=WITH_SUPPLIER),
        "The limit 'T + S <= 3' names S, which is categorical: limits are on numeric factors only.",
    ),
    "a limit on another factor": (
        "api_limits",
        _typed("T + Z <= 300"),
        "The limit 'T + Z <= 300' names Z, which is not one of your factors: T, P and F.",
    ),
    "a limit on no factor": ("api_limits", _typed("3 <= 5"), "The limit '3 <= 5' names none of your factors."),
    "a limit too long": (
        "api_limits",
        _typed("T + " * 130 + "P <= 3"),
        "The limit 'T + T + T + T + T + T + ...' is longer than 500 characters.",
    ),
    "limits with a categorical factor": (
        "api_limits",
        _typed("T <= 190", factors=WITH_SUPPLIER),
        (
            "Limits on the settings work with numeric factors only on this page, and S is categorical: remove the "
            "limits, or that factor."
        ),
    ),
    "limits in more factors than an optimal design takes": (
        "api_limits",
        _typed("X0 + X1 <= 1", factors=factors_named(*(f"X{i}" for i in range(9)))),
        "With limits on the settings, the page makes designs in up to 8 factors; you have 9.",
    ),
    "a limit on one factor alone": (
        "api_make_design",
        lambda _: {
            "design_type": "full_factorial",
            "factors": FACTORS[:1],
            "constraints": [{"expression": "T <= 190"}],
        },
        "A limit on T alone narrows its range: set its low and high to the settings you can run, and remove the limit.",
    ),
    "limits that leave nothing": (
        "api_make_design",
        _limited("d_optimal", "T + P >= 300"),
        "No setting in your ranges meets every limit: loosen a limit, or widen a range.",
    ),
    "limits with a design that cannot keep to them": (
        "api_make_design",
        _limited("box_behnken", "T + 25*P <= 250"),
        (
            "The Box-Behnken design cannot keep its runs inside limits on the settings: choose the D-optimal or "
            "I-optimal design, or remove the limits."
        ),
    ),
    # What only a script, or a doctored workbook, could send.
    "limits the page did not write": (
        "api_make_design",
        lambda _: {"design_type": "d_optimal", "factors": FACTORS, "constraints": "T <= 190"},
        "The page sent limits this page did not write.",
    ),
    "too many limits": (
        "api_limits",
        _typed(*(f"T <= {190 + i}" for i in range(9))),
        "A study can have at most 8 limits on its settings.",
    ),
    "a note too long": (
        "api_limits",
        lambda _: {"factors": FACTORS, "constraints": [{"expression": "T <= 190", "note": "x" * 81}]},
        "A limit's note is at most 80 characters, on one line.",
    ),
    "a marked row the table does not have": (
        "api_limits",
        _typed(marked=[9]),  # the worked example's table has 8 corners and the centre, rows 0 to 8
        "The page sent marked settings this page did not write.",
    ),
    "cuts from the centre, which is no corner": (
        "api_limits",
        _typed(marked=[8], cuts={"marked": [8], "corner": 8, "values": [190, None, None]}),
        "The page sent marked settings this page did not write.",
    ),
    "a slice along one factor twice": (
        "api_limits",
        _typed("T <= 190", slice={"x": "T", "y": "T", "at": {}}),
        "The page sent a slice of the settings this page did not write.",
    ),
    # Runs as given, which the design catalog opens: they must fit the factors.
    "given: no runs": (
        "api_make_design",
        _make(design_type="given", runs=[]),
        "The runs as given are not there: choose a design to make, or open the runs again.",
    ),
    "given: a run short of a factor": (
        "api_make_design",
        _make(design_type="given", runs=[[150, 1]]),
        "Run 1 does not give one setting for each of the 3 factors.",
    ),
    "given: a word for a number": (
        "api_make_design",
        _make(design_type="given", runs=[[150, 1, "fast"]]),
        "Factor F: each run's setting must be a number.",
    ),
    "given: too many runs": (
        "api_make_design",
        _make(design_type="given", runs=[[150, 1, 10]] * 2001),
        "Runs as given hold at most 2000 runs; these are 2001.",
    ),
    "given: too many factors": (
        "api_make_design",
        _make(
            design_type="given",
            factors=[{"name": f"X{i}", "low": 0, "high": 1} for i in range(41)],
            runs=[[0] * 41],
        ),
        "Runs as given take 1 to 40 factors on this page; you have 41.",
    ),
    "given: a block short": (
        "api_make_design",
        _make(design_type="given", runs=[[150, 1, 10], [200, 3, 20]], blocks=[1]),
        "The runs as given need one block for each run, or none at all.",
    ),
    "given: a block that is no whole number": (
        "api_make_design",
        _make(design_type="given", runs=[[150, 1, 10], [200, 3, 20]], blocks=[1, 1.5]),
        "Each run's block must be a whole number from 1.",
    ),
    "given: too many blocks": (
        "api_make_design",
        _make(design_type="given", runs=[[150, 1, 10]] * 66, blocks=[b for b in range(1, 34) for _ in (0, 1)]),
        "Runs as given come in at most 32 blocks on this page.",
    ),
    "given: a block for each run": (
        "api_make_design",
        _make(design_type="given", runs=[[150, 1, 10]] * 9, blocks=list(range(1, 10))),
        "Each run is in a block of its own, so the blocks leave nothing to compare: give the runs in fewer blocks.",
    ),
    # Runs added to a study, as Learn offers them.
    "add runs: an unknown way": (
        "api_add_runs",
        lambda ex: {**ex["held"], "kind": "teleport"},
        "Unknown way to add runs 'teleport'.",
    ),
    "add runs: a way the design does not take": (
        "api_add_runs",
        lambda ex: {**ex["held"], "kind": "foldover"},
        (
            "This study's runs cannot be extended that way: a fold-over takes a two-level design whose main effects "
            "are aliased with two-factor interactions."
        ),
    ),
    "add runs: outside the limits": (
        "api_add_runs",
        lambda _: {**HALF_FRACTION, "kind": "foldover"},
        "The runs of a fold-over would fall outside your limits on the settings.",
    ),
    "add runs: none called for": (
        "api_add_runs",
        _runs(kind="recommended"),  # the worked example's results call for a new study, not runs added to this one
        "The results call for no runs to add to this study: see what the next step says.",
    ),
    "add runs: a mixture": (
        "api_add_runs",
        _mixed(kind="replicate"),
        "A mixture's runs are blends, and the runs Learn adds to a design are not.",
    ),
    # Mixtures: the factors are the components of a blend, each a proportion.
    "mixture: a categorical factor": (
        "api_make_design",
        _blend({"name": "S", "low": "old", "high": "new", "units": ""}),
        (
            "A mixture's factors are its components, each a proportion, and S is categorical: remove it, or choose "
            "a design for factors with ranges."
        ),
    ),
    "mixture: a process variable": (
        "api_make_design",
        _blend(FACTORS[0]),
        (
            "Factor T runs from 150 to 200, which is no proportion: each component of a mixture lies between 0 and "
            "1, and a mixture on this page takes no process variables, such as a temperature."
        ),
    ),
    "mixture: a word for a bound": (
        "api_make_design",
        _blend({"name": "A", "high": "most"}),
        "Component A: enter its lower and upper proportions as numbers, or leave them blank for 0 and 1.",
    ),
    "mixture: a lower bound above the upper": (
        "api_make_design",
        _blend({"name": "A", "low": 0.6, "high": 0.4}, design_type="extreme_vertices"),
        "Component A: its lower proportion must be below its upper proportion.",
    ),
    "mixture: too few components": (
        "api_make_design",
        lambda _: {"design_type": "simplex_lattice", "factors": COMPONENTS[:2]},
        "The Simplex lattice (mixture) design takes 3 to 6 components on this page; you have 2.",
    ),
    "mixture: too many components": (
        "api_make_design",
        lambda _: {"design_type": "simplex_centroid", "factors": [{"name": n, "low": 0, "high": 1} for n in "ABCDE"]},
        "The Simplex centroid (mixture) design takes 3 to 4 components on this page; you have 5.",
    ),
    "mixture: lower bounds above 1 in all": (
        "api_make_design",
        _blend({"name": "A", "low": 0.5}, {"name": "B", "low": 0.4}, {"name": "C", "low": 0.3}),
        (
            "The lower proportions sum to 1.2, and must sum to less than 1: above 1 no blend gives every component "
            "its lower proportion, and at 1 only one blend does. Lower some of them."
        ),
    ),
    "mixture: upper bounds below 1 in all": (
        "api_make_design",
        _blend({"name": "A", "high": 0.3}, {"name": "B", "high": 0.3}, {"name": "C", "high": 0.3}),
        (
            "The upper proportions sum to 0.9, and must sum to more than 1: below 1 no blend of the components "
            "reaches 1, and at 1 only one does. Raise some of them."
        ),
    ),
    "mixture: a simplex design with bounds": (
        "api_make_design",
        _blend({"name": "A", "low": 0.1}, {"name": "C", "high": 0.8}),
        (
            "The Simplex lattice (mixture) design covers every blend, each component from 0 to 1, and A and C have "
            "bounds: choose extreme vertices for a mixture with bounds."
        ),
    ),
    "mixture: extreme vertices without a bound": (
        "api_make_design",
        _blend(design_type="extreme_vertices"),
        (
            "Extreme vertices are for a mixture with bounds: give a component a lower or upper proportion, or "
            "choose the simplex lattice or simplex centroid."
        ),
    ),
    "mixture: a lattice of degree 3": (
        "api_make_design",
        _blend(lattice_degree=3),
        "A simplex lattice's degree is 1 or 2 on this page.",
    ),
    "mixture: extreme vertices for the special cubic model": (
        "api_make_design",
        _blend({"name": "A", "high": 0.6}, design_type="extreme_vertices", mixture_model="scheffe_special_cubic"),
        "Extreme vertices are built for the linear or quadratic blending model, not 'scheffe_special_cubic'.",
    ),
    "mixture: limits on the settings": (
        "api_make_design",
        _blend(constraints=[{"expression": "A + B <= 0.8"}]),
        (
            "A mixture takes no limits on the settings on this page: give each component its lower and upper "
            "proportions instead, or remove the limits."
        ),
    ),
    "mixture: a model for factors with ranges": (
        "api_analyze_runs",
        _mixed(model="quadratic"),
        "A mixture is analysed with the linear, quadratic or special cubic blending model, not 'quadratic'.",
    ),
    "mixture: a process for factors with ranges": (
        "api_simulate",
        _mixed(scenario="reactor"),
        (
            "The process 'Reactor yield' is for factors with ranges: a mixture's results are simulated from 'A "
            "synergy and an antagonism'."
        ),
    ),
    "mixture: a mixture's process for factors with ranges": (
        "api_simulate",
        _runs(scenario="blend"),
        (
            "The process 'A synergy and an antagonism' is for a mixture's components: choose another for factors "
            "with ranges."
        ),
    ),
    "curves: a curve the page did not write": (
        "api_functional",
        _curved(curve={"label": "Colour"}),
        "The page sent a curve this page did not write.",
    ),
    "curves: a point along the curve that is no number": (
        "api_functional",
        _curved(curve={"argument": {"values": [0, "soon", 2]}}),
        "The page sent a curve this page did not write.",
    ),
    "curves: target settings that are not settings": (
        "api_functional",
        lambda ex: {**_curved()(ex), "target": [0, 1, 2], "held": "B"},
        "The page sent settings this page did not write.",
    ),
    "curves: points along the curve that do not rise": (
        "api_functional",
        _curved(curve={"argument": {"values": [0, 2, 1]}}),
        "A curve is read at 3 to 1000 points along it, each one after the point before.",
    ),
    "curves: for a study that measures none": (
        "api_functional",
        lambda ex: {**ex["held"], "curves": [None] * len(ex["held"]["rows"])},
        "This study measures no curve: describe the curve each run gives, then paste the curves.",
    ),
    "curves: of a mixture": (
        "api_functional",
        _curved("mixture"),
        "A mixture's curves are not analysed on this page yet.",
    ),
    "curves: fewer than the runs": (
        "api_functional",
        _curved(curves=[[0.0, 1.0, 2.0]]),
        "1 curves were sent for 15 runs.",
    ),
    "curves: a curve of the wrong length": (
        "api_functional",
        lambda ex: {**_curved()(ex), "curves": [[0.0, 1.0], *_curved()(ex)["curves"][1:]]},
        "Each run's curve holds 3 values, one per point along it, or blanks.",
    ),
    "curves: a target with no value": (
        "api_functional",
        _curved(target=[None, None, None]),
        "Enter the target curve: at least one value along it.",
    ),
    "curves: a target search at a level a factor lacks": (
        "api_functional",
        lambda ex: {**_curved("supplied")(ex), "model": "main_effects", "target": [0, 1, 2], "held": {"S": "Z"}},
        "S at Z is none of its levels, A and B.",
    ),
    "curves: a Curves sheet read at other points": (
        "api_read_workbook",
        _workbook(_curve_sheet(lambda runs: pd.DataFrame({"Run order": runs["Run order"], 0: 0.0, 1: 1.0, 5: 2.0}))),
        "The Curves sheet's columns are not the points along the curve this study reads it at.",
    ),
    "curves: a Curves sheet that lists a run twice": (
        "api_read_workbook",
        _workbook(_curve_sheet(lambda runs: pd.DataFrame({"Run order": 1, 0: 0.0, 1: [1.0] * len(runs), 2: 2.0}))),
        "The Curves sheet lists one run order twice: each run has one row there.",
    ),
    "curves: a Curves sheet without run orders, short of runs": (
        "api_read_workbook",
        _workbook(_curve_sheet(lambda _: pd.DataFrame({0: [0.0], 1: [1.0], 2: [2.0]}))),
        "The Curves sheet has 1 rows for 15 runs.",
    ),
    "correlation map: a mixture": (
        "api_correlations",
        _mixed(),
        "A mixture's components sum to one, so they move together by design: no map is drawn.",
    ),
    "correlation map: a factor of three levels": (
        "api_correlations",
        _spec(factors=[*FACTORS[:2], {"name": "S", "type": "categorical", "levels": ["A", "B", "C"]}]),
        "This map is for factors with ranges or of two levels, not for one of three levels or more.",
    ),
    "correlation map: more factors than the page models the interactions of": (
        "api_correlations",
        lambda _: _screening(13),
        "This map is drawn for 2 to 12 factors, and the study has 13.",
    ),
}


@pytest.fixture(scope="module")
def example(worked_example: dict[str, Any], api) -> Example:
    """Return the worked example with its workbook, a 9-run DSD that cannot fit the quadratic model, and a mixture.

    The mixture is a simplex lattice in three components, with results simulated from a blend.
    """
    dsd = api.ok("api_make_design", {"design_type": "dsd", "factors": FACTORS, "n_center_points": 1})
    lattice = api.ok("api_make_design", {"design_type": "simplex_lattice", "factors": COMPONENTS})
    mixture = {key: lattice[key] for key in ("spec", "columns", "rows")}
    responses = api.ok("api_simulate", mixture)["responses"]
    return {
        **worked_example,
        "workbook": workbook_of(api, worked_example["design"]),
        "dsd": {key: dsd[key] for key in ("spec", "columns", "rows")},
        "mixture": {**mixture, "responses": responses},
    }


def refusal(api, example: Example, case: str) -> str:
    """Return the message the page shows for one case of ``REFUSALS``.

    ``api_limits`` describes a limit that cannot be read, or limits no design can keep to, and
    goes on describing the others, so its message is the reply's ``refusal``, which holds the
    brief back; only a request the page did not write is refused outright.
    """
    fn, request, _ = REFUSALS[case]
    if fn == "api_plots":
        pytest.importorskip("plotly", reason="the page installs plotly for its first chart")
    answer = api.reply(fn, request(example))
    if fn == "api_limits" and answer["ok"]:
        assert answer["result"]["refusal"], "api_limits described limits it should hold the brief back for"
        return answer["result"]["refusal"]
    assert not answer["ok"], f"{fn} accepted a request it should refuse"
    return answer["error"]


@pytest.mark.parametrize("case", REFUSALS)
def test_each_refusal_says_what_is_wrong(api, example, case):
    assert refusal(api, example, case) == REFUSALS[case][2]


def _messages(node: ast.expr) -> list[ast.expr]:
    """List the messages a ``raise ValueError(...)`` can carry: both sides of a conditional message."""
    if isinstance(node, ast.IfExp):
        return [*_messages(node.body), *_messages(node.orelse)]
    return [node]


def _pattern(node: ast.expr) -> re.Pattern:
    """Match the text a message makes: an f-string's fields match anything."""
    if isinstance(node, ast.Constant):
        return re.compile(re.escape(str(node.value)))
    if isinstance(node, ast.JoinedStr):
        parts = [re.escape(str(v.value)) if isinstance(v, ast.Constant) else ".*?" for v in node.values]
        return re.compile("".join(parts), re.DOTALL)
    message = f"web/bootstrap.py raises ValueError with {ast.unparse(node)}, which this test cannot read"
    raise AssertionError(message)


def refusals_in_bootstrap() -> dict[str, re.Pattern]:
    """Every message ``web/bootstrap.py`` raises ``ValueError`` with: its source, and the pattern its text matches."""
    tree = ast.parse((WEB / "bootstrap.py").read_text(encoding="utf-8"))
    return {
        ast.unparse(message): _pattern(message)
        for node in ast.walk(tree)
        if isinstance(node, ast.Raise)
        and isinstance(node.exc, ast.Call)
        and getattr(node.exc.func, "id", None) == "ValueError"
        for message in _messages(node.exc.args[0])
    }


def test_a_model_term_from_the_library_is_checked_before_it_enters_a_formula(bootstrap):
    assert bootstrap._checked_term("I(T ** 2)", ["T"]) == "I(T ** 2)"
    with pytest.raises(ValueError, match="Unexpected model term"):
        bootstrap._checked_term("os.system", ["T"])


def test_every_refusal_bootstrap_can_make_has_a_case_here(api, bootstrap, example):
    """A ``raise ValueError`` added to ``web/bootstrap.py`` must come with a case in ``REFUSALS``."""
    shown = [refusal(api, example, case) for case in REFUSALS]
    # A guard on what the library returns, which no request reaches: its message is read off a direct call.
    with pytest.raises(ValueError, match="Unexpected model term") as guard:
        bootstrap._checked_term("os.system", ["T"])
    shown.append(str(guard.value))
    patterns = refusals_in_bootstrap()
    assert len(patterns) >= 25, "the reading of web/bootstrap.py found its refusals"
    untested = [text for text, p in patterns.items() if not any(p.fullmatch(message) for message in shown)]
    assert not untested, f"no case in REFUSALS makes the page show: {untested}"
