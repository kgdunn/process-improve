"""What the page tells a reader whose request it refuses.

Every refusal is a sentence the page shows as it is, so it must say what is wrong in the
reader's terms: which factor, which value, what to do instead. ``REFUSALS`` asks for each one
the way a reader (or a workbook) could, and the last test reads ``web/bootstrap.py`` for every
message it can refuse with, so a new refusal cannot ship without a case here.
"""

from __future__ import annotations

import ast
import base64
import json
import re
from typing import TYPE_CHECKING, Any

import pytest

from tests.web._support import FACTORS, WEB, edit_workbook, factors_named, fill_workbook, quadratic

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


def _spec(**change: object) -> Callable[[Example], dict]:
    """Build a request whose design specification, sent back by the page, has been changed."""
    return lambda ex: {**ex["held"], "spec": {**ex["held"]["spec"], **change}}


def _workbook(edit: Callable[[dict], object], **request: object) -> Callable[[Example], dict]:
    """Build an upload of the worked example's workbook, filled in and then changed as a reader could in Excel."""

    def payload(ex: Example) -> dict:
        filled = fill_workbook(ex["design"]["xlsx_base64"], quadratic)
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


def _supplied(**change: object) -> Callable[[Example], dict]:
    """Build a request about the runs of a design with a categorical factor, changed."""
    return lambda _: {**SUPPLIED, **change}


def _typed(*limits: str, factors: list[dict] = FACTORS, **change: object) -> Callable[[Example], dict]:
    """Build the request the limits panel sends as the reader types: the brief's factors, and these limits."""
    return lambda _: {"factors": factors, "constraints": [{"expression": text} for text in limits], **change}


def _limited(design_type: str, *limits: str) -> Callable[[Example], dict]:
    """Build a design request, in the worked example's factors, with limits on the settings."""
    sent = [{"expression": text} for text in limits]
    return lambda _: {"design_type": design_type, "factors": FACTORS, "constraints": sent}


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
    "one factor": ("api_make_design", _make(factors=FACTORS[:1]), "Enter at least two factors."),
    "repeated name": (
        "api_make_design",
        _make(factors=[FACTORS[0], {**FACTORS[1], "name": "T"}]),
        "Factor names must be unique and look like variable names (letters, digits, _).",
    ),
    "name with a space": (
        "api_make_design",
        _make(factors=[FACTORS[0], {**FACTORS[1], "name": "a b"}]),
        "Factor names must be unique and look like variable names (letters, digits, _).",
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
        "Factor S: its two labels must differ.",
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
            "fractional factorial or Plackett-Burman design."
        ),
    ),
    "a run at neither label": (
        "api_analyze_runs",
        _supplied(rows=[*SUPPLIED["rows"][:3], [4, 4, 200, "C", None]]),
        "Factor S: each run must be at A or B, not 'C'.",
    ),
    "the square of a categorical factor": (
        "api_analyze_runs",
        _supplied(model={"terms": [["T"], ["S", "S"]]}),
        "S is categorical, so it has no square: it is +1 at both its levels.",
    ),
    # The request itself, which only a script could get wrong.
    "a request that is not an object": (
        "api_make_design",
        lambda _: [1],
        "The page sent a request that is not a JSON object.",
    ),
    # The design's settings.
    "unknown design": ("api_make_design", _make(design_type="rocket"), "Unknown design type 'rocket'."),
    "response named as a factor": (
        "api_make_design",
        _make(response="T"),
        "The response name must be a variable name different from every factor.",
    ),
    "too few factors for the design": (
        "api_make_design",
        lambda _: {"design_type": "dsd", "factors": factors_named("A", "B")},
        "The Definitive screening (DSD) design takes 3 to 12 factors on this page; you have 2.",
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
    # The runs the page sends back.
    "a design the page did not make": (
        "api_simulate",
        _spec(design_type="rocket"),
        "The page sent a design this page did not make.",
    ),
    "response that is not a name": (
        "api_analyze_runs",
        lambda ex: {**_spec(response="__import__('os')")(ex), "responses": ex["simulated"]["responses"]},
        "The response name must be a variable name different from every factor.",
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
    "no noise": ("api_simulate", _runs(sigma=0), "The noise standard deviation must be a number above 0."),
    "no design for these factors": (
        "api_candidates",
        lambda _: {
            "factors": [
                *({"name": f"X{i}", "low": -1, "high": 1} for i in range(11)),
                {"name": "S", "low": "A", "high": "B"},
            ],
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
            factors=[{"name": f"X{i}", "low": 0, "high": 1} for i in range(13)],
            runs=[[0] * 13],
        ),
        "Runs as given take 2 to 12 factors on this page; you have 13.",
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
}


@pytest.fixture(scope="module")
def example(worked_example: dict[str, Any], api) -> Example:
    """Return the worked example, a 9-run DSD that cannot fit the quadratic model whatever its results, and a mixture.

    The mixture is a simplex lattice in three components, with results simulated from a blend.
    """
    dsd = api.ok("api_make_design", {"design_type": "dsd", "factors": FACTORS, "n_center_points": 1})
    lattice = api.ok("api_make_design", {"design_type": "simplex_lattice", "factors": COMPONENTS})
    mixture = {key: lattice[key] for key in ("spec", "columns", "rows")}
    responses = api.ok("api_simulate", mixture)["responses"]
    return {
        **worked_example,
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
