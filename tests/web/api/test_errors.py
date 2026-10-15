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
}


@pytest.fixture(scope="module")
def example(worked_example: dict[str, Any], api) -> Example:
    """Return the worked example, and a 9-run DSD that cannot fit the quadratic model whatever its results."""
    dsd = api.ok("api_make_design", {"design_type": "dsd", "factors": FACTORS, "n_center_points": 1})
    return {**worked_example, "dsd": {key: dsd[key] for key in ("spec", "columns", "rows")}}


def refusal(api, example: Example, case: str) -> str:
    """Return the message the page shows for one case of ``REFUSALS``."""
    fn, request, _ = REFUSALS[case]
    if fn == "api_plots":
        pytest.importorskip("plotly", reason="the page installs plotly for its first chart")
    return api.error(fn, request(example))


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
