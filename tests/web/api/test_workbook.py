"""The workbook: ``api_workbook`` writes the runs the page holds, ``api_read_workbook`` reads one back in.

Everything needed to analyse a campaign travels in the ``.xlsx`` file the reader keeps: the run
sheet, and a hidden sheet with the design's specification. A workbook can arrive by e-mail, so
what it stores is read as data and checked before anything uses it.
"""

from __future__ import annotations

import base64
import io
import json

import numpy as np
import pytest

from tests.web._support import (
    FACTORS,
    edit_workbook,
    factors_named,
    fill_workbook,
    held_of,
    quadratic,
    run_sheet,
    sheets_of,
)


# --------------------------------------------------------------------------- writing
def test_the_workbook_marks_simulated_results(api, held, simulated):
    sources = ["simulated"] * 14 + ["entered"]
    book = api.ok(
        "api_workbook", {**held, "responses": simulated["responses"], "sources": sources, "simulated": simulated}
    )
    sheets = sheets_of(book["xlsx_base64"])
    assert sheets["Runs"]["Result source"].fillna("").tolist() == ["simulated"] * 14 + [""]
    stored = json.loads(sheets["_pi_design"]["spec"].iloc[0])
    assert stored["simulated"] == {"scenario": "reactor", "sigma": 1.0, "seed": 1}
    np.testing.assert_allclose(sheets["Runs"]["y"], simulated["responses"])


def test_a_workbook_sent_without_sources_is_the_run_sheet_alone(api, held, simulated):
    book = api.ok("api_workbook", {**held, "responses": simulated["responses"]})
    assert list(sheets_of(book["xlsx_base64"])["Runs"].columns) == held["columns"]


def test_the_workbook_opens_on_its_run_sheet_with_the_design_hidden(design):
    openpyxl = pytest.importorskip("openpyxl")
    book = openpyxl.load_workbook(io.BytesIO(base64.b64decode(design["xlsx_base64"])))
    assert book.sheetnames == ["Runs", "_pi_design"]
    assert book["_pi_design"].sheet_state == "hidden"
    assert book["Runs"].freeze_panes == "A2", "the header row stays in view"


# --------------------------------------------------------------------------- reading back
def test_a_workbook_the_page_wrote_reads_back_as_the_page_held_it(api, held, simulated):
    responses = [*simulated["responses"][:13], 61.5, None]
    sources = ["simulated"] * 13 + ["entered", None]
    book = api.ok("api_workbook", {**held, "responses": responses, "sources": sources, "simulated": simulated})
    back = api.ok("api_read_workbook", book)
    assert back["columns"] == held["columns"]
    assert back["rows"] == held["rows"], "the runs, with the results column left blank"
    assert back["responses"] == responses
    assert back["sources"] == sources
    assert back["spec"] == {**held["spec"], "simulated": {"scenario": "reactor", "sigma": 1.0, "seed": 1}}


def test_a_fresh_workbook_reads_back_with_every_run_still_to_do(api, design):
    back = api.ok("api_read_workbook", {"xlsx_base64": design["xlsx_base64"]})
    assert back["spec"] == design["spec"]
    assert back["rows"] == design["rows"]
    assert back["responses"] == back["sources"] == [None] * design["n_runs"]


def test_results_typed_into_the_workbook_read_back_as_entered(api):
    design = api.ok("api_make_design", {"design_type": "box_behnken", "factors": FACTORS})
    back = api.ok(
        "api_read_workbook", {"xlsx_base64": fill_workbook(design["xlsx_base64"], quadratic, blank_rows=(0, 4))}
    )
    assert back["sources"] == [None if i in {0, 4} else "entered" for i in range(design["n_runs"])]
    typed = [v for v in back["responses"] if v is not None]
    np.testing.assert_allclose(typed, np.delete(quadratic(run_sheet(design)), [0, 4]))


def test_a_source_mark_without_a_result_is_not_a_result(api, held, simulated):
    """A reader who clears a simulated value in Excel leaves its mark behind: the run is still to do."""
    book = api.ok("api_workbook", {**held, "responses": simulated["responses"], "sources": ["simulated"] * 15})

    def clear_first(sheets):
        sheets["Runs"].loc[0, "y"] = np.nan

    back = api.ok("api_read_workbook", {"xlsx_base64": edit_workbook(book["xlsx_base64"], clear_first)})
    assert back["responses"][0] is None
    assert back["sources"] == [None] + ["simulated"] * 14


@pytest.mark.parametrize("named", ["factor", "response"])
def test_a_factor_or_response_named_source_keeps_its_values(api, named):
    """The column marking simulated results has a name no factor or response can take."""
    factors = factors_named("Source", "P", "F") if named == "factor" else FACTORS
    response = "Source" if named == "response" else "y"
    design = api.ok("api_make_design", {"design_type": "box_behnken", "factors": factors, "response": response})
    responses = [float(i) for i in range(design["n_runs"])]
    sources = ["simulated"] * design["n_runs"]
    book = api.ok("api_workbook", {**held_of(design), "responses": responses, "sources": sources})
    back = api.ok("api_read_workbook", book)
    assert (back["columns"], back["rows"]) == (design["columns"], design["rows"])
    assert back["responses"] == responses
    assert back["sources"] == sources


def test_a_workbook_marked_in_a_source_column_still_reads_its_marks(api, held, simulated):
    """Workbooks written before the marks column was renamed keep their simulated results marked."""
    book = api.ok("api_workbook", {**held, "responses": simulated["responses"]})

    def mark_as_before(sheets):
        sheets["Runs"]["Source"] = ["simulated"] * 14 + [None]

    back = api.ok("api_read_workbook", {"xlsx_base64": edit_workbook(book["xlsx_base64"], mark_as_before)})
    assert back["columns"] == held["columns"]
    assert back["sources"] == ["simulated"] * 14 + ["entered"]


@pytest.mark.parametrize("fn", ["api_read_workbook", "api_analyze"])
@pytest.mark.parametrize(
    ("factor", "model"),
    [("T", "__import__('os').system('true')"), ("print('run') or T", "quadratic")],
    ids=["model", "factor name"],
)
def test_code_in_a_workbook_is_refused_never_run(api, design, fn, factor, model):
    """A stored model is a name, never a formula; a stored factor name must be a variable name."""

    def tamper(sheets):
        stored = json.loads(sheets["_pi_design"]["spec"].iloc[0])
        stored["model"] = model
        stored["factors"][0]["name"] = factor
        sheets["_pi_design"]["spec"] = [json.dumps(stored)]
        sheets["Runs"] = sheets["Runs"].rename(columns={"T": factor}).assign(y=1.0)

    api.error(fn, {"xlsx_base64": edit_workbook(design["xlsx_base64"], tamper)})


@pytest.mark.parametrize("change", [{"design_type": "rocket"}, {"response": "T"}], ids=["design type", "response"])
def test_an_uploaded_workbook_must_hold_a_design_this_page_makes(api, design, change):
    def tamper(sheets):
        stored = json.loads(sheets["_pi_design"]["spec"].iloc[0])
        sheets["_pi_design"]["spec"] = [json.dumps({**stored, **change})]

    api.error("api_read_workbook", {"xlsx_base64": edit_workbook(design["xlsx_base64"], tamper)})


# --------------------------------------------------------------------------- analysing a workbook
def test_workbook_round_trip_recovers_the_model(api):
    design = api.ok("api_make_design", {"design_type": "box_behnken", "factors": FACTORS})
    result = api.ok("api_analyze", {"xlsx_base64": fill_workbook(design["xlsx_base64"], quadratic)})
    coef = {c["term"]: c["coefficient"] for c in result["coefficients"]}
    assert coef["T"] == pytest.approx(4, abs=0.2)
    assert coef["T:P"] == pytest.approx(2, abs=0.2)
    assert coef["I(T ** 2)"] == pytest.approx(-5, abs=0.3)
    assert result["equation"].startswith("y = ")


def test_partly_filled_workbook_reports_pending_runs(api):
    design = api.ok("api_make_design", {"design_type": "box_behnken", "factors": FACTORS})
    xlsx = fill_workbook(design["xlsx_base64"], quadratic, blank_rows=(0, 4))
    result = api.ok("api_analyze", {"xlsx_base64": xlsx, "model": "interactions"})
    assert result["pending_runs"] == [1, 5]
    assert result["n_completed"] == design["n_runs"] - 2


def test_analysing_the_page_s_runs_matches_analysing_its_workbook(api, held, simulated):
    responses = simulated["responses"]
    in_page = api.ok("api_analyze_runs", {**held, "responses": responses})
    book = api.ok("api_workbook", {**held, "responses": responses})
    from_book = api.ok("api_analyze", {"xlsx_base64": book["xlsx_base64"]})
    assert in_page["equation"] == from_book["equation"]
    assert in_page["summary"] == from_book["summary"]


def test_foreign_file_is_reported(api):
    assert "xlsx" in api.error("api_analyze", {"xlsx_base64": base64.b64encode(b"not a workbook").decode()})


def test_a_blocked_design_s_workbook_keeps_its_blocks(api):
    spec = {"design_type": "full_factorial", "factors": FACTORS, "n_center_points": 2, "n_blocks": 2}
    design = api.ok("api_make_design", spec)
    filled = fill_workbook(design["xlsx_base64"], quadratic)
    read = api.ok("api_read_workbook", {"xlsx_base64": filled})
    assert read["columns"] == design["columns"]
    assert [row[2] for row in read["rows"]] == [row[2] for row in design["rows"]], "each run keeps its block"
    analysis = api.ok("api_analyze", {"xlsx_base64": filled})
    assert "Block" in [a["source"] for a in analysis["anova"]]


@pytest.mark.parametrize("model", ["quadratic", {"terms": [["T"], ["P"], ["T", "P", "F"]]}])
def test_a_workbook_keeps_the_model_it_was_analysed_with(api, worked_example, model):
    held, simulated = worked_example["held"], worked_example["simulated"]
    written = api.ok("api_workbook", {**held, "responses": simulated["responses"], "model": model})
    read = api.ok("api_read_workbook", {"xlsx_base64": written["xlsx_base64"]})
    assert read["spec"]["analysis_model"] == model
    expected = api.ok("api_analyze_runs", {**held, "responses": simulated["responses"], "model": model})
    analysed = api.ok("api_analyze", {"xlsx_base64": written["xlsx_base64"]})
    assert analysed["formula"] == expected["formula"]


def test_a_workbook_analysed_as_designed_keeps_no_model(api, worked_example):
    held, simulated = worked_example["held"], worked_example["simulated"]
    written = api.ok("api_workbook", {**held, "responses": simulated["responses"], "model": ""})
    assert "analysis_model" not in api.ok("api_read_workbook", {"xlsx_base64": written["xlsx_base64"]})["spec"]


def test_a_categorical_factor_s_labels_go_through_the_workbook(api):
    factors = [*FACTORS[:2], {"name": "Supplier", "low": "old", "high": "new", "units": ""}]
    design = api.ok("api_make_design", {"design_type": "full_factorial", "factors": factors, "n_center_points": 2})
    filled = fill_workbook(design["xlsx_base64"], quadratic)
    read = api.ok("api_read_workbook", {"xlsx_base64": filled})
    assert read["spec"]["factors"][2] == factors[2]
    assert [row[4] for row in read["rows"]] == [row[4] for row in design["rows"]], "each run keeps its label"
    analysis = api.ok("api_analyze", {"xlsx_base64": filled})
    assert "Supplier" in [c["term"] for c in analysis["coefficients"]]
