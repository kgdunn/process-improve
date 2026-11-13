"""Packages the page installs when a call first needs them: openpyxl for a workbook, plotly for a chart.

Start-up in the browser leaves both out, so a visit that makes a run sheet and no more never downloads them. A
call that finds one missing answers ``{"ok": false, "needs": <package>}``, in place of its reply; the worker
(``web/worker.js``) installs the package and makes the call again. Here a package is made to look uninstalled,
as it is under Pyodide until then.
"""

from __future__ import annotations

import base64
import json
import sys

import pytest

from tests.web._support import FACTORS, workbook_of

#: The longest reply ``web/python-boot.js`` reads for what a call lacks; a real reply is longer, and is not read.
READ_BELOW = 200


@pytest.fixture
def lacking(monkeypatch):
    """Return a way to make a package look uninstalled for the rest of the test."""

    def without(*packages: str) -> None:
        for package in packages:
            monkeypatch.setitem(sys.modules, package, None)  # an import of it then fails as a missing module's does

    return without


@pytest.fixture
def studied(api, held, simulated) -> dict:
    """Return the worked example's runs with their results, as the page sends them."""
    return {**held, "responses": simulated["responses"]}


def needs(api, fn: str, payload: dict) -> str:
    """Return the package a call says it lacks, checking the reply is one the worker reads as that."""
    text = getattr(api.module, fn)(json.dumps(payload))
    reply = json.loads(text)
    assert reply == {"ok": False, "needs": reply.get("needs")}, "nothing for the reader: the worker acts on it"
    assert len(text) < READ_BELOW, "short enough for the worker to read it"
    return reply["needs"]


def test_a_design_its_comparison_and_its_analysis_need_neither_package(api, lacking, studied):
    """What most visits do works with the packages Python starts with."""
    lacking("openpyxl", "plotly")
    design = api.ok("api_make_design", {"design_type": "box_behnken", "factors": FACTORS})
    assert "xlsx_base64" not in design, "the workbook is written when the reader asks for it"
    assert api.ok("api_candidates", {"factors": FACTORS, "model": "interactions"})["made"]
    assert api.ok("api_analyze_runs", studied)["coefficients"]
    assert api.ok("api_best_settings", {**studied, "goal": "maximize"})["settings"]


def test_writing_and_reading_a_workbook_need_openpyxl(api, lacking, held, design):
    book = workbook_of(api, design)
    lacking("openpyxl")
    assert needs(api, "api_workbook", held) == "openpyxl"
    assert needs(api, "api_read_workbook", {"xlsx_base64": book}) == "openpyxl"
    assert needs(api, "api_analyze", {"xlsx_base64": book}) == "openpyxl"


def test_a_table_of_the_reader_s_own_needs_openpyxl_only_as_a_workbook(api, lacking, design):
    book = workbook_of(api, design)
    lacking("openpyxl")
    assert needs(api, "api_read_table", {"xlsx_base64": book}) == "openpyxl"
    read = api.ok("api_read_table", {"text": "A,B,y\n1,2,3\n2,3,4\n3,4,6\n"})
    assert [column["heading"] for column in read["columns"]] == ["A", "B", "y"], "text needs nothing more"


def test_the_charts_need_plotly(api, lacking, studied):
    lacking("plotly")
    assert needs(api, "api_plots", studied) == "plotly"


def test_a_package_that_is_there_and_cannot_be_imported_is_an_error_like_any_other(api, monkeypatch, held):
    """Installing openpyxl again would not help when it is something openpyxl imports that is missing."""

    def broken(name: str, *_: object) -> None:
        raise ModuleNotFoundError("No module named 'et_xmlfile'", name="et_xmlfile")

    monkeypatch.setattr(api.module.importlib, "import_module", broken)
    reply = api.reply("api_workbook", held)
    assert reply["error"] == "Python stopped with ModuleNotFoundError: No module named 'et_xmlfile'"
    assert "needs" not in reply


def test_a_workbook_that_is_not_one_is_refused_as_before_once_openpyxl_is_there(api):
    """The check for openpyxl comes before the file is read, so a file that cannot be read is still said to be so."""
    error = api.error("api_read_workbook", {"xlsx_base64": base64.b64encode(b"not a workbook").decode()})
    assert error == "That file could not be read as an .xlsx workbook."
