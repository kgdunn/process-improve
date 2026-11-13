"""Packages the page installs when a call first needs them: openpyxl for a workbook, plotly for a chart.

Start-up in the browser leaves both out, so a visit that makes a run sheet and no more never downloads them. A
call that finds one missing answers ``{"ok": false, "needs": <package>}``, in place of its reply; the worker
(``web/worker.js``) installs the package and makes the call again. Here a package is made to look uninstalled,
as it is under Pyodide until then.

statsmodels and scikit-learn, which a model is fitted with, are left out of start-up the same way, and asked for
the same way. The library imports them, not this page's Python, and only once a call first fits a model: that a
visit which makes a run sheet reaches for neither is ``test_browser_safe.py``'s to show, in an interpreter of its
own, since this one has long since imported both.
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


@pytest.mark.parametrize(
    ("missing", "needed"),
    [
        ("statsmodels", "statsmodels"),
        ("statsmodels.formula.api", "statsmodels"),  # Python names the module it looked for, which may be a part
        ("sklearn.utils", "sklearn"),
        ("threadpoolctl", "threadpoolctl"),  # which the release loads with scikit-learn, and the library imports itself
        ("plotly", "plotly"),
        ("openpyxl", "openpyxl"),
    ],
)
def test_a_call_stopped_for_want_of_a_package_start_up_leaves_out_says_which(bootstrap, missing, needed):
    """Wherever the import is: in the page's Python, or in a module of the library imported when first needed."""

    def stopped(_: dict) -> dict:
        raise ModuleNotFoundError(f"No module named {missing!r}", name=missing)

    text = bootstrap._reply(stopped, "{}")
    assert json.loads(text) == {"ok": False, "needs": needed}
    assert len(text) < READ_BELOW


@pytest.mark.parametrize(
    "error",
    [
        ModuleNotFoundError("No module named 'numba'", name="numba"),  # nothing the worker installs
        ModuleNotFoundError("No module named 'statsmodelsx'", name="statsmodelsx"),
        ModuleNotFoundError("import halted"),  # with no module named
        ImportError("cannot import name 'OLS' from 'statsmodels'", name="statsmodels"),  # it is there
    ],
)
def test_any_other_import_that_fails_is_an_error_like_any_other(bootstrap, error):
    def stopped(_: dict) -> dict:
        raise error

    reply = json.loads(bootstrap._reply(stopped, "{}"))
    assert reply["error"] == f"Python stopped with {type(error).__name__}: {error}"
    assert "needs" not in reply


def test_the_page_installs_something_for_every_package_it_may_ask_for(bootstrap):
    """Python asks by the name a package is imported by, and ``onDemand`` in web/python-boot.js is keyed by it."""
    assert bootstrap.ON_DEMAND == ("openpyxl", "plotly", "statsmodels", "sklearn", "joblib", "threadpoolctl")
    for package in bootstrap.ON_DEMAND:
        assert bootstrap._lacking(ModuleNotFoundError(name=package)) == package


def test_a_package_that_is_absent_at_the_start_is_refused_until_the_worker_says_it_is_in(bootstrap):
    """Its files are unpacked before its compiled modules are loaded: an import in between would break it."""
    guard = bootstrap._NotYet(["json", "a_package_not_yet_installed", "another_one"])
    assert guard.waiting == {"a_package_not_yet_installed", "another_one"}, "what is there at the start is let be"
    for module in ("a_package_not_yet_installed", "a_package_not_yet_installed.formula.api"):
        with pytest.raises(ModuleNotFoundError, match=f"No module named '{module}'") as refused:
            guard.find_spec(module)
        assert bootstrap._lacking(refused.value) is None, "nothing the worker installs: an error like any other"
        assert refused.value.name == module, "named as Python names a module that is not there"
    assert guard.find_spec("json") is None, "any other import is left to the finders after this one"
    assert guard.find_spec("a_package_not_yet_installed_at_all") is None, "a package is refused, not a prefix"


def test_the_worker_s_word_that_a_package_is_in_lets_its_imports_through(bootstrap, monkeypatch):
    guard = bootstrap._NotYet([])
    guard.waiting = {"statsmodels", "sklearn", "plotly"}
    monkeypatch.setattr(bootstrap, "_NOT_YET", guard)
    bootstrap.installed(json.dumps(["statsmodels", "sklearn", "joblib"]))
    assert guard.waiting == {"plotly"}
    assert guard.find_spec("statsmodels.api") is None
    with pytest.raises(ModuleNotFoundError):
        guard.find_spec("plotly.graph_objects")


def test_under_the_tests_own_python_every_package_is_there_and_none_is_refused(bootstrap):
    assert bootstrap._NOT_YET in sys.meta_path
    assert bootstrap._NOT_YET.waiting == set()


def test_a_package_pyodide_says_is_not_installed_counts_as_absent(bootstrap, monkeypatch):
    """Pyodide's finder raises for a package of its release that is not loaded, where Python's own return nothing."""

    def says_so(name: str) -> None:
        message = f"The module '{name}' is included in the Pyodide distribution, but it is not installed."
        raise ModuleNotFoundError(message)

    monkeypatch.setattr(bootstrap.importlib.util, "find_spec", says_so)
    assert bootstrap._absent("statsmodels")
    monkeypatch.undo()
    assert not bootstrap._absent("json")
    assert bootstrap._absent("a_package_not_yet_installed")


def test_a_design_that_lacks_a_package_is_not_left_out_of_the_comparison_as_one_that_cannot_be_made(api, monkeypatch):
    """The comparison notes a design that cannot be made and goes on; a package to install is no such fault."""

    def lacks_a_package(_: dict) -> dict:
        raise ModuleNotFoundError("No module named 'statsmodels'", name="statsmodels")

    monkeypatch.setattr(api.module, "make_design", lacks_a_package)
    assert needs(api, "api_candidates", {"factors": FACTORS, "model": "interactions"}) == "statsmodels"


def test_a_next_step_that_lacks_a_package_is_not_taken_for_an_older_release(api, monkeypatch, studied):
    """A release without the next step is told so in a sentence; a package to install is asked for instead."""
    assert api.ok("api_next_runs", studied)["recommendation"], "the worked example's results call for a next step"
    monkeypatch.setitem(sys.modules, "process_improve.experiments.next_step", None)  # as in a release before it
    older = api.ok("api_next_runs", studied)
    assert older["recommendation"] is None
    assert older["waiting"] == "This version of process-improve cannot say what the results call for yet."

    real_import = __import__

    def lacking_statsmodels(name: str, *args: object, **kwargs: object) -> object:
        if name == "process_improve.experiments.next_step":
            raise ModuleNotFoundError("No module named 'statsmodels'", name="statsmodels")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", lacking_statsmodels)
    assert needs(api, "api_next_runs", studied) == "statsmodels"


def test_the_worker_s_warm_up_imports_the_analysis_code_and_names_what_the_page_then_runs_on(api, bootstrap):
    """The worker calls it once it has fetched what a model is fitted with; the reply goes to the page's report."""
    reply = json.loads(bootstrap.warm())
    assert reply["ok"]
    assert reply["warnings"] == []
    assert reply["result"] == api.ok("api_catalogue")["runtime"], "here both are installed from the start"
    assert set(reply["result"]["packages"]) == set(bootstrap.RUNTIME_PACKAGES)
    assert "process_improve.experiments.analysis" in sys.modules


def test_a_workbook_that_is_not_one_is_refused_as_before_once_openpyxl_is_there(api):
    """The check for openpyxl comes before the file is read, so a file that cannot be read is still said to be so."""
    error = api.error("api_read_workbook", {"xlsx_base64": base64.b64encode(b"not a workbook").decode()})
    assert error == "That file could not be read as an .xlsx workbook."
