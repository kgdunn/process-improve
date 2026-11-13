"""The app needs nothing the browser does not have.

Pyodide has no WebAssembly build of some of the package's optional dependencies, and beyond the
Pyodide distribution the page installs only ``openpyxl``, with its first workbook, and plotly,
with its first chart (``web/worker.js``). Here a fresh interpreter refuses the other packages at
import and runs every step of the page but the charts, so a code path that reaches for one fails
here rather than on a reader's screen. ``tests/web/wasm`` runs the app's Python
under real Pyodide (``--wasm``).
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from tests.web._support import WEB

#: Packages the page cannot import: no WebAssembly build, or not installed by ``web/worker.js``.
BLOCKED = ("pulp", "pyoptex", "numba", "osqp", "skimage", "ruptures", "mcp", "matplotlib", "plotly", "pyDOE3")

#: Run in a fresh interpreter, with the web directory and the blocked packages as its arguments.
SCRIPT = """
import importlib.abc, json, sys

WEB, *BLOCKED = sys.argv[1:]


class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"{name} is not available in the browser")


sys.meta_path.insert(0, Block())
sys.path.insert(0, WEB)
import bootstrap


def call(fn, payload):
    reply = json.loads(getattr(bootstrap, fn)(json.dumps(payload)))
    if not reply["ok"]:  # no assert: one row of CI runs Python with -O, which leaves asserts out
        raise SystemExit(f"{fn}: {reply['error']}")
    return reply["result"]


def held(design):
    return {key: design[key] for key in ("spec", "columns", "rows")}


factors = [{"name": n, "low": 0, "high": 1} for n in "ABC"]
for design_type in bootstrap.DESIGNS:
    call("api_make_design", {"design_type": design_type, "factors": factors})

demo = call("api_catalogue", {})["demo"]
runs = held(call("api_make_design", demo["design"]))
simulated = call("api_simulate", {**runs, "scenario": demo["scenario"], "seed": demo["seed"]})
responses = simulated["responses"]
call("api_analyze_runs", {**runs, "responses": responses, "truth": simulated["truth"]})
call("api_best_settings", {**runs, "responses": responses})
sources =["simulated"] * len(responses)
book = call("api_workbook", {**runs, "responses": responses, "sources": sources, "simulated": simulated})
call("api_read_workbook", book)
call("api_analyze", book)

plan = call("api_compare_plan", {"factors": demo["design"]["factors"]})
made = [{"label": item["label"], **held(call("api_make_design", item["spec"]))} for item in plan["designs"]]
call("api_compare", {"model": plan["model"], "designs": made})

# The browser's copy of the wheel leaves out the reference catalogue's sources (NOT_FOR_THE_PAGE in
# scripts/build_web_app.py). One module reads them, and the page's Python must never load it.
if "process_improve.experiments.reference_catalogue" in sys.modules:
    raise SystemExit("the page's Python loaded the reference catalogue, which the browser's wheel leaves out")
"""


@pytest.mark.slow
def test_the_page_runs_without_the_packages_the_browser_lacks():
    result = subprocess.run(  # noqa: S603 - this interpreter, running the script above
        [sys.executable, "-c", SCRIPT, str(WEB), *BLOCKED],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},  # nothing written into web/
        encoding="locale",
    )
    assert result.returncode == 0, result.stderr


#: The packages a model is fitted with (``fitting`` in ``web/runtime.json``) and those only they bring with them,
#: by the names they are imported by. Start-up in the browser leaves them out, and the worker fetches them once the
#: page is ready. The first two are the ones Python may say a call lacks (``ON_DEMAND`` in ``web/bootstrap.py``).
FITTING = ("statsmodels", "sklearn", "joblib", "threadpoolctl")

#: Run in a fresh interpreter, with the web directory and the fitting packages as its arguments: a visit that ends
#: with a run sheet, then the first fit, before the worker has fetched what it needs and after.
RUN_SHEET_VISIT = """
import importlib.abc, json, sys

WEB, *FITTING = sys.argv[1:]


class NotYet(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in FITTING:  # as Python refuses a package that is not installed
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)


not_yet = NotYet()
sys.meta_path.insert(0, not_yet)
sys.path.insert(0, WEB)
import bootstrap


def reply(fn, payload):
    return json.loads(getattr(bootstrap, fn)(json.dumps(payload)))


def call(fn, payload):
    answer = reply(fn, payload)
    if not answer["ok"]:  # no assert: one row of CI runs Python with -O, which leaves asserts out
        raise SystemExit(f"{fn} of a visit that makes a run sheet: {answer}")
    return answer["result"]


def held(design):
    return {key: design[key] for key in ("spec", "columns", "rows")}


# Every step of a visit that ends with a run sheet: the designs the page offers, the comparison of them, limits on
# the settings, a table of the reader's own, more runs for a study, a workbook, and results made up to try it.
demo = call("api_catalogue", {})["demo"]
factors = demo["design"]["factors"]
abc = [{"name": n, "low": 0, "high": 1} for n in "ABC"]
for design_type in bootstrap.DESIGNS:
    call("api_make_design", {"design_type": design_type, "factors": abc})
bounds = {"A": (0.2, 0.7), "B": (0.1, 0.6), "C": (0.0, 0.5)}
bounded = [{"name": n, "low": low, "high": high} for n, (low, high) in bounds.items()]
for mixture in bootstrap.MIXTURES:
    call("api_make_design", {"design_type": mixture, "factors": bounded if mixture == "extreme_vertices" else abc})
levels = [*abc[:2], {"name": "Catalyst", "type": "categorical", "levels": ["X", "Y", "Z"]}]
call("api_make_design", {"design_type": "full_factorial", "factors": levels})
call("api_make_design", {"design_type": "full_factorial", "factors": abc, "n_blocks": 2})
limits = [{"expression": "T + 25*P <= 215"}, {"expression": "400 <= 3*T + 5*P <= 600"}]
call("api_limits", {"factors": factors, "constraints": limits})
limited = held(call("api_make_design", {"design_type": "d_optimal", "factors": factors, "constraints": limits}))
call("api_outside_limits", {"spec": limited["spec"], "settings": {"T": 200, "P": 3, "F": 20}})
call("api_candidates", {"factors": factors, "model": "main_quadratic", "search": True})
call("api_candidates", {"factors": factors, "model": "interactions", "constraints": limits})
plan = call("api_compare_plan", {"factors": factors})
made = [{"label": item["label"], **held(call("api_make_design", item["spec"]))} for item in plan["designs"]]
call("api_compare", {"model": plan["model"], "designs": made})
runs = held(call("api_make_design", demo["design"]))
call("api_correlations", runs)
call("api_relabel", {"design": runs, "factors": [{**factors[0], "name": "Temp"}, *factors[1:]]})
seven = [{"name": n, "low": -1, "high": 1} for n in "ABCDEFG"]
fraction = {"design_type": "fractional_factorial", "factors": seven, "resolution": 3, "n_center_points": 0}
call("api_add_runs", {**held(call("api_make_design", fraction)), "kind": "foldover"})
table = call("api_read_table", {"text": "T,P,Yield\\n150,1,60\\n200,1,64\\n150,3,61\\n200,3,69\\n175,2,65\\n"})
call("api_import_study", {"columns": table["columns"], "rows": table["rows"], "title": "Trials"})
book = call("api_workbook", runs)
call("api_read_workbook", book)
simulated = call("api_simulate", {**runs, "scenario": demo["scenario"], "seed": demo["seed"]})

reached = sorted({name.split(".")[0] for name in sys.modules} & set(FITTING))
if reached:
    raise SystemExit(f"a visit that makes a run sheet loaded {reached}")

# The first fit says what it lacks, in the reply the worker acts on; so does the worker's own warm-up.
request = {**runs, "responses": simulated["responses"]}
for lacking in (reply("api_analyze_runs", request), json.loads(bootstrap.warm())):
    if lacking.get("needs") not in bootstrap.ON_DEMAND or set(lacking) != {"ok", "needs"}:
        raise SystemExit(f"a fit without what a model is fitted with: {lacking}")

# The worker installs it. Its files are in place before its compiled modules are loaded, and a call answered in
# between must still be told it is lacking: Python refuses it until the worker says it is in.
sys.meta_path.remove(not_yet)
half_way = reply("api_analyze_runs", request)
if half_way.get("needs") not in bootstrap.ON_DEMAND:
    raise SystemExit(f"a fit while what a model is fitted with is still being installed: {half_way}")
bootstrap.installed(json.dumps(list(FITTING)))

# Then the call is made again, in the same Python: the import that failed left nothing broken.
warmed = json.loads(bootstrap.warm())
if not warmed["ok"] or "statsmodels" not in warmed["result"]["packages"]:
    raise SystemExit(f"the warm-up, once the packages are in: {warmed}")
fitted = call("api_analyze_runs", request)
if not fitted["coefficients"]:
    raise SystemExit("the fit made again answered without coefficients")
call("api_best_settings", request)
call("api_next_runs", request)
"""


@pytest.mark.slow
def test_a_visit_that_makes_a_run_sheet_needs_nothing_a_model_is_fitted_with():
    """The page is ready before statsmodels and scikit-learn are in: no step of designing may reach for them.

    A step that did would not fail on the page, it would wait: the worker installs what a call lacks and calls
    again. So the wait this guards against is silent, and only a test sees it.
    """
    result = subprocess.run(  # noqa: S603 - this interpreter, running the script above
        [sys.executable, "-c", RUN_SHEET_VISIT, str(WEB), *FITTING],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},  # nothing written into web/
        encoding="locale",
    )
    assert result.returncode == 0, result.stderr
