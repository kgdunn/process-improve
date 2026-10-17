"""The app needs nothing the browser does not have.

Pyodide has no WebAssembly build of some of the package's optional dependencies, and the page
installs only ``tqdm`` and ``openpyxl`` beyond the Pyodide distribution (``web/worker.js``).
Here a fresh interpreter refuses those packages at import and runs every step of the page but
the charts (``api_plots`` installs plotly on its first call), so a code path that reaches for
one fails here rather than on a reader's screen. ``tests/web/wasm`` runs the app's Python
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
    assert reply["ok"], (fn, reply["error"])
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
"""


@pytest.mark.slow
def test_the_page_runs_without_the_packages_the_browser_lacks():
    result = subprocess.run(  # noqa: S603 - this interpreter, running the script above
        [sys.executable, "-c", SCRIPT, str(WEB), *BLOCKED],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},  # nothing written into web/
    )
    assert result.returncode == 0, result.stderr
