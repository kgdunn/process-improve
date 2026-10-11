"""``web/runtime.json``: the one record of what the page loads into Pyodide.

``web/worker.js`` reads it, the wasm tier's runner reads it, and CI reads the Pyodide release
from it, so the browser, the wasm tier and CI run one Python. Every package from PyPI is pinned:
a release there would otherwise change the page with no commit here.
"""

from __future__ import annotations

import json
import re

from tests.web._support import ROOT, source

RUNTIME = json.loads(source("runtime.json"))


def test_the_worker_reads_runtime_json_and_keeps_no_copy_of_it():
    worker = source("worker.js")
    assert 'fetch("runtime.json"' in worker
    assert RUNTIME["pyodide"] not in worker, "no Pyodide release written out"
    for name in [*RUNTIME["packages"], *RUNTIME["micropip"]]:
        assert f'"{name}"' not in worker, f"{name} is written out in worker.js"


def test_the_pyodide_release_is_an_exact_version():
    assert re.fullmatch(r"\d+\.\d+\.\d+", RUNTIME["pyodide"])


def test_every_package_from_pypi_is_pinned_to_one_version():
    """The packages the Pyodide release lacks; its own (``packages``) are fixed by the release."""
    assert RUNTIME["micropip"], "openpyxl reads and writes the workbooks"
    for requirement in RUNTIME["micropip"]:
        assert re.fullmatch(r"[\w.-]+==[\w.]+", requirement), requirement


def test_no_package_is_loaded_both_from_the_release_and_from_pypi():
    from_pypi = {r.split("==")[0].lower().replace("_", "-") for r in RUNTIME["micropip"]}
    assert not from_pypi & {p.lower() for p in RUNTIME["packages"]}


def test_the_wasm_runner_reads_runtime_json():
    runner = (ROOT / "tests" / "web" / "wasm" / "runner.mjs").read_text(encoding="utf-8")
    assert "runtime.packages" in runner
    assert "runtime.micropip" in runner


def test_ci_reads_the_pyodide_release_from_runtime_json():
    workflow = (ROOT / ".github" / "workflows" / "run-tests.yml").read_text(encoding="utf-8")
    assert "jq -r .pyodide web/runtime.json" in workflow
    assert not re.search(r"pyodide@\d", workflow), "no release written out in the workflow"
