"""``web/runtime.json``: the one record of what the page loads into Pyodide.

``web/worker.js`` reads it, the wasm tier's runner reads it, and CI reads the Pyodide release
from it, so the browser, the wasm tier and CI run one Python. Every package from PyPI is pinned:
a release there would otherwise change the page with no commit here.
"""

from __future__ import annotations

import importlib.util
import json
import re
from typing import TYPE_CHECKING

import pytest

from tests.web._support import ROOT, source

if TYPE_CHECKING:
    from types import ModuleType

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


# --------------------------------------------------------------------------- the weekly check of the pins
@pytest.fixture(scope="module")
def checker() -> ModuleType:
    """Load ``tools/check_web_runtime.py``, which is a script, not a module of the package."""
    spec = importlib.util.spec_from_file_location("check_web_runtime", ROOT / "tools" / "check_web_runtime.py")
    assert spec
    assert spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_runtime_pinned_to_the_latest_releases_is_not_behind(checker):
    pins = dict(requirement.split("==") for requirement in RUNTIME["micropip"])
    assert checker.behind(RUNTIME, lambda: RUNTIME["pyodide"], pins.__getitem__) == []


def test_each_pin_that_is_not_the_latest_release_is_reported(checker):
    pins = dict(requirement.split("==") for requirement in RUNTIME["micropip"])
    assert checker.behind(RUNTIME, lambda: "9.0.0", lambda _: "99.0") == [
        f"Pyodide: {RUNTIME['pyodide']} is pinned, and 9.0.0 is the latest release.",
        *(f"{name}: {pin} is pinned, and 99.0 is the latest release on PyPI." for name, pin in pins.items()),
    ]


def test_the_weekly_check_tries_the_published_app_and_the_pins():
    workflow = (ROOT / ".github" / "workflows" / "web-app-live.yml").read_text(encoding="utf-8")
    assert "python tools/check_web_runtime.py" in workflow
    assert "APP: https://kgdunn.github.io/process-improve/app/" in workflow
    assert '--web-app="$APP"' in workflow
