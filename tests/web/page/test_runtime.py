"""``web/runtime.json``: the one record of what the page loads into Pyodide.

Until ``web/worker.js`` reads the file itself, the Pyodide release, its packages and what
micropip installs are also written out in ``worker.js``; CI reads the release from the file.
These tests hold the copies to it, so the browser, the wasm tier and CI run one Python.
"""

from __future__ import annotations

import json
import re

from tests.web._support import ROOT, source

RUNTIME = json.loads(source("runtime.json"))


def listed(pattern: str, text: str) -> list[str]:
    """Return the quoted names in the list ``pattern`` finds."""
    return re.findall(r'"([\w-]+)"', re.search(pattern, text, re.DOTALL).group(1))


def test_the_worker_starts_the_pyodide_release_runtime_json_names():
    assert re.search(r'const PYODIDE_VERSION = "([\d.]+)";', source("worker.js")).group(1) == RUNTIME["pyodide"]


def test_the_worker_loads_the_packages_runtime_json_names():
    assert listed(r"pyodide\.loadPackage\(\[(.*?)\]\)", source("worker.js")) == RUNTIME["packages"]


def test_the_worker_installs_what_runtime_json_names_from_pypi():
    assert listed(r"await micropip\.install\(\[(.*?)\]\)", source("worker.js")) == RUNTIME["micropip"]


def test_ci_reads_the_pyodide_release_from_runtime_json():
    workflow = (ROOT / ".github" / "workflows" / "run-tests.yml").read_text(encoding="utf-8")
    assert "jq -r .pyodide web/runtime.json" in workflow
    assert not re.search(r"pyodide@\d", workflow), "no release written out in the workflow"
