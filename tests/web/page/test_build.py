"""The build: ``scripts/build_web_app.py`` assembles the page, a wheel of this checkout and the worked example.

The docs workflow publishes what it assembles. ``demo.json`` is the worked example computed at
build time through ``web/bootstrap.py``, which the page draws at first paint while Python loads.
"""

from __future__ import annotations

import functools
import json
import re
from pathlib import Path

import pytest

from tests.web._support import APP_SOURCES, BUILD_SCRIPT, ROOT, WEB, no_bytecode, source


@pytest.fixture(scope="module")
def built_example(build_script) -> dict:
    """Run the worked example as the build does for ``demo.json``, and read it back as the page reads it."""
    pytest.importorskip("plotly", reason="the build draws the worked example's charts")
    with no_bytecode():
        return json.loads(json.dumps(build_script.worked_example()))


def test_every_module_the_page_imports_is_published(build_script):
    imported = set(re.findall(r'from "\./([\w.-]+)"', source("app.js")))
    assert imported, "app.js imports charts.js"
    assert imported <= set(build_script.APP_FILES)


@pytest.mark.slow
def test_the_built_worked_example_carries_what_the_page_draws(built_example):
    example = built_example
    assert example["version"]
    assert set(example["inputs"]) >= {"design", "scenario", "sigma", "seed", "model"}
    assert set(example["design"]) >= {"spec", "columns", "rows", "facts", "model", "n_runs"}
    assert "xlsx_base64" not in example["design"], "the page writes the workbook on demand"
    assert set(example["simulation"]) >= {"responses", "truth", "equation", "inactive", "label", "sigma", "seed"}
    assert all("true_value" in c for c in example["analysis"]["coefficients"])
    assert set(example["analysis"]) >= {"equation", "summary", "notes", "anova", "pending_runs", "n_completed"}
    assert set(example["plots"]["figures"]) == {"contour", "true_contour", "residuals"}
    assert example["plots"]["plotly_js"]


@pytest.mark.slow
def test_the_built_worked_example_includes_the_comparison(built_example):
    """The page draws step 4 at first paint too, from the comparison the build computes."""
    compared = built_example["comparison"]
    assert compared["model"] == "main_quadratic"
    assert [m["design_type"] for m in compared["made"]] == [
        "full_factorial",
        "fractional_factorial",
        "plackett_burman",
        "dsd",
        "box_behnken",
        "ccd",
        "d_optimal",
        "i_optimal",
    ]
    assert not any("xlsx_base64" in m["design"] for m in compared["made"]), "the page writes workbooks on demand"
    a_over_p = [m["key"] for m in compared["result"]["measures"]].index("a")
    assert compared["result"]["display"]["Box-Behnken [n=15]"][a_over_p] == "0.2173"


#: Files the build writes beside the page, rather than copying from web/.
BUILT = {"demo.json", "wheels/manifest.json"}


def _local(references: list[str]) -> set[str]:
    """Keep the references to files of the app itself: not other sites, not data: URLs."""
    return {ref for ref in references if "://" not in ref and not ref.startswith(("data:", "#", "$"))}


def test_every_file_the_page_loads_is_published(build_script):
    """What index.html, app.js and worker.js load must be copied by the build, or written by it."""
    loaded = _local(re.findall(r'(?:href|src)="([^"]+)"', source("index.html")))
    loaded |= _local(re.findall(r'from "\./([^"]+)"', source("app.js")))
    for name in ("app.js", "worker.js"):
        loaded |= _local(re.findall(r'(?:fetch|new Worker)\("([^"]+)"', source(name)))
    assert {"style.css", "app.js", "worker.js", "bootstrap.py", "demo.json"} <= loaded, "the search found the loads"
    assert loaded - BUILT <= set(build_script.APP_FILES)
    assert all((WEB / name).is_file() for name in build_script.APP_FILES)


def test_every_file_of_the_app_selects_these_tests(build_script):
    """A change to any file in web/ must run tests/web under --affected-since: each must be named in APP_SOURCES."""
    in_web = {p.name for p in WEB.iterdir() if p.is_file() and not p.name.startswith(".")}
    assert in_web | set(build_script.APP_FILES) | {BUILD_SCRIPT.name} <= set(APP_SOURCES)


@pytest.mark.slow
def test_a_change_to_any_file_of_the_app_runs_every_test_here(monkeypatch):
    """``--affected-since`` follows the names in APP_SOURCES from a changed file to the whole of tests/web."""
    from tests import _selection

    monkeypatch.setattr(_selection, "build_import_graph", functools.cache(_selection.build_import_graph))
    here = {p.resolve() for p in (ROOT / "tests" / "web").rglob("test_*.py")}
    for name in APP_SOURCES:
        changed = BUILD_SCRIPT if name == BUILD_SCRIPT.name else WEB / name
        selected = _selection.affected_test_files([changed.relative_to(ROOT).as_posix()], ROOT)
        assert selected is not None, name
        assert here <= selected, f"a change to {name} leaves out {sorted(p.name for p in here - selected)}"


def test_assemble_lays_out_what_the_page_and_its_worker_read(build_script, tmp_path, monkeypatch):
    """The page, the wheel under wheels/, the manifest naming both versions, and the worked example."""
    from importlib.metadata import version

    get_plotlyjs_version = pytest.importorskip(
        "plotly.offline", reason="the build names plotly's versions"
    ).get_plotlyjs_version

    def fake_wheel(_dist: Path) -> Path:
        wheel = tmp_path / "process_improve-9.9.9-py3-none-any.whl"
        wheel.write_bytes(b"a wheel")
        return wheel

    monkeypatch.setattr(build_script, "build_wheel", fake_wheel)
    monkeypatch.setattr(build_script, "worked_example", lambda: {"the": "worked example"})
    out = tmp_path / "app"
    (out / "stale").mkdir(parents=True)
    assert build_script.assemble(out) == "process_improve-9.9.9-py3-none-any.whl"

    published = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
    wheel = "wheels/process_improve-9.9.9-py3-none-any.whl"
    assert published == {*build_script.APP_FILES, *BUILT, wheel}, "and nothing left from an earlier build"
    for name in build_script.APP_FILES:
        assert (out / name).read_bytes() == (WEB / name).read_bytes(), name
    manifest = json.loads((out / "wheels" / "manifest.json").read_text())
    assert manifest == {
        "wheel": "process_improve-9.9.9-py3-none-any.whl",
        "plotly": version("plotly"),
        "plotly_js": get_plotlyjs_version(),
    }
    assert json.loads((out / "demo.json").read_text()) == {"the": "worked example"}


def test_the_page_reads_only_what_the_manifest_holds():
    """worker.js installs the wheel and the plotly release named there; app.js draws with the Plotly.js named there."""
    read = set(re.findall(r"const \{ (\w+)(?:: \w+)? \} = await manifest", source("worker.js")))
    read |= set(re.findall(r"\(await res\.json\(\)\)\.(\w+)", source("app.js")))
    assert read == {"wheel", "plotly", "plotly_js"}
