"""The build: ``scripts/build_web_app.py`` assembles the page, a wheel of this checkout and the worked example.

The docs workflow publishes what it assembles. ``demo.json`` is the worked example computed at
build time through ``web/bootstrap.py``, which the page draws at first paint while Python loads.
``book/<name>.json`` is each of the book's examples, made the same way, which ``?book=<name>`` opens.
"""

from __future__ import annotations

import functools
import hashlib
import json
import re
from pathlib import Path

import pytest

from tests.web._support import APP_SOURCES, BUILD_SCRIPT, ROOT, WEB, javascript, no_bytecode, source


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
    """The worked example lists the designs that can learn it, from the comparison the build computes."""
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


def _stand_in_catalog(out: Path) -> None:
    """Write the catalog's files as ``write_catalog`` lays them out, without describing 783 designs."""
    (out / "catalog" / "detail").mkdir(parents=True)
    (out / "catalog" / "index.json").write_text("{}")
    (out / "catalog" / "detail" / "generated.json").write_text("{}")


def test_assemble_lays_out_what_the_page_and_its_worker_read(build_script, tmp_path, monkeypatch):
    """The page, the wheel in a folder of its own under wheels/, the manifest naming it, and the worked example."""
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
    monkeypatch.setattr(build_script, "book_example", lambda entry: {"name": entry["name"], "title": "", "book": ""})
    monkeypatch.setattr(build_script, "published_example", lambda entry: {"name": entry["name"], "title": ""})
    monkeypatch.setattr(build_script, "write_catalog", _stand_in_catalog)
    out = tmp_path / "app"
    (out / "stale").mkdir(parents=True)
    assert build_script.assemble(out) == "process_improve-9.9.9-py3-none-any.whl"

    published = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
    wheel = f"{hashlib.sha256(b'a wheel').hexdigest()[:16]}/process_improve-9.9.9-py3-none-any.whl"
    examples = [*build_script.book_examples(), *build_script.published_examples()]
    book = {"book/index.json", *(f"book/{entry['name']}.json" for entry in examples)}
    catalog = {"catalog/index.json", "catalog/detail/generated.json"}
    assert published == {*build_script.APP_FILES, *BUILT, *book, *catalog, f"wheels/{wheel}"}, "and nothing from before"
    for name in build_script.APP_FILES:
        assert (out / name).read_bytes() == (WEB / name).read_bytes(), name
    manifest = json.loads((out / "wheels" / "manifest.json").read_text())
    assert manifest == {
        "wheel": wheel,
        "commit": build_script.commit(),
        "plotly": version("plotly"),
        "plotly_js": get_plotlyjs_version(),
    }
    assert json.loads((out / "demo.json").read_text()) == {"the": "worked example"}


def _assemble_around(build_script, out: Path, wheel: bytes, monkeypatch) -> dict:
    """Assemble the app into ``out`` around a stand-in wheel holding ``wheel``; return the manifest."""
    pytest.importorskip("plotly.offline", reason="the build names plotly's versions")
    built = out.parent / "dist" / "process_improve-9.9.9-py3-none-any.whl"
    built.parent.mkdir(exist_ok=True)
    built.write_bytes(wheel)
    monkeypatch.setattr(build_script, "build_wheel", lambda _dist: built)
    monkeypatch.setattr(build_script, "worked_example", dict)
    monkeypatch.setattr(build_script, "write_book", lambda _out: None)
    monkeypatch.setattr(build_script, "write_catalog", lambda _out: None)
    build_script.assemble(out)
    return json.loads((out / "wheels" / "manifest.json").read_text())


def test_each_build_s_wheel_has_a_url_of_its_own(build_script, tmp_path, monkeypatch):
    """A browser never installs a wheel it cached from an earlier deploy beside a newer ``bootstrap.py``."""
    published = []
    for wheel in (b"one build", b"the next build"):
        manifest = _assemble_around(build_script, tmp_path / "app", wheel, monkeypatch)
        assert (tmp_path / "app" / "wheels" / manifest["wheel"]).read_bytes() == wheel
        assert Path(manifest["wheel"]).name == "process_improve-9.9.9-py3-none-any.whl", "micropip reads the name"
        published.append(manifest["wheel"])
    assert published[0] != published[1]


def test_the_manifest_names_the_commit_the_app_was_built_from(build_script, tmp_path, monkeypatch):
    """The deploy waits for the published manifest to name the commit it pushed: CI's, or this checkout's."""
    monkeypatch.setenv("GITHUB_SHA", "0123abc")
    assert _assemble_around(build_script, tmp_path / "ci", b"a wheel", monkeypatch)["commit"] == "0123abc"
    monkeypatch.delenv("GITHUB_SHA")
    assert re.fullmatch(
        r"[0-9a-f]{40}", _assemble_around(build_script, tmp_path / "here", b"a wheel", monkeypatch)["commit"]
    )


def test_the_page_reads_only_what_the_manifest_holds():
    """worker.js installs the wheel and the plotly release named there; app.js draws with the Plotly.js named there."""
    read = set(re.findall(r"const \{ (\w+)(?:: \w+)? \} = await manifest", source("worker.js")))
    read |= set(re.findall(r"\(await res\.json\(\)\)\.(\w+)", source("app.js")))
    assert read == {"wheel", "plotly", "plotly_js"}


# --------------------------------------------------------------------------- the book's examples
@pytest.fixture(scope="module")
def book(build_script, tmp_path_factory) -> Path:
    """Write the book's examples as the build does, into a folder of their own."""
    out = tmp_path_factory.mktemp("app")
    with no_bytecode():
        build_script.write_book(out)
    return out / "book"


def test_the_index_names_every_example_and_its_page_in_the_book(build_script, book):
    """The book's own check reads the index: each name a link may carry, and the page that prints it.

    The examples of other published sources follow the book's, each with its source in place of a page.
    """
    index = json.loads((book / "index.json").read_text())["examples"]
    from_book = [entry["name"] for entry in build_script.book_examples()]
    names = [*from_book, *(entry["name"] for entry in build_script.published_examples())]
    assert [e["name"] for e in index] == names
    assert len(set(names)) == len(names), "each name opens one example"
    assert {p.name for p in book.iterdir()} == {"index.json", *(f"{name}.json" for name in names)}
    for example in index:
        assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", example["name"]), example["name"]
        assert example["title"]
        if example["name"] in from_book:
            assert example["book"].startswith("https://learnche.org/pid/design-analysis-experiments/"), example
            assert ".html" not in example["book"], "the book is served without the extension"
        else:
            assert "book" not in example, "the book's own check finds no page of its own here"
            assert example["source"]["url"].startswith("https://doi.org/"), example
            assert example["source"]["cite"], example
            assert example["source"]["where"], example


def test_two_examples_with_one_name_stop_the_build(build_script, monkeypatch, tmp_path):
    entry = build_script.published_examples()[0]
    monkeypatch.setattr(build_script, "published_examples", lambda: [entry, entry])
    monkeypatch.setattr(build_script, "book_examples", list)
    with pytest.raises(SystemExit, match=f"Two examples share a name: \\['{entry['name']}'\\]"), no_bytecode():
        build_script.write_book(tmp_path)


def test_the_page_reads_each_example_as_the_build_wrote_it(node, book):
    """Each file passes the page's own check, ``checkBookExample``, and keeps every run, result and setting."""
    written = {p.stem: json.loads(p.read_text()) for p in book.glob("*.json") if p.name != "index.json"}
    checked = javascript(node, source("campaign.js"), [("checkBookExample", [e, name]) for name, e in written.items()])
    for (name, example), read in zip(written.items(), checked, strict=True):
        campaign, opened = example["campaign"], read["campaign"]
        assert read["title"] == example["title"] == campaign["title"], name
        assert (read["book"] is None) == ("source" in example), f"{name}: a page of the book, or a source"
        assert read["source"] == example.get("source"), name
        assert opened["design"]["rows"] == campaign["design"]["rows"], name
        assert opened["responses"] == campaign["responses"], name
        assert all(v is not None for v in opened["responses"]), f"{name}: every run has the book's result"
        assert opened["model"] == campaign["model"], name
        columns, n = campaign["design"]["columns"], len(opened["responses"])
        std, run = (columns.index(c) for c in ("Std order", "Run order"))
        assert [row[std] for row in opened["design"]["rows"]] == list(range(1, n + 1)), "the book's table, row by row"
        assert sorted(row[run] for row in opened["design"]["rows"]) == list(range(1, n + 1)), name


def test_the_run_sheet_is_the_book_s_table(build_script):
    """Experiment by experiment, with the run order and the day the book prints for the side product."""
    entry = next(e for e in build_script.book_examples() if e["name"] == "side-product")
    with no_bytecode():
        opened = build_script.book_example(entry)["campaign"]
    assert opened["design"]["columns"] == ["Std order", "Run order", "Block", "A", "B", "Side_product"]
    assert [[*row[:-1], y] for row, y in zip(opened["design"]["rows"], opened["responses"], strict=True)] == [
        [1, 2, 1, 420, 10, 89],
        [2, 4, 2, 440, 10, 268],
        [3, 5, 2, 420, 12, 179],
        [4, 3, 1, 440, 12, 448],
        [5, 1, 1, 430, 11, 196],
        [6, 6, 2, 430, 11, 215],
    ]


def test_a_run_the_design_does_not_have_stops_the_build(build_script):
    entry = next(e for e in build_script.book_examples() if e["name"] == "conversion-interaction")
    with pytest.raises(SystemExit, match=r"its run 1, \[5, 5\], is not a run of the design"):
        build_script.book_example({**entry, "runs": [[5, 5, 1.0], *entry["runs"][1:]]})
    with pytest.raises(SystemExit, match="the design has 1 runs more than the book lists"):
        build_script.book_example({**entry, "runs": entry["runs"][1:]})
