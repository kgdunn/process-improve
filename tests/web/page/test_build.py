"""The build: ``scripts/build_web_app.py`` assembles the page, a wheel of this checkout and the worked example.

The docs workflow publishes what it assembles. ``demo.json`` is the worked example computed at
build time through ``web/bootstrap.py``, which the page draws at first paint while Python loads.
``book/<name>.json`` is each of the book's examples, made the same way, which ``?book=<name>`` opens.
"""

from __future__ import annotations

import functools
import hashlib
import io
import json
import re
import shutil
import zipfile
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
    assert {"contour", "true_contour", "residuals", "perturbation", "surface_3d"} <= set(example["plots"]["figures"])
    assert example["plots"]["plotly_js"]
    [fit] = example["predictors"]["responses"]
    assert len(fit["terms"]) == len(example["analysis"]["coefficients"]), "what if predicts from the model shown"


@pytest.mark.slow
def test_the_built_worked_example_includes_the_comparison(built_example):
    """The worked example lists the designs that can learn it, from the comparison the build computes."""
    compared = built_example["comparison"]
    assert compared["model"] == "main_quadratic"
    assert [m["design_type"] for m in compared["made"]] == [
        "full_factorial",
        "fractional_factorial",
        "plackett_burman",
        "taguchi",
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

#: What ``write_catalog`` writes that the page loads: the fields of the planning panel and their list
#: (``tests/web/page/test_field_pages.py`` checks it writes them).
CATALOG_WRITES = {"catalog/fields.json", "catalog/fields/index.html"}


def _local(references: list[str]) -> set[str]:
    """Keep the references to files of the app itself: not other sites, not data: URLs.

    A folder's address, such as ``catalog/``, is its ``index.html``, as the server answers it, and
    ``./``, the header's link home, is the page's own.
    """
    kept = [ref for ref in references if "://" not in ref and not ref.startswith(("data:", "#", "$"))]
    local = {ref.removeprefix("./") for ref in kept}
    return {f"{ref}index.html" if ref.endswith("/") or not ref else ref for ref in local}


def test_every_file_the_page_loads_is_published(build_script):
    """What index.html, app.js, the worker and the module that starts Python load must be copied by the build, or
    written by it.

    feedback.js imports the screenshot's module only when a reader asks for a screenshot: it is published all the same.
    """
    loaded = _local(re.findall(r'(?:href|src)="([^"]+)"', source("index.html")))
    for name in ("app.js", "worker.js"):
        loaded |= _local(re.findall(r'from "\./([^"]+)"', source(name)))
    loaded |= _local(re.findall(r'(?:fetch|new Worker)\("([^"]+)"', source("app.js")))
    for name in ("worker.js", "python-boot.js"):  # each names a file by its address beside the worker
        loaded |= _local(re.findall(r'new URL\("([\w./-]+\.\w+)", \w+\)', source(name)))
    loaded |= _local(re.findall(r'import\("\./([^"]+)"\)', source("feedback.js")))
    started = {"worker.js", "python-boot.js", "bootstrap.py", "runtime.json"}
    assert {"style.css", "app.js", "demo.json"} | started <= loaded, "the search found the loads"
    assert "vendor/modern-screenshot.mjs" in loaded
    assert loaded - BUILT - CATALOG_WRITES <= set(build_script.APP_FILES)
    assert all((WEB / name).is_file() for name in build_script.APP_FILES)


def _with_its_imports(name: str, found: list[str] | None = None) -> list[str]:
    """Return a module of the page and every module it imports, however far down, each once."""
    found = [] if found is None else found
    if name not in found:
        found.append(name)
        for imported in re.findall(r'from "\./([\w.-]+)"', source(name)):
            _with_its_imports(imported, found)
    return found


def test_the_page_asks_for_every_module_it_imports_at_once():
    """index.html preloads app.js and all it imports, so no module waits for the one that imports it to be read."""
    preloaded = re.findall(r'<link rel="modulepreload" href="([^"]+)">', source("index.html"))
    assert sorted(preloaded) == sorted(_with_its_imports("app.js")), "each module the page imports, and no other"
    assert len(preloaded) == len(set(preloaded))


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
    (out / "catalog" / "index.json").write_text("{}", encoding="utf-8")
    (out / "catalog" / "detail" / "generated.json").write_text("{}", encoding="utf-8")


def _stand_in_reproduction(out: Path) -> None:
    """Write ``catalog/reproduction.json`` where ``write_reproduction`` does, without recomputing every source."""
    (out / "catalog" / "reproduction.json").write_text("{}", encoding="utf-8")


def _refused(_name: str, _release: str) -> dict:
    """Stand in for PyPI where a test's build must not ask it."""
    raise OSError("no network in this test")


def _no_bytecode(build_script, monkeypatch) -> None:
    """Leave the app's Python as it is: a stand-in wheel is no zip, and compiling asks uv for another Python.

    The build asks nothing of the network either: PyPI is refused, so the manifest holds no exact files.
    """
    monkeypatch.setattr(build_script, "for_the_browser", lambda _wheel, _python: False)
    monkeypatch.setattr(build_script, "compile_bytecode", lambda _tree, _python: False)
    monkeypatch.setattr(build_script, "_pypi_release", _refused)


def test_assemble_lays_out_what_the_page_and_its_worker_read(build_script, tmp_path, monkeypatch):
    """The page, its Python in a folder of its own under wheels/, the manifest naming it, and the worked example."""
    from importlib.metadata import version

    get_plotlyjs_version = pytest.importorskip(
        "plotly.offline", reason="the build names plotly's versions"
    ).get_plotlyjs_version

    def fake_wheel(_dist: Path) -> Path:
        wheel = tmp_path / "process_improve-9.9.9-py3-none-any.whl"
        wheel.write_bytes(b"a wheel")
        return wheel

    monkeypatch.setattr(build_script, "build_wheel", fake_wheel)
    _no_bytecode(build_script, monkeypatch)
    monkeypatch.setattr(build_script, "worked_example", lambda: {"the": "worked example"})
    monkeypatch.setattr(build_script, "book_example", lambda entry: {"name": entry["name"], "title": "", "book": ""})
    monkeypatch.setattr(build_script, "published_example", lambda entry: {"name": entry["name"], "title": ""})
    monkeypatch.setattr(build_script, "write_catalog", _stand_in_catalog)
    monkeypatch.setattr(build_script, "write_reproduction", _stand_in_reproduction)
    out = tmp_path / "app"
    (out / "stale").mkdir(parents=True)
    assert build_script.assemble(out, dist=tmp_path) == "process_improve-9.9.9-py3-none-any.whl"

    published = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
    packed = (tmp_path / build_script.GLUE).read_bytes()
    folder = hashlib.sha256(b"a wheel" + packed).hexdigest()[:16]
    wheel, glue = f"{folder}/process_improve-9.9.9-py3-none-any.whl", f"{folder}/{build_script.GLUE}"
    with zipfile.ZipFile(io.BytesIO(packed)) as archive:
        assert archive.read("bootstrap.py") == (WEB / "bootstrap.py").read_bytes(), "the page's Python, as it is"
    examples = [*build_script.book_examples(), *build_script.published_examples()]
    book = {"book/index.json", *(f"book/{entry['name']}.json" for entry in examples)}
    catalog = {"catalog/index.json", "catalog/detail/generated.json", "catalog/reproduction.json"}
    python = {f"wheels/{wheel}", f"wheels/{glue}"}
    assert published == {*build_script.APP_FILES, *BUILT, *book, *catalog, *python}, "and nothing from before"
    for name in build_script.APP_FILES:
        assert (out / name).read_bytes() == (WEB / name).read_bytes(), name
    manifest = json.loads((out / "wheels" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest == {
        "wheel": wheel,
        "glue": glue,
        "bytecode": None,
        "service_worker": True,
        "commit": build_script.commit(),
        "plotly": version("plotly"),
        "plotly_js": get_plotlyjs_version(),
        "on_demand": None,
    }
    assert json.loads((out / "demo.json").read_text(encoding="utf-8")) == {"the": "worked example"}


def _assemble_around(build_script, out: Path, wheel: bytes, monkeypatch) -> dict:
    """Assemble the app into ``out`` around a stand-in wheel holding ``wheel``; return the manifest."""
    pytest.importorskip("plotly.offline", reason="the build names plotly's versions")
    built = out.parent / "dist" / "process_improve-9.9.9-py3-none-any.whl"
    built.parent.mkdir(exist_ok=True)
    built.write_bytes(wheel)
    monkeypatch.setattr(build_script, "build_wheel", lambda _dist: built)
    _no_bytecode(build_script, monkeypatch)
    monkeypatch.setattr(build_script, "worked_example", dict)
    monkeypatch.setattr(build_script, "write_book", lambda _out: None)
    monkeypatch.setattr(build_script, "write_catalog", lambda _out: None)
    monkeypatch.setattr(build_script, "write_reproduction", lambda _out: None)
    build_script.assemble(out, dist=built.parent)
    return json.loads((out / "wheels" / "manifest.json").read_text(encoding="utf-8"))


def test_each_build_s_python_has_a_url_of_its_own(build_script, tmp_path, monkeypatch):
    """A browser keeps the wheel and the page's Python for good, so neither may change at an address it has seen."""
    published = []
    for wheel in (b"one build", b"the next build"):
        manifest = _assemble_around(build_script, tmp_path / "app", wheel, monkeypatch)
        assert (tmp_path / "app" / "wheels" / manifest["wheel"]).read_bytes() == wheel
        assert Path(manifest["wheel"]).name == "process_improve-9.9.9-py3-none-any.whl", "micropip reads the name"
        assert Path(manifest["glue"]).parent == Path(manifest["wheel"]).parent, "one folder, named by both"
        published.append(manifest["wheel"])
    assert published[0] != published[1]


def test_a_change_to_the_page_s_python_alone_moves_both_to_a_new_address(build_script, tmp_path, monkeypatch):
    """The wheel a browser kept is never run beside a ``bootstrap.py`` of another build."""
    before = _assemble_around(build_script, tmp_path / "app", b"a wheel", monkeypatch)
    changed = tmp_path / "web"
    shutil.copytree(WEB, changed)
    with (changed / "bootstrap.py").open("a", encoding="utf-8") as glue:
        glue.write("# changed\n")
    monkeypatch.setattr(build_script, "WEB_SRC", changed)
    after = _assemble_around(build_script, tmp_path / "app", b"a wheel", monkeypatch)
    assert Path(before["wheel"]).parent != Path(after["wheel"]).parent


def _stand_in_wheel(path: Path, left_out: str) -> None:
    """Write a small wheel: a module, a file of data it keeps, and one in the folder the browser's copy leaves out."""
    files = {"pkg/__init__.py": "ANSWER = 42\n", "pkg/data/kept.csv": "1,2\n", f"{left_out}/big.json": "{}\n"}
    record = "".join(f"{name},,\n" for name in [*files, "pkg-1.0.dist-info/RECORD"])
    with zipfile.ZipFile(path, "w") as packed:
        for name, text in {**files, "pkg-1.0.dist-info/RECORD": record}.items():
            packed.writestr(name, text)


def test_the_browser_s_copy_of_the_wheel_leaves_out_what_the_page_never_reads(build_script, tmp_path, monkeypatch):
    """The catalogue's sources stay out, the rest stays in, and the wheel's RECORD lists what it then holds."""
    monkeypatch.setattr(build_script, "compile_bytecode", lambda _tree, _python: False)
    wheel = tmp_path / "pkg-1.0-py3-none-any.whl"
    _stand_in_wheel(wheel, build_script.NOT_FOR_THE_PAGE)
    assert not build_script.for_the_browser(wheel, "3.14"), "no bytecode was made"
    with zipfile.ZipFile(wheel) as packed:
        names = packed.namelist()
        record = packed.read("pkg-1.0.dist-info/RECORD").decode().splitlines()
    assert names == ["pkg/__init__.py", "pkg/data/kept.csv", "pkg-1.0.dist-info/RECORD"]
    assert record == ["pkg/__init__.py,,", "pkg/data/kept.csv,,", "pkg-1.0.dist-info/RECORD,,"]
    assert (ROOT / "src" / build_script.NOT_FOR_THE_PAGE).is_dir(), "the folder the package keeps them in"


@pytest.mark.slow
def test_the_wheel_s_bytecode_is_listed_in_it_and_never_checked_against_its_source(build_script, tmp_path, monkeypatch):
    """Compiled by the Python the page runs, which uv provides; skipped where uv cannot provide it.

    The build may run where ``PYTHONOPTIMIZE`` is set, as one row of CI does: the bytecode is the browser's all the
    same, which runs Python without ``-O`` and reads no ``.opt-1.pyc``.
    """
    monkeypatch.setenv("PYTHONOPTIMIZE", "1")
    python = build_script.runtime()["python"]
    wheel = tmp_path / "pkg-1.0-py3-none-any.whl"
    _stand_in_wheel(wheel, build_script.NOT_FOR_THE_PAGE)
    if not build_script.for_the_browser(wheel, python):
        pytest.skip(f"uv could not provide Python {python} here")
    compiled_name = f"pkg/__pycache__/__init__.cpython-{python.replace('.', '')}.pyc"
    with zipfile.ZipFile(wheel) as packed:
        names = packed.namelist()
        compiled = packed.read(compiled_name)
        record = packed.read("pkg-1.0.dist-info/RECORD").decode()
    assert names[-1] == "pkg-1.0.dist-info/RECORD", "a wheel lists its .dist-info folder last"
    assert int.from_bytes(compiled[4:8], "little") == 0b01, "hash-based, and not to be checked (PEP 552)"
    assert f"{compiled_name},sha256=" in record
    again = tmp_path / "again.whl"
    _stand_in_wheel(again, build_script.NOT_FOR_THE_PAGE)
    assert build_script.for_the_browser(again, python)
    assert again.read_bytes() == wheel.read_bytes(), "the same wheel gives the same copy, so the same address"


def test_the_manifest_names_the_commit_the_app_was_built_from(build_script, tmp_path, monkeypatch):
    """The deploy waits for the published manifest to name the commit it pushed: CI's, or this checkout's."""
    monkeypatch.setenv("GITHUB_SHA", "0123abc")
    assert _assemble_around(build_script, tmp_path / "ci", b"a wheel", monkeypatch)["commit"] == "0123abc"
    monkeypatch.delenv("GITHUB_SHA")
    assert re.fullmatch(
        r"[0-9a-f]{40}", _assemble_around(build_script, tmp_path / "here", b"a wheel", monkeypatch)["commit"]
    )


# --------------------------------------------------------------------------- the packages installed on demand
def _release(files: list[str], requires: list[str] | None = None) -> dict:
    """Return PyPI's record of a release, as far as the build reads it."""
    urls = [{"filename": name, "url": f"https://files.pythonhosted.org/packages/ab/cd/{name}"} for name in files]
    return {"urls": urls, "info": {"requires_dist": requires}}


#: PyPI's records of the releases the page installs on demand, with the kinds of requirement a release lists.
RELEASES = {
    ("openpyxl", "3.1.5"): _release(["openpyxl-3.1.5.tar.gz", "openpyxl-3.1.5-py2.py3-none-any.whl"], ["et-xmlfile"]),
    ("et-xmlfile", "2.0.0"): _release(["et_xmlfile-2.0.0-py3-none-any.whl"]),
    ("plotly", "7.1.0"): _release(
        ["plotly-7.1.0-py3-none-any.whl", "plotly-7.1.0.tar.gz"],
        [
            "narwhals>=1.15.1",
            "packaging",
            'kaleido>=1.1.0; extra == "kaleido"',
            'tzdata; sys_platform == "win32"',
            'typing_extensions; python_version < "3.11"',
        ],
    ),
}
PINS = {"openpyxl": ["openpyxl==3.1.5", "et-xmlfile==2.0.0"], "plotly": ["plotly==7.1.0"]}


def test_the_packages_installed_on_demand_are_resolved_to_exact_files(build_script, monkeypatch):
    """Each pin's wheel by its address, and what the wheels need of the Pyodide release, for the browser's Python."""
    monkeypatch.setattr(build_script, "_pypi_release", lambda name, release: RELEASES[name, release])
    at = "https://files.pythonhosted.org/packages/ab/cd/"
    assert build_script.exact_files(PINS, "3.14") == {
        "openpyxl": {
            "packages": [],
            "wheels": [f"{at}openpyxl-3.1.5-py2.py3-none-any.whl", f"{at}et_xmlfile-2.0.0-py3-none-any.whl"],
        },
        "plotly": {"packages": ["narwhals", "packaging"], "wheels": [f"{at}plotly-7.1.0-py3-none-any.whl"]},
    }
    old = build_script.exact_files({"plotly": ["plotly==7.1.0"]}, "3.10")
    assert old["plotly"]["packages"] == ["narwhals", "packaging", "typing-extensions"], "read for the Python named"


@pytest.mark.parametrize(
    ("pins", "pypi"),
    [
        (PINS, _refused),
        ({"plotly": ["plotly"]}, RELEASES.get),  # no pin to ask PyPI about
        (
            {"plotly": ["plotly==7.1.0"]},
            lambda *_: _release(["plotly-7.1.0-cp314-linux_x86_64.whl"]),
        ),  # no wheel for it
    ],
)
def test_a_build_that_cannot_resolve_them_goes_on_without_and_says_so(build_script, monkeypatch, capsys, pins, pypi):
    monkeypatch.setattr(build_script, "_pypi_release", pypi)
    assert build_script.exact_files(pins, "3.14") is None
    assert "note: built without the exact files of the packages installed on demand" in capsys.readouterr().out


def test_the_build_resolves_what_the_worker_installs_from_pypi_at_the_pins_the_page_has(build_script, node):
    """The worker's table (``onDemand`` in web/python-boot.js) holds a package's pins where PyPI has the package."""
    runtime = json.loads(source("runtime.json"))
    (table,) = javascript(node, source("python-boot.js"), [("onDemand", [runtime, {"plotly": "7.1.0"}])])
    from_pypi = {name: entry["pins"] for name, entry in table.items() if "pins" in entry}
    assert from_pypi == build_script.on_demand_pins("7.1.0")
    assert from_pypi == {"openpyxl": runtime["micropip"], "plotly": ["plotly==7.1.0"]}
    # What a model is fitted with is of the Pyodide release, whose folder holds its files: nothing to resolve.
    fitted_with = [entry["exact"] for entry in table.values() if "pins" not in entry]
    assert fitted_with == [{"packages": runtime["fitting"], "wheels": []}] * 4


def test_the_manifest_holds_the_exact_files_the_build_resolved(build_script, tmp_path, monkeypatch):
    manifest = _assemble_around(build_script, tmp_path / "app", b"a wheel", monkeypatch)
    assert manifest["on_demand"] is None, "PyPI could not be asked"
    monkeypatch.setattr(build_script, "exact_files", lambda pins, python: {"pins": pins, "python": python})
    built = tmp_path / "again" / "app"
    built.parent.mkdir()
    again = _assemble_around(build_script, built, b"a wheel", monkeypatch)
    from importlib.metadata import version

    python = json.loads(source("runtime.json"))["python"]
    assert again["on_demand"] == {"pins": build_script.on_demand_pins(version("plotly")), "python": python}


def test_the_page_reads_only_what_the_manifest_holds():
    """The worker installs the wheel and the plotly release named there, from the exact files in ``on_demand`` when
    the build resolved them; app.js draws with the Plotly.js named there, and a report from the Feedback dialog
    names the commit; the copy kept for use without a network follows ``service_worker``. ``bytecode`` is for a
    reader of the manifest, and for the tests.
    """
    read = {
        name
        for file in ("worker.js", "python-boot.js", "offline.js")
        for name in re.findall(r"(?<![/\w\"])manifest\.(\w+)", source(file))
    }
    for names in re.findall(r"const \{ ([^}]+) \} = manifest;", source("app.js")):
        read |= {name.split(":")[0].strip() for name in names.split(",")}
    assert read == {"wheel", "glue", "plotly", "plotly_js", "on_demand", "commit", "service_worker"}


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
    index = json.loads((book / "index.json").read_text(encoding="utf-8"))["examples"]
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
    written = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in book.glob("*.json") if p.name != "index.json"}
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
