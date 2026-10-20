"""Helpers shared by the browser app's tests: where the app lives, how to call it, and what its page holds.

The app's Python side, ``web/bootstrap.py``, is loaded from its file the way ``web/worker.js``
loads it into Pyodide, and called through its JSON entry points the way ``web/app.js`` calls
it: a JSON payload in, a JSON reply out. :class:`Api` makes those calls, and reads every reply
as the browser's ``JSON.parse`` reads it, so a reply the page could not read fails every test.
"""

from __future__ import annotations

import base64
import contextlib
import html.parser
import importlib.util
import io
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "web"
BUILD_SCRIPT = ROOT / "scripts" / "build_web_app.py"

#: Every file the app is built from: what ``web/`` holds, and the script that assembles it.
#: ``--affected-since`` maps a changed file outside ``src/`` and ``tests/`` to the test files
#: that name it (``tests/_selection.py``). Every test here imports this module, directly or
#: through ``conftest.py``, so naming the files here sends a change to any of them to all of
#: ``tests/web``. ``page/test_build.py`` checks that the list is complete.
APP_SOURCES = (
    "index.html",
    "about.html",
    "catalog/index.html",
    "catalog.js",
    "catalog-logic.js",
    "catalog-views.js",
    "style.css",
    "app.js",
    "logic.js",
    "limits.js",
    "corners.js",
    "picture.js",
    "charts.js",
    "python.js",
    "best.js",
    "report.js",
    "campaign.js",
    "store.js",
    "glossary.js",
    "terms.js",
    "dialogs.js",
    "feedback.js",
    "vendor/modern-screenshot.mjs",
    "worker.js",
    "bootstrap.py",
    "runtime.json",
    "favicon.svg",
    "apple-touch-icon.png",
    "icon-192.png",
    "icon-512.png",
    "og-card.png",
    "manifest.webmanifest",
    "book-examples.json",
    "published-examples.json",
    "build_web_app.py",
)

#: The names ``web/worker.js`` answers (its ``ENTRY_POINT``): the ``api_*`` entry points of ``bootstrap.py``.
ENTRY_POINT = re.compile(r"^api_[a-z_]+$")

#: Every design the page offers, in the order it lists them. The tests run over this list, and
#: ``test_designs`` checks it against ``bootstrap.DESIGNS``, so a new design cannot go untested.
ALL_DESIGNS = (
    "full_factorial",
    "fractional_factorial",
    "plackett_burman",
    "taguchi",
    "dsd",
    "box_behnken",
    "ccd",
    "d_optimal",
    "i_optimal",
    "maximin_lhs",
)

#: The designs the book has no page on: the page shows their hint without a link to it.
NO_BOOK_PAGE = frozenset({"taguchi", "maximin_lhs"})
#: Every mixture design the page offers, kept apart from ``ALL_DESIGNS`` as ``bootstrap.MIXTURES`` is from
#: ``DESIGNS``: its factors are the components of a blend. ``api/test_mixture.py`` checks it against ``MIXTURES``.
ALL_MIXTURES = ("simplex_lattice", "simplex_centroid", "extreme_vertices")

#: Every process the page simulates results from; ``test_simulation`` checks it against ``SCENARIOS``.
ALL_SCENARIOS = ("reactor", "main_effects", "interaction", "optimum", "nothing", "random")

#: Three factors in their own units, as the worked example has them.
FACTORS = [
    {"name": "T", "low": 150, "high": 200, "units": "degC"},
    {"name": "P", "low": 1, "high": 3, "units": "bar"},
    {"name": "F", "low": 10, "high": 20, "units": "L/min"},
]


def factors_named(*names: str) -> list[dict]:
    """Return factors with the given names, each from -1 to +1, so a setting and its coded value agree."""
    return [{"name": n, "low": -1, "high": 1, "units": ""} for n in names]


#: Runs as given in factors A, B and C, each from -1 to +1: A and B are equal in all but the first run, where B
#: is at its center, so the main effects model fits A and B with variance inflation factors of 10.8 and 11.
CORRELATED_RUNS = [[-1, 0, -1], [-1, -1, 1], [1, 1, -1], [1, 1, 1], [-1, -1, -1], [-1, -1, 1], [1, 1, -1], [1, 1, 1]]


def correlated_study() -> dict:
    """Return the request for :data:`CORRELATED_RUNS` as a design of runs as given."""
    return {"design_type": "given", "factors": factors_named("A", "B", "C"), "runs": CORRELATED_RUNS}


# --------------------------------------------------------------------------- loading the app's files
@contextlib.contextmanager
def no_bytecode() -> Iterator[None]:
    """Write no ``__pycache__`` beside the files loaded inside the block: ``web/`` and ``scripts/`` are not packages."""
    before, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        yield
    finally:
        sys.dont_write_bytecode = before


def load_module(name: str, path: Path) -> ModuleType:
    """Load a module from its file, as ``web/worker.js`` loads ``bootstrap.py`` into Pyodide."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    with no_bytecode():
        spec.loader.exec_module(module)
    return module


@cache
def source(name: str) -> str:
    """Return the text of one of the app's files in ``web/``."""
    return (WEB / name).read_text(encoding="utf-8")


# --------------------------------------------------------------------------- calling the app
def _strict(constant: str) -> NoReturn:
    """Refuse what Python's JSON reader accepts and the browser's ``JSON.parse`` does not: NaN and infinities."""
    message = f"the reply holds {constant}, which JSON.parse in the page cannot read"
    raise AssertionError(message)


@dataclass(frozen=True)
class Api:
    """The entry points of ``web/bootstrap.py``, called as the page calls them: a JSON payload in, a JSON reply out."""

    module: ModuleType

    def reply(self, fn: str, payload: object = None, *, text: str | None = None) -> dict[str, Any]:
        """Return the whole reply, ``ok`` then ``result`` or ``error``, and ``warnings``; ``text`` is sent as it is."""
        sent = json.dumps({} if payload is None else payload) if text is None else text
        return json.loads(getattr(self.module, fn)(sent), parse_constant=_strict)

    def ok(self, fn: str, payload: object = None) -> dict[str, Any]:
        """Return the result of a call the page expects to succeed."""
        answer = self.reply(fn, payload)
        assert answer["ok"], f"{fn} refused: {answer.get('error')}"
        return answer["result"]

    def error(self, fn: str, payload: object = None) -> str:
        """Return the message of a call the page expects to be refused."""
        answer = self.reply(fn, payload)
        assert not answer["ok"], f"{fn} accepted a request it should refuse"
        return answer["error"]


def held_of(design: dict) -> dict:
    """Return what the page holds of a generated design, and sends back with every later call."""
    return {"spec": design["spec"], "columns": design["columns"], "rows": design["rows"]}


def generate_all(
    api: Api, factors: list[dict], model: str, constraints: list[dict] | None = None
) -> tuple[list[dict], list[dict]]:
    """Generate every design of a comparison as the page does, one call each; return them, and what the page holds.

    With ``constraints``, the limits on the settings, only the designs that keep to them are made.
    """
    plan = api.ok("api_compare_plan", {"factors": factors, "model": model, "constraints": constraints or []})
    made = [{**item, "design": api.ok("api_make_design", item["spec"])} for item in plan["designs"]]
    return made, [{"label": m["label"], **held_of(m["design"])} for m in made]


# --------------------------------------------------------------------------- the run sheet and its workbook
def run_sheet(result: dict) -> pd.DataFrame:
    """Return the run sheet a reply describes, as the reader sees it."""
    return pd.DataFrame(result["rows"], columns=result["columns"])


def coded(runs: pd.DataFrame, factors: list[dict]) -> pd.DataFrame:
    """Return the runs' settings coded from each factor's range: -1 at its low value, +1 at its high value.

    A categorical factor's levels are labels: its low label is coded -1 and its high one +1.
    """

    def column(f: dict) -> pd.Series:
        if isinstance(f["low"], str):
            return runs[f["name"]].map({f["low"]: -1.0, f["high"]: 1.0})
        return (runs[f["name"]] - (f["high"] + f["low"]) / 2) / ((f["high"] - f["low"]) / 2)

    return pd.DataFrame({f["name"]: column(f) for f in factors})


def quadratic(runs: pd.DataFrame) -> np.ndarray:
    """Return a known response over ``FACTORS``: 50 + 4 T - 3 P + 2 T P - 5 T² in coded units, plus a little noise."""
    t, p = (runs["T"] - 175) / 25, runs["P"] - 2
    noise = np.random.default_rng(0).normal(0, 0.1, len(runs))
    return (50 + 4 * t - 3 * p + 2 * t * p - 5 * t**2 + noise).to_numpy()


def sheets_of(xlsx_base64: str) -> dict[str, pd.DataFrame]:
    """Read every sheet of a workbook the page wrote."""
    return pd.read_excel(io.BytesIO(base64.b64decode(xlsx_base64)), sheet_name=None)


def edit_workbook(xlsx_base64: str, edit: Callable[[dict[str, pd.DataFrame]], object]) -> str:
    """Change a workbook's sheets as a reader could in Excel, and return it as the page uploads it."""
    sheets = sheets_of(xlsx_base64)
    edit(sheets)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as xl:
        for name, frame in sheets.items():
            frame.to_excel(xl, sheet_name=name, index=False)
    return base64.b64encode(buf.getvalue()).decode()


def fill_workbook(
    xlsx_base64: str, response: Callable[[pd.DataFrame], object], blank_rows: tuple[int, ...] = ()
) -> str:
    """Fill the Runs sheet's response column as a reader would, in Excel, leaving the runs in ``blank_rows`` to do."""

    def fill(sheets: dict[str, pd.DataFrame]) -> None:
        runs = sheets["Runs"]
        name = json.loads(sheets["_pi_design"]["spec"].iloc[0])["response"]
        runs[name] = response(runs)
        runs.loc[list(blank_rows), name] = np.nan

    return edit_workbook(xlsx_base64, fill)


# --------------------------------------------------------------------------- the page
class Page(html.parser.HTMLParser):
    """``web/index.html`` as the tests read it: every element that has an id, the options of every <select>,
    the choices of every group of radio buttons, by its name, and the glossary entry of every word marked as a term.
    """

    def __init__(self, text: str) -> None:
        super().__init__()
        self.ids: dict[str, dict[str, str | None]] = {}
        self.options: dict[str, list[dict[str, str | None]]] = {}
        self.data_options: list[str] = []
        self.radios: dict[str, list[dict[str, str | None]]] = {}
        self.terms: list[str] = []
        self._select: str | None = None
        self.feed(text)
        self.close()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Keep the element's attributes under its id, and each <option> under its <select>."""
        found = dict(attrs)
        if found.get("id"):
            self.ids[str(found["id"])] = {"tag": tag, **found}
        if found.get("data-option"):
            self.data_options.append(str(found["data-option"]))
        if found.get("data-term"):
            self.terms.append(str(found["data-term"]))
        if tag == "select":
            self._select = str(found.get("id"))
            self.options[self._select] = []
        elif tag == "option" and self._select is not None:
            self.options[self._select].append({**found, "label": ""})
        elif tag == "input" and found.get("type") == "radio":
            self.radios.setdefault(str(found.get("name")), []).append(found)

    def handle_endtag(self, tag: str) -> None:
        """Leave a <select>."""
        if tag == "select":
            self._select = None

    def handle_data(self, data: str) -> None:
        """Read an option's label."""
        if self._select is not None and self.options[self._select]:
            option = self.options[self._select][-1]
            option["label"] = f"{option['label'] or ''}{data.strip()}"

    def values(self, select: str) -> list[str]:
        """Return the values of a <select>'s options, in order."""
        return [str(o["value"]) for o in self.options[select]]

    def selected(self, select: str) -> str:
        """Return the value a <select> opens on: the option marked ``selected``, else the first, as a browser does."""
        marked = [o for o in self.options[select] if "selected" in o]
        return str((marked or self.options[select])[0]["value"])

    def checked(self, name: str) -> str | None:
        """Return the value of the radio button of the group ``name`` the page opens on, or None for none."""
        marked = [r for r in self.radios[name] if "checked" in r]
        return str(marked[0]["value"]) if marked else None


@cache
def page() -> Page:
    """Return ``web/index.html``, read once."""
    return Page(source("index.html"))


# --------------------------------------------------------------------------- the page's JavaScript, under Node
#: Read a JSON request on stdin, import the module source it carries, call the functions it
#: names (or read the constants) and print what they return. The module is imported from a
#: data: URL, which Node reads
#: as an ES module whatever its version and whatever package.json sits above the checkout, so
#: nothing is written beside the source.
_DRIVER = """
let input = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => (input += chunk));
process.stdin.on("end", async () => {
  const { source, calls } = JSON.parse(input);
  const module = await import(`data:text/javascript;charset=utf-8,${encodeURIComponent(source)}`);
  const value = (name, args) => (typeof module[name] === "function" ? module[name](...args) : module[name]);
  process.stdout.write(JSON.stringify(calls.map(([name, args]) => value(name, args))));
});
"""


def javascript(node: str, module_source: str, calls: list[tuple[str, list]]) -> list:
    """Call functions of an ES module, given as source text, under Node; return what each call returns.

    A name that is not a function is read instead: ``("SPARE_RUNS", [])`` returns the constant.
    """
    done = subprocess.run(  # noqa: S603 - Node, found on PATH, runs the page's own code
        [node, "--input-type=module", "-e", _DRIVER],
        input=json.dumps({"source": module_source, "calls": calls}),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


# --------------------------------------------------------------------------- the design catalog
#: Published entries and generated designs that show each way the catalog opens a design, or does not:
#: a published experiment the page regenerates, with its results; a comparison, no design the page makes;
#: a generated design the page makes; one it does not; and the 2^3 factorial, which is the published
#: experiment's design.
CATALOG_SAMPLE = (
    "bhh-pilot-plant-2-3",
    "bhh-plant-trial-two-methods",
    "lib-ccd-k3-face-centered",
    "lib-omars-k4",
    "lib-full-factorial-k3",
)


#: A design printed in the book, whose page the catalog links to at its section: a case study of a fed-batch
#: bioreactor, with figures whose notes name their files. It is not in :data:`CATALOG_SAMPLE`; the tests of
#: a book design's place build a catalog of their own with it.
BOOK_DESIGN = "omars-fed-batch-bioreactor"


def catalog_sample(names: tuple[str, ...] = CATALOG_SAMPLE) -> list[dict]:
    """Describe the designs named, those of :data:`CATALOG_SAMPLE` by default, as ``catalog_records`` does."""
    from process_improve.experiments import design_catalog
    from process_improve.experiments.reference_catalogue import (
        reference_designs,
        reference_experiments,
        reference_sources,
    )

    sources = reference_sources()
    listed = [*(("experiments", e) for e in reference_experiments()), *(("designs", d) for d in reference_designs())]
    published = [
        {
            **design_catalog._published_record(e, sources[e["source"]]),
            "checks": design_catalog.printed_checks(e),
            "listed": where,
        }
        for where, e in listed
        if e["name"] in names
    ]
    generated = [
        {**design_catalog._generated_record(spec), "checks": None, "listed": None}
        for spec in design_catalog.GENERATED
        if design_catalog._spec_id(spec) in names
    ]
    return design_catalog._classified([*published, *generated])
