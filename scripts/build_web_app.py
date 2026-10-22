"""Assemble the no-install browser app into a directory that can be served.

Builds the wheel from this checkout, copies ``web/`` next to it and writes the
``wheels/manifest.json`` that ``web/worker.js`` reads, so the published page
always runs the code it was deployed from rather than the latest PyPI release.
The wheel goes in a folder named by its digest, so each deploy's wheel has a URL
of its own and a browser cannot install one it cached from an earlier deploy; the
manifest, fetched past the cache, names it and the commit the app was built from.
It also runs the page's worked example through ``web/bootstrap.py``, the code the
page runs, and writes the results to ``demo.json``: the page draws them at first
paint while Python loads in the browser. Each of the book's examples in
``web/book-examples.json`` is made the same way into ``book/<name>.json``, which the page
opens from ``?book=<name>``, a link the book prints; so is each example of another published
source in ``web/published-examples.json``, opened from ``?example=<name>`` with its citation.
The docs workflow runs this into
``docs/_build/html/app``.

    uv run python scripts/build_web_app.py --serve

``--serve`` starts a plain HTTP server: Pyodide fetches its files, so opening
``index.html`` from disk (``file://``) does not work.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import http.server
import importlib.util
import itertools
import json
import math
import os
import re
import shutil
import subprocess
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

ROOT = Path(__file__).resolve().parent.parent
WEB_SRC = ROOT / "web"
APP_FILES = (
    "index.html",
    "about.html",
    "catalog/index.html",
    "catalog.js",
    "catalog-logic.js",
    "catalog-views.js",
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
    "study-file.js",
    "store.js",
    "glossary.js",
    "terms.js",
    "dialogs.js",
    "feedback.js",
    "vendor/modern-screenshot.mjs",
    "worker.js",
    "runtime.json",
    "bootstrap.py",
    "style.css",
    "favicon.svg",
    "apple-touch-icon.png",
    "icon-192.png",
    "icon-512.png",
    "og-card.png",
    "manifest.webmanifest",
)


def build_wheel(dist: Path) -> Path:
    """Build the pure-Python wheel into ``dist`` and return its path."""
    shutil.rmtree(dist, ignore_errors=True)
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("uv not found on PATH; see https://docs.astral.sh/uv/")
    subprocess.run([uv, "build", "--wheel", "--out-dir", str(dist)], cwd=ROOT, check=True)  # noqa: S603
    wheels = sorted(dist.glob("process_improve-*.whl"))
    if not wheels:
        raise SystemExit(f"no wheel produced in {dist}")
    return wheels[-1]


def commit() -> str | None:
    """Name the commit the app is built from: the one CI checked out, else this checkout's, if git can tell."""
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    git = shutil.which("git")
    if git is None:
        return None
    head = subprocess.run([git, "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)  # noqa: S603
    return head.stdout.strip() or None


@functools.cache
def load_bootstrap() -> ModuleType:
    """Load ``web/bootstrap.py`` from its file, as ``web/worker.js`` loads it into Pyodide."""
    spec = importlib.util.spec_from_file_location("web_bootstrap", WEB_SRC / "bootstrap.py")
    if spec is None or spec.loader is None:
        raise SystemExit("web/bootstrap.py could not be loaded")
    bootstrap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bootstrap)
    return bootstrap


def caller(what: str) -> Callable[[str, dict], Any]:
    """Return a way to call an entry point of ``web/bootstrap.py`` as the page does; a refusal stops the build."""
    bootstrap = load_bootstrap()

    def call(fn: str, payload: dict) -> Any:  # noqa: ANN401
        reply = json.loads(getattr(bootstrap, fn)(json.dumps(payload)))
        if not reply["ok"]:
            raise SystemExit(f"{fn} failed on {what}: {reply['error']}")
        return reply["result"]

    return call


def worked_example() -> dict[str, Any]:
    """Run the page's worked example through ``web/bootstrap.py``, as the page runs it live.

    Every key the page reads at first paint is a reply of the same entry point the page
    calls, so the example drawn before Python loads and the one computed in the browser
    are the same code's output.
    """
    call = caller("the worked example")
    inputs = load_bootstrap().DEMO
    design = call("api_make_design", inputs["design"])
    held = {"spec": design["spec"], "columns": design["columns"], "rows": design["rows"]}
    simulation = call(
        "api_simulate", {**held, "scenario": inputs["scenario"], "sigma": inputs["sigma"], "seed": inputs["seed"]}
    )
    request = {**held, "responses": simulation["responses"], "model": inputs["model"], "truth": simulation["truth"]}
    design.pop("xlsx_base64")  # the page writes the workbook when it is asked for

    # The designs that can learn what the example aims to, as Plan lists them: every design the page offers for
    # the same factors, made and compared on the nine measures, with power for the smallest change its response names.
    comparison = call(
        "api_candidates",
        {
            "factors": inputs["design"]["factors"],
            "model": inputs["design"]["aim"],
            "response": inputs["design"]["response"],
            "response_info": inputs["design"]["response_info"],
        },
    )
    analysis = call("api_analyze_runs", request)
    best = call("api_best_settings", request)  # the page finds the best settings for the goal once the model is fitted
    following = call("api_next_runs", request)  # and then what the results call for next
    return {
        "version": version("process-improve"),
        "inputs": inputs,
        "design": design,
        "simulation": simulation,
        "analysis": analysis,
        "best": best,
        "next": following,
        "plots": call("api_plots", {**request, **({"best": best["coded"]} if best.get("coded") else {})}),
        "comparison": comparison,
    }


#: The book's examples that the page opens by name: ``?book=<name>`` opens ``book/<name>.json``, built from each.
BOOK_EXAMPLES = WEB_SRC / "book-examples.json"
#: Examples from published sources other than the book, each with its citation: ``?example=<name>``
#: opens one from the same ``book/`` folder and index as the book's.
PUBLISHED_EXAMPLES = WEB_SRC / "published-examples.json"
#: Where the book is published; each example names its page under here.
BOOK = "https://learnche.org/pid/"
#: The name ``?example=`` gives the worked example (``WORKED`` in ``web/campaign.js``), which no other may take.
WORKED = "worked"
#: The latest campaign format the page reads (``CAMPAIGN_VERSION`` in ``web/campaign.js``).
CAMPAIGN_VERSION = 3
#: The format of a study of one factor (``ONE_FACTOR_FORMAT`` in ``web/campaign.js``): a page from before it could
#: not analyse one, and refuses it in a sentence. A study of more factors keeps format 1.
ONE_FACTOR_FORMAT = 2
#: The format of a study with a categorical factor that lists its levels (``LEVELS_FORMAT`` in ``web/campaign.js``):
#: three or more, or two that read as numbers, which a page of an earlier format would misread.
LEVELS_FORMAT = 3


def study_format(spec: dict[str, Any]) -> int:
    """Return the lowest campaign format that holds a study of this specification (``studyFormat`` in campaign.js)."""
    if any("levels" in f for f in spec["factors"]):
        return LEVELS_FORMAT
    return ONE_FACTOR_FORMAT if len(spec["factors"]) == 1 else 1


#: How near, in coded units, a run of the book must come to a run of the design: the book prints
#: a rotatable design's axial distance, the square root of 2, as 1.41.
CODED_TOLERANCE = 0.01


def book_examples() -> list[dict[str, Any]]:
    """Return the book's examples as ``web/book-examples.json`` lists them."""
    return json.loads(BOOK_EXAMPLES.read_text(encoding="utf-8"))["examples"]


def published_examples() -> list[dict[str, Any]]:
    """Return the examples of published sources that the page opens, as ``web/published-examples.json`` lists them."""
    return json.loads(PUBLISHED_EXAMPLES.read_text(encoding="utf-8"))["examples"]


def coded(setting: float | str, factor: dict[str, Any]) -> float:
    """Return a run's setting of a factor in coded units: the low level -1 and the high level +1.

    A categorical factor's levels are spread evenly from -1 at its first to +1 at its last, as the page codes them.
    """
    bootstrap = load_bootstrap()
    if bootstrap._categorical(factor):
        levels = bootstrap._levels_of(factor)
        return -1.0 + 2.0 * bootstrap._level_index(setting, levels) / (len(levels) - 1)
    centre, half_range = (factor["high"] + factor["low"]) / 2, (factor["high"] - factor["low"]) / 2
    return (float(setting) - centre) / half_range


def in_book_order(design: dict[str, Any], entry: dict[str, Any], days: dict[int, int]) -> list[list] | str:
    """Return the design's runs in the order of the book's table, or say which of the book's runs it lacks.

    Each of the book's runs is found among the design's by its coded settings, and by its block
    where the book gives one: ``days`` maps the book's day to the design's block. The row's
    standard order becomes its place in the book's table, its run order the one the book
    prints (else that place), and its block the book's day.
    """
    columns, factors = design["columns"], design["spec"]["factors"]
    at = {name: i for i, name in enumerate(columns)}
    n = len(entry["runs"])
    order, blocks = entry.get("order", range(1, n + 1)), entry.get("blocks", [None] * n)
    unused, rows = list(design["rows"]), []
    for number, (run, ran, day) in enumerate(zip(entry["runs"], order, blocks, strict=True), start=1):

        def matches(row: list, settings: list = run[:-1], day: int | None = day) -> bool:
            near = all(
                abs(coded(row[at[f["name"]]], f) - s) <= CODED_TOLERANCE for f, s in zip(factors, settings, strict=True)
            )
            return near and (day is None or row[at["Block"]] == days[day])

        found = next((i for i, row in enumerate(unused) if matches(row)), None)
        if found is None:
            return f"its run {number}, {run[:-1]}, is not a run of the design"
        row = list(unused.pop(found))
        row[at["Std order"]], row[at["Run order"]] = number, ran
        if day is not None:
            row[at["Block"]] = day
        rows.append(row)
    if unused:
        return f"the design has {len(unused)} runs more than the book lists"
    return rows


def book_example(entry: dict[str, Any]) -> dict[str, Any]:
    """Make one of the book's examples into what the page opens from ``?book=<name>``: its campaign and its page."""
    return {
        "name": entry["name"],
        "title": entry["title"],
        "book": BOOK + entry["page"],
        "note": entry.get("note", ""),
        "campaign": campaign_of(entry, f"the book's example {entry['name']!r}"),
    }


def published_example(entry: dict[str, Any]) -> dict[str, Any]:
    """Make an example of a published source into what the page opens from ``?example=<name>``.

    In place of the book's page it carries its source, as the reference catalogue cites it: the
    citation, where in the source its runs are printed, and the address the source is read at.
    """
    from process_improve.experiments.reference_catalogue import (  # noqa: PLC0415 - the docs environment has it
        reference_sources,
    )

    source = reference_sources()[entry["source"]]
    return {
        "name": entry["name"],
        "title": entry["title"],
        "source": {"cite": source["cite"], "where": entry["where"], "url": source["url"]},
        "note": entry.get("note", ""),
        "campaign": campaign_of(entry, f"the published example {entry['name']!r}"),
    }


def campaign_of(entry: dict[str, Any], what: str) -> dict[str, Any]:
    """Return the campaign an example opens: its design, and its runs and results in the source's order.

    The design is made as Plan makes it, and its runs put in the order of the source's table,
    each with the source's result, so the page's run sheet reads as the source's table does. The
    design may number its blocks otherwise than the source numbers its days: each way of
    pairing them is tried.
    """
    design = caller(what)("api_make_design", entry["design"])
    labels = sorted(set(entry.get("blocks", [])))
    attempts = [
        in_book_order(design, entry, dict(zip(labels, paired, strict=True)))
        for paired in itertools.permutations(labels)
    ]
    rows = next((rows for rows in attempts if isinstance(rows, list)), None)
    if rows is None:
        raise SystemExit(f"{what}: {attempts[0]}")
    return {
        "v": study_format(design["spec"]),
        "title": entry["title"],
        "design": {
            "spec": design["spec"],
            "columns": design["columns"],
            "rows": rows,
            "model": design["model"],
            "n_runs": len(rows),
            "facts": design["facts"],
            "outside_range": design["outside_range"],
        },
        "responses": [run[-1] for run in entry["runs"]],
        "sources": ["entered"] * len(rows),
        "model": entry.get("model", ""),
    }


def write_book(out_dir: Path) -> None:
    """Write ``book/<name>.json`` for each example, the book's and the published ones, and ``book/index.json``."""
    examples = [book_example(entry) for entry in book_examples()]
    examples += [published_example(entry) for entry in published_examples()]
    names = [example["name"] for example in examples]
    if len(set(names)) != len(names):
        raise SystemExit(f"Two examples share a name: {sorted({n for n in names if names.count(n) > 1})}")
    if WORKED in names:
        raise SystemExit(f"An example is named {WORKED!r}, which ?example= keeps for the worked example.")
    (out_dir / "book").mkdir()
    for example in examples:
        (out_dir / "book" / f"{example['name']}.json").write_text(json.dumps(example, separators=(",", ":")) + "\n")
    (out_dir / "book" / "index.json").write_text(json.dumps(book_index(examples), indent=1) + "\n")


def book_index(examples: list[dict[str, Any]]) -> dict[str, Any]:
    """Return ``book/index.json``: each example's name, title, and its page of the book or its source."""
    keys = ("name", "title", "book", "source")
    return {"examples": [{key: example[key] for key in keys if key in example} for example in examples]}


# --------------------------------------------------------------------------- the design catalog
#: The format of ``catalog/index.json`` and its detail files, which ``web/catalog.js`` reads.
CATALOG_VERSION = 1
#: What the catalog page reads of every design at once beyond the library's facets: whether the page's
#: design has the source's runs, whether the page opens the design, the detail file that holds it, and
#: whether its detail holds the measures the page compares designs on.
CATALOG_FIELDS = ("regenerable", "open", "detail", "compared")
#: The longest title and units a campaign may carry (``checkCampaign`` in ``web/campaign.js``).
TITLE_MOST, UNITS_MOST = 200, 32


def _coded_runs(design: dict[str, Any]) -> Counter:
    """Count a design's runs by their coded settings, rounded, as the page made them."""
    columns, factors = design["columns"], design["spec"]["factors"]
    at = {name: i for i, name in enumerate(columns)}
    return Counter(tuple(round(coded(row[at[f["name"]]], f), 2) for f in factors) for row in design["rows"])


def _campaign(title: str, design: dict[str, Any], responses: list[float | None]) -> dict[str, Any]:
    """Return what a link opens: a design the page made, and a result, or a blank, for each run."""
    return {
        "v": study_format(design["spec"]),
        "title": title[:TITLE_MOST],
        "design": {
            key: design[key] for key in ("spec", "columns", "rows", "model", "n_runs", "facts", "outside_range")
        },
        "responses": responses,
        "sources": ["entered" if v is not None else None for v in responses],
        "model": "",
    }


def generated_campaign(record: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """Return the campaign that opens a generated design, made by the page in its own randomised run order.

    The page's design must have the record's runs: the catalog draws one design and opens the same one.
    """
    spec = record["spec"]
    designs = load_bootstrap().DESIGNS
    if spec["design_type"] not in designs:
        return None, "the page does not make this kind of design"
    options = dict(spec["options"])
    if "array" in spec:  # an orthogonal array, all its columns: the page makes it when it takes every column
        info = designs[spec["design_type"]]
        if not info["min_factors"] <= len(record["factors"]) <= info["max_factors"]:
            return None, f"the page makes this design for {info['min_factors']} to {info['max_factors']} factors"
        levels = taguchi_options(record)
        if isinstance(levels, str):
            return None, levels
        options |= levels
    factors = [page_factor({**f, "units": ""}) for f in record["factors"]]
    asked = {"design_type": spec["design_type"], "factors": factors, "response": "y", **options}
    design = caller(f"the catalog's design {record['id']!r}")("api_make_design", asked)
    design.pop("xlsx_base64")
    drawn = Counter(tuple(round(v, 2) for v in row) for row in record["coded_runs"])
    if _coded_runs(design) != drawn:
        raise SystemExit(f"the catalog's design {record['id']!r}: the page makes other runs than the library drew")
    return _campaign(record["title"], design, [None] * design["n_runs"]), ""


def taguchi_options(record: dict[str, Any]) -> dict[str, Any] | str:
    """Return the page's options for a Taguchi array with a record's runs, or say why the page cannot make it.

    The page runs every numeric factor of an array at two levels, or every one at three (``n_levels``), and
    each categorical factor at its own: so the numeric factors' runs must all take two levels, or all three.
    """
    runs = record["coded_runs"]
    numeric = [j for j, f in enumerate(record["factors"]) if f.get("type") == "numeric"]
    counts = {len({round(row[j], 6) for row in runs}) for j in numeric}
    if counts <= {2}:
        return {}
    if counts == {3}:
        return {"n_levels": 3}
    return "the page runs every numeric factor of a Taguchi array at 2 levels, or every one at 3"


def _page_levels(factor: dict[str, Any]) -> list[str] | None:
    """Return a categorical factor's levels as the page labels them, in the source's order; None if it refuses them.

    The source lists them, or gives the two as its low and high, which then go low first, as the catalog codes
    them. A label may read as a number, such as lot 1 or 2: the factor is categorical because the source says so.
    """
    from process_improve.experiments.design_catalog import _levels  # noqa: PLC0415 - the docs environment has it

    try:
        return list(load_bootstrap()._categorical_factor(str(factor["name"]), _levels(factor)).levels or [])
    except ValueError:
        return None


def page_factor(factor: dict[str, Any]) -> dict[str, Any]:
    """Write a source's factor as the page asks for it: a numeric factor's range and units, a categorical's levels.

    A source may write a range high first, coding its larger setting -1; the page's range goes low to high, so it
    codes that factor the other way round (see :func:`page_coded_runs`), and the runs, in the factor's own units,
    are the source's.
    """
    if factor.get("type") == "categorical":
        return {"name": factor["name"], "type": "categorical", "levels": _page_levels(factor), "units": ""}
    units = str(factor.get("units") or "")[:UNITS_MOST]
    low, high = sorted((factor["low"], factor["high"]))
    return {"name": factor["name"], "low": low, "high": high, "units": units}


def page_coded_runs(record: dict[str, Any]) -> list[list[float]]:
    """Return a record's coded runs as the page codes them, every numeric factor's range low to high.

    A source that writes a range high first codes the factor's larger setting -1, and :func:`page_factor` gives
    the page that range low to high, so the same setting is +1 on the page: its column is turned over. Each run
    keeps its settings in the factor's own units.
    """
    turned = [
        -1.0 if f.get("type") == "numeric" and _page_factor(f) and f["low"] > f["high"] else 1.0
        for f in record["factors"]
    ]
    return [[v * sign for v, sign in zip(row, turned, strict=True)] for row in record["coded_runs"]]


def _too_many_levels(factor: dict[str, Any]) -> bool:
    """Say whether a categorical factor has more levels than the page takes."""
    return factor.get("type") == "categorical" and len(factor.get("levels") or []) > load_bootstrap().MAX_LEVELS


def _page_factor(factor: dict[str, Any]) -> bool:
    """Say whether the page takes a factor as the source gives it: a name, and a numeric range or categorical levels."""
    if not str(factor["name"]).isidentifier():
        return False
    if factor.get("type") == "categorical":
        return _page_levels(factor) is not None
    low, high = factor.get("low"), factor.get("high")
    numbers = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (low, high))
    return factor.get("type") == "numeric" and numbers and low != high


def _factors_refused(factors: list[dict[str, Any]]) -> str:
    """Say why the page refuses a design's factors, or return "": too many levels, or a name, type or range."""
    if any(map(_too_many_levels, factors)):
        return f"a categorical factor has more than the {load_bootstrap().MAX_LEVELS} levels the page takes"
    if not all(map(_page_factor, factors)):
        return "a factor's name, type or range is not one the page accepts"
    return ""


def _blank_label(factor: dict[str, Any]) -> bool:
    """Say whether a factor has a label a spreadsheet reads back as a blank cell, such as None: the page refuses it."""
    bootstrap = load_bootstrap()
    raw = factor.get("levels") or [factor.get("low"), factor.get("high")]
    return any(bootstrap._label(v) in bootstrap.BLANK_LABELS for v in raw)


def _example_of(record: dict[str, Any]) -> dict[str, Any] | str:
    """Write a published record as the book's examples are written, for :func:`campaign_of`; or say why it cannot be.

    The design's specification is what the page would make for it: its type, its factors, a center run for
    each one the source prints and, where the source repeats every other run, that many replicates. A
    central composite design's axial distance is read off the runs, and so are a Taguchi array's levels.
    """
    designs = load_bootstrap().DESIGNS
    design_type, factors, runs = record["design_type"], record["factors"], record["coded_runs"]
    info = designs[design_type]
    if record["split_plot"] or record["blocked"]:
        whole_plots = "the page opens designs without whole plots"
        return whole_plots if record["split_plot"] else "the page makes its designs in blocks of its own"
    if not runs or any(v is None for row in runs for v in row):
        return "the source prints no complete table of runs"
    if not info["min_factors"] <= len(factors) <= info["max_factors"]:
        return f"the page makes this design for {info['min_factors']} to {info['max_factors']} factors"
    if why := _factors_refused(factors):
        return why
    runs = page_coded_runs(record)  # in the page's coding, so each result is at the source's settings
    names = [f["name"] for f in factors]
    response = next(
        (r["name"] for r in record["responses"] if r["name"].isidentifier() and r["name"] not in names), "y"
    )
    options: dict[str, Any] = {}
    if "n_center_points" in info["options"]:
        options["n_center_points"] = record["n_centre"] or 0
    repeats = Counter(tuple(row) for row in runs if any(v != 0 for v in row))
    if len(set(repeats.values())) == 1 and (times := next(iter(repeats.values()))) > 1:
        options["n_replicates"] = times
    if design_type == "ccd":
        axial = max(abs(v) for row in runs for v in row)
        options["alpha"] = "face_centered" if abs(axial - 1) < CODED_TOLERANCE else round(axial, 4)
    if design_type == "taguchi":
        levels = taguchi_options(record)
        if isinstance(levels, str):
            return levels
        options |= levels
    values = record["response_values"].get(response) or [None] * len(runs)
    return {
        "name": record["id"],
        "title": record["title"][:TITLE_MOST],
        "design": {
            "design_type": design_type,
            "factors": [page_factor(f) for f in factors],
            "response": response,
            **options,
        },
        "runs": [[*row, value] for row, value in zip(runs, values, strict=True)],
    }


def published_campaign(record: dict[str, Any]) -> tuple[dict[str, Any] | None, str, bool]:
    """Return the campaign that opens a published design, the reason there is none, and whether the page tried.

    The page makes the design from what the source prints, and it opens only when the page's design has
    exactly the source's runs, in the source's order and with its results: so a campaign also says that
    the library regenerates the published design, to within the source's rounding. A fraction is tried at
    each resolution the page makes.
    """
    if record["design_type"] not in load_bootstrap().DESIGNS:
        return None, "the page does not make this kind of design", False
    example = _example_of(record)
    if isinstance(example, str):
        return None, example, False
    k = len(record["factors"])
    resolutions = list(range(3, min(5, k) + 1)) if record["design_type"] == "fractional_factorial" else [None]
    reasons = []
    for resolution in resolutions:
        asked = (
            example if resolution is None else {**example, "design": {**example["design"], "resolution": resolution}}
        )
        try:
            campaign = campaign_of(asked, f"the catalog's design {record['id']!r}")
        except SystemExit as stop:
            reasons.append(str(stop).split(": ", 1)[-1])
            continue
        campaign["sources"] = ["entered" if v is not None else None for v in campaign["responses"]]
        return campaign, "", True
    return None, f"the page's design differs from the source's: {reasons[0]}", True


def _response_of(record: dict[str, Any]) -> str:
    """Name the response a design opens with: the source's first one that can name a column, else ``y``."""
    names = [f["name"] for f in record["factors"]]
    return next((r["name"] for r in record["responses"] if r["name"].isidentifier() and r["name"] not in names), "y")


def _given_setting(value: float, factor: dict[str, Any]) -> float | str | None:
    """Return a coded setting in the factor's own units, or its label; None for a setting between two labels.

    A categorical factor's levels lie evenly from -1 at its first to +1 at its last, as the source codes them, to
    the six decimals the catalog keeps.
    """
    if factor.get("type") == "categorical":
        levels = _page_levels(factor) or []
        place = (value + 1) / 2 * (len(levels) - 1)
        on = abs(place - round(place)) < CODED_TOLERANCE and 0 <= round(place) < len(levels)
        return levels[round(place)] if on else None
    middle, half = (factor["low"] + factor["high"]) / 2, (factor["high"] - factor["low"]) / 2
    return float(f"{middle + value * half:.10g}")


def _block_numbers(blocks: list[Any]) -> list[int]:
    """Give each run's block a number 1, 2, ... as the page takes them: by the source's numbers, else as printed.

    A source numbers its blocks, or names them (days, lots, operators): named blocks keep the order in which
    the run table first prints them.
    """
    from process_improve.experiments.design_catalog import _level_label  # noqa: PLC0415 - the docs environment has it

    labels = [_level_label(b) for b in blocks]
    order = list(dict.fromkeys(labels))
    numbers = {label: load_bootstrap()._number(label) for label in order}
    if all(n is not None for n in numbers.values()):
        order.sort(key=lambda label: numbers[label])
    place = {label: i for i, label in enumerate(order, 1)}
    return [place[label] for label in labels]


def _not_given(record: dict[str, Any]) -> str:
    """Say why the page cannot take a design's runs as given, or return "" when it can: the first reason that holds."""
    bootstrap = load_bootstrap()
    info, most = bootstrap.GIVEN_INFO, bootstrap.MAX_GIVEN_RUNS
    runs, factors, blocks = record["coded_runs"], record["factors"], record.get("blocks") or []
    low, high = info["min_factors"], info["max_factors"]
    complete = bool(runs) and all(v is not None for row in runs for v in row)
    reasons = (
        (lambda: record["split_plot"], "the page opens designs without whole plots"),
        (
            lambda: record["blocked"] and (not blocks or None in blocks),
            "the source runs the design in blocks, but does not print each run's block",
        ),
        (
            lambda: len(set(_block_numbers(blocks))) > bootstrap.MAX_GIVEN_BLOCKS,
            f"the page takes runs in at most {bootstrap.MAX_GIVEN_BLOCKS} blocks",
        ),
        (
            lambda: len(blocks) > 1 and len(set(_block_numbers(blocks))) == len(blocks),
            "each run is in a block of its own, so the blocks leave nothing to compare",
        ),
        (lambda: not complete, "the source prints no complete table of runs"),
        (lambda: record["n_mixture"], "the page makes mixtures of its own, but takes no mixture's runs as given"),
        (lambda: not low <= len(factors) <= high, f"the page opens designs of {low} to {high} factors"),
        (lambda: len(runs) > most, f"the page holds at most {most} runs"),
        (lambda: any(map(_blank_label, factors)), "a spreadsheet reads a label such as None as a blank cell"),
    )
    return next((why for holds, why in reasons if holds()), "") or _factors_refused(factors)


def given_campaign(record: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """Return the campaign that opens a design as its runs are given, and the reason there is none.

    It opens a design the page does not make, or makes differently, with the runs as the source prints
    them, or as the library made them: in their order, in the factors' own units, with the source's
    results, and in the source's blocks. The page takes runs without whole plots, of 1 to 12 numeric or
    categorical factors.
    """
    if why := _not_given(record):
        return None, why
    runs, factors = record["coded_runs"], record["factors"]
    rows = [[_given_setting(v, f) for v, f in zip(row, factors, strict=True)] for row in runs]
    if any(v is None for row in rows for v in row):
        return None, "a run sets a categorical factor between two of its levels"
    response = _response_of(record)
    asked = {
        "design_type": "given",
        "factors": [page_factor(f) for f in factors],
        "response": response,
        "runs": rows,
        **({"blocks": _block_numbers(record["blocks"])} if record.get("blocks") else {}),
    }
    design = caller(f"the catalog's design {record['id']!r}")("api_make_design", asked)
    design.pop("xlsx_base64")
    values = record["response_values"].get(response) or [None] * len(rows)
    return _campaign(record["title"], design, values), ""


# --------------------------------------------------------------------------- the numbers a source prints for a study
#: The number of the page's fit a check reads, by the kind of check: a key of the fit's summary.
FIT_NUMBERS = {
    "r_squared": "r_squared",
    "r_squared_adj": "r_squared_adj",
    "residual_df": "df_residual",
    "df_model": "df_model",
    "ms_residual": "mse_residual",
    "ss_residual": "ss_residual",
    "ss_regression": "ss_model",
    "ss_total": "ss_total",
    "f_regression": "f_statistic",
    "f_regression_p": "f_p_value",
}
#: The column of the page's analysis of variance a check reads, by the kind of check.
ANOVA_NUMBERS = {
    "anova_sum_sq": "sum_sq",
    "anova_mean_sq": "mean_sq",
    "anova_f": "F",
    "anova_p": "p_value",
    "anova_df": "df",
}
#: The kinds of check a row of the page's coefficients answers.
COEFFICIENT_NUMBERS = {"coefficient", "effect", "std_error", "t_value", "p_value", "ci_low", "ci_high"}
#: The kinds of check the page's comparison of one factor's levels answers.
COMPARISON_NUMBERS = {"tukey", "least_squares_means"}
#: Keys of a check that make its number one the page does not show: a transformed or summarised response, a
#: number worked from settings or read back on another scale, a one-sided test, a categorical factor coded otherwise.
NOT_SHOWN = (
    "summary",
    "transform",
    "settings",
    "squared",
    "complement",
    "exp",
    "log10",
    "back_transform",
    "at",
    "alternative",
    "scale",
    "categorical",
)


def _variable(name: str) -> str:
    """Return the variable a patsy factor reads: ``C(diet, Sum)`` reads diet."""
    found = re.match(r"C\(\s*(\w+)", name)
    return found.group(1) if found else name


def page_terms(formula: str, spec: dict[str, Any], block: tuple[str, ...] = ()) -> list[list[str]] | None:
    """Return a source's model, as patsy reads it, as the page's chosen terms; None for one the page cannot fit.

    The page cannot fit a model without an intercept, a numeric factor taken as levels, a transformed factor,
    a square of a categorical factor, or an interaction of four factors or more. A study in blocks is fitted
    with its blocks, so the model must name the block's columns as terms of their own, as the page's blocks
    are; a model without them, or with a block column in an interaction, is not the page's.
    """
    from patsy import ModelDesc  # noqa: PLC0415 - the docs environment has it

    bootstrap = load_bootstrap()
    factors = {f["name"]: f for f in spec["factors"]}
    categorical = {name for name, f in factors.items() if bootstrap._categorical(f)}
    desc = ModelDesc.from_formula(formula.split("~", 1)[-1])
    if not any(not t.factors for t in desc.rhs_termlist) or all(not t.factors for t in desc.rhs_termlist):
        return None  # no intercept, or the intercept alone
    terms, named_block = [], False
    for term in (t for t in desc.rhs_termlist if t.factors):
        variables = [_variable(f.name()) for f in term.factors]
        if set(variables) & set(block):
            if len(variables) > 1:
                return None  # the page fits no interaction with its blocks
            named_block = True
            continue
        parts = []
        for factor in term.factors:
            name = factor.name()
            square = re.fullmatch(r"I\(\s*(\w+)\s*\*\*\s*2\s*\)", name)
            variable = square.group(1) if square else _variable(name)
            numeric_as_levels = name.startswith("C(") and variable not in categorical
            transformed = not square and not name.startswith("C(") and name != variable
            if variable not in factors or numeric_as_levels or transformed or (square and variable in categorical):
                return None
            parts += [variable, variable] if square else [variable]
        squared = len(parts) == 2 and parts[0] == parts[1]  # a factor named twice is its square
        if not squared and (len(set(parts)) != len(parts) or len(parts) > 3):  # three factors at most
            return None
        terms.append(parts)
    return terms if terms and (not block or named_block) else None


def term_part(text: str) -> str:
    """Write one part of a term as a key: ``A²`` and ``I(A ** 2)`` as ``A^2``, ``C(diet, Sum)`` as ``diet``."""
    square = re.fullmatch(r"I\(\s*(\w+)\s*\*\*\s*2\s*\)", text) or re.fullmatch(r"(\w+)²", text)
    return f"{square.group(1)}^2" if square else _variable(text)


def _source_key(source: str) -> list[str]:
    """Return the parts of an analysis-of-variance source as keys, sorted: ``B:A`` as ``[A, B]``."""
    return sorted({term_part(p.strip()) for p in re.split(r"[:·]", source)})


def _level_parts(term: str) -> list[tuple[str, str | None]]:
    """Split a source's term into (factor, level) parts: ``diet[A]:T`` is [(diet, A), (T, None)]."""
    out = []
    for part in re.split(r"[:·]", term):
        level = re.fullmatch(r"(?:C\()?(\w+)(?:,[^\[]*\))?\[(?:S\.)?(.+)\]", part.strip())
        out.append((level.group(1), level.group(2)) if level else (term_part(part.strip()), None))
    return out


def _choices(term: str, spec: dict[str, Any]) -> list[list[tuple[str, float]]] | None:
    """Return, for each part of a source's term, the page's label parts that make it and their signs.

    A level of a categorical factor is its effect, its distance from the average level: the page's coefficient of
    that level, a two-level factor's at -1 or +1, and the last level's minus the sum of the others'.
    """
    bootstrap = load_bootstrap()
    factors = {f["name"]: f for f in spec["factors"]}
    choices = []
    for name, level in _level_parts(term):
        factor = factors.get(name)
        if level is None:
            choices.append([(name, 1.0)])
            continue
        levels = bootstrap._levels_of(factor) if factor and bootstrap._categorical(factor) else []
        at = bootstrap._level_index(level, levels) if levels else None
        if at is None:
            return None
        if len(levels) == 2:  # coded -1 and +1
            choices.append([(name, 1.0 if at else -1.0)])
        elif at < len(levels) - 1:
            choices.append([(f"{name}[{levels[at]}]", 1.0)])
        else:
            choices.append([(f"{name}[{other}]", -1.0) for other in levels[:-1]])
    return choices


def _asked(check: dict[str, Any]) -> dict[str, str] | None:
    """Read what a comparison check asks: its method, the two levels or the level, and the number."""
    if check["kind"] == "tukey":
        return {
            "method": "tukey",
            "head": "pairs",
            "a": str(check["level"]),
            "b": str(check["minus"]),
            "key": check["key"],
        }
    head, _, rest = check["key"].partition(".")
    item, _, key = rest.rpartition(".")
    a, _, b = item.partition(" - ")
    method = check.get("comparisons", "tukey")
    return {"method": method, "head": head, "a": a, "b": b, "key": key} if head in ("pairs", "by_level") else None


def _shown(check: dict[str, Any], entry: dict[str, Any], spec: dict[str, Any]) -> bool:
    """Say whether a check's number is one the page shows for this study."""
    kind = check.get("kind")
    known = kind in ANOVA_NUMBERS or kind in FIT_NUMBERS or kind in COEFFICIENT_NUMBERS or kind in COMPARISON_NUMBERS
    if not known or any(check.get(key) for key in NOT_SHOWN) or check.get("confidence", 0.95) != 0.95:
        return False
    own_units = any(f.get("coding") in ("none", "pseudocomponent") for f in entry["factors"]) or entry["runs"].get(
        "exact"
    )
    if kind in COEFFICIENT_NUMBERS and own_units:
        return False  # the source fits settings in their own units; the page, coded
    if kind in COMPARISON_NUMBERS:  # the page compares the levels of a study of one factor, on no other model
        alone = spec["factors"][0]["name"] if len(spec["factors"]) == 1 else None
        model = check["factor"] if kind == "tukey" and not check.get("block") else check.get("model")
        return alone is not None and model in (alone, f"C({alone})")
    return bool(check.get("model"))


def _reading(
    check: dict[str, Any], spec: dict[str, Any], block: tuple[str, ...], flipped: frozenset[str]
) -> dict[str, Any] | None:
    """Return where on the page's analysis a check's number is read, or None where the page has no such number.

    A factor the source codes high to low (``flipped``) is coded low to high on the page, so a coefficient of a
    term with an odd number of such factors has its sign turned, and each end of its interval is the other's.
    """
    kind = check["kind"]
    if kind in ANOVA_NUMBERS:
        return _anova_reading(check, block)
    if kind in FIT_NUMBERS:
        return {"table": "summary", "field": FIT_NUMBERS[kind]}
    if kind in COEFFICIENT_NUMBERS:
        return _coefficient_reading(check, spec, flipped)
    asked = _asked(check)
    return None if asked is None else {"table": "comparison", **asked}


def _anova_reading(check: dict[str, Any], block: tuple[str, ...]) -> dict[str, Any] | None:
    """Return the row and column of the page's analysis of variance a check reads; the total's df is the runs less 1."""
    if check["term"] == "Total" and check["kind"] == "anova_df":
        return {"table": "total_df"}
    wanted = _source_key(check["term"])
    if set(wanted) <= set(block):  # the page's row of its blocks, when they are one column
        if len(block) != 1:
            return None
        wanted = ["Block"]
    return {"table": "anova", "source": wanted, "field": ANOVA_NUMBERS[check["kind"]]}


def _coefficient_reading(check: dict[str, Any], spec: dict[str, Any], flipped: frozenset[str]) -> dict[str, Any] | None:
    """Return the rows of the page's coefficients a check reads, each with its sign, and the column; an effect twice."""
    sign = (-1) ** sum(name in flipped for name, _ in _level_parts(check["term"]))
    field = "coefficient" if check["kind"] == "effect" else check["kind"]
    if sign < 0 and field in ("ci_low", "ci_high"):
        field = "ci_high" if field == "ci_low" else "ci_low"
    choices = _choices(check["term"], spec)
    if choices is None:
        return None
    rows = [[[p for p, _ in combo], math.prod(s for _, s in combo)] for combo in itertools.product(*choices)]
    if len(rows) > 1 and field != "coefficient":
        return None  # a sum of coefficients has no row of its own
    unsigned = field in ("std_error", "p_value")
    times = 2 if check["kind"] == "effect" else 1
    return {"table": "coefficients", "rows": rows, "field": field, "sign": 1 if unsigned else sign, "times": times}


def _agreement(check: dict[str, Any], printed: object) -> dict[str, Any] | None:
    """Return the printed number, its bound and the rounding it allows, as the catalogue's checks hold a number to it.

    The rule is :func:`~process_improve.experiments.reference_catalogue._agrees`: half a unit in the last digit
    printed, or in ``decimals`` places, times ``of_rounded``. A number printed as an expression, or held to the
    rounding of the data, is not one the page compares.
    """
    from process_improve.experiments import reference_catalogue as rc  # noqa: PLC0415 - the docs environment has it

    if check.get("expression") or check.get("data_rounding") or isinstance(printed, list | dict):
        return None
    try:
        bound, digits = rc._number(str(printed))
    except ValueError:
        return None
    decimals = check.get("decimals")
    half = rc._tolerance(digits) if decimals is None else 0.5 * 10.0**-decimals
    side = {"<": "<", "<=": "<", "≤": "<", ">": ">", ">=": ">", "≥": ">"}.get(bound or "")
    return {"target": float(digits), "bound": side, "allowed": check.get("of_rounded", 1) * half}


def printed_numbers(record: dict[str, Any], entry: dict[str, Any], spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the numbers a source prints for a study that the page shows too, each with where the page reads it.

    Each holds what the number is and where it is printed (``what``, ``term``, ``printed``, ``where``), the model
    it is printed for as the page's terms (``model``), where on the page's analysis of that model the page's own
    number is (``read``), the rule it agrees by (``agree``), and the reason, where the catalogue knows the printed
    number not to reproduce (``reason``). A number of another response, or of some of the runs only, is not one
    the page shows; of the checks of one printed number, the first the page shows is kept.
    """
    from process_improve.experiments import design_catalog  # noqa: PLC0415 - the docs environment has it
    from process_improve.experiments import reference_catalogue as rc  # noqa: PLC0415

    response = _response_of(record)
    first = next((r["name"] for r in entry.get("responses") or [] if r["name"] in entry["runs"]["columns"]), None)
    block = tuple(design_catalog._block_columns(entry) or ()) if record.get("blocks") else ()
    flipped = frozenset(
        f["name"] for f in record["factors"] if _page_factor(f) and f.get("type") == "numeric" and f["low"] > f["high"]
    )
    out = []
    for expected in entry.get("expected", []):
        for check in rc._checks(expected):
            if not _shown(check, entry, spec) or check.get("runs") or (check.get("response") or first) != response:
                continue
            model = check["factor"] if check["kind"] == "tukey" else check["model"]
            terms = page_terms(model, spec, block)
            printed = check.get("printed", expected["value"])
            read, agree = (_reading(check, spec, block, flipped), _agreement(check, printed)) if terms else (None, None)
            if read is None or agree is None:
                continue
            where = expected.get("where") or {}
            out.append(
                {
                    "what": expected["what"],
                    "term": expected.get("term"),
                    "printed": str(printed),
                    "where": f"p. {where['page']}" if where.get("page") else "",
                    "model": terms,
                    "read": read,
                    "agree": agree,
                    "reason": check.get("disagreement") or None,
                }
            )
            break
    return out


def study_source(record: dict[str, Any], source: dict[str, Any], printed: int = 0) -> dict[str, Any]:
    """Name where a published study comes from, as a study the catalog opens keeps it.

    ``cite`` is the source's citation, ``where`` the place it prints the runs and ``url`` that place's address,
    else the source's; ``catalog`` is the study's id in the catalog, and ``printed`` how many of the numbers its
    source prints the page shows too, which ``catalog/printed/<id>.json`` holds.
    """
    if not re.fullmatch(r"[\w.-]+", record["id"]):
        raise SystemExit(f"The catalog's design {record['id']!r} has an id that cannot name a file.")
    return {
        "cite": source["cite"] or source["title"],
        "where": record["where"] or "",
        "url": record["where_url"] or source["url"] or "",
        "catalog": record["id"],
        "printed": printed,
    }


def write_catalog(out_dir: Path) -> None:
    """Write the catalog: ``catalog/index.json``, a detail file for each source and the rest, the errata and a feed.

    The index holds what the page searches, filters, maps and draws for every design at once, with the
    families, sources and applications it names; a detail file holds what it shows on opening one: its
    factors, coded runs, results, notes, where its source prints it and that place's address, the
    measures it is compared on and, where the page makes the design, the campaign that opens it, which names
    its source (see :func:`study_source`). ``printed/<id>.json`` holds the numbers a published study's source
    prints that the page shows too (see :func:`printed_numbers`), which Learn reads beside its own.
    ``errata.json`` lists every printed number the library does not reproduce, and ``feed.xml`` is an
    Atom feed of the sources.
    """
    from process_improve.experiments.compare import MEASURES  # noqa: PLC0415 - the docs environment has it
    from process_improve.experiments.design_catalog import (  # noqa: PLC0415
        INDEX_FIELDS,
        catalog_comparisons,
        catalog_errata,
        catalog_families,
        catalog_feed,
        catalog_records,
        catalog_sources,
        catalog_topics,
        shown_where,
    )
    from process_improve.experiments.reference_catalogue import (  # noqa: PLC0415
        reference_designs,
        reference_experiments,
    )

    records = catalog_records()
    comparisons = catalog_comparisons(records)
    sources = catalog_sources()
    entries = {e["name"]: e for e in [*reference_experiments(), *reference_designs()]}
    designs, details, printed = [], {}, {}
    for record in records:
        if record["collection"] == "generated":
            (campaign, reason), regenerable = generated_campaign(record), None
        else:
            campaign, reason, tried = published_campaign(record)
            regenerable = campaign is not None if tried else None
        # A design the page does not make, or makes otherwise, opens with its runs as given; the reason the page
        # did not make it is kept, and the reason it does not open is the given runs' when they cannot be taken.
        opens_as, unmade = ("made", "") if campaign else (None, reason)
        if campaign is None:
            campaign, why = given_campaign(record)
            opens_as, reason = ("given", "") if campaign else (None, why)
        if campaign and record["collection"] == "published":
            entry, results = entries.get(record["id"]), any(v is not None for v in campaign["responses"])
            numbers = printed_numbers(record, entry, campaign["design"]["spec"]) if entry and results else []
            campaign["source"] = study_source(record, sources[record["source"]], len(numbers))
            if numbers:
                printed[record["id"]] = numbers
        shard = record["source"] or "generated"
        designs.append(
            {
                **{key: record[key] for key in INDEX_FIELDS},
                "regenerable": regenerable,
                "open": bool(campaign),
                "detail": shard,
                "compared": record["id"] in comparisons,
            }
        )
        details.setdefault(shard, {})[record["id"]] = {
            "design_text": record["design_text"],
            "factors": record["factors"],
            "responses": record["responses"],
            "coded_runs": record["coded_runs"],
            "blocks": record.get("blocks"),
            "response_values": record["response_values"],
            "details": record["details"],
            "notes": record["notes"],
            "where": shown_where(record["where_full"]),
            "where_url": record["where_url"],
            "spec": record.get("spec"),
            "listed": record.get("listed"),
            "run_loss_model": record["run_loss_model"],
            "leverage": record["leverage"],
            "campaign": campaign,
            "opens_as": opens_as,
            "closed": reason,
            "unmade": unmade,
            "compare": comparisons.get(record["id"]),
        }
    index = {
        "version": CATALOG_VERSION,
        "library": version("process-improve"),
        "fields": [*INDEX_FIELDS, *CATALOG_FIELDS],
        "families": catalog_families(),
        "sources": catalog_sources(),
        "topics": catalog_topics(),
        "measures": [dataclasses.asdict(measure) for measure in MEASURES],
        "designs": designs,
    }
    folder = out_dir / "catalog"
    (folder / "detail").mkdir(parents=True)
    (folder / "printed").mkdir()

    def write(path: Path, data: dict[str, Any]) -> None:
        # The page reads these with JSON.parse, which refuses NaN: a value that is no number stops the build here.
        path.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n")

    write(folder / "index.json", index)
    for shard, held in details.items():
        write(folder / "detail" / f"{shard}.json", {"version": CATALOG_VERSION, "designs": held})
    for name, numbers in printed.items():
        write(folder / "printed" / f"{name}.json", {"version": CATALOG_VERSION, "printed": numbers})
    write(folder / "errata.json", {"version": CATALOG_VERSION, "errata": catalog_errata()})
    (folder / "feed.xml").write_text(catalog_feed(records), encoding="utf-8")


def assemble(out_dir: Path, dist: Path = ROOT / "dist" / "web") -> str:
    """Copy the app, a wheel freshly built in ``dist``, the book's examples and the worked example into ``out_dir``.

    Return the wheel's name. ``demo.json`` is written last, so its presence says the app is complete.
    """
    shutil.rmtree(out_dir, ignore_errors=True)
    (out_dir / "wheels").mkdir(parents=True)
    for name in APP_FILES:
        (out_dir / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(WEB_SRC / name, out_dir / name)
    wheel = build_wheel(dist)
    published = Path(hashlib.sha256(wheel.read_bytes()).hexdigest()[:16]) / wheel.name
    (out_dir / "wheels" / published.parent).mkdir()
    shutil.copy2(wheel, out_dir / "wheels" / published)
    from plotly.offline import get_plotlyjs_version  # noqa: PLC0415 - the docs environment has plotly

    manifest = {
        "wheel": published.as_posix(),
        "commit": commit(),
        "plotly": version("plotly"),
        "plotly_js": get_plotlyjs_version(),
    }
    (out_dir / "wheels" / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_book(out_dir)
    write_catalog(out_dir)
    (out_dir / "demo.json").write_text(json.dumps(worked_example(), separators=(",", ":")) + "\n")
    return wheel.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--out", type=Path, default=ROOT / "dist" / "web-app", help="output directory")
    parser.add_argument("--serve", action="store_true", help="serve the result over HTTP afterwards")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    wheel = assemble(args.out)
    print(f"Assembled {args.out} with {wheel}")
    if args.serve:
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(args.out))
        print(f"Serving on http://localhost:{args.port}/")
        http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler).serve_forever()


if __name__ == "__main__":
    main()
