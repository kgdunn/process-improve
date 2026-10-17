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
import os
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
    "campaign.js",
    "store.js",
    "glossary.js",
    "terms.js",
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
    return {
        "version": version("process-improve"),
        "inputs": inputs,
        "design": design,
        "simulation": simulation,
        "analysis": analysis,
        "best": best,
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
#: The campaign format the page reads (``CAMPAIGN_VERSION`` in ``web/campaign.js``).
CAMPAIGN_VERSION = 1
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
    """Return a run's setting of a factor in coded units: the low level -1 and the high level +1, a label too."""
    if isinstance(factor["low"], str):
        return -1.0 if setting == factor["low"] else 1.0
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
        "v": CAMPAIGN_VERSION,
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
        "v": CAMPAIGN_VERSION,
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
    if spec["design_type"] not in load_bootstrap().DESIGNS:
        return None, "the page does not make this kind of design"
    factors = [{"name": f["name"], "low": f["low"], "high": f["high"], "units": ""} for f in record["factors"]]
    asked = {"design_type": spec["design_type"], "factors": factors, "response": "y", **spec["options"]}
    design = caller(f"the catalog's design {record['id']!r}")("api_make_design", asked)
    design.pop("xlsx_base64")
    drawn = Counter(tuple(round(v, 2) for v in row) for row in record["coded_runs"])
    if _coded_runs(design) != drawn:
        raise SystemExit(f"the catalog's design {record['id']!r}: the page makes other runs than the library drew")
    return _campaign(record["title"], design, [None] * design["n_runs"]), ""


def _page_factor(factor: dict[str, Any]) -> bool:
    """Say whether the page takes a factor as the source gives it: a name, and a low and high value or label."""
    low, high = factor.get("low"), factor.get("high")
    numbers = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (low, high))
    labels = all(load_bootstrap()._label(v) is not None for v in (low, high))  # a label that reads as no number
    in_order = low < high if numbers else labels and low != high and not _blank_label(factor)  # type: ignore[operator]
    return str(factor["name"]).isidentifier() and factor.get("type") in {"numeric", "categorical"} and in_order


def _blank_label(factor: dict[str, Any]) -> bool:
    """Say whether a factor has a label a spreadsheet reads back as a blank cell, such as None: the page refuses it."""
    bootstrap = load_bootstrap()
    return any(bootstrap._label(v) in bootstrap.BLANK_LABELS for v in (factor.get("low"), factor.get("high")))


def _example_of(record: dict[str, Any]) -> dict[str, Any] | str:
    """Write a published record as the book's examples are written, for :func:`campaign_of`; or say why it cannot be.

    The design's specification is what the page would make for it: its type, its factors, a center run for
    each one the source prints and, where the source repeats every other run, that many replicates. A
    central composite design's axial distance is read off the runs.
    """
    designs = load_bootstrap().DESIGNS
    design_type, factors, runs = record["design_type"], record["factors"], record["coded_runs"]
    info = designs[design_type]
    if record["blocked"] or record["split_plot"]:
        return "the page opens designs without blocks or whole plots"
    if not runs or any(v is None for row in runs for v in row):
        return "the source prints no complete table of runs"
    if not info["min_factors"] <= len(factors) <= info["max_factors"]:
        return f"the page makes this design for {info['min_factors']} to {info['max_factors']} factors"
    if not all(_page_factor(f) for f in factors):
        return "a factor's name, type or range is not one the page accepts"
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
    values = record["response_values"].get(response) or [None] * len(runs)
    return {
        "name": record["id"],
        "title": record["title"][:TITLE_MOST],
        "design": {
            "design_type": design_type,
            "factors": [
                {"name": f["name"], "low": f["low"], "high": f["high"], "units": str(f.get("units") or "")[:UNITS_MOST]}
                for f in factors
            ],
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
    """Return a coded setting in the factor's own units, or its label; None for a label the factor does not have."""
    if isinstance(factor["low"], str):
        return {-1.0: factor["low"], 1.0: factor["high"]}.get(round(value, 6))
    middle, half = (factor["low"] + factor["high"]) / 2, (factor["high"] - factor["low"]) / 2
    return float(f"{middle + value * half:.10g}")


def _not_given(record: dict[str, Any]) -> str:
    """Say why the page cannot take a design's runs as given, or return "" when it can: the first reason that holds."""
    info, most = load_bootstrap().GIVEN_INFO, load_bootstrap().MAX_GIVEN_RUNS
    runs, factors = record["coded_runs"], record["factors"]
    low, high = info["min_factors"], info["max_factors"]
    complete = bool(runs) and all(v is not None for row in runs for v in row)
    reasons = (
        (lambda: record["blocked"] or record["split_plot"], "the page opens designs without blocks or whole plots"),
        (lambda: not complete, "the source prints no complete table of runs"),
        (lambda: record["n_mixture"], "the page has no model for a mixture's proportions"),
        (lambda: not low <= len(factors) <= high, f"the page opens designs of {low} to {high} factors"),
        (lambda: len(runs) > most, f"the page holds at most {most} runs"),
        (lambda: any(map(_blank_label, factors)), "a spreadsheet reads a label such as None as a blank cell"),
        (lambda: not all(map(_page_factor, factors)), "a factor's name, type or range is not one the page accepts"),
    )
    return next((why for holds, why in reasons if holds()), "")


def given_campaign(record: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """Return the campaign that opens a design as its runs are given, and the reason there is none.

    It opens a design the page does not make, or makes differently, with the runs as the source prints
    them, or as the library made them: in their order, in the factors' own units, with the source's
    results. The page takes runs without blocks or whole plots, of 2 to 12 numeric or two-level factors.
    """
    if why := _not_given(record):
        return None, why
    runs, factors = record["coded_runs"], record["factors"]
    rows = [[_given_setting(v, f) for v, f in zip(row, factors, strict=True)] for row in runs]
    if any(v is None for row in rows for v in row):
        return None, "a categorical factor has more than the two levels the page takes"
    response = _response_of(record)
    asked = {
        "design_type": "given",
        "factors": [
            {"name": f["name"], "low": f["low"], "high": f["high"], "units": str(f.get("units") or "")[:UNITS_MOST]}
            for f in factors
        ],
        "response": response,
        "runs": rows,
    }
    design = caller(f"the catalog's design {record['id']!r}")("api_make_design", asked)
    design.pop("xlsx_base64")
    values = record["response_values"].get(response) or [None] * len(rows)
    return _campaign(record["title"], design, values), ""


def write_catalog(out_dir: Path) -> None:
    """Write the catalog: ``catalog/index.json``, a detail file for each source and the rest, the errata and a feed.

    The index holds what the page searches, filters, maps and draws for every design at once, with the
    families, sources and applications it names; a detail file holds what it shows on opening one: its
    factors, coded runs, results, notes, where its source prints it and that place's address, the
    measures it is compared on and, where the page makes the design, the campaign that opens it.
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

    records = catalog_records()
    comparisons = catalog_comparisons(records)
    designs, details = [], {}
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

    def write(path: Path, data: dict[str, Any]) -> None:
        # The page reads these with JSON.parse, which refuses NaN: a value that is no number stops the build here.
        path.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n")

    write(folder / "index.json", index)
    for shard, held in details.items():
        write(folder / "detail" / f"{shard}.json", {"version": CATALOG_VERSION, "designs": held})
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
