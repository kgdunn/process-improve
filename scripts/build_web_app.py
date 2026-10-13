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
opens from ``?book=<name>``, a link the book prints. The docs workflow runs this into
``docs/_build/html/app``.

    uv run python scripts/build_web_app.py --serve

``--serve`` starts a plain HTTP server: Pyodide fetches its files, so opening
``index.html`` from disk (``file://``) does not work.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import http.server
import importlib.util
import itertools
import json
import os
import shutil
import subprocess
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
    "app.js",
    "logic.js",
    "charts.js",
    "python.js",
    "campaign.js",
    "store.js",
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

    # Step 4 for the same factors: every design the page offers, compared on the nine measures.
    model = "main_quadratic"
    plan = call("api_compare_plan", {"factors": inputs["design"]["factors"], "model": model})
    made = []
    for item in plan["designs"]:
        generated = call("api_make_design", item["spec"])
        generated.pop("xlsx_base64")
        made.append({**item, "design": generated})
    contenders = [
        {
            "label": m["label"],
            "spec": m["design"]["spec"],
            "columns": m["design"]["columns"],
            "rows": m["design"]["rows"],
        }
        for m in made
    ]
    return {
        "version": version("process-improve"),
        "inputs": inputs,
        "design": design,
        "simulation": simulation,
        "analysis": call("api_analyze_runs", request),
        "plots": call("api_plots", request),
        "comparison": {
            "model": model,
            "made": made,
            "notes": plan["notes"],
            "result": call("api_compare", {"model": model, "designs": contenders}),
        },
    }


#: The book's examples that the page opens by name: ``?book=<name>`` opens ``book/<name>.json``, built from each.
BOOK_EXAMPLES = WEB_SRC / "book-examples.json"
#: Where the book is published; each example names its page under here.
BOOK = "https://learnche.org/pid/"
#: The campaign format the page reads (``CAMPAIGN_VERSION`` in ``web/campaign.js``).
CAMPAIGN_VERSION = 1
#: How near, in coded units, a run of the book must come to a run of the design: the book prints
#: a rotatable design's axial distance, the square root of 2, as 1.41.
CODED_TOLERANCE = 0.01


def book_examples() -> list[dict[str, Any]]:
    """Return the book's examples as ``web/book-examples.json`` lists them."""
    return json.loads(BOOK_EXAMPLES.read_text(encoding="utf-8"))["examples"]


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
    """Make one of the book's examples into the campaign the page opens from ``?book=<name>``.

    The design is made as step 1 makes it, and its runs put in the order of the book's table,
    each with the book's result, so the page's run sheet reads as the book's table does. The
    design may number its blocks otherwise than the book numbers its days: each way of
    pairing them is tried.
    """
    what = f"the book's example {entry['name']!r}"
    design = caller(what)("api_make_design", entry["design"])
    labels = sorted(set(entry.get("blocks", [])))
    attempts = [
        in_book_order(design, entry, dict(zip(labels, paired, strict=True)))
        for paired in itertools.permutations(labels)
    ]
    rows = next((rows for rows in attempts if isinstance(rows, list)), None)
    if rows is None:
        raise SystemExit(f"{what}: {attempts[0]}")
    campaign = {
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
    return {
        "name": entry["name"],
        "title": entry["title"],
        "book": BOOK + entry["page"],
        "note": entry.get("note", ""),
        "campaign": campaign,
    }


def write_book(out_dir: Path) -> None:
    """Write ``book/<name>.json`` for each of the book's examples, and ``book/index.json`` naming them all."""
    examples = [book_example(entry) for entry in book_examples()]
    (out_dir / "book").mkdir()
    for example in examples:
        (out_dir / "book" / f"{example['name']}.json").write_text(json.dumps(example, separators=(",", ":")) + "\n")
    (out_dir / "book" / "index.json").write_text(json.dumps(book_index(examples), indent=1) + "\n")


def book_index(examples: list[dict[str, Any]]) -> dict[str, Any]:
    """Return ``book/index.json``: each example's name, title and page of the book, for the page and the book."""
    return {"examples": [{key: example[key] for key in ("name", "title", "book")} for example in examples]}


def assemble(out_dir: Path, dist: Path = ROOT / "dist" / "web") -> str:
    """Copy the app, a wheel freshly built in ``dist``, the book's examples and the worked example into ``out_dir``.

    Return the wheel's name. ``demo.json`` is written last, so its presence says the app is complete.
    """
    shutil.rmtree(out_dir, ignore_errors=True)
    (out_dir / "wheels").mkdir(parents=True)
    for name in APP_FILES:
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
