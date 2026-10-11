"""Assemble the no-install browser app into a directory that can be served.

Builds the wheel from this checkout, copies ``web/`` next to it and writes the
``wheels/manifest.json`` that ``web/worker.js`` reads, so the published page
always runs the code it was deployed from rather than the latest PyPI release.
The wheel goes in a folder named by its digest, so each deploy's wheel has a URL
of its own and a browser cannot install one it cached from an earlier deploy; the
manifest, fetched past the cache, names it and the commit the app was built from.
It also runs the page's worked example through ``web/bootstrap.py``, the code the
page runs, and writes the results to ``demo.json``: the page draws them at first
paint while Python loads in the browser. The docs workflow runs this into
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
import json
import os
import shutil
import subprocess
from importlib.metadata import version
from pathlib import Path
from typing import Any

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
    "qr.js",
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


def worked_example() -> dict[str, Any]:
    """Run the page's worked example through ``web/bootstrap.py``, as the page runs it live.

    Every key the page reads at first paint is a reply of the same entry point the page
    calls, so the example drawn before Python loads and the one computed in the browser
    are the same code's output.
    """
    spec = importlib.util.spec_from_file_location("web_bootstrap", WEB_SRC / "bootstrap.py")
    if spec is None or spec.loader is None:
        raise SystemExit("web/bootstrap.py could not be loaded")
    bootstrap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bootstrap)

    def call(fn: str, payload: dict) -> Any:  # noqa: ANN401
        reply = json.loads(getattr(bootstrap, fn)(json.dumps(payload)))
        if not reply["ok"]:
            raise SystemExit(f"{fn} failed on the worked example: {reply['error']}")
        return reply["result"]

    inputs = bootstrap.DEMO
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


def assemble(out_dir: Path, dist: Path = ROOT / "dist" / "web") -> str:
    """Copy the app, a wheel freshly built in ``dist`` and the worked example into ``out_dir``; return the wheel name.

    ``demo.json`` is written last, so its presence says the app is complete.
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
