"""Assemble the no-install browser app into a directory that can be served.

Builds the wheel from this checkout, copies ``web/`` next to it and writes the
``wheels/manifest.json`` that ``web/worker.js`` reads, so the published page
always runs the code it was deployed from rather than the latest PyPI release.
The docs workflow runs this into ``docs/_build/html/app``.

    uv run python scripts/build_web_app.py --serve

``--serve`` starts a plain HTTP server: Pyodide fetches its files, so opening
``index.html`` from disk (``file://``) does not work.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB_SRC = ROOT / "web"
APP_FILES = ("index.html", "app.js", "worker.js", "bootstrap.py", "style.css")


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


def assemble(out_dir: Path) -> str:
    """Copy the app and a freshly built wheel into ``out_dir``; return the wheel name."""
    shutil.rmtree(out_dir, ignore_errors=True)
    (out_dir / "wheels").mkdir(parents=True)
    for name in APP_FILES:
        shutil.copy2(WEB_SRC / name, out_dir / name)
    wheel = build_wheel(ROOT / "dist" / "web")
    shutil.copy2(wheel, out_dir / "wheels" / wheel.name)
    (out_dir / "wheels" / "manifest.json").write_text(json.dumps({"wheel": wheel.name}, indent=2) + "\n")
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
