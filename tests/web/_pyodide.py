"""The Pyodide distribution the page loads, kept on disk for Node.js: its standard library, lock file and wheels.

The page loads a Pyodide release from one folder of the release on the CDN, which ``web/runtime.json`` names as
``distribution``: ``full``, or ``pyc``, the same packages with their Python compiled to bytecode already, which
start in half the time. Node.js runs the release's npm package, whose interpreter is the same file in every folder.
What differs is kept beside it, in a directory of the folder's name: the standard library, the lock file, and each
wheel the page loads, checked against the lock file's digest and fetched again when it does not match.

    uv run python -m tests.web._pyodide     # fetch it now, as CI does before the tests
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "web" / "runtime.json"

#: The npm package of Pyodide for Node.js, unless ``PI_PYODIDE`` names another directory.
PACKAGE = Path.home() / ".cache" / "process-improve" / "node" / "node_modules" / "pyodide"

CDN = "https://cdn.jsdelivr.net/pyodide/v{pyodide}/{distribution}/"

#: Downloads of one file before giving up; a download cut short is the failure seen in CI.
ATTEMPTS = 4


def package() -> Path:
    """Return the directory of Pyodide's npm package: ``PI_PYODIDE``, or the cache the tests install it into."""
    return Path(os.environ.get("PI_PYODIDE") or PACKAGE)


def _normal(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _kept(target: Path, url: str, sound: Callable[[Path], bool]) -> None:
    """Make ``target`` hold what ``url`` serves: fetched unless it is sound already, and again if it arrives unsound."""
    for attempt in range(ATTEMPTS):
        if target.is_file() and sound(target):
            return
        if attempt:
            time.sleep(2**attempt)
        partial = target.with_name(f"{target.name}.{os.getpid()}.partial")
        try:
            urllib.request.urlretrieve(url, partial)
            partial.replace(target)
        except OSError as error:
            print(f"{target.name}: {error}", file=sys.stderr)  # noqa: T201 - the reason, should every attempt fail
            partial.unlink(missing_ok=True)
    if not (target.is_file() and sound(target)):
        message = f"{url} did not arrive whole in {ATTEMPTS} downloads"
        raise RuntimeError(message)


def _info(path: Path) -> dict | None:
    """Return what a lock file says of its interpreter (its Python, platform and ABI), or None if it is not one."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))["info"]
    except (OSError, ValueError, KeyError):
        return None


def _is_zip(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            return archive.testzip() is None
    except (zipfile.BadZipFile, OSError):
        return False


def needed(lock: dict, packages: list[str]) -> list[dict]:
    """Return the lock file's entry for each of ``packages`` and for everything they depend on."""
    entries = {_normal(name): entry for name, entry in lock["packages"].items()}
    todo, seen = [_normal(name) for name in packages], {}
    while todo:
        name = todo.pop()
        if name not in seen:
            seen[name] = entries[name]
            todo += [_normal(d) for d in entries[name]["depends"]]
    return list(seen.values())


def distribution(runtime: dict, npm: Path | None = None) -> Path:
    """Return the directory that holds the page's Pyodide distribution, fetching what it lacks from the CDN."""
    folder = (npm or package()) / runtime["distribution"]
    folder.mkdir(parents=True, exist_ok=True)
    cdn = CDN.format(**runtime)
    # The folder's lock file is of the interpreter the npm package runs: the same Python, platform and ABI.
    interpreter = _info(folder.parent / "pyodide-lock.json")
    _kept(folder / "pyodide-lock.json", f"{cdn}pyodide-lock.json", lambda path: _info(path) == interpreter)
    _kept(folder / "python_stdlib.zip", f"{cdn}python_stdlib.zip", _is_zip)
    lock = json.loads((folder / "pyodide-lock.json").read_text(encoding="utf-8"))
    for entry in needed(lock, [*runtime["packages"], *runtime["fitting"]]):
        digest = entry["sha256"]
        _kept(
            folder / entry["file_name"],
            cdn + entry["file_name"],
            lambda path, digest=digest: hashlib.sha256(path.read_bytes()).hexdigest() == digest,
        )
    return folder


if __name__ == "__main__":
    print(distribution(json.loads(RUNTIME.read_text(encoding="utf-8"))))  # noqa: T201 - where it is, for the log
