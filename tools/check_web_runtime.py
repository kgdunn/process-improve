"""Say whether the browser app's pinned runtime has fallen behind the latest releases.

``web/runtime.json`` pins the Pyodide release the page runs and each package it installs from
PyPI. Dependabot can see neither, so this compares them with the latest Pyodide release (on npm,
where the tests install it) and with the latest release of each package on PyPI. It prints what
is behind and exits 1, or says that nothing is and exits 0. Standard library only.

    python tools/check_web_runtime.py

To move a pin: change ``web/runtime.json``, then run the wasm tier (``pytest tests/web/wasm
--wasm``); a new Pyodide release can change what its own packages need from PyPI.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

RUNTIME = Path(__file__).resolve().parent.parent / "web" / "runtime.json"


def _read_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310 - fixed https URLs
        return json.load(response)


def latest_pyodide() -> str:
    """Return the latest Pyodide release, as npm tags it."""
    return _read_json("https://registry.npmjs.org/pyodide/latest")["version"]


def latest_on_pypi(name: str) -> str:
    """Return the latest release of a package on PyPI."""
    return _read_json(f"https://pypi.org/pypi/{name}/json")["info"]["version"]


def behind(
    runtime: dict, pyodide: Callable[[], str] = latest_pyodide, pypi: Callable[[str], str] = latest_on_pypi
) -> list[str]:
    """List, one line each, what ``runtime`` pins that is not the latest release."""
    lines = []
    if (latest := pyodide()) != runtime["pyodide"]:
        lines.append(f"Pyodide: {runtime['pyodide']} is pinned, and {latest} is the latest release.")
    for requirement in runtime["micropip"]:
        name, pinned = requirement.split("==")
        if (latest := pypi(name)) != pinned:
            lines.append(f"{name}: {pinned} is pinned, and {latest} is the latest release on PyPI.")
    return lines


def main() -> int:
    lines = behind(json.loads(RUNTIME.read_text(encoding="utf-8")))
    print("\n".join(lines) or "web/runtime.json pins the latest Pyodide release and the latest PyPI releases.")
    return 1 if lines else 0


if __name__ == "__main__":
    sys.exit(main())
