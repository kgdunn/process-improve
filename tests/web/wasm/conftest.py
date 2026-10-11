"""Fixtures for the app's Python under Pyodide in Node.js: ``runner.mjs``, asked as the page's worker is asked.

``pyodide`` starts the runner once per module with the Pyodide release ``web/runtime.json``
names (the npm package from ``PI_PYODIDE``, or else from the cache below), the built app's
wheel and its ``bootstrap.py``. It answers through the same ``Api`` as native Python, so a test
asks both the same thing and compares the answers.
"""

from __future__ import annotations

import collections
import json
import os
import queue
import subprocess
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.web._build import app_file, wheel
from tests.web._support import WEB, Api

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

PYODIDE_CACHE = Path.home() / ".cache" / "process-improve" / "node" / "node_modules" / "pyodide"
RUNNER = Path(__file__).with_name("runner.mjs")

#: Seconds Pyodide may take to start (packages from the CDN on a first run), and to answer one call.
BOOT_TIMEOUT = 600
CALL_TIMEOUT = 300


def pyodide_package(version: str) -> Path:
    """Return the directory of the npm package of Pyodide ``version``, failing with how to install it."""
    found = Path(os.environ.get("PI_PYODIDE") or PYODIDE_CACHE)
    install = f"`npm install --prefix ~/.cache/process-improve/node --no-save pyodide@{version}`"
    if not (found / "package.json").is_file():
        pytest.fail(f"Pyodide for Node.js is missing: {install}, or set PI_PYODIDE to its directory.")
    installed = json.loads((found / "package.json").read_text(encoding="utf-8"))["version"]
    if installed != version:
        pytest.fail(f"Pyodide {installed} is installed where web/runtime.json names {version}: {install}.")
    return found


class Runner:
    """A running ``runner.mjs``: each ``api_*`` attribute is a call answered under Pyodide."""

    def __init__(self, process: subprocess.Popen) -> None:
        self.process = process
        self.lines: queue.Queue[str | None] = queue.Queue()
        self.said: collections.deque[str] = collections.deque(maxlen=40)  # the last of what it printed aside
        threading.Thread(target=self._read, args=(process.stdout, self.lines.put), daemon=True).start()
        threading.Thread(target=self._read, args=(process.stderr, self.said.append), daemon=True).start()

    @staticmethod
    def _read(stream: object, keep: Callable[[str | None], object]) -> None:
        for line in stream:  # type: ignore[attr-defined]
            keep(line.rstrip("\n"))
        keep(None)

    def answer(self, timeout: float) -> str:
        """Return the runner's next line, failing with what it printed aside if it stops or hangs."""
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty:
            line = None
        if line is None:
            said = "\n".join(s for s in self.said if s)
            pytest.fail(f"Pyodide in Node.js stopped answering:\n{said}")
        return line

    def __getattr__(self, fn: str) -> Callable[[str], str]:
        """Return a call to ``fn`` under Pyodide, taking and returning JSON text as the page's worker does."""
        if not fn.startswith("api_"):
            raise AttributeError(fn)

        def call(payload: str) -> str:
            self.process.stdin.write(json.dumps({"fn": fn, "payload": payload}) + "\n")
            self.process.stdin.flush()
            return self.answer(CALL_TIMEOUT)

        return call


@pytest.fixture(scope="module")
def pyodide(node: str, web_app: Path | str, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Api]:
    """Start the app's Python under Pyodide in Node.js, once for the module, and call it as the page does."""
    runtime = WEB / "runtime.json"
    package = pyodide_package(json.loads(runtime.read_text(encoding="utf-8"))["pyodide"])
    scratch = tmp_path_factory.mktemp("app")
    files = [wheel(web_app, scratch), app_file(web_app, "bootstrap.py", scratch)]
    process = subprocess.Popen(  # noqa: S603 - Node.js running this directory's runner
        [node, str(RUNNER), str(package), str(runtime), *map(str, files)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    runner = Runner(process)
    runner.version = json.loads(runner.answer(BOOT_TIMEOUT))["ready"]
    yield Api(runner)
    process.stdin.close()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
