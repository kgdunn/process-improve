"""Running the page in Chromium for the browser tests: serving it, launching Chromium, watching the console.

Playwright is imported only by the fixtures that need it, so the rest of ``tests/web`` runs
without it. Locally it comes from ``uv run --with playwright==1.64.1``; the Chromium it drives
from ``playwright install chromium``, or from ``PI_CHROMIUM``, the path of another build.
"""

from __future__ import annotations

import functools
import hashlib
import http.server
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from tests.web._pyodide import CDN, LONGEST_WAIT, RUNTIME, package

if TYPE_CHECKING:
    from collections.abc import Iterator

PLAYWRIGHT = "playwright==1.64.1"
RUN_HINT = f"run the browser tests with `uv run --with {PLAYWRIGHT} pytest ...`"
INSTALL_HINT = f"install Chromium with `uv run --with {PLAYWRIGHT} playwright install chromium`, or set PI_CHROMIUM"


#: The Content-Security-Policy factori.al serves every page with (deploy/factorial/factorial.caddy): the browser tests
#: serve the app under it too, so a page that loads from, or sends to, a place it does not name fails here first.
POLICY = re.search(
    r'^\s*Content-Security-Policy "([^"]+)"$',
    (Path(__file__).resolve().parents[2] / "deploy" / "factorial" / "factorial.caddy").read_text(encoding="utf-8"),
    flags=re.MULTILINE,
)[1]


class _Quiet(http.server.SimpleHTTPRequestHandler):
    """Serve files without logging each request, without letting the browser cache them, and under the site's policy."""

    def log_message(self, *_: object) -> None:
        """Say nothing."""

    def end_headers(self) -> None:
        """Ask the browser not to cache, so each test sees the files as they are on disk; send the site's policy."""
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy", POLICY)
        super().end_headers()


def serve(directory: Path) -> Iterator[str]:
    """Serve ``directory`` on a free local port for as long as the generator is open; yield its URL."""
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_Quiet, directory=str(directory)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/"
    finally:
        server.shutdown()
        server.server_close()


def launch(headed: bool) -> Iterator[Any]:
    """Start Playwright and Chromium for as long as the generator is open; yield the browser.

    A test run without Playwright, or without a Chromium it can start, fails with the command
    that installs it, rather than skipping: a run asked to test the page must test it.
    """
    try:
        from playwright.sync_api import Error, sync_playwright
    except ImportError:
        pytest.fail(f"The browser tests need Playwright for Python: {RUN_HINT}.")
    manager = sync_playwright().start()
    # Behind an HTTPS proxy (as in a sandbox), the CDN goes through it and the local server does not.
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    args = [f"--proxy-server={proxy}", "--proxy-bypass-list=127.0.0.1;localhost"] if proxy else []
    try:
        if os.environ.get("PI_BROWSER") == "webkit":
            # Safari's engine, for the faults only it shows. It takes its proxy as a setting, not as a switch.
            route = {"server": proxy, "bypass": "127.0.0.1,localhost"} if proxy else None
            browser = manager.webkit.launch(headless=not headed, proxy=route)
        else:
            browser = manager.chromium.launch(
                headless=not headed, executable_path=os.environ.get("PI_CHROMIUM") or None, args=args
            )
    except Error as exc:  # a missing or broken browser
        manager.stop()
        pytest.fail(f"Chromium could not be started ({exc}); {INSTALL_HINT}.")
    try:
        yield browser
    finally:
        browser.close()
        manager.stop()


@dataclass
class Console:
    """What a page writes to its console as an error, and what it throws: a test fails on either."""

    errors: list[str] = field(default_factory=list)
    allowed: list[re.Pattern] = field(default_factory=list)

    def watch(self, page: Any) -> None:  # noqa: ANN401 - a Playwright Page
        """Listen to a page for errors."""

        def console(message: Any) -> None:  # noqa: ANN401 - a Playwright ConsoleMessage
            if message.type == "error":
                self.errors.append(message.text)

        page.on("console", console)
        page.on("pageerror", lambda error: self.errors.append(f"uncaught: {error}"))

    def allow(self, pattern: str) -> None:
        """Expect errors matching ``pattern``: the test is about them."""
        self.allowed.append(re.compile(pattern))

    def unexpected(self) -> list[str]:
        """Return the errors no ``allow`` covers."""
        return [e for e in self.errors if not any(p.search(e) for p in self.allowed)]


#: Each test's reports by phase, kept by ``pytest_runtest_makereport`` in ``tests/web/conftest.py``.
REPORTS = pytest.StashKey[dict[str, pytest.TestReport]]()


def failed(request: pytest.FixtureRequest) -> bool:
    """Say whether the test using a fixture has failed so far."""
    return any(report.failed for report in request.node.stash.get(REPORTS, {}).values())


def keep_artifacts(request: pytest.FixtureRequest, context: Any, pages: list[Any]) -> None:  # noqa: ANN401
    """Stop tracing a browser context, keeping the trace and a screenshot of each page if the test failed."""
    directory = request.config.getoption("--e2e-artifacts")
    if not directory:
        return
    if not failed(request):
        context.tracing.stop()
        return
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    name = re.sub(r"[^\w.-]+", "_", request.node.nodeid)
    for i, page in enumerate(pages):
        if not page.is_closed():
            page.screenshot(path=target / f"{name}-{i}.png", full_page=True)
    context.tracing.stop(path=target / f"{name}.zip")


#: How long Python may take to start in a page: Pyodide, numpy, pandas, scipy, statsmodels, the wheel (ms).
BOOT_TIMEOUT = 300_000

#: Downloads that never change once published: a Pyodide release's files, a Plotly.js release and
#: PyPI's files. Index pages, which do change, are always fetched.
IMMUTABLE = (
    re.compile(r"^https://cdn\.jsdelivr\.net/pyodide/v[\d.]+/"),
    re.compile(r"^https://cdn\.jsdelivr\.net/npm/plotly\.js-dist-min@[\d.]+/"),
    re.compile(r"^https://files\.pythonhosted\.org/packages/"),
)

#: The response headers a cached download is served with: its type, and leave for the page to read it.
KEPT_HEADERS = {"content-type", "access-control-allow-origin"}

#: Fetches of one download when the CDN answers with a server error (503 for some minutes, #1315): the waits
#: between them double from 2 s, about a minute in all.
FETCH_ATTEMPTS = 6

#: The type a file of the Pyodide release is served with when the page's copy comes from disk, by its extension.
RELEASE_TYPES = {
    ".wasm": "application/wasm",
    ".mjs": "text/javascript",
    ".js": "text/javascript",
    ".json": "application/json",
    ".zip": "application/zip",
    ".whl": "application/zip",
}


def _on_disk(url: str) -> Path | None:
    """Return the copy on disk of a file of the page's Pyodide release, which CI fetches before the tests; else None.

    ``tests/web/_pyodide.py`` keeps the release's folder beside its npm package, each wheel checked against the lock
    file's digest, and the npm package holds the interpreter, the same file in every folder.
    """
    runtime = json.loads(RUNTIME.read_text(encoding="utf-8"))
    cdn = CDN.format(**runtime)
    if not url.startswith(cdn):
        return None
    name = url.removeprefix(cdn)
    if "/" in name or Path(name).suffix not in RELEASE_TYPES:
        return None
    for path in (package() / runtime["distribution"] / name, package() / name):
        if path.is_file():
            return path
    return None


def _fetched(route: Any) -> Any:  # noqa: ANN401 - a Playwright Route and APIResponse
    """Fetch a download for the page, again after a wait while the server answers with an error of its own."""
    response = route.fetch()
    for attempt in range(1, FETCH_ATTEMPTS):
        if response.status < 500:  # an error of the server's own is 500 or more
            break
        time.sleep(min(2**attempt, LONGEST_WAIT))
        response = route.fetch()
    return response


def cache_downloads(context: Any, directory: Path) -> None:  # noqa: ANN401 - a Playwright BrowserContext
    """Serve the immutable downloads of a context's pages from a cache on disk, filling it on a miss.

    A file of the page's Pyodide release that CI fetched before the tests is served from that copy, so a CDN that
    answers with an error for some minutes (#1315) fails no test whose change is elsewhere.
    """
    directory.mkdir(parents=True, exist_ok=True)

    def handle(route: Any) -> None:  # noqa: ANN401 - a Playwright Route
        key = hashlib.sha256(route.request.url.encode()).hexdigest()
        body, meta = directory / key, directory / f"{key}.json"
        if body.is_file() and meta.is_file():
            route.fulfill(status=200, headers=json.loads(meta.read_text(encoding="utf-8")), body=body.read_bytes())
            return
        if (copy := _on_disk(route.request.url)) is not None:
            headers = {"content-type": RELEASE_TYPES[copy.suffix], "access-control-allow-origin": "*"}
            route.fulfill(status=200, headers=headers, body=copy.read_bytes())
            return
        response = _fetched(route)
        if response.ok:
            partial = directory / f"{key}.{os.getpid()}.partial"
            partial.write_bytes(response.body())
            partial.replace(body)
            kept = {k: v for k, v in response.headers.items() if k.lower() in KEPT_HEADERS}
            meta.write_text(json.dumps(kept), encoding="utf-8")
        route.fulfill(response=response)

    for pattern in IMMUTABLE:
        context.route(pattern, handle)
