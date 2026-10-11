"""Fixtures for the built app in Chromium, with Pyodide in its worker: the page as a reader gets it.

The app is what ``--web-app`` names (a directory, served here, or a URL) or a build of this
checkout. A module opens the page once and waits for Python to start, which takes as long as
half a minute on a first visit, and its tests share the page. With ``--cdn-cache`` (on unless
``off``), the downloads that never change (Pyodide's files, Plotly.js, PyPI's files) are kept
on disk between runs; the index pages are always fetched.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from tests.web._browser import Console, cache_downloads, failed, serve
from tests.web.e2e._page import status

if TYPE_CHECKING:
    from collections.abc import Iterator

HARNESS = Path(__file__).parents[1] / "harness.js"

#: How long Python may take to start: Pyodide, numpy, pandas, scipy, statsmodels, the wheel (ms).
BOOT_TIMEOUT = 300_000


@pytest.fixture(scope="session")
def app_url(web_app: Path | str) -> Iterator[str]:
    """Return the URL of the app under test, serving it first when it is a directory."""
    if isinstance(web_app, str):
        yield web_app
    else:
        yield from serve(web_app)


@dataclass
class Live:
    """The module's page, its console, its browser context, and the worked example the build drew."""

    page: Any
    console: Console
    context: Any
    demo: dict


@pytest.fixture(scope="module")
def live(chromium: Any, app_url: str, request: pytest.FixtureRequest) -> Iterator[Live]:  # noqa: ANN401
    """Open the app once for the module, and return it once Python has started and the first paint is done."""
    context = chromium.new_context(viewport={"width": 1280, "height": 900})
    context.add_init_script(path=HARNESS)  # the page's own worker, watched
    cache = request.config.getoption("--cdn-cache")
    if cache != "off":
        cache_downloads(context, Path(cache))
    tracing = bool(request.config.getoption("--e2e-artifacts"))
    if tracing:
        context.tracing.start(screenshots=True, snapshots=True)
    console = Console()
    page = context.new_page()
    console.watch(page)
    page.goto(app_url)
    page.wait_for_function(
        "/^(Ready|Could not)/.test(document.querySelector('#status .status__text').textContent)", timeout=BOOT_TIMEOUT
    )
    text, _ = status(page)
    if not text.startswith("Ready"):
        pytest.fail(f"Python did not start in the page: {text}")
    demo = page.evaluate("fetch('demo.json').then((r) => r.json())")
    charts = 1 + len(demo["comparison"]["result"]["designs"])
    page.wait_for_function(
        "window.__pi.count('api_catalogue') === 1 && window.__pi.idle()"
        f" && document.querySelectorAll('#compare-charts .js-plotly-plot').length === {charts}",
        timeout=60_000,
    )
    yield Live(page, console, context, demo)
    if tracing:
        context.tracing.stop()
    context.close()


@pytest.fixture
def page(live: Live, request: pytest.FixtureRequest) -> Iterator[Any]:
    """Return the module's page for one test; an error it writes to the console during the test fails it."""
    seen = len(live.console.errors)
    tracing = bool(request.config.getoption("--e2e-artifacts"))
    if tracing:
        live.context.tracing.start_chunk()
    yield live.page
    if tracing:
        if failed(request):
            target = Path(request.config.getoption("--e2e-artifacts"))
            target.mkdir(parents=True, exist_ok=True)
            name = re.sub(r"[^\w.-]+", "_", request.node.nodeid)
            live.page.screenshot(path=target / f"{name}.png", full_page=True)
            live.context.tracing.stop_chunk(path=target / f"{name}.zip")
        else:
            live.context.tracing.stop_chunk()
    errors = [e for e in live.console.errors[seen:] if not any(p.search(e) for p in live.console.allowed)]
    if errors:
        pytest.fail(f"The page wrote errors to its console: {errors}")


@pytest.fixture
def demo(live: Live) -> dict:
    """Return the worked example as the build computed it, natively, into ``demo.json``."""
    return live.demo
