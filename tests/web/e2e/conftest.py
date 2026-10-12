"""Fixtures for the page in Chromium, its Python answered natively.

Each test gets a fresh page. Its worker is never started: ``harness.js`` answers every call in
this process, from ``web/bootstrap.py``, through the binding ``window.__piNative``, so these
tests need neither Pyodide nor the network. ``demo.json``, the book's examples, the manifest
and Plotly.js are served from here too, and any other request leaving the page is refused: a page that starts
reaching for another site fails here until the tests serve what it asks for. A test about the
worker itself parametrizes ``transport`` to ``worker``: the page's own worker then runs, and
finds Pyodide's download refused.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from tests.web._browser import Console, keep_artifacts, serve
from tests.web._support import ENTRY_POINT, WEB, no_bytecode
from tests.web.e2e._page import first_paint_done

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from types import ModuleType

HARNESS = Path(__file__).parents[1] / "harness.js"


@pytest.fixture(scope="session")
def site() -> Iterator[str]:
    """Serve ``web/`` as it is in the checkout: with Python answered natively, nothing needs building."""
    yield from serve(WEB)


@pytest.fixture(scope="session")
def first_paint(build_script: ModuleType) -> dict[str, Any]:
    """Return ``demo.json`` as the build writes it: the worked example, computed through ``web/bootstrap.py``."""
    with no_bytecode():
        return json.loads(json.dumps(build_script.worked_example()))


@pytest.fixture(scope="session")
def book_files(build_script: ModuleType) -> dict[str, Any]:
    """Return each ``book/<name>.json``, and ``book/index.json`` as ``index``, as the build writes them."""
    with no_bytecode():
        examples = {entry["name"]: build_script.book_example(entry) for entry in build_script.book_examples()}
    return {**examples, "index": build_script.book_index(list(examples.values()))}


@pytest.fixture(scope="session")
def plotly_js() -> tuple[str, bytes]:
    """Return the Plotly.js the Python plotly package ships, and its version: what the page draws with."""
    plotly = pytest.importorskip("plotly", reason="the page draws its charts with Plotly.js")
    from plotly.offline import get_plotlyjs_version

    return get_plotlyjs_version(), (Path(plotly.__file__).parent / "package_data" / "plotly.min.js").read_bytes()


def _native(bootstrap: ModuleType) -> Callable[[str, str], str]:
    """Answer the page's calls as ``web/worker.js`` does, with ``web/bootstrap.py`` run here."""

    def answer(fn: str, payload: str) -> str:
        entry = getattr(bootstrap, fn, None) if ENTRY_POINT.match(fn) else None
        if not callable(entry):
            return json.dumps({"ok": False, "error": f"{json.dumps(fn)} is not one of the page's entry points."})
        try:
            return entry(payload)
        except Exception as exc:  # noqa: BLE001 - worker.js sends any exception back as the error, in one line
            return json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    return answer


def _book(route: Any, book_files: dict[str, Any]) -> None:  # noqa: ANN401 - a Playwright Route
    """Answer ``book/<name>.json`` and ``book/index.json`` as the published app does: what the build wrote, or 404."""
    name = route.request.url.rsplit("/", 1)[1].removesuffix(".json")
    if name in book_files:
        route.fulfill(json=book_files[name])
    else:
        route.fulfill(status=404, body="Not Found")


@pytest.fixture
def console() -> Console:
    """Collect what the page writes to its console as an error; declare expected ones with ``allow``."""
    return Console()


@pytest.fixture
def context_options() -> dict[str, Any]:
    """Return the browser context's settings: a desktop window. A test can parametrize this to change them."""
    return {"viewport": {"width": 1280, "height": 900}}


@pytest.fixture
def transport() -> str:
    """Return who answers the page's calls: ``native``, this process; ``worker``, the page's own worker."""
    return "native"


@pytest.fixture
def open_context(  # noqa: PLR0913 - a fixture's arguments are the fixtures it needs
    chromium: Any,  # noqa: ANN401 - a Playwright Browser
    context_options: dict[str, Any],
    transport: str,
    bootstrap: ModuleType,
    site: str,
    first_paint: dict[str, Any],
    book_files: dict[str, Any],
    plotly_js: tuple[str, bytes],
    console: Console,
    request: pytest.FixtureRequest,
) -> Iterator[Callable[[], Any]]:
    """Return a way to open a browser context whose pages are answered natively; fail on any console error.

    Each context is a browser of its own, its storage empty: a test about what one reader's browser
    keeps, or what reaches another reader's, opens a second.
    """
    version, script = plotly_js
    opened: list[Any] = []

    def open_one() -> Any:  # noqa: ANN401 - a Playwright BrowserContext
        context = chromium.new_context(**context_options)
        context.add_init_script(f"window.__piTransport = {json.dumps(transport)};")
        context.add_init_script(path=HARNESS)
        context.expose_function("__piNative", _native(bootstrap))
        # The last route added is tried first, so the refusal of everything else goes in first.
        context.route(
            "**/*",
            lambda route: route.continue_() if route.request.url.startswith(site) else route.abort("blockedbyclient"),
        )
        context.route(f"{site}demo.json", lambda route: route.fulfill(json=first_paint))
        context.route(re.compile(re.escape(f"{site}book/") + r"[^/]+\.json$"), lambda route: _book(route, book_files))
        context.route(f"{site}wheels/manifest.json", lambda route: route.fulfill(json={"plotly_js": version}))
        context.route(
            re.compile(r"/plotly\.min\.js$"), lambda route: route.fulfill(body=script, content_type="text/javascript")
        )
        context.on("page", console.watch)
        if request.config.getoption("--e2e-artifacts"):
            context.tracing.start(screenshots=True, snapshots=True)
        opened.append(context)
        return context

    yield open_one
    for context in opened:
        keep_artifacts(request, context, context.pages)
        context.close()
    unexpected = console.unexpected()
    if unexpected:
        pytest.fail(f"The page wrote errors to its console: {unexpected}")


@pytest.fixture
def context(open_context: Callable[[], Any]) -> Any:  # noqa: ANN401 - a Playwright BrowserContext
    """Open a browser context whose pages are answered natively; fail the test on any console error."""
    return open_context()


@pytest.fixture
def page(context: Any, site: str, first_paint: dict[str, Any]) -> Any:  # noqa: ANN401 - a Playwright Page
    """Open the page, and return it once it has drawn the worked example and Python has answered."""
    page = context.new_page()
    page.goto(site)
    first_paint_done(page, first_paint)
    return page


@pytest.fixture
def blank_page(context: Any, site: str, console: Console) -> Any:  # noqa: ANN401 - a Playwright Page
    """Open the page as a plain preview of ``web/`` shows it: no worked example, so every step waits for a design."""
    console.allow(r"404")  # the browser reports the missing demo.json
    context.route(f"{site}demo.json", lambda route: route.fulfill(status=404))
    page = context.new_page()
    page.goto(site)
    page.wait_for_function("window.__pi.count('api_catalogue') === 1 && window.__pi.idle()")
    return page
