"""Fixtures for the page in Chromium, its Python answered natively.

Each test gets a fresh page. Its worker is never started: ``harness.js`` answers every call in
this process, from ``web/bootstrap.py``, through the binding ``window.__piNative``, so these
tests need neither Pyodide nor the network. ``demo.json``, the manifest and Plotly.js are
served from here too, and any other request leaving the page is refused: a page that starts
reaching for another site fails here until the tests serve what it asks for.
"""

from __future__ import annotations

import json
import re
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from tests.web._browser import Console, keep_artifacts, serve
from tests.web._support import WEB, no_bytecode
from tests.web.e2e._page import first_paint_done

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from types import ModuleType

HARNESS = Path(__file__).with_name("harness.js")


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
def plotly_js() -> tuple[str, bytes]:
    """Return the Plotly.js the Python plotly package ships, and its version: what the page draws with."""
    plotly = pytest.importorskip("plotly", reason="the page draws its charts with Plotly.js")
    from plotly.offline import get_plotlyjs_version

    return get_plotlyjs_version(), (Path(plotly.__file__).parent / "package_data" / "plotly.min.js").read_bytes()


def _native(bootstrap: ModuleType) -> Callable[[str, str], str]:
    """Answer the page's calls as ``web/worker.js`` does, with ``web/bootstrap.py`` run here."""

    def answer(fn: str, payload: str) -> str:
        try:
            return getattr(bootstrap, fn)(payload)
        except Exception as exc:  # noqa: BLE001 - worker.js sends any exception back as the error
            return json.dumps({"ok": False, "error": "".join(traceback.format_exception(exc))})

    return answer


@pytest.fixture
def console() -> Console:
    """Collect what the page writes to its console as an error; declare expected ones with ``allow``."""
    return Console()


@pytest.fixture
def context(  # noqa: PLR0913 - a fixture's arguments are the fixtures it needs
    chromium: Any,  # noqa: ANN401 - a Playwright Browser
    bootstrap: ModuleType,
    site: str,
    first_paint: dict[str, Any],
    plotly_js: tuple[str, bytes],
    console: Console,
    request: pytest.FixtureRequest,
) -> Iterator[Any]:
    """Open a browser context whose pages are answered natively; fail the test on any console error."""
    version, script = plotly_js
    context = chromium.new_context(viewport={"width": 1280, "height": 900})
    context.add_init_script("window.__piTransport = 'native';")
    context.add_init_script(path=HARNESS)
    context.expose_function("__piNative", _native(bootstrap))
    # The last route added is tried first, so the refusal of everything else goes in first.
    context.route(
        "**/*",
        lambda route: route.continue_() if route.request.url.startswith(site) else route.abort("blockedbyclient"),
    )
    context.route(f"{site}demo.json", lambda route: route.fulfill(json=first_paint))
    context.route(f"{site}wheels/manifest.json", lambda route: route.fulfill(json={"plotly_js": version}))
    context.route(
        re.compile(r"/plotly\.min\.js$"), lambda route: route.fulfill(body=script, content_type="text/javascript")
    )
    context.on("page", console.watch)
    if request.config.getoption("--e2e-artifacts"):
        context.tracing.start(screenshots=True, snapshots=True)
    yield context
    keep_artifacts(request, context, context.pages)
    context.close()
    unexpected = console.unexpected()
    if unexpected:
        pytest.fail(f"The page wrote errors to its console: {unexpected}")


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
