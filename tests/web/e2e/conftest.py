"""Fixtures for the page in Chromium, its Python answered natively.

Each test gets a fresh page. Its worker is never started: ``harness.js`` answers every call in
this process, from ``web/bootstrap.py``, through the binding ``window.__piNative``, so these
tests need neither Pyodide nor the network. ``demo.json``, the book's examples, the catalog of a
few designs, the manifest and Plotly.js are served from here too, and any other request leaving
the page is refused: a page that starts reaching for another site fails here until the tests
serve what it asks for. A test about the worker itself parametrizes ``transport`` to ``worker``:
the page's own worker then runs, and finds Pyodide's download refused.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from process_improve.experiments import design_catalog
from tests.web._browser import Console, keep_artifacts, serve
from tests.web._support import (
    BLOCKED_DESIGN,
    BOOK_DESIGN,
    CATALOG_SAMPLE,
    ENTRY_POINT,
    FACTORS,
    MANY_FACTOR_DESIGNS,
    PAIRED_DESIGN,
    WEB,
    catalog_sample,
    no_bytecode,
)
from tests.web.e2e._page import candidates_done, first_paint_done, set_factors

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
        examples |= {
            entry["name"]: build_script.published_example(entry) for entry in build_script.published_examples()
        }
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


@pytest.fixture(scope="session")
def catalog_files(build_script: ModuleType, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write the catalog as the build does, for a few designs (``CATALOG_SAMPLE``), and return its folder."""
    out = tmp_path_factory.mktemp("catalog")
    sample = catalog_sample()
    with pytest.MonkeyPatch.context() as patch, no_bytecode():
        patch.setattr(design_catalog, "catalog_records", lambda: sample)
        build_script.write_catalog(out)
    return out / "catalog"


@pytest.fixture(scope="session")
def book_catalog_files(build_script: ModuleType, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write the catalog of ``CATALOG_SAMPLE`` and ``BOOK_DESIGN``, a design the book prints; return its folder."""
    out = tmp_path_factory.mktemp("book-catalog")
    sample = catalog_sample((*CATALOG_SAMPLE, BOOK_DESIGN))
    with pytest.MonkeyPatch.context() as patch, no_bytecode():
        patch.setattr(design_catalog, "catalog_records", lambda: sample)
        build_script.write_catalog(out)
    return out / "catalog"


@pytest.fixture(scope="session")
def many_catalog_files(build_script: ModuleType, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write the catalog of ``CATALOG_SAMPLE`` and ``MANY_FACTOR_DESIGNS``, of 31 and 40 factors; return its folder."""
    out = tmp_path_factory.mktemp("many-catalog")
    sample = catalog_sample((*CATALOG_SAMPLE, *MANY_FACTOR_DESIGNS))
    with pytest.MonkeyPatch.context() as patch, no_bytecode():
        patch.setattr(design_catalog, "catalog_records", lambda: sample)
        build_script.write_catalog(out)
    return out / "catalog"


@pytest.fixture(scope="session")
def blocked_catalog_files(build_script: ModuleType, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write the catalog of ``CATALOG_SAMPLE`` with ``BLOCKED_DESIGN`` and ``PAIRED_DESIGN``; return its folder."""
    out = tmp_path_factory.mktemp("blocked-catalog")
    sample = catalog_sample((*CATALOG_SAMPLE, BLOCKED_DESIGN, PAIRED_DESIGN))
    with pytest.MonkeyPatch.context() as patch, no_bytecode():
        patch.setattr(design_catalog, "catalog_records", lambda: sample)
        build_script.write_catalog(out)
    return out / "catalog"


def _catalog(route: Any, folder: Path) -> None:  # noqa: ANN401 - a Playwright Route
    """Answer ``catalog/index.json`` and ``catalog/detail/<name>.json`` from what the build wrote, or 404."""
    path = folder / route.request.url.split("/catalog/", 1)[1].split("?")[0]
    if path.resolve().is_relative_to(folder.resolve()) and path.is_file():
        route.fulfill(body=path.read_bytes(), content_type="application/json")
    else:
        route.fulfill(status=404, body="Not Found")


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
    catalog_files: Path,
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
        context.route(
            re.compile(re.escape(f"{site}catalog/") + r".+\.json$"), lambda route: _catalog(route, catalog_files)
        )
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
def book_context(context: Any, site: str, book_catalog_files: Path) -> Any:  # noqa: ANN401 - a Playwright BrowserContext
    """Return a browser context whose catalog also holds ``BOOK_DESIGN``: this route is tried before the sample's."""
    context.route(
        re.compile(re.escape(f"{site}catalog/") + r".+\.json$"), lambda route: _catalog(route, book_catalog_files)
    )
    return context


@pytest.fixture
def many_context(context: Any, site: str, many_catalog_files: Path) -> Any:  # noqa: ANN401 - a Playwright BrowserContext
    """Return a browser context whose catalog also holds designs of 31 and 40 factors, routed before the sample's."""
    context.route(
        re.compile(re.escape(f"{site}catalog/") + r".+\.json$"), lambda route: _catalog(route, many_catalog_files)
    )
    return context


@pytest.fixture
def blocked_context(context: Any, site: str, blocked_catalog_files: Path) -> Any:  # noqa: ANN401 - a Playwright BrowserContext
    """Return a browser context whose catalog also holds the designs in blocks, routed before the sample's."""
    context.route(
        re.compile(re.escape(f"{site}catalog/") + r".+\.json$"), lambda route: _catalog(route, blocked_catalog_files)
    )
    return context


@pytest.fixture
def page(context: Any, site: str) -> Any:  # noqa: ANN401 - a Playwright Page
    """Open the worked example, ``?example=worked``; return the page once its charts are drawn and Python answered."""
    page = context.new_page()
    page.goto(f"{site}?example=worked")
    first_paint_done(page)
    return page


@pytest.fixture
def fresh_page(context: Any, site: str) -> Any:  # noqa: ANN401 - a Playwright Page
    """Open the page as a first visit finds it, once Python has answered: a new study, nothing typed in yet."""
    page = context.new_page()
    page.goto(site)
    page.wait_for_function("window.__pi.count('api_catalogue') === 1 && window.__pi.idle()")
    return page


@pytest.fixture
def blank_page(fresh_page: Any) -> Any:  # noqa: ANN401 - a Playwright Page
    """Return a first visit with the worked example's three factors typed in and the response left blank, ``y``.

    The study has no run sheet yet; the designs that can learn the brief are listed before it is returned.
    """
    set_factors(fresh_page, FACTORS)
    candidates_done(fresh_page)
    return fresh_page
