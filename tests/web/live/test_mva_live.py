"""The multivariate page of the built app, with Python in its own worker under Pyodide.

``tests/web/e2e/test_mva_page.py`` answers the page's Python natively; this is the page as a
reader gets it: Pyodide started in ``mva/worker.js``, the app's one wheel installed from the
app's ``wheels/manifest.json``, and ``mva/bootstrap.py`` answering. It is not a smoke test, so
a fault here does not hold back the designed-experiments page's deploy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from tests.web._browser import BOOT_TIMEOUT, Console
from tests.web.live.conftest import new_context

if TYPE_CHECKING:
    from collections.abc import Iterator

UPSET = ["52", "53", "54"]
QUALITY = ["Conv", "Mn", "Mw", "LCB", "SCB"]


@pytest.fixture
def mva(chromium: Any, app_url: str, request: pytest.FixtureRequest) -> Iterator[tuple[Any, Console]]:  # noqa: ANN401
    """Open the built multivariate page and wait until Python has started in its worker."""
    context = new_context(chromium, request)
    console = Console()
    page = context.new_page()
    console.watch(page)
    page.goto(f"{app_url}mva/index.html")
    page.wait_for_function(
        "/^(Ready: process-improve|Could not)/.test(document.getElementById('status').textContent)",
        timeout=BOOT_TIMEOUT,
    )
    yield page, console
    context.close()


def test_the_built_page_fits_excludes_and_explains_under_pyodide(mva) -> None:
    page, console = mva
    assert page.text_content("#status").startswith("Ready: process-improve")
    page.click("#open-example")
    page.wait_for_function("document.getElementById('scores').data?.length > 0", timeout=120_000)
    assert "54 rows fitted" in page.text_content("#model-summary")

    page.evaluate("cols => window.__mva.store.select('cols', cols)", QUALITY)
    page.click("#use-y")
    page.wait_for_function("window.__mva.state.fit?.variables.length === 14", timeout=60_000)
    page.evaluate("ids => window.__mva.store.select('rows', ids)", UPSET)
    page.click("#exclude")
    page.wait_for_function(
        "document.getElementById('scores').data.some((t) => t.name === 'Excluded, projected')", timeout=60_000
    )
    assert "51 rows fitted, 3 excluded" in page.text_content("#model-summary")

    page.evaluate("ids => window.__mva.store.select('rows', ids)", ["48", "49", "50", "51", *UPSET])
    page.click("#explain")
    page.wait_for_selector("#explanation:not([hidden])", timeout=60_000)
    assert page.text_content("#summary tr:nth-child(2) td:first-child") == "z2"
    assert console.unexpected() == []
