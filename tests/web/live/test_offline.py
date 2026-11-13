"""The built app without a network: it opens from the copy its service worker keeps, and with the whole copy it works.

A first visit registers ``sw.js`` and, once Python has started, copies what the visit loaded: the page's own files,
the Pyodide release with its packages, the wheel. The reader asks for the charts and workbooks too, which need
plotly, openpyxl and Plotly.js. Then the network goes: the page opens again, Python starts from the copy, and a
design is made, analysed, drawn and written to a workbook.

The context here has the service worker and no routes: a request the worker answers passes by any route, so the
downloads are the real ones, as a reader's are.
"""

from __future__ import annotations

import json
import shutil
import time
from typing import Any

import pytest

from tests.web._browser import Console, serve
from tests.web._support import FACTORS
from tests.web.e2e._page import candidates_done, generate, practise, reveal, run_sheet, set_factors, settled
from tests.web.live.conftest import HARNESS
from tests.web.live.test_live import started

SAID = "document.querySelector('#offline span').textContent"
COPIES = "caches.keys().then((names) => names.filter((name) => name.startsWith('factorial-')))"
REGISTERED = "navigator.serviceWorker.getRegistration().then((found) => found !== undefined)"


def until(page: Any, expression: str, wanted: object, timeout: float = 60) -> object:  # noqa: ANN401 - a Playwright Page
    """Evaluate an expression that resolves in time, until it gives ``wanted`` or ``timeout`` seconds pass; return it.

    ``page.wait_for_function`` cannot wait for a promise: one counts as true at once, whatever it resolves to.
    """
    deadline = time.monotonic() + timeout
    while (found := page.evaluate(expression)) != wanted and time.monotonic() < deadline:
        page.wait_for_timeout(200)
    return found


def test_the_page_opens_and_works_without_a_network_once_its_copy_is_kept(chromium, app_url):
    context = chromium.new_context(viewport={"width": 1280, "height": 900})
    context.add_init_script(path=HARNESS)
    console = Console()
    console.allow(r"ERR_INTERNET_DISCONNECTED")  # the browser reports each request the lost network refuses
    page = context.new_page()
    console.watch(page)
    try:
        page.goto(app_url)
        started(page)
        page.wait_for_function(
            f"{SAID}.startsWith('Factorial opens on this device without a network')", timeout=180_000
        )
        page.click("#offline-keep")
        page.wait_for_function(f"{SAID}.startsWith('Factorial opens and works on this device')", timeout=300_000)
        assert page.is_hidden("#offline-keep"), "nothing more to keep"
        assert page.is_visible("#offline-remove")

        context.set_offline(True)
        page.reload()
        started(page)  # Python, its packages and the wheel come from the copy
        assert page.evaluate(f"{SAID}").startswith("Factorial opens and works on this device"), "said from the first"
        set_factors(page, FACTORS)
        candidates_done(page)
        generate(page, design_type="full_factorial")
        assert len(run_sheet(page)) == 11
        practise(page)
        settled(page, timeout=120_000)
        page.wait_for_function(
            "document.querySelectorAll('#charts .js-plotly-plot').length >= 3", timeout=120_000
        )  # plotly, installed from the copy, and Plotly.js, served from it
        reveal(page, "#download")
        with page.expect_download(timeout=120_000) as download:
            page.click("#download")  # openpyxl, installed from the copy
        assert download.value.suggested_filename == "full_factorial-11-runs.xlsx"
        assert not console.unexpected(), console.unexpected()

        context.set_offline(False)
        page.click("#offline-remove")
        page.wait_for_function(f"{SAID} === 'Factorial keeps no copy of itself on this device.'")
        assert until(page, REGISTERED, wanted=False) is False
        assert page.evaluate(COPIES) == []
    finally:
        context.close()


def test_a_build_made_without_the_worker_retires_it_in_a_browser_that_has_it(chromium, web_app, tmp_path):
    """The way out of a worker gone wrong: the next build says so in its manifest, and the copy goes with the worker."""
    if isinstance(web_app, str):
        pytest.skip("the manifest of a published app cannot be changed from here")
    app = tmp_path / "app"
    shutil.copytree(web_app, app)
    manifest = app / "wheels" / "manifest.json"
    for url in serve(app):
        context = chromium.new_context(viewport={"width": 1280, "height": 900})
        try:
            page = context.new_page()
            page.goto(url)
            started(page)
            page.wait_for_function(f"{SAID}.startsWith('Factorial opens on this device')", timeout=180_000)
            assert page.evaluate(REGISTERED)
            assert sorted(page.evaluate(COPIES)) == ["factorial-app-1", "factorial-kept-1"]

            retired = {**json.loads(manifest.read_text(encoding="utf-8")), "service_worker": False}
            manifest.write_text(json.dumps(retired), encoding="utf-8")  # as --no-service-worker builds it
            page.reload()
            started(page)
            assert until(page, REGISTERED, wanted=False) is False
            assert until(page, COPIES, wanted=[]) == []
            assert page.is_hidden("#offline")
        finally:
            context.close()
