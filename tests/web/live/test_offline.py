"""The built app without a network: it opens from the copy the page keeps, and with the whole copy it works.

Once Python has started, a first visit registers ``sw.js`` and copies what it loaded: the page's own files, the
Pyodide release with its packages, the wheel. The reader asks for the charts and workbooks too,
which need plotly, openpyxl and Plotly.js. Then the network goes: the page opens again, Python starts from the
copy, and a design is made, analysed, drawn and written to a workbook.

The context here has the service worker and no routes: a request the worker answers passes by any route, so the
downloads are the real ones, as a reader's are. ``tests/web/js`` runs the same code against a stand-in for the
browser, state by state; these tests are what a real browser makes of it.
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
#: Ask for the app's wheel, a file the worker serves from the copy: resolves once the whole of it has arrived.
THE_WHEEL_AGAIN = """fetch('wheels/manifest.json').then((got) => got.json())
    .then((manifest) => fetch(`wheels/${manifest.wheel}`)).then((got) => got.arrayBuffer()).then(() => true)"""


def until(page: Any, expression: str, wanted: object, timeout: float = 60) -> object:  # noqa: ANN401 - a Playwright Page
    """Evaluate an expression that resolves in time, until it gives ``wanted`` or ``timeout`` seconds pass; return it.

    ``page.wait_for_function`` cannot wait for a promise: one counts as true at once, whatever it resolves to.
    """
    deadline = time.monotonic() + timeout
    while (found := page.evaluate(expression)) != wanted and time.monotonic() < deadline:
        page.wait_for_timeout(200)
    return found


def said(page: Any, start: str, timeout: float = 180_000) -> None:  # noqa: ANN401 - a Playwright Page
    """Wait until the footer's words on the copy start with ``start``."""
    page.wait_for_function(f"{SAID}.startsWith({json.dumps(start)})", timeout=timeout)


def a_study_with_its_charts_and_its_workbook(page: Any) -> None:  # noqa: ANN401 - a Playwright Page
    """Make a design, simulate and analyse it, draw its charts and download its workbook: all that needs Python."""
    set_factors(page, FACTORS)
    candidates_done(page)
    generate(page, design_type="full_factorial")
    assert len(run_sheet(page)) == 11
    practise(page)
    settled(page, timeout=120_000)
    reveal(page, "#charts")
    page.wait_for_function(
        "document.querySelectorAll('#charts .js-plotly-plot').length >= 3", timeout=120_000
    )  # plotly, installed, and Plotly.js
    reveal(page, "#download")
    with page.expect_download(timeout=120_000) as download:
        page.click("#download")  # openpyxl, installed
    assert download.value.suggested_filename == "full_factorial-11-runs.xlsx"


def watched(chromium: Any) -> tuple[Any, Any, Console]:  # noqa: ANN401 - Playwright's browser, context and page
    """Open a context that has the service worker, and a page in it whose console is watched."""
    context = chromium.new_context(viewport={"width": 1280, "height": 900})
    context.add_init_script(path=HARNESS)
    console = Console()
    console.allow(r"ERR_INTERNET_DISCONNECTED")  # the browser reports each request the lost network refuses
    page = context.new_page()
    console.watch(page)
    return context, page, console


def test_the_page_opens_and_works_without_a_network_once_its_copy_is_kept(chromium, app_url):
    context, page, console = watched(chromium)
    try:
        page.goto(app_url)
        started(page)
        said(page, "Factorial opens on this device without a network")
        assert page.evaluate(REGISTERED)
        page.click("#offline-keep")
        said(page, "Factorial opens and works on this device", timeout=300_000)
        assert page.is_hidden("#offline-keep"), "nothing more to keep"
        assert page.is_visible("#offline-remove")

        context.set_offline(True)
        page.reload()
        started(page)  # Python, its packages and the wheel come from the copy
        said(page, "Factorial opens and works on this device")  # read from the copy, with no network to ask
        a_study_with_its_charts_and_its_workbook(page)  # plotly and openpyxl come from the copy: no index is asked
        reveal(page, "#charts")
        page.locator("#charts figure.chart .expand").first.click()  # the window a figure opens in, from the copy too
        page.wait_for_selector("dialog.figure[open] .main-svg")
        page.keyboard.press("Escape")
        listed = context.new_page()  # what Factorial does: a page of the app, and the document it reads, from the copy
        listed.goto(f"{app_url}features.html")
        listed.wait_for_selector("#features h2")
        listed.close()
        assert not console.unexpected(), console.unexpected()

        context.set_offline(False)
        page.click("#offline-remove")
        said(page, "Factorial keeps no copy of itself on this device.")
        assert until(page, REGISTERED, wanted=False) is False
        assert page.evaluate(COPIES) == []
        # The worker serves this page until it is closed, and looks in the copy for each file asked of it. In
        # Chromium a look by a cache's name makes the cache: the copy's name came back that way, empty.
        page.evaluate(THE_WHEEL_AGAIN)
        assert page.evaluate(COPIES) == [], "a file asked for after the copy was removed brings no copy back"
    finally:
        context.close()


def test_the_whole_copy_is_whole_whatever_the_visit_had_installed_before_it_was_asked_for(chromium, app_url):
    """A copy asked for after the charts and a workbook were made holds their packages all the same.

    Python had installed plotly and openpyxl while no copy was kept, so nothing of them passed through the worker
    into one. The page copies each file of the whole copy by its address, whatever Python has installed.
    """
    context, page, console = watched(chromium)
    try:
        page.goto(app_url)
        started(page)
        said(page, "Factorial opens on this device without a network")
        page.click("#offline-remove")
        said(page, "Factorial keeps no copy of itself on this device.")
        a_study_with_its_charts_and_its_workbook(page)  # installed with no copy to keep them in
        assert page.evaluate(COPIES) == [], "nothing is kept that the reader removed"

        page.click("#offline-keep")  # "Keep a copy, to open it without a network"
        said(page, "Factorial opens on this device without a network")
        page.click("#offline-keep")  # "Keep its charts and workbooks too"
        said(page, "Factorial opens and works on this device", timeout=300_000)

        context.set_offline(True)
        page.reload()
        started(page)
        a_study_with_its_charts_and_its_workbook(page)
        assert not console.unexpected(), console.unexpected()
    finally:
        context.close()


def test_a_build_made_without_the_worker_retires_it_in_a_browser_that_has_it(chromium, web_app, tmp_path):
    """The way out of a worker gone wrong, as ``--no-service-worker`` builds it: in two parts, each enough alone.

    The build's ``sw.js`` is the retiring worker, which the browser fetches at the next visit to any page of the
    app, whatever the page's own code does: here a page that runs none of the app's. And its manifest tells the
    app's page to unregister the worker and delete the copies itself.
    """
    if isinstance(web_app, str):
        pytest.skip("the files of a published app cannot be changed from here")
    app = tmp_path / "app"
    shutil.copytree(web_app, app)
    manifest, worker = app / "wheels" / "manifest.json", app / "sw.js"
    kept = {"manifest": manifest.read_text(encoding="utf-8"), "worker": worker.read_text(encoding="utf-8")}

    def a_copy_is_kept(page: Any) -> None:  # noqa: ANN401 - a Playwright Page
        started(page)
        said(page, "Factorial opens on this device")
        assert page.evaluate(REGISTERED)
        assert sorted(page.evaluate(COPIES)) == ["factorial-app-1", "factorial-kept-1", "factorial-made-1"]

    for url in serve(app):
        context = chromium.new_context(viewport={"width": 1280, "height": 900})
        try:
            page = context.new_page()
            page.goto(url)
            a_copy_is_kept(page)

            worker.write_text((app / "sw-retire.js").read_text(encoding="utf-8"), encoding="utf-8")
            page.goto(f"{url}about.html")  # a page of the app that runs none of its code
            assert until(page, REGISTERED, wanted=False) is False, "the retiring worker unregistered"
            assert until(page, COPIES, wanted=[]) == [], "and deleted the copies"

            worker.write_text(kept["worker"], encoding="utf-8")
            page.goto(url)
            a_copy_is_kept(page)
            retired = {**json.loads(kept["manifest"]), "service_worker": False}
            manifest.write_text(json.dumps(retired), encoding="utf-8")
            page.reload()
            started(page)
            assert until(page, REGISTERED, wanted=False) is False, "the page unregistered the worker"
            assert until(page, COPIES, wanted=[]) == []
            assert page.is_hidden("#offline")
            page.evaluate(THE_WHEEL_AGAIN)  # through the worker, which serves this page until it is closed
            assert page.evaluate(COPIES) == [], "and a file asked for afterwards brings no copy back"
        finally:
            context.close()
