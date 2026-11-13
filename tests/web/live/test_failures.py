"""What the page says when PyPI is refused: Python starts all the same, and only what needs PyPI says so.

Nothing at start-up comes from PyPI: Pyodide and its packages come from the CDN, and the wheel and the page's own
Python from the app. openpyxl comes from PyPI with the first workbook, and plotly with the first chart. A network
that reaches the CDN but not PyPI therefore still makes and analyses a design; the workbook is refused in one
line, and is written once PyPI answers.
"""

from __future__ import annotations

import re

from tests.web._browser import Console
from tests.web._support import FACTORS
from tests.web.e2e._page import candidates_done, generate, reveal, run_sheet, set_factors, status
from tests.web.live.test_live import started

PYPI = re.compile(r"^https://(pypi\.org|files\.pythonhosted\.org)/")


def test_a_refused_package_index_stops_only_the_call_that_needs_it(own_context, app_url):
    own_context.route(PYPI, lambda route: route.abort("blockedbyclient"))  # routed last, so tried first
    console = Console()
    console.allow(r"ERR_BLOCKED_BY_CLIENT")  # the browser reports each refused download
    page = own_context.new_page()
    console.watch(page)
    page.goto(app_url)
    started(page)
    set_factors(page, FACTORS)
    candidates_done(page)
    generate(page, design_type="full_factorial")
    assert len(run_sheet(page)) == 11, "a design needs nothing from PyPI"

    reveal(page, "#download")
    page.click("#download")
    page.wait_for_function("document.querySelector('#status').dataset.state === 'error'", timeout=120_000)
    text, _ = status(page)
    assert text.startswith("Could not install openpyxl: "), text
    assert "\n" not in text, "one line, not a traceback"

    own_context.unroute(PYPI)
    with page.expect_download(timeout=120_000) as download:
        page.click("#download")  # the install is tried again
    assert download.value.suggested_filename == "full_factorial-11-runs.xlsx"
    assert status(page) == ("Installed openpyxl for Excel workbooks.", "ok")
    assert not console.unexpected(), console.unexpected()
