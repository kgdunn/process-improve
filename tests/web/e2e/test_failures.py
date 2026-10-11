"""What the page says when Python cannot start: the reason, in one line, and nothing left waiting.

These tests run the page's own worker, which finds Pyodide's download refused, as a firewall
or a lost connection would refuse it. The worked example drawn at first paint stays readable.
"""

from __future__ import annotations

import re
import time

import pytest

from tests.web.e2e._page import status

pytestmark = pytest.mark.parametrize("transport", ["worker"])


def failed_to_start(page) -> str:
    """Wait for the status line to report that Python could not start, and return what it says."""
    page.wait_for_function("document.querySelector('#status').dataset.state === 'error'", timeout=15_000)
    text, _ = status(page)
    assert text.startswith("Could not start Python: "), text
    assert "\n" not in text, "one line, not a traceback"
    return text


@pytest.mark.xfail(strict=True, reason="defect: a refused Pyodide download leaves the page waiting for Python")
def test_a_refused_python_download_is_reported_and_nothing_waits_for_it(context, site, console):
    console.allow(r"ERR_BLOCKED_BY_CLIENT|pyodide\.js")  # the browser reports the refused download
    refused = []

    def hold(route) -> None:
        """Keep Pyodide's download waiting until the reader has asked for Python."""
        refused.append(route)

    context.route(re.compile(r"/pyodide\.js$"), hold)
    page = context.new_page()
    page.goto(site)
    page.click("#run-example")
    assert page.text_content("#run-example") == "Waiting for Python to start..."
    deadline = time.monotonic() + 15
    while not refused:  # the worker asks for Pyodide as it starts
        assert time.monotonic() < deadline, "the worker never asked for Pyodide"
        page.wait_for_timeout(50)
    refused[0].abort("blockedbyclient")
    assert "pyodide.js" in failed_to_start(page)
    assert page.text_content("#run-example") == "Run the worked example"
    assert page.is_disabled("#run-example")
    for control in ("#generate", "#simulate", "#compare", "#upload"):
        assert page.is_disabled(control), control
    assert page.locator("#design-table tbody tr").count() == 15, "the worked example drawn at first paint stays"
