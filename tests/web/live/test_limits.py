"""Limits on the settings in the built app, read, pictured and proposed by Pyodide in the page's own worker.

The native tier checks the panel against Python answering natively; here the same reader's steps
run against the app as a reader gets it, so the worker's queue, the wheel's solvers under
WebAssembly and the page's drawing are checked together.
"""

from __future__ import annotations

from tests.web._browser import Console
from tests.web._support import FACTORS
from tests.web.e2e._page import candidates_done, corners_done, mark_corners, reveal, run_sheet, set_factors, type_limit
from tests.web.live.test_live import started

HEAT = "T + 25 * P <= 250"


def test_a_limit_typed_proposed_and_kept_to_under_pyodide(own_context, app_url):
    """Typed, read and pictured; a corner marked and its limit proposed; a run sheet made inside both limits."""
    console = Console()
    page = own_context.new_page()
    console.watch(page)
    page.goto(app_url)
    started(page)
    set_factors(page, FACTORS)
    candidates_done(page)
    type_limit(page, "T+25*P<=250")
    row = page.locator("#limit-list .limit").first
    assert row.locator("[name=expression]").input_value() == HEAT
    assert row.locator(".limit-effect").text_content() == "Rules out 13% of your ranges: 2 of the 8 corners."
    assert page.locator("#region-panels .region__pick").count() == 3, "a panel for each pair of the three factors"

    corners_done(page)
    mark_corners(page, 3)
    assert page.text_content("#limit-proposals .proposal__text") == "5 * P + F <= 30"
    page.get_by_role("button", name="Add this limit").click()
    page.wait_for_function("document.querySelectorAll('#limit-list .limit-effect:not(:empty)').length === 2")
    candidates_done(page)

    reveal(page, "#generate")
    page.click("#generate")
    page.wait_for_function("document.querySelectorAll('#design-table tbody tr').length > 0", timeout=120_000)
    for run in run_sheet(page):
        t, p, f = (float(v) for v in run[2:5])
        assert t + 25 * p <= 250 + 1e-9, run
        assert 5 * p + f <= 30 + 1e-9, run
    assert not console.errors, console.errors
