"""The steps that cross the worker under Pyodide: a refusal, the workbook out and back in, and the comparison.

Each compares what Pyodide computed in the browser with what native Python computes, or with
what the build computed: the same numbers, to rounding.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tests.web._browser import BOOT_TIMEOUT, Console
from tests.web.e2e._page import last_reply, run_sheet, settled, status


def test_a_refusal_from_python_reaches_the_status_line(page):
    name = page.locator("#factors tbody tr").first.locator("[name=name]")
    name.fill("a b")
    try:
        page.click("#generate")
        settled(page)
        assert status(page) == (
            "Factor names must be unique and look like variable names (letters, digits, _).",
            "error",
        )
    finally:
        name.fill("T")


def test_the_workbook_goes_out_and_comes_back_in_under_pyodide(page, demo, tmp_path):
    with page.expect_download() as download:
        page.click("#download")
    path = tmp_path / download.value.suggested_filename
    download.value.save_as(path)
    runs = pd.read_excel(path, sheet_name="Runs")
    np.testing.assert_allclose(runs["y"], [float(row[-1]) for row in run_sheet(page)])

    page.click("#tab-real")
    page.set_input_files("#upload", path)
    page.wait_for_function("window.__pi.count('api_read_workbook') === 1")
    settled(page, timeout=120_000)
    read = last_reply(page, "api_read_workbook")["result"]
    assert read["responses"] == pytest.approx(runs["y"].tolist())
    assert "15 completed runs." in page.text_content("#analysis")


def test_the_comparison_under_pyodide_scores_as_the_build_did(page, demo):
    page.click("#compare")
    page.wait_for_function("window.__pi.count('api_compare') === 1 && window.__pi.idle()", timeout=300_000)
    settled(page)
    live = last_reply(page, "api_compare")["result"]
    built = demo["comparison"]["result"]
    assert live["designs"] == built["designs"]
    for label in built["designs"]:
        for key, value in built["values"][label].items():
            if isinstance(value, float):
                assert live["values"][label][key] == pytest.approx(value, rel=1e-6, abs=1e-9), (label, key)


@pytest.mark.parametrize("fn", ["_reply", "make_design", "json", "api_nothing"])
def test_the_worker_answers_only_the_page_s_entry_points(page, fn):
    """Nothing in bootstrap.py's namespace but its api_* entry points can be called from the page."""
    assert page.evaluate("(fn) => window.__pi.ask(fn, {})", fn) == {
        "ok": False,
        "error": f"{json.dumps(fn)} is not one of the page's entry points.",
    }



def started(page) -> None:
    """Wait until Python has started in the page, or failed to."""
    page.wait_for_function(
        "/^(Ready|Could not|Analysis complete)/.test(document.querySelector('#status .status__text').textContent)",
        timeout=BOOT_TIMEOUT,
    )
    assert not status(page)[0].startswith("Could not"), status(page)[0]


def test_a_design_comes_back_and_is_analysed_again_under_pyodide(own_context, app_url):
    """IndexedDB keeps the design and its results; the page opened again analyses them in its own Pyodide.

    A browser of its own, so the module's shared page keeps the worked example the other tests read.
    """
    console = Console()
    first = own_context.new_page()
    console.watch(first)
    first.goto(app_url)
    started(first)
    first.select_option("#design-type", "full_factorial")
    first.fill("#n_center_points", "2")
    first.click("#generate")
    settled(first, timeout=120_000)
    first.click("#simulate")
    first.wait_for_function("document.querySelector('#library-count').textContent === '1'", timeout=60_000)
    settled(first, timeout=120_000)
    coefficients = first.text_content("#analysis table")

    again = own_context.new_page()
    console.watch(again)
    again.goto(app_url)
    again.wait_for_selector("#campaign-banner", state="visible")
    started(again)
    again.wait_for_function("window.__pi.count('api_analyze_runs') >= 1 && window.__pi.idle()", timeout=120_000)
    settled(again, timeout=120_000)
    assert again.text_content("#campaign-banner").startswith("Picked up where you left off: Full factorial")
    assert run_sheet(again) == run_sheet(first)
    assert again.text_content("#analysis table") == coefficients
    assert not console.errors, console.errors
