"""The steps that cross the worker under Pyodide: a refusal, the workbook out and back in, and the comparison.

Each compares what Pyodide computed in the browser with what native Python computes, or with
what the build computed: the same numbers, to rounding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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
