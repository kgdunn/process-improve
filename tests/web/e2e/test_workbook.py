"""The workbook from the page: downloaded with the runs and results the page holds, and read back in.

A workbook can come back by upload or by being dropped on the page; either way it fills every
step, and a file that is not one of the page's workbooks is refused in the status line.
"""

from __future__ import annotations

import base64
import json

import numpy as np
import pandas as pd

from tests.web._support import FACTORS, fill_workbook, quadratic
from tests.web.e2e._page import factor_table, run_sheet, settled, status


def test_the_downloaded_workbook_holds_the_page_s_runs_and_results(page, first_paint):
    with page.expect_download() as download:
        page.click("#download")
    assert download.value.suggested_filename == "box_behnken-15-runs.xlsx"
    sheets = pd.read_excel(download.value.path(), sheet_name=None)
    runs = sheets["Runs"]
    np.testing.assert_allclose(runs["y"], first_paint["simulation"]["responses"])
    assert runs["Result source"].tolist() == ["simulated"] * 15
    assert json.loads(sheets["_pi_design"]["spec"].iloc[0])["simulated"] == {
        "scenario": "reactor",
        "sigma": 1.0,
        "seed": 1,
    }


def filled_ccd(api) -> tuple[dict, bytes]:
    """Make a central composite design's workbook, filled in as in Excel except for its first run."""
    design = api.ok("api_make_design", {"design_type": "ccd", "factors": FACTORS})
    return design, base64.b64decode(fill_workbook(design["xlsx_base64"], quadratic, blank_rows=(0,)))


def assert_the_page_holds(page, design: dict) -> None:
    """Check that the page holds the workbook's design, its results, and their analysis."""
    assert page.input_value("#design-type") == "ccd"
    assert [row["name"] for row in factor_table(page)] == [f["name"] for f in FACTORS]
    assert page.text_content("#design-summary p") == f"{design['n_runs']} runs, read from the workbook."
    results = [row[-1] for row in run_sheet(page)]
    assert results[0] == ""
    assert all(results[1:])
    assert page.locator("#design-table input.sim").count() == 0, "results from a workbook are entered"
    assert f"{design['n_runs'] - 1} completed runs." in page.text_content("#analysis")
    assert "Still to run: 1." in page.text_content("#analysis")
    assert page.is_hidden("#demo-banner")
    assert page.is_hidden("#truth"), "nothing is known about the process behind measured results"


def test_an_uploaded_workbook_fills_every_step(page, api, tmp_path):
    design, workbook = filled_ccd(api)
    path = tmp_path / "filled.xlsx"
    path.write_bytes(workbook)
    page.click("#tab-real")
    page.set_input_files("#upload", path)
    settled(page)
    assert_the_page_holds(page, design)


def test_a_workbook_dropped_on_the_page_is_read_as_an_uploaded_one(page, api):
    design, workbook = filled_ccd(api)
    page.click("#tab-real")
    page.evaluate(
        """([name, data]) => {
             const transfer = new DataTransfer();
             transfer.items.add(new File([Uint8Array.from(atob(data), (c) => c.charCodeAt(0))], name));
             const drop = document.querySelector('#drop');
             drop.dispatchEvent(new DragEvent('dragover', { dataTransfer: transfer, bubbles: true, cancelable: true }));
             drop.dispatchEvent(new DragEvent('drop', { dataTransfer: transfer, bubbles: true, cancelable: true }));
           }""",
        ["filled.xlsx", base64.b64encode(workbook).decode()],
    )
    page.wait_for_function("window.__pi.count('api_read_workbook') === 1")
    settled(page)
    assert_the_page_holds(page, design)
    assert "over" not in (page.get_attribute("#drop", "class") or "").split(), "the drop zone is back to rest"


def test_a_file_that_is_not_a_workbook_is_refused(page, tmp_path):
    path = tmp_path / "notes.xlsx"
    path.write_text("not a workbook")
    page.click("#tab-real")
    page.set_input_files("#upload", path)
    settled(page)
    assert status(page) == ("That file could not be read as an .xlsx workbook.", "error")
    assert page.is_visible("#demo-banner"), "the page keeps what it held"
