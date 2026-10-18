"""Results pasted into the run sheet from a spreadsheet, as it copies them: a line a row, a tab between cells.

A column goes in place, down the sheet as it is shown from the box pasted into, and a block of several
columns across the responses from it. A block whose first column holds the run numbers, or whose
headings name the Run order and the responses, is matched to the runs by number, whatever its order.
A block that does not fit is not pasted at all, and the status line says why.
"""

from __future__ import annotations

from typing import Any

from tests.web.e2e._page import generate, headers, reveal, run_sheet, settled, sort_by, status

#: Pasted into a box as a reader's paste reaches it: a paste event carrying the text a spreadsheet copied.
PASTE = """(box, text) => {
    const data = new DataTransfer();
    data.setData("text/plain", text);
    box.focus();
    box.dispatchEvent(new ClipboardEvent("paste", { clipboardData: data, bubbles: true, cancelable: true }));
}"""


def paste(page: Any, text: str, row: int = 0) -> None:  # noqa: ANN401 - a Playwright Page
    """Paste ``text`` into the result box of the run sheet's ``row``-th row as shown, and let the page settle."""
    reveal(page, "#design-table")
    page.locator("#design-table tbody tr").nth(row).locator("input[data-response]").first.evaluate(PASTE, text)
    settled(page)


def column(page: Any, name: str) -> list[str]:  # noqa: ANN401
    """Read a column of the run sheet as shown: each cell's text, or the result in its box."""
    at = headers(page).index(name)
    return [row[at] for row in run_sheet(page)]


def test_a_column_goes_down_the_sheet_as_shown_and_is_analysed(blank_page):
    page = blank_page
    generate(page, design_type="box_behnken")
    results = [f"{60 + i / 10:g}" for i in range(15)]
    paste(page, "\n".join(results) + "\n")
    assert column(page, "y") == results
    assert status(page) == ("Pasted 15 results of y, down the run sheet from run 1.", "ok")
    assert page.text_content("#badge-run") == "15 of 15"
    page.wait_for_function("window.__pi.count('api_analyze_runs') === 1")  # a moment after the paste, as for typing
    settled(page)
    sent = page.evaluate("window.__pi.calls.findLast((c) => c.fn === 'api_analyze_runs').payload")
    assert sent["responses"] == [float(r) for r in results]


def test_a_sorted_sheet_takes_a_column_in_the_order_it_shows(blank_page):
    page = blank_page
    generate(page, design_type="box_behnken")
    sort_by(page, "Std order")
    paste(page, "1\n2\n3", row=2)
    standard = sorted(run_sheet(page), key=lambda r: int(r[0]))
    assert [r[-1] for r in standard[:5]] == ["", "", "1", "2", "3"], "from the third row shown, down"


def test_a_spreadsheet_s_whole_sheet_with_headings_is_matched_to_the_runs_by_run_order(blank_page):
    """Copied with its headings, the factors and Std order along, and in another order: each result finds its run."""
    page = blank_page
    generate(page, design_type="box_behnken")
    sheet = run_sheet(page)
    heads = headers(page)
    by_std = sorted(sheet, key=lambda r: int(r[heads.index("Std order")]))
    lines = ["\t".join([*heads[:-1], "y"])]
    lines += ["\t".join([*r[:-1], str(100 + int(r[heads.index("Run order")]))]) for r in by_std]
    paste(page, "\n".join(lines), row=7)
    runs = column(page, "Run order")
    assert column(page, "y") == [str(100 + int(run)) for run in runs]
    assert status(page)[0] == "Pasted 15 results of y, matched to the runs by the Run order column."


def test_a_block_that_does_not_fit_is_not_pasted_and_the_status_says_why(blank_page):
    page = blank_page
    generate(page, design_type="box_behnken")
    paste(page, "\n".join(["61"] * 16))
    assert status(page) == (
        "You pasted 16 lines, and the run sheet shows 15 runs from the one pasted into down. Nothing was pasted.",
        "error",
    )
    assert column(page, "y") == [""] * 15


def test_one_value_written_with_a_decimal_comma_goes_in_its_box(blank_page):
    page = blank_page
    generate(page, design_type="box_behnken")
    before = status(page)
    paste(page, "61,5", row=4)
    assert column(page, "y")[4] == "61.5"
    assert status(page) == before, "a box's own paste needs no word"
