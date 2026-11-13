"""The run sheet in Run: the runs in the order to make them, sortable by any column, each heading with its units.

A sort never loses a result: the results column is read from what the reader typed, and a
run still to do sorts last either way.
"""

from __future__ import annotations

from pathlib import Path

from tests.web.e2e._page import reveal, run_sheet, sort_by, type_results


def sorted_on(page) -> dict[str, str]:
    """Return each column heading's aria-sort."""
    return dict(
        zip(
            page.locator("#design-table th button").all_text_contents(),
            [th.get_attribute("aria-sort") for th in page.locator("#design-table th").all()],
            strict=True,
        )
    )


def test_the_runs_are_listed_in_the_order_to_make_them(page):
    assert [int(row[1]) for row in run_sheet(page)] == list(range(1, 16))
    assert sorted_on(page)["Run order"] == "ascending"


def test_a_column_heading_sorts_the_sheet_up_then_down(page):
    sort_by(page, "T")
    settings = [float(row[2]) for row in run_sheet(page)]
    assert settings == sorted(settings)
    assert sorted_on(page)["T"] == "ascending"
    assert sorted_on(page)["Run order"] == "none"
    sort_by(page, "T")
    assert [float(row[2]) for row in run_sheet(page)] == sorted(settings, reverse=True)
    assert sorted_on(page)["T"] == "descending"


def test_runs_still_to_do_sort_last_either_way(page):
    type_results(page, {0: "", 1: ""})
    for _ in range(2):
        sort_by(page, "Yield")
        results = [row[-1] for row in run_sheet(page)]
        assert results[-2:] == ["", ""]
        assert all(results[:-2])


def test_each_heading_carries_its_units_and_the_response_is_named_as_the_reader_names_it(page):
    units = page.locator("#design-table thead th").evaluate_all(
        "ths => ths.map((th) => th.querySelector('.col-unit')?.textContent ?? '')"
    )
    assert units == ["", "", "degC", "bar", "L/min", "%"]
    page.fill("#response", "Yield of the reactor")
    assert page.locator("#design-table thead th button.sort").last.text_content() == "Yield of the reactor"
    assert page.get_attribute('#design-table input[data-index="0"]', "aria-label") == "Yield of the reactor, run 1"


def test_a_typed_result_survives_a_sort(page):
    type_results(page, {3: "123.5"})
    sort_by(page, "Yield")
    assert run_sheet(page)[-1][-1] == "123.5", "the largest result, sorted last"


def test_the_run_sheet_downloads_as_csv_with_its_results(page):
    """Made in the page, without Python: the columns as listed, a line a run, each result in its response's column."""
    reveal(page, "#csv")
    asked = page.evaluate("window.__pi.calls.length")
    with page.expect_download() as caught:
        page.click("#csv")
    saved = caught.value
    assert saved.suggested_filename == "box_behnken-15-runs.csv"
    lines = Path(saved.path()).read_bytes().decode().split("\r\n")
    assert lines[0] == "Std order,Run order,T,P,F,Yield"
    assert len(lines) == 17, "the headings, fifteen runs, and the last line's end"
    assert all(line.split(",")[-1] for line in lines[1:16]), "every run has its result"
    assert page.evaluate("window.__pi.calls.length") == asked, "Python is not asked"
