"""The run sheet in Run: the runs in the order to make them, sortable by any column, each heading with its units.

A sort never loses a result: the results column is read from what the reader typed, and a
run still to do sorts last either way.
"""

from __future__ import annotations

from pathlib import Path

from tests.web.e2e._page import open_question, reveal, run_sheet, sort_by, type_results


def sorted_on(page) -> dict[str, str]:
    """Return each column heading's aria-sort."""
    return dict(
        zip(
            page.locator("#design-table th button").all_text_contents(),
            [th.get_attribute("aria-sort") for th in page.locator("#design-table th[aria-sort]").all()],
            strict=True,
        )
    )


def test_the_runs_are_listed_in_the_order_to_make_them(page):
    assert [int(row[1]) for row in run_sheet(page)] == list(range(1, 16))
    assert sorted_on(page)["Run order"] == "ascending"


def test_a_column_heading_sorts_the_sheet_up_then_down(page):
    sort_by(page, "Temperature")
    settings = [float(row[2]) for row in run_sheet(page)]
    assert settings == sorted(settings)
    assert sorted_on(page)["Temperature"] == "ascending"
    assert sorted_on(page)["Run order"] == "none"
    sort_by(page, "Temperature")
    assert [float(row[2]) for row in run_sheet(page)] == sorted(settings, reverse=True)
    assert sorted_on(page)["Temperature"] == "descending"


def test_runs_still_to_do_sort_last_either_way(page):
    type_results(page, {0: "", 1: ""})
    for _ in range(2):
        sort_by(page, "Yield")
        results = [row[-1] for row in run_sheet(page)]
        assert results[-2:] == ["", ""]
        assert all(results[:-2])


def test_each_heading_carries_its_units_and_the_response_is_named_as_the_reader_names_it(page):
    units = page.locator("#design-table thead th[aria-sort]").evaluate_all(
        "ths => ths.map((th) => th.querySelector('.col-unit')?.textContent ?? '')"
    )
    assert units == ["", "", "degC", "bar", "L/min", "%"]
    open_question(page)
    page.fill("#response", "Yield of the reactor")
    assert page.locator("#design-table thead th button.sort").last.text_content() == "Yield of the reactor"
    assert page.get_attribute('#design-table input[data-index="0"]', "aria-label") == "Yield of the reactor, run 1"


#: The run sheet's head under and over its names: the column groups, and each factor's range and response's goal.
HEAD = """(t) => [...t.tHead.rows].filter((tr) => !tr.querySelector('th[aria-sort]'))
  .map((tr) => [...tr.cells].map((cell) => [cell.textContent, cell.colSpan]))"""

#: Each run's place on the sheet by its run order, whether its row is picked out as the best so far, and the results
#: marked as missing their goal, by their run order.
MARKS = """(t) => ({
  best: [...t.querySelectorAll('tr.run-best')].map((tr) => tr.dataset.run),
  misses: [...t.querySelectorAll('td.misses input')].map((box) => Number(box.value)),
  legend: !document.querySelector('#sheet-legend').hidden,
})"""


def test_the_head_groups_the_columns_and_writes_each_range_and_goal_in_words(page):
    groups, goals = page.eval_on_selector("#design-table", HEAD)
    assert groups == [["", 2], ["Factors: what we set", 3], ["Responses", 1]], "the orders stand apart, under no group"
    assert goals == [
        ["Range or goal", 1],
        ["", 1],
        ["150 to 200", 1],
        ["1 to 3", 1],
        ["10 to 20", 1],
        ["highest", 1],
    ], "the units are under the names; Yield is maximized, with no window typed"


def test_a_result_outside_its_window_is_marked_and_the_best_run_is_picked_out(page):
    yields = [float(row[-1]) for row in run_sheet(page)]
    marks = page.eval_on_selector("#design-table", MARKS)
    assert marks["misses"] == [], "with no window, nothing misses"
    assert not marks["legend"]
    [best] = marks["best"]
    assert float(run_sheet(page)[int(best) - 1][-1]) == max(yields), "the run with the highest Yield, its goal"
    assert page.text_content("#design-table tr.run-best .best-so-far").strip() == "best so far"

    reveal(page, "#window")
    page.fill('#window-limits tr[data-response="Yield"] [name="low"]', "60")
    page.wait_for_function(
        "document.querySelector('#design-table .sheet-goals td[data-response]').textContent === 'at least 60'"
    )
    marks = page.eval_on_selector("#design-table", MARKS)
    assert sorted(marks["misses"]) == sorted(y for y in yields if y < 60), "each result under the window, and no other"
    assert marks["legend"]
    reveal(page, "#design-table")
    missed = page.locator("#design-table td.misses").first
    assert missed.locator(".visually-hidden").text_content() == " misses its goal", "said to a screen reader too"


#: Where the What-if rows sit in the run sheet: whether they come before the runs, and the left edge of each cell of
#: the What-if row and of each heading over the runs, by the column's name.
PLACED = """(t) => {
  const body = t.querySelector('#what-if-settings');
  const left = (cell) => Math.round(cell.getBoundingClientRect().left);
  const heads = [...t.querySelectorAll('thead th[aria-sort]')];
  const row = t.querySelector('#what-if-settings tr.what-if-row');
  return {
    first: body.compareDocumentPosition(t.querySelector('tr[data-run]')) === Node.DOCUMENT_POSITION_FOLLOWING,
    heads: Object.fromEntries(heads.map((th) => [th.querySelector('button.sort').textContent, left(th)])),
    cells: Object.fromEntries([...row.querySelectorAll('[data-factor], [data-response]')]
      .map((cell) => [cell.dataset.factor ?? cell.dataset.response, left(cell)])),
    predicted: row.querySelector('[data-response]').textContent,
  };
}"""


def test_what_if_sits_over_the_runs_in_their_columns_whatever_the_sort(page):
    reveal(page, "#design-table")
    page.wait_for_selector("#what-if-settings .goal-bar")
    placed = page.eval_on_selector("#design-table", PLACED)
    assert placed["first"], "the What-if rows come before the runs"
    assert placed["cells"] == {name: placed["heads"][name] for name in ("Temperature", "Pressure", "Flow", "Yield")}
    assert placed["predicted"] == page.text_content("#what-if-settings .what-if-value strong").rstrip("%")
    sort_by(page, "Yield")
    assert page.eval_on_selector("#design-table", PLACED)["first"], "and stay on top when a heading sorts the runs"


def test_learn_points_to_the_what_if_row_over_the_runs(page):
    reveal(page, "#where")
    assert page.is_visible("#what-if-link"), "Where to run has the way to it"
    page.click("#to-what-if")
    assert page.get_attribute("#workspace", "data-phase") == "run"
    page.wait_for_function(
        "(() => { const box = document.querySelector('#what-if-settings').getBoundingClientRect();"
        " return box.top < innerHeight && box.bottom > 0; })()"
    )


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
    assert lines[0] == "Std order,Run order,Temperature,Pressure,Flow,Yield"
    assert len(lines) == 17, "the headings, fifteen runs, and the last line's end"
    assert all(line.split(",")[-1] for line in lines[1:16]), "every run has its result"
    assert page.evaluate("window.__pi.calls.length") == asked, "Python is not asked"
