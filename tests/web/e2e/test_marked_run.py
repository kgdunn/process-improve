"""One run, marked wherever it is shown: on every chart that draws the runs, in the run sheet, in the per-run table.

A point out of line on one chart is a run, and the reader wants that run on the other charts and in the run sheet.
The run the pointer rests on is marked on all of them; a press keeps it marked until it is pressed again, or Escape.
"""

from __future__ import annotations

import pytest

from tests.web.e2e._page import generate, reveal, settled
from tests.web.e2e.test_charts import open_more
from tests.web.e2e.test_figure_window import SHOWN

#: The charts of the worked example that draw a point for each run.
RUN_CHARTS = ["contour", "true_contour", "residuals", "normal_probability", "residuals_order"]

#: The run each chart marks, by the chart's key: null for a chart that draws the runs and marks none.
MARKS = """() => Object.fromEntries([...document.querySelectorAll('#charts figure.chart')].flatMap((card) => {
  const plot = card.querySelector('.plot');
  const runs = (plot.data ?? []).filter((trace) => trace.meta === 'runs');
  if (!runs.length) return [];
  const marked = runs.flatMap((trace) => (trace.selectedpoints ?? []).map((i) => [trace.customdata[i]].flat()[0]));
  return [[card.dataset.chart, marked.length ? marked[0] : null]];
}))"""

#: Whether the chart of the residuals marks a run: every chart marks it by then.
MARKED = f"({MARKS})().residuals !== null"
LET_GO = f"({MARKS})().residuals === null"

TOUCH = {"viewport": {"width": 820, "height": 1100}, "has_touch": True, "is_mobile": True}


def marks(page) -> dict:
    return page.evaluate(MARKS)


def marked_rows(page) -> dict[str, list[str]]:
    """Return the runs of the rows marked in the run sheet and in the table of each run."""
    rows = "(rows) => rows.map((row) => row.dataset.run)"
    return {
        "sheet": page.locator("#design-table tr.run-marked").evaluate_all(rows),
        "each run": page.locator("#diagnostics tr.run-marked").evaluate_all(rows),
    }


def point(page, chart: str, at: int):
    """Return the point drawn ``at`` a place in a chart's trace of the runs, and the run it is."""
    plot = page.locator(f"figure[data-chart={chart}] .plot")
    held = plot.evaluate("(plot, at) => [plot.data.find((t) => t.meta === 'runs').customdata[at]].flat()[0]", at)
    plot.evaluate("(plot) => plot.scrollIntoView({ block: 'center' })")  # clear of the study bar at the window's top
    return plot.locator(".scatterlayer .trace.scatter:has(.point) .point").nth(at), held


def rest_on(page, locator) -> None:
    locator.evaluate("(part) => part.scrollIntoView({ block: 'center' })")
    box = locator.bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)


def test_a_run_rested_on_is_marked_on_every_chart_of_the_runs_and_in_both_tables(page):
    open_more(page)
    assert marks(page) == dict.fromkeys(RUN_CHARTS), "these draw the runs, and none is marked at first"
    spot, run = point(page, "residuals", 4)
    rest_on(page, spot)
    page.wait_for_function(MARKED)
    assert marks(page) == dict.fromkeys(RUN_CHARTS, run), "the same run, on each of them"
    assert marked_rows(page) == {"sheet": [str(run)], "each run": [str(run)]}

    page.mouse.move(2, 2)
    page.wait_for_function(LET_GO)
    assert marks(page) == dict.fromkeys(RUN_CHARTS)
    assert marked_rows(page) == {"sheet": [], "each run": []}


@pytest.mark.parametrize("chart", RUN_CHARTS)
def test_each_chart_of_the_runs_marks_the_run_its_own_point_is(page, chart):
    """A line through the points, or the surface under them, does not take the pointer from the run."""
    open_more(page)
    spot, run = point(page, chart, 3)
    rest_on(page, spot)
    page.wait_for_function(MARKED)
    assert marks(page)[chart] == run
    assert marked_rows(page)["sheet"] == [str(run)]


def test_the_marked_run_is_drawn_larger_in_the_warning_colour_and_the_others_fainter(page):
    open_more(page)
    spot, _ = point(page, "normal_probability", 2)
    rest_on(page, spot)
    page.wait_for_function(MARKED)
    styles = page.locator("figure[data-chart=residuals_order] .plot").evaluate(
        """(plot) => {
             const { selected, unselected, marker } = plot._fullData.find((trace) => trace.meta === 'runs');
             return [selected.marker.color, selected.marker.size, marker.size, unselected.marker.opacity];
           }"""
    )
    warn = page.evaluate("getComputedStyle(document.documentElement).getPropertyValue('--warn').trim()")
    assert styles == [warn, 13, 8, 0.55]


def test_a_press_keeps_the_run_marked_until_it_is_pressed_again(page):
    open_more(page)
    spot, run = point(page, "residuals", 6)
    spot.click(force=True)
    page.wait_for_function(MARKED)
    page.mouse.move(2, 2)
    other, _ = point(page, "residuals", 1)
    rest_on(page, other)
    page.wait_for_timeout(300)
    assert marks(page) == dict.fromkeys(RUN_CHARTS, run), "the pointer on another run leaves the pressed one marked"

    page.wait_for_timeout(800)  # a second press is the reader's own, not the browser's echo of the first
    spot.click(force=True)
    page.mouse.move(2, 2)
    page.wait_for_function(LET_GO)
    assert marked_rows(page) == {"sheet": [], "each run": []}


def test_escape_lets_go_of_a_run_a_press_marked(page):
    open_more(page)
    spot, run = point(page, "residuals_order", 3)
    spot.click(force=True)
    page.wait_for_function(MARKED)
    page.mouse.move(2, 2)
    assert marked_rows(page)["sheet"] == [str(run)]
    page.keyboard.press("Escape")
    page.wait_for_function(LET_GO)
    assert marks(page) == dict.fromkeys(RUN_CHARTS)


def test_a_row_of_the_run_sheet_marks_its_run_on_the_charts(page):
    reveal(page, "#design-table")
    row = page.locator("#design-table tbody tr").nth(5)
    run = int(row.get_attribute("data-run"))
    rest_on(page, row.locator("td").first)
    page.wait_for_function(MARKED)
    assert marks(page) == dict.fromkeys(["contour", "true_contour", "residuals"], run), "on the charts drawn so far"
    assert marked_rows(page)["sheet"] == [str(run)]
    page.mouse.move(2, 2)
    page.wait_for_function(LET_GO)


def test_a_row_of_the_table_of_each_run_is_pressed_to_mark_it_and_stays_marked_as_the_table_is_sorted(page):
    reveal(page, "#diagnostics")
    page.click("#diagnostics > summary")
    assert "press it to keep it marked" in page.text_content("#diagnostics .hint")
    row = page.locator("#diagnostics tbody tr").nth(8)
    run = row.get_attribute("data-run")
    row.locator("td").nth(1).click()
    page.mouse.move(2, 2)
    page.wait_for_function(MARKED)
    assert marked_rows(page) == {"sheet": [run], "each run": [run]}

    page.click("#diagnostics thead .sort >> nth=2")  # by residual: the rows are written again
    assert marked_rows(page)["each run"] == [run], "the run's row, wherever the order puts it"
    page.locator("#diagnostics tr.run-marked td").nth(1).click()
    page.wait_for_function(LET_GO)
    assert marked_rows(page) == {"sheet": [], "each run": []}


def test_a_result_typed_in_the_run_sheet_marks_no_run_for_good(page):
    """A press on a box to type in is the box's own: the run is marked only while the pointer rests on its row."""
    reveal(page, "#design-table")
    page.locator("#design-table tbody tr").nth(2).locator("input").first.click()
    page.mouse.move(2, 2)
    page.wait_for_timeout(300)
    assert marked_rows(page) == {"sheet": [], "each run": []}


def test_a_new_design_lets_go_of_the_run_marked_in_the_one_before(page):
    reveal(page, "#design-table")
    page.locator("#design-table tbody tr").nth(4).locator("td").first.click()
    page.wait_for_function("document.querySelectorAll('#design-table tr.run-marked').length === 1")
    generate(page, design_type="full_factorial")
    settled(page)
    assert page.locator("tr.run-marked").count() == 0
    assert set(marks(page).values()) == {None}


def test_a_figure_s_window_shows_the_run_marked(page):
    open_more(page)
    spot, run = point(page, "residuals", 9)
    spot.click(force=True)
    page.wait_for_function(MARKED)
    page.click("figure[data-chart=normal_probability] .expand")
    page.wait_for_function(SHOWN)
    in_window = """(plot) => {
      const runs = plot.data.find((trace) => trace.meta === 'runs');
      return runs.selectedpoints.map((i) => runs.customdata[i]);
    }"""
    assert page.locator("dialog.figure .plot").evaluate(in_window) == [run]


@pytest.mark.parametrize("context_options", [TOUCH])
def test_under_a_finger_a_tap_on_a_point_marks_its_run_and_a_tap_again_lets_it_go(page):
    """A tap raises Plotly's click twice, for the touch and for the mouse the browser plays after it: one press."""
    open_more(page)
    spot, run = point(page, "residuals", 5)
    spot.tap(force=True)
    page.wait_for_function(MARKED)
    page.wait_for_timeout(800)
    assert marks(page)["normal_probability"] == run, "still marked once the browser has played the mouse"
    assert marked_rows(page)["sheet"] == [str(run)]
    spot.tap(force=True)
    page.wait_for_function(LET_GO)
