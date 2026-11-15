"""What one response costs another, in Learn: the trade-offs between the responses of a balance.

Under the balance of a study of three responses, the page tries settings on the models and draws each as a point
on two responses' axes: paler where the models guess, with a line through the best one response can be for each
value of the other, the runs made, and a star at the balanced settings. The reader picks the axes and a response to
colour the points by, types limits that keep only the settings that meet them, and presses a point to choose its
settings: the what-if panel moves there, and confirmation runs can be added there.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.web.e2e._page import last_reply, practise, reveal, run_sheet, settled
from tests.web.e2e.test_responses import three_responses

PLOT = "document.querySelector('#trade-chart .plot')"
TRIED = f"{PLOT}.data.filter((trace) => trace.meta === 'tried')"
POINTS = f"{TRIED}.reduce((sum, trace) => sum + trace.x.length, 0)"


@pytest.fixture
def traded(fresh_page: Any) -> Any:  # noqa: ANN401 - a Playwright Page
    """Return a study of three responses, its results simulated, in Learn with its trade-offs drawn."""
    three_responses(fresh_page)
    practise(fresh_page)
    reveal(fresh_page, "#balance")
    fresh_page.wait_for_selector("#trade-chart .js-plotly-plot")
    settled(fresh_page)
    return fresh_page


def names(page: Any) -> list[str]:  # noqa: ANN401
    """Return the names of the chart's traces, as its legend lists them."""
    return page.evaluate("document.querySelector('#trade-chart .plot').data.map((trace) => trace.name)")


def test_the_settings_tried_are_drawn_with_the_best_trade_off_the_runs_and_the_balance(traded):
    page = traded
    tried = last_reply(page, "api_trade_offs")["result"]
    assert page.evaluate(POINTS) == len(tried["candidates"]["d"]) == 2000
    assert names(page) == [
        "Tried, far from your runs",
        "Settings tried",
        "Best trade-off",
        "Runs you made",
        "Balanced settings",
    ]
    assert page.input_value("#trade-across") == "Yield"
    assert page.input_value("#trade-up") == "Cost"
    axes = page.evaluate(f"(({{ xaxis, yaxis }}) => [xaxis.title.text, yaxis.title.text])({PLOT}.layout)")
    assert axes == ["Yield (%): more is better", "Cost ($/kg): less is better"]
    runs = page.evaluate(
        "document.querySelector('#trade-chart .plot').data.find((trace) => trace.meta === 'runs').x.length"
    )
    assert runs == len(run_sheet(page))
    assert "2,000 settings tried on the models" in page.text_content("#trade-about")
    assert page.text_content("#trade-count") == "", "no limit is typed"


def test_a_limit_keeps_only_the_settings_that_meet_it_and_the_line_is_drawn_again(traded):
    page = traded
    before = page.evaluate(POINTS)
    page.fill('#trade-limits tr[data-response="Purity"] [name="low"]', "62")
    page.fill('#trade-limits tr[data-response="Purity"] [name="high"]', "64")
    page.wait_for_function(f"{POINTS} < {before}")
    kept = page.evaluate(POINTS)
    assert page.text_content("#trade-count") == f"{kept:,} of the 2,000 settings tried meet your limits."
    page.fill('#trade-limits tr[data-response="Purity"] [name="low"]', "1000")
    page.wait_for_function("document.querySelector('#trade-count').textContent.startsWith('None of the settings')")
    assert page.evaluate(POINTS) == 0


def test_the_reader_picks_the_axes_and_what_colours_the_points(traded):
    page = traded
    page.select_option("#trade-up", "Purity")
    page.wait_for_function("document.querySelector('#trade-chart .plot').layout.yaxis.title.text.startsWith('Purity')")
    assert (
        page.evaluate("document.querySelector('#trade-chart .plot').layout.yaxis.title.text")
        == "Purity (%): nearer 63 is better"
    )
    page.select_option("#trade-across", "Purity")  # the response on the other axis: the two change places
    page.wait_for_function("document.querySelector('#trade-chart .plot').layout.xaxis.title.text.startsWith('Purity')")
    assert [page.input_value("#trade-across"), page.input_value("#trade-up")] == ["Purity", "Yield"]
    page.select_option("#trade-colour", "Cost")
    page.wait_for_function(f"{TRIED}.at(-1).marker.showscale === true")
    assert page.evaluate(f"{TRIED}.at(-1).marker.colorbar.title.text") == "Cost ($/kg)"


def test_a_point_pressed_is_chosen_and_the_what_if_panel_moves_to_its_settings(traded):
    page = traded
    tried = last_reply(page, "api_trade_offs")["result"]["candidates"]
    page.evaluate(f"{PLOT}.emit('plotly_click', {{ points: [{{ data: {{ meta: 'tried' }}, customdata: 7 }}] }})")
    page.wait_for_selector("#trade-chosen table")
    assert "Settings chosen" in names(page)
    said = page.text_content("#trade-chosen .lead")
    assert said.startswith("Settings chosen: T = ")
    what_if = page.evaluate(
        "[...document.querySelectorAll('#what-if-settings input[type=number]')].map((box) => Number(box.value))"
    )
    assert what_if == pytest.approx([tried["settings"][name][7] for name in ("T", "P", "F")], rel=1e-3)
    before = len(run_sheet(page))
    page.click("#trade-chosen button[data-act=confirm]")
    page.wait_for_function(f"document.querySelectorAll('#design-table tbody tr').length === {before + 3}")
    settled(page)
    page.click("#trade-chosen button[data-act=back]")
    assert page.locator("#trade-chosen table").count() == 0
    assert "Settings chosen" not in names(page)
