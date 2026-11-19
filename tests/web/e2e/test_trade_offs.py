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

from tests.web._support import FACTORS
from tests.web.e2e._page import (
    candidates_done,
    generate,
    last_reply,
    name_response,
    practise,
    reveal,
    run_sheet,
    saved_studies,
    set_factors,
    settled,
)
from tests.web.e2e.test_rename import rename
from tests.web.e2e.test_responses import add_response, three_responses

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


def test_other_settings_nearly_as_good_are_listed_each_with_a_button_that_chooses_it(traded):
    """The practice study has one best, so Python's reply is given an alternative: the page lists it and chooses it."""
    page = traded
    assert page.locator("#trade-nearly table").count() == 0, "the balance has one best"
    tried = last_reply(page, "api_trade_offs")["result"]
    balance = last_reply(page, "api_balance")["result"]
    other = {
        "settings": {name: values[7] for name, values in tried["candidates"]["settings"].items()},
        "responses": {name: values[7] for name, values in tried["candidates"]["responses"].items()},
        "d": dict.fromkeys(tried["candidates"]["responses"], 0.9),
        "composite": 0.97 * balance["composite"],
    }
    page.evaluate(
        "(reply) => window.__pi.fake('api_trade_offs', reply)",
        {"ok": True, "result": {**tried, "alternatives": [other]}},
    )
    before = page.evaluate("window.__pi.count('api_trade_offs')")
    page.fill('#balance-goals tr[data-response="Cost"] [name="high"]', "70")
    page.wait_for_function(f"window.__pi.count('api_trade_offs') > {before}")
    page.wait_for_selector("#trade-nearly table")
    assert "Nearly as good" in names(page)
    row = page.locator("#trade-nearly tbody tr")
    assert row.locator("td").first.text_content().startswith("T = ")
    row.get_by_role("button", name="Choose the settings of row 1").click()
    page.wait_for_selector("#trade-chosen table")
    assert page.text_content("#trade-chosen .lead") == f"Settings chosen: {row.locator('td').first.text_content()}."


def test_the_settings_chosen_go_when_a_factor_is_renamed(traded):
    """The settings chosen name the factors as they were: a rename lets go of them, and the chart is drawn again."""
    page = traded
    page.evaluate(f"{PLOT}.emit('plotly_click', {{ points: [{{ data: {{ meta: 'tried' }}, customdata: 7 }}] }})")
    page.wait_for_selector("#trade-chosen table")
    before = page.evaluate("window.__pi.count('api_trade_offs')")
    rename(page, 0, "Temperature")
    page.wait_for_function(f"window.__pi.count('api_trade_offs') > {before}")
    settled(page)
    assert page.locator("#trade-chosen table").count() == 0
    assert "Settings chosen" not in names(page)
    assert "Temperature" in last_reply(page, "api_trade_offs")["result"]["candidates"]["settings"]


def test_a_categorical_factor_s_level_is_picked_and_the_chart_is_drawn_at_it(fresh_page):
    """The chart is drawn at the balanced level at first; another level is tried on the models there, with no star."""
    page = fresh_page
    name_response(page, "Yield", "%", goal="maximize")
    add_response(page, "Cost", "$/kg", "minimize", None)
    set_factors(page, [*FACTORS[:2], {"name": "Supplier", "low": "A", "high": "B"}])
    candidates_done(page)
    generate(page, design_type="full_factorial")
    practise(page)
    reveal(page, "#balance")
    page.wait_for_selector("#trade-chart .js-plotly-plot")
    settled(page)
    first = last_reply(page, "api_trade_offs")["result"]
    box = page.locator("#trade-levels select")
    assert box.count() == 1
    balanced = first["held"][0]["value"]
    other = "A" if balanced == "B" else "B"
    assert box.input_value() == balanced
    assert f"{balanced} (balanced)" in box.locator("option").all_text_contents()
    assert "Balanced settings" in names(page)
    asked = page.evaluate("window.__pi.count('api_trade_offs')")
    box.select_option(other)
    page.wait_for_function(f"window.__pi.count('api_trade_offs') > {asked}")
    settled(page)
    again = last_reply(page, "api_trade_offs")["result"]
    assert again["held"] == [{"name": "Supplier", "value": other}]
    assert "Balanced settings" not in names(page), "the balanced settings are at the other level"
    assert page.text_content("#trade-chart figcaption").startswith(f"Drawn with Supplier at {other}")
    assert f"with Supplier at {other}" in page.text_content("#trade-about")
    assert page.locator("#trade-levels select").input_value() == other


def test_the_chart_s_axes_colour_limits_and_choice_are_kept_with_the_study(traded):
    page = traded
    page.select_option("#trade-colour", "Cost")
    before = page.evaluate(POINTS)
    page.fill('#trade-limits tr[data-response="Purity"] [name="low"]', "62")
    page.wait_for_function(f"{POINTS} < {before}")
    page.evaluate(f"{PLOT}.emit('plotly_click', {{ points: [{{ data: {{ meta: 'tried' }}, customdata: 7 }}] }})")
    page.wait_for_selector("#trade-chosen table")
    chosen = page.text_content("#trade-chosen .lead")
    kept = saved_studies(page, lambda studies: bool(studies) and "chosen" in studies[0].get("trade", {}))
    assert kept[0]["v"] == 14
    assert kept[0]["trade"]["limits"] == {"Purity": [62, None]}
    assert kept[0]["trade"]["colour"] == "Cost"

    page.reload()
    page.wait_for_selector("#trade-chart .js-plotly-plot")
    settled(page)
    page.wait_for_selector("#trade-chosen table")
    assert page.input_value("#trade-colour") == "Cost"
    assert page.input_value('#trade-limits tr[data-response="Purity"] [name="low"]') == "62"
    assert page.text_content("#trade-chosen .lead") == chosen, "the same settings, found again among those tried"
