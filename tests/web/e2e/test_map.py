"""The map beside the dials of What if: the response over two factors, which a press or a drag sets together.

The map is the fitted surface of the response Learn shows, with the other factors where the dials have them. It marks
where the dials are, stars the best settings when it passes through them, and fades where the settings lie beyond the
runs. The dials and the map always agree: a drag on the map moves two dials, and a dial moves the mark or, for a
factor the map is not drawn over, the surface.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.web._support import FACTORS
from tests.web.e2e._page import generate, practise, reveal, set_factors, settled
from tests.web.e2e.test_profiles import setting
from tests.web.e2e.test_what_if import rows, set_number

PLOT = "#what-if-map-plot"

#: The map as Plotly holds it: its surface, where it fades, each mark by its name, and where its axes are drawn.
MAP = """() => { const plot = document.querySelector('#what-if-map-plot');
  const marks = Object.fromEntries(plot.data.filter((trace) => trace.type === 'scatter' && trace.x[0] !== null)
    .map((trace) => [trace.name, [trace.x[0], trace.y[0]]]));
  const drawn = (axis) => ({ title: axis.title.text, range: axis.range, offset: axis._offset, length: axis._length });
  return { z: plot.data[0].z, x: plot.data[0].x, y: plot.data[0].y, marks,
    beyond: plot.data.find((trace) => trace.meta === 'beyond')?.z ?? null,
    axes: [plot._fullLayout.xaxis, plot._fullLayout.yaxis].map(drawn) }; }"""

#: Whether the map marks the dials where the number boxes have them: a box holds its setting to four figures.
AGREES = """() => { const plot = document.querySelector('#what-if-map-plot');
  const set = plot.data?.at(-1);
  if (!set || !plot._fullLayout) return false;
  const boxes = Object.fromEntries([...document.querySelectorAll('#what-if-settings td.what-if-factor')]
    .map((row) => [row.dataset.factor, Number(row.querySelector('input[name=number]')?.value)]));
  return [plot._fullLayout.xaxis, plot._fullLayout.yaxis].every((axis, i) => {
    const name = Object.keys(boxes).find((factor) => axis.title.text.startsWith(factor));
    const marked = [set.x[0], set.y[0]][i];
    return name && Math.abs(marked - boxes[name]) <= 1e-3 * Math.max(1, Math.abs(boxes[name])); }); }"""


def shown_map(page: Any) -> dict:  # noqa: ANN401 - a Playwright Page
    """Return the map as it is drawn, once it shows where the dials are now."""
    page.wait_for_function(AGREES)
    return page.evaluate(MAP)


def point_at(page: Any, x: float, y: float) -> tuple[float, float]:  # noqa: ANN401
    """Return where on the screen the map draws the settings ``x`` and ``y`` of its two factors."""
    drawn = shown_map(page)
    box = page.locator(PLOT).bounding_box()
    across, up = drawn["axes"]
    share = lambda axis, value: (value - axis["range"][0]) / (axis["range"][1] - axis["range"][0])  # noqa: E731
    return (
        box["x"] + across["offset"] + share(across, x) * across["length"],
        box["y"] + up["offset"] + (1 - share(up, y)) * up["length"],
    )


@pytest.fixture
def mapped(page: Any) -> Any:  # noqa: ANN401
    """Return the worked example with What if on show and its map drawn, at the best settings."""
    reveal(page, "#what-if")
    page.wait_for_selector(f"#what-if-map:not([hidden]) {PLOT}.js-plotly-plot")
    page.locator("#what-if-map").scroll_into_view_if_needed()
    return page


@pytest.mark.parametrize("context_options", [{"viewport": {"width": 1440, "height": 900}}])
def test_the_map_is_under_the_run_sheet_over_two_factors(mapped):
    page = mapped
    assert page.text_content("#what-if-map-title") == "Yield over Temperature and Pressure"
    sheet = page.locator("#design-table").bounding_box()
    chart = page.locator("#what-if-map").bounding_box()
    assert sheet["y"] + sheet["height"] <= chart["y"], "under the run sheet, whose What-if row it sets"
    drawn = shown_map(page)
    assert [axis["title"] for axis in drawn["axes"]] == ["Temperature (degC)", "Pressure (bar)"]
    assert [axis["range"] for axis in drawn["axes"]] == [[150, 200], [1, 3]]


def test_the_map_marks_where_the_dials_are_and_stars_the_best_settings(mapped):
    page = mapped
    drawn = shown_map(page)
    assert drawn["marks"]["The What-if row"] == [200, 1], "the panel opens at the best settings"
    assert drawn["marks"]["The best settings"] == [200, 1]
    # The surface under the mark is the table's prediction: the map's corner there is the grid's own point.
    assert f"{drawn['z'][0][-1]:.4g}%" == rows(page)[0][1]

    set_number(page, "Temperature", 160)
    assert shown_map(page)["marks"]["The What-if row"] == [160, 1], "a dial moves the mark"
    assert shown_map(page)["z"] == drawn["z"], "and not the surface, which is drawn over that factor"

    # Flow is not an axis of the map: moving it gives another surface, which no longer passes through the best.
    set_number(page, "Flow", 12)
    moved = shown_map(page)
    assert moved["z"] != drawn["z"]
    assert "The best settings" not in moved["marks"]


def test_a_press_and_a_drag_on_the_map_set_both_dials(mapped):
    page = mapped
    calls = page.evaluate("window.__pi.count('api_predictors')")
    page.mouse.click(*point_at(page, 175, 2))
    assert (setting(page, "Temperature"), setting(page, "Pressure")) == (175, 2)
    assert setting(page, "Flow") == pytest.approx(18.04), "the third dial is where it was"
    assert shown_map(page)["marks"]["The What-if row"] == [175, 2]

    page.mouse.move(*point_at(page, 175, 2))
    page.mouse.down()
    page.mouse.move(*point_at(page, 160.2, 2.61), steps=6)
    page.mouse.up()
    assert setting(page, "Temperature") == pytest.approx(160, abs=0.5), "to the slider's step of 0.5"
    assert setting(page, "Pressure") == pytest.approx(2.61, abs=0.011), "and of 0.02: 2.60 or 2.62, a pixel decides"
    assert shown_map(page)["marks"]["The What-if row"] == [setting(page, "Temperature"), setting(page, "Pressure")]
    assert page.evaluate("window.__pi.count('api_predictors')") == calls, "in the page's own JavaScript"

    page.mouse.move(*point_at(page, 175, 2))
    page.mouse.down()
    box = page.locator(PLOT).bounding_box()
    page.mouse.move(box["x"] + box["width"] + 40, box["y"] - 40, steps=4)  # past the map's top right corner
    page.mouse.up()
    assert (setting(page, "Temperature"), setting(page, "Pressure")) == (200, 3), "held at the ends of the ranges"


def test_the_map_fades_where_the_table_says_the_settings_lie_beyond_the_runs(mapped):
    page = mapped
    drawn = shown_map(page)
    assert drawn["beyond"] is not None, "the best settings, in a corner, lie beyond the runs"
    assert drawn["beyond"][0][-1] == 1
    assert rows(page)[0][4].split(", ")[1].startswith("above")
    for factor, middle in (("Temperature", 175), ("Pressure", 2), ("Flow", 15)):
        set_number(page, factor, middle)
    centred = shown_map(page)
    assert rows(page)[0][4].split(", ")[1].startswith("within")
    middle = len(centred["x"]) // 2
    assert centred["beyond"] is None or centred["beyond"][middle][middle] == 0, "clear where the dials are"


def test_the_map_is_over_the_two_factors_the_surface_chart_was_given(mapped):
    page = mapped
    reveal(page, "#contour-factor-2")
    page.select_option("#contour-factor-2", "Flow")
    settled(page)
    page.wait_for_function(
        "document.querySelector('#what-if-map-title').textContent === 'Yield over Temperature and Flow'"
    )
    drawn = shown_map(page)
    assert [axis["title"] for axis in drawn["axes"]] == ["Temperature (degC)", "Flow (L/min)"]
    assert drawn["marks"]["The What-if row"] == pytest.approx(
        [setting(page, "Temperature"), setting(page, "Flow")], rel=1e-3
    )


def test_a_study_with_one_numeric_factor_has_dials_and_no_map(blank_page):
    page = blank_page
    catalyst = {"name": "Catalyst", "type": "categorical", "levels": ["Old", "New"]}
    set_factors(page, [FACTORS[0], catalyst])
    generate(page, design_type="full_factorial", n_center_points=0, n_replicates=2)
    practise(page)
    reveal(page, "#what-if")
    page.wait_for_selector("#what-if:not([hidden])")
    settled(page)
    assert page.locator("#what-if-settings svg.profile").count() == 2
    assert page.is_hidden("#what-if-map")


@pytest.mark.parametrize(
    "context_options", [{"viewport": {"width": 390, "height": 844}, "is_mobile": True, "has_touch": True}]
)
def test_on_a_phone_the_map_is_under_the_dials_and_a_tap_sets_both(mapped):
    page = mapped
    dials = page.locator("#what-if-settings").bounding_box()
    chart = page.locator("#what-if-map").bounding_box()
    assert dials["y"] + dials["height"] <= chart["y"], "the dials, then the map"
    assert chart["x"] + chart["width"] <= 390
    assert page.locator(PLOT).evaluate("(plot) => getComputedStyle(plot).touchAction") == "pan-y"
    page.locator(PLOT).evaluate("(plot) => plot.scrollIntoView({ block: 'center' })")  # clear of the study's bar
    page.touchscreen.tap(*point_at(page, 175, 2))
    page.wait_for_function("Number(document.querySelector('#what-if-settings input[name=number]').value) === 175")
    assert (setting(page, "Temperature"), setting(page, "Pressure")) == (175, 2)
    assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
