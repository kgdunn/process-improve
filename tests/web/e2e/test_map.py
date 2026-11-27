"""The map beside the dials of What if: the response over two factors, which a press or a drag sets together.

The map is the fitted surface of the response Learn shows, with the other factors where the dials have them. It marks
where the dials are, stars the best settings when it passes through them, and fades where the settings lie beyond the
runs. The dials and the map always agree: a drag on the map moves two dials, and a dial moves the mark or, for a
factor the map is not drawn over, the surface.

The map can be detached from the page: it then floats where the reader puts it, at the size they give it, and is put
back at a press. The browser keeps where it floated.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.web._support import FACTORS
from tests.web.e2e._page import first_paint_done, generate, overflowing, practise, reveal, set_factors, settled
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


#: Where the map, the run sheet's box and the What-if row are in the window: each as its left, top, right and bottom.
PLACES = """() => Object.fromEntries(Object.entries(
  { map: '#what-if-map', sheet: '.sheet-and-map > .scroll', row: '#what-if-settings tr.what-if-row' })
  .map(([name, part]) => { const box = document.querySelector(part).getBoundingClientRect();
    return [name, { left: box.left, top: box.top, right: box.right, bottom: box.bottom }]; }))"""


@pytest.mark.parametrize(
    "context_options",
    [{"viewport": {"width": 1440, "height": 900}}, {"viewport": {"width": 1200, "height": 800}}],
    ids=["a wide laptop", "a laptop"],
)
def test_on_a_wide_screen_the_map_is_beside_the_run_sheet_in_view_with_the_what_if_row(mapped):
    page = mapped
    assert page.text_content("#what-if-map-title") == "Yield over Temperature and Pressure"
    high = page.viewport_size["height"]
    page.locator("#what-if-settings tr.what-if-row").evaluate("(row) => row.scrollIntoView({ block: 'center' })")
    at = page.evaluate(PLACES)
    assert at["sheet"]["right"] <= at["map"]["left"], "side by side"
    for part in ("map", "row"):
        assert at[part]["top"] >= 0, f"the {part} is in view: a drag on the map and the row it sets, together"
        assert at[part]["bottom"] <= high
    sheet = page.locator(".sheet-and-map > .scroll")
    assert sheet.evaluate("(box) => box.scrollWidth <= box.clientWidth"), "and the run sheet still fits its box"
    drawn = shown_map(page)
    assert [axis["title"] for axis in drawn["axes"]] == ["Temperature (degC)", "Pressure (bar)"]
    assert [axis["range"] for axis in drawn["axes"]] == [[150, 200], [1, 3]]

    # The runs scroll past, and the map stays under the study's bar while the run sheet is in view.
    page.locator("#design-table tr[data-run]").nth(10).evaluate("(row) => row.scrollIntoView({ block: 'center' })")
    later = page.evaluate(PLACES)
    assert later["row"]["top"] < at["row"]["top"], "the What-if row has gone up with the page"
    bar = page.locator("#studybar").bounding_box()
    assert bar["y"] + bar["height"] <= later["map"]["top"] <= 100, "the map is held under the bar"
    assert later["map"]["bottom"] <= high


@pytest.mark.parametrize("context_options", [{"viewport": {"width": 1000, "height": 800}}])
def test_on_a_narrower_screen_the_map_is_under_the_run_sheet(mapped):
    at = mapped.evaluate(PLACES)
    assert at["sheet"]["bottom"] <= at["map"]["top"], "under the run sheet, whose What-if row it sets"
    assert at["map"]["left"] == at["sheet"]["left"]


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


# --------------------------------------------------------------------------- the map, detached from the page
DETACH, MOVE, GRIP = "#what-if-map-detach", "#what-if-map-move", "#what-if-map-grip"
STEP = 24  # app.js MAP_STEP: how far a press of an arrow key moves or resizes the floating map

#: The map's box in the window and whether it floats there, the run sheet's width, the room its chart has and the
#: size the chart is drawn at, the foot of the study's bar, and what the browser keeps of it.
FLOATS = """() => { const map = document.querySelector('#what-if-map');
  const plot = document.querySelector('#what-if-map-plot');
  const box = map.getBoundingClientRect();
  return { floats: getComputedStyle(map).position === 'fixed',
    box: { left: box.left, top: box.top, width: box.width, height: box.height },
    sheet: document.querySelector('.sheet-and-map > .scroll').getBoundingClientRect().width,
    room: [plot.clientWidth, plot.clientHeight],
    drawn: [Math.round(plot._fullLayout.width), Math.round(plot._fullLayout.height)],
    bar: document.querySelector('#studybar').getBoundingClientRect().bottom,
    kept: JSON.parse(localStorage.getItem('factorial-map')) }; }"""

LAPTOP = pytest.mark.parametrize("context_options", [{"viewport": {"width": 1440, "height": 900}}])


def floats(page: Any) -> dict:  # noqa: ANN401
    """Return where the map is, once its chart is drawn to the room it has."""
    page.wait_for_function(
        "(() => { const plot = document.querySelector('#what-if-map-plot'); return plot._fullLayout"
        " && Math.round(plot._fullLayout.width) === plot.clientWidth"
        " && Math.round(plot._fullLayout.height) === plot.clientHeight; })()"
    )
    return page.evaluate(FLOATS)


def drag(page: Any, part: str, across: float, down: float) -> None:  # noqa: ANN401
    """Drag a part of the map by its middle, ``across`` and ``down`` pixels."""
    box = page.locator(part).bounding_box()
    x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    page.mouse.move(x, y)
    page.mouse.down()
    page.mouse.move(x + across, y + down, steps=4)
    page.mouse.up()


@pytest.fixture
def detached(mapped: Any) -> Any:  # noqa: ANN401
    """Return the worked example with its map detached where it stood, beside the run sheet."""
    mapped.locator("#what-if-settings tr.what-if-row").evaluate("(row) => row.scrollIntoView({ block: 'center' })")
    mapped.click(DETACH)
    return mapped


@LAPTOP
def test_the_map_detaches_where_it_stands_and_is_put_back_at_a_press(mapped):
    page = mapped
    page.locator("#what-if-settings tr.what-if-row").evaluate("(row) => row.scrollIntoView({ block: 'center' })")
    docked = floats(page)
    assert not docked["floats"]
    assert docked["kept"] is None
    assert page.is_hidden(MOVE), "a map in its place has nothing to move it by"
    assert page.is_hidden(GRIP)
    assert page.text_content(DETACH) == "Detach"

    page.click(DETACH)
    now = floats(page)
    assert now["floats"]
    assert now["box"] == pytest.approx(docked["box"], abs=0.5), "it floats where it stood, to a whole pixel"
    assert now["sheet"] > docked["sheet"], "and the run sheet has the page's whole width"
    assert (page.text_content(DETACH), page.is_visible(MOVE), page.is_visible(GRIP)) == ("Put back", True, True)
    assert page.get_attribute(DETACH, "aria-label") == "Put the map back in its place on the page"
    assert now["kept"] == {**now["box"], "floating": True}

    # It stays where it is as the page scrolls, and a press on it still sets both factors.
    page.evaluate("scrollBy(0, 300)")
    assert floats(page)["box"] == now["box"]
    page.mouse.click(*point_at(page, 175, 2))
    assert (setting(page, "Temperature"), setting(page, "Pressure")) == (175, 2)

    page.click(DETACH)
    back = floats(page)
    assert not back["floats"]
    assert back["sheet"] == docked["sheet"]
    assert (page.text_content(DETACH), page.is_hidden(MOVE), page.is_hidden(GRIP)) == ("Detach", True, True)
    assert back["kept"] == {**now["box"], "floating": False}, "its box is kept for the next press of Detach"
    assert back["room"][1] == 300, "in its place it is as tall as it was"


@LAPTOP
def test_a_detached_map_is_moved_by_its_head_and_resized_by_its_corner(detached):
    page = detached
    start = floats(page)["box"]
    drag(page, "#what-if-map-title", -400, 120)
    moved = floats(page)
    assert moved["box"] == {**start, "left": start["left"] - 400, "top": start["top"] + 120}
    drag(page, MOVE, 60, -40)
    assert floats(page)["box"] == {**start, "left": start["left"] - 340, "top": start["top"] + 80}, "or by Move"

    drag(page, GRIP, 150, 90)
    sized = floats(page)
    assert (sized["box"]["width"], sized["box"]["height"]) == (start["width"] + 150, start["height"] + 90)
    assert (sized["box"]["left"], sized["box"]["top"]) == (start["left"] - 340, start["top"] + 80), "its corner alone"
    assert sized["drawn"] == sized["room"], "the chart is drawn again to the room it has"
    assert sized["room"][1] > moved["room"][1]
    assert sized["kept"] == {**sized["box"], "floating": True}
    page.mouse.click(*point_at(page, 160, 2.5))
    assert (setting(page, "Temperature"), setting(page, "Pressure")) == (160, 2.5), "and still sets both factors"


@LAPTOP
def test_a_detached_map_stays_inside_the_window_and_under_the_bar(detached):
    page = detached
    drag(page, "#what-if-map-title", -3000, -3000)
    held = floats(page)
    assert (held["box"]["left"], held["box"]["top"]) == (0, round(held["bar"])), "under the study's bar, never over it"
    drag(page, GRIP, 3000, 3000)
    full = floats(page)["box"]
    assert (full["left"] + full["width"], full["top"] + full["height"]) == (1440, 900), "as large as the window"
    drag(page, GRIP, -3000, -3000)
    small = floats(page)["box"]
    assert (small["width"], small["height"]) == (260, 260), "and no smaller than it can be read at"
    drag(page, "#what-if-map-title", 3000, 3000)
    cornered = floats(page)["box"]
    assert (cornered["left"], cornered["top"]) == (1440 - 260, 900 - 260)

    page.set_viewport_size({"width": 1000, "height": 700})
    inside = floats(page)["box"]
    foot = (inside["left"] + inside["width"], inside["top"] + inside["height"])
    assert foot == (1000, 700), "a smaller window keeps it"


@LAPTOP
def test_the_arrow_keys_move_and_resize_a_detached_map(detached):
    page = detached
    start = floats(page)["box"]
    page.focus(MOVE)
    for key in ("ArrowLeft", "ArrowLeft", "ArrowDown"):
        page.keyboard.press(key)
    assert floats(page)["box"] == {**start, "left": start["left"] - 2 * STEP, "top": start["top"] + STEP}
    scrolled = page.evaluate("scrollY")
    page.focus(GRIP)
    for key in ("ArrowRight", "ArrowUp"):
        page.keyboard.press(key)
    sized = floats(page)
    assert (sized["box"]["width"], sized["box"]["height"]) == (start["width"] + STEP, start["height"] - STEP)
    assert page.evaluate("scrollY") == scrolled, "the arrows are the map's: the page is not scrolled"
    assert sized["kept"] == {**sized["box"], "floating": True}
    for part, label in ((MOVE, "Move the map: drag it, or press the arrow keys"), (GRIP, "Resize the map: ")):
        assert page.get_attribute(part, "aria-label").startswith(label)


@LAPTOP
def test_a_map_left_floating_floats_there_on_the_next_visit(detached):
    page = detached
    drag(page, "#what-if-map-title", -300, 60)
    drag(page, GRIP, 100, 40)
    left = floats(page)["box"]
    for _ in range(2):  # opened again in Learn, where the map is out of sight, then shown in Run
        page.reload()
        first_paint_done(page)
        reveal(page, "#design-table")
        page.wait_for_selector(f"#what-if-map:not([hidden]) {PLOT}.js-plotly-plot")
        again = floats(page)
        assert again["floats"]
        assert again["box"] == left
        assert again["drawn"] == again["room"]
    page.click(DETACH)
    page.reload()
    first_paint_done(page)
    reveal(page, "#design-table")
    page.wait_for_selector(f"#what-if-map:not([hidden]) {PLOT}.js-plotly-plot")
    assert not floats(page)["floats"], "put back, it opens in its place"
    page.click(DETACH)
    assert floats(page)["box"] == left, "and Detach floats it where it last floated"


@pytest.mark.parametrize("context_options", [{"viewport": {"width": 390, "height": 844}, "is_mobile": True}])
def test_on_a_phone_a_detached_map_fits_the_window(mapped):
    page = mapped
    page.click(DETACH)
    box = floats(page)["box"]
    assert box["left"] >= 0
    assert box["left"] + box["width"] <= 390
    assert overflowing(page) == []
    for part in (DETACH, MOVE):
        side = page.locator(part).bounding_box()
        assert min(side["width"], side["height"]) >= 24, "large enough for a finger"
