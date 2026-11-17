"""What if as a row of dials: each factor's profile over its slider, which a drag, the keyboard and a typed value set.

A profile is the response Learn shows against one factor, the others where the panel has them, with its 95%
confidence band. The line is dashed where the settings lie beyond the runs. A mark stands where the factor is set, at
what the table under the dials predicts. Every profile of the row is drawn to one scale, so a steeper line is a
larger effect, and the scale holds still while the reader drags.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.web._support import FACTORS
from tests.web.e2e._page import generate, practise, reveal, set_factors
from tests.web.e2e.test_responses import three_responses
from tests.web.e2e.test_what_if import rows, set_number

PROFILES = "#what-if-settings svg.profile"
PAD = 10  # app.js PROFILE.pad: the room about a profile's plot

#: Each profile as the page draws it: what it says, its size inside its border, the ends of its scale, its mark, and
#: its line's pieces.
DRAWN = """(profiles) => profiles.map((one) => {
  const mark = one.querySelector('.profile__set');
  const points = (line) => line.getAttribute('points').split(' ').map((point) => point.split(',').map(Number));
  return {
    factor: one.closest('[data-factor]').dataset.factor,
    response: one.dataset.response,
    says: one.getAttribute('aria-label'),
    width: one.clientWidth, height: one.clientHeight,
    ends: [...one.querySelectorAll('.profile__end')].map((end) => end.textContent),
    mark: mark && ['cx', 'cy'].map((at) => Number(mark.getAttribute(at))),
    band: one.querySelectorAll('.profile__band').length,
    pieces: [...one.querySelectorAll('.profile__line')].map((line) => (
      { beyond: line.classList.contains('profile__line--beyond'), points: points(line) })),
    levels: [...one.querySelectorAll('.profile__level')].map((dot) => dot.classList.contains('profile__level--set')),
  };
})"""


def drawn(page: Any, response: str | None = None) -> dict[str, dict]:  # noqa: ANN401 - a Playwright Page
    """Return each factor's profile of a response as the page draws it, by the factor's name: the first response's."""
    every = page.locator(PROFILES).evaluate_all(DRAWN)
    response = response or every[0]["response"]
    return {one["factor"]: one for one in every if one["response"] == response}


def number(text: str) -> float:
    """Read a number the page wrote with its units after it: ``68.25%`` or ``1.5 bar``."""
    return float(text.split(maxsplit=1)[0].rstrip("%"))


def setting(page: Any, factor: str) -> float:  # noqa: ANN401
    """Return what a factor's number box holds."""
    return float(page.input_value(f'#what-if-settings [data-factor="{factor}"] input[name="number"]'))


def press_at(page: Any, factor: str, share: float, to: float | None = None) -> None:  # noqa: ANN401
    """Press on a factor's profile at a share of its plot's width, and drag to another share when ``to`` is given."""
    box = page.locator(f'#what-if-settings [data-factor="{factor}"] svg.profile').bounding_box()
    across = lambda at: box["x"] + 1 + PAD + at * (box["width"] - 2 - 2 * PAD)  # noqa: E731 - inside its 1 px border
    middle = box["y"] + box["height"] / 2
    page.mouse.move(across(share), middle)
    page.mouse.down()
    if to is not None:
        page.mouse.move(across(to), middle, steps=6)
    page.mouse.up()


@pytest.fixture
def what_if(page: Any) -> Any:  # noqa: ANN401
    """Return the worked example with its What if panel on show, at the best settings."""
    reveal(page, "#what-if")
    page.wait_for_selector("#what-if:not([hidden])")
    page.locator("#what-if").scroll_into_view_if_needed()
    return page


def test_each_factor_has_a_profile_of_the_response_on_show(what_if):
    page = what_if
    profiles = drawn(page)
    assert list(profiles) == ["Temperature", "Pressure", "Flow"]
    assert profiles["Temperature"]["says"].startswith(
        "Yield against Temperature, the other factors where they are set: "
    )
    assert " at 150 degC, " in profiles["Temperature"]["says"]
    assert "highest at 68.25% at " in profiles["Flow"]["says"], "the best settings are a turn of the curve over Flow"
    assert all(one["band"] == 1 for one in profiles.values()), "each with its 95% band"
    assert page.text_content("#what-if-about").startswith("Each picture is Yield against one factor")
    # One scale for the row: every profile has the same two ends, and the best prediction lies between them.
    ends = {tuple(one["ends"]) for one in profiles.values()}
    assert len(ends) == 1
    top, bottom = (number(end) for end in ends.pop())
    assert bottom < number(rows(page)[0][1]) < top


def test_a_profile_s_mark_stands_at_the_setting_and_at_the_table_s_prediction(what_if):
    page = what_if
    set_number(page, "Temperature", 180)
    set_number(page, "Flow", 12.5)
    profiles = drawn(page)
    for factor, low, high in (("Temperature", 150, 200), ("Pressure", 1, 3), ("Flow", 10, 20)):
        one = profiles[factor]
        across, up = one["mark"]
        share = (setting(page, factor) - low) / (high - low)
        assert across == pytest.approx(PAD + share * (one["width"] - 2 * PAD), abs=0.6), factor
        top, bottom = (number(end) for end in one["ends"])
        predicted = bottom + (one["height"] - PAD - up) / (one["height"] - 2 * PAD) * (top - bottom)
        assert predicted == pytest.approx(number(rows(page)[0][1]), abs=0.08), f"{factor}: the table's prediction"


def test_a_drag_on_a_profile_sets_its_factor_and_every_profile_follows(what_if):
    page = what_if
    before = drawn(page)
    calls = page.evaluate("window.__pi.count('api_predictors')")
    press_at(page, "Temperature", 0.9, to=0.5)
    assert setting(page, "Temperature") == 175
    assert page.input_value('#what-if-settings [data-factor="Temperature"] input[name="slider"]') == "175"
    after = drawn(page)
    assert after["Temperature"]["mark"][0] == pytest.approx(after["Temperature"]["width"] / 2, abs=0.6)
    # Temperature and Pressure act together, so Pressure's profile is another line now; the table follows too.
    assert after["Pressure"]["pieces"] != before["Pressure"]["pieces"]
    assert rows(page)[0][1] != "68.25%"
    assert page.is_visible("#what-if-back")
    assert page.evaluate("window.__pi.count('api_predictors')") == calls, (
        "in the page's own JavaScript: no call to Python"
    )

    page.click("#what-if-back")
    assert setting(page, "Temperature") == 200
    assert drawn(page)["Pressure"]["pieces"] == before["Pressure"]["pieces"]


def test_a_press_on_a_profile_sets_the_factor_to_the_slider_s_step(what_if):
    page = what_if
    press_at(page, "Temperature", 0.333)
    assert setting(page, "Temperature") == 166.5, "a third of the way from 150 to 200, to the slider's step of 0.5"
    press_at(page, "Pressure", 0.5, to=1.4)  # dragged past the plot's end, and off the picture
    assert setting(page, "Pressure") == 3


def test_the_keyboard_and_a_typed_value_move_the_mark(what_if):
    page = what_if
    slider = page.locator('#what-if-settings [data-factor="Pressure"] input[name="slider"]')
    assert slider.get_attribute("aria-label") == "Pressure, from 1 to 3 bar"
    start = drawn(page)["Pressure"]["mark"][0]
    slider.focus()
    for _ in range(10):
        page.keyboard.press("ArrowRight")
    moved = drawn(page)["Pressure"]["mark"][0]
    assert setting(page, "Pressure") == pytest.approx(1.2)
    assert moved > start
    set_number(page, "Pressure", 2)
    assert drawn(page)["Pressure"]["mark"][0] == pytest.approx(drawn(page)["Pressure"]["width"] / 2, abs=0.6)


def piece_at_the_mark(profile: dict) -> dict:
    """Return the piece of a profile's line that its mark stands on: a solid one when the mark is where two meet."""
    across = profile["mark"][0]
    on = [
        piece for piece in profile["pieces"] if piece["points"][0][0] - 0.05 <= across <= piece["points"][-1][0] + 0.05
    ]
    return min(on, key=lambda piece: piece["beyond"])


def test_the_line_is_dashed_where_the_settings_lie_beyond_the_runs(what_if):
    """The table says whether the settings are within the runs' leverage: the mark is on a dashed piece when not."""
    page = what_if
    assert rows(page)[0][4].split(", ")[1].startswith("above"), "the best settings lie beyond the runs"
    assert all(piece_at_the_mark(one)["beyond"] for one in drawn(page).values())
    assert all("Part of it lies beyond the runs." in one["says"] for one in drawn(page).values())
    for factor, middle in (("Temperature", 175), ("Pressure", 2), ("Flow", 15)):
        set_number(page, factor, middle)
    assert rows(page)[0][4].split(", ")[1].startswith("within"), "the center of the ranges is within them"
    profiles = drawn(page)
    assert not any(piece_at_the_mark(one)["beyond"] for one in profiles.values())
    # A piece beyond the runs starts where the solid one ends, so the line has no gap.
    for one in profiles.values():
        for left, right in zip(one["pieces"], one["pieces"][1:], strict=False):
            assert left["points"][-1] == right["points"][0]
            assert left["beyond"] != right["beyond"]


def test_the_scale_holds_still_while_the_reader_drags_and_fits_again_after(what_if):
    page = what_if
    top, bottom = (number(end) for end in drawn(page)["Temperature"]["ends"])
    box = page.locator('#what-if-settings [data-factor="Pressure"] svg.profile').bounding_box()
    middle = box["y"] + box["height"] / 2
    page.mouse.move(box["x"] + PAD, middle)
    page.mouse.down()
    page.mouse.move(box["x"] + box["width"] - PAD, middle, steps=8)
    held_top, held_bottom = (number(end) for end in drawn(page)["Temperature"]["ends"])
    assert held_top >= top, "the scale only grows while the pointer is down"
    assert held_bottom <= bottom
    page.mouse.up()
    fresh_top, fresh_bottom = (number(end) for end in drawn(page)["Temperature"]["ends"])
    assert setting(page, "Pressure") == 3
    assert fresh_top < top, "at 3 bar the yield is lower everywhere: the scale now fits the profiles as they are"
    assert (fresh_top, fresh_bottom) != (held_top, held_bottom)


def test_a_factor_with_levels_has_a_dot_for_each_and_a_press_chooses_one(blank_page):
    page = blank_page
    catalyst = {"name": "Catalyst", "type": "categorical", "levels": ["Old", "New"]}
    set_factors(page, [FACTORS[0], catalyst])
    generate(page, design_type="full_factorial", n_center_points=0, n_replicates=2)
    practise(page)
    reveal(page, "#what-if")
    page.wait_for_selector("#what-if:not([hidden])")
    page.locator("#what-if").scroll_into_view_if_needed()
    one = drawn(page)["Catalyst"]
    assert len(one["levels"]) == 2
    assert one["says"].startswith("y at each level of Catalyst: ")
    chosen = page.input_value('#what-if-settings [data-factor="Catalyst"] select')
    assert one["levels"] == [chosen == "Old", chosen == "New"], "the level the list holds is the one marked"
    press_at(page, "Catalyst", 0.0 if chosen == "New" else 1.0)
    other = "Old" if chosen == "New" else "New"
    assert page.input_value('#what-if-settings [data-factor="Catalyst"] select') == other
    assert drawn(page)["Catalyst"]["levels"] == [other == "Old", other == "New"]
    page.select_option('#what-if-settings [data-factor="Catalyst"] select', chosen)
    assert drawn(page)["Catalyst"]["levels"] == [chosen == "Old", chosen == "New"], "the list moves the mark too"


def test_with_several_responses_each_has_its_row_of_profiles_on_its_own_scale(fresh_page):
    page = fresh_page
    three_responses(page)
    practise(page)
    reveal(page, "#what-if")
    page.wait_for_selector("#what-if:not([hidden])")
    assert page.locator(PROFILES).count() == 9, "three responses over three factors"
    assert page.text_content("#what-if-about").startswith("Each picture is a response against one factor")
    scales = {}
    for response in ("Yield", "Cost", "Purity"):
        row = drawn(page, response)
        assert list(row) == ["T", "P", "F"]
        assert row["T"]["says"].startswith(f"{response} against T, ")
        ends = {tuple(one["ends"]) for one in row.values()}
        assert len(ends) == 1, f"{response}: one scale for its row"
        scales[response] = ends.pop()
    assert len(set(scales.values())) == 3, "each response has a scale of its own"
    # Each picture says which response it is of; the three of a factor stand one under another, over its slider.
    names = page.locator('#what-if-settings [data-factor="T"] svg.profile .profile__name').all_text_contents()
    assert names == ["Yield", "Cost", "Purity"]
    tops = page.locator('#what-if-settings [data-factor="T"] svg.profile').evaluate_all(
        "(all) => all.map((one) => one.getBoundingClientRect().top)"
    )
    slider = page.locator('#what-if-settings [data-factor="T"] input[name="slider"]').bounding_box()
    assert tops == sorted(tops)
    assert tops[-1] < slider["y"]
    assert len(rows(page)) == 3, "the table under the dials has every response too"

    # A press on one response's picture sets the factor for them all.
    before = [drawn(page, response)["T"]["mark"][0] for response in ("Yield", "Cost", "Purity")]
    was = setting(page, "T")
    costs = page.locator('#what-if-settings [data-factor="T"] svg.profile[data-response="Cost"]')
    costs.evaluate("(one) => one.scrollIntoView({ block: 'center' })")  # clear of the study's bar
    box = costs.bounding_box()
    at = 0.25 if was > 175 else 0.75
    page.mouse.click(box["x"] + box["width"] * at, box["y"] + box["height"] / 2)
    assert setting(page, "T") != was
    marks = [drawn(page, response)["T"]["mark"][0] for response in ("Yield", "Cost", "Purity")]
    assert len(set(marks)) == 1, "one setting, marked on every row"
    assert marks != before, "and every row's mark has moved"


def test_one_response_s_pictures_do_not_name_it(what_if):
    assert what_if.locator("#what-if-settings .profile__name").count() == 0
    assert what_if.locator(PROFILES).first.evaluate("(one) => one.clientHeight") > 120


@pytest.mark.parametrize("context_options", [{"viewport": {"width": 390, "height": 844}, "is_mobile": True}])
def test_on_a_phone_the_dials_are_one_under_another_and_a_swipe_still_scrolls(what_if):
    page = what_if
    profiles = drawn(page)
    tops = page.locator(PROFILES).evaluate_all("(all) => all.map((one) => one.getBoundingClientRect().top)")
    assert tops == sorted(tops)
    assert len(set(tops)) == 3, "one dial a row"
    assert all(one["width"] <= 390 for one in profiles.values())
    assert page.locator(PROFILES).first.evaluate("(one) => getComputedStyle(one).touchAction") == "pan-y"
    assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
    for one in profiles.values():
        assert one["mark"][0] <= one["width"], "drawn to the dial's own width"


PHONE = {"viewport": {"width": 390, "height": 844}, "is_mobile": True, "has_touch": True}


@pytest.mark.parametrize("context_options", [PHONE])
def test_a_finger_that_lands_on_a_profile_sets_nothing_until_it_lifts(what_if):
    """An up or down swipe that starts on a profile is the page's: the browser cancels it, and no factor has moved."""
    page = what_if
    profile = page.locator('#what-if-settings [data-factor="Temperature"] svg.profile')
    profile.evaluate("(one) => one.scrollIntoView({ block: 'center' })")  # clear of the study's bar
    box = profile.bounding_box()
    at = {"x": box["x"] + box["width"] * 0.3, "y": box["y"] + box["height"] / 2}
    touch = page.context.new_cdp_session(page)
    touch.send("Input.dispatchTouchEvent", {"type": "touchStart", "touchPoints": [at]})
    assert setting(page, "Temperature") == 200, "the finger is down, and nothing is set"
    touch.send("Input.dispatchTouchEvent", {"type": "touchCancel", "touchPoints": []})
    assert setting(page, "Temperature") == 200, "a swipe the browser took for the page sets nothing"
    page.touchscreen.tap(at["x"], at["y"])
    page.wait_for_function("Number(document.querySelector('#what-if-settings input[name=number]').value) < 200")
    assert setting(page, "Temperature") == pytest.approx(165, abs=1.5), "a tap sets the factor where it landed"
