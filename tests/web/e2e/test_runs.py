"""A prediction as 20 runs, in What if: what a reader might get at the settings the What-if row holds.

Each fitted response's prediction is drawn against its goal: its 95% prediction interval as a line, and above it 20
runs, each as likely as the next. The least and the greatest of them are the two ends of that interval, so the runs
and the interval cannot disagree. A run inside the response's goal is filled and one outside it is a ring, so the
reader sees how many of 20 runs the goal would pass.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.web._support import FACTORS
from tests.web.e2e._page import generate, practise, reveal, set_factors, settled
from tests.web.e2e.test_responses import three_responses
from tests.web.e2e.test_what_if import predictions, rows, set_number

RUNS = "#what-if-settings .goal-bar"

#: Each response against its goal as the page draws it: what the drawing says, the interval written beside it, how
#: many runs there are and how many are filled, whether a goal is shaded, and how far right it reaches.
DRAWN = """(bars) => bars.map((bar) => ({
  response: bar.closest('.what-if-goal').dataset.response,
  says: bar.getAttribute('aria-label'),
  interval: bar.closest('.what-if-goal').querySelector('.what-if-interval').textContent,
  dots: bar.querySelectorAll('.goal-bar__run').length,
  filled: bar.querySelectorAll('.goal-bar__run--inside').length,
  band: bar.querySelectorAll('.goal-bar__band').length,
  right: bar.getBoundingClientRect().right,
}))"""


def drawn(page: Any) -> list[dict]:  # noqa: ANN401 - a Playwright Page
    """Return each response against its goal as the page draws it, in the page's order."""
    return page.locator(RUNS).evaluate_all(DRAWN)


@pytest.fixture
def shown(page: Any) -> Any:  # noqa: ANN401
    """Return the worked example with What if on show."""
    reveal(page, "#what-if")
    page.wait_for_selector("#what-if:not([hidden]) .goal-bar")
    return page


def test_the_runs_reach_as_far_as_the_prediction_interval_beside_them(shown):
    page = shown
    [row] = rows(page)
    [runs] = drawn(page)
    interval = row[3]  # the 95% prediction interval, as the table writes it: "61.2 to 68.4%"
    assert f" Yield as 20 runs you might get: {interval}." in runs["says"]
    assert runs["interval"] == f"({interval.rstrip('%')})", "the least and the greatest, beside the prediction"
    assert runs["dots"] == 20
    assert runs["filled"] == 20, "with no goal to judge them by, no run is outside one"
    assert runs["band"] == 0
    assert "higher is better" in runs["says"], "the worked example maximizes Yield, with no window typed"


def test_the_runs_follow_the_dials_without_a_call_to_python(shown):
    page = shown
    before = drawn(page)[0]["says"]
    calls = predictions(page)
    set_number(page, "Temperature", 160)
    [runs] = drawn(page)
    assert runs["says"] != before
    assert f" Yield as 20 runs you might get: {rows(page)[0][3]}." in runs["says"]
    assert predictions(page) == calls, "in the page's own JavaScript"


def test_a_run_inside_the_response_s_goal_is_filled_and_one_outside_is_a_ring(fresh_page):
    page = fresh_page
    three_responses(page)
    practise(page)
    reveal(page, "#what-if")
    page.wait_for_selector("#what-if:not([hidden]) .goal-bar")
    table = rows(page)
    assert [one["response"] for one in drawn(page)] == [row[0] for row in table], "a drawing for each response"
    assert all(
        f" {row[0]} as 20 runs you might get: " in one["says"] for one, row in zip(drawn(page), table, strict=True)
    )
    assert all(one["band"] == 1 for one in drawn(page)), "each has its acceptable range in the balance as its goal"
    assert all(" of them are inside its goal." in one["says"] for one in drawn(page))

    # A window that starts at the prediction passes the half of the runs above it: the runs are spread evenly about it.
    predicted = table[0][1].rstrip("%")
    reveal(page, "#window")
    page.fill('#window-limits tr[data-response="Yield"] [name="low"]', predicted)
    page.wait_for_function(
        "document.querySelector('#what-if-settings .goal-bar')?.getAttribute('aria-label').includes('your window')"
    )
    first = drawn(page)[0]
    assert first["filled"] in (10, 11), "ten above the prediction, and one more when the page rounded it down"
    assert first["dots"] == 20
    assert first["says"].endswith(f" {first['filled']} of them are inside your window.")

    page.click("#window-clear")
    page.wait_for_function(
        "!document.querySelector('#what-if-settings .goal-bar').getAttribute('aria-label').includes('your window')"
    )
    assert "inside its goal" in drawn(page)[0]["says"], "the balance's range is its goal again"


def test_a_model_with_no_run_left_to_estimate_the_noise_has_no_runs_to_draw(blank_page):
    page = blank_page
    set_factors(page, FACTORS[:2])
    generate(page, design_type="full_factorial", n_center_points=0)
    practise(page)
    reveal(page, "#model")
    page.select_option("#model", "interactions")  # four terms from four runs
    settled(page)
    reveal(page, "#what-if")
    page.wait_for_selector("#what-if-out table", state="attached")
    assert rows(page)[0][3] == "none", "no prediction interval"
    [one] = drawn(page)
    assert one["dots"] == 0, "no runs to draw"
    assert "with no interval: no run is left to estimate the noise" in one["says"]


@pytest.mark.parametrize("context_options", [{"viewport": {"width": 390, "height": 844}, "is_mobile": True}])
def test_on_a_phone_the_runs_fit_the_screen(shown):
    page = shown
    assert drawn(page)[0]["right"] <= 390
    assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
