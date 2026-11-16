"""A prediction as 20 runs, in What if: what a reader might get at the settings the dials hold.

Under the table of What if, each fitted response's prediction is drawn as 20 runs, each as likely as the next, piled
up where a run is likelier. The least and the greatest of them are the two ends of the 95% prediction interval the
table gives, so the drawing and the table cannot disagree. A run inside the response's window is filled and one
outside it is a ring, so the reader sees how many of 20 runs the window would pass.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.web._support import FACTORS
from tests.web.e2e._page import generate, practise, reveal, set_factors, settled
from tests.web.e2e.test_responses import three_responses
from tests.web.e2e.test_what_if import predictions, rows, set_number

RUNS = "#what-if-runs figure"

#: Each response's runs as the page draws them: what the words say, the two numbers under the line, how many dots
#: there are and how many are filled, whether a window is shaded, and whether a screen reader is read the drawing.
DRAWN = """(figures) => figures.map((figure) => ({
  response: figure.dataset.response,
  says: figure.querySelector('figcaption').textContent,
  ends: [...figure.querySelectorAll('.runs__end')].map((end) => end.textContent),
  dots: figure.querySelectorAll('.runs__run').length,
  filled: figure.querySelectorAll('.runs__run--inside').length,
  window: figure.querySelectorAll('.runs__window').length,
  hidden: figure.querySelector('svg').getAttribute('aria-hidden'),
  right: figure.getBoundingClientRect().right,
}))"""


def drawn(page: Any) -> list[dict]:  # noqa: ANN401 - a Playwright Page
    """Return each response's runs as the page draws them, in the page's order."""
    return page.locator(RUNS).evaluate_all(DRAWN)


@pytest.fixture
def shown(page: Any) -> Any:  # noqa: ANN401
    """Return the worked example with What if on show."""
    reveal(page, "#what-if")
    page.wait_for_selector("#what-if-runs:not([hidden]) figure")
    return page


def test_the_runs_reach_as_far_as_the_prediction_interval_beside_them(shown):
    page = shown
    [row] = rows(page)
    [runs] = drawn(page)
    interval = row[3]  # the 95% prediction interval, as the table writes it: "61.2 to 68.4%"
    assert runs["says"] == f"Yield as 20 runs you might get: {interval}."
    assert runs["ends"] == [end.rstrip("%") for end in interval.split(" to ")], "the least and the greatest, under them"
    assert runs["dots"] == 20
    assert runs["filled"] == 20, "with no window typed, no run is outside one"
    assert runs["window"] == 0
    assert runs["hidden"] == "true", "the words say what the drawing shows"


def test_the_runs_follow_the_dials_without_a_call_to_python(shown):
    page = shown
    before = drawn(page)[0]["says"]
    calls = predictions(page)
    set_number(page, "Temperature", 160)
    [runs] = drawn(page)
    assert runs["says"] != before
    assert runs["says"] == f"Yield as 20 runs you might get: {rows(page)[0][3]}."
    assert predictions(page) == calls, "in the page's own JavaScript"


def test_a_run_inside_the_response_s_window_is_filled_and_one_outside_is_a_ring(fresh_page):
    page = fresh_page
    three_responses(page)
    practise(page)
    reveal(page, "#what-if")
    page.wait_for_selector("#what-if-runs:not([hidden]) figure")
    table = rows(page)
    assert [one["response"] for one in drawn(page)] == [row[0] for row in table], "a drawing for each response"
    assert all(
        one["says"].startswith(f"{row[0]} as 20 runs you might get: ")
        for one, row in zip(drawn(page), table, strict=True)
    )

    # A window that starts at the prediction passes the half of the runs above it: the runs are spread evenly about it.
    predicted = table[0][1].rstrip("%")
    reveal(page, "#window")
    page.fill('#window-limits tr[data-response="Yield"] [name="low"]', predicted)
    page.wait_for_function("document.querySelector('#what-if-runs figure .runs__window') !== null")
    first, *others = drawn(page)
    assert first["filled"] in (10, 11), "ten above the prediction, and one more when the page rounded it down"
    assert first["dots"] == 20
    assert first["says"].endswith(f" {first['filled']} of them are inside your window.")
    assert all(one["filled"] == 20 and one["window"] == 0 for one in others), "a response with no window loses none"

    page.click("#window-clear")
    page.wait_for_function("document.querySelector('#what-if-runs figure .runs__window') === null")
    assert drawn(page)[0]["filled"] == 20
    assert "window" not in drawn(page)[0]["says"]


def test_a_model_with_no_run_left_to_estimate_the_noise_has_no_runs_to_draw(blank_page):
    page = blank_page
    set_factors(page, FACTORS[:2])
    generate(page, design_type="full_factorial", n_center_points=0)
    practise(page)
    reveal(page, "#model")
    page.select_option("#model", "interactions")  # four terms from four runs
    settled(page)
    reveal(page, "#what-if")
    page.wait_for_selector("#what-if-out table")
    assert rows(page)[0][3] == "none", "no prediction interval"
    assert page.is_hidden("#what-if-runs")


@pytest.mark.parametrize("context_options", [{"viewport": {"width": 390, "height": 844}, "is_mobile": True}])
def test_on_a_phone_the_runs_fit_the_screen(shown):
    page = shown
    assert drawn(page)[0]["right"] <= 390
    assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
