"""The settings that meet a specification, in Learn: a window for each response, and how surely a new run meets it.

The reader types a window for a response, or for several. The page says over what share of the settings the models
predict every response inside its window, at what share one new run meets every window with a chance of 50% and of
90% or more, and where it is surest to. A chart shades the settings that meet the windows over two factors, the
others where the what-if panel has them, with a line for each chance, the runs made and a star at those settings.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.web._support import FACTORS
from tests.web.e2e._page import generate, last_reply, practise, reveal, run_sheet, set_factors, settled
from tests.web.e2e.test_responses import three_responses

PLOT = "document.querySelector('#window-chart .plot')"
ANSWERED = "window.__pi.replies.filter((r) => r.fn === 'api_specification').length"


@pytest.fixture
def studied(fresh_page: Any) -> Any:  # noqa: ANN401 - a Playwright Page
    """Return a study of three responses, its results simulated, in Learn at the section of the windows."""
    three_responses(fresh_page)
    practise(fresh_page)
    reveal(fresh_page, "#window")
    settled(fresh_page)
    return fresh_page


def window(page: Any, response: str, end: str, value: str) -> dict:  # noqa: ANN401
    """Type an end of a response's window, wait for the page to judge it, and return what Python answered."""
    before = page.evaluate(ANSWERED)
    page.fill(f'#window-limits tr[data-response="{response}"] [name="{end}"]', value)
    page.wait_for_function(f"{ANSWERED} > {before}")
    page.wait_for_selector("#window-out .lead")
    return last_reply(page, "api_specification")["result"]


def names(page: Any) -> list[str]:  # noqa: ANN401
    """Return the names of the chart's traces that its legend lists."""
    return page.evaluate(f"{PLOT}.data.filter((trace) => trace.showlegend !== false).map((trace) => trace.name)")


def test_a_window_typed_says_what_share_of_the_settings_meets_it_and_how_surely(studied):
    page = studied
    assert page.locator("#window-out .lead").count() == 0, "nothing is said before a window is typed"
    found = window(page, "Yield", "low", "58")
    said = [p.text_content() for p in page.locator("#window-out p").all()]
    assert said[0].startswith("By the models, ")
    assert said[0].endswith(" of the settings inside your ranges give the response inside its window.")
    assert said[1].startswith("One new run meets its window with a chance of 90% or more at ")
    assert said[2].startswith("A new run is surest to meet its window at T = ")
    chance = found["surest"]["probability"]
    assert said[2].endswith(f": a chance of {'more than 99' if chance >= 0.995 else round(100 * chance)}%.")
    page.wait_for_selector("#window-chart .js-plotly-plot")
    assert names(page) == [
        "Every response inside its window",
        "Yield: 58% and above",
        "50% chance a new run meets every window",
        "90% chance a new run meets every window",
        "Runs you made",
        "What-if settings",
    ]
    assert "Shaded where every response is predicted inside its window" in page.text_content("#window-chart figcaption")
    code = page.text_content("#py-window code")
    assert 'windows = {"Yield": (58, None)}' in code
    assert "judged = feasible_region(models, windows, " in code


def test_windows_on_two_responses_are_met_together_and_cleared(studied):
    page = studied
    alone = window(page, "Yield", "low", "58")
    both = window(page, "Cost", "high", "62")
    assert both["share"] < alone["share"], "a second window leaves fewer settings"
    assert [r["name"] for r in both["responses"]] == ["Yield", "Cost"]
    assert page.text_content("#window-out .lead").endswith("give every response inside every window.")
    assert names(page)[1:3] == ["Yield: 58% and above", "Cost: 62 $/kg and below"]
    page.click("#window-clear")
    page.wait_for_function("document.querySelector('#window-out').childElementCount === 0")
    assert page.input_value('#window-limits tr[data-response="Yield"] [name="low"]') == ""
    assert page.locator("#window-chart").is_hidden()
    assert page.locator("#py-window").is_hidden()


def test_the_chart_is_drawn_through_the_what_if_settings_and_follows_them(studied):
    page = studied
    found = window(page, "Yield", "low", "58")
    held = found["overlay"]["held"]
    assert [h["name"] for h in held] == ["F"], "the chart is over T and P, the two factors that matter most"
    before = page.evaluate(ANSWERED)
    reveal(page, "#what-if-settings")  # the What-if row, over the runs in Run
    box = page.locator('#what-if-settings [data-factor="F"] input[type=number]')
    box.fill("12")
    page.wait_for_function(f"{ANSWERED} > {before}")
    reveal(page, "#window-chart")
    moved = last_reply(page, "api_specification")["result"]
    assert moved["overlay"]["held"] == [{"name": "F", "value": 12}]
    assert moved["share"] == pytest.approx(found["share"]), "the share is over every setting, wherever What if is"
    page.wait_for_function("document.querySelector('#window-chart figcaption').textContent.includes('F at 12 L/min')")
    page.select_option("#window-up", "F")
    page.wait_for_function(f"{PLOT}.layout.yaxis.title.text.startsWith('F')")
    assert [page.input_value("#window-across"), page.input_value("#window-up")] == ["T", "F"]


def test_the_surest_settings_go_to_what_if_and_are_confirmed(studied):
    page = studied
    surest = window(page, "Yield", "low", "58")["surest"]
    page.click("#window-what-if")
    what_if = page.evaluate(
        "[...document.querySelectorAll('#what-if-settings input[type=number]')].map((box) => Number(box.value))"
    )
    assert what_if == pytest.approx([surest["settings"][name] for name in ("T", "P", "F")], rel=1e-3)
    assert page.get_attribute("#workspace", "data-phase") == "run", "the What-if row is over the runs, in Run"
    before = len(run_sheet(page))
    reveal(page, "#window-confirm")
    page.click("#window-confirm")
    page.wait_for_function(f"document.querySelectorAll('#design-table tr[data-run]').length === {before + 3}")
    settled(page)
    assert page.locator("#window-confirm").is_hidden(), "the runs are there: their results are what is wanted"
    assert page.text_content("#window-out .lead").startswith("By the models, ")


def test_a_window_refused_and_then_emptied_leaves_nothing_behind(studied):
    """A window upside down is refused in a sentence; with both ends emptied the sentence goes, and nothing is asked."""
    page = studied
    yields = '#window-limits tr[data-response="Yield"]'
    page.fill(f'{yields} [name="low"]', "70")
    before = page.evaluate(ANSWERED)
    page.fill(f'{yields} [name="high"]', "60")
    page.wait_for_function(f"{ANSWERED} > {before}")
    page.wait_for_selector("#window-problems:not([hidden])")
    assert page.text_content("#window-problems") == "The window of Yield must run from a lower value to a higher one."
    page.fill(f'{yields} [name="low"]', "")
    page.fill(f'{yields} [name="high"]', "")
    page.wait_for_selector("#window-problems", state="hidden")
    assert page.locator("#window-clear").is_hidden()
    assert page.locator("#window-out .lead").count() == 0


def test_a_study_with_a_categorical_factor_says_what_the_window_gives_at_each_level(fresh_page):
    """A level is chosen, not moved to: beside the level What if has, each other level has its share and its chance."""
    page = fresh_page
    set_factors(page, [*FACTORS[:2], {"name": "Supplier", "low": "A", "high": "B"}])
    generate(page, design_type="full_factorial", n_center_points=0, n_replicates=2)
    practise(page)
    page.wait_for_selector("#window-limits tbody tr", state="attached")  # the section comes once What if has its fits
    reveal(page, "#window")
    settled(page)
    low = page.evaluate("Number(document.querySelector('#window-limits tr td:last-child').textContent.split(' ')[0])")
    response = page.get_attribute("#window-limits tbody tr", "data-response")
    found = window(page, response, "low", str(low + 1))
    assert page.text_content("#window-out h4") == "At each level"
    heads = page.locator("#window-out table thead th").all_text_contents()
    assert [head.strip() for head in heads] == ["Supplier", "Settings that meet it", "Highest chance", "Under What if"]
    rows = [row.locator("td").all_text_contents() for row in page.locator("#window-out table tbody tr").all()]
    assert [row[0] for row in rows] == ["A", "B"]
    here = found["held"][0]["value"]
    assert {row[0]: row[3] for row in rows} == {here: "yes", "B" if here == "A" else "A": ""}
    assert [row["share"] for row in found["levels"] if row["here"]] == [found["share"]]
