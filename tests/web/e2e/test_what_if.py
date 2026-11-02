"""What if, in Learn: the reader sets each factor, and every response's model predicts there.

The panel opens at the best settings, or at the balanced settings once two responses or more have a
goal, and predicts there what they show. Moving a setting changes every response at once, in the
page's own JavaScript, without a call to Python; settings past a limit on the settings name it.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from tests.web._support import FACTORS
from tests.web.e2e._page import (
    candidates_done,
    generate,
    last_reply,
    practise,
    reveal,
    set_factors,
    settled,
    type_limit,
)
from tests.web.e2e.test_best_settings import num, table_rows
from tests.web.e2e.test_layout import WIDE_FONT, overflowing
from tests.web.e2e.test_responses import PHONE, three_responses


def rows(page: Any) -> list[list[str]]:  # noqa: ANN401
    """Read the what-if table, a row per response, as the reader sees it."""
    return table_rows(page, "#what-if-out")


def predictions(page: Any) -> int:  # noqa: ANN401
    """Count the calls the page made for the fits it predicts from."""
    return page.evaluate("window.__pi.count('api_predictors')")


def set_number(page: Any, factor: str, value: float) -> None:  # noqa: ANN401
    """Type a setting in a factor's number box and leave it, as a reader would."""
    box = page.locator(f'#what-if-settings [data-factor="{factor}"] input[name="number"]')
    box.fill(str(value))
    box.press("Tab")


def predicted_by_hand(fit: dict, factors: list[dict], settings: dict) -> float:
    """Work out a fit's prediction at numeric settings from the numbers Python sent: a term is its columns' product."""
    middle = {f["name"]: ((f["high"] + f["low"]) / 2, (f["high"] - f["low"]) / 2) for f in factors}
    coded = {name: (settings[name] - centre) / half for name, (centre, half) in middle.items()}
    terms = zip(fit["terms"], fit["coefficients"], strict=True)
    return sum(c * math.prod(coded[p] for p in parts) for parts, c in terms)


@pytest.fixture
def three(fresh_page: Any) -> Any:  # noqa: ANN401 - a Playwright Page
    """Return a study of a yield, a cost and a purity, each with a goal, its results simulated and analysed."""
    three_responses(fresh_page)
    practise(fresh_page)
    return fresh_page


def test_the_worked_example_opens_at_its_best_settings_and_follows_each_move(page, first_paint):
    reveal(page, "#what-if")
    page.wait_for_selector("#what-if:not([hidden])")
    best = first_paint["best"]
    assert page.text_content("#h-what-if") == "What if"
    assert rows(page) == [rows(page)[0]], "one row: the worked example measures one response"
    assert rows(page)[0][:2] == ["Yield", f"{num(best['predicted'])}%"], "the prediction the best settings show"
    shown = {s["name"]: num(s["value"]) for s in best["settings"]}
    boxes = page.evaluate(
        "() => Object.fromEntries([...document.querySelectorAll('#what-if-settings [data-factor]')]"
        ".map((row) => [row.dataset.factor, row.querySelector('input[name=number]').value]))"
    )
    assert boxes == shown
    assert page.is_hidden("#what-if-back"), "at the best settings there is nowhere to come back to"

    before = predictions(page)
    set_number(page, "T", 160)
    moved = {s["name"]: s["value"] for s in best["settings"]} | {"T": 160}
    [fit] = first_paint["predictors"]["responses"]
    expected = predicted_by_hand(fit, first_paint["predictors"]["factors"], moved)
    assert rows(page)[0][1] == f"{num(expected)}%"
    assert predictions(page) == before, "the page predicts in its own JavaScript, without asking Python"
    assert page.text_content("#what-if-back") == "Back to the best settings"

    page.click("#what-if-back")
    assert rows(page)[0][1] == f"{num(best['predicted'])}%"
    assert page.is_hidden("#what-if-back")
    assert "new_points=at" in page.text_content("#py-what-if code"), "the code that makes the same predictions"


def test_a_number_past_its_range_is_held_at_the_end_of_it(page, first_paint):
    reveal(page, "#what-if")
    page.wait_for_selector("#what-if:not([hidden])")
    set_number(page, "T", 260)
    box = page.locator('#what-if-settings [data-factor="T"] input[name="number"]')
    assert box.input_value() == "200", "the models hold inside the range they were fitted over"


def test_with_three_responses_the_panel_opens_at_the_balanced_settings(three):
    page = three
    reveal(page, "#what-if")
    page.wait_for_selector("#what-if:not([hidden])")
    balanced = last_reply(page, "api_balance")["result"]
    table = rows(page)
    assert [row[0] for row in table] == ["Yield", "Cost", "Purity"]
    units = {"Yield": "%", "Cost": " $/kg", "Purity": "%"}
    assert [row[1] for row in table] == [f"{num(r['predicted'])}{units[r['name']]}" for r in balanced["responses"]]
    set_number(page, "P", 2.5)
    assert page.text_content("#what-if-back") == "Back to the balanced settings"


def test_settings_past_a_limit_name_it(fresh_page):
    page = fresh_page
    set_factors(page, FACTORS)
    candidates_done(page)
    type_limit(page, "T + 25 * P <= 215")
    generate(page, design_type="d_optimal")
    practise(page)
    reveal(page, "#what-if")
    page.wait_for_selector("#what-if:not([hidden])")
    before = page.evaluate("window.__pi.count('api_outside_limits')")
    set_number(page, "T", 200)
    set_number(page, "P", 3)
    page.wait_for_function(f"window.__pi.count('api_outside_limits') > {before}")
    settled(page)
    limit = last_reply(page, "api_outside_limits")["result"]["broken"][0]["expression"]
    warnings = page.locator("#what-if-out ul.warnings li").all_text_contents()
    assert f"These settings break your limit 1, {limit}." in warnings


@pytest.mark.parametrize("context_options", [PHONE])
def test_on_a_phone_the_panel_fits(page):
    page.add_style_tag(content=WIDE_FONT)
    reveal(page, "#what-if")
    page.wait_for_selector("#what-if:not([hidden])")
    assert overflowing(page) == []
