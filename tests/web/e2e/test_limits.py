"""Limits on the settings: typed under the factors, read by Python, and kept to by every design the page makes.

A limit is typed in the factors' own units. Python reads it as the library will, says under it what
it rules out, and gives back its canonical text when the reader leaves the box; a limit that cannot be
read says why, and holds the brief back. With any limit, only the D-optimal and I-optimal designs are
offered, since only they keep every run inside, and the run sheet's runs all meet the limits.
"""

from __future__ import annotations

from typing import Any

from tests.web._support import FACTORS
from tests.web.e2e._page import (
    calls,
    candidates_done,
    generate,
    reveal,
    run_sheet,
    saved_studies,
    set_factors,
    settled,
    type_limit,
)

HEAT = "T + 25 * P <= 250"


def limit_row(page: Any, n: int = 0) -> Any:  # noqa: ANN401
    return page.locator("#limit-list .limit").nth(n)


def test_a_typed_limit_says_what_it_rules_out_and_takes_python_s_text(blank_page):
    page = blank_page
    type_limit(page, "T+25*P<=250", note="the product chars")
    row = limit_row(page)
    assert row.locator("[name=expression]").input_value() == HEAT, "the canonical text, once the reader leaves the box"
    assert row.locator(".limit-effect").text_content() == "Rules out 13% of your ranges: 2 of the 8 corners."
    assert page.text_content("#limits-summary").startswith("The limits leave 87% of your ranges to run.")
    assert page.text_content("#limits-count") == "1"
    assert row.locator("[name=high]").input_value() == "250", "a straight limit's bound, to tune without retyping"


def test_with_a_limit_only_the_designs_that_keep_to_it_are_offered(blank_page):
    page = blank_page
    type_limit(page, HEAT)
    candidates_done(page)
    listed = page.locator("#candidates tbody tr").evaluate_all("(rows) => rows.map((r) => r.dataset.design)")
    assert sorted(listed) == ["d_optimal", "i_optimal"]
    reveal(page, "#design-type")
    disabled = page.locator("#design-type option").evaluate_all(
        "(options) => options.filter((o) => o.disabled).map((o) => o.value)"
    )
    assert "box_behnken" in disabled
    assert "d_optimal" not in disabled
    assert calls(page, "api_candidates")[-1]["payload"]["constraints"] == [{"expression": HEAT}]


def test_every_run_of_the_run_sheet_meets_the_limit(blank_page):
    page = blank_page
    type_limit(page, HEAT)
    candidates_done(page)
    generate(page)
    reveal(page, "#design-table")
    sheet = run_sheet(page)
    assert sheet
    for run in sheet:
        t, p = float(run[2]), float(run[3])
        assert t + 25 * p <= 250 + 1e-9, run
    reveal(page, "#design-summary")
    assert "Every run is inside your limits on the settings." in page.text_content("#design-summary")


def test_a_limit_that_cannot_be_read_says_why_and_holds_the_brief_back(blank_page):
    page = blank_page
    type_limit(page, "T + Z <= 300")
    row = limit_row(page)
    message = "The limit 'T + Z <= 300' names Z, which is not one of your factors: T, P and F."
    assert row.locator(".limit-problem").text_content() == message
    assert row.locator("[name=expression]").get_attribute("aria-invalid") is not None
    reveal(page, "#generate")
    assert page.locator("#generate").is_disabled()
    assert page.text_content("#candidates") == message


def test_a_limit_typed_with_a_phone_s_signs_is_read_as_typed(blank_page):
    page = blank_page
    type_limit(page, "T − 25·P ≥ 100")
    assert limit_row(page).locator("[name=expression]").input_value() == "T - 25 * P >= 100"


def test_renaming_a_factor_renames_it_inside_the_limits_and_keeps_the_run_sheet(blank_page):
    page = blank_page
    type_limit(page, HEAT)
    candidates_done(page)
    generate(page)
    reveal(page, "#factors")
    box = page.locator("#factors tbody tr").nth(0).locator("[name=name]")
    box.fill("Temp")
    box.press("Tab")
    settled(page)
    reveal(page, "#limits")
    assert limit_row(page).locator("[name=expression]").input_value() == "Temp + 25 * P <= 250"
    assert page.locator("#design-stale").is_hidden(), "a rename is a label: the run sheet keeps its runs"


def test_a_bound_box_rewrites_the_limit(blank_page):
    page = blank_page
    type_limit(page, HEAT)
    limit_row(page).locator("[name=high]").fill("240")
    page.wait_for_timeout(400)
    settled(page)
    assert limit_row(page).locator("[name=expression]").input_value() == "T + 25 * P <= 240"
    assert limit_row(page).locator(".limit-effect").text_content().startswith("Rules out")


def test_the_factor_buttons_type_a_name_at_the_caret(blank_page):
    page = blank_page
    reveal(page, "#limits")
    page.click("#limits summary")
    page.click("#add-limit")
    box = limit_row(page).locator("[name=expression]")
    for piece in ("Type T", "Type at most, <="):
        page.get_by_role("button", name=piece, exact=True).click()
    box.press("End")
    box.type(" 190")
    assert box.input_value() == "T <= 190"
    assert page.evaluate("document.activeElement.name") == "expression", "the keyboard stays in the box"


def test_the_limits_come_back_with_the_draft(blank_page, site):
    page = blank_page
    type_limit(page, HEAT, note="the product chars")
    page.wait_for_timeout(400)  # the draft is written a moment after the last change
    page.reload()
    page.wait_for_function("window.__pi.count('api_catalogue') === 1 && window.__pi.idle()")
    reveal(page, "#limits")
    assert page.evaluate("document.querySelector('#limits').open")
    assert limit_row(page).locator("[name=expression]").input_value() == HEAT
    assert limit_row(page).locator("[name=note]").input_value() == "the product chars"


def test_a_saved_study_keeps_its_limits(blank_page):
    page = blank_page
    type_limit(page, HEAT)
    candidates_done(page)
    generate(page)
    saved = saved_studies(page, lambda studies: bool(studies) and "constraints" in studies[0]["design"]["spec"])
    assert saved[0]["design"]["spec"]["constraints"] == [{"expression": HEAT, "type": "linear"}]
    page.reload()
    page.wait_for_function("window.__pi.count('api_catalogue') === 1 && window.__pi.idle()")
    settled(page)
    reveal(page, "#limits")
    assert limit_row(page).locator("[name=expression]").input_value() == HEAT
    assert page.locator("#design-stale").is_hidden(), "the study opens as it was made"


def test_set_factors_keeps_the_limits_when_the_ranges_change(blank_page):
    """A limit is a physical statement in the factors' units: a range changed leaves its text, and its effect changes."""
    page = blank_page
    type_limit(page, HEAT)
    before = limit_row(page).locator(".limit-effect").text_content()
    set_factors(page, [{**FACTORS[0], "high": 210}, *FACTORS[1:]])
    page.wait_for_timeout(400)
    settled(page)
    assert limit_row(page).locator("[name=expression]").input_value() == HEAT
    assert limit_row(page).locator(".limit-effect").text_content() != before
