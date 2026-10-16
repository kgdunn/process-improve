"""Limits on the settings: typed under the factors, read by Python, and kept to by every design the page makes.

A limit is typed in the factors' own units. Python reads it as the library will, says under it what
it rules out, and gives back its canonical text when the reader leaves the box; a limit that cannot be
read says why, and holds the brief back. With any limit, only the D-optimal and I-optimal designs are
offered, since only they keep every run inside, and the run sheet's runs all meet the limits.

A limit can also be had by marking the corners of the ranges that cannot be run: Python proposes the
straight limit that rules out just those, which the reader adds as it is or adjusts factor by factor.
"""

from __future__ import annotations

from typing import Any

from tests.web._support import FACTORS, factors_named
from tests.web.e2e._page import (
    calls,
    candidates_done,
    corners_done,
    generate,
    limits_done,
    mark_corners,
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
    type_limit(page, "T \u2212 25\u00b7P \u2265 100")
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
    """A limit is a statement in the factors' units: a range changed leaves its text, and changes its effect."""
    page = blank_page
    type_limit(page, HEAT)
    before = limit_row(page).locator(".limit-effect").text_content()
    set_factors(page, [{**FACTORS[0], "high": 210}, *FACTORS[1:]])
    page.wait_for_timeout(400)
    settled(page)
    assert limit_row(page).locator("[name=expression]").input_value() == HEAT
    assert limit_row(page).locator(".limit-effect").text_content() != before


# --------------------------------------------------------------------------- the settings marked "cannot run"
def proposals(page: Any) -> list[dict[str, str | None]]:  # noqa: ANN401
    """Read each proposal as the reader sees it: who it is for, its limit, what it does, and its problem or warning."""
    return page.evaluate(
        """() => [...document.querySelectorAll('#limit-proposals .proposal')].map((card) => Object.fromEntries(
             ['for', 'text', 'effect', 'also', 'problem'].map((part) => {
               const p = card.querySelector(`.proposal__${part}`);
               return [part, p && !p.hidden ? p.textContent : null];
             })))"""
    )


def corner_box(page: Any, row: int) -> Any:  # noqa: ANN401
    return page.locator(f'#corner-table tr[data-row="{row}"] input')


#: The row of the table of corners the keyboard is in.
FOCUSED_ROW = "() => Number(document.activeElement.closest('tr')?.dataset.row)"


def test_marking_corners_proposes_the_limit_that_rules_out_just_those(blank_page):
    page = blank_page
    corners_done(page)
    assert page.locator("#corner-table tbody tr").count() == 9, "the 8 corners of the ranges, and their center"
    centre = corner_box(page, 8).get_attribute("aria-label")
    assert centre == "Cannot run: the center of the ranges, T 175 degC, P 2 bar, F 15 L/min"
    mark_corners(page, 6, 7)
    assert calls(page, "api_limits")[-1]["payload"]["marked"] == [6, 7]
    assert proposals(page) == [
        {
            "for": "For the 2 corners where T is 200 degC and P is 3 bar:",
            "text": HEAT,
            "effect": "Rules out 13% of your ranges: 2 of the 8 corners.",
            "also": None,
            "problem": None,
        }
    ]
    page.get_by_role("button", name="Add this limit").click()
    limits_done(page)
    assert limit_row(page).locator("[name=expression]").input_value() == HEAT
    assert page.evaluate("document.activeElement.name") == "expression", "the keyboard goes to the limit added"
    assert proposals(page) == [], "the corners marked are ruled out now"
    ruled = corner_box(page, 7).get_attribute("aria-label")
    assert ruled == "Cannot run: T 200 degC, P 3 bar, F 20 L/min, ruled out by limit 1"
    assert page.text_content('#corner-table tr[data-row="7"] .corner__by') == "1"


def test_a_corner_a_limit_rules_out_stays_ticked(blank_page):
    page = blank_page
    type_limit(page, HEAT)
    corners_done(page)
    box = corner_box(page, 7)
    assert box.is_checked()
    asked = len(calls(page, "api_limits"))
    box.click(force=True)  # it reads as dimmed, which Playwright will not click unless told to; a reader can
    page.wait_for_timeout(400)
    assert box.is_checked(), "a limit decides it, not the tick"
    assert len(calls(page, "api_limits")) == asked


def test_when_no_straight_cut_rules_out_the_corners_marked_the_page_says_why(blank_page):
    page = blank_page
    set_factors(page, factors_named(*"ABCD"))
    corners_done(page)
    mark_corners(page, 0, 1, 2, 5)
    [proposal] = proposals(page)
    assert proposal["problem"] == (
        "No single straight cut rules out these corners and keeps the others. Mark them in smaller groups, or type a "
        "limit."
    )
    assert proposal["text"] is None
    assert page.locator("#limit-proposals .proposal .actions").is_hidden(), "nothing to add"


def test_a_proposal_is_adjusted_factor_by_factor_from_its_corner(blank_page):
    page = blank_page
    corners_done(page)
    mark_corners(page, 7)
    assert proposals(page)[0]["text"] == "T + 25 * P + 5 * F <= 350"
    page.get_by_role("button", name="Adjust it factor by factor").click()
    cuts = page.locator("#limit-proposals .cuts")
    questions = cuts.locator(".cut__question").all_text_contents()
    assert questions == ["T can go up to  degC", "P can go up to  bar", "F can go up to  L/min"]
    assert [cuts.locator(f"[name=cut-{n}]").input_value() for n in "TPF"] == ["175", "2", "15"]
    cuts.locator("[name=free-F]").check()
    cuts.locator("[name=cut-P]").fill("2.5")
    box = cuts.locator("[name=cut-T]")
    box.fill("190")
    corners_done(page)
    assert page.evaluate("document.activeElement.name") == "cut-T", "the keyboard stays in the box while Python answers"
    assert box.input_value() == "190"
    [proposal] = proposals(page)
    assert proposal["text"] == "T + 20 * P <= 250"
    also = "It also rules out a setting you did not mark: the corner T 200 degC, P 3 bar, F 10 L/min."
    assert proposal["also"] == also
    assert calls(page, "api_limits")[-1]["payload"]["cuts"] == {"marked": [7], "corner": 7, "values": [190, 2.5, None]}
    page.get_by_role("button", name="Add this limit").click()
    limits_done(page)
    assert limit_row(page).locator("[name=expression]").input_value() == "T + 20 * P <= 250"


def test_a_cut_no_limit_goes_through_says_why_under_the_questions(blank_page):
    page = blank_page
    corners_done(page)
    mark_corners(page, 7)
    page.get_by_role("button", name="Adjust it factor by factor").click()
    page.locator("#limit-proposals [name=cut-T]").fill("200")
    corners_done(page)
    assert page.text_content("#limit-proposals .cuts__problem").startswith("The cut of 'T' at 200 is at or beyond")
    assert proposals(page)[0]["text"] == "T + 25 * P + 5 * F <= 350", "the limit shown is the one proposed"


def test_the_table_of_corners_works_from_the_keyboard(blank_page):
    page = blank_page
    corners_done(page)
    corner_box(page, 0).focus()
    for key, row in (("ArrowDown", 1), ("End", 8), ("ArrowUp", 7), ("Home", 0)):
        page.keyboard.press(key)
        assert page.evaluate(FOCUSED_ROW) == row, key
    page.keyboard.press("End")
    page.keyboard.press("ArrowUp")
    page.keyboard.press("Space")
    assert page.get_attribute('#corner-table tr[data-row="7"]', "class") == "marked", "marked at once"
    corners_done(page)
    assert page.evaluate(FOCUSED_ROW) == 7, "the keyboard stays on the row while Python proposes a limit"
    assert proposals(page)[0]["text"] == "T + 25 * P + 5 * F <= 350"
    page.keyboard.press("Tab")
    page.keyboard.press("Tab")
    assert page.evaluate("document.activeElement.id") == "clear-marks", "past the center, out of the table"
    page.keyboard.press("Enter")
    assert page.evaluate(FOCUSED_ROW) == 0, "the marks are cleared, and the keyboard goes back to the table"
    corners_done(page)
    assert proposals(page) == []


def test_with_five_factors_or_more_a_face_of_the_ranges_is_marked_at_once(blank_page):
    """The filters narrow the table to a face of the ranges, and one button marks every corner on it."""
    page = blank_page
    set_factors(page, factors_named(*"ABCDEF"))
    corners_done(page)
    assert page.locator("#corner-table tbody tr").count() == 33, "32 of the 64 corners, and the center"
    assert page.locator("#mark-shown").is_hidden(), "marking every corner would leave no settings"
    page.click("#more-corners")
    assert page.locator("#corner-table tbody tr").count() == 65
    assert page.evaluate(FOCUSED_ROW) == 32, "the button is gone: the keyboard is on the first corner it showed"
    for name in "AB":
        page.select_option(f'#corner-filters select[name="{name}"]', "high")
    assert page.locator("#corner-table tbody tr").count() == 17
    page.click("#mark-shown")
    corners_done(page)
    assert page.get_attribute("#mark-shown", "aria-disabled") == "true", "every corner on the face is marked"
    [proposal] = proposals(page)
    assert proposal["for"] == "For the 16 corners where A is 1 and B is 1:"
    assert proposal["text"] == "A + B <= 1"
