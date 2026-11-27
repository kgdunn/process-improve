"""Factors that are hard to change, in the browser (#1225).

A tick on a factor in the plan's settings makes a split-plot design: the run sheet lists the runs whole plot
by whole plot, and Learn judges each term against the spread it meets.
"""

from __future__ import annotations

from typing import Any

from tests.web.e2e._page import choose, generate, practise, reveal, run_sheet, saved_studies, set_factors, settled

FACTORS = [
    {"name": "T", "low": 150, "high": 200, "units": "degC"},
    {"name": "P", "low": 1, "high": 3, "units": "bar"},
    {"name": "S", "low": 10, "high": 30, "units": "rpm"},
]


def tick(page: Any, name: str) -> Any:  # noqa: ANN401 - a Playwright Page and Locator
    """Return the tick of one factor in the plan's settings."""
    return page.locator("#hard-factors label", has_text=name).locator("input")


def planned(page: Any, design_type: str = "d_optimal") -> None:  # noqa: ANN401 - a Playwright Page
    set_factors(page, FACTORS)
    choose(page, design_type=design_type)
    settled(page)


def columns(page: Any) -> list[str]:  # noqa: ANN401 - a Playwright Page
    return page.evaluate("[...document.querySelectorAll('#design-table thead th button')].map((b) => b.textContent)")


def test_the_ticks_are_offered_with_the_designs_that_take_them(blank_page):
    page = blank_page
    planned(page, "full_factorial")
    assert page.is_hidden("#hard-to-change"), "a full factorial randomizes every run"
    choose(page, design_type="d_optimal")
    settled(page)
    reveal(page, "#hard-to-change")
    assert page.locator("#hard-factors label").all_text_contents() == [" T", " P", " S"]
    assert not any(tick(page, name).is_checked() for name in "TPS"), "no factor is hard to change until ticked"


def test_a_ticked_factor_is_set_once_for_each_whole_plot(blank_page):
    page = blank_page
    planned(page)
    tick(page, "T").check()
    generate(page)
    assert "Whole plot" in columns(page)[2]
    rows = run_sheet(page)
    plots: dict[str, set[str]] = {}
    for row in rows:
        plots.setdefault(row[2], set()).add(row[3])
    assert len(plots) > 1
    assert all(len(settings) == 1 for settings in plots.values()), "T has one setting in each whole plot"
    assert [row[2] for row in rows] == sorted((row[2] for row in rows), key=int), "one whole plot after another"
    said = page.locator("#design-summary").text_content()
    assert f"{len(rows)} runs, in the order given." in said
    assert f"T is hard to change, so the runs are made in {len(plots)} groups, called whole plots" in said


def test_a_tick_changed_marks_the_run_sheet_as_from_an_earlier_design(blank_page):
    page = blank_page
    planned(page)
    generate(page)
    stale = "() => !document.getElementById('design-stale').hidden"  # the note is in Design, the ticks in Plan
    assert not page.evaluate(stale)
    reveal(page, "#hard-to-change")
    tick(page, "P").check()
    assert page.evaluate(stale)
    tick(page, "P").uncheck()
    assert not page.evaluate(stale)


def test_learn_says_what_each_term_is_judged_against(blank_page):
    page = blank_page
    planned(page)
    tick(page, "T").check()
    generate(page)
    practise(page)
    reveal(page, "#analysis")
    said = page.locator("#analysis").text_content()
    assert "T was set once for each of the" in said
    assert " judged against the first, and every other term against the second." in said


def test_a_study_opened_again_keeps_its_ticks_and_a_renamed_factor_keeps_its_own(blank_page):
    page = blank_page
    planned(page)
    tick(page, "T").check()
    generate(page)
    practise(page)
    saved_studies(page, bool)  # the study is kept in the browser before the page is opened again
    page.reload()
    page.wait_for_selector("#design-table tr[data-run]", state="attached")
    settled(page)
    reveal(page, "#hard-to-change")
    assert tick(page, "T").is_checked()
    assert not tick(page, "P").is_checked()
    name = page.locator("#factors tbody tr").first.locator('[name="name"]')
    name.fill("Temp")
    name.press("Tab")
    settled(page)
    assert page.locator("#hard-factors label").all_text_contents() == [" Temp", " P", " S"]
    assert tick(page, "Temp").is_checked()
    assert "Temp is hard to change" in page.locator("#design-summary").text_content()
