"""Run, a run at a time: the run to make now, its settings in large type, its result, and whether it failed.

The card's boxes are the run sheet's own. A run that failed stays on the sheet, marked, with no result: the fit goes
without it and the next step does not wait for it.
"""

from __future__ import annotations

import pytest

from tests.web.e2e._page import generate, reveal, settled

PHONE = {"viewport": {"width": 390, "height": 844}, "is_mobile": True}


def sheet_results(page) -> list[str]:
    return page.locator("#design-table input[data-response]").evaluate_all("(all) => all.map((one) => one.value)")


def a_run_sheet(page) -> None:
    generate(page)
    page.click("#go-run")
    if not page.evaluate("document.querySelector('#bench').open"):
        page.click("#bench > summary")


def test_on_a_desk_the_card_is_closed_until_it_is_opened(blank_page):
    generate(blank_page)
    blank_page.click("#go-run")
    assert blank_page.is_visible("#bench > summary")
    assert blank_page.text_content("#bench > summary") == "One run at a time"
    assert not blank_page.evaluate("document.querySelector('#bench').open"), "the whole sheet is the desk's way"


@pytest.mark.parametrize("context_options", [PHONE])
def test_on_a_phone_the_card_shows_the_first_run_to_make_with_its_settings_in_large_type(blank_page):
    page = blank_page
    generate(page)
    page.click("#go-run")
    assert page.evaluate("document.querySelector('#bench').open")
    assert page.text_content("#bench-run") == "Run 1"
    runs = page.locator("#design-table tbody tr").count()
    assert page.text_content("#bench-of") == f"1 of {runs}, to make. {runs} still to make."
    names = page.locator("#bench-settings dt").all_text_contents()
    assert len(names) >= 2
    size = page.locator("#bench-settings dd").first.evaluate("(one) => parseFloat(getComputedStyle(one).fontSize)")
    assert size >= 22, "read at arm's length"
    box = page.locator("#bench-results input").first.bounding_box()
    assert box["height"] >= 44, "a target a thumb can hit"
    assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
    assert page.is_visible("#design-table"), "the whole sheet stays under it"


def test_a_result_typed_on_the_card_is_the_run_sheet_s_and_next_run_moves_on(blank_page):
    page = blank_page
    a_run_sheet(page)
    page.fill("#bench-results input", "61.5")
    settled(page)
    first = page.evaluate(
        """() => { const rows = [...document.querySelectorAll('#design-table tbody tr')];
        const one = rows.find((row) => row.querySelector('input[data-response]').value !== '');
        return one.querySelector('input[data-response]').value; }"""
    )
    assert first == "61.5"
    assert page.text_content("#bench-of").startswith("1 of "), "the card stays on the run while it is typed"
    assert ", made." in page.text_content("#bench-of")
    page.click("#bench-next")
    assert page.text_content("#bench-run") == "Run 2"
    assert page.input_value("#bench-results input") == ""
    page.click("#bench-prev")
    assert page.input_value("#bench-results input") == "61.5"
    assert page.is_disabled("#bench-prev")


def test_a_result_typed_on_the_run_sheet_shows_on_the_card(blank_page):
    page = blank_page
    a_run_sheet(page)
    place = page.evaluate(
        """() => { const run = document.querySelector('#bench-run').textContent.replace('Run ', '');
        const rows = [...document.querySelectorAll('#design-table tbody tr')];
        const heads = [...document.querySelectorAll('#design-table thead th')];
        const head = heads.findIndex((th) => th.textContent.includes('Run order'));
        return rows.findIndex((row) => row.cells[head].textContent.trim() === run); }"""
    )
    page.locator("#design-table tbody tr").nth(place).locator("input[data-response]").first.fill("58")
    settled(page)
    assert page.input_value("#bench-results input") == "58"


def test_a_run_that_failed_stays_on_the_sheet_with_no_result_and_is_not_waited_for(blank_page):
    page = blank_page
    a_run_sheet(page)
    page.click("#tab-simulate")
    page.click("#simulate")
    settled(page)
    assert all(sheet_results(page))
    assert page.is_visible("#run-done")
    total = page.text_content("#badge-run")
    runs = int(total.split(" of ")[1])
    page.check("#bench-failed")
    settled(page)
    results = sheet_results(page)
    assert results.count("") == 1, "its result is gone"
    assert page.locator("#design-table input[data-response]:disabled").count() == 1
    assert page.locator("#design-table input[data-response]:disabled").get_attribute("placeholder") == "failed"
    assert page.is_disabled("#bench-results input")
    assert ", failed." in page.text_content("#bench-of")
    assert page.text_content("#badge-run") == f"{runs - 1} of {runs - 1}", "the count goes without it"
    assert page.is_visible("#run-done"), "every other run has its result"
    assert page.locator("#design-table tbody tr").count() == runs, "it stays on the sheet"
    page.click("#go-learn")
    page.wait_for_selector("#answer:not([hidden])")
    reveal(page, "#next")
    assert "Once every run has its result" not in page.text_content("#next-out"), "the next step does not wait for it"
    page.click("#go-run")
    page.uncheck("#bench-failed")
    settled(page)
    assert page.locator("#design-table input[data-response]:disabled").count() == 0
    assert page.text_content("#badge-run") == f"{runs - 1} of {runs}", "a run to make again"


def test_the_card_stays_open_or_closed_as_the_reader_left_it(blank_page):
    page = blank_page
    a_run_sheet(page)
    assert page.evaluate("document.querySelector('#bench').open")
    page.reload()
    page.wait_for_selector("#bench > summary", state="attached")
    assert page.evaluate("localStorage.getItem('factorial-bench')") == "open"
