"""Results the reader entered never leave the page without a clear question.

A reader typed four results, pressed "Make the run sheet" again, and met a blank sheet in a new order, with no word
that it was a new study and that the results were in Your studies. A run sheet that holds entered results is kept
when the design is unchanged; when the design has changed the reader is asked, and told where the study is kept.
Practice results fill only the runs that have none of the reader's own. These tests walk each way a run sheet can
be made again.
"""

from __future__ import annotations

import pytest

from tests.web.e2e._page import choose, generate, practise, reveal, settled, status, type_results

PHONE = {"viewport": {"width": 390, "height": 844}, "is_mobile": True}
TYPED = {0: 23, 1: 45, 2: 78, 3: 98}


def results(page) -> list[str]:
    return page.locator("#design-table input[data-response]").evaluate_all("(all) => all.map((one) => one.value)")


def settings(page) -> list[str]:
    """Each run's settings, in the sheet's order: the sheet is the same sheet when these are as they were."""
    return page.locator("#design-table tbody tr").evaluate_all(
        """(rows) => rows.map((row) => [...row.cells].filter((cell) => !cell.querySelector('input'))
             .map((cell) => cell.textContent).join('|'))"""
    )


def a_sheet_with_typed_results(page) -> tuple[list[str], list[str]]:
    generate(page)
    type_results(page, TYPED)
    return results(page), settings(page)


def press_make(page) -> None:
    reveal(page, "#generate")
    page.click("#generate")


@pytest.mark.parametrize("context_options", [{}, PHONE], ids=["a desk", "a phone"])
def test_make_the_run_sheet_again_with_the_design_unchanged_keeps_the_sheet_and_its_results(blank_page):
    page = blank_page
    typed, runs = a_sheet_with_typed_results(page)
    assert [value for value in typed if value] == ["23", "45", "78", "98"]
    press_make(page)
    settled(page)
    assert page.is_hidden("#replace-dialog"), "nothing to ask: nothing would change"
    assert page.get_attribute("#workspace", "data-phase") == "run", "the sheet is shown"
    assert results(page) == typed
    assert settings(page) == runs, "the same runs in the same order"
    assert status(page)[0] == "The run sheet for this design is made already: your results are kept."


@pytest.mark.parametrize("context_options", [{}, PHONE], ids=["a desk", "a phone"])
def test_a_changed_design_asks_before_it_clears_entered_results_and_keeps_them_by_default(blank_page):
    page = blank_page
    typed, runs = a_sheet_with_typed_results(page)
    choose(page, design_type="box_behnken")  # chosen by name: the page's own pick of a design does not undo it
    settled(page)
    press_make(page)
    dialog = page.locator("#replace-dialog")
    dialog.wait_for(state="visible")
    assert page.text_content("#h-replace") == "Start a new study?"
    said = page.text_content("#replace-says")
    assert "holds 4 results you entered" in said
    assert "This study and its results stay in Your studies" in said
    assert page.evaluate("document.activeElement.id") == "replace-keep", "the safe choice has the keyboard"
    page.click("#replace-keep")
    settled(page)
    assert dialog.is_hidden()
    page.click("#go-run")
    assert results(page) == typed
    assert settings(page) == runs


def test_escape_closes_the_question_and_keeps_the_results(blank_page):
    page = blank_page
    typed, _ = a_sheet_with_typed_results(page)
    choose(page, design_type="box_behnken")  # chosen by name: the page's own pick of a design does not undo it
    settled(page)
    press_make(page)
    page.locator("#replace-dialog").wait_for(state="visible")
    page.keyboard.press("Escape")
    settled(page)
    page.click("#go-run")
    assert results(page) == typed


def test_the_reader_can_start_the_new_study_and_the_study_before_keeps_its_results(blank_page):
    page = blank_page
    typed, _ = a_sheet_with_typed_results(page)
    choose(page, design_type="box_behnken")  # chosen by name: the page's own pick of a design does not undo it
    settled(page)
    press_make(page)
    page.locator("#replace-dialog").wait_for(state="visible")
    page.click("#replace-go")
    settled(page)
    page.click("#go-run")
    assert len(results(page)) > len(typed), "the new design's runs: a Box-Behnken design has more"
    assert not any(results(page)), "blank, as the question said"
    kept = page.evaluate(
        """async () => (await (await import('./store.js')).savedCampaigns())
             .map((one) => one.responses.filter((value) => value !== null))"""
    )
    assert [23, 45, 78, 98] in kept, "the study before is in Your studies, with its results"


def test_one_result_is_enough_to_be_asked(blank_page):
    page = blank_page
    generate(page)
    type_results(page, {2: 61.5})
    choose(page, design_type="box_behnken")  # chosen by name: the page's own pick of a design does not undo it
    settled(page)
    press_make(page)
    page.locator("#replace-dialog").wait_for(state="visible")
    assert "holds 1 result you entered" in page.text_content("#replace-says")
    page.click("#replace-keep")


def test_practice_results_are_made_again_without_a_question(blank_page):
    page = blank_page
    generate(page)
    practise(page)
    assert all(results(page))
    choose(page, design_type="box_behnken")  # chosen by name: the page's own pick of a design does not undo it
    settled(page)
    press_make(page)
    settled(page)
    assert page.is_hidden("#replace-dialog"), "simulated results are no loss: they are made again"


def test_a_sheet_with_no_results_is_made_again_without_a_question(blank_page):
    page = blank_page
    generate(page)
    choose(page, design_type="box_behnken")  # chosen by name: the page's own pick of a design does not undo it
    settled(page)
    press_make(page)
    settled(page)
    assert page.is_hidden("#replace-dialog")


def test_simulating_fills_the_other_runs_and_leaves_entered_results_as_they_are(blank_page):
    page = blank_page
    generate(page)
    type_results(page, {0: 23, 2: 78})
    before = results(page)
    practise(page)
    after = results(page)
    assert all(after), "every run has a result now"
    kept = [i for i, value in enumerate(before) if value]
    assert [after[i] for i in kept] == [before[i] for i in kept], "the reader's own are as typed"
    boxes = page.locator("#design-table input[data-response]")
    assert boxes.evaluate_all("(all) => all.filter((one) => one.classList.contains('sim')).length") == len(after) - 2


def test_a_practice_round_does_not_clear_entered_results(blank_page):
    page = blank_page
    typed, _ = a_sheet_with_typed_results(page)
    reveal(page, "#practice-round")
    page.click("#practice-round")
    settled(page)
    assert page.is_hidden("#replace-dialog")
    page.click("#go-run")
    assert results(page) == typed, "the design is unchanged: the sheet and its results stay"


def test_entered_results_are_still_there_after_a_reload(blank_page):
    page = blank_page
    typed, runs = a_sheet_with_typed_results(page)
    page.wait_for_function("async () => (await (await import('./store.js')).savedCampaigns()).length > 0")
    page.reload()
    page.wait_for_selector("#design-table tbody tr", state="attached")
    settled(page)
    page.click("#go-run")
    assert results(page) == typed
    assert settings(page) == runs
