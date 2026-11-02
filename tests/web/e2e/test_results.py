"""Results typed into the run sheet, one at a time as the runs come in.

The page analyses the completed runs a moment after the reader stops typing, lists the runs
still to do, and takes the analysis away again when no result is left.
"""

from __future__ import annotations

from tests.web.e2e._page import calls, enter_results, generate, reveal, settled, status, type_results

#: Fifteen results for the worked example's Box-Behnken design.
RESULTS = [61.2, 58.9, 66.4, 63.0, 55.7, 60.3, 64.8, 59.1, 62.5, 57.6, 65.2, 61.9, 62.4, 61.8, 62.1]


def analyses(page) -> int:
    return len(calls(page, "api_analyze_runs"))


def test_typed_results_are_analysed_once_typing_pauses(blank_page):
    page = blank_page
    generate(page, design_type="box_behnken")
    type_results(page, dict(enumerate(RESULTS)))
    assert analyses(page) == 1, "one analysis after the last keystroke, not one per keystroke"
    assert calls(page, "api_analyze_runs")[0]["payload"]["responses"] == RESULTS
    assert "15 completed runs" in page.text_content("#analysis")
    assert page.locator("#design-table input.sim").count() == 0


def test_the_runs_still_to_do_are_named(blank_page):
    page = blank_page
    generate(page, design_type="box_behnken")
    type_results(page, dict(enumerate(RESULTS[:13])))
    fit = page.locator("#analysis ul.fit li").all_text_contents()
    assert fit[:2] == ["13 completed runs", "Still to run: 14, 15"], "a line each, as a list"
    assert fit[-1].startswith("Residual degrees of freedom = ")
    assert page.text_content("#run-source") == "13 of 15 results in."


def test_clearing_every_result_takes_the_analysis_away(page):
    type_results(page, dict.fromkeys(range(15), ""))
    reveal(page, "#analysis")
    assert page.is_hidden("#analysis")
    assert page.is_visible("#analyse-empty")
    assert page.is_hidden("#charts")
    assert page.is_hidden("#badge-learn"), "no fit to show on the tab"


def test_a_result_typed_over_a_simulated_one_is_entered(page):
    type_results(page, {0: 70})
    assert page.locator('#design-table input[data-index="0"]').get_attribute("class") == ""
    assert page.locator("#design-table input.sim").count() == 14
    assert calls(page, "api_analyze_runs")[-1]["payload"]["responses"][0] == 70


def test_an_analysis_overtaken_by_clearing_the_results_is_dropped(page):
    page.evaluate("window.__pi.hold('api_analyze_runs')")
    enter_results(page, {0: 70})  # its analysis is held on the way back
    enter_results(page, dict.fromkeys(range(15), ""))
    page.evaluate("window.__pi.release('api_analyze_runs')")
    settled(page)
    assert page.is_hidden("#analysis")
    assert page.is_hidden("#charts")
    assert status(page) == ("No results to analyse yet.", "ok")


def test_an_analysis_overtaken_by_a_new_design_is_dropped(blank_page):
    """Entered results, not simulated ones: a new design has no results until the reader adds them."""
    page = blank_page
    generate(page, design_type="box_behnken")
    page.evaluate("window.__pi.hold('api_analyze_runs')")
    enter_results(page, {0: 70})  # its analysis is held on the way back
    reveal(page, "#generate")
    page.click("#generate")
    page.wait_for_function("window.__pi.replies.filter((r) => r.fn === 'api_make_design').length === 2")
    page.evaluate("window.__pi.release('api_analyze_runs')")
    settled(page)
    assert page.is_hidden("#analysis")
    assert page.is_hidden("#charts")
    assert status(page) == ("15 runs generated.", "ok")
