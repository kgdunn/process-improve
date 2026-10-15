"""The designs that can learn the brief: made and scored as it is typed, picked from a list, compared in detail.

Design asks Python, once the brief has two factors and the typing pauses, for every design the
page offers for these factors, each made as Design makes it and all scored on the model the study
aims to learn, with power for the response's smallest change worth finding. The designs that fit
are listed by their runs, one picked for a reader who has not chosen; the detailed comparison
shows the nine measures, the FDS curves and a correlation map per design, and any design can be
taken from it into the plan.
"""

from __future__ import annotations

import pytest

from tests.web._support import FACTORS, factors_named
from tests.web.e2e._page import (
    calls,
    candidates_done,
    generate,
    last_reply,
    name_response,
    practise,
    reveal,
    set_aim,
    set_factors,
    settled,
)


def listed(page) -> list[str]:
    """Return the designs listed as able to learn the brief, in the order shown."""
    return page.locator("#candidates tbody tr").evaluate_all("rows => rows.map((tr) => tr.dataset.design)")


def picked(page) -> str:
    return page.get_attribute('#candidates tbody tr[aria-selected="true"]', "data-design")


def design_columns(page) -> list[str]:
    heads = page.locator("#comparison thead th").all_text_contents()
    assert heads[:2] == ["Measure", "Better"]
    return [head.removesuffix("Use this design") for head in heads[2:]]


def test_the_designs_that_can_learn_the_brief_are_made_and_scored_as_it_is_typed(fresh_page):
    page = fresh_page
    name_response(page, "Yield", "%", noise=1, change=2)
    assert calls(page, "api_candidates") == [], "nothing to learn about before two factors"
    set_factors(page, FACTORS)
    candidates_done(page)
    (request,) = [c["payload"] for c in calls(page, "api_candidates")]
    assert request["model"] == "main_quadratic", "each factor's curvature, for a study of three factors"
    assert request["factors"] == FACTORS
    assert request["response_info"]["smallest_change"] == 2
    reply = last_reply(page, "api_candidates")["result"]
    fitting = [
        m["design_type"]
        for m, label in zip(reply["made"], reply["result"]["designs"], strict=True)
        if reply["result"]["values"][label]["fits"]
    ]
    assert sorted(listed(page)) == sorted(fitting)
    runs = page.locator("#candidates tbody tr td:nth-child(2)").all_text_contents()
    assert [int(r) for r in runs] == sorted(int(r) for r in runs), "sorted by their runs"
    assert picked(page) == "dsd", "the fewest runs that leave three or more to estimate the noise"
    assert page.input_value("#design-type") == "dsd"
    assert "cannot fit this model" in page.text_content("#candidates")
    assert page.text_content("#candidates thead th[colspan]") == "Power to find a change of 2 %"


def test_another_aim_lists_the_designs_that_can_learn_it(blank_page):
    page = blank_page
    set_aim(page, "main_effects")
    candidates_done(page)
    assert calls(page, "api_candidates")[-1]["payload"]["model"] == "main_effects"
    assert "fractional_factorial" in listed(page), "a two-level fraction can learn the main effects"
    assert picked(page) == "fractional_factorial"
    headings = page.locator("#candidates thead th").all_text_contents()
    power = "Power to find a change of two noise standard deviations"  # no smallest change given
    assert headings == ["Design", "Runs", "Spare runs", power, "Main effect"]


def test_power_is_for_the_smallest_change_worth_finding(blank_page):
    page = blank_page
    name_response(page, "Purity", "%", noise=1.5, change=6)
    page.wait_for_function("window.__pi.count('api_candidates') === 2")
    candidates_done(page)
    reply = last_reply(page, "api_candidates")["result"]
    assert reply["result"]["effect_size"] == 2.0, "a change of 6 is a coefficient of 3, two noise standard deviations"
    assert page.text_content("#candidates thead th[colspan]") == "Power to find a change of 6 %"


def test_picking_a_design_puts_it_in_the_plan_and_the_reader_s_choice_is_kept(blank_page):
    page = blank_page
    page.click('#candidates tbody tr[data-design="box_behnken"]')
    assert page.input_value("#design-type") == "box_behnken"
    assert picked(page) == "box_behnken"
    set_aim(page, "quadratic")
    candidates_done(page)
    assert picked(page) == "box_behnken", "a design the reader picked stays picked for another aim"


def test_the_designs_are_made_only_while_plan_is_on_show(blank_page):
    page = blank_page
    page.click("#go-run")
    page.locator("#factors tbody tr").first.locator("[name=name]").fill("Temp")
    page.wait_for_timeout(700)
    assert len(calls(page, "api_candidates")) == 1, "nothing is made for a list nobody is looking at"
    page.click("#go-plan")
    candidates_done(page)
    assert len(calls(page, "api_candidates")) == 2
    assert calls(page, "api_candidates")[-1]["payload"]["factors"][0]["name"] == "Temp"


def test_a_brief_changed_while_its_designs_are_made_is_listed_again(blank_page):
    page = blank_page
    page.evaluate("window.__pi.hold('api_candidates')")
    name = page.locator("#factors tbody tr").first.locator("[name=name]")
    name.fill("Temp")
    name.press("Tab")  # a name is taken as the reader leaves its box
    page.wait_for_function("window.__pi.count('api_candidates') === 2")
    name.fill("Heat")
    name.press("Tab")
    assert page.get_attribute("#candidates", "data-stale") == "true", "the last list stays, dimmed"
    page.evaluate("window.__pi.release('api_candidates')")
    page.wait_for_function("window.__pi.count('api_candidates') === 3")
    page.evaluate("window.__pi.release('api_candidates')")
    candidates_done(page)
    assert calls(page, "api_candidates")[-1]["payload"]["factors"][0]["name"] == "Heat"
    assert page.get_attribute("#candidates", "data-stale") == "false"


def test_when_the_designs_cannot_be_made_the_page_says_why(blank_page):
    page = blank_page
    page.evaluate("window.__pi.fake('api_candidates', {ok: false, error: 'the solver gave up', warnings: []})")
    page.locator("#factors tbody tr").first.locator("[name=name]").fill("Temp")
    page.keyboard.press("Tab")
    page.wait_for_function("window.__pi.count('api_candidates') === 2")
    candidates_done(page)
    assert page.text_content("#candidates") == "the solver gave up"
    assert page.is_disabled("#compare")


@pytest.mark.slow
def test_with_five_factors_the_optimal_designs_wait_to_be_searched_for(fresh_page):
    page = fresh_page
    set_factors(page, factors_named(*"ABCDE"))
    candidates_done(page)
    assert page.is_checked('input[name="aim"][value="main_effects"]'), "a screening aim for five factors"
    assert "d_optimal" in listed(page), "an optimal design for main effects is quick to search for"
    set_aim(page, "main_quadratic")
    candidates_done(page)
    assert "d_optimal" not in listed(page)
    assert "designs take a while to search for with 5 factors" in page.text_content("#candidates")
    page.click("#search")
    page.wait_for_function("window.__pi.count('api_candidates') === 3", timeout=10_000)
    candidates_done(page)
    assert calls(page, "api_candidates")[-1]["payload"]["search"] is True
    assert {"d_optimal", "i_optimal"} <= set(listed(page))
    assert page.is_hidden("#search")


def test_the_comparison_in_detail_has_a_column_per_design_and_a_row_per_measure(page, first_paint):
    reveal(page, "#compare")
    page.click("#compare")
    page.wait_for_function("document.querySelectorAll('#compare-charts .js-plotly-plot').length > 1")
    compared = first_paint["comparison"]["result"]
    assert design_columns(page) == compared["designs"]
    shown = [
        m for row, m in enumerate(compared["measures"]) if any(compared["display"][d][row] for d in compared["designs"])
    ]
    groups = page.locator("#comparison tbody tr.group").all_text_contents()
    assert groups == list(dict.fromkeys(m["group"] for m in shown))
    assert page.locator("#comparison tbody tr:not(.group)").count() == len(shown)
    charts = 1 + len(compared["designs"])  # the FDS plot and a correlation map per design
    page.wait_for_function(f"document.querySelectorAll('#compare-charts .js-plotly-plot').length === {charts}")
    assert [c["fn"] for c in calls(page)] == ["api_catalogue"], "the build compared them"


def test_a_design_taken_from_the_detailed_comparison_reaches_the_run_sheet_as_made(page, first_paint):
    chosen = next(m for m in first_paint["comparison"]["made"] if m["design_type"] == "d_optimal")
    reveal(page, "#compare")
    page.click("#compare")
    column = page.locator("#comparison thead th", has_text=chosen["label"])
    column.get_by_role("button", name="Use this design").click()
    assert page.is_hidden("#compare-dialog")
    assert page.input_value("#design-type") == "d_optimal"
    assert page.text_content("#badge-plan") == "changed", "the run sheet on the page is the Box-Behnken design's"
    page.click("#generate")
    settled(page)
    assert calls(page, "api_make_design") == [], "the comparison made it already"
    assert page.locator("#design-table tbody tr").count() == chosen["design"]["n_runs"]
    assert calls(page, "api_simulate"), "practising: the same process, noise and seed, on the new runs"
    assert page.locator("#design-table input.sim").count() == chosen["design"]["n_runs"]


def test_a_design_made_for_the_brief_keeps_the_response_as_the_reader_named_it(blank_page):
    page = blank_page
    name_response(page, "Tensile strength", "MPa")
    candidates_done(page)
    generate(page)
    assert page.locator("#design-table thead th button.sort").last.text_content() == "Tensile strength"
    practise(page)
    assert page.text_content("#truth-equation").startswith("Tensile strength = ")
