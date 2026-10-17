"""A Taguchi orthogonal array, made, simulated and analysed as a reader would, in the browser.

The array takes one setting of its own, the levels of each numeric factor; its facts name the
array, and Learn fits the main effects and squares a three-level array is made for, saying once
that the interactions are aliased with them.
"""

from __future__ import annotations

from tests.web._support import FACTORS
from tests.web.e2e._page import calls, choose, generate, headers, last_reply, practise, reveal, run_sheet, settled


def facts(page) -> list[str]:
    return page.locator("#design-summary ul.facts li").all_text_contents()


def test_the_levels_setting_shows_for_a_taguchi_array_only(blank_page):
    page = blank_page
    choose(page, design_type="taguchi")
    assert page.is_visible("#n_levels")
    assert page.input_value("#n_levels") == "2", "two levels, until the reader asks for three"
    choose(page, design_type="full_factorial")
    assert page.is_hidden("#n_levels")


def test_a_three_level_array_is_made_simulated_and_analysed_for_its_main_effects_and_squares(blank_page):
    page = blank_page
    generate(page, design_type="taguchi", n_levels=3)
    request = calls(page, "api_make_design")[-1]["payload"]
    assert (request["design_type"], request["n_levels"]) == ("taguchi", 3)
    assert len(run_sheet(page)) == 9
    assert facts(page)[0] == "Taguchi L9 (3^4): 9 runs, main effects and squares."
    assert "levels=[150, 175, 200]" in page.text_content("#py-design code")

    practise(page)
    reveal(page, "#analysis")
    assert page.input_value("#model") == "", "analysed as designed"
    analysis = last_reply(page, "api_analyze_runs")
    assert analysis["warnings"] == []
    assert analysis["result"]["formula"] == "T + P + F + I(T ** 2) + I(P ** 2) + I(F ** 2)"
    note = (
        "An orthogonal array estimates the main effects and squares; the interactions are aliased with them, "
        "in full or in part, so a large effect may belong to an interaction."
    )
    assert page.locator("#analysis ul.facts li").all_text_contents() == [note]
    terms = [
        row.locator("td").first.text_content()
        for row in page.locator("#analysis table").first.locator("tbody tr").all()
    ]
    assert terms == ["Intercept", "T", "P", "F", "T²", "P²", "F²"]
    assert "True value" in headers(page, "#analysis table"), "each estimate beside the simulated process's"


def test_a_change_of_levels_makes_the_run_sheet_out_of_date(blank_page):
    page = blank_page
    generate(page, design_type="taguchi")
    assert facts(page)[0] == "Taguchi L4 (2^3): 4 runs, main effects."
    choose(page, n_levels=3)
    assert page.is_visible("#plan-note")
    assert page.text_content("#plan-note") == "The run sheet in Run is from an earlier design."


def test_the_array_picked_from_the_designs_for_the_brief_keeps_its_three_levels(blank_page):
    """The brief aims at main effects and squares, so the comparison lists the array at three levels."""
    page = blank_page
    page.click('#candidates tbody tr[data-design="taguchi"]')
    assert page.input_value("#design-type") == "taguchi"
    assert page.input_value("#n_levels") == "3"
    reveal(page, "#generate")
    page.click("#generate")
    settled(page)
    assert len(run_sheet(page)) == 9
    assert facts(page)[0] == "Taguchi L9 (3^4): 9 runs, main effects and squares."
    assert [f["name"] for f in FACTORS] == headers(page)[2:5]
