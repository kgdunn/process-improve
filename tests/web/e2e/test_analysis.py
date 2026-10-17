"""The analysis in Learn as the page shows it: the fitted equation, the tables, and the sentences around them.

When the results were simulated, each coefficient stands beside the true value of what it
estimates. A saturated two-level fit is judged with Lenth's method; replicated center runs
give a lack-of-fit test; a model the design cannot fit is refused in a sentence.
"""

from __future__ import annotations

import base64
import re

from tests.web._support import correlated_study, fill_workbook
from tests.web.e2e._page import (
    calls,
    choose,
    definition,
    generate,
    headers,
    practise,
    reveal,
    settled,
    status,
    type_results,
)


def coefficient_rows(page) -> list[list[str]]:
    table = page.locator("#analysis table").first
    return [row.locator("td").all_text_contents() for row in table.locator("tbody tr").all()]


def test_each_estimate_stands_beside_its_true_value(page, first_paint):
    analysis = first_paint["analysis"]
    assert page.text_content("#analysis .equation") == analysis["equation"]
    assert headers(page, "#analysis table")[:6] == [
        "Term",
        "Coefficient",
        "Std. error",
        "95% interval",
        "p-value",
        "True value",
    ]
    rows = coefficient_rows(page)
    assert [row[0] for row in rows] == [c["label"] for c in analysis["coefficients"]]
    assert [float(row[-1]) for row in rows] == [c["true_value"] for c in analysis["coefficients"]]
    shown = definition(page, '#analysis table th .term[data-term="true-value"]')
    assert shown.startswith("What the coefficient estimates for the process the results were simulated from")


def test_results_typed_in_have_no_true_values(blank_page):
    page = blank_page
    generate(page, design_type="full_factorial")
    type_results(page, {run: 50 + run for run in range(11)})
    assert "True value" not in headers(page, "#analysis table")


def test_lack_of_fit_is_tested_against_the_repeated_center_runs(page):
    text = page.text_content("#analysis")
    assert "Lack of fit: F = " in text
    assert "(pure error from the replicated runs: 2 degrees of freedom)" in text


def test_a_saturated_two_level_design_is_judged_with_lenth_s_method(blank_page):
    page = blank_page
    generate(page, design_type="plackett_burman", n_center_points=0)
    type_results(page, {0: 12.1, 1: 15.4, 2: 9.8, 3: 14.0})
    text = page.text_content("#analysis")
    assert "Effects, judged with Lenth's method" in text
    assert re.search(r"PSE = [-\d.]+; ME = [-\d.]+; SME = [-\d.]+\.", text)
    assert definition(page, '#analysis h3 .term[data-term="effect"]').endswith("twice its coefficient.")
    assert "Predicted R² cannot be computed: at least one run cannot be predicted from the others" in text
    lenth = page.locator("#analysis table").nth(1)
    assert lenth.locator("thead th").all_text_contents() == ["Term", "Effect", "Beyond ME", "Beyond SME"]
    assert page.get_attribute("#analysis h3 a", "href").endswith("#doe-lenth-method")


def test_an_orthogonal_design_shows_no_vif_column_and_no_note(page):
    """The worked example's terms are close to uncorrelated: a column of VIFs near 1 would say nothing."""
    assert "VIF" not in headers(page, "#analysis table")
    assert "variance inflation factor" not in page.text_content("#analysis")
    assert "evaluate_design" not in page.text_content("#py-analyse code")


def test_correlated_terms_show_their_vifs_in_a_column_and_a_note(page, api, tmp_path):
    """Runs as given in which A and B move together: the note names them, and the table shows each term's VIF."""
    design = api.ok("api_make_design", correlated_study())
    path = tmp_path / "correlated.xlsx"
    filled = fill_workbook(design["xlsx_base64"], lambda runs: 10 + 2 * runs["A"] - runs["C"] + 0.1 * runs.index)
    path.write_bytes(base64.b64decode(filled))
    reveal(page, "#tab-real")
    page.click("#tab-real")
    page.set_input_files("#upload", path)
    settled(page)
    coefficients = page.locator("#analysis table").first.locator("thead th button.sort").all_text_contents()
    assert coefficients == ["Term", "Coefficient", "Std. error", "95% interval", "p-value", "VIF"]
    assert {row[0]: row[-1] for row in coefficient_rows(page)} == {
        "Intercept": "",
        "A": "10.80",
        "B": "11.00",
        "C": "1.200",
    }
    notes = page.locator("#analysis ul.facts li").all_text_contents()
    (note,) = [n for n in notes if "VIF" in n]
    assert note.startswith("A (VIF 11) and B (VIF 11) are correlated with other terms in these runs.")
    # Once the results are read, the mouse that pressed "Enter results" can rest on a word of Learn's and open
    # its definition: move it away, and let that close, before resting it on the heading.
    page.mouse.move(0, 0)
    page.locator(".term-tip").wait_for(state="hidden")
    assert definition(page, '#analysis table th .term[data-term="vif"]').startswith("The variance inflation factor:")
    assert page.locator('#analysis ul.facts .term[data-term="vif"]').text_content() == "variance inflation factor"
    assert 'vif = evaluate_design(X, model="A + B + C", metric="vif")["vif"]' in page.text_content("#py-analyse code")


def test_choosing_a_model_analyses_the_results_again(page):
    choose(page, model="main_effects")
    settled(page)
    assert calls(page, "api_analyze_runs")[-1]["payload"]["model"] == "main_effects"
    assert [row[0] for row in coefficient_rows(page)] == ["Intercept", "T", "P", "F"]


def test_a_model_the_design_cannot_fit_is_refused_in_a_sentence(page):
    generate(page, design_type="dsd", n_center_points=1)  # nine runs, simulated again
    choose(page, model="quadratic")
    settled(page)
    assert status(page) == ("The results could not be analysed.", "error")
    assert "cannot fit the quadratic model" in page.text_content("#analysis p.error")
    assert page.is_hidden("#charts")


def test_a_blocked_design_shows_each_run_s_block_and_fits_the_blocks(blank_page):
    page = blank_page
    generate(page, design_type="full_factorial", n_center_points=2, n_blocks=2)
    assert headers(page)[:3] == ["Std order", "Run order", "Block"]
    assert calls(page, "api_make_design")[-1]["payload"]["n_blocks"] == 2
    practise(page)
    assert "Block 1" in [row[0] for row in coefficient_rows(page)]
    anova = page.locator("#analysis table").last.locator("tbody tr td:first-child").all_text_contents()
    assert "Block" in anova
    assert "The blocks are fitted as a fixed effect" in page.text_content("#analysis")
    code = page.text_content("#py-analyse code")
    assert '"Block": runs["Block"],' in code, "the blocks are fitted as the page fits them"


def ticked(page) -> list[str]:
    boxes = page.locator("#term-list input:checked")
    return [boxes.nth(i).locator("xpath=..").text_content().strip() for i in range(boxes.count())]


def test_the_terms_to_fit_can_be_ticked_three_factor_interactions_included(blank_page):
    page = blank_page
    generate(page, design_type="full_factorial", n_center_points=2)
    practise(page)
    assert page.is_hidden("#terms")
    choose(page, model="terms")
    settled(page)
    assert page.is_visible("#terms")
    assert ticked(page) == ["T", "P", "F", "T·P", "T·F", "P·F"], "the list starts from the design's own model"
    before = page.evaluate("window.__pi.count('api_analyze_runs')")
    page.get_by_label("T·P·F", exact=True).check()
    page.get_by_label("P·F", exact=True).uncheck()
    page.wait_for_function(f"window.__pi.count('api_analyze_runs') > {before}")
    settled(page)
    assert calls(page, "api_analyze_runs")[-1]["payload"]["model"] == {
        "terms": [["T"], ["P"], ["F"], ["T", "P"], ["T", "F"], ["T", "P", "F"]]
    }
    assert [row[0] for row in coefficient_rows(page)] == ["Intercept", "T", "P", "F", "T·P", "T·F", "T·P·F"]
    assert 'model="T + P + F + T:P + T:F + T:P:F"' in page.text_content("#py-analyse code")


def test_the_terms_start_from_the_named_model_chosen_before(blank_page):
    page = blank_page
    generate(page, design_type="box_behnken")
    choose(page, model="quadratic")
    choose(page, model="terms")
    assert ticked(page)[-3:] == ["T²", "P²", "F²"]
    choose(page, model="")
    assert page.is_hidden("#terms")


def test_a_design_that_can_fit_the_aim_is_analysed_for_it(blank_page):
    """A DSD made to learn each factor's curvature is analysed with the squares, not with its own main effects."""
    page = blank_page
    generate(page, design_type="dsd")
    assert page.input_value("#model") == "terms"
    assert ticked(page) == ["T", "P", "F", "T²", "P²", "F²"]
    practise(page)
    assert [row[0] for row in coefficient_rows(page)] == ["Intercept", "T", "P", "F", "T²", "P²", "F²"]


def test_a_design_that_cannot_fit_the_aim_is_analysed_as_designed(blank_page):
    page = blank_page
    generate(page, design_type="fractional_factorial")
    assert page.input_value("#model") == "", "a two-level design cannot tell the squares apart"
