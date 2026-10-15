"""The analysis in Learn as the page shows it: the fitted equation, the tables, and the sentences around them.

When the results were simulated, each coefficient stands beside the true value of what it
estimates. A saturated two-level fit is judged with Lenth's method; replicated center runs
give a lack-of-fit test; a model the design cannot fit is refused in a sentence.
"""

from __future__ import annotations

import re

from tests.web.e2e._page import calls, choose, generate, headers, practise, settled, status, type_results


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
    tip = page.locator("#analysis table th button.tip")
    assert tip.get_attribute("aria-label") == "About true value", "the column's definition, on request"
    tip.click()
    shown = page.locator(".tip-text:popover-open").text_content()
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
    assert "Each effect is twice its coefficient" in text, "the method's definition, behind a ? in its heading"
    assert "Predicted R² cannot be computed: at least one run cannot be predicted from the others" in text
    lenth = page.locator("#analysis table").nth(1)
    assert lenth.locator("thead th").all_text_contents() == ["Term", "Effect", "Beyond ME", "Beyond SME"]
    assert page.get_attribute("#analysis h3 a", "href").endswith("#doe-lenth-method")


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
