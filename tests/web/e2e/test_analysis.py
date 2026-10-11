"""Step 3's analysis as the page shows it: the fitted equation, the tables, and the sentences around them.

When the results were simulated, each coefficient stands beside the true value of what it
estimates. A saturated two-level fit is judged with Lenth's method; replicated center runs
give a lack-of-fit test; a model the design cannot fit is refused in a sentence.
"""

from __future__ import annotations

from tests.web.e2e._page import calls, choose, generate, headers, settled, status, type_results


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
    assert "The true value is what the coefficient estimates" in page.text_content("#analysis")


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
    assert "Pseudo standard error PSE = " in text
    assert "Each effect is twice its coefficient" in text
    assert "Predicted R² cannot be computed: at least one run cannot be predicted from the others." in text
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
    page.click("#simulate")
    settled(page)
    assert "Block 1" in [row[0] for row in coefficient_rows(page)]
    anova = page.locator("#analysis table").last.locator("tbody tr td:first-child").all_text_contents()
    assert "Block" in anova
    assert "The blocks are fitted as a fixed effect" in page.text_content("#analysis")
    code = page.text_content("#py-analyse code")
    assert 'result.design[names + ["Block"]]' in code
