"""A space-filling design in the page: chosen from the design list, made, practised on and analysed.

It is made for no model, so it opens on ten runs a factor rather than on a model's coefficients, its
Runs box says which model Learn fits (the one the study aims to learn), and the list of designs for
a model never shows it. Like the other designs but the optimal ones, it keeps to the factors' ranges
and takes no limits on the settings.
"""

from __future__ import annotations

from tests.web.e2e._page import calls, candidates_done, choose, generate, practise, reveal, set_aim, type_limit


def runs_box(page) -> tuple[str, str]:
    """Return the Runs box's number and the hint beside it."""
    return page.input_value("#budget"), page.text_content("#budget-hint")


def test_it_opens_on_ten_runs_a_factor_and_says_which_model_learn_fits(blank_page):
    page = blank_page
    choose(page, design_type="maximin_lhs")
    assert runs_box(page) == (
        "30",
        "10 runs a factor is the usual start; Learn fits the model of your aim, which has 10 coefficients.",
    )
    set_aim(page, "main_effects")
    assert runs_box(page) == (
        "30",
        "10 runs a factor is the usual start; Learn fits the model of your aim, which has 4 coefficients.",
    ), "the runs stay, as the design is made for no model"
    choose(page, design_type="d_optimal")
    assert runs_box(page) == ("8", "The model has 4 coefficients."), "a D-optimal design's runs follow its model"


def test_its_run_sheet_spreads_each_factor_and_learn_fits_the_aim_s_model(blank_page):
    page = blank_page
    set_aim(page, "interactions")
    generate(page, design_type="maximin_lhs", budget=12)
    made = calls(page, "api_make_design")[-1]["payload"]
    assert (made["design_type"], made["budget"], made["aim"]) == ("maximin_lhs", 12, "interactions")
    summary = page.text_content("#design-summary")
    assert "Each factor takes 12 different settings, evenly spaced across its range" in summary
    practise(page)
    reveal(page, "#analysis")
    assert page.locator("#analysis table").first.locator("tbody tr").count() == 7, "the intercept, 3 mains and 3 pairs"
    code = page.text_content("#py-run code")
    assert code.startswith("import math\n"), "the code rounds the settings as the run sheet lists them"
    seed = made["random_state"]
    assert f'generate_design(factors, design_type="maximin_lhs", random_state={seed}, budget=12)' in code


def test_the_designs_for_a_model_never_list_it_and_limits_rule_it_out(blank_page):
    page = blank_page
    listed = page.locator("#candidates tbody tr").evaluate_all("rows => rows.map((tr) => tr.dataset.design)")
    assert "maximin_lhs" not in listed
    type_limit(page, "T + 25 * P <= 250")
    candidates_done(page)
    assert page.locator('#design-type option[value="maximin_lhs"]').evaluate("(option) => option.disabled")
