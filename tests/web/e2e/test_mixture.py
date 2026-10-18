"""A mixture in the browser: its components typed into the brief, its run sheet of blends, and its analysis.

A reader names the components of a blend, leaves their bounds blank (0 and 1), chooses a simplex
lattice, makes the run sheet, practises with simulated results and reads the Scheffé model's
analysis, with the fitted response drawn over the triangle of blends.
"""

from __future__ import annotations

import pytest

from tests.web._support import FACTORS
from tests.web.e2e._page import (
    calls,
    choose,
    factor_table,
    generate,
    headers,
    name_response,
    practise,
    reveal,
    run_sheet,
    set_factors,
    settled,
    type_limit,
)
from tests.web.e2e.test_layout import WIDE_FONT, overflowing
from tests.web.e2e.test_responses import add_response, pick, tabs

#: Three components with their bounds left blank, as a reader types them.
BLEND = [{"name": "Water"}, {"name": "Oil"}, {"name": "Wax"}]


def charts(page) -> list[str]:
    """Return the titles of the charts in Learn, once each is drawn."""
    page.wait_for_function(
        "document.querySelectorAll('#charts .js-plotly-plot').length"
        " === document.querySelectorAll('#charts figure').length"
    )
    return page.locator("#charts figure h3").all_text_contents()


def made_mixture(page) -> None:
    """Type the three components, choose a simplex lattice, make its run sheet and practise on it."""
    set_factors(page, BLEND)
    choose(page, design_type="simplex_lattice")
    generate(page)
    practise(page)


def test_a_simplex_lattice_is_made_simulated_and_drawn_over_its_triangle(fresh_page):
    page = fresh_page
    set_factors(page, BLEND)
    choose(page, design_type="simplex_lattice")
    reveal(page, "#factors")
    assert page.locator("#factors thead th").all_text_contents()[2:4] == ["Lower", "Upper"]
    assert page.get_attribute("#factors tbody tr [name=low]", "placeholder") == "0"
    assert page.is_hidden("#factor-problems"), "a blank bound is 0 or 1"
    assert "a mixture's components are not" in page.text_content("#candidates")
    assert page.is_disabled('input[name="aim"][value="main_quadratic"]'), "a mixture's design sets its model"
    assert page.is_enabled("#generate")

    generate(page)
    sheet = run_sheet(page)
    assert len(sheet) == 7
    for row in sheet:
        assert sum(float(v) for v in row[2:5]) == pytest.approx(1.0, abs=1e-5), row
    assert "The {3, 2} simplex lattice" in page.text_content("#design-summary")
    assert page.input_value("#scenario") == "blend"
    assert page.locator("#scenario option:not([hidden])").all_text_contents() == ["A synergy and an antagonism"]

    practise(page)
    page.click("#go-learn")
    equation = page.text_content("#analysis .equation")
    assert equation.startswith("y = ")
    assert "Water·Oil" in equation
    assert "Intercept" not in page.text_content("#analysis")
    assert "Linear blending" in page.text_content("#analysis")
    assert "Effects along each Cox direction" in page.text_content("#analysis")
    assert "Lenth's method and the effects of two-level factors do not apply" in page.text_content("#analysis")
    assert charts(page) == [
        "Coefficients",
        "Fitted response over the blends",
        "True response over the blends",
        "Residuals against fitted values",
    ]
    types = page.evaluate(
        "[...document.querySelectorAll('#charts .js-plotly-plot')].map((d) => d.data.map((t) => t.type))"
    )
    assert types[1] == ["scatterternary", "scatterternary"], "the fitted response over the blends, and the runs"
    assert page.locator("#charts .views button").all_text_contents() == ["Estimates"], "no Pareto plot"
    shown = page.locator("#model option:not([hidden])").all_text_contents()
    assert shown == ["As designed", "Linear blending", "Quadratic blending", "Special cubic blending"]


def test_a_model_the_lattice_cannot_estimate_is_refused_saying_why(fresh_page):
    page = fresh_page
    set_factors(page, BLEND)
    generate(page, design_type="simplex_lattice", lattice_degree=1)
    assert len(run_sheet(page)) == 4, "the pure components and the centroid"
    practise(page)
    page.click("#go-learn")
    assert page.text_content("#analysis .equation").count("·") == 3, "a coefficient per component, and no blend"
    choose(page, model="scheffe_quadratic")
    settled(page)
    assert page.text_content("#analysis .error") == (
        "This design cannot fit the quadratic blending model: only 4 of its 6 coefficients can be estimated from "
        "these 4 runs. Choose a smaller model."
    )


def test_four_components_are_traced_along_their_cox_directions(fresh_page):
    page = fresh_page
    set_factors(page, [*BLEND, {"name": "Salt"}])
    generate(page, design_type="simplex_centroid")
    assert len(run_sheet(page)) == 15
    practise(page)
    page.click("#go-learn")
    assert "Fitted response along each Cox direction" in charts(page)


def test_bounds_choose_extreme_vertices_and_a_process_variable_is_refused(fresh_page):
    page = fresh_page
    set_factors(page, [{"name": "Water", "low": "0.2"}, {"name": "Oil", "high": "0.6"}, {"name": "Wax"}])
    choose(page, design_type="simplex_lattice")
    reveal(page, "#factors")
    assert "Water and Oil have bounds: choose extreme vertices" in page.text_content("#factor-problems")
    assert page.is_disabled("#generate")
    choose(page, design_type="extreme_vertices")
    assert page.is_hidden("#factor-problems")
    generate(page)
    assert "extreme vertices of the blends your bounds allow" in page.text_content("#design-summary")
    set_factors(page, [FACTORS[0], {"name": "Oil"}, {"name": "Wax"}])
    reveal(page, "#factors")
    assert "T runs from 150 to 200, which is no proportion" in page.text_content("#factor-problems")


def test_a_categorical_factor_or_a_limit_rules_out_a_mixture_saying_why(fresh_page):
    page = fresh_page
    set_factors(page, BLEND)
    choose(page, design_type="simplex_lattice")
    set_factors(page, [*BLEND[:2], {"name": "Supplier", "low": "old", "high": "new"}])
    disabled = "[...document.querySelectorAll('#design-type option')].filter((o) => o.disabled).map((o) => o.value)"
    assert {"simplex_lattice", "simplex_centroid", "extreme_vertices"} <= set(page.evaluate(disabled))
    assert page.text_content("#design-hint") == (
        "A mixture's factors are its components, each a proportion, and Supplier is categorical: remove it, or "
        "choose a design for factors with ranges."
    )
    assert page.is_disabled("#generate")
    page.locator("#factors tbody tr").nth(2).locator(".switch").click()  # numeric again: a component
    set_factors(page, BLEND)
    assert page.is_enabled("#generate")
    type_limit(page, "Water + Oil <= 0.8")
    assert page.text_content("#design-hint").startswith("A mixture takes no limits on the settings on this page")
    assert page.is_visible("#limits-mixture"), "the limits panel says so too"
    assert page.is_disabled("#generate")


def test_a_mixture_comes_back_from_a_saved_study_as_a_mixture(fresh_page, context, site):
    page = fresh_page
    made_mixture(page)
    page.wait_for_timeout(500)  # the study is saved a moment after its last change
    again = context.new_page()
    again.goto(site)
    again.wait_for_function("window.__pi.count('api_analyze_runs') >= 1 && window.__pi.idle()", timeout=30_000)
    assert again.input_value("#design-type") == "simplex_lattice"
    assert [row["name"] for row in factor_table(again)] == ["Water", "Oil", "Wax"]
    assert "Fitted response over the blends" in charts(again)


@pytest.mark.parametrize("context_options", [{"viewport": {"width": 390, "height": 844}, "is_mobile": True}])
def test_a_mixture_on_a_phone_scrolls_only_downwards(fresh_page):
    page = fresh_page
    page.add_style_tag(content=WIDE_FONT)
    made_mixture(page)
    page.click("#go-learn")
    charts(page)
    for phase in ("plan", "run", "learn"):
        page.click(f"#go-{phase}")
        assert overflowing(page) == [], phase
        assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth"), phase


def test_a_mixture_measures_two_responses_and_says_its_best_blend_is_not_searched(fresh_page):
    page = fresh_page
    name_response(page, "Hardness", "HV", goal="maximize")
    add_response(page, "Cost", "$/kg", "minimize")
    made_mixture(page)
    assert headers(page)[-2:] == ["Hardness", "Cost"], "a column for each response"
    page.click("#go-learn")
    settled(page)
    reveal(page, "#response-picker")
    assert tabs(page) == [["Hardness", "true"], ["Cost", "false"]]
    assert page.text_content("#analysis .equation").startswith("Hardness = ")
    assert page.text_content("#best-out") == (
        "The page does not yet search a mixture's blends for the one that best meets the goal for Hardness. "
        "The charts below show the fitted Hardness over the blends."
    )
    assert page.is_hidden("#best-chart")
    # The balance is asked for once the charts are drawn, which can be after the page reads as settled.
    page.wait_for_function("window.__pi.count('api_balance') > 0")
    assert calls(page, "api_balance"), "the page asked, and was told it is a mixture"
    assert page.is_hidden("#balance"), "two goals, and no balance searched over a mixture's blends"
    pick(page, "Cost")
    assert page.text_content("#analysis .equation").startswith("Cost = ")
    assert "Fitted response over the blends" in charts(page)
    assert "for Cost" in page.text_content("#best-out")
