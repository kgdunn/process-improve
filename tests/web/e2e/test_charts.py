"""The charts of step 3: the page's coefficient chart, then the library's own plots, drawn with Plotly.js.

Each chart sits beside a table that holds its numbers, so a chart that cannot be drawn costs
the picture and not the result.
"""

from __future__ import annotations

import re

from tests.web._support import factors_named
from tests.web.e2e._page import generate, set_factors, type_results

TITLES = ["Coefficients", "Fitted response surface", "True response surface", "Residuals against fitted values"]


def plots(page, box: str = "#charts") -> list[list[dict]]:
    """Return each drawn chart's traces, as Plotly holds them."""
    return page.evaluate(
        """(box) => [...document.querySelectorAll(`${box} .js-plotly-plot`)].map((d) => d.data.map((t) => ({
             type: t.type, name: t.name, zmin: t.zmin, zmax: t.zmax, size: t.contours?.size })))""",
        box,
    )


def test_each_chart_has_its_title_and_caption(page):
    assert page.locator("#charts figure h3").all_text_contents() == TITLES
    captions = page.locator("#charts figcaption").all_text_contents()
    assert "over T and P, the two factors with the largest main effects" in captions[1]


def test_the_coefficient_chart_marks_the_true_values(page):
    assert [trace["name"] for trace in plots(page)[0]] == ["Estimate, with its 95% interval", "True value"]


def test_the_fitted_and_true_surfaces_share_one_colour_scale(page):
    fitted, true = plots(page)[1][0], plots(page)[2][0]
    assert fitted["type"] == true["type"] == "contour"
    assert (fitted["zmin"], fitted["zmax"], fitted["size"]) == (true["zmin"], true["zmax"], true["size"])


def test_a_saturated_two_level_design_gets_the_half_normal_plot(blank_page):
    page = blank_page
    set_factors(page, factors_named(*"ABCDEFG"))
    generate(page, design_type="plackett_burman", n_center_points=0)
    type_results(page, {run: 10 + 3 * (run % 2) - (run % 3) for run in range(8)})
    page.wait_for_function("document.querySelectorAll('#charts .js-plotly-plot').length === 3")
    assert page.locator("#charts figure h3").all_text_contents() == [
        "Coefficients",
        "Fitted response surface",
        "Half-normal plot of the effects",
    ]


def test_without_plotly_the_tables_still_hold_the_numbers(context, site, console):
    console.allow(r"plotly|ERR_FAILED")
    context.route(re.compile(r"/plotly\.min\.js$"), lambda route: route.abort())
    page = context.new_page()
    page.goto(site)
    page.wait_for_function("document.querySelector('#compare-charts p.hint')")
    assert "The charts could not be drawn" in page.text_content("#charts")
    assert "the tables above hold the numbers" in page.text_content("#charts")
    assert page.locator("#analysis table").count() >= 2, "the coefficients and the ANOVA"
    assert "The charts could not be drawn" in page.text_content("#compare-charts")
