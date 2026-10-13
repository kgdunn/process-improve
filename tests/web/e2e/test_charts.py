"""The charts of step 3: the page's coefficient chart, then the library's own plots, drawn with Plotly.js.

Each chart sits beside a table that holds its numbers, so a chart that cannot be drawn costs
the picture and not the result.
"""

from __future__ import annotations

import re

import pytest

from tests.web._support import factors_named
from tests.web.e2e._page import first_paint_done, generate, set_factors, type_results

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


# --------------------------------------------------------------------------- the coefficients, two ways
def coefficient_plot(page):
    return page.locator("#charts .chart.wide .js-plotly-plot").first


def term_labels(page) -> list[str]:
    """Return the term labels the coefficient chart shows down its side."""
    return coefficient_plot(page).locator(".ytick text").all_text_contents()


def view(page) -> str:
    return page.locator("#charts .views button[aria-pressed='true']").text_content()


@pytest.mark.parametrize("context_options", [{"viewport": {"width": 390, "height": 844}, "is_mobile": True}])
def test_on_a_phone_every_term_is_labelled_and_the_legend_clears_the_axis(page, first_paint):
    terms = [c["label"] for c in first_paint["analysis"]["coefficients"] if c["term"] != "Intercept"]
    shown = term_labels(page)
    assert sorted(shown) == sorted(terms), "every term labelled, none skipped for want of room"
    legend = coefficient_plot(page).locator(".legend").bounding_box()
    title = coefficient_plot(page).locator(".xtitle").bounding_box()
    assert legend["y"] + legend["height"] <= title["y"], "the legend sits above the plot, the axis title below"


def test_the_coefficients_switch_to_the_book_s_pareto_plot_and_back(context, page, site, first_paint):
    assert view(page) == "Estimates"
    page.click("#charts .views button[data-view='pareto']")
    page.wait_for_function("document.querySelector('#charts .chart.wide .js-plotly-plot').data[0].type === 'bar'")
    assert view(page) == "Pareto plot"
    traces = plots(page)[0]
    assert [t["name"] for t in traces] == ["Positive", "Negative", "True size"]
    colours = coefficient_plot(page).evaluate("d => d.data.filter((t) => t.type === 'bar').map((t) => t.marker.color)")
    assert colours == ["#E69F00", "#0072B2"], "the book's orange and blue"
    sizes = coefficient_plot(page).evaluate("d => d.layout.yaxis.categoryarray")
    by_size = sorted(
        (c for c in first_paint["analysis"]["coefficients"] if c["term"] != "Intercept"),
        key=lambda c: -abs(c["coefficient"]),
    )
    assert sizes == [c["label"] for c in by_size], "the largest at the top"
    caption = page.locator("#charts .chart.wide figcaption").text_content()
    assert "as the book's Pareto plot draws them" in caption
    assert "95% interval" in caption, "the worked example leaves degrees of freedom for intervals"

    again = context.new_page()
    again.goto(site)
    first_paint_done(again, first_paint)
    assert view(again) == "Pareto plot", "the view chosen is kept in this browser"
    again.click("#charts .views button[data-view='estimates']")
    again.wait_for_function("document.querySelector('#charts .chart.wide .js-plotly-plot').data[0].type === 'scatter'")


def test_with_no_degrees_of_freedom_the_pareto_plot_draws_lenth_s_margins(blank_page):
    page = blank_page
    set_factors(page, factors_named(*"ABCDEFG"))
    generate(page, design_type="plackett_burman", n_center_points=0)
    type_results(page, {0: 12.1, 1: 15.4, 2: 9.8, 3: 14.0, 4: 11.2, 5: 13.3, 6: 10.1, 7: 16.0})
    page.wait_for_function("document.querySelectorAll('#charts .js-plotly-plot').length === 3")
    page.click("#charts .views button[data-view='pareto']")
    page.wait_for_function("document.querySelector('#charts .chart.wide .js-plotly-plot').data[0].type === 'bar'")
    marks = coefficient_plot(page).evaluate("d => d.layout.annotations.map((a) => a.text)")
    assert marks == ["ME", "SME"]
    assert "Lenth's margins of error" in page.locator("#charts .chart.wide figcaption").text_content()
