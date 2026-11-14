"""The charts in Learn: the page's coefficient chart, then the library's own plots, drawn with Plotly.js.

Each chart sits beside a table that holds its numbers, so a chart that cannot be drawn costs
the picture and not the result.
"""

from __future__ import annotations

import re

import pytest

from tests.web._support import factors_named
from tests.web.e2e._page import (
    calls,
    enter_results,
    first_paint_done,
    generate,
    reveal,
    set_factors,
    status,
    type_results,
)

TITLES = ["Coefficients", "Fitted response surface", "True response surface", "Residuals against fitted values"]


def plots(page, box: str = "#charts") -> list[list[dict]]:
    """Return each drawn chart's traces, as Plotly holds them."""
    return page.evaluate(
        """(box) => [...document.querySelectorAll(`${box} .js-plotly-plot`)].map((d) => d.data.map((t) => ({
             type: t.type, name: t.name, zmin: t.zmin, zmax: t.zmax, size: t.contours?.size })))""",
        box,
    )


def test_the_charts_say_the_analysis_is_complete_after_python_s_note_on_them(blank_page):
    """Python says it is installing plotly while the first charts are on their way.

    The charts' end replaces that note.
    """
    page = blank_page
    generate(page, design_type="full_factorial", n_center_points=0)
    plotted = page.evaluate("window.__pi.count('api_plots')")
    page.evaluate("window.__pi.hold('api_plots')")
    enter_results(page, {i: 50 + i for i in range(8)})
    page.wait_for_function(f"window.__pi.count('api_plots') > {plotted}")
    page.evaluate("window.__pi.note('Installing plotly for the charts...')")
    assert status(page) == ("Installing plotly for the charts...", "busy")
    page.evaluate("window.__pi.release('api_plots')")
    page.wait_for_function("window.__pi.idle() && document.querySelectorAll('#charts .js-plotly-plot').length > 1")
    assert status(page) == ("Analysis complete.", "ok")


def test_each_chart_has_its_title_and_caption(page):
    assert page.locator("#charts figure h3").all_text_contents() == TITLES
    assert page.locator("#charts .chart.wide figcaption").count() == 0, "the coefficients' legend says it all"
    captions = page.locator("#charts figcaption").all_text_contents()
    assert "over T and P, the two factors with the largest main effects" in captions[0]


def test_the_coefficient_chart_marks_the_true_values(page):
    legend = coefficient_plot(page).evaluate(
        "d => [...d.data].sort((a, b) => a.legendrank - b.legendrank).map((t) => t.name)"
    )
    assert legend == ["Estimate", "95% interval", "True value"]
    assert plots(page)[0][0]["name"] == "True value", "drawn under the estimates"


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
    page.goto(f"{site}?example=worked")
    page.wait_for_function("document.querySelector('#charts p.hint')")
    assert "The charts could not be drawn" in page.text_content("#charts")
    assert "the tables above hold the numbers" in page.text_content("#charts")
    assert page.locator("#analysis table").count() >= 2, "the coefficients and the ANOVA"
    reveal(page, "#compare")
    page.click("#compare")
    page.wait_for_function("document.querySelector('#compare-charts p.hint')")
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
    assert [t["name"] for t in traces] == ["Positive", "Negative", "95% interval", "True size"]
    colours = coefficient_plot(page).evaluate("d => d.data.filter((t) => t.type === 'bar').map((t) => t.marker.color)")
    assert colours == ["#E69F00", "#0072B2"], "the book's orange and blue"
    sizes = coefficient_plot(page).evaluate("d => d.layout.yaxis.categoryarray")
    by_size = sorted(
        (c for c in first_paint["analysis"]["coefficients"] if c["term"] != "Intercept"),
        key=lambda c: -abs(c["coefficient"]),
    )
    assert sizes == [c["label"] for c in by_size], "the largest at the top"

    again = context.new_page()
    again.goto(f"{site}?example=worked")
    first_paint_done(again)
    assert view(again) == "Pareto plot", "the view chosen is kept in this browser"
    again.click("#charts .views button[data-view='estimates']")
    again.wait_for_function("document.querySelector('#charts .chart.wide .js-plotly-plot').data[0].type === 'scatter'")


def test_with_no_degrees_of_freedom_the_pareto_plot_draws_lenth_s_margins(blank_page):
    page = blank_page
    set_factors(page, factors_named(*"ABCDEFG"))
    generate(page, design_type="plackett_burman", n_center_points=0)
    type_results(page, {0: 12.1, 1: 15.4, 2: 9.8, 3: 14.0, 4: 11.2, 5: 13.3, 6: 10.1, 7: 16.0})
    page.wait_for_function("document.querySelectorAll('#charts .js-plotly-plot').length === 3")
    reveal(page, "#charts")
    page.click("#charts .views button[data-view='pareto']")
    page.wait_for_function("document.querySelector('#charts .chart.wide .js-plotly-plot').data[0].type === 'bar'")
    marks = coefficient_plot(page).evaluate("d => d.layout.annotations.map((a) => a.text)")
    assert marks == ["ME", "SME"]
    names = [t["name"] for t in plots(page)[0]]
    assert "Margin of error (ME)" in names, "the legend names the lines"
    assert "95% interval" not in names, "no degrees of freedom are left for an interval"


def test_charts_drawn_while_learn_was_hidden_fit_its_width(blank_page):
    """Results typed in Run are analysed and drawn there and then; showing Learn fits each chart to its card."""
    page = blank_page
    generate(page, design_type="box_behnken")
    type_results(page, {run: 60 + (run % 5) for run in range(15)})
    page.wait_for_function("document.querySelectorAll('#charts .js-plotly-plot').length === 3")
    reveal(page, "#charts")
    fitted = """() => [...document.querySelectorAll('#charts figure')].every((card) => {
         const chart = card.querySelector('.js-plotly-plot .main-svg').getBoundingClientRect().width;
         return chart <= card.clientWidth && chart >= 0.8 * card.clientWidth;
       })"""
    page.wait_for_function(fitted, timeout=5_000)  # Plotly fits a chart a moment after it is asked to


# --------------------------------------------------------------------------- more charts, and the surface's factors
MORE_TITLES = [
    "Each factor's effect on the prediction",
    "Average result at each setting",
    "Interaction of the two factors",
    "Fitted response surface, in three dimensions",
    "Predictions at the corners",
    "Residuals against the normal distribution",
    "Residuals in the order the runs were made",
    "Where the predictions are least certain",
]


def open_more(page) -> None:
    """Open "More charts" and wait until each of its charts is drawn."""
    page.click("#more-charts > summary")
    page.wait_for_function(
        "(n) => document.querySelectorAll('#more-charts .js-plotly-plot').length === n", arg=len(MORE_TITLES)
    )


def test_more_charts_are_drawn_when_their_heading_is_opened(page):
    assert page.text_content("#more-charts > summary") == "More charts (8)"
    assert page.locator("#more-charts figure").count() == 0, "a chart drawn while hidden has no width to fit"
    open_more(page)
    assert page.locator("#more-charts figure h3").all_text_contents() == MORE_TITLES
    assert all(caption.endswith(".") for caption in page.locator("#more-charts figcaption").all_text_contents())
    kinds = [traces[0]["type"] for traces in plots(page, "#more-charts")]
    assert kinds == ["scatter", "scatter", "scatter", "surface", "scatter3d", "scatter", "scatter", "contour"]


def test_each_factor_s_effect_is_drawn_with_the_band_of_its_interval_which_its_caption_explains(page):
    open_more(page)
    card = page.locator("figure[data-chart=perturbation]")
    drawn = card.locator(".plot").evaluate(
        "(plot) => plot.data.map((trace) => [trace.name ?? '', trace.meta ?? '', trace.fill ?? ''])"
    )
    assert [name for name, meta, _ in drawn if meta != "interval"] == ["T", "P", "F"]
    assert [fill for _, meta, fill in drawn if meta == "interval" and fill] == ["tonexty"] * 3, "a band a line"
    assert drawn[-1][:2] == ["95% interval", "interval"], "which the legend names"
    assert (
        card.locator("figcaption")
        .text_content()
        .endswith(
            "The band around a line is how well the model knows it: a wide band is a line that could lie elsewhere."
        )
    )


#: The first and the last colour of each chart's ramp, for the charts drawn in bands or shades of the response.
ENDS = """() => Object.fromEntries([...document.querySelectorAll('#charts figure.chart')].flatMap((card) => {
  const ramp = card.querySelector('.plot').data?.find((trace) => Array.isArray(trace.colorscale))?.colorscale;
  return ramp ? [[card.dataset.chart, [ramp[0][1], ramp.at(-1)[1]].map((colour) => colour.toLowerCase())]] : [];
}))"""

#: The ends of the library's ramp, near white for low, and of the dark page's, as dark as the page for low.
LIGHT_ENDS, DARK_ENDS = ["#eff6ff", "#1e3a8a"], ["#1a2436", "#bfdbfe"]
SURFACES = ["contour", "true_contour", "surface_3d", "prediction_variance"]


def test_on_a_dark_page_a_surface_is_dark_where_the_response_is_low_and_saved_as_a_light_page_s(page, tmp_path):
    """A ramp that starts in near white glares on a dark page, so there it starts dark. An image is on white."""
    open_more(page)
    assert page.evaluate(ENDS) == dict.fromkeys(SURFACES, LIGHT_ENDS)
    page.emulate_media(color_scheme="dark")
    page.wait_for_function(f"({ENDS})().prediction_variance[0] === '{DARK_ENDS[0]}'")
    assert page.evaluate(ENDS) == dict.fromkeys(SURFACES, DARK_ENDS)

    page.evaluate(
        """() => { const save = Plotly.toImage;
          Plotly.toImage = (figure, ...how) => { window.__saved = figure; return save(figure, ...how); }; }"""
    )
    page.click("figure[data-chart=contour] .expand")
    page.wait_for_function("document.querySelector('dialog.figure .plot.js-plotly-plot')")
    assert page.locator("dialog.figure .plot").evaluate("(plot) => plot.data[0].colorscale[0][1]") == DARK_ENDS[0]
    with page.expect_download() as svg:
        page.click("dialog.figure [data-do=svg]")
    svg.value.save_as(tmp_path / "figure.svg")
    saved = page.evaluate("[window.__saved.data[0].colorscale[0][1], window.__saved.data[0].contours.labelfont.color]")
    assert [colour.lower() for colour in saved] == [LIGHT_ENDS[0], "#1d1d1f"]


def rgb(colour: str) -> str:
    """Return a colour written ``#rrggbb`` as a browser writes it in a computed style."""
    return "rgb({}, {}, {})".format(*(int(colour[at : at + 2], 16) for at in (1, 3, 5)))


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_the_numbers_on_a_surface_s_lines_have_a_halo_in_the_page_s_colour(page, scheme):
    """A ramp runs from as light as the page to as dark as its ink: no one colour reads on every band without it."""
    page.emulate_media(color_scheme=scheme)
    open_more(page)
    ink, panel = page.evaluate(
        "['--fg', '--panel'].map((name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim())"
    )
    for chart in ("contour", "prediction_variance"):
        plot = page.locator(f"figure[data-chart={chart}] .plot")
        page.wait_for_function(
            "([plot, ink]) => plot.data[0].contours.labelfont?.color === ink", arg=[plot.element_handle(), ink]
        )
        font = plot.evaluate("(plot) => plot.data[0].contours.labelfont")
        assert font["shadow"].count(panel) == 4, "a shadow to each corner"
        labels = plot.locator(".contourlabels text")
        assert labels.count() >= 3
        drawn = labels.first.evaluate("(label) => [getComputedStyle(label).fill, getComputedStyle(label).textShadow]")
        assert drawn[0] == rgb(ink)
        assert drawn[1].count(rgb(panel)) == 4, "the browser draws it"


def test_the_reader_chooses_the_two_factors_the_surface_is_drawn_over(page):
    assert [page.input_value(f"#contour-factor-{at}") for at in (1, 2)] == ["T", "P"]
    open_more(page)
    plotted = page.evaluate("window.__pi.count('api_plots')")
    page.select_option("#contour-factor-2", "F")
    page.wait_for_function(f"window.__pi.count('api_plots') > {plotted} && window.__pi.idle()")
    page.wait_for_function("document.querySelector('#charts figcaption').textContent.includes('you chose')")
    assert calls(page, "api_plots")[-1]["payload"]["contour_factors"] == ["T", "F"]
    assert "over T and F, the two factors you chose" in page.locator("#charts figcaption").first.text_content()
    assert page.locator("#more-charts").evaluate("(box) => box.open"), "it stays open as the charts are drawn again"
    page.wait_for_function(
        "(n) => document.querySelectorAll('#more-charts .js-plotly-plot').length === n", arg=len(MORE_TITLES)
    )


def test_choosing_the_factor_the_other_box_holds_makes_the_two_change_places(page):
    page.select_option("#contour-factor-1", "P")
    page.wait_for_function("document.querySelector('#charts figcaption').textContent.includes('you chose')")
    assert [page.input_value(f"#contour-factor-{at}") for at in (1, 2)] == ["P", "T"]


def test_a_button_takes_the_reader_s_choice_of_factors_back_to_the_two_largest(page):
    assert page.locator("#contour-largest").count() == 0, "nothing is chosen, so there is nothing to take back"
    page.select_option("#contour-factor-2", "F")
    page.wait_for_function("document.querySelector('#charts figcaption').textContent.includes('you chose')")
    plotted = page.evaluate("window.__pi.count('api_plots')")
    page.click("#contour-largest")
    page.wait_for_function(f"window.__pi.count('api_plots') > {plotted} && window.__pi.idle()")
    page.wait_for_function("document.querySelector('#charts figcaption').textContent.includes('largest main effects')")
    assert "contour_factors" not in calls(page, "api_plots")[-1]["payload"]
    assert [page.input_value(f"#contour-factor-{at}") for at in (1, 2)] == ["T", "P"]
    assert page.locator("#contour-largest").count() == 0
    page.wait_for_function("document.activeElement.id === 'contour-factor-1'")  # the button went with the choice
