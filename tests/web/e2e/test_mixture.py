"""A mixture in the browser: its components typed into the brief, its run sheet of blends, and its analysis.

A reader names the components of a blend, leaves their bounds blank (0 and 1), chooses a simplex
lattice, makes the run sheet, practises with simulated results and reads the Scheffé model's
analysis, with the fitted response drawn over the triangle of blends.
"""

from __future__ import annotations

import pytest

from tests.web._support import FACTORS
from tests.web.e2e._page import (
    choose,
    factor_table,
    generate,
    headers,
    last_reply,
    name_response,
    practise,
    reveal,
    run_sheet,
    set_factors,
    settled,
    status,
    type_limit,
)
from tests.web.e2e.test_best_settings import num, table_rows
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
    shown = page.locator("#model option").all_text_contents()
    assert shown == ["As designed", "Linear blending", "Quadratic blending", "Special cubic blending"], (
        "the models a mixture takes, and no other in the list: an option cannot be hidden in every browser"
    )
    assert page.locator("#scenario option").all_text_contents() == ["A synergy and an antagonism"]


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
    # Beside it, the triangle of the three that move the response most, the fourth held, and the reader's own three.
    assert "Fitted response over the blends" in charts(page)
    first = [page.input_value(f"#triangle-component-{i}") for i in (1, 2, 3)]
    held = next(name for name in [b["name"] for b in BLEND] + ["Salt"] if name not in first)
    caption = page.locator("figure[data-chart=ternary] figcaption").text_content()
    assert f"{held} held at" in caption
    assert "the other three share the rest" in caption
    page.select_option("#triangle-component-1", held)
    page.wait_for_function(f"document.querySelector('#triangle-component-1')?.value === {held!r}")
    page.wait_for_selector("#triangle-largest")
    assert held not in page.locator("figure[data-chart=ternary] figcaption").text_content().split(" held at")[0]
    page.click("#triangle-largest")
    page.wait_for_selector("#triangle-largest", state="detached")
    assert [page.input_value(f"#triangle-component-{i}") for i in (1, 2, 3)] == first


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


def blend_stars(page) -> list[list[float]]:
    """Return where each chart of Learn puts the best blend's star: its proportions on the triangle."""
    return page.evaluate(
        """() => [...document.querySelectorAll('#charts .js-plotly-plot')].flatMap((div) =>
             div.data.filter((t) => t.name === 'Best blend').map((t) => [t.a[0], t.b[0], t.c[0]]))"""
    )


def test_a_mixture_measures_two_responses_and_finds_each_one_s_best_blend(fresh_page):
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
    best = last_reply(page, "api_best_settings")["result"]
    reveal(page, "#best")
    assert page.text_content("#h-best") == "The best blend for your goal"
    assert page.text_content("#best .lead") == (
        f"This blend gives the highest Hardness the model predicts: {num(best['predicted'])} HV."
    )
    assert headers(page, "#best-out table")[:3] == ["Component", "Proportion", "Where it lies"]
    rows = table_rows(page, "#best-out table.data:not(.known-table)")
    assert [row[:2] for row in rows] == [[s["name"], num(s["value"])] for s in best["settings"]]
    assert sum(s["value"] for s in best["settings"]) == pytest.approx(1.0)
    assert page.text_content("#best-chart .term") == "How tightly to weigh out each component"
    assert "Fitted response over the blends" in charts(page)
    assert page.is_visible("#best-chart")
    assert blend_stars(page) == [[best["coded"][n] for n in ("Water", "Oil", "Wax")]], "on the fitted triangle"
    caption = page.locator("#charts figure", has_text="Fitted response over the blends").locator("figcaption")
    assert caption.text_content().endswith("the white dots are the runs, and the star the best blend.")
    assert "DesignRegion(factors)" in page.text_content("#py-best code")
    # The balance is asked for once the charts are drawn, which can be after the page reads as settled.
    page.wait_for_selector("#balance-chart .js-plotly-plot")
    b = last_reply(page, "api_balance")["result"]
    reveal(page, "#balance")
    assert page.text_content("#balance .lead").startswith(
        "This blend balances the responses best: an overall desirability"
    )
    assert headers(page, "#balance-out table")[:3] == ["Component", "Proportion", "Where it lies"]
    traces = page.evaluate(
        "() => document.querySelector('#balance-chart .js-plotly-plot').data.map((t) => ({name: t.name, type: t.type, "
        "a: t.a, b: t.b, c: t.c}))"
    )
    assert {t["type"] for t in traces} == {"scatterternary"}, "the overlay is over the triangle"
    star = next(t for t in traces if t["name"] == "Balanced blend")
    assert [star["a"][0], star["b"][0], star["c"][0]] == b["overlay"]["at"]
    shaded = next(t for t in traces if t["name"] == "Every response acceptable")
    assert len(shaded["a"]) == sum(b["overlay"]["acceptable"])
    assert page.text_content("#balance-chart figcaption").startswith("Each corner is a pure component and each point")
    assert page.text_content("#add-balance-confirmation") == "Add 3 confirmation runs at this blend"
    assert "region=DesignRegion(factors)" in page.text_content("#py-balance code")
    pick(page, "Cost")
    assert page.text_content("#analysis .equation").startswith("Cost = ")
    assert "Fitted response over the blends" in charts(page)
    assert page.text_content("#best .lead").startswith("This blend gives the lowest Cost the model predicts: ")


def test_confirmation_runs_at_the_best_blend_are_weighed_out_simulated_and_judged(fresh_page):
    page = fresh_page
    name_response(page, "Hardness", "HV", goal="maximize")
    made_mixture(page)
    page.click("#go-learn")
    settled(page)
    reveal(page, "#best")
    at = last_reply(page, "api_best_settings")["result"]["confirm_at"]
    assert sum(round(v * 1000) for v in at.values()) == 1000, "weighed out to a thousandth, summing to 1"
    assert page.text_content("#add-confirmation") == "Add 3 confirmation runs at this blend"
    page.click("#add-confirmation")
    settled(page)
    assert status(page)[0] == "3 confirmation runs added at the best blend, their results simulated."
    reveal(page, "#design-table")
    names = headers(page)
    rows = [row for row in run_sheet(page) if row[names.index("Confirmation run")] == "yes"]
    assert [[float(row[names.index(n)]) for n in ("Water", "Oil", "Wax")] for row in rows] == [list(at.values())] * 3
    reveal(page, "#best")
    assert page.text_content("#confirm-out .hint").startswith(
        "The confirmation runs weigh out each proportion to 0.001"
    )
    verdict = page.locator("#confirm-out .verdict")
    assert verdict.count() == 1
    assert verdict.get_attribute("data-state") in {"inside", "outside"}
    assert "this blend" in verdict.text_content()
    assert page.is_hidden("#add-confirmation"), "runs at this blend are on the sheet"


# --------------------------------------------------------------------------- a target and a specification
#: The names of a chart's traces that its legend lists, and the kinds of trace it is drawn with.
LISTED = "(plot) => plot.data.filter((trace) => trace.showlegend !== false).map((trace) => trace.name)"
KINDS = "(plot) => [...new Set(plot.data.map((trace) => trace.type))]"


def test_four_components_have_a_target_and_a_specification_drawn_over_three_the_others_held(fresh_page):
    """The triangle goes through the blend found; one picker, under each chart, sets the three of both."""
    page = fresh_page
    set_factors(page, [*BLEND, {"name": "Salt"}])
    generate(page, design_type="simplex_centroid")
    practise(page)
    page.wait_for_selector("#target-goals tbody tr", state="attached")
    reveal(page, "#target")
    settled(page)
    response = page.get_attribute("#target-goals tbody tr", "data-response")
    slider = page.locator('#target-goals [name="slider"]')
    middle = (float(slider.get_attribute("min")) + float(slider.get_attribute("max"))) / 2
    page.fill('#target-goals [name="target"]', f"{middle:.4g}")
    page.wait_for_selector("#target-chart .js-plotly-plot")
    found = last_reply(page, "api_target_settings")["result"]
    overlay = found["overlay"]
    plot = page.locator("#target-chart .plot")
    assert plot.evaluate(KINDS) == ["scatterternary"]
    caption = page.text_content("#target-chart figcaption")
    (held,) = overlay["held"]
    assert caption.startswith(f"Each point is a blend with {held} held at ")
    shown = [page.input_value(f"#target-triangle-component-{i}") for i in (1, 2, 3)]
    assert shown == overlay["components"]
    assert page.locator("#target-out .notes, #target-out ul").filter(has_text="no such blend").count() == 0

    reveal(page, "#window")
    page.fill(f'#window-limits tr[data-response="{response}"] [name="low"]', f"{middle:.4g}")
    page.wait_for_selector("#window-chart .js-plotly-plot")
    assert page.locator("#window-chart .plot").evaluate(KINDS) == ["scatterternary"]
    # Pick the held component into the triangle: both charts are found again over the new three.
    page.select_option("#window-triangle-component-1", held)
    page.wait_for_function(
        f"window.__pi.replies.filter((r) => r.fn === 'api_specification').at(-1)?.reply?.result?.overlay?.components?.[0] === {held!r}"
    )
    page.wait_for_function(
        f"window.__pi.replies.filter((r) => r.fn === 'api_target_settings').at(-1)?.reply?.result?.overlay?.components?.[0] === {held!r}"
    )
    page.wait_for_selector("#window-triangle-largest")
    assert page.input_value("#target-triangle-component-1") == held, "one picker's three, under every chart"
    page.click("#window-triangle-largest")
    page.wait_for_selector("#window-triangle-largest", state="detached")


def test_a_mixture_has_the_blend_for_a_target_and_the_blends_that_meet_a_specification(fresh_page):
    """Both questions are answered in blends: no factor has a range to move alone, and no what-if panel holds one."""
    page = fresh_page
    made_mixture(page)
    page.wait_for_selector("#target-goals tbody tr", state="attached")  # the sections come once Learn has its fits
    reveal(page, "#target")
    settled(page)
    response = page.get_attribute("#target-goals tbody tr", "data-response")
    slider = page.locator('#target-goals [name="slider"]')
    middle = (float(slider.get_attribute("min")) + float(slider.get_attribute("max"))) / 2
    page.fill('#target-goals [name="target"]', f"{middle:.4g}")
    page.wait_for_selector("#target-out .lead")
    found = last_reply(page, "api_target_settings")["result"]
    assert page.text_content("#target-out .lead") == "This blend gives the target, by the models."
    assert headers(page, "#target-out table")[:3] == ["Component", "Proportion", "Where it lies"]
    assert sum(s["value"] for s in found["settings"]) == pytest.approx(1.0, abs=1e-9)
    assert "blends tried give the target" in page.text_content("#target-out")
    page.wait_for_selector("#target-chart .js-plotly-plot")
    plot = page.locator("#target-chart .plot")
    assert plot.evaluate(KINDS) == ["scatterternary"], "drawn over the triangle of the three components"
    assert plot.evaluate(LISTED)[-2:] == ["Runs you made", "Blend found"]
    assert page.text_content("#target-chart figcaption").startswith("Each corner is a pure component")
    assert page.locator("#target-chart .trade-view").is_hidden(), "the triangle has no two factors to pick"
    assert page.text_content("#target-confirm") == "Add 3 confirmation runs at this blend"
    for part in ("#target-what-if", "#target-solve"):
        assert page.locator(part).is_hidden(), part
    assert "DesignRegion(factors).sample(" in page.text_content("#py-target code"), "the blends, as the page draws them"

    reveal(page, "#window")
    page.fill(f'#window-limits tr[data-response="{response}"] [name="low"]', f"{middle:.4g}")
    page.wait_for_selector("#window-out .lead")
    judged = last_reply(page, "api_specification")["result"]
    lead = page.text_content("#window-out .lead")
    assert lead.startswith("By the models, ")
    assert lead.endswith(" of the blends within your bounds give the response inside its window.")
    assert 0 < judged["share"] < 1
    page.wait_for_selector("#window-chart .js-plotly-plot")
    plot = page.locator("#window-chart .plot")
    assert plot.evaluate(KINDS) == ["scatterternary"]
    listed = plot.evaluate(LISTED)
    assert listed[0] == "Every response inside its window"
    assert listed[-2:] == ["Runs you made", "Surest blend"]
    assert page.text_content("#window-confirm") == "Add 3 confirmation runs at the surest blend"
    assert page.locator("#window-what-if").is_hidden()
    assert "feasible_region(models, windows, tried" in page.text_content("#py-window code"), "judged as the page does"
    before = len(run_sheet(page))
    page.click("#window-confirm")
    page.wait_for_function(f"document.querySelectorAll('#design-table tbody tr').length === {before + 3}")
    settled(page)
    added = dict(zip(headers(page), run_sheet(page)[-1], strict=False))
    assert sum(float(added[f["name"]]) for f in BLEND) == pytest.approx(1.0, abs=1e-9), "weighed out to sum to 1"
    assert [added[f["name"]] for f in BLEND] == [num(v) for v in judged["surest"]["confirm_at"].values()]
