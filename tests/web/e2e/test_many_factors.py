"""Studies of more than 12 factors in the browser: the brief, the run sheet, Learn, a phone and the catalog.

A reader can brief a screening design of up to 40 factors. Past 12 the page models no
interactions, and each part that keeps a lower cap of its own says so on the page, in a sentence:
the aims and the fraction's resolution, the comparison of the designs, the trade-off grid, the
models Learn fits, and in the catalog the correlation map and the comparison of designs.
"""

from __future__ import annotations

import pytest

from process_improve.experiments.designs_screening import plackett_burman_runs
from tests.web._support import MANY_FACTOR_DESIGNS
from tests.web.e2e._page import (
    WIDE_FONT,
    calls,
    candidates_done,
    choose,
    generate,
    overflowing,
    practise,
    reveal,
    run_sheet,
    saved_studies,
    set_factors,
    settled,
)

#: The numbers of factors the page is tested at past its old cap of 12: one more, 24, and the most it takes.
MANY = (13, 24, 40)

GIVEN_31, ARRAY_40 = MANY_FACTOR_DESIGNS


def factors(k: int) -> list[dict]:
    """Return ``k`` numeric factors, X1 to Xk, each from 0 to 10 u."""
    return [{"name": f"X{i + 1}", "low": 0, "high": 10, "units": "u"} for i in range(k)]


def disabled(page, selector: str) -> list[str]:
    """Return the values of the options or inputs ``selector`` finds that cannot be chosen."""
    return page.evaluate(
        "(s) => [...document.querySelectorAll(s)].filter((o) => o.disabled).map((o) => o.value)", selector
    )


def coefficients(page) -> list[str]:
    """Return the terms of the coefficients table Learn shows, the intercept first."""
    table = page.locator("#analysis table").first
    return [row.locator("td").first.text_content() for row in table.locator("tbody tr").all()]


@pytest.mark.parametrize("k", MANY)
def test_a_screening_design_of_many_factors_is_briefed_made_and_analysed(fresh_page, bootstrap, k):
    page = fresh_page
    set_factors(page, factors(k))
    candidates_done(page)
    designs = [*bootstrap.DESIGNS.items(), *bootstrap.MIXTURES.items()]
    assert disabled(page, "#design-type option") == [
        d for d, i in designs if not i["min_factors"] <= k <= i["max_factors"]
    ]
    # The aims past 12 factors: the main effects alone, and why.
    assert disabled(page, 'input[name="aim"]') == ["interactions", "main_quadratic", "quadratic"]
    assert page.is_checked('input[name="aim"][value="main_effects"]')
    said = page.text_content("#aim-terms")
    assert (
        f"Past 12 factors, a study finds which factors matter, and a fraction is made at resolution III or IV: {k} "
        f"factors have {k * (k - 1) // 2} two-factor interactions" in said
    )
    # The designs listed: scored on power alone, and why; nothing more to compare in detail.
    assert "Past 12 factors, the designs are compared on their runs, spare runs and power alone" in page.text_content(
        "#candidates"
    )
    assert page.is_hidden("#compare")
    generate(page, design_type="plackett_burman")
    assert len(run_sheet(page)) == plackett_burman_runs(k) + int(page.input_value("#n_center_points"))
    practise(page)
    reveal(page, "#model")
    assert disabled(page, "#model option") == ["interactions", "quadratic"]
    assert page.text_content("#model-many").startswith(
        "Past 12 factors, Learn fits the main effects, or the terms you choose:"
    )
    assert coefficients(page) == ["Intercept", *(f["name"] for f in factors(k))]
    saved = saved_studies(page, lambda studies: bool(studies) and len(studies[0]["design"]["spec"]["factors"]) == k)
    assert saved[0]["v"] == 4, "saved in the format an earlier Factorial refuses as a later one's"


def test_the_brief_stops_at_40_factors_and_a_fraction_at_24_each_saying_so(fresh_page):
    page = fresh_page
    set_factors(page, factors(40))
    assert page.is_disabled("#add-factor")
    assert page.is_disabled("#add-categorical")
    assert page.text_content("#factors-full") == "A study has at most 40 factors."
    assert page.is_visible("#factors-full")
    choose(page, design_type="fractional_factorial")
    assert (
        page.text_content("#design-hint")
        == "Fractional factorial (2-level) takes 3 to 24 factors on this page; you have 40."
    )
    page.locator("#factors tbody tr").last.get_by_role("button", name="Remove factor").click()
    assert not page.is_disabled("#add-factor")
    assert page.is_hidden("#factors-full")


def test_a_fraction_of_24_factors_stops_at_resolution_iv_and_the_grid_says_where_it_stops(fresh_page):
    page = fresh_page
    set_factors(page, factors(24))
    choose(page, design_type="fractional_factorial")
    assert disabled(page, "#resolution option") == ["5"]
    assert page.input_value("#resolution") == "4"
    assert page.text_content("#trade-off-hint") == (
        "The grid shows fractions in 3 to 11 factors; the page finds the fraction in your 24 when it makes the design."
    )
    generate(page)
    assert calls(page, "api_make_design")[-1]["payload"]["resolution"] == 4
    assert len(run_sheet(page)) == 64 + 3, "the 2^(24-18) fraction of resolution IV, and its center runs"


def test_past_12_factors_learn_fits_an_interaction_the_reader_names(fresh_page):
    page = fresh_page
    set_factors(page, factors(13))
    generate(page, design_type="dsd")  # its main effects are clear of every interaction, so X1·X2 can join them
    practise(page)
    choose(page, model="terms")
    settled(page)
    assert page.is_visible("#term-adder")
    assert page.locator("#term-list h4").all_text_contents() == ["Main effects", "Squares"], "no interaction yet"
    page.select_option("#term-first", "X1")
    page.select_option("#term-second", "X1")
    assert page.is_disabled("#add-term"), "a factor has no interaction with itself"
    page.select_option("#term-second", "X2")
    before = page.evaluate("window.__pi.count('api_analyze_runs')")
    page.click("#add-term")
    page.wait_for_function(f"window.__pi.count('api_analyze_runs') > {before}")
    settled(page)
    assert ["X1", "X2"] in calls(page, "api_analyze_runs")[-1]["payload"]["model"]["terms"]
    assert page.locator("#term-list h4").all_text_contents() == ["Main effects", "Interactions you name", "Squares"]
    assert page.get_by_label("X1·X2", exact=True).is_checked()
    assert "X1·X2" in coefficients(page)


@pytest.mark.parametrize("k", [24, 40])
@pytest.mark.parametrize("context_options", [{"viewport": {"width": 390, "height": 844}, "is_mobile": True}])
def test_on_a_phone_the_brief_and_the_run_sheet_of_many_factors_scroll_only_downwards(fresh_page, k):
    page = fresh_page
    page.add_style_tag(content=WIDE_FONT)
    set_factors(page, factors(k))
    candidates_done(page)
    generate(page, design_type="plackett_burman")
    practise(page)
    for phase in ("plan", "run", "learn"):
        page.click(f"#go-{phase}")
        # A chart drawn while its phase was hidden takes Plotly's own width until the phase shows and fits it.
        page.wait_for_function("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
        assert overflowing(page) == [], phase


# --------------------------------------------------------------------------- the catalog
def open_catalog(context, site: str, fragment: str = "") -> object:
    """Open the catalog, and wait until it lists its designs."""
    page = context.new_page()
    page.goto(f"{site}catalog/{fragment}")
    page.wait_for_selector(".result")
    return page


def test_a_catalog_design_of_31_factors_says_where_its_detail_stops_and_opens_as_given(many_context, site):
    page = open_catalog(many_context, site, f"#d={GIVEN_31}")
    page.wait_for_selector("#detail[open]")
    text = " ".join(page.text_content("#detail-body").split())
    assert (
        "The map of the correlations between a design's terms is drawn for up to 12 factors: this design's 31 have "
        "527 terms, too many to read in one map." in text
    )
    assert page.locator(f"#detail-body [data-compare='{GIVEN_31}']").count() == 0
    assert (
        "It cannot join a comparison: the catalog compares designs whose factors are all numeric, with at most 12 "
        "factors and 1000 runs, since it scores each on models up to the full quadratic" in text
    )
    page.click(f"#detail-body [data-open='{GIVEN_31}']")
    page.wait_for_selector("#campaign-banner", state="visible")
    page.wait_for_function("window.__pi.count('api_catalogue') === 1 && window.__pi.idle()")
    settled(page)
    assert len(run_sheet(page)) == 32
    assert page.input_value("#design-type") == "given"
    assert page.locator("#factors tbody tr").count() == 31


def test_a_catalog_design_of_40_factors_opens_as_the_page_makes_it(many_context, site):
    page = open_catalog(many_context, site, f"#d={ARRAY_40}")
    page.wait_for_selector("#detail[open]")
    page.click(f"#detail-body [data-open='{ARRAY_40}']")
    page.wait_for_selector("#campaign-banner", state="visible")
    page.wait_for_function("window.__pi.count('api_catalogue') === 1 && window.__pi.idle()")
    settled(page)
    assert len(run_sheet(page)) == 81
    assert page.input_value("#design-type") == "taguchi"
    assert page.locator("#factors tbody tr").count() == 40
