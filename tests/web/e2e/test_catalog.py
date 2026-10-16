"""The design catalog in the browser: search and filter the designs, read one, and open it in Factorial.

The catalog served here is the build's, for the few designs of ``CATALOG_SAMPLE``: a published
experiment the page regenerates, with its results; a comparison; a generated design the page makes;
and one it does not.
"""

from __future__ import annotations

import pytest

from tests.web._support import CATALOG_SAMPLE
from tests.web.e2e._page import run_sheet, settled

PILOT, COMPARISON, CCD, OMARS, CUBE = CATALOG_SAMPLE


def open_catalog(context, site: str, fragment: str = "") -> object:
    """Open the catalog, and wait until it lists its designs."""
    page = context.new_page()
    page.goto(f"{site}catalog/{fragment}")
    page.wait_for_selector(".result")
    return page


def shown(page) -> list[str]:
    """Return the designs the list shows, in order, by their ids."""
    return page.eval_on_selector_all(".result__title", "titles => titles.map((t) => t.dataset.id)")


def counted(page, text: str) -> None:
    page.wait_for_function(f"document.querySelector('#count').textContent.startsWith({text!r})")


def test_the_app_s_header_opens_the_catalog(fresh_page):
    fresh_page.click("#catalog-link")
    fresh_page.wait_for_url("**/catalog/")
    fresh_page.wait_for_selector(".result")
    assert fresh_page.text_content("#count") == "5 of 5 designs: 2 published, 3 generated"
    assert set(shown(fresh_page)) == {PILOT, COMPARISON, CCD, OMARS, CUBE}
    assert shown(fresh_page)[-1] == COMPARISON, "a comparison is listed after the designs"


def test_its_address_without_the_slash_leads_to_it(context, site):
    """factori.al/catalog is the folder's address: the server sends it on to factori.al/catalog/."""
    page = context.new_page()
    page.goto(f"{site}catalog")
    page.wait_for_selector(".result")
    assert page.url == f"{site}catalog/"


def test_the_toggle_and_the_search_narrow_the_list_and_the_address_keeps_them(context, site):
    page = open_catalog(context, site)
    page.click("#collection [data-value=generated]")
    counted(page, "3 of 5")
    assert set(shown(page)) == {CCD, OMARS, CUBE}
    page.fill("#query", "quadratic")
    counted(page, "1 of 5")
    assert shown(page) == [CCD]
    assert "col=generated" in page.url
    assert "q=quadratic" in page.url


def test_a_filtered_address_opens_with_its_filters(context, site):
    page = open_catalog(context, site, "#q=pilot&col=published")
    assert shown(page) == [PILOT]
    assert page.input_value("#query") == "pilot"
    assert page.get_attribute("#collection [data-value=published]", "aria-checked") == "true"


def test_a_family_s_count_is_what_ticking_it_shows(context, site):
    page = open_catalog(context, site)
    family = page.locator("#families label", has=page.locator("input[value=full_factorial]"))
    assert family.locator(".count").text_content() == "2"
    family.locator("input").check()
    counted(page, "2 of 5")
    assert set(shown(page)) == {PILOT, CUBE}
    assert "fam=full_factorial" in page.url


def test_a_design_s_detail_shows_its_source_and_its_checks(context, site):
    page = open_catalog(context, site)
    page.click(f".result__title[data-id='{PILOT}']")
    page.wait_for_selector("#detail[open]")
    text = " ".join(page.text_content("#detail-body").split())
    assert "Box, G. E. P., Hunter, W. G. and Hunter, J. S. (1978)" in text
    assert "recomputed numbers agree" in text
    assert "The library makes this design" in text
    assert "d=bhh-pilot-plant-2-3" in page.url
    page.keyboard.press("Escape")
    page.wait_for_selector("#detail:not([open])", state="attached")
    assert "d=" not in page.url


def test_a_design_s_detail_says_what_it_withstands_and_where_else_it_is(context, site):
    """The pilot plant is the library's 2^3 factorial: each lists the other as the same design."""
    page = open_catalog(context, site)
    badge = page.locator(".result", has=page.locator(f".result__title[data-id='{PILOT}']")).locator(".badge")
    assert "same design in 1 more place" in badge.all_text_contents()
    page.click(f".result__title[data-id='{PILOT}']")
    page.wait_for_selector("#detail[open]")
    text = " ".join(page.text_content("#detail-body").split())
    assert "If any one run is lost, the design still estimates two-factor interactions" in text
    assert "Its two-level runs hold every combination of its 3 factors' levels." in text
    assert page.locator("#detail-body canvas.leverage").count() == 1
    page.click(f"#detail-body .detail__twins [data-id='{CUBE}']")
    page.wait_for_function(
        "document.querySelector('#detail-title').textContent.startsWith('Full factorial, 3 factors')"
    )
    assert "d=lib-full-factorial-k3" in page.url


def test_the_robustness_filters_keep_the_designs_that_withstand_more(context, site):
    page = open_catalog(context, site)
    page.select_option("#projection", "3")
    counted(page, "1 of 5")
    assert shown(page) == [CCD]
    assert "qp=3" in page.url
    page.select_option("#projection", "0")
    page.select_option("#projectivity", "3")
    counted(page, "3 of 5")
    assert set(shown(page)) == {PILOT, CUBE, CCD}
    survives = page.locator("label", has=page.locator("[data-switch=survives]"))
    assert survives.locator(".count").text_content() == "3", "each of the three survives losing a run"


def test_a_design_the_page_cannot_take_says_why_it_does_not_open(context, site):
    page = open_catalog(context, site, f"#d={COMPARISON}")
    page.wait_for_selector("#detail[open]")
    assert "Factorial does not open this design: the page opens designs of 2 to 12 factors." in page.text_content(
        "#detail-body"
    )
    assert page.locator(f"#detail-body [data-open='{COMPARISON}']").count() == 0


def test_a_design_the_page_does_not_make_opens_with_its_runs_as_given(context, site):
    """The page makes no OMARS design: Factorial opens the library's runs as they are, and lists them in Design."""
    page = open_catalog(context, site, f"#d={OMARS}")
    page.wait_for_selector("#detail[open]")
    assert "Factorial opens these runs as the library made them, in their order" in page.text_content("#detail-body")
    page.click(f"#detail-body [data-open='{OMARS}']")
    page.wait_for_selector("#campaign-banner", state="visible")
    page.wait_for_function("window.__pi.count('api_catalogue') === 1 && window.__pi.idle()")
    settled(page)
    assert len(run_sheet(page)) == 9
    assert page.input_value("#design-type") == "given"
    shown = page.eval_on_selector("#design-type option[value=given]", "o => !o.hidden")
    assert shown, "the list offers runs as given while they are open"


def test_a_published_design_opens_in_factorial_with_the_source_s_results(context, site):
    page = open_catalog(context, site)
    page.click(f"[data-open='{PILOT}']")
    page.wait_for_selector("#campaign-banner", state="visible")
    page.wait_for_function("window.__pi.count('api_catalogue') === 1 && window.__pi.idle()")
    settled(page)
    assert len(run_sheet(page)) == 8
    assert page.get_attribute("#workspace", "data-phase") == "learn", "the source's results, analysed"


def test_a_generated_design_opens_in_factorial_ready_to_run(context, site):
    page = open_catalog(context, site)
    page.click(f"[data-open='{CCD}']")
    page.wait_for_selector("#campaign-banner", state="visible")
    page.wait_for_function("window.__pi.count('api_catalogue') === 1 && window.__pi.idle()")
    settled(page)
    assert len(run_sheet(page)) == 17


def test_the_map_draws_a_mark_for_every_design_listed(context, site):
    page = open_catalog(context, site)
    page.wait_for_selector("#map .scatterlayer")
    marks = page.evaluate(
        "document.querySelector('#map').data.filter((t) => t.customdata).flatMap((t) => t.customdata).sort()"
    )
    assert marks == sorted(CATALOG_SAMPLE)


@pytest.mark.parametrize("context_options", [{"viewport": {"width": 390, "height": 844}, "is_mobile": True}])
def test_it_reads_on_a_phone_without_scrolling_sideways(context, site):
    page = open_catalog(context, site)
    wider = page.evaluate(
        """() => [...document.querySelectorAll('body *')]
             .filter((e) => e.getBoundingClientRect().right > document.documentElement.clientWidth + 1)
             .filter((e) => !e.closest('.js-plotly-plot'))
             .map((e) => e.tagName.toLowerCase() + (e.className ? '.' + e.className : ''))"""
    )
    assert wider == []


# --------------------------------------------------------------------------- phase 3: the views over the list
def row(page, label: str) -> list[str]:
    """Return a row of the comparison: each design's value of the measure with this label."""
    cells = page.locator("table.compare tr", has=page.locator("th", has_text=label)).locator("td")
    return cells.all_text_contents()[1:]


def toast_text(page) -> str:
    page.wait_for_selector("#toast.shown")
    return page.text_content("#toast")


def test_two_designs_of_one_size_are_compared_on_the_model_both_fit(context, site):
    page = open_catalog(context, site)
    page.click(f"[data-compare='{CCD}']")
    page.wait_for_selector("#tray:not([hidden])")
    assert page.is_disabled("#tray-open"), "one design is nothing to compare"
    page.click(f"[data-compare='{CUBE}']")
    assert page.text_content("#tray-text") == "2 designs of 3 factors"
    page.click("#tray-open")
    page.wait_for_selector("#compare[open] table.compare")
    assert page.input_value("#compare-model") == "interactions", "the 2^3 cannot fit the quadratic"
    heads = page.eval_on_selector_all("table.compare thead th", "cells => cells.map((c) => c.textContent)")
    assert heads[2:] == ["Central composite, 3 factors, face-centered [n=17]", "Full factorial, 3 factors [n=8]"]
    assert row(page, "Runs") == ["17", "8"]
    assert "cmp=" in page.url
    assert "view=compare" in page.url
    page.wait_for_selector("#compare .compare__fds .scatterlayer")
    page.select_option("#compare-model", "quadratic")
    assert row(page, "Fits the model?") == ["yes", "no (rank 7)"], "the squares of a 2^3 are its intercept"


def test_a_design_of_another_size_does_not_join_the_comparison(context, site):
    page = open_catalog(context, site)
    page.click(f"[data-compare='{CCD}']")
    page.click(f"[data-compare='{OMARS}']")
    assert "the designs compared have 3 factors, and this one has 4" in toast_text(page)
    assert page.locator("#tray-list li").count() == 1


def test_a_design_s_projections_say_what_any_two_or_three_factors_estimate(context, site):
    page = open_catalog(context, site, f"#d={CCD}")
    page.wait_for_selector("#detail[open] .explorer")
    text = page.locator(".explorer__text")
    assert text.text_content() == (
        "In A, B and C, the runs sit at 15 distinct settings and estimate the full quadratic in them (10 terms). "
        "They hold all 8 corners: a full 2^3 factorial in these factors."
    )
    assert page.locator(".projection__point").count() == 15
    page.click(".explorer__picks [data-column='2']")
    assert text.text_content().startswith(
        "In A and B, the runs sit at 9 distinct settings and estimate the full quadratic"
    )
    assert "They hold all 4 corners" in text.text_content()
    assert "The largest |r| between two second-order terms is 0.51." in page.text_content("#detail-body")


def test_a_design_leads_to_the_families_it_can_be_augmented_to(context, site):
    page = open_catalog(context, site, f"#d={CUBE}")
    page.wait_for_selector("#detail[open] .detail__steps")
    page.click(".detail__steps a:text('Central composite')")
    page.wait_for_selector("#detail:not([open])", state="attached")
    counted(page, "1 of 5")
    assert shown(page) == [CCD]
    assert "fam=ccd" in page.url


def test_the_errata_list_every_printed_number_that_does_not_reproduce(context, site):
    page = open_catalog(context, site)
    page.click("#errata-open")
    page.wait_for_selector("#errata[open] .errata__table")
    total = page.text_content("#errata-count")
    assert total.startswith(total.split(" of ")[1].split()[0]), "every printed number, before a filter"
    page.select_option("#errata-source", "box-hunter-hunter-1978")
    page.fill("#errata-words", "already rounded 3.65")
    page.wait_for_function("document.querySelector('#errata-count').textContent.startsWith('1 of')")
    assert "view=errata" in page.url
    page.click(f"#errata [data-id='{COMPARISON}']")
    page.wait_for_selector("#detail[open]")
    assert page.text_content("#detail-title").startswith("An industrial experiment")


def test_the_gaps_are_the_generated_designs_no_source_prints(context, site):
    page = open_catalog(context, site)
    page.click(".try:text('gaps')")
    counted(page, "2 of 5")
    assert set(shown(page)) == {CCD, OMARS}, "the library's 2^3 is Box, Hunter and Hunter's pilot plant"
    page.fill("#query", "")
    page.select_option("#source", "box-hunter-hunter-1978")
    counted(page, "2 of 5")
    assert set(shown(page)) == {PILOT, COMPARISON}
    assert "src=box-hunter-hunter-1978" in page.url


def test_a_bar_of_the_timeline_keeps_its_years(context, site):
    page = open_catalog(context, site)
    page.wait_for_selector("#year-bars .bin")
    page.locator("#year-bars .bin[data-from='1975']").click()
    counted(page, "2 of 5")
    assert (page.input_value("#year-min"), page.input_value("#year-max")) == ("1975", "1979")
    assert "year=1975-1979" in page.url


# --------------------------------------------------------------------------- phase 4: limits on the settings
def test_a_limit_keeps_the_designs_whose_every_run_meets_it(context, site):
    """A + B <= 2 holds for every coded run of a cube; the comparison has one factor, so the limit names no B of its."""
    page = open_catalog(context, site)
    page.fill("#limit", "A + B <= 2")
    counted(page, "4 of 5")
    assert COMPARISON not in shown(page)
    page.fill("#limit", "A + B <= 1")
    counted(page, "0 of 5")
    page.check("#limit-all")
    counted(page, "4 of 5")
    badge = page.locator(".result", has=page.locator(f".result__title[data-id='{CCD}']")).locator(".badge--warn")
    assert badge.text_content() == "2 of 17 runs outside the limit"
    assert "lim=A+%2B+B+%3C%3D+1" in page.url
    page.fill("#limit", "A + B")
    assert page.text_content("#limit-error").startswith("The limit cannot be read: write <= for at most")
    counted(page, "5 of 5")


def test_a_design_s_runs_outside_the_limit_are_marked_and_it_opens_with_the_limit(context, site):
    page = open_catalog(context, site, f"#lim=A+%2B+B+%3C%3D+1&lima=1&d={CCD}")
    page.wait_for_selector("#detail[open] .detail__warn")
    line = page.text_content("#detail-body .detail__warn")
    assert line.endswith("are outside the limit A + B <= 1, of its 17 runs.")
    assert page.locator("#detail-body tr.row--outside").count() == 2
    page.click(f"[data-open-limited='{CCD}']")
    page.wait_for_selector("#campaign-banner", state="visible")
    page.wait_for_function("window.__pi.count('api_catalogue') === 1 && window.__pi.idle()")
    settled(page)
    assert len(run_sheet(page)) == 17
    assert page.locator("#limit-list .limit [name=expression]").first.input_value() == "A + B <= 1", (
        "a CCD from -1 to +1: the limit in its own units is the limit in coded units"
    )
    assert "outside your limits on the settings: choose the D-optimal or I-optimal design" in page.text_content(
        "#design-summary"
    )


def test_a_link_that_names_a_design_in_its_query_opens_it(context, site):
    """The book links to a design as catalog/?d=<id>: its theme reads a fragment as a CSS selector."""
    page = context.new_page()
    page.goto(f"{site}catalog/?d={CCD}")
    page.wait_for_selector("#detail[open]")
    assert page.text_content("#detail-title").startswith("Central composite, 3 factors")
    assert page.url == f"{site}catalog/#d={CCD}", "the address keeps it in the fragment, which no server sees"
