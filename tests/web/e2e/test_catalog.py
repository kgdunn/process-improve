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


def test_a_design_the_page_does_not_make_says_why_it_does_not_open(context, site):
    page = open_catalog(context, site, f"#d={OMARS}")
    page.wait_for_selector("#detail[open]")
    assert "Factorial does not open this design: the page does not make this kind of design." in page.text_content(
        "#detail-body"
    )
    assert page.locator(f"#detail-body [data-open='{OMARS}']").count() == 0


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
