"""The design catalog in the browser: search and filter the designs, read one, and open it in Factorial.

The catalog served here is the build's, for the few designs of ``CATALOG_SAMPLE``: a published
experiment the page regenerates, with its results; a comparison; a generated design the page makes;
and one it does not.
"""

from __future__ import annotations

import pytest

from tests.web._support import CATALOG_SAMPLE
from tests.web.e2e._page import run_sheet, settled

PILOT, COMPARISON, CCD, OMARS = CATALOG_SAMPLE


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
    assert fresh_page.text_content("#count") == "4 of 4 designs: 2 published, 2 generated"
    assert set(shown(fresh_page)) == {PILOT, COMPARISON, CCD, OMARS}
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
    counted(page, "2 of 4")
    assert set(shown(page)) == {CCD, OMARS}
    page.fill("#query", "quadratic")
    counted(page, "1 of 4")
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
    assert family.locator(".count").text_content() == "1"
    family.locator("input").check()
    counted(page, "1 of 4")
    assert shown(page) == [PILOT]
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
    assert marks == sorted([PILOT, COMPARISON, CCD, OMARS])


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
