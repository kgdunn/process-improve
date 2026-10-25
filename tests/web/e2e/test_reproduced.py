"""Reproduced from the books, in the browser: how closely the library reproduces each published case.

The page reads ``catalog/reproduction.json`` as the build writes it, here for the sources of
``REPRODUCTION_SAMPLE``: Lenth (1989), with a case reproduced in full and one whose misprinted alias chain
is not a number, and Núñez Ares and Goos (2019), with a case within 1% and one within 5%.
"""

from __future__ import annotations

import json
import re

import pytest

TENSILE = "lenth-example-ii-tensile-strength"


@pytest.fixture(scope="module")
def reproduction(reproduction_file) -> dict:
    return json.loads(reproduction_file.read_text(encoding="utf-8"))


def open_page(context, site: str, fragment: str = "") -> object:
    """Open the page, and wait until it has drawn the cases."""
    page = context.new_page()
    page.goto(f"{site}catalog/reproduced/{fragment}")
    page.wait_for_selector("#tiers tbody tr")
    return page


def tier_counts(page) -> dict[str, int]:
    """Return the cases the table counts in each tier."""
    return page.eval_on_selector_all(
        "#tiers tbody tr",
        "rows => Object.fromEntries(rows.map((r) => [r.dataset.tier, Number(r.querySelector('strong').textContent)]))",
    )


def test_the_catalog_leads_to_the_page_and_its_table_counts_every_case(reproduced_context, site, reproduction):
    page = reproduced_context.new_page()
    page.goto(f"{site}catalog/")
    page.click("text=Reproduced from the books")
    page.wait_for_url("**/catalog/reproduced/")
    page.wait_for_selector("#tiers tbody tr")
    recomputed = [c for c in reproduction["cases"] if c["tier"]]
    expected = {tier: sum(c["tier"] == tier for c in recomputed) for tier in reproduction["tiers"]}
    assert tier_counts(page) == expected
    assert all(expected.values()), "the sample has a case in every tier"
    numbers = sum(c["checked"] for c in recomputed)
    assert page.text_content("#stat-numbers") == f"{numbers:,}"
    assert page.text_content("#stat-cases") == str(len(recomputed))
    assert page.locator("#sources .source").count() == 2


def test_a_tier_opens_its_list_and_a_case_shows_each_number_that_differs_and_why(reproduced_context, site):
    page = open_page(reproduced_context, site)
    page.click("#tiers tr[data-tier=not_reproduced] button")
    assert page.locator("#tier-not_reproduced").get_attribute("open") is not None
    page.wait_for_function("location.hash.includes('open=not_reproduced')")
    case = page.locator(f"#case-{TENSILE}")
    assert "worst not a number" in case.locator("summary").text_content()
    case.locator("summary").click()
    me = case.locator(".number").filter(has=page.locator(".number__what", has_text=re.compile(r"^ME:")))
    assert me.locator(".number__error").text_content() == "3.3%"
    assert me.locator("dd code").first.text_content() == ".56"
    assert "misprint" in me.locator(".number__reason").text_content()
    href = case.locator("a", has_text="Open its design").get_attribute("href")
    assert href.endswith(f"catalog/#d={TENSILE}")
    page.wait_for_function(f"location.hash.includes('case={TENSILE}')")


def test_an_address_opens_the_case_it_names(reproduced_context, site):
    page = open_page(reproduced_context, site, f"#case={TENSILE}")
    case = page.locator(f"#case-{TENSILE}")
    case.locator(".number").first.wait_for()
    assert case.get_attribute("open") is not None
    assert page.locator("#tier-not_reproduced").get_attribute("open") is not None


def test_a_source_chosen_narrows_the_page_and_the_address_keeps_it(reproduced_context, site):
    page = open_page(reproduced_context, site)
    page.click("#sources .source:has-text('Lenth')")
    page.wait_for_function("location.hash.includes('src=lenth-1989')")
    assert tier_counts(page) == {"reproduced": 1, "within_1": 0, "within_5": 0, "not_reproduced": 1}
    assert page.locator("#sources .source[aria-pressed=true]").count() == 1
    assert page.input_value("#source") == "lenth-1989"
    page.reload()
    page.wait_for_selector("#tiers tbody tr")
    assert tier_counts(page)["not_reproduced"] == 1
    page.click("#scope .link")
    assert sum(tier_counts(page).values()) > 2, "Show every case clears the source"


def test_a_reason_searched_for_keeps_the_cases_whose_numbers_differ_for_it(reproduced_context, site):
    page = open_page(reproduced_context, site)
    page.click(".try:text-is('misprint')")
    page.wait_for_function("location.hash.includes('q=misprint')")
    assert page.input_value("#query") == "misprint"
    assert page.locator(f"#case-{TENSILE}").count() == 1, "the lists of the tiers that differ open"
    assert page.locator("#tier-reproduced .case").count() == 0
    page.fill("#query", "no word of any case")
    page.wait_for_function("document.querySelector('#count').textContent.startsWith('0 cases')")


@pytest.mark.parametrize("context_options", [{"viewport": {"width": 390, "height": 844}}])
def test_on_a_phone_the_page_never_scrolls_sideways(reproduced_context, site):
    page = open_page(reproduced_context, site, "#open=within_1,within_5,not_reproduced")
    for case in page.locator(".case summary").all():
        case.click()
    assert page.evaluate("document.documentElement.scrollWidth") <= 390
