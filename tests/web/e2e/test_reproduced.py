"""Reproduced from the books, in the browser: how closely the library reproduces each published case.

The page reads ``catalog/reproduction.json`` as the build writes it, here for the sources of
``REPRODUCTION_SAMPLE``: Lenth (1989), with a case reproduced in full and one within 1% once the paper's two
misprints are set aside, and Núñez Ares, Schoen and Goos (2023), with a case within 5%, one the library is
unable to reproduce, and a search whose numbers the library cannot compute yet.
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


def cause_counts(page, table: str) -> dict[str, int]:
    """Return the numbers a table of causes counts for each cause."""
    return page.eval_on_selector_all(
        f"#{table} tbody tr",
        "rows => Object.fromEntries(rows.map((r) => [r.dataset.cause, Number(r.querySelector('strong').textContent)]))",
    )


def totals(reproduction: dict, key: str) -> dict[str, int]:
    """Count the causes of the differences (``differs``) or of the numbers not recomputed (``unchecked``)."""
    out = dict.fromkeys(reproduction["causes"][key], 0)
    for case in reproduction["cases"]:
        if key == "differs":
            for number in case["differs"]:
                out[number["cause"]] += 1
        else:
            for cause, n in case["unchecked"].items():
                out[cause] += n
    return out


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
    assert page.text_content("#stat-errors") == str(totals(reproduction, "differs")["source_error"])
    assert page.text_content("#stat-missing") == str(totals(reproduction, "unchecked")["library_gap"])
    assert page.locator("#bysource .source").count() == 2
    assert page.locator("#bysource tfoot th").text_content() == "All 2 sources"


def test_a_tier_opens_its_list_and_a_case_shows_each_number_that_differs_and_why(reproduced_context, site):
    page = open_page(reproduced_context, site)
    page.click("#tiers tr[data-tier=within_1] button")
    assert page.locator("#tier-within_1").get_attribute("open") is not None
    page.wait_for_function("location.hash.includes('open=within_1')")
    case = page.locator(f"#case-{TENSILE}")
    assert "worst 0.25%" in case.locator("summary").text_content(), "the misprints do not hold the case back"
    case.locator("summary").click()
    me = case.locator(".number").filter(has=page.locator(".number__what", has_text=re.compile(r"^ME:")))
    assert me.locator(".number__cause").text_content() == "Error in the source"
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
    assert page.locator("#tier-within_1").get_attribute("open") is not None


def test_a_source_chosen_narrows_the_page_and_the_address_keeps_it(reproduced_context, site):
    page = open_page(reproduced_context, site)
    page.click("#bysource .source:has-text('Lenth')")
    page.wait_for_function("location.hash.includes('src=lenth-1989')")
    assert tier_counts(page) == {"reproduced": 1, "within_1": 1, "within_5": 0, "not_reproduced": 0}
    assert page.locator("#bysource .source[aria-pressed=true]").count() == 1
    assert page.input_value("#source") == "lenth-1989"
    page.reload()
    page.wait_for_selector("#tiers tbody tr")
    assert tier_counts(page)["within_1"] == 1
    page.click("#scope .link")
    assert sum(tier_counts(page).values()) > 2, "Show every case clears the source"


def test_the_causes_are_counted_for_all_the_sources_together_only(reproduced_context, site, reproduction):
    """A source chosen narrows the tiers, but never the counts of errors in the sources, nor the table by source."""
    page = open_page(reproduced_context, site)
    differs, unchecked = totals(reproduction, "differs"), totals(reproduction, "unchecked")
    assert cause_counts(page, "causes-differs") == differs
    assert cause_counts(page, "causes-unchecked") == unchecked
    assert differs["source_error"] > 0
    page.click("#bysource .source:has-text('Lenth')")
    page.wait_for_function("location.hash.includes('src=lenth-1989')")
    assert cause_counts(page, "causes-differs") == differs, "the totals stay those of every source"
    assert "Error in the source" not in page.text_content("#bysource")


def test_what_the_library_lacks_is_listed_by_reason_with_its_cases(reproduced_context, site, reproduction):
    page = open_page(reproduced_context, site)
    missing = [n for c in reproduction["cases"] for n in c["missing"]]
    assert missing, "the sample holds numbers the library cannot compute yet"
    groups = page.locator("#missing > li").count()
    assert 1 <= groups <= min(20, len({n["reason"] for n in missing})), (
        "the numbers of one missing method fall together"
    )
    counts = [int(t.split()[0]) for t in page.locator("#missing .missing__item .badge").all_text_contents()]
    assert counts == sorted(counts, reverse=True), "what would recompute the most numbers comes first"
    assert page.text_content("#missing-count").startswith(f"{len(missing)} printed numbers")
    first = page.locator("#missing .missing__item").first
    first.locator("summary").click()
    assert first.locator(".missing__cases a").first.get_attribute("href").startswith(f"{site}catalog/#d=")
    page.click("#bysource .source:has-text('Lenth')")
    page.wait_for_function("location.hash.includes('src=lenth-1989')")
    assert page.locator("#missing > li").count() == 0, "the list follows the source chosen"


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


def test_the_sources_add_up_by_kind_and_a_kind_chosen_keeps_its_cases(reproduced_context, site, reproduction):
    """Both sources of the sample are journal papers: one group, with the page's totals, and bars on totals only."""
    page = open_page(reproduced_context, site)
    group = page.locator("#bysource .bysource__group")
    assert group.count() == 1
    assert group.locator("th").text_content() == "Journal papers 2 sources"
    recomputed = [c for c in reproduction["cases"] if c["tier"]]
    assert group.locator(".bysource__cases").text_content() == str(len(recomputed))
    assert page.locator("#bysource tr[data-source] .source__bar").count() == 0, "no bar beside a single source"
    assert page.locator("#bysource .bysource__group .source__bar span").count() > 0
    group.locator("button").click()
    page.wait_for_function("location.hash.includes('form=paper')")
    assert page.input_value("#source") == "form:paper"
    assert sum(tier_counts(page).values()) == len(recomputed)
