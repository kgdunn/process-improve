"""Reproduced from the books, in the browser: how many printed numbers Factorial gives as its sources print them.

The page reads ``catalog/reproduction.json`` as the build writes it, here for the sources of
``REPRODUCTION_SAMPLE``: Lenth (1989), with a case reproduced in full and one within 1% once the paper's two
misprints are set aside, and Núñez Ares, Schoen and Goos (2023), with a case within 5%, one the library is
unable to reproduce, and a search whose numbers the library cannot compute yet.
"""

from __future__ import annotations

import itertools
import json
import re

import pytest

TENSILE = "lenth-example-ii-tensile-strength"

#: The causes the page counts, in the order of its bar, as ``REASONS`` in ``web/reproduced-logic.js`` has them.
REASONS = ("source_rounding", "source_error", "method", "library", "incorrect_option", "unexplained", "library_gap")

#: The lists of cases behind "Search every case", one for each state a case can be in (``CASE_STATES``).
LISTS = (
    "Matches the book",
    "Differs, and we know why",
    "Multiple choice, wrong on purpose",
    "Differs, reason not known yet",
)

#: A number an example's box shows, as ``A_NUMBER`` in ``web/reproduced-logic.js`` has it.
A_NUMBER = re.compile("[-\N{MINUS SIGN}+]?[\\d,]*\\.?\\d+%?")


@pytest.fixture(scope="module")
def reproduction(reproduction_file) -> dict:
    return json.loads(reproduction_file.read_text(encoding="utf-8"))


def open_page(context, site: str, fragment: str = "", *, cases: bool = False) -> object:
    """Open the page and wait until it has drawn the sources; with ``cases``, open "Search every case" too."""
    page = context.new_page()
    page.goto(f"{site}catalog/reproduced/{fragment}")
    page.wait_for_selector("#shelf .shelf__row", state="attached")
    if cases and page.get_attribute("#numbers", "open") is None:
        page.click("#numbers > summary")
    return page


def counts(reproduction: dict, source: str | None = None) -> dict:
    """Count the printed numbers of a source, or of them all: recomputed, matching, and no match by cause."""
    cases = [c for c in reproduction["cases"] if source in (None, c["source"])]
    causes = dict.fromkeys(REASONS, 0)
    for case in cases:
        causes["library_gap"] += case["unchecked"].get("library_gap", 0)
        for number in case["differs"]:
            causes[number["cause"]] += 1
    checked = sum(c["checked"] for c in cases)
    return {"checked": checked, "match": checked - sum(len(c["differs"]) for c in cases), "causes": causes}


def fitting(reproduction: dict, cause: str) -> list[dict]:
    """Return the numbers of a cause that fit an example's box: plain numbers of 14 characters at most."""

    def fits(value: object) -> bool:
        return isinstance(value, str) and len(value) <= 14 and A_NUMBER.fullmatch(value) is not None

    out = []
    for case in reproduction["cases"]:
        numbers = case["missing"] if cause == "library_gap" else [n for n in case["differs"] if n["cause"] == cause]
        out += [
            n for n in numbers if n["where"] and fits(n["printed"]) and (cause == "library_gap" or fits(n["found"]))
        ]
    return out


def reason_counts(page) -> dict[str, int]:
    """Return the numbers the cascade counts for each reason it has a row for, in the rows' order."""
    return page.eval_on_selector_all(
        "#reasons .reason[data-cause]",
        "rows => Object.fromEntries(rows.map((r) => [r.dataset.cause, "
        "Number(r.querySelector('.badge').textContent.replaceAll(',', ''))]))",
    )


def test_the_catalog_leads_to_the_page_and_its_bar_counts_every_number(reproduced_context, site, reproduction):
    """The catalog's door has the footer's name for validation, and opens it: the books are one step on."""
    page = reproduced_context.new_page()
    page.goto(f"{site}catalog/")
    page.click(".door:has-text('Is it accurate?')")
    page.wait_for_url("**/validation.html")
    page.click("text=each with its pages")
    page.wait_for_url("**/catalog/reproduced/")
    page.wait_for_selector("#shelf .shelf__row", state="attached")
    all_ = counts(reproduction)
    assert page.text_content("#tally-value") == f"{all_['match']:,} of {all_['checked']:,}"
    assert all_["causes"]["source_error"] > 0, "the sample has a misprint"
    assert all_["causes"]["library_gap"] > 0, "and numbers the library cannot compute yet"
    states = page.eval_on_selector_all("#wholebar .wholebar__part", "parts => parts.map((p) => p.dataset.state)")
    assert states == ["match", "known", "gap"], "the states the sample has numbers in, in the bar's order"
    share = 100 * all_["match"] / all_["checked"]
    assert page.text_content("#wholebar [data-state=match]") == f"{share:.1f}% match"
    assert f"Not in Factorial yet: {all_['causes']['library_gap']}" in page.get_attribute("#wholebar", "aria-label")
    assert page.locator("#shelf .shelf__row").count() == 2
    assert page.locator("#forms button").all_text_contents() == ["All 2", "Journal papers 2"]


def test_the_reasons_are_a_cascade_and_each_shows_a_number_of_its_own(reproduced_context, site, reproduction):
    page = open_page(reproduced_context, site)
    causes = counts(reproduction)["causes"]
    held = {cause: n for cause, n in causes.items() if n}
    assert reason_counts(page) == held, "a row for each reason with a number, in the order of the bar"
    differ = sum(held.values()) - causes["library_gap"]
    assert page.text_content("#reasons-title") == (
        f"The {differ} numbers that differ, and {causes['library_gap']} more that are not in Factorial yet"
    )
    spans = page.eval_on_selector_all(
        "#reasons .reason[data-cause] .reason__bar",
        "bars => bars.map((b) => [parseFloat(b.style.left), parseFloat(b.style.width)])",
    )
    assert spans[0][0] == 0
    for (left, width), (start, _) in itertools.pairwise(spans):
        assert start == pytest.approx(left + width, abs=0.01), "a bar starts where the one before it ended"
    assert sum(spans[-1]) == pytest.approx(100, abs=0.01)
    assert page.text_content("#reasons .reason__mid") == f"{differ} differ"

    misprint = page.locator("#reasons .reason[data-cause=source_error]")
    assert misprint.locator(".reason__name").text_content() == f"A misprint or slip in the book{causes['source_error']}"
    assert misprint.locator(".example__label").text_content() == "Example:"
    printed = {
        (n["printed"], n["found"]) for c in reproduction["cases"] for n in c["differs"] if n["cause"] == "source_error"
    }
    assert misprint.locator(".example__box").text_content().startswith("Lenth (1989), p. ")
    assert tuple(misprint.locator(".example__values dd").all_text_contents()) in printed
    what = misprint.locator(".example__where").text_content().split(" · ")[1]
    assert what in reproduction["quantities"].values(), "the quantity by its plain name"

    lacking = page.locator("#reasons .reason[data-cause=library_gap]")
    first = lacking.locator(".example__box").text_content()
    assert first.endswith("no method yet")
    assert len(fitting(reproduction, "library_gap")) > 1
    lacking.locator(".example__more").click()
    assert lacking.locator(".example__box").text_content() != first, "See another puts the next number in the box"
    assert lacking.locator(".example__values dd").last.text_content() == "no method yet"
    one = [cause for cause in held if len(fitting(reproduction, cause)) == 1]
    assert one, "the sample has a reason with one number that fits the box"
    assert page.locator(f"#reasons .reason[data-cause={one[0]}] .example__more").count() == 0, "it offers no other"


def test_a_source_s_row_opens_onto_its_counts_and_leads_to_its_page(reproduced_context, site, reproduction):
    page = open_page(reproduced_context, site)
    lenth = counts(reproduction, "lenth-1989")
    row = page.locator("#shelf .shelf__row[data-source=lenth-1989]")
    assert row.locator(".shelf__name").text_content() == "Lenth · 1989"
    assert row.locator("summary .shelf__numbers").text_content() == str(lenth["checked"])
    assert row.locator(".shelf__share").text_content() == f"{100 * lenth['match'] / lenth['checked']:.1f}%"
    assert row.get_attribute("open") is None
    row.locator("summary").click()
    facts = row.locator(".shelf__body .states li").all_text_contents()
    assert facts == [
        f"Matches the book {lenth['match']}",
        f"The book rounds sooner {lenth['causes']['source_rounding']}",
        f"A misprint or slip in the book {lenth['causes']['source_error']}",
    ]
    assert "Cases that match in every number: 1 of 2." in row.locator(".shelf__body").text_content()
    assert row.locator(".shelf__body a").get_attribute("href") == "../books/lenth-1989/"
    assert page.locator(".receipt").count() == 0, "the book the page has a table of is not in the sample"

    page.select_option("#source-order", "numbers")
    most = max(("lenth-1989", "nunez-ares-et-al-2023"), key=lambda key: counts(reproduction, key)["checked"])
    assert page.locator("#shelf .shelf__row").first.get_attribute("data-source") == most
    assert page.locator("#shelf .shelf__row[data-source=lenth-1989]").get_attribute("open") is not None, (
        "a row stays open when the order changes"
    )
    page.click("#forms button:has-text('Journal papers')")
    assert page.get_attribute("#forms button:has-text('Journal papers')", "aria-pressed") == "true"
    assert page.locator("#shelf .shelf__row").count() == 2


def test_the_cases_wait_behind_a_fold_and_an_address_that_names_one_opens_it(reproduced_context, site):
    """The page opens on its bar and its sources; the cases open on asking, or for an address that names one."""
    page = open_page(reproduced_context, site)
    assert page.get_attribute("#numbers", "open") is None
    assert page.is_hidden("#lists"), "the lists of cases are behind Search every case"
    page.click("#numbers > summary")
    assert page.is_visible("#lists")

    page = open_page(reproduced_context, site, f"#case={TENSILE}")
    case = page.locator(f"#case-{TENSILE}")
    case.locator(".number").first.wait_for()
    assert page.get_attribute("#numbers", "open") is not None
    assert case.get_attribute("open") is not None
    assert page.locator("#state-known").get_attribute("open") is not None, "the list of the case's state"


def test_a_state_opens_its_list_and_a_case_shows_each_number_that_differs_and_why(reproduced_context, site):
    """The cases are listed by state, as the bar counts their numbers.

    Lenth's tensile strength differs, for known reasons, and its worst difference leaves its misprints out.
    """
    page = open_page(reproduced_context, site, cases=True)
    assert tuple(page.locator("#lists .statelist__name").all_text_contents()) == LISTS
    page.click("#state-known > summary")
    page.wait_for_function("location.hash.includes('open=known')")
    case = page.locator(f"#case-{TENSILE}")
    assert "worst 0.25%" in case.locator("summary").text_content(), "the misprints do not hold the case back"
    case.locator("summary").click()
    what = re.compile(r"^A margin of error:")
    me = case.locator(".number").filter(has=page.locator(".number__what", has_text=what))
    assert me.locator(".number__cause").text_content() == "A misprint or slip in the book"
    assert me.locator(".number__error").text_content() == "3.3%"
    assert me.locator("dd code").first.text_content() == ".56"
    assert "misprint" in me.locator(".number__reason").text_content()
    href = case.locator("a", has_text="Open its design").get_attribute("href")
    assert href.endswith(f"catalog/#d={TENSILE}")
    page.wait_for_function(f"location.hash.includes('case={TENSILE}')")


def test_a_source_chosen_narrows_the_cases_and_never_the_counts_above_them(reproduced_context, site, reproduction):
    """A source chosen keeps its cases in the lists; the bar and the reasons stay those of every source."""
    page = open_page(reproduced_context, site, cases=True)
    recomputed = sum(1 for c in reproduction["cases"] if c["tier"])
    assert page.text_content("#count") == f"{recomputed} cases, in four lists."
    tally, reasons = page.text_content("#tally-value"), reason_counts(page)
    page.select_option("#source", "lenth-1989")
    page.wait_for_function("location.hash.includes('src=lenth-1989')")
    assert page.text_content("#count") == f"2 cases of {recomputed} kept."
    assert page.text_content("#tally-value") == tally
    assert reason_counts(page) == reasons
    page.reload()
    page.wait_for_selector("#shelf .shelf__row", state="attached")
    assert page.get_attribute("#numbers", "open") is not None, "an address naming a source shows its cases"
    assert page.input_value("#source") == "lenth-1989"
    page.click("#clear")
    assert page.text_content("#count") == f"{recomputed} cases, in four lists."


def test_what_the_library_lacks_is_listed_by_reason_with_its_cases(reproduced_context, site, reproduction):
    page = open_page(reproduced_context, site, cases=True)
    missing = [n for c in reproduction["cases"] for n in c["missing"]]
    assert missing, "the sample holds numbers the library cannot compute yet"
    assert reason_counts(page)["library_gap"] == len(missing), "the cascade counts them too"
    groups = page.locator("#missing > li").count()
    assert 1 <= groups <= min(20, len({n["reason"] for n in missing})), (
        "the numbers of one missing method fall together"
    )
    counted = [int(t.split()[0]) for t in page.locator("#missing .missing__item .badge").all_text_contents()]
    assert counted == sorted(counted, reverse=True), "what would recompute the most numbers comes first"
    assert page.text_content("#missing-count").startswith(f"{len(missing)} printed numbers")
    first = page.locator("#missing .missing__item").first
    first.locator("summary").click()
    assert first.locator(".missing__cases a").first.get_attribute("href").startswith(f"{site}catalog/#d=")
    page.select_option("#source", "lenth-1989")
    page.wait_for_function("location.hash.includes('src=lenth-1989')")
    assert page.locator("#missing > li").count() == 0, "the list follows the source chosen"


def test_a_reason_searched_for_keeps_the_cases_whose_numbers_differ_for_it(reproduced_context, site):
    page = open_page(reproduced_context, site, cases=True)
    page.click(".try:text-is('misprint')")
    page.wait_for_function("location.hash.includes('q=misprint')")
    assert page.input_value("#query") == "misprint"
    assert page.locator(f"#case-{TENSILE}").count() == 1, "the lists of the states that differ open"
    assert page.locator("#state-match .case").count() == 0
    page.fill("#query", "no word of any case")
    page.wait_for_function("document.querySelector('#count').textContent.startsWith('0 cases')")


def test_a_form_chosen_keeps_the_cases_of_its_sources(reproduced_context, site, reproduction):
    """Both sources of the sample are journal papers: the form keeps every case, and the address keeps the form."""
    page = open_page(reproduced_context, site, cases=True)
    page.select_option("#source", "form:paper")
    page.wait_for_function("location.hash.includes('form=paper')")
    recomputed = sum(1 for c in reproduction["cases"] if c["tier"])
    assert page.text_content("#count") == f"{recomputed} cases, in four lists."
    assert page.is_visible("#clear")


@pytest.mark.parametrize("context_options", [{"viewport": {"width": 390, "height": 844}}])
def test_on_a_phone_the_page_never_scrolls_sideways(reproduced_context, site):
    page = open_page(reproduced_context, site, "#open=known,quiz,open")
    for row in page.locator("#shelf .shelf__row > summary").all():
        row.click()
    for case in page.locator(".case summary").all():
        case.click()
    assert page.locator("#reasons .reason__link").first.is_hidden(), "on a phone the bars are joined by their starts"
    assert page.evaluate("document.documentElement.scrollWidth") <= 390
