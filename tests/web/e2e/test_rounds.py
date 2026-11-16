"""The campaign, on the fourth stage, Next: a study as a round, and what its results call for.

A study started from another, with its ranges moved, names the study it builds on. Next lists the rounds of that
campaign from the studies the browser keeps: each round's runs and its best result so far, with the way to open
it. The link between two rounds goes with the study through a reload, a file and a link, and the page says when a
run or a result of an earlier round changed after a later one was built on it. A study that stands alone is one
round, and lists none. What the results call for, which Learn used to end with, is on this stage.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from tests.web.e2e._page import (
    enter_results,
    generate,
    open_library,
    open_question,
    practise,
    reveal,
    saved_studies,
    settled,
)
from tests.web.e2e.test_study_files import BANNER, opened_page

TITLE = "Yield, maximise: Box-Behnken in T, P and F"


def rounds(page: Any) -> list[list[str]]:  # noqa: ANN401 - a Playwright Page
    """Read the rounds as the page lists them, a row each: the round, its study, its runs, its best, and the way in."""
    return page.evaluate(
        "() => [...document.querySelectorAll('#rounds-out tbody tr')]"
        ".map((tr) => [...tr.cells].map((cell) => cell.textContent))"
    )


def next_of(page: Any) -> Any:  # noqa: ANN401
    """Show Next once the rounds are listed on it."""
    reveal(page, "#next")
    page.wait_for_selector("#rounds:not([hidden]) tbody tr")
    return page


def start_study(page: Any) -> None:  # noqa: ANN401
    """Make a Box-Behnken study that maximises Yield, and simulate its results: they call for a study further on."""
    open_question(page)
    page.fill("#response", "Yield")
    page.select_option("#response-goal", "maximize")
    generate(page, design_type="box_behnken")
    practise(page)


@pytest.fixture
def two_rounds(blank_page: Any) -> Any:  # noqa: ANN401
    """Return a study started from another, both with results, with Next on show: the second round of two."""
    page = blank_page
    start_study(page)
    reveal(page, "#next")
    page.click("#next-study")
    settled(page)
    generate(page, design_type="box_behnken")
    practise(page)
    return next_of(page)


def test_next_is_the_fourth_stage_and_holds_what_the_results_call_for(page):
    tabs = page.locator("#studybar [role=tab]")
    assert tabs.evaluate_all("(all) => all.map((tab) => tab.id)") == ["go-plan", "go-run", "go-learn", "go-next"]
    assert re.sub(r"\s+", " ", tabs.last.text_content()).strip() == "4 Next"
    assert page.locator("#step-analyse #next").count() == 0, "the next step has left Learn"
    assert page.text_content("#learn-to-next") == "Next: what the results call for"
    page.click("#learn-to-next")
    assert page.get_attribute("#workspace", "data-phase") == "next"
    assert page.get_attribute("#go-next", "aria-selected") == "true"
    assert page.is_visible("#step-next #next")
    assert page.text_content("#h-next-stage") == "What comes next"
    assert page.text_content("#next-out .next__called h4") == "What the results call for"
    assert page.is_hidden("#rounds"), "the worked example stands alone: no rounds to list"
    assert page.is_hidden("#next-empty")
    assert page.is_hidden("#badge-next")

    # The stages answer the arrow keys, the fourth as the others do.
    page.focus("#go-learn")
    page.keyboard.press("ArrowRight")
    assert page.evaluate("document.activeElement.id") == "go-next"


def test_before_any_analysis_next_says_what_it_will_hold(blank_page):
    page = blank_page
    page.click("#go-next")
    assert page.is_visible("#next-empty")
    assert page.text_content("#next-empty") == "What the results call for is here once Learn has analysed them."
    assert page.is_hidden("#next")
    assert page.is_hidden("#rounds")


def test_a_study_started_from_another_is_the_next_round_of_its_campaign(two_rounds):
    page = two_rounds
    listed = rounds(page)
    assert [row[0] for row in listed] == ["Round 1", "Round 2"]
    assert [row[1] for row in listed] == [TITLE, TITLE]
    assert [row[2] for row in listed] == ["15 runs, all results in", "15 runs, all results in"]
    assert all(re.fullmatch(r"[\d.]+, in run \d+", row[3]) for row in listed), "each round's best result so far"
    assert [row[4] for row in listed] == ["Open", "This study"]
    assert page.text_content("#badge-next") == "round 2"
    assert page.is_visible("#next"), "what this round's results call for is under the rounds"

    first, second = sorted(saved_studies(page, lambda saved: len(saved) == 2), key=lambda study: study["created"])
    assert "builds_on" not in first
    assert first["v"] == 1, "a study that stands alone keeps the format it had"
    [on] = second["builds_on"]
    assert on["id"] == first["id"]
    assert re.fullmatch(r"[0-9a-f]{64}", on["sha256"])
    assert on["title"] == TITLE
    assert second["v"] == 9, "saved in the format that holds the link"


def test_the_round_before_is_kept_through_a_reload_before_the_run_sheet_is_made(blank_page):
    page = blank_page
    start_study(page)
    reveal(page, "#next")
    page.click("#next-study")
    settled(page)
    page.reload()
    page.wait_for_function("window.__pi.count('api_catalogue') === 1 && window.__pi.idle()")
    assert page.get_attribute("#workspace", "data-phase") == "plan", "the new study's question, as a draft"
    generate(page, design_type="box_behnken")
    practise(page)
    assert [row[0] for row in rounds(next_of(page))] == ["Round 1", "Round 2"]


def test_a_round_is_opened_from_the_list_and_lists_the_same_rounds(two_rounds):
    page = two_rounds
    page.click(f'[aria-label="Open round 1: {TITLE}"]')
    page.wait_for_selector("#campaign-banner", state="visible")
    settled(page)
    listed = rounds(next_of(page))
    assert [row[4] for row in listed] == ["This study", "Open"]
    assert page.text_content("#badge-next") == "round 1"


def test_a_result_changed_in_an_earlier_round_is_said_of_the_round_built_on_it(two_rounds):
    page = two_rounds
    assert page.locator("#rounds-out .warnings li").count() == 0
    page.click(f'[aria-label="Open round 1: {TITLE}"]')
    page.wait_for_selector("#campaign-banner", state="visible")
    settled(page)
    enter_results(page, {0: 12.5})
    settled(page)
    listed = next_of(page)
    assert listed.locator("#rounds-out .warnings li").all_text_contents() == [
        "Round 2 was built on round 1 as it was then: a run or a result of round 1 has changed since."
    ]


def test_a_study_s_own_file_carries_the_rounds_before_it_and_opens_the_study(two_rounds, open_context, site, tmp_path):
    page = two_rounds
    saved_studies(page, lambda saved: len(saved) == 2 and any(study.get("builds_on") for study in saved))
    open_library(page)
    later = page.locator("#library-list li").filter(has_text=TITLE).first  # the list has the newest study first
    with page.expect_download() as download:
        later.get_by_role("button", name=f"Download {TITLE}").click()
    name = download.value.suggested_filename
    download.value.save_as(tmp_path / name)
    other = opened_page(open_context, site)
    other.set_input_files("#study-upload", str(tmp_path / name))
    other.wait_for_selector(BANNER, state="visible")
    settled(other)
    assert other.text_content(BANNER) == (
        f"Opened {TITLE} from {name}. It and the round before it are saved in this browser, under Your studies."
    )
    assert other.text_content("#library-count") == "2"
    listed = rounds(next_of(other))
    assert [(row[0], row[4]) for row in listed] == [("Round 1", "Open"), ("Round 2", "This study")]


def test_a_link_carries_the_one_study_and_names_the_round_before(two_rounds, open_context):
    page = two_rounds
    reveal(page, "#share")
    page.click("#share")
    page.wait_for_selector("#share-out", state="visible")
    elsewhere = open_context().new_page()
    elsewhere.goto(page.input_value("#share-url"))
    elsewhere.wait_for_selector("#campaign-banner", state="visible")
    elsewhere.wait_for_function("window.__pi.count('api_catalogue') === 1 && window.__pi.idle()")
    settled(elsewhere)
    listed = rounds(next_of(elsewhere))
    assert listed[0][:2] == ["Round 1", f"{TITLE} (not in this browser)"]
    assert listed[0][4] == "", "no study to open"
    assert (listed[1][0], listed[1][4]) == ("Round 2", "This study")


def test_the_report_holds_the_rounds(two_rounds):
    page = two_rounds
    reveal(page, "#learn-report-view")
    with page.expect_popup() as popup:
        page.click("#learn-report-view")
    report = popup.value
    report.wait_for_url("blob:**")
    text = report.text_content("body")
    assert "The campaign so far" in text
    assert "Round 2" in text
    listed = report.locator("table").filter(has_text="Round 2").first.text_content()
    assert "Open" not in listed, "the way to open a round is the page's own"


@pytest.mark.parametrize("context_options", [{"viewport": {"width": 390, "height": 844}, "is_mobile": True}])
def test_on_a_phone_the_four_stages_and_the_rounds_fit(two_rounds):
    page = two_rounds
    assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
    right = page.locator("#go-next").bounding_box()
    assert right["x"] + right["width"] <= 390
