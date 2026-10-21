"""What Factorial keeps: each study and its results, in the reader's own browser, and in a link.

A context is one browser with storage of its own, so a page opened again in the same context is
the reader coming back, and a page in a second context is another reader, or another machine. A
brief not yet made into a run sheet is kept too, as a draft.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tests.web._support import FACTORS
from tests.web.e2e._page import (
    choose,
    factor_table,
    first_paint_done,
    generate,
    name_response,
    practise,
    reveal,
    run_sheet,
    saved_studies,
    set_factors,
    settled,
    status,
)

BANNER = "#campaign-banner"
#: How a study made from the worked example's brief is titled: its response, then its design and factors.
YIELD = "Yield (%), maximise: "


def results(page) -> list[str]:
    """Read the results column of the run sheet as the reader sees it, in the sheet's order."""
    return page.eval_on_selector_all("#design-table input[data-index]", "inputs => inputs.map((i) => i.value)")


def saved(page, count: int) -> None:
    """Wait until the browser keeps ``count`` designs, as the list says."""
    page.wait_for_function(f"document.querySelector('#library-count').textContent === '{count}'")


def come_back(context) -> object:
    """Open the page again in the same browser, and wait until it has picked up and analysed the design."""
    page = context.new_page()
    page.goto(page_url(context))
    page.wait_for_selector(BANNER, state="visible")
    page.wait_for_function("window.__pi.count('api_catalogue') === 1 && window.__pi.idle()")
    settled(page)
    return page


def page_url(context) -> str:
    return context.pages[0].url.split("#")[0]


def hide(page) -> None:
    """Hide the page, as closing the tab, reloading it or putting the phone away does."""
    page.evaluate(
        """() => {
            Object.defineProperty(document, "visibilityState", { value: "hidden", configurable: true });
            document.dispatchEvent(new Event("visibilitychange"));
        }"""
    )


def test_the_worked_example_is_not_saved(context, page):
    assert page.text_content("#library-count") == "0"
    again = context.new_page()
    again.goto(page.url)
    first_paint_done(again)
    assert again.is_visible("#demo-banner"), "its address shows it again"
    assert again.is_hidden(BANNER)
    assert again.text_content("#library-count") == "0"


def test_a_reader_who_comes_back_picks_up_where_they_left_off(context, page):
    generate(page, design_type="full_factorial", n_center_points=2)
    practise(page)
    saved(page, 1)
    sheet, typed = run_sheet(page), results(page)
    coefficients = page.text_content("#analysis table")

    assert page.url == page_url(context), "the run sheet is the reader's own: the address no longer names the example"
    again = come_back(context)
    assert again.text_content(BANNER).startswith(
        f"Picked up where you left off: {YIELD}Full factorial (2-level) in T, P and F"
    )
    assert again.is_hidden("#demo-banner")
    assert again.get_attribute("#workspace", "data-phase") == "learn", "a study with results opens on its analysis"
    assert run_sheet(again) == sheet
    assert results(again) == typed, "the simulated results, as they were"
    assert again.text_content("#analysis table") == coefficients, "analysed again once Python started"
    assert again.input_value("#design-type") == "full_factorial"


def test_trying_designs_keeps_one_entry_until_a_result_is_typed_in(blank_page):
    """Each design tried while simulating takes the last one's place; a design with a typed result is kept."""
    page = blank_page
    generate(page, design_type="full_factorial", n_center_points=0)
    saved(page, 1)
    generate(page, design_type="full_factorial", n_center_points=3)
    practise(page)
    generate(page, design_type="box_behnken")  # simulated again, on the new runs
    page.wait_for_timeout(600)
    saved(page, 1)
    page.fill("#design-table input[data-index='0']", "12.5")
    generate(page, design_type="ccd")
    saved(page, 2)


def test_a_study_s_runs_are_ordered_with_the_moment_it_began_kept_for_its_life(blank_page):
    """The seed is the second the study began: kept while designs are tried, new once a result is typed in."""
    page = blank_page

    def seed() -> int:
        (found,) = re.findall(r"random_state=(\d+)", page.text_content("#py-design code"))
        return int(found)

    generate(page, design_type="full_factorial", n_center_points=0)
    first = seed()
    now = page.evaluate("Math.floor(Date.now() / 1000)")
    assert now - 600 <= first <= now, "the second the study began, when its brief was first compared"
    generate(page, design_type="box_behnken")
    assert seed() == first, "the design tried in its place keeps the study's seed"
    page.wait_for_timeout(1100)  # the next study begins in a later second
    page.fill("#design-table input[data-index='0']", "12.5")
    generate(page, design_type="ccd")
    assert seed() > first, "a study with a typed result is kept, and the next one has a seed of its own"


def test_a_result_typed_just_before_the_next_design_is_saved_with_its_design(context, site):
    """Making the next design saves the one before at once, with the result typed a moment earlier.

    The page saves a moment after the last change. The page's timers are stopped between typing
    the result and the next design, so the result can only be saved by the switch itself.
    """
    context.clock.install()
    page = context.new_page()
    page.goto(f"{site}?example=worked")
    first_paint_done(page)
    generate(page, design_type="full_factorial", n_center_points=0)
    saved(page, 1)
    choose(page, design_type="ccd", n_center_points=2)
    page.clock.pause_at(page.evaluate("Date.now()"))
    reveal(page, "#design-table")
    page.fill("#design-table input[data-index='0']", "12.5")
    reveal(page, "#generate")
    page.click("#generate")
    page.wait_for_function("window.__pi.count('api_make_design') === 2 && window.__pi.idle()")
    page.clock.resume()
    saved(page, 2)
    page.click("#library summary")
    page.click("#library-list >> text=Full factorial (2-level) in T, P and F")
    settled(page)
    assert results(page)[0] == "12.5"


def test_a_result_typed_just_before_the_page_is_hidden_is_saved(context, site):
    """Closing the tab or putting the phone away straight after typing a result keeps it."""
    context.clock.install()
    page = context.new_page()
    page.goto(f"{site}?example=worked")
    first_paint_done(page)
    generate(page, design_type="full_factorial", n_center_points=0)
    saved(page, 1)
    page.clock.pause_at(page.evaluate("Date.now()"))
    page.fill("#design-table input[data-index='0']", "12.5")
    hide(page)
    # The browser writes it in the background; the page's timers stay stopped.
    assert saved_studies(page, lambda saved: saved[0]["responses"][0] == 12.5)[0]["responses"][0] == 12.5


def test_your_studies_opens_an_earlier_study_and_deletes_one(page):
    generate(page, design_type="full_factorial", n_center_points=0)
    practise(page)
    page.fill("#design-table input[data-index='0']", "12.5")  # a result typed in: this study is kept
    generate(page, design_type="ccd", n_center_points=2)
    saved(page, 2)
    page.click("#library summary")
    page.click("#library-list >> text=Full factorial (2-level) in T, P and F")
    settled(page)
    assert page.input_value("#design-type") == "full_factorial"
    assert page.text_content(BANNER) == f"Opened {YIELD}Full factorial (2-level) in T, P and F."
    assert len(run_sheet(page)) == 8
    assert all(results(page)), "with its results"

    page.click(f"#library-list >> role=button[name='Delete {YIELD}Central composite (CCD) in T, P and F']")
    page.wait_for_selector(f"#library-list >> text=Deleted {YIELD}Central composite (CCD) in T, P and F.")
    saved(page, 1)
    page.click("#library-list >> role=button[name='Undo']")
    saved(page, 2)


@pytest.mark.parametrize(
    "context_options",
    [{"viewport": {"width": 744, "height": 1000}}, {"viewport": {"width": 390, "height": 844}, "is_mobile": True}],
    ids=["ipad-mini", "phone"],
)
def test_your_studies_puts_delete_at_the_right_end_of_each_row(page):
    """Each row's Delete, and its Undo once pressed, ends where the row ends, however long the title is."""
    generate(page, design_type="full_factorial", n_center_points=0)
    practise(page)
    page.fill("#design-table input[data-index='0']", "12.5")  # a result typed in: this study is kept
    generate(page, design_type="ccd", n_center_points=2)
    saved(page, 2)
    page.click("#library summary")

    def right_ends(name: str) -> tuple[float, float]:
        button = page.locator("#library-list").get_by_role("button", name=name)
        row = button.locator("xpath=ancestor::li")
        return tuple(round(b["x"] + b["width"]) for b in (button.bounding_box(), row.bounding_box()))

    end, row = right_ends(f"Delete {YIELD}Full factorial (2-level) in T, P and F")
    assert end == row
    page.click(f"#library-list >> role=button[name='Delete {YIELD}Full factorial (2-level) in T, P and F']")
    end, row = right_ends("Undo")
    assert end == row


def test_a_link_opens_the_design_and_its_results_in_another_browser(open_context, context, page, site):
    generate(page, design_type="full_factorial", n_center_points=2)
    practise(page)
    page.click("#share")
    page.wait_for_selector("#share-out", state="visible")
    link = page.input_value("#share-url")
    assert link.startswith(f"{site}#c=")
    assert "Whoever opens it gets this study and its results" in page.text_content("#share-note")

    elsewhere = open_context().new_page()
    elsewhere.goto(link)
    elsewhere.wait_for_selector(BANNER, state="visible")
    elsewhere.wait_for_function("window.__pi.count('api_catalogue') === 1 && window.__pi.idle()")
    settled(elsewhere)
    assert elsewhere.text_content(BANNER).startswith(
        f"Opened from a link: {YIELD}Full factorial (2-level) in T, P and F."
    )
    assert elsewhere.input_value("#response") == "Yield", "the response as the reader described it travels too"
    assert run_sheet(elsewhere) == run_sheet(page)
    assert results(elsewhere) == results(page)
    assert elsewhere.text_content("#truth-equation") == page.text_content("#truth-equation")
    assert elsewhere.url == site, "the address loses the link, so a reload opens the saved copy"
    saved(elsewhere, 1)


def test_a_copied_link_says_so_where_the_reader_is_looking(context, page, site):
    context.grant_permissions(["clipboard-read", "clipboard-write"], origin=site.rstrip("/"))
    generate(page, design_type="full_factorial", n_center_points=0)
    page.click("#share")
    page.wait_for_selector("#toast", state="visible")
    assert page.text_content("#toast") == "Link copied to the clipboard: paste it into an email or a chat."
    assert page.get_attribute("#toast", "data-state") == "ok"
    assert page.evaluate("navigator.clipboard.readText()") == page.input_value("#share-url")
    assert page.text_content("#share-note").startswith("Whoever opens it gets"), "the toast says it was copied"
    page.wait_for_selector("#toast", state="hidden", timeout=6_000)


def test_a_link_the_browser_will_not_copy_is_selected_to_copy_by_hand(page):
    page.evaluate(
        "() => { navigator.clipboard.writeText = () =>"
        " Promise.reject(new DOMException('Write permission denied.', 'NotAllowedError')); }"
    )
    generate(page, design_type="full_factorial", n_center_points=0)
    page.click("#share")
    page.wait_for_selector("#toast", state="visible")
    assert page.get_attribute("#toast", "data-state") == "error"
    assert page.text_content("#toast").startswith("The browser would not copy the link")
    assert page.text_content("#share-note").startswith("Copy the link from the box.")


def test_a_link_without_its_results_carries_the_design_only(open_context, page):
    generate(page, design_type="full_factorial", n_center_points=0)
    practise(page)
    page.uncheck("#share-results")
    page.click("#share")
    page.wait_for_selector("#share-out", state="visible")
    assert "without its results" in page.text_content("#share-note")
    elsewhere = open_context().new_page()
    elsewhere.goto(page.input_value("#share-url"))
    elsewhere.wait_for_selector(BANNER, state="visible")
    settled(elsewhere)
    assert [row[:-1] for row in run_sheet(elsewhere)] == [row[:-1] for row in run_sheet(page)]
    assert results(elsewhere) == [""] * 8
    assert elsewhere.get_attribute("#workspace", "data-phase") == "run", (
        "a study without results opens on its run sheet"
    )
    assert elsewhere.is_hidden("#analysis")


def test_a_damaged_link_is_refused_and_a_new_study_started(context, site):
    page = context.new_page()
    page.goto(f"{site}#c=bm90IGEgZGVzaWdu")
    page.wait_for_function("window.__pi.count('api_catalogue') === 1 && window.__pi.idle()")
    assert page.text_content(BANNER) == "This design could not be opened: the link has been cut or changed."
    assert page.get_attribute(BANNER, "class") == "banner error"
    assert page.get_attribute("#workspace", "data-phase") == "plan"
    assert page.is_hidden("#demo-banner")
    assert status(page)[1] == "ok"


def test_the_notebook_holds_each_step_s_code(page, tmp_path: Path):
    generate(page, design_type="full_factorial", n_center_points=2)
    practise(page)
    with page.expect_download() as download:
        page.click("#notebook")
    assert download.value.suggested_filename == "full_factorial-10-runs.ipynb"
    target = tmp_path / "book.ipynb"
    download.value.save_as(target)
    book = json.loads(target.read_text(encoding="utf-8"))
    assert book["nbformat"] == 4
    code = ["".join(cell["source"]) for cell in book["cells"] if cell["cell_type"] == "code"]
    assert code[1] == page.text_content("#py-design code"), "the design's code, after the install line"
    analysis = page.text_content("#py-analyse code")

    def body(cell: str) -> str:
        return "\n".join(line for line in cell.splitlines() if not line.startswith(("import ", "from "))).strip()

    assert all(body(cell) in analysis for cell in code[1:4]), "a cell a step, as the analysis panel holds them all"
    best = page.text_content("#py-best code")
    assert all(body(cell) in best for cell in code[1:]), "and the best settings, as their panel holds every step"
    assert len(code) == 5, "the install line, the design, the results, the analysis and the best settings"


@pytest.mark.parametrize("context_options", [{"viewport": {"width": 390, "height": 844}, "is_mobile": True}])
def test_on_a_phone_the_link_and_the_list_fit(page):
    generate(page, design_type="full_factorial", n_center_points=0)
    practise(page)
    page.click("#share")
    page.wait_for_selector("#share-out", state="visible")
    assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")


def test_results_sent_back_by_link_fill_in_the_design_kept_here(open_context, blank_page):
    """The engineer shares the design; the operator types the results on a phone and sends a link back."""
    engineer = blank_page
    generate(engineer, design_type="full_factorial", n_center_points=0)
    saved(engineer, 1)
    engineer.click("#share")
    engineer.wait_for_selector("#share-out", state="visible")

    operator = open_context().new_page()
    operator.goto(engineer.input_value("#share-url"))
    operator.wait_for_selector(BANNER, state="visible")
    settled(operator)
    typed = [str(10 + i) for i in range(8)]
    for i, value in enumerate(typed):
        operator.fill(f"#design-table input[data-index='{i}']", value)
    settled(operator)
    operator.click("#share")
    operator.wait_for_selector("#share-out", state="visible")

    back = engineer.context.new_page()
    back.goto(operator.input_value("#share-url"))
    back.wait_for_selector(BANNER, state="visible")
    settled(back)
    assert back.text_content(BANNER) == (
        "Added 8 results from a link to Full factorial (2-level) in T, P and F, kept in this browser."
    )
    assert results(back) == typed
    saved(back, 1)  # filled in, not saved again as a copy
    assert "8 completed runs" in back.text_content("#analysis")


def test_a_brief_not_yet_made_into_a_run_sheet_is_kept_as_a_draft(context, fresh_page):
    """A reload, or a tab closed and opened again, finds the brief as the reader left it, in Design."""
    name_response(fresh_page, "Purity", "%", goal="target", target=99.5)
    set_factors(fresh_page, FACTORS[:2])
    fresh_page.check('input[name="aim"][value="interactions"]')
    fresh_page.wait_for_timeout(500)  # the draft is written a moment after the last change
    again = context.new_page()
    again.goto(page_url(context))
    again.wait_for_function("window.__pi.count('api_catalogue') === 1 && window.__pi.idle()")
    assert again.input_value("#response") == "Purity"
    assert again.input_value("#response-target") == "99.5"
    assert [f["name"] for f in factor_table(again)] == ["T", "P"]
    assert again.is_checked('input[name="aim"][value="interactions"]')
    assert again.text_content("#study-title") == "Purity (%), target 99.5"


def test_a_brief_typed_just_before_the_page_is_hidden_is_kept(context, site):
    """A press on the logo, or a reload, straight after typing in the brief keeps what was typed.

    The page writes the draft a moment after the last change. Its timers are stopped once the page
    has opened, so the draft can only be written by the page being hidden.
    """
    context.clock.install()
    page = context.new_page()
    page.goto(site)
    page.wait_for_function("window.__pi.count('api_catalogue') === 1 && window.__pi.idle()")
    page.clock.pause_at(page.evaluate("Date.now()"))
    page.fill("#response", "Purity")
    assert page.evaluate("localStorage.getItem('factorial-draft')") is None, "not yet: its moment has not come"
    hide(page)
    assert json.loads(page.evaluate("localStorage.getItem('factorial-draft')"))["response"]["label"] == "Purity"


def test_the_logo_opens_the_home_page_afresh(page, site):
    """The mark and the name link to the page's own address: what a reload finds there, not the worked example."""
    with page.expect_navigation():
        page.click(".brand__home .mark")
    page.wait_for_function("window.__pi.count('api_catalogue') === 1 && window.__pi.idle()")
    assert page.url == site
    assert page.text_content("#study-title") == "A new study", "the worked example is not kept"
    name_response(page, "Purity", "%")
    with page.expect_navigation():
        page.click(".brand__home h1")
    page.wait_for_function("window.__pi.count('api_catalogue') === 1 && window.__pi.idle()")
    assert page.input_value("#response") == "Purity", "the brief, as it was left"


def test_making_the_run_sheet_turns_the_draft_into_a_saved_study(context, blank_page):
    generate(blank_page, design_type="full_factorial")
    saved(blank_page, 1)
    again = come_back(context)
    assert again.text_content(BANNER).startswith("Picked up where you left off: Full factorial (2-level) in T, P and F")
    assert again.get_attribute("#workspace", "data-phase") == "run"


def test_a_new_study_keeps_the_last_one_and_starts_at_its_response(page):
    generate(page, design_type="full_factorial", n_center_points=0)
    saved(page, 1)
    page.click("#new-study")
    assert page.get_attribute("#workspace", "data-phase") == "plan"
    assert page.input_value("#response") == ""
    assert factor_table(page) == [{"name": "", "low": "", "high": "", "units": ""}] * 2
    assert page.is_hidden("#run-out")
    assert page.text_content("#study-title") == "A new study"
    page.click("#library summary")
    assert page.locator("#library-list li").count() == 1, "the study before is kept"
