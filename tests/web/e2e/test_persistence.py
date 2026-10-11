"""What Factorial keeps: each design and its results, in the reader's own browser, and in a link.

A context is one browser with storage of its own, so a page opened again in the same context is
the reader coming back, and a page in a second context is another reader, or another machine.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.web.e2e._page import first_paint_done, generate, run_sheet, settled, status

BANNER = "#campaign-banner"


def results(page) -> list[str]:
    """Read the results column of the run sheet as the reader sees it, in the sheet's order."""
    return page.eval_on_selector_all("#design-table input[data-index]", "inputs => inputs.map((i) => i.value)")


def simulate(page) -> None:
    page.click("#simulate")
    settled(page)


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


def test_the_worked_example_is_not_saved(context, page, first_paint):
    assert page.is_hidden("#library")
    again = context.new_page()
    again.goto(page.url)
    first_paint_done(again, first_paint)
    assert again.is_visible("#demo-banner"), "a reader who only looked comes back to the worked example"
    assert again.is_hidden(BANNER)


def test_a_reader_who_comes_back_picks_up_where_they_left_off(context, page):
    generate(page, design_type="full_factorial", n_center_points=2)
    simulate(page)
    saved(page, 1)
    sheet, typed = run_sheet(page), results(page)
    coefficients = page.text_content("#analysis table")

    again = come_back(context)
    assert again.text_content(BANNER).startswith("Picked up where you left off: Full factorial (2-level) in T, P and F")
    assert again.is_hidden("#demo-banner")
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
    simulate(page)
    generate(page, design_type="box_behnken")  # simulated again, on the new runs
    page.wait_for_timeout(600)
    saved(page, 1)
    page.fill("#design-table input[data-index='0']", "12.5")
    generate(page, design_type="ccd")
    saved(page, 2)


def test_your_designs_opens_an_earlier_design_and_deletes_one(page):
    generate(page, design_type="full_factorial", n_center_points=0)
    simulate(page)
    page.fill("#design-table input[data-index='0']", "12.5")  # a result typed in: this design is kept
    generate(page, design_type="ccd", n_center_points=2)
    saved(page, 2)
    page.click("#library summary")
    page.click("#library-list >> text=Full factorial (2-level) in T, P and F")
    settled(page)
    assert page.input_value("#design-type") == "full_factorial"
    assert page.text_content(BANNER) == "Opened Full factorial (2-level) in T, P and F."
    assert len(run_sheet(page)) == 8
    assert all(results(page)), "with its results"

    page.click("#library-list >> role=button[name='Delete Central composite (CCD) in T, P and F']")
    page.wait_for_selector("#library-list >> text=Deleted Central composite (CCD) in T, P and F.")
    saved(page, 1)
    page.click("#library-list >> role=button[name='Undo']")
    saved(page, 2)


def test_a_link_opens_the_design_and_its_results_in_another_browser(open_context, context, page, site):
    generate(page, design_type="full_factorial", n_center_points=2)
    simulate(page)
    page.click("#share")
    page.wait_for_selector("#share-out", state="visible")
    link = page.input_value("#share-url")
    assert link.startswith(f"{site}#c=")
    assert "Whoever opens it gets this design and its results" in page.text_content("#share-note")

    elsewhere = open_context().new_page()
    elsewhere.goto(link)
    elsewhere.wait_for_selector(BANNER, state="visible")
    elsewhere.wait_for_function("window.__pi.count('api_catalogue') === 1 && window.__pi.idle()")
    settled(elsewhere)
    assert elsewhere.text_content(BANNER).startswith("Opened from a link: Full factorial (2-level) in T, P and F.")
    assert run_sheet(elsewhere) == run_sheet(page)
    assert results(elsewhere) == results(page)
    assert elsewhere.text_content("#truth-equation") == page.text_content("#truth-equation")
    assert elsewhere.url == site, "the address loses the link, so a reload opens the saved copy"
    saved(elsewhere, 1)


def test_a_link_without_its_results_carries_the_design_only(open_context, page):
    generate(page, design_type="full_factorial", n_center_points=0)
    simulate(page)
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
    assert elsewhere.is_visible("#analyse-empty")


def test_a_damaged_link_is_refused_and_the_worked_example_shown(context, site, first_paint):
    page = context.new_page()
    page.goto(f"{site}#c=bm90IGEgZGVzaWdu")
    first_paint_done(page, first_paint)
    assert page.text_content(BANNER) == (
        "This design could not be opened: the link has been cut or changed. Here is the worked example instead."
    )
    assert page.get_attribute(BANNER, "class") == "banner error"
    assert page.is_visible("#demo-banner")
    assert status(page)[1] == "ok"


def test_the_notebook_holds_each_step_s_code(page, tmp_path: Path):
    generate(page, design_type="full_factorial", n_center_points=2)
    simulate(page)
    with page.expect_download() as download:
        page.click("#notebook")
    assert download.value.suggested_filename == "full_factorial-10-runs.ipynb"
    target = tmp_path / "book.ipynb"
    download.value.save_as(target)
    book = json.loads(target.read_text(encoding="utf-8"))
    assert book["nbformat"] == 4
    code = ["".join(cell["source"]) for cell in book["cells"] if cell["cell_type"] == "code"]
    shown = [page.text_content(f"#py-{step} code") for step in ("design", "run", "analyse")]
    assert code[1:] == shown, "the panels' code, in order, after the install line"


@pytest.mark.parametrize("context_options", [{"viewport": {"width": 390, "height": 844}, "is_mobile": True}])
def test_on_a_phone_the_link_and_the_list_fit(page):
    generate(page, design_type="full_factorial", n_center_points=0)
    simulate(page)
    page.click("#share")
    page.wait_for_selector("#share-out", state="visible")
    assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")


def test_the_printed_run_sheet_has_a_box_per_result_and_a_code_that_opens_the_design(open_context, page, tmp_path):
    generate(page, design_type="full_factorial", n_center_points=2)
    page.evaluate("() => { window.print = () => { window.__printed = (window.__printed ?? 0) + 1; }; }")
    page.click("#print")
    page.wait_for_function("window.__printed === 1")
    rows = page.locator("#print-sheet table").nth(1).locator("tbody tr")
    assert rows.count() == 10
    assert page.locator("#print-sheet td.result").all_text_contents() == [""] * 10
    assert page.text_content("#print-sheet h1") == "Full factorial (2-level) in T, P and F"

    page.emulate_media(media="print")
    assert page.is_visible("#print-sheet")
    assert page.is_hidden("#step-design"), "on paper, the run sheet alone"
    shot = tmp_path / "qr.png"
    page.locator("#print-sheet .qr img").screenshot(path=shot)

    zxingcpp = pytest.importorskip("zxingcpp", reason="zxing-cpp reads the printed code")
    image = pytest.importorskip("PIL.Image")
    read = zxingcpp.read_barcodes(image.open(shot))
    assert len(read) == 1
    link = read[0].text
    elsewhere = open_context().new_page()
    elsewhere.goto(link)
    elsewhere.wait_for_selector(BANNER, state="visible")
    settled(elsewhere)
    assert [row[:-1] for row in run_sheet(elsewhere)] == [row[:-1] for row in run_sheet(page)]
    assert results(elsewhere) == [""] * 10, "the code carries the design, not its results"


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
    assert "8 completed runs." in back.text_content("#analysis")
