"""The study as a report: one HTML page to read or to save as a PDF, stamped with how to get its numbers again.

Learn views it in a tab of its own, and saves it as a PDF through the browser's own print. It holds the brief, the
design and its run sheet with the results, what Learn makes of each response's results (the analysis, its charts as
images, the best settings), the next step and the Python code, under a stamp: when and with which releases it was
made, the run order's seed, where the results came from, the SHA-256 of the study's inputs, and a link that opens the
study again.
"""

from __future__ import annotations

import itertools
import platform
from pathlib import Path
from typing import Any

from tests.web.e2e._page import practise, reveal, settled, status, study_menu
from tests.web.e2e.test_responses import pick, tabs, three_responses


def report_html(page: Any) -> str:  # noqa: ANN401 - a Playwright Page
    """Take the study's report from Learn, once there is an analysis, as the file a reader gets: return its text.

    The page gives the file where the browser opens no tab, which is how it is taken here. The report's own tab is at
    a blob: address, which Playwright's WebKit will not open while the test answers the page's requests; the tab is
    tested where it is the subject, below.
    """
    reveal(page, "#learn-report-view")
    page.evaluate("window.opensTabs = window.open; window.open = () => null")
    with page.expect_download() as download:
        page.click("#learn-report-view")
    page.evaluate("window.open = window.opensTabs")
    settled(page)
    return Path(download.value.path()).read_text(encoding="utf-8")


def viewed_report(page: Any) -> Any:  # noqa: ANN401 - a Playwright Page
    """Open the study's report from its menu, as a reader does at any stage, results or none: return its tab."""
    with page.expect_popup() as popup:
        study_menu(page, "#menu-report")
    report = popup.value
    report.wait_for_url("blob:**")
    report.wait_for_load_state()
    settled(page)
    return report


def opened(page: Any, html: str) -> Any:  # noqa: ANN401
    """Open a report in a page of its own, in the same browser, with no network: return that page."""
    report = page.context.new_page()
    report.route("**/*", lambda route: route.abort())
    report.set_content(html)
    return report


def stamp(report: Any) -> dict[str, str]:  # noqa: ANN401
    """Return the report's stamp by its rows' labels, with the software versions at its foot as ``On``."""
    return report.evaluate(
        """() => {
          const cells = (tr) => [tr.cells[0].textContent, tr.cells[1].textContent];
          const rows = [...document.querySelectorAll('table.stamp tr')].map(cells);
          const said = document.querySelector('footer .versions')?.textContent;
          const foot = said?.match(/^Software versions: (.*)\\.$/)?.[1];
          return Object.fromEntries(foot ? [...rows, ['On', foot]] : rows);
        }"""
    )


def headings(report: Any) -> list[str]:  # noqa: ANN401
    return report.locator("h2").all_text_contents()


def test_the_report_names_what_a_model_was_fitted_with_once_the_worker_has_fetched_it(page):
    """Start-up leaves statsmodels out, so the catalog names what Python ran on then; the worker says the rest."""
    runtime = {"python": "3.14.2", "packages": {"numpy": "2.4.6", "statsmodels": "0.14.6"}, "pyodide": "314.0.7"}
    page.evaluate("(runtime) => window.__pi.warmed(runtime)", runtime)
    said = "Python 3.14.2 in Pyodide 314.0.7, with numpy 2.4.6 and statsmodels 0.14.6"
    assert stamp(opened(page, report_html(page)))["On"] == said
    page.evaluate("window.__pi.warmed(false)")  # a worker started again that could not fetch it changes nothing
    assert stamp(opened(page, report_html(page)))["On"] == said


def test_the_worked_example_s_report_holds_the_study_and_what_learn_makes_of_it(page, first_paint):
    html = report_html(page)
    report = opened(page, html)
    assert report.text_content("h1") == "Yield (%), maximize: Box-Behnken in Temperature, Pressure and Flow", (
        "the study's title"
    )
    assert headings(report) == [
        "The brief",
        "The design: Box-Behnken, 15 runs",
        "What the results say",
        "The next step",
        "The Python code",
        "Terms used",
    ]
    # The words the report used that the page defines, once each, in the order the report used them.
    terms = report.locator("dl.terms dt").all_text_contents()
    assert terms[:3] == ["Coded units", "R²", "Adjusted R²"]
    assert {"Leverage", "Lack of fit", "Confidence interval"} <= set(terms)
    assert len(terms) == len(set(terms)) == report.locator("dl.terms dd").count()
    assert "Terms in the model" not in terms, "how to use the page is of no use on paper"
    assert report.locator("dl.terms dd").first.text_content().startswith("Each factor centered and scaled")
    said = stamp(report)
    assert said["With"].startswith("Factorial (process-improve")
    assert said["On"].startswith(f"Python {platform.python_version()}, with numpy ")
    assert said["Run order"] == "In random order (seed 42)."
    label, sigma, seed = (first_paint["simulation"][key] for key in ("label", "sigma", "seed"))
    assert said["Results"] == f"15 of 15 runs, simulated for practice ({label}, noise {sigma:g}, seed {seed})."
    assert len(said["Fingerprint (SHA-256)"]) == 64

    sheet = report.locator("section:nth-of-type(2) table")
    assert sheet.locator("tbody tr").count() == 15
    assert sheet.locator("td.simulated").count() == 15, "every result simulated, in italics"
    learn = report.locator("section:nth-of-type(3)")
    heads = ["What each factor does", "Coefficients", "ANOVA", "The best settings for your goal"]
    assert learn.locator("h3").all_text_contents() == heads
    assert learn.locator("li").first.text_content().startswith("Temperature matters most: ")
    assert learn.locator("details.diagnostics[open] tbody tr").count() == 15, "each run, opened for paper"
    assert learn.locator(".equation").text_content() == page.text_content("#analysis .equation")
    coefficients = learn.locator("table").first.locator("thead th").all_text_contents()
    assert coefficients == ["Term", "95% low", "Coefficient", "95% high", "Std. error", "p-value", "True value"]
    assert learn.locator("table").first.locator("tbody td.strong").count() == 10, (
        "each estimate stands out on paper too"
    )
    assert "Download this table" not in learn.text_content(), "paper has no button to press"
    images = report.locator("figure img")
    assert images.count() >= 3, "the coefficients, the response surface and the residuals at least"
    assert all(src.startswith("data:image/png;base64,") for src in images.evaluate_all("(i) => i.map((x) => x.src)"))
    assert report.locator("button, input, select, script").count() == 0, "nothing to use, only to read"
    assert "from process_improve.experiments import" in report.text_content("pre code")


def test_the_report_opens_in_a_tab_of_its_own_to_read(page):
    """From Learn, beside the analysis, and from the study's menu, at any stage."""
    reveal(page, "#learn-report-view")
    with page.expect_popup() as popup:
        page.click("#learn-report-view")
    report = popup.value
    report.wait_for_url("blob:**")
    report.wait_for_load_state()
    for shown in (report, viewed_report(page)):
        assert shown.text_content("h1") == "Yield (%), maximize: Box-Behnken in Temperature, Pressure and Flow"
        assert "The next step" in headings(shown)
        assert status(page)[0].startswith("The report is open in a tab of its own")
        shown.close()


def test_save_as_pdf_opens_the_report_with_the_browser_s_print_over_it(page):
    """A page cannot write a PDF file: the browser's print can, so the button opens the report and asks for it."""
    # In place of the print dialog, which no test can answer, each tab counts how often it was asked to print.
    page.context.add_init_script("window.print = () => { window.printed = (window.printed ?? 0) + 1; };")
    reveal(page, "#learn-report-pdf")
    assert page.text_content("#learn-report-pdf") == "Save as PDF"
    with page.expect_popup() as popup:
        page.click("#learn-report-pdf")
    report = popup.value
    report.wait_for_url("blob:**")
    report.wait_for_function("window.printed === 1")
    assert report.title() == "Yield (%), maximize: Box-Behnken in Temperature, Pressure and Flow: report", (
        "the name the browser gives the PDF"
    )
    assert "@media print" in report.text_content("style"), "laid out for paper by the report's own styles"
    assert status(page)[0] == (
        'The report is open in a tab of its own, with the print dialog over it: choose "Save as PDF" there.'
    )
    report.wait_for_timeout(400)
    assert report.evaluate("window.printed") == 1, "asked once, and no more"
    report.close()


def test_where_the_browser_opens_no_tab_the_report_is_downloaded_as_a_file(page):
    page.evaluate("window.open = () => null")  # as a browser that blocks a new tab answers
    for button in ("#learn-report-view", "#learn-report-pdf"):
        reveal(page, button)
        with page.expect_download() as download:
            page.click(button)
        settled(page)
        assert download.value.suggested_filename == "box_behnken-15-runs-report.html"
        html = Path(download.value.path()).read_text(encoding="utf-8")
        assert opened(page, html).text_content("h1").startswith("Yield (%), maximize: Box-Behnken")
        assert status(page)[0].startswith("The browser opened no tab, so the report is downloaded")


def test_the_three_ways_to_take_the_study_away_are_one_control_named_report(page):
    reveal(page, "#learn-report")
    group = page.locator("#learn-report [role=group]")
    assert page.text_content("#learn-report-name") == "Report"
    assert group.get_attribute("aria-labelledby") == "learn-report-name"
    buttons = group.locator("button")
    assert buttons.all_text_contents() == ["View", "Save as PDF", "Python notebook"]
    for button in buttons.all():
        assert button.text_content() in button.get_attribute("aria-label"), "its full name holds the word it shows"
    boxes = [button.bounding_box() for button in buttons.all()]
    assert len({box["y"] for box in boxes}) == 1, "one row"
    for left, right in itertools.pairwise(boxes):
        assert right["x"] - (left["x"] + left["width"]) <= 0, "each touches the next"


def test_the_digest_is_of_what_the_link_carries(page):
    html = report_html(page)
    report = opened(page, html)
    digest = stamp(report)["Fingerprint (SHA-256)"]
    link = report.get_attribute("table.stamp a", "href")
    assert link.startswith(page.url.split("?")[0] + "#c=")
    again = page.evaluate(
        """async (packed) => {
            const { unpackCampaign } = await import('./campaign.js');
            const { canonicalJson, sha256 } = await import('./report.js');
            return sha256(canonicalJson(await unpackCampaign(packed)));
        }""",
        link.split("#c=")[1],
    )
    assert again == digest, "the study the link opens is the one the digest is of"


def test_a_report_is_the_same_study_until_the_study_changes(page):
    first = report_html(page)
    second = report_html(page)
    digest = [stamp(opened(page, html))["Fingerprint (SHA-256)"] for html in (first, second)]
    assert digest[0] == digest[1], "made twice, it is the same report"
    reveal(page, "#design-table")
    box = page.locator("#design-table input[data-response]").first
    box.fill("99")
    box.press("Tab")
    changed = report_html(page)
    assert stamp(opened(page, changed))["Fingerprint (SHA-256)"] != digest[0], "a result changed is another study"
    assert stamp(opened(page, changed))["Results"].startswith("15 of 15 runs, 1 entered and 14 simulated")


def test_a_report_holds_each_response_s_analysis_and_learn_keeps_the_one_it_showed(fresh_page):
    page = fresh_page
    three_responses(page)
    practise(page)
    pick(page, "Cost")
    html = report_html(page)
    report = opened(page, html)
    assert headings(report)[2:5] == [
        "What the results say of Yield",
        "What the results say of Cost",
        "What the results say of Purity",
    ]
    for section, label in zip(report.locator("section").all()[2:5], ["Yield", "Cost", "Purity"], strict=True):
        assert section.locator(".equation").text_content().startswith(f"{label} = ")
    assert tabs(page)[1] == ["Cost", "true"], "Learn shows the response it showed before"
    assert page.text_content("#analysis .equation").startswith("Cost = ")


def test_learn_offers_the_report_only_with_an_analysis(fresh_page):
    page = fresh_page
    assert page.locator("#learn-report").is_hidden()
    three_responses(page)
    assert page.locator("#learn-report").is_hidden(), "a run sheet with no results has nothing for Learn to report"
    report = viewed_report(page)  # the study's menu still opens one, of the brief and the design
    assert headings(report) == ["The brief", "The design: Box-Behnken, 15 runs", "The Python code"]
    assert stamp(report)["Results"] == "None yet."
