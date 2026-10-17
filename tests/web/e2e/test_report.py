"""The study as a report: one HTML page to keep or to send, with a stamp that says how to get its numbers again.

Run and Learn each download it. It holds the brief, the design and its run sheet with the results, what
Learn makes of each response's results (the analysis, its charts as images, the best settings), the next
step and the Python code, under a stamp: when and with which releases it was made, the run order's seed,
where the results came from, the SHA-256 of the study's inputs, and a link that opens the study again.
"""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

from tests.web.e2e._page import practise, reveal, settled
from tests.web.e2e.test_responses import pick, tabs, three_responses


def download_report(page: Any, button: str = "#report") -> tuple[str, str]:  # noqa: ANN401 - a Playwright Page
    """Click a report button, as a reader would; return the file's name and its text."""
    reveal(page, button)
    with page.expect_download() as download:
        page.click(button)
    settled(page)
    return download.value.suggested_filename, Path(download.value.path()).read_text(encoding="utf-8")


def opened(page: Any, html: str) -> Any:  # noqa: ANN401
    """Open a report in a page of its own, in the same browser, with no network: return that page."""
    report = page.context.new_page()
    report.route("**/*", lambda route: route.abort())
    report.set_content(html)
    return report


def stamp(report: Any) -> dict[str, str]:  # noqa: ANN401
    return report.evaluate(
        "() => Object.fromEntries([...document.querySelectorAll('table.stamp tr')]"
        ".map((tr) => [tr.cells[0].textContent, tr.cells[1].textContent]))"
    )


def headings(report: Any) -> list[str]:  # noqa: ANN401
    return report.locator("h2").all_text_contents()


def test_the_worked_example_s_report_holds_the_study_and_what_learn_makes_of_it(page, first_paint):
    name, html = download_report(page)
    assert name == "box_behnken-15-runs-report.html"
    report = opened(page, html)
    assert report.text_content("h1") == "Yield (%), maximise: Box-Behnken in T, P and F", "the study's title"
    assert headings(report) == [
        "The brief",
        "The design: Box-Behnken, 15 runs",
        "What the results say",
        "The next step",
        "The Python code",
    ]
    said = stamp(report)
    assert said["With"].startswith("process-improve")
    assert said["On"].startswith(f"Python {platform.python_version()}, with numpy ")
    assert said["Run order"] == "Drawn at random with seed 42."
    label, sigma, seed = (first_paint["simulation"][key] for key in ("label", "sigma", "seed"))
    assert said["Results"] == f"15 of 15 runs, simulated: {label}, with noise {sigma:g} and seed {seed}."
    assert len(said["Inputs, SHA-256"]) == 64

    sheet = report.locator("section:nth-of-type(2) table")
    assert sheet.locator("tbody tr").count() == 15
    assert sheet.locator("td.simulated").count() == 15, "every result simulated, in italics"
    learn = report.locator("section:nth-of-type(3)")
    assert learn.locator("h3").all_text_contents() == ["Coefficients", "ANOVA", "The best settings for your goal"]
    assert learn.locator(".equation").text_content() == page.text_content("#analysis .equation")
    coefficients = learn.locator("table").first.locator("thead th").all_text_contents()
    assert coefficients == ["Term", "Coefficient", "Std. error", "95% interval", "p-value", "True value"]
    images = report.locator("figure img")
    assert images.count() >= 3, "the coefficients, the response surface and the residuals at least"
    assert all(src.startswith("data:image/png;base64,") for src in images.evaluate_all("(i) => i.map((x) => x.src)"))
    assert report.locator("button, input, select, script").count() == 0, "nothing to use, only to read"
    assert "from process_improve.experiments import" in report.text_content("pre code")


def test_the_digest_is_of_what_the_link_carries(page):
    _, html = download_report(page)
    report = opened(page, html)
    digest = stamp(report)["Inputs, SHA-256"]
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
    _, first = download_report(page)
    _, second = download_report(page, "#learn-report-button")
    digest = [stamp(opened(page, html))["Inputs, SHA-256"] for html in (first, second)]
    assert digest[0] == digest[1], "Run's button and Learn's make the same report"
    reveal(page, "#design-table")
    box = page.locator("#design-table input[data-response]").first
    box.fill("99")
    box.press("Tab")
    _, changed = download_report(page)
    assert stamp(opened(page, changed))["Inputs, SHA-256"] != digest[0], "a result changed is another study"
    assert stamp(opened(page, changed))["Results"].startswith("15 of 15 runs, 1 entered and 14 simulated")


def test_a_report_holds_each_response_s_analysis_and_learn_keeps_the_one_it_showed(fresh_page):
    page = fresh_page
    three_responses(page)
    practise(page)
    pick(page, "Cost")
    _, html = download_report(page, "#learn-report-button")
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
    name, html = download_report(page)
    assert name == "box_behnken-15-runs-report.html"
    report = opened(page, html)
    assert headings(report) == ["The brief", "The design: Box-Behnken, 15 runs", "The Python code"]
    assert stamp(report)["Results"] == "None yet."
