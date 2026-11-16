"""Learn's model, its tables and each run are one fold, "For the statistician", closed until it is opened.

Learn leads with the answer, its checks and what the results say in sentences. The equation, the fit statistics,
the coefficients, the analysis of variance and each run as the model sees it are evidence: a press away, none
of it removed, and all of it in the report.
"""

from __future__ import annotations

import pytest

from tests.web.e2e._page import first_paint_done, open_statistician, settled

FOLD = "#statistician"


@pytest.fixture
def page(page):
    """Return the worked example as a reader who has never opened the fold finds it.

    The tests' harness opens the fold once a tab, so the tests of Learn's tables see them. Here the choice is taken
    back and the page loaded again.
    """
    page.evaluate("localStorage.removeItem('factorial-statistician')")
    page.reload()
    first_paint_done(page)
    return page


def test_the_fold_is_closed_and_learn_reads_in_sentences_first(page):
    assert not page.locator(FOLD).evaluate("(fold) => fold.open")
    assert page.text_content(f"{FOLD} > summary") == "For the statistician: the model, its coefficients and each run"
    assert page.is_visible("#findings"), "what the results say, in sentences"
    for part in ("#analysis .equation", "#analysis table", "#diagnostics"):
        assert page.locator(part).first.is_hidden(), part
    fold, findings, charts = (page.locator(part).bounding_box() for part in (FOLD, "#findings", "#charts"))
    assert findings["y"] < fold["y"] < charts["y"], "after the findings, before the charts"


def test_the_fold_opens_by_keyboard_and_holds_the_model_its_tables_and_each_run(page):
    page.focus(f"{FOLD} > summary")
    page.keyboard.press("Enter")
    assert page.is_visible("#analysis .equation")
    held = page.locator(f"{FOLD} h3").all_text_contents()
    assert held[0] == "Coefficients"
    assert "ANOVA" in held
    assert page.locator(f"{FOLD} .fit").text_content().count("R²") == 3, "R², adjusted and predicted"
    assert page.is_visible("#diagnostics > summary"), "each run, as the model sees it"
    assert page.locator(f"{FOLD} table").count() >= 2


def refit(page, model: str) -> None:
    """Choose a model, and wait until Learn shows the fit of it: another equation than the one on the page."""
    before = page.text_content("#analysis .equation")
    page.select_option("#model", model)
    page.wait_for_function("(was) => document.querySelector('#analysis .equation')?.textContent !== was", arg=before)
    settled(page)


def test_the_fold_stays_as_the_reader_has_it_through_a_new_fit(page):
    open_statistician(page)
    refit(page, "main_effects")
    assert page.locator(FOLD).evaluate("(fold) => fold.open")
    page.click(f"{FOLD} > summary")
    refit(page, "quadratic")
    assert not page.locator(FOLD).evaluate("(fold) => fold.open")


def test_the_fold_a_reader_opened_is_open_at_the_next_visit(page):
    open_statistician(page)
    page.reload()
    first_paint_done(page)
    assert page.locator(FOLD).evaluate("(fold) => fold.open")
    assert page.is_visible("#analysis .equation")
    page.click(f"{FOLD} > summary")
    page.reload()
    first_paint_done(page)
    assert not page.locator(FOLD).evaluate("(fold) => fold.open"), "and closed again once the reader closed it"


def test_a_check_s_evidence_inside_the_fold_opens_the_fold(page):
    runs = page.locator("#checks [data-check=runs]")
    runs.locator("summary").click()
    runs.get_by_role("button", name="See each run as the model sees it").click()
    page.wait_for_selector("#diagnostics.shown-part")
    assert page.locator(FOLD).evaluate("(fold) => fold.open")
    assert page.is_visible("#diagnostics > summary")


def test_the_report_holds_every_table_whether_the_fold_is_open_or_not(page):
    with page.expect_popup() as popup:
        page.click("#learn-report-view")
    report = popup.value
    report.wait_for_url("blob:**")
    text = report.text_content("body")
    for held in ("Coefficients", "ANOVA", "Each run, as the model sees it", page.text_content("#analysis .equation")):
        assert held in text, held


@pytest.mark.parametrize("context_options", [{"viewport": {"width": 1440, "height": 900}}])
def test_with_the_fold_closed_learn_is_shorter_than_with_it_open(page):
    closed = page.evaluate("document.documentElement.scrollHeight")
    open_statistician(page)
    opened = page.evaluate("document.documentElement.scrollHeight")
    assert opened - closed > 900, f"the tables are more than a screen of their own: {closed} px closed, {opened} open"
