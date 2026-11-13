"""Whole studies, start to finish: what a reader does in one sitting, each step handing its work to the next.

The other browser tests take one part of the page each. These follow a reader through a study instead, and check
what passes between the parts: the brief into the design, the design into the run sheet, the results into what
Learn says, a choice made in Learn into the report, the study into the browser's store and out again by link.
A change that breaks a hand-off fails here, even when each part still works alone.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.web._support import FACTORS, fill_workbook, quadratic
from tests.web.e2e._page import (
    calls,
    candidates_done,
    enter_results,
    generate,
    headers,
    last_reply,
    name_response,
    practise,
    reveal,
    run_sheet,
    saved_studies,
    set_factors,
    settled,
    status,
)
from tests.web.e2e.test_analysis import coefficient_rows, set_aside, terms_fitted
from tests.web.e2e.test_balance import balanced
from tests.web.e2e.test_import import paste
from tests.web.e2e.test_levels import listed_designs, more_charts, set_level
from tests.web.e2e.test_mixture import BLEND
from tests.web.e2e.test_next_step import add, offered
from tests.web.e2e.test_persistence import BANNER, come_back, results
from tests.web.e2e.test_report import download_report, headings, opened
from tests.web.e2e.test_responses import pick, tabs, three_responses
from tests.web.e2e.test_workbook import upload


def findings(page: Any) -> list[str]:  # noqa: ANN401 - a Playwright Page
    """Read what the results say, in Learn, sentence by sentence."""
    return page.locator("#findings li").all_text_contents()


def typed_from(page: Any, response: Any) -> np.ndarray:  # noqa: ANN401
    """Type each run's result as a known process gives it for the run sheet's settings; return the results."""
    sheet = pd.DataFrame(run_sheet(page), columns=headers(page)).apply(pd.to_numeric, errors="coerce")
    values = np.round(response(sheet), 2)
    enter_results(page, dict(enumerate(values)))
    settled(page)
    return values


def download(page: Any, button: str) -> Any:  # noqa: ANN401
    """Press a button that saves a file, as a reader would; return the download."""
    reveal(page, button)
    with page.expect_download() as caught:
        page.click(button)
    return caught.value


# --------------------------------------------------------------------------- a study of the reader's own
#: The terms the search keeps of the quadratic model, for results that follow ``quadratic``.
KEPT = sorted(["Intercept", "T", "P", "T·P", "T²"])


def strength_study(page: Any) -> np.ndarray:  # noqa: ANN401
    """Describe a strength to maximise in the worked example's factors, make a Box-Behnken design, type its results."""
    name_response(page, "Strength", "MPa", goal="maximize")
    set_factors(page, FACTORS)
    candidates_done(page)
    generate(page, design_type="box_behnken")
    return typed_from(page, quadratic)


def simplified(page: Any) -> None:  # noqa: ANN401
    """Ask for a simpler model and accept it, as a reader would; wait for its fit."""
    reveal(page, "#simplify")
    page.click("#simplify-ask")
    page.wait_for_function("document.querySelector('#simplify-words').textContent.startsWith('The results support')")
    before = page.evaluate("window.__pi.count('api_analyze_runs')")
    page.click("#simplify-apply")
    page.wait_for_function(f"window.__pi.count('api_analyze_runs') > {before}")
    settled(page)


def test_a_study_goes_from_its_brief_to_its_report(fresh_page):
    """Brief, designs compared, run sheet, results, what they say, a simpler model, a run set aside, the report."""
    page = fresh_page
    name_response(page, "Strength", "MPa", goal="maximize")
    set_factors(page, FACTORS)
    candidates_done(page)
    assert page.locator("#candidates tbody tr").count() > 3, "the designs that can learn the brief are compared"
    generate(page, design_type="box_behnken")
    assert page.get_attribute("#workspace", "data-phase") == "run", "a run sheet made opens on its runs"
    assert headers(page) == ["Std order", "Run order", "T", "P", "F", "Strength"]
    blank = Path(download(page, "#csv").path()).read_bytes().decode().split("\r\n")
    assert blank[0] == "Std order,Run order,T,P,F,Strength"
    assert all(line.endswith(",") for line in blank[1:16]), "no run has its result yet"

    typed_from(page, quadratic)
    assert page.get_attribute("#workspace", "data-phase") == "run", "typing a result does not move the reader"
    reveal(page, "#analysis")
    said = findings(page)
    assert said[0].startswith("T matters most: raising T from 150 to 200 degC raises Strength by ")
    assert any(line.startswith("The effect of T depends on P") for line in said), "the interaction in the process"
    assert "15 completed runs" in page.text_content("#analysis")

    simplified(page)
    assert terms_fitted(page) == KEPT
    set_aside(page, 4)
    assert "14 completed runs" in page.text_content("#analysis")
    assert terms_fitted(page) == KEPT, "the model chosen is the one fitted without the run"
    assert page.text_content("#best .lead").startswith("These settings give the highest Strength the model predicts")

    name, html = download_report(page)
    assert name == "box_behnken-15-runs-report.html"
    report = opened(page, html)
    assert report.text_content("h1").startswith("Strength (MPa), maximise: Box-Behnken in T, P and F")
    assert "What the results say" in headings(report)
    words = report.text_content("body")
    assert "Run 4 is set aside" in words, "the report says which run the fit leaves out"
    assert findings(page)[0] in words, "and says what Learn says"


def test_a_study_is_kept_as_it_was_left_and_opens_the_same_from_a_link(open_context, context, fresh_page, site):
    """The results, the model chosen and the run set aside come back on a later visit, and in another browser."""
    page = fresh_page
    typed = strength_study(page)
    reveal(page, "#analysis")
    simplified(page)
    set_aside(page, 4)
    learned = findings(page)
    saved = saved_studies(page, lambda s: bool(s) and bool(s[0].get("set_aside")))
    assert saved[0]["set_aside"], "the study is kept with the run set aside"

    again = come_back(context)
    assert again.text_content(BANNER).startswith("Picked up where you left off: Strength (MPa), maximise: ")
    assert again.get_attribute("#workspace", "data-phase") == "learn", "a study with results opens on its analysis"
    assert [float(v) for v in results(again)] == list(typed), "every result as it was typed"
    assert terms_fitted(again) == KEPT, "the terms chosen are kept with the study"
    assert again.text_content("#aside-runs").startswith("Run 4 is set aside: ")
    assert findings(again) == learned

    reveal(again, "#share")
    again.click("#share")
    again.wait_for_selector("#share-out", state="visible")
    elsewhere = open_context().new_page()
    elsewhere.goto(again.input_value("#share-url"))
    elsewhere.wait_for_selector(BANNER, state="visible")
    elsewhere.wait_for_function("window.__pi.count('api_catalogue') === 1 && window.__pi.idle()")
    settled(elsewhere)
    assert elsewhere.url == site, "the address loses the link once the study is kept there"
    assert [float(v) for v in results(elsewhere)] == list(typed)
    assert findings(elsewhere) == learned, "another browser learns the same from the link"
    assert elsewhere.text_content("#aside-runs").startswith("Run 4 is set aside: ")


# --------------------------------------------------------------------------- the reader's own table
#: Eight corners in three factors, each made twice, and a yield that follows the pressure most.
TWO_LEVELS = "Temp (degC),Pressure (bar),Speed (rpm),Yield (%)\n" + "".join(
    f"{t},{p},{s},{60 + 0.04 * (t - 175) - 3 * (p - 2) + 0.005 * (s - 300) + 0.1 * ((i * 7) % 5)}\n"
    for i, (t, p, s) in enumerate([(t, p, s) for t in (150, 200) for p in (1, 3) for s in (200, 400)] * 2)
)


def test_a_table_of_the_reader_s_own_is_analysed_and_grows_by_the_runs_it_lacks(fresh_page):
    """A pasted table opens as a study, Learn says what it shows and what it lacks, and the runs added wait in Run."""
    page = fresh_page
    paste(page, TWO_LEVELS)
    page.fill("#import-title", "Line 3, March")
    page.click("#import-open")
    page.wait_for_function("document.querySelector('#workspace').dataset.phase === 'learn'")
    settled(page)
    assert page.input_value("#design-type") == "given", "the runs are taken as they are"
    assert findings(page)[0].startswith("Pressure matters most: ")
    assert len(run_sheet(page)) == 16

    reveal(page, "#next")
    assert "center" in offered(page), "corners only, and no run between them"
    add(page, "center")
    assert page.get_attribute("#workspace", "data-phase") == "run", "runs still to make are shown where they are made"
    sheet = run_sheet(page)
    assert len(sheet) == 19
    assert [row[-1] for row in sheet[16:]] == ["", "", ""], "the runs added wait for their results"
    assert all(row[-1] for row in sheet[:16]), "the table's own results stay with their runs"
    assert status(page) == ("3 runs added at the end of the run sheet: enter their results in Run.", "ok")

    lines = Path(download(page, "#csv").path()).read_bytes().decode().split("\r\n")
    assert len(lines) == 21, "the headings, nineteen runs, and the last line's end"
    assert lines[17].split(",")[-4:-1] == ["175", "2", "300"], "a run at the center of the table's ranges"

    enter_results(page, {16: 60.2, 17: 60.4, 18: 60.1})
    settled(page)
    fit = calls(page, "api_analyze_runs")[-1]["payload"]
    assert len(fit["responses"]) == 19
    assert None not in fit["responses"]
    reveal(page, "#analysis")
    assert "19 completed runs" in page.text_content("#analysis")
    saved = saved_studies(
        page, lambda s: bool(s) and len(s[0]["design"]["rows"]) == 19 and None not in s[0]["responses"]
    )
    assert saved[0]["title"] == "Line 3, March", "the grown study is kept under the name the reader gave the table"


# --------------------------------------------------------------------------- a workbook, to the lab and back
def test_a_run_sheet_goes_to_the_lab_as_a_workbook_and_comes_back_with_its_results(blank_page, tmp_path):
    """The workbook downloaded holds the runs; filled in elsewhere and uploaded, it fills Run and Learn."""
    page = blank_page
    generate(page, design_type="ccd")
    taken = download(page, "#download")
    assert taken.suggested_filename.endswith("-runs.xlsx")
    book = Path(taken.path()).read_bytes()
    runs = pd.read_excel(io.BytesIO(book), sheet_name="Runs")
    assert len(runs) == len(run_sheet(page))
    assert runs["y"].isna().all(), "no results yet"

    filled = tmp_path / "from the lab.xlsx"
    filled.write_bytes(base64.b64decode(fill_workbook(base64.b64encode(book).decode(), quadratic)))
    upload(page, filled)
    page.wait_for_function("window.__pi.count('api_read_workbook') === 1")
    settled(page)
    assert page.get_attribute("#workspace", "data-phase") == "learn", "a workbook with results opens on its analysis"
    assert all(row[-1] for row in run_sheet(page))
    assert findings(page)[0].startswith("T matters most: ")
    assert page.locator("#design-table input.sim").count() == 0, "results from a workbook are the reader's own"

    back = pd.read_excel(download(page, "#download").path(), sheet_name="Runs")
    np.testing.assert_allclose(back["y"], pd.read_excel(filled, sheet_name="Runs")["y"])


# --------------------------------------------------------------------------- the reading level, along a study
def test_the_reading_level_changes_what_is_offered_and_never_what_was_learned(blank_page):
    """A study made at Beginner reads the same at Adventurous, which only adds to what the page offers."""
    page = blank_page
    set_level(page, 1)
    assert "d_optimal" not in listed_designs(page)
    generate(page, design_type="full_factorial", n_center_points=3)
    practise(page)
    reveal(page, "#analysis")
    said, table = findings(page), coefficient_rows(page)
    assert said, "a beginner is told what the results say"
    assert page.locator("#simplify").is_hidden(), "simplifying the model waits for a higher level"
    asked = page.evaluate("window.__pi.count('api_analyze_runs')")

    set_level(page, 3)
    assert "d_optimal" in listed_designs(page)
    assert page.locator("#simplify").is_visible()
    assert "cube" in more_charts(page), "a two-level design in three factors has its cube"
    assert findings(page) == said
    assert coefficient_rows(page) == table
    assert page.evaluate("window.__pi.count('api_analyze_runs')") == asked, "a level is no reason to fit again"

    set_level(page, 1)
    assert page.input_value("#design-type") == "full_factorial", "the study's design stays at every level"
    assert findings(page) == said


# --------------------------------------------------------------------------- stopping, and carrying on
def test_stopping_a_slow_analysis_keeps_every_result_and_the_next_change_is_analysed(blank_page):
    """Python ended in the middle of an analysis loses nothing typed, and answers the reader's next change."""
    page = blank_page
    generate(page, design_type="full_factorial", n_center_points=0)
    page.evaluate("window.__pi.hold('api_analyze_runs')")
    enter_results(page, {i: 50 + (i * 7) % 11 for i in range(8)})
    page.wait_for_selector("#stop:not([hidden])", timeout=10_000)
    page.click("#stop")
    page.wait_for_function("document.querySelector('#stop').hidden")
    assert [row[-1] for row in run_sheet(page)] == [str(50 + (i * 7) % 11) for i in range(8)]
    page.evaluate("window.__pi.release('api_analyze_runs')")  # the reply of the worker that was ended: none is heard

    enter_results(page, {0: 58})
    settled(page)
    reveal(page, "#analysis")
    assert "8 completed runs" in page.text_content("#analysis")
    assert calls(page, "api_analyze_runs")[-1]["payload"]["responses"][0] == 58
    assert page.is_hidden("#stop")
    saved = saved_studies(page, lambda s: bool(s) and s[0]["responses"][0] == 58)
    assert saved[0]["responses"] == [58, *[50 + (i * 7) % 11 for i in range(1, 8)]]


# --------------------------------------------------------------------------- several responses
def test_three_responses_are_analysed_balanced_reported_and_kept_together(context, fresh_page):
    """One run sheet, a result for each response in each run, each response's model; then their balance."""
    page = fresh_page
    three_responses(page)
    practise(page)
    reveal(page, "#response-picker")
    assert [name for name, _ in tabs(page)] == ["Yield", "Cost", "Purity"]
    first = findings(page)
    pick(page, "Cost")
    assert findings(page) != first, "each response says its own"
    assert calls(page, "api_analyze_runs")[-1]["payload"]["response"] == "Cost"

    reveal(page, "#balance")
    page.wait_for_selector("#balance-chart .js-plotly-plot")
    balance = balanced(page)
    assert [r["name"] for r in balance["responses"]] == ["Yield", "Cost", "Purity"]
    assert 0 < balance["composite"] <= 1

    _, html = download_report(page)
    report = opened(page, html)
    said = report.text_content("body")
    assert all(f"{name}" in said for name in ("Yield", "Cost", "Purity"))
    assert f"{balance['composite']:.2f}" in said, "the report holds the balance the page found"
    report.close()

    saved_studies(page, lambda s: bool(s) and len(s[0].get("more", {})) == 2)
    again = come_back(context)
    reveal(again, "#response-picker")
    assert [name for name, _ in tabs(again)] == ["Yield", "Cost", "Purity"]
    assert again.locator("#design-table input[data-response]").count() == 3 * 15, "every response's results came back"


# --------------------------------------------------------------------------- a mixture
def test_a_mixture_goes_from_its_components_to_its_best_blend(fresh_page):
    """Components in the brief, blends on the run sheet, the blending model in Learn, and the best blend sums to one."""
    page = fresh_page
    name_response(page, "Gloss", "", goal="maximize")
    set_factors(page, BLEND)
    generate(page, design_type="simplex_lattice")
    for row in run_sheet(page):
        assert sum(float(v) for v in row[2:5]) == pytest.approx(1.0, abs=1e-5), "every run is a whole blend"
    practise(page)
    reveal(page, "#analysis")
    assert "Scheffé" in page.text_content("#analysis") or "blend" in page.text_content("#analysis")
    best = last_reply(page, "api_best_settings")["result"]
    assert sum(s["value"] for s in best["settings"]) == pytest.approx(1.0, abs=1e-6)
    assert page.text_content("#best .lead").startswith("This blend gives the highest Gloss the model predicts")
    name, html = download_report(page)
    assert name.endswith("-report.html")
    assert "Gloss" in opened(page, html).text_content("h1")
