"""The first paint: the worked example the build computed, drawn while Python starts.

A reader sees a whole campaign at once (the design, its simulated results, the analysis
beside the true process, and the comparison of every design) before Pyodide has loaded,
and can press nothing that needs Python until it has.
"""

from __future__ import annotations

import pytest

from tests.web.e2e._page import factor_table, first_paint_done, run_sheet, status, type_results

#: Controls that need Python, enabled once it has started.
NEEDS_PYTHON = ("#generate", "#simulate", "#reroll", "#compare", "#upload")


def open_while_python_starts(context, site):
    """Open the page with the catalogue held back, as Python still loading."""
    context.add_init_script("window.__pi.hold('api_catalogue');")
    page = context.new_page()
    page.goto(site)
    page.wait_for_function("document.querySelectorAll('#compare-charts .js-plotly-plot').length > 1")
    return page


def test_the_worked_example_is_drawn_before_python_has_started(context, site, first_paint):
    page = open_while_python_starts(context, site)
    assert page.is_visible("#demo-banner")
    assert page.text_content("#demo-version") == f"process-improve {first_paint['version']}"
    assert status(page) == ("Showing the worked example while Python starts...", "busy")
    assert len(run_sheet(page)) == len(first_paint["design"]["rows"])
    assert page.text_content("#truth-equation") == first_paint["simulation"]["equation"]
    assert page.text_content("#analysis .equation") == first_paint["analysis"]["equation"]
    assert page.locator("#charts .js-plotly-plot").count() == 4, "coefficients, both surfaces, residuals"
    for control in NEEDS_PYTHON:
        assert page.is_disabled(control), control


def test_python_starting_enables_what_needs_it(context, site, first_paint):
    page = open_while_python_starts(context, site)
    page.evaluate("window.__pi.release('api_catalogue')")
    first_paint_done(page, first_paint)
    assert status(page) == ("Ready: process-improve (native)", "ok")
    for control in NEEDS_PYTHON:
        assert page.is_enabled(control), control


def test_while_python_starts_the_status_line_lights_up_the_mark_s_runs(context, site, first_paint):
    """The loader is the app's mark: its runs light up in a random run order until Python answers."""
    page = open_while_python_starts(context, site)
    assert page.is_visible("#status .status__icon svg")
    assert page.locator("#status .status__icon circle").count() == 7, "the mark's six corners and its centre run"
    page.evaluate("window.__pi.release('api_catalogue')")
    first_paint_done(page, first_paint)
    assert page.is_hidden("#status .status__icon svg"), "a tick once Python is ready"


@pytest.mark.slow
def test_the_status_line_fades_after_a_success_and_a_quick_analysis_does_not_bring_it_back(page):
    """Its room goes back to the page five seconds after Python is ready; an error brings it back."""
    assert page.is_visible("#status")
    page.wait_for_selector("#status", state="hidden", timeout=8_000)
    assert status(page)[1] == "ok"
    type_results(page, {0: 50})
    assert status(page) == ("Analysis complete.", "ok")
    assert page.is_hidden("#status"), "a quick analysis does not make the page jump"
    page.select_option("#model", "quadratic")
    type_results(page, dict.fromkeys(range(5, 15), ""))
    page.wait_for_function("document.querySelector('#status').dataset.state === 'error'")
    assert page.is_visible("#status"), "an error is shown at once"


def test_the_form_holds_the_worked_example_s_settings(page, first_paint):
    inputs = first_paint["inputs"]
    assert factor_table(page) == [
        {key: str(f[key]) for key in ("name", "low", "high", "units")} for f in inputs["design"]["factors"]
    ]
    assert page.input_value("#design-type") == inputs["design"]["design_type"]
    assert page.input_value("#n_center_points") == str(inputs["design"]["n_center_points"])
    assert page.input_value("#scenario") == inputs["scenario"]
    assert float(page.input_value("#sigma")) == inputs["sigma"]


def test_the_results_are_marked_as_simulated(page, first_paint):
    simulated = page.locator("#design-table input.sim")
    assert simulated.count() == len(first_paint["simulation"]["responses"])
    assert "simulated from" in page.text_content("#run-source")


def test_without_a_built_example_the_page_starts_empty(blank_page):
    """A plain preview of web/ has no demo.json: the steps wait for a design."""
    assert blank_page.is_visible("#run-empty")
    assert blank_page.is_visible("#analyse-empty")
    assert blank_page.is_hidden("#demo-banner")
    assert status(blank_page) == ("Ready: process-improve (native)", "ok")
