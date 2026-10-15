"""The first paint: a new study at its response, or the worked example the build computed, drawn while Python starts.

A first visit opens a new study in Design, its response asked for first. ``?example=worked``, or
Explore a worked example, shows a whole study at once (its brief, the designs that can learn it,
its run sheet with simulated results, and the analysis beside the true process) before Pyodide
has loaded, and nothing that needs Python can be pressed until it has.
"""

from __future__ import annotations

import pytest

from tests.web.e2e._page import (
    EXAMPLE_CHARTS,
    calls,
    enter_results,
    factor_table,
    first_paint_done,
    reveal,
    run_sheet,
    settled,
    status,
    type_results,
)

#: Controls that need Python, enabled once it has started.
NEEDS_PYTHON = ("#generate", "#simulate", "#reroll", "#upload")


def open_while_python_starts(context, site):
    """Open the worked example with the catalogue held back, as Python still loading."""
    context.add_init_script("window.__pi.hold('api_catalogue');")
    page = context.new_page()
    page.goto(f"{site}?example=worked")
    page.wait_for_function(f"document.querySelectorAll('#charts .js-plotly-plot').length === {EXAMPLE_CHARTS}")
    return page


def phase(page) -> str:
    return page.get_attribute("#workspace", "data-phase")


def test_a_first_visit_starts_a_new_study_at_its_response(fresh_page):
    page = fresh_page
    assert phase(page) == "plan"
    assert page.evaluate("document.activeElement.id") == "response", "the first question is what the study measures"
    assert page.text_content("#study-title") == "A new study"
    assert factor_table(page) == [{"name": "", "low": "", "high": "", "units": ""}] * 2, "two factors to type in"
    assert page.is_visible("#plan-empty"), "the designs wait for the brief"
    assert page.is_hidden("#design-card")
    assert page.is_hidden("#demo-banner")
    assert page.is_hidden("#factor-problems"), "a blank row is no factor yet, and needs nothing"
    assert status(page) == ("Ready: process-improve (native)", "ok")
    assert [c["fn"] for c in calls(page)] == ["api_catalogue"]


def test_the_worked_example_is_drawn_before_python_has_started(context, site, first_paint):
    page = open_while_python_starts(context, site)
    assert page.is_visible("#demo-banner")
    assert page.text_content("#demo-version") == f"process-improve {first_paint['version']}"
    assert status(page) == ("Showing the worked example while Python starts...", "busy")
    assert phase(page) == "learn", "the worked example opens on what it learned"
    assert len(run_sheet(page)) == len(first_paint["design"]["rows"])
    assert page.text_content("#truth-equation") == first_paint["simulation"]["equation"]
    assert page.text_content("#analysis .equation") == first_paint["analysis"]["equation"]
    assert page.locator("#charts .js-plotly-plot").count() == EXAMPLE_CHARTS, "coefficients, both surfaces, residuals"
    for control in NEEDS_PYTHON:
        assert page.is_disabled(control), control
    assert page.is_enabled("#compare"), "the comparison the build made needs no Python to read"


def test_python_starting_enables_what_needs_it(context, site):
    page = open_while_python_starts(context, site)
    page.evaluate("window.__pi.release('api_catalogue')")
    first_paint_done(page)
    assert status(page) == ("Ready: process-improve (native)", "ok")
    for control in NEEDS_PYTHON:
        assert page.is_enabled(control), control
    assert [c["fn"] for c in calls(page)] == ["api_catalogue"], "the worked example's designs are not made again"


def test_while_python_starts_the_status_line_lights_up_the_mark_s_runs(context, site):
    """The loader is the app's mark: its runs light up in a random run order until Python answers."""
    page = open_while_python_starts(context, site)
    assert page.is_visible("#status .status__icon svg")
    assert page.locator("#status .status__icon circle").count() == 7, "the mark's six corners and its centre run"
    page.evaluate("window.__pi.release('api_catalogue')")
    first_paint_done(page)
    assert page.is_hidden("#status .status__icon svg"), "a tick once Python is ready"


@pytest.mark.slow
def test_the_status_line_fades_after_a_success_and_comes_back_for_a_wait_or_an_error(page):
    """Its room goes back to the page five seconds after Python is ready, until there is news worth the room."""
    assert page.is_visible("#status")
    page.wait_for_selector("#status", state="hidden", timeout=8_000)
    assert status(page)[1] == "ok", "faded, its message kept"
    type_results(page, dict.fromkeys(range(15), ""))
    assert status(page) == ("No results to analyse yet.", "ok")
    assert page.is_hidden("#status"), "a success that the page itself shows brings nothing back"
    page.evaluate("window.__pi.hold('api_analyze_runs')")
    enter_results(page, {0: 50})
    page.wait_for_selector("#status", state="visible")
    assert status(page) == ("Analysing...", "busy"), "a wait that lasts is shown"
    page.evaluate("window.__pi.release('api_analyze_runs')")
    page.wait_for_function("document.querySelector('#status').dataset.state === 'error'")
    assert page.is_visible("#status"), "one result cannot fit the model: the error stays in view"


def test_the_brief_holds_the_worked_example_s_response_factors_and_aim(page, first_paint):
    inputs = first_paint["inputs"]
    described = inputs["design"]["response_info"]
    assert page.input_value("#response") == described["label"]
    assert page.input_value("#response-units") == described["units"]
    assert page.is_checked(f'input[name="goal"][value="{described["goal"]}"]')
    assert float(page.input_value("#response-typical")) == described["typical"]
    assert float(page.input_value("#response-noise")) == described["noise"]
    assert float(page.input_value("#response-change")) == described["smallest_change"]
    assert page.text_content("#study-title") == "Yield (%), maximise"
    assert factor_table(page) == [
        {key: str(f[key]) for key in ("name", "low", "high", "units")} for f in inputs["design"]["factors"]
    ]
    assert page.is_checked(f'input[name="aim"][value="{inputs["design"]["aim"]}"]')
    assert page.input_value("#design-type") == inputs["design"]["design_type"]
    assert page.input_value("#n_center_points") == str(inputs["design"]["n_center_points"])
    assert page.input_value("#scenario") == inputs["scenario"]
    assert float(page.input_value("#sigma")) == inputs["sigma"]


def test_the_designs_the_build_compared_are_listed_in_plan(page, first_paint):
    reveal(page, "#candidates")
    compared = first_paint["comparison"]["result"]
    fitting = [label for label in compared["designs"] if compared["values"][label]["fits"]]
    rows = page.locator("#candidates tbody tr")
    assert rows.count() == len(fitting)
    chosen = page.locator('#candidates tbody tr[aria-selected="true"]')
    assert chosen.get_attribute("data-design") == "box_behnken", "the design the worked example uses"


def test_the_results_are_marked_as_simulated(page, first_paint):
    simulated = page.locator("#design-table input.sim")
    assert simulated.count() == len(first_paint["simulation"]["responses"])
    assert "simulated from" in page.text_content("#run-source")
    reveal(page, "#run-source")
    assert page.get_attribute("#tab-simulate", "aria-selected") == "true", "the worked example is practice"


def test_explore_a_worked_example_shows_it_at_once(fresh_page, first_paint):
    page = fresh_page
    page.click("#show-example")
    page.wait_for_function(f"document.querySelectorAll('#charts .js-plotly-plot').length === {EXAMPLE_CHARTS}")
    settled(page)
    assert page.is_visible("#demo-banner")
    assert phase(page) == "learn"
    assert page.url.endswith("?example=worked"), "a reload shows it again"
    assert [c["fn"] for c in calls(page)] == ["api_catalogue"], "computed when the page was built"
    assert page.text_content("#analysis .equation") == first_paint["analysis"]["equation"]


def test_without_a_built_example_the_worked_example_runs_live(context, site, console, first_paint):
    """A plain preview of web/ has no demo.json: Explore a worked example computes it here instead."""
    console.allow(r"404")  # the browser reports the missing demo.json
    context.route(f"{site}demo.json", lambda route: route.fulfill(status=404))
    page = context.new_page()
    page.goto(site)
    page.wait_for_function("window.__pi.count('api_catalogue') === 1 && window.__pi.idle()")
    page.click("#show-example")
    page.wait_for_function(f"document.querySelectorAll('#charts .js-plotly-plot').length === {EXAMPLE_CHARTS}")
    settled(page)
    fns = [c["fn"] for c in calls(page)]
    assert fns[-4:] == ["api_make_design", "api_simulate", "api_analyze_runs", "api_plots"]
    assert page.text_content("#analysis .equation") == first_paint["analysis"]["equation"]
    assert phase(page) == "learn"
    assert page.is_hidden("#library"), "the worked example run here is not saved"
