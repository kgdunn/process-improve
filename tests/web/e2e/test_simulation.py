"""Simulating the results in step 2: a known process over the reader's factors, plus noise.

The true process sits behind a reveal; new noise draws the same process with the next seed;
a change of process, noise or design simulates again, so the truth and the analysis always
describe the runs on the page.
"""

from __future__ import annotations

from tests.web._support import FACTORS
from tests.web.e2e._page import calls, choose, generate, run_sheet, set_factors, settled


def simulated_results(page) -> list[str]:
    return page.locator("#design-table input.sim").evaluate_all("(inputs) => inputs.map((i) => i.value)")


def test_simulating_fills_every_run_and_keeps_the_true_process_to_hand(blank_page):
    page = blank_page
    generate(page, design_type="full_factorial")
    assert page.is_hidden("#truth")
    page.click("#simulate")
    settled(page)
    assert len(simulated_results(page)) == len(run_sheet(page)) == 11
    assert page.is_visible("#truth")
    assert page.text_content("#truth-equation").startswith("y = 62 + 4·T - 3.1·P + 1.75·F + 1.5·T·P")
    assert 'simulated from "Reactor yield", with noise of standard deviation 1 (seed 1)' in page.text_content(
        "#run-source"
    )
    assert "TrueProcess(" in page.text_content("#py-run code")


def test_new_noise_draws_the_same_process_with_the_next_seed(page, first_paint):
    page.click("#reroll")
    settled(page)
    assert calls(page, "api_simulate")[-1]["payload"]["seed"] == first_paint["inputs"]["seed"] + 1
    assert page.text_content("#truth-equation") == first_paint["simulation"]["equation"]
    assert [float(v) for v in simulated_results(page)] != first_paint["simulation"]["responses"]


def test_another_process_or_noise_simulates_again(page, bootstrap):
    choose(page, scenario="interaction")
    settled(page)
    assert calls(page, "api_simulate")[-1]["payload"]["scenario"] == "interaction"
    assert page.text_content("#scenario-hint") == bootstrap.SCENARIOS["interaction"]["hint"]
    page.fill("#sigma", "2.5")
    page.locator("#sigma").blur()
    settled(page)
    assert calls(page, "api_simulate")[-1]["payload"]["sigma"] == 2.5


def test_the_true_process_names_the_factors_it_leaves_out(page):
    set_factors(page, [*FACTORS, {"name": "Salt", "low": -1, "high": 1}])
    generate(page, design_type="dsd")
    assert page.text_content("#truth-inactive") == "Salt does not affect y in this process."


def test_a_new_design_is_simulated_from_the_same_process(page):
    """Results simulated for the old runs are simulated again for the new ones, over the factors as they are now."""
    page.locator("#factors tbody tr").first.locator("[name=name]").fill("Temp")
    generate(page)
    fns = [c["fn"] for c in calls(page)]
    assert fns[-4:] == ["api_make_design", "api_simulate", "api_analyze_runs", "api_plots"]
    assert page.text_content("#truth-equation").startswith("y = 62 + 4·Temp")
    assert page.is_hidden("#design-stale")
