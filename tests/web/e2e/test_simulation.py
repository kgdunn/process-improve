"""Practising with simulated results in Run: a known process over the reader's factors, plus noise.

The true process sits behind a reveal; new noise draws the same process with the next seed;
a change of process, noise or design simulates again while the reader practises, so the truth
and the analysis always describe the runs on the page. Practice results are on the reader's
scale: centred on the response's typical value, with its run-to-run noise.
"""

from __future__ import annotations

from tests.web._support import FACTORS
from tests.web.e2e._page import (
    calls,
    choose,
    generate,
    name_response,
    practise,
    reveal,
    run_sheet,
    set_factors,
    settled,
    type_results,
)


def simulated_results(page) -> list[str]:
    return page.locator("#design-table input.sim").evaluate_all("(inputs) => inputs.map((i) => i.value)")


def test_simulating_fills_every_run_and_keeps_the_true_process_to_hand(blank_page):
    page = blank_page
    generate(page, design_type="full_factorial")
    assert page.is_hidden("#truth")
    assert page.get_attribute("#tab-real", "aria-selected") == "true", "results are entered unless the reader practises"
    assert calls(page, "api_simulate") == [], "nothing is simulated until asked for"
    practise(page)
    assert len(simulated_results(page)) == len(run_sheet(page)) == 11
    assert page.is_visible("#truth")
    assert page.text_content("#truth-equation").startswith("y = 62 + 4·T - 3.1·P + 1.75·F + 1.5·T·P")
    assert page.text_content("#run-source") == (
        '11 of 11 results simulated from "Reactor yield" (noise 1, seed 1), in italics: '
        "type over any to enter your own."
    )
    assert "TrueProcess(" in page.text_content("#py-run code")


def test_new_noise_draws_the_same_process_with_the_next_seed(page, first_paint):
    reveal(page, "#reroll")
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
    reveal(page, "#sigma")
    page.fill("#sigma", "2.5")
    page.locator("#sigma").blur()
    settled(page)
    assert calls(page, "api_simulate")[-1]["payload"]["sigma"] == 2.5


def test_the_true_process_names_the_factors_it_leaves_out(page):
    set_factors(page, [*FACTORS, {"name": "Salt", "low": -1, "high": 1}])
    generate(page, design_type="dsd")
    assert page.text_content("#truth-inactive") == "Salt does not affect Yield in this process."


def test_a_new_design_is_simulated_from_the_same_process(page):
    """Results simulated for the old runs are simulated again for the new ones, over the factors as they are now."""
    page.locator("#factors tbody tr").first.locator("[name=name]").fill("Temp")
    generate(page)
    fns = [c["fn"] for c in calls(page)]
    assert fns[-4:] == ["api_make_design", "api_simulate", "api_analyze_runs", "api_plots"]
    assert page.text_content("#truth-equation").startswith("Yield = 62 + 4·Temp")
    assert page.is_hidden("#design-stale")


def test_the_true_process_goes_once_no_simulated_result_is_left(page):
    reveal(page, "#truth")
    assert page.is_visible("#truth")
    type_results(page, {run: 60 + run for run in range(14)})
    assert page.is_visible("#truth"), "one simulated result is left"
    type_results(page, {14: 74})
    assert page.locator("#design-table input.sim").count() == 0
    assert page.is_hidden("#truth")


def test_practice_results_are_on_the_reader_s_scale(blank_page):
    """Centred on the typical value, with the run-to-run noise: the process a reader with no numbers gets, scaled."""
    page = blank_page
    generate(page, design_type="box_behnken")
    practise(page)
    plain = [float(v) for v in simulated_results(page)]
    name_response(page, "Purity", "%", typical=97, noise=0.4)
    page.wait_for_function("window.__pi.calls.filter((c) => c.fn === 'api_simulate').at(-1).payload.sigma === 0.4")
    settled(page)
    assert calls(page, "api_simulate")[-1]["payload"]["sigma"] == 0.4, "the practice noise follows the response's"
    scaled = [float(v) for v in simulated_results(page)]
    assert scaled != plain
    assert abs(sum(scaled) / len(scaled) - 97) < 2, "centred near the typical value"
    assert page.text_content("#truth-equation").startswith("Purity = 97 + ")
