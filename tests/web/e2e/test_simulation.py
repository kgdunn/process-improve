"""Practising with simulated results in Run: a known process over the reader's factors, plus noise.

One button simulates, and a press always changes the results: the first simulates them, and one
more draws new noise for the same process, with the next seed. The true process sits behind a
reveal, beside the button. A change of process, noise or design simulates again while the reader
practises, so the truth and the analysis always describe the runs on the page. Practice results
are on the reader's scale: centred on the response's typical value, with its run-to-run noise.
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
    status,
    type_results,
)


def simulated_results(page) -> list[str]:
    return page.locator("#design-table input.sim").evaluate_all("(inputs) => inputs.map((i) => i.value)")


def test_simulating_fills_every_run_and_keeps_the_true_process_to_hand(blank_page):
    page = blank_page
    generate(page, design_type="full_factorial")
    assert page.is_hidden("#truth-toggle")
    assert page.get_attribute("#tab-real", "aria-selected") == "true", "results are entered unless the reader practises"
    assert calls(page, "api_simulate") == [], "nothing is simulated until asked for"
    practise(page)
    assert len(simulated_results(page)) == len(run_sheet(page)) == 11
    assert page.is_visible("#truth-toggle")
    assert page.is_hidden("#truth"), "behind its button until the reader asks"
    assert page.text_content("#truth-equation").startswith("y = 62 + 4·T - 3.1·P + 1.75·F + 1.5·T·P")
    assert page.text_content("#run-source") == (
        '11 of 11 results simulated from "Reactor yield" (noise 1, seed 1), in italics: '
        "type over any to enter your own."
    )
    assert "TrueProcess(" in page.text_content("#py-run code")


def test_results_simulated_for_a_study_left_meanwhile_stay_out_of_the_new_one(blank_page):
    """A new study, started while the results are simulated, drops them: it has no run sheet, and no busy note."""
    page = blank_page
    generate(page, design_type="full_factorial")
    page.evaluate("window.__pi.hold('api_simulate')")
    reveal(page, "#panel-simulate")
    page.click("#tab-simulate")
    page.click("#simulate")
    assert status(page) == ("Simulating the results...", "busy")
    page.click("#new-study")
    page.evaluate("window.__pi.release('api_simulate')")
    settled(page)
    assert page.get_attribute("#workspace", "data-phase") == "plan"
    assert page.is_hidden("#run-out")
    assert page.is_hidden("#status")


def test_simulating_again_draws_new_noise_for_the_same_process_with_the_next_seed(page, first_paint):
    reveal(page, "#simulate")
    assert page.text_content("#simulate") == "Simulate again, with new noise", "the button says what a press does"
    page.click("#simulate")
    settled(page)
    assert calls(page, "api_simulate")[-1]["payload"]["seed"] == first_paint["inputs"]["seed"] + 1
    assert page.text_content("#truth-equation") == first_paint["simulation"]["equation"]
    assert [float(v) for v in simulated_results(page)] != first_paint["simulation"]["responses"]


def test_every_press_of_the_button_changes_the_results(blank_page):
    """A second press with the seed as it was gave the same results, and looked as if it did nothing."""
    page = blank_page
    generate(page, design_type="full_factorial")
    reveal(page, "#panel-simulate")
    page.click("#tab-simulate")
    assert page.text_content("#simulate") == "Simulate the results"
    seen = []
    for _ in range(3):
        page.click("#simulate")
        settled(page)
        seen.append(simulated_results(page))
        assert page.text_content("#simulate") == "Simulate again, with new noise"
    assert len({tuple(results) for results in seen}) == 3, "new results at each press"
    assert [call["payload"]["seed"] for call in calls(page, "api_simulate")] == [1, 2, 3]
    assert page.text_content("#run-source").startswith(
        '11 of 11 results simulated from "Reactor yield" (noise 1, seed 3)'
    )
    assert page.locator("#panel-simulate button").all_text_contents() == [
        "Simulate again, with new noise",
        "Show the true process",
    ], "one button simulates: there is no second one for new noise"


def test_the_true_process_is_shown_and_hidden_by_the_button_beside_the_one_that_simulates(page):
    reveal(page, "#truth-toggle")
    toggle = page.locator("#truth-toggle")
    assert (toggle.text_content(), toggle.get_attribute("aria-expanded")) == ("Show the true process", "false")
    assert toggle.get_attribute("aria-controls") == "truth"
    toggle.click()
    assert page.is_visible("#truth-equation")
    assert (toggle.text_content(), toggle.get_attribute("aria-expanded")) == ("Hide the true process", "true")
    page.click("#simulate")
    settled(page)
    assert page.is_visible("#truth-equation"), "it stays open as the results are simulated again"
    toggle.click()
    assert page.is_hidden("#truth")
    assert toggle.text_content() == "Show the true process"


def test_the_process_and_its_noise_are_one_row_and_so_are_the_two_buttons(page):
    """The noise's label took two lines, which put its box below the process's; the buttons were on two rows."""
    reveal(page, "#panel-simulate")
    page.set_viewport_size({"width": 820, "height": 1100})

    def bottom(selector: str) -> float:
        box = page.locator(selector).bounding_box()
        return box["y"] + box["height"]

    def middle(selector: str) -> float:
        box = page.locator(selector).bounding_box()
        return box["y"] + box["height"] / 2

    assert abs(bottom("#scenario") - bottom("#sigma")) <= 1, "the two boxes at one height"
    labels = page.locator("#panel-simulate .row > label > span:first-child")
    tops = labels.evaluate_all("(spans) => spans.map((span) => span.getBoundingClientRect().top)")
    assert abs(tops[0] - tops[1]) <= 1, "and their labels on one line"
    assert labels.nth(1).evaluate("(span) => span.getClientRects().length") == 1, "the noise's label is not wrapped"
    assert abs(middle("#simulate") - middle("#truth-toggle")) <= 1, "the reveal beside the button"


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
    first = page.locator("#factors tbody tr").first
    first.locator("[name=name]").fill("Temp")
    first.locator("[name=name]").press("Tab")
    settled(page)
    first.locator("[name=high]").fill("210")
    generate(page)
    fns = [c["fn"] for c in calls(page)]
    # The next step is asked for once the best settings are known, before the charts are drawn; the what-if
    # panel's fits come last, so it opens at the best settings.
    assert fns[-7:] == [
        "api_make_design",
        "api_simulate",
        "api_analyze_runs",
        "api_best_settings",
        "api_next_runs",
        "api_plots",
        "api_predictors",
    ]
    assert page.text_content("#truth-equation").startswith("Yield = 62 + 4·Temp")
    assert page.is_hidden("#design-stale")


def test_the_true_process_goes_once_no_simulated_result_is_left(page):
    reveal(page, "#truth-toggle")
    page.click("#truth-toggle")
    assert page.is_visible("#truth")
    type_results(page, {run: 60 + run for run in range(14)})
    reveal(page, "#truth-toggle")
    assert page.is_visible("#truth"), "one simulated result is left"
    type_results(page, {14: 74})
    assert page.locator("#design-table input.sim").count() == 0
    reveal(page, "#simulate")
    assert page.is_hidden("#truth-toggle")
    assert page.is_hidden("#truth"), "shut with its button, so it is shut when results are simulated again"
    assert page.text_content("#simulate") == "Simulate the results", "nothing simulated is left to draw new noise for"


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
