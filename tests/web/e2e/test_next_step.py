"""The next step, in Learn: what the results call for, a new study further along, and runs added to this one.

Once every run has its result, Learn says what the results call for next, with one button that does it:
runs added to the study, confirmation runs at the optimum, or a new study with its ranges moved or
widened. When the best settings sit at the ends of the ranges and the model predicts better past them,
the new study's ranges move along the steepest ascent, and it starts in Design. Learn also offers the
ways of adding runs that suit the design: a fold-over for a fraction whose main effects are aliased with
two-factor interactions, center runs for a two-level design with none, and others. The runs added go at
the end of the run sheet; the runs before keep their results.
"""

from __future__ import annotations

from tests.web._support import FACTORS
from tests.web.e2e._page import (
    calls,
    choose,
    factor_table,
    generate,
    headers,
    name_response,
    practise,
    reveal,
    run_sheet,
    saved_studies,
    set_factors,
    settled,
    status,
    type_results,
)


def num(x: float) -> str:
    """Write a number as the page's tables do: four significant figures, no trailing zeros."""
    return f"{float(f'{x:.4g}'):g}"


def next_rows(page) -> list[list[str]]:
    """Read the table of the next study's ranges as the reader sees it."""
    return page.evaluate(
        "() => [...document.querySelectorAll('#next-out table tbody tr')]"
        ".map((tr) => [...tr.cells].map((c) => c.textContent))"
    )


def offered(page) -> list[str]:
    """Return the kinds of the ways of adding runs Learn offers, in its order."""
    return page.evaluate("() => [...document.querySelectorAll('#next-out .offer')].map((o) => o.dataset.kind)")


def add(page, kind: str) -> None:
    """Press the button of one way of adding runs, as a reader would, and wait for the page to settle."""
    reveal(page, "#next")
    page.click(f'#next-out .offer[data-kind="{kind}"] button.button')
    settled(page)


# --------------------------------------------------------------------------- a new study further along
def called_for(page) -> str | None:
    """Return the step the results call for, as the next step's first part marks it, or None."""
    reveal(page, "#next")
    return page.evaluate("() => document.querySelector('#next-out .next__called')?.dataset.action ?? null")


def practise_on(page, scenario: str) -> None:
    """Simulate the run sheet's results from one of the page's processes, as a reader practising would."""
    reveal(page, "#panel-simulate")
    page.click("#tab-simulate")
    page.select_option("#scenario", scenario)
    page.click("#simulate")
    settled(page)


def do_what_is_called_for(page) -> None:
    """Press the button of what the results call for, as a reader would, and wait for the page to settle."""
    reveal(page, "#next")
    page.click("#next-out .next__called button.button")
    settled(page)


def test_the_worked_example_s_results_call_for_a_new_study_further_along(page, first_paint):
    assert called_for(page) == "move_region"
    assert page.text_content("#h-next") == "The next step"
    assert page.text_content("#next-out .next__called h4") == "What the results call for"
    assert headers(page, "#next-out table") == ["Factor", "Range now", "Next study"]
    moved = {r["name"]: r for r in first_paint["next"]["recommendation"]["ranges"]}
    assert next_rows(page) == [
        ["T", "150 to 200 degC", "175 to 225 degC"],
        ["P", "1 to 3 bar", f"{num(moved['P']['low'])} to {num(moved['P']['high'])} bar"],
        ["F", "10 to 20 L/min", "10 to 20 L/min"],
    ]
    assert page.is_hidden("#next-out .next__why p"), "why, and what then, are folded until opened"
    assert offered(page) == [], "its Box-Behnken design fits the full quadratic with runs to spare"


def test_the_next_step_lies_inside_its_card(page):
    reveal(page, "#next")
    card = page.locator("#next").bounding_box()
    parts = page.locator("#next-out > div").all()
    assert parts, "the new study further along"
    for part in parts:
        assert part.bounding_box()["width"] > card["width"] / 2, "each part spans the card, not a numbered badge"
    for inside in (page.locator("#next-out table"), page.locator("#next-study")):
        box = inside.bounding_box()
        assert card["x"] <= box["x"]
        assert box["x"] + box["width"] <= card["x"] + card["width"]
        assert card["y"] <= box["y"]
        assert box["y"] + box["height"] <= card["y"] + card["height"], "below the card, as on a tablet before"


def test_the_new_study_opens_in_design_with_the_moved_ranges(page):
    reveal(page, "#next")
    page.click("#next-study")
    settled(page)
    assert page.get_attribute("#workspace", "data-phase") == "plan"
    table = factor_table(page)
    assert [f["name"] for f in table] == ["T", "P", "F"]
    assert (table[0]["low"], table[0]["high"]) == ("175", "225")
    assert (table[2]["low"], table[2]["high"]) == ("10", "20")
    assert page.input_value("#response") == "Yield", "the brief is this study's"
    assert page.input_value("#design-type") == "box_behnken"
    assert page.evaluate("document.querySelector('#run-out').hidden"), "its runs are not made yet"
    assert status(page) == ("A new study, its ranges moved: check they are safe to run, then make its run sheet.", "ok")


# --------------------------------------------------------------------------- runs added to this study
def test_a_fraction_is_folded_over_with_its_results_kept(blank_page):
    page = blank_page
    generate(page, design_type="fractional_factorial", resolution=3, n_center_points=3)
    practise(page)
    before = [row[-1] for row in run_sheet(page)]
    assert len(before) == 7
    reveal(page, "#next")
    assert offered(page)[0] == "foldover"
    add(page, "foldover")
    sheet = run_sheet(page)
    assert len(sheet) == 11
    assert headers(page)[:3] == ["Std order", "Run order", "Block"]
    assert [row[-1] for row in sheet[:7]] == before, "the results before stay with their runs"
    assert all(row[-1] != "" for row in sheet[7:]), "a practice study's runs added are simulated"
    assert [row[2] for row in sheet] == ["1"] * 7 + ["2"] * 4
    assert status(page) == ("4 runs added at the end of the run sheet, their results simulated.", "ok")
    reveal(page, "#design-summary")
    summary = page.text_content("#design-summary")
    assert "Runs 8 to 11 fold the design over" in summary
    assert "11 runs, 3 of them at the center of the ranges, in random run order." in summary
    assert "No main effect or two-factor interaction is fully aliased with another." in summary  # the full 2^3 now
    assert page.is_hidden("#design-stale"), "the plan is the grown study's runs, so the run sheet is up to date"
    assert page.input_value("#design-type") == "given"
    fit = calls(page, "api_analyze_runs")[-1]["payload"]
    assert len(fit["responses"]) == 11
    assert "foldover" not in offered(page), "its main effects are clear now"


def test_runs_added_to_a_real_study_wait_in_run_for_their_results(blank_page):
    page = blank_page
    generate(page, design_type="full_factorial", n_center_points=0)
    type_results(page, {i: 50 + i for i in range(8)})
    reveal(page, "#next")
    assert "center" in offered(page)
    add(page, "center")
    assert page.get_attribute("#workspace", "data-phase") == "run"
    sheet = run_sheet(page)
    assert len(sheet) == 11
    assert [row[-1] for row in sheet[8:]] == ["", "", ""]
    assert [row[2:5] for row in sheet[8:]] == [["175", "2", "15"]] * 3
    assert "Block" not in headers(page), "center runs are no block of their own"
    assert status(page) == ("3 runs added at the end of the run sheet: enter their results in Run.", "ok")
    saved = saved_studies(page, lambda s: bool(s) and len(s[0]["design"]["rows"]) == 11)
    assert len(saved[0]["design"]["rows"]) == 11, "the grown study is saved"
    assert saved[0]["responses"][:8] == [50 + i for i in range(8)]


def test_a_refusal_to_add_runs_says_why(blank_page):
    page = blank_page
    generate(page, design_type="full_factorial", n_center_points=0)
    type_results(page, {i: 50 + i for i in range(8)})
    reveal(page, "#next")
    refused = {"ok": False, "error": "Center runs would fall outside your limits on the settings.", "warnings": []}
    page.evaluate("(reply) => window.__pi.fake('api_add_runs', reply)", refused)
    add(page, "center")
    assert status(page) == ("Center runs would fall outside your limits on the settings.", "error")
    assert len(run_sheet(page)) == 8, "the run sheet is as it was"


def test_a_mixture_shows_no_next_step(fresh_page):
    """A mixture's blends take none of the ways of adding runs, and its best settings are not searched."""
    page = fresh_page
    set_factors(page, [{"name": "Water"}, {"name": "Oil"}, {"name": "Wax"}])
    choose(page, design_type="simplex_lattice")
    generate(page)
    practise(page)
    reveal(page, "#analysis")
    assert page.is_hidden("#next")
    assert calls(page, "api_next_runs") == []


# --------------------------------------------------------------------------- following the results
def test_a_practice_study_follows_its_results_to_a_confirmed_optimum(blank_page):
    """Curvature at the center calls for axial runs; the response surface, for confirmation runs; they agree."""
    page = blank_page
    name_response(page, "Yield", "%", goal="maximize")
    generate(page, design_type="full_factorial", n_center_points=3)
    practise_on(page, "optimum")
    assert called_for(page) == "augment_to_rsm"
    do_what_is_called_for(page)
    assert len(run_sheet(page)) == 19
    assert status(page) == ("8 runs added at the end of the run sheet, their results simulated.", "ok")
    assert calls(page, "api_add_runs")[-1]["payload"]["kind"] == "recommended"
    assert called_for(page) == "confirm"
    assert headers(page, "#next-out .next__called table") == ["Factor", "Setting"]
    do_what_is_called_for(page)
    assert len(run_sheet(page)) == 22
    assert status(page) == (
        "3 confirmation runs added at the settings the results call for, their results simulated.",
        "ok",
    )
    assert called_for(page) == "stop"
    assert page.locator("#next-out .next__called .actions button").count() == 0, (
        "a study that stops has no step to press"
    )


def test_a_run_without_its_result_holds_the_next_step_back(blank_page):
    page = blank_page
    name_response(page, "Yield", "%", goal="maximize")
    generate(page, design_type="full_factorial", n_center_points=0)
    type_results(page, {i: 50 + i for i in range(7)})
    reveal(page, "#next")
    assert called_for(page) is None
    assert page.text_content("#next-out > p.hint") == "Once every run has its result, the results say what to run next."


def test_a_study_in_which_nothing_stands_out_widens_its_ranges_once(blank_page):
    """Its next study starts with wider ranges; the run sheet made for it says so, and the next step reads it."""
    page = blank_page
    set_factors(page, [*FACTORS, {"name": "S", "low": -1, "high": 1}])
    name_response(page, "Yield", "%", goal="maximize")
    generate(page, design_type="full_factorial", n_center_points=3)
    practise_on(page, "nothing")
    assert called_for(page) == "widen_ranges"
    do_what_is_called_for(page)
    assert status(page) == (
        "A new study, its ranges widened: check they are safe to run, then make its run sheet.",
        "ok",
    )
    assert next((f["low"], f["high"]) for f in factor_table(page)) == ("137.5", "212.5")
    generate(page)
    practise_on(page, "nothing")
    assert calls(page, "api_next_runs")[-1]["payload"]["spec"]["options"]["widened"] is True


def test_the_factors_of_the_worked_example_are_the_brief_s(first_paint):
    """The worked example's factors are the ones the tests above move: T, P and F, as tests/web/_support.py has them."""
    assert [f["name"] for f in first_paint["inputs"]["design"]["factors"]] == [f["name"] for f in FACTORS]
