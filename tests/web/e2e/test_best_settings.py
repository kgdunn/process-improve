"""The best settings for the response's goal, in Learn: where to run the process, and how far to trust the model there.

The worked example maximises its yield. Learn says what the model predicts at the best settings,
lists the settings and where each lies in its range, gives the two intervals and the leverage there,
draws how tightly to hold each factor, and puts a star on the response surface, which is drawn
through the best settings. Confirmation runs at the best settings go at the end of the run sheet,
stay out of the fit, and are judged against the model once their results are in.
"""

from __future__ import annotations

import pytest

from tests.web._support import FACTORS
from tests.web.e2e._page import (
    calls,
    definition,
    generate,
    headers,
    last_reply,
    name_response,
    reveal,
    run_sheet,
    saved_studies,
    settled,
    status,
    type_results,
)


def num(x: float) -> str:
    """Write a number as the page's sentences do: four significant figures, no trailing zeros."""
    return f"{float(f'{x:.4g}'):g}"


def table_rows(page, selector: str) -> list[list[str]]:
    """Read a table's body as the reader sees it, a list of cells per row."""
    return page.evaluate(
        "(s) => [...document.querySelectorAll(`${s} tbody tr`)].map((tr) => [...tr.cells].map((c) => c.textContent))",
        selector,
    )


def stars(page) -> list[list[float]]:
    """Return where each chart of Learn puts a star: the response surfaces, at the best settings."""
    return page.evaluate(
        """() => [...document.querySelectorAll('#charts .js-plotly-plot')].flatMap((div) =>
             div.data.filter((t) => t.name === 'Best settings').map((t) => [t.x[0], t.y[0]]))"""
    )


def set_goal(page, goal: str, target: object = None) -> None:
    """Choose the response's goal in the brief, as a reader would, and wait for the best settings it asks for."""
    before = page.evaluate("window.__pi.count('api_best_settings')")
    reveal(page, "#goal")
    page.check(f'input[name="goal"][value="{goal}"]')
    if target is not None:
        page.fill("#response-target", str(target))
    page.wait_for_function(f"window.__pi.count('api_best_settings') > {before}")
    settled(page)
    reveal(page, "#best")


# --------------------------------------------------------------------------- the worked example
def test_the_worked_example_says_where_to_run_its_process(page, first_paint):
    best = first_paint["best"]
    reveal(page, "#best")
    assert page.is_visible("#best")
    assert page.text_content("#h-best") == "The best settings for your goal"
    assert (
        page.text_content("#best .lead")
        == f"These settings give the highest Yield the model predicts: {num(best['predicted'])}%."
    )
    assert headers(page, "#best-out table")[:3] == ["Factor", "Best setting", "Where it lies"]
    f = next(s for s in best["settings"] if s["name"] == "F")
    assert table_rows(page, "#best-out table.data:not(.known-table)") == [
        ["T", "200 degC", "high end; a higher Yield may lie above"],
        ["P", "1 bar", "low end; a higher Yield may lie below"],
        ["F", f"{num(f['value'])} L/min", "inside its range, at its peak"],
    ]
    low, high = best["prediction"]
    known = table_rows(page, "#best-out .known-table")
    assert [row[0] for row in known] == [
        "95% confidence interval for the mean",
        "95% prediction interval for one new run",
        "Leverage",
    ]
    assert known[1][1] == f"{num(low)} to {num(high)}%"
    two = lambda h: f"{float(f'{h:.2g}'):g}"  # noqa: E731 - a leverage, as the page writes it
    assert known[2][1] == f"{two(best['leverage'])}, above the runs': {two(best['max_leverage'])} at most"
    text = page.text_content("#best-out")
    assert "A next study could move the ranges along the steepest ascent: T up 25 degC, P down" in text
    assert "These settings lie beyond the runs, so the prediction leans on the model there" in text
    assert page.locator("#best-out .warnings li").count() == 1, "a Box-Behnken design has no run at a corner"
    assert calls(page, "api_best_settings") == [], "the build found them, with the rest of the worked example"


def test_its_words_open_their_definitions(page):
    reveal(page, "#best")
    assert definition(page, '#best th .term[data-term="prediction-interval"]').startswith(
        "The range that one new run at these settings falls in 95% of the time."
    )
    assert definition(page, '#best th .term[data-term="leverage"]').startswith("How far settings sit from the runs")
    assert definition(page, '#confirm h4 .term[data-term="confirmation-run"]').startswith(
        "A run made at the best settings after the analysis"
    )


def test_how_tightly_to_hold_each_factor_is_drawn_as_bars(page, first_paint):
    reveal(page, "#best")
    assert page.locator("#best-chart .js-plotly-plot").count() == 1
    traces = page.evaluate("document.querySelector('#best-chart .js-plotly-plot').data.map((t) => [t.name, t.y, t.x])")
    sensitivity = {s["name"]: s for s in first_paint["best"]["sensitivity"]}
    (down, names, lowered), (up, _, raised), (pale, *_) = traces
    assert (down, up, pale) == ("Lowered 5% of its range", "Raised 5% of its range", "Outside your ranges or limits")
    assert names == ["T", "P", "F"], "the factor that moves the prediction most at the top"
    assert lowered == [sensitivity[n]["down"] for n in names]
    assert raised == [sensitivity[n]["up"] for n in names]
    assert page.is_hidden("#best-chart figcaption"), "its legend, its axis and its heading's definition say the rest"


def test_a_star_marks_the_best_settings_on_the_surface_drawn_through_them(page, first_paint):
    best = first_paint["best"]
    assert stars(page) == [[best["coded"]["T"], best["coded"]["P"]]] * 2, "on the fitted and the true surface"
    caption = page.locator("#charts figure", has_text="Fitted response surface").locator("figcaption").text_content()
    f = next(s for s in best["settings"] if s["name"] == "F")
    assert caption == (
        f"Drawn over T and P, the two factors with the largest main effects, with F at {num(f['value'])} L/min, its "
        "best setting; the star marks the best settings; the dots are the runs."
    )


def test_the_python_shown_finds_the_same_best_settings(page):
    reveal(page, "#py-best")
    page.click("#py-best summary")
    code = page.text_content("#py-best code")
    assert 'goal = {"response": "Yield", "goal": "maximize"' in code
    assert "best = optimize_responses(" in code
    assert 'analysis_type="prediction"' in code
    assert code.index("analysis = analyze_experiment(") < code.index("best = optimize_responses("), "after step 3"


# --------------------------------------------------------------------------- the goal in the brief
def test_the_goal_in_the_brief_is_the_goal_searched(page):
    set_goal(page, "minimize")
    assert page.text_content("#best .lead").startswith("These settings give the lowest Yield the model predicts:")
    payload = calls(page, "api_plots")[-1]["payload"]
    assert payload["best"] == last_reply(page, "api_best_settings")["result"]["coded"], "the surface drawn through them"
    set_goal(page, "target")
    assert (
        page.text_content("#best-out")
        == "Type the target for Yield in the brief, and the page finds the settings that reach it."
    )
    set_goal(page, "target", 60)
    assert page.text_content("#best .lead") == "These settings give Yield on its target, 60%, as the model predicts."
    assert "Other settings reach the target too" in page.text_content("#best-out")


def test_a_goal_of_learning_says_how_to_see_the_best_settings(page):
    set_goal(page, "learn")
    assert page.text_content("#best-out") == (
        "Set a goal for Yield in the brief, to maximise, minimise or hit a target, and the page finds the best "
        "settings here."
    )
    for hidden in ("#best-chart", "#confirm", "#py-best"):
        assert page.is_hidden(hidden), hidden
    assert stars(page) == [], "no best settings, no star"


# --------------------------------------------------------------------------- confirmation runs
def confirmation_rows(page) -> list[list[str]]:
    """Return the run sheet's confirmation runs, as the reader sees them."""
    at = headers(page).index("Confirmation run")
    return [row for row in run_sheet(page) if row[at] == "yes"]


def test_confirmation_runs_added_while_practising_are_simulated_and_judged(page, first_paint):
    reveal(page, "#best")
    assert page.text_content("#add-confirmation") == "Add 3 confirmation runs at these settings"
    page.click("#add-confirmation")
    settled(page)
    assert status(page)[0] == "3 confirmation runs added at the best settings, their results simulated."
    reveal(page, "#design-table")
    assert headers(page)[-1] == "Confirmation run"
    at = first_paint["best"]["confirm_at"]
    rows = confirmation_rows(page)
    assert [row[:5] for row in rows] == [["", str(order), "200", "1", num(at["F"])] for order in (16, 17, 18)]
    assert page.locator("#design-table input.sim").count() == 18, "their results simulated too"
    assert "15 completed runs" in page.text_content("#analysis"), "they do not enter the fit"
    reveal(page, "#best")
    verdict = page.locator("#confirm-out .verdict")
    assert verdict.count() == 1
    assert verdict.get_attribute("data-state") in {"inside", "outside"}
    assert "The average of their 3 runs" in verdict.text_content()
    assert page.is_hidden("#add-confirmation"), "runs at these settings are on the sheet"
    assert page.is_visible("#remove-confirmation")
    assert "# The confirmation runs are left out" in page.text_content("#py-best code")

    page.click("#remove-confirmation")
    settled(page)
    assert "Confirmation run" not in headers(page)
    assert len(run_sheet(page)) == 15
    assert page.locator("#confirm-out .verdict").count() == 0
    assert page.is_visible("#add-confirmation")


def test_confirmation_runs_made_by_hand_say_whether_the_model_holds_there(blank_page):
    page = blank_page
    name_response(page, "Yield", "%", goal="maximize")
    generate(page, design_type="box_behnken")
    sheet = run_sheet(page)
    names = headers(page)
    at = {name: names.index(name) for name in ("T", "P", "F")}

    def surface(row: list[str]) -> float:
        t, p, f = (
            (float(row[at[n]]) - (fa["low"] + fa["high"]) / 2) / ((fa["high"] - fa["low"]) / 2)
            for n, fa in zip("TPF", FACTORS, strict=True)
        )
        return round(60 + 3 * t - 2 * p - 2 * t**2 - 1.5 * f**2 + 0.3 * ((len(row[0]) % 3) - 1), 2)

    type_results(page, {i: surface(row) for i, row in enumerate(sheet)})
    reveal(page, "#best")
    page.click("#add-confirmation")
    settled(page)
    assert status(page)[0] == "3 confirmation runs added at the end of the run sheet: enter their results in Run."
    group = last_reply(page, "api_best_settings")["result"]["confirmation"][0]
    low, high = group["interval"]
    verdict = page.locator("#confirm-out .verdict")
    assert verdict.get_attribute("data-state") == "waiting"
    assert verdict.text_content().endswith(
        f"When their average is between {num(low)} and {num(high)}%, the model holds there."
    )

    inside = round(group["predicted"], 2)
    type_results(page, {15: inside, 16: inside + 0.1, 17: inside - 0.1})
    reveal(page, "#best")
    assert verdict.get_attribute("data-state") == "inside"
    assert verdict.text_content().endswith("the model holds at these settings.")
    assert "18 runs" not in page.text_content("#study-about"), "the study is the design's runs"

    type_results(page, {15: inside - 30, 16: inside - 31, 17: inside - 29})
    reveal(page, "#best")
    assert verdict.get_attribute("data-state") == "outside"
    assert "the process differs from the model at these settings" in verdict.text_content()
    saved = saved_studies(page, lambda kept: bool(kept) and "Confirmation run" in kept[0]["design"]["columns"])
    assert saved[0]["design"]["rows"][-1][-1] == "yes", "the study kept in this browser keeps them"


@pytest.mark.parametrize("context_options", [{"viewport": {"width": 390, "height": 844}, "is_mobile": True}])
def test_on_a_phone_the_best_settings_fit_the_screen(page):
    reveal(page, "#best")
    fits = "document.documentElement.scrollWidth <= document.documentElement.clientWidth"
    assert page.evaluate(fits), "no sideways scroll"
    page.click("#add-confirmation")
    settled(page)
    reveal(page, "#best")
    assert page.evaluate(fits), "nor once the confirmation runs are judged"
    box = page.locator("#best-chart").bounding_box()
    assert box["x"] >= 0
    assert box["x"] + box["width"] <= 390
