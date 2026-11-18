"""Factors that are hard to change (#1225): a split-plot design from the page, and how Learn judges its terms.

A furnace temperature is set once for a group of runs, a whole plot. The library makes the design
(``generate_design(hard_to_change=...)``) and judges each coefficient against the spread it meets
(``analysis_type="split_plot"``); these tests are of what the page makes of both.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.web._support import fill_workbook, held_of, run_sheet, workbook_of

FACTORS = [
    {"name": "T", "low": 150, "high": 200, "units": "degC"},
    {"name": "P", "low": 1, "high": 3, "units": "bar"},
    {"name": "S", "low": 10, "high": 30, "units": "rpm"},
]
PLAN = {"design_type": "d_optimal", "factors": FACTORS, "model_type": "interactions", "budget": 12}
PLOT = "Whole plot"


@pytest.fixture
def design(api) -> dict:
    return api.ok("api_make_design", {**PLAN, "hard_to_change": ["T"]})


def results(design: dict, upto: int | None = None) -> list[float | None]:
    """Return results with a shift of their own for each whole plot, larger than the run-to-run spread."""
    sheet = run_sheet(design)
    rng = np.random.default_rng(0)
    shift = rng.normal(0, 2, int(sheet[PLOT].max()) + 1)
    y = 50 + 0.2 * sheet["T"] + 3 * sheet["P"] - 0.2 * sheet["S"] + shift[sheet[PLOT]] + rng.normal(0, 1, len(sheet))
    return [v if upto is None or i < upto else None for i, v in enumerate(y.round(3).tolist())]


def study(design: dict, **more: object) -> dict:
    return {**held_of(design), "responses": results(design), "model": "interactions", **more}


# --------------------------------------------------------------------------- the design
def test_the_run_sheet_lists_the_runs_whole_plot_by_whole_plot(design):
    sheet = run_sheet(design)
    assert design["columns"][:3] == ["Std order", "Run order", PLOT]
    assert sheet[PLOT].is_monotonic_increasing, "the runs of a whole plot are made one after another"
    assert (sheet.groupby(PLOT)["T"].nunique() == 1).all(), "T is set once for each whole plot"
    assert sheet.groupby(PLOT)["P"].nunique().max() > 1, "and P changes inside one"
    assert design["spec"]["options"]["hard_to_change"] == ["T"]


def test_the_notes_say_how_the_runs_are_made(design):
    plots = run_sheet(design)[PLOT].nunique()
    assert design["facts"]["notes"][-1].startswith(
        f"T is hard to change, so the runs are made in {plots} groups, called whole plots: set T once for each"
    )


def test_the_factors_are_kept_in_the_study_s_order(api):
    made = api.ok("api_make_design", {**PLAN, "budget": 16, "hard_to_change": ["P", "T"]})
    assert made["spec"]["options"]["hard_to_change"] == ["T", "P"]
    sheet = run_sheet(made)
    assert (sheet.groupby(PLOT)[["T", "P"]].nunique() == 1).all().all()


@pytest.mark.parametrize("hard", [None, []])
def test_a_design_with_no_such_factor_is_as_before(api, hard):
    made = api.ok("api_make_design", {**PLAN, "hard_to_change": hard})
    assert PLOT not in made["columns"]
    assert "hard_to_change" not in made["spec"]["options"]


def test_a_design_that_takes_no_such_factor_leaves_it_out(api):
    made = api.ok("api_make_design", {"design_type": "full_factorial", "factors": FACTORS, "hard_to_change": ["T"]})
    assert PLOT not in made["columns"]
    assert "hard_to_change" not in made["spec"]["options"]


# --------------------------------------------------------------------------- Learn
def test_each_term_is_judged_against_the_spread_it_meets(api, design):
    fit = api.ok("api_analyze_runs", study(design))
    told = fit["split_plot"]
    assert told["hard_to_change"] == ["T"]
    assert told["n_whole_plots"] == run_sheet(design)[PLOT].nunique()
    assert told["sd_whole_plot"] > told["sd_run"] > 0, "the results were made so"
    assert told["whole_plot_terms"] == ["T"]
    by_term = {c["term"]: c for c in fit["coefficients"]}
    assert by_term["T"]["stratum"] == "whole_plot"
    assert by_term["P"]["stratum"] == "subplot"
    assert by_term["T"]["std_error"] > 2 * by_term["P"]["std_error"], "T is known less well than P"
    assert fit["anova"] == [], "an ANOVA table has one error, and these runs have two"
    assert fit["lack_of_fit"] is None
    assert fit["notes"][-1].startswith("T was set once for each of the")
    assert "T is judged against the first, and every other term against the second." in fit["notes"][-1]


def test_the_ordinary_fit_makes_the_hard_factor_look_more_certain(api, design):
    ordinary = api.ok("api_analyze_runs", study({**design, "spec": _without(design["spec"])}))
    split = api.ok("api_analyze_runs", study(design))

    def error_of(fit: dict) -> float:
        return next(c["std_error"] for c in fit["coefficients"] if c["term"] == "T")

    assert error_of(split) > error_of(ordinary)


def _without(spec: dict) -> dict:
    options = {key: value for key, value in spec["options"].items() if key != "hard_to_change"}
    return {**spec, "options": options}


def test_too_few_results_are_judged_against_one_spread_and_the_notes_say_so(api, design):
    fit = api.ok("api_analyze_runs", {**study(design), "responses": results(design, upto=5), "model": "main_effects"})
    assert "split_plot" not in fit
    assert "the results so far are too few to judge it against the spread from whole plot" in fit["notes"][-1]
    assert fit["anova"], "the ordinary fit stands"


def test_a_workbook_keeps_the_whole_plots_and_is_analysed_alike(api, design):
    y = results(design)
    book = fill_workbook(workbook_of(api, design), lambda runs: y)
    back = api.ok("api_read_workbook", {"xlsx_base64": book})
    assert back["columns"] == design["columns"]
    assert back["spec"]["options"]["hard_to_change"] == ["T"]
    from_book = api.ok("api_analyze", {"xlsx_base64": book, "model": "interactions"})
    from_page = api.ok("api_analyze_runs", study(design))
    assert from_book["split_plot"] == from_page["split_plot"]
    assert from_book["coefficients"] == from_page["coefficients"]


def test_the_other_parts_of_learn_still_answer(api, design):
    held = study(design)
    held["spec"] = {**held["spec"], "response_info": {"goal": "maximize"}}
    best = api.ok("api_best_settings", held)
    assert {s["name"] for s in best["settings"]} == {"T", "P", "S"}
    assert api.ok("api_plots", held)


def test_learn_offers_no_runs_to_add(api, design):
    offered = api.ok("api_next_runs", study(design))
    assert offered["offers"] == []
    assert offered["notes"] == [
        "This design sets a factor once for each group of runs, and the runs Learn adds to a design are made one "
        "by one, so it offers none here."
    ]
