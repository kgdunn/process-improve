"""A Taguchi orthogonal array: ``api_make_design`` with ``design_type="taguchi"``, and what the page does with it.

The library chooses the smallest standard array with a column for each factor; the page runs each
numeric factor at two levels, or at three (``n_levels``), and a categorical factor at its two labels.
The array is analysed for its main effects, and at three levels for the numeric factors' squares too,
and the analysis says once that the interactions are aliased with them.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from tests.web._support import FACTORS, coded, factors_named, fill_workbook, held_of, run_sheet

SUPPLIER = {"name": "Supplier", "low": "A", "high": "B", "units": ""}


def named(k: int) -> list[dict]:
    """Return ``k`` numeric factors, each from -1 to +1."""
    return factors_named(*(f"X{i + 1}" for i in range(k)))


def make(api, factors: list[dict], **options: object) -> dict:
    return api.ok("api_make_design", {"design_type": "taguchi", "factors": factors, **options})


def x_of(design: dict) -> np.ndarray:
    """Return a design's runs in coded units, its categorical factors coded -1 and +1."""
    return coded(run_sheet(design), design["spec"]["factors"]).to_numpy(dtype=float)


# --------------------------------------------------------------------------- the array chosen
@pytest.mark.parametrize(
    ("factors", "options", "array", "runs"),
    [
        (named(3), {}, "Taguchi L4 (2^3): 4 runs, main effects.", 4),
        (named(7), {}, "Taguchi L8 (2^7): 8 runs, main effects.", 8),
        (named(11), {}, "Taguchi L12 (2^11): 12 runs, main effects.", 12),
        (named(4), {"n_levels": 3}, "Taguchi L9 (3^4): 9 runs, main effects and squares.", 9),
    ],
    ids=["3 factors, two levels", "7 factors, two levels", "11 factors, two levels", "4 factors, three levels"],
)
def test_the_array_is_the_smallest_with_a_column_for_each_factor(api, factors, options, array, runs):
    design = make(api, factors, **options)
    assert design["n_runs"] == runs
    assert design["facts"]["notes"][0] == array, "the design's facts name the array first"
    assert design["facts"]["center_runs"] == 0, "an orthogonal array has no center runs"


def test_one_more_factor_takes_the_next_array(api):
    assert make(api, named(4))["n_runs"] == 8
    assert make(api, named(12))["facts"]["notes"][0] == "Taguchi L16 (2^15): 16 runs, main effects."
    assert (
        make(api, named(5), n_levels=3)["facts"]["notes"][0]
        == "Taguchi L18 (2^1 3^7): 18 runs, main effects and squares."
    )
    assert (
        make(api, named(8), n_levels=3)["facts"]["notes"][0] == "Taguchi L27 (3^13): 27 runs, main effects and squares."
    )


@pytest.mark.parametrize(
    ("factors", "options"),
    [
        (named(7), {}),
        (named(11), {}),
        (named(4), {"n_levels": 3}),
        ([*named(6), SUPPLIER], {"n_levels": 3}),
    ],
    ids=["L8", "L12", "L9", "L18 with a categorical factor"],
)
def test_each_column_is_balanced_and_every_pair_orthogonal(api, factors, options):
    """Each level of a factor is run equally often, and so is each pair of levels of any two factors."""
    x = x_of(make(api, factors, **options))
    for column in x.T:
        _, counts = np.unique(column, return_counts=True)
        assert len(set(counts)) == 1, "every level of a factor equally often"
    for i, j in itertools.combinations(range(x.shape[1]), 2):
        _, counts = np.unique(x[:, [i, j]], axis=0, return_counts=True)
        levels = len(np.unique(x[:, i])) * len(np.unique(x[:, j]))
        assert len(counts) == levels, ("every pair of levels is run", i, j)
        assert len(set(counts)) == 1, ("each pair equally often", i, j)
    centred = x - x.mean(axis=0)
    gram = centred.T @ centred
    np.testing.assert_allclose(gram - np.diag(np.diag(gram)), 0, atol=1e-9, err_msg="the main effects are orthogonal")


def test_three_levels_are_the_low_the_center_and_the_high(api):
    design = make(api, FACTORS, n_levels=3)
    assert sorted(set(run_sheet(design)["T"])) == [150.0, 175.0, 200.0]
    assert design["spec"]["options"] == {"random_state": 42, "n_levels": 3}
    assert design["outside_range"] == []


def test_two_levels_are_the_default_and_store_no_option(api):
    design = make(api, FACTORS)
    assert sorted(set(run_sheet(design)["T"])) == [150.0, 200.0]
    assert design["spec"]["options"] == {"random_state": 42}
    assert make(api, FACTORS, n_levels=2)["spec"] == design["spec"]


# --------------------------------------------------------------------------- categorical factors
def test_a_categorical_factor_takes_a_two_level_column(api):
    """Beside numeric factors at three levels, a categorical factor runs at its two labels: the L18 has the column."""
    design = make(api, [*FACTORS, SUPPLIER], n_levels=3)
    sheet = run_sheet(design)
    assert design["n_runs"] == 18
    assert sheet["Supplier"].value_counts().to_dict() == {"A": 9, "B": 9}
    assert design["facts"]["notes"][0] == "Taguchi L18 (2^1 3^7): 18 runs, main effects and squares."
    assert design["spec"]["factors"][3] == SUPPLIER


def test_two_categorical_factors_at_three_levels_take_the_l36(api):
    """The L18 has one two-level column: two categorical factors beside three-level ones need the L36's."""
    other = {**SUPPLIER, "name": "Catalyst", "low": "old", "high": "new"}
    design = make(api, [*FACTORS, SUPPLIER, other], n_levels=3)
    assert design["facts"]["notes"][0] == "Taguchi L36 (2^11 3^12): 36 runs, main effects and squares."


def test_categorical_factors_alone_take_a_two_level_array(api):
    factors = [{**SUPPLIER, "name": f"S{i}"} for i in range(5)]
    design = make(api, factors, n_levels=3)
    assert design["facts"]["notes"][0] == "Taguchi L8 (2^7): 8 runs, main effects."
    assert design["model"] == "main_effects", "no numeric factor, so no square"


# --------------------------------------------------------------------------- the model it is made for
def test_a_two_level_array_is_made_for_the_main_effects(api):
    assert make(api, named(7))["model"] == "main_effects"


def test_a_three_level_array_is_made_for_the_main_effects_and_the_numeric_factors_squares(api):
    design = make(api, [*FACTORS, SUPPLIER], n_levels=3)
    assert design["model"] == design["spec"]["model"] == "main_quadratic"
    responses = np.random.default_rng(1).normal(50, 1, design["n_runs"]).tolist()
    reply = api.reply("api_analyze_runs", {**held_of(design), "responses": responses})
    analysis = reply["result"]
    assert analysis["formula"] == "T + P + F + Supplier + I(T ** 2) + I(P ** 2) + I(F ** 2)", "no square of a category"
    assert analysis["summary"]["df_residual"] == 18 - 8
    assert reply["warnings"] == [], "no warning about aliasing the model does not reach"


def test_the_designed_model_recovers_a_process_of_main_effects_and_squares(api):
    """With no noise, the fit returns the process's coefficients: the array estimates each term apart."""
    design = make(api, FACTORS, n_levels=3)
    x = coded(run_sheet(design), FACTORS)
    y = 60 + 3 * x["T"] - 2 * x["P"] + 0.5 * x["F"] - 4 * x["T"] ** 2 + 1.5 * x["F"] ** 2
    analysis = api.ok("api_analyze_runs", {**held_of(design), "responses": y.tolist()})
    found = {c["term"]: c["coefficient"] for c in analysis["coefficients"]}
    expected = {"Intercept": 60, "T": 3, "P": -2, "F": 0.5, "I(T ** 2)": -4, "I(P ** 2)": 0, "I(F ** 2)": 1.5}
    assert found == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize(
    ("factors", "options", "estimates"),
    [(named(7), {}, "main effects"), (FACTORS, {"n_levels": 3}, "main effects and squares")],
    ids=["two levels", "three levels"],
)
def test_the_analysis_says_once_that_interactions_are_aliased_with_what_the_array_estimates(
    api, factors, options, estimates
):
    design = make(api, factors, **options)
    responses = np.random.default_rng(2).normal(50, 1, design["n_runs"]).tolist()
    reply = api.reply("api_analyze_runs", {**held_of(design), "responses": responses})
    notes = reply["result"]["notes"]
    note = (
        f"An orthogonal array estimates the {estimates}; the interactions are aliased with them, in full or in part, "
        "so a large effect may belong to an interaction."
    )
    assert [n for n in notes if "aliased" in n] == [note]
    assert reply["warnings"] == []


def test_a_model_with_interactions_has_the_alias_chains_note_instead(api):
    design = make(api, named(4))
    responses = np.random.default_rng(3).normal(50, 1, design["n_runs"]).tolist()
    notes = api.ok("api_analyze_runs", {**held_of(design), "responses": responses, "model": "interactions"})["notes"]
    assert not any(n.startswith("An orthogonal array") for n in notes)
    assert any(n.startswith("This design cannot tell these terms apart") for n in notes)


def test_too_few_results_name_the_model_the_array_was_made_for(api):
    design = make(api, named(4), n_levels=3)
    responses = [1.0, 2.0, 3.0, None, None, None, None, None, None]
    message = api.error("api_analyze_runs", {**held_of(design), "responses": responses})
    assert message.startswith("The main effects and squares model has 9 coefficients, so it needs at least 9 ")


# --------------------------------------------------------------------------- what the page refuses
def test_the_page_takes_the_factor_counts_the_arrays_take(bootstrap):
    info = bootstrap.DESIGNS["taguchi"]
    assert (info["min_factors"], info["max_factors"]) == (2, 12)
    assert info["model"] == "main_effects"


def test_an_array_takes_no_limits_on_the_settings(api):
    request = {"design_type": "taguchi", "factors": FACTORS, "constraints": [{"expression": "T + 25*P <= 250"}]}
    assert api.error("api_make_design", request) == (
        "The Taguchi orthogonal array design cannot keep its runs inside limits on the settings: choose the "
        "D-optimal or I-optimal design, or remove the limits."
    )


@pytest.mark.parametrize("levels", [4, 1, 2.5, "three"])
def test_levels_the_page_does_not_offer_are_refused(api, levels):
    assert api.error("api_make_design", {"design_type": "taguchi", "factors": FACTORS, "n_levels": levels}) == (
        "A Taguchi array runs each numeric factor at 2 levels or at 3 on this page."
    )


# --------------------------------------------------------------------------- the comparison
@pytest.mark.parametrize(
    ("model", "options"),
    [("main_effects", {}), ("interactions", {}), ("main_quadratic", {"n_levels": 3}), ("quadratic", {"n_levels": 3})],
)
def test_the_comparison_runs_the_array_at_three_levels_for_a_model_with_squares(api, model, options):
    plan = api.ok("api_compare_plan", {"factors": FACTORS, "model": model})
    taguchi = next(d for d in plan["designs"] if d["design_type"] == "taguchi")
    assert taguchi["label"] == "Taguchi"
    assert {key: taguchi["spec"].get(key) for key in options} == options
    assert ("n_levels" in taguchi["spec"]) == bool(options)


@pytest.mark.slow
def test_the_comparison_scores_the_three_level_array_on_the_main_effects_and_squares(api):
    reply = api.reply("api_candidates", {"factors": named(4), "model": "main_quadratic"})
    result = reply["result"]["result"]
    label = next(d for d in result["designs"] if d.startswith("Taguchi"))
    assert label == "Taguchi [n=9]"
    assert result["values"][label]["fits"] is True
    assert label in result["fds"], "a design that fits the model has its FDS curve"
    made = next(m for m in reply["result"]["made"] if m["design_type"] == "taguchi")
    assert made["warnings"] == []


# --------------------------------------------------------------------------- the workbook and renaming
def test_the_workbook_keeps_the_levels_and_the_model(api):
    design = make(api, [*FACTORS, SUPPLIER], n_levels=3)
    filled = fill_workbook(design["xlsx_base64"], lambda runs: np.arange(len(runs), dtype=float) % 5)
    back = api.ok("api_read_workbook", {"xlsx_base64": filled})
    assert back["spec"] == design["spec"]
    assert back["rows"] == design["rows"]
    from_book = api.ok("api_analyze", {"xlsx_base64": filled})
    from_page = api.ok("api_analyze_runs", {**held_of(back), "responses": back["responses"]})
    assert from_book["formula"] == from_page["formula"] == "T + P + F + Supplier + I(T ** 2) + I(P ** 2) + I(F ** 2)"
    assert from_book["coefficients"] == from_page["coefficients"]


def test_renaming_keeps_the_runs_and_names_the_array_as_before(api):
    factors = [*FACTORS, SUPPLIER]
    design = make(api, factors, n_levels=3)
    renamed = [{**f, "name": {"T": "Temperature", "Supplier": "Vendor"}.get(f["name"], f["name"])} for f in factors]
    request = {**held_of(design), "facts": design["facts"], "outside_range": design["outside_range"]}
    reply = api.ok("api_relabel", {"design": request, "factors": renamed})
    assert reply["columns"] == ["Std order", "Run order", "Temperature", "P", "F", "Vendor", "y"]
    assert reply["facts"]["notes"] == make(api, renamed, n_levels=3)["facts"]["notes"]
    assert reply["spec"]["options"] == {"random_state": 42, "n_levels": 3}
    responses = np.random.default_rng(4).normal(50, 1, design["n_runs"]).tolist()
    analysis = api.ok(
        "api_analyze_runs",
        {"spec": reply["spec"], "columns": reply["columns"], "rows": design["rows"], "responses": responses},
    )
    assert analysis["formula"] == ("Temperature + P + F + Vendor + I(Temperature ** 2) + I(P ** 2) + I(F ** 2)"), (
        "the squares follow the new names"
    )
