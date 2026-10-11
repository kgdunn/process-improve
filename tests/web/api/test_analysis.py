"""Analysing the results: ``api_analyze_runs``, which step 3 of the page calls, and ``api_analyze`` for a workbook.

Factors are coded to -1 and +1 from their ranges. Terms a design cannot tell apart are
estimated once, as their sum; a saturated two-level fit is judged with Lenth's method; and
when the results were simulated, each coefficient carries the true value of what it estimates.
"""

from __future__ import annotations

import pytest

from tests.web._support import FACTORS, factors_named, fill_workbook, held_of, quadratic, run_sheet


def analyse(api, spec: dict, model: str = "") -> dict:
    """Generate a design, fill its workbook with the ``quadratic`` response, and return the analysis reply."""
    design = api.ok("api_make_design", spec)
    return api.reply("api_analyze", {"xlsx_base64": fill_workbook(design["xlsx_base64"], quadratic), "model": model})


def by_label(result: dict) -> dict:
    """Index a result's coefficients by the label the page shows."""
    return {c["label"]: c for c in result["coefficients"]}


# --------------------------------------------------------------------------- the runs the page holds
def test_the_worked_example_recovers_the_true_process(api, held, simulated):
    analysis = api.ok("api_analyze_runs", {**held, "responses": simulated["responses"]})
    truth = {t["term"]: t["coefficient"] for t in simulated["truth"]}
    for c in analysis["coefficients"]:
        assert c["ci_low"] <= truth.get(c["term"], 0.0) <= c["ci_high"], c["term"]
    assert analysis["model"] == "quadratic"


def test_true_values_are_what_each_coefficient_estimates(api, held, simulated):
    """In a Box-Behnken design every term is estimated on its own, so the true values are the process's."""
    request = {**held, "responses": simulated["responses"], "truth": simulated["truth"]}
    analysis = api.ok("api_analyze_runs", request)
    truth = {t["term"]: t["coefficient"] for t in simulated["truth"]}
    assert {c["term"]: c["true_value"] for c in analysis["coefficients"]} == {
        c["term"]: truth.get(c["term"], 0.0) for c in analysis["coefficients"]
    }, "exact, with no rounding error left on the zeros"


def test_without_the_true_process_no_coefficient_has_a_true_value(api, held, simulated):
    """Results typed in or uploaded have no known process: the page then leaves the "True value" column out."""
    analysis = api.ok("api_analyze_runs", {**held, "responses": simulated["responses"]})
    assert all("true_value" not in c for c in analysis["coefficients"])


def test_an_aliased_estimate_is_set_beside_the_sum_it_estimates(api):
    """In a 2^(3-1) design F is aliased with T:P, so its estimate is centred on F + T:P."""
    spec = {
        "design_type": "fractional_factorial",
        "resolution": 3,
        "n_center_points": 0,
        "factors": factors_named(*"TPF"),
    }
    held = held_of(api.ok("api_make_design", spec))
    sim = api.ok("api_simulate", {**held, "scenario": "reactor", "sigma": 0.01})
    analysis = api.ok(
        "api_analyze_runs", {**held, "responses": sim["responses"], "model": "interactions", "truth": sim["truth"]}
    )
    true = {c["label"]: c["true_value"] for c in analysis["coefficients"]}
    assert true["F + T·P"] == pytest.approx(1.75 + 1.5)
    assert true["Intercept"] == pytest.approx(62 - 2.25), "F squared is 1 at every run of a two-level design"


def test_the_model_chosen_replaces_the_one_the_design_was_made_for(api, held, simulated):
    request = {**held, "responses": simulated["responses"]}
    assert api.ok("api_analyze_runs", request)["model"] == held["spec"]["model"] == "quadratic"
    main = api.ok("api_analyze_runs", {**request, "model": "main_effects"})
    assert main["model"] == "main_effects"
    assert [c["term"] for c in main["coefficients"]] == ["Intercept", "T", "P", "F"]


def test_runs_still_to_do_are_left_out_and_listed(api, held, simulated):
    responses = [*simulated["responses"][:-2], None, None]
    analysis = api.ok("api_analyze_runs", {**held, "responses": responses})
    assert analysis["n_completed"] == 13
    assert analysis["pending_runs"] == [14, 15]


def test_a_mismatched_response_list_is_refused(api, held):
    assert "2 responses were sent for 15 runs" in api.error("api_analyze_runs", {**held, "responses": [1.0, 2.0]})


@pytest.mark.parametrize(
    "spec_change",
    [
        {"response": "__import__('os')"},
        {"model": "y ~ T"},
        {"factors": [{"name": "T; import os", "low": 0, "high": 1}, {"name": "P", "low": 0, "high": 1}]},
    ],
)
def test_a_specification_is_data_never_code(api, held, spec_change):
    api.error("api_analyze_runs", {**held, "spec": {**held["spec"], **spec_change}, "responses": [1.0] * 15})


@pytest.mark.parametrize("model", ["__import__('os').system('true')", "y ~ T", ""])
def test_model_name_from_the_request_is_never_a_formula(api, model):
    reply = analyse(api, {"design_type": "box_behnken", "factors": FACTORS}, model=model)
    assert reply["ok"] is (model == ""), "an empty choice means 'as designed'; anything else not in MODELS is refused"


# --------------------------------------------------------------------------- what the fit can and cannot estimate
def test_too_few_runs_says_how_many_are_needed(api):
    design = api.ok("api_make_design", {"design_type": "box_behnken", "factors": FACTORS})
    xlsx = fill_workbook(design["xlsx_base64"], quadratic, blank_rows=tuple(range(8)))
    assert "10 coefficients" in api.error("api_analyze", {"xlsx_base64": xlsx, "model": "quadratic"})


def test_aliased_terms_are_estimated_once_as_their_chain(api):
    """In a 2^(3-1) each main effect shares its column with an interaction: the data give their sum, once.

    The fit used to list every term, each with half the sum, so T read 2 where it is 4.
    """
    reply = analyse(api, {"design_type": "fractional_factorial", "factors": FACTORS}, model="interactions")
    assert reply["ok"], reply.get("error")
    coef = by_label(reply["result"])
    assert set(coef) == {"Intercept", "T + P·F", "P + T·F", "F + T·P"}
    assert coef["T + P·F"]["coefficient"] == pytest.approx(4, abs=0.2)
    assert coef["F + T·P"]["coefficient"] == pytest.approx(2, abs=0.2)
    assert coef["T + P·F"]["aliases"] == ["T", "P:F"]
    assert any("cannot tell these terms apart" in note for note in reply["result"]["notes"])


def test_squares_of_a_two_level_design_are_one_chain(api):
    """With center points the squares of a two-level design are one column, so only their total is estimable."""
    spec = {"design_type": "full_factorial", "factors": FACTORS, "n_center_points": 3}
    result = analyse(api, spec, model="quadratic")["result"]
    coef = by_label(result)
    assert coef["T² + P² + F²"]["coefficient"] == pytest.approx(-5, abs=0.3)
    assert "T²" not in coef
    assert "(T² + P² + F²)" in result["equation"]


def test_squares_without_center_points_are_not_estimable(api):
    """Without center points every square is 1 at every run, the same as the intercept."""
    spec = {"design_type": "full_factorial", "factors": FACTORS, "n_center_points": 0}
    result = analyse(api, spec, model="quadratic")["result"]
    assert sorted(result["not_estimable"]) == ["F²", "P²", "T²"]
    assert {"T", "P", "F", "T·P", "T·F", "P·F"} <= set(by_label(result))


def test_lack_of_fit_is_tested_against_the_repeated_center_runs(api, held, simulated):
    """Three center runs give two degrees of freedom of pure error, which the page states beside the test."""
    analysis = api.ok("api_analyze_runs", {**held, "responses": simulated["responses"]})
    assert analysis["lack_of_fit"]["df_pure_error"] == 2
    assert 0 <= analysis["lack_of_fit"]["p_value"] <= 1


def test_saturated_two_level_fit_is_judged_with_lenth(api):
    """Four runs for four coefficients leave no error to test against; Lenth's method judges the effects."""
    spec = {"design_type": "plackett_burman", "factors": FACTORS, "n_center_points": 0}
    reply = analyse(api, spec, model="main_effects")
    assert reply["ok"], reply.get("error")
    result = reply["result"]
    assert result["summary"]["df_residual"] == 0
    assert {"PSE", "ME", "SME"} <= set(result["lenth"])
    assert sorted(e["term"] for e in result["lenth"]["effects"]) == ["F", "P", "T"]
    assert any("Lenth" in note for note in result["notes"])
    assert not any("divide by zero" in w for w in reply["warnings"]), "numerical noise is not a warning for the reader"
    assert any("only 3 effects" in note for note in result["notes"]), "three effects give Lenth's margins one df"


def test_a_saturated_fit_has_no_predicted_r_squared(api):
    """Every run is fitted exactly, so none can be predicted from the others; the page explains the gap."""
    spec = {"design_type": "plackett_burman", "factors": FACTORS, "n_center_points": 0}
    summary = analyse(api, spec, model="main_effects")["result"]["summary"]
    assert summary["r_squared_pred"] is None


def test_lenth_margins_are_not_called_wide_with_seven_effects(api):
    spec = {"design_type": "plackett_burman", "factors": factors_named(*"ABCDEFG"), "n_center_points": 0}
    design = api.ok("api_make_design", spec)
    workbook = fill_workbook(design["xlsx_base64"], lambda runs: (10 + 3 * runs["A"] - 2 * runs["C"]).to_numpy())
    result = api.ok("api_analyze", {"xlsx_base64": workbook, "model": "main_effects"})
    assert len(result["lenth"]["effects"]) == 7
    assert not any("margins are wide" in note for note in result["notes"])


def test_saturated_fit_of_a_three_level_design_says_nothing_can_be_tested(api):
    """Lenth's method needs orthogonal two-level effects, which a saturated D-optimal design does not have."""
    spec = {"design_type": "d_optimal", "factors": FACTORS[:2], "model_type": "quadratic", "budget": 6}
    reply = analyse(api, spec, model="quadratic")
    assert reply["ok"], reply.get("error")
    result = reply["result"]
    assert result["summary"]["df_residual"] == 0
    assert result["lenth"] is None
    assert any("no degrees of freedom" in note for note in result["notes"])


#: The results of the book's breaded-chicken exercise, by coded setting of D, R and P: they make
#: the D·P and R·P interactions exactly zero.
CHICKEN = {
    (-1, -1, -1): 3,
    (1, -1, -1): 9,
    (-1, 1, -1): 3,
    (1, 1, -1): 7,
    (-1, -1, 1): 3,
    (1, -1, 1): 10,
    (-1, 1, 1): 4,
    (1, 1, 1): 7,
}


@pytest.mark.xfail(strict=True, reason="a zero coefficient is shown as round-off, 6.106e-16")
def test_a_coefficient_the_results_make_zero_reads_zero(api):
    """Least squares returns an exactly zero coefficient as round-off; the page shows 0, as the book prints it."""
    spec = {"design_type": "full_factorial", "factors": factors_named(*"DRP"), "n_center_points": 0}
    design = api.ok("api_make_design", spec)
    responses = [CHICKEN[tuple(int(v) for v in row)] for row in run_sheet(design)[["D", "R", "P"]].to_numpy()]
    result = api.ok("api_analyze_runs", {**held_of(design), "responses": responses, "model": "interactions"})
    coefficients = by_label(result)
    assert coefficients["D·P"]["coefficient"] == 0
    assert coefficients["R·P"]["coefficient"] == 0
    assert "e-1" not in result["equation"]


@pytest.mark.xfail(strict=True, reason="a zero effect is shown as round-off, 8.882e-16")
def test_a_zero_effect_reads_zero_in_lenth_s_table(api):
    """A saturated fit whose results make C exactly zero: its effect reads 0, as its coefficient does."""
    spec = {"design_type": "fractional_factorial", "factors": factors_named(*"ABC"), "resolution": 3}
    design = api.ok("api_make_design", {**spec, "n_center_points": 0})
    responses = [10 + 2 * a + b for a, b in run_sheet(design)[["A", "B"]].to_numpy()]
    result = api.ok("api_analyze_runs", {**held_of(design), "responses": responses, "model": "main_effects"})
    assert {e["label"]: e["effect"] for e in result["lenth"]["effects"]}["C"] == 0
    assert by_label(result)["C"]["coefficient"] == 0
    assert "e-1" not in result["equation"]


def test_a_model_the_design_cannot_fit_is_refused_plainly(api):
    """A 9-run DSD in 3 factors cannot fit the 10-term quadratic model, whatever the data."""
    reply = analyse(api, {"design_type": "dsd", "factors": FACTORS, "n_center_points": 1}, model="quadratic")
    assert not reply["ok"]
    assert "cannot fit the quadratic model" in reply["error"]


def test_too_few_results_for_the_model_are_reported_in_the_page_s_words(api, held, simulated):
    """The runs may be the page's, typed in, and the runs still to do are named as the analysis names them."""
    responses = [None] * 8 + simulated["responses"][8:]
    assert api.error("api_analyze_runs", {**held, "responses": responses, "model": "quadratic"}) == (
        "The quadratic model has 10 coefficients, so it needs at least 10 completed runs, and 7 are complete. "
        "Still to run: 1, 2, 3, 4, 5, 6, 7, 8."
    )
