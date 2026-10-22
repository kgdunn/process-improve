"""Categorical factors of two to eight levels, each level any text: a catalyst A, B or C, or a lot 1, 2, 3 or 4.

A categorical factor says its type and lists its levels, so a label that reads as a number stays a label. The
full factorial and the Taguchi arrays hold a factor of three levels or more; the two-level designs refuse it.
The analysis fits such a factor as its contrasts, a column per level but the last, names each by its level,
and tests the factor as one source with a degree of freedom per level but one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

from tests.web._support import FACTORS, fill_workbook, held_of, run_sheet, sheets_of

CATALYST = {"name": "Catalyst", "type": "categorical", "levels": ["A", "B", "C"], "units": ""}
LOT = {"name": "Lot", "type": "categorical", "levels": ["1", "2", "3", "4"], "units": ""}
T = FACTORS[0]

#: How far each catalyst's response lies from the average catalyst's: A on it, B 3 above, C 3 below.
CATALYST_EFFECTS = {"A": 0.0, "B": 3.0, "C": -3.0}


def made(api, factors: list[dict], design_type: str = "full_factorial", **options: object) -> dict:
    """Make a design in these factors, with no center runs unless asked."""
    return api.ok("api_make_design", {"design_type": design_type, "factors": factors, "n_center_points": 0, **options})


def results(design: dict, noise: float = 0.0, seed: int = 1) -> list[float]:
    """Return results with the catalysts' effects and T's slope of 2 per coded unit, and noise if asked."""
    sheet = run_sheet(design)
    rng = np.random.default_rng(seed)
    return [
        50 + CATALYST_EFFECTS[c] + 2 * (t - 175) / 25 + noise * rng.normal()
        for c, t in zip(sheet["Catalyst"], sheet["T"], strict=True)
    ]


def test_a_full_factorial_runs_every_combination_of_the_levels(api):
    design = made(api, [CATALYST, T], n_replicates=2)
    sheet = run_sheet(design)
    assert design["n_runs"] == 12
    assert sorted(zip(sheet["Catalyst"], sheet["T"], strict=True)) == sorted(
        [(c, t) for c in "ABC" for t in (150, 200)] * 2
    )
    assert design["spec"]["factors"][0] == CATALYST, "the factor says its type and lists its levels"


def test_labels_that_read_as_numbers_stay_labels(api):
    design = made(api, [LOT, T])
    assert sorted(set(run_sheet(design)["Lot"])) == ["1", "2", "3", "4"], "the library reads them as 1.0, 2.0, ..."
    two = made(api, [{"name": "Lot", "type": "categorical", "levels": ["1", "2"]}], n_replicates=2)
    assert two["spec"]["factors"][0] == {
        "name": "Lot",
        "type": "categorical",
        "levels": ["1", "2"],
        "low": "1",
        "high": "2",
        "units": "",
    }, "two levels keep their low and high too, as every two-level design reads them"


def test_two_text_labels_are_stored_as_before(api):
    design = made(api, [{"name": "S", "type": "categorical", "levels": ["old", "new"]}, T])
    assert design["spec"]["factors"][0] == {"name": "S", "low": "old", "high": "new", "units": ""}


def test_a_taguchi_array_holds_a_factor_of_three_levels(api):
    design = made(api, [CATALYST, T], design_type="taguchi")
    assert design["facts"]["notes"][0].startswith("Taguchi L18 (2^1 3^7)")
    assert set(run_sheet(design)["Catalyst"]) == {"A", "B", "C"}


def test_runs_as_given_are_read_at_their_levels_a_number_as_its_label(api):
    runs = [["1"], [2], [3.0], ["4"], ["1"], ["2"]]
    design = api.ok("api_make_design", {"design_type": "given", "factors": [LOT], "runs": runs})
    assert run_sheet(design)["Lot"].tolist() == ["1", "2", "3", "4", "1", "2"]
    assert design["facts"]["notes"][1] == (
        "The runs compare Lot at 1, 2, 3 and 4: 2 runs at 1, 2 runs at 2, 1 run at 3 and 1 run at 4."
    )
    error = api.error("api_make_design", {"design_type": "given", "factors": [LOT], "runs": [["5"], ["1"]]})
    assert error == "Factor Lot: each run must be at 1, 2, 3 or 4, not '5'."


@pytest.fixture(scope="module")
def catalysts(api) -> tuple[dict, dict]:
    """Return a replicated full factorial in the catalyst and T, and its analysis with the interactions model."""
    design = made(api, [CATALYST, T], n_replicates=2)
    analysis = api.ok(
        "api_analyze_runs", {**held_of(design), "responses": results(design, noise=0.5), "model": "interactions"}
    )
    return design, analysis


def test_a_factor_of_several_levels_is_fitted_as_a_contrast_per_level_but_the_last(catalysts):
    _, analysis = catalysts
    labels = [c["label"] for c in analysis["coefficients"]]
    assert labels == ["Intercept", "Catalyst[A]", "Catalyst[B]", "T", "Catalyst[A]·T", "Catalyst[B]·T"]
    effects = {c["label"]: c["coefficient"] for c in analysis["coefficients"]}
    assert effects["Catalyst[A]"] == pytest.approx(0.0, abs=0.6), "A lies on the average catalyst"
    assert effects["Catalyst[B]"] == pytest.approx(3.0, abs=0.6)
    assert effects["T"] == pytest.approx(2.0, abs=0.3)
    assert analysis["equation"].startswith("y = ")
    assert "Catalyst[B]" in analysis["equation"]


def test_the_anova_tests_the_factor_as_one_source_of_a_degree_of_freedom_per_level_but_one(catalysts):
    _, analysis = catalysts
    rows = {a["source"]: a for a in analysis["anova"]}
    assert list(rows) == ["Catalyst", "T", "Catalyst:T", "Residual"]
    assert (rows["Catalyst"]["df"], rows["Catalyst:T"]["df"], rows["Residual"]["df"]) == (2, 2, 6)
    assert rows["Catalyst"]["p_value"] < 0.001
    assert rows["Catalyst:T"]["p_value"] > 0.05, "the catalysts do not change T's slope"


def test_the_grouped_sum_of_squares_is_what_leaving_the_factor_out_costs(api, catalysts):
    """A balanced design: the catalyst's sum of squares is the spread of its averages, times the runs at each."""
    design, analysis = catalysts
    sheet = run_sheet(design)
    y = np.array(results(design, noise=0.5))
    means = {c: y[sheet["Catalyst"] == c].mean() for c in "ABC"}
    expected = 4 * sum((m - y.mean()) ** 2 for m in means.values())
    catalyst = next(a for a in analysis["anova"] if a["source"] == "Catalyst")
    assert catalyst["sum_sq"] == pytest.approx(expected)
    assert catalyst["mean_sq"] == pytest.approx(expected / 2), "its mean square is over its two degrees of freedom"
    residual = next(a for a in analysis["anova"] if a["source"] == "Residual")
    assert catalyst["F"] == pytest.approx(catalyst["mean_sq"] / residual["mean_sq"])


def test_the_anova_is_of_type_ii_as_statsmodels_gives_it_for_unbalanced_runs(api):
    """Three runs left out, unevenly: each source's sum of squares is statsmodels' Type II, the catalyst's grouped."""
    design = made(api, [CATALYST, T], n_replicates=3)
    y = results(design, noise=0.5)
    for i in (0, 4, 7):
        y[i] = None
    analysis = api.ok("api_analyze_runs", {**held_of(design), "responses": y, "model": "interactions"})
    sheet = run_sheet(design)
    frame = pd.DataFrame({"Catalyst": sheet["Catalyst"], "T": (sheet["T"] - 175) / 25, "y": y}).dropna()
    table = anova_lm(ols("y ~ C(Catalyst, Sum) * T", frame).fit(), typ=2)
    rows = {a["source"]: a for a in analysis["anova"]}
    for source, row in [("Catalyst", "C(Catalyst, Sum)"), ("T", "T"), ("Catalyst:T", "C(Catalyst, Sum):T")]:
        assert rows[source]["sum_sq"] == pytest.approx(table.loc[row, "sum_sq"]), source
        assert rows[source]["F"] == pytest.approx(table.loc[row, "F"]), source


SUPPLIER = {"name": "Supplier", "type": "categorical", "levels": ["1", "2", "3"], "units": ""}
BATCH = {"name": "Batch", "type": "categorical", "levels": ["a", "b", "c", "d"], "units": ""}
NESTED = {"terms": [["Supplier"], ["Supplier", "Batch"]]}


@pytest.fixture(scope="module")
def batches(api) -> tuple[dict, list[float]]:
    """Batches a to d of each of three suppliers, twice each, a supplier's batches its own: Batch nested in Supplier."""
    design = made(api, [SUPPLIER, BATCH], n_replicates=2)
    sheet = run_sheet(design)
    rng = np.random.default_rng(3)
    return design, (sheet["Supplier"].astype(int) + rng.normal(size=len(sheet))).round(3).tolist()


def test_a_factor_nested_in_another_is_fitted_within_each_of_its_levels_as_statsmodels_fits_it(api, batches):
    """Supplier + Supplier:Batch: Batch's effect within each supplier, 3 x 3 columns, as patsy codes it."""
    design, y = batches
    analysis = api.ok("api_analyze_runs", {**held_of(design), "responses": y, "model": NESTED})
    sheet = run_sheet(design)
    frame = pd.DataFrame({"S": sheet["Supplier"].astype(str), "B": sheet["Batch"], "y": y})
    table = anova_lm(ols("y ~ S + S:B", frame).fit(), typ=2)
    rows = {a["source"]: a for a in analysis["anova"]}
    assert (rows["Supplier"]["df"], rows["Supplier:Batch"]["df"], rows["Residual"]["df"]) == (2, 9, 12)
    for source, row in [("Supplier", "S"), ("Supplier:Batch", "S:B"), ("Residual", "Residual")]:
        assert rows[source]["sum_sq"] == pytest.approx(table.loc[row, "sum_sq"]), source
    labels = [c["label"] for c in analysis["coefficients"]]
    assert "Supplier[3]·Batch[a]" in labels, "batch a's effect within the third supplier"


def test_a_term_of_two_level_factors_is_their_product_with_or_without_the_terms_below_it(api):
    """A two-level design's three-factor effect: T·S·P without S·P is the product of the -1/+1 columns, not S nested."""
    supplier = {"name": "S", "low": "A", "high": "B", "units": ""}
    design = made(api, [T, supplier, FACTORS[1]])
    y = [float(i % 3) for i in range(len(run_sheet(design)))]
    model = {"terms": [["T"], ["S"], ["P"], ["T", "S"], ["T", "S", "P"]]}
    analysis = api.ok("api_analyze_runs", {**held_of(design), "responses": y, "model": model})
    assert analysis["formula"].endswith("T:S:P"), "one column, the effect a two-level design estimates"


def test_a_nested_model_s_best_settings_hold_the_outer_factor_s_indicators(api, batches):
    design, y = batches
    held = held_of(design)
    held["spec"]["response_info"] = {"goal": "maximize"}
    best = api.ok("api_best_settings", {**held, "responses": y, "model": NESTED})
    kept = set(best["search"]["held"])
    assert {"Supplier_I1", "Supplier_I2", "Supplier_I3", "Supplier_L1", "Supplier_L2"} <= kept
    plain = api.ok("api_best_settings", {**held, "responses": y, "model": "main_effects"})
    assert not any("_I" in name for name in plain["search"]["held"]), "a model with no nesting holds no indicators"


def test_a_model_of_chosen_terms_takes_the_factor_whole(api, catalysts):
    design, _ = catalysts
    chosen = {"terms": [["Catalyst"], ["T"]]}
    analysis = api.ok("api_analyze_runs", {**held_of(design), "responses": results(design, 0.5), "model": chosen})
    assert [c["label"] for c in analysis["coefficients"]] == ["Intercept", "Catalyst[A]", "Catalyst[B]", "T"]


def test_the_workbook_keeps_labels_that_read_as_numbers(api):
    design = made(api, [LOT, T], n_replicates=2)
    sheet = run_sheet(design)
    y = [10.0 + int(lot) + (t - 175) / 25 for lot, t in zip(sheet["Lot"], sheet["T"], strict=True)]
    workbook = fill_workbook(design["xlsx_base64"], lambda _: y)
    runs = sheets_of(workbook)["Runs"]
    assert sorted(set(runs["Lot"].astype(str))) == ["1", "2", "3", "4"]
    analysis = api.ok("api_analyze", {"xlsx_base64": workbook, "model": "main_effects"})
    assert [a["source"] for a in analysis["anova"]] == ["Lot", "T", "Residual"]


# --------------------------------------------------------------------------- one factor of several levels compared
#: Box, Hunter and Hunter's blood coagulation times (their Table 6.1) for four diets, of unequal runs.
DIETS = {
    "A": [62, 60, 63, 59],
    "B": [63, 67, 71, 64, 65, 66],
    "C": [68, 66, 71, 67, 68, 68],
    "D": [56, 62, 60, 61, 63, 64, 63, 59],
}
DIET = {"name": "Diet", "type": "categorical", "levels": list(DIETS), "units": ""}


def compared(api, factor: dict, runs: list[str], y: list[float], control: str | None = None) -> dict:
    """Return the comparison of one factor's levels, its runs as given."""
    design = api.ok("api_make_design", {"design_type": "given", "factors": [factor], "runs": [[r] for r in runs]})
    request = {**held_of(design), "responses": y, **({"control": control} if control else {})}
    return api.ok("api_analyze_runs", request)["comparison"]


@pytest.fixture(scope="module")
def diets(api) -> dict:
    runs = [diet for diet, times in DIETS.items() for _ in times]
    return compared(api, DIET, runs, [t for times in DIETS.values() for t in times])


def test_the_levels_are_tested_together_by_the_one_way_anova(diets):
    test = diets["several"]["test"]
    assert (test["df_between"], test["df_within"]) == (3, 20)
    assert test["f_value"] == pytest.approx(13.57, abs=0.01), "Box, Hunter and Hunter's F"
    assert test["p_value"] < 0.001
    assert diets["difference"] is None


def test_tukey_s_intervals_are_every_pair_s_as_scipy_gives_them(diets):
    from scipy import stats

    expected = stats.tukey_hsd(*DIETS.values())
    interval = expected.confidence_interval(0.95)
    names = list(DIETS)
    for pair in diets["several"]["pairs"]:
        i, j = names.index(pair["from"]), names.index(pair["to"])
        assert pair["estimate"] == pytest.approx(-expected.statistic[i, j]), "the later level less the earlier"
        assert pair["p_value"] == pytest.approx(expected.pvalue[i, j], abs=1e-4)
        assert [pair["ci_low"], pair["ci_high"]] == pytest.approx([-interval.high[i, j], -interval.low[i, j]], abs=1e-3)
    assert len(diets["several"]["pairs"]) == 6


def test_dunnett_s_intervals_are_scipy_s_for_levels_of_equal_runs(api):
    from scipy import stats

    rng = np.random.default_rng(4)
    groups = {level: list(50 + i + rng.normal(0, 1, 4)) for i, level in enumerate("ABCD")}
    factor = {"name": "Diet", "type": "categorical", "levels": list("ABCD")}
    runs = [level for level, values in groups.items() for _ in values]
    found = compared(api, factor, runs, [v for values in groups.values() for v in values], control="B")["several"]
    assert found["control"] == "B"
    others = [groups[level] for level in "ACD"]
    # scipy integrates by randomized quasi-Monte Carlo: seeded, its interval is the same on every run.
    expected = stats.dunnett(*others, control=groups["B"], random_state=1).confidence_interval(0.95)
    assert [v["level"] for v in found["versus"]] == ["A", "C", "D"]
    for v, low, high in zip(found["versus"], expected.low, expected.high, strict=True):
        assert [v["ci_low"], v["ci_high"]] == pytest.approx([low, high], abs=2e-3)
        assert v["differs"] == (low > 0 or high < 0)


def test_dunnett_s_intervals_for_levels_of_unequal_runs_are_scipy_s(diets):
    """Box, Hunter and Hunter's diets, 4, 6, 6 and 8 animals: each comparison with diet A as scipy gives it."""
    from scipy import stats

    others = [DIETS[d] for d in "BCD"]
    expected = stats.dunnett(*others, control=DIETS["A"], random_state=1).confidence_interval(0.95)
    for v, low, high in zip(diets["several"]["versus"], expected.low, expected.high, strict=True):
        assert [v["ci_low"], v["ci_high"]] == pytest.approx([low, high], abs=5e-3), v["level"]


def test_a_control_of_more_runs_lowers_the_correlation_of_the_comparisons(api):
    """Eight runs at the control, four at each other level: the comparisons correlate by 4 / (4 + 8), not 1/2."""
    from scipy import stats

    rng = np.random.default_rng(5)
    groups = {"A": list(50 + rng.normal(0, 1, 8)), **{level: list(52 + rng.normal(0, 1, 4)) for level in "BCD"}}
    factor = {"name": "Diet", "type": "categorical", "levels": list("ABCD")}
    runs = [level for level, values in groups.items() for _ in values]
    found = compared(api, factor, runs, [v for values in groups.values() for v in values])["several"]
    expected = stats.dunnett(*(groups[x] for x in "BCD"), control=groups["A"], random_state=1).confidence_interval()
    for v, low, high in zip(found["versus"], expected.low, expected.high, strict=True):
        assert [v["ci_low"], v["ci_high"]] == pytest.approx([low, high], abs=2e-3), v["level"]


def test_the_control_is_the_first_level_unless_another_is_named(diets, api):
    assert diets["several"]["control"] == "A"
    runs = [diet for diet, times in DIETS.items() for _ in times]
    y = [t for times in DIETS.values() for t in times]
    assert compared(api, DIET, runs, y, control="nobody")["several"]["control"] == "A"


def test_labels_that_read_as_numbers_are_compared_as_levels(api):
    lots = {"name": "Lot", "type": "categorical", "levels": ["3", "1", "2"]}
    found = compared(api, lots, ["1", "1", "2", "2", "3", "3"], [10, 11, 12, 13, 20, 21])
    assert [level["level"] for level in found["levels"]] == ["3", "1", "2"], "in the order listed, not as numbers"
    assert found["several"]["control"] == "3"


# --------------------------------------------------------------------------- simulated, and the best settings
def test_a_simulated_process_spreads_a_factor_s_effect_over_its_levels(api):
    """The reactor moves the response by -c at the first catalyst and +c at the last: over the contrasts, -c and 0."""
    design = made(api, [CATALYST, T], n_replicates=2)
    held = held_of(design)
    simulated = api.ok("api_simulate", {**held, "scenario": "main_effects", "sigma": 0.5, "seed": 2})
    truth = {t["term"]: t["coefficient"] for t in simulated["truth"]}
    assert set(truth) >= {"Catalyst_L1", "Catalyst_L2", "T"}
    assert truth["Catalyst_L2"] == 0, "the middle level lies on the average"
    assert "Catalyst[A]" in simulated["equation"]
    analysis = api.ok("api_analyze_runs", {**held, "responses": simulated["responses"], "truth": simulated["truth"]})
    true = {c["label"]: c["true_value"] for c in analysis["coefficients"]}
    assert true["Catalyst[A]"] == truth["Catalyst_L1"]


def test_the_best_settings_hold_each_level_in_turn(api):
    design = made(api, [CATALYST, T, FACTORS[1]])
    held = held_of(design)
    held["spec"]["response_info"] = {"goal": "maximize"}
    sheet = run_sheet(design)
    effects = {"A": 0.0, "B": 4.0, "C": -2.0}
    y = [
        50 + effects[c] + 2 * (t - 175) / 25 - (p - 2)
        for c, t, p in zip(sheet["Catalyst"], sheet["T"], sheet["P"], strict=True)
    ]
    best = api.ok("api_best_settings", {**held, "responses": y, "model": "main_effects"})
    assert {s["name"]: s["value"] for s in best["settings"]} == {"Catalyst": "B", "T": 200, "P": 1}
    assert best["predicted"] == pytest.approx(57.0)
    plots = api.ok("api_plots", {**held, "responses": y, "model": "main_effects", "best": best["coded"]})
    assert plots["contour_factors"] == ["T", "P"]
    assert plots["held"] == [{"name": "Catalyst", "value": "B"}], "the surface is drawn at the best catalyst"


def test_renaming_a_factor_keeps_its_levels_and_changing_a_level_needs_a_new_run_sheet(api):
    design = made(api, [CATALYST, T], n_replicates=2)
    held = held_of(design)
    renamed = api.ok("api_relabel", {"design": held, "factors": [{**CATALYST, "name": "Cat"}, T]})
    assert {"Cat", "T"} <= set(renamed["columns"])
    assert renamed["spec"]["factors"][0]["levels"] == ["A", "B", "C"]
    for levels in (["A", "B", "D"], ["C", "B", "A"]):
        message = api.error("api_relabel", {"design": held, "factors": [{**CATALYST, "levels": levels}, T]})
        assert "making the run sheet again" in message


# --------------------------------------------------------------------------- the designs Design lists
def candidates(api, factors: list[dict], model: str = "main_effects") -> dict:
    return api.ok("api_candidates", {"factors": factors, "model": model, "response": "y", "random_state": 7})


def test_one_factor_of_three_levels_is_sized_for_the_power_of_its_f_test(api):
    found = candidates(api, [CATALYST])
    plan = found["levels"]
    assert plan["n"] == 6
    assert [row["runs"] for row in plan["rows"][:3]] == [6, 9, 12], "three runs a row, one at each level"
    [design] = found["made"]
    assert design["design"]["n_runs"] == 18
    [sized] = found["result"]["designs"]
    assert found["result"]["values"][sized]["power_main"] == pytest.approx(
        next(row["power"] for row in plan["rows"] if row["n"] == 6)
    ), "the plan's power is the F test's the design list scores"


def test_the_two_level_designs_are_left_out_saying_why(api):
    found = candidates(api, [CATALYST, T, FACTORS[1]], model="interactions")
    reasons = {d["design_type"]: d["reason"] for d in found["left_out"]}
    assert reasons["fractional_factorial"] == reasons["plackett_burman"] == "levels"
    assert [m["design_type"] for m in found["made"]] == ["full_factorial", "taguchi"]


def test_designs_with_a_factor_of_several_levels_are_scored_on_its_terms_together(api):
    found = candidates(api, [CATALYST, T, FACTORS[1]], model="interactions")
    values = found["result"]["values"]
    full = next(v for label, v in values.items() if label.startswith("Full factorial"))
    assert full["fits"] is True
    assert 0 < full["power_interaction"] <= full["power_main"] < 1
    assert found["result"]["terms"][:3] == ["Intercept", "Catalyst_L1", "Catalyst_L2"]
