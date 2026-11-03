"""Optimal inversion: the inputs a PLS or TPLS model vouches for, under rules, with each rule's shadow price."""

from __future__ import annotations

import dataclasses
import itertools
import json
import pathlib
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import OptimizeResult
from sklearn.utils import Bunch

from process_improve.multivariate import (
    PLS,
    TPLS,
    InversionRules,
    LinearRule,
    MCUVScaler,
    affine_model,
    optimal_inversion,
)
from process_improve.multivariate import _optimal_inversion as oi
from process_improve.multivariate._tpls import DataFrameDict

# The tests written before 2.4.0 search inputs without both bounds, which warns until 3.0 makes it an error. The tests
# of that warning record it with pytest.warns, which this filter does not reach.
pytestmark = pytest.mark.filterwarnings("ignore:.*no lower or no upper bound:FutureWarning")

DATA = pathlib.Path(__file__).parents[1] / "src" / "process_improve" / "datasets" / "multivariate"
GROUPS = [f"Group{g}" for g in range(1, 6)]
CHEESE_COSTS = {"Acetic": 1.0, "H2S": 3.0, "Lactic": 8.0}


# ---------------------------------------------------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cheese() -> PLS:
    """Fit the cheddar-cheese PLS of the book's inversion section: two components, the first four cheeses held out."""
    data = pd.read_csv(DATA / "cheddar-cheese.csv")
    return PLS(n_components=2).fit(data[["Acetic", "H2S", "Lactic"]].iloc[4:], data[["Taste"]].iloc[4:])


@pytest.fixture(scope="module")
def pyphi() -> tuple[TPLS, dict]:
    """Fit the bundled TPLS example (Garcia-Munoz's pyphi data): 105 batches blended from five materials' lots."""
    d = {g: pd.read_csv(DATA / "tpls-pyphi" / f"properties_{g}.csv", index_col=0) for g in GROUPS}
    f = {g: pd.read_csv(DATA / "tpls-pyphi" / f"formulas_{g}.csv", index_col=0) for g in GROUPS}
    z = {"Conditions": pd.read_csv(DATA / "tpls-pyphi" / "process_conditions.csv", index_col=0)}
    y = {"Quality": pd.read_csv(DATA / "tpls-pyphi" / "quality_indicators.csv", index_col=0)}
    model = TPLS(n_components=4, d_matrix=d).fit(DataFrameDict({"F": f, "Z": z, "Y": y}))
    return model, {"F": f, "Z": z, "Y": y}


MIXTURE = ["a", "b", "c", "d"]


@pytest.fixture(scope="module")
def mixture() -> PLS:
    """
    Fit a four-ingredient mixture whose quality is linear in the fractions: a PLS of two components on 40 blends.

    Its optimum under the counting rules can be checked by enumerating every choice of ingredients.
    """
    rng = np.random.default_rng(7)
    fractions = rng.dirichlet([1.0, 1.0, 1.0, 1.0], size=40)
    quality = fractions @ np.array([10.0, 4.0, 7.0, 2.0]) + rng.normal(0, 0.2, 40)
    return PLS(n_components=2).fit(pd.DataFrame(fractions, columns=MIXTURE), pd.DataFrame({"q": quality}))


def _mixture_rules(**changes: object) -> InversionRules:
    rules = InversionRules(
        costs={"a": 9.0, "b": 2.0, "c": 4.0, "d": 1.0},
        windows={"q": (6.0, None)},
        bounds=dict.fromkeys(MIXTURE, (0.0, 1.0)),
        rules=(LinearRule("mixture", dict.fromkeys(MIXTURE, 1.0), lower=1.0, upper=1.0),),
        conf_level=0.99,
    )
    return dataclasses.replace(rules, **changes)


# ---------------------------------------------------------------------------------------------------------------------
# diagnose's residuals, and the affine maps read from them
# ---------------------------------------------------------------------------------------------------------------------
def test_pls_diagnose_returns_the_residuals_whose_length_is_spe(cheese: PLS) -> None:
    """Each row's residual is what the components leave of it in the scaled space; SPE is its length."""
    rows = pd.DataFrame({"Acetic": [5.0, 6.5], "H2S": [4.0, 8.0], "Lactic": [1.2, 1.9]})

    out = cheese.diagnose(rows)

    assert list(out.residuals.columns) == ["Acetic", "H2S", "Lactic"]
    np.testing.assert_allclose(np.linalg.norm(out.residuals, axis=1), out.spe, rtol=1e-12)
    scaled = cheese._scaled_rows(rows).to_numpy()
    np.testing.assert_allclose(out.residuals, scaled - out.scores.to_numpy() @ cheese.x_loadings_.to_numpy().T)


@pytest.mark.dataset
def test_tpls_diagnose_returns_each_blocks_residuals_whose_length_is_spe(pyphi: tuple[TPLS, dict]) -> None:
    """For every F and Z group, the length of each row's residual is that group's last SPE."""
    model, data = pyphi
    rows = DataFrameDict(
        {"F": {g: data["F"][g].iloc[:5] for g in GROUPS}, "Z": {"Conditions": data["Z"]["Conditions"].iloc[:5]}}
    )

    out = model.diagnose(rows)

    for block, groups in out.residuals.items():
        for group, residual in groups.items():
            np.testing.assert_allclose(np.linalg.norm(residual, axis=1), out.spe[block][group].iloc[:, -1], rtol=1e-10)


def test_the_affine_maps_reproduce_pls_at_new_rows(cheese: PLS) -> None:
    """Predictions, T² and SPE from the maps equal diagnose's at rows the maps were not read at."""
    affine = affine_model(cheese)
    rows = np.array([[4.8, 3.9, 1.1], [6.1, 7.5, 1.8], [5.0, 9.0, 1.3]])

    out = cheese.diagnose(pd.DataFrame(rows, columns=affine.names))

    np.testing.assert_allclose([affine.predict(r) for r in rows], out.y_hat, rtol=1e-9)
    np.testing.assert_allclose([affine.hotellings_t2(r) for r in rows], out.hotellings_t2, rtol=1e-9)
    np.testing.assert_allclose([affine.spe(r)[0] for r in rows], out.spe, rtol=1e-9)


def test_only_a_model_that_can_describe_itself_is_read() -> None:
    """A model without an inversion probe (here, a scaler) is refused by name."""
    with pytest.raises(TypeError, match="reads PLS and TPLS models, not MCUVScaler"):
        affine_model(MCUVScaler())


def test_maps_that_disagree_with_the_model_are_refused(cheese: PLS, monkeypatch: pytest.MonkeyPatch) -> None:
    """A model that is not affine in its inputs (here, one that squares them) is not read as maps."""
    probe = PLS._inversion_probe

    def curved(model: PLS, conf_level: float) -> oi.Probe:
        read = probe(model, conf_level)
        return dataclasses.replace(read, diagnose=lambda rows: read.diagnose(rows**2))

    monkeypatch.setattr(PLS, "_inversion_probe", curved)

    with pytest.raises(ValueError, match="could not be read as affine maps"):
        affine_model(cheese)


# ---------------------------------------------------------------------------------------------------------------------
# PLS: targets, costs, windows, limits
# ---------------------------------------------------------------------------------------------------------------------
def test_a_target_is_met_by_the_most_typical_of_its_null_space(cheese: PLS) -> None:
    """Every point of the null space hits the target; the answer is the one of least T², no worse than any of them."""
    result = cheese.optimal_inversion(InversionRules(targets={"Taste": 20.9}))
    walked = [cheese.invert(20.9, null_space_coordinates=[s]).hotellings_t2 for s in np.linspace(-3, 3, 601)]

    assert result.feasible
    assert result.proven_optimal
    assert result.y_hat["Taste"] == pytest.approx(20.9, abs=1e-4)
    assert result.hotellings_t2 <= min(walked) + 1e-6
    assert result.hotellings_t2 <= cheese.invert(20.9).hotellings_t2  # the minimum-norm point is not the least T²


def test_the_cheapest_inputs_beat_every_point_the_window_and_limits_allow(cheese: PLS) -> None:
    """Sweep each acceptable taste's null space, as the book's specification region does: none is cheaper."""
    rules = InversionRules(costs=CHEESE_COSTS, windows={"Taste": (20.0, 30.0)})
    t2_limit, spe_limit = cheese.hotellings_t2_limit(0.95), cheese.spe_limit(conf_level=0.95)
    steps = np.linspace(-3, 3, 241)[:, None]
    scaled = []
    for target in np.linspace(20, 30, 41):  # every point of each acceptable taste's null space, as invert builds it
        found = cheese.invert(target)
        scores = found.scores.to_numpy() + steps * found.null_space_basis.to_numpy().ravel()
        scaled.append(scores @ cheese.x_loadings_.to_numpy().T)
    points = cheese._x_scaler.inverse_transform(np.vstack(scaled))
    points.columns = ["Acetic", "H2S", "Lactic"]
    check = cheese.diagnose(points)
    inside = (check.hotellings_t2 <= t2_limit).to_numpy() & (check.spe <= spe_limit).to_numpy()
    swept = points[inside] @ pd.Series(CHEESE_COSTS)[points.columns]

    result = optimal_inversion(cheese, rules)

    assert result.feasible
    assert result.proven_optimal
    assert result.objective <= min(swept) + 1e-6
    check = cheese.diagnose(result.x_new.to_frame().T)
    assert float(check.y_hat.iloc[0, 0]) == pytest.approx(result.y_hat["Taste"])
    assert float(check.hotellings_t2.iloc[0]) <= t2_limit * (1 + 1e-4)
    assert float(check.spe.iloc[0]) <= spe_limit * (1 + 1e-4)
    assert set(result.constraints.loc[result.constraints.binding != "", "constraint"]) == {"Taste", "T²", "SPE X"}


def test_each_shadow_price_forecasts_the_change_on_solving_again(cheese: PLS) -> None:
    """Move each binding rule 1% the way that relaxes it: the objective moves as its shadow price says."""
    rules = InversionRules(costs=CHEESE_COSTS, windows={"Taste": (20.0, 30.0)})

    report = cheese.inversion_sensitivity(rules)

    assert set(report.checked["constraint"]) == {"Taste", "T²", "SPE X"}
    np.testing.assert_allclose(report.checked["actual_change"], report.checked["forecast_change"], rtol=0.02)
    assert (report.checked["actual_change"] < 0).all()  # relaxing a binding rule can only save


def test_the_cost_falls_as_the_t2_limit_widens_and_the_answer_holds_its_shape_over_each_price_range(
    cheese: PLS,
) -> None:
    """The cost front is falling; within each input's price range, the same rules bind."""
    rules = InversionRules(costs=CHEESE_COSTS, windows={"Taste": (20.0, 30.0)})

    report = oi.inversion_sensitivity(cheese, rules, front=("T²", [2.0, 4.0, 6.0, 8.0]), prices=["Lactic"])
    taste = oi.inversion_sensitivity(cheese, rules, front=("Taste", [20.0, 22.0, 24.0, 31.0])).front
    spe = oi.inversion_sensitivity(cheese, rules, front=("SPE X", [1.0, 1.5])).front

    assert report.front["feasible"].all()
    assert (np.diff(report.front["objective"]) < 0).all()
    assert (np.diff(taste["objective"].iloc[:3]) > 0).all()  # the binding lower side moves: tastier costs more
    assert not taste["feasible"].iloc[3]  # past the window's upper side, 30, there is nothing to meet
    assert spe["objective"].iloc[1] < spe["objective"].iloc[0]
    lactic = report.prices.iloc[0]
    assert -1.0 <= lactic["falls_by"] <= 0.0 <= lactic["rises_by"] <= 1.0
    assert lactic["used"]


def test_a_safety_limit_binds_is_priced_and_traced(cheese: PLS) -> None:
    """A legal limit on a weighted sum of the inputs is a linear rule: it binds, has a price, and can be traced."""
    rules = InversionRules(
        costs={"Acetic": 1.0, "H2S": -2.0, "Lactic": 1.0},  # H2S earns: only the limit holds it back
        windows={"Taste": (20.0, 30.0)},
        rules=(LinearRule("sulfide limit", {"H2S": 1.0, "Lactic": 0.5}, upper=6.5),),
    )

    report = oi.inversion_sensitivity(cheese, rules, front=("sulfide limit", [6.0, 6.5, 7.0]))
    table = report.inversion.constraints.set_index("constraint")

    assert table.loc["sulfide limit", "binding"] == "upper"
    assert table.loc["sulfide limit", "shadow_price"] < 0
    assert (np.diff(report.front["objective"]) < 0).all()
    row = report.checked.set_index("constraint").loc["sulfide limit"]
    assert row["actual_change"] == pytest.approx(row["forecast_change"], rel=0.02)


def test_no_answer_says_whether_the_linear_rules_or_the_limits_stand_in_the_way(cheese: PLS) -> None:
    """Out of reach within the inputs' bounds: proven at once. Reachable only far from the data: the limits named."""
    bounded = InversionRules(targets={"Taste": 25.0}, windows={"Taste": (1000.0, None)},
                             bounds={"Acetic": (4, 7), "H2S": (3, 10), "Lactic": (0.8, 2.1)})  # fmt: skip
    unbounded = InversionRules(targets={"Taste": 25.0}, windows={"Taste": (80.0, None)})

    out_of_reach, far = cheese.optimal_inversion(bounded), cheese.optimal_inversion(unbounded)

    assert not out_of_reach.feasible
    assert out_of_reach.proven_optimal  # proven: no answer exists
    assert "No inputs within their bounds meet the linear rules" in out_of_reach.message
    assert not far.feasible
    assert far.proven_optimal
    assert "only far from the data the model was built from" in far.message
    assert "T²" in far.message


def test_a_fixed_input_stays_put_and_its_price_is_reported(cheese: PLS) -> None:
    """Holding lactic acid at 1.5: the answer keeps it there, and says what freeing it would be worth."""
    rules = InversionRules(costs=CHEESE_COSTS, windows={"Taste": (20.0, 30.0)}, fixed={"Lactic": 1.5})

    result = cheese.optimal_inversion(rules)

    assert result.x_new["Lactic"] == 1.5
    fixed = result.constraints.set_index("constraint").loc["Lactic"]
    assert (fixed["kind"], fixed["binding"]) == ("fixed", "equal")
    assert np.isfinite(fixed["shadow_price"])


def test_rules_the_model_cannot_read_are_refused_with_every_reason(cheese: PLS) -> None:
    """Unknown names, a window upside down, no objective or two, a bad count: each named."""
    bad = InversionRules(
        costs={"Acetic": 1.0, "Unknown": 2.0}, targets={"Taste": 1.0}, windows={"Missing": (1, 2), "Taste": (5, 1)},
        bounds={"Acetic": 5},
        max_count={"nowhere": 0}, spe_limits={"Y": 1.0},
    )  # fmt: skip

    with pytest.raises(ValueError, match="cannot be used") as refused:
        cheese.optimal_inversion(bad)

    message = str(refused.value)
    for reason in ("'Missing' is not a quality", "'Unknown' is not an input", "'nowhere' is not a group",
                   "Taste: a window is", "give costs or targets", "Acetic: bounds are (low, high)",
                   "'Y' is not a residual block", "allow at least one input"):  # fmt: skip
        assert reason in message


def test_with_nothing_to_pay_nothing_binds_and_the_answer_is_the_model_centre(cheese: PLS) -> None:
    """A cost of zero and no window: the search stays at the centre, where no rule binds and nothing has a price."""
    result = cheese.optimal_inversion(InversionRules(costs={"Acetic": 0.0}))

    assert result.feasible
    assert (result.constraints["binding"] == "").all()
    assert result.kkt_residual == 0.0
    assert result.hotellings_t2 == pytest.approx(0.0, abs=1e-12)


def test_a_first_try_that_stops_short_starts_again_from_the_least_excess(
    cheese: PLS, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the first search from the centre stops outside a limit, the least-excess point is the next start."""
    rules = InversionRules(costs=CHEESE_COSTS, windows={"Taste": (20.0, 30.0)})
    expected = cheese.optimal_inversion(rules).objective
    monkeypatch.setattr(oi, "_FIRST_TRY_ITERATIONS", 1)

    assert cheese.optimal_inversion(rules).objective == pytest.approx(expected, rel=1e-6)


def test_a_restart_that_finds_nothing_better_keeps_the_point_in_hand(
    cheese: PLS, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Away from the optimum the search starts again; a restart that ends outside the rules changes nothing."""
    problem = oi._Problem(affine_model(cheese), InversionRules(costs=CHEESE_COSTS, windows={"Taste": (20.0, 30.0)}))
    bounds, rows = problem.continuous_bounds(None), problem.core_rows()
    objective = problem.objective[problem.keep()]
    point = problem.start(bounds)  # the model's centre: not the optimum
    search = oi._Search(lambda w: objective @ w, lambda _w: objective, point, bounds, [], problem.scale(bounds))
    monkeypatch.setattr(oi, "_slsqp", lambda s, _maxiter: s.start + 1e3)

    assert oi._kkt_residual(problem, point, bounds) > oi._KKT_TOLERANCE
    assert oi._settled(problem, search, point, rows) is point


def test_a_search_that_never_settles_says_so(cheese: PLS, monkeypatch: pytest.MonkeyPatch) -> None:
    """Should no point the search ends at pass the final check, the answer says the search did not settle."""
    monkeypatch.setattr(oi, "_meets", lambda *_: False)

    result = cheese.optimal_inversion(InversionRules(costs=CHEESE_COSTS, windows={"Taste": (20.0, 30.0)}))

    assert not result.feasible
    assert not result.proven_optimal  # a search that did not settle proves nothing
    assert "did not settle" in result.message


def test_the_most_typical_step_keeps_the_answer_when_it_cannot_improve_on_it(
    cheese: PLS, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tie-break that ends outside the rules leaves the best answer found as it was."""
    problem = oi._Problem(affine_model(cheese), InversionRules(targets={"Taste": 20.9}))
    solution = oi._solve(problem)
    monkeypatch.setattr(oi, "_slsqp", lambda search, _maxiter: search.start + 100.0)

    assert oi._most_typical(problem, solution) is solution


def test_sensitivity_of_rules_with_no_answer_is_empty(cheese: PLS) -> None:
    """No answer, nothing to be sensitive: the report holds the reason and empty tables."""
    report = cheese.inversion_sensitivity(InversionRules(targets={"Taste": 25.0}, windows={"Taste": (80.0, None)}))

    assert not report.inversion.feasible
    assert report.checked.empty
    assert report.counts.empty


def test_a_model_read_at_one_confidence_is_not_used_at_another(cheese: PLS) -> None:
    """The limits are read with the maps: rules asking for another confidence level are refused."""
    affine = affine_model(cheese, conf_level=0.95)
    rules = InversionRules(costs=CHEESE_COSTS, windows={"Taste": (20.0, 30.0)}, conf_level=0.99)

    with pytest.raises(ValueError, match=r"read at conf_level 0\.95, the rules ask for 0\.99"):
        optimal_inversion(affine, rules)


def test_maps_stored_as_json_search_as_the_model_does(cheese: PLS) -> None:
    """Store the maps, read them back, and search without the model: the same answer, and the same sensitivity."""
    rules = InversionRules(costs=CHEESE_COSTS, windows={"Taste": (20.0, 30.0)})
    stored = json.loads(json.dumps(affine_model(cheese).to_dict()))

    affine = oi.AffineModel.from_dict(stored)

    assert optimal_inversion(affine, rules).objective == pytest.approx(cheese.optimal_inversion(rules).objective)
    assert set(oi.inversion_sensitivity(affine, rules).checked["constraint"]) == {"Taste", "T²", "SPE X"}


def test_the_methods_are_the_functions_and_the_model_still_pickles(cheese: PLS) -> None:
    """``pls.optimal_inversion`` forwards to the function; adding it does not stop the model pickling."""
    rules = InversionRules(costs=CHEESE_COSTS, windows={"Taste": (20.0, 30.0)})

    again = pickle.loads(pickle.dumps(cheese))  # noqa: S301 - our own object

    assert again.optimal_inversion(rules).objective == pytest.approx(optimal_inversion(cheese, rules).objective)


# ---------------------------------------------------------------------------------------------------------------------
# Counting rules: the number of ingredients
# ---------------------------------------------------------------------------------------------------------------------
def _every_choice(model: PLS, rules: InversionRules, most: int) -> dict[frozenset[str], float]:
    """Solve every choice of at most ``most`` ingredients, each at 10% or more and the others at 0: its best cost."""
    costs = {}
    for size in range(1, most + 1):
        for chosen in itertools.combinations(MIXTURE, size):
            bounds = {n: ((0.1, 1.0) if n in chosen else (0.0, 0.0)) for n in MIXTURE}
            found = optimal_inversion(model, dataclasses.replace(rules, bounds=bounds, max_count={}, min_amount={}))
            if found.feasible:
                costs[frozenset(chosen)] = found.objective
    return costs


def _cheapest_by_enumeration(model: PLS, rules: InversionRules, most: int) -> float:
    """Return the best of every choice of at most ``most`` ingredients: the optimum."""
    return min(_every_choice(model, rules, most).values(), default=np.inf)


def test_at_most_two_ingredients_is_the_best_of_every_choice_of_two(mixture: PLS) -> None:
    """Outer approximation's answer equals the best over every choice of one or two ingredients, each at 10%+."""
    rules = _mixture_rules(groups={"all": MIXTURE}, max_count={"all": 2}, min_amount={"all": 0.1})

    result = mixture.optimal_inversion(rules)

    assert result.feasible
    assert result.proven_optimal
    assert len(result.used["all"]) <= 2
    assert all(result.x_new[n] >= 0.1 - 1e-6 for n in result.used["all"])
    assert result.objective == pytest.approx(_cheapest_by_enumeration(mixture, rules, 2), rel=1e-5)
    assert result.x_new.sum() == pytest.approx(1.0)


def test_one_more_ingredient_never_costs_more(mixture: PLS) -> None:
    """The value of one more ingredient: allowing a third can only help, and one alone may not be enough."""
    rules = _mixture_rules(groups={"all": MIXTURE}, max_count={"all": 2}, min_amount={"all": 0.1})

    report = mixture.inversion_sensitivity(rules)

    counts = report.counts.set_index("max_count")
    assert counts.loc[3, "change"] <= 1e-6
    assert counts.loc[1, "objective"] >= report.inversion.objective - 1e-6 or not counts.loc[1, "feasible"]


def test_one_of_a_pair_allowed_prices_only_the_second(mixture: PLS) -> None:
    """Either a or b, not both: with one allowed there is no 'one fewer', so only one more is priced."""
    rules = _mixture_rules(groups={"pair": ["a", "b"]}, max_count={"pair": 1}, min_amount={"pair": 0.1})

    report = mixture.inversion_sensitivity(rules)

    assert report.inversion.feasible
    assert len(report.inversion.used["pair"]) <= 1
    assert list(report.counts["max_count"]) == [2]


def test_an_ingredient_held_at_its_minimum_is_reported_as_that_not_as_a_bound(mixture: PLS) -> None:
    """At its group's minimum amount an ingredient is held by the counting rule, not moved one input at a time."""
    rules = _mixture_rules(costs={"a": 1.0, "b": 2.0, "c": 4.0, "d": 9.0}, groups={"all": MIXTURE},
                           max_count={"all": 2}, min_amount={"all": 0.25})  # fmt: skip

    report = mixture.inversion_sensitivity(rules)

    held = report.inversion.constraints.set_index("constraint").loc["b"]
    assert (held["kind"], held["binding"], held["lower"]) == ("min_amount", "lower", 0.25)
    assert "b" not in set(report.checked["constraint"])


def test_too_few_ingredients_for_the_window_has_no_answer_and_says_so(mixture: PLS) -> None:
    """One ingredient at most, but the quality window needs a blend: the counting rules are named."""
    rules = _mixture_rules(windows={"q": (6.5, 7.0)}, groups={"all": MIXTURE}, max_count={"all": 1},
                           min_amount={"all": 0.1})  # fmt: skip

    result = mixture.optimal_inversion(rules)

    assert not result.feasible
    assert result.proven_optimal  # the master ran out of choices: proven that none meets the rules
    assert "none do with them" in result.message


# ---------------------------------------------------------------------------------------------------------------------
# Alternatives: the next best answers, each with a different set of ingredients (no-good cuts)
# ---------------------------------------------------------------------------------------------------------------------
COUNTED = {"groups": {"all": MIXTURE}, "max_count": {"all": 2}, "min_amount": {"all": 0.1}}


def test_each_alternative_is_the_cheapest_ingredient_set_the_earlier_answers_allow(mixture: PLS) -> None:
    """Every alternative leaves out an ingredient of each earlier answer; no allowed set, solved alone, is cheaper."""
    # Arrange
    rules = _mixture_rules(**COUNTED)
    costs = _every_choice(mixture, rules, 2)

    # Act
    result = mixture.optimal_inversion(rules, alternatives=2)

    # Assert
    answers = [result, *result.alternatives]
    sets = [frozenset(r.used["all"]) for r in answers]
    assert all(r.feasible and r.proven_optimal for r in answers)
    for k, answer in enumerate(answers):
        allowed = [cost for s, cost in costs.items() if not any(earlier <= s for earlier in sets[:k])]
        assert not any(earlier <= sets[k] for earlier in sets[:k])
        assert answer.objective == pytest.approx(min(allowed), rel=1e-5)
    assert [r.objective for r in answers] == sorted(r.objective for r in answers)


def test_leaving_out_two_shares_at_most_the_rest_of_each_earlier_set(mixture: PLS) -> None:
    """With leave_out=2, a later answer keeps at most |S| - 2 of each earlier answer's ingredients S."""
    # Arrange
    rules = _mixture_rules(**COUNTED)
    costs = _every_choice(mixture, rules, 2)

    # Act
    result = mixture.optimal_inversion(rules, alternatives=1, leave_out=2)

    # Assert
    first, second = frozenset(result.used["all"]), result.alternatives[0]
    allowed = [cost for s, cost in costs.items() if len(first & s) <= len(first) - 2]
    assert second.feasible
    assert len(first & set(second.used["all"])) <= len(first) - 2
    assert second.objective == pytest.approx(min(allowed), rel=1e-5)


def test_the_alternatives_end_with_the_reason_once_no_ingredient_set_is_left(mixture: PLS) -> None:
    """Asked for more than exist, the list stops at a result that says why, after every allowed set was offered."""
    # Arrange
    rules = _mixture_rules(**COUNTED)
    costs = _every_choice(mixture, rules, 2)

    # Act
    result = mixture.optimal_inversion(rules, alternatives=10)

    # Assert
    found = [r for r in result.alternatives if r.feasible]
    last = result.alternatives[-1]
    assert len(result.alternatives) == len(found) + 1
    assert not last.feasible
    assert last.proven_optimal
    assert "No further answer leaves out 1" in last.message
    sets = [frozenset(r.used["all"]) for r in [result, *found]]
    assert all(any(earlier <= s for earlier in sets) for s in costs)  # every feasible set is used up


def test_alternatives_need_counted_inputs_and_whole_numbers(mixture: PLS) -> None:
    """Nothing to switch off, nothing to differ in; counts are whole numbers; no answer, no alternatives."""
    # Arrange, Act, Assert
    with pytest.raises(ValueError, match="need counted inputs"):
        mixture.optimal_inversion(_mixture_rules(), alternatives=1)
    with pytest.raises(ValueError, match="leave_out is a whole number"):
        mixture.optimal_inversion(_mixture_rules(**COUNTED), alternatives=1, leave_out=0)
    with pytest.raises(ValueError, match="alternatives is a whole number"):
        mixture.optimal_inversion(_mixture_rules(**COUNTED), alternatives=-1)
    unreachable = mixture.optimal_inversion(_mixture_rules(**COUNTED, windows={"q": (20.0, None)}), alternatives=2)
    assert not unreachable.feasible
    assert unreachable.alternatives == []
    assert "alternatives" not in mixture.optimal_inversion(_mixture_rules(**COUNTED))


def test_a_counted_input_needs_zero_inside_its_bounds(mixture: PLS) -> None:
    """'Not used' is 0: an input that must be at least 0.2 cannot be counted."""
    rules = _mixture_rules(bounds=dict.fromkeys(MIXTURE, (0.2, 1.0)) | {"d": (0.0, 1.0)},
                           windows={"q": (4.0, None)}, groups={"all": MIXTURE}, max_count={"all": 3})  # fmt: skip

    with pytest.raises(ValueError, match="A counted input must be allowed 0"):
        mixture.optimal_inversion(rules)


def test_a_search_that_runs_out_of_rounds_says_it_is_not_proven(mixture: PLS, monkeypatch: pytest.MonkeyPatch) -> None:
    """With a single round of outer approximation the best point is returned, flagged as not proven optimal."""
    monkeypatch.setattr(oi, "_MAX_ROUNDS", 1)
    rules = _mixture_rules(groups={"all": MIXTURE}, max_count={"all": 2}, min_amount={"all": 0.1})

    result = mixture.optimal_inversion(rules)

    assert result.feasible
    assert not result.proven_optimal
    assert "not proven optimal" in result.message
    assert "ran 1 rounds" in result.message


def test_a_master_problem_that_cannot_be_solved_proves_nothing(mixture: PLS, monkeypatch: pytest.MonkeyPatch) -> None:
    """A time limit in the counting rules' master is not an infeasibility: the answer says the search stopped."""
    real = oi.optimize.milp

    def timed_out(*args: object, **kwargs: object) -> OptimizeResult:
        if kwargs.get("integrality") is None:
            return real(*args, **kwargs)
        return OptimizeResult(x=None, status=1, message="Time limit reached.", fun=None)

    monkeypatch.setattr(oi.optimize, "milp", timed_out)
    rules = _mixture_rules(groups={"all": MIXTURE}, max_count={"all": 2}, min_amount={"all": 0.1})

    result = mixture.optimal_inversion(rules)

    assert not result.feasible
    assert not result.proven_optimal
    assert "could not be solved (Time limit reached.)" in result.message
    assert "none is ruled out" in result.message


def test_a_mixture_price_range_holds_the_answers_shape_inside_it(mixture: PLS) -> None:
    """Inside an ingredient's price range the same ingredients are used; past its edge, the answer changes."""
    rules = _mixture_rules()
    base = mixture.optimal_inversion(rules)

    report = mixture.inversion_sensitivity(rules, prices=["b"])
    row = report.prices.iloc[0]

    def shape(fraction: float) -> frozenset[str]:
        costs = dict(rules.costs) | {"b": rules.costs["b"] * (1 + fraction)}
        return oi._usage(mixture.optimal_inversion(dataclasses.replace(rules, costs=costs)))

    assert shape(row["rises_by"] / 2) == oi._usage(base)
    if row["rises_by"] < 1.0:
        assert shape(row["rises_by"] + 0.02) != oi._usage(base)


def test_a_price_with_no_answer_is_a_change_of_shape() -> None:
    """Price ranging compares shapes: a search with no answer at some price is a shape of its own."""
    assert oi._usage(Bunch(feasible=False)) == frozenset({"no answer"})


def test_prices_are_ranged_only_for_a_cost_objective(cheese: PLS) -> None:
    """A distance to targets has no prices to range; tracing something that is not a rule is refused."""
    with pytest.raises(ValueError, match="only be ranged when the objective is cost"):
        cheese.inversion_sensitivity(InversionRules(targets={"Taste": 25.0}), prices=["Acetic"])
    with pytest.raises(ValueError, match="cannot be traced"):
        cheese.inversion_sensitivity(InversionRules(costs=CHEESE_COSTS, windows={"Taste": (20, 30)}), front=("x", [1]))


# ---------------------------------------------------------------------------------------------------------------------
# TPLS: which lots, at what fractions
# ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.dataset
@pytest.mark.slow
def test_tpls_alternative_blends_use_other_lots_and_come_no_closer(pyphi: tuple[TPLS, dict]) -> None:
    """Each alternative blend leaves out a lot of every earlier blend, stays inside the limits, and is no closer."""
    # Arrange
    model, data = pyphi
    rules = InversionRules(
        targets=data["Y"]["Quality"].loc["L050"].to_dict(),
        max_count=dict.fromkeys(GROUPS, 2), min_amount=dict.fromkeys(GROUPS, 0.1),
    )  # fmt: skip

    # Act
    result = model.optimal_inversion(rules, alternatives=2)

    # Assert
    answers = [result, *result.alternatives]
    lots = [frozenset(lot for g in GROUPS for lot in r.used[g]) for r in answers]
    assert all(r.feasible and r.proven_optimal for r in answers)
    assert all(not lots[j] <= lots[k] for k in range(len(lots)) for j in range(k))
    assert all(r.hotellings_t2 <= r.t2_limit * (1 + 1e-4) for r in answers)
    assert all(r.objective >= result.objective - 1e-6 for r in answers)


@pytest.mark.dataset
def test_tpls_closest_blend_with_two_lots_per_material_is_what_tpls_predicts(pyphi: tuple[TPLS, dict]) -> None:
    """Like batch L050, at most two lots of each material, each at 10%+: every group sums to 1, inside the limits."""
    model, data = pyphi
    rules = InversionRules(
        targets=data["Y"]["Quality"].loc["L050"].to_dict(),
        max_count=dict.fromkeys(GROUPS, 2), min_amount=dict.fromkeys(GROUPS, 0.1),
    )  # fmt: skip

    result = model.optimal_inversion(rules)

    assert result.feasible
    assert result.proven_optimal
    for group in GROUPS:
        lots = result.x_new[model.material_names[group]]
        assert lots.sum() == pytest.approx(1.0, abs=1e-6)
        assert 1 <= len(result.used[group]) <= 2
        assert (lots[result.used[group]] >= 0.1 - 1e-6).all()
    rows = DataFrameDict({
        "F": {g: result.x_new[model.material_names[g]].to_frame().T for g in GROUPS},
        "Z": {"Conditions": result.x_new[model.condition_names["Conditions"]].to_frame().T},
    })  # fmt: skip
    check = model.diagnose(rows)
    np.testing.assert_allclose(check.hat["Quality"].iloc[0], result.y_hat, rtol=1e-8)
    assert float(check.hotellings_t2.iloc[0, -1]) <= result.t2_limit * (1 + 1e-4)


@pytest.mark.dataset
def test_tpls_cheapest_blend_beats_every_training_blend_that_qualifies(pyphi: tuple[TPLS, dict]) -> None:
    """Minimise cost under two windows: no training blend inside the windows and the limits is cheaper."""
    model, data = pyphi
    affine = affine_model(model)
    lots = [n for g in GROUPS for n in model.material_names[g]]
    costs = dict(zip(lots, np.random.default_rng(0).uniform(1, 5, len(lots)), strict=True))
    quality = data["Y"]["Quality"]
    windows = {"Y1": (float(quality["Y1"].median()), None), "Y3": (None, float(quality["Y3"].median()))}
    training = pd.concat([*(data["F"][g] for g in GROUPS), data["Z"]["Conditions"]], axis=1)[list(affine.names)]
    price = np.array([costs.get(n, 0.0) for n in affine.names])
    limits = oi._limits(affine, InversionRules(costs=costs))
    qualifying = [
        float(x @ price) for x in training.to_numpy()
        if affine.predict(x)[0] >= windows["Y1"][0] and affine.predict(x)[2] <= windows["Y3"][1]
        and all(lim.value(x) <= lim.limit for lim in limits)
    ]  # fmt: skip

    result = optimal_inversion(affine, InversionRules(costs=costs, windows=windows))

    assert qualifying
    assert result.feasible
    assert result.objective <= min(qualifying) + 1e-6
    assert result.y_hat["Y1"] >= windows["Y1"][0] - 1e-6


@pytest.mark.dataset
@pytest.mark.slow
def test_tpls_cheapest_blend_of_two_lots_per_material_costs_no_less_than_the_free_blend(
    pyphi: tuple[TPLS, dict],
) -> None:
    """The counting rules can only add cost; the answer keeps to them and is proven optimal."""
    model, data = pyphi
    affine = affine_model(model)
    lots = [n for g in GROUPS for n in model.material_names[g]]
    costs = dict(zip(lots, np.random.default_rng(0).uniform(1, 5, len(lots)), strict=True))
    quality = data["Y"]["Quality"]
    rules = InversionRules(costs=costs, windows={"Y1": (float(quality["Y1"].median()), None)})
    counted = dataclasses.replace(rules, max_count=dict.fromkeys(GROUPS, 2), min_amount=dict.fromkeys(GROUPS, 0.1))

    free, result = optimal_inversion(affine, rules), optimal_inversion(affine, counted)

    assert result.feasible
    assert result.proven_optimal
    assert result.n_cuts > 0
    assert all(1 <= len(result.used[g]) <= 2 for g in GROUPS)
    assert result.objective >= free.objective - 1e-6


@pytest.mark.dataset
@pytest.mark.slow
def test_a_cheaper_lot_never_loses_the_answer(pyphi: tuple[TPLS, dict]) -> None:
    """
    Found by price ranging: with one lot 6.25% cheaper the search once ended with a false 'no answer'. Its master
    had no bound on the process conditions, the cost ignores them, and a tangent plane taken at an extreme left
    HiGHS rejecting blends that meet every rule. A cheaper lot can only lower the cost.
    """
    model, data = pyphi
    affine = affine_model(model)
    lots = [n for g in GROUPS for n in model.material_names[g]]
    costs = dict(zip(lots, np.round(np.random.default_rng(0).uniform(1, 5, len(lots)), 2), strict=True))
    quality = data["Y"]["Quality"]
    rules = InversionRules(
        costs=costs, windows={"Y1": (float(quality["Y1"].median()), None), "Y3": (None, float(quality["Y3"].median()))},
        max_count=dict.fromkeys(GROUPS, 2), min_amount=dict.fromkeys(GROUPS, 0.1),
    )  # fmt: skip
    base = optimal_inversion(affine, rules)

    cheaper = optimal_inversion(affine, dataclasses.replace(rules, costs=costs | {"A0013": costs["A0013"] * 0.9375}))

    assert base.feasible
    assert cheaper.feasible
    assert cheaper.proven_optimal
    assert cheaper.objective <= base.objective + 1e-9


def _small_tpls(lots_b: list[str]) -> TPLS:
    """Fit a two-material TPLS without process conditions: 30 blends of three lots each."""
    rng = np.random.default_rng(3)
    lots = {"A": ["A1", "A2", "A3"], "B": lots_b}
    d = {g: pd.DataFrame(rng.normal(size=(3, 2)), index=names, columns=["p1", "p2"]) for g, names in lots.items()}
    f = {g: pd.DataFrame(rng.dirichlet([1, 1, 1], 30), columns=names) for g, names in lots.items()}
    y = pd.DataFrame({"q": f["A"].to_numpy() @ [3.0, 1.0, 2.0] + f["B"].to_numpy() @ [1.0, 2.0, 0.5]})
    return TPLS(n_components=2, d_matrix=d).fit(DataFrameDict({"F": f, "Y": {"Quality": y}}))


def test_a_tpls_model_without_conditions_blends_its_lots(recwarn: pytest.WarningsRecorder) -> None:
    """With no Z block the inputs are the lots alone, and each material's fractions still sum to 1."""
    model = _small_tpls(["B1", "B2", "B3"])

    result = model.optimal_inversion(InversionRules(targets={"q": 3.5}))

    assert result.feasible
    assert list(result.spe.index) == ["F/A", "F/B"]
    assert result.x_new[["A1", "A2", "A3"]].sum() == pytest.approx(1.0)
    assert result.y_hat["q"] == pytest.approx(3.5, abs=1e-3)


def test_without_closure_a_materials_fractions_are_free_to_total_anything() -> None:
    """``closure=False`` drops the 'sums to 1' rule: the constraint table has none."""
    model = _small_tpls(["B1", "B2", "B3"])

    result = model.optimal_inversion(InversionRules(targets={"q": 3.5}, closure=False))

    assert result.feasible
    assert "closure" not in set(result.constraints["kind"])


def test_a_lot_at_its_whole_material_is_held_by_the_total_and_its_bound_is_implied() -> None:
    """
    One lot per material, so each is at 100%: its bound and the material's total hold it together. The total keeps
    the price; the bound, reported with the input's own bounds, is implied (NaN) and not moved to check it.
    """
    model = _small_tpls(["B1", "B2", "B3"])
    lots = ["A1", "A2", "A3", "B1", "B2", "B3"]
    rules = InversionRules(
        costs=dict(zip(lots, [3.0, 1.0, 2.0, 1.0, 2.0, 3.0], strict=True)),
        max_count={"A": 1, "B": 1},
        min_amount={"A": 0.1, "B": 0.1},
        t2_limit=1e6,
        spe_limits={"F/A": 1e6, "F/B": 1e6},
    )  # fmt: skip  (a pure lot is atypical)

    report = model.inversion_sensitivity(rules)
    table = report.inversion.constraints
    whole = table.query("kind == 'bound'")

    assert report.inversion.feasible
    assert report.inversion.proven_optimal
    assert len(whole) == 2
    assert (whole["lower"] == 0.0).all()
    assert (whole["upper"] == 1.0).all()
    assert whole["shadow_price"].isna().all()
    assert table.query("kind == 'closure'")["shadow_price"].notna().all()
    assert not set(whole["constraint"]) & set(report.checked["constraint"])


def test_tpls_groups_that_share_a_lot_name_are_refused() -> None:
    """Inputs are named by their lot alone, so two materials may not share a lot name."""
    model = _small_tpls(["A1", "B2", "B3"])

    with pytest.raises(ValueError, match="share a lot or condition name"):
        affine_model(model)


# ---------------------------------------------------------------------------------------------------------------------
# Each input's extreme limits, and the proof that an answer is optimal
# ---------------------------------------------------------------------------------------------------------------------
CHEESE_BOUNDS = {"Acetic": (4.0, 7.0), "H2S": (3.0, 10.0), "Lactic": (0.8, 2.1)}


def test_an_input_without_both_bounds_is_named_at_the_callers_line(cheese: PLS) -> None:
    """Only T² and SPE would hold it. The warning names it, at the caller's own line, through the method too."""
    rules = InversionRules(costs=CHEESE_COSTS, windows={"Taste": (20.0, 30.0)})

    with pytest.warns(FutureWarning, match="Acetic, H2S, Lactic have no lower or no upper bound") as record:
        cheese.optimal_inversion(rules)
    with pytest.warns(FutureWarning, match="error from 3.0") as sensitivity:
        cheese.inversion_sensitivity(rules)

    assert {w.filename for w in [*record, *sensitivity]} == {__file__}


def test_a_fixed_input_needs_no_bounds_and_one_side_is_not_enough(cheese: PLS) -> None:
    """A fixed input is held at its value; an input with a lower bound alone can still run off the other way."""
    rules = InversionRules(
        costs=CHEESE_COSTS, windows={"Taste": (20.0, 30.0)},
        bounds={"Acetic": (4.0, None), "H2S": (3.0, 10.0)}, fixed={"Lactic": 1.5},
    )  # fmt: skip

    with pytest.warns(FutureWarning, match=r"Acetic has no lower or no upper bound") as record:
        optimal_inversion(cheese, rules)

    assert "H2S" not in str(record[0].message)
    assert "Lactic" not in str(record[0].message)


def test_inputs_with_both_bounds_search_without_a_warning(cheese: PLS) -> None:
    """Every input given its extreme limits: nothing to warn of, and the answer keeps inside them."""
    rules = InversionRules(costs=CHEESE_COSTS, windows={"Taste": (20.0, 30.0)}, bounds=CHEESE_BOUNDS)

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        result = optimal_inversion(cheese, rules)

    assert result.feasible
    assert all(low - 1e-9 <= result.x_new[n] <= high + 1e-9 for n, (low, high) in CHEESE_BOUNDS.items())


@pytest.mark.dataset
def test_tpls_lot_fractions_have_bounds_already_and_the_conditions_need_them(pyphi: tuple[TPLS, dict]) -> None:
    """A lot's fraction is 0 to 1 unless given; a process condition has no natural range, so it is named."""
    model, data = pyphi
    rules = InversionRules(targets=data["Y"]["Quality"].loc["L050"].to_dict())
    conditions = data["Z"]["Conditions"]
    lots = [n for g in GROUPS for n in model.material_names[g]]

    with pytest.warns(FutureWarning, match="var1, var2, var3, var4, var5 and 5 more have no lower") as record:
        model.optimal_inversion(rules)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        bounded = model.optimal_inversion(
            dataclasses.replace(rules, bounds={c: (conditions[c].min(), conditions[c].max()) for c in conditions})
        )

    assert not any(lot in str(record[0].message) for lot in lots)
    assert bounded.feasible


@pytest.mark.dataset
def test_tangent_planes_prove_an_optimum_whose_multipliers_do_not_settle(pyphi: tuple[TPLS, dict]) -> None:
    """
    Bounded by its training range, the cheapest blend has many lots at a bound. The binding rows are dependent and
    the least-squares multipliers miss, though the point is optimal: the linear program over the limits' tangent
    planes does no better, and the answer is proven.
    """
    model, data = pyphi
    training = pd.concat([*(data["F"][g] for g in GROUPS), data["Z"]["Conditions"]], axis=1)
    lots = [n for g in GROUPS for n in model.material_names[g]]
    quality = data["Y"]["Quality"]
    rules = InversionRules(
        costs=dict(zip(lots, np.random.default_rng(0).uniform(1, 5, len(lots)), strict=True)),
        windows={"Y1": (float(quality["Y1"].median()), None), "Y3": (None, float(quality["Y3"].median()))},
        bounds={n: (float(training[n].min()), float(training[n].max())) for n in training},
    )  # fmt: skip

    result = model.optimal_inversion(rules)

    assert result.feasible
    assert result.kkt_residual > 1e-3  # the multipliers alone could not prove it
    assert result.proven_optimal
    assert result.objective == pytest.approx(6.955066, rel=1e-5)


def test_tangent_planes_refuse_a_blend_that_meets_the_rules_at_a_higher_cost(mixture: PLS) -> None:
    """
    The cheapest blend for a stricter window also meets this window, at a higher cost. The planes at the optimum
    prove it; the planes at the costlier blend find a cheaper point, so prove nothing.
    """
    rules = _mixture_rules()
    problem = oi._Problem(affine_model(mixture, rules.conf_level), rules)
    bounds = problem.continuous_bounds(None)
    best = optimal_inversion(mixture, rules)
    stricter = optimal_inversion(mixture, _mixture_rules(windows={"q": (7.0, None)}))

    assert stricter.objective > best.objective + 1e-3
    assert stricter.y_hat["q"] >= 6.0
    assert oi._proven_by_planes(problem, best.x_new.to_numpy(), bounds)
    assert not oi._proven_by_planes(problem, stricter.x_new.to_numpy(), bounds)
