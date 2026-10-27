"""Edge cases of the analysis side: GLM options, split-plot fits, catalogue checks, plots and tool wrappers.

Each test drives one branch the rest of the suite does not reach and checks what it does: the
error and its message, or the number, against a value worked out by hand or by another route.
"""

from __future__ import annotations

import importlib
import itertools
import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import optimize, stats

from process_improve.experiments import (
    analyze_experiment,
    least_squares_means,
    optimize_responses,
    reference_catalogue,
)
from process_improve.experiments._lm import UnsafeFormulaError, validate_formula_is_safe
from process_improve.experiments.random_effects import variance_components
from process_improve.experiments.visualization.plots.registry import create_plot
from process_improve.multivariate import PLS, MCUVScaler
from process_improve.multivariate._pls import _held_out_scores
from process_improve.regression import box_tidwell
from process_improve.regression.nonlinear import nonlinear_regression
from process_improve.simulation.batch import BioreactorSimulator
from process_improve.tool_spec import execute_tool_call
from process_improve.univariate.classical import dunnett_value

box_tidwell_module = importlib.import_module("process_improve.regression.box_tidwell")
variance_module = importlib.import_module("process_improve.regression._variance_components")


def _quiet(function, *args, **kwargs):
    """Call ``function`` with its warnings silenced: the tests check results, not the advice printed beside them."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function(*args, **kwargs)


# --------------------------------------------------------------------------- generalized linear models
@pytest.fixture(scope="module")
def doses() -> pd.DataFrame:
    """Twenty settings of one factor, ten trials at each: more settings than Hosmer and Lemeshow's ten groups."""
    successes = [1, 0, 2, 1, 3, 2, 3, 4, 3, 5, 6, 5, 7, 6, 8, 7, 9, 8, 9, 9]
    return pd.DataFrame({"x": np.linspace(-1.0, 1.0, 20), "s": np.array(successes, dtype=float), "n": 10.0})


@pytest.fixture(scope="module")
def amounts() -> pd.DataFrame:
    """Return a positive response whose spread grows with its size, for the gamma family."""
    y = [2.1, 3.5, 2.8, 4.9, 6.2, 5.1, 8.3, 7.7, 9.9, 12.4]
    return pd.DataFrame({"x": np.linspace(-1.0, 1.0, 10), "y": y})


def _glm(df: pd.DataFrame, response: str, **options) -> dict:
    return analyze_experiment(df, response_column=response, model=f"{response} ~ x", analysis_type="glm", **options)[
        "glm"
    ]


@pytest.mark.parametrize("trials", [0, [10, 10, 10, -1, 10, 10]])
def test_a_number_of_trials_that_is_not_positive_is_refused(trials):
    """Trials given as one number, or one per run, must all be positive."""
    df = pd.DataFrame({"x": [-1.0, 1.0, -1.0, 1.0, 0.0, 0.0], "s": [2.0, 8.0, 3.0, 7.0, 5.0, 4.0]})
    with pytest.raises(ValueError, match="Every run's number of trials must be positive"):
        _glm(df, "s", family="binomial", trials=trials)


def test_hosmer_lemeshow_cuts_more_than_ten_settings_into_ten_groups(doses):
    """Twenty runs of equal trials, sorted by fitted risk, make ten groups of two; the test has 10 - 2 df."""
    found = _glm(doses, "s", family="binomial", trials="n")["goodness_of_fit"]["hosmer_lemeshow"]
    successes = doses["s"].to_numpy()
    fit = sm.GLM(np.column_stack([successes, 10 - successes]), sm.add_constant(doses["x"]), sm.families.Binomial())
    p = fit.fit().mu
    order = np.argsort(p)
    statistic = 0.0
    for pair in order.reshape(10, 2):
        observed, expected = successes[pair].sum(), 10 * p[pair].sum()
        statistic += (observed - expected) ** 2 / (expected * (1 - expected / 20))
    assert found["df"] == 8
    assert found["statistic"] == pytest.approx(statistic, rel=1e-9)
    assert found["p_value"] == pytest.approx(stats.chi2.sf(statistic, 8), rel=1e-9)


def test_a_dispersion_given_as_a_number_is_held_fixed(amounts):
    """Gamma with a log link has unit weights, so each variance is phi over the column's sum of squares."""
    out = _glm(amounts, "y", family="gamma", link="log", dispersion=0.5)
    assert (out["dispersion"], out["dispersion_method"]) == (0.5, "fixed")
    x = amounts["x"].to_numpy()
    intercept, slope = out["coefficients"]
    assert intercept["std_error"] == pytest.approx(np.sqrt(0.5 / len(x)), rel=1e-6)
    assert slope["std_error"] == pytest.approx(np.sqrt(0.5 / np.sum(x**2)), rel=1e-6)
    assert out["aic"] == pytest.approx(-2 * out["log_likelihood"] + 2 * 2)  # a fixed dispersion is not estimated


def test_the_deviance_dispersion_is_the_deviance_over_the_residual_df(amounts):
    out = _glm(amounts, "y", family="gamma", link="log", dispersion="deviance")
    y = amounts["y"].to_numpy()
    mu = np.array([row["predicted"] for row in out["fitted"]])
    deviance = 2 * np.sum(-np.log(y / mu) + (y - mu) / mu)
    assert out["dispersion_method"] == "deviance"
    assert out["dispersion"] == pytest.approx(deviance / (len(y) - 2), rel=1e-6)


def test_a_probit_link_has_no_odds_ratios(doses):
    """Only a log or logit link gives exponentiated coefficients; the probit fit is statsmodels' own."""
    out = _glm(doses, "s", family="binomial", link="probit", trials="n")
    successes = doses["s"].to_numpy()
    probit = sm.families.Binomial(sm.families.links.Probit())
    fit = sm.GLM(np.column_stack([successes, 10 - successes]), sm.add_constant(doses["x"]), probit).fit()
    for row, estimate, std_error in zip(out["coefficients"], fit.params, fit.bse, strict=True):
        assert "exp_coefficient" not in row
        assert (row["coefficient"], row["std_error"]) == pytest.approx((estimate, std_error), rel=1e-6)


# --------------------------------------------------------------------------- least squares: ANOVA and selection
def test_a_main_effect_with_no_intercept_is_tested_on_its_uncorrected_sum_of_squares():
    """In y ~ -1 + A + A:B, A is adjusted for nothing: its sum of squares is that of the A means about zero."""
    df = pd.DataFrame(
        {
            "A": ["a1"] * 4 + ["a2"] * 4,
            "B": ["b1", "b1", "b2", "b2", "b3", "b3", "b4", "b4"],
            "y": [1.0, 2.0, 4.0, 5.0, 7.0, 8.0, 12.0, 13.0],
        }
    )
    out = _quiet(analyze_experiment, df, response_column="y", model="y ~ -1 + A + A:B", analysis_type="anova")
    rows = {row["source"]: row for row in out["anova_table"]}
    assert rows["A"]["df"] == 2
    assert rows["A"]["sum_sq"] == pytest.approx(4 * 3.0**2 + 4 * 10.0**2)  # the A means are 3 and 10
    assert rows["A:B"]["sum_sq"] == pytest.approx(36.0 - 2.0)  # within-A less within-cell sums of squares
    assert rows["Residual"]["sum_sq"] == pytest.approx(2.0)
    assert rows["A"]["F"] == pytest.approx(436.0 / 2 / (2.0 / 4))


def test_a_linear_mixture_model_leaves_the_search_nothing_to_move():
    """A linear Scheffe model has no blending terms to add or drop; its AICc is that of the three proportions."""
    rows = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]
    rows += [(1 / 3, 1 / 3, 1 / 3), (2 / 3, 1 / 6, 1 / 6), (1 / 6, 2 / 3, 1 / 6), (1 / 6, 1 / 6, 2 / 3)]
    df = pd.DataFrame(rows, columns=list("ABC")).assign(y=[11.0, 9.2, 15.1, 10.8, 13.9, 12.5, 12.1, 11.9, 10.4, 13.6])
    found = analyze_experiment(df, response_column="y", model="scheffe_linear", analysis_type="model_selection")
    found = found["model_selection"]
    x, y = df[list("ABC")].to_numpy(), df["y"].to_numpy()
    rss = float(np.sum((y - x @ np.linalg.lstsq(x, y, rcond=None)[0]) ** 2))
    n, k = len(y), 4  # three coefficients and the error variance
    assert found["selected_formula"] == "y ~ -1 + A + B + C"
    assert found["criterion_value"] == pytest.approx(
        n * np.log(2 * np.pi * rss / n) + n + 2 * k + 2 * k * (k + 1) / (n - k - 1), rel=1e-12
    )


def test_a_supersaturated_design_keeps_the_interactions_model_it_cannot_estimate():
    """With more factors than runs not even the main effects are estimable, so there is nothing to fall back to."""
    signs = np.array(
        [
            [1, 1, 1, -1, 1, -1, -1, 1],
            [1, -1, -1, 1, 1, 1, -1, -1],
            [-1, 1, -1, 1, -1, 1, 1, 1],
            [-1, -1, 1, -1, -1, -1, 1, -1],
            [1, 1, -1, -1, -1, 1, 1, -1],
            [-1, -1, 1, 1, 1, -1, -1, 1],
        ],
        dtype=float,
    )
    df = pd.DataFrame(signs, columns=list("ABCDEFGH")).assign(y=[3.1, 2.4, 5.0, 4.2, 3.3, 2.9])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = analyze_experiment(df, response_column="y", analysis_type="coefficients")
    assert out["model_summary"]["model"] == "interactions"
    assert out["model_summary"]["formula"] == "y ~ (A + B + C + D + E + F + G + H) ** 2"
    assert not [w for w in caught if "main-effects model was fitted instead" in str(w.message)]


def test_block_contrasts_skip_a_name_a_factor_already_has():
    """A factor named Block1 keeps its name; the contrast of the two blocks is Block1_, half their difference."""
    df = pd.DataFrame(
        {
            "A": [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
            "Block1": [-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0],
            "Block": [1, 1, 1, 1, 2, 2, 2, 2],
            "y": [10.0, 14.0, 11.0, 16.0, 12.0, 17.0, 14.0, 18.0],
        }
    )
    out = analyze_experiment(df, response_column="y", model="main_effects", analysis_type=["coefficients", "anova"])
    assert out["model_summary"]["formula"] == "y ~ Block1_ + A + Block1"
    coefficients = {row["term"]: row["coefficient"] for row in out["coefficients"]}
    assert coefficients["Block1_"] == pytest.approx((12.75 - 15.25) / 2)  # the two blocks' means
    assert coefficients["Block1"] == pytest.approx((11 + 16 + 14 + 18 - 10 - 14 - 12 - 17) / 8)
    block = next(row for row in out["anova_table"] if row["source"] == "Block")
    assert (block["df"], block["sum_sq"]) == (1, pytest.approx(8 * 1.25**2))


def test_a_level_a_factor_does_not_have_is_refused_in_at():
    df = pd.DataFrame(
        {
            "treatment": ["t1", "t2", "t3"] * 4,
            "block": ["B1"] * 3 + ["B2"] * 3 + ["B3"] * 3 + ["B4"] * 3,
            "y": [5.0, 6.1, 7.2, 5.4, 6.0, 7.9, 4.8, 6.6, 7.1, 5.2, 5.9, 7.5],
        }
    )
    with pytest.raises(ValueError, match="block has no level 'B9'"):
        least_squares_means(df, "y ~ treatment + block", "treatment", at={"block": "B9"})


def test_one_side_of_a_formula_may_be_empty_and_the_other_is_still_checked():
    validate_formula_is_safe("~ A + I(A ** 2)", ["A"], allow_transforms=True)  # a one-sided formula is accepted
    with pytest.raises(UnsafeFormulaError, match=r"unknown name\(s\) \['os'\]"):
        validate_formula_is_safe("~ A + I(os)", ["A"], allow_transforms=True)


# --------------------------------------------------------------------------- split-plot analyses
def _strip_plot(seed: int = 3) -> pd.DataFrame:
    """Six rows and six columns, x1 set by row and x2 by column; the rows vary, the columns do not."""
    rng = np.random.default_rng(seed)
    rows = [
        {"Row": f"r{r}", "Column": f"c{c}", "x1": [-1, 1][r % 2], "x2": [-1, 1][c % 2]}
        for r in range(6)
        for c in range(6)
    ]
    df = pd.DataFrame(rows)
    row_effect = {f"r{r}": rng.normal() for r in range(6)}
    return df.assign(y=3 + df["x1"] + 2 * df["x2"] + df["Row"].map(row_effect) + rng.normal(scale=0.5, size=36))


def _split_plot(df: pd.DataFrame, model: str, whole_plot, **options) -> dict:
    return _quiet(
        analyze_experiment,
        df,
        response_column="y",
        model=model,
        whole_plot=whole_plot,
        analysis_type="split_plot",
        **options,
    )


def test_a_crossed_random_effect_estimated_as_zero_is_named_and_left_out():
    """With the column variance at zero, the two-way fit is the split-plot fit on the rows alone."""
    df = _strip_plot()
    crossed = _split_plot(df, "x1 + x2", ["Row", "Column"])
    single = _split_plot(df, "x1 + x2", "Row")["split_plot"]["variance_components"]
    variances = crossed["split_plot"]["variance_components"]
    assert variances["Column"] == 0.0
    assert crossed["split_plot_note"].startswith("The variance of 'Column' is estimated as zero")
    assert (variances["Row"], variances["residual"]) == pytest.approx((single["whole_plot"], single["residual"]))


def test_crossed_random_effects_need_every_term_estimable():
    df = _strip_plot().assign(x3=lambda d: d["x1"])
    with pytest.raises(ValueError, match="has 4 columns but rank 3, so some terms are aliased"):
        _split_plot(df, "x1 + x2 + x3", ["Row", "Column"])


def test_the_expected_information_stands_in_when_the_observed_is_not_positive_definite():
    """At a whole-plot variance of zero, on its boundary, the observed information is not positive definite."""
    rng = np.random.default_rng(2)
    rows = [
        {"plot": f"w{w}", "A": [-1, 1][w % 2], "B": [-1, 1][s % 2], "C": [-1, 1][(w // 2) % 2]}
        for w in range(6)
        for s in range(4)
    ]
    df = pd.DataFrame(rows)
    whole = {f"w{w}": rng.normal(scale=0.3) for w in range(6)}
    df = df.assign(y=1 + df["A"] + 0.5 * df["B"] + df["plot"].map(whole) + rng.normal(size=24))
    observed = _split_plot(df, "y ~ A + B + C", "plot", information="observed")["split_plot"]
    expected = _split_plot(df, "y ~ A + B + C", "plot", information="expected")["split_plot"]
    assert observed["variance_components"]["whole_plot"] == 0.0
    assert [row["df"] for row in observed["coefficients"]] == pytest.approx(
        [row["df"] for row in expected["coefficients"]]
    )


# --------------------------------------------------------------------------- random effects
@pytest.fixture(scope="module")
def gauge() -> pd.DataFrame:
    """Four parts, each read once by three operators: the part-by-operator cells hold one reading each."""
    readings = [[10.1, 10.5, 9.8], [12.0, 12.4, 11.7], [9.0, 9.6, 9.1], [11.2, 11.0, 11.5]]
    rows = [{"part": f"p{p}", "operator": f"o{o}", "y": v} for p, row in enumerate(readings) for o, v in enumerate(row)]
    return pd.DataFrame(rows)


def test_with_no_residual_df_the_error_variance_has_a_wald_interval(gauge):
    """part:operator takes every reading, so the residual is the part-by-operator mean square, with a Wald interval."""
    out = _quiet(variance_components, gauge, "y ~ operator", ["part", "part:operator"])
    table = gauge["y"].to_numpy().reshape(4, 3)  # a row per part, a column per operator
    interaction = table - table.mean(axis=1, keepdims=True) - table.mean(axis=0) + table.mean()
    mean_square = float(np.sum(interaction**2)) / ((4 - 1) * (3 - 1))
    std_error = mean_square * np.sqrt(2 / 6)  # an ANOVA mean square on 6 df
    z = stats.norm.ppf(0.975)
    assert out["components"]["residual"] == pytest.approx(mean_square, rel=1e-8)
    assert out["std_errors"]["residual"] == pytest.approx(std_error, rel=1e-6)
    assert out["intervals"]["residual"] == pytest.approx([mean_square - z * std_error, mean_square + z * std_error])


def test_a_formula_without_a_response_is_refused(gauge):
    with pytest.raises(ValueError, match="must name one response on the left of '~'"):
        variance_components(gauge, "~ operator", ["part"])


def test_an_unbounded_search_steps_back_from_a_covariance_that_is_not_positive_definite(monkeypatch):
    """Group means nearly equal: the unbounded REML estimates are the ANOVA ones, the batch variance negative."""
    groups = [[-1.0, 0.5, 1.0, -0.4], [0.9, -1.1, 0.3, -0.1], [-0.6, 1.2, -0.8, 0.2], [0.4, -0.3, 1.1, -1.2]]
    groups.append([-0.2, 0.7, -1.0, 0.5])
    df = pd.DataFrame({"y": np.concatenate(groups), "batch": np.repeat([f"b{k}" for k in range(5)], 4)})
    infinite = []
    criterion = variance_module._Profile.criterion

    def watched(self, psi):
        value = criterion(self, psi)
        infinite.extend([value] if np.isinf(value) else [])
        return value

    monkeypatch.setattr(variance_module._Profile, "criterion", watched)
    out = variance_components(df, "y ~ 1", ["batch"], bounded=False)
    within = sum(float(np.sum((np.array(g) - np.mean(g)) ** 2)) for g in groups) / (5 * 3)
    between = 4 * float(np.var([np.mean(g) for g in groups], ddof=1))
    assert infinite, "the search tried a step past where V is positive definite"
    assert out["components"]["residual"] == pytest.approx(within, rel=1e-9)
    assert out["components"]["batch"] == pytest.approx((between - within) / 4, rel=1e-9)


# --------------------------------------------------------------------------- desirability and optimisation
def _model(**terms: float) -> dict:
    """Return a fitted model of factors A and B from coefficients named as the terms are."""
    return {
        "response_name": "y",
        "factor_names": ["A", "B"],
        "coefficients": [{"term": term, "coefficient": value} for term, value in terms.items()],
    }


@pytest.mark.parametrize(
    ("goal", "message"),
    [
        ({"response": "y", "low": 0, "high": 1}, "Goal dict for response 'y' is missing the required 'goal' key"),
        ({"response": "y", "goal": "target", "low": 0, "high": 1}, "A goal of 'target' for response 'y' requires"),
    ],
)
def test_a_goal_without_its_type_or_target_is_refused(goal, message):
    with pytest.raises(ValueError, match=message):
        optimize_responses([_model(Intercept=1.0, A=1.0)], goals=[goal], method="desirability")


def test_no_alternatives_are_offered_when_no_setting_is_desirable():
    """The goal starts above anything the model reaches, so every desirability is 0."""
    goal = [{"response": "y", "goal": "maximize", "low": 50, "high": 60}]
    found = _quiet(optimize_responses, [_model(Intercept=1.0, A=1.0)], goals=goal, n_alternatives=2)
    assert found["desirability"]["composite_desirability"] == 0.0
    assert found["desirability"]["alternatives"] == []


def test_a_single_peak_has_no_alternative_however_many_are_asked_for():
    """The response 3 - A^2 - B^2 peaks once, at the centre, where d = 3/4: every start ends there."""
    peak = _model(**{"Intercept": 3.0, "I(A ** 2)": -1.0, "I(B ** 2)": -1.0})
    goal = [{"response": "y", "goal": "maximize", "low": 0, "high": 4}]  # d > 0 over the whole square
    found = optimize_responses([peak], goals=goal, n_alternatives=1)["desirability"]
    assert found["composite_desirability"] == pytest.approx(0.75)
    assert list(found["optimal_coded"].values()) == pytest.approx([0.0, 0.0], abs=1e-6)
    assert found["alternatives"] == []


def test_the_ridge_bracket_narrows_when_the_slope_on_the_leading_axis_is_tiny():
    """On the unit circle, y = 1e-9 A + B - A^2 - 2 B^2 = sin t - 1 - sin^2 t peaks at sin t = 1/2."""
    model = _model(**{"Intercept": 0.0, "A": 1e-9, "B": 1.0, "I(A ** 2)": -1.0, "I(B ** 2)": -2.0})
    end = optimize_responses([model], method="ridge_analysis")["ridge_analysis"]["path"][-1]
    assert end["radius"] == pytest.approx(1.0)
    assert (end["coded"]["A"], end["coded"]["B"]) == pytest.approx((np.sqrt(0.75), 0.5), abs=1e-6)
    assert end["predicted_response"] == pytest.approx(0.5 - 1 - 0.25, abs=1e-6)


# --------------------------------------------------------------------------- the reference catalogue's checks
def _plane() -> dict:
    """Return a 2^2 with a centre point, y = 10 + 3 x1 - x2 exactly, x1 from 100 to 200 and x2 from 125 to 175."""
    runs = [(-1, -1), (-1, 1), (1, -1), (1, 1), (0, 0)]
    return {
        "name": "edge-plane",
        "factors": [
            {"name": "x1", "type": "numeric", "low": 100, "high": 200},
            {"name": "x2", "type": "numeric", "low": 125, "high": 175},
        ],
        "responses": [{"name": "y"}],
        "runs": {"columns": ["x1", "x2", "y"], "rows": [[u, v, 10 + 3 * u - v] for u, v in runs], "coded": True},
    }


def test_a_path_stepped_against_the_gradient_is_refused():
    """The gradient in x2 is -1, so a source's step of +25 in x2 cannot be on the path of steepest ascent."""
    check = {"kind": "steepest_path", "model": "x1 + x2", "per": "x2", "change": 25, "steps": 1, "key": "x1"}
    with pytest.raises(ValueError, match="The path moves x2 the other way from the source's change of 25"):
        reference_catalogue._reproduce(_plane(), {"value": "0"}, check, reference_catalogue._Fits())


def test_a_model_the_library_cannot_fit_leaves_only_the_runs():
    out = reference_catalogue._Fits()(_plane(), {"model": "x1 + x9"})
    assert set(out) == {"y", "settings"}
    assert out["y"].tolist() == [8.0, 6.0, 14.0, 12.0, 10.0]


def test_a_projection_onto_a_missing_entry_or_another_shape_is_said_so():
    entry = _plane()
    with pytest.raises(KeyError, match="The catalogue has no entry 'no-such-design'"):
        reference_catalogue._reproduce(entry, {"value": "x"}, {"kind": "projection", "of": "no-such-design"}, None)
    check = {"kind": "projection", "columns": ["x1", "x2"], "into": ["x1"]}
    found = reference_catalogue._reproduce(entry, {"value": "x"}, check, None)
    assert found == ("5 runs and 2 columns against 5 and 1", False)


def test_a_design_metric_of_a_named_design_is_read_from_the_design_the_library_makes():
    """A 2^2 full factorial with its three centre points: X'X is diag(7, 4, 4) for the main effects."""
    check = {
        "kind": "design_metric",
        "design": {"design_type": "full_factorial", "k": 2},
        "model": "main_effects",
        "metric": "information_matrix",
        "key": "xtx",
    }
    printed = [["7", "0", "0"], ["0", "4", "0"], ["0", "0", "4"]]
    _, agrees = reference_catalogue._reproduce({"name": "named"}, {"value": printed}, check, None)
    assert agrees


def test_a_design_metric_on_some_runs_carries_those_runs_groups():
    """Runs 1 to 4 lie in two whole plots, so X' V^-1 X uses V = I + Z Z' for those four runs alone."""
    rows = [["p1", -1, -1, 0], ["p1", -1, 1, 0], ["p2", 1, -1, 0], ["p2", 1, 1, 0], ["p3", -1, -1, 0], ["p3", 1, 1, 0]]
    entry = {
        "name": "edge-split-plot",
        "factors": [{"name": n, "type": "numeric", "low": -1, "high": 1} for n in "AB"],
        "responses": [{"name": "y"}],
        "runs": {"columns": ["plot", "A", "B", "y"], "rows": rows, "coded": True},
    }
    check = {
        "kind": "design_metric",
        "metric": "information_matrix",
        "model": "A + B",
        "options": {"random_effects": {"plot": 1.0}},
        "groups": ["plot"],
        "runs": [1, 2, 3, 4],
        "key": "xtx",
    }
    found = reference_catalogue._design_metric_array(entry, check)
    x = np.array([[1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]], dtype=float)
    z = np.kron(np.eye(2), np.ones((2, 1)))
    np.testing.assert_allclose(found, x.T @ np.linalg.solve(np.eye(4) + z @ z.T, x), atol=1e-12)


# --------------------------------------------------------------------------- plots
def _factorial(k: int) -> list[dict[str, float]]:
    return [dict(zip("ABCDEF", p, strict=False)) for p in itertools.product([-1.0, 1.0], repeat=k)]


def test_a_design_quality_plot_takes_the_model_of_the_fitted_formula():
    spec = create_plot(
        "fds_plot", design_data=_factorial(3), analysis_results={"model_summary": {"formula": "y ~ A + B"}}
    ).to_spec()
    assert spec.metadata["model"] == "A + B"


def test_a_half_fraction_is_judged_on_its_main_effects():
    """A 2^(3-1) has four runs: the seven terms of the interactions model do not fit, the four main-effect ones do."""
    half = [row for row in _factorial(3) if row["A"] * row["B"] * row["C"] == 1]
    spec = create_plot("power_curve", design_data=[*half, *half]).to_spec()
    assert (spec.metadata["model"], spec.metadata["n_terms"], spec.metadata["residual_df"]) == ("main_effects", 4, 4)


@pytest.mark.parametrize(
    ("plot_type", "title"),
    [("fds_plot", "FDS Plot - no factor columns"), ("power_curve", "Power Curve - no design information")],
)
def test_a_design_of_responses_alone_has_no_design_quality(plot_type, title):
    spec = create_plot(plot_type, design_data=[{"y": 1.0}, {"y": 2.0}], response_column="y").to_spec()
    assert spec.title == title


def _desirability_surface(coefficient: float) -> list[list[float]]:
    responses = [{"coefficients": [{"term": "Intercept", "coefficient": coefficient}], "goal": "maximize"}]
    plot = create_plot(
        "desirability_contour", analysis_results={"optimization": {"responses": responses}}, factors_to_plot=["A", "B"]
    )
    return plot.to_spec().panels[0].layers[0].style["z_matrix"]


def test_a_flat_response_ramps_over_one_unit_about_its_level():
    """Low and high are 5 -/+ 0.5, so the level itself sits half way up the ramp."""
    assert np.unique(_desirability_surface(5.0)).tolist() == [0.5]


def test_a_response_with_no_finite_prediction_ramps_from_0_to_1():
    """Every prediction is infinite, past the high end of the [0, 1] ramp, so every setting is fully desirable."""
    assert np.unique(_desirability_surface(np.inf)).tolist() == [1.0]


def test_an_overlay_with_neither_responses_nor_coefficients_says_so():
    plot = create_plot("overlay", analysis_results={"model_summary": {"r_squared": 0.9}}, factors_to_plot=["A", "B"])
    assert plot.to_spec().title == "Overlay Plot - no response data"


def _mixed_design() -> list[dict]:
    lo = [(-1, -1), (1, -1), (-1, 1), (1, 1), (0, 0), (0, 0)]
    return [{"A": a, "B": b, "C": "lo"} for a, b in lo] + [{"A": a, "B": b, "C": "hi"} for a, b in [(-1, -1), (1, 1)]]


@pytest.mark.parametrize("axes", [["A", "Q"], ["A", "C"]])
def test_the_prediction_variance_axes_must_be_quantitative_factors(axes):
    plot = create_plot("prediction_variance", design_data=_mixed_design(), factors_to_plot=axes, model="main_effects")
    with pytest.raises(ValueError, match=f"factors_to_plot names '{axes[1]}', which is not a quantitative factor"):
        plot.to_spec()


def test_a_categorical_factor_not_plotted_is_held_at_its_first_runs_level():
    """C is held at 'lo', the level of the first run: its indicator (reference 'hi') is 1 across the grid."""
    rows = _mixed_design()
    spec = create_plot("prediction_variance", design_data=rows, model="main_effects").to_spec()
    style = spec.panels[0].layers[0].style
    x = np.array([[1.0, r["A"], r["B"], float(r["C"] == "lo")] for r in rows])
    inverse = np.linalg.inv(x.T @ x)
    gx, gy = np.meshgrid(style["x_grid"], style["y_grid"])
    points = np.column_stack([np.ones(gx.size), gx.ravel(), gy.ravel(), np.ones(gx.size)])
    expected = len(rows) * np.einsum("ij,jk,ik->i", points, inverse, points)
    np.testing.assert_allclose(np.ravel(style["z_matrix"]), expected, rtol=1e-10)


# --------------------------------------------------------------------------- regression and multivariate
def test_a_held_out_row_that_sees_only_two_copies_of_one_column_has_no_two_scores():
    """Two identical observed columns cannot fix two scores: that row stays NaN, the others are projected."""
    rng = np.random.default_rng(0)
    base = rng.normal(size=(30, 3))
    x = pd.DataFrame(np.column_stack([base[:, 0], base]), columns=["a", "a_copy", "b", "c"])
    y = pd.DataFrame({"y": base @ [1.0, -1.0, 0.5] + 0.1 * rng.normal(size=30)})
    xs, ys = MCUVScaler().fit_transform(x), MCUVScaler().fit_transform(y)
    model = PLS(n_components=2, scale=False).fit(xs.iloc[:25], ys.iloc[:25])
    test = xs.iloc[25:].to_numpy().copy()
    test[0, 2:] = np.nan  # only a and its copy are seen
    test[1, 3] = np.nan
    scores = _held_out_scores(model, xs.iloc[:25].to_numpy(), test, 2)
    assert np.isnan(scores[0]).all()
    np.testing.assert_allclose(scores[1], model.project(pd.DataFrame(test[1:2], columns=x.columns)).scores.iloc[0])
    np.testing.assert_allclose(scores[2:], test[2:] @ model.direct_weights_.to_numpy()[:, :2])


def test_a_trial_power_whose_columns_overflow_is_halved_and_the_search_goes_on(monkeypatch):
    """The response (ln x - 4.25)^2 overshoots to a power past 4000; the least-squares power is near -0.87."""
    x = np.geomspace(1.0, 1e3, 15)
    y = (np.log(x) - 4.25) ** 2
    overflowed = []
    columns = box_tidwell_module._power_columns

    def watched(values, powers):
        found = columns(values, powers)
        overflowed.extend([] if np.isfinite(found[0]).all() else [powers.copy()])
        return found

    monkeypatch.setattr(box_tidwell_module, "_power_columns", watched)
    found = _quiet(box_tidwell, x, y)

    def sse(power: float) -> float:
        model = np.column_stack([np.ones_like(x), x**power])
        return float(np.sum((y - model @ np.linalg.lstsq(model, y, rcond=None)[0]) ** 2))

    best = optimize.minimize_scalar(sse, bounds=(-2.0, -0.2), method="bounded", options={"xatol": 1e-10})
    assert overflowed, "a trial power overflowed the columns"
    assert found["power"]["x"] == pytest.approx(best.x, abs=1e-6)
    assert found["sse"] == pytest.approx(best.fun, rel=1e-8)


def test_a_search_that_runs_off_to_infinity_does_not_converge():
    """The fit improves for ever as a and b grow, so the search spends every evaluation it has."""
    x = np.linspace(0.0, 5.0, 10)
    with pytest.raises(ValueError, match=r"did not converge from the start values \{'a': 0.0, 'b': 0.0\}"):
        nonlinear_regression(lambda x, a, b: np.exp(-a) * x + np.exp(-b), x, np.zeros(10), {"a": 0.0, "b": 0.0})


@pytest.mark.slow  # two optimiser searches of the simulator: about 2.5 s under coverage
def test_a_later_start_that_does_worse_does_not_replace_the_best():
    """With random_state=1 the random start ends below the nominal one, so the nominal start's optimum stands."""
    sim = BioreactorSimulator()
    nominal = sim.optimal_trajectory(n_knots=2, n_starts=1)
    two = sim.optimal_trajectory(n_knots=2, n_starts=2, random_state=1)
    assert two.titer == nominal.titer
    pd.testing.assert_frame_equal(two.trajectory, nominal.trajectory)


@pytest.mark.parametrize("correlation", [1.0, -0.1, [0.5, 1.2, 0.5]])
def test_dunnett_needs_each_correlation_in_0_to_1(correlation):
    with pytest.raises(ValueError, match=r"correlation in \[0, 1\)"):
        dunnett_value(0.95, 3, 12.0, correlation=correlation)


# --------------------------------------------------------------------------- tool wrappers
def test_the_omars_tool_gives_no_formula_for_a_response_a_formula_cannot_name():
    from process_improve.experiments import Factor, TrueProcess, generate_design

    factors = [Factor(name=name, low=-1, high=1) for name in "ABCDEFGH"]
    design = generate_design(factors, design_type="dsd", n_center_points=3).design.drop(columns="RunOrder")
    truth = TrueProcess({"Intercept": 50, "A": 6, "C": -4, "F": 5, "A:C": 4}, sigma=0.5)
    rows = design.assign(**{"yield (%)": truth.simulate(design, random_state=5)}).to_dict("records")
    result = execute_tool_call("analyze_omars", {"design_matrix": rows, "response_column": "yield (%)"})
    assert result["success"] is True
    assert sorted(result["active_main_effects"]) == ["A", "C", "F"]
    assert result["model_formula"] is None


def test_a_boxplot_keeps_the_title_it_is_given():
    data = [{"w": v} for v in (9.8, 10.1, 10.0, 10.4, 9.9)]
    result = execute_tool_call(
        "boxplot", {"data": data, "value_columns": ["w"], "title": "Fill weights", "backend": "plotly"}
    )
    assert result["title"] == "Fill weights"
    assert result["data"]["quartiles"][0]["q_stats"][2] == pytest.approx(10.0)  # the median


def test_a_model_grown_with_more_components_says_how_many_it_had():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(30, 4)), columns=list("abcd"), index=[f"r{i}" for i in range(30)])
    frame.index.name = "id"
    session = execute_tool_call("mva_open", {"source": {"csv": frame.to_csv(), "name": "plant.csv"}})["session"]
    first = execute_tool_call(
        "mva_fit", {"session": session, "name": "M1", "kind": "PCA", "n_components": 2, "cross_validate": False}
    )
    second = execute_tool_call(
        "mva_fit",
        {"session": first["session"], "name": "M2", "parent": "M1", "n_components": 3, "cross_validate": False},
    )
    assert second["model"]["changes"] == ["n_components 3 (was 2)"]
