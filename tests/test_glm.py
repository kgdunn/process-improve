"""Generalized linear models in ``analyze_experiment``, checked against Montgomery (2013), section 15.1.

The coupon redemptions (Table 15.1) are fitted by Minitab's binary logistic regression, the grill
defects (Table 15.4) by a Poisson model and the worsted yarn's cycles to failure (Table 15.6) by a
gamma model with a log link, both in SAS's GENMOD.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import reference_catalogue
from process_improve.experiments._analyses.glm import run_glm
from process_improve.experiments._tools.analyze_experiment import AnalyzeExperimentInput, analyze_experiment_tool
from process_improve.experiments.analysis import analyze_experiment


def _runs(name: str) -> pd.DataFrame:
    source = json.loads((reference_catalogue._CATALOGUE_DIR / "montgomery-2013.json").read_text(encoding="utf-8"))
    entry = next(e for e in source["experiments"] if e["name"] == name)
    return pd.DataFrame(entry["runs"]["rows"], columns=entry["runs"]["columns"]).astype(float)


@pytest.fixture(scope="module")
def coupons() -> pd.DataFrame:
    return _runs("montgomery-coupon-logistic-regression").assign(n=1000.0)


def test_logistic_regression_reproduces_minitab(coupons):
    """Minitab's reduced model: coefficients, standard errors, z, odds ratios and the fit statistics."""
    out = analyze_experiment(
        coupons,
        response_column="redeemed",
        model="redeemed ~ A + B + C + B:C",
        analysis_type="glm",
        family="binomial",
        trials="n",
    )["glm"]
    rows = {row["term"]: row for row in out["coefficients"]}
    assert rows["A"]["coefficient"] == pytest.approx(0.168675, abs=5e-7)
    assert rows["A"]["std_error"] == pytest.approx(0.0254235, abs=5e-8)
    assert rows["B:C"]["z_value"] == pytest.approx(-1.61, abs=0.005)
    assert rows["C"]["p_value"] == pytest.approx(0.364, abs=5e-4)
    assert (rows["A"]["exp_coefficient"], rows["A"]["exp_ci_low"], rows["A"]["exp_ci_high"]) == pytest.approx(
        (1.18, 1.13, 1.24), abs=0.005
    )
    assert out["log_likelihood_trials"] == pytest.approx(-4615.346, abs=5e-4)
    fit = out["goodness_of_fit"]
    assert (fit["pearson"]["statistic"], fit["pearson"]["df"], fit["pearson"]["p_value"]) == pytest.approx(
        (1.53593, 3, 0.674), abs=5e-4
    )
    assert fit["deviance"]["statistic"] == pytest.approx(1.53602, abs=5e-5)
    hosmer = fit["hosmer_lemeshow"]  # eight distinct runs, each its own group
    assert (hosmer["statistic"], hosmer["df"], hosmer["p_value"]) == pytest.approx((1.53593, 6, 0.957), abs=5e-4)


def test_one_trial_per_row_gives_the_same_fit_as_the_counts(coupons):
    """Successes out of trials, or each trial as its own 0/1 row: the same coefficients and standard errors."""
    small = coupons.assign(n=20.0, redeemed=lambda d: np.round(d["redeemed"] / 50))
    rows = [{"A": r.A, "B": r.B, "C": r.C, "y": float(i < r.redeemed)} for r in small.itertuples() for i in range(20)]
    grouped = run_glm(small, "redeemed ~ A + B", "binomial", trials="n")
    single = run_glm(pd.DataFrame(rows), "y ~ A + B", "binomial")
    for a, b in zip(grouped["coefficients"], single["coefficients"], strict=True):
        assert (a["coefficient"], a["std_error"]) == pytest.approx((b["coefficient"], b["std_error"]), rel=1e-6)


def test_poisson_means_and_their_intervals_reproduce_genmod():
    """The grill defects: the fitted mean of runs 1 and 2 with its 95% interval, from exp of the linear predictor's."""
    out = run_glm(_runs("montgomery-grill-defects-glm"), "c ~ D + F + B:G", "poisson")
    assert [row["coefficient"] for row in out["coefficients"]] == pytest.approx(
        [1.128, -0.896, -1.176, -0.737], abs=5e-4
    )
    first, second = out["fitted"][:2]
    assert (first["predicted"], first["ci_low"], first["ci_high"]) == pytest.approx((51.26, 42.45, 61.90), abs=0.005)
    assert (second["predicted"], second["ci_low"], second["ci_high"]) == pytest.approx((11.74, 8.14, 16.94), abs=0.005)
    assert out["dispersion"] == 1.0


def test_gamma_intervals_follow_the_package_convention_asked_for():
    """GENMOD's maximum-likelihood dispersion and observed information reproduce Montgomery's intervals; R's do not."""
    yarn = _runs("montgomery-worsted-yarn-glm")
    sas = run_glm(yarn, "cycles ~ x1 + x2 + x3", "gamma", link="log", dispersion="ml", information="observed")
    first = sas["fitted"][0]
    assert (first["predicted"], first["ci_low"], first["ci_high"]) == pytest.approx((680.52, 583.83, 793.22), abs=0.005)
    r_like = run_glm(yarn, "cycles ~ x1 + x2 + x3", "gamma", link="log")
    assert r_like["fitted"][0]["predicted"] == pytest.approx(first["predicted"])  # the same fit,
    assert r_like["fitted"][0]["ci_low"] < first["ci_low"]  # a wider interval
    assert r_like["dispersion"] == pytest.approx(r_like["pearson_chi2"] / r_like["df_residual"])
    assert [row["coefficient"] for row in sas["coefficients"]] == pytest.approx([6.35, 0.84, -0.63, -0.39], abs=0.005)


def test_observed_and_expected_information_agree_under_the_canonical_link():
    defects = _runs("montgomery-grill-defects-glm")
    expected = run_glm(defects, "c ~ D + F", "poisson", information="expected")
    observed = run_glm(defects, "c ~ D + F", "poisson", information="observed")
    for a, b in zip(expected["coefficients"], observed["coefficients"], strict=True):
        assert a["std_error"] == pytest.approx(b["std_error"], rel=1e-8)


def test_a_mean_is_predicted_at_new_settings_with_its_interval():
    yarn = _runs("montgomery-worsted-yarn-glm")
    centre = pd.DataFrame({"x1": [0.0], "x2": [0.0], "x3": [0.0]})
    out = analyze_experiment(
        yarn,
        response_column="cycles",
        model="cycles ~ x1 + x2 + x3",
        analysis_type="glm",
        family="gamma",
        link="log",
        new_points=centre,
    )["glm"]
    predicted = out["predictions"][0]
    assert predicted["predicted"] == pytest.approx(np.exp(out["coefficients"][0]["coefficient"]))
    assert predicted["ci_low"] < predicted["predicted"] < predicted["ci_high"]


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({}, "needs a family"),
        ({"family": "normal"}, "family must be one of"),
        ({"family": "binomial", "link": "nope"}, "link must be one of"),
        ({"family": "poisson", "trials": 1000}, "for the binomial family"),
        ({"family": "binomial", "trials": "missing"}, "not in the data"),
        ({"family": "gamma", "dispersion": "moments"}, "dispersion must be"),
        ({"family": "gamma", "dispersion": -1.0}, "must be positive"),
        ({"family": "gamma", "information": "fisher"}, "information must be"),
        ({"family": "gamma", "transform": "log"}, "without a transform"),
    ],
)
def test_a_glm_that_cannot_be_fitted_as_asked_is_refused(coupons, options, message):
    with pytest.raises(ValueError, match=message):
        analyze_experiment(coupons, response_column="redeemed", analysis_type="glm", **options)


def test_the_trials_column_is_not_a_factor(coupons):
    out = analyze_experiment(
        coupons, response_column="redeemed", model="main_effects", analysis_type="glm", family="binomial", trials="n"
    )
    assert out["factor_names"] == ["A", "B", "C"]
    assert [row["term"] for row in out["glm"]["coefficients"]] == ["Intercept", "A", "B", "C"]


def test_the_tool_fits_a_glm(coupons):
    spec = AnalyzeExperimentInput(
        design_matrix=coupons.drop(columns="n").to_dict("records"),
        response_column="redeemed",
        model="main_effects",
        analysis_type="glm",
        family="binomial",
        trials=1000,
    )
    result = analyze_experiment_tool(spec)
    assert result["glm"]["coefficients"][1]["exp_coefficient"] == pytest.approx(1.18, abs=0.005)
