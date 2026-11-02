"""The Box-Cox interval for lambda: the likelihood ratio's, and the t-based allowance the books draw (#801 F1).

Box, Hunter and Hunter (1978, sections 7.8 and 10.10) and Montgomery (2013, Equation 15.2) bound
the interval by the critical sum of squares S_min (1 + t^2 / nu), and read its limits off a plot
of S(lambda). Their critical values reproduce wherever their S_min does; their limits, read by
eye, agree with the solved ones to within 0.03.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest
from patsy import dmatrix
from scipy import stats

from process_improve.experiments import analyze_experiment, reference_catalogue
from process_improve.experiments._analyses.box_cox import _profile, box_cox_profile


def _runs(source: str, name: str) -> pd.DataFrame:
    path = reference_catalogue._CATALOGUE_DIR / f"{source}.json"
    entry = next(e for e in json.loads(path.read_text(encoding="utf-8"))["experiments"] if e["name"] == name)
    return pd.DataFrame(entry["runs"]["rows"], columns=entry["runs"]["columns"])


def _box_cox(runs: pd.DataFrame, factors: list[str], model: str) -> dict:
    """Return the box_cox analysis, as ``analyze_experiment`` gives it for the model."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the model's own fit warns of nothing the interval depends on
        return analyze_experiment(runs[factors], runs["y"], model=model, analysis_type="box_cox")["box_cox"]


#: Each source's runs, the factors and model of its Box-Cox analysis, the library's t limits, and the printed ones.
BOOKS = [
    pytest.param(
        ("box-hunter-hunter-1978", "bhh-toxic-agents", ["poison", "treatment"], "poison + treatment"),
        (-1.168, -0.325),
        (-1.18, -0.32),
        id="bhh-7.8-toxic-agents",
    ),
    pytest.param(
        ("box-hunter-hunter-1978", "bhh-daniel-drill-2-4", ["A", "B", "C", "D"], "A + B + C + D"),
        (-0.304, 0.196),
        (-0.32, 0.21),
        id="bhh-10.10-drill",
    ),
    pytest.param(
        ("montgomery-2013", "montgomery-peak-discharge-box-cox", ["method"], "method"),
        (0.299, 0.792),
        (0.27, 0.77),
        id="montgomery-15.1.1",
    ),
]


@pytest.mark.parametrize(("entry", "solved", "printed"), BOOKS)
def test_the_t_interval_is_the_one_the_books_read_off_their_plots(entry, solved, printed):
    source, name, factors, model = entry
    found = _box_cox(_runs(source, name), factors, model)
    t_interval = found["intervals"]["t"]
    nu = found["residual_df"]
    assert t_interval["critical_sse"] == pytest.approx(found["sse"] * (1 + stats.t.ppf(0.975, nu) ** 2 / nu))
    assert t_interval["lambda_ci"] == pytest.approx(solved, abs=5e-4)
    assert t_interval["lambda_ci"] == pytest.approx(printed, abs=0.03), "read off a plot, by eye"
    likelihood = found["intervals"]["likelihood"]
    assert found["lambda_ci"] == likelihood["lambda_ci"], "lambda_ci stays the likelihood ratio's"
    assert likelihood["lambda_ci"][0] > t_interval["lambda_ci"][0]
    assert likelihood["lambda_ci"][1] < t_interval["lambda_ci"][1], "the t allowance widens it, on few df"


def test_the_critical_sums_of_squares_the_books_print():
    """BHH 7.8 prints 0.322 (1 + 4.08/42) = 0.353. BHH 10.10's 2.59 (1 + t^2/11) = 3.73 and Montgomery's 35.00 (1 +
    2.086^2/20) = 42.61 start from an S_min read off their plots; the runs give 2.539 and 35.27.
    """
    toxic = _box_cox(_runs("box-hunter-hunter-1978", "bhh-toxic-agents"), ["poison", "treatment"], "poison + treatment")
    assert (toxic["sse"], toxic["residual_df"]) == (pytest.approx(0.322, abs=5e-4), 42)
    assert toxic["intervals"]["t"]["critical_sse"] == pytest.approx(0.353, abs=5e-4)
    drill = _box_cox(_runs("box-hunter-hunter-1978", "bhh-daniel-drill-2-4"), ["A", "B", "C", "D"], "A + B + C + D")
    assert (drill["sse"], drill["residual_df"]) == (pytest.approx(2.539, abs=5e-4), 11)
    peak = _box_cox(_runs("montgomery-2013", "montgomery-peak-discharge-box-cox"), ["method"], "method")
    assert (peak["sse"], peak["residual_df"]) == (pytest.approx(35.27, abs=5e-3), 20)
    assert peak["intervals"]["t"]["critical_sse"] == pytest.approx(35.27 * (1 + 2.086**2 / 20), abs=0.01)


@pytest.mark.parametrize("interval", ["likelihood", "t"])
def test_each_end_is_where_s_lambda_meets_the_critical_value(interval):
    """The ends are solved for, not taken at a point of the 0.001 grid."""
    peak = _runs("montgomery-2013", "montgomery-peak-discharge-box-cox")
    y, exog = peak["y"].to_numpy(dtype=float), np.asarray(dmatrix("C(method)", peak), dtype=float)
    profile = _profile(y, exog)
    lmbda, (low, high) = box_cox_profile(y, exog, interval=interval)
    assert lmbda == profile.lmbda
    found = _box_cox(peak, ["method"], "method")["intervals"][interval]
    assert [low, high] == pytest.approx(found["lambda_ci"])
    for end in (low, high):
        assert profile.sse(end) == pytest.approx(found["critical_sse"], rel=1e-9)
    if interval == "likelihood":
        # 2 (L(lambda_hat) - L(lambda)) = chi2(1) at each end, with L(lambda) = -n/2 log S(lambda)
        assert len(y) * np.log(profile.sse(low) / profile.s_min) == pytest.approx(stats.chi2.ppf(0.95, 1))


def test_a_model_that_leaves_no_residual_df_has_no_t_interval():
    runs = pd.DataFrame({"A": [-1.0, 1.0, -1.0, 1.0], "B": [-1.0, -1.0, 1.0, 1.0], "y": [3.0, 5.0, 4.0, 9.0]})
    found = _box_cox(runs, ["A", "B"], "A + B + A:B")
    assert found["residual_df"] == 0
    assert found["intervals"]["t"] is None
    exog = np.asarray(dmatrix("A + B + A:B", runs), dtype=float)
    with pytest.raises(ValueError, match="needs residual degrees of freedom"):
        box_cox_profile(runs["y"].to_numpy(dtype=float), exog, interval="t")


def test_an_unknown_convention_is_refused():
    runs = _runs("montgomery-2013", "montgomery-peak-discharge-box-cox")
    exog = np.asarray(dmatrix("C(method)", runs), dtype=float)
    with pytest.raises(ValueError, match="interval must be one of"):
        box_cox_profile(runs["y"].to_numpy(dtype=float), exog, interval="symmetric")
