"""Standard errors from an error standard deviation known before the runs (#789 B6).

Each printed value comes from a source that takes sigma from past records or earlier
experiments, not from the runs' residual. They are entries of the reference catalogue,
read here from its data.
"""

from __future__ import annotations

import doctest
import json
import math
from importlib import resources

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from process_improve.experiments import analyze_experiment

pytestmark = pytest.mark.dataset


def _runs(source: str, name: str) -> pd.DataFrame:
    """Return an entry's runs from the reference catalogue."""
    catalogue = resources.files("process_improve.datasets.experiments") / "reference_catalogue" / source
    entry = next(e for e in json.loads(catalogue.read_text(encoding="utf-8"))["experiments"] if e["name"] == name)
    return pd.DataFrame(entry["runs"]["rows"], columns=entry["runs"]["columns"])


def _rows(result: dict, key: str = "coefficients") -> dict[str, dict]:
    return {row["term"]: row for row in result[key]}


@pytest.fixture(scope="module")
def first_order() -> pd.DataFrame:
    """BHH Table 15.1: a 2^2 factorial with three centre runs, in coded units."""
    runs = _runs("box-hunter-hunter-1978.json", "bhh-rsm-first-order-and-steepest-ascent")
    return runs.iloc[:7][["time", "temperature", "y"]]


class TestCoefficients:
    def test_bhh_15_4_the_fitted_plane(self, first_order: pd.DataFrame) -> None:
        """The plane 62.01 + 2.35 x1 + 4.50 x2, (±0.57) (±0.75) (±0.75), with sigma = 1.5: sigma/sqrt(7), sigma/2."""
        result = analyze_experiment(
            first_order, response_column="y", model="main_effects", analysis_type="coefficients", sigma=1.5
        )
        rows = _rows(result)
        assert rows["Intercept"]["coefficient"] == pytest.approx(62.014, abs=1e-3)
        assert rows["Intercept"]["std_error"] == pytest.approx(1.5 / math.sqrt(7))
        assert round(rows["Intercept"]["std_error"], 2) == 0.57
        assert rows["time"]["std_error"] == pytest.approx(0.75)
        assert rows["temperature"]["std_error"] == pytest.approx(0.75)
        assert result["model_summary"]["prior_sigma"] == 1.5

    def test_bhh_15_7_the_interaction_check(self, first_order: pd.DataFrame) -> None:
        """b12 = -0.65 ± 0.75."""
        result = analyze_experiment(
            first_order, response_column="y", model="interactions", analysis_type="coefficients", sigma=1.5
        )
        row = _rows(result)["time:temperature"]
        assert (round(row["coefficient"], 2), row["std_error"]) == (-0.65, pytest.approx(0.75))

    def test_the_tests_and_intervals_read_the_normal_distribution(self, first_order: pd.DataFrame) -> None:
        """Sigma is not estimated from the runs, so no degrees of freedom are lost to it."""
        result = analyze_experiment(
            first_order, response_column="y", model="main_effects", analysis_type="coefficients", sigma=1.5
        )
        row = _rows(result)["time"]
        z = row["coefficient"] / 0.75
        assert row["t_value"] == pytest.approx(z)
        assert row["p_value"] == pytest.approx(2 * stats.norm.sf(abs(z)))
        half = stats.norm.ppf(0.975) * 0.75
        assert (row["ci_low"], row["ci_high"]) == pytest.approx((row["coefficient"] - half, row["coefficient"] + half))

    def test_davies_11_5_a_three_by_three(self) -> None:
        """Sigma² = 0.6 from earlier experiments: b0 = 81.22 ± 0.58 and b1 = 1.97 ± 0.32, V(b0) = 0.5555 sigma²."""
        runs = _runs("davies-1956.json", "davies-near-stationary-3x3").drop(columns="Trial")
        result = analyze_experiment(
            runs, response_column="y", model="quadratic", analysis_type="coefficients", sigma=math.sqrt(0.6)
        )
        rows = _rows(result)
        assert (round(rows["Intercept"]["coefficient"], 2), round(rows["Intercept"]["std_error"], 2)) == (81.22, 0.58)
        assert (round(rows["x1"]["coefficient"], 2), round(rows["x1"]["std_error"], 2)) == (1.97, 0.32)
        assert rows["Intercept"]["std_error"] ** 2 / 0.6 == pytest.approx(5 / 9)

    def test_davies_11_7_the_composite_design(self) -> None:
        """V(b0) = (224/288) sigma², so b0 has a standard error of about 0.9 with sigma = 1."""
        runs = _runs("davies-1956.json", "davies-composite-three-factor").drop(columns="Trial")
        result = analyze_experiment(
            runs, response_column="y", model="quadratic", analysis_type="coefficients", coding="actual", sigma=1.0
        )
        intercept = _rows(result)["Intercept"]["std_error"]
        assert intercept**2 == pytest.approx(224 / 288)
        assert round(intercept, 1) == 0.9

    def test_bhh_15_19_the_block_effect(self) -> None:
        """The two blocks of six runs differ by an amount whose standard error is sqrt(sigma²/6 + sigma²/6) = 0.87."""
        runs = _runs("box-hunter-hunter-1978.json", "bhh-rsm-composite-design")
        frame = runs[["block", "time", "temperature", "y"]].rename(columns={"block": "Block"})
        result = analyze_experiment(
            frame, response_column="y", model="quadratic", analysis_type="coefficients", sigma=1.5
        )
        difference = 2 * _rows(result)["Block1"]["std_error"]  # the blocks' contrast is half their difference
        assert difference == pytest.approx(1.5 / math.sqrt(3), rel=1e-3)
        assert round(difference, 2) == 0.87

    def test_the_coefficients_in_the_factors_own_units_use_it_too(self) -> None:
        runs = pd.DataFrame({"T": [100, 150, 100, 150, 125], "P": [1, 1, 3, 3, 2], "y": [10.0, 14.0, 11.0, 17.0, 13.0]})
        result = analyze_experiment(
            runs, response_column="y", model="main_effects", analysis_type="coefficients", coding="coded", sigma=0.5
        )
        actual = {row["term"]: row for row in result["coefficients_actual"]}
        assert actual["T"]["std_error"] == pytest.approx(0.5 / 25 / 2)  # sigma/sqrt(4) per coded unit, 25 K each
        assert _rows(result)["T"]["std_error"] == pytest.approx(0.25)


class TestEffectsAndSignificance:
    @pytest.fixture(scope="class")
    def bicycle(self) -> pd.DataFrame:
        """BHH Table 12.6: the first 2^(7-4) fraction, eight runs for seven factors."""
        return _runs("box-hunter-hunter-1978.json", "bhh-bicycle-2-7-4").iloc[:8].drop(columns="No.")

    def test_bhh_12_a_saturated_design_gets_standard_errors(self, bicycle: pd.DataFrame) -> None:
        """Sigma = 3 s from past experience: each effect has a standard error of sqrt(3²/4 + 3²/4) = 2.1."""
        result = analyze_experiment(
            bicycle, response_column="y", model="main_effects", analysis_type=["effects", "significance"], sigma=3.0
        )
        assert result["model_summary"]["df_residual"] == 0
        assert result["effect_std_errors"] == pytest.approx(dict.fromkeys(result["effects"], 3 / math.sqrt(2)))
        assert round(result["effect_std_errors"]["gear"], 1) == 2.1
        assert result["significant_terms"] == ["dynamo", "gear"], "BHH: only the dynamo and the gear matter"
        assert "significance_note" not in result

    def test_without_sigma_the_saturated_design_has_none(self, bicycle: pd.DataFrame) -> None:
        result = analyze_experiment(
            bicycle, response_column="y", model="main_effects", analysis_type=["effects", "significance"]
        )
        assert "effect_std_errors" not in result
        assert "no term can be tested" in result["significance_note"]
        assert "prior_sigma" not in result["model_summary"]

    def test_davies_the_limits_of_an_effect(self) -> None:
        """Process variance 6.25 from past records: the effect of 10% more ammonium chloride is 5.88 ± 2.45."""
        runs = _runs("davies-1956.json", "davies-organic-chemical-confounded")
        result = analyze_experiment(
            runs[["A", "B", "C", "Yield"]],
            response_column="Yield",
            model="main_effects",
            analysis_type=["effects", "confidence_intervals"],
            sigma=2.5,
        )
        assert abs(result["effects"]["B"]) == pytest.approx(5.875)
        assert stats.norm.ppf(0.975) * result["effect_std_errors"]["B"] == pytest.approx(2.45, abs=5e-3)
        row = _rows(result, "confidence_intervals")["B"]
        assert row["ci_high"] - row["ci_low"] == pytest.approx(2.45, abs=5e-3), "the coefficient is half the effect"


class TestWhatKeepsTheResidual:
    def test_the_anova_and_lenths_method_do_not_change(self, first_order: pd.DataFrame) -> None:
        types = ["anova", "lenth_method", "lack_of_fit"]
        given = analyze_experiment(first_order, response_column="y", model="main_effects", analysis_type=types)
        known = analyze_experiment(
            first_order, response_column="y", model="main_effects", analysis_type=types, sigma=1.5
        )
        for key in ("anova_table", "lenth_method", "lack_of_fit"):
            # Through JSON, so that a NaN in the residual row compares equal to itself
            assert json.dumps(known[key], sort_keys=True) == json.dumps(given[key], sort_keys=True), key

    def test_predictions_use_it(self, first_order: pd.DataFrame) -> None:
        """At the centre the fitted value has variance sigma²/7, and a new run sigma² more."""
        known = analyze_experiment(
            first_order,
            response_column="y",
            model="main_effects",
            analysis_type="prediction",
            new_points=pd.DataFrame({"time": [0.0], "temperature": [0.0]}),
            sigma=1.5,
        )
        row = known["predictions"][0]
        z = stats.norm.ppf(0.975)
        assert row["ci_high"] - row["predicted"] == pytest.approx(z * 1.5 / math.sqrt(7))
        assert row["pi_high"] - row["predicted"] == pytest.approx(z * 1.5 * math.sqrt(1 + 1 / 7))


@pytest.mark.parametrize("sigma", [0, -1.5, float("nan"), float("inf"), "1.5", True])
def test_sigma_must_be_a_positive_number(first_order: pd.DataFrame, sigma: object) -> None:
    with pytest.raises(ValueError, match="sigma must be a positive number"):
        analyze_experiment(first_order, response_column="y", sigma=sigma)  # type: ignore[arg-type]


def test_a_numpy_number_is_a_sigma(first_order: pd.DataFrame) -> None:
    result = analyze_experiment(first_order, response_column="y", analysis_type="coefficients", sigma=np.float64(1.5))
    assert result["model_summary"]["prior_sigma"] == 1.5


def test_the_docstring_example_prints_what_it_shows() -> None:
    runner = doctest.DocTestRunner()
    for test in doctest.DocTestFinder().find(analyze_experiment):
        runner.run(test)
    failed, attempted = runner.summarize(verbose=False)
    assert attempted > 0
    assert failed == 0
