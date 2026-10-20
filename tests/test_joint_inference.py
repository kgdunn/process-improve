"""The correlations of the coefficient estimates and their joint confidence region (#789 B10)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from process_improve.experiments import analyze_experiment

# Box, Hunter and Hunter, Table 14.5: impurity against two concentrations, fitted through the origin.
IMPURITY = pd.DataFrame(
    {
        "x1": [0.34, 0.34, 0.58, 1.26, 1.26, 1.82],
        "x2": [0.73, 0.73, 0.69, 0.97, 0.97, 0.46],
        "y": [5.75, 4.79, 5.44, 9.09, 8.59, 5.09],
    }
)


def _impurity(**options: object) -> dict:
    return analyze_experiment(
        IMPURITY, response_column="y", model="y ~ x1 + x2 - 1", analysis_type="coefficients", coding="actual", **options
    )


def test_bhh_14_41_the_correlation_of_the_two_estimates() -> None:
    correlations = _impurity()["estimate_correlations"]
    assert round(correlations["x1"]["x2"], 4) == -0.8250
    assert correlations["x2"]["x1"] == correlations["x1"]["x2"]
    assert correlations["x1"]["x1"] == correlations["x2"]["x2"] == 1.0


def test_bhh_14_45_the_contour_of_the_90_percent_joint_region() -> None:
    """S = S_R (1 + p F / (n - p)): BHH print 1.3308 (1 + 2/4 (4.32)) = 4.2053, with F rounded from 4.3246."""
    region = _impurity(significance_level=0.10)["joint_confidence_region"]
    assert region["residual_sum_of_squares"] == pytest.approx(1.3308, abs=1e-4)
    f_point = stats.f.ppf(0.90, 2, 4)
    assert region["sum_of_squares"] == pytest.approx(region["residual_sum_of_squares"] * (1 + 2 / 4 * f_point))
    assert region["sum_of_squares"] == pytest.approx(4.2053, abs=5e-3)
    assert region["confidence_level"] == pytest.approx(0.90)


def test_an_orthogonal_design_has_uncorrelated_estimates() -> None:
    runs = pd.DataFrame({"A": [-1, 1, -1, 1, 0], "B": [-1, -1, 1, 1, 0], "y": [3.0, 5.0, 4.0, 7.0, 4.5]})
    correlations = analyze_experiment(runs, response_column="y", model="main_effects", analysis_type="coefficients")[
        "estimate_correlations"
    ]
    off_diagonal = [value for a, row in correlations.items() for b, value in row.items() if a != b]
    assert np.allclose(off_diagonal, 0.0)


def test_a_prior_sigma_leaves_the_correlations_and_bounds_the_region_by_chi_square() -> None:
    """With sigma known the region is S_R + sigma^2 chi^2 on p degrees of freedom."""
    given, known = _impurity(), _impurity(sigma=0.5)
    assert known["estimate_correlations"] == given["estimate_correlations"]
    region = known["joint_confidence_region"]
    expected = region["residual_sum_of_squares"] + 0.25 * stats.chi2.ppf(0.95, 2)
    assert region["sum_of_squares"] == pytest.approx(expected)


def test_a_saturated_design_has_correlations_but_no_region_without_sigma() -> None:
    runs = pd.DataFrame({"A": [-1, 1, -1, 1], "B": [-1, -1, 1, 1], "y": [3.0, 5.0, 4.0, 7.0]})
    result = analyze_experiment(runs, response_column="y", model="interactions", analysis_type="coefficients")
    assert math.isnan(result["joint_confidence_region"]["sum_of_squares"])
    assert set(result["estimate_correlations"]) == {"Intercept", "A", "B", "A:B"}
    known = analyze_experiment(runs, response_column="y", model="interactions", analysis_type="coefficients", sigma=1.0)
    assert math.isfinite(known["joint_confidence_region"]["sum_of_squares"])


def test_the_names_are_those_of_the_coefficient_table() -> None:
    runs = pd.DataFrame(
        {"T": [100, 150, 100, 150, 125, 125], "K": ["a", "a", "b", "b", "c", "c"], "y": [1.0, 2.0, 2.5, 3.1, 2.0, 2.2]}
    )
    result = analyze_experiment(runs, response_column="y", model="main_effects", analysis_type="coefficients")
    assert list(result["estimate_correlations"]) == [row["term"] for row in result["coefficients"]]
