"""The overall F test and the likelihood criteria in ``analyze_experiment``'s model summary."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from process_improve.experiments.analysis import analyze_experiment

# Montgomery's 2^2 with three replicates (Table 6.1): reactant concentration A, catalyst B, yield.
RUNS = pd.DataFrame(
    {
        "A": [-1, 1, -1, 1] * 3,
        "B": [-1, -1, 1, 1] * 3,
        "y": [28, 36, 18, 31, 25, 32, 19, 30, 27, 32, 23, 29],
    }
)


def _summary(model: str) -> dict:
    return analyze_experiment(RUNS, response_column="y", model=model, analysis_type="coefficients", coding="actual")[
        "model_summary"
    ]


def test_the_overall_f_test_compares_the_model_with_the_intercept_alone():
    found = _summary("y ~ A + B + A:B")
    y = RUNS["y"].to_numpy(dtype=float)
    x = np.column_stack([np.ones(12), RUNS["A"], RUNS["B"], RUNS["A"] * RUNS["B"]])
    fitted = x @ np.linalg.lstsq(x, y, rcond=None)[0]
    regression, residual = float(((fitted - y.mean()) ** 2).sum()), float(((y - fitted) ** 2).sum())
    f = (regression / 3) / (residual / 8)
    assert found["f_statistic"] == pytest.approx(f)
    assert found["f_p_value"] == pytest.approx(stats.f.sf(f, 3, 8))


def test_the_aics_differ_by_a_constant_and_rank_models_alike():
    full, reduced = _summary("y ~ A + B + A:B"), _summary("y ~ A + B")
    n = 12
    for found, p in ((full, 4), (reduced, 3)):
        rss = found["mse_residual"] * found["df_residual"]
        assert found["aic_step"] == pytest.approx(n * np.log(rss / n) + 2 * p)  # R's step and extractAIC
        assert found["log_likelihood"] == pytest.approx(-n / 2 * (np.log(2 * np.pi * rss / n) + 1))
        assert found["aic"] == pytest.approx(-2 * found["log_likelihood"] + 2 * (p + 1))  # R's AIC
        assert found["bic"] == pytest.approx(-2 * found["log_likelihood"] + np.log(n) * (p + 1))
        assert found["aic"] - found["aic_step"] == pytest.approx(n * (np.log(2 * np.pi) + 1) + 2)
    assert (full["aic"] < reduced["aic"]) == (full["aic_step"] < reduced["aic_step"])


def test_without_an_intercept_the_f_test_is_against_zero():
    """With no intercept, and columns that do not span one, R tests the model against zero."""
    found = _summary("y ~ A + B - 1")
    y = RUNS["y"].to_numpy(dtype=float)
    x = RUNS[["A", "B"]].to_numpy(dtype=float)
    fitted = x @ np.linalg.lstsq(x, y, rcond=None)[0]
    f = (float(fitted @ fitted) / 2) / (float(((y - fitted) ** 2).sum()) / 10)
    assert found["f_statistic"] == pytest.approx(f)


def test_the_corrected_aic_adds_the_small_sample_term_and_is_none_where_it_is_undefined():
    found, n, k = _summary("y ~ A + B + A:B"), 12, 5  # four coefficients and the error variance
    assert found["aicc"] == pytest.approx(found["aic"] + 2 * k * (k + 1) / (n - k - 1))
    few = analyze_experiment(RUNS.iloc[:4], response_column="y", model="y ~ A + B", coding="actual")
    assert few["model_summary"]["aicc"] is None  # n - k - 1 = 4 - 4 - 1 < 0
