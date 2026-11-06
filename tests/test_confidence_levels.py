"""Every function that takes a confidence level refuses one outside (0, 1), with the same message (#1132).

A level between -1 and 0 gave a reversed interval (``robust_regression`` returned (1.994, 1.947) for
``conflevel=-0.5``), a level of 0 an interval of zero width, and a level of 1 or more an infinite or
NaN one, each without a word. One shared check now refuses them all before any work is done.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import Comparisons, fit_across_blocks, least_squares_means
from process_improve.experiments.random_effects import intraclass_correlation_interval
from process_improve.experiments.visualization.plots.registry import create_plot
from process_improve.multivariate import PLS, MCUVScaler
from process_improve.regression.methods import OLS, multiple_linear_regression, robust_regression
from process_improve.regression.nonlinear import nonlinear_regression
from process_improve.sensory.analysis import product_means
from process_improve.tool_safety import ToolInputInvalidError
from process_improve.tool_spec import execute_tool_call
from process_improve.univariate.classical import ttest_one_sample, variance_interval
from process_improve.univariate.metrics import (
    confidence_interval,
    ttest_independent,
    ttest_independent_from_df,
    ttest_paired,
    ttest_paired_from_df,
)

X = np.array([1.0, 2, 3, 4, 5, 6, 7, 8])
Y = np.array([2.1, 3.9, 6.2, 7.8, 10.1, 12.2, 13.8, 16.1])
ONE_GROUP = pd.DataFrame({"g": ["a"] * 8, "y": Y})


def _pls() -> tuple[PLS, pd.DataFrame]:
    rng = np.random.default_rng(1132)
    frame = pd.DataFrame(rng.standard_normal((12, 3)), columns=["a", "b", "c"])
    scaled = MCUVScaler().fit_transform(frame)
    target = MCUVScaler().fit_transform((frame["a"] + 0.1 * rng.standard_normal(12)).to_frame("y"))
    return PLS(n_components=1).fit(scaled, target), scaled


def _across_blocks(level: float) -> pd.DataFrame:
    runs = pd.DataFrame(
        {
            "A": [-1, 1, -1, 1, 0, -1, 1, -1, 1, 0],
            "B": [-1, -1, 1, 1, 0, -1, -1, 1, 1, 0],
            "day": ["d1"] * 5 + ["d2"] * 5,
            "y": [3.1, 7.2, 1.0, 5.1, 4.0, 3.5, 7.4, 1.6, 5.5, 4.6],
        }
    )
    fit = fit_across_blocks(runs, ["y"], {"linear": "A + B"}, block="day")
    return fit.predict(pd.DataFrame({"A": [0.5], "B": [0.5]}), conf_level=level)


CALLS: dict[str, tuple[str, Callable[[float], Any]]] = {
    "robust_regression least squares": (
        "conflevel",
        lambda c: robust_regression(X, Y, conflevel=c, intervals="least_squares"),
    ),
    "robust_regression bootstrap": (
        "conflevel",
        lambda c: robust_regression(X, Y, conflevel=c, intervals="bootstrap", n_boot=9),
    ),
    "robust_regression default": ("conflevel", lambda c: robust_regression(X, Y, conflevel=c)),
    "multiple_linear_regression": ("conflevel", lambda c: multiple_linear_regression(X, Y, conflevel=c)),
    "OLS.fit": ("conflevel", lambda c: OLS(conflevel=c).fit(X, Y)),
    "OLS.prediction_interval": ("conflevel", lambda c: OLS().fit(X, Y).prediction_interval(4.5, conflevel=c)),
    "nonlinear_regression": (
        "conf_level",
        lambda c: nonlinear_regression(lambda x, slope: slope * x, X, Y, {"slope": 1.0}, conf_level=c),
    ),
    "ttest_one_sample": ("conflevel", lambda c: ttest_one_sample(Y, mu=8.0, conflevel=c)),
    "ttest_independent": ("conflevel", lambda c: ttest_independent(pd.Series(X), pd.Series(Y), conflevel=c)),
    "ttest_paired": ("conflevel", lambda c: ttest_paired(pd.Series(Y - X), conflevel=c)),
    # One group: there is no pair to compare, and the level was checked only inside the loop over the pairs.
    "ttest_independent_from_df": ("conflevel", lambda c: ttest_independent_from_df(ONE_GROUP, "g", "y", conflevel=c)),
    "ttest_paired_from_df": ("conflevel", lambda c: ttest_paired_from_df(ONE_GROUP, "g", "y", conflevel=c)),
    "confidence_interval": ("conflevel", lambda c: confidence_interval(pd.DataFrame({"y": Y}), "y", conflevel=c)),
    "variance_interval": ("conflevel", lambda c: variance_interval(2.0, 7, conflevel=c)),
    "product_means": (
        "conf_level",
        lambda c: product_means(
            pd.DataFrame({"product": ["p"] * 3, "attribute": ["sweet"] * 3, "score": [4.0, 5.0, 6.0]}),
            conf_level=c,
        ),
    ),
    "least_squares_means": (
        "conflevel",
        lambda c: least_squares_means(
            pd.DataFrame({"t": ["a", "a", "b", "b", "c", "c"], "y": [1.0, 2.0, 3.0, 4.5, 5.0, 7.0]}),
            "y ~ t",
            "t",
            Comparisons(conflevel=c),
        ),
    ),
    "intraclass_correlation_interval": (
        "conflevel",
        lambda c: intraclass_correlation_interval(29.73, 1.90, 3, 4, conflevel=c),
    ),
    "BlockedFit.predict": ("conf_level", _across_blocks),
    "create_plot": ("confidence_level", lambda c: create_plot("pareto", confidence_level=c)),
    "PLS.selectivity_ratio": ("conf_level", lambda c: _pls()[0].selectivity_ratio(_pls()[1], conf_level=c)),
}


@pytest.mark.parametrize("level", [-0.5, 0.0, 1.0, 1.5, 95.0, np.inf, np.nan], ids=str)
@pytest.mark.parametrize("call", list(CALLS), ids=str)
def test_a_confidence_level_outside_0_1_is_refused(call: str, level: float) -> None:
    """The level is refused with a message that names the parameter and the value given."""
    name, function = CALLS[call]
    with pytest.raises(ValueError, match=rf"^{name} must lie in \(0, 1\); got {level}\.$"):
        function(level)


@pytest.mark.filterwarnings("ignore:process_improve.regression.robust_regression's default:FutureWarning")
@pytest.mark.parametrize("call", list(CALLS), ids=str)
def test_a_confidence_level_inside_0_1_is_accepted(call: str) -> None:
    """The check refuses only levels outside (0, 1): each call runs at 0.9."""
    _name, function = CALLS[call]
    function(0.9)


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("robust_regression", {"x": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 4.1, 5.9, 8.2], "confidence_level": 1.5}),
        ("visualize_doe", {"plot_type": "pareto", "confidence_level": 95}),
    ],
)
def test_a_tool_refuses_a_confidence_level_outside_0_1_at_its_schema(tool: str, arguments: dict) -> None:
    """A hosted caller is refused by the input contract, before any worker starts."""
    with pytest.raises(ToolInputInvalidError, match="confidence_level"):
        execute_tool_call(tool, arguments)


@pytest.mark.parametrize("function", [ttest_independent_from_df, ttest_paired_from_df])
def test_a_pairwise_test_on_one_group_refuses_an_unknown_correction(function: Callable[..., pd.DataFrame]) -> None:
    """The correction was checked only once there were comparisons to correct (#1169)."""
    with pytest.raises(ValueError, match=r"^correction must be one of 'holm', 'bh', or None; got 'bonferroni'\.$"):
        function(ONE_GROUP, "g", "y", correction="bonferroni")
