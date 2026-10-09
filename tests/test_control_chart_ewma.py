"""Tests for the EWMA (exponentially weighted moving average) control chart variant."""

import pathlib

import numpy as np
import pandas as pd
import pytest

from process_improve.monitoring.control_charts import ControlChart
from process_improve.tool_spec import execute_tool_call

RUBBER_COLOUR = pathlib.Path(__file__).parents[1] / "src" / "process_improve" / "datasets" / "monitoring"


def fit_ewma(y: np.ndarray | pd.Series, **kwargs: float) -> ControlChart:
    """Return an EWMA chart fitted to ``y``."""
    chart = ControlChart(variant="ewma")
    chart.calculate_limits(y, **kwargs)
    return chart


def shifted_series() -> np.ndarray:
    """Sixty in-control samples from N(50, 2), then forty after a one-sigma shift to N(52, 2)."""
    rng = np.random.default_rng(1)
    return np.concatenate([rng.normal(50, 2, 60), rng.normal(52, 2, 40)])


def test_statistic_and_limits_follow_the_textbook_formulas() -> None:
    """z_t = w y_t + (1 - w) z_(t-1) from z_0 = target; limits 3 s sqrt(w / (2 - w) (1 - (1 - w)^(2t)))."""
    chart = fit_ewma(np.array([1.0, 2.0, 3.0, 4.0]), target=0.0, s=1.0, ld_1=0.5)

    assert chart.df["ewma"].tolist() == pytest.approx([0.5, 1.25, 2.125, 3.0625])
    t = np.arange(1, 5)
    half_width = 3.0 * np.sqrt(0.5 / 1.5 * (1.0 - 0.25**t))
    assert chart.df["ewma_ucl"].to_numpy() == pytest.approx(half_width)
    assert chart.df["ewma_lcl"].to_numpy() == pytest.approx(-half_width)
    # 2.125 > 1.72 and 3.06 > 1.73: the statistic, not the raw value, crosses the limit.
    assert chart.idx_outside_3S == [2, 3]


def test_limits_widen_to_their_steady_state() -> None:
    """The limits start narrow and grow monotonically to target +/- 3 s sqrt(w / (2 - w))."""
    chart = fit_ewma(shifted_series(), target=50.0, s=2.0, ld_1=0.2)

    ucl = chart.df["ewma_ucl"].to_numpy()
    assert np.all(np.diff(ucl) >= 0)
    assert ucl[0] < ucl[-1]
    # With w = 0.2 the steady-state half-width is exactly s, because 3 sqrt(0.2 / 1.8) = 1.
    assert ucl[-1] == pytest.approx(52.0, abs=1e-9)
    assert chart.df["ewma_lcl"].iloc[-1] == pytest.approx(48.0, abs=1e-9)


def test_weight_of_one_is_the_shewhart_individuals_chart() -> None:
    """With w = 1 the statistic is the observation and the limits are target +/- 3 s."""
    y = shifted_series()
    ewma = fit_ewma(y, ld_1=1.0)
    shewhart = ControlChart(variant="xbar.no.subgroup")
    shewhart.calculate_limits(y)

    assert ewma.df["ewma"].to_numpy() == pytest.approx(y)
    assert ewma.df["ewma_ucl"].to_numpy() == pytest.approx(ewma.target + 3 * ewma.s)
    assert (ewma.target, ewma.s) == (shewhart.target, shewhart.s)
    assert ewma.idx_outside_3S == shewhart.idx_outside_3S


def test_detects_a_one_sigma_shift_that_the_shewhart_chart_misses() -> None:
    """A sustained one-sigma shift at sample 60 alarms the EWMA chart, not the Shewhart chart."""
    y = shifted_series()
    ewma = fit_ewma(y, target=50.0, s=2.0, ld_1=0.2)
    shewhart = fit_ewma(y, target=50.0, s=2.0, ld_1=1.0)

    assert ewma.idx_outside_3S[0] == 74  # 14 samples after the shift, and no false alarm before it
    assert len(ewma.idx_outside_3S) == 10
    assert shewhart.idx_outside_3S == []


def test_missing_value_carries_the_statistic_forward_and_is_never_flagged() -> None:
    """A gap holds the statistic and its limits, and is not reported even when the statistic is outside."""
    chart = fit_ewma(np.array([0.0, 0.0, 10.0, np.nan, 0.0]), target=0.0, s=1.0, ld_1=0.5)

    assert chart.df["ewma"].tolist() == pytest.approx([0.0, 0.0, 5.0, 5.0, 2.5])
    # The gap does not count as an observation, so the limit at index 3 equals the one at index 2.
    ucl = chart.df["ewma_ucl"].to_numpy()
    assert ucl[3] == ucl[2]
    assert ucl[4] == pytest.approx(3.0 * np.sqrt(0.5 / 1.5 * (1.0 - 0.25**4)))
    assert chart.idx_outside_3S == [2, 4]


def test_leading_missing_values_start_from_the_target() -> None:
    """Before the first observation the statistic sits on the target with zero-width limits, unflagged."""
    chart = fit_ewma(np.array([np.nan, np.nan, 1.0, 1.0]), target=0.0, s=1.0, ld_1=0.5)

    assert chart.df["ewma"].tolist() == pytest.approx([0.0, 0.0, 0.5, 0.75])
    assert chart.df["ewma_ucl"].iloc[:2].tolist() == [0.0, 0.0]
    assert chart.idx_outside_3S == []


@pytest.mark.parametrize(
    ("style", "expected_target", "expected_s"),
    [
        ("robust", np.median, lambda y: 1.4826 * np.median(np.abs(y - np.median(y)))),
        ("regular", np.mean, lambda y: np.std(y, ddof=1)),
    ],
)
def test_estimates_target_and_s_as_the_shewhart_chart_does(style, expected_target, expected_s) -> None:
    """Without a given target and s, both come from the data, robustly or not."""
    y = shifted_series()
    chart = ControlChart(variant="ewma", style=style)
    chart.calculate_limits(y)

    assert chart.target == pytest.approx(expected_target(y))
    assert chart.s == pytest.approx(expected_s(y))


def test_a_given_target_or_s_is_kept_and_only_the_other_is_estimated() -> None:
    """Each of target and s is used when given and estimated only when missing."""
    y = shifted_series()

    only_target = fit_ewma(y, target=50.0)
    assert only_target.target == 50.0
    assert only_target.s == pytest.approx(1.4826 * np.median(np.abs(y - np.median(y))))

    only_s = fit_ewma(y, s=2.0)
    assert only_s.target == pytest.approx(np.median(y))
    assert only_s.s == 2.0


def test_default_weight_is_0_2() -> None:
    """Without ``ld_1`` the weight is 0.2, and the weight used is recorded on the chart."""
    assert fit_ewma(shifted_series()).ld_1 == 0.2


@pytest.mark.parametrize("weight", [0.0, -0.1, 1.5])
def test_rejects_a_weight_outside_zero_to_one(weight: float) -> None:
    """The weight must satisfy 0 < ld_1 <= 1."""
    with pytest.raises(ValueError, match="0 < ld_1 <= 1"):
        fit_ewma(shifted_series(), ld_1=weight)


def test_rejects_the_holt_winters_trend_weight() -> None:
    """An EWMA chart has a single weight; ``ld_2`` belongs to the Holt-Winters chart."""
    with pytest.raises(ValueError, match="Holt-Winters chart only"):
        fit_ewma(shifted_series(), ld_2=0.3)


def test_refitting_a_chart_matches_a_fresh_chart() -> None:
    """Nothing from a first fit (target, s, weight) leaks into a second fit on the same chart."""
    y = shifted_series()
    reused = ControlChart(variant="EWMA")  # the variant is case-insensitive
    reused.calculate_limits(y, target=50.0, s=2.0, ld_1=0.5)
    reused.calculate_limits(y[:40])

    fresh = fit_ewma(y[:40])
    assert (reused.target, reused.s, reused.ld_1) == (fresh.target, fresh.s, 0.2)
    pd.testing.assert_frame_equal(reused.df, fresh.df)


@pytest.mark.dataset
def test_rubber_colour_data_matches_pandas_ewm() -> None:
    """On real data, the statistic equals pandas' ``ewm`` started at the target, and alarms where Shewhart does not.

    The target 238.78 and standard deviation 10.43234 are those of R's ``qcc(type="xbar.one")``
    for the same data, used as known Phase I values.
    """
    y = pd.read_csv(RUBBER_COLOUR / "rubber-colour.csv")["Colour"]
    chart = fit_ewma(y, target=238.78, s=10.43234, ld_1=0.2)

    started_at_target = pd.concat([pd.Series([238.78]), y], ignore_index=True)
    reference = started_at_target.ewm(alpha=0.2, adjust=False).mean().iloc[1:].to_numpy()
    assert chart.df["ewma"].to_numpy() == pytest.approx(reference, abs=1e-9)
    assert chart.df["ewma_ucl"].iloc[-1] == pytest.approx(238.78 + 10.43234)
    assert chart.idx_outside_3S == [69]
    assert fit_ewma(y, target=238.78, s=10.43234, ld_1=1.0).idx_outside_3S == []


def test_control_chart_tool_reports_ewma_limits_and_statistic() -> None:
    """The tool's EWMA limits are the steady-state ones, and it reports the statistic at each alarm."""
    rng = np.random.default_rng(9)
    # A long in-control history, then a sustained one-sigma shift over the last 20 samples.
    values = [float(v) for v in np.concatenate([rng.normal(50, 2, 80), rng.normal(52, 2, 20)])]

    result = execute_tool_call("control_chart", {"values": values, "chart_type": "ewma"})
    shewhart = execute_tool_call("control_chart", {"values": values, "chart_type": "shewhart"})

    assert "error" not in result
    assert result["ewma_weight"] == 0.2
    # With w = 0.2 the steady-state half-width is s, a third of the Shewhart chart's 3 s.
    half_width = result["upper_control_limit"] - result["target"]
    assert half_width == pytest.approx(result["spread"])
    assert half_width == pytest.approx((shewhart["upper_control_limit"] - shewhart["target"]) / 3)
    # Target and s come from the same data, yet the EWMA chart flags the shift and the Shewhart chart does not.
    assert result["out_of_control_indices"] == list(range(91, 100))  # from 11 samples after the shift
    assert shewhart["out_of_control_indices"] == []
    assert len(result["out_of_control_ewma"]) == result["n_out_of_control"]
    assert all(v > result["upper_control_limit"] for v in result["out_of_control_ewma"])
