"""Tests for the tabular CUSUM (cumulative sum) control chart variant."""

import numpy as np
import pytest

from process_improve.monitoring.control_charts import ControlChart, _one_sided_cusum

# Montgomery's tabular CUSUM example (Introduction to Statistical Quality Control, Example 9.1):
# 20 samples from N(10, 1), then 10 after the mean shifts to 11, charted with k = 0.5 and h = 5.
MONTGOMERY = np.array(
    [
        *(9.45, 7.99, 9.29, 11.66, 12.16, 10.18, 8.04, 11.46, 9.20, 10.34, 9.03, 11.47, 10.51, 9.40, 10.08),
        *(9.37, 10.62, 10.31, 8.52, 10.84, 10.90, 9.33, 12.29, 11.50, 10.60, 11.08, 10.38, 11.62, 11.31, 10.52),
    ]
)
# The upper sum he tabulates for periods 1 to 29.
MONTGOMERY_UPPER = [
    *(0.0, 0.0, 0.0, 1.16, 2.82, 2.50, 0.04, 1.00, 0.0, 0.0, 0.0, 0.97, 0.98, 0.0, 0.0),
    *(0.0, 0.12, 0.0, 0.0, 0.34, 0.74, 0.0, 1.79, 2.79, 2.89, 3.47, 3.35, 4.47, 5.28),
]


def fit_cusum(y: np.ndarray, **kwargs: float) -> ControlChart:
    """Return a CUSUM chart fitted to ``y``."""
    chart = ControlChart(variant="cusum")
    chart.calculate_limits(y, **kwargs)
    return chart


def shifted_series(shift: float = 2.0) -> np.ndarray:
    """Sixty samples from N(50, 2), then forty after the mean moves by ``shift``."""
    rng = np.random.default_rng(1)
    return np.concatenate([rng.normal(50, 2, 60), rng.normal(50 + shift, 2, 40)])


def run_length(deviations: np.ndarray, k: float, h: float) -> int:
    """Return the sample, counted from 1, of the first alarm raised by either one-sided sum."""
    upward, downward = _one_sided_cusum(deviations, k, h)[1], _one_sided_cusum(-deviations, k, h)[1]
    return min(alarms[0][0] for alarms in (upward, downward) if alarms) + 1


def test_reproduces_the_published_example() -> None:
    """The upper sum matches Montgomery's table, alarms at period 29, and dates the shift and its size."""
    chart = fit_cusum(MONTGOMERY, target=10.0, s=1.0)

    assert chart.df["cusum_upper"].iloc[:29].round(2).tolist() == MONTGOMERY_UPPER
    assert chart.idx_outside_3S == [28]  # period 29
    alarm = chart.cusum_alarms.loc[28]
    assert alarm["direction"] == "up"
    assert alarm["shift_start"] == 22  # period 23: the mean moved between periods 22 and 23
    assert alarm["new_mean"] == pytest.approx(10 + 0.5 + 5.28 / 7, abs=0.005)  # 11.25, as he estimates


def test_the_sum_restarts_after_an_alarm() -> None:
    """After an alarm the sum that raised it restarts from zero, so each alarm is a separate detection.

    Montgomery's table keeps accumulating instead (5.30 at period 30).
    """
    chart = fit_cusum(MONTGOMERY, target=10.0, s=1.0)
    assert chart.df["cusum_upper"].iloc[29] == pytest.approx(10.52 - 10.5)


@pytest.mark.parametrize(("h", "published"), [(5.0, 10.4), (4.0, 8.38)])
def test_run_length_after_a_one_sigma_shift_matches_the_published_value(h: float, published: float) -> None:
    """With k = 0.5 a one-sigma shift is detected after 10.4 samples on average for h = 5, and 8.38 for h = 4.

    These are the published average run lengths of the tabular CUSUM (Montgomery, chapter 9);
    4000 simulated runs give a mean with a standard error near 0.1.
    """
    rng = np.random.default_rng(0)
    runs = [run_length(rng.normal(1.0, 1.0, 80), k=0.5, h=h) for _ in range(4000)]
    assert np.mean(runs) == pytest.approx(published, abs=0.25)


@pytest.mark.slow
def test_in_control_run_length_matches_the_published_value() -> None:
    """With k = 0.5 and h = 5, an in-control process raises a false alarm once in 465 samples on average."""
    rng = np.random.default_rng(0)
    runs = [run_length(rng.normal(0.0, 1.0, 4000), k=0.5, h=5.0) for _ in range(600)]
    assert np.mean(runs) == pytest.approx(465, rel=0.1)


def test_dates_a_shift_and_estimates_the_new_mean() -> None:
    """A one-sigma shift at sample 60 is detected at 73, dated to sample 60, and sized close to the true 52."""
    chart = fit_cusum(shifted_series(), target=50.0, s=2.0)

    first = chart.cusum_alarms.iloc[0]
    assert chart.cusum_alarms.index[0] == 73
    assert (first["direction"], first["shift_start"]) == ("up", 60)
    assert first["new_mean"] == pytest.approx(51.78, abs=0.005)
    assert chart.idx_outside_3S == chart.cusum_alarms.index.tolist()


def test_a_downward_shift_raises_a_downward_alarm() -> None:
    """The lower sum catches a shift down, and estimates a new mean below the target."""
    chart = fit_cusum(shifted_series(shift=-2.0), target=50.0, s=2.0)

    assert set(chart.cusum_alarms["direction"]) == {"down"}
    assert chart.cusum_alarms.index.min() >= 60
    assert (chart.cusum_alarms["new_mean"] < 50.0).all()


def test_a_missing_value_holds_the_sums_and_is_not_counted() -> None:
    """A gap holds both sums, is never flagged, and does not count towards the run that dates a shift."""
    chart = fit_cusum(np.array([0.0, 3.0, np.nan, 3.0, 3.0]), target=0.0, s=1.0)

    assert chart.df["cusum_upper"].tolist() == [0.0, 2.5, 2.5, 5.0, 7.5]
    alarm = chart.cusum_alarms.loc[4]
    assert alarm["shift_start"] == 1
    assert alarm["new_mean"] == pytest.approx(0.5 + 7.5 / 3)  # three observations in the run: 3.0
    assert chart.idx_outside_3S == [4]


def test_defaults_and_estimates() -> None:
    """Without tuning parameters k = 0.5 and h = 5; without a target and s both come from the data."""
    y = shifted_series()
    chart = fit_cusum(y)

    assert (chart.k, chart.h) == (0.5, 5.0)
    assert chart.target == pytest.approx(np.median(y))
    assert chart.s == pytest.approx(1.4826 * np.median(np.abs(y - np.median(y))))


@pytest.mark.parametrize("parameters", [{"k": -0.1}, {"h": 0.0}, {"h": np.inf}])
def test_rejects_invalid_parameters(parameters: dict[str, float]) -> None:
    """The reference value cannot be negative, and the decision interval must be positive and finite."""
    with pytest.raises(ValueError, match="k >= 0 and h > 0"):
        fit_cusum(shifted_series(), **parameters)


@pytest.mark.parametrize(
    ("variant", "parameter", "accepted"),
    [("cusum", "ld_1", r"\['h', 'k'\]"), ("ewma", "k", r"\['ld_1'\]"), ("hw", "h", r"\['ld_1', 'ld_2'\]")],
)
def test_each_variant_accepts_only_its_own_parameters(variant: str, parameter: str, accepted: str) -> None:
    """A tuning parameter of another chart is rejected, with the ones this chart accepts named."""
    chart = ControlChart(variant=variant)
    with pytest.raises(ValueError, match=rf"unexpected keyword argument.*{accepted}"):
        chart.calculate_limits(shifted_series(), **{parameter: 0.3})


def test_an_unknown_style_raises_instead_of_charting() -> None:
    """A style other than 'robust' or 'regular' estimates no target or s, so nothing is charted."""
    chart = ControlChart(variant="cusum", style="Robust")  # the style is case-sensitive
    with pytest.raises(ValueError, match="could not be estimated"):
        chart.calculate_limits(shifted_series())
    assert "cusum_upper" not in chart.df.columns


def test_refitting_a_chart_matches_a_fresh_chart() -> None:
    """Nothing from a first fit (target, s, k, h, alarms) leaks into a second fit on the same chart."""
    y = shifted_series()
    reused = ControlChart(variant="CUSUM")  # the variant is case-insensitive
    reused.calculate_limits(y, target=50.0, s=2.0, k=1.0, h=4.0)
    reused.calculate_limits(y[:50])

    fresh = fit_cusum(y[:50])
    assert (reused.target, reused.s, reused.k, reused.h) == (fresh.target, fresh.s, 0.5, 5.0)
    assert reused.cusum_alarms.equals(fresh.cusum_alarms)
