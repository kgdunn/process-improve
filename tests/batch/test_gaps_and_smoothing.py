"""Gap filling and trajectory smoothing for batch data (#200).

The point of both functions is what they do *better* than the one-liner they
replace, so the tests compare against that one-liner rather than against
themselves: `fill_gaps` against `bfill().ffill()`, and `smooth_trajectories`
against filtering a concatenated frame and against a moving average.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.signal import savgol_coeffs, savgol_filter

from process_improve.batch import fill_gaps, smooth_trajectories


@pytest.fixture
def ramp_with_a_gap() -> dict[str, pd.DataFrame]:
    """Return a linear temperature ramp with two samples missing out of the middle."""
    return {"B1": pd.DataFrame({"T": [20.0, 25.0, np.nan, np.nan, 40.0, 45.0]})}


class TestFillGaps:
    """`fill_gaps` against the `bfill().ffill()` it replaces."""

    def test_a_ramp_is_interpolated_not_held_flat(self, ramp_with_a_gap: dict[str, pd.DataFrame]) -> None:
        """Holding the last value flat manufactures a step the feature layer reads as real.

        `f_slope` and `f_rupture` measure exactly the shape that a
        forward-filled plateau invents, so this is the difference between a
        feature describing the process and a feature describing the fill.
        """
        filled, _report = fill_gaps(ramp_with_a_gap)
        assert filled["B1"]["T"].tolist() == [20.0, 25.0, 30.0, 35.0, 40.0, 45.0]

        held_flat = ramp_with_a_gap["B1"]["T"].ffill()
        assert held_flat.tolist() == [20.0, 25.0, 25.0, 25.0, 40.0, 45.0]
        assert held_flat.diff().max() == 15.0, "a step the raw ramp never takes"
        slopes = filled["B1"]["T"].diff().dropna().to_numpy()
        assert slopes == pytest.approx(slopes[0]), "the filled ramp keeps one constant slope"

    def test_a_gap_longer_than_the_limit_stays_missing(self) -> None:
        """Two missing samples and two hundred are not the same claim about the process."""
        long_gap = {"B1": pd.DataFrame({"T": [1.0, *[np.nan] * 8, 10.0]})}
        filled, report = fill_gaps(long_gap, limit=5)

        assert filled["B1"]["T"].isna().sum() == 8
        assert report.loc["B1", "T"] == "0/8"

        everything, report_all = fill_gaps(long_gap, limit=None)
        assert everything["B1"]["T"].isna().sum() == 0
        assert report_all.loc["B1", "T"] == "8/0"

    def test_short_and_long_gaps_in_one_column_are_judged_separately(self) -> None:
        """The decision is made on the original gaps, not on what an earlier pass left."""
        mixed = {"B1": pd.DataFrame({"T": [0.0, np.nan, 2.0, *[np.nan] * 6, 9.0]})}
        filled, report = fill_gaps(mixed, limit=3)

        assert filled["B1"]["T"].iloc[1] == pytest.approx(1.0), "the one-sample gap is filled"
        assert filled["B1"]["T"].iloc[3:9].isna().all(), "the six-sample gap is not"
        assert report.loc["B1", "T"] == "1/6"

    def test_the_edges_are_left_alone_by_default(self) -> None:
        """Filling the start of a batch reads backwards from the future."""
        edges = {"B1": pd.DataFrame({"T": [np.nan, np.nan, 5.0, 6.0, np.nan]})}

        left, _ = fill_gaps(edges)
        assert left["B1"]["T"].isna().tolist() == [True, True, False, False, True]

        extended, _ = fill_gaps(edges, edge="nearest")
        assert extended["B1"]["T"].tolist() == [5.0, 5.0, 5.0, 6.0, 6.0]

    def test_pchip_does_not_invent_an_impossible_value(self) -> None:
        """A plain cubic overshoots below zero on a rising concentration; pchip cannot.

        Shape preservation is the reason to offer pchip at all: an interpolated
        negative concentration is not a smoother answer, it is a wrong one.
        """
        rising = pd.Series([0.0, 0.02, np.nan, np.nan, 0.9, 3.0, 6.0])
        assert rising.interpolate(method="cubic", limit_area="inside").min() < -0.1

        filled, _ = fill_gaps({"B1": pd.DataFrame({"C": rising})}, method="pchip")
        assert filled["B1"]["C"].min() >= 0.0
        assert filled["B1"]["C"].is_monotonic_increasing

    def test_non_numeric_columns_are_never_touched(self) -> None:
        """A phase label travelling alongside the trajectory is not a measurement."""
        mixed = {"B1": pd.DataFrame({"T": [1.0, np.nan, 3.0], "phase": ["heat", None, "cool"]})}
        filled, report = fill_gaps(mixed)

        assert filled["B1"]["T"].tolist() == [1.0, 2.0, 3.0]
        assert filled["B1"]["phase"].isna().sum() == 1
        assert "phase" not in report.columns

    def test_the_report_accounts_for_every_batch(self) -> None:
        """The report is the deliverable: how much was measured and how much invented."""
        batches = {
            "clean": pd.DataFrame({"T": [1.0, 2.0, 3.0, 4.0]}),
            "patchy": pd.DataFrame({"T": [1.0, np.nan, 3.0, 4.0]}),
            "broken": pd.DataFrame({"T": [1.0, np.nan, np.nan, np.nan]}),
        }
        _filled, report = fill_gaps(batches, limit=1, edge="leave")

        assert list(report.index) == ["clean", "patchy", "broken"]
        assert report.loc["clean", "T"] == "0/0"
        assert report.loc["patchy", "T"] == "1/0"
        assert report.loc["broken", "T"] == "0/3", "a trailing gap is not filled, whatever its length"
        assert report["n_samples"].tolist() == [4, 4, 4]

    def test_an_all_missing_column_survives(self) -> None:
        """Nothing to interpolate between, so nothing is invented."""
        empty = {"B1": pd.DataFrame({"T": [np.nan] * 5})}
        filled, report = fill_gaps(empty, limit=None)
        assert filled["B1"]["T"].isna().all()
        assert report.loc["B1", "T"] == "0/5"

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [({"limit": 0}, "positive number of samples"), ({"columns": ["nope"]}, "not in the batch data")],
    )
    def test_bad_arguments_are_rejected(
        self, ramp_with_a_gap: dict[str, pd.DataFrame], kwargs: dict, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            fill_gaps(ramp_with_a_gap, **kwargs)

    def test_no_batches_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="At least one batch"):
            fill_gaps({})


class TestSmoothTrajectories:
    """`smooth_trajectories` against the ways of doing this that go wrong quietly."""

    def test_smoothing_never_crosses_a_batch_boundary(self) -> None:
        """The bug this function exists to make impossible.

        Two batches, each perfectly constant. Filtering them as one concatenated
        series drags a ramp across the join; smoothing per batch leaves both
        exactly where they were.
        """
        batches = {
            "B1": pd.DataFrame({"y": np.zeros(40)}),
            "B2": pd.DataFrame({"y": np.full(40, 100.0)}),
        }
        out = smooth_trajectories(batches, window=11, polyorder=2)

        assert out["B1"]["y"].to_numpy() == pytest.approx(0.0, abs=1e-9)
        assert out["B2"]["y"].to_numpy() == pytest.approx(100.0, abs=1e-9)

        concatenated = savgol_filter(np.r_[batches["B1"]["y"], batches["B2"]["y"]], 11, 2)
        assert abs(concatenated[39]) > 30.0, "the naive route pulls the end of B1 up"
        assert concatenated[40] < 70.0, "and the start of B2 down"

    def test_savgol_keeps_a_peak_that_a_moving_average_flattens(self) -> None:
        """Peak height and area are what gets measured next, so the smoother must keep them.

        Savitzky-Golay fits a local polynomial, which reproduces a quadratic peak
        exactly up to its order; a moving average of the same width cannot.
        """
        rng = np.random.default_rng(3)
        t = np.linspace(-4, 4, 121)
        peak = np.exp(-(t**2))
        noisy = peak + rng.normal(scale=0.02, size=t.size)

        filtered = smooth_trajectories({"B": pd.DataFrame({"y": noisy})}, window=21, polyorder=2)["B"]["y"]
        moving_average = pd.Series(noisy).rolling(21, center=True, min_periods=1).mean()

        assert abs(filtered.max() - 1.0) < 0.03
        assert abs(moving_average.max() - 1.0) > 0.10
        assert abs(filtered.max() - 1.0) < abs(moving_average.max() - 1.0) / 5

    def test_savgol_removes_exactly_the_noise_its_coefficients_predict(self) -> None:
        """Noise reduction is checked against theory, not against a round number.

        A linear filter multiplies the noise standard deviation by the root sum
        of squares of its coefficients. For Savitzky-Golay that quantity is
        known in closed form, so the filter can be checked against what it is
        supposed to do rather than against an arbitrary threshold that would
        pass for a filter doing something subtly different.
        """
        rng = np.random.default_rng(0)
        clean = np.linspace(0, 10, 200) ** 0.5
        noisy = clean + rng.normal(scale=0.05, size=200)

        out = smooth_trajectories({"B": pd.DataFrame({"y": noisy})}, window=11, polyorder=2)["B"]["y"].to_numpy()

        predicted = float(np.sqrt(np.sum(savgol_coeffs(11, 2) ** 2)))
        assert predicted == pytest.approx(0.4555, abs=1e-3)

        # Away from the ends, and away from the square root's cusp at zero,
        # where the remaining curvature is bias rather than noise.
        interior = slice(20, 180)
        observed = float(np.std((out - clean)[interior]) / np.std((noisy - clean)[interior]))
        assert observed == pytest.approx(predicted, rel=0.25)

    def test_lowess_rejects_spikes_that_savgol_smears(self) -> None:
        """The reason to offer both. Savitzky-Golay spreads each spike over its window."""
        rng = np.random.default_rng(5)
        base = np.linspace(0, 5, 100)
        spiked = base + rng.normal(scale=0.05, size=100)
        spiked[[30, 31, 70]] += 20.0
        data = {"B": pd.DataFrame({"y": spiked})}
        away_from_spikes = np.setdiff1d(np.arange(100), [29, 30, 31, 32, 69, 70, 71])

        robust = smooth_trajectories(data, method="lowess", frac=0.2)["B"]["y"].to_numpy()
        local_fit = smooth_trajectories(data, method="savgol", window=11, polyorder=2)["B"]["y"].to_numpy()

        def rmse(values: np.ndarray) -> float:
            return float(np.sqrt(np.mean((values[away_from_spikes] - base[away_from_spikes]) ** 2)))

        assert rmse(robust) < 0.1
        assert rmse(local_fit) > 0.5
        assert rmse(robust) < rmse(local_fit) / 10

    def test_lowess_reports_when_its_robustness_collapses(self) -> None:
        """A noiseless trajectory with spikes drives the robustness scale to zero.

        LOWESS weights by ``6 * median(|residual|)``. With the local fits exact
        away from the spikes that median is zero, every weight degenerates, and
        statsmodels hands the column straight back: finite, right length, still
        spiked. Nothing else in the pipeline would notice.
        """
        noiseless = np.linspace(0, 5, 100)
        spiked = noiseless.copy()
        spiked[[30, 31, 70]] += 20.0

        with pytest.warns(UserWarning, match="robustness weights collapsed"):
            out = smooth_trajectories({"B1": pd.DataFrame({"y": spiked})}, method="lowess", frac=0.2)["B1"]["y"]

        assert not np.allclose(out.to_numpy(), spiked), "it must not hand the column back untouched"
        assert out.iloc[30] < spiked[30] / 2, "the spike is at least smoothed, even if not rejected"

    def test_a_batch_shorter_than_the_window_is_smoothed_and_reported(self) -> None:
        """Batches differ in length before alignment, so a short one must not fail."""
        rng = np.random.default_rng(1)
        batches = {
            "long": pd.DataFrame({"y": rng.normal(size=50)}),
            "tiny": pd.DataFrame({"y": rng.normal(size=7)}),
        }
        with pytest.warns(UserWarning, match="shorter than window"):
            out = smooth_trajectories(batches, window=21, polyorder=2)

        assert len(out["tiny"]) == 7
        assert out["tiny"]["y"].notna().all()

    def test_missing_samples_stay_missing(self) -> None:
        """Neither filter can see through a gap, and neither pretends to."""
        rng = np.random.default_rng(2)
        values = np.r_[rng.normal(size=20), np.full(3, np.nan), rng.normal(size=20)]
        data = {"B": pd.DataFrame({"y": values})}

        for method in ("savgol", "lowess"):
            out = smooth_trajectories(data, method=method)["B"]["y"]
            assert out.isna().to_numpy().nonzero()[0].tolist() == [20, 21, 22]
            assert out.notna().sum() == 40

    def test_columns_can_be_selected(self) -> None:
        rng = np.random.default_rng(4)
        raw = pd.DataFrame({"a": rng.normal(size=40), "b": rng.normal(size=40)})
        out = smooth_trajectories({"B": raw}, columns=["a"])["B"]

        assert not np.allclose(out["a"], raw["a"])
        assert np.allclose(out["b"], raw["b"])

    def test_the_input_is_not_mutated(self) -> None:
        rng = np.random.default_rng(6)
        raw = pd.DataFrame({"y": rng.normal(size=40)})
        before = raw["y"].copy()
        smooth_trajectories({"B": raw})
        assert np.array_equal(raw["y"].to_numpy(), before.to_numpy())

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"method": "kalman"}, "must be 'savgol' or 'lowess'"),
            ({"window": 3, "polyorder": 5}, "must be below window"),
            ({"frac": 0.0}, r"must be in \(0, 1\]"),
            ({"frac": 1.5}, r"must be in \(0, 1\]"),
        ],
    )
    def test_bad_arguments_are_rejected(self, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            smooth_trajectories({"B": pd.DataFrame({"y": np.zeros(20)})}, **kwargs)

    def test_no_batches_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="At least one batch"):
            smooth_trajectories({})


@pytest.mark.integration
def test_fill_then_smooth_is_the_intended_order() -> None:
    """Smoothing cannot see through a gap, so the fill comes first.

    Running them the other way round leaves the gap in place and the samples
    around it unsmoothed; running them in order gives a complete, smooth
    trajectory with the long gap still honestly missing.
    """
    rng = np.random.default_rng(7)
    clean = np.linspace(0, 10, 80) ** 0.5
    values = clean + rng.normal(scale=0.05, size=80)
    values[[10, 11]] = np.nan  # short: fillable
    values[40:52] = np.nan  # long: left alone
    batches = {"B1": pd.DataFrame({"y": values})}

    filled, report = fill_gaps(batches, limit=5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        smoothed = smooth_trajectories(filled, window=9, polyorder=2)["B1"]["y"]

    assert report.loc["B1", "y"] == "2/12"
    assert smoothed.iloc[10:12].notna().all(), "the short gap is closed before smoothing"
    assert smoothed.iloc[40:52].isna().all(), "the long gap is still missing, for the model to handle"

    # Compare only where the raw data was measured: the filled samples have no
    # raw counterpart to be better or worse than.
    measured = np.isfinite(values) & smoothed.notna().to_numpy()
    assert np.std(smoothed.to_numpy()[measured] - clean[measured]) < np.std(values[measured] - clean[measured])
