"""Tests for online (Nomikos-MacGregor) batch monitoring."""

import numpy as np
import pandas as pd
import pytest

from process_improve.batch._batch_monitor import BatchMonitor
from process_improve.batch._batch_pca import BatchPCA
from process_improve.batch.datasets import load_nylon
from process_improve.batch.preprocessing import resample_to_reference


@pytest.fixture
def aligned_nylon() -> dict:
    """Nylon batches resampled to a common length."""
    batches = load_nylon()
    tags = list(next(iter(batches.values())).columns)
    return resample_to_reference(batches, columns_to_align=tags, reference_batch=1)


@pytest.fixture
def good_model_and_monitor(aligned_nylon: dict) -> tuple[BatchPCA, BatchMonitor, dict]:
    """Build a BatchPCA + fitted BatchMonitor from a good subset of nylon."""
    good = {k: v for k, v in aligned_nylon.items() if 1 <= k <= 36}
    model = BatchPCA(n_components=3).fit(good)
    monitor = BatchMonitor(model, conf_level=0.99).fit(good)
    return model, monitor, good


def test_predict_online_matches_diagnose_when_complete(aligned_nylon: dict) -> None:
    """At full observation the PMP projection equals the offline diagnosis."""
    model = BatchPCA(n_components=3).fit(aligned_nylon)
    batch = aligned_nylon[48]
    online = model.predict_online(batch, upto_k=model.n_timesteps_)
    offline = model.diagnose({48: batch})
    np.testing.assert_allclose(online.scores.to_numpy(), offline.scores.iloc[0].to_numpy(), atol=1e-6)
    assert np.isclose(online.hotellings_t2, offline.hotellings_t2.iloc[0, -1], atol=1e-6)
    assert np.isclose(online.spe, offline.spe.iloc[0], atol=1e-6)


def test_predict_online_bad_upto_k(aligned_nylon: dict) -> None:
    """upto_k outside the valid range is rejected."""
    model = BatchPCA(n_components=2).fit(aligned_nylon)
    with pytest.raises(ValueError, match="upto_k"):
        model.predict_online(aligned_nylon[1], upto_k=0)
    with pytest.raises(ValueError, match="upto_k"):
        model.predict_online(aligned_nylon[1], upto_k=model.n_timesteps_ + 1)


def test_monitor_limits_shapes(good_model_and_monitor: tuple) -> None:
    """The fitted limits span every time sample."""
    _, monitor, _ = good_model_and_monitor
    assert monitor.spe_limit_over_time_.shape == (monitor.n_timesteps_,)
    assert monitor.t2_limit_over_time_.shape == (monitor.n_timesteps_,)
    assert np.all(monitor.spe_limit_over_time_ > 0)


def test_good_batch_stays_mostly_in_limit(good_model_and_monitor: tuple) -> None:
    """A good training batch rarely breaches its own control limits."""
    _, monitor, good = good_model_and_monitor
    result = monitor.monitor(good[20])
    assert result.spe.shape == (monitor.n_timesteps_,)
    # good batches define the 99% limit, so alarms should be rare
    assert result.spe_alarm.mean() < 0.1


def test_injected_fault_breaches_spe(good_model_and_monitor: tuple, aligned_nylon: dict) -> None:
    """A large sustained fault in a batch trips the SPE limit in that window."""
    _, monitor, _ = good_model_and_monitor
    faulty = aligned_nylon[10].copy()
    tag = faulty.columns[7]
    window = slice(60, 80)
    faulty.iloc[window, faulty.columns.get_loc(tag)] += 5.0 * faulty[tag].std()
    result = monitor.monitor(faulty)
    # the fault window (1-based samples 61..80) should raise SPE alarms
    fault_mask = (result.time >= 61) & (result.time <= 80)
    assert result.spe_alarm[fault_mask].any()


def test_monitor_upto_k_truncates(good_model_and_monitor: tuple) -> None:
    """upto_k truncates the returned traces to that many samples."""
    _, monitor, good = good_model_and_monitor
    result = monitor.monitor(good[20], upto_k=50)
    assert len(result.time) == 50
    assert len(result.spe) == 50
    assert result.time[-1] == 50


def test_monitor_bad_upto_k(good_model_and_monitor: tuple) -> None:
    """An out-of-range upto_k for monitor is rejected."""
    _, monitor, good = good_model_and_monitor
    with pytest.raises(ValueError, match="upto_k"):
        monitor.monitor(good[20], upto_k=monitor.n_timesteps_ + 5)


def test_monitor_with_initial_conditions(aligned_nylon: dict) -> None:
    """Monitoring threads an initial-conditions block through end to end."""
    good = {k: v for k, v in aligned_nylon.items() if 1 <= k <= 20}
    z = pd.DataFrame({"charge": [float(k) for k in good]}, index=list(good.keys()))
    model = BatchPCA(n_components=2).fit(good, initial_conditions=z)
    monitor = BatchMonitor(model, conf_level=0.95).fit(good, initial_conditions=z)
    z_one = pd.Series({"charge": 5.0})
    result = monitor.monitor(good[5], initial_conditions=z_one)
    assert len(result.spe) == monitor.n_timesteps_


def test_predict_online_rejects_unexpected_initial_conditions(aligned_nylon: dict) -> None:
    """Passing a Z block to a model fitted without one is rejected."""
    model = BatchPCA(n_components=2).fit({k: v for k, v in aligned_nylon.items() if k <= 10})
    with pytest.raises(ValueError, match="without initial conditions"):
        model.predict_online(aligned_nylon[1], upto_k=10, initial_conditions=pd.Series({"charge": 1.0}))


def test_reference_t2_averages_a_times_n_minus_one_over_n(good_model_and_monitor: tuple) -> None:
    """The mean T2 of the reference batches is A(N-1)/N at every sample.

    The T2 at sample k is t_k' S_k^-1 t_k with S_k the reference batches'
    score covariance at that sample. Summing over the N reference batches
    gives the trace of S_k^-1 S_k scaled by N - 1, so the mean is
    A(N-1)/N, whatever the batch data. A departure from this value would
    mean the covariance and the scores it standardises are out of step.
    """
    model, monitor, good = good_model_and_monitor
    n_components, n_reference = model.n_components, len(good)
    assert monitor.n_reference_batches_ == n_reference
    expected = n_components * (n_reference - 1) / n_reference
    np.testing.assert_allclose(monitor.t2_mean_over_time_, expected, rtol=1e-9)
    assert np.unique(monitor.t2_limit_over_time_).size == 1
    assert monitor.score_covariance_over_time_.shape == (monitor.n_timesteps_, n_components, n_components)


def test_spe_window_pools_neighbouring_samples(good_model_and_monitor: tuple) -> None:
    """``spe_window`` pools the reference SPE of neighbouring samples before each limit is fitted.

    A window of 0 is the per-sample fit; a window that spans the whole batch
    pools every reference value into one limit; a window of 1 at an interior
    sample fits the limit to the three samples' values, which is checked
    against the reference traces directly. The mean trace is never pooled.
    """
    from process_improve.multivariate._limits import spe_calculation

    model, monitor, good = good_model_and_monitor
    same = BatchMonitor(model, conf_level=0.99, spe_window=0).fit(good)
    np.testing.assert_allclose(same.spe_limit_over_time_, monitor.spe_limit_over_time_)

    everything = BatchMonitor(model, conf_level=0.99, spe_window=monitor.n_timesteps_).fit(good)
    assert np.unique(everything.spe_limit_over_time_.round(9)).size == 1
    np.testing.assert_allclose(everything.spe_mean_over_time_, monitor.spe_mean_over_time_)

    one = BatchMonitor(model, conf_level=0.99, spe_window=1).fit(good)
    reference_spe = np.stack(
        [np.asarray(model.predict_online_trace(batch).spe, dtype=float) for batch in good.values()]
    )
    k = monitor.n_timesteps_ // 2
    expected = spe_calculation(reference_spe[:, k - 1 : k + 2].ravel(), conf_level=0.99)
    assert one.spe_limit_over_time_[k] == pytest.approx(expected, rel=1e-12)
    assert one.spe_limit_over_time_[0] == pytest.approx(
        spe_calculation(reference_spe[:, :2].ravel(), conf_level=0.99), rel=1e-12
    )

    with pytest.raises(ValueError, match="spe_window"):
        BatchMonitor(model, conf_level=0.99, spe_window=-1).fit(good)
