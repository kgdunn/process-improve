"""Tests for the on-line (batch-so-far) API of BatchPLS, BatchPCA and BatchMonitor.

``predict_online`` estimates the scores, the final quality and the residual of
a batch that has run for ``upto_k`` samples; ``predict_online_trace`` does so
at every sample of a complete batch in one call; ``online_rmse`` averages the
evolving prediction error over a set of batches; and ``BatchMonitor`` turns the
traces of reference batches into per-sample control limits. The bundled dryer
data are aligned, thinned to every fourth sample and cut to 24 batches so the
whole module runs in about a second; every check here is an exact identity or
a shape, not a number that depends on the data.
"""

import numpy as np
import pandas as pd
import pytest

from process_improve.batch import BatchMonitor, BatchPCA, BatchPLS, load_dryer, resample_to_reference

pytestmark = pytest.mark.dataset

N_COMPONENTS = 2
N_BATCHES = 24
THIN_EVERY = 4
UPTO_K = 7


@pytest.fixture(scope="module")
def small_dryer() -> dict:
    """Return the first 24 dryer batches, aligned, with every fourth sample kept (37 samples x 10 tags)."""
    batches = load_dryer()
    tags = [c for c in next(iter(batches.values())).columns if c != "ClockTime"]
    trimmed = {k: v[tags] for k, v in batches.items()}
    aligned = resample_to_reference(trimmed, columns_to_align=tags, reference_batch=1)
    ids = list(aligned)[:N_BATCHES]
    return {k: aligned[k].iloc[::THIN_EVERY].reset_index(drop=True) for k in ids}


@pytest.fixture(scope="module")
def dryer_quality(small_dryer: dict) -> pd.DataFrame:
    """Return a synthetic quality block driven by the mean dryer temperature."""
    return pd.DataFrame(
        {"final": [float(b["DryerTemp"].mean()) for b in small_dryer.values()]},
        index=list(small_dryer.keys()),
    )


@pytest.fixture(scope="module")
def pls_model(small_dryer: dict, dryer_quality: pd.DataFrame) -> BatchPLS:
    """Fit a 2-component BatchPLS on every batch of the small dryer set."""
    return BatchPLS(n_components=N_COMPONENTS).fit(small_dryer, dryer_quality)


@pytest.fixture(scope="module")
def pca_model(small_dryer: dict) -> BatchPCA:
    """Fit a 2-component BatchPCA on every batch of the small dryer set."""
    return BatchPCA(n_components=N_COMPONENTS).fit(small_dryer)


@pytest.fixture(scope="module")
def pls_monitor(pls_model: BatchPLS, small_dryer: dict) -> BatchMonitor:
    """Fit a cumulative-SPE monitor on the PLS model, with the training batches as the reference set."""
    return BatchMonitor(pls_model, conf_level=0.99).fit(small_dryer)


def first_batch(batches: dict) -> tuple[int, pd.DataFrame]:
    """Return the identifier and trajectories of the first batch in the dictionary."""
    batch_id = next(iter(batches))
    return batch_id, batches[batch_id]


def sequence_of(index: pd.MultiIndex) -> np.ndarray:
    """Time sample of each unfolded column, -1 for an initial-condition column."""
    return np.array([-1 if s == "" else int(s) for s in index.get_level_values("sequence")])


# ---------------------------------------------------------------------------
# predict_online and predict_online_trace
# ---------------------------------------------------------------------------


def test_complete_batch_matches_offline_predict(pls_model: BatchPLS, small_dryer: dict) -> None:
    """With every sample observed the on-line estimate is the ordinary prediction."""
    batch_id, batch = first_batch(small_dryer)
    online = pls_model.predict_online(batch, pls_model.n_timesteps_)
    offline = pls_model.predict({batch_id: batch})
    np.testing.assert_allclose(online.scores.to_numpy(), offline.scores.loc[batch_id].to_numpy(), rtol=1e-12, atol=0)
    np.testing.assert_allclose(online.y_hat.to_numpy(), offline.y_hat.loc[batch_id].to_numpy(), rtol=1e-12, atol=0)
    assert list(online.scores.index) == list(pls_model.scores_.columns)
    assert list(online.y_hat.index) == list(pls_model.target_names_)
    assert not online.residuals.isna().any()


def test_trace_ends_at_the_fitted_attributes(pls_model: BatchPLS, small_dryer: dict) -> None:
    """The last row of a training batch's trace is its entry in predictions_ and scores_."""
    batch_id, batch = first_batch(small_dryer)
    trace = pls_model.predict_online_trace(batch)
    n = pls_model.n_timesteps_
    assert trace.y_hat.shape == (n, len(pls_model.target_names_))
    assert trace.y_hat.index.name == "upto_k"
    assert list(trace.y_hat.index) == list(range(1, n + 1))
    assert trace.scores.shape == (n, N_COMPONENTS)
    np.testing.assert_array_equal(trace.time, np.arange(1, n + 1))
    np.testing.assert_allclose(
        trace.y_hat.iloc[-1].to_numpy(), pls_model.predictions_.loc[batch_id].to_numpy(), rtol=1e-12, atol=0
    )
    np.testing.assert_allclose(
        trace.scores.iloc[-1].to_numpy(), pls_model.scores_.loc[batch_id].to_numpy(), rtol=1e-12, atol=0
    )
    for key in ("hotellings_t2", "spe", "spe_instantaneous", "condition_number"):
        assert np.asarray(trace[key]).shape == (n,)


def test_truncated_batch_matches_full_batch_at_upto_k(pls_model: BatchPLS, small_dryer: dict) -> None:
    """Only the first ``upto_k`` rows are used, so the batch so far and the complete batch agree."""
    _, batch = first_batch(small_dryer)
    so_far = pls_model.predict_online(batch.iloc[:UPTO_K], UPTO_K)
    complete = pls_model.predict_online(batch, UPTO_K)
    np.testing.assert_array_equal(so_far.scores.to_numpy(), complete.scores.to_numpy())
    np.testing.assert_array_equal(so_far.y_hat.to_numpy(), complete.y_hat.to_numpy())
    assert so_far.spe == complete.spe
    assert so_far.spe_instantaneous == complete.spe_instantaneous
    assert so_far.hotellings_t2 == complete.hotellings_t2
    assert so_far.condition_number == complete.condition_number
    np.testing.assert_array_equal(so_far.residuals.to_numpy(), complete.residuals.to_numpy())
    np.testing.assert_array_equal(so_far.forecast.to_numpy(), complete.forecast.to_numpy())
    # And row upto_k of the trace is the same estimate again.
    trace = pls_model.predict_online_trace(batch)
    np.testing.assert_allclose(trace.y_hat.loc[UPTO_K].to_numpy(), complete.y_hat.to_numpy(), rtol=1e-12, atol=0)
    np.testing.assert_allclose(trace.scores.iloc[UPTO_K - 1].to_numpy(), complete.scores.to_numpy(), rtol=1e-12, atol=0)
    assert trace.spe[UPTO_K - 1] == pytest.approx(complete.spe, rel=1e-12)
    assert trace.spe_instantaneous[UPTO_K - 1] == pytest.approx(complete.spe_instantaneous, rel=1e-12)


def test_forecast_keeps_the_observed_samples_and_imputes_the_rest(pls_model: BatchPLS, small_dryer: dict) -> None:
    """The forecast is the batch's own values up to ``upto_k`` and the model's imputation after."""
    _, batch = first_batch(small_dryer)
    n = pls_model.n_timesteps_
    forecast = pls_model.predict_online(batch, UPTO_K).forecast
    assert forecast.shape == (n, pls_model.n_tags_)
    assert list(forecast.columns) == list(pls_model.tag_names_)
    np.testing.assert_array_equal(forecast.iloc[:UPTO_K].to_numpy(), batch.iloc[:UPTO_K].to_numpy())
    remainder = forecast.iloc[UPTO_K:].to_numpy()
    assert np.isfinite(remainder).all()
    assert not np.allclose(remainder, batch.iloc[UPTO_K:].to_numpy())
    # With the whole batch observed there is nothing left to impute.
    complete = pls_model.predict_online(batch, n).forecast
    np.testing.assert_array_equal(complete.to_numpy(), batch.to_numpy())


def test_instantaneous_spe_is_the_newest_sample_only(
    pls_model: BatchPLS, pca_model: BatchPCA, small_dryer: dict
) -> None:
    """After one sample the two SPE statistics coincide; afterwards the instantaneous one is never larger."""
    _, batch = first_batch(small_dryer)
    for model in (pls_model, pca_model):
        trace = model.predict_online_trace(batch)
        assert trace.spe_instantaneous[0] == pytest.approx(trace.spe[0], rel=1e-12)
        assert np.all(trace.spe_instantaneous <= trace.spe * (1 + 1e-12))
        single = model.predict_online(batch, 1)
        assert single.spe_instantaneous == pytest.approx(single.spe, rel=1e-12)
        assert single.spe == pytest.approx(trace.spe[0], rel=1e-12)


def test_residuals_are_nan_exactly_where_unobserved(
    pls_model: BatchPLS, pca_model: BatchPCA, small_dryer: dict, dryer_quality: pd.DataFrame
) -> None:
    """The residual is NaN for the samples still to come and nowhere else; Z columns count as observed."""
    batch_id, batch = first_batch(small_dryer)
    for model in (pls_model, pca_model):
        residuals = model.predict_online(batch, UPTO_K).residuals
        assert list(residuals.index) == list(model.feature_columns_)
        np.testing.assert_array_equal(residuals.isna().to_numpy(), sequence_of(residuals.index) >= UPTO_K)

    ids = list(small_dryer)
    z = pd.DataFrame({"charge": [float(i) for i in range(len(ids))]}, index=ids)
    with_z = BatchPLS(n_components=N_COMPONENTS).fit(small_dryer, dryer_quality, initial_conditions=z)
    residuals = with_z.predict_online(batch, UPTO_K, initial_conditions=z.loc[[batch_id]]).residuals
    sequence = sequence_of(residuals.index)
    assert (sequence == -1).sum() == 1
    np.testing.assert_array_equal(residuals.isna().to_numpy(), sequence >= UPTO_K)
    # A Series of initial conditions is accepted too, with the same result.
    as_series = with_z.predict_online(batch, UPTO_K, initial_conditions=z.loc[batch_id])
    np.testing.assert_array_equal(as_series.residuals.to_numpy(), residuals.to_numpy())


def test_upto_k_out_of_range_raises(pls_model: BatchPLS, small_dryer: dict) -> None:
    """``upto_k`` must lie in 1 .. n_timesteps_."""
    _, batch = first_batch(small_dryer)
    with pytest.raises(ValueError, match="upto_k"):
        pls_model.predict_online(batch, 0)
    with pytest.raises(ValueError, match="upto_k"):
        pls_model.predict_online(batch, pls_model.n_timesteps_ + 1)
    with pytest.raises(ValueError, match="rows"):
        pls_model.predict_online(batch.iloc[:3], 5)


def test_wrong_columns_raise(pls_model: BatchPLS, small_dryer: dict) -> None:
    """A batch whose tags differ from the training tags is rejected by both entry points."""
    _, batch = first_batch(small_dryer)
    wrong = batch.rename(columns={batch.columns[0]: "nope"})
    with pytest.raises(ValueError, match="training tag"):
        pls_model.predict_online(wrong, UPTO_K)
    with pytest.raises(ValueError, match="training column layout"):
        pls_model.predict_online_trace(wrong)


# ---------------------------------------------------------------------------
# online_rmse
# ---------------------------------------------------------------------------


def test_online_rmse_ends_at_the_fitted_rmse(
    pls_model: BatchPLS, small_dryer: dict, dryer_quality: pd.DataFrame
) -> None:
    """One curve per target over the batch; its last value is the RMSE of the fit."""
    rmse = pls_model.online_rmse(small_dryer, dryer_quality)
    n = pls_model.n_timesteps_
    assert rmse.shape == (n, len(pls_model.target_names_))
    assert rmse.index.name == "upto_k"
    assert list(rmse.index) == list(range(1, n + 1))
    assert list(rmse.columns) == list(pls_model.target_names_)
    assert (rmse.to_numpy() > 0).all()
    fitted = np.sqrt(((pls_model.predictions_.reindex(dryer_quality.index) - dryer_quality) ** 2).mean())
    np.testing.assert_allclose(rmse.iloc[-1].to_numpy(), fitted.to_numpy(), rtol=1e-9)
    np.testing.assert_allclose(rmse.iloc[-1].to_numpy(), pls_model.rmse_.iloc[:, -1].to_numpy(), rtol=1e-9)


def test_online_rmse_validates_the_quality_block(
    pls_model: BatchPLS, small_dryer: dict, dryer_quality: pd.DataFrame
) -> None:
    """Y must be a DataFrame with the training targets and a row for every batch."""
    with pytest.raises(TypeError, match="DataFrame"):
        pls_model.online_rmse(small_dryer, dryer_quality.to_numpy())
    with pytest.raises(ValueError, match="training targets"):
        pls_model.online_rmse(small_dryer, dryer_quality.rename(columns={"final": "other"}))
    with pytest.raises(ValueError, match="no row"):
        pls_model.online_rmse(small_dryer, dryer_quality.iloc[:-1])


# ---------------------------------------------------------------------------
# BatchMonitor on a BatchPLS model
# ---------------------------------------------------------------------------


def test_monitor_fit_attributes(pls_monitor: BatchMonitor, small_dryer: dict) -> None:
    """Every per-sample attribute spans the batch, and the reference T2 averages A(N-1)/N."""
    n = pls_monitor.n_timesteps_
    n_reference = len(small_dryer)
    assert pls_monitor.n_reference_batches_ == n_reference
    for name in ("spe_limit_over_time_", "t2_limit_over_time_", "spe_mean_over_time_", "t2_mean_over_time_"):
        assert getattr(pls_monitor, name).shape == (n,)
    assert pls_monitor.score_covariance_over_time_.shape == (n, N_COMPONENTS, N_COMPONENTS)
    assert np.all(pls_monitor.spe_limit_over_time_ > 0)
    assert np.all(pls_monitor.spe_limit_over_time_ > pls_monitor.spe_mean_over_time_)
    # The T2 is standardised by the reference covariance at each sample, so
    # its limit is the same at every sample and the reference batches' mean
    # T2 is A(N-1)/N everywhere (the trace of S^-1 S, spread over N batches).
    assert np.unique(pls_monitor.t2_limit_over_time_).size == 1
    expected = N_COMPONENTS * (n_reference - 1) / n_reference
    np.testing.assert_allclose(pls_monitor.t2_mean_over_time_, expected, rtol=1e-9)
    assert pls_monitor.t2_limit_over_time_[0] > expected


def test_monitor_returns_scores_and_boolean_alarms(pls_monitor: BatchMonitor, small_dryer: dict) -> None:
    """monitor() carries the score trace and one boolean alarm per statistic and sample."""
    _, batch = first_batch(small_dryer)
    n = pls_monitor.n_timesteps_
    result = pls_monitor.monitor(batch)
    assert set(result.keys()) >= {"time", "scores", "hotellings_t2", "spe", "t2_limit", "spe_limit"}
    assert result.scores.shape == (n, N_COMPONENTS)
    assert list(result.scores.columns) == list(pls_monitor.model.scores_.columns)
    for name in ("t2_alarm", "spe_alarm"):
        alarm = result[name]
        assert alarm.dtype == np.bool_
        assert alarm.shape == (n,)
    np.testing.assert_array_equal(result.t2_alarm, result.hotellings_t2 > result.t2_limit)
    np.testing.assert_array_equal(result.spe_alarm, result.spe > result.spe_limit)
    truncated = pls_monitor.monitor(batch, upto_k=10)
    assert truncated.scores.shape == (10, N_COMPONENTS)
    assert len(truncated.time) == len(truncated.t2_alarm) == len(truncated.spe_alarm) == 10


def test_training_batch_t2_is_within_the_limit_early_on(pls_monitor: BatchMonitor, small_dryer: dict) -> None:
    """A reference batch sits inside its own T2 limit after five samples."""
    _, batch = first_batch(small_dryer)
    result = pls_monitor.monitor(batch, upto_k=5)
    assert result.hotellings_t2[-1] < result.t2_limit[-1]
    assert not result.t2_alarm[-1]


def test_invalid_spe_statistic_raises_at_fit(pls_model: BatchPLS, small_dryer: dict) -> None:
    """The SPE statistic is validated when the limits are built."""
    with pytest.raises(ValueError, match="spe_statistic"):
        BatchMonitor(pls_model, spe_statistic="nope").fit(small_dryer)


def test_instantaneous_spe_monitor(pls_model: BatchPLS, pls_monitor: BatchMonitor, small_dryer: dict) -> None:
    """The per-interval SPE monitor fits; at the first sample it coincides with the cumulative one."""
    instantaneous = BatchMonitor(pls_model, conf_level=0.99, spe_statistic="instantaneous").fit(small_dryer)
    assert instantaneous.spe_mean_over_time_[0] == pytest.approx(pls_monitor.spe_mean_over_time_[0], rel=1e-12)
    assert instantaneous.spe_limit_over_time_[0] == pytest.approx(pls_monitor.spe_limit_over_time_[0], rel=1e-12)
    # The newest sample's residual is a part of the cumulative residual, so
    # the instantaneous mean never exceeds the cumulative mean.
    assert np.all(instantaneous.spe_mean_over_time_ <= pls_monitor.spe_mean_over_time_ * (1 + 1e-12))
    assert instantaneous.spe_mean_over_time_[-1] < pls_monitor.spe_mean_over_time_[-1]
    # The T2 side is untouched by the choice of SPE statistic.
    np.testing.assert_array_equal(instantaneous.t2_limit_over_time_, pls_monitor.t2_limit_over_time_)
    _, batch = first_batch(small_dryer)
    result = instantaneous.monitor(batch)
    trace = pls_model.predict_online_trace(batch)
    np.testing.assert_allclose(result.spe, trace.spe_instantaneous, rtol=1e-12)


def test_too_few_reference_batches_raise(pls_model: BatchPLS, small_dryer: dict) -> None:
    """The per-sample score covariance needs more reference batches than components."""
    ids = list(small_dryer)[:N_COMPONENTS]
    with pytest.raises(ValueError, match="reference batches"):
        BatchMonitor(pls_model).fit({k: small_dryer[k] for k in ids})


def test_monitor_on_a_batch_pca_model(pca_model: BatchPCA, small_dryer: dict) -> None:
    """The same monitor wraps a BatchPCA model (the detailed nylon tests live in test_batch_monitor.py)."""
    monitor = BatchMonitor(pca_model, conf_level=0.99, spe_statistic="instantaneous").fit(small_dryer)
    n = monitor.n_timesteps_
    assert monitor.n_reference_batches_ == len(small_dryer)
    assert monitor.score_covariance_over_time_.shape == (n, N_COMPONENTS, N_COMPONENTS)
    _, batch = first_batch(small_dryer)
    result = monitor.monitor(batch)
    assert result.scores.shape == (n, N_COMPONENTS)
    assert result.spe.shape == result.hotellings_t2.shape == (n,)
    assert result.t2_alarm.dtype == np.bool_
    assert result.spe_alarm.dtype == np.bool_


# ---------------------------------------------------------------------------
# column layouts
# ---------------------------------------------------------------------------


def test_group_by_batch_layout_gives_the_same_online_result(
    pls_model: BatchPLS, small_dryer: dict, dryer_quality: pd.DataFrame
) -> None:
    """A model unfolded as (sequence, tag) projects a batch-so-far exactly like the (tag, sequence) one.

    The unfolded column order is a permutation, so every sign-free quantity
    (the prediction, the two SPEs, T2 and the forecast) must agree; only the
    label bookkeeping in the online helpers differs.
    """
    _, batch = first_batch(small_dryer)
    grouped = BatchPLS(n_components=N_COMPONENTS, group_by_batch=True).fit(small_dryer, dryer_quality)
    assert list(grouped.feature_columns_.names) == ["sequence", "tag"]
    expected = pls_model.predict_online(batch, UPTO_K)
    result = grouped.predict_online(batch, UPTO_K)
    pd.testing.assert_series_equal(result.y_hat, expected.y_hat, rtol=1e-9)
    assert result.spe == pytest.approx(expected.spe, rel=1e-9)
    assert result.spe_instantaneous == pytest.approx(expected.spe_instantaneous, rel=1e-9)
    assert result.hotellings_t2 == pytest.approx(expected.hotellings_t2, rel=1e-9)
    pd.testing.assert_frame_equal(result.forecast, expected.forecast, rtol=1e-9)
    trace = grouped.predict_online_trace(batch)
    np.testing.assert_allclose(trace.spe, pls_model.predict_online_trace(batch).spe, rtol=1e-9)
