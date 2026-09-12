"""Pin the numbers the batch case-study scripts print, on the objects their functions return.

Every value here is quoted in the narrative pages under
``docs/user_guide/case_studies/batch/``; the DuPont and SBR values also match
the 2011-2012 course notes the case studies were rebuilt from. The data are
downloaded from openmv.net, so the tests skip offline.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from process_improve.batch import BatchMonitor, BatchPLS
from tests._case_study_scripts import SBR_URL_OVERRIDE, load_or_skip, load_script

pytestmark = [pytest.mark.dataset, pytest.mark.slow]


@pytest.fixture(scope="module")
def dupont_script():
    return load_script("dupont_batch_pca")


@pytest.fixture(scope="module")
def dupont_batches(dupont_script) -> dict:
    return load_or_skip(dupont_script.load_data)


@pytest.fixture(scope="module")
def dupont_model_a(dupont_script, dupont_batches):
    return dupont_script.fit_model_a(dupont_batches)


@pytest.fixture(scope="module")
def sbr_script():
    return load_script("sbr_batch_pls")


@pytest.fixture(scope="module")
def sbr_data(sbr_script) -> tuple[dict, pd.DataFrame]:
    return load_or_skip(lambda: sbr_script.load_data(SBR_URL_OVERRIDE))


@pytest.fixture(scope="module")
def sbr_model(sbr_script, sbr_data):
    return sbr_script.fit_model(*sbr_data)


SBR_FAULT_BATCHES = (34, 37)
SBR_CONF_LEVEL = 0.99
ALARM_RUN = 3


@pytest.fixture(scope="module")
def sbr_reference(sbr_data) -> tuple[dict, BatchPLS]:
    """Return the normal batches (34 and 37 left out) and the 2-component reference model fitted on them."""
    trajectories, quality = sbr_data
    normal = {batch_id: batch for batch_id, batch in trajectories.items() if batch_id not in SBR_FAULT_BATCHES}
    return normal, BatchPLS(n_components=2).fit(normal, quality.loc[list(normal)])


@pytest.fixture(scope="module")
def sbr_monitor_instantaneous(sbr_reference) -> BatchMonitor:
    """Per-sample limits from the normal batches, charting the per-interval (instantaneous) SPE."""
    normal, reference = sbr_reference
    return BatchMonitor(reference, conf_level=SBR_CONF_LEVEL, spe_statistic="instantaneous").fit(normal)


@pytest.fixture(scope="module")
def sbr_monitor_cumulative(sbr_reference) -> BatchMonitor:
    """Fit the same limits, charting the SPE accumulated over every sample observed so far."""
    normal, reference = sbr_reference
    return BatchMonitor(reference, conf_level=SBR_CONF_LEVEL, spe_statistic="cumulative").fit(normal)


def first_sustained_alarm(alarm: np.ndarray, run: int = ALARM_RUN) -> int | None:
    """Return the 1-based sample at which ``run`` consecutive alarms begin, or None if there is no such run.

    A single sample above a 99% limit is expected now and then on a normal
    batch, so the case study only acts on three consecutive alarms.
    """
    flags = np.asarray(alarm, dtype=bool)
    for start in range(len(flags) - run + 1):
        if flags[start : start + run].all():
            return start + 1
    return None


class TestDuPont:
    """Batch PCA outlier hunt on the DuPont polymerization reactor (#155)."""

    def test_model_a_matches_the_narrative(self, dupont_model_a) -> None:
        """38.3% and 17.6% per component, 55.9% cumulative; 50-55 in the scores; 49 in the SPE."""
        np.testing.assert_allclose(dupont_model_a.r2_per_component_.to_numpy(), [0.3828, 0.1758], atol=5e-4)
        assert dupont_model_a.r2_cumulative_.iloc[-1] == pytest.approx(0.5586, abs=5e-4)
        assert set(dupont_model_a.scores_.iloc[:, 0].abs().nlargest(4).index) == {50, 51, 52, 54}
        assert set(dupont_model_a.scores_.iloc[:, 1].abs().nlargest(3).index) == {50, 53, 55}
        spe = dupont_model_a.spe_.iloc[:, -1]
        assert spe.idxmax() == 49
        assert spe.max() > dupont_model_a.spe_limit(conf_level=0.95)

    def test_batch_49_is_a_short_event_in_the_heating_cooling_and_pressure_tags(
        self, dupont_script, dupont_model_a, dupont_batches
    ) -> None:
        """The raw view blames Flow-1; the SPE contributions do not."""
        _spe_share, by_tag, by_time = dupont_script.diagnose_spe_outlier(dupont_model_a, dupont_batches)
        share = by_tag / by_tag.sum()
        assert share.idxmax() == "TempC-1"
        assert share["Flow-1"] < 0.05
        assert set(by_time.nlargest(7).index) <= set(range(55, 65))

    def test_rebuilt_models_match_the_narrative(self, dupont_script, dupont_batches) -> None:
        """Excluding 49-55 exposes a second cluster; excluding it too gives an even model."""
        model_b = dupont_script.fit_model_b(dupont_batches)
        np.testing.assert_allclose(model_b.r2_per_component_.to_numpy(), [0.3331, 0.1328, 0.0851], atol=5e-4)
        assert {37, 44, 46, 48} <= set(model_b.scores_.iloc[:, 1].abs().nlargest(6).index)
        assert {39, 43, 45, 46, 47} <= set(model_b.scores_.iloc[:, 2].abs().nlargest(6).index)
        model_c = dupont_script.fit_model_c(dupont_batches)
        assert model_c.n_batches_ == 40
        np.testing.assert_allclose(model_c.r2_per_component_.to_numpy(), [0.3752, 0.1143, 0.0637], atol=5e-4)

    def test_model_c_flags_every_batch_left_out_of_it(self, dupont_script, dupont_batches) -> None:
        """The 15 batches removed before model C all lie above its SPE limit; seven above its T2 limit too."""
        model_c = dupont_script.fit_model_c(dupont_batches)
        table = dupont_script.verify_left_out(model_c, dupont_batches)
        assert sorted(table.index) == [37, 39, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]
        assert (table["SPE"] > model_c.spe_limit(conf_level=0.95)).all()
        above_t2 = set(table.index[table["T2"] > model_c.hotellings_t2_limit(conf_level=0.95)])
        assert above_t2 == {37, 50, 51, 52, 53, 54, 55}

    def test_poor_quality_batches_are_inside_both_limits(self, dupont_script, dupont_batches) -> None:
        """Observability: dupont_batches 38, 40, 41, 42 leave no trace in the trajectories."""
        table = dupont_script.observability_table(dupont_script.fit_model_c(dupont_batches))
        assert list(table.index) == [38, 40, 41, 42]
        assert (table["T2"] < table["T2 limit"]).all()
        assert (table["SPE"] < table["SPE limit"]).all()

    @pytest.mark.usefixtures("dupont_batches")
    def test_script_runs_end_to_end(self, dupont_script, tmp_path: pathlib.Path) -> None:
        """The whole script executes and writes its figures (skipped with the data when offline)."""
        assert dupont_script.main(["--output-dir", str(tmp_path)]) == 0
        assert len(list(tmp_path.glob("*.html"))) >= 15


class TestSBR:
    """Batch PLS fault diagnosis on the simulated SBR reactor (#156)."""

    def test_model_matches_the_narrative(self, sbr_script, sbr_model) -> None:
        """R2X 24.5% and 12.7%, R2Y 65.3% and 6.9%; batches 37 and 34 flagged in the scores, not in the SPE."""
        r2y = sbr_script.per_component(sbr_model.r2_cumulative_.to_numpy())
        r2x = sbr_script.per_component(sbr_model.r2_per_variable_.mean(axis=0).to_numpy())
        np.testing.assert_allclose(r2x, [0.2447, 0.1272], atol=5e-4)
        np.testing.assert_allclose(r2y, [0.6531, 0.0689], atol=5e-4)
        t1, t2 = sbr_model.scores_.iloc[:, 0], sbr_model.scores_.iloc[:, 1]
        assert set(t1.nsmallest(3).index) == {34, 37, 38}
        assert t1.idxmin() == 37
        assert t2.idxmax() == 34
        for batch_id in (34, 37):
            assert sbr_model.hotellings_t2_.loc[batch_id].iloc[-1] > sbr_model.hotellings_t2_limit(conf_level=0.95)
            assert sbr_model.spe_.loc[batch_id].iloc[-1] < sbr_model.spe_limit(conf_level=0.95)

    def test_contributions_name_the_fault_variables(self, sbr_script, sbr_model, sbr_data) -> None:
        """Batch 37: conversion and latex density low throughout; batch 34: the heat balance."""
        trajectories, _quality = sbr_data
        t1 = sbr_script.diagnose_batch_37(sbr_model, trajectories)
        by_tag_37 = t1.loc[37].groupby(level="tag", sort=False).sum()
        assert set(by_tag_37.nsmallest(2).index) == {"Conversion", "LatexDensity"}
        t2 = sbr_script.diagnose_batch_34(sbr_model, trajectories)
        by_tag_34 = t2.loc[34].groupby(level="tag", sort=False).sum()
        assert set(by_tag_34.nlargest(3).index) == {"EnergyReleased", "JacketTemp", "CoolingTemp"}

    def test_faulty_batches_have_poor_observed_and_fitted_quality(self, sbr_script, sbr_model, sbr_data) -> None:
        """Both batches rank at the low end of the quality attributes, observed and predicted."""
        _trajectories, quality = sbr_data
        table = sbr_script.compare_predictions(sbr_model, quality)
        rank = table.loc["rank of observed"]
        assert rank.loc[37, "Branching"] == 1
        assert rank.loc[34, "ParticleSize"] == 1
        predicted_rank = sbr_model.predictions_.rank()
        assert predicted_rank.loc[37].max() <= 3

    @pytest.mark.usefixtures("sbr_data")
    def test_script_runs_end_to_end(self, sbr_script, tmp_path: pathlib.Path) -> None:
        """The whole script executes and writes its figures (skipped with the data when offline)."""
        argv = ["--output-dir", str(tmp_path)]
        if SBR_URL_OVERRIDE:
            argv += ["--data-url", SBR_URL_OVERRIDE]
        assert sbr_script.main(argv) == 0
        assert len(list(tmp_path.glob("*.html"))) >= 15

    @pytest.mark.dataset
    @pytest.mark.slow
    def test_reference_monitor_flags_batch_37_in_the_scores_early(
        self, sbr_reference, sbr_monitor_instantaneous, sbr_data
    ) -> None:
        """Batch 37 is off from the start: its T2 stays above the limit from about sample 23 onwards."""
        normal, reference = sbr_reference
        trajectories, _quality = sbr_data
        n_reference = len(normal)
        assert n_reference == 51
        assert reference.n_components == 2
        monitor = sbr_monitor_instantaneous
        assert monitor.n_reference_batches_ == n_reference
        expected_mean = reference.n_components * (n_reference - 1) / n_reference
        np.testing.assert_allclose(monitor.t2_mean_over_time_, expected_mean, rtol=1e-9)
        assert monitor.t2_limit_over_time_[0] == pytest.approx(10.54, abs=0.05)
        first_t2 = first_sustained_alarm(monitor.monitor(trajectories[37]).t2_alarm)
        assert first_t2 is not None
        assert 15 < first_t2 <= 30
        assert first_t2 == pytest.approx(23, abs=2)

    @pytest.mark.dataset
    @pytest.mark.slow
    def test_batch_34_is_caught_by_the_instantaneous_spe_when_its_fault_begins(
        self, sbr_monitor_instantaneous, sbr_data
    ) -> None:
        """The fault enters batch 34 around sample 100; the per-interval SPE sustains an alarm from about 105."""
        trajectories, _quality = sbr_data
        result = sbr_monitor_instantaneous.monitor(trajectories[34])
        first_spe = first_sustained_alarm(result.spe_alarm)
        assert first_spe is not None
        assert 100 <= first_spe <= 115
        assert first_sustained_alarm(result.spe_alarm[:95]) is None
        # The scores react much later: the departure is off the reference plane, not along it.
        first_t2 = first_sustained_alarm(result.t2_alarm)
        assert first_t2 is None or first_t2 > 150

    @pytest.mark.dataset
    @pytest.mark.slow
    def test_cumulative_spe_alarms_later_than_the_instantaneous_spe(
        self, sbr_monitor_instantaneous, sbr_monitor_cumulative, sbr_data
    ) -> None:
        """Accumulating the residual over the whole batch so far dilutes a fresh fault, so it alarms later."""
        trajectories, _quality = sbr_data
        first_instantaneous = first_sustained_alarm(sbr_monitor_instantaneous.monitor(trajectories[34]).spe_alarm)
        first_cumulative = first_sustained_alarm(sbr_monitor_cumulative.monitor(trajectories[34]).spe_alarm)
        assert first_instantaneous is not None
        assert first_cumulative is not None
        assert first_cumulative > first_instantaneous

    @pytest.mark.dataset
    @pytest.mark.slow
    def test_normal_batches_rarely_cross_the_limits(self, sbr_reference, sbr_monitor_instantaneous) -> None:
        """At the 99% level the normal batches spend well under 3% of their samples above either limit."""
        normal, _reference = sbr_reference
        results = [sbr_monitor_instantaneous.monitor(batch) for batch in normal.values()]
        t2_fraction = float(np.mean([result.t2_alarm.mean() for result in results]))
        spe_fraction = float(np.mean([result.spe_alarm.mean() for result in results]))
        assert t2_fraction < 0.03
        assert spe_fraction < 0.03

    @pytest.mark.dataset
    @pytest.mark.slow
    def test_evolving_prediction_of_batch_4_converges_to_the_fitted_value(self, sbr_model, sbr_data) -> None:
        """With the 53-batch model the trace of batch 4 ends at predictions_, and the RMSEE shrinks over the batch."""
        trajectories, quality = sbr_data
        trace = sbr_model.predict_online_trace(trajectories[4])
        np.testing.assert_allclose(
            trace.y_hat.iloc[-1].to_numpy(), sbr_model.predictions_.loc[4].to_numpy(), rtol=1e-12, atol=0
        )
        evolving = trace.y_hat["ParticleSize"]
        np.testing.assert_allclose(
            evolving.loc[[10, 25, 50, 100, 150, 200]].to_numpy(),
            [1256.8, 1255.4, 1256.5, 1256.4, 1257.4, 1257.1],
            atol=0.05,
        )
        rmse = sbr_model.online_rmse(trajectories, quality)["ParticleSize"]
        assert rmse.index.name == "upto_k"
        assert rmse.iloc[-1] == pytest.approx(float(sbr_model.rmse_.loc["ParticleSize"].iloc[-1]), rel=1e-9)
        np.testing.assert_allclose(
            rmse.loc[[10, 50, 100, 150, 200]].to_numpy(), [2.94, 2.72, 2.63, 1.93, 1.87], atol=0.01
        )
        assert rmse.loc[10] > rmse.iloc[-1]
        # Trimmed score regression shrinks toward the mean batch while little of
        # the batch has been seen, so the estimation error never exceeds the
        # error of predicting the average batch every time. The estimator this
        # replaced had no such information and ran to 4.7 standard deviations
        # after one sample.
        assert (rmse <= float(quality["ParticleSize"].std(ddof=1))).all()
