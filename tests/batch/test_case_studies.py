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
