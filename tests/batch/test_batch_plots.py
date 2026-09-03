"""Tests for the batch model plots (time-varying loadings, contribution-at-time)."""

import numpy as np
import pandas as pd
import pytest

from process_improve.batch._batch_monitor import BatchMonitor
from process_improve.batch._batch_pca import BatchPCA
from process_improve.batch._batch_plots import (
    contribution_at_time_plot,
    online_monitoring_plot,
    time_varying_loading_plot,
    unfolded_contribution_plot,
)
from process_improve.batch._batch_pls import BatchPLS
from process_improve.batch.data_input import dict_to_wide
from process_improve.batch.datasets import load_nylon
from process_improve.batch.preprocessing import resample_to_reference
from process_improve.multivariate import PCA, MCUVScaler

go = pytest.importorskip("plotly.graph_objects")


@pytest.fixture
def aligned_nylon() -> dict:
    """Nylon batches resampled to a common length."""
    batches = load_nylon()
    tags = list(next(iter(batches.values())).columns)
    return resample_to_reference(batches, columns_to_align=tags, reference_batch=1)


@pytest.fixture
def fitted_model(aligned_nylon: dict) -> BatchPCA:
    """Return a BatchPCA fitted on aligned nylon data."""
    return BatchPCA(n_components=3).fit(aligned_nylon)


@pytest.fixture
def spe_contributions(fitted_model: BatchPCA, aligned_nylon: dict) -> pd.DataFrame:
    """SPE contributions of every training batch, on the (tag, sequence) index."""
    return fitted_model.spe_contributions(fitted_model.unfold_and_scale(aligned_nylon))


def test_time_varying_loading_plot(fitted_model: BatchPCA) -> None:
    """One trace per tag, each spanning the full batch length."""
    fig = time_varying_loading_plot(fitted_model, component=1)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == fitted_model.n_tags_
    assert len(fig.data[0].x) == fitted_model.n_timesteps_


def test_time_varying_loading_plot_bad_component(fitted_model: BatchPCA) -> None:
    """An out-of-range component index is rejected."""
    with pytest.raises(ValueError, match="components"):
        time_varying_loading_plot(fitted_model, component=99)


def test_time_varying_loading_plot_with_initial_conditions() -> None:
    """A model with a Z block adds a marker group for the initial conditions."""
    batches = load_nylon()
    tags = list(next(iter(batches.values())).columns)
    aligned = resample_to_reference(batches, columns_to_align=tags, reference_batch=1)
    z = pd.DataFrame({"charge": [float(i) for i in range(len(aligned))]}, index=list(aligned.keys()))
    model = BatchPCA(n_components=2).fit(aligned, initial_conditions=z)
    fig = time_varying_loading_plot(model, component=1)
    assert len(fig.data) == model.n_tags_ + 1


def test_contribution_at_time_plot(fitted_model: BatchPCA) -> None:
    """One bar per tag at the requested time sample."""
    batches = load_nylon()
    tags = list(next(iter(batches.values())).columns)
    aligned = resample_to_reference(batches, columns_to_align=tags, reference_batch=1)
    scaled = fitted_model._scaled_wide(aligned, None)
    contributions = fitted_model.spe_contributions(scaled)
    fig = contribution_at_time_plot(contributions, k=57, batch_id=49)
    assert isinstance(fig, go.Figure)
    assert len(fig.data[0].x) == fitted_model.n_tags_


def test_contribution_at_time_plot_defaults_to_first_batch(fitted_model: BatchPCA) -> None:
    """With no batch_id the first row is used."""
    batches = load_nylon()
    tags = list(next(iter(batches.values())).columns)
    aligned = resample_to_reference(batches, columns_to_align=tags, reference_batch=1)
    scaled = fitted_model._scaled_wide(aligned, None)
    contributions = fitted_model.spe_contributions(scaled)
    fig = contribution_at_time_plot(contributions, k=0)
    assert len(fig.data[0].x) == fitted_model.n_tags_


def test_contribution_at_time_plot_rejects_flat_columns() -> None:
    """A contribution frame without the 2-level column index is rejected."""
    flat = pd.DataFrame(np.zeros((2, 3)), columns=["a", "b", "c"])
    with pytest.raises(ValueError, match="2-level"):
        contribution_at_time_plot(flat, k=0)


def test_contribution_at_time_plot_unknown_batch_id(fitted_model: BatchPCA) -> None:
    """An unknown batch_id is rejected with a clear error."""
    batches = load_nylon()
    tags = list(next(iter(batches.values())).columns)
    aligned = resample_to_reference(batches, columns_to_align=tags, reference_batch=1)
    contributions = fitted_model.spe_contributions(fitted_model._scaled_wide(aligned, None))
    with pytest.raises(ValueError, match="not a row"):
        contribution_at_time_plot(contributions, k=0, batch_id=99999)


def test_contribution_at_time_plot_out_of_range_time(fitted_model: BatchPCA) -> None:
    """A time sample beyond the batch length has no contributions and is rejected."""
    batches = load_nylon()
    tags = list(next(iter(batches.values())).columns)
    aligned = resample_to_reference(batches, columns_to_align=tags, reference_batch=1)
    contributions = fitted_model.spe_contributions(fitted_model._scaled_wide(aligned, None))
    with pytest.raises(ValueError, match="No contributions at time"):
        contribution_at_time_plot(contributions, k=fitted_model.n_timesteps_ + 100)


def test_online_monitoring_plot() -> None:
    """The online monitoring plot draws the trace, limit, and good-batch mean."""
    batches = load_nylon()
    tags = list(next(iter(batches.values())).columns)
    aligned = resample_to_reference(batches, columns_to_align=tags, reference_batch=1)
    good = {k: v for k, v in aligned.items() if 1 <= k <= 36}
    model = BatchPCA(n_components=3).fit(good)
    monitor = BatchMonitor(model, conf_level=0.99).fit(good)
    fig = online_monitoring_plot(monitor, aligned[49], "spe")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 3  # good-batch mean, limit, and the batch trace
    fig_t2 = online_monitoring_plot(monitor, aligned[49], "t2")
    assert isinstance(fig_t2, go.Figure)


def test_online_monitoring_plot_bad_statistic() -> None:
    """An unknown statistic name is rejected."""
    batches = load_nylon()
    tags = list(next(iter(batches.values())).columns)
    aligned = resample_to_reference(batches, columns_to_align=tags, reference_batch=1)
    good = {k: v for k, v in aligned.items() if 1 <= k <= 20}
    model = BatchPCA(n_components=2).fit(good)
    monitor = BatchMonitor(model, conf_level=0.95).fit(good)
    with pytest.raises(ValueError, match="statistic must be"):
        online_monitoring_plot(monitor, aligned[1], "nonsense")


def test_unfolded_contribution_plot_one_trace_per_tag(fitted_model: BatchPCA, spe_contributions: pd.DataFrame) -> None:
    """The full plot has one bar trace per tag covering every unfolded cell, with tag ticks."""
    fig = unfolded_contribution_plot(spe_contributions**2, batch_id=49)
    assert isinstance(fig, go.Figure)
    assert [trace.name for trace in fig.data] == [str(tag) for tag in fitted_model.tag_names_]
    assert sum(len(trace.x) for trace in fig.data) == fitted_model.n_tags_ * fitted_model.n_timesteps_
    assert list(fig.layout.xaxis.ticktext) == [str(tag) for tag in fitted_model.tag_names_]
    assert "49" in fig.layout.title.text


def test_unfolded_contribution_plot_by_tag_sums_over_time(
    fitted_model: BatchPCA, spe_contributions: pd.DataFrame
) -> None:
    """The summary plot has one bar per tag holding the signed sum over time."""
    squared = spe_contributions**2
    fig = unfolded_contribution_plot(squared, batch_id=49, by_tag=True)
    assert len(fig.data) == 1
    assert list(fig.data[0].x) == [str(tag) for tag in fitted_model.tag_names_]
    expected = squared.loc[49].groupby(level="tag", sort=False).sum().to_numpy()
    assert np.allclose(fig.data[0].y, expected)
    assert np.isclose(sum(fig.data[0].y), fitted_model.spe_.loc[49].iloc[-1] ** 2)


def test_unfolded_contribution_plot_with_initial_conditions(aligned_nylon: dict) -> None:
    """A model fitted with a Z block draws the initial-condition cells as their own trace."""
    z = pd.DataFrame({"charge": [float(i) for i in range(len(aligned_nylon))]}, index=list(aligned_nylon.keys()))
    model = BatchPCA(n_components=2).fit(aligned_nylon, initial_conditions=z)
    contributions = model.score_contributions(model.unfold_and_scale(aligned_nylon, initial_conditions=z))
    fig = unfolded_contribution_plot(contributions, batch_id=1)
    assert fig.data[0].name == "initial conditions"
    assert len(fig.data) == model.n_tags_ + 1


def test_unfolded_contribution_plot_defaults_to_first_row(spe_contributions: pd.DataFrame) -> None:
    """With no batch_id the first row is drawn."""
    fig = unfolded_contribution_plot(spe_contributions)
    assert str(spe_contributions.index[0]) in fig.layout.title.text


def test_unfolded_contribution_plot_rejects_flat_columns() -> None:
    """A contribution frame without the 2-level column index is rejected."""
    flat = pd.DataFrame(np.zeros((2, 3)), columns=["a", "b", "c"])
    with pytest.raises(ValueError, match="2-level"):
        unfolded_contribution_plot(flat)


def test_unfolded_contribution_plot_unknown_batch_id(spe_contributions: pd.DataFrame) -> None:
    """An unknown batch_id is rejected with a clear error."""
    with pytest.raises(ValueError, match="not a row"):
        unfolded_contribution_plot(spe_contributions, batch_id=99999)


def test_time_varying_loading_plot_accepts_a_plain_pca_on_unfolded_data(aligned_nylon: dict) -> None:
    """A multivariate PCA fitted on a dict_to_wide matrix (columns re-attached) plots like BatchPCA."""
    wide = dict_to_wide(aligned_nylon)
    scaled = MCUVScaler().fit_transform(wide)
    scaled.columns = wide.columns  # MCUVScaler drops the 2-level labels
    pca = PCA(n_components=2).fit(scaled)
    fig = time_varying_loading_plot(pca, component=2)
    assert len(fig.data) == len(wide.columns.get_level_values("tag").unique())
    assert len(fig.data[0].x) == wide.columns.get_level_values("sequence").nunique()
    assert "loadings" in fig.layout.title.text

    flat = PCA(n_components=2).fit(MCUVScaler().fit_transform(wide))  # labels lost: rejected with guidance
    with pytest.raises(ValueError, match="2-level"):
        time_varying_loading_plot(flat)


def test_time_varying_loading_plot_shows_batch_pls_weights(aligned_nylon: dict) -> None:
    """For a BatchPLS the profiles are the X-weights and the axis says so."""
    # One target that is an exact trajectory sample leaves no Y variance after one component.
    quality = pd.DataFrame(
        {"final": [float(b["Tag01"].iloc[-1]) for b in aligned_nylon.values()]}, index=list(aligned_nylon)
    )
    model = BatchPLS(n_components=1).fit(aligned_nylon, quality)
    fig = time_varying_loading_plot(model, component=1)
    assert len(fig.data) == model.n_tags_
    assert "weights" in fig.layout.title.text
    assert fig.layout.yaxis.title.text.endswith("w1")
