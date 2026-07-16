"""Tests for the batch model plots (time-varying loadings, contribution-at-time)."""

import numpy as np
import pandas as pd
import pytest

from process_improve.batch._batch_pca import BatchPCA
from process_improve.batch._batch_plots import contribution_at_time_plot, time_varying_loading_plot
from process_improve.batch.datasets import load_nylon
from process_improve.batch.preprocessing import resample_to_reference

go = pytest.importorskip("plotly.graph_objects")


@pytest.fixture
def fitted_model() -> BatchPCA:
    """Return a BatchPCA fitted on aligned nylon data."""
    batches = load_nylon()
    tags = list(next(iter(batches.values())).columns)
    aligned = resample_to_reference(batches, columns_to_align=tags, reference_batch=1)
    return BatchPCA(n_components=3).fit(aligned)


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
