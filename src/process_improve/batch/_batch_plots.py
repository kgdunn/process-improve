# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.
"""Model-level plots for batchwise-unfolded (multiway) PCA of batch data.

These complement the batch score / SPE / T2 plots, which are inherited from the
multivariate package (they operate on the internal PCA model). Two plots are
specific to the batch (unfolded) structure:

- :func:`time_varying_loading_plot`: the loadings of one component drawn as a
  function of time, one trace per tag, so the reader sees how each variable
  contributes to a component over the batch evolution.
- :func:`contribution_at_time_plot`: for one batch, the per-tag contribution to
  SPE or Hotelling's T2 at a chosen time sample, drawn as a bar chart to
  diagnose which variable drives an abnormal event.
"""

from __future__ import annotations

import typing

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover - exercised via env-without-plotly
    from process_improve._extras import _MissingExtra

    go = _MissingExtra("plotly", "plotting")  # type: ignore[assignment]

from ..visualization.themes import DEFAULT_THEME, REFERENCE_LINE_COLOR

if typing.TYPE_CHECKING:
    from ._batch_pca import BatchPCA


def _split_loadings(model: BatchPCA, component: int) -> tuple[pd.DataFrame, pd.Series]:
    """Split one component's loadings into a (tag x time) grid and a Z series.

    Returns the trajectory loadings reshaped to rows = tags, columns = time
    (in fitted order), and the initial-condition loadings as a plain Series
    (empty when the model was fitted without a Z block).
    """
    loading = model.loadings_.iloc[:, component - 1]
    sequence = loading.index.get_level_values("sequence")
    is_traj = sequence != ""
    traj = loading[is_traj]
    # Reshape to tags x time, preserving the fitted tag and time order.
    grid = traj.unstack(level="sequence")  # noqa: PD010 - direct inverse of the unfold; pivot_table would aggregate
    grid = grid.reindex(index=model.tag_names_, columns=model.time_index_)
    z_loadings = loading[~is_traj]
    z_loadings.index = z_loadings.index.get_level_values("tag")
    return grid, z_loadings


def time_varying_loading_plot(
    model: BatchPCA,
    component: int = 1,
    fig: go.Figure | None = None,
    show_initial_conditions: bool = True,
) -> go.Figure:
    """Plot one component's loadings as a function of time, one trace per tag.

    The batchwise-unfolded model has a separate loading for every
    ``(tag, time)`` cell, so a component's loadings can be read as a set of
    time-varying weight profiles: how strongly each variable loads on the
    component at each point in the batch. Initial-condition (Z) loadings, which
    have no time axis, are drawn as a marker group to the left of time zero.

    Parameters
    ----------
    model : BatchPCA
        A fitted :class:`process_improve.batch.BatchPCA` model.
    component : int, default=1
        1-based component index whose loadings to plot.
    fig : plotly.graph_objects.Figure, optional
        Figure to draw into; a new one is created when omitted.
    show_initial_conditions : bool, default=True
        Draw the initial-condition loadings (if the model has any) as a marker
        group before time zero.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if not 0 < component <= model.n_components:
        raise ValueError(f"The model has {model.n_components} components; need 1 <= component <= {model.n_components}.")
    grid, z_loadings = _split_loadings(model, component)

    if fig is None:
        fig = go.Figure()
    for tag in model.tag_names_:
        fig.add_trace(
            go.Scatter(
                x=list(model.time_index_),
                y=grid.loc[tag].to_numpy(),
                mode="lines",
                name=str(tag),
            )
        )
    if show_initial_conditions and len(z_loadings) > 0:
        fig.add_trace(
            go.Scatter(
                x=[-1] * len(z_loadings),
                y=z_loadings.to_numpy(),
                mode="markers",
                marker={"symbol": "diamond", "size": 9},
                text=[str(name) for name in z_loadings.index],
                name="initial conditions",
            )
        )
    fig.add_hline(y=0, line_color=REFERENCE_LINE_COLOR, line_width=1)
    fig.update_layout(
        template=DEFAULT_THEME,
        title=f"Time-varying loadings for component {component}",
        xaxis_title="Time [sequence order]",
        yaxis_title=f"Loading p{component}",
    )
    return fig


def contribution_at_time_plot(
    contributions: pd.DataFrame,
    k: int,
    batch_id: typing.Hashable | None = None,
    fig: go.Figure | None = None,
) -> go.Figure:
    """Bar chart of per-tag contributions at one time sample, for one batch.

    Takes the output of :meth:`process_improve.batch.BatchPCA.spe_contributions`
    or :meth:`~process_improve.batch.BatchPCA.t2_contributions` (one row per
    batch, columns indexed by the 2-level ``(tag, sequence)`` unfolded index)
    and shows, for a single batch and a single time sample ``k``, how much each
    tag contributes. This localizes an abnormal event to the responsible
    variable(s).

    Parameters
    ----------
    contributions : pd.DataFrame
        Contribution matrix from ``BatchPCA.spe_contributions`` /
        ``t2_contributions``: one row per batch, a 2-level ``(tag, sequence)``
        column index.
    k : int
        The time sample (sequence value) at which to show the contributions.
    batch_id : Hashable, optional
        Which batch (row) to plot. Defaults to the first row; required to be a
        valid row label when the matrix has more than one batch.
    fig : plotly.graph_objects.Figure, optional
        Figure to draw into; a new one is created when omitted.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if contributions.columns.nlevels != 2 or set(contributions.columns.names) != {"tag", "sequence"}:
        raise ValueError(
            "contributions must have a 2-level (tag, sequence) column index, as returned by "
            "BatchPCA.spe_contributions / t2_contributions."
        )
    if batch_id is None:
        batch_id = contributions.index[0]
    elif batch_id not in contributions.index:
        raise ValueError(f"batch_id {batch_id!r} is not a row of the contributions matrix.")

    position = typing.cast("int", contributions.index.get_loc(batch_id))
    row = typing.cast("pd.Series", contributions.iloc[position])
    at_k = row[row.index.get_level_values("sequence") == k]
    if at_k.empty:
        raise ValueError(f"No contributions at time sample k={k}; available samples run over the batch length.")
    tags = [str(label[0]) for label in at_k.index]
    values = at_k.to_numpy(dtype=float)

    if fig is None:
        fig = go.Figure()
    fig.add_trace(go.Bar(x=tags, y=values, marker_color=np.where(values >= 0, "#2563EB", "#DC2626")))
    fig.add_hline(y=0, line_color=REFERENCE_LINE_COLOR, line_width=1)
    fig.update_layout(
        template=DEFAULT_THEME,
        title=f"Contributions at time {k} (batch {batch_id})",
        xaxis_title="Tag",
        yaxis_title="Contribution",
    )
    return fig
