# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.
"""Model-level plots for batchwise-unfolded (multiway) PCA and PLS of batch data.

These complement the batch score / SPE / T2 plots, which are inherited from the
multivariate package (they operate on the internal PCA or PLS model). Three
plots are specific to the batch (unfolded) structure:

- :func:`time_varying_loading_plot`: the loadings (or PLS weights) of one
  component drawn as a function of time, one trace per tag, so the reader sees
  how each variable contributes to a component over the batch evolution.
- :func:`unfolded_contribution_plot`: for one batch, its whole contribution
  vector over the unfolded ``(tag, time)`` axis, grouped and coloured by tag,
  or summed into one bar per tag. This is the classic batch contribution plot
  that answers "which variables, and when".
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

from ..visualization.themes import DEFAULT_THEME, LIMIT_LINE_COLOR, REFERENCE_LINE_COLOR

if typing.TYPE_CHECKING:
    from sklearn.base import BaseEstimator

    from ._batch_monitor import BatchMonitor
    from ._batch_pca import BatchPCA
    from ._batch_pls import BatchPLS

_UNFOLDED_INDEX_ERROR = (
    "needs the 2-level (tag, sequence) index that process_improve.batch.dict_to_wide produces. "
    "When scaling the unfolded matrix yourself, re-attach ``wide.columns`` after MCUVScaler, "
    "which drops the multi-level labels."
)


def _unfolded_loadings(model: BaseEstimator) -> pd.DataFrame:
    """Return the model's loadings (PCA) or X-weights (PLS) on the unfolded index, or raise."""
    loadings = getattr(model, "loadings_", None)
    if loadings is None:
        loadings = getattr(model, "x_weights_", None)
    if loadings is None:
        raise ValueError("Model is not fitted, or has neither loadings_ nor x_weights_. Call fit() first.")
    is_unfolded = isinstance(loadings, pd.DataFrame) and set(loadings.index.names) == {"tag", "sequence"}
    if not is_unfolded:
        raise ValueError(f"The model's loadings index {_UNFOLDED_INDEX_ERROR}")
    return loadings


def _split_loadings(model: BaseEstimator, component: int) -> tuple[pd.DataFrame, pd.Series]:
    """Split one component's loadings into a (tag x time) grid and a Z series.

    Returns the trajectory loadings reshaped to rows = tags, columns = time
    (in fitted order), and the initial-condition loadings as a plain Series
    (empty when the model was fitted without a Z block). Works for the batch
    classes and for any :mod:`process_improve.multivariate` model fitted on a
    :func:`process_improve.batch.dict_to_wide` matrix: the tag and time order
    are read from the model when it records them and from the index otherwise.
    """
    loading = _unfolded_loadings(model).iloc[:, component - 1]
    sequence = loading.index.get_level_values("sequence")
    is_traj = sequence != ""
    traj = loading[is_traj]
    # Reshape to tags x time, preserving the fitted tag and time order.
    grid = traj.unstack(level="sequence")  # noqa: PD010 - direct inverse of the unfold; pivot_table would aggregate
    tag_names = getattr(model, "tag_names_", None) or list(dict.fromkeys(traj.index.get_level_values("tag")))
    time_index = getattr(model, "time_index_", None) or list(dict.fromkeys(traj.index.get_level_values("sequence")))
    grid = grid.reindex(index=tag_names, columns=time_index)
    z_loadings = loading[~is_traj]
    z_loadings.index = z_loadings.index.get_level_values("tag")
    return grid, z_loadings


def time_varying_loading_plot(
    model: BatchPCA | BatchPLS | BaseEstimator,
    component: int = 1,
    fig: go.Figure | None = None,
    show_initial_conditions: bool = True,
) -> go.Figure:
    """Plot one component's loadings (or weights) as a function of time, one trace per tag.

    The batchwise-unfolded model has a separate loading for every
    ``(tag, time)`` cell, so a component's loadings can be read as a set of
    time-varying weight profiles: how strongly each variable loads on the
    component at each point in the batch. Initial-condition (Z) loadings, which
    have no time axis, are drawn as a marker group to the left of time zero.

    Parameters
    ----------
    model : BatchPCA, BatchPLS or a fitted multivariate model
        A fitted :class:`process_improve.batch.BatchPCA` (loadings ``p``), a
        fitted :class:`process_improve.batch.BatchPLS` (weights ``w``), or any
        :mod:`process_improve.multivariate` model whose ``loadings_`` or
        ``x_weights_`` carry the 2-level ``(tag, sequence)`` index of a
        :func:`process_improve.batch.dict_to_wide` matrix.
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
    n_components = _unfolded_loadings(model).shape[1]
    if not 0 < component <= n_components:
        raise ValueError(f"The model has {n_components} components; need 1 <= component <= {n_components}.")
    grid, z_loadings = _split_loadings(model, component)
    symbol, kind = ("w", "weights") if hasattr(model, "x_weights_") else ("p", "loadings")

    if fig is None:
        fig = go.Figure()
    for tag in grid.index:
        fig.add_trace(
            go.Scatter(
                x=list(grid.columns),
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
        title=f"Time-varying {kind} for component {component}",
        xaxis_title="Time [sequence order]",
        yaxis_title=f"{kind.capitalize()[:-1]} {symbol}{component}",
    )
    return fig


def _contribution_row(
    contributions: pd.DataFrame, batch_id: typing.Hashable | None
) -> tuple[typing.Hashable, pd.Series]:
    """Validate a contribution matrix and return one batch's row (default: the first)."""
    if contributions.columns.nlevels != 2 or set(contributions.columns.names) != {"tag", "sequence"}:
        raise ValueError(
            "contributions must have a 2-level (tag, sequence) column index, as returned by "
            "the score_contributions / spe_contributions / t2_contributions methods of BatchPCA and BatchPLS."
        )
    if batch_id is None:
        batch_id = contributions.index[0]
    elif batch_id not in contributions.index:
        raise ValueError(f"batch_id {batch_id!r} is not a row of the contributions matrix.")
    position = typing.cast("int", contributions.index.get_loc(batch_id))
    return batch_id, typing.cast("pd.Series", contributions.iloc[position])


def unfolded_contribution_plot(
    contributions: pd.DataFrame,
    batch_id: typing.Hashable | None = None,
    *,
    by_tag: bool = False,
    fig: go.Figure | None = None,
) -> go.Figure:
    """Bar chart of one batch's contributions over the whole unfolded ``(tag, time)`` axis.

    Takes a contribution matrix (one row per batch, the 2-level
    ``(tag, sequence)`` column index of the unfolded data) and draws one
    batch's row as bars in unfolded column order, one trace per tag so the
    legend toggles tags and the colour identifies them. The tag names are
    written under the centre of each tag's block of samples. Reading the
    plot left to right answers "which variables, and at which time" for the
    score, SPE or T2 of that batch.

    With ``by_tag=True`` the bars are summed over time, one bar per tag,
    which is the compact summary used to rank the variables. The sum is
    signed: for score contributions it is the tag's contribution to the
    score; for SPE contributions (signed residuals) pass ``contributions ** 2``
    to get each tag's share of the SPE.

    Parameters
    ----------
    contributions : pd.DataFrame
        Output of ``score_contributions``, ``spe_contributions`` or
        ``t2_contributions`` on :class:`process_improve.batch.BatchPCA` or
        :class:`process_improve.batch.BatchPLS`, or of the standalone
        :mod:`process_improve.multivariate` functions on a model fitted to a
        :func:`process_improve.batch.dict_to_wide` matrix whose column index
        was re-attached after scaling.
    batch_id : Hashable, optional
        Which batch (row) to plot. Defaults to the first row.
    by_tag : bool, default=False
        Sum the contributions over time and draw one bar per tag.
    fig : plotly.graph_objects.Figure, optional
        Figure to draw into; a new one is created when omitted.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    batch_id, row = _contribution_row(contributions, batch_id)
    values = row.to_numpy(dtype=float)
    tags = row.index.get_level_values("tag")
    sequence = row.index.get_level_values("sequence")
    is_traj = np.asarray(sequence != "")
    tag_order = list(dict.fromkeys(tags[is_traj]))

    if fig is None:
        fig = go.Figure()
    if by_tag:
        sums = row.groupby(level="tag", sort=False).sum()
        labels = [str(tag) for tag in sums.index]
        totals = sums.to_numpy(dtype=float)
        fig.add_trace(go.Bar(x=labels, y=totals, marker_color=np.where(totals >= 0, "#2563EB", "#DC2626")))
        fig.update_layout(
            template=DEFAULT_THEME,
            title=f"Contributions for batch {batch_id}, summed over time",
            xaxis_title="Tag",
            yaxis_title="Contribution",
        )
    else:
        positions = np.arange(len(row))
        if not is_traj.all():
            z_mask = ~is_traj
            fig.add_trace(
                go.Bar(
                    x=positions[z_mask],
                    y=values[z_mask],
                    name="initial conditions",
                    text=[str(tag) for tag in tags[z_mask]],
                    hovertemplate="%{text}<br>%{y:.3g}<extra>initial conditions</extra>",
                )
            )
        tick_positions, tick_labels = [], []
        for tag in tag_order:
            mask = np.asarray(tags == tag) & is_traj
            fig.add_trace(
                go.Bar(
                    x=positions[mask],
                    y=values[mask],
                    name=str(tag),
                    customdata=np.asarray(sequence[mask]),
                    hovertemplate="%{customdata}<br>%{y:.3g}<extra>" + str(tag) + "</extra>",
                )
            )
            tick_positions.append(float(positions[mask].mean()))
            tick_labels.append(str(tag))
        fig.update_xaxes(tickvals=tick_positions, ticktext=tick_labels)
        fig.update_layout(
            template=DEFAULT_THEME,
            title=f"Contributions for batch {batch_id}",
            xaxis_title="Unfolded (tag, time) cells",
            yaxis_title="Contribution",
            barmode="overlay",
            bargap=0,
        )
    fig.add_hline(y=0, line_color=REFERENCE_LINE_COLOR, line_width=1)
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
    batch_id, row = _contribution_row(contributions, batch_id)
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


def online_monitoring_plot(
    monitor: BatchMonitor,
    batch: pd.DataFrame,
    statistic: str = "spe",
    *,
    initial_conditions: pd.Series | pd.DataFrame | None = None,
    fig: go.Figure | None = None,
) -> go.Figure:
    """Plot a batch's online SPE or T2 trace against the time-varying limit.

    Tracks the batch through the fitted
    :class:`process_improve.batch.BatchMonitor` and draws its statistic over
    time overlaid on the control limit and the mean good-batch trace, with the
    alarm samples marked. This is the online (real-time) monitoring chart of
    Nomikos and MacGregor.

    Parameters
    ----------
    monitor : BatchMonitor
        A fitted :class:`process_improve.batch.BatchMonitor`.
    batch : pd.DataFrame
        A single aligned batch to monitor.
    statistic : {"spe", "t2"}, default="spe"
        Which statistic to plot.
    initial_conditions : pd.Series or pd.DataFrame, optional
        The Z block for this batch; required if the model was fitted with one.
    fig : plotly.graph_objects.Figure, optional
        Figure to draw into; a new one is created when omitted.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    statistic = statistic.lower()
    if statistic not in {"spe", "t2"}:
        raise ValueError(f"statistic must be 'spe' or 't2'; got {statistic!r}.")

    result = monitor.monitor(batch, initial_conditions=initial_conditions)
    time = result.time
    if statistic == "spe":
        trace, limit, alarm = result.spe, result.spe_limit, result.spe_alarm
        mean_trace = monitor.spe_mean_over_time_[: len(time)]
        label = "SPE"
    else:
        trace, limit, alarm = result.hotellings_t2, result.t2_limit, result.t2_alarm
        mean_trace = monitor.t2_mean_over_time_[: len(time)]
        label = "Hotelling's T2"

    if fig is None:
        fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=time, y=mean_trace, mode="lines", name="good-batch mean", line={"color": REFERENCE_LINE_COLOR})
    )
    fig.add_trace(
        go.Scatter(
            x=time,
            y=limit,
            mode="lines",
            name=f"{int(monitor.conf_level * 100)}% limit",
            line={"color": LIMIT_LINE_COLOR, "dash": "dash"},
        )
    )
    fig.add_trace(go.Scatter(x=time, y=trace, mode="lines", name=label, line={"color": "#2563EB"}))
    if bool(np.any(alarm)):
        fig.add_trace(
            go.Scatter(
                x=time[alarm],
                y=np.asarray(trace)[alarm],
                mode="markers",
                name="alarm",
                marker={"color": LIMIT_LINE_COLOR, "size": 8, "symbol": "x"},
            )
        )
    fig.update_layout(
        template=DEFAULT_THEME,
        title=f"Online {label} monitoring",
        xaxis_title="Time [sequence order]",
        yaxis_title=label,
    )
    return fig
