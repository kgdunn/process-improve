"""Batch PLS fault diagnosis on the simulated SBR batch reactor.

Case study for issue #156. Styrene-butadiene rubber is made by emulsion
polymerization in a batch reactor; five quality attributes of the latex are
measured at the end of each batch. Because these 53 batches were simulated
from a first-principles model, the fault is known: batches 34 and 37 both
received 30% more organic impurity in the butadiene feed, from the very
start of batch 37 and midway through batch 34. This script fits a
batchwise-unfolded PLS from the six trajectories to the five quality
attributes and follows the diagnosis of the course notes: the score plot
flags both batches, the whole-batch SPE does not, the weights and the
contribution plots name the variables, and the raw trajectories confirm the
story. The same fault lands in two different places of the score plot
because it started at two different times. The last two sections ask what
the model would have shown while a batch was still running: the evolving
prediction of the final quality, and per-sample T2 and SPE charts built
from a reference model of the normal batches.

Data: https://openmv.net/info/sbr-batch-reactor (53 batches x 200 samples x
9 tags, plus 5 quality attributes), downloaded when the script runs.

Source: Nomikos, P., "Statistical process control of batch processes", PhD
thesis, McMaster University, 1995, and the ConnectMV latent-variable course
notes (2011-2012, CC BY-SA 3.0).

Run from the repository root::

    uv run python docs/user_guide/case_studies/batch/sbr_batch_pls.py --output-dir case-study-output/sbr

Every figure is written as a self-contained HTML file to the output directory.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from process_improve.batch import (
    BatchMonitor,
    BatchPLS,
    contribution_at_time_plot,
    load_sbr,
    online_monitoring_plot,
    time_varying_loading_plot,
    unfolded_contribution_plot,
)
from process_improve.batch.plotting import plot_all_batches_per_tag

# -- section: constants --
CONF_LEVEL = 0.95
FAULT_FROM_START = 37
FAULT_MID_BATCH = 34
FAULT_BATCHES = [FAULT_MID_BATCH, FAULT_FROM_START]
INSPECT_SAMPLE = 120  # a sample well after the fault in batch 34 has developed
HIGHLIGHT = '{"color": "red", "width": 4}'  # Plotly line style, JSON-encoded, for the highlighted batches
LABELS = {"show_labels": True}
# -- end: constants --


# -- section: load --
def load_data(url: str | None = None) -> tuple[dict, pd.DataFrame]:
    """Download the batches and keep the six trajectories the original study modelled.

    The two feed flows are constant in the simulation and the feed
    temperature barely moves, so they carry no batch-to-batch information.
    """
    sbr = load_sbr(url=url)
    trajectories = {batch_id: batch[sbr.trajectory_tags] for batch_id, batch in sbr.X.items()}
    first = next(iter(trajectories.values()))
    print(
        f"{len(trajectories)} batches x {first.shape[0]} samples x {first.shape[1]} tags; quality block {sbr.Y.shape}"
    )
    return trajectories, sbr.Y


# -- end: load --


# -- section: raw --
def plot_raw(trajectories: dict, tag: str, highlight: list[int]) -> go.Figure:
    """Overlay one tag for every batch, with the batches of interest drawn in red."""
    return plot_all_batches_per_tag(
        trajectories, tag, batches_to_highlight={HIGHLIGHT: highlight}, extra_info=f"highlighting {highlight}"
    )


# -- end: raw --


# -- section: model --
def fit_model(trajectories: dict, quality: pd.DataFrame) -> BatchPLS:
    """Two-component batch PLS from the unfolded trajectories to the five quality attributes.

    Each batch is one row of 6 tags x 200 samples = 1200 columns. Every
    column is scaled to unit variance, so the mean of the per-column R2 is the
    R2 of the whole trajectory block.
    """
    model = BatchPLS(n_components=2).fit(trajectories, quality)
    r2y = per_component(model.r2_cumulative_.to_numpy())
    r2x = per_component(model.r2_per_variable_.mean(axis=0).to_numpy())
    print(f"R2X per component = {r2x[0]:.3f}, {r2x[1]:.3f}; R2Y per component = {r2y[0]:.3f}, {r2y[1]:.3f}")
    t1, t2 = model.scores_.iloc[:, 0], model.scores_.iloc[:, 1]
    print(f"lowest t1 = {t1.nsmallest(3).index.tolist()}; highest t2 = {t2.nlargest(2).index.tolist()}")
    t2_limit = model.hotellings_t2_limit(conf_level=CONF_LEVEL)
    spe_limit = model.spe_limit(conf_level=CONF_LEVEL)
    for batch_id in FAULT_BATCHES:
        hotelling = model.hotellings_t2_.loc[batch_id].iloc[-1]
        spe = model.spe_.loc[batch_id].iloc[-1]
        print(f"batch {batch_id}: T2 = {hotelling:.1f} (limit {t2_limit:.1f}), SPE = {spe:.1f} (limit {spe_limit:.1f})")
    return model


def per_component(cumulative: np.ndarray) -> np.ndarray:
    """Turn a cumulative R2 vector into the increment each component adds."""
    return np.diff(np.concatenate([[0.0], cumulative]))


# -- end: model --


# -- section: r2-breakdown --
def r2_breakdown(model: BatchPLS) -> pd.DataFrame:
    """R2 of every (tag, time) cell after two components, as a tags x time grid."""
    grid = model.r2_per_variable_.iloc[:, -1].unstack(level="sequence")  # noqa: PD010 - inverse of the unfold
    print(
        "R2 per tag, averaged over time: " + ", ".join(f"{tag} {value:.2f}" for tag, value in grid.mean(axis=1).items())
    )
    return grid


def plot_r2_over_time(grid: pd.DataFrame) -> go.Figure:
    """One line per tag: how much of each trajectory the model explains at every sample."""
    fig = go.Figure()
    for tag, row in grid.iterrows():
        fig.add_trace(go.Scatter(x=list(grid.columns), y=row.to_numpy(), mode="lines", name=str(tag)))
    fig.update_layout(
        title="R2 of each trajectory over the batch", xaxis_title="Time [sequence order]", yaxis_title="R2"
    )
    return fig


# -- end: r2-breakdown --


# -- section: batch-37 --
def diagnose_batch_37(model: BatchPLS, trajectories: dict) -> pd.DataFrame:
    """Score contributions to t1: why batch 37 sits at the low end of t1."""
    contributions = model.score_contributions(model.unfold_and_scale(trajectories), component=1)
    row = contributions.loc[FAULT_FROM_START]
    by_tag = row.groupby(level="tag", sort=False).sum()
    print(f"batch {FAULT_FROM_START}: t1 contributions per tag = " + describe(by_tag))
    print(f"batch {FAULT_FROM_START}: share of the t1 contribution per fifth of the batch = {share_per_fifth(row)}")
    print(
        f"batch {FAULT_FROM_START}: first sustained departure from the other batches: {sustained_departure(trajectories, FAULT_FROM_START)}"
    )
    return contributions


def describe(by_tag: pd.Series) -> str:
    """Format per-tag contributions as one line."""
    return ", ".join(f"{tag} {value:+.1f}" for tag, value in by_tag.items())


def share_per_fifth(row: pd.Series) -> str:
    """Split a batch's contribution vector into five equal time blocks and give each block's share."""
    by_time = row.groupby(level="sequence").sum()
    fifths = by_time.groupby(np.arange(len(by_time)) * 5 // len(by_time)).sum()
    return ", ".join(f"{share:.0%}" for share in fifths / fifths.sum())


def sustained_departure(trajectories: dict, batch_id: int, n_sd: float = 2.0, run: int = 20) -> dict[str, int | None]:
    """Return the first sample from which each tag of one batch stays outside the band of the normal batches.

    The band is the mean of the batches without the fault plus or minus ``n_sd``
    standard deviations, sample by sample. A tag has departed once it stays
    outside the band for ``run`` consecutive samples, a tenth of the batch by
    default, so that a single noisy excursion does not count. ``None`` means
    the tag never departs.
    """
    others = np.stack([batch.to_numpy() for key, batch in trajectories.items() if key not in FAULT_BATCHES])
    z = (trajectories[batch_id].to_numpy() - others.mean(axis=0)) / others.std(axis=0, ddof=1)
    outside = (np.abs(z) > n_sd).astype(int)
    window = np.ones(run, dtype=int)
    onset: dict[str, int | None] = {}
    for j, tag in enumerate(trajectories[batch_id].columns):
        runs = np.convolve(outside[:, j], window, mode="valid") == run
        onset[tag] = int(np.argmax(runs)) if runs.any() else None
    return onset


# -- end: batch-37 --


# -- section: batch-34 --
def diagnose_batch_34(model: BatchPLS, trajectories: dict) -> pd.DataFrame:
    """Score contributions to t2: the same fault, but starting midway through batch 34."""
    contributions = model.score_contributions(model.unfold_and_scale(trajectories), component=2)
    row = contributions.loc[FAULT_MID_BATCH]
    by_tag = row.groupby(level="tag", sort=False).sum()
    print(f"batch {FAULT_MID_BATCH}: t2 contributions per tag = " + describe(by_tag))
    print(f"batch {FAULT_MID_BATCH}: share of the t2 contribution per fifth of the batch = {share_per_fifth(row)}")
    print(
        f"batch {FAULT_MID_BATCH}: first sustained departure from the other batches: {sustained_departure(trajectories, FAULT_MID_BATCH)}"
    )
    return contributions


# -- end: batch-34 --


# -- section: predictions --
def compare_predictions(model: BatchPLS, quality: pd.DataFrame) -> pd.DataFrame:
    """Observed and fitted quality of the two faulty batches, with the rank of each observed value."""
    table = pd.concat(
        {
            "observed": quality.loc[FAULT_BATCHES],
            "predicted": model.predictions_.loc[FAULT_BATCHES],
            "rank of observed": quality.rank().loc[FAULT_BATCHES].astype(int),
        },
        names=["value", "batch_id"],
    )
    print(f"quality of the faulty batches (rank 1 = lowest of {len(quality)} batches)")
    print(table.to_string(float_format=lambda value: f"{value:.4g}"))
    return table


# -- end: predictions --


# -- section: online-prediction --
AVERAGE_BATCH = 4  # the batch whose trajectories lie closest to the average
PREDICTION_SAMPLES = [10, 25, 50, 100, 150, 200]
ERROR_SAMPLES = [10, 50, 100, 150, 200]


def evolving_prediction(
    model: BatchPLS, trajectories: dict, quality: pd.DataFrame, batch_id: int = AVERAGE_BATCH
) -> pd.DataFrame:
    """Predict the final quality after every sample of one batch, as it would have been seen in real time.

    The unfolded row of a running batch is complete up to the newest sample
    and missing after it. ``predict_online_trace`` estimates the scores from
    the observed cells alone (trimmed score regression by default) and maps
    them to the quality attributes, once per sample.
    """
    y_hat = model.predict_online_trace(trajectories[batch_id]).y_hat
    actual = quality.loc[batch_id, "ParticleSize"]
    fitted = model.predictions_.loc[batch_id, "ParticleSize"]
    print(f"batch {batch_id}: ParticleSize measured {actual:.1f}, fitted from the complete batch {fitted:.1f}")
    print(
        f"batch {batch_id}: ParticleSize predicted after "
        + ", ".join(f"{k} samples {y_hat.loc[k, 'ParticleSize']:.1f}" for k in PREDICTION_SAMPLES)
    )
    return y_hat


def evolving_error(model: BatchPLS, trajectories: dict, quality: pd.DataFrame) -> pd.DataFrame:
    """RMSEE of the evolving prediction at every sample, relative to the standard deviation of each attribute.

    ``online_rmse`` traces every training batch and pools the squared errors
    per sample, so this is the estimation error (RMSEE) as a function of how
    much of the batch has been observed. A ratio of about 1 is the error of
    predicting the average batch.
    """
    ratio = model.online_rmse(trajectories, quality) / quality.std(ddof=1)
    for target in ("ParticleSize", "Branching"):
        print(
            f"RMSEE / sd of {target} after "
            + ", ".join(f"{k} samples {ratio.loc[k, target]:.2f}" for k in ERROR_SAMPLES)
        )
    return ratio


def plot_online_rmse(ratio: pd.DataFrame) -> go.Figure:
    """One line per quality attribute: the RMSEE relative to that attribute's standard deviation."""
    fig = go.Figure()
    for target in ratio.columns:
        fig.add_trace(go.Scatter(x=ratio.index, y=ratio[target].to_numpy(), mode="lines", name=str(target)))
    fig.add_hline(y=1.0, line_dash="dash", annotation_text="predicting the average batch")
    fig.update_layout(
        title="RMSEE of the evolving quality prediction (training batches)",
        xaxis_title="Samples observed",
        yaxis_title="RMSEE / standard deviation",
    )
    return fig


def plot_online_prediction(y_hat: pd.DataFrame, quality: pd.DataFrame, batch_id: int) -> go.Figure:
    """Draw the evolving prediction of one batch's particle size against its measured value and the average batch."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_hat.index, y=y_hat["ParticleSize"].to_numpy(), mode="lines", name="predicted"))
    fig.add_hline(y=quality.loc[batch_id, "ParticleSize"], line_dash="dash", annotation_text="measured")
    fig.add_hline(y=quality["ParticleSize"].mean(), line_dash="dot", annotation_text="average of all batches")
    fig.update_layout(
        title=f"ParticleSize of batch {batch_id}, predicted as the batch runs",
        xaxis_title="Samples observed",
        yaxis_title="ParticleSize",
    )
    return fig


# -- end: online-prediction --


# -- section: online-monitoring --
MONITOR_CONF_LEVEL = 0.99
ALARM_RUN = 3  # consecutive samples above the limit before an alarm counts
FORECAST_FROM = {FAULT_FROM_START: ("Conversion", [30, 60]), FAULT_MID_BATCH: ("CoolingTemp", [60, 115])}


def normal_batches(trajectories: dict) -> dict:
    """Return the batches without the injected fault."""
    return {batch_id: batch for batch_id, batch in trajectories.items() if batch_id not in FAULT_BATCHES}


def online_monitoring(trajectories: dict, quality: pd.DataFrame) -> tuple[BatchMonitor, dict]:
    """Per-sample T2 and SPE charts of the two faulty batches against a model of the normal batches.

    A reference model must describe normal operation, so batches 34 and 37
    are left out of it. ``BatchMonitor`` projects every reference batch after
    1, 2, ..., 200 samples with the same missing-data estimator and builds a
    limit at each sample from the spread of those projections. The SPE
    charted is the instantaneous one, the residual of the newest sample only,
    which reacts in the sample a fault begins. An alarm counts once the
    statistic stays above its limit for three consecutive samples.
    """
    normal = normal_batches(trajectories)
    reference = BatchPLS(n_components=2).fit(normal, quality.loc[list(normal)])
    monitor = BatchMonitor(reference, conf_level=MONITOR_CONF_LEVEL, spe_statistic="instantaneous").fit(normal)
    print(f"reference model on {len(normal)} batches; T2 limit {monitor.t2_limit_over_time_[0]:.2f} at every sample")
    alarms: dict[int, dict[str, int | None]] = {}
    for batch_id in FAULT_BATCHES:
        result = monitor.monitor(trajectories[batch_id])
        alarms[batch_id] = {
            "T2": first_sustained_alarm(result.t2_alarm),
            "SPE": first_sustained_alarm(result.spe_alarm),
        }
        print(
            f"batch {batch_id}: first {ALARM_RUN} consecutive samples above the limit: "
            f"T2 after {alarms[batch_id]['T2']} samples, SPE after {alarms[batch_id]['SPE']} samples"
        )
    t2_rate, spe_rate = alarm_rates(monitor, normal)
    print(f"normal batches: {t2_rate:.2%} of the T2 values and {spe_rate:.2%} of the SPE values above their limits")
    spe_alarm = alarms[FAULT_MID_BATCH]["SPE"]
    if spe_alarm is None:
        raise RuntimeError(
            f"batch {FAULT_MID_BATCH} raised no sustained SPE alarm; the residual shares below need one."
        )
    for k in (spe_alarm, spe_alarm + 4):
        shares = residual_shares(reference, trajectories[FAULT_MID_BATCH], k)
        print(f"batch {FAULT_MID_BATCH}: share of the SPE after {k} samples per tag = " + describe_shares(shares))
    for batch_id, (tag, sample_points) in FORECAST_FROM.items():
        for k in sample_points:
            forecast = reference.predict_online(trajectories[batch_id], k).forecast
            print(
                f"batch {batch_id} {tag}, mean of the samples after {k}: "
                f"forecast {forecast[tag].iloc[k:].mean():.4f}, actual {trajectories[batch_id][tag].iloc[k:].mean():.4f}, "
                f"average of the normal batches {np.mean([batch[tag].iloc[k:].mean() for batch in normal.values()]):.4f}"
            )
    return monitor, alarms


def first_sustained_alarm(alarm: np.ndarray, run: int = ALARM_RUN) -> int | None:
    """Return the 1-based sample after which a statistic first stays above its limit for ``run`` samples in a row."""
    runs = np.convolve(np.asarray(alarm, dtype=int), np.ones(run, dtype=int), mode="valid") == run
    return int(np.argmax(runs)) + 1 if runs.any() else None


def alarm_rates(monitor: BatchMonitor, batches: dict) -> tuple[float, float]:
    """Fraction of the (batch, sample) points of these batches above the T2 limit and above the SPE limit."""
    results = [monitor.monitor(batch) for batch in batches.values()]
    t2_rate = float(np.mean([result.t2_alarm for result in results]))
    spe_rate = float(np.mean([result.spe_alarm for result in results]))
    return t2_rate, spe_rate


def residual_shares(model: BatchPLS, batch: pd.DataFrame, k: int) -> pd.Series:
    """Share of the squared instantaneous SPE after ``k`` samples carried by each tag."""
    squared = model.predict_online(batch, k).residuals.xs(k - 1, level="sequence") ** 2
    return squared / squared.sum()


def describe_shares(shares: pd.Series) -> str:
    """Format per-tag shares as one line, largest first."""
    return ", ".join(f"{tag} {share:.0%}" for tag, share in shares.sort_values(ascending=False).items())


def plot_forecasts(reference: BatchPLS, trajectories: dict) -> go.Figure:
    """Draw the rest of each faulty batch as forecast from its scores at two points, next to what happened."""
    normal = normal_batches(trajectories)
    titles = [f"batch {batch_id}: {tag}" for batch_id, (tag, _) in FORECAST_FROM.items()]
    fig = make_subplots(rows=1, cols=2, subplot_titles=titles)
    for col, (batch_id, (tag, sample_points)) in enumerate(FORECAST_FROM.items(), start=1):
        n = len(trajectories[batch_id])
        time = np.arange(1, n + 1)
        average = np.mean([batch[tag].to_numpy() for batch in normal.values()], axis=0)
        fig.add_trace(go.Scatter(x=time, y=average, mode="lines", name="average of the normal batches"), row=1, col=col)
        fig.add_trace(
            go.Scatter(x=time, y=trajectories[batch_id][tag].to_numpy(), mode="lines", name=f"batch {batch_id}"),
            row=1,
            col=col,
        )
        for k in sample_points:
            forecast = reference.predict_online(trajectories[batch_id], k).forecast[tag]
            fig.add_trace(
                go.Scatter(
                    x=time[k:],
                    y=forecast.iloc[k:].to_numpy(),
                    mode="lines",
                    line={"dash": "dash"},
                    name=f"forecast from {k} samples",
                ),
                row=1,
                col=col,
            )
        fig.update_xaxes(title_text="Time [sequence order]", row=1, col=col)
        fig.update_yaxes(title_text=tag, row=1, col=col)
    fig.update_layout(title="The rest of the batch, forecast from the reference model")
    return fig


# -- end: online-monitoring --


def main(argv: list[str] | None = None) -> int:
    """Run the whole case study and write its figures."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("case-study-output/sbr"))
    parser.add_argument("--data-url", default=None, help="override the openmv.net URL (for example a file:// copy)")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def save(fig: go.Figure, name: str) -> None:
        fig.write_html(args.output_dir / f"{name}.html", include_plotlyjs="cdn")

    trajectories, quality = load_data(args.data_url)
    for tag in ("Conversion", "LatexDensity", "CoolingTemp"):
        save(plot_raw(trajectories, tag, FAULT_BATCHES), f"raw-{tag}")

    model = fit_model(trajectories, quality)
    save(model.score_plot(settings=LABELS), "scores")
    save(model.spe_plot(settings=LABELS), "spe")
    save(plot_r2_over_time(r2_breakdown(model)), "r2-over-time")
    save(time_varying_loading_plot(model, component=1), "weights-w1")
    save(time_varying_loading_plot(model, component=2), "weights-w2")

    t1_contributions = diagnose_batch_37(model, trajectories)
    save(unfolded_contribution_plot(t1_contributions, FAULT_FROM_START), "contributions-37-t1")
    save(unfolded_contribution_plot(t1_contributions, FAULT_FROM_START, by_tag=True), "contributions-37-t1-by-tag")
    for tag in ("LatexDensity", "Conversion"):
        save(plot_raw(trajectories, tag, [FAULT_FROM_START]), f"raw-{tag}-batch-37")

    t2_contributions = diagnose_batch_34(model, trajectories)
    save(unfolded_contribution_plot(t2_contributions, FAULT_MID_BATCH), "contributions-34-t2")
    save(
        contribution_at_time_plot(t2_contributions, k=INSPECT_SAMPLE, batch_id=FAULT_MID_BATCH),
        "contributions-34-t2-at-sample",
    )
    for tag in ("CoolingTemp", "JacketTemp", "EnergyReleased"):
        save(plot_raw(trajectories, tag, [FAULT_MID_BATCH]), f"raw-{tag}-batch-34")

    compare_predictions(model, quality)
    for variable in ("Composition", "ParticleSize"):
        save(model.predictions_vs_observed_plot(quality, variable=variable), f"observed-vs-predicted-{variable}")

    y_hat = evolving_prediction(model, trajectories, quality)
    save(plot_online_prediction(y_hat, quality, AVERAGE_BATCH), f"online-prediction-batch-{AVERAGE_BATCH}")
    save(plot_online_rmse(evolving_error(model, trajectories, quality)), "online-rmse")

    monitor, _alarms = online_monitoring(trajectories, quality)
    save(online_monitoring_plot(monitor, trajectories[FAULT_FROM_START], "t2"), f"online-t2-batch-{FAULT_FROM_START}")
    save(online_monitoring_plot(monitor, trajectories[FAULT_MID_BATCH], "spe"), f"online-spe-batch-{FAULT_MID_BATCH}")
    save(plot_forecasts(monitor.model, trajectories), f"forecast-batch-{FAULT_FROM_START}-and-{FAULT_MID_BATCH}")
    print(f"figures written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
