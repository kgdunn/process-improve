"""Batch PLS fault diagnosis on the simulated SBR batch reactor.

Case study for issue #156. Styrene-butadiene rubber is made by emulsion
polymerization in a batch reactor; five quality attributes of the latex are
measured at the end of each batch. Because these 53 batches were simulated
from a first-principles model, the fault is known: batches 34 and 37 both
received 30% more organic impurity in the butadiene feed, from the very
start of batch 37 and from midway through batch 34. This script fits a
batchwise-unfolded PLS from the six trajectories to the five quality
attributes and follows the diagnosis of the course notes: the score plot
flags both batches, the whole-batch SPE does not, the weights and the
contribution plots name the variables, and the raw trajectories confirm the
story. The same fault lands in two different places of the score plot
because it started at two different times.

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

from process_improve.batch import (
    BatchPLS,
    contribution_at_time_plot,
    load_sbr,
    time_varying_loading_plot,
    unfolded_contribution_plot,
)
from process_improve.batch.plotting import plot_all_batches_per_tag

# -- section: constants --
CONF_LEVEL = 0.95
FAULT_FROM_START = 37
FAULT_MID_BATCH = 34
FAULT_BATCHES = [FAULT_MID_BATCH, FAULT_FROM_START]
INSPECT_SAMPLE = 120  # a sample well after the mid-batch fault has developed
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
    by_tag = contributions.loc[FAULT_FROM_START].groupby(level="tag", sort=False).sum()
    print(
        f"batch {FAULT_FROM_START}: t1 contributions per tag = "
        + ", ".join(f"{tag} {value:+.1f}" for tag, value in by_tag.items())
    )
    return contributions


# -- end: batch-37 --


# -- section: batch-34 --
def diagnose_batch_34(model: BatchPLS, trajectories: dict) -> pd.DataFrame:
    """Score contributions to t2: the same fault, but starting midway through batch 34."""
    contributions = model.score_contributions(model.unfold_and_scale(trajectories), component=2)
    row = contributions.loc[FAULT_MID_BATCH]
    by_tag = row.groupby(level="tag", sort=False).sum()
    by_time = row.groupby(level="sequence").sum()
    onset = int(by_time.index[(by_time.cumsum() > 0.05 * by_time.sum()).to_numpy().argmax()])
    print(
        f"batch {FAULT_MID_BATCH}: t2 contributions per tag = "
        + ", ".join(f"{tag} {value:+.1f}" for tag, value in by_tag.items())
    )
    print(f"batch {FAULT_MID_BATCH}: 5% of the t2 contribution has accumulated by sample {onset}")
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
    print(f"figures written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
