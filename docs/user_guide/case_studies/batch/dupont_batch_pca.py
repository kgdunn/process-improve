"""Batch PCA outlier diagnosis on the DuPont batch polymerization reactor.

Case study for issue #155. Industrial nylon is made in a batch autoclave, and
the quality of a batch is only known from a laboratory analysis about twelve
hours after the batch ends, so nothing can be corrected while it runs. This
script builds a batchwise-unfolded PCA on the trajectories of 55 batches and
works through the classic outlier hunt of Nomikos and MacGregor (1995): flag
batches in the score plot and in the SPE, find the responsible variables and
the moment they went wrong with contribution plots, tell an abnormal batch
from a batch that was merely operated differently, and rebuild the model
until the score distribution is even.

Data: https://openmv.net/info/polymerization (55 batches x 100 aligned
samples x 10 tags, scaled for confidentiality), downloaded when the script
runs.

Source: Nomikos, P. and MacGregor, J.F., "Multivariate SPC Charts for
Monitoring Batch Processes", Technometrics, 37(1), 41-59, 1995, and the
ConnectMV latent-variable course notes (2011-2012, CC BY-SA 3.0).

Run from the repository root::

    uv run python docs/user_guide/case_studies/batch/dupont_batch_pca.py --output-dir case-study-output/dupont

Every figure is written as a self-contained HTML file to the output directory.
"""

from __future__ import annotations

import argparse
import pathlib

import pandas as pd
import plotly.graph_objects as go

from process_improve.batch import BatchPCA, load_dupont, time_varying_loading_plot, unfolded_contribution_plot
from process_improve.batch.plotting import plot_all_batches_per_tag

# -- section: constants --
CONF_LEVEL = 0.95
SPE_OUTLIER = 49
SCORE_OUTLIERS = [50, 51, 52, 53, 54, 55]
DIFFERENT_BUT_ACCEPTABLE = [37, 39, 43, 44, 45, 46, 47, 48]
POOR_QUALITY_NOT_VISIBLE = [38, 40, 41, 42]
HIGHLIGHT = '{"color": "red", "width": 4}'  # Plotly line style, JSON-encoded, for the highlighted batches
LABELS = {"show_labels": True}
# -- end: constants --


# -- section: load --
def load_data(url: str | None = None) -> dict:
    """Download the 55 aligned batches; each is a 100-sample by 10-tag frame."""
    batches = load_dupont(url=url)
    first = next(iter(batches.values()))
    print(f"{len(batches)} batches x {first.shape[0]} samples x {first.shape[1]} tags")
    return batches


# -- end: load --


# -- section: raw --
def plot_raw(batches: dict, tag: str, highlight: list[int]) -> go.Figure:
    """Overlay one tag for every batch, with the batches of interest drawn in red."""
    return plot_all_batches_per_tag(
        batches, tag, batches_to_highlight={HIGHLIGHT: highlight}, extra_info=f"highlighting {highlight}"
    )


# -- end: raw --


# -- section: model-a --
def fit_model_a(batches: dict) -> BatchPCA:
    """Two-component batch PCA on all 55 batches.

    Every batch becomes one row of 10 tags x 100 samples = 1000 columns; the
    columns are centred (removing the average trajectory) and scaled to unit
    variance, so the model describes batch-to-batch deviations.
    """
    model = BatchPCA(n_components=2).fit(batches)
    per_component = model.r2_per_component_.to_numpy()
    print(
        f"Model A: R2 per component = {per_component[0]:.3f}, {per_component[1]:.3f}; cumulative = {per_component.sum():.3f}"
    )
    print(
        f"Model A: largest |t1| = {abs_top(model.scores_.iloc[:, 0], 4)}; largest |t2| = {abs_top(model.scores_.iloc[:, 1], 3)}"
    )
    spe = model.spe_.iloc[:, -1]
    print(
        f"Model A: largest SPE = batch {spe.idxmax()} ({spe.max():.1f} vs {CONF_LEVEL:.0%} limit {model.spe_limit(conf_level=CONF_LEVEL):.1f})"
    )
    return model


def abs_top(values: pd.Series, n: int) -> list:
    """Batch identifiers with the largest absolute values, largest first."""
    return values.abs().nlargest(n).index.tolist()


# -- end: model-a --


# -- section: spe-49 --
def diagnose_spe_outlier(
    model: BatchPCA, batches: dict, batch_id: int = SPE_OUTLIER
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Which variables, and when, push one batch off the model plane.

    The signed SPE contributions are the residuals of every (tag, time) cell;
    their squares add up to the batch's SPE, so the squares are each cell's
    share of it. Summing the shares per tag ranks the variables; summing them
    per time sample locates the event.
    """
    spe_share = model.spe_contributions(model.unfold_and_scale(batches)) ** 2
    by_tag = spe_share.loc[batch_id].groupby(level="tag", sort=False).sum()
    by_time = spe_share.loc[batch_id].groupby(level="sequence").sum()
    print(
        f"Batch {batch_id}: SPE share per tag = "
        + ", ".join(f"{tag} {share:.0%}" for tag, share in (by_tag / by_tag.sum()).items())
    )
    peak = by_time.nlargest(7).index
    print(f"Batch {batch_id}: the seven largest per-sample shares sit at samples {sorted(int(k) for k in peak)}")
    return spe_share, by_tag, by_time


def plot_share_over_time(by_time: pd.Series, batch_id: int) -> go.Figure:
    """Bar chart of a batch's SPE share per time sample."""
    fig = go.Figure(go.Bar(x=list(by_time.index), y=by_time.to_numpy()))
    fig.update_layout(
        title=f"SPE share per time sample, batch {batch_id}",
        xaxis_title="Time [sequence order]",
        yaxis_title="Share of SPE",
    )
    return fig


# -- end: spe-49 --


# -- section: score-outliers --
def diagnose_score_outliers(model: BatchPCA, batches: dict) -> dict[str, pd.DataFrame]:
    """Score contributions for the batches that stand out on t1 and t2."""
    scaled = model.unfold_and_scale(batches)
    contributions = {
        "t1": model.score_contributions(scaled, component=1),
        "t2": model.score_contributions(scaled, component=2),
    }
    for batch_id, component in ((54, "t1"), (55, "t2")):
        by_tag = contributions[component].loc[batch_id].groupby(level="tag", sort=False).sum()
        print(
            f"Batch {batch_id}: {component} contributions per tag = "
            + ", ".join(f"{tag} {value:+.1f}" for tag, value in by_tag.items())
        )
    return contributions


# -- end: score-outliers --


# -- section: model-b --
def fit_model_b(batches: dict) -> BatchPCA:
    """Rebuild without batches 49 to 55: a second cluster appears on t2 and t3."""
    kept = {batch_id: batch for batch_id, batch in batches.items() if batch_id < SPE_OUTLIER}
    model = BatchPCA(n_components=3).fit(kept)
    per_component = model.r2_per_component_.to_numpy()
    print(f"Model B ({len(kept)} batches): R2 per component = " + ", ".join(f"{value:.3f}" for value in per_component))
    print(
        f"Model B: largest |t2| = {abs_top(model.scores_.iloc[:, 1], 6)}; largest |t3| = {abs_top(model.scores_.iloc[:, 2], 6)}"
    )
    return model


def diagnose_different_batch(model: BatchPCA, batches: dict, batch_id: int = 39) -> pd.DataFrame:
    """Return the t3 contributions of a batch from the second cluster."""
    kept = {key: value for key, value in batches.items() if key in model.batch_ids_}
    contributions = model.score_contributions(model.unfold_and_scale(kept), component=3)
    by_tag = contributions.loc[batch_id].groupby(level="tag", sort=False).sum()
    print(
        f"Batch {batch_id}: t3 contributions per tag = "
        + ", ".join(f"{tag} {value:+.1f}" for tag, value in by_tag.items())
    )
    return contributions


# -- end: model-b --


# -- section: model-c --
def fit_model_c(batches: dict) -> BatchPCA:
    """Rebuild once more without the second cluster: the reference model."""
    excluded = set(range(SPE_OUTLIER, 56)) | set(DIFFERENT_BUT_ACCEPTABLE)
    kept = {batch_id: batch for batch_id, batch in batches.items() if batch_id not in excluded}
    model = BatchPCA(n_components=3).fit(kept)
    per_component = model.r2_per_component_.to_numpy()
    print(f"Model C ({len(kept)} batches): R2 per component = " + ", ".join(f"{value:.3f}" for value in per_component))
    return model


def observability_table(model: BatchPCA) -> pd.DataFrame:
    """T2 and SPE of the poor-quality batches that the trajectories do not reveal."""
    table = pd.DataFrame(
        {
            "T2": model.hotellings_t2_.loc[POOR_QUALITY_NOT_VISIBLE].iloc[:, -1],
            "T2 limit": model.hotellings_t2_limit(conf_level=CONF_LEVEL),
            "SPE": model.spe_.loc[POOR_QUALITY_NOT_VISIBLE].iloc[:, -1],
            "SPE limit": model.spe_limit(conf_level=CONF_LEVEL),
        }
    )
    print("Model C: poor-quality batches against the limits\n" + table.round(2).to_string())
    return table


# -- end: model-c --


def main(argv: list[str] | None = None) -> int:
    """Run the whole case study and write its figures."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("case-study-output/dupont"))
    parser.add_argument("--data-url", default=None, help="override the openmv.net URL (for example a file:// copy)")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def save(fig: go.Figure, name: str) -> None:
        fig.write_html(args.output_dir / f"{name}.html", include_plotlyjs="cdn")

    batches = load_data(args.data_url)
    save(plot_raw(batches, "TempC-1", SCORE_OUTLIERS), "raw-TempC-1-score-outliers")
    save(plot_raw(batches, "Press-1", SCORE_OUTLIERS), "raw-Press-1-score-outliers")

    model_a = fit_model_a(batches)
    save(model_a.score_plot(settings=LABELS), "model-a-scores")
    save(model_a.spe_plot(settings=LABELS), "model-a-spe")

    save(plot_raw(batches, "Flow-1", [SPE_OUTLIER]), "raw-Flow-1-batch-49")
    save(plot_raw(batches, "TempC-1", [SPE_OUTLIER]), "raw-TempC-1-batch-49")
    spe_share, _by_tag, by_time = diagnose_spe_outlier(model_a, batches)
    save(unfolded_contribution_plot(spe_share, SPE_OUTLIER), "spe-contributions-49")
    save(unfolded_contribution_plot(spe_share, SPE_OUTLIER, by_tag=True), "spe-contributions-49-by-tag")
    save(plot_share_over_time(by_time, SPE_OUTLIER), "spe-contributions-49-by-time")

    save(time_varying_loading_plot(model_a, component=1), "model-a-loadings-p1")
    contributions = diagnose_score_outliers(model_a, batches)
    save(unfolded_contribution_plot(contributions["t1"], 54), "score-contributions-54-t1")
    save(plot_raw(batches, "Press-2", [54]), "raw-Press-2-batch-54")
    save(unfolded_contribution_plot(contributions["t2"], 55), "score-contributions-55-t2")

    model_b = fit_model_b(batches)
    save(model_b.score_plot(pc_horiz=2, pc_vert=3, settings=LABELS), "model-b-scores-t2-t3")
    save(
        unfolded_contribution_plot(diagnose_different_batch(model_b, batches), 39, by_tag=True),
        "score-contributions-39-t3",
    )
    save(plot_raw(batches, "Press-3", [39]), "raw-Press-3-batch-39")

    model_c = fit_model_c(batches)
    save(model_c.score_plot(settings=LABELS), "model-c-scores")
    observability_table(model_c)
    print(f"figures written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
