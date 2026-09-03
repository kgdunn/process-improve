"""Multiblock batch PLS on the FMC batch dryer.

Case study for issue #154. An agricultural chemical is dried in an industrial
batch dryer: wet cake (solid plus embedded solvent) is charged, dried through
three recipe phases (solvent collection, a temperature ramp, cool-down), and
the solvent is collected in a side tank. Chemical changes take place in the
solid during drying, and the operators adjust a few set points. Besides the
ten process trajectories there are three one-row-per-batch blocks: the
chemistry of the cake before the batch (Zchem), the operating conditions and
recipe timings (Zop), and eight final quality attributes (Y).

This script follows the ladder of models in the original course material, two
components each: PCA on the quality block, PLS from each initial-condition
block, multiblock PLS on both, batch PCA on the trajectories, batch PLS to
quality, and finally the batch multiblock PLS that joins all three X blocks.
The trajectories were aligned within each phase before the data were
archived; ``ClockTime``, the wall time at each aligned sample, is carried
along as a trajectory so the warping itself is part of the data.

The data contain genuine missing cells, so the models here are the
:mod:`process_improve.multivariate` estimators, whose NIPALS path handles
missing values; the batch classes ``BatchPCA`` and ``BatchPLS`` require
complete data.

Data: https://openmv.net/info/batch-dryer (59 batches x 325 aligned samples,
four blocks), downloaded when the script runs. Thirteen batches without
chemistry measurements are excluded, as in the original study.

Source: Garcia-Munoz, S., Kourti, T., MacGregor, J.F., Mateos, A.G. and
Murphy, G., "Troubleshooting of an Industrial Batch Process Using
Multivariate Methods", Industrial and Engineering Chemistry Research, 42,
3592-3601, 2003, and the ConnectMV latent-variable course notes (2011-2012,
CC BY-SA 3.0).

Run from the repository root::

    uv run python docs/user_guide/case_studies/batch/fmc_multiblock_batch_pls.py --output-dir case-study-output/fmc

Every figure is written as a self-contained HTML file to the output directory.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.utils import Bunch

from process_improve.batch import dict_to_wide, load_fmc, time_varying_loading_plot, unfolded_contribution_plot
from process_improve.batch.plotting import plot_all_batches_per_tag
from process_improve.multivariate import PCA, PLS, MCUVScaler
from process_improve.multivariate.methods import MBPLS

# -- section: constants --
N_COMPONENTS = 2
CONF_LEVEL = 0.95
QUALITY_GROUP = [61, 14]  # one batch from each group in the quality score plot
OPERATING_OUTLIER = 20  # stands out on the operating conditions and on the trajectories
TRAJECTORY_BATCHES = [13, 5, 7]  # batches examined in the batch PLS
HIGHLIGHT = '{"color": "red", "width": 4}'  # Plotly line style, JSON-encoded, for the highlighted batches
LABELS = {"show_labels": True}
# -- end: constants --


# -- section: load --
def load_and_exclude(url: str | None = None) -> Bunch:
    """Download the four blocks and drop the batches without chemistry data."""
    fmc = load_fmc(url=url)
    keep = [batch_id for batch_id in fmc.batch_ids if batch_id not in fmc.missing_chemistry]
    data = Bunch(
        X={batch_id: fmc.X[batch_id] for batch_id in keep},
        Y=fmc.Y.loc[keep],
        Zop=fmc.Zop.loc[keep],
        Zchem=fmc.Zchem.loc[keep],
    )
    incomplete = [batch_id for batch_id, batch in data.X.items() if batch.isna().any().any()]
    print(
        f"{len(keep)} batches kept; missing cells: Y {int(data.Y.isna().sum().sum())}, Zchem {int(data.Zchem.isna().sum().sum())}, X in batches {incomplete}"
    )
    return data


# -- end: load --


# -- section: raw --
def plot_raw(trajectories: dict, tag: str, highlight: list[int]) -> go.Figure:
    """Overlay one tag for every batch, with the batches of interest drawn in red."""
    return plot_all_batches_per_tag(
        trajectories, tag, batches_to_highlight={HIGHLIGHT: highlight}, extra_info=f"highlighting {highlight}"
    )


# -- end: raw --


# -- section: quality --
def pca_on_quality(quality: pd.DataFrame) -> tuple[PCA, pd.DataFrame]:
    """Two-component PCA of the quality block, to see how the batches group in quality space."""
    y_scaled = MCUVScaler().fit_transform(quality)  # missing cells pass through; PCA switches to NIPALS
    model = PCA(n_components=N_COMPONENTS).fit(y_scaled)
    print(f"PCA on Y: R2 cumulative = {cumulative(model.r2_cumulative_)}")
    contributions = model.score_contributions(y_scaled, component=1)
    for batch_id in QUALITY_GROUP:
        print(f"batch {batch_id}: t1 contributions = " + describe(contributions.loc[batch_id]))
    return model, y_scaled


def describe(values: pd.Series) -> str:
    """Format a contribution vector, naming the cells that are missing in the data."""
    return ", ".join(f"{name} {'missing' if pd.isna(value) else f'{value:+.2f}'}" for name, value in values.items())


def cumulative(values: pd.Series | np.ndarray) -> str:
    """Format a cumulative R2 vector."""
    return ", ".join(f"{value:.3f}" for value in np.asarray(values, dtype=float))


def plot_bars(values: pd.Series, title: str) -> go.Figure:
    """Bar chart of one contribution vector."""
    fig = go.Figure(go.Bar(x=[str(name) for name in values.index], y=values.to_numpy(dtype=float)))
    fig.update_layout(title=title, yaxis_title="Contribution")
    return fig


# -- end: quality --


# -- section: initial-conditions --
def pls_from_initial_conditions(data: Bunch, y_scaled: pd.DataFrame) -> tuple[PLS, PLS, pd.DataFrame]:
    """PLS from each initial-condition block to quality, one block at a time.

    The blocks are scaled with ``MCUVScaler`` first and the models fitted with
    ``scale=False``, so every later contribution plot works in the same scaled
    space as the model.
    """
    zchem_scaled = MCUVScaler().fit_transform(data.Zchem)
    zop_scaled = MCUVScaler().fit_transform(data.Zop)
    pls_chem = PLS(n_components=N_COMPONENTS, scale=False).fit(zchem_scaled, y_scaled)
    pls_op = PLS(n_components=N_COMPONENTS, scale=False).fit(zop_scaled, y_scaled)
    print(f"PLS Zchem -> Y: R2Y cumulative = {cumulative(pls_chem.r2_cumulative_)}")
    print(f"PLS Zop -> Y: R2Y cumulative = {cumulative(pls_op.r2_cumulative_)}")
    contributions = pls_op.score_contributions(zop_scaled, component=1).loc[OPERATING_OUTLIER]
    print(
        f"batch {OPERATING_OUTLIER} on Zop: t1 contributions = "
        + ", ".join(f"{name} {value:+.2f}" for name, value in contributions.items())
    )
    return pls_chem, pls_op, zop_scaled


# -- end: initial-conditions --


# -- section: multiblock-z --
def mbpls_on_initial_conditions(data: Bunch) -> MBPLS:
    """Multiblock PLS from both initial-condition blocks to quality.

    Each block is scaled on its own and then weighted by 1 / sqrt(K_b), so the
    eleven chemistry columns and the nine operating columns pull on the
    super-score with equal total weight.
    """
    blocks = {"Zchem": data.Zchem, "Zop": data.Zop}
    model = MBPLS(n_components=N_COMPONENTS).fit(blocks, data.Y)
    print(
        f"MBPLS Z -> Y: R2Y cumulative = {cumulative(model.r2_y_cumulative_)}; R2X per block after {N_COMPONENTS} components = "
        + ", ".join(f"{name} {value:.3f}" for name, value in model.r2_x_per_block_cumulative_.iloc[:, -1].items())
    )
    for name, block_contributions in model.score_contributions(blocks, component=1).items():
        row = block_contributions.loc[OPERATING_OUTLIER]
        print(
            f"batch {OPERATING_OUTLIER}, block {name}: t1 contributions = "
            + ", ".join(f"{col} {value:+.2f}" for col, value in row.items())
        )
    return model


# -- end: multiblock-z --


# -- section: unfold --
def unfold_trajectories(trajectories: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Unfold the trajectories batchwise and scale every column.

    One row per batch, 10 tags x 325 samples = 3250 columns. ``MCUVScaler``
    returns flat column labels, so the 2-level ``(tag, sequence)`` index is
    re-attached; the batch plots need it.
    """
    wide = dict_to_wide({batch_id: batch.drop(columns="ClockTime") for batch_id, batch in trajectories.items()})
    x_scaled = MCUVScaler().fit_transform(wide)
    x_scaled.columns = wide.columns
    print(
        f"unfolded trajectories: {wide.shape[0]} batches x {wide.shape[1]} columns, {int(wide.isna().sum().sum())} missing cells"
    )
    return wide, x_scaled


# -- end: unfold --


# -- section: batch-pca --
def batch_pca_on_trajectories(x_scaled: pd.DataFrame) -> tuple[PCA, pd.DataFrame]:
    """Two-component batch PCA on the trajectories alone."""
    model = PCA(n_components=N_COMPONENTS).fit(x_scaled)  # NIPALS, because of the missing cells
    print(f"batch PCA on X: R2 cumulative = {cumulative(model.r2_cumulative_)}")
    spe_share = model.spe_contributions(x_scaled) ** 2  # all-NaN rows for the batches with missing cells
    complete = spe_share.dropna(how="all")
    worst = model.spe_.loc[complete.index].iloc[:, -1].idxmax()
    by_tag = spe_share.loc[worst].groupby(level="tag", sort=False).sum()
    print(
        f"largest SPE among the complete batches: batch {worst}; share per tag = "
        + ", ".join(f"{tag} {share:.0%}" for tag, share in (by_tag / by_tag.sum()).items())
    )
    return model, spe_share


# -- end: batch-pca --


# -- section: batch-pls --
def batch_pls_to_quality(x_scaled: pd.DataFrame, y_scaled: pd.DataFrame) -> tuple[PLS, pd.DataFrame]:
    """Two-component batch PLS from the unfolded trajectories to quality."""
    model = PLS(n_components=N_COMPONENTS, scale=False).fit(x_scaled, y_scaled)
    print(f"batch PLS X -> Y: R2Y cumulative = {cumulative(model.r2_cumulative_)}")
    contributions = model.score_contributions(x_scaled, component=1)
    by_tag = contributions.loc[TRAJECTORY_BATCHES[0]].groupby(level="tag", sort=False).sum()
    print(
        f"batch {TRAJECTORY_BATCHES[0]}: t1 contributions per tag = "
        + ", ".join(f"{tag} {value:+.1f}" for tag, value in by_tag.items())
    )
    return model, contributions


# -- end: batch-pls --


# -- section: batch-mbpls --
def batch_mbpls(data: Bunch, wide: pd.DataFrame) -> tuple[MBPLS, dict]:
    """Batch multiblock PLS: chemistry, operating conditions and trajectories to quality."""
    blocks = {"Zchem": data.Zchem, "Zop": data.Zop, "X": wide}
    model = MBPLS(n_components=N_COMPONENTS).fit(blocks, data.Y)
    print(
        f"batch MBPLS: R2Y cumulative = {cumulative(model.r2_y_cumulative_)}; R2X per block after {N_COMPONENTS} components = "
        + ", ".join(f"{name} {value:.3f}" for name, value in model.r2_x_per_block_cumulative_.iloc[:, -1].items())
    )
    print("super VIP per block: " + ", ".join(f"{name} {value:.2f}" for name, value in model.super_vip_.items()))
    return model, blocks


# -- end: batch-mbpls --


def main(argv: list[str] | None = None) -> int:
    """Run the whole case study and write its figures."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("case-study-output/fmc"))
    parser.add_argument("--data-url", default=None, help="override the openmv.net URL (for example a file:// copy)")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def save(fig: go.Figure, name: str) -> None:
        fig.write_html(args.output_dir / f"{name}.html", include_plotlyjs="cdn")

    data = load_and_exclude(args.data_url)
    for tag in ("D-Temp", "J-Temp", "CTankLvl", "ClockTime"):
        save(plot_raw(data.X, tag, [OPERATING_OUTLIER]), f"raw-{tag}")

    pca_y, y_scaled = pca_on_quality(data.Y)
    save(pca_y.score_plot(settings=LABELS), "quality-pca-scores")
    y_contributions = pca_y.score_contributions(y_scaled, component=1)
    for batch_id in QUALITY_GROUP:
        save(
            plot_bars(y_contributions.loc[batch_id], f"t1 contributions of batch {batch_id} (PCA on Y)"),
            f"quality-pca-contributions-{batch_id}",
        )

    _pls_chem, pls_op, zop_scaled = pls_from_initial_conditions(data, y_scaled)
    save(pls_op.score_plot(settings=LABELS), "pls-zop-scores")
    save(
        plot_bars(
            pls_op.score_contributions(zop_scaled, component=1).loc[OPERATING_OUTLIER],
            f"t1 contributions of batch {OPERATING_OUTLIER} (PLS Zop)",
        ),
        "pls-zop-contributions-20",
    )

    mbpls_z = mbpls_on_initial_conditions(data)
    save(mbpls_z.super_score_plot(), "mbpls-z-super-scores")
    save(mbpls_z.super_weights_bar_plot(component=1), "mbpls-z-super-weights")

    wide, x_scaled = unfold_trajectories(data.X)
    pca_x, spe_share = batch_pca_on_trajectories(x_scaled)
    save(pca_x.score_plot(settings=LABELS), "batch-pca-scores")
    save(pca_x.spe_plot(settings=LABELS), "batch-pca-spe")
    save(time_varying_loading_plot(pca_x, component=1), "batch-pca-loadings-p1")
    worst = pca_x.spe_.loc[spe_share.dropna(how="all").index].iloc[:, -1].idxmax()
    save(unfolded_contribution_plot(spe_share, worst, by_tag=True), f"batch-pca-spe-contributions-{worst}")
    for tag in ("D-Temp", "Power", "Torque"):
        save(plot_raw(data.X, tag, [OPERATING_OUTLIER]), f"raw-{tag}-batch-20")

    pls_x, x_contributions = batch_pls_to_quality(x_scaled, y_scaled)
    save(pls_x.score_plot(settings=LABELS), "batch-pls-scores")
    save(unfolded_contribution_plot(x_contributions, TRAJECTORY_BATCHES[0]), "batch-pls-contributions-13")
    for batch_id in TRAJECTORY_BATCHES:
        save(plot_raw(data.X, "D-Temp", [batch_id]), f"raw-D-Temp-batch-{batch_id}")
    save(
        pls_x.predictions_vs_observed_plot(y_observed=y_scaled, variable="SolventConc"),
        "batch-pls-observed-vs-predicted",
    )

    mbpls_x, blocks = batch_mbpls(data, wide)
    save(mbpls_x.super_score_plot(), "batch-mbpls-super-scores")
    save(mbpls_x.super_weights_bar_plot(component=1), "batch-mbpls-super-weights")
    save(
        unfolded_contribution_plot(mbpls_x.score_contributions(blocks, component=1)["X"], TRAJECTORY_BATCHES[0]),
        "batch-mbpls-x-contributions-13",
    )
    save(mbpls_x.predictions_vs_observed_plot(data.Y, variable="SolventConc"), "batch-mbpls-observed-vs-predicted")
    print(f"figures written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
