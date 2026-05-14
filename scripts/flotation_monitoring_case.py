"""
Multivariate process monitoring case study for pid-book §7.3.

Fits a 2-component PCA on the phase-1 stretch (first 479 observations,
15 December 2004) of the openmv flotation-cell dataset, then monitors
the phase-2 stretch (16 December onwards, 2443 observations) with
Hotelling's T^2 and SPE. Also produces a univariate Shewhart trace on
the Feed rate column for comparison.

Generates five PNG figures referenced by
`product-development-product-improvement/multivariate-process-monitoring.rst`
into the sister `figures` repository (`figures/monitoring/`).
"""

from __future__ import annotations

import pathlib
from math import gamma, sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from process_improve.multivariate import PCA, MCUVScaler

FIGURES_DIR = pathlib.Path("/home/user/figures/monitoring")
DATA_PATH = pathlib.Path("/tmp/flotation-cell.csv")

N_PHASE1 = 479
N_SUB = 4
CONF_LEVEL = 0.95


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df


def split_phases(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    num = df.drop(columns=["Date and time"])
    return num.iloc[:N_PHASE1].copy(), num.iloc[N_PHASE1:].copy()


def subgroup(x, n_sub: int) -> np.ndarray:
    n_groups = len(x) // n_sub
    return np.asarray(x[: n_groups * n_sub]).reshape((n_groups, n_sub))


def shewhart_limits(xbar_p1: np.ndarray, s_p1: np.ndarray, n_sub: int):
    target = float(xbar_p1.mean())
    sbar = float(s_p1.mean())
    a_n = sqrt(2) * gamma(n_sub / 2) / (sqrt(n_sub - 1) * gamma((n_sub - 1) / 2))
    sigma_hat = sbar / a_n
    lcl = target - 3 * sigma_hat / sqrt(n_sub)
    ucl = target + 3 * sigma_hat / sqrt(n_sub)
    return target, lcl, ucl


def figure_shewhart(phase1, phase2, out: pathlib.Path) -> None:
    sub_p1 = subgroup(phase1["Feed rate"].values, N_SUB)
    sub_p2 = subgroup(phase2["Feed rate"].values, N_SUB)
    xbar_p1 = sub_p1.mean(axis=1)
    s_p1 = sub_p1.std(axis=1, ddof=1)
    xbar_p2 = sub_p2.mean(axis=1)
    target, lcl, ucl = shewhart_limits(xbar_p1, s_p1, N_SUB)
    first_alarm_p2 = int(np.where((xbar_p2 < lcl) | (xbar_p2 > ucl))[0][0])

    fig, ax = plt.subplots(figsize=(11, 4.4))
    idx_p1 = np.arange(len(xbar_p1))
    idx_p2 = np.arange(len(xbar_p1), len(xbar_p1) + len(xbar_p2))
    ax.plot(idx_p1, xbar_p1, "k.-", linewidth=1.0, markersize=4, label="Phase 1 (15 Dec)")
    ax.plot(idx_p2, xbar_p2, "C0.-", linewidth=1.0, markersize=3, label="Phase 2 (16 Dec onwards)")
    ax.axhline(target, color="grey", linestyle=":", linewidth=1, label="target")
    ax.axhline(ucl, color="red", linestyle="--", linewidth=1, label="UCL / LCL (3 sigma)")
    ax.axhline(lcl, color="red", linestyle="--", linewidth=1)
    ax.axvline(len(xbar_p1) - 0.5, color="grey", linestyle=":", linewidth=1)
    ax.set_xlabel("Subgroup index (2 min each)")
    ax.set_ylabel("Feed rate (subgroup mean, t/h)")
    ax.set_title(
        f"Shewhart chart on Feed rate (subgroup size {N_SUB}); "
        f"first phase-2 alarm at subgroup {first_alarm_p2}"
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def figure_score_phase1(model, phase1, out: pathlib.Path) -> None:
    scores = model.scores_
    ci_x, ci_y = model.ellipse_coordinates(score_horiz=1, score_vert=2, conf_level=CONF_LEVEL)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(scores.iloc[:, 0], scores.iloc[:, 1], "k.", markersize=5, alpha=0.7, label="Phase 1 scores")
    ax.plot(ci_x, ci_y, color="palevioletred", linewidth=1.8, label="95% T^2 ellipse")
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(0, color="grey", linewidth=0.5)
    ax.set_xlabel("$t_1$")
    ax.set_ylabel("$t_2$")
    ax.set_title(f"Phase-1 score plot ({len(phase1)} observations, 15 Dec)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def figure_loadings(model, out: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    for k, ax in enumerate(axes):
        p = model.loadings_.iloc[:, k]
        colors = ["#1f77b4" if v >= 0 else "#d62728" for v in p.values]
        ax.bar(p.index, p.values, color=colors)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_ylabel(f"$p_{k + 1}$ loading")
        ax.tick_params(axis="x", rotation=25)
        ax.set_title(f"Loading {k + 1}")
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def figure_t2_spe(t2, spe, t2_lim, spe_lim, out: pathlib.Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(t2.index, t2.values, color="#1f77b4", linewidth=0.9)
    axes[0].axhline(t2_lim, color="red", linestyle="--", linewidth=1, label="95% limit")
    axes[0].set_ylabel("Hotelling's $T^2$")
    axes[0].set_title("Phase-2 multivariate monitoring chart")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(spe.index, spe.values, color="#d62728", linewidth=0.9)
    axes[1].axhline(spe_lim, color="red", linestyle="--", linewidth=1, label="95% limit")
    axes[1].set_ylabel("SPE")
    axes[1].set_xlabel("Observation index (30 s spacing)")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def figure_contribution(contribs: pd.Series, first_alarm_local: int, out: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(contribs.index, contribs.values, color="#d62728")
    ax.set_ylabel("$(x_k - \\hat{x}_k)^2$ in scaled units")
    ax.set_title(
        f"Per-variable SPE contributions at phase-2 obs {first_alarm_local} "
        f"(16 Dec, first SPE alarm)"
    )
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    phase1, phase2 = split_phases(df)
    print(f"phase1 shape: {phase1.shape}  phase2 shape: {phase2.shape}")

    # 1. Univariate Shewhart on Feed rate
    figure_shewhart(phase1, phase2, FIGURES_DIR / "Flotation-MSPC-shewhart.png")
    print("[1/5] shewhart fig done")

    # 2-5. Multivariate PCA on phase 1
    scaler = MCUVScaler().fit(phase1)
    model = PCA(n_components=2).fit(scaler.transform(phase1))
    print(f"R^2 cumulative: {model.r2_cumulative_.values}")

    figure_score_phase1(model, phase1, FIGURES_DIR / "Flotation-MSPC-score-phase1.png")
    print("[2/5] score plot done")

    figure_loadings(model, FIGURES_DIR / "Flotation-MSPC-loadings.png")
    print("[3/5] loadings done")

    result = model.predict(scaler.transform(phase2))
    t2 = result.hotellings_t2.iloc[:, -1]
    spe = result.spe
    t2_lim = float(model.hotellings_t2_limit(conf_level=CONF_LEVEL))
    spe_lim = float(model.spe_limit(conf_level=CONF_LEVEL))
    print(f"95% T^2 limit: {t2_lim:.2f}   95% SPE limit: {spe_lim:.2f}")

    figure_t2_spe(t2, spe, t2_lim, spe_lim,
                  FIGURES_DIR / "Flotation-MSPC-t2-spe.png")
    print("[4/5] T^2/SPE done")

    flagged_spe = spe[spe > spe_lim]
    first_alarm = int(flagged_spe.index[0])
    first_alarm_local = first_alarm - N_PHASE1
    row_scaled = scaler.transform(phase2).loc[first_alarm].values
    row_hat = row_scaled @ model.loadings_.values @ model.loadings_.values.T
    contribs = pd.Series(
        (row_scaled - row_hat) ** 2,
        index=phase1.columns,
        name="SPE contribution",
    )
    print(f"first SPE alarm at phase-2 obs {first_alarm_local}")
    print(contribs.round(3).to_dict())
    figure_contribution(contribs, first_alarm_local,
                        FIGURES_DIR / "Flotation-MSPC-contributions.png")
    print("[5/5] contributions done")

    print()
    print("Wrote 5 PNG figures to:", FIGURES_DIR)


if __name__ == "__main__":
    main()
