"""Generate the committed PNGs for the adaptive soft-sensor case study.

Mirrors the analysis in the pid-book chapter
``product-development-product-improvement/multivariate-process-monitoring.rst``
(the "Keeping a model current" section): a static PLS soft sensor drifts, an
adaptive PLS tracks it, monitoring statistics flag the drift, the adaptation
splits into a preprocessing part and a kernel part, and the distance
metric shows the model ageing. The chapter shows the equivalent Plotly code; the
committed figures are these matplotlib renderings.

Usage::

    python scripts/adaptive-softsensor-figures.py [output_dir]

``output_dir`` defaults to ``figures/monitoring`` (the kgdunn/figures repo).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from process_improve.multivariate import AdaptivePLS

DATA_URL = "https://openmv.net/file/vapor-pressure.csv"
A = 3  # number of PLS components, used throughout
DARK_BLUE = "#1f3d7a"  # default line colour for all figures
ORANGE = "#c55a11"     # the "Testing data" divider and the kernel part
GREEN = "#2e6f3e"      # the static model in the payoff figure
GREY = "0.45"
plt.rcParams.update({
    "font.size": 11, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 140,
    "axes.prop_cycle": mpl.cycler(color=[DARK_BLUE]),
})
XLABEL = "Time since start [months]"


def bias_std_rmsep(err: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    e = err[mask]
    return float(e.mean()), float(e.std()), float(np.sqrt((e**2).mean()))


def testing_divider(ax, x0: float, y: float, reach: float) -> None:
    """Orange dashed testing boundary: a right-pointing arrow starting on the line,
    with the 'Testing data' label centred above the arrow (not across it).
    """
    ax.axvline(x0, color=ORANGE, ls="--", lw=1.2)
    # horizontal arrow starting on the dashed line, pointing right
    ax.annotate("", xy=(x0 + reach, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.4))
    # label centred above the arrow
    ax.annotate("Testing data", xy=(x0 + reach / 2, y), xytext=(0, 5),
                textcoords="offset points", color=ORANGE, fontsize=9,
                fontweight="bold", ha="center", va="bottom")


def main(out_dir: Path) -> None:  # noqa: PLR0915
    out_dir.mkdir(parents=True, exist_ok=True)
    vp = pd.read_csv(DATA_URL)
    vp["month"] = vp["hours_elapsed"] / 730.5
    tags = [c for c in vp.columns if c not in ("hours_elapsed", "month", "vapour_pressure_kpa", "current_estimator")]
    lab_rows = np.where(vp["vapour_pressure_kpa"].notna().to_numpy())[0]
    lab = vp.iloc[lab_rows].reset_index(drop=True)
    y_lab = lab["vapour_pressure_kpa"].to_numpy()
    n_train = len(lab) // 2
    train = lab.index < n_train
    drift_month = float(np.quantile(lab["month"], 0.60))
    post = lab["month"].to_numpy() >= drift_month

    Xrow = vp[tags].to_numpy()
    month = vp["month"].to_numpy()

    def stream(model, learn=None, y_update=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pred = np.zeros(len(vp))
        t2 = np.zeros(len(vp))
        spe = np.zeros(len(vp))
        dist = np.zeros(len(vp))
        for i in range(len(vp)):
            may_learn = True if learn is None else bool(learn[i])
            if may_learn:
                yv = None if (y_update is None or np.isnan(y_update[i])) else np.array([y_update[i]])
                out = model.update(Xrow[i], y_row=yv)
                pred[i], t2[i], spe[i], dist[i] = out.prediction[0], out.hotellings_t2, out.spe, out.distance
            else:
                pred[i] = model.predict(vp[tags].iloc[[i]]).to_numpy().ravel()[0]
                dist[i] = dist[i - 1] if i else model.n_components
        return pred, t2, spe, dist

    static = AdaptivePLS(
        n_components=A, forgetting_factor=0, gamma=0, lambda_center=0, alpha_scale=0,
        lambda_center_y=0, alpha_scale_y=0, adaptive_spe_limit=False, conf_level=0.99,
    ).fit(lab.loc[train, tags], lab.loc[train, ["vapour_pressure_kpa"]])
    static_pred, static_t2, static_spe, _ = stream(static)
    t2_lim = float(static.hotellings_t2_limit(conf_level=0.99))
    spe_lim = float(static.update(Xrow[0]).spe_limit)
    learn = static_spe < 2.5 * spe_lim

    def ewma_smooth(values, lam=0.35) -> np.ndarray:
        out = values.astype(float).copy()
        idx = np.where(~np.isnan(out))[0]
        for k in range(1, len(idx)):
            out[idx[k]] = lam * out[idx[k]] + (1 - lam) * out[idx[k - 1]]
        return out

    adaptive = AdaptivePLS(
        n_components=A, forgetting_factor=0.01, gamma=0.05, lambda_center=0.003, alpha_scale=0.012,
        lambda_center_y=0.12, alpha_scale_y=0.05, update_when_out_of_control=True, conf_level=0.99,
    ).fit(lab.loc[train, tags], lab.loc[train, ["vapour_pressure_kpa"]])
    # frozen training snapshot for the channel decomposition
    mx0, sx0, my0, sy0 = adaptive.mx_.copy(), adaptive.sx_.copy(), adaptive.my_.copy(), adaptive.sy_.copy()
    beta0 = adaptive._beta_scaled.copy()
    norm_beta0 = float(np.linalg.norm(beta0))
    y_update = ewma_smooth(vp["vapour_pressure_kpa"].to_numpy())

    prep_ch = np.full(len(vp), np.nan)
    kernel_ch = np.full(len(vp), np.nan)
    center_shift = np.zeros(len(vp))
    beta_shift = np.zeros(len(vp))
    adaptive_pred = np.zeros(len(vp))
    distance = np.zeros(len(vp))
    for i in range(len(vp)):
        x0 = Xrow[i]
        xs_cur = np.nan_to_num((x0 - adaptive.mx_) / adaptive.sx_, nan=0.0)
        xs_train = np.nan_to_num((x0 - mx0) / sx0, nan=0.0)
        y_train = float((my0 + sy0 * (beta0.T @ xs_train))[0])
        y_prep = float((adaptive.my_ + adaptive.sy_ * (beta0.T @ xs_cur))[0])
        y_full = float((adaptive.my_ + adaptive.sy_ * (adaptive._beta_scaled.T @ xs_cur))[0])
        center_shift[i] = float(np.linalg.norm((adaptive.mx_ - mx0) / sx0))
        beta_shift[i] = float(np.linalg.norm(adaptive._beta_scaled - beta0)) / norm_beta0
        if learn[i]:
            prep_ch[i] = y_prep - y_train
            kernel_ch[i] = y_full - y_prep
            yv = None if np.isnan(y_update[i]) else np.array([y_update[i]])
            out = adaptive.update(x0, y_row=yv)
            adaptive_pred[i], distance[i] = out.prediction[0], out.distance
        else:
            adaptive_pred[i] = y_full
            distance[i] = distance[i - 1] if i else A

    err_static = static_pred[lab_rows] - y_lab
    err_adaptive = adaptive_pred[lab_rows] - y_lab
    lab_month = lab["month"].to_numpy()

    # Fig 1: motivation
    fig, ax = plt.subplots(figsize=(8.2, 3.3))
    ax.plot(lab_month, y_lab, ".", ms=4, color=GREY, label="Lab reference")
    ax.plot(month, static_pred, "-", lw=0.9, color=DARK_BLUE, label="Static PLS prediction")
    ax.set_ylim(15, 100)
    testing_divider(ax, drift_month, 86, 5.0)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel("Vapour pressure [kPa]")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_dir / "adaptive-softsensor-motivation.png")
    plt.close(fig)

    # Fig 2: monitoring, with asterisks marking the lab-sample times
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.2, 4.2), sharex=True)
    a1.plot(month, static_t2, lw=0.5, color=DARK_BLUE)
    a1.axhline(t2_lim, color="k", lw=1)
    a1.plot(lab_month, np.full_like(lab_month, 30.0), "*", ms=5, color=ORANGE, label="Lab sample")
    a1.set_ylim(0, 3 * t2_lim)
    a1.set_ylabel("Hotelling's $T^2$")
    a1.set_title("99% limits", fontsize=10)
    a1.legend(loc="upper right", fontsize=8)
    a2.plot(month, static_spe, lw=0.5, color=DARK_BLUE)
    a2.axhline(spe_lim, color="k", lw=1)
    a2.plot(lab_month, np.full_like(lab_month, 20.0), "*", ms=5, color=ORANGE)
    a2.set_ylim(0, 3 * spe_lim)
    a2.set_ylabel("SPE")
    a2.set_xlabel(XLABEL)
    for a in (a1, a2):
        a.axvline(drift_month, color="k", ls="--", lw=1)
    fig.tight_layout()
    fig.savefig(out_dir / "adaptive-softsensor-monitoring.png")
    plt.close(fig)

    # Fig 3: payoff (static = dark-green squares, adaptive = blue circles)
    fig, ax = plt.subplots(figsize=(8.2, 3.3))
    ax.axhspan(-3, 3, color="0.85", alpha=0.6)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.plot(lab_month, err_static, "s", ms=4.5, color=GREEN, alpha=0.7, label="Static PLS")
    ax.plot(lab_month, err_adaptive, "o", ms=4.5, color=DARK_BLUE, alpha=0.7, label="Adaptive PLS")
    ax.set_ylim(-20, 22)
    testing_divider(ax, drift_month, 18, 5.0)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel("Prediction error [kPa]")
    ax.legend(loc="lower left", fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "adaptive-softsensor-payoff.png")
    plt.close(fig)

    # Fig 4: diagnostics (distance metric), zoomed to the meaningful band
    fig, ax = plt.subplots(figsize=(8.2, 3.0))
    ax.plot(month, distance, lw=0.8, color=DARK_BLUE)
    ax.axhline(A, color="0.6", ls=":", lw=1)
    ax.set_ylim(1.5, 3.05)
    testing_divider(ax, drift_month, 2.65, 4.0)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel("Subspace overlap [components]")
    fig.tight_layout()
    fig.savefig(out_dir / "adaptive-softsensor-diagnostics.png")
    plt.close(fig)

    # Fig 5: adaptation decomposition (preprocessing vs kernel parts + state drift)
    fig, (b1, b2) = plt.subplots(2, 1, figsize=(8.2, 5.0), sharex=True)
    b1.axhline(0, color="0.6", lw=0.8)
    b1.plot(month, prep_ch, lw=0.8, color=DARK_BLUE, label="Preprocessing part")
    b1.plot(month, kernel_ch, lw=0.8, color=ORANGE, label="Kernel part")
    b1.set_ylim(-25, 25)
    b1.set_ylabel("Correction vs static [kPa]")
    b1.axvline(drift_month, color="k", ls="--", lw=1)
    b1.legend(loc="upper left", fontsize=8, framealpha=0.9)
    b1.set_title("What drives the adaptation", fontsize=10)
    # testing-data divider on the bottom panel only (the top panel is already busy)
    testing_divider(b2, drift_month, 11.5, 4.0)
    ln1, = b2.plot(month, center_shift, lw=1.0, color=DARK_BLUE, label="Centre migration (preprocessing)")
    b2.set_ylabel("Centre migration [training SD]", color=DARK_BLUE)
    b2.tick_params(axis="y", labelcolor=DARK_BLUE)
    b2.set_ylim(0, 13)
    b2b = b2.twinx()
    ln2, = b2b.plot(month, A - distance, lw=1.0, color=ORANGE, label="Subspace rotation (kernel)")
    b2b.set_ylabel("Subspace rotation [components]", color=ORANGE)
    b2b.tick_params(axis="y", labelcolor=ORANGE)
    b2b.set_ylim(0, A)
    b2b.grid(visible=False)
    b2.set_xlabel(XLABEL)
    b2.legend(handles=[ln1, ln2], loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_dir / "adaptive-softsensor-decomposition.png")
    plt.close(fig)

    print(f"wrote 5 figures to {out_dir}")
    print("static post-drift  bias/std/RMSEP:", tuple(round(v, 1) for v in bias_std_rmsep(err_static, post)))
    print("adaptive post-drift bias/std/RMSEP:", tuple(round(v, 1) for v in bias_std_rmsep(err_adaptive, post)))
    print("distance ages", round(distance[0], 2), "->", round(distance[-1], 2))
    valid_prep = prep_ch[~np.isnan(prep_ch)]
    valid_ker = kernel_ch[~np.isnan(kernel_ch)]
    print("part magnitudes (median |kPa|): preprocessing",
          round(float(np.median(np.abs(valid_prep))), 2), "| kernel", round(float(np.median(np.abs(valid_ker))), 2))
    print("centre migration ends at", round(center_shift[-1], 2), "training SD; beta change",
          round(beta_shift[-1] * 100, 0), "%")


if __name__ == "__main__":
    default = Path(__file__).resolve().parents[1] / "figures" / "monitoring"
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else default)
