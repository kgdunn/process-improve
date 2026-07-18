"""Generate the committed PNGs for the adaptive soft-sensor case study.

Mirrors the analysis in the pid-book chapter
``product-development-product-improvement/multivariate-process-monitoring.rst``
(the "Keeping a model current" section): a static PLS soft sensor drifts, an
adaptive PLS tracks it, monitoring statistics flag the drift, and the distance
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
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 140})


def bias_std_rmsep(err: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    e = err[mask]
    return float(e.mean()), float(e.std()), float(np.sqrt((e**2).mean()))


def main(out_dir: Path) -> None:  # noqa: PLR0915
    out_dir.mkdir(parents=True, exist_ok=True)
    vp = pd.read_csv(DATA_URL)
    vp["month"] = vp["hours_elapsed"] / 730.5
    tags = [c for c in vp.columns if c not in ("hours_elapsed", "month", "vapour_pressure_kpa", "current_estimator")]
    lab_rows = np.where(vp["vapour_pressure_kpa"].notna().to_numpy())[0]
    lab = vp.iloc[lab_rows].reset_index(drop=True)
    y_lab = lab["vapour_pressure_kpa"].to_numpy()
    n_seed = len(lab) // 2
    seed = lab.index < n_seed
    drift_month = float(np.quantile(lab["month"], 0.60))
    post = lab["month"].to_numpy() >= drift_month

    Xrow = vp[tags].to_numpy()

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
        n_components=3, forgetting_factor=0, gamma=0, lambda_center=0, alpha_scale=0,
        lambda_center_y=0, alpha_scale_y=0, adaptive_spe_limit=False, conf_level=0.99,
    ).fit(lab.loc[seed, tags], lab.loc[seed, ["vapour_pressure_kpa"]])
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
        n_components=3, forgetting_factor=0.003, gamma=0.1, lambda_center=0.012, alpha_scale=0.012,
        lambda_center_y=0.12, alpha_scale_y=0.05, update_when_out_of_control=True, conf_level=0.99,
    ).fit(lab.loc[seed, tags], lab.loc[seed, ["vapour_pressure_kpa"]])
    adaptive_pred, _, _, distance = stream(
        adaptive, learn=learn, y_update=ewma_smooth(vp["vapour_pressure_kpa"].to_numpy())
    )

    month = vp["month"].to_numpy()
    err_static = static_pred[lab_rows] - y_lab
    err_adaptive = adaptive_pred[lab_rows] - y_lab

    # Fig 1: motivation
    fig, ax = plt.subplots(figsize=(8.2, 3.3))
    ax.plot(lab["month"], y_lab, ".", ms=4, color="0.35", label="Lab reference")
    ax.plot(month, static_pred, "-", lw=0.9, color="tab:red", label="Static PLS prediction")
    ax.axvline(drift_month, color="k", ls="--", lw=1)
    ax.text(drift_month + 0.3, 92, "drift established", fontsize=9)
    ax.set_ylim(15, 100)
    ax.set_xlabel("Months since start of record")
    ax.set_ylabel("Vapour pressure (kPa)")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_dir / "adaptive-softsensor-motivation.png")
    plt.close(fig)

    # Fig 2: monitoring
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.2, 4.2), sharex=True)
    a1.plot(month, static_t2, lw=0.5, color="tab:blue")
    a1.axhline(t2_lim, color="k", lw=1)
    a1.set_ylim(0, 3 * t2_lim)
    a1.set_ylabel("Hotelling's $T^2$")
    a1.set_title("99% limits", fontsize=10)
    a2.plot(month, static_spe, lw=0.5, color="tab:red")
    a2.axhline(spe_lim, color="k", lw=1)
    a2.set_ylim(0, 3 * spe_lim)
    a2.set_ylabel("SPE")
    a2.set_xlabel("Months since start of record")
    for a in (a1, a2):
        a.axvline(drift_month, color="k", ls="--", lw=1)
    fig.tight_layout()
    fig.savefig(out_dir / "adaptive-softsensor-monitoring.png")
    plt.close(fig)

    # Fig 3: payoff
    fig, ax = plt.subplots(figsize=(8.2, 3.3))
    ax.axhspan(-3, 3, color="0.85", alpha=0.6)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.plot(lab["month"], err_static, "o", ms=4.5, color="tab:red", alpha=0.6, label="Static PLS")
    ax.plot(lab["month"], err_adaptive, "o", ms=4.5, color="tab:blue", alpha=0.6, label="Adaptive PLS")
    ax.axvline(drift_month, color="k", ls="--", lw=1)
    ax.set_ylim(-20, 22)
    ax.set_xlabel("Months since start of record")
    ax.set_ylabel("Prediction error (kPa)")
    ax.legend(loc="lower left", fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "adaptive-softsensor-payoff.png")
    plt.close(fig)

    # Fig 4: diagnostics (distance metric)
    fig, ax = plt.subplots(figsize=(8.2, 3.0))
    ax.plot(month, distance, lw=0.8, color="tab:purple")
    ax.axhline(3, color="0.6", ls=":", lw=1)
    ax.axvline(drift_month, color="k", ls="--", lw=1)
    ax.set_ylim(0, 3.2)
    ax.set_xlabel("Months since start of record")
    ax.set_ylabel("Subspace overlap with seed model")
    fig.tight_layout()
    fig.savefig(out_dir / "adaptive-softsensor-diagnostics.png")
    plt.close(fig)

    print(f"wrote 4 figures to {out_dir}")
    print("static post-drift  bias/std/RMSEP:", tuple(round(v, 1) for v in bias_std_rmsep(err_static, post)))
    print("adaptive post-drift bias/std/RMSEP:", tuple(round(v, 1) for v in bias_std_rmsep(err_adaptive, post)))
    print("distance ages", round(distance[0], 2), "->", round(distance[-1], 2))


if __name__ == "__main__":
    default = Path(__file__).resolve().parents[1] / "figures" / "monitoring"
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else default)
