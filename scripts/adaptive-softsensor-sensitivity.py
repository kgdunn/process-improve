"""Parameter-sensitivity figures for the adaptive soft-sensor case study.

Two committed figures for the pid-book "Choosing the adaptation settings"
subsection. The logic mirrors the chapter's Plotly code exactly (same public
API, same observation-selection mask and smoothed reference), so a reader who
runs the chapter reproduces these figures.

- ``adaptive-softsensor-sensitivity.png``: one-at-a-time sweeps of the four
  tuning parameters, scored by the leakage-free prequential (one-step-ahead)
  RMSEP on an inner validation window. The model is fitted on an early
  tune-train block; the inner window is later, and the far testing data is never
  touched. Validates which parameters matter (components and the centring rate)
  and which do not (gamma).
- ``adaptive-softsensor-distance-roughness.png``: the distance metric of the
  deployed model streamed at the chosen forgetting factor (smooth) and at a
  too-large one (jagged), validating that a jagged trace means the factor is too
  high.

Usage::

    python scripts/adaptive-softsensor-sensitivity.py [output_dir]
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
A = 3
DARK_BLUE, ORANGE = "#1f3d7a", "#c55a11"
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 140})
CFG = dict(n_components=A, forgetting_factor=0.01, gamma=0.05, lambda_center=0.003,
           alpha_scale=0.012, lambda_center_y=0.12, alpha_scale_y=0.05)


def main(out_dir: Path) -> None:  # noqa: PLR0915, C901
    out_dir.mkdir(parents=True, exist_ok=True)
    vp = pd.read_csv(DATA_URL)
    vp["month"] = vp["hours_elapsed"] / 730.5
    tags = [c for c in vp.columns if c not in ("hours_elapsed", "month", "vapour_pressure_kpa", "current_estimator")]
    lab_rows = np.where(vp["vapour_pressure_kpa"].notna().to_numpy())[0]
    lab = vp.iloc[lab_rows].reset_index(drop=True)
    y_lab = lab["vapour_pressure_kpa"].to_numpy()
    n_train = len(lab) // 2
    train = lab.index < n_train
    Xrow = vp[tags].to_numpy()

    def ewma(v, lam=0.35) -> np.ndarray:
        o = v.astype(float).copy()
        idx = np.where(~np.isnan(o))[0]
        for k in range(1, len(idx)):
            o[idx[k]] = lam * o[idx[k]] + (1 - lam) * o[idx[k - 1]]
        return o

    y_upd = ewma(vp["vapour_pressure_kpa"].to_numpy())

    def stream(model, learn) -> tuple[np.ndarray, np.ndarray]:
        pred = np.zeros(len(vp))
        dist = np.zeros(len(vp))
        for i in range(len(vp)):
            if learn is None or bool(learn[i]):
                yv = None if np.isnan(y_upd[i]) else np.array([y_upd[i]])
                out = model.update(Xrow[i], y_row=yv)
                pred[i], dist[i] = out.prediction[0], out.distance
            else:
                pred[i] = model.predict(vp[tags].iloc[[i]]).to_numpy().ravel()[0]
                dist[i] = dist[i - 1] if i else model.n_components
        return pred, dist

    # observation-selection mask from the deployed static model (as in the chapter)
    static = AdaptivePLS(n_components=A, forgetting_factor=0, gamma=0, lambda_center=0, alpha_scale=0,
                         lambda_center_y=0, alpha_scale_y=0, adaptive_spe_limit=False, conf_level=0.99)
    static.fit(lab.loc[train, tags], lab.loc[train, ["vapour_pressure_kpa"]])
    static_spe = np.array([static.update(Xrow[i]).spe for i in range(len(vp))])
    spe_lim = float(static.update(Xrow[0]).spe_limit)
    learn = static_spe < 2.5 * spe_lim

    # ---- sensitivity: tune on an early block, score a later inner window ----
    n_tune = int(0.40 * len(lab))
    inner = np.arange(n_tune + 8, int(0.75 * len(lab)))

    def prequential(**changes) -> float:
        cfg = dict(CFG, **changes)
        m = AdaptivePLS(update_when_out_of_control=True, conf_level=0.99, **cfg)
        m.fit(lab.iloc[:n_tune][tags], lab.iloc[:n_tune][["vapour_pressure_kpa"]])
        pred, _ = stream(m, learn)
        err = pred[lab_rows] - y_lab
        return float(np.sqrt((err[inner] ** 2).mean()))

    sweeps = {
        "n_components": ([2, 3, 4, 5], False),
        "forgetting_factor": ([0.003, 0.01, 0.03, 0.1], True),
        "lambda_center": ([0.001, 0.003, 0.01, 0.03], True),
        "gamma": ([0.0, 0.05, 0.1, 0.2], False),
    }
    results = {name: [(v, prequential(**{name: v})) for v in vals] for name, (vals, _) in sweeps.items()}
    for name, rows in results.items():
        print(name, [(v, round(r, 2)) for v, r in rows], flush=True)

    def elasticity(vals, metric, base) -> float:
        i = int(np.argmin(np.abs(np.array(vals) - base)))
        if i in (0, len(vals) - 1) or np.any(np.array(vals[i - 1:i + 2]) <= 0):
            return np.nan
        v, m = np.array(vals, float), np.array(metric, float)
        return float((np.log(m[i + 1]) - np.log(m[i - 1])) / (np.log(v[i + 1]) - np.log(v[i - 1])))

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.1), sharey=True)
    for ax, (name, (vals, is_log)) in zip(axes, sweeps.items(), strict=True):
        arr = np.array(results[name])
        ax.plot(arr[:, 0], arr[:, 1], "o-", color=DARK_BLUE, lw=1.5)
        ax.axvline(CFG[name], color=ORANGE, ls=":", lw=1.4)
        if is_log:
            ax.set_xscale("log")
        e = elasticity(vals, [r for _, r in results[name]], CFG[name])
        ax.set_title(f"{name}\nelasticity = {e:+.2f}" if np.isfinite(e) else f"{name}\n(flat: no effect)", fontsize=9)
        ax.set_xlabel(name, fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Prequential RMSEP [kPa]")
    fig.suptitle("Parameter sensitivity: prequential one-step-ahead RMSEP on the inner validation window "
                 "(orange = chosen value)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "adaptive-softsensor-sensitivity.png")
    plt.close(fig)

    # ---- distance roughness of the deployed model: chosen vs too-large mu ----
    def deployed_distance(mu) -> np.ndarray:
        m = AdaptivePLS(update_when_out_of_control=True, conf_level=0.99, **dict(CFG, forgetting_factor=mu))
        m.fit(lab.loc[train, tags], lab.loc[train, ["vapour_pressure_kpa"]])
        return stream(m, learn)[1]

    month = vp["month"].to_numpy()
    d_ok = deployed_distance(0.01)
    d_big = deployed_distance(0.10)
    r_ok, r_big = float(np.std(np.diff(d_ok))), float(np.std(np.diff(d_big)))
    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    ax.plot(month, d_big, lw=0.7, color=ORANGE, label=f"forgetting_factor = 0.10 (jagged, roughness {r_big:.4f})")
    ax.plot(month, d_ok, lw=0.9, color=DARK_BLUE, label=f"forgetting_factor = 0.01 (chosen, roughness {r_ok:.4f})")
    ax.set_xlabel("Time since start [months]")
    ax.set_ylabel("Subspace overlap [components]")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_dir / "adaptive-softsensor-distance-roughness.png")
    plt.close(fig)
    print(f"roughness: chosen(0.01)={r_ok:.4f}  too-large(0.10)={r_big:.4f}", flush=True)
    print("wrote 2 sensitivity figures to", out_dir, flush=True)


if __name__ == "__main__":
    default = Path(__file__).resolve().parents[1] / "figures" / "monitoring"
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else default)
