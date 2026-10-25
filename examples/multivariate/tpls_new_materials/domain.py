# (c) Kevin Dunn, 2010-2026. MIT License.
"""Applicability domain: tell in advance which new materials will be predicted badly.

For every left-out material of the simulation study, its Hotelling's T2 and SPE in a PCA of
the *other* materials' properties are compared with the prediction error of its blends. Run
after ``study.py``, which writes ``results/per_material_errors.csv``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from simulate import simulate
from study import RESULTS, SCENARIOS

METHOD = "1. projection, centred F, no D deflation"


def material_domain(d: pd.DataFrame, material: str, n_pc: int = 3) -> tuple[float, float, float, float]:
    """T2 and SPE of ``material`` in a PCA of the other rows of ``d``, with their 95% limits."""
    train = d.drop(index=material)
    mu, sd = train.mean(), train.std()
    x = ((train - mu) / sd).to_numpy()
    new = ((d.loc[[material]] - mu) / sd).to_numpy()
    _, sv, vt = np.linalg.svd(x, full_matrices=False)
    p = vt[:n_pc].T
    var = sv[:n_pc] ** 2 / (len(x) - 1)
    t_new = new @ p
    t2 = float(np.sum(t_new**2 / var))
    spe = float(np.sum((new - t_new @ p.T) ** 2))
    n = len(x)
    t2_lim = n_pc * (n - 1) * (n + 1) / (n * (n - n_pc)) * stats.f.ppf(0.95, n_pc, n - n_pc)
    spe_train = np.sum((x - x @ p @ p.T) ** 2, axis=1)
    g, h = spe_train.var() / (2 * spe_train.mean()), 2 * spe_train.mean() ** 2 / spe_train.var()
    return t2, t2_lim, spe, g * stats.chi2.ppf(0.95, h)


if __name__ == "__main__":
    errors = pd.read_csv(RESULTS / "per_material_errors.csv")
    errors = errors[(errors["method"] == METHOD) & (errors["A"] == errors["A"].max())]
    rows = []
    for (scenario, rep), grp in errors.groupby(["scenario", "rep"]):
        sim = simulate(seed=1000 * rep + 7, **SCENARIOS[scenario])
        for material, mse in zip(grp["material"], grp["mse"], strict=True):
            t2, t2_lim, spe, spe_lim = material_domain(sim.D["A"], material)
            rows.append(
                {
                    "scenario": scenario,
                    "rep": rep,
                    "material": material,
                    "rmse_over_noise": np.sqrt(mse) / sim.noise,
                    "T2_over_limit": t2 / t2_lim,
                    "SPE_over_limit": spe / spe_lim,
                }
            )
    dom = pd.DataFrame(rows)
    dom.to_csv(RESULTS / "domain_vs_error.csv", index=False)
    dom["outside"] = (dom["T2_over_limit"] > 1) | (dom["SPE_over_limit"] > 1)
    summary = dom.groupby("scenario").apply(
        lambda d: pd.Series(
            {
                "Spearman(error, T2)": stats.spearmanr(d["rmse_over_noise"], d["T2_over_limit"])[0],
                "Spearman(error, SPE)": stats.spearmanr(d["rmse_over_noise"], d["SPE_over_limit"])[0],
                "materials outside 95%": int(d["outside"].sum()),
                "error inside": d.loc[~d["outside"], "rmse_over_noise"].median(),
                "error outside": d.loc[d["outside"], "rmse_over_noise"].median(),
            }
        ),
        include_groups=False,
    )
    print(f"Per-material RMSEP / noise SD of '{METHOD}', against T2 and SPE of the material")
    print(summary.round(2))
    summary.round(3).to_csv(RESULTS / "domain_summary.csv")
