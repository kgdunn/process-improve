# (c) Kevin Dunn, 2010-2026. MIT License.
"""Applicability domain: tell in advance which new materials will be predicted badly.

Each left-out material's Hotelling's T2 and SPE, in a PCA of the *other* materials' properties,
are compared with the prediction error of its blends: for the ordinary materials of the main
study (run ``study.py`` first, which writes ``results/per_material_errors.csv``), and for a
material placed far from all the others.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from simulate import simulate
from study import RESULTS, SCENARIOS, ZERO_SHOT, leave_one_material_out

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


def ordinary_materials() -> pd.DataFrame:
    """T2 and SPE against error, for every left-out material of the main study (8 components)."""
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
                    "material": material,
                    "rmse_over_noise": np.sqrt(mse) / sim.noise,
                    "T2_over_limit": t2 / t2_lim,
                    "SPE_over_limit": spe / spe_lim,
                }
            )
    dom = pd.DataFrame(rows)
    dom.to_csv(RESULTS / "domain_vs_error.csv", index=False)
    return dom.groupby("scenario").apply(
        lambda d: pd.Series(
            {
                "beyond T2 limit": (d["T2_over_limit"] > 1).mean(),
                "beyond SPE limit": (d["SPE_over_limit"] > 1).mean(),
                "Spearman(error, T2)": stats.spearmanr(d["rmse_over_noise"], d["T2_over_limit"])[0],
            }
        ),
        include_groups=False,
    )


def _far_one(rep: int, methods: dict) -> pd.DataFrame:
    sim = simulate(seed=1000 * rep + 7, outlier_shift=4.0)
    data = {"D": sim.D, "F": sim.F, "Z": sim.Z, "Y": sim.Y}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        err = leave_one_material_out(data, "A", 8, methods)
    per = err.groupby(["method", "material"])["sq"].mean().pow(0.5).div(sim.noise).rename("rmse").reset_index()
    dom = {m: material_domain(sim.D["A"], m) for m in sim.D["A"].index}
    per["T2/limit"] = per["material"].map(lambda m: dom[m][0] / dom[m][1])
    per["SPE/limit"] = per["material"].map(lambda m: dom[m][2] / dom[m][3])
    return per


def far_material(n_repeats: int = 6) -> pd.DataFrame:
    """One material moved 4 trait standard deviations away from the rest: median error, T2 and SPE."""
    names = (METHOD, "3a. mixing-rule PLS", "4. hierarchical offsets")
    methods = {k: ZERO_SHOT[k] for k in names}
    out = pd.concat(Parallel(n_jobs=-1)(delayed(_far_one)(r, methods) for r in range(n_repeats)))
    out["material is"] = np.where(out["material"] == "A-out", "far from the others", "one of the other 13")
    return out.groupby(["material is", "method"])[["rmse", "T2/limit", "SPE/limit"]].median()


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    ordinary = ordinary_materials()
    print(f"Ordinary materials ({METHOD}, 8 components): share beyond the 95% limits, and rank correlation")
    print(ordinary.round(2))
    ordinary.round(3).to_csv(RESULTS / "domain_summary.csv")
    far = far_material()
    print("\nA material far from the others: median RMSEP / noise SD, T2 / limit and SPE / limit")
    print(far.round(2))
    far.round(3).to_csv(RESULTS / "far_material.csv")
