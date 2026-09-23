"""Check every compare_cv_criteria rule on data whose latent structure is known.

Each scenario is a :class:`~process_improve.simulation.LatentStructure`. For every
simulated data set the script records what each selection rule recommends, and checks it
against two kinds of truth:

* structural: the number of latent variables that drive Y (``n_relevant``);
* oracle: 4000 fresh rows from the same process, which give the component count that
  predicts new rows best and the SPE alarm rate of each limit on new rows.

A second study places a weak but real second component in the band ``0 < s_a < 1/2``
and counts how often the held-out score correlation beats its permutation null there.
These are the numbers quoted in the user guide (``cross_validation.rst``).

Usage
-----
    python scripts/cv_criteria_recovery.py [n_datasets] [n_band_datasets]

The defaults, 30 data sets per scenario and 40 per setting of the band study, give the
quoted numbers; the full run takes about 20 minutes on four cores.
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

from process_improve.multivariate import PLS, compare_cv_criteria
from process_improve.simulation import LatentStructure

A_MAX = 5
N_FRESH = 4000
RULES = ["q2_max", "q2_1se", "van_der_voet", "score_correlation", "covariance_permutation", "subspace_stability"]
TWO_LATENT = LatentStructure(x_sd=[3.0, 1.0], y_coefficients=[1.0, 1.0])

#: name -> (process, training rows, fraction of cells missing)
SCENARIOS: dict[str, tuple[LatentStructure, int, float]] = {
    "S1: 2 LVs drive y": (TWO_LATENT, 60, 0.0),
    "S2: y is noise": (LatentStructure(x_sd=[3.0, 1.0], y_coefficients=[0.0, 0.0]), 60, 0.0),
    "S3: 1 LV drives y, stronger LV Y-orthogonal": (
        LatentStructure(x_sd=[2.0, 3.0, 1.5], y_coefficients=[1.0, 0.0, 0.0]),
        60,
        0.0,
    ),
    "S5: 2 equal LVs, M = 2 (swap)": (LatentStructure(x_sd=[1.0, 1.0], y_coefficients=np.eye(2)), 60, 0.0),
    "S6: S1, 15% cells missing": (TWO_LATENT, 60, 0.15),
    "S6: S1, 30% cells missing": (TWO_LATENT, 60, 0.30),
    "S7: S1 with N = 30": (TWO_LATENT, 30, 0.0),
}


def one_data_set(process: LatentStructure, n: int, missing: float, seed: int) -> dict:
    """Recommendations for one simulated data set, with the fresh-row truth beside them."""
    train = process.sample(n, missing_fraction=missing, random_state=seed)
    fresh = process.sample(N_FRESH, random_state=100_000 + seed)
    result = compare_cv_criteria(
        train.X, train.Y, max_components=A_MAX, random_state=seed, n_permutations=499, n_cv_permutations=199
    )
    models = [PLS(n_components=a).fit(train.X, train.Y) for a in range(1, A_MAX + 1)]
    mse = np.array([np.mean((m.predict(fresh.X).to_numpy() - fresh.Y.to_numpy()) ** 2) for m in models])
    a_ref = max(process.n_relevant, 1)
    spe_fresh = models[a_ref - 1].diagnose(fresh.X).spe.to_numpy()
    limits = result.spe_limits.loc[a_ref]
    w1 = models[0].x_weights_.to_numpy()[:, 0]
    return {
        **result.recommendations["n_components"].to_dict(),
        "oracle": int(np.argmin(mse)) + 1,
        "two_beats_one": bool(mse[1] < mse[0]),
        "w1_tilt_deg": float(np.degrees(np.arccos(min(1.0, abs(w1 @ process.loadings.iloc[:, 0].to_numpy()))))),
        "alarm_full": float(np.mean(spe_fresh > limits["full_data"])),
        "alarm_refitted": float(np.mean(spe_fresh > limits["pseudo_validation"])),
    }


def recovery_table(n_datasets: int) -> str:
    """Markdown table: the fraction of data sets in which each rule recovers the structural truth."""
    header = ["Scenario", "Truth", "Oracle A (mode)", *(f"`{r}`" for r in RULES)]
    header += ["SPE alarms on fresh rows: model limit / refitted limit", "2 beat 1 on fresh rows", "median w1 tilt"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for name, (process, n, missing) in SCENARIOS.items():
        runs = pd.DataFrame([one_data_set(process, n, missing, seed) for seed in range(n_datasets)])
        oracle = runs["oracle"].mode().iloc[0]
        cells = " | ".join(f"{np.mean(runs[r] == process.n_relevant):.0%}" for r in RULES)
        lines.append(
            f"| {name} | {process.n_relevant} | {oracle} ({np.mean(runs['oracle'] == oracle):.0%}) | {cells} | "
            f"{runs['alarm_full'].mean():.3f} / {runs['alarm_refitted'].mean():.3f} | "
            f"{int(runs['two_beats_one'].sum())} of {n_datasets} | {runs['w1_tilt_deg'].median():.1f} deg |"
        )
        print(lines[-1], flush=True)
    return "\n".join(lines)


def slope_band_study(n_datasets: int) -> str:
    """How often a component with 0 < s_2 < 1/2 has a held-out correlation beating its null."""
    rows = []
    for n, k, b2, y_noise in [(50, 16, 0.25, 0.5), (50, 16, 0.4, 1.0), (50, 32, 0.25, 0.5), (200, 64, 0.25, 1.0)]:
        process = LatentStructure(
            x_sd=[2.0, 1.0], y_coefficients=[1.0, b2], n_features=k, noise_sd=0.5, y_noise_sd=y_noise
        )
        for seed in range(n_datasets):
            train = process.sample(n, random_state=seed)
            table = compare_cv_criteria(
                train.X, train.Y, max_components=2, random_state=seed, n_permutations=19, n_cv_permutations=199
            ).table
            rows.append({"s": table.loc[2, "slope_ratio"], "p": table.loc[2, "r_cv_p"]})
    runs = pd.DataFrame(rows)
    band = runs[(runs["s"] > 0) & (runs["s"] < 0.5)]
    above = runs[runs["s"] >= 0.5]
    return (
        f"0 < s_2 < 1/2: {int((band['p'] < 0.05).sum())} of {len(band)} components reach p < 0.05; "
        f"s_2 >= 1/2: {int((above['p'] < 0.05).sum())} of {len(above)}."
    )


if __name__ == "__main__":
    n_datasets = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    n_band = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    warnings.simplefilter("ignore")  # NIPALS convergence notes on noise components
    print(recovery_table(n_datasets))
    print()
    print(slope_band_study(n_band))
