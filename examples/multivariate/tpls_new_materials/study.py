# (c) Kevin Dunn, 2010-2026. MIT License.
"""Leave-one-material-out study of how well each method predicts blends that use a new material.

For every material ``j`` of the chosen group, the blends that use ``j`` are the test set, and
the model is trained on all other blends, with ``j`` removed from D and F. Only the properties
of ``j`` are known at prediction time. This is exactly the situation of a new material.

Run ``python study.py`` from this folder (about 40 minutes on 4 cores); it prints the tables
quoted in README.md and writes them to ``results/``.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import methods as mt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from new_materials import centre_f, predict
from simulate import simulate

from process_improve.multivariate.methods import PLS, DataFrameDict

ZERO_SHOT = {
    "ignore the material": mt.ignore_material,
    "nearest known material": mt.nearest_material,
    "zero-padded refit": mt.zero_padded_refit,
    "1. projection, autoscaled F": mt.projection_autoscaled,
    "1. projection, centred F": mt.projection_centred,
    "1. projection, autoscaled F, no D deflation": mt.projection_autoscaled_no_d_deflation,
    "1. projection, centred F, no D deflation": mt.projection_centred_no_d_deflation,
    "2. pseudo-recipe": mt.pseudo_recipe_default,
    "2. pseudo-recipe, no D deflation": mt.pseudo_recipe_no_d_deflation,
    "3a. mixing-rule PLS": mt.mixing_rule_pls,
    "3b. mixing-rule PLS + moments": mt.mixing_moments_pls,
    "4. hierarchical offsets": mt.hierarchical,
}
SCENARIOS = {
    "ideal mixing": {},
    "uneven usage": {"usage_skew": 1.0},
    "hidden trait": {"hidden": 1.0},
    "non-ideal mixing": {"nonideal": 1.0},
}
RESULTS = Path(__file__).parent / "results"


def fit_ceiling(n_components: int = 8, n_repeats: int = 6) -> pd.DataFrame:
    """In-sample R2 of each TPLS variant and of PLS on [Z, F D], with and without Z and group B.

    The truth is reachable: its R2 is in the last column. The D-block deflation is what stops
    TPLS short of it once the super score carries Z, or more than one group of materials.
    """
    configs = {
        "group A only": ({"z_effect": 0.0, "b_effect": 0.0}, ["A"], False),
        "group A + Z": ({"b_effect": 0.0}, ["A"], True),
        "groups A + B": ({"z_effect": 0.0}, ["A", "B"], False),
        "groups A + B + Z": ({}, ["A", "B"], True),
    }
    rows = []
    for label, (kwargs, groups, use_z) in configs.items():
        for rep in range(n_repeats):
            sim = simulate(seed=1000 * rep + 7, **kwargs)
            y = sim.Y["Y"]

            def r2(pred: pd.DataFrame, y: pd.DataFrame = y) -> float:
                return float(1 - ((y - pred.to_numpy()) ** 2).sum().sum() / ((y - y.mean()) ** 2).sum().sum())

            data = {
                "D": {g: sim.D[g] for g in groups},
                "F": {g: sim.F[g] for g in groups},
                "Z": sim.Z if use_z else {},
                "Y": sim.Y,
            }
            row = {"data": label, "truth": r2(sim.y_true)}
            for name, centred, deflate_d in (
                ("TPLS (default)", False, True),
                ("TPLS, centred F", True, True),
                ("TPLS, centred F, no D deflation", True, False),
            ):
                mt.FIT_CACHE.clear()
                model, means = mt.fit_tpls(data, n_components, centred=centred, deflate_d=deflate_d)
                f = centre_f(data["F"], means) if centred else data["F"]
                row[name] = r2(predict(model, DataFrameDict({"F": f, "Z": data["Z"], "Y": {}})))
            stats = {g: (d.mean(), d.std()) for g, d in data["D"].items()}
            d_std = {g: (d - stats[g][0]) / stats[g][1] for g, d in data["D"].items()}
            x = pd.concat([*data["Z"].values(), mt.mixture_descriptors(data["F"], d_std)], axis=1)
            row["PLS on [Z, F D]"] = r2(PLS(n_components=n_components).fit(x, y).predict(x))
            rows.append(row)
    table = pd.DataFrame(rows).groupby("data", sort=False).mean()
    return table[[c for c in table.columns if c != "truth"] + ["truth"]]


def _pivot(values: pd.Series, index: str | list, columns: str, aggfunc: str = "mean") -> pd.DataFrame:
    """Spread a multi-indexed Series into a table (the mean over any index level not named)."""
    long = values.rename("value").reset_index()
    return long.pivot_table(index=index, columns=columns, values="value", aggfunc=aggfunc)


def invariance_checks(n_components: int = 6) -> pd.DataFrame:
    """Run the two tests a model must pass to take new materials; changes are in Y standard deviations.

    1. Split one lot into two identical half-lots (the twin entered as a new material).
    2. Add to D a material that no blend uses, and predict the training blends again.

    Each entry is the largest change of any prediction; a model that passes shows 0.
    """
    sim = simulate(seed=7)
    data = {"D": sim.D, "F": sim.F, "Z": sim.Z, "Y": sim.Y}
    sd = sim.Y["Y"].std().to_numpy()
    lot = sim.F["A"].std().idxmin()  # the lot whose usage varied least, so autoscaling matters most
    twin = sim.D["A"].loc[[lot]].rename(index={lot: "twin"})
    unused = sim.D["A"].iloc[[0]].rename(index={sim.D["A"].index[0]: "unused"}) * 1.05
    split_f = {**sim.F, "A": sim.F["A"].assign(**{lot: sim.F["A"][lot] / 2, "twin": sim.F["A"][lot] / 2})}
    padded_f = {**sim.F, "A": sim.F["A"].assign(unused=0.0)}
    rows = {}
    variants = (
        ("TPLS, autoscaled F", False, True),
        ("TPLS, centred F", True, True),
        ("TPLS, centred F, no D deflation", True, False),
    )
    for name, centred, deflate_d in variants:
        mt.FIT_CACHE.clear()
        model, means = mt.fit_tpls(data, n_components, centred=centred, deflate_d=deflate_d)

        def run(f: dict, d_new: dict | None = None, model=model, means=means, centred=centred) -> np.ndarray:
            f = centre_f(f, means) if centred else f
            return predict(model, DataFrameDict({"F": f, "Z": sim.Z, "Y": {}}), d_new).to_numpy()

        base = run(sim.F)
        rows[name] = {
            "split a lot in two": np.max(np.abs(run(split_f, {"A": twin}) - base) / sd),
            "add an unused material, projected": np.max(np.abs(run(padded_f, {"A": unused}) - base) / sd),
        }
        padded = {**data, "D": {**sim.D, "A": pd.concat([sim.D["A"], unused])}, "F": padded_f}
        mt.FIT_CACHE.clear()
        refit, refit_means = mt.fit_tpls(padded, n_components, centred=centred, deflate_d=deflate_d)
        f = centre_f(padded_f, refit_means) if centred else padded_f
        new = predict(refit, DataFrameDict({"F": f, "Z": sim.Z, "Y": {}})).to_numpy()
        rows[name]["add an unused material, refit"] = np.max(np.abs(new - base) / sd)
    return pd.DataFrame(rows).T


def split(data: dict, material: str, group: str) -> tuple[dict, dict, dict, dict, pd.Index]:
    """Training blocks without ``material``; the test blends that use it; its properties."""
    uses = data["F"][group][material] > 0
    train = {
        "D": {g: d.drop(index=material) if g == group else d for g, d in data["D"].items()},
        "F": {g: (f.drop(columns=material) if g == group else f).loc[~uses] for g, f in data["F"].items()},
        "Z": {k: z.loc[~uses] for k, z in data["Z"].items()},
        "Y": {k: y.loc[~uses] for k, y in data["Y"].items()},
    }
    test_f = {g: f.loc[uses] for g, f in data["F"].items()}
    test_z = {k: z.loc[uses] for k, z in data["Z"].items()}
    d_new = {group: data["D"][group].loc[[material]]}
    return train, test_f, test_z, d_new, uses[uses].index


def leave_one_material_out(data: dict, group: str, n_components: int, which: dict = ZERO_SHOT) -> pd.DataFrame:
    """Squared errors of every method for every (left-out material, blend that uses it) pair.

    Returns a long frame: method, material, blend, Y column and squared error.
    """
    key = next(iter(data["Y"]))
    errors = []
    for material in data["F"][group].columns:
        mt.FIT_CACHE.clear()
        train, test_f, test_z, d_new, rows = split(data, material, group)
        if len(rows) == 0 or len(rows) == len(data["F"][group]):
            continue
        for name, method in which.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pred = method(train, test_f, test_z, d_new, n_components)
            sq = (pred - data["Y"][key].loc[rows].to_numpy()) ** 2
            errors.append(
                sq.rename_axis("blend")
                .reset_index()
                .melt("blend", var_name="y", value_name="sq")
                .assign(method=name, material=material)
            )
    return pd.concat(errors, ignore_index=True)


def leave_one_blend_out(data: dict, n_components: int) -> pd.DataFrame:
    """Predict each blend with approach 1 from all other blends, so every material has been seen (reference)."""
    names = next(iter(data["F"].values())).index
    out = []
    for blend in names:
        mt.FIT_CACHE.clear()
        keep = names != blend
        train = {"D": data["D"], **{b: {k: v.loc[keep] for k, v in data[b].items()} for b in "FZY"}}
        test_f = {g: f.loc[[blend]] for g, f in data["F"].items()}
        test_z = {k: z.loc[[blend]] for k, z in data["Z"].items()}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out.append(mt.projection_centred_no_d_deflation(train, test_f, test_z, {}, n_components))
    return pd.concat(out)


REFERENCE = "reference: material seen in training"


def _one_dataset(scenario: str, rep: int, components: tuple) -> pd.DataFrame:
    sim = simulate(seed=1000 * rep + 7, **SCENARIOS[scenario])
    data = {"D": sim.D, "F": sim.F, "Z": sim.Z, "Y": sim.Y}
    errs = []
    for n_comp in components:
        err = leave_one_material_out(data, "A", n_comp)
        # The reference, scored on the same (material, blend) pairs as the methods.
        sq = ((leave_one_blend_out(data, n_comp) - sim.Y["Y"]) ** 2).rename_axis("blend").reset_index()
        pairs = err.loc[err["method"] == next(iter(ZERO_SHOT)), ["blend", "y", "material"]]
        ref = pairs.merge(sq.melt("blend", var_name="y", value_name="sq"), on=["blend", "y"]).assign(method=REFERENCE)
        err = pd.concat([err, ref], ignore_index=True)
        errs.append(err.assign(A=n_comp, scenario=scenario, rep=rep, noise=sim.noise))
    return pd.concat(errs)


def simulation_zero_shot(n_repeats: int = 6, components: tuple = (4, 6, 8)) -> tuple[pd.DataFrame, ...]:
    """RMSEP / noise SD (1.0 is the best possible) per scenario and method, at each method's best A."""
    jobs = [(s, r) for s in SCENARIOS for r in range(n_repeats)]
    errors = pd.concat(Parallel(n_jobs=-1)(delayed(_one_dataset)(s, r, components) for s, r in jobs), ignore_index=True)
    # RMSEP per data set (pooled over materials, blends and Y columns), divided by the noise SD.
    per_rep = errors.groupby(["scenario", "rep", "A", "method"]).apply(
        lambda d: np.sqrt(d["sq"].mean()) / d["noise"].iloc[0], include_groups=False
    )
    mean_a = _pivot(per_rep, ["scenario", "method"], "A")
    order = [*ZERO_SHOT, REFERENCE]
    best = _pivot(mean_a.min(axis=1), "method", "scenario").loc[order, list(SCENARIOS)]
    best_a = _pivot(mean_a.idxmin(axis=1), "method", "scenario", "first").loc[order, list(SCENARIOS)]
    return best, best_a, errors


def few_shot(data: dict, group: str, n_components: int, shots: tuple = (1, 2, 3), n_draws: int = 3) -> pd.DataFrame:
    """Prediction after measuring ``n`` blends that use the new material (approach 5 and refits)."""
    key = next(iter(data["Y"]))
    rng = np.random.default_rng(0)
    rows = []
    for material in data["F"][group].columns:
        mt.FIT_CACHE.clear()
        train, test_f, test_z, d_new, blends = split(data, material, group)
        if len(blends) < max(shots) + 2:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            zero, sigma2, tau2 = mt.hierarchical(train, test_f, test_z, d_new, n_components, return_parts=True)
        y = data["Y"][key].loc[blends]
        for n in shots:
            for _ in range(n_draws):
                shot = pd.Index(rng.choice(blends, n, replace=False))
                rest = blends.difference(shot)
                preds = {"4. zero-shot (no measurement)": zero.loc[rest]}
                upd = mt.few_shot_update(zero, test_f[group][material], y.loc[shot], tau2, sigma2)
                preds["5. update the offset (BLUP)"] = upd.loc[rest]
                aug = {
                    "D": {g: pd.concat([d, d_new[g]]) if g in d_new else d for g, d in train["D"].items()},
                    "F": {
                        g: pd.concat([f.reindex(columns=test_f[g].columns, fill_value=0.0), test_f[g].loc[shot]])
                        for g, f in train["F"].items()
                    },
                    "Z": {k: pd.concat([z, test_z[k].loc[shot]]) for k, z in train["Z"].items()},
                    "Y": {key: pd.concat([train["Y"][key], y.loc[shot]])},
                }
                rest_f = {g: f.loc[rest] for g, f in test_f.items()}
                rest_z = {k: z.loc[rest] for k, z in test_z.items()}
                mt.FIT_CACHE.clear()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model, _ = mt.fit_tpls(aug, n_components, centred=False)
                    preds["refit default TPLS with the shots"] = mt.predict(
                        model, DataFrameDict({"F": rest_f, "Z": rest_z, "Y": {}})
                    )
                    preds["refit 4. with the shots"] = mt.hierarchical(aug, rest_f, rest_z, {}, n_components)
                for name, p in preds.items():
                    rows.append(
                        {
                            "material": material,
                            "shots": n,
                            "method": name,
                            "sse": float(((p - y.loc[rest].to_numpy()) ** 2).to_numpy().sum()),
                            "count": p.size,
                        }
                    )
    return pd.DataFrame(rows)


def _few_shot_dataset(rep: int, n_components: int) -> pd.DataFrame:
    sim = simulate(seed=1000 * rep + 7, hidden=1.0)
    data = {"D": sim.D, "F": sim.F, "Z": sim.Z, "Y": sim.Y}
    return few_shot(data, "A", n_components).assign(rep=rep, noise=sim.noise)


def simulation_few_shot(n_repeats: int = 6, n_components: int = 8) -> pd.DataFrame:
    """RMSEP / noise SD on the blends not used as shots, in the hidden-trait scenario."""
    out = pd.concat(Parallel(n_jobs=-1)(delayed(_few_shot_dataset)(r, n_components) for r in range(n_repeats)))
    per_rep = out.groupby(["rep", "shots", "method"]).apply(
        lambda d: np.sqrt(d["sse"].sum() / d["count"].sum()) / d["noise"].iloc[0], include_groups=False
    )
    return _pivot(per_rep, "method", "shots")


def load_pyphi() -> dict:
    """Load the T-shaped example data set bundled with process-improve (from pyphi, Garcia-Munoz)."""
    base = Path(__file__).parents[3] / "src/process_improve/datasets/multivariate/tpls-pyphi"
    groups = [f"Group {g}" for g in range(1, 6)]
    return {
        "D": {g: pd.read_csv(base / f"properties_{g.replace(' ', '')}.csv", index_col=0) for g in groups},
        "F": {g: pd.read_csv(base / f"formulas_{g.replace(' ', '')}.csv", index_col=0) for g in groups},
        "Z": {"Conditions": pd.read_csv(base / "process_conditions.csv", index_col=0).astype(float)},
        "Y": {"Quality": pd.read_csv(base / "quality_indicators.csv", index_col=0)},
    }


def _real_one(group: str, n_comp: int) -> pd.DataFrame:
    data = load_pyphi()
    err = leave_one_material_out(data, group, n_comp)
    return err.assign(group=group, A=n_comp)


def real_data(components: tuple = (1, 2, 3, 4)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-lot-out on the pyphi data: Q2 = 1 - PRESS / SS, Y columns standardised, per group."""
    data = load_pyphi()
    y = data["Y"]["Quality"]
    jobs = [(g, a) for g in data["F"] for a in components]
    errors = pd.concat(Parallel(n_jobs=-1)(delayed(_real_one)(g, a) for g, a in jobs), ignore_index=True)
    errors["sq_std"] = errors["sq"] / errors["y"].map(y.var())
    # One A per method, the one with the best Q2 over all lots; then Q2 per group at that A.
    pooled = _pivot(1 - errors.groupby(["A", "method"])["sq_std"].mean(), "method", "A")
    best_a = pooled.idxmax(axis=1)
    q2 = (1 - errors.groupby(["group", "A", "method"])["sq_std"].mean()).rename("value").reset_index()
    q2 = q2[q2["A"] == q2["method"].map(best_a)]
    table = q2.pivot_table(index="method", columns="group", values="value").loc[list(ZERO_SHOT)]
    table["all lots"] = pooled.max(axis=1).loc[list(ZERO_SHOT)]
    return table, best_a.loc[list(ZERO_SHOT)]


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    RESULTS.mkdir(exist_ok=True)
    t0 = time.time()

    checks = invariance_checks()
    print("\nLargest change of any prediction, in Y standard deviations (0 = the test is passed)")
    print(checks.round(4))
    checks.round(4).to_csv(RESULTS / "invariance_checks.csv")

    ceiling = fit_ceiling()
    print("\nIn-sample R2 with 8 components, mean of 6 simulated data sets")
    print(ceiling.round(3))
    ceiling.round(3).to_csv(RESULTS / "fit_ceiling.csv")

    best, best_a, errors = simulation_zero_shot()
    print("\nSimulation, zero-shot: RMSEP / noise SD (1.0 = the noise floor), best A in {4, 6, 8}")
    print(best.round(2))
    print(best_a)
    best.round(3).to_csv(RESULTS / "simulation_zero_shot.csv")
    per_mat = errors.groupby(["scenario", "rep", "A", "method", "material"])["sq"].mean().rename("mse").reset_index()
    per_mat.to_csv(RESULTS / "per_material_errors.csv", index=False)
    print(f"[{time.time() - t0:.0f} s]")

    fs = simulation_few_shot()
    print("\nSimulation, hidden-trait scenario, after n measured blends: RMSEP / noise SD")
    print(fs.round(2))
    fs.round(3).to_csv(RESULTS / "simulation_few_shot.csv")
    print(f"[{time.time() - t0:.0f} s]")

    real, real_a = real_data()
    print("\npyphi data, leave-one-lot-out: Q2 (1 is perfect, 0 is predicting the mean), A in 1..4 best overall")
    print(real.round(3))
    print(real_a)
    real.round(4).to_csv(RESULTS / "pyphi_leave_one_lot_out.csv")
    print(f"[{time.time() - t0:.0f} s]")
