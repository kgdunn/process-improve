# (c) Kevin Dunn, 2010-2026. MIT License.
"""The candidate ways to predict blends that use a new material, behind one calling convention.

Every zero-shot method is ``method(train, test_f, test_z, d_new, n_components) -> predicted Y``:

* ``train`` holds the training blocks ``D, F, Z, Y`` (dicts of DataFrames, as for TPLS), with
  the new material absent from both D and F;
* ``test_f`` / ``test_z`` are the blends to predict, whose F block has the new material's column;
* ``d_new`` holds the new material's properties, per group.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from new_materials import TPLSWithoutDDeflation, centre_f, predict, pseudo_recipe, substitute

from process_improve.multivariate.methods import PLS, TPLS, DataFrameDict

#: Fitted models of the current fold, shared by the methods that use the same fit. Cleared per fold.
FIT_CACHE: dict = {}


def fit_tpls(
    train: dict, n_components: int, centred: bool, deflate_d: bool = True
) -> tuple[TPLS, dict[str, pd.Series] | None]:
    """Fit TPLS with autoscaled F (the default) or centred F, with or without the D-block deflation."""
    key = (id(train), n_components, centred, deflate_d)
    if key in FIT_CACHE:
        return FIT_CACHE[key]
    cls = TPLS if deflate_d else TPLSWithoutDDeflation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # zero-variance columns are expected: unused materials
        if not centred:
            fitted = cls(n_components, train["D"]).fit(DataFrameDict({k: train[k] for k in "FZY"})), None
        else:
            means = {g: f.mean() for g, f in train["F"].items()}
            blocks = DataFrameDict({"F": centre_f(train["F"], means), "Z": train["Z"], "Y": train["Y"]})
            fitted = cls(n_components, train["D"], skip_f_matrix_preprocessing=True).fit(blocks), means
    FIT_CACHE[key] = fitted
    return fitted


def _diagnose(model: TPLS, f: dict, z: dict) -> pd.DataFrame:
    return predict(model, DataFrameDict({"F": f, "Z": z, "Y": {}}))


# ---- Baselines: what one tries first --------------------------------------------------------------


def ignore_material(train, test_f, test_z, d_new, n_components):
    """Drop the new material and renormalise the remaining fractions of its group."""
    model, _ = fit_tpls(train, n_components, centred=False)
    f = {}
    for g, df in test_f.items():
        known = df[model.material_names[g]]
        f[g] = known.div(known.sum(axis=1).replace(0, 1), axis=0) if g in d_new else known
    return _diagnose(model, f, test_z)


def nearest_material(train, test_f, test_z, d_new, n_components):
    """Use the training material nearest in (autoscaled) property space instead of the new one."""
    model, _ = fit_tpls(train, n_components, centred=False)
    f = {}
    for g, df in test_f.items():
        f[g] = df[model.material_names[g]].copy()
        if g in d_new:
            pre = model.preproc_["D"][g]
            train_d = ((model.d_matrix[g] - pre["center"]) / pre["scale"]).fillna(0)
            for name, row in ((d_new[g] - pre["center"]) / pre["scale"]).fillna(0).iterrows():
                nearest = ((train_d - row) ** 2).sum(axis=1).idxmin()
                f[g][nearest] += df[name]
    return _diagnose(model, f, test_z)


def zero_padded_refit(train, test_f, test_z, d_new, n_components):
    """Add the new material to D, with an all-zero column in F, and refit the default TPLS."""
    padded = {
        "D": {g: pd.concat([d, d_new[g]]) if g in d_new else d for g, d in train["D"].items()},
        "F": {
            g: f.reindex(columns=f.columns.tolist() + list(d_new[g].index), fill_value=0.0) if g in d_new else f
            for g, f in train["F"].items()
        },
        "Z": train["Z"],
        "Y": train["Y"],
    }
    model, _ = fit_tpls(padded, n_components, centred=False)
    return _diagnose(model, {g: df[model.material_names[g]] for g, df in test_f.items()}, test_z)


# ---- Approach 1: project the new material onto the D block ---------------------------------------


def projection_autoscaled(train, test_f, test_z, d_new, n_components, deflate_d: bool = True):
    """Approach 1 on the default model: the new material's F scale has to be guessed (median)."""
    model, _ = fit_tpls(train, n_components, centred=False, deflate_d=deflate_d)
    return predict(model, DataFrameDict({"F": test_f, "Z": test_z, "Y": {}}), d_new)


def projection_autoscaled_no_d_deflation(train, test_f, test_z, d_new, n_components):
    """Approach 1 on autoscaled F, without the D-block deflation."""
    return projection_autoscaled(train, test_f, test_z, d_new, n_components, deflate_d=False)


def projection_centred(train, test_f, test_z, d_new, n_components, deflate_d: bool = True):
    """Approach 1: TPLS with centred F; the new material is scored through the D block."""
    model, means = fit_tpls(train, n_components, centred=True, deflate_d=deflate_d)
    return predict(model, DataFrameDict({"F": centre_f(test_f, means), "Z": test_z, "Y": {}}), d_new)


def projection_centred_no_d_deflation(train, test_f, test_z, d_new, n_components):
    """Approach 1, on TPLS without the D-block deflation (see ``TPLSWithoutDDeflation``)."""
    return projection_centred(train, test_f, test_z, d_new, n_components, deflate_d=False)


# ---- Approach 2: pseudo-recipe, the new material as a mixture of known ones -------------------------


def pseudo_recipe_default(train, test_f, test_z, d_new, n_components, deflate_d: bool = True):
    """Approach 2 on the unchanged default TPLS (autoscaled F)."""
    model, _ = fit_tpls(train, n_components, centred=False, deflate_d=deflate_d)
    f = {}
    for g, df in test_f.items():
        f[g] = substitute(df, pseudo_recipe(model, d_new[g], g)) if g in d_new else df[model.material_names[g]]
    return _diagnose(model, f, test_z)


def pseudo_recipe_no_d_deflation(train, test_f, test_z, d_new, n_components):
    """Approach 2 on autoscaled-F TPLS without the D-block deflation."""
    return pseudo_recipe_default(train, test_f, test_z, d_new, n_components, deflate_d=False)


# ---- Approach 3: collapse the T to an L with mixing rules, then PLS ---------------------------------


def mixture_descriptors(f: dict, d: dict, latent: dict | None = None) -> pd.DataFrame:
    """Blend descriptors computed from the properties of the materials it uses (mixing rules).

    ``mean``: the fraction-weighted mean of every property (the ideal mixing rule). With
    ``latent`` (per group, a loadings matrix that maps preprocessed properties to material
    scores), also ``cov``: the second central moments of the blend's material scores, i.e.
    their fraction-weighted (co)variances. A blend of two very different materials and a
    single material with their average properties have the same mean, but not the same
    spread: these moments describe departures from ideal mixing.

    ``d`` must already be centred and scaled with the training statistics, and cover every material.
    """
    parts = []
    for g, f_g in f.items():
        d_g = d[g].loc[f_g.columns].fillna(0.0).to_numpy()
        w = f_g.to_numpy()
        mean = w @ d_g
        parts.append(pd.DataFrame(mean, index=f_g.index, columns=[f"{g}: mean {c}" for c in d[g].columns]))
        if latent is not None:
            m = d_g @ latent[g]
            m_bar = w @ m
            pairs = [(a, b) for a in range(m.shape[1]) for b in range(a, m.shape[1])]
            cov = np.column_stack([w @ (m[:, a] * m[:, b]) - m_bar[:, a] * m_bar[:, b] for a, b in pairs])
            parts.append(pd.DataFrame(cov, index=f_g.index, columns=[f"{g}: cov {a + 1}{b + 1}" for a, b in pairs]))
    return pd.concat(parts, axis=1)


def _mixing_rule_pls(train, test_f, test_z, d_new, n_components, moments, n_pc: int = 3) -> pd.DataFrame:
    d_all = {g: pd.concat([d, d_new[g]]) if g in d_new else d for g, d in train["D"].items()}
    stats = {g: (d.mean(), d.std().replace(0, 1)) for g, d in train["D"].items()}
    d_std = {g: (d - stats[g][0]) / stats[g][1] for g, d in d_all.items()}
    latent = None
    if moments:  # leading principal directions of the training materials' properties, per group
        latent = {}
        for g, d in train["D"].items():
            x = ((d - stats[g][0]) / stats[g][1]).fillna(0.0).to_numpy()
            latent[g] = np.linalg.svd(x, full_matrices=False)[2][: min(n_pc, x.shape[1])].T
    zkey = next(iter(train["Z"]))
    x_train = pd.concat([train["Z"][zkey], mixture_descriptors(train["F"], d_std, latent)], axis=1)
    x_test = pd.concat([test_z[zkey], mixture_descriptors(test_f, d_std, latent)], axis=1)
    keep = x_train.std() > 1e-12
    model = PLS(n_components=n_components).fit(x_train.loc[:, keep], next(iter(train["Y"].values())))
    return model.predict(x_test.loc[:, keep])


def mixing_rule_pls(train, test_f, test_z, d_new, n_components):
    """Approach 3a: PLS on [Z, fraction-weighted mean properties per group] (ideal mixing rule)."""
    return _mixing_rule_pls(train, test_f, test_z, d_new, n_components, moments=False)


def mixing_moments_pls(train, test_f, test_z, d_new, n_components):
    """Approach 3b: as 3a, plus the spread of the materials in each blend, for non-ideal mixing."""
    return _mixing_rule_pls(train, test_f, test_z, d_new, n_components, moments=True)


# ---- Approach 4: hierarchical, property-based effect + a shrunken offset per known material ----------


def ridge_gcv(x: np.ndarray, e: np.ndarray, grid: np.ndarray | None = None) -> tuple[np.ndarray, ...]:
    """Ridge regression of each column of ``e`` on ``x``, with the penalty chosen by GCV per column.

    Returns the coefficients (columns of ``x`` by columns of ``e``) and, per column of ``e``, the
    noise variance ``sigma2`` and the prior variance of a coefficient implied by the chosen
    penalty, ``tau2 = sigma2 / lambda`` (the ridge solution is then the BLUP).
    """
    grid = np.logspace(-3, 3, 31) if grid is None else grid
    u, sv, vt = np.linalg.svd(x, full_matrices=False)
    n = x.shape[0]
    coef = np.zeros((x.shape[1], e.shape[1]))
    tau2, sigma2 = np.zeros(e.shape[1]), np.zeros(e.shape[1])
    for k in range(e.shape[1]):
        uty = u.T @ e[:, k]
        best = None
        for lam in grid:
            shrink = sv**2 / (sv**2 + lam)
            fitted = u @ (shrink * uty)
            dof = shrink.sum()
            gcv = n * np.sum((e[:, k] - fitted) ** 2) / (n - dof) ** 2
            if best is None or gcv < best[0]:
                best = (gcv, lam, shrink, dof, fitted)
        _, lam, shrink, dof, fitted = best
        coef[:, k] = vt.T @ (sv / (sv**2 + lam) * uty)  # = shrink / sv, without 0/0 for a zero column
        sigma2[k] = np.sum((e[:, k] - fitted) ** 2) / (n - dof)
        tau2[k] = sigma2[k] / lam
    return coef, sigma2, tau2


def material_offsets(model: TPLS, means: dict, train: dict) -> tuple[dict[str, pd.DataFrame], np.ndarray, np.ndarray]:
    """Per known material, the part of its effect that its properties do not explain (BLUP, ridge by GCV).

    The residuals of the property-based model are regressed on the raw fractions of all materials.
    """
    fitted = _diagnose(model, centre_f(train["F"], means), train["Z"])
    resid = (next(iter(train["Y"].values())) - fitted).to_numpy()
    f_all = pd.concat(list(train["F"].values()), axis=1)
    coef, sigma2, tau2 = ridge_gcv(f_all.to_numpy(), resid)
    offsets, start = {}, 0
    for g, f_g in train["F"].items():
        offsets[g] = pd.DataFrame(coef[start : start + f_g.shape[1]], index=f_g.columns, columns=fitted.columns)
        start += f_g.shape[1]
    return offsets, sigma2, tau2


def hierarchical(train, test_f, test_z, d_new, n_components, return_parts: bool = False):
    """Approach 4: approach 1 (no D deflation) plus the offsets of the known materials in each blend.

    The new material's own offset is 0: nothing is known about it beyond its properties.
    """
    model, means = fit_tpls(train, n_components, centred=True, deflate_d=False)
    base = predict(model, DataFrameDict({"F": centre_f(test_f, means), "Z": test_z, "Y": {}}), d_new)
    offsets, sigma2, tau2 = material_offsets(model, means, train)
    known = sum(test_f[g][offsets[g].index].to_numpy() @ offsets[g].to_numpy() for g in offsets)
    pred = base + known
    if return_parts:
        return pred, sigma2, tau2
    return pred


def few_shot_update(
    pred_zero: pd.DataFrame, f_new: pd.Series, y_shot: pd.DataFrame, tau2: np.ndarray, sigma2: np.ndarray
) -> pd.DataFrame:
    """Approach 5: posterior mean of the new material's offset from a few measured blends.

    ``pred_zero`` are the zero-shot predictions (approach 4) for all blends with the new material,
    ``f_new`` its fraction in each, and ``y_shot`` the measured quality of the blends used as shots.
    The offset prior is N(0, tau2) per Y column, the noise N(0, sigma2): the classic BLUP.
    """
    shots = y_shot.index
    resid = (y_shot - pred_zero.loc[shots]).to_numpy()
    f_s = f_new.loc[shots].to_numpy()
    delta = (f_s @ resid) / (np.sum(f_s**2) + sigma2 / tau2)
    return pred_zero + np.outer(f_new.to_numpy(), delta)
