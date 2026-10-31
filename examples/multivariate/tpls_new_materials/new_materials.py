# (c) Kevin Dunn, 2010-2026. MIT License.
"""Predict with a fitted TPLS model when a blend uses a material that was not in training.

A new material is a new row of D (its properties are known) and a new column of F (its
fractions in the blends to predict). ``TPLS.diagnose`` refuses such a column. The functions
here extend a *fitted* model to new materials without refitting it.

The key fact, used by every function here, is that TPLS treats the materials (rows of D)
exactly as a PCA/PLS model treats observations: each material gets scores ``r`` from its
properties, through the weights ``S`` and deflation loadings ``V`` of the D block. So a new
material is projected like a new observation, component by component. Its F-deflation
loading is set to zero, which is the value TPLS would have computed for a column of F that
is zero in every training blend, and which leaves every existing prediction unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from process_improve.multivariate.methods import TPLS, DataFrameDict


class TPLSWithoutDDeflation(TPLS):
    """TPLS that deflates F, Z and Y between components, but never D (a prototype for this study).

    The D-block deflation ``D <- D - r v'`` removes a direction of the material space after one
    component, whether or not Y has been fully explained along it. When the super score also
    carries Z, or several groups of materials, Y usually needs that direction again, and it is
    gone: the fit stops improving. Deflating F by the super score already removes from the
    mixture properties ``F D`` exactly what the component explained, so the D deflation is not
    needed for prediction. Without it, a one-group model is a PLS on ``[Z, F D]``.
    """

    deflate_d = False

    def _calculate_and_store_deflation_matrices(self, pc_a, t_super_i, q_super_i, r_i) -> None:
        """Run the TPLS deflation, then put D back as it was."""
        d_before = {k: np.asarray(v, dtype=float).copy() for k, v in self.d_mats.items()}
        super()._calculate_and_store_deflation_matrices(pc_a, t_super_i, q_super_i, r_i)
        self.d_mats = d_before


def project_materials(model: TPLS, d_new: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Score new materials on the D block of a fitted TPLS model.

    Parameters
    ----------
    model : TPLS
        A fitted model.
    d_new : dict[str, pd.DataFrame]
        Per group, the properties of the new materials: rows are materials, columns the
        same properties as the training D block of that group. Missing values are allowed.

    Returns
    -------
    dict[str, pd.DataFrame]
        Per group, a frame with one row per new material and columns ``r_1..r_A`` (the
        material scores, on the scale of ``model.r_loadings_f``), ``T2`` (Hotelling's T2
        of the material among the training materials) and ``SPE`` (sum of squares of the
        properties the scores do not reproduce, preprocessed scale). Score the training
        materials the same way to get the reference distribution of both.
    """
    out = {}
    n_comp = model.n_components_
    deflate_d = getattr(model, "deflate_d", True)
    for group, df in d_new.items():
        pre = model.preproc_["D"][group]
        raw = df[pre["center"].index].to_numpy(dtype=float)
        present = ~np.isnan(raw)
        # A property with no variance in training has a NaN scale; training zeroed it, so do the same.
        d_tilde = np.nan_to_num((raw - pre["center"].to_numpy()) / pre["scale"].to_numpy() / pre["block"])
        s_mat = model.s_loadings_d[group].to_numpy()
        v_mat = model.v_loadings_d[group].to_numpy()
        resid = d_tilde.copy()
        scores = np.zeros((raw.shape[0], n_comp))
        for a in range(n_comp):
            # Same masked projection as TPLS step 5, r = D s / s's, then the step-14 deflation.
            denom = np.sum((s_mat[:, a][None, :] * present) ** 2, axis=1)
            scores[:, a] = resid @ s_mat[:, a] / np.where(denom > 0, denom, np.nan)
            if deflate_d:
                resid = resid - np.outer(scores[:, a], v_mat[:, a]) * present
        if not deflate_d:
            # Without the deflation the scores are not orthogonal: the residual is what the span of S misses.
            resid = d_tilde - d_tilde @ s_mat @ np.linalg.pinv(s_mat)
        train = model.r_loadings_f[group].to_numpy()
        # T2 among the training materials of the group; their scores are centred, since D is.
        inv_cov = np.linalg.pinv(np.cov(train, rowvar=False).reshape(n_comp, n_comp))
        frame = pd.DataFrame(scores, index=df.index, columns=[f"r_{a + 1}" for a in range(n_comp)])
        frame["T2"] = np.einsum("ia,ab,ib->i", scores, inv_cov, scores)
        frame["SPE"] = np.sum(resid**2, axis=1)
        out[group] = frame
    return out


def predict(
    model: TPLS,
    X: DataFrameDict,
    d_new: dict[str, pd.DataFrame] | None = None,
    f_scale_new: float | dict[str, float] | None = None,
) -> pd.DataFrame:
    """Predict Y for blends that may use materials not in training (approach 1, latent projection).

    Mirrors ``TPLS.diagnose`` (complete F and Z rows assumed), with two additions for every
    material named in ``d_new``: its scores ``r`` come from :func:`project_materials`, and
    its F-deflation loading is zero. The denominator of the blend score is the training
    ``r'r``, so a new material's column does not rescale the scores of other blends.

    Parameters
    ----------
    model : TPLS
        A fitted model.
    X : DataFrameDict
        The blends to predict. Each F group holds the training materials plus any new ones.
    d_new : dict[str, pd.DataFrame] or None
        Properties of the new materials, per group. Groups may be omitted.
    f_scale_new : float, dict or None
        Only for a model fitted with autoscaled F (the default). The F scale of a new
        material is undefined: it never varied in training. ``None`` uses the median scale
        of the training materials of its group. A model fitted with
        ``skip_f_matrix_preprocessing=True`` ignores this and uses the fractions as given.

    Returns
    -------
    pd.DataFrame
        Predicted Y (first Y block), original units, one row per blend.
    """
    d_new = d_new or {}
    r_new = project_materials(model, d_new) if d_new else {}
    n_comp = model.n_components_
    groups = list(model.material_names)
    f_x, r_all, p_all, r_den = {}, {}, {}, {}
    for g in groups:
        known = model.material_names[g]
        extra = list(r_new[g].index) if g in r_new else []
        f_g = X["F"][g][known + extra].to_numpy(dtype=float)
        if not model.skip_f_matrix_preprocessing_:
            centre = model.preproc_["F"][g]["center"].to_numpy()
            scale = model.preproc_["F"][g]["scale"].to_numpy()
            s_new = f_scale_new.get(g) if isinstance(f_scale_new, dict) else f_scale_new
            s_new = float(np.nanmedian(scale)) if s_new is None else s_new
            # A new material was never used in training: its training mean fraction is 0.
            centre = np.r_[centre, np.zeros(len(extra))]
            scale = np.r_[scale, np.full(len(extra), s_new)]
            f_g = np.nan_to_num((f_g - centre) / scale)
        f_x[g] = f_g
        r_known = model.r_loadings_f[g].to_numpy()
        r_extra = r_new[g][[f"r_{a + 1}" for a in range(n_comp)]].to_numpy() if extra else np.zeros((0, n_comp))
        r_all[g] = np.vstack([r_known, r_extra])
        p_all[g] = np.vstack([model.p_loadings_f[g].to_numpy(), np.zeros((len(extra), n_comp))])
        r_den[g] = np.sum(r_known**2, axis=0)

    z_x = {}
    for k in model.condition_names:
        pre = model.preproc_["Z"][k]
        z_x[k] = np.nan_to_num(((X["Z"][k][model.condition_names[k]] - pre["center"]) / pre["scale"]).to_numpy(float))

    n_obs = next(iter(f_x.values())).shape[0]
    scores = np.zeros((n_obs, n_comp))
    w_super = model.w_loadings_super.to_numpy()
    for a in range(n_comp):
        t_f = sum(f_x[g] @ r_all[g][:, a] for g in groups) / sum(r_den[g][a] for g in groups)
        if model.n_conditions > 0:
            t_z = sum(z_x[k] @ model.w_loadings_z[k].to_numpy()[:, a] for k in z_x)
            t_z = t_z / sum(np.sum(model.w_loadings_z[k].to_numpy()[:, a] ** 2) for k in z_x)
            t_a = np.c_[t_z, t_f] @ w_super[:, a]
        else:
            t_a = t_f * w_super[0, a]
        scores[:, a] = t_a
        f_x = {g: f_g - np.outer(t_a, p_all[g][:, a]) for g, f_g in f_x.items()}
        z_x = {k: z_k - np.outer(t_a, model.p_loadings_z[k].to_numpy()[:, a]) for k, z_k in z_x.items()}

    key = next(iter(model.quality_names))
    pre = model.preproc_["Y"][key]
    y_hat = scores @ model.q_loadings_y[key].to_numpy().T * pre["scale"].to_numpy() + pre["center"].to_numpy()
    return pd.DataFrame(y_hat, index=X["F"][groups[0]].index, columns=model.quality_names[key])


def centre_f(f_blocks: dict[str, pd.DataFrame], means: dict[str, pd.Series]) -> dict[str, pd.DataFrame]:
    """Subtract the training mean fraction of each material; a material absent from ``means`` has mean 0.

    Use with a model fitted with ``skip_f_matrix_preprocessing=True`` on centred F, which is
    the preprocessing under which a material's effect depends on its properties only.
    """
    return {g: df - means[g].reindex(df.columns).fillna(0.0) for g, df in f_blocks.items()}


def material_coefficients(model: TPLS, group: str, props: pd.DataFrame) -> pd.DataFrame:
    """Change in predicted Y per unit fraction of a material, for any material with known properties.

    A TPLS prediction is affine in the fractions, and for a model fitted with centred F the
    coefficient of a material is a function of its properties alone: ``b(d)``, the model's
    implicit property-to-performance map. Computed as a difference of two predictions, so it
    needs no algebra beyond :func:`predict`.
    """
    names = [f"probe {i}" for i in range(props.shape[0])]
    n_rows = len(names) + 1  # row 0 uses nothing; row i uses one unit of probe i
    f = {g: pd.DataFrame(np.zeros((n_rows, len(m))), columns=m) for g, m in model.material_names.items()}
    f[group] = f[group].join(pd.DataFrame(np.eye(n_rows)[:, 1:], columns=names))
    z = {k: pd.DataFrame([model.preproc_["Z"][k]["center"]] * n_rows) for k in model.condition_names}
    y_hat = predict(model, DataFrameDict({"F": f, "Z": z, "Y": {}}), {group: props.set_axis(names)})
    return pd.DataFrame(y_hat.to_numpy()[1:] - y_hat.to_numpy()[0], index=props.index, columns=y_hat.columns)


def pseudo_recipe(
    model: TPLS, d_new: pd.DataFrame, group: str, n_neighbours: int = 8, space: str = "properties"
) -> pd.DataFrame:
    """Write each new material as a non-negative, sum-to-one mixture of training materials (approach 2).

    The weights reconstruct the new material's preprocessed properties (``space="properties"``)
    or its D-block scores (``space="scores"``) from its ``n_neighbours`` nearest training
    materials. A blend that uses ``x`` of the new material then uses ``x * alpha_k`` of
    training material ``k`` instead, and the unchanged model predicts it. This works with any
    F preprocessing, including the default autoscaling.

    Returns
    -------
    pd.DataFrame
        Rows: new materials. Columns: training materials of the group (weights, rows sum
        to 1), plus ``residual``, the reconstruction error (zero when the new material lies
        in the convex hull of its neighbours).
    """
    pre = model.preproc_["D"][group]
    names = model.material_names[group]

    def prep(df: pd.DataFrame) -> np.ndarray:
        x = np.nan_to_num(((df[pre["center"].index] - pre["center"]) / pre["scale"]).to_numpy(float))
        if space == "scores":
            sc = project_materials(model, {group: df})[group]
            return sc[[c for c in sc.columns if c.startswith("r_")]].to_numpy()
        return x

    train = prep(model.d_matrix[group])
    new = prep(d_new)
    weights = np.zeros((new.shape[0], len(names)))
    resid = np.zeros(new.shape[0])
    for i, x in enumerate(new):
        nearest = np.argsort(np.sum((train - x) ** 2, axis=1))[:n_neighbours]
        # Sum-to-one as a heavily weighted extra row of a non-negative least-squares problem.
        big = 1e3 * max(1.0, float(np.abs(train).max()))
        a_mat = np.vstack([train[nearest].T, big * np.ones((1, len(nearest)))])
        alpha, _ = nnls(a_mat, np.r_[x, big])
        alpha /= alpha.sum()
        weights[i, nearest] = alpha
        resid[i] = float(np.sqrt(np.sum((alpha @ train[nearest] - x) ** 2)))
    out = pd.DataFrame(weights, index=d_new.index, columns=names)
    out["residual"] = resid
    return out


def substitute(f_block: pd.DataFrame, recipe: pd.DataFrame) -> pd.DataFrame:
    """Replace each new-material column of ``f_block`` by its pseudo-recipe over training materials."""
    weights = recipe.drop(columns="residual")
    out = f_block[weights.columns].copy()
    for name in recipe.index:
        out += np.outer(f_block[name].to_numpy(), weights.loc[name].to_numpy())
    return out
