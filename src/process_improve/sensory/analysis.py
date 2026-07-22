"""(c) Kevin Dunn, 2010-2026. MIT License.

Relate descriptive panel attributes to the product.

:func:`analyze_descriptive` runs the proof-of-concept pipeline on a validated
dataset: score and (optionally) correct the panel, then relate each sensory
attribute to the product. The relate step dispatches on the validation mode:

* **observational** (supported) - the product has measured descriptors but
  unknown formulation, so the attribute block is related to the descriptors
  with PLS (:class:`process_improve.multivariate.PLS` plus VIP) and
  per-descriptor correlations, reported as association rather than causation.
* **designed** (stub, not implemented yet) - the product is a controlled
  experimental run; the plan is to regress each attribute on the design factors
  via :func:`process_improve.experiments.analyze_experiment` for factor effects.
  See :func:`relate_designed`; it raises ``NotImplementedError`` for now.

The observational relate corrects across the family of tests with
Benjamini-Hochberg FDR and returns supporting product means with confidence
intervals and a PCA sensory map. Both the marginal associations and the
cross-validated discriminator are additionally gated on a leave-one-out
jackknife, so an association or predictive coefficient that rests on a single
high-leverage observation (a predictor that is non-zero on only one product,
common in sparse, wide descriptor blocks) is demoted rather than reported. The
jackknife adds no threshold of its own: it reuses the same ``alpha`` and the
number of observations, so a genuine multi-observation driver is unaffected.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.stats import t as t_dist

from process_improve.multivariate.methods import PCA, PLS, MCUVScaler, selectivity_ratio, vip
from process_improve.sensory.mam import MAMResult, align_scores, mixed_assessor_model
from process_improve.sensory.panel import PanelScorecard, apply_correction, panel_scorecard
from process_improve.sensory.validation import ValidationResult
from process_improve.univariate.metrics import benjamini_hochberg, confidence_interval


@dataclass
class AnalysisResult:
    """Outcome of :func:`analyze_descriptive`.

    Attributes
    ----------
    mode : str
        ``"designed"`` or ``"observational"``.
    panel : PanelScorecard
        The per-panelist scorecard and flags.
    dropped : list of str
        Panelists removed before the relate step.
    mam : MAMResult
        Mixed Assessor Model: per-panelist scaling coefficients and the MAM vs
        classical product-effect F-tests.
    correction : str
        The panel correction applied before relating: ``"none"``, ``"align"``,
        or ``"drop"``.
    relate : dict
        Mode-specific relate results; see :func:`analyze_descriptive`.
    product_means : pandas.DataFrame
        Per product-by-attribute mean with a confidence interval.
    pca : dict
        Product sensory map: ``scores``, ``loadings``, ``explained_variance``.
    config : dict
        The options the analysis ran with.
    """

    mode: str
    panel: PanelScorecard
    dropped: list[str]
    mam: MAMResult
    correction: str
    relate: dict[str, Any]
    product_means: pd.DataFrame
    pca: dict[str, Any]
    config: dict[str, Any] = field(default_factory=dict)


def aggregate_to_product(panel: pd.DataFrame) -> pd.DataFrame:
    """Return a product-by-attribute table of mean scores.

    Parameters
    ----------
    panel : pandas.DataFrame
        Validated ``descriptive_long`` panel data.

    Returns
    -------
    pandas.DataFrame
        Index ``product``, one column per attribute, values the mean score
        over panelists and replicates.
    """
    wide = panel.pivot_table(
        index="product", columns="attribute", values="score", aggfunc="mean", observed=True
    )
    wide.index = wide.index.astype(str)
    wide.columns.name = None
    return wide


#: Minimum usable observations before the leave-one-out jackknife of an
#: association is defined; below this an association cannot be certified as
#: influence-robust (mirrors the ``len < 4`` guard in ``panel._mad_bands``).
_MIN_OBS_FOR_JACKKNIFE = 4

#: Correlations are clipped off +/-1 before the Fisher-z transform so ``arctanh``
#: stays finite.
_R_CLIP = 1.0 - 1e-12


def _attach_fdr(records: list[dict[str, Any]], alpha: float) -> list[dict[str, Any]]:
    """Attach Benjamini-Hochberg q-values (and reject flags) to ``records``."""
    pvals = [r["p_value"] for r in records]
    if not pvals:
        return records
    bh = benjamini_hochberg(np.asarray(pvals), alpha=alpha)
    for rec, q, rej in zip(records, bh.p_adjusted, bh.reject, strict=True):
        rec["q_value"] = float(q)
        # Harden the marginal significance: an association counts only when it also
        # survives the leave-one-out jackknife, so a single high-leverage
        # observation cannot manufacture a "significant" correlation.
        rec["significant"] = bool(rej) and bool(rec.get("influence_robust", True))
    return records


def _jackknife_correlation(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[float, bool, int]:
    """Leave-one-out jackknife significance of a Pearson correlation.

    Returns ``(jackknife_se, influence_robust, n_supporting)``. The correlation is
    Fisher-z transformed, and ``influence_robust`` is ``True`` when the two-sided
    ``alpha`` jackknife confidence interval for the transformed correlation excludes
    zero, i.e. the association does not collapse when any single observation is
    removed. A predictor that is non-zero on a single high-leverage observation has
    one deletion that drives the correlation to zero, which inflates the jackknife
    standard error until the interval spans zero and the association is demoted. The
    rule introduces no threshold of its own: it reuses ``alpha`` and the number of
    observations, so a genuine multi-observation driver stays significant while a
    single-support spike does not.
    """
    n = int(x.size)
    if n < _MIN_OBS_FOR_JACKKNIFE:
        return float("nan"), False, n

    def _z(r_value: float) -> float:
        return float(np.arctanh(np.clip(r_value, -_R_CLIP, _R_CLIP)))

    z_full = _z(float(pearsonr(x, y)[0]))
    pseudo = np.empty(n)
    for i in range(n):
        keep = np.arange(n) != i
        xi, yi = x[keep], y[keep]
        # Removing the sole support of a spike leaves a constant column: no variance
        # means no correlation, so the deleted-sample estimate is zero.
        r_i = float(pearsonr(xi, yi)[0]) if xi.std() > 0 and yi.std() > 0 else 0.0
        pseudo[i] = n * z_full - (n - 1) * _z(r_i)
    z_mean = float(pseudo.mean())
    se = float(pseudo.std(ddof=1) / np.sqrt(n))
    if not np.isfinite(se) or se <= 0.0:
        return se, False, n
    t_crit = float(t_dist.ppf(1.0 - alpha / 2.0, df=n - 1))
    return se, bool(abs(z_mean) > t_crit * se), n


def _fit_pls_safe(x: pd.DataFrame, y: pd.DataFrame, n_components: int) -> tuple[PLS | None, int]:
    """Fit PLS, stepping the component count down on a near-collinear (singular) block.

    Near-duplicate descriptor columns (a proxy that is almost an exact function
    of a driver) can make the high-order PLS deflation singular. Retry with one
    fewer component until the fit succeeds, returning ``(model, components_used)``
    or ``(None, 0)`` if even a single component fails.
    """
    for k in range(max(1, n_components), 0, -1):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                return PLS(n_components=k).fit(x, y), k
        except np.linalg.LinAlgError:  # noqa: PERF203 - retry on singular block; loop is a few iterations
            continue
    return None, 0


def _collinear_clusters(x_block: pd.DataFrame, threshold: float) -> dict[str, int]:
    """Group descriptors into clusters of mutually high absolute correlation.

    Single-linkage connected components on the descriptor ``|corr|`` matrix:
    two descriptors join the same cluster when their absolute Pearson
    correlation is at least ``threshold``. Returns ``{descriptor: cluster_id}``
    with cluster ids assigned in column order (a singleton descriptor gets its
    own id). Collinear proxies therefore share an id, which is how the
    discriminator reports that they cannot be told apart.
    """
    cols = list(x_block.columns)
    n = len(cols)
    corr = np.nan_to_num(x_block.corr().abs().to_numpy(), nan=0.0)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if corr[i, j] >= threshold:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)

    roots: dict[int, int] = {}
    cluster_of: dict[str, int] = {}
    for i, name in enumerate(cols):
        root = find(i)
        if root not in roots:
            roots[root] = len(roots)
        cluster_of[str(name)] = roots[root]
    return cluster_of


def discriminate_observational(  # noqa: PLR0913, PLR0915
    agg: pd.DataFrame,
    covariates: pd.DataFrame,
    *,
    n_components: int = 2,
    alpha: float = 0.05,
    n_permutations: int = 199,
    random_state: int = 0,
    cluster_threshold: float = 0.95,
    max_components_cv: int = 4,
) -> dict[str, Any]:
    """Cross-validated discriminator of which descriptors carry predictive signal.

    The marginal associations (:func:`relate_observational`) flag every
    descriptor that correlates with an attribute in-sample, genuine drivers and
    proxies alike. This step adds out-of-sample evidence:

    1. a per-attribute cross-validated Q-squared gate (is the attribute
       predictable from the descriptor block at all),
    2. a selectivity ratio per descriptor on the target-projected predictive
       direction, with a permutation p-value (Benjamini-Hochberg corrected
       across the whole family), so a descriptor that merely correlates by
       chance but does not enter the predictive direction is demoted, and
    3. a collinear-cluster id per descriptor.

    What it cannot do is rank descriptors *within* a collinear cluster: two
    descriptors that carry the same information predict equally well out of
    sample, so they share a cluster id and both stay significant. Separating
    them needs an external dataset or a designed experiment.

    Parameters
    ----------
    agg : pandas.DataFrame
        Product-by-attribute mean table (index ``product``).
    covariates : pandas.DataFrame
        One row per product with the measured descriptors (plus a ``product``
        column, which is dropped here).
    n_components : int
        Latent components for the in-sample selectivity-ratio fit.
    alpha : float
        Target false-discovery rate for the permutation family.
    n_permutations : int
        Number of label permutations for the selectivity-ratio null.
    random_state : int
        Seed for the permutations and the cross-validation folds.
    cluster_threshold : float
        Absolute-correlation threshold for the collinear clustering.
    max_components_cv : int
        Cap on the component count the Q-squared gate may select.

    Returns
    -------
    dict
        ``per_attribute`` (the Q-squared gate per attribute), ``descriptors``
        (the per attribute-descriptor selectivity ratio, permutation q-value,
        ``jackknife_significant`` flag from the leave-one-out beta confidence
        interval, ``discriminator_significant`` flag and ``cluster_id``),
        ``clusters`` (the descriptor-to-cluster map), and the settings used. A
        descriptor is ``discriminator_significant`` only when it also survives the
        jackknife, so a coefficient carried by a single product is demoted.
    """
    x_all = covariates.loc[agg.index]
    descriptors = [c for c in x_all.columns if c != "product" and pd.api.types.is_numeric_dtype(x_all[c])]
    x_block = x_all[descriptors].astype(float)
    clusters = _collinear_clusters(x_block, cluster_threshold)

    rng = np.random.default_rng(random_state)
    per_attribute: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for attr in agg.columns:
        y = agg[attr].astype(float)
        mask = y.notna()
        x_attr = x_block.loc[mask]
        y_attr = y.loc[mask]
        n_rows = x_attr.shape[0]
        cap = max(1, min(max_components_cv, x_attr.shape[1], n_rows - 2))
        a = max(1, min(n_components, cap))

        # Fit one PLS for this attribute, reused for the Q-squared gate and the
        # selectivity ratio. ``_fit_pls_safe`` steps the component count down if
        # the near-collinear descriptor block makes the fit singular.
        x_scaled = MCUVScaler().fit_transform(x_attr)
        y_scaled = MCUVScaler().fit_transform(y_attr.to_frame())
        pls, a = _fit_pls_safe(x_scaled, y_scaled, a)

        # 1. Leave-one-out cross-validated Q-squared gate: is the attribute
        #    predictable from the descriptor block out of sample? The same LOO
        #    refit also yields a jackknife confidence interval per descriptor
        #    coefficient (Martens' uncertainty test); it is reused below so a
        #    descriptor whose predictive weight rests on a single high-leverage
        #    product is demoted even when it survives the permutation null.
        predictable = False
        q2_cv = float("nan")
        rmsep_cv = float("nan")
        jack_significant: dict[str, bool] = {}
        if pls is not None and n_rows >= 5 and y_attr.std() > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                cv = pls.cross_validate(
                    x_scaled, y_scaled, cv="loo", conf_level=1.0 - alpha, show_progress=False
                )
            q2_cv = float(cv.q_squared.iloc[0])
            rmsep_cv = float(cv.rmse_cv.iloc[0])
            predictable = q2_cv > 0.0
            jack_significant = {str(name): bool(flag) for name, flag in cv.significant.iloc[:, 0].items()}
        per_attribute.append(
            {
                "attribute": str(attr),
                "n_components_cv": a if pls is not None else 0,
                "q2_cv": q2_cv,
                "rmsep_cv": rmsep_cv,
                "predictable": bool(predictable),
            }
        )

        # 2. Selectivity ratio on the predictive direction, with a permutation
        #    null. Multiplicity across descriptors is controlled by the
        #    max-statistic (Westfall-Young) permutation: for each label
        #    permutation, the *largest* selectivity ratio over all descriptors
        #    forms the null. An observed SR above that null is significant after
        #    correction, so even a single genuine driver is detectable without
        #    the resolution loss of a per-test Benjamini-Hochberg floor. The
        #    permutation loop is skipped for attributes the Q-squared gate found
        #    unpredictable: none of their descriptors can be flagged anyway.
        k = len(descriptors)
        if pls is None:  # degenerate block: nothing to relate for this attribute
            sr_values = np.zeros(k)
            p_raw = np.ones(k)
            p_maxt = np.ones(k)
        else:
            sr_values = np.asarray(selectivity_ratio(pls, x_scaled).reindex(descriptors), dtype=float)
            if predictable:
                y_values = y_scaled.to_numpy().ravel()
                ge_each = np.zeros(k)  # per-descriptor null exceedances
                ge_max = np.zeros(k)  # exceedances of the family-wide max null
                done = 0
                for _ in range(n_permutations):
                    permuted = pd.DataFrame(
                        y_values[rng.permutation(n_rows)], index=x_scaled.index, columns=y_scaled.columns
                    )
                    pls_p, _ = _fit_pls_safe(x_scaled, permuted, a)
                    if pls_p is None:
                        continue  # a degenerate permutation contributes nothing to the null
                    sr_p = selectivity_ratio(pls_p, x_scaled).reindex(descriptors).to_numpy()
                    ge_each += sr_p >= sr_values
                    ge_max += np.nanmax(sr_p) >= sr_values
                    done += 1
                denom = done + 1.0
                p_raw = (ge_each + 1.0) / denom
                p_maxt = (ge_max + 1.0) / denom
            else:
                p_raw = np.ones(k)
                p_maxt = np.ones(k)
        for i, desc in enumerate(descriptors):
            desc_robust = jack_significant.get(str(desc), False)
            records.append(
                {
                    "attribute": str(attr),
                    "descriptor": str(desc),
                    "selectivity_ratio": float(sr_values[i]),
                    "p_value": float(p_raw[i]),
                    "q_value": float(p_maxt[i]),
                    "jackknife_significant": bool(desc_robust),
                    "discriminator_significant": bool(p_maxt[i] <= alpha and predictable and desc_robust),
                    "cluster_id": clusters[str(desc)],
                }
            )

    return {
        "per_attribute": per_attribute,
        "descriptors": records,
        "clusters": clusters,
        "alpha": alpha,
        "n_permutations": n_permutations,
        "cluster_threshold": cluster_threshold,
    }


def relate_designed(
    agg: pd.DataFrame,
    covariates: pd.DataFrame,
    *,
    model: str = "main_effects",
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Relate attributes to controlled design factors (not implemented yet).

    Stub for a later release. The plan is to regress each attribute on the
    design factors via :func:`process_improve.experiments.analyze_experiment`
    (or ``analyze_omars`` for DSD/OMARS designs) and report factor effects with
    Benjamini-Hochberg correction. For now use ``mode="observational"``.

    Raises
    ------
    NotImplementedError
        Always, until the designed-mode relate step is built.
    """
    del agg, covariates, model, alpha
    raise NotImplementedError(
        "Designed (DoE/OMARS) relate is not implemented yet; use "
        "mode='observational'. Planned for a later release."
    )


def relate_observational(  # noqa: PLR0913
    agg: pd.DataFrame,
    covariates: pd.DataFrame,
    *,
    n_components: int = 2,
    alpha: float = 0.05,
    discriminator: bool = True,
    n_permutations: int = 199,
    random_state: int = 0,
) -> dict[str, Any]:
    """Relate the attribute block to measured descriptors with PLS plus correlations.

    Each marginal association carries a Pearson ``r``, an FDR ``q_value`` and, from a
    leave-one-out jackknife, ``jackknife_se``, ``influence_robust`` and
    ``n_supporting``. ``significant`` requires both FDR rejection and jackknife
    robustness, so a correlation created by a single high-leverage observation is not
    reported as significant.
    """
    x_block = covariates.loc[agg.index].astype(float)
    y_block = agg.astype(float)
    max_comp = max(1, min(n_components, x_block.shape[1], x_block.shape[0] - 1))
    pls = PLS(n_components=max_comp).fit(x_block, y_block)
    vips = vip(pls)

    drivers: list[dict[str, Any]] = [
        {"descriptor": str(name), "vip": float(value)}
        for name, value in vips.items()
    ]
    drivers.sort(key=lambda r: float(r["vip"]), reverse=True)

    # Per (attribute, descriptor) association, BH-corrected across the family.
    assoc: list[dict[str, Any]] = []
    for attr in y_block.columns:
        for desc in x_block.columns:
            pair = pd.concat([y_block[attr], x_block[desc]], axis=1).dropna()
            if pair.shape[0] >= 3 and pair.iloc[:, 0].std() > 0 and pair.iloc[:, 1].std() > 0:
                yv = pair.iloc[:, 0].to_numpy(dtype=float)
                xv = pair.iloc[:, 1].to_numpy(dtype=float)
                r, p = pearsonr(yv, xv)
                jack_se, robust, n_support = _jackknife_correlation(xv, yv, alpha)
                assoc.append(
                    {
                        "attribute": str(attr),
                        "descriptor": str(desc),
                        "r": float(r),
                        "p_value": float(p),
                        "jackknife_se": float(jack_se),
                        "influence_robust": bool(robust),
                        "n_supporting": int(n_support),
                    }
                )
    assoc = _attach_fdr(assoc, alpha)
    result: dict[str, Any] = {
        "mode": "observational",
        "n_components": max_comp,
        "alpha": alpha,
        "vip": drivers,
        "associations": assoc,
    }
    if discriminator:
        result["discriminator"] = discriminate_observational(
            agg,
            covariates.loc[agg.index],
            n_components=max_comp,
            alpha=alpha,
            n_permutations=n_permutations,
            random_state=random_state,
        )
    return result


def product_means(panel: pd.DataFrame, conf_level: float = 0.95) -> pd.DataFrame:
    """Return per product-by-attribute mean with a confidence interval."""
    rows: list[dict[str, Any]] = []
    for (prod, attr), grp in panel.groupby(["product", "attribute"], observed=True):
        scores = grp[["score"]].dropna()
        center = float(scores["score"].mean())
        if scores.shape[0] >= 2:
            lo, hi = confidence_interval(scores, "score", conflevel=conf_level, style="regular")
        else:
            lo = hi = float("nan")
        rows.append(
            {"product": str(prod), "attribute": str(attr), "mean": center, "ci_low": float(lo), "ci_high": float(hi)}
        )
    return pd.DataFrame(rows)


def _pca_map(agg: pd.DataFrame, n_components: int = 2) -> dict[str, Any]:
    """Fit a PCA on the product-by-attribute means and return the map."""
    filled = agg.dropna(axis=1, how="any")
    max_comp = max(1, min(n_components, filled.shape[0] - 1, filled.shape[1]))
    pca = PCA(n_components=max_comp).fit(filled)
    return {
        "scores": pca.scores_,
        "loadings": pca.loadings_,
        "explained_variance": [float(v) for v in pca.r2_per_component_],
    }


def analyze_descriptive(  # noqa: PLR0913
    validated: ValidationResult,
    *,
    drop_panelists: str | list[str] | None = None,
    correction: str = "none",
    align_method: str = "both",
    model: str = "main_effects",
    n_components: int = 2,
    conf_level: float = 0.95,
    alpha: float = 0.05,
    discriminator: bool = True,
    n_permutations: int = 199,
    random_state: int = 0,
) -> AnalysisResult:
    """Run the descriptive pipeline: panel check, correction, and relate.

    Parameters
    ----------
    validated : ValidationResult
        A passing result from
        :func:`process_improve.sensory.validate_descriptive`.
    drop_panelists : {"auto", None} or list of str
        ``"auto"`` drops every flagged panelist; a list drops exactly those
        ids; ``None`` keeps all panelists.
    correction : {"none", "align", "drop"}
        Panel correction before relating. ``"none"`` (default) leaves scores as
        is; ``"align"`` applies the Mixed Assessor Model scale alignment to all
        panelists (:func:`process_improve.sensory.mam.align_scores`); ``"drop"``
        is a synonym for using ``drop_panelists``. Alignment and dropping
        compose: panelists are aligned first, then any dropped.
    align_method : {"both", "location", "scale"}
        Which MAM lever to apply when ``correction="align"``.
    model : str
        Design model for the ``designed`` relate step (default
        ``"main_effects"``).
    n_components : int
        Components for the PLS relate step and the PCA map.
    conf_level : float
        Confidence level for the product-mean intervals.
    alpha : float
        Target false-discovery rate for the relate step.
    discriminator : bool
        Whether to run the cross-validated discriminator
        (:func:`discriminate_observational`) in the observational relate step.
    n_permutations : int
        Permutations for the discriminator's selectivity-ratio null.
    random_state : int
        Seed for the discriminator's permutations and cross-validation folds.

    Returns
    -------
    AnalysisResult
        See the class docstring.

    Raises
    ------
    ValueError
        If ``validated`` did not pass validation.
    """
    if not validated.ok or validated.normalized_df is None or validated.covariates is None:
        raise ValueError(
            "analyze_descriptive requires a validated dataset; "
            "validate_descriptive reported errors: "
            f"{validated.errors}"
        )

    panel = validated.normalized_df
    card = panel_scorecard(panel)
    mam = mixed_assessor_model(panel)

    # Correction: align all panelists onto a common scale (MAM), then drop.
    working = align_scores(panel, method=align_method) if correction == "align" else panel
    if drop_panelists == "auto":
        dropped = list(card.flagged)
    elif isinstance(drop_panelists, list):
        dropped = drop_panelists
    else:
        dropped = []
    clean = apply_correction(working, dropped)

    agg = aggregate_to_product(clean)
    if validated.mode == "designed":
        relate = relate_designed(agg, validated.covariates, model=model, alpha=alpha)
    else:
        relate = relate_observational(
            agg,
            validated.covariates,
            n_components=n_components,
            alpha=alpha,
            discriminator=discriminator,
            n_permutations=n_permutations,
            random_state=random_state,
        )

    return AnalysisResult(
        mode=validated.mode,
        panel=card,
        dropped=dropped,
        mam=mam,
        correction=correction,
        relate=relate,
        product_means=product_means(clean, conf_level=conf_level),
        pca=_pca_map(agg, n_components=n_components),
        config={
            "model": model,
            "correction": correction,
            "align_method": align_method,
            "n_components": n_components,
            "conf_level": conf_level,
            "alpha": alpha,
            "discriminator": discriminator,
            "n_permutations": n_permutations,
            "random_state": random_state,
            "content_hash": validated.content_hash,
        },
    )
