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
intervals and a PCA sensory map.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

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


def _attach_fdr(records: list[dict[str, Any]], alpha: float) -> list[dict[str, Any]]:
    """Attach Benjamini-Hochberg q-values (and reject flags) to ``records``."""
    pvals = [r["p_value"] for r in records]
    if not pvals:
        return records
    bh = benjamini_hochberg(np.asarray(pvals), alpha=alpha)
    for rec, q, rej in zip(records, bh.p_adjusted, bh.reject, strict=True):
        rec["q_value"] = float(q)
        rec["significant"] = bool(rej)
    return records


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


def discriminate_observational(  # noqa: PLR0913
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
        ``discriminator_significant`` flag and ``cluster_id``), ``clusters``
        (the descriptor-to-cluster map), and the settings used.
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

        # 1. Cross-validated Q-squared gate (raw inputs; the splitter scales inside folds).
        cap = max(1, min(max_components_cv, x_attr.shape[1], n_rows - 2))
        predictable = False
        q2_cv = float("nan")
        rmsep_cv = float("nan")
        a = max(1, min(n_components, cap))
        if n_rows >= 5 and y_attr.std() > 0:
            sel = PLS.select_n_components(
                x_attr,
                y_attr,
                max_components=cap,
                cv=5,
                n_repeats=3,
                random_state=random_state,
            )
            a = int(sel.n_components)
            q2_cv = float(sel.r2y_validated.loc[a, "total"])
            rmsep_cv = float(sel.rmsecv.loc[a, "total"])
            predictable = q2_cv > 0.0
        per_attribute.append(
            {
                "attribute": str(attr),
                "n_components_cv": a,
                "q2_cv": q2_cv,
                "rmsep_cv": rmsep_cv,
                "predictable": bool(predictable),
            }
        )

        # 2. Selectivity ratio on the predictive direction, with a permutation
        #    null. The permutation loop is the costly part, so it is skipped for
        #    attributes the Q-squared gate already found unpredictable: none of
        #    their descriptors can be flagged anyway.
        x_scaled = MCUVScaler().fit_transform(x_attr)
        y_scaled = MCUVScaler().fit_transform(y_attr.to_frame())
        pls = PLS(n_components=a).fit(x_scaled, y_scaled)
        sr_obs = selectivity_ratio(pls, x_scaled).reindex(descriptors)
        if predictable:
            y_values = y_scaled.to_numpy().ravel()
            ge_counts = np.zeros(len(descriptors))
            for _ in range(n_permutations):
                permuted = pd.DataFrame(
                    y_values[rng.permutation(n_rows)], index=x_scaled.index, columns=y_scaled.columns
                )
                pls_p = PLS(n_components=a).fit(x_scaled, permuted)
                sr_p = selectivity_ratio(pls_p, x_scaled).reindex(descriptors).to_numpy()
                ge_counts += sr_p >= sr_obs.to_numpy()
            perm_p = (ge_counts + 1.0) / (n_permutations + 1.0)
        else:
            perm_p = np.ones(len(descriptors))
        for i, desc in enumerate(descriptors):
            records.append(
                {
                    "attribute": str(attr),
                    "descriptor": str(desc),
                    "selectivity_ratio": float(sr_obs[desc]),
                    "p_value": float(perm_p[i]),
                    "cluster_id": clusters[str(desc)],
                    "_predictable": bool(predictable),
                }
            )

    # 3. Benjamini-Hochberg per attribute (each attribute is its own family of
    #    tests), then gate the significance flag on the attribute being
    #    predictable out of sample.
    by_attribute: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        by_attribute.setdefault(rec["attribute"], []).append(rec)
    for group in by_attribute.values():
        _attach_fdr(group, alpha)
    for rec in records:
        rec["discriminator_significant"] = bool(rec.pop("significant") and rec.pop("_predictable"))

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
    """Relate the attribute block to measured descriptors with PLS plus correlations."""
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
                r, p = pearsonr(pair.iloc[:, 0], pair.iloc[:, 1])
                assoc.append(
                    {"attribute": str(attr), "descriptor": str(desc), "r": float(r), "p_value": float(p)}
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
