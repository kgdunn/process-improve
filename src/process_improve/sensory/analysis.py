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

from process_improve.multivariate.methods import PCA, PLS, vip
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


def relate_observational(
    agg: pd.DataFrame,
    covariates: pd.DataFrame,
    *,
    n_components: int = 2,
    alpha: float = 0.05,
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
    return {
        "mode": "observational",
        "n_components": max_comp,
        "alpha": alpha,
        "vip": drivers,
        "associations": assoc,
    }


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
    model: str = "main_effects",
    n_components: int = 2,
    conf_level: float = 0.95,
    alpha: float = 0.05,
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
    model : str
        Design model for the ``designed`` relate step (default
        ``"main_effects"``).
    n_components : int
        Components for the PLS relate step and the PCA map.
    conf_level : float
        Confidence level for the product-mean intervals.
    alpha : float
        Target false-discovery rate for the relate step.

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

    if drop_panelists == "auto":
        dropped = list(card.flagged)
    elif isinstance(drop_panelists, list):
        dropped = drop_panelists
    else:
        dropped = []
    clean = apply_correction(panel, dropped)

    agg = aggregate_to_product(clean)
    if validated.mode == "designed":
        relate = relate_designed(agg, validated.covariates, model=model, alpha=alpha)
    else:
        relate = relate_observational(agg, validated.covariates, n_components=n_components, alpha=alpha)

    return AnalysisResult(
        mode=validated.mode,
        panel=card,
        dropped=dropped,
        relate=relate,
        product_means=product_means(clean, conf_level=conf_level),
        pca=_pca_map(agg, n_components=n_components),
        config={
            "model": model,
            "n_components": n_components,
            "conf_level": conf_level,
            "alpha": alpha,
            "content_hash": validated.content_hash,
        },
    )
