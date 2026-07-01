"""(c) Kevin Dunn, 2010-2026. MIT License.

Agent-callable tool wrappers for the descriptive panel-data pipeline.

Each function is decorated with ``@tool_spec`` so it can be passed straight to
an LLM tool-use API. Inputs are plain JSON (lists of row-records), and every
result is a JSON-serialisable ``dict``. ``analyze_descriptive`` validates its
input first and refuses to run when validation fails, mirroring the in-process
gate enforced by :func:`process_improve.sensory.validate_descriptive`.
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from process_improve.sensory.analysis import analyze_descriptive as _analyze_descriptive
from process_improve.sensory.ingest import reshape_to_long as _reshape_to_long
from process_improve.sensory.mam import align_scores as _align_scores
from process_improve.sensory.mam import mixed_assessor_model as _mixed_assessor_model
from process_improve.sensory.panel import panel_scorecard as _panel_scorecard
from process_improve.sensory.validation import DESCRIPTIVE_LONG_COLUMNS
from process_improve.sensory.validation import validate_descriptive as _validate_descriptive
from process_improve.tool_spec import clean, get_tool_specs, tool_spec

_SENSORY_TOOL_NAMES: list[str] = []


def _register(name: str) -> None:
    _SENSORY_TOOL_NAMES.append(name)


class _ReshapeInput(BaseModel):
    """Input contract for ``sensory_reshape_to_long``."""

    model_config = ConfigDict(extra="forbid")

    data: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        description=(
            "The parsed panel table as row-records. The spreadsheet should already be read into rows "
            "(by the front end or a code sandbox); this tool only reshapes, it does not read files."
        ),
    )
    layout: Literal["long", "wide_by_attribute", "wide_by_product"] = Field(
        ...,
        description=(
            "'wide_by_attribute' when there is one column per attribute (rows are panelist x product x "
            "replicate); 'wide_by_product' when there is one column per product and an attribute label "
            "column (one panelist x product matrix per attribute, stacked); 'long' when there is already "
            "one row per score with attribute and score columns."
        ),
    )
    panelist_id: str = Field(..., description="Name of the column holding the panelist / assessor id.")
    product: str | None = Field(
        None,
        description="Column holding the product / sample id. Required for long and wide_by_attribute.",
    )
    session: str | None = Field(None, description="Optional column holding the session; defaults to 1 if absent.")
    replicate: str | None = Field(None, description="Optional column holding the replicate; defaults to 1 if absent.")
    attributes: list[str] | None = Field(
        None,
        description="wide_by_attribute: the attribute column names. If omitted, all non-id columns are attributes.",
    )
    products: list[str] | None = Field(
        None,
        description="wide_by_product: the product column names. If omitted, all non-id columns are products.",
    )
    attribute: str | None = Field(
        None,
        description=(
            "The attribute column: the attribute-names column for 'long', or the block-label column "
            "for 'wide_by_product'."
        ),
    )
    score: str | None = Field(None, description="long: the column holding the score.")
    ignore: list[str] | None = Field(
        None,
        description=(
            "Optional nuisance columns to drop before reshaping (e.g. a site or batch code). When the "
            "attribute / product list is omitted, all remaining columns excludes these."
        ),
    )


@tool_spec(
    name="sensory_reshape_to_long",
    description=(
        "Deterministically reshape parsed panel data into the descriptive_long schema (panelist_id, "
        "session, product, attribute, replicate, score). Handles already-long data and the common "
        "wide-by-attribute layout (one column per attribute). You supply an explicit column mapping; "
        "the tool melts if needed and verifies round-trip invariants (grand mean, per-attribute and "
        "per-panelist means, and cell count are identical before and after), failing if the mapping is "
        "wrong rather than silently corrupting the data. Run this before sensory_validate_descriptive. "
        "Returns: on success {ok: true, checks, long}, where 'long' is the reshaped rows in the "
        "descriptive_long schema and 'checks' holds the round-trip invariants (ok, grand_mean_before, "
        "grand_mean_after, grand_mean_diff, per_attribute_max_diff, per_panelist_max_diff, "
        "n_cells_before, n_cells_after). On a bad mapping it returns {ok: false, errors: [str]}."
    ),
    input_model=_ReshapeInput,
    category="sensory",
)
def sensory_reshape_to_long(spec: _ReshapeInput) -> dict:
    """Reshape to descriptive_long with round-trip checks; see tool spec for details."""
    mapping = {
        "panelist_id": spec.panelist_id,
        "product": spec.product,
        "session": spec.session,
        "replicate": spec.replicate,
        "attributes": spec.attributes,
        "products": spec.products,
        "attribute": spec.attribute,
        "score": spec.score,
        "ignore": spec.ignore,
    }
    try:
        long_df, checks = _reshape_to_long(pd.DataFrame(spec.data), layout=spec.layout, mapping=mapping)
    except ValueError as exc:
        return clean({"ok": False, "errors": [str(exc)]})
    return clean({"ok": True, "checks": checks, "long": long_df.to_dict(orient="records")})


class _ValidateInput(BaseModel):
    """Input contract for ``sensory_validate_descriptive``."""

    model_config = ConfigDict(extra="forbid")

    panel: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        description=(
            "Panel data as a list of row-records, each with keys panelist_id, "
            "session, product, attribute, replicate, score."
        ),
    )
    covariates: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        description=(
            "Product-covariate table as row-records, each with a 'product' key "
            "plus the design factors (designed mode) or measured descriptors "
            "(observational mode)."
        ),
    )
    mode: Literal["observational"] = Field(
        ...,
        description=(
            "Only 'observational' (measured product descriptors) is supported for now. "
            "Designed (controlled factor levels) is planned for a later release."
        ),
    )
    score_min: float | None = Field(None, description="Optional lower bound for the score scale.")
    score_max: float | None = Field(None, description="Optional upper bound for the score scale.")


@tool_spec(
    name="sensory_validate_descriptive",
    description=(
        "Validate descriptive panel data against the descriptive_long schema and a product-covariate "
        "table. Checks required columns, dtypes, score range, panel balance, and label encoding, plus "
        "mode-specific covariate checks. Run this before sensory_analyze_descriptive. "
        "Returns: {ok: bool, mode, warnings: [str], errors: [str], content_hash: str, stats: object}. "
        "'ok' gates the rest of the pipeline; 'errors' is non-empty only when ok is false, while "
        "'warnings' (for example panel-imbalance notes) can appear even when ok is true."
    ),
    input_model=_ValidateInput,
    category="sensory",
)
def sensory_validate_descriptive(spec: _ValidateInput) -> dict:
    """Validate panel + covariate inputs; see tool spec for details."""
    result = _validate_descriptive(
        pd.DataFrame(spec.panel),
        pd.DataFrame(spec.covariates),
        mode=spec.mode,
        score_min=spec.score_min,
        score_max=spec.score_max,
    )
    return clean(
        {
            "ok": result.ok,
            "mode": result.mode,
            "warnings": result.warnings,
            "errors": result.errors,
            "content_hash": result.content_hash,
            "stats": result.stats,
        }
    )


class _AnalyzeInput(BaseModel):
    """Input contract for ``sensory_analyze_descriptive``."""

    model_config = ConfigDict(extra="forbid")

    panel: list[dict[str, Any]] = Field(..., min_length=1, description="Panel data row-records (see validate).")
    covariates: list[dict[str, Any]] = Field(
        ..., min_length=1, description="Product-covariate row-records (see validate)."
    )
    mode: Literal["observational"] = Field(
        ...,
        description="Only 'observational' is supported for now; designed mode is planned for later.",
    )
    drop_flagged: bool = Field(
        False,
        description="When true, drop every panelist the scorecard flags before relating to the product.",
    )
    drop_panelists: list[str] = Field(
        default_factory=list,
        description="Explicit panelist ids to drop (used when drop_flagged is false).",
    )
    correction: Literal["none", "align", "drop"] = Field(
        "none",
        description=(
            "Panel correction before relating. 'align' applies the Mixed Assessor Model scale "
            "alignment to all panelists; 'drop' relies on drop_flagged / drop_panelists; 'none' "
            "leaves scores unchanged. Align and drop compose."
        ),
    )
    align_method: Literal["both", "location", "scale"] = Field(
        "both",
        description="Which MAM lever to apply when correction='align'.",
    )
    model: str = Field("main_effects", description="Design model for the designed relate step.")
    n_components: int = Field(2, ge=1, description="Components for the PLS relate step and PCA map.")
    conf_level: float = Field(0.95, gt=0, lt=1, description="Confidence level for product-mean intervals.")
    alpha: float = Field(0.05, gt=0, lt=1, description="Target false-discovery rate for the relate step.")
    discriminator: bool = Field(
        True,
        description=(
            "Run the cross-validated discriminator: a per-attribute out-of-sample Q-squared gate, a "
            "selectivity ratio per descriptor with a permutation test, and collinear-cluster grouping. "
            "Set false to skip it (faster)."
        ),
    )
    n_permutations: int = Field(
        199, ge=1, description="Permutations for the discriminator's selectivity-ratio null."
    )
    random_state: int = Field(0, description="Seed for the discriminator's permutations and CV folds.")
    score_min: float | None = Field(None, description="Optional lower bound for the score scale.")
    score_max: float | None = Field(None, description="Optional upper bound for the score scale.")


@tool_spec(
    name="sensory_analyze_descriptive",
    description=(
        "Run the descriptive panel pipeline: validate, score and optionally correct the panel (drop "
        "anomalous panelists and/or apply Mixed Assessor Model scale alignment), then relate each "
        "attribute to the product. Observational mode relates attributes to measured descriptors with "
        "PLS and correlations (association). Returns the panel scorecard flags, dropped panelists, the "
        "MAM scaling coefficients and product F-tests, the relate results with Benjamini-Hochberg "
        "q-values, and (unless disabled) a cross-validated discriminator: a per-attribute Q-squared "
        "gate, a selectivity ratio per descriptor with a permutation q-value, and collinear-cluster ids "
        "that mark proxies which cannot be separated from a genuine driver. Also returns product means "
        "with CIs and a PCA map. Refuses to run if validation fails. "
        "(Designed/DoE mode is planned for a later release.) "
        "Returns: on validation failure {ok: false, errors: [str], warnings: [str]}. On success "
        "{ok: true, mode, warnings, flagged, flag_reasons, scale_bands, dropped, correction, mam, relate, "
        "product_means, pca}. 'flagged'/'dropped' are panelist-id lists; 'scale_bands' is rows of "
        "panelist_id, scale_use_band, offset_band (each 'low'/'normal'/'high') for colouring a panel map; "
        "'mam' has 'scaling' and "
        "'ftests' (as in sensory_panel_check); 'product_means' is rows of product, attribute, mean, "
        "ci_low, ci_high; 'pca' has 'explained_variance' and 'scores'. 'relate' (observational) holds "
        "{mode, n_components, alpha, vip, associations, discriminator}: 'vip' is rows of descriptor, vip; "
        "'associations' is the marginal table, rows of attribute, descriptor, r, p_value, q_value, "
        "significant. 'discriminator' (present unless disabled) holds {per_attribute, descriptors, "
        "clusters, alpha, n_permutations, cluster_threshold}: 'per_attribute' is rows of attribute, "
        "n_components_cv, q2_cv, rmsep_cv, predictable; 'descriptors' is rows of attribute, descriptor, "
        "selectivity_ratio, p_value, q_value, discriminator_significant, cluster_id; 'clusters' maps "
        "each descriptor to an integer collinear-cluster id (descriptors sharing an id cannot be told "
        "apart, so a significant one may be a proxy for another in the same cluster)."
    ),
    input_model=_AnalyzeInput,
    category="sensory",
)
def sensory_analyze_descriptive(spec: _AnalyzeInput) -> dict:
    """Validate then analyse; see tool spec for details."""
    validated = _validate_descriptive(
        pd.DataFrame(spec.panel),
        pd.DataFrame(spec.covariates),
        mode=spec.mode,
        score_min=spec.score_min,
        score_max=spec.score_max,
    )
    if not validated.ok:
        return clean({"ok": False, "errors": validated.errors, "warnings": validated.warnings})

    drop: str | list[str] | None = "auto" if spec.drop_flagged else (spec.drop_panelists or None)
    result = _analyze_descriptive(
        validated,
        drop_panelists=drop,
        correction=spec.correction,
        align_method=spec.align_method,
        model=spec.model,
        n_components=spec.n_components,
        conf_level=spec.conf_level,
        alpha=spec.alpha,
        discriminator=spec.discriminator,
        n_permutations=spec.n_permutations,
        random_state=spec.random_state,
    )
    scores = result.pca["scores"].reset_index().rename(columns={"index": "product"})
    return clean(
        {
            "ok": True,
            "mode": result.mode,
            "warnings": validated.warnings,
            "flagged": result.panel.flagged,
            "flag_reasons": result.panel.reasons,
            "scale_bands": result.panel.table[["scale_use_band", "offset_band"]]
            .reset_index()
            .to_dict(orient="records"),
            "dropped": result.dropped,
            "correction": result.correction,
            "mam": {
                "scaling": result.mam.scaling.to_dict(orient="records"),
                "ftests": result.mam.ftests.to_dict(orient="records"),
            },
            "relate": result.relate,
            "product_means": result.product_means.to_dict(orient="records"),
            "pca": {
                "explained_variance": result.pca["explained_variance"],
                "scores": scores.to_dict(orient="records"),
            },
        }
    )


class _PanelCheckInput(BaseModel):
    """Input contract for ``sensory_panel_check``."""

    model_config = ConfigDict(extra="forbid")

    panel: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        description=(
            "Panel data row-records, each with keys panelist_id, session, product, attribute, "
            "replicate, score. No product-covariate table is needed for a panel check."
        ),
    )
    align: bool = Field(
        False,
        description="When true, also return the panel rescaled onto a common scale (MAM alignment).",
    )
    align_method: Literal["both", "location", "scale"] = Field(
        "both",
        description="Which MAM lever to apply when align is true.",
    )


@tool_spec(
    name="sensory_panel_check",
    description=(
        "Assess panel quality from descriptive panel data alone (no product covariates needed). "
        "Returns the per-panelist scorecard (discrimination, agreement, scale use, drift) with the "
        "flagged panelists and reasons, and the Mixed Assessor Model results: each panelist's scaling "
        "coefficient beta per attribute (beta<1 compresses the scale, >1 expands it) and the MAM versus "
        "classical product-effect F-tests. With align=true, also returns the panel rescaled onto a "
        "common scale so scale-usage differences are removed while genuine disagreement is preserved. "
        "Returns: {ok: true, scorecard, flagged, flag_reasons, mam}. 'scorecard' is one row per "
        "panelist (panelist_id, discrimination, agreement, scale_shift, scale_spread, drift, plus the "
        "two-sided outlier bands scale_use_band and offset_band, each 'low' / 'normal' / 'high'); use the "
        "bands to colour or label a panel map (low/high scale_use_band = compresses/expands the range; "
        "low/high offset_band = rates consistently low/high) instead of re-deriving thresholds, and only "
        "call out the non-normal ones. 'flagged' "
        "is the list of anomalous panelist ids and 'flag_reasons' maps each to its list of reasons; "
        "'mam' has "
        "'scaling' (rows of attribute, panelist_id, beta, offset, mean) and 'ftests' (rows of attribute, "
        "f_product_mam, p_product_mam, f_product_classical, p_product_classical, df_product, "
        "df_disagreement). With align=true an 'aligned_panel' (rescaled descriptive_long rows) is added. "
        "On missing columns it returns {ok: false, errors: [str]}."
    ),
    input_model=_PanelCheckInput,
    category="sensory",
)
def sensory_panel_check(spec: _PanelCheckInput) -> dict:
    """Panel scorecard plus Mixed Assessor Model; see tool spec for details."""
    df = pd.DataFrame(spec.panel)
    missing = [c for c in DESCRIPTIVE_LONG_COLUMNS if c not in df.columns]
    if missing:
        return clean({"ok": False, "errors": [f"Panel data is missing required columns: {missing}."]})
    for col in ("panelist_id", "product", "attribute"):
        df[col] = df[col].astype(str).str.strip()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")

    card = _panel_scorecard(df)
    mam = _mixed_assessor_model(df)
    out: dict[str, Any] = {
        "ok": True,
        "scorecard": card.table.reset_index().to_dict(orient="records"),
        "flagged": card.flagged,
        "flag_reasons": card.reasons,
        "mam": {
            "scaling": mam.scaling.to_dict(orient="records"),
            "ftests": mam.ftests.to_dict(orient="records"),
        },
    }
    if spec.align:
        out["aligned_panel"] = _align_scores(df, method=spec.align_method).to_dict(orient="records")
    return clean(out)


_register("sensory_reshape_to_long")
_register("sensory_validate_descriptive")
_register("sensory_analyze_descriptive")
_register("sensory_panel_check")


def get_sensory_tool_specs() -> list[dict]:
    """Return tool specs for all sensory tools registered in this module."""
    return get_tool_specs(names=_SENSORY_TOOL_NAMES)
