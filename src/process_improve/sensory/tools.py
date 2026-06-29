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
    layout: Literal["long", "wide_by_attribute"] = Field(
        ...,
        description=(
            "'wide_by_attribute' when there is one column per attribute (rows are panelist x product x "
            "replicate); 'long' when there is already one row per score with attribute and score columns."
        ),
    )
    panelist_id: str = Field(..., description="Name of the column holding the panelist / assessor id.")
    product: str = Field(..., description="Name of the column holding the product / sample id.")
    session: str | None = Field(None, description="Optional column holding the session; defaults to 1 if absent.")
    replicate: str | None = Field(None, description="Optional column holding the replicate; defaults to 1 if absent.")
    attributes: list[str] | None = Field(
        None,
        description="wide_by_attribute: the attribute column names. If omitted, all non-id columns are attributes.",
    )
    attribute: str | None = Field(None, description="long: the column holding the attribute names.")
    score: str | None = Field(None, description="long: the column holding the score.")


@tool_spec(
    name="sensory_reshape_to_long",
    description=(
        "Deterministically reshape parsed panel data into the descriptive_long schema (panelist_id, "
        "session, product, attribute, replicate, score). Handles already-long data and the common "
        "wide-by-attribute layout (one column per attribute). You supply an explicit column mapping; "
        "the tool melts if needed and verifies round-trip invariants (grand mean, per-attribute and "
        "per-panelist means, and cell count are identical before and after), failing if the mapping is "
        "wrong rather than silently corrupting the data. Run this before sensory_validate_descriptive."
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
        "attribute": spec.attribute,
        "score": spec.score,
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
        "mode-specific covariate checks. Returns ok/warnings/errors, a content hash, and summary counts. "
        "Run this before sensory_analyze_descriptive."
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
        "q-values, product means with CIs, and a PCA map. Refuses to run if validation fails. "
        "(Designed/DoE mode is planned for a later release.)"
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
    )
    scores = result.pca["scores"].reset_index().rename(columns={"index": "product"})
    return clean(
        {
            "ok": True,
            "mode": result.mode,
            "warnings": validated.warnings,
            "flagged": result.panel.flagged,
            "flag_reasons": result.panel.reasons,
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
        "common scale so scale-usage differences are removed while genuine disagreement is preserved."
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
