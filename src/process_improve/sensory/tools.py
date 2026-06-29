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
from process_improve.sensory.validation import validate_descriptive as _validate_descriptive
from process_improve.tool_spec import clean, get_tool_specs, tool_spec

_SENSORY_TOOL_NAMES: list[str] = []


def _register(name: str) -> None:
    _SENSORY_TOOL_NAMES.append(name)


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
    mode: Literal["designed", "observational"] = Field(
        ...,
        description="'designed' for controlled factor levels; 'observational' for measured descriptors.",
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
    mode: Literal["designed", "observational"] = Field(..., description="Covariate-table interpretation.")
    drop_flagged: bool = Field(
        False,
        description="When true, drop every panelist the scorecard flags before relating to the product.",
    )
    drop_panelists: list[str] = Field(
        default_factory=list,
        description="Explicit panelist ids to drop (used when drop_flagged is false).",
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
        "Run the descriptive panel pipeline: validate, score and optionally drop anomalous panelists, "
        "then relate each attribute to the product. Designed mode regresses attributes on the design "
        "factors (effects); observational mode relates them to measured descriptors with PLS and "
        "correlations (association). Returns the panel scorecard flags, dropped panelists, the relate "
        "results with Benjamini-Hochberg q-values, product means with CIs, and a PCA map. Refuses to run "
        "if validation fails."
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
            "relate": result.relate,
            "product_means": result.product_means.to_dict(orient="records"),
            "pca": {
                "explained_variance": result.pca["explained_variance"],
                "scores": scores.to_dict(orient="records"),
            },
        }
    )


_register("sensory_validate_descriptive")
_register("sensory_analyze_descriptive")


def get_sensory_tool_specs() -> list[dict]:
    """Return tool specs for all sensory tools registered in this module."""
    return get_tool_specs(names=_SENSORY_TOOL_NAMES)
