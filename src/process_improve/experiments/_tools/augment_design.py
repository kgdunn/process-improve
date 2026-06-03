# (c) Kevin Dunn, 2010-2026. MIT License.
"""MCP tool wrapper: ``augment_design`` (ENG-02)."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from process_improve.experiments._tools import _TOOL_EXPECTED_EXCEPTIONS, _register, logger
from process_improve.tool_spec import clean, tool_spec


class AugmentDesignInput(BaseModel):
    """Input contract for ``augment_design``."""

    model_config = ConfigDict(extra="forbid")

    existing_design: list[dict[str, Any]] = Field(
        ...,
        min_length=2,
        description=(
            "Current design matrix as list of dicts with factor names as keys "
            "and coded values (-1/+1) as values. "
            "Example: [{'A': -1, 'B': -1}, {'A': 1, 'B': -1}, ...]"
        ),
    )
    augmentation_type: Literal[
        "foldover",
        "semifold",
        "add_center_points",
        "add_axial_points",
        "add_runs_optimal",
        "upgrade_to_rsm",
        "add_blocks",
        "replicate",
    ] = Field(
        ...,
        description="Type of augmentation to apply to the design.",
    )
    target_model: Literal["main_effects", "interactions", "quadratic"] | None = Field(
        None,
        description=(
            "Desired model after augmentation. Used by 'add_runs_optimal' "
            "and 'upgrade_to_rsm'. Default: 'interactions'."
        ),
    )
    n_additional_runs: int | None = Field(
        None,
        ge=1,
        description=(
            "Budget for additional runs. Interpretation depends on type: "
            "number of center points, D-optimal runs, replicates, or blocks."
        ),
    )
    fold_on: str | None = Field(
        None,
        description=(
            "Factor name to fold on (semifold only). "
            "If omitted, the best factor is auto-selected."
        ),
    )
    alpha: str | float | None = Field(
        None,
        description=(
            "Axial distance for add_axial_points or upgrade_to_rsm. "
            "'rotatable', 'face_centered', 'orthogonal', or a numeric value."
        ),
    )
    generators: list[str] | None = Field(
        None,
        description=(
            "Generator strings from the original fractional factorial design "
            "(e.g. ['D=ABC']). Needed for foldover/semifold alias analysis."
        ),
    )


@tool_spec(
    name="augment_design",
    description=(
        "Extend or modify an existing experimental design. Supports foldover (de-alias all "
        "2-factor interactions), semifold (de-alias specific interactions with fewer runs), "
        "adding center points (test for curvature), adding axial/star points (upgrade to CCD "
        "for response surface modeling), D-optimal augmentation (add runs to maximize information), "
        "upgrade to RSM (convert screening design to response surface design), add blocks "
        "(retroactively confound block effects with high-order interactions), and replication "
        "(improve precision estimates). "
        "Always returns the augmented design matrix plus an explanation of what changed in the "
        "alias structure and design properties."
    ),
    input_model=AugmentDesignInput,
    examples="""
    # "Fold over my 2^(4-1) design to de-alias two-factor interactions"
        -> ``augment_design(existing_design=[...], augmentation_type="foldover",
                generators=["D=ABC"])``

    # "Add 5 center points to test for curvature"
        -> ``augment_design(existing_design=[...], augmentation_type="add_center_points",
                n_additional_runs=5)``

    # "Upgrade my screening design to a CCD for response surface modeling"
        -> ``augment_design(existing_design=[...], augmentation_type="upgrade_to_rsm",
                alpha="rotatable", target_model="quadratic")``

    # "Add 6 D-optimal runs to improve my design"
        -> ``augment_design(existing_design=[...], augmentation_type="add_runs_optimal",
                n_additional_runs=6, target_model="interactions")``
    """,
    category="experiments",
)
def augment_design_tool(spec: AugmentDesignInput) -> dict[str, Any]:
    """Augment an existing design."""
    try:
        from process_improve.experiments.augment import augment_design  # noqa: PLC0415

        df = pd.DataFrame(spec.existing_design)
        result = augment_design(
            existing_design=df,
            augmentation_type=spec.augmentation_type,
            target_model=spec.target_model,
            n_additional_runs=spec.n_additional_runs,
            fold_on=spec.fold_on,
            alpha=spec.alpha,
            generators=spec.generators,
        )
        return clean(result)
    except _TOOL_EXPECTED_EXCEPTIONS as e:
        logger.exception("Tool augment_design failed")
        return {"error": str(e)}


_register("augment_design")
