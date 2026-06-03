# (c) Kevin Dunn, 2010-2026. MIT License.
"""MCP tool wrapper: ``generate_design`` (ENG-02)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from process_improve.experiments._tools import _TOOL_EXPECTED_EXCEPTIONS, _register, logger
from process_improve.tool_spec import clean, tool_spec


class GenerateDesignInput(BaseModel):
    """Input contract for ``generate_design``."""

    model_config = ConfigDict(extra="forbid")

    factors: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        description="List of factor specifications (name, type, low/high or levels, units).",
    )
    design_type: Literal[
        "full_factorial",
        "fractional_factorial",
        "plackett_burman",
        "box_behnken",
        "ccd",
        "dsd",
        "d_optimal",
        "i_optimal",
        "a_optimal",
        "mixture",
        "taguchi",
    ] | None = Field(
        None,
        description="Design type. If omitted, auto-selected based on factors and budget.",
    )
    budget: int | None = Field(
        None,
        ge=1,
        description="Maximum number of experimental runs.",
    )
    center_points: int = Field(
        3,
        ge=0,
        description="Number of center point replicates (default: 3).",
    )
    replicates: int = Field(
        1,
        ge=1,
        description="Number of full replicates (default: 1).",
    )
    resolution: int | None = Field(
        None,
        ge=3,
        le=5,
        description="Minimum resolution for fractional factorials (3, 4, or 5).",
    )
    alpha: Literal["rotatable", "face_centered", "orthogonal"] | None = Field(
        None,
        description="Axial distance for CCD designs.",
    )
    random_seed: int = Field(
        42,
        description="Seed for reproducible randomization (default: 42).",
    )


@tool_spec(
    name="generate_design",
    description=(
        "Generate an experimental design matrix for a designed experiment. "
        "Supports full factorial, fractional factorial, Plackett-Burman, Box-Behnken, "
        "Central Composite (CCD), Definitive Screening (DSD), D-optimal, mixture, "
        "and Taguchi designs. "
        "Each factor needs a name and type ('continuous', 'categorical', or 'mixture'). "
        "Continuous factors require 'low' and 'high' bounds. Categorical factors require 'levels'. "
        "If design_type is not specified, one is auto-selected based on the number of factors and budget. "
        "Returns the design matrix in both coded (-1/+1) and actual units, run order, and metadata."
    ),
    input_model=GenerateDesignInput,
    examples="""
    # "Create a 2-factor CCD for Temperature (150-200 degC) and Pressure (1-5 bar)"
        -> ``generate_design(factors=[{"name": "Temperature", "low": 150, "high": 200, "units": "degC"},
                                      {"name": "Pressure", "low": 1, "high": 5, "units": "bar"}],
                             design_type="ccd", alpha="rotatable")``

    # "Screen 7 factors with minimal runs"
        -> ``generate_design(factors=[{"name": "A", "low": -1, "high": 1}, ...7 factors...],
                             design_type="plackett_burman")``

    # "Create a 2^(5-2) fractional factorial at resolution III"
        -> ``generate_design(factors=[{"name": f, "low": -1, "high": 1} for f in "ABCDE"],
                             design_type="fractional_factorial", resolution=3)``
    """,
    category="experiments",
)
def generate_design_tool(spec: GenerateDesignInput) -> dict[str, Any]:
    """Generate an experimental design."""
    try:
        from process_improve.experiments.designs import generate_design  # noqa: PLC0415
        from process_improve.experiments.factor import Factor  # noqa: PLC0415

        factor_objects = [Factor(**f) for f in spec.factors]

        result = generate_design(
            factors=factor_objects,
            design_type=spec.design_type,
            budget=spec.budget,
            center_points=spec.center_points,
            replicates=spec.replicates,
            resolution=spec.resolution,
            alpha=spec.alpha,
            random_seed=spec.random_seed,
        )

        design_coded = result.design.drop(columns=["RunOrder"], errors="ignore")
        design_actual = result.design_actual.drop(columns=["RunOrder"], errors="ignore")

        output: dict[str, Any] = {
            "design_coded": design_coded.to_dict(orient="records"),
            "design_actual": design_actual.to_dict(orient="records"),
            "run_order": result.run_order,
            "design_type": result.design_type,
            "n_runs": result.n_runs,
            "n_factors": result.n_factors,
            "factor_names": result.factor_names,
        }
        if result.generators:
            output["generators"] = result.generators
        if result.defining_relation:
            output["defining_relation"] = result.defining_relation
        if result.resolution is not None:
            output["resolution"] = result.resolution
        if result.alpha is not None:
            output["alpha"] = result.alpha

        return clean(output)
    except _TOOL_EXPECTED_EXCEPTIONS as e:
        logger.exception("Tool generate_design failed")
        return {"error": str(e)}


_register("generate_design")
