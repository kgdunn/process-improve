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
    design_type: (
        Literal[
            "full_factorial",
            "fractional_factorial",
            "plackett_burman",
            "box_behnken",
            "ccd",
            "dsd",
            "omars",
            "omars_ilp",
            "d_optimal",
            "i_optimal",
            "a_optimal",
            "e_optimal",
            "mixture",
            "taguchi",
            "supersaturated",
            "latin_hypercube",
            "maximin_lhs",
            "uniform",
            "sobol",
            "halton",
            "maximin",
        ]
        | None
    ) = Field(
        None,
        description="Design type. If omitted, auto-selected based on factors and budget.",
    )
    budget: int | None = Field(
        None,
        ge=1,
        description="Maximum number of experimental runs.",
    )
    n_center_points: int = Field(
        3,
        ge=0,
        description="Number of center point replicates (default: 3).",
    )
    n_replicates: int = Field(
        1,
        ge=1,
        description="Number of full replicates (default: 1).",
    )
    n_blocks: int | None = Field(
        None,
        ge=2,
        description=(
            "Number of blocks. A two-level factorial is blocked by confounding high-order interactions "
            "(never a main effect); other designs by exchange. Runs are randomised within blocks."
        ),
    )
    resolution: int | None = Field(
        None,
        ge=3,
        le=8,
        description="Minimum resolution for fractional factorials (3 for III, 4 for IV, 5 for V, ...).",
    )
    generators: list[str] | None = Field(
        None,
        description='Explicit fractional-factorial generators, e.g. ["D=ABC", "E=AC"].',
    )
    alpha: Literal["rotatable", "face_centered", "inscribed", "orthogonal"] | float | None = Field(
        None,
        description="Axial distance for CCD designs: a name or a positive number.",
    )
    cube: Literal["full", "fractional"] = Field(
        "full",
        description="CCD cube portion: the full 2^k factorial or a resolution V fraction.",
    )
    model_type: str = Field(
        "interactions",
        description=(
            "Model the optimal and mixture designs are built for: 'main_effects', 'interactions' or "
            "'quadratic' (mixtures also accept 'linear', 'special_cubic' and the 'scheffe_*' names)."
        ),
    )
    constraints: list[str] | None = Field(
        None,
        description=(
            'Inequalities on the factors in actual units, e.g. ["3*T + 5*D <= 600", "400 <= 3*T + 5*D"]. '
            "Enforced by the optimal designs, constrained mixtures and the sobol, halton and maximin designs."
        ),
    )
    candidates: list[dict[str, Any]] | None = Field(
        None,
        description=(
            "Settings the runs must be chosen from (optimal designs only), one record per candidate, in actual "
            "units: historical operating points, equipment settings, or blends that can be made."
        ),
    )
    fixed_runs: list[dict[str, Any]] | None = Field(
        None,
        description=(
            "Runs to keep (optimal designs only), one record per run, continuous factors in coded [-1, 1] units "
            "and categorical factors as labels. They count towards the budget."
        ),
    )
    hard_to_change: list[str] | None = Field(
        None,
        description="Names of hard-to-change factors (split-plot structure; needs the optional pyoptex package).",
    )
    random_state: int = Field(
        42,
        ge=0,
        description="Seed for the run order and any random search (default: 42).",
    )
    random_seed: int | None = Field(
        None,
        ge=0,
        description="Deprecated since 1.97.0, removed in 2.0: use random_state.",
        json_schema_extra={"deprecated": True},
    )


def _jsonable(value: Any) -> Any:  # noqa: ANN401
    """Turn design metadata into JSON-friendly values: dataclasses to dicts, frames to records."""
    import dataclasses  # noqa: PLC0415

    import pandas as pd  # noqa: PLC0415

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


@tool_spec(
    name="generate_design",
    description=(
        "Generate an experimental design matrix for a designed experiment. "
        "Supports full factorial, fractional factorial, Plackett-Burman, Box-Behnken, "
        "Central Composite (CCD), Definitive Screening (DSD), D/I/A/E-optimal, mixture, "
        "OMARS, Taguchi, supersaturated (fewer runs than factors) and space-filling designs "
        "(Latin hypercube, maximin, uniform, Sobol, Halton). Constraints in actual units, a candidate set, "
        "fixed runs and blocks are supported. "
        "Each factor needs a name and type ('continuous', 'categorical', or 'mixture'). "
        "Continuous factors require 'low' and 'high' bounds. Categorical factors require 'levels'. "
        "If design_type is not specified, one is auto-selected based on the number of factors and budget. "
        "Returns the design matrix in both coded (-1/+1) and actual units, run order, and metadata, "
        "including metadata['region'] to pass on to evaluate_design and optimize_responses."
    ),
    input_model=GenerateDesignInput,
    rng={"uses_rng": True, "seed_param": "random_state", "default_seed": 42},
    examples="""
    # "Create a 2-factor CCD for Temperature (150-200 degC) and Pressure (1-5 bar)"
        -> ``generate_design(factors=[{"name": "Temperature", "low": 150, "high": 200, "units": "degC"},
                                      {"name": "Pressure", "low": 1, "high": 5, "units": "bar"}],
                             design_type="ccd", alpha="rotatable")``

    # "Screen 7 factors with minimal runs"
        -> ``generate_design(factors=[{"name": "A", "low": -1, "high": 1}, ...7 factors...],
                             design_type="plackett_burman")``

    # "Design 12 runs for Temperature and Dose, with 3*T + 5*D at most 600"
        -> ``generate_design(factors=[{"name": "T", "low": 100, "high": 150},
                                      {"name": "D", "low": 20, "high": 60}],
                             design_type="i_optimal", budget=12, model_type="quadratic",
                             constraints=["3*T + 5*D <= 600"])``

    # "Create a 2^(5-2) fractional factorial at resolution III"
        -> ``generate_design(factors=[{"name": f, "low": -1, "high": 1} for f in "ABCDE"],
                             design_type="fractional_factorial", resolution=3)``
    """,
    category="experiments",
)
def generate_design_tool(spec: GenerateDesignInput) -> dict[str, Any]:
    """Generate an experimental design."""
    try:
        import pandas as pd  # noqa: PLC0415

        from process_improve.experiments.designs import generate_design  # noqa: PLC0415
        from process_improve.experiments.factor import Constraint, Factor  # noqa: PLC0415

        factor_objects = [Factor(**f) for f in spec.factors]
        result = generate_design(
            factors=factor_objects,
            design_type=spec.design_type,
            budget=spec.budget,
            n_center_points=spec.n_center_points,
            n_replicates=spec.n_replicates,
            n_blocks=spec.n_blocks,
            resolution=spec.resolution,
            generators=spec.generators,
            alpha=spec.alpha,
            cube=spec.cube,
            constraints=[Constraint(expression=e) for e in spec.constraints] if spec.constraints else None,
            hard_to_change=spec.hard_to_change,
            model_type=spec.model_type,
            fixed_runs=pd.DataFrame(spec.fixed_runs) if spec.fixed_runs else None,
            candidates=pd.DataFrame(spec.candidates) if spec.candidates else None,
            random_state=spec.random_seed if spec.random_seed is not None else spec.random_state,
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
        # The region (for evaluate_design and optimize_responses), backend, criterion values,
        # blocking, construction and selected candidates.
        output["metadata"] = _jsonable(result.metadata)

        return clean(output)
    except _TOOL_EXPECTED_EXCEPTIONS as e:
        logger.exception("Tool generate_design failed")
        return {"error": str(e)}


_register("generate_design")
