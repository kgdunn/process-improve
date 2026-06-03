# (c) Kevin Dunn, 2010-2026. MIT License.
"""MCP tool wrapper: ``create_factorial_design`` (ENG-02)."""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from process_improve.experiments._tools import _TOOL_EXPECTED_EXCEPTIONS, _register, logger
from process_improve.tool_spec import clean, tool_spec


class CreateFactorialDesignInput(BaseModel):
    """Input contract for ``create_factorial_design``."""

    model_config = ConfigDict(extra="forbid")

    n_factors: int = Field(
        ...,
        ge=2,
        le=10,
        description="Number of factors in the design (2 to 10).",
    )
    factor_names: list[str] | None = Field(
        None,
        description=(
            "Optional list of factor names. Must have exactly n_factors entries. "
            "If not provided, factors are named A, B, C, ..."
        ),
    )


@tool_spec(
    name="create_factorial_design",
    description=(
        "Create a full factorial (2^k) experimental design with factors coded as -1 and +1. "
        "Returns the design matrix as a list of dictionaries (one per run), the number of runs, "
        "the number of factors, and the factor names. "
        "Use this when planning a designed experiment to systematically explore the effect of "
        "2 to 10 factors, each at two levels."
    ),
    input_model=CreateFactorialDesignInput,
    examples="""
    # "Create a 2-factor full factorial design"
        -> ``create_factorial_design(n_factors=2)``

    # "Create a 3-factor design with named factors"
        -> ``create_factorial_design(n_factors=3, factor_names=["Temperature", "Pressure", "Time"])``
    """,
    category="experiments",
)
def create_factorial_design(spec: CreateFactorialDesignInput) -> dict[str, Any]:
    """Create a full factorial design."""
    try:
        from process_improve.experiments.designs_factorial import full_factorial  # noqa: PLC0415

        columns = full_factorial(spec.n_factors, names=spec.factor_names)
        design = pd.concat(columns, axis=1)
        names = list(design.columns)
        return clean({
            "design": design.to_dict(orient="records"),
            "n_runs": len(design),
            "n_factors": spec.n_factors,
            "factor_names": names,
        })
    except _TOOL_EXPECTED_EXCEPTIONS as e:
        logger.exception("Tool create_factorial_design failed")
        return {"error": str(e)}


_register("create_factorial_design")
