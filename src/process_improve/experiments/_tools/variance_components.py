# (c) Kevin Dunn, 2010-2026. MIT License.
"""MCP tool wrapper: ``variance_components``, how much of a result's spread each source gives (#1267)."""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from process_improve.experiments._tools import _TOOL_EXPECTED_EXCEPTIONS, _register, logger
from process_improve.tool_spec import clean, tool_spec

#: The most rows the REML fit takes from a tool call.
MAX_TOOL_ROWS = 2000


class VarianceComponentsInput(BaseModel):
    """Input contract for ``variance_components``."""

    model_config = ConfigDict(extra="forbid")

    data: list[dict[str, Any]] = Field(
        ...,
        min_length=3,
        description=(
            f"One dict per observation, at most {MAX_TOOL_ROWS}: the response and every factor named in formula "
            "and random. Example: [{'batch': 'B1', 'operator': 'Ann', 'y': 10.2}, ...]"
        ),
    )
    formula: str = Field(
        ...,
        description=(
            "The response and the fixed terms: 'y ~ 1' for the mean alone, 'y ~ operator' for a fixed factor. "
            "Factors are taken as levels, coded to sum to zero; 'a * b' expands to both and their interaction."
        ),
    )
    random: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "The random terms: a factor ('batch'), 'a:b' for the random interaction of two factors, or 'b(a)' "
            "for b nested in a (each sample within its batch, labels may repeat from one batch to the next)."
        ),
    )
    bounded: bool = Field(
        True,
        description=(
            "Keep every component at or above zero, as lme4 does. False lets one go negative, as JMP's unbounded "
            "estimates do: in a balanced design they are then the analysis-of-variance estimates."
        ),
    )
    significance_level: float = Field(0.05, gt=0.0, lt=1.0, description="For the intervals and the tests.")


@tool_spec(
    name="variance_components",
    description=(
        "Fit a linear model with random intercepts by REML and report its variance components: how much of the "
        "spread in a result comes from each random source (batch, operator, day, sample within batch) and how "
        "much is left to the measurement. Use it for a gauge study, a nested sampling plan, or any data where "
        "a factor's levels are a sample of many. Returns 'components' (each source's variance and the "
        "'residual'), 'percent' (each one's share of the total), their 'std_errors' and 'intervals', the fixed "
        "terms' 'fixed_effects' and F 'tests' (Satterthwaite's degrees of freedom), 'blups' (each random "
        "level's predicted effect), 'reml_criterion', 'r_squared' and 'r_squared_adj'."
    ),
    input_model=VarianceComponentsInput,
    examples="""
    # "How much of the spread comes from the batch, and how much from the test?"
        -> ``variance_components(data=[{"batch": "B1", "y": 10.2}, ...], formula="y ~ 1", random=["batch"])``

    # "Samples nested in batches, with the operator fixed"
        -> ``variance_components(data=[...], formula="y ~ operator", random=["batch", "sample(batch)"])``
    """,
    category="experiments",
)
def variance_components_tool(spec: VarianceComponentsInput) -> dict[str, Any]:
    """Estimate the variance components of the data."""
    try:
        from process_improve.experiments.random_effects import _group_columns, variance_components  # noqa: PLC0415

        if len(spec.data) > MAX_TOOL_ROWS:
            return {"error": f"data has {len(spec.data)} rows; the tool takes at most {MAX_TOOL_ROWS}."}
        table = pd.DataFrame(spec.data)
        missing = sorted({c for term in spec.random for c in _group_columns(term)} - set(table.columns))
        if missing:
            return {"error": f"The random terms name columns the data do not have: {missing}."}
        result = variance_components(
            table,
            spec.formula,
            spec.random,
            bounded=spec.bounded,
            significance_level=spec.significance_level,
        )
        return clean(result)
    except _TOOL_EXPECTED_EXCEPTIONS as e:
        logger.exception("Tool variance_components failed")
        return {"error": str(e)}


_register("variance_components")
