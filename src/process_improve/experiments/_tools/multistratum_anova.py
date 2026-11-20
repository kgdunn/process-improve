# (c) Kevin Dunn, 2010-2026. MIT License.
"""MCP tool wrapper: ``multistratum_anova``, the analysis of variance of runs made in groups (#1267)."""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from process_improve.experiments._tools import _TOOL_EXPECTED_EXCEPTIONS, _register, logger
from process_improve.tool_spec import clean, tool_spec

#: The most runs the analysis takes from a tool call.
MAX_TOOL_ROWS = 2000


class MultistratumAnovaInput(BaseModel):
    """Input contract for ``multistratum_anova``."""

    model_config = ConfigDict(extra="forbid")

    data: list[dict[str, Any]] = Field(
        ...,
        min_length=2,
        description=(
            f"The runs, at most {MAX_TOOL_ROWS}: the treatment factors, the columns that name the units "
            "(replicate, whole plot, batch), and the response if there is one. Every factor is taken as levels."
        ),
    )
    model: str = Field(..., description="The treatment terms, a plain formula over the columns: 'S * T * A'.")
    strata: str | list[str] = Field(
        ...,
        description=(
            "The block structure as error terms: 'rep/plot' for whole plots within replicates, 'S:T' for a whole "
            "plot at each combination of S and T, 'row + column' for a strip plot."
        ),
    )
    response_column: str | None = Field(
        None,
        description="The response; without one the result is the skeleton: each stratum's degrees of freedom.",
    )


@tool_spec(
    name="multistratum_anova",
    description=(
        "The analysis of variance of a design whose runs are made in groups of several sizes (split-plot, "
        "strip-plot, split-split-plot, runs in blocks), each term tested in the stratum where its contrasts "
        "fall, against that stratum's own error. Without a response it returns the skeleton analysis of "
        "variance: which terms each stratum estimates and on how many degrees of freedom, so a design can be "
        "judged before it is run. Returns 'strata' (each error term, then 'Within', with its degrees of "
        "freedom, its units, its terms and its residual; with a response each term's ss, ms, f and p_value), "
        "'df' (each stratum's terms mapped to their degrees of freedom) and 'table' (every row)."
    ),
    input_model=MultistratumAnovaInput,
    examples="""
    # "How many degrees of freedom does each whole-plot term get in this split-plot design?"
        -> ``multistratum_anova(data=[{"rep": 1, "plot": 1, "M": "a", "T": 200}, ...],
                model="M * T", strata="rep/plot")``

    # "Test my strip-plot experiment's terms"
        -> ``multistratum_anova(data=[...], model="A * B", strata="row + column", response_column="y")``
    """,
    category="experiments",
)
def multistratum_anova_tool(spec: MultistratumAnovaInput) -> dict[str, Any]:
    """Return the multistratum analysis of variance, or its skeleton."""
    try:
        from process_improve.experiments.multistratum import multistratum_anova  # noqa: PLC0415

        if len(spec.data) > MAX_TOOL_ROWS:
            return {"error": f"data has {len(spec.data)} runs; the tool takes at most {MAX_TOOL_ROWS}."}
        result = multistratum_anova(pd.DataFrame(spec.data), spec.model, spec.strata, spec.response_column)
        return clean({"strata": result["strata"], "df": result["df"], "table": result["table"].to_dict("records")})
    except _TOOL_EXPECTED_EXCEPTIONS as e:
        logger.exception("Tool multistratum_anova failed")
        return {"error": str(e)}


_register("multistratum_anova")
