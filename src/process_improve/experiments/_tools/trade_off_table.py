# (c) Kevin Dunn, 2010-2026. MIT License.
"""MCP tool wrapper: ``trade_off_table`` (ENG-02)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from process_improve.experiments._tools import _TOOL_EXPECTED_EXCEPTIONS, _register, logger
from process_improve.tool_spec import clean, tool_spec

# Beyond this the exhaustive minimum-aberration search behind each cell stops
# being interactive, so the tool refuses the request at the schema layer.
_MAX_RUNS = 128
_MAX_FACTORS = 12


class TradeOffTableInput(BaseModel):
    """Input contract for ``trade_off_table``."""

    model_config = ConfigDict(extra="forbid")

    runs: list[int] = Field(
        default=[4, 8, 16, 32, 64],
        description=("Run budgets, one per row of the table. Each must be a power of two (4, 8, 16, 32, 64, 128)."),
        min_length=1,
        max_length=8,
    )
    factors: list[int] = Field(
        default=[3, 4, 5, 6, 7, 8, 9],
        description="Factor counts, one per column of the table. Each between 2 and 12.",
        min_length=1,
        max_length=11,
    )


@tool_spec(
    name="trade_off_table",
    description=(
        "Build the two-level factorial trade-off table: for each combination of run budget "
        "and number of factors, the design that fits and what it costs in aliasing. "
        "Each cell reports the design label (e.g. '2^(7-4) III'), its resolution, and the "
        "generators that build it; cells where the budget exceeds the full factorial report "
        "replication instead ('2^3 (twice)'), and impossible combinations are blank. "
        "Use this tool when the user is choosing how many experiments to run, asks how many "
        "factors they can screen on a given budget, or asks what they give up by running "
        "fewer experiments. For the full alias chains of one specific design, use "
        "evaluate_design or generate_design instead."
    ),
    input_model=TradeOffTableInput,
    examples="""
    # "I can afford 16 runs. How many factors can I screen, and what do I lose?"
        -> ``trade_off_table(runs=[16], factors=[4, 5, 6, 7, 8, 9])``

    # "Show me the standard runs-against-factors trade-off table."
        -> ``trade_off_table()``

    # "Is it worth doubling from 8 to 16 runs for my 6 factors?"
        -> ``trade_off_table(runs=[8, 16], factors=[6])``
    """,
    category="experiments",
)
def trade_off_table_tool(spec: TradeOffTableInput) -> dict[str, Any]:
    """Return the runs-against-factors trade-off table, with per-cell detail."""
    try:
        from process_improve.experiments.tradeoff import tradeoff  # noqa: PLC0415

        for n_runs in spec.runs:
            if n_runs < 2 or n_runs > _MAX_RUNS or (n_runs & (n_runs - 1)) != 0:
                raise ValueError(f"Each run budget must be a power of 2 between 2 and {_MAX_RUNS}; got {n_runs}.")
        for n_factors in spec.factors:
            if not (2 <= n_factors <= _MAX_FACTORS):
                raise ValueError(f"Each factor count must be between 2 and {_MAX_FACTORS}; got {n_factors}.")

        table: dict[str, dict[str, str]] = {}
        cells: list[dict[str, Any]] = []
        for n_runs in spec.runs:
            row: dict[str, str] = {}
            for n_factors in spec.factors:
                try:
                    result = tradeoff(runs=n_runs, factors=n_factors, display=False)
                except ValueError as exc:
                    # No such design: too many factors for the budget. The cell
                    # is blank in the table, and the reason is kept in `cells`.
                    row[str(n_factors)] = ""
                    cells.append(
                        {
                            "runs": n_runs,
                            "factors": n_factors,
                            "label": "",
                            "exists": False,
                            "reason": str(exc),
                        }
                    )
                    continue
                row[str(n_factors)] = result.label
                cells.append(
                    {
                        "runs": n_runs,
                        "factors": n_factors,
                        "label": result.label,
                        "exists": True,
                        "resolution": result.resolution,
                        "roman": result.roman,
                        "n_generators": result.n_generators,
                        "generators": result.generators,
                        "replicates": result.replicates,
                    }
                )
            table[str(n_runs)] = row

        return clean(
            {
                "table": table,
                "cells": cells,
                "reading_guide": (
                    "Rows are run budgets, columns are factor counts. Going down a column costs "
                    "more experiments but buys resolution; going across a row studies more factors "
                    "for the same money, at the cost of heavier aliasing. Resolution III aliases "
                    "main effects with two-factor interactions; resolution IV keeps main effects "
                    "clear of them; resolution V separates two-factor interactions from each other."
                ),
            }
        )
    except _TOOL_EXPECTED_EXCEPTIONS as e:
        logger.exception("Tool trade_off_table failed")
        return {"error": str(e)}


_register("trade_off_table")
