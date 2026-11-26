# (c) Kevin Dunn, 2010-2026. MIT License.
"""MCP tool wrapper: ``yates_table``, Yates's algorithm on a full two- or three-level factorial, and its reverse."""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from process_improve.experiments._tools import _TOOL_EXPECTED_EXCEPTIONS, _register, logger
from process_improve.tool_spec import clean, tool_spec

#: Columns a design carries that are not factors.
_NOT_FACTORS = frozenset({"RunOrder", "Block"})

#: The settings of each factor of a 3^k factorial.
_THREE_SETTINGS = 3


class YatesTableInput(BaseModel):
    """Input contract for ``yates_table``."""

    model_config = ConfigDict(extra="forbid")

    design_matrix: list[dict[str, Any]] = Field(
        ...,
        min_length=2,
        description=(
            "One dict per run: the factors and the response. Every factor is coded -1 and +1 (a 2^k factorial) or "
            "-1, 0 and +1 (a 3^k factorial), and each run of the full factorial appears once, in any order; average "
            "replicates first. Example: [{'T': -1, 'C': -1, 'K': -1, 'y': 60}, {'T': 1, 'C': -1, 'K': -1, 'y': 72}, "
            "...]"
        ),
    )
    response_column: str = Field(..., description="Name of the response column in design_matrix.")
    factor_names: list[str] | None = Field(
        None,
        description=(
            "The factors, the first the one that changes fastest in standard order. For a fraction, its basic "
            "factors. Default: every column except the response, RunOrder and Block."
        ),
    )
    keep: list[str] | None = Field(
        None,
        description=(
            "Two-level factorials only: the effects a model keeps, named as the table names them, e.g. "
            "['T', 'C', 'T:K']. The algorithm is then also run in reverse, giving the model's fitted values and "
            "residuals. The mean is always kept."
        ),
    )


@tool_spec(
    name="yates_table",
    description=(
        "Yates's algorithm on a full two-level (2^k) or three-level (3^k) factorial, run once each: the table a "
        "textbook prints when it works the analysis by hand (Box, Hunter and Hunter, Table 10.5; Box and Draper, "
        "Table 7G.1). A design whose every factor has three settings is worked as a 3^k. Returns 'table', one row "
        "per run in standard order: the algorithm's columns (1) to (k), the effect the row estimates ('mean', 'A', "
        "'B', 'A:B', ...; for a 3^k also 'A^2', the quadratic polynomial 3x^2 - 2), its 'estimate' and 'sum_sq'. "
        "For a 2^k the estimate is the effect (the difference of two averages), and 'sum_of_squares' holds the "
        "check on the arithmetic: each column's sum of squares is twice the one before. For a 3^k each effect has "
        "one degree of freedom; its 'divisor' z'z, 'estimate' z'y/z'z, 'standardized' component and "
        "'normal_score' (its position on a normal plot) are given too. With 'keep', a 2^k is also worked in "
        "reverse: 'reverse' holds the fitted values and residuals of the model that keeps those effects. Use it "
        "to show or check the working; for an ANOVA with p-values use analyze_experiment."
    ),
    input_model=YatesTableInput,
    examples="""
    # "Work Yates's algorithm on my 2^3 factorial"
        -> ``yates_table(design_matrix=[{"T": -1, "C": -1, "K": -1, "y": 60}, ...], response_column="y")``

    # "Fitted values of the model with T, C and the TK interaction"
        -> ``yates_table(design_matrix=[...], response_column="y", keep=["T", "C", "T:K"])``
    """,
    category="experiments",
)
def yates_table_tool(spec: YatesTableInput) -> dict[str, Any]:
    """Work Yates's algorithm, forwards and, with ``keep``, in reverse."""
    try:
        from process_improve.experiments.yates import reverse_yates, yates_table, yates_table_3k  # noqa: PLC0415

        df = pd.DataFrame(spec.design_matrix)
        if spec.response_column not in df.columns:
            raise ValueError(f"response_column {spec.response_column!r} is not a column of design_matrix.")
        names = spec.factor_names or [c for c in df.columns if c != spec.response_column and c not in _NOT_FACTORS]
        missing = [name for name in names if name not in df.columns]
        if missing:
            raise ValueError(f"factor_names {missing} are not columns of design_matrix.")
        three_level = bool(names) and all(df[name].nunique() == _THREE_SETTINGS for name in names)
        if three_level and spec.keep is not None:
            raise ValueError("keep runs the algorithm in reverse, which is for a two-level factorial only.")
        table = (yates_table_3k if three_level else yates_table)(df, spec.response_column, factors=names)
        output: dict[str, Any] = {
            "factor_names": names,
            "response_name": spec.response_column,
            "levels": 3 if three_level else 2,
            "table": table.to_dict(orient="records"),
        }
        if not three_level:
            output["sum_of_squares"] = table.attrs["sum_of_squares"]
        if spec.keep is not None:
            reverse = reverse_yates(table, spec.keep)
            output["reverse"] = {
                "table": reverse.to_dict(orient="records"),
                "sum_of_squares": reverse.attrs["sum_of_squares"],
            }
        return clean(output)
    except _TOOL_EXPECTED_EXCEPTIONS as e:
        logger.exception("Tool yates_table failed")
        return {"error": str(e)}


_register("yates_table")
