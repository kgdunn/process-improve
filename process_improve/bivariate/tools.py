"""(c) Kevin Dunn, 2010-2026. MIT License.

Agent-callable tool wrappers for bivariate analysis.

Pydantic input contract (ENG-04 / ENG-10): each tool pairs its
``@tool_spec`` decorator with a ``BaseModel`` carrying
``ConfigDict(extra="forbid")``; the function receives the parsed
model as its single positional argument.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from process_improve.tool_spec import clean, get_tool_specs, tool_spec

_BIVARIATE_TOOL_NAMES: list[str] = []


def _register(name: str) -> None:
    _BIVARIATE_TOOL_NAMES.append(name)


# ---------------------------------------------------------------------------
# find_elbow
# ---------------------------------------------------------------------------


class FindElbowInput(BaseModel):
    """Input contract for ``find_elbow``."""

    model_config = ConfigDict(extra="forbid")

    x: list[float] = Field(
        ...,
        min_length=6,
        description="X-axis values (e.g. component numbers, cluster counts).",
    )
    y: list[float] = Field(
        ...,
        min_length=6,
        description="Y-axis values (e.g. explained variance, SSE). Must be same length as x.",
    )


@tool_spec(
    name="find_elbow",
    description=(
        "Find the elbow (knee) point in an x-y curve - the point where the curve transitions "
        "from steep to flat (or vice versa). "
        "Common use cases: selecting the number of PCA components from a scree plot, determining "
        "optimal cluster count from an elbow plot, or finding a saturation point. "
        "Uses a robust line-fitting approach to identify where two linear regimes meet. "
        "Returns the index and coordinates of the elbow point."
    ),
    input_model=FindElbowInput,
    examples="""
    # "Where is the elbow in my scree plot? Components [1,2,3,4,5,6,7,8], variance [40,25,15,8,5,3,2,2]"
        -> ``find_elbow(x=[1,2,3,4,5,6,7,8], y=[40,25,15,8,5,3,2,2])``
    """,
    category="bivariate",
)
def find_elbow(spec: FindElbowInput) -> dict[str, Any]:
    """Find the elbow point in an x-y curve."""
    from process_improve.bivariate.methods import find_elbow_point  # noqa: PLC0415

    try:
        x_arr = np.asarray(spec.x, dtype=float)
        y_arr = np.asarray(spec.y, dtype=float)
        elbow_idx = find_elbow_point(x_arr, y_arr)
        elbow_idx = int(elbow_idx)
        return clean({
            "elbow_index": elbow_idx,
            "elbow_x": spec.x[elbow_idx],
            "elbow_y": spec.y[elbow_idx],
            "n": len(spec.x),
        })
    except (ValueError, TypeError, IndexError) as exc:
        return {"error": str(exc)}


_register("find_elbow")


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------


def get_bivariate_tool_specs() -> list[dict]:
    """Return tool specs for all bivariate tools registered in this module."""
    return get_tool_specs(names=_BIVARIATE_TOOL_NAMES)
