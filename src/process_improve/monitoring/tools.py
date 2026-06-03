"""(c) Kevin Dunn, 2010-2026. MIT License.

Agent-callable tool wrappers for process monitoring.

Pydantic input contract (ENG-04 / ENG-10): each tool pairs its
``@tool_spec`` decorator with a ``BaseModel`` carrying
``ConfigDict(extra="forbid")``; the function receives the parsed
model as its single positional argument.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from process_improve.tool_spec import clean, get_tool_specs, tool_spec

_MONITORING_TOOL_NAMES: list[str] = []


def _register(name: str) -> None:
    _MONITORING_TOOL_NAMES.append(name)


# ---------------------------------------------------------------------------
# control_chart
# ---------------------------------------------------------------------------


class ControlChartInput(BaseModel):
    """Input contract for ``control_chart``."""

    model_config = ConfigDict(extra="forbid")

    values: list[float] = Field(
        ...,
        min_length=5,
        description="Time-ordered sequence of numeric observations.",
    )
    chart_type: Literal["shewhart", "cusum", "holt_winters"] = Field(
        "holt_winters",
        description=(
            "Type of control chart. 'shewhart': individual observations chart. "
            "'cusum': cumulative sum chart. 'holt_winters' (default): a blend of "
            "Shewhart and CUSUM properties."
        ),
    )
    style: Literal["robust", "regular"] = Field(
        "robust",
        description="'robust' (default): uses median/MAD. 'regular': uses mean/std.",
    )


@tool_spec(
    name="control_chart",
    description=(
        "Build a control chart for a sequence of numeric observations and identify out-of-control points. "
        "Supports Shewhart (xbar), CUSUM, and Holt-Winters (HW) chart types. "
        "The robust style (default) uses median and MAD instead of mean and std, making it resistant "
        "to outliers in the Phase-I data. "
        "Returns the calculated target (center line), upper and lower control limits, and indices of "
        "any out-of-control observations."
    ),
    input_model=ControlChartInput,
    examples="""
    # "Are any of these measurements out of control? [10.1, 10.3, 10.2, 10.0, 15.5, 10.1]"
        -> ``control_chart(values=[10.1, 10.3, 10.2, 10.0, 15.5, 10.1])``

    # "Build a Shewhart chart for my process data"
        -> ``control_chart(values=[...], chart_type="shewhart")``
    """,
    category="monitoring",
)
def control_chart(spec: ControlChartInput) -> dict[str, Any]:
    """Build a control chart and return limits + out-of-control points."""
    from process_improve.monitoring.control_charts import ControlChart  # noqa: PLC0415

    variant_map = {
        "shewhart": "xbar.no.subgroup",
        "cusum": "cusum",
        "holt_winters": "hw",
    }
    variant = variant_map.get(spec.chart_type, "hw")

    try:
        cc = ControlChart(style=spec.style, variant=variant)
        y = pd.Series(np.asarray(spec.values, dtype=float))
        cc.calculate_limits(y)

        target = cc.target
        s = cc.s
        ucl = target + 3 * s if s is not None and target is not None else None
        lcl = target - 3 * s if s is not None and target is not None else None
        ooc_indices = list(cc.idx_outside_3S) if cc.idx_outside_3S else []
        ooc_values = [float(spec.values[i]) for i in ooc_indices if i < len(spec.values)]

        return clean({
            "target": target,
            "upper_control_limit": ucl,
            "lower_control_limit": lcl,
            "spread": s,
            "out_of_control_indices": ooc_indices,
            "out_of_control_values": ooc_values,
            "n_out_of_control": len(ooc_indices),
            "n_observations": len(spec.values),
            "chart_type": spec.chart_type,
            "style": spec.style,
        })
    except (ValueError, TypeError, KeyError) as exc:
        return {"error": str(exc)}


_register("control_chart")


# ---------------------------------------------------------------------------
# process_capability
# ---------------------------------------------------------------------------


class ProcessCapabilityInput(BaseModel):
    """Input contract for ``process_capability``."""

    model_config = ConfigDict(extra="forbid")

    values: list[float] = Field(
        ...,
        min_length=5,
        description="Process measurement values.",
    )
    lower_spec: float | None = Field(
        None,
        description="Lower specification limit. Omit if there is no lower limit.",
    )
    upper_spec: float | None = Field(
        None,
        description="Upper specification limit. Omit if there is no upper limit.",
    )
    robust: bool = Field(
        True,
        description=(
            "If true (default), use robust statistics (median/Sn) for the calculation. "
            "Set to false for classical mean/std calculation."
        ),
    )


@tool_spec(
    name="process_capability",
    description=(
        "Calculate the process capability index (Cpk) for a set of measurements against "
        "specification limits. Cpk measures how well a process fits within its specification "
        "limits, accounting for process centering. "
        "A Cpk >= 1.33 is generally considered capable; Cpk >= 1.67 is excellent. "
        "Cpk < 1.0 means the process is producing out-of-spec output. "
        "Provide at least one of lower_spec or upper_spec."
    ),
    input_model=ProcessCapabilityInput,
    examples="""
    # "What is the Cpk for my data [10.1, 10.2, 9.9, 10.0] with spec limits 9.5 to 10.5?"
        -> ``process_capability(values=[10.1, 10.2, 9.9, 10.0], lower_spec=9.5, upper_spec=10.5)``
    """,
    category="monitoring",
)
def process_capability(spec: ProcessCapabilityInput) -> dict[str, Any]:
    """Calculate Cpk process capability index."""
    from process_improve.monitoring.metrics import calculate_cpk  # noqa: PLC0415

    try:
        df = pd.DataFrame({"value": np.asarray(spec.values, dtype=float)})
        trim = 2.5 if spec.robust else 0.0
        specs = (
            spec.lower_spec if spec.lower_spec is not None else np.nan,
            spec.upper_spec if spec.upper_spec is not None else np.nan,
        )

        cpk_result = calculate_cpk(df, which_column="value", specifications=specs, trim_percentile=trim)
        cpk_float = float(cpk_result.cpk)

        if cpk_float >= 1.67:
            interpretation = f"Cpk = {cpk_float:.3f}. Excellent capability - process is well within spec limits."
        elif cpk_float >= 1.33:
            interpretation = f"Cpk = {cpk_float:.3f}. Good capability - process fits within spec limits."
        elif cpk_float >= 1.0:
            interpretation = f"Cpk = {cpk_float:.3f}. Marginal capability - process barely fits within spec limits."
        else:
            interpretation = f"Cpk = {cpk_float:.3f}. Poor capability - process is producing out-of-spec output."

        return clean({
            "cpk": cpk_float,
            "rsd": float(cpk_result.rsd),
            "interpretation": interpretation,
            "lower_spec": spec.lower_spec,
            "upper_spec": spec.upper_spec,
            "n": len(spec.values),
            "robust": spec.robust,
        })
    except (ValueError, TypeError, KeyError) as exc:
        return {"error": str(exc)}


_register("process_capability")


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------


def get_monitoring_tool_specs() -> list[dict]:
    """Return tool specs for all monitoring tools registered in this module."""
    return get_tool_specs(names=_MONITORING_TOOL_NAMES)
