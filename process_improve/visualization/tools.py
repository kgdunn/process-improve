"""(c) Kevin Dunn, 2010-2026. MIT License.

Agent-callable visualization tools.

Pydantic input contract (ENG-04 / ENG-10): each tool pairs its
``@tool_spec`` decorator with a ``BaseModel`` carrying
``ConfigDict(extra="forbid")``; the function receives the parsed
model as its single positional argument.

Each function returns a JSON-serialisable dict that matches the
shape of ``visualize_doe`` (``plot_type``, ``title``, ``data``,
``plotly``, ``echarts``), so frontends can render them without
dispatch.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from process_improve.tool_spec import clean, tool_spec
from process_improve.visualization.charts.boxplot import BoxPlot, BoxStats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _box_from_values(
    group: str,
    values: np.ndarray,
    *,
    id_series: pd.Series | None,
    original_index: np.ndarray,
) -> BoxStats:
    """Compute the five-number summary + outliers for one group.

    Whiskers use the 1.5*IQR convention: the lower whisker is the
    smallest value >= Q1 - 1.5*IQR, the upper whisker is the largest
    value <= Q3 + 1.5*IQR; anything beyond is an outlier.
    """
    arr = np.asarray(values, dtype=float)
    q1, median, q3 = (float(x) for x in np.percentile(arr, [25, 50, 75]))
    iqr = q3 - q1
    lo_fence = q1 - 1.5 * iqr
    hi_fence = q3 + 1.5 * iqr

    inlier_mask = (arr >= lo_fence) & (arr <= hi_fence)
    inliers = arr[inlier_mask]
    lower = float(inliers.min()) if inliers.size else float(arr.min())
    upper = float(inliers.max()) if inliers.size else float(arr.max())

    outlier_rows: list[dict[str, Any]] = []
    for is_in, val, idx in zip(inlier_mask, arr, original_index, strict=True):
        if is_in:
            continue
        row: dict[str, Any] = {"value": float(val)}
        if id_series is not None:
            row["id"] = str(id_series.loc[idx])
        outlier_rows.append(row)

    return BoxStats(
        group=group,
        q_stats=[lower, q1, median, q3, upper],
        outliers=outlier_rows,
    )


def _collect_boxes(
    df: pd.DataFrame,
    *,
    value_columns: list[str],
    group_by: str | None,
    id_column: str | None,
) -> list[BoxStats]:
    """Build the per-group :class:`BoxStats` list.

    - With ``group_by``: the *first* entry of ``value_columns`` is the
      response; one box per unique group value (insertion-ordered).
    - Without ``group_by``: one box per column in ``value_columns``.
    """
    id_series = df[id_column] if id_column else None

    if group_by is not None:
        response = value_columns[0]
        boxes: list[BoxStats] = []
        for group_value, sub in df.groupby(group_by, sort=False):
            vals = sub[response].to_numpy(dtype=float)
            if vals.size == 0:
                continue
            boxes.append(
                _box_from_values(
                    group=str(group_value),
                    values=vals,
                    id_series=id_series,
                    original_index=sub.index.to_numpy(),
                ),
            )
        return boxes

    return [
        _box_from_values(
            group=col,
            values=df[col].to_numpy(dtype=float),
            id_series=id_series,
            original_index=df.index.to_numpy(),
        )
        for col in value_columns
    ]


# ---------------------------------------------------------------------------
# Tool: boxplot
# ---------------------------------------------------------------------------


class BoxplotInput(BaseModel):
    """Input contract for ``boxplot``."""

    model_config = ConfigDict(extra="forbid")

    data: list[dict[str, Any]] = Field(
        ...,
        description="Tabular data as a list of record dicts (one per observation).",
    )
    value_columns: list[str] = Field(
        ...,
        description=(
            "Numeric columns to summarise. Without group_by, each column "
            "becomes a separate box. With group_by, only the first column "
            "is used and one box is drawn per unique group value."
        ),
    )
    group_by: str | None = Field(
        None,
        description=(
            "Optional column name to group by. When provided, the first "
            "entry of value_columns is plotted per unique group value."
        ),
    )
    id_column: str | None = Field(
        None,
        description=(
            "Optional stable observation id column. When set together with "
            "link_group, outlier points carry the id so a brush selection "
            "can be relayed to other charts in the link group."
        ),
    )
    show_points: bool = Field(
        True,
        description="Overlay outlier points on the boxes.",
    )
    link_group: str | None = Field(
        None,
        description=(
            "Cross-chart linking key. Charts sharing this key form a "
            "brushing group on the frontend."
        ),
    )
    title: str = Field(
        "",
        description="Chart title.",
    )
    backend: Literal["both", "plotly", "echarts"] = Field(
        "both",
        description="Which rendering backend(s) to include in output.",
    )


@tool_spec(
    name="boxplot",
    description=(
        "Generate a boxplot (optionally grouped) for one or more numeric columns of a "
        "tabular dataset. Use this to inspect distributions, compare groups, and spot "
        "outliers (values beyond 1.5*IQR). Returns the five-number summary and a "
        "ready-to-render ECharts option plus an optional Plotly figure dict. "
        "If a stable per-row id column exists, pass it as id_column and set link_group "
        "to enable brushing across linked charts."
    ),
    input_model=BoxplotInput,
    examples="""
    # "Show a boxplot of tensile_strength by lot"
        -> ``boxplot(data=[...], value_columns=["tensile_strength"], group_by="lot")``

    # "Compare the distributions of height, weight, and age"
        -> ``boxplot(data=[...], value_columns=["height", "weight", "age"])``
    """,
    category="visualization",
)
def boxplot(spec: BoxplotInput) -> dict[str, Any]:  # noqa: PLR0911
    """Generate a boxplot."""
    try:
        if not spec.data:
            return {"error": "data is empty"}
        if not spec.value_columns:
            return {"error": "value_columns is empty"}

        df = pd.DataFrame(spec.data)

        missing = [c for c in spec.value_columns if c not in df.columns]
        if missing:
            return {"error": f"value_columns not found in data: {missing}"}
        if spec.group_by is not None and spec.group_by not in df.columns:
            return {"error": f"group_by column not found in data: {spec.group_by!r}"}
        if spec.id_column is not None and spec.id_column not in df.columns:
            return {"error": f"id_column not found in data: {spec.id_column!r}"}

        boxes = _collect_boxes(
            df,
            value_columns=spec.value_columns,
            group_by=spec.group_by,
            id_column=spec.id_column,
        )
        if not boxes:
            return {"error": "no non-empty groups to plot"}

        default_title = spec.title
        if not default_title:
            default_title = (
                f"{spec.value_columns[0]} by {spec.group_by}"
                if spec.group_by is not None
                else ", ".join(spec.value_columns)
            )

        y_title = spec.value_columns[0] if spec.group_by is not None else "value"
        x_title = spec.group_by or "variable"

        chart = BoxPlot(
            boxes=boxes,
            title=default_title,
            x_title=x_title,
            y_title=y_title,
            show_points=spec.show_points,
            link_group=spec.link_group,
        )

        chart_spec = chart.to_spec()

        result: dict[str, Any] = {
            "plot_type": "boxplot",
            "title": default_title,
            "data": {
                "quartiles": [
                    {"group": b.group, "q_stats": list(b.q_stats)} for b in boxes
                ],
                "outliers": [
                    {"group": b.group, **o} for b in boxes for o in b.outliers
                ],
            },
            "link_group": spec.link_group,
            "point_ids": chart_spec.point_ids or [],
        }

        result["plotly"] = chart.to_plotly() if spec.backend in ("both", "plotly") else None
        result["echarts"] = chart.to_echarts() if spec.backend in ("both", "echarts") else None

        return clean(result)
    except (ValueError, TypeError, KeyError) as e:
        logger.exception("Tool boxplot failed")
        return {"error": str(e)}
