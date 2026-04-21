"""(c) Kevin Dunn, 2010-2026. MIT License.

Agent-callable visualization tools.

Each function in this module is decorated with ``@tool_spec`` so it can be
passed directly to an LLM tool-use API.  The wrappers accept plain
JSON-serialisable inputs (lists of dicts, strings, numbers) and always
return JSON-serialisable dict results that match the shape of
``visualize_doe`` (``plot_type``, ``title``, ``data``, ``plotly``,
``echarts``), so frontends can render them without dispatch.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

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
    input_schema={
        "json": {
            "type": "object",
            "properties": {
                "data": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": (
                        "Tabular data as a list of record dicts (one per observation)."
                    ),
                },
                "value_columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Numeric columns to summarise. Without group_by, each column "
                        "becomes a separate box. With group_by, only the first column "
                        "is used and one box is drawn per unique group value."
                    ),
                },
                "group_by": {
                    "type": "string",
                    "description": (
                        "Optional column name to group by. When provided, the first "
                        "entry of value_columns is plotted per unique group value."
                    ),
                },
                "id_column": {
                    "type": "string",
                    "description": (
                        "Optional stable observation id column. When set together with "
                        "link_group, outlier points carry the id so a brush selection "
                        "can be relayed to other charts in the link group."
                    ),
                },
                "show_points": {
                    "type": "boolean",
                    "description": "Overlay outlier points on the boxes.",
                    "default": True,
                },
                "link_group": {
                    "type": "string",
                    "description": (
                        "Cross-chart linking key. Charts sharing this key form a "
                        "brushing group on the frontend."
                    ),
                },
                "title": {
                    "type": "string",
                    "description": "Chart title.",
                },
                "backend": {
                    "type": "string",
                    "enum": ["both", "plotly", "echarts"],
                    "description": "Which rendering backend(s) to include in output.",
                    "default": "both",
                },
            },
            "required": ["data", "value_columns"],
        },
    },
    examples="""
    # "Show a boxplot of tensile_strength by lot"
        -> ``boxplot(data=[...], value_columns=["tensile_strength"], group_by="lot")``

    # "Compare the distributions of height, weight, and age"
        -> ``boxplot(data=[...], value_columns=["height", "weight", "age"])``
    """,
    category="visualization",
)
def boxplot(  # noqa: PLR0911, PLR0913
    *,
    data: list[dict[str, Any]],
    value_columns: list[str],
    group_by: str | None = None,
    id_column: str | None = None,
    show_points: bool = True,
    link_group: str | None = None,
    title: str = "",
    backend: str = "both",
) -> dict[str, Any]:
    """Generate a boxplot; see tool spec for details."""
    try:
        if not data:
            return {"error": "data is empty"}
        if not value_columns:
            return {"error": "value_columns is empty"}

        df = pd.DataFrame(data)

        missing = [c for c in value_columns if c not in df.columns]
        if missing:
            return {"error": f"value_columns not found in data: {missing}"}
        if group_by is not None and group_by not in df.columns:
            return {"error": f"group_by column not found in data: {group_by!r}"}
        if id_column is not None and id_column not in df.columns:
            return {"error": f"id_column not found in data: {id_column!r}"}

        boxes = _collect_boxes(
            df,
            value_columns=value_columns,
            group_by=group_by,
            id_column=id_column,
        )
        if not boxes:
            return {"error": "no non-empty groups to plot"}

        default_title = title
        if not default_title:
            default_title = (
                f"{value_columns[0]} by {group_by}"
                if group_by is not None
                else ", ".join(value_columns)
            )

        y_title = value_columns[0] if group_by is not None else "value"
        x_title = group_by or "variable"

        chart = BoxPlot(
            boxes=boxes,
            title=default_title,
            x_title=x_title,
            y_title=y_title,
            show_points=show_points,
            link_group=link_group,
        )

        spec = chart.to_spec()

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
            "link_group": link_group,
            "point_ids": spec.point_ids or [],
        }

        result["plotly"] = chart.to_plotly() if backend in ("both", "plotly") else None
        result["echarts"] = chart.to_echarts() if backend in ("both", "echarts") else None

        return clean(result)
    except Exception as e:
        logger.exception("Tool boxplot failed")
        return {"error": str(e)}
