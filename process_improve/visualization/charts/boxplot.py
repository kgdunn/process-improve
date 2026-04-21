"""Generic boxplot chart.

:class:`BoxPlot` consumes pre-computed quartiles and outlier rows, builds
a :class:`~process_improve.visualization.spec.ChartSpec`, and renders to
Plotly or ECharts via the shared adapters.  Statistic computation lives
in the tool wrapper (:mod:`process_improve.visualization.tools`) so the
chart class itself stays purely presentational.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from process_improve.visualization.adapters.echarts_adapter import EChartsAdapter
from process_improve.visualization.adapters.plotly_adapter import PlotlyAdapter
from process_improve.visualization.spec import (
    ChartSpec,
    Encoding,
    LayerSpec,
    PanelSpec,
)
from process_improve.visualization.types import MarkType, ScaleType


@dataclass
class BoxStats:
    """Five-number summary plus outliers for a single box.

    Parameters
    ----------
    group : str
        Category label rendered on the x-axis.
    q_stats : list[float]
        ``[lower_whisker, Q1, median, Q3, upper_whisker]`` — the order
        used by both ECharts boxplot series and Plotly's pre-computed
        :class:`plotly.graph_objects.Box` (``lowerfence``, ``q1``,
        ``median``, ``q3``, ``upperfence``).
    outliers : list[dict]
        Each entry has ``value`` (float) and optional ``id`` (str) — the
        stable observation id used for cross-chart brushing.
    """

    group: str
    q_stats: list[float]
    outliers: list[dict[str, Any]] = field(default_factory=list)


class BoxPlot:
    """Boxplot chart over one or more groups.

    Parameters
    ----------
    boxes : list[BoxStats]
        Pre-computed box summaries, one per category.
    title : str
        Chart title.
    y_title : str
        Y-axis label (typically the response/value column).
    x_title : str
        X-axis label (typically the group column).
    show_points : bool
        Whether to overlay outlier (and optionally jitter) points.
    link_group : str or None
        Cross-chart linking key.  When set, the rendered ECharts option
        gains a ``brush`` component and ``__link_group`` field.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        boxes: list[BoxStats],
        title: str = "",
        y_title: str = "",
        x_title: str = "",
        show_points: bool = False,
        link_group: str | None = None,
    ) -> None:
        self.boxes = boxes
        self.title = title
        self.y_title = y_title
        self.x_title = x_title
        self.show_points = show_points
        self.link_group = link_group

    # ------------------------------------------------------------------
    # Spec construction
    # ------------------------------------------------------------------

    def to_spec(self) -> ChartSpec:
        """Build the backend-agnostic :class:`ChartSpec`."""
        box_data = [
            {"group": b.group, "q_stats": list(b.q_stats)} for b in self.boxes
        ]
        layers: list[LayerSpec] = [
            LayerSpec(
                mark=MarkType.boxplot,
                data=box_data,
                x=Encoding(field="group", title=self.x_title, scale=ScaleType.category),
                y=Encoding(field="value", title=self.y_title),
                name="boxplot",
            ),
        ]

        point_ids: list[str] = []
        if self.show_points:
            point_rows: list[dict[str, Any]] = []
            for b in self.boxes:
                for o in b.outliers:
                    pid = str(o.get("id", "")) if o.get("id") is not None else ""
                    point_rows.append({
                        "group": b.group,
                        "value": o["value"],
                        "id": pid,
                    })
                    point_ids.append(pid)
            if point_rows:
                layers.append(
                    LayerSpec(
                        mark=MarkType.scatter,
                        data=point_rows,
                        x=Encoding(field="group", title=self.x_title, scale=ScaleType.category),
                        y=Encoding(field="value", title=self.y_title),
                        name="outliers",
                        style={"size": 5},
                    ),
                )

        panel = PanelSpec(
            layers=layers,
            title=self.title,
            x_title=self.x_title,
            y_title=self.y_title,
        )

        return ChartSpec(
            panels=[panel],
            title=self.title,
            plot_type="boxplot",
            link_group=self.link_group,
            point_ids=point_ids if (self.link_group and point_ids) else None,
        )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def to_plotly(self) -> dict[str, Any]:
        """Render to a Plotly figure dict."""
        return PlotlyAdapter().render(self.to_spec())

    def to_echarts(self) -> dict[str, Any]:
        """Render to an ECharts option dict."""
        return EChartsAdapter().render(self.to_spec())
