"""ECharts backend adapter: ChartSpec → ECharts option dict.

Produces raw Python dicts that match the `ECharts option specification
<https://echarts.apache.org/en/option.html>`_.  No pyecharts dependency
— dicts are JSON-serialisable and can be passed directly to
``echarts.setOption()`` on the SvelteKit frontend.
"""

from __future__ import annotations

from typing import Any

from process_improve.experiments.visualization.adapters.base import AbstractAdapter
from process_improve.experiments.visualization.colors import (
    DOE_PALETTE,
    ECHARTS_VISUAL_MAP_COLORS,
)
from process_improve.experiments.visualization.spec import (
    Annotation,
    ChartSpec,
    LayerSpec,
    PanelSpec,
)
from process_improve.experiments.visualization.types import AnnotationType, MarkType


class EChartsAdapter(AbstractAdapter):
    """Translate a :class:`ChartSpec` to an ECharts option dict."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(self, spec: ChartSpec) -> dict[str, Any]:
        """Convert the full chart spec to an ECharts option dict.

        Parameters
        ----------
        spec : ChartSpec
            The backend-agnostic chart specification.

        Returns
        -------
        dict
            ECharts option dict with ``title``, ``xAxis``, ``yAxis``,
            ``series``, etc.
        """
        n = len(spec.panels)
        if n == 0:
            return {"title": {"text": spec.title}, "series": []}

        if n == 1:
            return self._single_panel(spec.panels[0], spec.title)
        return self._multi_panel(spec)

    def render_panel(self, panel: PanelSpec) -> dict[str, Any]:
        """Convert a single panel to an ECharts option dict.

        Parameters
        ----------
        panel : PanelSpec
            One chart panel.

        Returns
        -------
        dict
            ECharts option dict.
        """
        return self._single_panel(panel, panel.title)

    # ------------------------------------------------------------------
    # Single-panel rendering
    # ------------------------------------------------------------------

    def _single_panel(self, panel: PanelSpec, title: str = "") -> dict[str, Any]:
        series: list[dict[str, Any]] = []
        has_3d = False

        for layer in panel.layers:
            s, is_3d = self._layer_to_series(layer)
            if is_3d:
                has_3d = True
            series.append(s)

        # Collect annotations as markLines / markAreas on the first series
        mark_lines, mark_areas = self._collect_annotations(panel.annotations)
        if series and (mark_lines or mark_areas):
            if mark_lines:
                series[0].setdefault("markLine", {})
                series[0]["markLine"]["data"] = mark_lines
                series[0]["markLine"]["silent"] = True
                series[0]["markLine"]["symbol"] = "none"
            if mark_areas:
                series[0].setdefault("markArea", {})
                series[0]["markArea"]["data"] = mark_areas
                series[0]["markArea"]["silent"] = True

        option: dict[str, Any] = {
            "title": {"text": title or panel.title, "left": "center"},
            "tooltip": {"trigger": "axis" if not has_3d else "item"},
            "series": series,
            "toolbox": {
                "feature": {
                    "saveAsImage": {},
                    "dataZoom": {},
                    "restore": {},
                },
            },
        }

        if has_3d:
            # 3D plots use a different axis system
            option["xAxis3D"] = {"name": panel.x_title}
            option["yAxis3D"] = {"name": panel.y_title}
            option["zAxis3D"] = {"name": panel.z_title}
            option["grid3D"] = {}
        else:
            x_axis = self._build_x_axis(panel)
            option["xAxis"] = x_axis

            if panel.secondary_y:
                option["yAxis"] = [
                    {"type": "value", "name": panel.y_title},
                    {"type": "value", "name": panel.secondary_y_title, "position": "right"},
                ]
            else:
                option["yAxis"] = {"type": "value", "name": panel.y_title}

            option["grid"] = {"containLabel": True}

        return option

    # ------------------------------------------------------------------
    # Multi-panel rendering
    # ------------------------------------------------------------------

    def _multi_panel(self, spec: ChartSpec) -> dict[str, Any]:
        n = len(spec.panels)
        cols = min(spec.columns, n)
        rows = (n + cols - 1) // cols

        grids: list[dict] = []
        x_axes: list[dict] = []
        y_axes: list[dict] = []
        all_series: list[dict] = []

        panel_width = 100.0 / cols
        panel_height = 100.0 / rows

        for idx, panel in enumerate(spec.panels):
            row = idx // cols
            col = idx % cols

            grid = {
                "left": f"{col * panel_width + 5}%",
                "top": f"{row * panel_height + 8}%",
                "width": f"{panel_width - 10}%",
                "height": f"{panel_height - 16}%",
            }
            grids.append(grid)

            x_axis = self._build_x_axis(panel)
            x_axis["gridIndex"] = idx
            x_axes.append(x_axis)

            y_axes.append({
                "type": "value",
                "name": panel.y_title,
                "gridIndex": idx,
            })

            for layer in panel.layers:
                s, _ = self._layer_to_series(layer)
                s["xAxisIndex"] = idx
                s["yAxisIndex"] = idx
                all_series.append(s)

            mark_lines, mark_areas = self._collect_annotations(panel.annotations)
            if all_series and (mark_lines or mark_areas):
                last_series = all_series[-1]
                if mark_lines:
                    last_series.setdefault("markLine", {})
                    last_series["markLine"]["data"] = mark_lines
                    last_series["markLine"]["silent"] = True
                    last_series["markLine"]["symbol"] = "none"
                if mark_areas:
                    last_series.setdefault("markArea", {})
                    last_series["markArea"]["data"] = mark_areas
                    last_series["markArea"]["silent"] = True

        return {
            "title": {"text": spec.title, "left": "center"},
            "tooltip": {"trigger": "axis"},
            "grid": grids,
            "xAxis": x_axes,
            "yAxis": y_axes,
            "series": all_series,
            "toolbox": {"feature": {"saveAsImage": {}, "restore": {}}},
        }

    # ------------------------------------------------------------------
    # Layer → ECharts series
    # ------------------------------------------------------------------

    def _layer_to_series(self, layer: LayerSpec) -> tuple[dict[str, Any], bool]:
        """Convert a :class:`LayerSpec` to an ECharts series dict.

        Returns
        -------
        tuple[dict, bool]
            The series dict and whether it is a 3D chart.
        """
        mark = layer.mark if isinstance(layer.mark, MarkType) else MarkType(layer.mark)

        if mark == MarkType.bar:
            return self._bar_series(layer), False

        if mark == MarkType.line:
            return self._line_series(layer), False

        if mark == MarkType.scatter:
            return self._scatter_series(layer), False

        if mark in (MarkType.contour, MarkType.heatmap):
            return self._heatmap_series(layer), False

        if mark == MarkType.surface:
            return self._surface_series(layer), True

        if mark == MarkType.wireframe:
            return self._wireframe_series(layer), True

        if mark == MarkType.text:
            return self._scatter_series(layer), False

        # Fallback
        return self._scatter_series(layer), False

    def _bar_series(self, layer: LayerSpec) -> dict[str, Any]:
        data = [row[layer.y.field] for row in layer.data] if layer.y else []
        series: dict[str, Any] = {
            "type": "bar",
            "name": layer.name,
            "data": data,
        }
        colors = layer.style.get("colors")
        if colors:
            series["itemStyle"] = {"color": None}
            series["data"] = [
                {"value": v, "itemStyle": {"color": c}} for v, c in zip(data, colors)
            ]
        elif layer.color:
            series["itemStyle"] = {"color": layer.color}
        return series

    def _line_series(self, layer: LayerSpec) -> dict[str, Any]:
        data = self._paired_data(layer)
        return {
            "type": "line",
            "name": layer.name,
            "data": data,
            "smooth": layer.style.get("smooth", False),
            "lineStyle": {
                "color": layer.color or DOE_PALETTE["primary"],
                "type": _echarts_dash(layer.style.get("dash", "solid")),
                "width": layer.style.get("width", 2),
            },
            "symbol": "none" if not layer.style.get("show_points", False) else "circle",
        }

    def _scatter_series(self, layer: LayerSpec) -> dict[str, Any]:
        data = self._paired_data(layer)
        size = layer.style.get("size", 8)
        colors = layer.style.get("colors")
        series: dict[str, Any] = {
            "type": "scatter",
            "name": layer.name,
            "data": data,
            "symbolSize": size,
        }
        if colors:
            series["data"] = [
                {"value": d, "itemStyle": {"color": c}} for d, c in zip(data, colors)
            ]
        elif layer.color:
            series["itemStyle"] = {"color": layer.color}
        return series

    def _heatmap_series(self, layer: LayerSpec) -> dict[str, Any]:
        z_matrix = layer.style.get("z_matrix", [])
        x_grid = layer.style.get("x_grid", [])
        y_grid = layer.style.get("y_grid", [])

        # ECharts heatmap needs [x_idx, y_idx, value] triples
        data = []
        for i, y_val in enumerate(y_grid):
            row_data = z_matrix[i] if i < len(z_matrix) else []
            for j, x_val in enumerate(x_grid):
                z_val = row_data[j] if j < len(row_data) else 0
                data.append([x_val, y_val, z_val])

        return {
            "type": "heatmap",
            "name": layer.name,
            "data": data,
            "emphasis": {"itemStyle": {"shadowBlur": 10}},
        }

    def _surface_series(self, layer: LayerSpec) -> dict[str, Any]:
        z_matrix = layer.style.get("z_matrix", [])
        return {
            "type": "surface",
            "name": layer.name,
            "data": z_matrix,
            "shading": "color",
        }

    def _wireframe_series(self, layer: LayerSpec) -> dict[str, Any]:
        data = []
        for row in layer.data:
            point: list[Any] = []
            if layer.x:
                point.append(row.get(layer.x.field, 0))
            if layer.y:
                point.append(row.get(layer.y.field, 0))
            if layer.z:
                point.append(row.get(layer.z.field, 0))
            data.append(point)

        return {
            "type": "scatter3D",
            "name": layer.name,
            "data": data,
            "symbolSize": 8,
            "lineStyle": {"width": 2},
        }

    # ------------------------------------------------------------------
    # Annotations → markLine / markArea
    # ------------------------------------------------------------------

    def _collect_annotations(
        self,
        annotations: list[Annotation],
    ) -> tuple[list[dict], list[list[dict]]]:
        """Convert annotations to ECharts markLine and markArea data."""
        mark_lines: list[dict[str, Any]] = []
        mark_areas: list[list[dict[str, Any]]] = []

        for ann in annotations:
            at = ann.annotation_type
            if isinstance(at, str):
                at = AnnotationType(at)

            color = ann.style.get("color", DOE_PALETTE["threshold_me"])
            dash = ann.style.get("dash", "solid")

            if at in (AnnotationType.reference_line, AnnotationType.significance_threshold):
                if ann.value is None:
                    continue
                line_item: dict[str, Any] = {
                    "lineStyle": {
                        "color": color,
                        "type": _echarts_dash(dash),
                        "width": ann.style.get("width", 2),
                    },
                    "label": {"formatter": ann.label, "position": "end"},
                }
                if ann.axis == "y":
                    line_item["yAxis"] = ann.value
                else:
                    line_item["xAxis"] = ann.value
                mark_lines.append(line_item)

            elif at == AnnotationType.reference_band:
                if ann.value is None or ann.value_end is None:
                    continue
                fill = ann.style.get("fill_color", "rgba(37, 99, 235, 0.1)")
                if ann.axis == "y":
                    mark_areas.append([
                        {"yAxis": ann.value, "itemStyle": {"color": fill}},
                        {"yAxis": ann.value_end},
                    ])
                else:
                    mark_areas.append([
                        {"xAxis": ann.value, "itemStyle": {"color": fill}},
                        {"xAxis": ann.value_end},
                    ])

            elif at == AnnotationType.constraint_region:
                fill = ann.style.get("color", "rgba(220, 38, 38, 0.15)")
                x_min = ann.style.get("x_min")
                x_max = ann.style.get("x_max")
                y_min = ann.style.get("y_min")
                y_max = ann.style.get("y_max")
                if x_min is not None and x_max is not None:
                    mark_areas.append([
                        {"xAxis": x_min, "itemStyle": {"color": fill}},
                        {"xAxis": x_max},
                    ])
                if y_min is not None and y_max is not None:
                    mark_areas.append([
                        {"yAxis": y_min, "itemStyle": {"color": fill}},
                        {"yAxis": y_max},
                    ])

        return mark_lines, mark_areas

    # ------------------------------------------------------------------
    # Axis builders
    # ------------------------------------------------------------------

    def _build_x_axis(self, panel: PanelSpec) -> dict[str, Any]:
        """Infer x-axis type from the first layer's data."""
        axis: dict[str, Any] = {"name": panel.x_title}

        # Check if the first layer uses category data
        if panel.layers:
            first = panel.layers[0]
            if first.x and first.x.scale.value == "category":
                axis["type"] = "category"
                axis["data"] = [row[first.x.field] for row in first.data]
            else:
                axis["type"] = "value"
        else:
            axis["type"] = "value"

        return axis

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _paired_data(self, layer: LayerSpec) -> list[list]:
        """Build ECharts ``[[x, y], ...]`` paired data from a layer."""
        if not layer.x or not layer.y:
            return []
        return [
            [row.get(layer.x.field, 0), row.get(layer.y.field, 0)]
            for row in layer.data
        ]


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _echarts_dash(dash: str) -> str:
    """Map Plotly-style dash names to ECharts ``lineStyle.type``."""
    mapping = {
        "solid": "solid",
        "dash": "dashed",
        "dot": "dotted",
        "dashdot": "dashed",
        "longdash": "dashed",
    }
    return mapping.get(dash, "solid")
