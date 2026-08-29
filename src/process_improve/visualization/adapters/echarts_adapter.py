"""ECharts backend adapter: ChartSpec → ECharts option dict.

Produces raw Python dicts that match the `ECharts option specification
<https://echarts.apache.org/en/option.html>`_.  No pyecharts dependency
- dicts are JSON-serialisable and can be passed directly to
``echarts.setOption()`` on the SvelteKit frontend.
"""

from __future__ import annotations

from typing import Any

from process_improve.visualization.adapters.base import AbstractAdapter
from process_improve.visualization.colors import (
    DOE_PALETTE,
)
from process_improve.visualization.spec import (
    Annotation,
    ChartSpec,
    LayerSpec,
    PanelSpec,
)
from process_improve.visualization.types import AnnotationType, MarkType


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
            option: dict[str, Any] = {"title": {"text": spec.title}, "series": []}
        elif n == 1:
            option = self._single_panel(spec.panels[0], spec.title)
        else:
            option = self._multi_panel(spec)

        if spec.link_group:
            self._inject_brush(option, spec.link_group)
        return option

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
        self._attach_annotations(series, panel.annotations)

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

            y_axes.append(
                {
                    "type": "value",
                    "name": panel.y_title,
                    "gridIndex": idx,
                }
            )

            # Build this panel's own series first, then attach this panel's
            # annotations to them (never to a previous panel's series).
            panel_series: list[dict[str, Any]] = []
            for layer in panel.layers:
                s, _ = self._layer_to_series(layer)
                panel_series.append(s)

            self._attach_annotations(panel_series, panel.annotations)

            for s in panel_series:
                s["xAxisIndex"] = idx
                s["yAxisIndex"] = idx
            all_series.extend(panel_series)

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

    def _layer_to_series(self, layer: LayerSpec) -> tuple[dict[str, Any], bool]:  # noqa: PLR0911
        """Convert a :class:`LayerSpec` to an ECharts series dict.

        Returns
        -------
        tuple[dict, bool]
            The series dict and whether it is a 3D chart.

        Raises
        ------
        NotImplementedError
            If the layer uses :attr:`MarkType.area`, which is declared in
            the spec vocabulary but not implemented here.
        """
        mark = layer.mark if isinstance(layer.mark, MarkType) else MarkType(layer.mark)

        if mark == MarkType.area:
            msg = "MarkType.area is declared in the spec vocabulary but not implemented in the ECharts adapter."
            raise NotImplementedError(msg)

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

        if mark == MarkType.boxplot:
            return self._boxplot_series(layer), False

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
            _check_per_point_style(layer, "colors", colors, len(data))
            series["itemStyle"] = {"color": None}
            series["data"] = [{"value": v, "itemStyle": {"color": c}} for v, c in zip(data, colors, strict=True)]
        elif layer.color:
            series["itemStyle"] = {"color": layer.color}

        error_y = layer.style.get("error_y")
        if error_y and layer.x:
            _check_per_point_style(layer, "error_y", error_y, len(data))
            categories = [row[layer.x.field] for row in layer.data]
            mark_data: list[list[dict[str, Any]]] = []
            for cat, value, err in zip(categories, data, error_y, strict=True):
                if err is None:
                    continue
                err_abs = abs(float(err))
                if err_abs <= 0:
                    continue
                low = max(0.0, float(value) - err_abs)
                high = float(value) + err_abs
                mark_data.append(
                    [
                        {"coord": [cat, low], "symbol": "none"},
                        {"coord": [cat, high], "symbol": "none"},
                    ]
                )
            if mark_data:
                series["markLine"] = {
                    "silent": True,
                    "symbol": ["none", "none"],
                    "lineStyle": {"color": "#333", "width": 1.5},
                    "data": mark_data,
                }
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
            _check_per_point_style(layer, "colors", colors, len(data))
            series["data"] = [{"value": d, "itemStyle": {"color": c}} for d, c in zip(data, colors, strict=True)]
        elif layer.color:
            series["itemStyle"] = {"color": layer.color}
        return series

    def _heatmap_series(self, layer: LayerSpec) -> dict[str, Any]:
        """Build an ECharts heatmap series from grid-style layer data.

        Raises
        ------
        ValueError
            If ``style['z_matrix']`` does not have exactly one row per
            ``y_grid`` entry and one value per ``x_grid`` entry. A ragged
            matrix would otherwise silently plot missing cells as 0.
        """
        z_matrix = layer.style.get("z_matrix", [])
        x_grid = layer.style.get("x_grid", [])
        y_grid = layer.style.get("y_grid", [])

        if len(z_matrix) != len(y_grid) or any(len(row) != len(x_grid) for row in z_matrix):
            msg = (
                f"style['z_matrix'] on layer {layer.name!r} must have shape "
                f"({len(y_grid)}, {len(x_grid)}) to match y_grid and x_grid; "
                f"got {len(z_matrix)} rows of lengths {[len(row) for row in z_matrix]}."
            )
            raise ValueError(msg)

        # ECharts heatmap needs [x_idx, y_idx, value] triples
        data = []
        for i, y_val in enumerate(y_grid):
            for j, x_val in enumerate(x_grid):
                data.append([x_val, y_val, z_matrix[i][j]])

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

    def _boxplot_series(self, layer: LayerSpec) -> dict[str, Any]:
        """Build an ECharts boxplot series.

        Each row in ``layer.data`` must provide the five-number summary
        under ``q_stats`` as ``[min, Q1, median, Q3, max]`` (the order
        ECharts expects).  The category axis is picked up from
        :meth:`_build_x_axis` via ``layer.x`` with ``ScaleType.category``.
        """
        data = [list(row["q_stats"]) for row in layer.data]
        series: dict[str, Any] = {
            "type": "boxplot",
            "name": layer.name,
            "data": data,
        }
        if layer.color:
            series["itemStyle"] = {"color": layer.color}
        return series

    def _wireframe_series(self, layer: LayerSpec) -> dict[str, Any]:
        """Build a 3D scatter series from a wireframe layer.

        Raises
        ------
        KeyError
            If a data row is missing an encoded field, matching the
            Plotly adapter's behaviour for the same layer.
        """
        data = []
        for row in layer.data:
            point: list[Any] = []
            if layer.x:
                point.append(row[layer.x.field])
            if layer.y:
                point.append(row[layer.y.field])
            if layer.z:
                point.append(row[layer.z.field])
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

    def _collect_annotations(  # noqa: C901, PLR0912
        self,
        annotations: list[Annotation],
    ) -> tuple[list[dict], list[list[dict]]]:
        """Convert annotations to ECharts markLine and markArea data.

        Raises
        ------
        NotImplementedError
            If an annotation uses :attr:`AnnotationType.label`, which is
            declared in the spec vocabulary but not implemented here.
        """
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
                    mark_areas.append(
                        [
                            {"yAxis": ann.value, "itemStyle": {"color": fill}},
                            {"yAxis": ann.value_end},
                        ]
                    )
                else:
                    mark_areas.append(
                        [
                            {"xAxis": ann.value, "itemStyle": {"color": fill}},
                            {"xAxis": ann.value_end},
                        ]
                    )

            elif at == AnnotationType.constraint_region:
                fill = ann.style.get("color", "rgba(220, 38, 38, 0.15)")
                x_min = ann.style.get("x_min")
                x_max = ann.style.get("x_max")
                y_min = ann.style.get("y_min")
                y_max = ann.style.get("y_max")
                if x_min is not None and x_max is not None:
                    mark_areas.append(
                        [
                            {"xAxis": x_min, "itemStyle": {"color": fill}},
                            {"xAxis": x_max},
                        ]
                    )
                if y_min is not None and y_max is not None:
                    mark_areas.append(
                        [
                            {"yAxis": y_min, "itemStyle": {"color": fill}},
                            {"yAxis": y_max},
                        ]
                    )

            elif at == AnnotationType.label:
                msg = (
                    "AnnotationType.label is declared in the spec vocabulary but not implemented "
                    "in the ECharts adapter."
                )
                raise NotImplementedError(msg)

        return mark_lines, mark_areas

    # ------------------------------------------------------------------
    # Cross-chart linking
    # ------------------------------------------------------------------

    def _inject_brush(self, option: dict[str, Any], link_group: str) -> None:
        """Attach a ``brush`` component and record the link group key.

        The frontend link coordinator reads ``__link_group`` to decide
        which charts belong to the same brushing group.
        """
        option["__link_group"] = link_group

        toolbox = option.setdefault("toolbox", {})
        feature = toolbox.setdefault("feature", {})
        feature.setdefault("brush", {})

        option.setdefault(
            "brush",
            {
                "toolbox": ["rect", "polygon", "clear"],
                "xAxisIndex": "all",
                "throttleType": "debounce",
                "throttleDelay": 100,
            },
        )

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

    def _merge_mark_lines(
        self,
        series: dict[str, Any],
        new_lines: list[dict[str, Any]],
    ) -> None:
        """Append annotation markLines to a series, preserving any existing
        per-series markLines (e.g. bar-layer error bars).
        """
        existing = series.setdefault("markLine", {})
        if "data" in existing:
            existing["data"] = list(existing["data"]) + list(new_lines)
        else:
            existing["data"] = list(new_lines)
            existing.setdefault("silent", True)
            existing.setdefault("symbol", "none")

    def _paired_data(self, layer: LayerSpec) -> list[list]:
        """Build ECharts ``[[x, y], ...]`` paired data from a layer.

        Raises
        ------
        KeyError
            If a data row is missing the encoded x or y field, matching
            the Plotly adapter's behaviour for the same layer.
        """
        if not layer.x or not layer.y:
            return []
        return [[row[layer.x.field], row[layer.y.field]] for row in layer.data]

    def _attach_annotations(
        self,
        series: list[dict[str, Any]],
        annotations: list[Annotation],
    ) -> None:
        """Attach a panel's annotations to a series belonging to that panel.

        The single-panel and multi-panel paths both call this with only the
        current panel's series, so annotations can never end up on another
        panel. When the panel has annotations but no layers, an empty,
        invisible line series is created to carry them.

        Parameters
        ----------
        series : list[dict]
            The series built for the current panel; may be extended in place.
        annotations : list[Annotation]
            The panel's annotations.
        """
        mark_lines, mark_areas = self._collect_annotations(annotations)
        if not (mark_lines or mark_areas):
            return

        if series:
            target = series[0]
        else:
            target = {"type": "line", "name": "", "data": [], "silent": True}
            series.append(target)

        if mark_lines:
            self._merge_mark_lines(target, mark_lines)
        if mark_areas:
            target.setdefault("markArea", {})
            target["markArea"]["data"] = mark_areas
            target["markArea"]["silent"] = True


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _check_per_point_style(layer: LayerSpec, key: str, values: Any, n_points: int) -> None:  # noqa: ANN401
    """Check that a per-point style list has one entry per data point.

    A shorter list would otherwise silently truncate the series (marks
    simply vanish from the chart), and a longer list indicates the style
    was built for different data.

    Parameters
    ----------
    layer : LayerSpec
        The layer being rendered, used to name the offender in the message.
    key : str
        The ``layer.style`` key being checked (e.g. ``"colors"``).
    values : Any
        The per-point style entries.
    n_points : int
        Number of data points in the layer.

    Raises
    ------
    ValueError
        If ``len(values)`` differs from ``n_points``.
    """
    if len(values) != n_points:
        msg = (
            f"style[{key!r}] on layer {layer.name!r} has {len(values)} entries "
            f"but the layer has {n_points} data points; provide one entry per point."
        )
        raise ValueError(msg)


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
