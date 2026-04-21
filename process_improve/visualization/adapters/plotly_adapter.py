"""Plotly backend adapter: ChartSpec → Plotly figure dict.

Converts a :class:`ChartSpec` into a Plotly-compatible dict that can be
passed directly to ``plotly.graph_objects.Figure(data_dict)`` or
serialised to JSON via ``json.dumps``.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from process_improve.visualization.adapters.base import AbstractAdapter
from process_improve.visualization.colors import (
    DOE_PALETTE,
    SURFACE_COLORSCALE,
)
from process_improve.visualization.spec import (
    Annotation,
    ChartSpec,
    LayerSpec,
    PanelSpec,
)
from process_improve.visualization.types import AnnotationType, MarkType


class PlotlyAdapter(AbstractAdapter):
    """Translate a :class:`ChartSpec` to a Plotly figure dict."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(self, spec: ChartSpec) -> dict[str, Any]:
        """Convert the full chart spec to a Plotly figure dict.

        Parameters
        ----------
        spec : ChartSpec
            The backend-agnostic chart specification.

        Returns
        -------
        dict
            Plotly figure dict with ``data`` and ``layout`` keys.
        """
        n = len(spec.panels)
        if n == 0:
            return go.Figure().to_dict()

        if n == 1:  # noqa: SIM108
            fig = self._single_panel(spec.panels[0], spec.title)
        else:
            fig = self._multi_panel(spec)

        return fig.to_dict()

    def render_panel(self, panel: PanelSpec) -> dict[str, Any]:
        """Convert a single panel to a Plotly figure dict.

        Parameters
        ----------
        panel : PanelSpec
            One chart panel.

        Returns
        -------
        dict
            Plotly figure dict.
        """
        return self._single_panel(panel, panel.title).to_dict()

    # ------------------------------------------------------------------
    # Single-panel rendering
    # ------------------------------------------------------------------

    def _single_panel(self, panel: PanelSpec, title: str = "") -> go.Figure:
        if panel.secondary_y:  # noqa: SIM108
            fig = make_subplots(specs=[[{"secondary_y": True}]])
        else:
            fig = go.Figure()

        for layer in panel.layers:
            trace, on_secondary = self._layer_to_trace(layer)
            if panel.secondary_y and on_secondary:
                fig.add_trace(trace, secondary_y=True)
            else:
                fig.add_trace(trace)

        for ann in panel.annotations:
            self._add_annotation(fig, ann)

        fig.update_layout(
            title=dict(text=title or panel.title),
            xaxis=dict(title=panel.x_title),
            width=panel.width,
            height=panel.height,
            template="plotly_white",
            font=dict(size=12),
        )

        if panel.secondary_y:
            fig.update_yaxes(title_text=panel.y_title, secondary_y=False)
            fig.update_yaxes(title_text=panel.secondary_y_title, secondary_y=True)
        else:
            fig.update_layout(yaxis=dict(title=panel.y_title))

        return fig

    # ------------------------------------------------------------------
    # Multi-panel rendering
    # ------------------------------------------------------------------

    def _multi_panel(self, spec: ChartSpec) -> go.Figure:
        n = len(spec.panels)
        cols = min(spec.columns, n)
        rows = (n + cols - 1) // cols

        subtitles = [p.title for p in spec.panels]
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=subtitles)

        for idx, panel in enumerate(spec.panels):
            row = idx // cols + 1
            col = idx % cols + 1

            for layer in panel.layers:
                trace, _ = self._layer_to_trace(layer)
                fig.add_trace(trace, row=row, col=col)

            fig.update_xaxes(title_text=panel.x_title, row=row, col=col)
            fig.update_yaxes(title_text=panel.y_title, row=row, col=col)

            for ann in panel.annotations:
                self._add_annotation(fig, ann, row=row, col=col)

        fig.update_layout(
            title=dict(text=spec.title),
            template="plotly_white",
            font=dict(size=12),
            showlegend=True,
        )

        return fig

    # ------------------------------------------------------------------
    # Layer → Plotly trace
    # ------------------------------------------------------------------

    def _layer_to_trace(self, layer: LayerSpec) -> tuple[go.BaseTraceType, bool]:  # noqa: PLR0911
        """Convert a :class:`LayerSpec` to a Plotly trace.

        Returns
        -------
        tuple[go.BaseTraceType, bool]
            The trace and whether it targets the secondary y-axis.
        """
        on_secondary = layer.style.get("secondary_y", False)
        mark = layer.mark if isinstance(layer.mark, MarkType) else MarkType(layer.mark)

        # Marks that don't read from layer.x.field / layer.y.field at the
        # row level (they use layer.style grids or row-level ``q_stats``).
        if mark == MarkType.contour:
            return self._contour_trace(layer), on_secondary
        if mark == MarkType.surface:
            return self._surface_trace(layer), on_secondary
        if mark == MarkType.heatmap:
            return self._heatmap_trace(layer), on_secondary
        if mark == MarkType.wireframe:
            return self._wireframe_trace(layer), on_secondary
        if mark == MarkType.boxplot:
            return self._boxplot_trace(layer), on_secondary

        x_vals = [row[layer.x.field] for row in layer.data] if layer.x else []
        y_vals = [row[layer.y.field] for row in layer.data] if layer.y else []

        if mark == MarkType.bar:
            return self._bar_trace(layer, x_vals, y_vals), on_secondary

        if mark == MarkType.line:
            return self._line_trace(layer, x_vals, y_vals), on_secondary

        if mark == MarkType.scatter:
            return self._scatter_trace(layer, x_vals, y_vals), on_secondary

        if mark == MarkType.text:
            return self._text_trace(layer, x_vals, y_vals), on_secondary

        # Fallback to scatter
        return self._scatter_trace(layer, x_vals, y_vals), on_secondary

    def _bar_trace(
        self,
        layer: LayerSpec,
        x_vals: list,
        y_vals: list,
    ) -> go.Bar:
        colors = layer.style.get("colors")
        return go.Bar(
            x=x_vals,
            y=y_vals,
            name=layer.name,
            marker=dict(color=colors or layer.color or DOE_PALETTE["primary"]),
            opacity=layer.opacity,
        )

    def _line_trace(
        self,
        layer: LayerSpec,
        x_vals: list,
        y_vals: list,
    ) -> go.Scatter:
        dash = layer.style.get("dash", "solid")
        width = layer.style.get("width", 2)
        return go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines",
            name=layer.name,
            line=dict(
                color=layer.color or DOE_PALETTE["primary"],
                dash=dash,
                width=width,
            ),
            opacity=layer.opacity,
        )

    def _scatter_trace(
        self,
        layer: LayerSpec,
        x_vals: list,
        y_vals: list,
    ) -> go.Scatter:
        size = layer.style.get("size", 8)
        symbol = layer.style.get("symbol", "circle")
        colors = layer.style.get("colors")
        return go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="markers",
            name=layer.name,
            marker=dict(
                color=colors or layer.color or DOE_PALETTE["primary"],
                size=size,
                symbol=symbol,
            ),
            opacity=layer.opacity,
        )

    def _contour_trace(self, layer: LayerSpec) -> go.Contour:
        x_vals = layer.style.get("x_grid", [])
        y_vals = layer.style.get("y_grid", [])
        z_matrix = layer.style.get("z_matrix", [])
        return go.Contour(
            x=x_vals,
            y=y_vals,
            z=z_matrix,
            name=layer.name,
            colorscale=SURFACE_COLORSCALE,
            contours=dict(showlabels=True),
        )

    def _surface_trace(self, layer: LayerSpec) -> go.Surface:
        x_vals = layer.style.get("x_grid", [])
        y_vals = layer.style.get("y_grid", [])
        z_matrix = layer.style.get("z_matrix", [])
        return go.Surface(
            x=x_vals,
            y=y_vals,
            z=z_matrix,
            name=layer.name,
            colorscale=SURFACE_COLORSCALE,
        )

    def _heatmap_trace(self, layer: LayerSpec) -> go.Heatmap:
        x_vals = layer.style.get("x_grid", [])
        y_vals = layer.style.get("y_grid", [])
        z_matrix = layer.style.get("z_matrix", [])
        return go.Heatmap(
            x=x_vals,
            y=y_vals,
            z=z_matrix,
            name=layer.name,
            colorscale=SURFACE_COLORSCALE,
        )

    def _text_trace(
        self,
        layer: LayerSpec,
        x_vals: list,
        y_vals: list,
    ) -> go.Scatter:
        text_vals = [row.get("text", "") for row in layer.data]
        return go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="text",
            text=text_vals,
            name=layer.name,
            textfont=dict(size=layer.style.get("size", 12)),
        )

    def _boxplot_trace(self, layer: LayerSpec) -> go.Box:
        """Build a Plotly box trace from pre-computed quartiles.

        Each row in ``layer.data`` provides ``group`` (category label)
        and ``q_stats`` as ``[min, Q1, median, Q3, max]`` — matching
        the wire format used by the ECharts adapter.
        """
        groups = [row["group"] for row in layer.data]
        lower = [row["q_stats"][0] for row in layer.data]
        q1 = [row["q_stats"][1] for row in layer.data]
        median = [row["q_stats"][2] for row in layer.data]
        q3 = [row["q_stats"][3] for row in layer.data]
        upper = [row["q_stats"][4] for row in layer.data]
        return go.Box(
            x=groups,
            lowerfence=lower,
            q1=q1,
            median=median,
            q3=q3,
            upperfence=upper,
            name=layer.name,
            marker=dict(color=layer.color or DOE_PALETTE["primary"]),
            opacity=layer.opacity,
            boxpoints=False,
        )

    def _wireframe_trace(self, layer: LayerSpec) -> go.Scatter3d:
        x_vals = [row[layer.x.field] for row in layer.data] if layer.x else []
        y_vals = [row[layer.y.field] for row in layer.data] if layer.y else []
        z_vals = [row[layer.z.field] for row in layer.data] if layer.z else []
        return go.Scatter3d(
            x=x_vals,
            y=y_vals,
            z=z_vals,
            mode="lines+markers+text",
            name=layer.name,
            line=dict(color=layer.color or DOE_PALETTE["primary"], width=3),
            marker=dict(size=5),
            text=[row.get("text", "") for row in layer.data],
            textposition="top center",
        )

    # ------------------------------------------------------------------
    # Annotations → Plotly shapes / annotations
    # ------------------------------------------------------------------

    def _add_annotation(  # noqa: C901
        self,
        fig: go.Figure,
        ann: Annotation,
        *,
        row: int | None = None,
        col: int | None = None,
    ) -> None:
        at = ann.annotation_type
        if isinstance(at, str):
            at = AnnotationType(at)

        color = ann.style.get("color", DOE_PALETTE["threshold_me"])
        dash = ann.style.get("dash", "dash")
        width = ann.style.get("width", 2)

        if at in (AnnotationType.reference_line, AnnotationType.significance_threshold):
            if ann.value is None:
                return
            if ann.axis == "y":
                fig.add_hline(
                    y=ann.value,
                    line_dash=dash,
                    line_color=color,
                    line_width=width,
                    annotation_text=ann.label,
                    annotation_position="top right",
                    row=row or "all",
                    col=col or "all",
                )
            else:
                fig.add_vline(
                    x=ann.value,
                    line_dash=dash,
                    line_color=color,
                    line_width=width,
                    annotation_text=ann.label,
                    annotation_position="top right",
                    row=row or "all",
                    col=col or "all",
                )

        elif at == AnnotationType.reference_band:
            if ann.value is None or ann.value_end is None:
                return
            band_color = ann.style.get("fill_color", "rgba(37, 99, 235, 0.1)")
            if ann.axis == "y":
                fig.add_hrect(
                    y0=ann.value,
                    y1=ann.value_end,
                    fillcolor=band_color,
                    line_width=0,
                    row=row or "all",
                    col=col or "all",
                )
            else:
                fig.add_vrect(
                    x0=ann.value,
                    x1=ann.value_end,
                    fillcolor=band_color,
                    line_width=0,
                    row=row or "all",
                    col=col or "all",
                )

        elif at == AnnotationType.constraint_region:
            x_min = ann.style.get("x_min")
            x_max = ann.style.get("x_max")
            y_min = ann.style.get("y_min")
            y_max = ann.style.get("y_max")
            fill_color = ann.style.get("color", "rgba(220, 38, 38, 0.15)")
            if x_min is not None and x_max is not None:
                fig.add_vrect(
                    x0=x_min,
                    x1=x_max,
                    fillcolor=fill_color,
                    line_width=0,
                    annotation_text=ann.label,
                    row=row or "all",
                    col=col or "all",
                )
            if y_min is not None and y_max is not None:
                fig.add_hrect(
                    y0=y_min,
                    y1=y_max,
                    fillcolor=fill_color,
                    line_width=0,
                    annotation_text=ann.label,
                    row=row or "all",
                    col=col or "all",
                )
