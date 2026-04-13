"""Backend-agnostic chart specification (ChartSpec IR).

The intermediate representation captures *what* to draw — data, visual
encodings, annotations — while backend adapters handle *how* to draw it.
Inspired by Vega-Lite's declarative approach, with DOE-specific
primitives (significance thresholds, constraint regions) as first-class
concepts.

Dataclasses are used over Pydantic to keep this layer dependency-free
and fast to construct.  All fields are JSON-serialisable via
:meth:`ChartSpec.to_dict`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from process_improve.experiments.visualization.types import (
    AnnotationType,
    MarkType,
    ScaleType,
)


# ---------------------------------------------------------------------------
# Encoding: maps a data field to a visual channel
# ---------------------------------------------------------------------------


@dataclass
class Encoding:
    """Map a data field to a visual channel (x, y, colour, size, …).

    Parameters
    ----------
    field : str
        Column name in the layer's row-oriented data.
    title : str
        Human-readable axis or legend label.
    scale : ScaleType
        Axis scale (linear, log, category).
    domain : tuple[float, float] or None
        Explicit axis min/max override.
    format_str : str
        Number format (e.g. ``".2f"``).
    """

    field: str
    title: str = ""
    scale: ScaleType = ScaleType.linear
    domain: tuple[float, float] | None = None
    format_str: str = ""


# ---------------------------------------------------------------------------
# LayerSpec: one visual layer (trace) in a chart
# ---------------------------------------------------------------------------


@dataclass
class LayerSpec:
    """A single visual layer — one trace in Plotly, one series in ECharts.

    Parameters
    ----------
    mark : MarkType
        Visual mark type (bar, line, scatter, contour, …).
    data : list[dict]
        Row-oriented records for this layer.
    x : Encoding or None
        Horizontal-axis encoding.
    y : Encoding or None
        Vertical-axis encoding.
    z : Encoding or None
        Depth / colour-intensity encoding (contour, surface, heatmap).
    color : str or None
        Literal CSS colour string, or field name for colour encoding.
    name : str
        Trace / series name for the legend.
    opacity : float
        Layer opacity (0–1).
    style : dict
        Catch-all for extra visual properties (line dash, marker size,
        bar width, …).
    """

    mark: MarkType
    data: list[dict[str, Any]]
    x: Encoding | None = None
    y: Encoding | None = None
    z: Encoding | None = None
    color: str | None = None
    name: str = ""
    opacity: float = 1.0
    style: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Annotations: reference lines, thresholds, regions
# ---------------------------------------------------------------------------


@dataclass
class Annotation:
    """An overlay annotation on a chart panel.

    Parameters
    ----------
    annotation_type : AnnotationType
        What kind of annotation this is.
    axis : str
        ``"x"`` or ``"y"`` — which axis the annotation references.
    value : float or None
        Position for reference lines / thresholds.
    value_end : float or None
        End position for bands / regions.
    label : str
        Annotation text label.
    style : dict
        Visual overrides (color, dash, width, …).
    """

    annotation_type: AnnotationType
    axis: str = "y"
    value: float | None = None
    value_end: float | None = None
    label: str = ""
    style: dict[str, Any] = field(default_factory=dict)


def significance_threshold(
    value: float,
    *,
    alpha: float = 0.05,
    label: str | None = None,
    name: str = "ME",
) -> Annotation:
    """Create a DOE significance-threshold annotation.

    Used on Pareto and half-normal plots to indicate the margin of error
    (ME) or simultaneous margin of error (SME) from Lenth's method.

    Parameters
    ----------
    value : float
        Threshold position on the y-axis.
    alpha : float
        Significance level (for the label).
    label : str or None
        Override label (default: ``"ME (α=0.05)"``).
    name : str
        Threshold name (``"ME"`` or ``"SME"``).

    Returns
    -------
    Annotation
    """
    return Annotation(
        annotation_type=AnnotationType.significance_threshold,
        axis="y",
        value=value,
        label=label or f"{name} (α={alpha})",
        style={"color": "#F59E0B", "dash": "dash", "width": 2},
    )


def constraint_region(
    *,
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    label: str = "Infeasible",
) -> Annotation:
    """Create a DOE constraint-region annotation.

    Used on contour and overlay plots to shade infeasible areas.

    Parameters
    ----------
    x_min, x_max, y_min, y_max : float or None
        Region boundaries.
    label : str
        Region label.

    Returns
    -------
    Annotation
    """
    return Annotation(
        annotation_type=AnnotationType.constraint_region,
        axis="xy",
        label=label,
        style={
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "color": "rgba(220, 38, 38, 0.15)",
        },
    )


# ---------------------------------------------------------------------------
# PanelSpec: one chart panel in a layout
# ---------------------------------------------------------------------------


@dataclass
class PanelSpec:
    """A single chart panel — becomes one Plotly subplot or ECharts grid.

    Parameters
    ----------
    layers : list[LayerSpec]
        Visual layers drawn in this panel.
    annotations : list[Annotation]
        Overlaid annotations.
    title : str
        Panel title.
    x_title : str
        Horizontal-axis label.
    y_title : str
        Vertical-axis label.
    z_title : str
        Depth-axis label (3D plots only).
    secondary_y : bool
        Whether a secondary y-axis is needed.
    secondary_y_title : str
        Label for the secondary y-axis.
    width : int
        Panel width in pixels.
    height : int
        Panel height in pixels.
    backend_hints : dict
        Escape hatch for backend-specific options.
    """

    layers: list[LayerSpec] = field(default_factory=list)
    annotations: list[Annotation] = field(default_factory=list)
    title: str = ""
    x_title: str = ""
    y_title: str = ""
    z_title: str = ""
    secondary_y: bool = False
    secondary_y_title: str = ""
    width: int = 700
    height: int = 500
    backend_hints: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ChartSpec: the top-level specification (may contain multiple panels)
# ---------------------------------------------------------------------------


@dataclass
class ChartSpec:
    """Top-level chart specification — the IR that adapters consume.

    A ``ChartSpec`` may contain one or more :class:`PanelSpec` panels.
    Single-panel charts (Pareto, contour, …) have ``len(panels) == 1``.
    Multi-panel charts (diagnostic trio, linked contours) have 2–4
    panels with a grid layout.

    Parameters
    ----------
    panels : list[PanelSpec]
        One or more chart panels.
    title : str
        Overall chart / dashboard title.
    plot_type : str
        DOE plot type name (for metadata).
    layout : str
        ``"single"``, ``"row"``, ``"column"``, or ``"grid"``.
    columns : int
        Number of columns for grid layout.
    linked : bool
        Whether panels share brush / zoom interactions.
    metadata : dict
        Extra metadata passed through to the output.
    """

    panels: list[PanelSpec] = field(default_factory=list)
    title: str = ""
    plot_type: str = ""
    layout: str = "single"
    columns: int = 2
    linked: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- helpers -----------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the entire spec to a plain dict (JSON-safe).

        Enum values are converted to their string representations.
        """
        raw = asdict(self)
        return _clean_enums(raw)

    def to_data_dict(self) -> dict[str, Any]:
        """Extract computed data arrays from the spec.

        Returns a lightweight dict containing just the raw data from each
        panel's layers — useful when the consumer wants to render with a
        custom frontend.
        """
        panels_data: list[dict[str, Any]] = []
        for panel in self.panels:
            layers_data = []
            for layer in panel.layers:
                layers_data.append({
                    "mark": layer.mark.value if isinstance(layer.mark, MarkType) else layer.mark,
                    "data": layer.data,
                    "name": layer.name,
                })
            panels_data.append({
                "title": panel.title,
                "layers": layers_data,
            })
        return {
            "plot_type": self.plot_type,
            "title": self.title,
            "panels": panels_data,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _clean_enums(obj: Any) -> Any:  # noqa: ANN401
    """Recursively convert Enum values to their ``.value`` string."""
    if isinstance(obj, dict):
        return {k: _clean_enums(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_enums(v) for v in obj]
    if isinstance(obj, Enum):
        return obj.value
    return obj
