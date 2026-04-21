"""Backend-agnostic visualization layer.

A small intermediate representation (:class:`~process_improve.visualization.spec.ChartSpec`)
describes *what* to draw; backend adapters
(:class:`~process_improve.visualization.adapters.PlotlyAdapter`,
:class:`~process_improve.visualization.adapters.EChartsAdapter`) turn the
spec into Plotly figure dicts or ECharts option dicts.

This package is the shared substrate used by the DOE plots in
:mod:`process_improve.experiments.visualization` and by the generic chart
classes in :mod:`process_improve.visualization.charts`.
"""

from process_improve.visualization.spec import (
    Annotation,
    ChartSpec,
    Encoding,
    LayerSpec,
    PanelSpec,
)
from process_improve.visualization.types import (
    AnnotationType,
    MarkType,
    ScaleType,
)

__all__ = [
    "Annotation",
    "AnnotationType",
    "ChartSpec",
    "Encoding",
    "LayerSpec",
    "MarkType",
    "PanelSpec",
    "ScaleType",
]
