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

from process_improve.visualization.raincloud import raincloud
from process_improve.visualization.spec import (
    Annotation,
    ChartSpec,
    Encoding,
    LayerSpec,
    PanelSpec,
)
from process_improve.visualization.themes import (
    DEFAULT_THEME,
    THEME_BRAND,
    THEME_ECONOMIST,
    THEME_JOURNAL,
    THEME_NAMES,
    THEME_TUFTE,
    register_themes,
    set_theme,
)
from process_improve.visualization.types import (
    AnnotationType,
    MarkType,
    ScaleType,
)

# Register the base themes and apply the package default on import. When
# the ``[plotting]`` extra is not installed (ENG-13 / #295), plotly is
# absent and the registration is silently skipped; calling ``raincloud``
# or any other plot helper will then raise the documented "install the
# extra" ImportError.
try:
    import plotly  # noqa: F401  - presence check
except ImportError:
    pass
else:
    register_themes()
    set_theme(DEFAULT_THEME)

__all__ = [
    "DEFAULT_THEME",
    "THEME_BRAND",
    "THEME_ECONOMIST",
    "THEME_JOURNAL",
    "THEME_NAMES",
    "THEME_TUFTE",
    "Annotation",
    "AnnotationType",
    "ChartSpec",
    "Encoding",
    "LayerSpec",
    "MarkType",
    "PanelSpec",
    "ScaleType",
    "raincloud",
    "register_themes",
    "set_theme",
]
