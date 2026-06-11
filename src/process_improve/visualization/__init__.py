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

# Register the base themes on import, but deliberately do NOT change
# ``plotly.io.templates.default``. Setting the global default would
# silently restyle every other Plotly figure in the same process (the
# library's own plots pass ``template="pi_journal"`` explicitly, so they
# never relied on the global default). Registration is additive and the
# theme names are ``pi_``-namespaced, so it is safe on import. Callers who
# want a process-improve theme as their global default can opt in with
# :func:`set_theme`.
#
# When the ``[plotting]`` extra is not installed (ENG-13 / #295), plotly is
# absent and the registration is silently skipped; calling ``raincloud``
# or any other plot helper will then raise the documented "install the
# extra" ImportError.
try:
    import plotly  # noqa: F401  - presence check
except ImportError:  # pragma: no cover - exercised via env-without-plotly
    pass
else:
    register_themes()

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
