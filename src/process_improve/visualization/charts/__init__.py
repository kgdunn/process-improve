"""Generic chart classes built on the :mod:`process_improve.visualization` IR.

Each class takes pre-computed numeric inputs, builds a
:class:`~process_improve.visualization.spec.ChartSpec`, and exposes
:meth:`to_plotly` / :meth:`to_echarts` rendering helpers.  Tool wrappers
in :mod:`process_improve.visualization.tools` are the LLM-facing entry
points; chart classes can also be used directly from notebooks.
"""

from process_improve.visualization.charts.boxplot import BoxPlot, BoxStats

__all__ = ["BoxPlot", "BoxStats"]
