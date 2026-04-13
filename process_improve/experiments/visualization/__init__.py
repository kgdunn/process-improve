"""DOE visualization: dual-backend (Plotly + ECharts) chart generation.

Provides :func:`visualize_doe`, the public API for generating DOE plots
from fitted models or design matrices.  Each plot type is computed once
via a backend-agnostic :class:`ChartSpec` intermediate representation,
then rendered to Plotly figures and/or ECharts option dicts through
thin adapter classes.

Architecture
------------
Plot classes (one per DOE plot type) compute statistics and build a
:class:`ChartSpec`.  Backend adapters translate the spec into native
Plotly or ECharts JSON.  This keeps DOE logic DRY while allowing
backend-specific tuning.

Quick start
-----------
::

    from process_improve.experiments.visualization import visualize_doe

    result = visualize_doe(
        plot_type="pareto",
        analysis_results=analysis_output,
    )
    plotly_fig = result["plotly"]   # Plotly figure dict
    echarts_opt = result["echarts"] # ECharts option dict
"""

from process_improve.experiments.visualization.api import visualize_doe

__all__ = ["visualize_doe"]
