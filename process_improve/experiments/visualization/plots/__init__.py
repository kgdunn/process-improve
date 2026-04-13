"""DOE plot classes: one module per logical plot group.

Re-exports all plot classes so consumers can do::

    from process_improve.experiments.visualization.plots import ParetoPlot

The :func:`~process_improve.experiments.visualization.plots.registry.create_plot`
factory is the preferred entry point for programmatic creation.
"""

from process_improve.experiments.visualization.plots.registry import (
    BasePlot,
    create_plot,
    get_available_plot_types,
)

__all__ = [
    "BasePlot",
    "create_plot",
    "get_available_plot_types",
]
