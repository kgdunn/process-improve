"""Plot registry and base class for DOE plot types.

Every concrete plot class decorates itself with ``@register_plot`` to
join the global registry.  The :func:`create_plot` factory is then used
by :func:`~process_improve.experiments.visualization.api.visualize_doe`
to dispatch by ``plot_type`` string.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from process_improve.visualization.adapters.echarts_adapter import EChartsAdapter
from process_improve.visualization.adapters.plotly_adapter import PlotlyAdapter
from process_improve.visualization.spec import ChartSpec

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_PLOT_REGISTRY: dict[str, type[BasePlot]] = {}


def register_plot(plot_type: str):  # noqa: ANN201
    """Class decorator that registers a plot class under *plot_type*.

    Parameters
    ----------
    plot_type : str
        The DOE plot type name (e.g. ``"pareto"``).

    Returns
    -------
    Callable
        The original class, unchanged.
    """

    def decorator(cls: type[BasePlot]) -> type[BasePlot]:
        _PLOT_REGISTRY[plot_type] = cls
        return cls

    return decorator


def create_plot(plot_type: str, **kwargs: Any) -> BasePlot:  # noqa: ANN401
    """Instantiate a plot class by its registered name.

    Parameters
    ----------
    plot_type : str
        DOE plot type name.
    **kwargs
        Forwarded to the plot class constructor.

    Returns
    -------
    BasePlot
        An instance of the registered plot class.

    Raises
    ------
    ValueError
        If *plot_type* is not registered.
    """
    _ensure_discovery()
    if plot_type not in _PLOT_REGISTRY:
        available = sorted(_PLOT_REGISTRY)
        raise ValueError(
            f"Unknown plot_type {plot_type!r}. Available: {available}"
        )
    return _PLOT_REGISTRY[plot_type](**kwargs)


def get_available_plot_types() -> list[str]:
    """Return all registered plot type names.

    Returns
    -------
    list[str]
        Sorted list of plot type names.
    """
    _ensure_discovery()
    return sorted(_PLOT_REGISTRY)


# ---------------------------------------------------------------------------
# Lazy discovery
# ---------------------------------------------------------------------------

_discovery_done = False


def _ensure_discovery() -> None:
    """Import all plot modules to populate the registry.

    Called lazily the first time :func:`create_plot` is invoked.
    """
    global _discovery_done  # noqa: PLW0603
    if _discovery_done:
        return

    import contextlib  # noqa: PLC0415
    import importlib  # noqa: PLC0415

    for module in [
        "process_improve.experiments.visualization.plots.effects",
        "process_improve.experiments.visualization.plots.significance",
        "process_improve.experiments.visualization.plots.diagnostics",
        "process_improve.experiments.visualization.plots.surfaces",
        "process_improve.experiments.visualization.plots.cube_plot",
        "process_improve.experiments.visualization.plots.optimization_plots",
        "process_improve.experiments.visualization.plots.design_quality",
    ]:
        with contextlib.suppress(ImportError):
            importlib.import_module(module)

    _discovery_done = True


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BasePlot(ABC):
    """Abstract base for all DOE plot classes.

    Subclasses must implement :meth:`to_spec`.  The :meth:`to_plotly`
    and :meth:`to_echarts` methods use the adapters to convert the spec.

    Parameters
    ----------
    analysis_results : dict or None
        Output from :func:`analyze_experiment`.
    design_data : list[dict] or None
        Raw design matrix rows.
    response_column : str or None
        Name of the response column.
    factors_to_plot : list[str] or None
        Factor subset to display.
    hold_values : dict or None
        Fixed values for non-plotted factors.
    highlight_significant : bool
        Whether to highlight significant effects.
    confidence_level : float
        Confidence level for reference lines.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        analysis_results: dict[str, Any] | None = None,
        design_data: list[dict[str, Any]] | None = None,
        response_column: str | None = None,
        factors_to_plot: list[str] | None = None,
        hold_values: dict[str, float] | None = None,
        highlight_significant: bool = True,
        confidence_level: float = 0.95,
    ) -> None:
        self.analysis_results = analysis_results or {}
        self.design_data = design_data or []
        self.response_column = response_column
        self.factors_to_plot = factors_to_plot
        self.hold_values = hold_values or {}
        self.highlight_significant = highlight_significant
        self.confidence_level = confidence_level

    @abstractmethod
    def to_spec(self) -> ChartSpec:
        """Build the backend-agnostic :class:`ChartSpec`.

        Returns
        -------
        ChartSpec
        """

    def to_plotly(self) -> dict[str, Any]:
        """Render to a Plotly figure dict.

        Returns
        -------
        dict
            Plotly figure dict (``data`` + ``layout``).
        """
        return PlotlyAdapter().render(self.to_spec())

    def to_echarts(self) -> dict[str, Any]:
        """Render to an ECharts option dict.

        Returns
        -------
        dict
            ECharts option dict.
        """
        return EChartsAdapter().render(self.to_spec())

    # ------------------------------------------------------------------
    # Convenience helpers for subclasses
    # ------------------------------------------------------------------

    def _get_effects(self) -> dict[str, float]:
        """Extract effects dict from analysis results.

        Returns
        -------
        dict[str, float]
            Term name → effect value.
        """
        return dict(self.analysis_results.get("effects", {}))

    def _get_coefficients(self) -> list[dict[str, Any]]:
        """Extract coefficient list from analysis results.

        Returns
        -------
        list[dict]
            Each dict has ``term``, ``coefficient``, ``p_value``, etc.
        """
        return list(self.analysis_results.get("coefficients", []))

    def _get_residual_diagnostics(self) -> dict[str, Any]:
        """Extract residual diagnostics from analysis results.

        Returns
        -------
        dict
            Keys: ``residuals``, ``fitted_values``, ``cooks_distance``,
            ``leverage``, etc.
        """
        return dict(self.analysis_results.get("residual_diagnostics", {}))

    def _get_lenth(self) -> dict[str, Any]:
        """Extract Lenth's method results from analysis results.

        Returns
        -------
        dict
            Keys: ``PSE``, ``ME``, ``SME``, ``effects`` list.
        """
        return dict(self.analysis_results.get("lenth_method", {}))

    def _get_factor_names(self) -> list[str]:
        """Infer factor names from analysis results or design data.

        Returns
        -------
        list[str]
        """
        # From model summary
        if "model_summary" in self.analysis_results:
            formula = self.analysis_results["model_summary"].get("formula", "")
            if "~" in formula:
                rhs = formula.split("~")[1].strip()
                # Extract bare factor names (before : or ** or I(...))
                import re  # noqa: PLC0415

                names = re.findall(r"\b([A-Za-z_]\w*)\b", rhs)
                # Remove reserved words
                reserved = {"I", "np", "power"}
                return list(dict.fromkeys(n for n in names if n not in reserved))

        # From design data
        if self.design_data:
            all_cols = list(self.design_data[0].keys())
            if self.response_column and self.response_column in all_cols:
                all_cols.remove(self.response_column)
            return all_cols

        return []
