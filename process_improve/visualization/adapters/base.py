"""Abstract base class for chart-spec adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from process_improve.visualization.spec import ChartSpec, PanelSpec


class AbstractAdapter(ABC):
    """Convert a :class:`ChartSpec` to a backend-native representation.

    Subclasses implement :meth:`render` (full spec) and
    :meth:`render_panel` (single panel).
    """

    @abstractmethod
    def render(self, spec: ChartSpec) -> Any:  # noqa: ANN401
        """Convert a full :class:`ChartSpec` to the native format.

        Parameters
        ----------
        spec : ChartSpec
            The backend-agnostic chart specification.

        Returns
        -------
        Any
            Native representation (e.g. Plotly figure dict or ECharts
            option dict).
        """

    @abstractmethod
    def render_panel(self, panel: PanelSpec) -> Any:  # noqa: ANN401
        """Convert a single :class:`PanelSpec` to the native format.

        Parameters
        ----------
        panel : PanelSpec
            One chart panel.

        Returns
        -------
        Any
            Native representation for this panel.
        """
