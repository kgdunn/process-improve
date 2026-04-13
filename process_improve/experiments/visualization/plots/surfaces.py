"""Response-surface plots: contour, 3D surface, prediction variance.

These plots evaluate a fitted model over a 2D grid of two factors
(holding others at their centre or ``hold_values``).  They answer
questions like Q47 ("How do I draw a contour plot?") and Q121
("BBD contour plots and optimal salt/temperature/time").
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from process_improve.experiments.visualization.colors import DOE_PALETTE, SURFACE_COLORSCALE
from process_improve.experiments.visualization.plots.registry import BasePlot, register_plot
from process_improve.experiments.visualization.spec import (
    ChartSpec,
    Encoding,
    LayerSpec,
    PanelSpec,
)
from process_improve.experiments.visualization.types import MarkType


# ---------------------------------------------------------------------------
# Shared model evaluator
# ---------------------------------------------------------------------------


def _evaluate_model(
    coef_map: dict[str, float],
    point: dict[str, float],
) -> float:
    """Evaluate a polynomial model at a single point.

    Parameters
    ----------
    coef_map : dict[str, float]
        Term name → coefficient value (from ``analysis_results["coefficients"]``).
    point : dict[str, float]
        Factor name → coded value.

    Returns
    -------
    float
        Predicted response.
    """
    y_hat = coef_map.get("Intercept", 0.0)

    for term, coef in coef_map.items():
        if term == "Intercept":
            continue

        # Quadratic: I(A ** 2)
        m = re.match(r"I\((\w+)\s*\*\*\s*2\)", term)
        if m:
            name = m.group(1)
            y_hat += coef * point.get(name, 0.0) ** 2
            continue

        # Interaction: A:B
        if ":" in term:
            parts = term.split(":")
            val = 1.0
            for p in parts:
                val *= point.get(p, 0.0)
            y_hat += coef * val
            continue

        # Linear: plain factor name
        y_hat += coef * point.get(term, 0.0)

    return y_hat


def _build_coef_map(coefficients: list[dict[str, Any]]) -> dict[str, float]:
    """Extract a term → coefficient dict from analysis results."""
    return {c["term"]: c["coefficient"] for c in coefficients}


def _compute_grid(
    coef_map: dict[str, float],
    factor_x: str,
    factor_y: str,
    hold_values: dict[str, float],
    n_grid: int = 50,
) -> tuple[list[float], list[float], list[list[float]]]:
    """Evaluate model over a meshgrid for two factors.

    Parameters
    ----------
    coef_map : dict[str, float]
        Coefficient map.
    factor_x, factor_y : str
        The two factors to sweep.
    hold_values : dict[str, float]
        Fixed values for all other factors.
    n_grid : int
        Grid resolution per axis.

    Returns
    -------
    tuple[list[float], list[float], list[list[float]]]
        x_grid, y_grid, z_matrix (row-major).
    """
    x_grid = np.linspace(-1, 1, n_grid).tolist()
    y_grid = np.linspace(-1, 1, n_grid).tolist()

    z_matrix = []
    for y_val in y_grid:
        row = []
        for x_val in x_grid:
            point = dict(hold_values)
            point[factor_x] = x_val
            point[factor_y] = y_val
            row.append(_evaluate_model(coef_map, point))
        z_matrix.append(row)

    return x_grid, y_grid, z_matrix


# ---------------------------------------------------------------------------
# Contour plot
# ---------------------------------------------------------------------------


@register_plot("contour")
class ContourPlot(BasePlot):
    """Contour plot of the response surface for two factors.

    Evaluates the fitted model on a grid of two factors, holding
    remaining factors at their centre (or at ``hold_values``).

    Data sources
    ------------
    Requires ``analysis_results`` with ``"coefficients"`` key and
    ``factors_to_plot`` (exactly 2 factors).
    """

    def to_spec(self) -> ChartSpec:
        """Build a contour ChartSpec.

        Returns
        -------
        ChartSpec
        """
        coefficients = self._get_coefficients()
        if not coefficients:
            return ChartSpec(title="Contour Plot — no coefficients")

        factors = self.factors_to_plot or self._get_factor_names()
        if len(factors) < 2:
            return ChartSpec(title="Contour Plot — need at least 2 factors")

        factor_x, factor_y = factors[0], factors[1]
        coef_map = _build_coef_map(coefficients)
        x_grid, y_grid, z_matrix = _compute_grid(
            coef_map, factor_x, factor_y, self.hold_values,
        )

        contour_layer = LayerSpec(
            mark=MarkType.contour,
            data=[],  # Data is in style for grid-based plots
            x=Encoding(field="x", title=factor_x),
            y=Encoding(field="y", title=factor_y),
            name="Response",
            style={
                "x_grid": x_grid,
                "y_grid": y_grid,
                "z_matrix": z_matrix,
            },
        )

        panel = PanelSpec(
            layers=[contour_layer],
            title=f"Contour Plot: {factor_x} × {factor_y}",
            x_title=factor_x,
            y_title=factor_y,
        )

        return ChartSpec(
            panels=[panel],
            title=f"Contour Plot: {factor_x} × {factor_y}",
            plot_type="contour",
            metadata={
                "factors": [factor_x, factor_y],
                "hold_values": self.hold_values,
            },
        )


# ---------------------------------------------------------------------------
# 3D Surface plot
# ---------------------------------------------------------------------------


@register_plot("surface_3d")
class Surface3DPlot(BasePlot):
    """3D response surface for two factors.

    Same computation as :class:`ContourPlot` but rendered as an
    interactive 3D surface.

    Data sources
    ------------
    Requires ``analysis_results`` with ``"coefficients"`` key and
    ``factors_to_plot`` (exactly 2 factors).
    """

    def to_spec(self) -> ChartSpec:
        """Build a 3D surface ChartSpec.

        Returns
        -------
        ChartSpec
        """
        coefficients = self._get_coefficients()
        if not coefficients:
            return ChartSpec(title="3D Surface — no coefficients")

        factors = self.factors_to_plot or self._get_factor_names()
        if len(factors) < 2:
            return ChartSpec(title="3D Surface — need at least 2 factors")

        factor_x, factor_y = factors[0], factors[1]
        coef_map = _build_coef_map(coefficients)
        x_grid, y_grid, z_matrix = _compute_grid(
            coef_map, factor_x, factor_y, self.hold_values,
        )

        surface_layer = LayerSpec(
            mark=MarkType.surface,
            data=[],
            x=Encoding(field="x", title=factor_x),
            y=Encoding(field="y", title=factor_y),
            z=Encoding(field="z", title="Response"),
            name="Response Surface",
            style={
                "x_grid": x_grid,
                "y_grid": y_grid,
                "z_matrix": z_matrix,
            },
        )

        panel = PanelSpec(
            layers=[surface_layer],
            title=f"Response Surface: {factor_x} × {factor_y}",
            x_title=factor_x,
            y_title=factor_y,
            z_title="Response",
        )

        return ChartSpec(
            panels=[panel],
            title=f"Response Surface: {factor_x} × {factor_y}",
            plot_type="surface_3d",
            metadata={
                "factors": [factor_x, factor_y],
                "hold_values": self.hold_values,
                "requires_gl": True,
            },
        )


# ---------------------------------------------------------------------------
# Prediction Variance plot
# ---------------------------------------------------------------------------


@register_plot("prediction_variance")
class PredictionVariancePlot(BasePlot):
    """Prediction variance (scaled) contour for two factors.

    Shows the scaled prediction variance ``n * Var(ŷ) / σ²`` across the
    design space, which depends only on the design geometry, not the
    response values.

    Data sources
    ------------
    Requires ``design_data`` (the design matrix rows) and
    ``factors_to_plot`` (exactly 2 factors).
    """

    def to_spec(self) -> ChartSpec:
        """Build a prediction-variance ChartSpec.

        Returns
        -------
        ChartSpec
        """
        import pandas as pd  # noqa: PLC0415

        if not self.design_data:
            return ChartSpec(title="Prediction Variance — no design data")

        df = pd.DataFrame(self.design_data)
        factors = self.factors_to_plot or [c for c in df.columns if c != self.response_column]
        if len(factors) < 2:
            return ChartSpec(title="Prediction Variance — need at least 2 factors")

        factor_x, factor_y = factors[0], factors[1]

        # Build model matrix (main effects + interactions + quadratic)
        X = df[factors].values.astype(float)
        n, k = X.shape

        # Add intercept, interactions, quadratics for a full second-order
        X_terms = [np.ones(n)]
        for i in range(k):
            X_terms.append(X[:, i])
        for i in range(k):
            for j in range(i + 1, k):
                X_terms.append(X[:, i] * X[:, j])
        for i in range(k):
            X_terms.append(X[:, i] ** 2)

        X_model = np.column_stack(X_terms)

        try:
            XtX_inv = np.linalg.inv(X_model.T @ X_model)
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(X_model.T @ X_model)

        # Evaluate scaled prediction variance on a grid
        n_grid = 40
        x_grid = np.linspace(-1, 1, n_grid).tolist()
        y_grid = np.linspace(-1, 1, n_grid).tolist()

        fx_idx = factors.index(factor_x) if factor_x in factors else 0
        fy_idx = factors.index(factor_y) if factor_y in factors else 1

        z_matrix = []
        for y_val in y_grid:
            row = []
            for x_val in x_grid:
                point = np.zeros(k)
                point[fx_idx] = x_val
                point[fy_idx] = y_val
                # Hold other factors at centre (0)
                for fi in range(k):
                    if fi != fx_idx and fi != fy_idx:
                        point[fi] = self.hold_values.get(factors[fi], 0.0)

                # Build model row
                x_row = [1.0]
                for i in range(k):
                    x_row.append(point[i])
                for i in range(k):
                    for j in range(i + 1, k):
                        x_row.append(point[i] * point[j])
                for i in range(k):
                    x_row.append(point[i] ** 2)

                x_vec = np.array(x_row)
                spv = float(n * x_vec @ XtX_inv @ x_vec)
                row.append(spv)
            z_matrix.append(row)

        contour_layer = LayerSpec(
            mark=MarkType.contour,
            data=[],
            x=Encoding(field="x", title=factor_x),
            y=Encoding(field="y", title=factor_y),
            name="SPV",
            style={
                "x_grid": x_grid,
                "y_grid": y_grid,
                "z_matrix": z_matrix,
            },
        )

        panel = PanelSpec(
            layers=[contour_layer],
            title=f"Prediction Variance: {factor_x} × {factor_y}",
            x_title=factor_x,
            y_title=factor_y,
        )

        return ChartSpec(
            panels=[panel],
            title=f"Scaled Prediction Variance: {factor_x} × {factor_y}",
            plot_type="prediction_variance",
        )
