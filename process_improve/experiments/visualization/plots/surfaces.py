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

from process_improve._linalg import is_singular
from process_improve.experiments.visualization.plots.registry import BasePlot, register_plot
from process_improve.visualization.spec import (
    Annotation,
    ChartSpec,
    Encoding,
    LayerSpec,
    PanelSpec,
)
from process_improve.visualization.types import AnnotationType, MarkType

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
    remaining factors at their centre (or at ``hold_values``).  When
    ``design_data`` is also provided the experimental points are
    overlaid as a scatter layer with hover text giving every factor
    level and the observed response.  Replicated points are jittered
    slightly so they are individually visible.

    Data sources
    ------------
    Requires ``analysis_results`` with ``"coefficients"`` key and
    ``factors_to_plot`` (exactly 2 factors).  Optionally accepts
    ``design_data`` for the experimental-point overlay and
    ``factor_labels`` (a ``{symbol: full_name}`` mapping) for the
    axis titles.
    """

    def to_spec(self) -> ChartSpec:
        """Build a contour ChartSpec.

        Returns
        -------
        ChartSpec
        """
        coefficients = self._get_coefficients()
        if not coefficients:
            return ChartSpec(title="Contour Plot - no coefficients")

        factors = self.factors_to_plot or self._get_factor_names()
        if len(factors) < 2:
            return ChartSpec(title="Contour Plot - need at least 2 factors")

        factor_x, factor_y = factors[0], factors[1]
        x_title = self._axis_title(factor_x)
        y_title = self._axis_title(factor_y)

        coef_map = _build_coef_map(coefficients)
        x_grid, y_grid, z_matrix = _compute_grid(
            coef_map, factor_x, factor_y, self.hold_values,
        )

        contour_layer = LayerSpec(
            mark=MarkType.contour,
            data=[],  # Data is in style for grid-based plots
            x=Encoding(field="x", title=x_title),
            y=Encoding(field="y", title=y_title),
            name="Response",
            style={
                "x_grid": x_grid,
                "y_grid": y_grid,
                "z_matrix": z_matrix,
            },
        )

        layers: list[LayerSpec] = [contour_layer]
        point_layer = self._build_design_point_layer(factor_x, factor_y)
        if point_layer is not None:
            layers.append(point_layer)

        panel = PanelSpec(
            layers=layers,
            annotations=_zero_reference_lines(),
            title=f"Contour Plot: {x_title} x {y_title}",
            x_title=x_title,
            y_title=y_title,
            backend_hints={"equal_aspect": True},
        )

        return ChartSpec(
            panels=[panel],
            title=f"Contour Plot: {x_title} x {y_title}",
            plot_type="contour",
            metadata={
                "factors": [factor_x, factor_y],
                "factor_labels": {factor_x: x_title, factor_y: y_title},
                "hold_values": self.hold_values,
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _axis_title(self, factor: str) -> str:
        """Return the full-name axis label for *factor*, falling back to the symbol."""
        label = self.factor_labels.get(factor)
        if not label or label == factor:
            return factor
        return f"{label} ({factor})"

    def _build_design_point_layer(
        self,
        factor_x: str,
        factor_y: str,
    ) -> LayerSpec | None:
        """Return a scatter layer of experimental points, or ``None``.

        Replicated points (same coded location) are jittered so they do
        not overlap exactly.  Hover text exposes every factor level for
        the run plus the measured response.
        """
        if not self.design_data:
            return None

        rows = [r for r in self.design_data if factor_x in r and factor_y in r]
        if not rows:
            return None

        # Stable jitter: deterministic per-replicate offset so plots are
        # reproducible across runs.
        rng = np.random.default_rng(seed=0)
        jitter_amplitude = 0.02

        # Group rows by (rounded) coded location to detect replicates.
        groups: dict[tuple[float, float], list[int]] = {}
        for idx, r in enumerate(rows):
            key = (round(float(r[factor_x]), 6), round(float(r[factor_y]), 6))
            groups.setdefault(key, []).append(idx)

        data: list[dict[str, Any]] = []
        for indices in groups.values():
            n_replicates = len(indices)
            for k, idx in enumerate(indices):
                r = rows[idx]
                if n_replicates > 1:
                    dx = float(rng.uniform(-jitter_amplitude, jitter_amplitude))
                    dy = float(rng.uniform(-jitter_amplitude, jitter_amplitude))
                else:
                    dx = dy = 0.0
                point: dict[str, Any] = {
                    "x": float(r[factor_x]) + dx,
                    "y": float(r[factor_y]) + dy,
                    "hover": _format_point_hover(
                        r,
                        factor_x,
                        factor_y,
                        self.response_column,
                        replicate=(k + 1, n_replicates) if n_replicates > 1 else None,
                    ),
                }
                if self.response_column and self.response_column in r:
                    point["response"] = r[self.response_column]
                data.append(point)

        return LayerSpec(
            mark=MarkType.scatter,
            data=data,
            x=Encoding(field="x"),
            y=Encoding(field="y"),
            name="Experimental runs",
            color="#111827",
            opacity=0.55,
            style={
                "size": 9,
                "symbol": "circle",
                "hover_field": "hover",
                "edge_color": "#111111",
                "edge_width": 1,
            },
        )


def _zero_reference_lines() -> list[Annotation]:
    """Solid black axes through the origin (issue #5)."""
    style = {"color": "#111111", "dash": "solid", "width": 1}
    return [
        Annotation(
            annotation_type=AnnotationType.reference_line,
            axis="x",
            value=0.0,
            style=dict(style),
        ),
        Annotation(
            annotation_type=AnnotationType.reference_line,
            axis="y",
            value=0.0,
            style=dict(style),
        ),
    ]


def _format_point_hover(
    row: dict[str, Any],
    factor_x: str,
    factor_y: str,
    response_column: str | None,
    replicate: tuple[int, int] | None = None,
) -> str:
    """Build the hover string for an experimental point.

    Lists every factor level (not just the two being plotted) so that
    overlapping runs in a fractional or replicated design can still be
    distinguished - see issue #23.
    """
    lines: list[str] = []
    for key, value in row.items():
        if key == response_column:
            continue
        marker = " *" if key in (factor_x, factor_y) else ""
        try:
            val_str = f"{float(value):g}"
        except (TypeError, ValueError):
            val_str = str(value)
        lines.append(f"{key}{marker}: {val_str}")
    if response_column and response_column in row:
        try:
            resp_str = f"{float(row[response_column]):g}"
        except (TypeError, ValueError):
            resp_str = str(row[response_column])
        lines.append(f"{response_column}: {resp_str}")
    if replicate is not None:
        lines.append(f"replicate {replicate[0]} of {replicate[1]}")
    return "<br>".join(lines)


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
            return ChartSpec(title="3D Surface - no coefficients")

        factors = self.factors_to_plot or self._get_factor_names()
        if len(factors) < 2:
            return ChartSpec(title="3D Surface - need at least 2 factors")

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
            title=f"Response Surface: {factor_x} x {factor_y}",
            x_title=factor_x,
            y_title=factor_y,
            z_title="Response",
        )

        return ChartSpec(
            panels=[panel],
            title=f"Response Surface: {factor_x} x {factor_y}",
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

    def to_spec(self) -> ChartSpec:  # noqa: C901, PLR0912
        """Build a prediction-variance ChartSpec.

        Returns
        -------
        ChartSpec
        """
        import pandas as pd  # noqa: PLC0415

        if not self.design_data:
            return ChartSpec(title="Prediction Variance - no design data")

        df = pd.DataFrame(self.design_data)
        factors = self.factors_to_plot or [c for c in df.columns if c != self.response_column]
        if len(factors) < 2:
            return ChartSpec(title="Prediction Variance - need at least 2 factors")

        factor_x, factor_y = factors[0], factors[1]

        # Build model matrix (main effects + interactions + quadratic)
        X = df[factors].values.astype(float)
        n, k = X.shape

        # Add intercept, interactions, quadratics for a full second-order
        X_terms = [np.ones(n)]
        for i in range(k):
            X_terms.append(X[:, i])  # noqa: PERF401
        for i in range(k):
            for j in range(i + 1, k):
                X_terms.append(X[:, i] * X[:, j])  # noqa: PERF401
        for i in range(k):
            X_terms.append(X[:, i] ** 2)  # noqa: PERF401

        X_model = np.column_stack(X_terms)

        # Fall back to the pseudo-inverse not just for an exactly-singular X'X but
        # also for an ill-conditioned one, where np.linalg.inv would silently
        # return overflow-driven garbage.
        XtX = X_model.T @ X_model
        XtX_inv = np.linalg.pinv(XtX) if is_singular(XtX) else np.linalg.inv(XtX)

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
                    if fi != fx_idx and fi != fy_idx:  # noqa: PLR1714
                        point[fi] = self.hold_values.get(factors[fi], 0.0)

                # Build model row
                x_row = [1.0]
                for i in range(k):
                    x_row.append(point[i])  # noqa: PERF401
                for i in range(k):
                    for j in range(i + 1, k):
                        x_row.append(point[i] * point[j])  # noqa: PERF401
                for i in range(k):
                    x_row.append(point[i] ** 2)  # noqa: PERF401

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
            title=f"Prediction Variance: {factor_x} x {factor_y}",
            x_title=factor_x,
            y_title=factor_y,
        )

        return ChartSpec(
            panels=[panel],
            title=f"Scaled Prediction Variance: {factor_x} x {factor_y}",
            plot_type="prediction_variance",
        )
