"""Optimisation plots: desirability contour, overlay, ridge trace, steepest ascent.

These plots support multi-response optimisation workflows.  They
consume fitted model coefficients, desirability goals, and constraint
specifications from :func:`~process_improve.experiments.optimization.optimize_responses`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from process_improve.experiments.visualization.colors import (
    DESIRABILITY_COLORSCALE,
    DOE_PALETTE,
    FACTOR_COLORS,
)
from process_improve.experiments.visualization.plots.registry import BasePlot, register_plot
from process_improve.experiments.visualization.plots.surfaces import _build_coef_map, _compute_grid, _evaluate_model
from process_improve.experiments.visualization.spec import (
    Annotation,
    ChartSpec,
    Encoding,
    LayerSpec,
    PanelSpec,
)
from process_improve.experiments.visualization.types import AnnotationType, MarkType

# ---------------------------------------------------------------------------
# Shared desirability helpers
# ---------------------------------------------------------------------------


def _desirability_maximize(y: float, low: float, high: float, weight: float = 1.0) -> float:
    """Individual desirability for a maximise goal."""
    if y <= low:
        return 0.0
    if y >= high:
        return 1.0
    return ((y - low) / (high - low)) ** weight


def _desirability_minimize(y: float, low: float, high: float, weight: float = 1.0) -> float:
    """Individual desirability for a minimise goal."""
    if y <= low:
        return 1.0
    if y >= high:
        return 0.0
    return ((high - y) / (high - low)) ** weight


def _desirability_target(  # noqa: PLR0913
    y: float,
    low: float,
    target: float,
    high: float,
    weight_low: float = 1.0,
    weight_high: float = 1.0,
) -> float:
    """Individual desirability for a target goal."""
    if y <= low or y >= high:
        return 0.0
    if y <= target:
        return ((y - low) / (target - low)) ** weight_low
    return ((high - y) / (high - target)) ** weight_high


def _individual_desirability(y: float, goal: dict[str, Any]) -> float:
    """Compute individual desirability for a single response."""
    goal_type = goal.get("goal", "maximize")
    weight = goal.get("weight", 1.0)

    if goal_type == "maximize":
        return _desirability_maximize(y, goal.get("low", 0.0), goal.get("high", 1.0), weight)
    if goal_type == "minimize":
        return _desirability_minimize(y, goal.get("low", 0.0), goal.get("high", 1.0), weight)
    if goal_type == "target":
        return _desirability_target(
            y,
            goal.get("low", 0.0),
            goal.get("target", 0.5),
            goal.get("high", 1.0),
            weight,
            goal.get("weight_high", weight),
        )
    return 0.0


def _composite_desirability(d_values: list[float], importances: list[float] | None = None) -> float:
    """Weighted geometric mean of individual desirabilities."""
    if not d_values:
        return 0.0
    if any(d == 0.0 for d in d_values):
        return 0.0
    weights = importances or [1.0] * len(d_values)
    total_w = sum(weights)
    product = 1.0
    for d, w in zip(d_values, weights):  # noqa: B905
        product *= d ** (w / total_w)
    return product


# ---------------------------------------------------------------------------
# Desirability contour plot
# ---------------------------------------------------------------------------


@register_plot("desirability_contour")
class DesirabilityContourPlot(BasePlot):
    """Contour plot of composite desirability over two factors.

    Evaluates multiple fitted models over a 2D grid, computes individual
    desirabilities for each response, then shows the composite
    desirability as a filled contour.

    Data sources
    ------------
    Requires ``analysis_results`` with ``"coefficients"`` and
    ``"optimization"`` keys.  The ``"optimization"`` dict should contain
    ``"responses"`` — a list of dicts each with ``"coefficients"``,
    ``"goal"``, ``"low"``, ``"high"``, and optionally ``"target"``,
    ``"weight"``, ``"importance"``.
    """

    def to_spec(self) -> ChartSpec:
        """Build a desirability contour ChartSpec.

        Returns
        -------
        ChartSpec
        """
        responses = self._get_optimization_responses()
        if not responses:
            return ChartSpec(title="Desirability Contour — no optimisation data")

        factors = self.factors_to_plot or self._get_factor_names()
        if len(factors) < 2:
            return ChartSpec(title="Desirability Contour — need at least 2 factors")

        factor_x, factor_y = factors[0], factors[1]
        n_grid = 50
        x_grid = np.linspace(-1, 1, n_grid).tolist()
        y_grid = np.linspace(-1, 1, n_grid).tolist()

        importances = [r.get("importance", 1.0) for r in responses]

        # Build coefficient maps for each response
        coef_maps = []
        for resp in responses:
            coeffs = resp.get("coefficients", [])
            coef_maps.append(_build_coef_map(coeffs))

        # Evaluate composite desirability over grid
        z_matrix: list[list[float]] = []
        for y_val in y_grid:
            row: list[float] = []
            for x_val in x_grid:
                point = dict(self.hold_values)
                point[factor_x] = x_val
                point[factor_y] = y_val

                d_values = []
                for cm, resp in zip(coef_maps, responses):  # noqa: B905
                    y_hat = _evaluate_model(cm, point)
                    d = _individual_desirability(y_hat, resp)
                    d_values.append(d)

                row.append(_composite_desirability(d_values, importances))
            z_matrix.append(row)

        contour_layer = LayerSpec(
            mark=MarkType.contour,
            data=[],
            x=Encoding(field="x", title=factor_x),
            y=Encoding(field="y", title=factor_y),
            name="Composite Desirability",
            style={
                "x_grid": x_grid,
                "y_grid": y_grid,
                "z_matrix": z_matrix,
                "colorscale": DESIRABILITY_COLORSCALE,
                "zmin": 0.0,
                "zmax": 1.0,
            },
        )

        panel = PanelSpec(
            layers=[contour_layer],
            title=f"Desirability: {factor_x} x {factor_y}",
            x_title=factor_x,
            y_title=factor_y,
        )

        return ChartSpec(
            panels=[panel],
            title=f"Desirability Contour: {factor_x} x {factor_y}",
            plot_type="desirability_contour",
            metadata={
                "factors": [factor_x, factor_y],
                "hold_values": self.hold_values,
                "n_responses": len(responses),
            },
        )

    def _get_optimization_responses(self) -> list[dict[str, Any]]:
        """Extract response definitions from analysis_results."""
        if not self.analysis_results:
            return []
        opt = self.analysis_results.get("optimization", {})
        responses = opt.get("responses", [])
        if responses:
            return responses

        # Fall back: single-response from top-level coefficients
        coefficients = self._get_coefficients()
        if coefficients:
            return [{
                "coefficients": coefficients,
                "goal": "maximize",
                "low": float("-inf"),
                "high": float("inf"),
                "weight": 1.0,
            }]
        return []


# ---------------------------------------------------------------------------
# Overlay plot
# ---------------------------------------------------------------------------


@register_plot("overlay")
class OverlayPlot(BasePlot):
    """Overlay contour plot for multiple responses.

    Draws contour lines from each response model on the same axes,
    with optional constraint regions shaded.  Useful for identifying
    a feasible operating region that satisfies all specifications.

    Data sources
    ------------
    Requires ``analysis_results`` with ``"optimization"`` containing
    ``"responses"`` — each with ``"coefficients"``, and optionally
    ``"low"``/``"high"`` bounds for feasibility shading.
    """

    def to_spec(self) -> ChartSpec:
        """Build an overlay ChartSpec.

        Returns
        -------
        ChartSpec
        """
        responses = self._get_overlay_responses()
        if not responses:
            return ChartSpec(title="Overlay Plot — no response data")

        factors = self.factors_to_plot or self._get_factor_names()
        if len(factors) < 2:
            return ChartSpec(title="Overlay Plot — need at least 2 factors")

        factor_x, factor_y = factors[0], factors[1]
        n_grid = 50
        x_grid = np.linspace(-1, 1, n_grid).tolist()
        y_grid = np.linspace(-1, 1, n_grid).tolist()

        layers: list[LayerSpec] = []

        for i, resp in enumerate(responses):
            coeffs = resp.get("coefficients", [])
            if not coeffs:
                continue

            coef_map = _build_coef_map(coeffs)
            _, _, z_matrix = _compute_grid(
                coef_map, factor_x, factor_y, self.hold_values, n_grid,
            )

            resp_name = resp.get("name", f"Response {i + 1}")
            color = FACTOR_COLORS[i % len(FACTOR_COLORS)]

            layer = LayerSpec(
                mark=MarkType.contour,
                data=[],
                x=Encoding(field="x", title=factor_x),
                y=Encoding(field="y", title=factor_y),
                name=resp_name,
                color=color,
                style={
                    "x_grid": x_grid,
                    "y_grid": y_grid,
                    "z_matrix": z_matrix,
                    "contours_coloring": "lines",
                    "ncontours": 8,
                },
            )
            layers.append(layer)

        annotations: list[Annotation] = []
        # Add constraint region annotations if bounds are specified
        for resp in responses:
            low = resp.get("low")
            high = resp.get("high")
            if low is not None and high is not None:
                resp_name = resp.get("name", "Response")
                annotations.append(Annotation(
                    annotation_type=AnnotationType.label,
                    label=f"{resp_name}: [{low:.2f}, {high:.2f}]",
                    style={"color": DOE_PALETTE["neutral"]},
                ))

        panel = PanelSpec(
            layers=layers,
            annotations=annotations,
            title=f"Overlay: {factor_x} x {factor_y}",
            x_title=factor_x,
            y_title=factor_y,
        )

        return ChartSpec(
            panels=[panel],
            title=f"Overlay Plot: {factor_x} x {factor_y}",
            plot_type="overlay",
            metadata={
                "factors": [factor_x, factor_y],
                "hold_values": self.hold_values,
                "n_responses": len(responses),
            },
        )

    def _get_overlay_responses(self) -> list[dict[str, Any]]:
        """Extract multi-response data for overlay."""
        if not self.analysis_results:
            return []
        opt = self.analysis_results.get("optimization", {})
        return opt.get("responses", [])


# ---------------------------------------------------------------------------
# Ridge trace plot
# ---------------------------------------------------------------------------


@register_plot("ridge_trace")
class RidgeTracePlot(BasePlot):
    """Ridge trace plot for response-surface models.

    Traces the optimal predicted response and factor settings along
    spheres of increasing radius from the design centre.  Useful for
    understanding how the optimum shifts as we move further from the
    centre of the design space.

    Data sources
    ------------
    Requires ``analysis_results`` with ``"coefficients"`` for a
    second-order (quadratic) model.
    """

    def to_spec(self) -> ChartSpec:  # noqa: C901
        """Build a ridge trace ChartSpec.

        Returns
        -------
        ChartSpec
        """
        coefficients = self._get_coefficients()
        if not coefficients:
            return ChartSpec(title="Ridge Trace — no coefficients")

        factors = self.factors_to_plot or self._get_factor_names()
        if not factors:
            return ChartSpec(title="Ridge Trace — no factors")

        coef_map = _build_coef_map(coefficients)
        b0, b_vec, b_mat = self._extract_b_and_B(coef_map, factors)

        radii = np.linspace(0, 1.5, 30)
        response_trace: list[float] = []
        factor_traces: dict[str, list[float]] = {f: [] for f in factors}

        for r in radii:
            if r == 0:
                # At centre
                response_trace.append(b0)
                for f in factors:
                    factor_traces[f].append(0.0)
                continue

            # Solve (B + mu*I)x = -b/2 for x on sphere ||x|| = r
            # Use eigenvalue approach: find mu such that ||x(mu)|| = r
            best_y = float("-inf")
            best_x = np.zeros(len(factors))
            for mu in np.linspace(-10, 10, 200):
                try:
                    mat = b_mat + mu * np.eye(len(factors))
                    x_opt = np.linalg.solve(mat, -0.5 * b_vec)
                    norm_x = float(np.linalg.norm(x_opt))
                    if norm_x > 0:  # noqa: SIM108
                        x_scaled = x_opt * (r / norm_x)
                    else:
                        x_scaled = x_opt
                    y_val = b0 + float(b_vec @ x_scaled) + float(x_scaled @ b_mat @ x_scaled)
                    if y_val > best_y:
                        best_y = y_val
                        best_x = x_scaled
                except np.linalg.LinAlgError:  # noqa: PERF203
                    continue

            response_trace.append(float(best_y))
            for j, f in enumerate(factors):
                factor_traces[f].append(float(best_x[j]))

        # Panel 1: Response vs radius
        resp_data = [
            {"radius": float(r), "response": float(y)}
            for r, y in zip(radii, response_trace)  # noqa: B905
        ]
        resp_layer = LayerSpec(
            mark=MarkType.line,
            data=resp_data,
            x=Encoding(field="radius", title="Radius"),
            y=Encoding(field="response", title="Predicted Response"),
            name="Response",
            color=DOE_PALETTE["primary"],
            style={"width": 2},
        )

        resp_panel = PanelSpec(
            layers=[resp_layer],
            title="Response vs Radius",
            x_title="Radius from Centre",
            y_title="Predicted Response",
        )

        # Panel 2: Factor settings vs radius
        factor_layers: list[LayerSpec] = []
        for i, f in enumerate(factors):
            fdata = [
                {"radius": float(r), "coded_level": float(v)}
                for r, v in zip(radii, factor_traces[f])  # noqa: B905
            ]
            color = FACTOR_COLORS[i % len(FACTOR_COLORS)]
            flayer = LayerSpec(
                mark=MarkType.line,
                data=fdata,
                x=Encoding(field="radius", title="Radius"),
                y=Encoding(field="coded_level", title="Coded Level"),
                name=f,
                color=color,
                style={"width": 2},
            )
            factor_layers.append(flayer)

        factor_panel = PanelSpec(
            layers=factor_layers,
            title="Factor Settings vs Radius",
            x_title="Radius from Centre",
            y_title="Coded Factor Level",
        )

        return ChartSpec(
            panels=[resp_panel, factor_panel],
            title="Ridge Trace",
            plot_type="ridge_trace",
            layout="column",
            columns=1,
            linked=True,
            metadata={"factors": factors},
        )

    def _extract_b_and_B(  # noqa: N802
        self,
        coef_map: dict[str, float],
        factors: list[str],
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Extract intercept, linear vector b, and quadratic matrix B.

        For the model y = b0 + b'x + x'Bx, extracts b0, b, and B.
        """
        k = len(factors)
        b0 = coef_map.get("Intercept", 0.0)
        b_vec = np.zeros(k)
        b_mat = np.zeros((k, k))

        for i, fi in enumerate(factors):
            b_vec[i] = coef_map.get(fi, 0.0)

        for i, fi in enumerate(factors):
            # Quadratic: I(fi ** 2)
            quad_key = f"I({fi} ** 2)"
            b_mat[i, i] = coef_map.get(quad_key, 0.0)

            # Interactions: fi:fj
            for j, fj in enumerate(factors):
                if j <= i:
                    continue
                int_key1 = f"{fi}:{fj}"
                int_key2 = f"{fj}:{fi}"
                int_val = coef_map.get(int_key1, coef_map.get(int_key2, 0.0))
                b_mat[i, j] = int_val / 2.0
                b_mat[j, i] = int_val / 2.0

        return b0, b_vec, b_mat


# ---------------------------------------------------------------------------
# Steepest ascent path plot
# ---------------------------------------------------------------------------


@register_plot("steepest_ascent_path")
class SteepestAscentPathPlot(BasePlot):
    """Steepest ascent (or descent) path visualisation.

    Shows the trajectory from the design centre along the steepest
    ascent direction derived from a first-order model.  Displays both
    the path in factor space and the predicted response along the path.

    Data sources
    ------------
    Requires ``analysis_results`` with ``"coefficients"`` (first-order
    model) or ``"steepest_path"`` pre-computed results.
    """

    def to_spec(self) -> ChartSpec:
        """Build a steepest ascent path ChartSpec.

        Returns
        -------
        ChartSpec
        """
        path_data = self._get_path_data()
        if path_data is None:
            return ChartSpec(title="Steepest Ascent — no path data")

        steps = path_data.get("steps", [])
        if not steps:
            return ChartSpec(title="Steepest Ascent — no steps computed")

        direction = path_data.get("direction", "ascent")
        direction_label = direction.capitalize()

        factors = self.factors_to_plot or self._get_factor_names()

        # Panel 1: Predicted response along path
        resp_data = [
            {"step": s["step"], "predicted": s.get("predicted_response", 0.0)}
            for s in steps
        ]
        resp_layer = LayerSpec(
            mark=MarkType.line,
            data=resp_data,
            x=Encoding(field="step", title="Step"),
            y=Encoding(field="predicted", title="Predicted Response"),
            name="Predicted",
            color=DOE_PALETTE["primary"],
            style={"width": 2, "show_points": True},
        )

        # Add actual response overlay if available
        resp_layers: list[LayerSpec] = [resp_layer]
        has_actual_response = any("actual_response" in s for s in steps)
        if has_actual_response:
            actual_data = [
                {"step": s["step"], "actual": s["actual_response"]}
                for s in steps
                if "actual_response" in s
            ]
            actual_layer = LayerSpec(
                mark=MarkType.scatter,
                data=actual_data,
                x=Encoding(field="step"),
                y=Encoding(field="actual", title="Actual Response"),
                name="Actual",
                color=DOE_PALETTE["negative"],
                style={"size": 10, "symbol": "diamond"},
            )
            resp_layers.append(actual_layer)

        resp_panel = PanelSpec(
            layers=resp_layers,
            title=f"Response Along Steepest {direction_label} Path",
            x_title="Step Number",
            y_title="Response",
        )

        # Panel 2: Factor trajectories
        factor_layers: list[LayerSpec] = []
        # Determine which factors to show
        path_factors = factors
        if not path_factors and steps:
            coded = steps[0].get("coded", {})
            path_factors = list(coded.keys())

        for i, f in enumerate(path_factors):
            fdata = [
                {"step": s["step"], "coded_level": s.get("coded", {}).get(f, 0.0)}
                for s in steps
            ]
            color = FACTOR_COLORS[i % len(FACTOR_COLORS)]
            flayer = LayerSpec(
                mark=MarkType.line,
                data=fdata,
                x=Encoding(field="step", title="Step"),
                y=Encoding(field="coded_level", title="Coded Level"),
                name=f,
                color=color,
                style={"width": 2, "show_points": True},
            )
            factor_layers.append(flayer)

        factor_panel = PanelSpec(
            layers=factor_layers,
            title="Factor Settings Along Path",
            x_title="Step Number",
            y_title="Coded Factor Level",
        )

        return ChartSpec(
            panels=[resp_panel, factor_panel],
            title=f"Steepest {direction_label} Path",
            plot_type="steepest_ascent_path",
            layout="column",
            columns=1,
            linked=True,
            metadata={
                "direction": direction,
                "n_steps": len(steps),
                "factors": path_factors,
                "direction_vector": path_data.get("direction_vector", {}),
            },
        )

    def _get_path_data(self) -> dict[str, Any] | None:
        """Extract or compute steepest path data."""
        # Try pre-computed path from analysis_results
        if self.analysis_results:
            path = self.analysis_results.get("steepest_path")
            if path:
                return path

        # Compute from first-order coefficients
        coefficients = self._get_coefficients()
        if not coefficients:
            return None

        factors = self.factors_to_plot or self._get_factor_names()
        if not factors:
            return None

        coef_map = _build_coef_map(coefficients)

        # Extract linear coefficients as direction vector
        direction_vec = {}
        for f in factors:
            direction_vec[f] = coef_map.get(f, 0.0)

        norm = sum(v**2 for v in direction_vec.values()) ** 0.5
        if norm == 0:
            return None

        # Normalise direction
        direction_norm = {f: v / norm for f, v in direction_vec.items()}

        # Generate steps
        step_size = 0.5
        n_steps = 10
        steps = []
        for step_num in range(n_steps + 1):
            coded = {f: direction_norm[f] * step_size * step_num for f in factors}
            point = dict(self.hold_values)
            point.update(coded)
            predicted = _evaluate_model(coef_map, point)
            steps.append({
                "step": step_num,
                "coded": coded,
                "predicted_response": float(predicted),
            })

        return {
            "direction": "ascent",
            "direction_vector": direction_vec,
            "step_size": step_size,
            "steps": steps,
        }
