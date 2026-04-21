"""Cube plot: 3D wireframe with response values at 2^3 design vertices.

A cube plot displays the predicted (or observed) response at each
corner of the design cube for a 3-factor experiment.  Fully custom —
no charting library provides this natively.
"""

from __future__ import annotations

import itertools

from process_improve.experiments.visualization.plots.registry import BasePlot, register_plot
from process_improve.experiments.visualization.plots.surfaces import _build_coef_map, _evaluate_model
from process_improve.visualization.colors import DOE_PALETTE
from process_improve.visualization.spec import (
    ChartSpec,
    Encoding,
    LayerSpec,
    PanelSpec,
)
from process_improve.visualization.types import MarkType


@register_plot("cube_plot")
class CubePlot(BasePlot):
    """Cube plot for 3-factor designs.

    Displays a 3D cube with predicted response values at each of the
    8 vertices (2^3 = 8 corners).  The cube wireframe is drawn with
    :class:`MarkType.wireframe` (``go.Scatter3d`` in Plotly,
    ``scatter3D`` in ECharts).

    Data sources
    ------------
    Requires ``analysis_results`` with ``"coefficients"`` key, or
    ``design_data`` with ``response_column`` and exactly 3 factors.
    """

    def to_spec(self) -> ChartSpec:
        """Build a cube plot ChartSpec.

        Returns
        -------
        ChartSpec
        """
        factors = self.factors_to_plot or self._get_factor_names()
        if len(factors) < 3:
            return ChartSpec(title="Cube Plot — need exactly 3 factors")

        f1, f2, f3 = factors[0], factors[1], factors[2]

        # Get response values at all 8 vertices
        vertices = list(itertools.product([-1, 1], repeat=3))
        vertex_values = self._get_vertex_values(f1, f2, f3, vertices)

        if vertex_values is None:
            return ChartSpec(title="Cube Plot — cannot compute vertex values")

        # Build vertex data with labels
        vertex_data = []
        for (x, y, z), val in zip(vertices, vertex_values):  # noqa: B905
            vertex_data.append({
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "text": f"{val:.2f}",
                "value": val,
            })

        vertex_layer = LayerSpec(
            mark=MarkType.wireframe,
            data=vertex_data,
            x=Encoding(field="x", title=f1),
            y=Encoding(field="y", title=f2),
            z=Encoding(field="z", title=f3),
            name="Vertices",
            color=DOE_PALETTE["primary"],
            style={"size": 8},
        )

        # Edge connections: 12 edges of a cube
        edges = [
            # Bottom face
            (0, 1), (2, 3), (0, 2), (1, 3),
            # Top face
            (4, 5), (6, 7), (4, 6), (5, 7),
            # Verticals
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]

        edge_data = []
        for i, j in edges:
            v1 = vertices[i]
            v2 = vertices[j]
            edge_data.append({"x": float(v1[0]), "y": float(v1[1]), "z": float(v1[2]), "text": ""})
            edge_data.append({"x": float(v2[0]), "y": float(v2[1]), "z": float(v2[2]), "text": ""})
            # Add None separator for Plotly (NaN for ECharts)
            edge_data.append({"x": None, "y": None, "z": None, "text": ""})

        edge_layer = LayerSpec(
            mark=MarkType.wireframe,
            data=edge_data,
            x=Encoding(field="x"),
            y=Encoding(field="y"),
            z=Encoding(field="z"),
            name="Edges",
            color=DOE_PALETTE["grid"],
            style={"width": 1, "size": 0},
        )

        panel = PanelSpec(
            layers=[edge_layer, vertex_layer],
            title=f"Cube Plot: {f1} x {f2} x {f3}",
            x_title=f1,
            y_title=f2,
            z_title=f3,
        )

        return ChartSpec(
            panels=[panel],
            title=f"Cube Plot: {f1} x {f2} x {f3}",
            plot_type="cube_plot",
            metadata={
                "factors": [f1, f2, f3],
                "vertices": [
                    {"levels": list(v), "value": val}
                    for v, val in zip(vertices, vertex_values)  # noqa: B905
                ],
                "requires_gl": True,
            },
        )

    def _get_vertex_values(
        self,
        f1: str,
        f2: str,
        f3: str,
        vertices: list[tuple[int, ...]],
    ) -> list[float] | None:
        """Compute response values at cube vertices.

        Tries model prediction first, then raw data lookup.
        """
        # Try model-based prediction
        coefficients = self._get_coefficients()
        if coefficients:
            coef_map = _build_coef_map(coefficients)
            values = []
            for v in vertices:
                point = dict(self.hold_values)
                point[f1] = float(v[0])
                point[f2] = float(v[1])
                point[f3] = float(v[2])
                values.append(_evaluate_model(coef_map, point))
            return values

        # Try raw data lookup
        if self.design_data and self.response_column:
            import pandas as pd  # noqa: PLC0415

            df = pd.DataFrame(self.design_data)
            if all(c in df.columns for c in [f1, f2, f3, self.response_column]):
                values = []
                for v in vertices:
                    mask = (df[f1] == v[0]) & (df[f2] == v[1]) & (df[f3] == v[2])
                    subset = df[mask]
                    if len(subset) > 0:
                        values.append(float(subset[self.response_column].mean()))
                    else:
                        values.append(float("nan"))
                return values

        return None
