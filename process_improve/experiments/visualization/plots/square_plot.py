"""Square plot: 2D box with response values at the four 2^2 design vertices.

A square plot is the 2-factor analogue of :class:`CubePlot`: it displays
the predicted (or observed) response at each corner of a 2x2 design.
Common in introductory DoE worksheets where readers reason about main
effects and interactions by inspecting the four corner responses.
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


@register_plot("square_plot")
class SquarePlot(BasePlot):
    """Square plot for 2-factor designs.

    Displays a 2D square with predicted response values at each of the
    four vertices of a 2x2 design.  The four edges are drawn as straight
    lines; each corner is annotated with its response value.

    Data sources
    ------------
    Requires ``analysis_results`` with a ``"coefficients"`` key, or
    ``design_data`` with ``response_column`` and exactly 2 factors.
    """

    def to_spec(self) -> ChartSpec:
        """Build a square plot ChartSpec.

        Returns
        -------
        ChartSpec
        """
        factors = self.factors_to_plot or self._get_factor_names()
        if len(factors) < 2:
            return ChartSpec(title="Square Plot - need exactly 2 factors")

        f1, f2 = factors[0], factors[1]

        vertices = list(itertools.product([-1, 1], repeat=2))
        vertex_values = self._get_vertex_values(f1, f2, vertices)

        if vertex_values is None:
            return ChartSpec(title="Square Plot - cannot compute vertex values")

        vertex_data = [
            {
                "x": float(x),
                "y": float(y),
                "text": f"{val:.2f}",
                "value": val,
            }
            for (x, y), val in zip(vertices, vertex_values)  # noqa: B905
        ]

        vertex_layer = LayerSpec(
            mark=MarkType.scatter,
            data=vertex_data,
            x=Encoding(field="x", title=f1),
            y=Encoding(field="y", title=f2),
            name="Vertices",
            color=DOE_PALETTE["primary"],
            style={"size": 12, "show_text": True},
        )

        # Edge order: traverse the square ((-1,-1) -> (1,-1) -> (1,1) -> (-1,1) -> close)
        edge_order = [(-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)]
        edge_data = [
            {"x": float(x), "y": float(y), "text": ""}
            for x, y in edge_order
        ]

        edge_layer = LayerSpec(
            mark=MarkType.line,
            data=edge_data,
            x=Encoding(field="x"),
            y=Encoding(field="y"),
            name="Edges",
            color=DOE_PALETTE["grid"],
            style={"width": 1, "size": 0},
        )

        panel = PanelSpec(
            layers=[edge_layer, vertex_layer],
            title=f"Square Plot: {f1} x {f2}",
            x_title=f1,
            y_title=f2,
        )

        return ChartSpec(
            panels=[panel],
            title=f"Square Plot: {f1} x {f2}",
            plot_type="square_plot",
            metadata={
                "factors": [f1, f2],
                "vertices": [
                    {"levels": list(v), "value": val}
                    for v, val in zip(vertices, vertex_values)  # noqa: B905
                ],
            },
        )

    def _get_vertex_values(
        self,
        f1: str,
        f2: str,
        vertices: list[tuple[int, ...]],
    ) -> list[float] | None:
        """Compute response values at the four square vertices.

        Tries model prediction first, then raw-data lookup, mirroring
        :class:`CubePlot._get_vertex_values`.
        """
        coefficients = self._get_coefficients()
        if coefficients:
            coef_map = _build_coef_map(coefficients)
            values = []
            for v in vertices:
                point = dict(self.hold_values)
                point[f1] = float(v[0])
                point[f2] = float(v[1])
                values.append(_evaluate_model(coef_map, point))
            return values

        if self.design_data and self.response_column:
            import pandas as pd  # noqa: PLC0415

            df = pd.DataFrame(self.design_data)
            if all(c in df.columns for c in [f1, f2, self.response_column]):
                values = []
                for v in vertices:
                    mask = (df[f1] == v[0]) & (df[f2] == v[1])
                    subset = df[mask]
                    if len(subset) > 0:
                        values.append(float(subset[self.response_column].mean()))
                    else:
                        values.append(float("nan"))
                return values

        return None
