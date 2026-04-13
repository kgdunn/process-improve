"""Factor-effect plots: main effects, interaction, and perturbation.

These plots show how each factor (and factor pairs) affect the response.
They operate on raw design data or analysis results.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from process_improve.experiments.visualization.colors import DOE_PALETTE, FACTOR_COLORS
from process_improve.experiments.visualization.plots.registry import BasePlot, register_plot
from process_improve.experiments.visualization.spec import (
    Annotation,
    ChartSpec,
    Encoding,
    LayerSpec,
    PanelSpec,
)
from process_improve.experiments.visualization.types import AnnotationType, MarkType, ScaleType


# ---------------------------------------------------------------------------
# Main-effects plot
# ---------------------------------------------------------------------------


@register_plot("main_effects")
class MainEffectsPlot(BasePlot):
    """Main-effects plot: mean response at each factor level.

    For each factor, plots the average response at the low (-1) and
    high (+1) levels.  A flat line indicates no effect; a steep line
    indicates a large effect.

    Data sources
    ------------
    Requires ``design_data`` with ``response_column``, or
    ``analysis_results`` containing ``effects``.
    """

    def to_spec(self) -> ChartSpec:
        """Build a main-effects ChartSpec.

        Returns
        -------
        ChartSpec
        """
        df = self._get_design_df()
        if df is None or df.empty:
            return ChartSpec(title="Main Effects Plot — no data")

        response = self.response_column or self._infer_response(df)
        if response is None or response not in df.columns:
            return ChartSpec(title="Main Effects Plot — no response column")

        factors = self.factors_to_plot or self._get_factor_names()
        if not factors:
            factor_cols = [c for c in df.columns if c != response]
            factors = factor_cols

        # Calculate means at each level per factor
        layers: list[LayerSpec] = []
        for i, factor in enumerate(factors):
            if factor not in df.columns:
                continue
            levels = sorted(df[factor].unique())
            means = [float(df[df[factor] == lv][response].mean()) for lv in levels]
            level_strs = [str(lv) for lv in levels]

            data = [
                {"level": ls, "mean_response": m, "factor": factor}
                for ls, m in zip(level_strs, means)
            ]

            color = FACTOR_COLORS[i % len(FACTOR_COLORS)]
            layer = LayerSpec(
                mark=MarkType.line,
                data=data,
                x=Encoding(field="level", title="Factor Level", scale=ScaleType.category),
                y=Encoding(field="mean_response", title=f"Mean {response}"),
                name=factor,
                color=color,
                style={"show_points": True, "width": 2},
            )
            layers.append(layer)

        # Grand mean reference line
        grand_mean = float(df[response].mean())
        annotations = [
            Annotation(
                annotation_type=AnnotationType.reference_line,
                axis="y",
                value=grand_mean,
                label=f"Grand Mean = {grand_mean:.2f}",
                style={"color": DOE_PALETTE["zero_line"], "dash": "dot", "width": 1},
            ),
        ]

        panel = PanelSpec(
            layers=layers,
            annotations=annotations,
            title="Main Effects Plot",
            x_title="Factor Level",
            y_title=f"Mean {response}",
        )

        return ChartSpec(
            panels=[panel],
            title="Main Effects Plot",
            plot_type="main_effects",
        )

    def _get_design_df(self) -> pd.DataFrame | None:
        """Build a DataFrame from design_data or analysis_results."""
        if self.design_data:
            return pd.DataFrame(self.design_data)
        return None

    def _infer_response(self, df: pd.DataFrame) -> str | None:
        """Guess the response column (last column or 'y')."""
        if "y" in df.columns:
            return "y"
        # Last column that isn't a typical factor name
        return df.columns[-1]


# ---------------------------------------------------------------------------
# Interaction plot
# ---------------------------------------------------------------------------


@register_plot("interaction")
class InteractionPlot(BasePlot):
    """Interaction plot for two factors.

    For each level of factor A, plots the mean response across levels
    of factor B.  Non-parallel lines indicate an interaction effect.

    Data sources
    ------------
    Requires ``design_data`` with ``response_column`` and
    ``factors_to_plot`` (exactly 2 factors).
    """

    def to_spec(self) -> ChartSpec:
        """Build an interaction ChartSpec.

        Returns
        -------
        ChartSpec
        """
        df = self._get_design_df()
        if df is None or df.empty:
            return ChartSpec(title="Interaction Plot — no data")

        response = self.response_column or self._infer_response(df)
        if response is None or response not in df.columns:
            return ChartSpec(title="Interaction Plot — no response column")

        factors = self.factors_to_plot or self._get_factor_names()
        if len(factors) < 2:
            return ChartSpec(title="Interaction Plot — need at least 2 factors")

        factor_a, factor_b = factors[0], factors[1]
        if factor_a not in df.columns or factor_b not in df.columns:
            return ChartSpec(title=f"Interaction Plot — factors {factor_a}, {factor_b} not in data")

        levels_a = sorted(df[factor_a].unique())
        levels_b = sorted(df[factor_b].unique())

        # One line per level of factor_a, x-axis = factor_b levels
        layers: list[LayerSpec] = []
        for i, lv_a in enumerate(levels_a):
            subset = df[df[factor_a] == lv_a]
            means = []
            for lv_b in levels_b:
                cell = subset[subset[factor_b] == lv_b]
                means.append(float(cell[response].mean()) if len(cell) > 0 else None)

            data = [
                {"level_b": str(lb), "mean_response": m}
                for lb, m in zip(levels_b, means)
                if m is not None
            ]

            color = FACTOR_COLORS[i % len(FACTOR_COLORS)]
            layer = LayerSpec(
                mark=MarkType.line,
                data=data,
                x=Encoding(field="level_b", title=factor_b, scale=ScaleType.category),
                y=Encoding(field="mean_response", title=f"Mean {response}"),
                name=f"{factor_a} = {lv_a}",
                color=color,
                style={"show_points": True, "width": 2},
            )
            layers.append(layer)

        panel = PanelSpec(
            layers=layers,
            title=f"Interaction: {factor_a} × {factor_b}",
            x_title=factor_b,
            y_title=f"Mean {response}",
        )

        return ChartSpec(
            panels=[panel],
            title=f"Interaction Plot: {factor_a} × {factor_b}",
            plot_type="interaction",
        )

    def _get_design_df(self) -> pd.DataFrame | None:
        """Build a DataFrame from design_data."""
        if self.design_data:
            return pd.DataFrame(self.design_data)
        return None

    def _infer_response(self, df: pd.DataFrame) -> str | None:
        """Guess the response column."""
        if "y" in df.columns:
            return "y"
        return df.columns[-1]


# ---------------------------------------------------------------------------
# Perturbation plot
# ---------------------------------------------------------------------------


@register_plot("perturbation")
class PerturbationPlot(BasePlot):
    """Perturbation plot: one factor at a time, others held at centre.

    Sweeps each factor from -1 to +1 while holding all others at their
    centre (0 in coded units, or at ``hold_values``).  Requires a fitted
    model (coefficients from analysis results).

    Data sources
    ------------
    Requires ``analysis_results`` with ``"coefficients"`` key.
    """

    def to_spec(self) -> ChartSpec:
        """Build a perturbation ChartSpec.

        Returns
        -------
        ChartSpec
        """
        coefficients = self._get_coefficients()
        if not coefficients:
            return ChartSpec(title="Perturbation Plot — no coefficients")

        factors = self.factors_to_plot or self._get_factor_names()
        if not factors:
            return ChartSpec(title="Perturbation Plot — no factors")

        # Build a simple model evaluator from coefficients
        coef_map = {c["term"]: c["coefficient"] for c in coefficients}
        intercept = coef_map.get("Intercept", 0.0)

        sweep = np.linspace(-1, 1, 50)
        layers: list[LayerSpec] = []

        for i, factor in enumerate(factors):
            predictions = []
            for x_val in sweep:
                y_hat = intercept
                for term, coef in coef_map.items():
                    if term == "Intercept":
                        continue
                    if term == factor:
                        y_hat += coef * x_val
                    elif ":" in term:
                        # Interaction: use hold_values (default 0)
                        parts = term.split(":")
                        val = 1.0
                        for p in parts:
                            if p == factor:
                                val *= x_val
                            else:
                                val *= self.hold_values.get(p, 0.0)
                        y_hat += coef * val
                    elif term.startswith("I(") and factor in term:
                        # Quadratic term for this factor
                        y_hat += coef * x_val**2
                    else:
                        # Other main effects at hold value
                        y_hat += coef * self.hold_values.get(term, 0.0)
                predictions.append(float(y_hat))

            data = [
                {"coded_level": float(x), "predicted": p}
                for x, p in zip(sweep, predictions)
            ]
            color = FACTOR_COLORS[i % len(FACTOR_COLORS)]
            layer = LayerSpec(
                mark=MarkType.line,
                data=data,
                x=Encoding(field="coded_level", title="Coded Factor Level"),
                y=Encoding(field="predicted", title="Predicted Response"),
                name=factor,
                color=color,
                style={"width": 2},
            )
            layers.append(layer)

        panel = PanelSpec(
            layers=layers,
            title="Perturbation Plot",
            x_title="Coded Factor Level (Deviation from Centre)",
            y_title="Predicted Response",
        )

        return ChartSpec(
            panels=[panel],
            title="Perturbation Plot",
            plot_type="perturbation",
        )
