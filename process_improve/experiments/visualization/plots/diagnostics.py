"""Residual diagnostic plots and Box-Cox transformation plot.

The four diagnostic plots assess model adequacy:

- Residuals vs fitted: check for patterns (should be random scatter)
- Normal probability: check normality of residuals
- Residuals vs order: check for time-dependent patterns
- Box-Cox: find the optimal power transformation

All consume the ``residual_diagnostics`` or ``box_cox`` keys from
:func:`~process_improve.experiments.analysis.analyze_experiment`.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from process_improve.experiments.visualization.colors import DIAGNOSTIC_COLORS, DOE_PALETTE
from process_improve.experiments.visualization.plots.registry import BasePlot, register_plot
from process_improve.experiments.visualization.spec import (
    Annotation,
    ChartSpec,
    Encoding,
    LayerSpec,
    PanelSpec,
)
from process_improve.experiments.visualization.types import AnnotationType, MarkType

# ---------------------------------------------------------------------------
# Residuals vs Fitted
# ---------------------------------------------------------------------------


@register_plot("residuals_vs_fitted")
class ResidualsVsFittedPlot(BasePlot):
    """Residuals vs fitted values plot.

    A random scatter with no pattern indicates a good model.  Funnelling
    suggests heteroscedasticity; curvature suggests a missing term.

    Data sources
    ------------
    Requires ``analysis_results`` with ``"residual_diagnostics"`` key
    containing ``residuals`` and ``fitted_values`` arrays.
    """

    def to_spec(self) -> ChartSpec:
        """Build a residuals-vs-fitted ChartSpec.

        Returns
        -------
        ChartSpec
        """
        diag = self._get_residual_diagnostics()
        residuals = diag.get("residuals", [])
        fitted = diag.get("fitted_values", [])

        if not residuals or not fitted:
            return ChartSpec(title="Residuals vs Fitted — no data")

        # Scatter layer
        scatter_data = [
            {"fitted": f, "residual": r}
            for f, r in zip(fitted, residuals)  # noqa: B905
        ]
        scatter_layer = LayerSpec(
            mark=MarkType.scatter,
            data=scatter_data,
            x=Encoding(field="fitted", title="Fitted Values"),
            y=Encoding(field="residual", title="Residuals"),
            name="Residuals",
            color=DIAGNOSTIC_COLORS["residual"],
            style={"size": 8},
        )

        # Zero reference line
        annotations = [
            Annotation(
                annotation_type=AnnotationType.reference_line,
                axis="y",
                value=0.0,
                label="",
                style={"color": DIAGNOSTIC_COLORS["reference_line"], "dash": "dash", "width": 1},
            ),
        ]

        panel = PanelSpec(
            layers=[scatter_layer],
            annotations=annotations,
            title="Residuals vs Fitted Values",
            x_title="Fitted Values",
            y_title="Residuals",
        )

        return ChartSpec(
            panels=[panel],
            title="Residuals vs Fitted Values",
            plot_type="residuals_vs_fitted",
        )


# ---------------------------------------------------------------------------
# Normal Probability Plot
# ---------------------------------------------------------------------------


@register_plot("normal_probability")
class NormalProbabilityPlot(BasePlot):
    """Normal probability (Q-Q) plot of residuals.

    Points should fall approximately on the reference line if residuals
    are normally distributed.

    Data sources
    ------------
    Requires ``analysis_results`` with ``"residual_diagnostics"`` key
    containing ``residuals``.
    """

    def to_spec(self) -> ChartSpec:
        """Build a normal-probability ChartSpec.

        Returns
        -------
        ChartSpec
        """
        diag = self._get_residual_diagnostics()
        residuals = diag.get("residuals", [])

        if not residuals:
            return ChartSpec(title="Normal Probability Plot — no data")

        # scipy.stats.probplot returns (quantiles, ordered_residuals)
        (theoretical, ordered), (slope, intercept, _r) = stats.probplot(residuals, dist="norm")

        scatter_data = [
            {"theoretical": float(t), "ordered": float(o)}
            for t, o in zip(theoretical, ordered)  # noqa: B905
        ]
        scatter_layer = LayerSpec(
            mark=MarkType.scatter,
            data=scatter_data,
            x=Encoding(field="theoretical", title="Theoretical Quantiles"),
            y=Encoding(field="ordered", title="Ordered Residuals"),
            name="Residuals",
            color=DIAGNOSTIC_COLORS["residual"],
            style={"size": 8},
        )

        # Reference line
        t_min, t_max = min(theoretical), max(theoretical)
        ref_data = [
            {"theoretical": float(t_min), "ordered": intercept + slope * t_min},
            {"theoretical": float(t_max), "ordered": intercept + slope * t_max},
        ]
        ref_layer = LayerSpec(
            mark=MarkType.line,
            data=ref_data,
            x=Encoding(field="theoretical"),
            y=Encoding(field="ordered"),
            name="Reference Line",
            color=DIAGNOSTIC_COLORS["normal_line"],
            style={"dash": "dash"},
        )

        panel = PanelSpec(
            layers=[scatter_layer, ref_layer],
            title="Normal Probability Plot",
            x_title="Theoretical Quantiles",
            y_title="Ordered Residuals",
        )

        return ChartSpec(
            panels=[panel],
            title="Normal Probability Plot",
            plot_type="normal_probability",
        )


# ---------------------------------------------------------------------------
# Residuals vs Run Order
# ---------------------------------------------------------------------------


@register_plot("residuals_vs_order")
class ResidualsVsOrderPlot(BasePlot):
    """Residuals vs run order plot.

    Time-dependent patterns (trends, cycles) indicate violated
    independence assumptions.

    Data sources
    ------------
    Requires ``analysis_results`` with ``"residual_diagnostics"`` key
    containing ``residuals``.
    """

    def to_spec(self) -> ChartSpec:
        """Build a residuals-vs-order ChartSpec.

        Returns
        -------
        ChartSpec
        """
        diag = self._get_residual_diagnostics()
        residuals = diag.get("residuals", [])

        if not residuals:
            return ChartSpec(title="Residuals vs Order — no data")

        scatter_data = [
            {"run_order": i + 1, "residual": float(r)}
            for i, r in enumerate(residuals)
        ]
        scatter_layer = LayerSpec(
            mark=MarkType.scatter,
            data=scatter_data,
            x=Encoding(field="run_order", title="Run Order"),
            y=Encoding(field="residual", title="Residuals"),
            name="Residuals",
            color=DIAGNOSTIC_COLORS["residual"],
            style={"size": 8},
        )

        # Connect points with thin line to show trends
        line_layer = LayerSpec(
            mark=MarkType.line,
            data=scatter_data,
            x=Encoding(field="run_order"),
            y=Encoding(field="residual"),
            name="Trend",
            color=DIAGNOSTIC_COLORS["residual"],
            opacity=0.4,
            style={"width": 1},
        )

        annotations = [
            Annotation(
                annotation_type=AnnotationType.reference_line,
                axis="y",
                value=0.0,
                label="",
                style={"color": DIAGNOSTIC_COLORS["reference_line"], "dash": "dash", "width": 1},
            ),
        ]

        panel = PanelSpec(
            layers=[line_layer, scatter_layer],
            annotations=annotations,
            title="Residuals vs Run Order",
            x_title="Run Order",
            y_title="Residuals",
        )

        return ChartSpec(
            panels=[panel],
            title="Residuals vs Run Order",
            plot_type="residuals_vs_order",
        )


# ---------------------------------------------------------------------------
# Box-Cox Transformation Plot
# ---------------------------------------------------------------------------


@register_plot("box_cox")
class BoxCoxPlot(BasePlot):
    """Box-Cox log-likelihood profile plot.

    Shows the log-likelihood as a function of lambda with a confidence
    interval.  The optimal lambda is at the peak.

    Data sources
    ------------
    Requires ``design_data`` with ``response_column`` (all positive
    response values), or ``analysis_results`` containing
    ``residual_diagnostics`` with residuals and fitted values.
    """

    def to_spec(self) -> ChartSpec:
        """Build a Box-Cox ChartSpec.

        Returns
        -------
        ChartSpec
        """
        # Get response values
        y = self._get_response_values()
        if y is None or len(y) < 3:
            return ChartSpec(title="Box-Cox Plot — insufficient data")

        if np.any(np.array(y) <= 0):
            return ChartSpec(title="Box-Cox Plot — requires all positive response values")

        y_arr = np.array(y, dtype=float)

        # Compute log-likelihood over a range of lambdas
        lambdas = np.linspace(-3, 3, 200)
        log_likes = []
        n = len(y_arr)
        for lam in lambdas:
            if abs(lam) < 1e-10:  # noqa: SIM108
                y_t = np.log(y_arr)
            else:
                y_t = (y_arr**lam - 1.0) / lam
            # Log-likelihood (proportional)
            ss = float(np.sum((y_t - y_t.mean()) ** 2))
            ll = -n / 2.0 * np.log(ss / n) + (lam - 1) * np.sum(np.log(y_arr))
            log_likes.append(float(ll))

        # Optimal lambda
        best_idx = int(np.argmax(log_likes))
        best_lambda = float(lambdas[best_idx])
        best_ll = log_likes[best_idx]

        # 95% CI: log-likelihood within chi2(1)/2 of maximum
        from scipy.stats import chi2  # noqa: PLC0415

        ci_threshold = best_ll - chi2.ppf(self.confidence_level, 1) / 2.0

        # Build data
        line_data = [
            {"lambda": float(l), "log_likelihood": ll}
            for l, ll in zip(lambdas, log_likes)  # noqa: B905, E741
        ]
        line_layer = LayerSpec(
            mark=MarkType.line,
            data=line_data,
            x=Encoding(field="lambda", title="λ"),
            y=Encoding(field="log_likelihood", title="Log-Likelihood"),
            name="Log-Likelihood",
            color=DOE_PALETTE["primary"],
        )

        # Optimal point
        opt_data = [{"lambda": best_lambda, "log_likelihood": best_ll}]
        opt_layer = LayerSpec(
            mark=MarkType.scatter,
            data=opt_data,
            x=Encoding(field="lambda"),
            y=Encoding(field="log_likelihood"),
            name=f"Optimal λ = {best_lambda:.2f}",
            color=DOE_PALETTE["negative"],
            style={"size": 12, "symbol": "diamond"},
        )

        # Annotations: optimal lambda line + CI threshold
        annotations = [
            Annotation(
                annotation_type=AnnotationType.reference_line,
                axis="x",
                value=best_lambda,
                label=f"λ = {best_lambda:.2f}",
                style={"color": DOE_PALETTE["negative"], "dash": "dash", "width": 1},
            ),
            Annotation(
                annotation_type=AnnotationType.reference_line,
                axis="y",
                value=ci_threshold,
                label=f"{self.confidence_level * 100:.0f}% CI",
                style={"color": DOE_PALETTE["zero_line"], "dash": "dot", "width": 1},
            ),
        ]

        # Common lambda reference lines
        for special_lam, label in [(-1, "1/y"), (0, "ln(y)"), (0.5, "√y"), (1, "y")]:
            annotations.append(
                Annotation(
                    annotation_type=AnnotationType.reference_line,
                    axis="x",
                    value=special_lam,
                    label=label,
                    style={"color": DOE_PALETTE["grid"], "dash": "dot", "width": 1},
                )
            )

        panel = PanelSpec(
            layers=[line_layer, opt_layer],
            annotations=annotations,
            title="Box-Cox Transformation",
            x_title="λ (Power Parameter)",
            y_title="Log-Likelihood",
        )

        return ChartSpec(
            panels=[panel],
            title="Box-Cox Transformation",
            plot_type="box_cox",
            metadata={"optimal_lambda": best_lambda},
        )

    def _get_response_values(self) -> list[float] | None:
        """Extract response values from design data or analysis results."""
        if self.design_data and self.response_column:
            import pandas as pd  # noqa: PLC0415

            df = pd.DataFrame(self.design_data)
            if self.response_column in df.columns:
                return [float(v) for v in df[self.response_column].values]

        # Try to reconstruct from fitted + residuals
        diag = self._get_residual_diagnostics()
        fitted = diag.get("fitted_values", [])
        residuals = diag.get("residuals", [])
        if fitted and residuals and len(fitted) == len(residuals):
            return [f + r for f, r in zip(fitted, residuals)]  # noqa: B905

        return None
