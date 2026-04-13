"""Design-quality plots: FDS (fraction of design space) and power curve.

These plots assess the quality of an experimental design *before*
running experiments.  They help practitioners evaluate whether the
design provides adequate coverage and statistical power.
"""

from __future__ import annotations

import numpy as np

from process_improve.experiments.visualization.colors import DOE_PALETTE
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
# FDS (Fraction of Design Space) plot
# ---------------------------------------------------------------------------


@register_plot("fds_plot")
class FDSPlot(BasePlot):
    """Fraction of Design Space (FDS) plot.

    Shows the cumulative distribution of scaled prediction variance
    (SPV) across the design space.  The x-axis is the fraction of the
    space (0 to 1), and the y-axis is the SPV value at that fraction.
    A good design has low SPV across most of the design space.

    Data sources
    ------------
    Requires ``design_data`` with factor columns (coded values).
    """

    def to_spec(self) -> ChartSpec:  # noqa: C901, PLR0912
        """Build an FDS ChartSpec.

        Returns
        -------
        ChartSpec
        """
        import pandas as pd  # noqa: PLC0415

        if not self.design_data:
            return ChartSpec(title="FDS Plot — no design data")

        df = pd.DataFrame(self.design_data)
        factors = self.factors_to_plot or [
            c for c in df.columns if c != self.response_column
        ]
        if len(factors) < 2:
            return ChartSpec(title="FDS Plot — need at least 2 factors")

        # Build model matrix (intercept + linear + interactions + quadratic)
        x_design = df[factors].values.astype(float)
        n, k = x_design.shape

        x_terms = [np.ones(n)]
        for i in range(k):
            x_terms.append(x_design[:, i])  # noqa: PERF401
        for i in range(k):
            for j in range(i + 1, k):
                x_terms.append(x_design[:, i] * x_design[:, j])  # noqa: PERF401
        for i in range(k):
            x_terms.append(x_design[:, i] ** 2)  # noqa: PERF401

        x_model = np.column_stack(x_terms)

        try:
            xtx_inv = np.linalg.inv(x_model.T @ x_model)
        except np.linalg.LinAlgError:
            xtx_inv = np.linalg.pinv(x_model.T @ x_model)

        # Sample random points in the design space and compute SPV
        n_samples = 5000
        rng = np.random.default_rng(42)
        random_points = rng.uniform(-1, 1, size=(n_samples, k))

        spv_values = []
        for point in random_points:
            # Build model row for this point
            x_row = [1.0]
            for i in range(k):
                x_row.append(point[i])  # noqa: PERF401
            for i in range(k):
                for j in range(i + 1, k):
                    x_row.append(point[i] * point[j])  # noqa: PERF401
            for i in range(k):
                x_row.append(point[i] ** 2)  # noqa: PERF401

            x_vec = np.array(x_row)
            spv = float(n * x_vec @ xtx_inv @ x_vec)
            spv_values.append(spv)

        # Sort SPV values for CDF
        spv_sorted = sorted(spv_values)
        fractions = [(i + 1) / n_samples for i in range(n_samples)]

        # Downsample for plotting (every 50th point)
        step = max(1, n_samples // 100)
        plot_data = [
            {"fraction": fractions[i], "spv": spv_sorted[i]}
            for i in range(0, n_samples, step)
        ]
        # Ensure last point included
        if plot_data[-1]["fraction"] != fractions[-1]:
            plot_data.append({"fraction": fractions[-1], "spv": spv_sorted[-1]})

        fds_layer = LayerSpec(
            mark=MarkType.line,
            data=plot_data,
            x=Encoding(field="fraction", title="Fraction of Design Space"),
            y=Encoding(field="spv", title="Scaled Prediction Variance"),
            name="FDS",
            color=DOE_PALETTE["primary"],
            style={"width": 2},
        )

        # Reference lines at common thresholds
        median_spv = spv_sorted[n_samples // 2]
        annotations = [
            Annotation(
                annotation_type=AnnotationType.reference_line,
                axis="y",
                value=float(median_spv),
                label=f"Median SPV = {median_spv:.2f}",
                style={"color": DOE_PALETTE["zero_line"], "dash": "dash", "width": 1},
            ),
            Annotation(
                annotation_type=AnnotationType.reference_line,
                axis="x",
                value=0.5,
                label="50%",
                style={"color": DOE_PALETTE["grid"], "dash": "dot", "width": 1},
            ),
        ]

        panel = PanelSpec(
            layers=[fds_layer],
            annotations=annotations,
            title="Fraction of Design Space Plot",
            x_title="Fraction of Design Space",
            y_title="Scaled Prediction Variance (n·Var/σ²)",
        )

        return ChartSpec(
            panels=[panel],
            title="Fraction of Design Space (FDS) Plot",
            plot_type="fds_plot",
            metadata={
                "n_points": n,
                "n_factors": k,
                "median_spv": float(median_spv),
                "max_spv": float(spv_sorted[-1]),
            },
        )


# ---------------------------------------------------------------------------
# Power curve plot
# ---------------------------------------------------------------------------


@register_plot("power_curve")
class PowerCurvePlot(BasePlot):
    """Statistical power curve for a factorial design.

    Shows the probability of detecting an effect of a given size as
    a function of the signal-to-noise ratio (Δ/σ).  Helps determine
    whether a design has sufficient runs to detect practically
    important effects.

    Data sources
    ------------
    Requires ``design_data`` with factor columns, or ``analysis_results``
    with design information (``n_runs``, ``n_factors``).
    """  # noqa: RUF002

    def to_spec(self) -> ChartSpec:
        """Build a power curve ChartSpec.

        Returns
        -------
        ChartSpec
        """
        from scipy.stats import f as f_dist  # noqa: PLC0415

        design_info = self._get_design_info()
        if design_info is None:
            return ChartSpec(title="Power Curve — no design information")

        n_runs = design_info["n_runs"]
        n_terms = design_info["n_terms"]
        alpha = 1.0 - self.confidence_level

        # Degrees of freedom
        df1 = 1  # Single effect test
        df2 = max(1, n_runs - n_terms)  # Residual df

        # Signal-to-noise ratios (delta/sigma)
        sn_ratios = np.linspace(0, 4, 100)

        # F critical value
        f_crit = float(f_dist.ppf(1 - alpha, df1, df2))

        # Power for each signal-to-noise ratio
        # Non-centrality parameter: lambda = n * (delta/sigma)^2 / 4
        # for a 2-level design with n runs
        power_values = []
        for sn in sn_ratios:
            ncp = n_runs * sn**2 / 4.0
            power = 1.0 - float(f_dist.cdf(f_crit, df1, df2, loc=0, scale=1)  # noqa: RUF034
                                if ncp == 0
                                else f_dist.cdf(f_crit, df1, df2, loc=0, scale=1))
            # Use non-central F
            if ncp > 0:
                from scipy.stats import ncf  # noqa: PLC0415
                power = 1.0 - float(ncf.cdf(f_crit, df1, df2, ncp))
            else:
                power = alpha
            power_values.append(power)

        plot_data = [
            {"sn_ratio": float(sn), "power": float(p)}
            for sn, p in zip(sn_ratios, power_values)  # noqa: B905
        ]

        power_layer = LayerSpec(
            mark=MarkType.line,
            data=plot_data,
            x=Encoding(field="sn_ratio", title="Signal-to-Noise Ratio (Δ/σ)"),  # noqa: RUF001
            y=Encoding(field="power", title="Power"),
            name=f"n={n_runs}, df={df2}",
            color=DOE_PALETTE["primary"],
            style={"width": 2},
        )

        # Reference lines
        annotations = [
            Annotation(
                annotation_type=AnnotationType.reference_line,
                axis="y",
                value=0.8,
                label="Power = 0.80",
                style={"color": DOE_PALETTE["positive"], "dash": "dash", "width": 1},
            ),
            Annotation(
                annotation_type=AnnotationType.reference_line,
                axis="y",
                value=alpha,
                label=f"α = {alpha:.2f}",  # noqa: RUF001
                style={"color": DOE_PALETTE["negative"], "dash": "dot", "width": 1},
            ),
        ]

        panel = PanelSpec(
            layers=[power_layer],
            annotations=annotations,
            title="Power Curve",
            x_title="Signal-to-Noise Ratio (Δ/σ)",  # noqa: RUF001
            y_title="Power (1 − β)",  # noqa: RUF001
        )

        return ChartSpec(
            panels=[panel],
            title="Power Curve",
            plot_type="power_curve",
            metadata={
                "n_runs": n_runs,
                "n_terms": n_terms,
                "residual_df": df2,
                "alpha": alpha,
            },
        )

    def _get_design_info(self) -> dict[str, int] | None:
        """Extract design size information."""
        import pandas as pd  # noqa: PLC0415

        if self.design_data:
            df = pd.DataFrame(self.design_data)
            factors = self.factors_to_plot or [
                c for c in df.columns if c != self.response_column
            ]
            n_runs = len(df)
            k = len(factors)
            # Full second-order model terms: intercept + k + k*(k-1)/2 + k
            n_terms = 1 + k + k * (k - 1) // 2 + k
            return {"n_runs": n_runs, "n_terms": n_terms}

        if self.analysis_results:
            summary = self.analysis_results.get("model_summary", {})
            n_runs = summary.get("n_obs", 0)
            n_terms = summary.get("n_params", 0)
            if n_runs > 0 and n_terms > 0:
                return {"n_runs": n_runs, "n_terms": n_terms}

        return None
