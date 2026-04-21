"""Significance plots: Pareto, half-normal, and Daniel (normal QQ).

These plots help identify which effects are statistically significant in
unreplicated or replicated factorial experiments.  They consume the
``effects`` and/or ``lenth_method`` keys from
:func:`~process_improve.experiments.analysis.analyze_experiment`.
"""

from __future__ import annotations

from scipy import stats

from process_improve.experiments.visualization.plots.registry import BasePlot, register_plot
from process_improve.visualization.colors import DOE_PALETTE
from process_improve.visualization.spec import (
    Annotation,
    ChartSpec,
    Encoding,
    LayerSpec,
    PanelSpec,
    significance_threshold,
)
from process_improve.visualization.types import MarkType, ScaleType

# ---------------------------------------------------------------------------
# Pareto plot
# ---------------------------------------------------------------------------


@register_plot("pareto")
class ParetoPlot(BasePlot):
    """Pareto chart of absolute effects with cumulative percentage line.

    Displays horizontal bars for the absolute value of each effect,
    sorted descending, with a cumulative-percentage overlay on a
    secondary y-axis.  Optionally adds Lenth ME and SME threshold lines.

    Data sources
    ------------
    Requires ``analysis_results`` with at least ``"effects"`` key.
    If ``"lenth_method"`` is also present (with ``ME`` and ``SME``),
    significance thresholds are drawn.
    """

    def to_spec(self) -> ChartSpec:
        """Build a Pareto ChartSpec.

        Returns
        -------
        ChartSpec
        """
        effects = self._get_effects()
        lenth = self._get_lenth()

        if not effects:
            return ChartSpec(title="Pareto Chart — no effects data")

        # Sort by absolute effect (descending)
        sorted_items = sorted(effects.items(), key=lambda kv: abs(kv[1]), reverse=True)
        names = [n for n, _ in sorted_items]
        abs_vals = [abs(v) for _, v in sorted_items]

        # Cumulative percentage
        total = sum(abs_vals)
        cum_pct = []
        running = 0.0
        for v in abs_vals:
            running += v
            cum_pct.append(100.0 * running / total if total > 0 else 0.0)

        # Bar colours: green for positive, red for negative
        bar_colors = [
            DOE_PALETTE["positive"] if v >= 0 else DOE_PALETTE["negative"]
            for _, v in sorted_items
        ]

        # --- Layers ---
        bar_data = [{"name": n, "abs_effect": v} for n, v in zip(names, abs_vals)]  # noqa: B905
        bar_layer = LayerSpec(
            mark=MarkType.bar,
            data=bar_data,
            x=Encoding(field="name", scale=ScaleType.category),
            y=Encoding(field="abs_effect", title="|Effect|"),
            name="Absolute Effect",
            style={"colors": bar_colors},
        )

        cum_data = [{"name": n, "cum_pct": p} for n, p in zip(names, cum_pct)]  # noqa: B905
        cum_layer = LayerSpec(
            mark=MarkType.line,
            data=cum_data,
            x=Encoding(field="name", scale=ScaleType.category),
            y=Encoding(field="cum_pct", title="Cumulative %"),
            name="Cumulative %",
            color=DOE_PALETTE["cumulative"],
            style={"secondary_y": True, "show_points": True},
        )

        # --- Annotations ---
        annotations: list[Annotation] = []
        if lenth and self.highlight_significant:
            me = lenth.get("ME")
            sme = lenth.get("SME")
            if me is not None:
                annotations.append(significance_threshold(me, alpha=1 - self.confidence_level, name="ME"))
            if sme is not None:
                annotations.append(significance_threshold(
                    sme,
                    alpha=1 - self.confidence_level,
                    name="SME",
                    label=f"SME (α={1 - self.confidence_level})",  # noqa: RUF001
                ))
                # Override SME colour to red
                annotations[-1].style["color"] = DOE_PALETTE["threshold_sme"]

        panel = PanelSpec(
            layers=[bar_layer, cum_layer],
            annotations=annotations,
            title="Pareto Chart of Effects",
            x_title="Effect",
            y_title="|Effect|",
            secondary_y=True,
            secondary_y_title="Cumulative %",
        )

        return ChartSpec(
            panels=[panel],
            title="Pareto Chart of Effects",
            plot_type="pareto",
        )


# ---------------------------------------------------------------------------
# Half-normal plot
# ---------------------------------------------------------------------------


@register_plot("half_normal")
class HalfNormalPlot(BasePlot):
    """Half-normal probability plot of absolute effects.

    Plots ordered absolute effects against half-normal quantiles.
    Significant effects deviate from the reference line.

    Data sources
    ------------
    Requires ``analysis_results`` with ``"effects"`` key.  Uses
    ``"lenth_method"`` for the ME/SME threshold if available.
    """

    def to_spec(self) -> ChartSpec:
        """Build a half-normal ChartSpec.

        Returns
        -------
        ChartSpec
        """
        effects = self._get_effects()
        lenth = self._get_lenth()

        if not effects:
            return ChartSpec(title="Half-Normal Plot — no effects data")

        abs_effects = {n: abs(v) for n, v in effects.items()}
        sorted_items = sorted(abs_effects.items(), key=lambda kv: kv[1])
        names = [n for n, _ in sorted_items]
        abs_vals = [v for _, v in sorted_items]

        n = len(abs_vals)
        # Half-normal quantiles: use probabilities (i - 0.5) / n
        probs = [(i + 0.5) / n for i in range(n)]
        quantiles = [float(stats.halfnorm.ppf(p)) for p in probs]

        # Determine significance for colouring
        significant_set: set[str] = set()
        if lenth and self.highlight_significant:
            me = lenth.get("ME", float("inf"))
            for item in lenth.get("effects", []):
                if item.get("active_ME", False):
                    significant_set.add(item["term"])

        point_colors = [
            DOE_PALETTE["negative"] if n in significant_set else DOE_PALETTE["primary"]
            for n in names
        ]

        # --- Layers ---
        scatter_data = [
            {"quantile": q, "abs_effect": v, "name": name}
            for q, v, name in zip(quantiles, abs_vals, names)  # noqa: B905
        ]
        scatter_layer = LayerSpec(
            mark=MarkType.scatter,
            data=scatter_data,
            x=Encoding(field="quantile", title="Half-Normal Quantile"),
            y=Encoding(field="abs_effect", title="|Effect|"),
            name="Effects",
            style={"colors": point_colors, "size": 10},
        )

        # Reference line through the non-significant effects
        if len(abs_vals) > 1:
            non_sig_q = [q for q, n in zip(quantiles, names) if n not in significant_set]  # noqa: B905
            non_sig_v = [v for v, n in zip(abs_vals, names) if n not in significant_set]  # noqa: B905
            if len(non_sig_q) >= 2:
                slope = (non_sig_v[-1] - non_sig_v[0]) / (non_sig_q[-1] - non_sig_q[0]) if non_sig_q[-1] != non_sig_q[0] else 0  # noqa: E501
                intercept = non_sig_v[0] - slope * non_sig_q[0]
                line_q = [0, quantiles[-1] * 1.1]
                line_v = [intercept, intercept + slope * line_q[1]]
            else:
                line_q = [0, quantiles[-1] * 1.1]
                line_v = [0, abs_vals[-1]]
        else:
            line_q = [0, 1]
            line_v = [0, abs_vals[0] if abs_vals else 1]

        ref_data = [
            {"quantile": q, "abs_effect": v} for q, v in zip(line_q, line_v)  # noqa: B905
        ]
        ref_layer = LayerSpec(
            mark=MarkType.line,
            data=ref_data,
            x=Encoding(field="quantile"),
            y=Encoding(field="abs_effect"),
            name="Reference",
            color=DOE_PALETTE["zero_line"],
            style={"dash": "dash"},
        )

        # Label layer for effect names
        label_data = [
            {"quantile": q, "abs_effect": v, "text": name}
            for q, v, name in zip(quantiles, abs_vals, names)  # noqa: B905
            if name in significant_set
        ]
        label_layer = LayerSpec(
            mark=MarkType.text,
            data=label_data,
            x=Encoding(field="quantile"),
            y=Encoding(field="abs_effect"),
            name="Labels",
            style={"size": 11},
        )

        # --- Annotations ---
        annotations: list[Annotation] = []
        if lenth and self.highlight_significant:
            me = lenth.get("ME")
            if me is not None:
                annotations.append(significance_threshold(me, alpha=1 - self.confidence_level, name="ME"))

        panel = PanelSpec(
            layers=[scatter_layer, ref_layer, label_layer],
            annotations=annotations,
            title="Half-Normal Plot of Effects",
            x_title="Half-Normal Quantile",
            y_title="|Effect|",
        )

        return ChartSpec(
            panels=[panel],
            title="Half-Normal Plot of Effects",
            plot_type="half_normal",
        )


# ---------------------------------------------------------------------------
# Daniel (normal QQ) plot
# ---------------------------------------------------------------------------


@register_plot("daniel")
class DanielPlot(BasePlot):
    """Daniel plot: normal Q-Q plot of effects.

    Non-zero (significant) effects deviate from the straight line
    through the origin.  This is the full-normal variant of the
    half-normal plot.

    Data sources
    ------------
    Requires ``analysis_results`` with ``"effects"`` key.
    """

    def to_spec(self) -> ChartSpec:
        """Build a Daniel plot ChartSpec.

        Returns
        -------
        ChartSpec
        """
        effects = self._get_effects()
        lenth = self._get_lenth()

        if not effects:
            return ChartSpec(title="Daniel Plot — no effects data")

        sorted_items = sorted(effects.items(), key=lambda kv: kv[1])
        names = [n for n, _ in sorted_items]
        vals = [v for _, v in sorted_items]

        n = len(vals)
        probs = [(i + 0.5) / n for i in range(n)]
        quantiles = [float(stats.norm.ppf(p)) for p in probs]

        # Determine significance for colouring
        significant_set: set[str] = set()
        if lenth and self.highlight_significant:
            for item in lenth.get("effects", []):
                if item.get("active_ME", False):
                    significant_set.add(item["term"])

        point_colors = [
            DOE_PALETTE["negative"] if name in significant_set else DOE_PALETTE["primary"]
            for name in names
        ]

        # --- Layers ---
        scatter_data = [
            {"quantile": q, "effect": v, "name": name}
            for q, v, name in zip(quantiles, vals, names)  # noqa: B905
        ]
        scatter_layer = LayerSpec(
            mark=MarkType.scatter,
            data=scatter_data,
            x=Encoding(field="quantile", title="Normal Quantile"),
            y=Encoding(field="effect", title="Effect"),
            name="Effects",
            style={"colors": point_colors, "size": 10},
        )

        # Reference line (fit through all points)
        if len(vals) >= 2:
            slope, intercept, *_ = stats.linregress(quantiles, vals)
            line_q = [quantiles[0] * 1.1, quantiles[-1] * 1.1]
            line_v = [intercept + slope * q for q in line_q]
        else:
            line_q = [-2, 2]
            line_v = [vals[0], vals[0]] if vals else [0, 0]

        ref_data = [{"quantile": q, "effect": v} for q, v in zip(line_q, line_v)]  # noqa: B905
        ref_layer = LayerSpec(
            mark=MarkType.line,
            data=ref_data,
            x=Encoding(field="quantile"),
            y=Encoding(field="effect"),
            name="Reference Line",
            color=DOE_PALETTE["zero_line"],
            style={"dash": "dash"},
        )

        # Labels for significant effects
        label_data = [
            {"quantile": q, "effect": v, "text": name}
            for q, v, name in zip(quantiles, vals, names)  # noqa: B905
            if name in significant_set
        ]
        label_layer = LayerSpec(
            mark=MarkType.text,
            data=label_data,
            x=Encoding(field="quantile"),
            y=Encoding(field="effect"),
            name="Labels",
            style={"size": 11},
        )

        panel = PanelSpec(
            layers=[scatter_layer, ref_layer, label_layer],
            title="Daniel Plot (Normal Q-Q of Effects)",
            x_title="Normal Quantile",
            y_title="Effect",
        )

        return ChartSpec(
            panels=[panel],
            title="Daniel Plot (Normal Q-Q of Effects)",
            plot_type="daniel",
        )
