"""Design-quality plots: FDS (fraction of design space) and power curve.

These plots assess the quality of an experimental design *before*
running experiments.  They help practitioners evaluate whether the
design provides adequate coverage and statistical power.

Both are computed by :func:`~process_improve.experiments.evaluate_design`'s
machinery for an explicit model (see :func:`design_model`), over every factor
column of the design (``RunOrder`` and ``Block`` are not factors).
"""

from __future__ import annotations

import contextlib
from typing import Any

import numpy as np
import pandas as pd

from process_improve.experiments.evaluate import (
    _build_context,
    _coefficient_power,
    _EvalContext,
    _EvalRequest,
    _is_intercept_col,
    evaluate_design,
)
from process_improve.experiments.visualization.plots.registry import BasePlot, register_plot
from process_improve.visualization.colors import DOE_PALETTE
from process_improve.visualization.spec import (
    Annotation,
    ChartSpec,
    Encoding,
    LayerSpec,
    PanelSpec,
)
from process_improve.visualization.types import AnnotationType, MarkType

#: Bookkeeping columns of a generated design, which are not factors.
_RUN_COLUMNS = ("RunOrder", "Block")


# ---------------------------------------------------------------------------
# Shared helpers: the design, its model, and the model's information matrix
# ---------------------------------------------------------------------------


def design_frame(plot: BasePlot) -> tuple[pd.DataFrame, list[str]]:
    """Return the factor columns of ``plot.design_data`` and their names.

    Every column except the response, ``RunOrder`` and ``Block`` is a factor, whatever
    ``factors_to_plot`` says: a design-quality statistic belongs to the whole design.
    """
    df = pd.DataFrame(plot.design_data)
    factors = [c for c in df.columns if c != plot.response_column and c not in _RUN_COLUMNS]
    return df[factors], factors


def design_model(plot: BasePlot, df: pd.DataFrame, factors: list[str]) -> str:
    """Name the model a design-quality plot is drawn for.

    The caller's ``model``; else the right-hand side of the fitted formula in
    ``analysis_results``; else the richest of ``"quadratic"``, ``"interactions"`` and
    ``"main_effects"`` that the design can estimate (``"quadratic"`` only when every
    quantitative factor has at least three distinct settings). The plot reports the
    model in its metadata.
    """
    if plot.model:
        return plot.model
    formula = plot.analysis_results.get("model_summary", {}).get("formula", "")
    if "~" in formula:
        return formula.split("~", 1)[1].strip()
    numeric = [f for f in factors if pd.api.types.is_numeric_dtype(df[f])]
    ladder = ["interactions", "main_effects"]
    if numeric and all(df[f].nunique() >= 3 for f in numeric):
        ladder.insert(0, "quadratic")
    for model in ladder[:-1]:
        with contextlib.suppress(ValueError):
            design_context(df, factors, model)
            return model
    return ladder[-1]


def design_context(df: pd.DataFrame, factors: list[str], model: str) -> _EvalContext:
    """Build the model matrix and ``(X'X)^-1`` for *model*; raise when it is not estimable."""
    ctx = _build_context(
        _EvalRequest(
            design_df=df,
            factor_names=factors,
            model=model,
            generators=None,
            defining_relation=None,
            resolution=None,
            effect_size=None,
            alpha=0.05,
            sigma=None,
        )
    )
    if ctx.is_singular:
        raise ValueError(
            f"The design cannot estimate the model {model!r}: X'X is singular. Pass model= with fewer terms "
            "(for example 'main_effects' or 'interactions'), or add runs."
        )
    return ctx


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

    The curve is ``evaluate_design(..., metric="fds")`` for the plot's model
    (see :func:`design_model`) over the cuboidal region.

    Parameters
    ----------
    random_state : int, numpy.random.Generator or None
        Seed for the region sampler (default 42, as in ``evaluate_design``).
    n_samples : int
        Number of points sampled uniformly over the region (default 20,000).
    **kwargs
        The :class:`BasePlot` arguments.

    Data sources
    ------------
    Requires ``design_data`` with factor columns (coded values).
    """

    def __init__(
        self,
        *,
        random_state: int | np.random.Generator | None = 42,
        n_samples: int = 20_000,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(**kwargs)
        self.random_state = random_state
        self.n_samples = n_samples

    def to_spec(self) -> ChartSpec:
        """Build an FDS ChartSpec.

        Returns
        -------
        ChartSpec

        Raises
        ------
        ValueError
            If the design cannot estimate the model.
        """
        if not self.design_data:
            return ChartSpec(title="FDS Plot - no design data")

        df, factors = design_frame(self)
        if not factors:
            return ChartSpec(title="FDS Plot - no factor columns")
        model = design_model(self, df, factors)
        design_context(df, factors, model)  # raises when the model is not estimable
        fds = evaluate_design(
            df,
            model=model,
            metric="fds",
            n_samples=self.n_samples,
            fds_resolution=101,
            random_state=self.random_state,
        )["fds"]
        curve = fds["curve"]
        plot_data = [
            {"fraction": float(f), "spv": float(v)}
            for f, v in zip(curve["fraction"], curve["scaled_prediction_variance"], strict=True)
        ]
        median_spv = float(fds["quantiles"]["0.5"]) * len(df)

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
        annotations = [
            Annotation(
                annotation_type=AnnotationType.reference_line,
                axis="y",
                value=median_spv,
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
                "n_points": len(df),
                "n_factors": len(factors),
                "model": model,
                "median_spv": median_spv,
                "max_spv": float(fds["scaled_max_prediction_variance"]),
            },
        )


# ---------------------------------------------------------------------------
# Power curve plot
# ---------------------------------------------------------------------------


@register_plot("power_curve")
class PowerCurvePlot(BasePlot):
    """Statistical power curve for a designed experiment.

    Shows the probability of detecting an effect of a given size as
    a function of the signal-to-noise ratio (Δ/σ), where Δ is the
    high-minus-low effect, twice the coefficient in -1 / +1 coding.  The
    power of each term's t-test uses that term's own variance in
    ``(X'X)^-1`` for the plot's model (see :func:`design_model`), as
    ``evaluate_design(..., metric="power")`` does; terms with the same
    variance share a line.

    Data sources
    ------------
    Requires ``design_data`` with factor columns, or ``analysis_results``
    whose ``model_summary`` gives ``n_obs`` and ``n_terms``; the latter
    assumes an orthogonal two-level design (every coefficient variance
    ``sigma^2 / N``).
    """  # noqa: RUF002

    def to_spec(self) -> ChartSpec:
        """Build a power curve ChartSpec.

        Returns
        -------
        ChartSpec

        Raises
        ------
        ValueError
            If the model is not estimable, or leaves no residual degrees of freedom.
        """
        alpha = round(1.0 - self.confidence_level, 10)
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"confidence_level must lie strictly between 0 and 1; got {self.confidence_level!r}.")
        design_info = self._get_design_info()
        if design_info is None:
            return ChartSpec(title="Power Curve - no design information")

        n_runs = design_info["n_runs"]
        n_terms = design_info["n_terms"]
        df2 = n_runs - n_terms
        if df2 <= 0:
            raise ValueError(
                f"The model has {n_terms} terms for {n_runs} runs, which leaves no residual degrees of freedom to "
                "test an effect against; use a model with fewer terms or a design with more runs."
            )

        sn_ratios = np.linspace(0.0, 4.0, 101)
        layers: list[LayerSpec] = []
        colours = [DOE_PALETTE["primary"], DOE_PALETTE["secondary"], DOE_PALETTE["cumulative"], DOE_PALETTE["negative"]]
        for k, (terms, c_jj) in enumerate(design_info["groups"]):
            power = np.atleast_1d(_coefficient_power(sn_ratios / 2.0, c_jj, df2, alpha))
            layers.append(
                LayerSpec(
                    mark=MarkType.line,
                    data=[{"sn_ratio": float(sn), "power": float(p)} for sn, p in zip(sn_ratios, power, strict=True)],
                    x=Encoding(field="sn_ratio", title="Signal-to-Noise Ratio (Δ/σ)"),  # noqa: RUF001
                    y=Encoding(field="power", title="Power"),
                    name=", ".join(terms),
                    color=colours[k % len(colours)],
                    style={"width": 2},
                )
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
                label=f"α = {alpha:g}",  # noqa: RUF001
                style={"color": DOE_PALETTE["negative"], "dash": "dot", "width": 1},
            ),
        ]

        panel = PanelSpec(
            layers=layers,
            annotations=annotations,
            title=f"Power Curve (n={n_runs}, residual df={df2})",
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
                "model": design_info["model"],
            },
        )

    def _get_design_info(self) -> dict[str, Any] | None:
        """Return the run and term counts, the model, and the terms grouped by coefficient variance."""
        if self.design_data:
            df, factors = design_frame(self)
            if not factors:
                return None
            model = design_model(self, df, factors)
            ctx = design_context(df, factors, model)
            assert ctx.XtX_inv is not None  # design_context raises when singular
            groups: dict[float, list[str]] = {}
            for j, name in enumerate(ctx.column_names):
                if not _is_intercept_col(name):
                    groups.setdefault(round(float(ctx.XtX_inv[j, j]), 10), []).append(name)
            return {
                "n_runs": ctx.N,
                "n_terms": ctx.p,
                "model": model,
                "groups": [(terms, c_jj) for c_jj, terms in groups.items()],
            }

        if self.analysis_results:
            summary = self.analysis_results.get("model_summary", {})
            n_runs = summary.get("n_obs", 0)
            n_terms = summary.get("n_terms", summary.get("n_params", 0))
            if n_runs > 0 and n_terms > 0:
                # Without the design, assume an orthogonal two-level design: c_jj = 1 / N.
                return {
                    "n_runs": n_runs,
                    "n_terms": n_terms,
                    "model": summary.get("formula"),
                    "groups": [(["every term (orthogonal design assumed)"], 1.0 / n_runs)],
                }

        return None
