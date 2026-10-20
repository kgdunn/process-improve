"""Response-surface plots: contour, 3D surface, prediction variance.

These plots evaluate a fitted model over a 2D grid of two factors
(holding others at their centre or ``hold_values``).  They answer
questions like Q47 ("How do I draw a contour plot?") and Q121
("BBD contour plots and optimal salt/temperature/time").

Given a ``region`` with limits on the settings, the contour and
prediction-variance plots shade the settings that cannot be run, and the
3D surface leaves them out.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from process_improve.experiments.visualization.plots.registry import BasePlot, register_plot
from process_improve.visualization.colors import CANNOT_RUN_FILL, DOE_PALETTE
from process_improve.visualization.spec import (
    Annotation,
    ChartSpec,
    Encoding,
    LayerSpec,
    PanelSpec,
)
from process_improve.visualization.types import AnnotationType, MarkType

if TYPE_CHECKING:
    from process_improve.experiments.region import DesignRegion

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

        # Quadratic: ``I(A ** 2)`` (older statsmodels) or
        # ``np.power(A, 2)`` / ``power(A, 2)`` (newer). SEC-27 (#276).
        m = re.match(r"I\((\w+)\s*\*\*\s*2\)", term) or re.match(r"(?:np\.)?power\((\w+)\s*,\s*2\)", term)
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


def _extent(plot: BasePlot, factor: str) -> float:
    """Return how far the grid reaches along ``factor`` from the centre, in coded units (#799).

    It is ``plot.extent`` when given (one number, or one per factor), else the largest
    absolute coded value of the factor over ``design_data``, and at least 1: a composite
    design's axial runs at plus or minus alpha, which carry its curvature, stay in view.
    """
    given = plot.extent.get(factor) if isinstance(plot.extent, dict) else plot.extent
    if given is not None:
        if not (np.isfinite(given) and given > 0):
            raise ValueError(f"The extent of {factor!r} must be a positive number of coded units; got {given!r}.")
        return float(given)
    values = [r[factor] for r in plot.design_data if isinstance(r.get(factor), int | float) and np.isfinite(r[factor])]
    return max([1.0, *(abs(float(v)) for v in values)])


def _compute_grid(
    coef_map: dict[str, float],
    axes: tuple[str, str],
    hold_values: dict[str, float],
    n_grid: int = 50,
    extents: tuple[float, float] = (1.0, 1.0),
) -> tuple[list[float], list[float], list[list[float]]]:
    """Evaluate model over a meshgrid for two factors.

    Parameters
    ----------
    coef_map : dict[str, float]
        Coefficient map.
    axes : tuple[str, str]
        The two factors to sweep, along x and along y.
    hold_values : dict[str, float]
        Fixed values for all other factors.
    n_grid : int
        Grid resolution per axis.
    extents : tuple[float, float]
        How far the grid reaches from the centre along x and along y, in coded units.

    Returns
    -------
    tuple[list[float], list[float], list[list[float]]]
        x_grid, y_grid, z_matrix (row-major).
    """
    factor_x, factor_y = axes
    x_grid = np.linspace(-extents[0], extents[0], n_grid).tolist()
    y_grid = np.linspace(-extents[1], extents[1], n_grid).tolist()

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


def _limit_slack(
    region: DesignRegion,
    axes: tuple[str, str],
    grid: tuple[list[float], list[float]],
    hold_values: dict[str, Any],
) -> np.ndarray:
    """Return how far each grid point is from meeting every limit: ``<= 0`` exactly where it can be run.

    ``axes`` names the factors along x and y, and ``grid`` holds their values; rows follow
    the y values and columns the x values, as in a plot's ``z_matrix``. The grid and
    ``hold_values`` are in coded units, which the region's factor ranges define; a factor
    of the region that is neither plotted nor held is at the centre of its range.
    A straight limit's slack is its signed distance from its boundary, linear over the
    grid, so a contour of it at zero is drawn straight. Where a curved limit cannot be
    evaluated (``sqrt`` or ``log`` of a negative number) or is infinite, the setting
    cannot be run, and its slack is set one above the largest finite slack (an infinite
    allowed value one below the smallest): a figure's JSON writes ``inf`` and ``NaN`` as
    ``null``, which Plotly leaves unshaded and the ECharts mask drops.

    Raises
    ------
    ValueError
        If the region names neither plotted factor, so it cannot vary over the plot.
    """
    names = region.names
    factor_x, factor_y = axes
    x_grid, y_grid = grid
    if factor_x not in names and factor_y not in names:
        raise ValueError(
            f"The region's factors {names} include neither {factor_x!r} nor {factor_y!r}, the factors plotted; "
            "give the region over the same factors as the plot."
        )
    gx, gy = np.meshgrid(np.asarray(x_grid, dtype=float), np.asarray(y_grid, dtype=float))
    coded = np.zeros((gx.size, len(names)))
    for j, name in enumerate(names):
        if name == factor_x:
            coded[:, j] = gx.ravel()
        elif name == factor_y:
            coded[:, j] = gy.ravel()
        else:
            coded[:, j] = float(hold_values.get(name, 0.0))
    return _finite(region._limit_slack(coded).reshape(gx.shape))


def _finite(slack: np.ndarray) -> np.ndarray:
    """Return the slack with ``NaN`` and ``+inf`` one above its largest finite value and 0, ``-inf`` one below."""
    finite = slack[np.isfinite(slack)]
    above = max(float(finite.max()) if finite.size else 0.0, 0.0) + 1.0
    below = min(float(finite.min()) if finite.size else 0.0, 0.0) - 1.0
    return np.where(np.isnan(slack) | (slack == np.inf), above, np.where(slack == -np.inf, below, slack))


def _cannot_run_layer(
    slack: np.ndarray, x_grid: list[float], y_grid: list[float], x_title: str, y_title: str
) -> LayerSpec:
    """Return the layer that shades the settings that cannot be run: a constraint contour of the slack at zero.

    The settings where the slack is at most zero are kept clear; the rest are shaded
    with :data:`~process_improve.visualization.colors.CANNOT_RUN_FILL`, translucent so
    the response stays readable beneath.
    """
    return LayerSpec(
        mark=MarkType.contour,
        data=[],
        x=Encoding(field="x", title=x_title),
        y=Encoding(field="y", title=y_title),
        name="Cannot be run",
        style={
            "x_grid": list(x_grid),
            "y_grid": list(y_grid),
            "z_matrix": slack.tolist(),
            "contours": {"type": "constraint", "operation": "<=", "value": 0},
            "fillcolor": CANNOT_RUN_FILL,
            "hoverinfo": "skip",
            "showscale": False,
            "showlabels": False,
        },
    )


def _shading(
    plot: BasePlot,
    axes: tuple[str, str],
    grid: tuple[list[float], list[float]],
    titles: tuple[str, str],
    shown: tuple[list[float], list[float]] | None = None,
) -> tuple[LayerSpec | None, dict[str, Any]]:
    """Return a plot's shading layer and the metadata saying how much it covers, or ``(None, {})`` without limits.

    ``grid`` is in coded units; ``shown`` is the same grid in the plot's axis units, when they differ.
    """
    region = plot._get_region()
    if region is None or not region.constraints:
        return None, {}
    slack = _limit_slack(region, axes, grid, plot.hold_values)
    layer = _cannot_run_layer(slack, *(shown or grid), *titles)
    return layer, {"cannot_run_share": float(np.mean(~(slack <= 0)))}


# ---------------------------------------------------------------------------
# Contour plot
# ---------------------------------------------------------------------------


#: The layers a contour plot can add from the canonical analysis of its model (#799).
_OVERLAYS = ("stationary_point", "canonical_axes")


def _clip_line(
    point: tuple[float, float], direction: tuple[float, float], extents: tuple[float, float]
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Return where the line through ``point`` along ``direction`` enters and leaves the plotted box.

    The box is ``[-ex, ex] x [-ey, ey]`` for ``extents`` ``(ex, ey)``, in coded units. None
    when the direction has no part in the plotted plane (an axis across the slice) or the
    line misses the box.
    """
    (px, py), (dx, dy) = point, direction
    if math.hypot(dx, dy) < 1e-9:
        return None
    enter, leave = -math.inf, math.inf
    for position, step, reach in ((px, dx, extents[0]), (py, dy, extents[1])):
        if abs(step) < 1e-12:
            if abs(position) > reach:
                return None
            continue
        first, second = (-reach - position) / step, (reach - position) / step
        enter, leave = max(enter, min(first, second)), min(leave, max(first, second))
    if not enter < leave:
        return None
    return (px + enter * dx, py + enter * dy), (px + leave * dx, py + leave * dy)


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

    With a ``region`` that has limits on the settings, a second layer, after
    the response, shades the settings that cannot be run with the other
    factors held at ``hold_values``: a constraint contour of the limits' slack
    at zero (``"Cannot be run"``). The experimental points are drawn last, on
    top. Without a region, or with one without limits, the figure is the same
    as before.

    The grid covers every run in ``design_data`` (see ``extent``), so a
    composite design's axial runs lie inside the plot (#799). ``overlays``
    draws the canonical analysis of the model
    (``optimize_responses(..., method="canonical_analysis")``), which shows,
    for example, where a ridge runs: ``"stationary_point"`` marks the
    stationary point with its classification, and ``"canonical_axes"`` draws
    each canonical axis through it with its eigenvalue. With more than two
    factors the plot is a slice at ``hold_values``, and the point and axes are
    drawn at their plotted coordinates; the slice passes through the
    stationary point only when the other factors are held at its coordinates.

    Parameters
    ----------
    units : {"coded", "actual"}
        The units of the axes. ``"actual"`` draws every layer in the factors'
        own units, from ``analysis_results["factor_coding"]``, which
        :func:`~process_improve.experiments.analysis.analyze_experiment`
        returns, or else from the ranges of the ``region``'s factors. The
        ``design_data`` and ``hold_values`` stay in coded units. Default
        ``"coded"``.
    overlays : sequence of str
        Any of ``"stationary_point"`` and ``"canonical_axes"``; they need a
        second-order model. Default none.
    **kwargs
        The parameters of
        :class:`~process_improve.experiments.visualization.plots.registry.BasePlot`.
    """

    def __init__(self, *, units: str = "coded", overlays: Sequence[str] = (), **kwargs: Any) -> None:  # noqa: ANN401
        super().__init__(**kwargs)
        if units not in ("coded", "actual"):
            raise ValueError(f"units must be 'coded' or 'actual'; got {units!r}.")
        overlays = (overlays,) if isinstance(overlays, str) else tuple(overlays)
        unknown = [name for name in overlays if name not in _OVERLAYS]
        if unknown:
            raise ValueError(f"Unknown overlays {unknown}; choose from {list(_OVERLAYS)}.")
        self.units = units
        self.overlays = overlays

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
        shown = (self._to_axis(factor_x), self._to_axis(factor_y))

        coef_map = _build_coef_map(coefficients)
        extents = (_extent(self, factor_x), _extent(self, factor_y))
        x_grid, y_grid, z_matrix = _compute_grid(coef_map, (factor_x, factor_y), self.hold_values, extents=extents)
        grid_shown = (shown[0](x_grid), shown[1](y_grid))

        contour_layer = LayerSpec(
            mark=MarkType.contour,
            data=[],  # Data is in style for grid-based plots
            x=Encoding(field="x", title=x_title),
            y=Encoding(field="y", title=y_title),
            name="Response",
            style={
                "x_grid": grid_shown[0],
                "y_grid": grid_shown[1],
                "z_matrix": z_matrix,
            },
        )

        layers: list[LayerSpec] = [contour_layer]
        axes = (factor_x, factor_y)
        shading, shaded = _shading(self, axes, (x_grid, y_grid), (x_title, y_title), grid_shown)
        if shading is not None:
            layers.append(shading)
        overlay_layers, overlaid = self._overlay_layers(axes, extents, shown)
        layers.extend(overlay_layers)
        point_layer = self._build_design_point_layer(factor_x, factor_y, shown)
        if point_layer is not None:
            layers.append(point_layer)

        panel = PanelSpec(
            layers=layers,
            annotations=_zero_reference_lines(shown[0]([0.0])[0], shown[1]([0.0])[0]),
            title=f"Contour Plot: {x_title} x {y_title}",
            x_title=x_title,
            y_title=y_title,
            backend_hints={"equal_aspect": self.units == "coded"},
        )

        return ChartSpec(
            panels=[panel],
            title=f"Contour Plot: {x_title} x {y_title}",
            plot_type="contour",
            metadata={
                "factors": [factor_x, factor_y],
                "factor_labels": {factor_x: x_title, factor_y: y_title},
                "hold_values": self.hold_values,
                "units": self.units,
                "extent": {factor_x: extents[0], factor_y: extents[1]},
                **shaded,
                **overlaid,
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

    def _to_axis(self, factor: str) -> Callable[[Sequence[float]], list[float]]:
        """Return the map from coded values of ``factor`` to the units of its axis."""
        if self.units == "coded":
            return lambda values: [float(v) for v in values]
        low, high = self._actual_range(factor)
        centre, half = (low + high) / 2.0, (high - low) / 2.0
        return lambda values: [centre + half * float(v) for v in values]

    def _actual_range(self, factor: str) -> tuple[float, float]:
        """Return the actual values of ``factor`` at coded -1 and +1, from the fit's coding or the region."""
        coding = (self.analysis_results.get("factor_coding") or {}).get(factor)
        if coding is not None:
            return float(coding["low"]), float(coding["high"])
        region = self._get_region()
        if region is not None and factor in region.names:
            j = region.names.index(factor)
            return float(region._low[j]), float(region._high[j])
        raise ValueError(
            f"units='actual' needs the range of {factor!r}: pass analysis_results from analyze_experiment, whose "
            "factor_coding holds it, or a region over the plotted factors."
        )

    def _canonical_analysis(self) -> dict[str, Any]:
        """Return ``optimize_responses(..., method="canonical_analysis")`` for the plotted model."""
        from process_improve.experiments.optimization import optimize_responses  # noqa: PLC0415

        coefficients = self._get_coefficients()
        names = list(self.analysis_results.get("factor_names") or self._get_factor_names())
        if not names:  # a bare coefficient list: its first-order terms name the factors
            names = [
                c["term"] for c in coefficients if re.fullmatch(r"[A-Za-z_]\w*", c["term"]) and c["term"] != "Intercept"
            ]
        model: dict[str, Any] = {
            "response_name": self.analysis_results.get("response_name") or self.response_column or "y",
            "factor_names": names,
            "coefficients": coefficients,
        }
        if "factor_coding" in self.analysis_results:
            model["factor_coding"] = self.analysis_results["factor_coding"]
        problem = f"The overlays {list(self.overlays)} need the canonical analysis of the model"
        try:
            result = optimize_responses([model], method="canonical_analysis")
        except ValueError as err:
            raise ValueError(f"{problem}: {err}") from err
        failed = result["canonical_analysis"].get("error") or result["stationary_point"].get("error")
        if failed:  # a first-order model has no second-order part to analyse
            raise ValueError(f"{problem}: {failed}")
        return result

    def _overlay_layers(
        self,
        axes: tuple[str, str],
        extents: tuple[float, float],
        shown: tuple[Callable[[Sequence[float]], list[float]], ...],
    ) -> tuple[list[LayerSpec], dict[str, Any]]:
        """Return the stationary point and canonical axes layers that ``overlays`` asks for, and their metadata."""
        if not self.overlays:
            return [], {}
        result = self._canonical_analysis()
        point = result["stationary_point"]
        sx, sy = (float(point["stationary_point_coded"][name]) for name in axes)
        kind = str(point["classification"]).replace("_", " ")
        layers: list[LayerSpec] = []
        metadata: dict[str, Any] = {}
        if "stationary_point" in self.overlays:
            at = (shown[0]([sx])[0], shown[1]([sy])[0])
            hover = f"Stationary point, a {kind}<br>{axes[0]}: {at[0]:.4g}<br>{axes[1]}: {at[1]:.4g}"
            hover += f"<br>predicted: {float(point['predicted_response']):.4g}"
            layers.append(
                LayerSpec(
                    mark=MarkType.scatter,
                    data=[{"x": at[0], "y": at[1], "hover": hover}],
                    x=Encoding(field="x"),
                    y=Encoding(field="y"),
                    name=f"Stationary point ({kind})",
                    color=DOE_PALETTE["negative"],
                    style={"size": 14, "symbol": "x", "hover_field": "hover", "edge_color": "#ffffff", "edge_width": 1},
                )
            )
            metadata["stationary_point"] = {
                "coded": {axes[0]: sx, axes[1]: sy},
                "classification": point["classification"],
                "predicted_response": point["predicted_response"],
            }
        if "canonical_axes" in self.overlays:
            analysis = result["canonical_analysis"]
            drawn = []
            for k, (axis, value) in enumerate(zip(analysis["canonical_axes"], analysis["eigenvalues"], strict=True), 1):
                direction = (float(axis["coefficients"][axes[0]]), float(axis["coefficients"][axes[1]]))
                ends = _clip_line((sx, sy), direction, extents)
                if ends is None:
                    continue
                (x0, y0), (x1, y1) = ends
                layers.append(
                    LayerSpec(
                        mark=MarkType.line,
                        data=[
                            {"x": shown[0]([x0])[0], "y": shown[1]([y0])[0]},
                            {"x": shown[0]([x1])[0], "y": shown[1]([y1])[0]},
                        ],
                        x=Encoding(field="x"),
                        y=Encoding(field="y"),
                        name=f"Canonical axis W{k}, eigenvalue {float(value):.3g}",
                        color=DOE_PALETTE["secondary"],
                        style={"width": 2, "dash": "dash"},
                    )
                )
                drawn.append(
                    {"axis": f"W{k}", "eigenvalue": float(value), "direction": dict(zip(axes, direction, strict=True))}
                )
            metadata["canonical_axes"] = drawn
        return layers, metadata

    def _build_design_point_layer(
        self,
        factor_x: str,
        factor_y: str,
        shown: tuple[Callable[[Sequence[float]], list[float]], ...] | None = None,
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
                x_coded, y_coded = float(r[factor_x]) + dx, float(r[factor_y]) + dy
                point: dict[str, Any] = {
                    "x": shown[0]([x_coded])[0] if shown else x_coded,
                    "y": shown[1]([y_coded])[0] if shown else y_coded,
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


def _zero_reference_lines(x_centre: float = 0.0, y_centre: float = 0.0) -> list[Annotation]:
    """Solid black axes through the centre of the design, coded 0 (issue #5)."""
    style = {"color": "#111111", "dash": "solid", "width": 1}
    return [
        Annotation(
            annotation_type=AnnotationType.reference_line,
            axis="x",
            value=x_centre,
            style=dict(style),
        ),
        Annotation(
            annotation_type=AnnotationType.reference_line,
            axis="y",
            value=y_centre,
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
    interactive 3D surface. With a ``region`` that has limits on the settings,
    the surface is ``NaN`` (not drawn) over the settings that cannot be run.

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
            coef_map,
            (factor_x, factor_y),
            self.hold_values,
            extents=(_extent(self, factor_x), _extent(self, factor_y)),
        )
        shaded: dict[str, Any] = {}
        region = self._get_region()
        if region is not None and region.constraints:
            slack = _limit_slack(region, (factor_x, factor_y), (x_grid, y_grid), self.hold_values)
            ruled_out = ~(slack <= 0)
            z_matrix = np.where(ruled_out, np.nan, np.asarray(z_matrix, dtype=float)).tolist()
            shaded = {"cannot_run_share": float(np.mean(ruled_out))}

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
                **shaded,
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
    response values.  The model (see
    :func:`~process_improve.experiments.visualization.plots.design_quality.design_model`)
    is built from every factor of the design; the factors not plotted are held at
    ``hold_values`` (default: 0 for a quantitative factor, the first level for a
    categorical one). With a ``region`` that has limits on the settings, a second
    layer shades the settings that cannot be run, as in :class:`ContourPlot`.

    Data sources
    ------------
    Requires ``design_data`` (the design matrix rows); ``factors_to_plot``
    picks the two axes (default: the first two factors).
    """

    def to_spec(self) -> ChartSpec:
        """Build a prediction-variance ChartSpec.

        Returns
        -------
        ChartSpec

        Raises
        ------
        ValueError
            If the design cannot estimate the model, or a plotted factor is not a
            quantitative factor of the design.
        """
        import pandas as pd  # noqa: PLC0415
        from patsy import build_design_matrices  # noqa: PLC0415

        from process_improve.experiments.evaluate import _prediction_variance_at_points  # noqa: PLC0415
        from process_improve.experiments.visualization.plots.design_quality import (  # noqa: PLC0415
            design_context,
            design_frame,
            design_model,
        )

        if not self.design_data:
            return ChartSpec(title="Prediction Variance - no design data")

        df, factors = design_frame(self)
        axes = self.factors_to_plot or factors
        if len(axes) < 2:
            return ChartSpec(title="Prediction Variance - need at least 2 factors")
        factor_x, factor_y = axes[0], axes[1]
        for f in (factor_x, factor_y):
            if f not in factors or not pd.api.types.is_numeric_dtype(df[f]):
                raise ValueError(f"factors_to_plot names {f!r}, which is not a quantitative factor of the design.")

        model = design_model(self, df, factors)
        ctx = design_context(df, factors, model)
        assert ctx.XtX_inv is not None  # design_context raises when singular

        # Evaluate scaled prediction variance on a grid; rows follow y, columns follow x.
        n_grid = 40
        x_axis = np.linspace(-_extent(self, factor_x), _extent(self, factor_x), n_grid)
        y_axis = np.linspace(-_extent(self, factor_y), _extent(self, factor_y), n_grid)
        gx, gy = np.meshgrid(x_axis, y_axis)
        points: dict[str, Any] = {}
        for f in factors:
            if f == factor_x:
                points[f] = gx.ravel()
            elif f == factor_y:
                points[f] = gy.ravel()
            elif pd.api.types.is_numeric_dtype(df[f]):
                points[f] = np.full(gx.size, float(self.hold_values.get(f, 0.0)))
            else:
                points[f] = np.full(gx.size, self.hold_values.get(f, df[f].iloc[0]), dtype=object)
        (expanded,) = build_design_matrices([ctx.design_info], pd.DataFrame(points), return_type="matrix")
        spv = ctx.N * _prediction_variance_at_points(np.asarray(expanded, dtype=float), ctx.XtX_inv)
        z_matrix = spv.reshape(n_grid, n_grid).tolist()
        x_grid = x_axis.tolist()
        y_grid = y_axis.tolist()

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

        layers: list[LayerSpec] = [contour_layer]
        shading, shaded = _shading(self, (factor_x, factor_y), (x_grid, y_grid), (factor_x, factor_y))
        if shading is not None:
            layers.append(shading)

        panel = PanelSpec(
            layers=layers,
            title=f"Prediction Variance: {factor_x} x {factor_y}",
            x_title=factor_x,
            y_title=factor_y,
        )

        return ChartSpec(
            panels=[panel],
            title=f"Scaled Prediction Variance: {factor_x} x {factor_y}",
            plot_type="prediction_variance",
            metadata={"model": model, "hold_values": self.hold_values, **shaded},
        )
