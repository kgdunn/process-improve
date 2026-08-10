#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["process-improve[expt,plotting]", "kaleido"]
# ///
"""Render a ``visualize_doe`` result to an image or an interactive page.

``visualize_doe`` returns a chart specification with both a Plotly and an
ECharts rendering of the same figure; it does not write files. This turns
either one into something a person can look at.

Usage
-----
Go straight from an analysis to a picture::

    python render_plot.py --analysis analysis.json --type pareto --output pareto.png

Render a chart spec that ``visualize_doe`` already produced::

    python render_plot.py --spec plot.json --output plot.png

Formats follow the extension: ``.png``, ``.svg``, ``.pdf``, ``.jpg`` are
static images (needs kaleido); ``.html`` is a self-contained interactive page
that opens in any browser with no server; ``.json`` writes the raw chart spec.

Plot types
----------
Significance:  pareto, half_normal, daniel
Factor effects: main_effects, interaction, perturbation, cube_plot, square_plot
Diagnostics:   residuals_vs_fitted, normal_probability, residuals_vs_order, box_cox
Response surface: contour, surface_3d, prediction_variance
Optimisation:  desirability_contour, overlay, ridge_trace, steepest_ascent_path
Design quality: fds_plot, power_curve

Exit codes
----------
0   the figure was written
1   bad usage (missing input, unknown plot type, unwritable path)
2   visualize_doe reported an error
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

EXIT_USAGE = 1
EXIT_TOOL_ERROR = 2

_STATIC_SUFFIXES = {".png", ".svg", ".pdf", ".jpg", ".jpeg", ".webp"}


def _read_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    if not path.is_file():
        sys.exit(f"No such file: {path}")
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        sys.exit(f"{path} is not valid JSON: {exc}")
    if not isinstance(payload, dict):
        sys.exit(f"{path} must contain a JSON object.")
    return payload


def _build_spec(args: argparse.Namespace) -> dict[str, Any]:
    """Return the chart spec, either loaded or produced by visualize_doe."""
    if args.spec:
        return _read_json(Path(args.spec))

    try:
        from process_improve.tool_spec import discover_tools, execute_tool_call
    except ImportError as exc:  # pragma: no cover - environment guard
        sys.exit(
            f"Could not import process_improve ({exc}).\nInstall it with:  pip install 'process-improve[expt,plotting]'"
        )
    discover_tools()

    payload: dict[str, Any] = {"plot_type": args.type}
    if args.analysis:
        payload["analysis_results"] = _read_json(Path(args.analysis))
    if args.design:
        import pandas as pd

        payload["design_data"] = pd.read_csv(args.design).to_dict("records")
    if args.response:
        payload["response_column"] = args.response
    if args.factors:
        payload["factors_to_plot"] = args.factors.split(",")

    result = execute_tool_call("visualize_doe", payload)
    if isinstance(result, dict) and "error" in result:
        print(f"visualize_doe failed: {result['error']}", file=sys.stderr)
        raise SystemExit(EXIT_TOOL_ERROR)
    return result


def _write(spec: dict[str, Any], output: Path) -> None:
    """Write the chart spec out in whatever form the extension asks for."""
    suffix = output.suffix.lower()

    if suffix == ".json":
        output.write_text(json.dumps(spec, indent=2, default=str))
        return

    figure_spec = spec.get("plotly")
    if not figure_spec:
        sys.exit("The chart spec has no 'plotly' figure to render. Write it as .json instead.")

    try:
        import plotly.graph_objects as go
    except ImportError:  # pragma: no cover - environment guard
        sys.exit("Rendering needs plotly. Install with: pip install 'process-improve[plotting]'")

    figure = go.Figure(figure_spec)

    if suffix == ".html":
        figure.write_html(str(output), include_plotlyjs="inline", full_html=True)
        return

    if suffix not in _STATIC_SUFFIXES:
        sys.exit(
            f"Unsupported output extension {suffix!r}. Use one of: {', '.join(sorted(_STATIC_SUFFIXES))}, .html, .json"
        )

    try:
        figure.write_image(str(output), scale=2)
    except (ImportError, ValueError, RuntimeError) as exc:
        sys.exit(
            f"Static image export failed ({exc}).\n"
            "It needs kaleido:  pip install kaleido\n"
            "Or write an .html file, which needs nothing extra."
        )


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(
        prog="render_plot.py",
        description="Turn a visualize_doe chart spec into a picture.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--spec", help="a chart spec that visualize_doe already produced")
    source.add_argument("--analysis", help="an analyze_experiment / fit_linear_model result to plot")
    parser.add_argument("--type", help="plot type; required unless --spec is used")
    parser.add_argument("--design", help="design CSV, for plots that need the raw runs")
    parser.add_argument("--response", help="response column name, for design-data plots")
    parser.add_argument("--factors", help="comma-separated factors to plot, for contour and surface")
    parser.add_argument("--output", required=True, type=Path, help="output path; format follows the extension")
    args = parser.parse_args(argv)

    if args.analysis and not args.type:
        parser.error("--type is required when rendering from --analysis")

    spec = _build_spec(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write(spec, args.output)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
