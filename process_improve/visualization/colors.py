"""DOE colour palettes shared across both backends.

Provides consistent, accessible colours for all DOE plot types.
The palettes are designed for readability on white backgrounds and
are colourblind-friendly where possible.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Core palette
# ---------------------------------------------------------------------------

#: Primary palette for DOE visualisations.
DOE_PALETTE: dict[str, str] = {
    "primary": "#2563EB",          # Main lines, bars
    "secondary": "#7C3AED",        # Secondary traces
    "positive": "#059669",         # Positive effects
    "negative": "#DC2626",         # Negative effects
    "neutral": "#6B7280",          # Non-significant
    "threshold_me": "#F59E0B",     # Margin of error line
    "threshold_sme": "#DC2626",    # Simultaneous ME line
    "grid": "#E5E7EB",             # Grid lines
    "background": "#FFFFFF",       # Background
    "zero_line": "#9CA3AF",        # Zero / reference line
    "cumulative": "#F97316",       # Cumulative % line (Pareto)
}

# ---------------------------------------------------------------------------
# Factor colours (up to 10 factors)
# ---------------------------------------------------------------------------

#: Distinct colours for individual factors in main-effects / interaction.
FACTOR_COLORS: list[str] = [
    "#2563EB",  # blue
    "#DC2626",  # red
    "#059669",  # green
    "#F59E0B",  # amber
    "#7C3AED",  # violet
    "#EC4899",  # pink
    "#14B8A6",  # teal
    "#F97316",  # orange
    "#6366F1",  # indigo
    "#84CC16",  # lime
]

# ---------------------------------------------------------------------------
# Contour / surface colour scales
# ---------------------------------------------------------------------------

#: Plotly-style continuous colour scale for contour and surface plots.
SURFACE_COLORSCALE: list[list[float | str]] = [
    [0.0, "#EFF6FF"],
    [0.25, "#93C5FD"],
    [0.5, "#3B82F6"],
    [0.75, "#1D4ED8"],
    [1.0, "#1E3A8A"],
]

#: ECharts-compatible colour stops (same hues as Plotly scale).
ECHARTS_VISUAL_MAP_COLORS: list[str] = [
    "#EFF6FF",
    "#93C5FD",
    "#3B82F6",
    "#1D4ED8",
    "#1E3A8A",
]

# ---------------------------------------------------------------------------
# Diagnostic plot colours
# ---------------------------------------------------------------------------

#: Colours specifically for residual diagnostic plots.
DIAGNOSTIC_COLORS: dict[str, str] = {
    "residual": "#2563EB",
    "fitted_line": "#DC2626",
    "reference_line": "#9CA3AF",
    "high_leverage": "#F59E0B",
    "high_cooks": "#DC2626",
    "normal_line": "#059669",
    "confidence_band": "rgba(37, 99, 235, 0.15)",
}

# ---------------------------------------------------------------------------
# Desirability colour scale (red → yellow → green)
# ---------------------------------------------------------------------------

DESIRABILITY_COLORSCALE: list[list[float | str]] = [
    [0.0, "#DC2626"],   # d = 0 (worst)
    [0.5, "#F59E0B"],   # d = 0.5
    [1.0, "#059669"],   # d = 1 (best)
]
