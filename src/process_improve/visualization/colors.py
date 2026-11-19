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
    "primary": "#2563EB",  # Main lines, bars
    "secondary": "#7C3AED",  # Secondary traces
    "positive": "#059669",  # Positive effects
    "negative": "#DC2626",  # Negative effects
    "neutral": "#6B7280",  # Non-significant
    "threshold_me": "#F59E0B",  # Margin of error line
    "threshold_sme": "#DC2626",  # Simultaneous ME line
    "grid": "#E5E7EB",  # Grid lines
    "background": "#FFFFFF",  # Background
    "zero_line": "#9CA3AF",  # Zero / reference line
    "cumulative": "#F97316",  # Cumulative % line (Pareto)
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

#: The Scientific colour maps a surface is drawn in (Crameri, F., 2018, Scientific colour maps, Zenodo,
#: doi:10.5281/zenodo.1243862; MIT licence, copyright (c) Fabio Crameri): nine colours each, sampled evenly along
#: the map, dark for one end and light for the other. They are even to the eye, so a step in the colour is a step
#: in the value, and a reader who does not tell red from green reads them, as does a print in grey. ``devon`` and
#: ``grayC`` end in white, which a page is, so they are sampled along their first nine tenths.
COLOUR_MAPS: dict[str, list[str]] = {
    "batlow": ["#011959", "#114360", "#226061", "#4d734d", "#828231", "#c09036", "#f29d6d", "#fdb4b6", "#faccfa"],
    "devon": ["#2c1a4c", "#293467", "#275186", "#376aae", "#6583d2", "#9b9ce8", "#bbb5f1", "#d2cef6", "#eae8fb"],
    "lajolla": ["#191900", "#372411", "#67342a", "#a64644", "#d9604e", "#e58851", "#edae54", "#f7da74", "#fffecb"],
    "grayC": ["#000000", "#232323", "#3d3d3d", "#565656", "#6e6e6e", "#848484", "#9c9c9c", "#b9b9b9", "#dcdcdc"],
}

#: The map a surface is drawn in unless another is asked for.
DEFAULT_COLOUR_MAP = "batlow"


def colorscale(name: str = DEFAULT_COLOUR_MAP, *, dark_page: bool = False) -> list[list[float | str]]:
    """Return one of :data:`COLOUR_MAPS` as a Plotly colour scale, a low value fading into the page.

    Parameters
    ----------
    name : str
        A key of :data:`COLOUR_MAPS`.
    dark_page : bool
        Whether the figure is on a dark page. A low value is then the map's dark end; on a light page, the
        default, it is the light end.

    Returns
    -------
    list of [float, str]
        The colour at each of nine evenly spaced fractions from 0 to 1.
    """
    if name not in COLOUR_MAPS:
        raise ValueError(f"Unknown colour map {name!r}: choose one of {', '.join(COLOUR_MAPS)}.")
    colours = COLOUR_MAPS[name] if dark_page else COLOUR_MAPS[name][::-1]
    return [[i / (len(colours) - 1), colour] for i, colour in enumerate(colours)]


#: Plotly-style continuous colour scale for contour and surface plots: batlow, light for low and dark for high.
SURFACE_COLORSCALE: list[list[float | str]] = colorscale()

#: ECharts-compatible colour stops (same hues as Plotly scale).
ECHARTS_VISUAL_MAP_COLORS: list[str] = [str(colour) for _, colour in SURFACE_COLORSCALE]

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
    [0.0, "#DC2626"],  # d = 0 (worst)
    [0.5, "#F59E0B"],  # d = 0.5
    [1.0, "#059669"],  # d = 1 (best)
]

# ---------------------------------------------------------------------------
# Sweet-spot fill: a single flat green, used behind overlaid response contours
# to shade the region where every response is within specification. It is one
# colour rather than a scale because the mask it renders is binary.
# ---------------------------------------------------------------------------

SWEET_SPOT_COLORSCALE: list[list[float | str]] = [
    [0.0, "#059669"],
    [1.0, "#059669"],
]

# ---------------------------------------------------------------------------
# Settings that cannot be run: a translucent grey laid over a contour plot where
# the limits on the settings rule a point out, so the response stays readable
# beneath it and the settings that can be run are left clear.
# ---------------------------------------------------------------------------

CANNOT_RUN_FILL: str = "rgba(75, 85, 99, 0.45)"

# ---------------------------------------------------------------------------
# Font stacks (shared by the registered Plotly base themes)
# ---------------------------------------------------------------------------

#: Serif stack - used by the Tufte theme.
FONT_SERIF: str = "Georgia, 'Times New Roman', serif"

#: Modern sans stack - used by the process-improve brand theme.
FONT_SANS: str = "Inter, 'Helvetica Neue', Arial, sans-serif"

#: Heavy sans stack - used by the Economist theme.
FONT_SANS_HEAVY: str = "Verdana, Arial, sans-serif"

#: Compact sans stack - used by the scientific-journal theme.
FONT_SANS_COMPACT: str = "Helvetica, Arial, sans-serif"
