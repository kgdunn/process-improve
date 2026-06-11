"""Plotly base themes for process-improve.

This module registers a small family of Plotly templates so that every
figure produced by the library shares a consistent, professional look.
Importing :mod:`process_improve.visualization` registers the templates but
does **not** change ``plotly.io.templates.default``: doing so would
silently restyle every other Plotly figure in the same process. The
library's own plots request :data:`DEFAULT_THEME` explicitly. Call
:func:`set_theme` to opt a whole session into a process-improve theme.

Four themes are provided:

``pi_tufte``
    Minimal, maximum data-ink. White background, no gridlines,
    range-frame axes, a muted colourway and a serif font.
``pi_economist``
    Editorial style after *The Economist*: a pale blue-grey panel,
    horizontal-only white gridlines, a heavy sans font and the
    signature Economist colourway (deep blue with a red accent).
``pi_journal``
    Scientific-journal style (Nature / IEEE): white background, light
    full grid, a black mirrored box and the Okabe-Ito colourblind-safe
    palette.
``pi_brand``
    The process-improve house style, built on the existing
    :data:`~process_improve.visualization.colors.FACTOR_COLORS` palette
    so latent-variable plots and DOE plots match.

Examples
--------
>>> from process_improve.visualization import set_theme
>>> set_theme("pi_economist")          # change the global default
>>> pca.score_plot()                    # now uses the Economist theme
>>> pca.score_plot(settings={"template": "pi_tufte"})  # per-call override
"""

from __future__ import annotations

try:
    import plotly.graph_objects as go
    import plotly.io as pio
except ImportError:  # pragma: no cover - exercised via env-without-plotly
    from process_improve._extras import _MissingExtra

    go = _MissingExtra("plotly", "plotting")  # type: ignore[assignment]
    pio = _MissingExtra("plotly", "plotting")  # type: ignore[assignment]

from process_improve.visualization.colors import (
    FACTOR_COLORS,
    FONT_SANS,
    FONT_SANS_COMPACT,
    FONT_SANS_HEAVY,
    FONT_SERIF,
)

# ---------------------------------------------------------------------------
# Theme names
# ---------------------------------------------------------------------------

THEME_TUFTE: str = "pi_tufte"
THEME_ECONOMIST: str = "pi_economist"
THEME_JOURNAL: str = "pi_journal"
THEME_BRAND: str = "pi_brand"

#: Names of every theme registered by :func:`register_themes`.
THEME_NAMES: tuple[str, ...] = (THEME_TUFTE, THEME_ECONOMIST, THEME_JOURNAL, THEME_BRAND)

#: Theme applied as the Plotly default when the package is imported.
DEFAULT_THEME: str = THEME_JOURNAL

# ---------------------------------------------------------------------------
# Semantic line colours
# ---------------------------------------------------------------------------
# Reference and confidence-limit lines are drawn as layout shapes, so they are
# not coloured by a template ``colorway``. These constants give plot code a
# single, theme-agnostic source for those semantic colours.

#: Colour for confidence-limit lines (SPE limit, Hotelling's T2 limit, ...).
LIMIT_LINE_COLOR: str = "#DC2626"

#: Colour for neutral reference lines (zero lines, parity diagonals, ...).
REFERENCE_LINE_COLOR: str = "#9CA3AF"


def _tufte_template() -> go.layout.Template:
    axis = dict(
        showgrid=False,
        zeroline=False,
        showline=True,
        linecolor="#2A2A2A",
        linewidth=1,
        ticks="outside",
        ticklen=6,
        tickcolor="#2A2A2A",
        mirror=False,
    )
    return go.layout.Template(
        layout=dict(
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            colorway=["#5B7C99", "#B4543A", "#6B8F71", "#8A7A9B", "#A9883E"],
            font=dict(family=FONT_SERIF, size=13, color="#2A2A2A"),
            title=dict(font=dict(family=FONT_SERIF, size=18, color="#2A2A2A"), x=0.0, xanchor="left"),
            xaxis=axis,
            yaxis=axis,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0.0,
                font=dict(size=12),
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
            ),
            margin=dict(l=72, r=36, t=84, b=64),
            hoverlabel=dict(font=dict(family=FONT_SERIF, size=12)),
        ),
    )


def _economist_template() -> go.layout.Template:
    # The Economist red leads the accent palette but is kept out of the first
    # slot: the first colourway entry styles the primary data series, and a red
    # series would clash with the red confidence-limit lines (LIMIT_LINE_COLOR).
    return go.layout.Template(
        layout=dict(
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#D9E9F1",
            colorway=["#006BA2", "#E3120B", "#3EBCD2", "#379A8B", "#EBB434", "#9A607F"],
            font=dict(family=FONT_SANS_HEAVY, size=13, color="#121212"),
            title=dict(font=dict(family=FONT_SANS_HEAVY, size=20, color="#121212"), x=0.0, xanchor="left"),
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showline=True,
                linecolor="#121212",
                linewidth=1,
                ticks="outside",
                ticklen=5,
                tickcolor="#121212",
                mirror=False,
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="#FFFFFF",
                gridwidth=1,
                zeroline=False,
                showline=False,
                ticks="",
                mirror=False,
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0.0,
                font=dict(size=12),
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
            ),
            margin=dict(l=72, r=40, t=84, b=64),
            hoverlabel=dict(font=dict(family=FONT_SANS_HEAVY, size=12)),
        ),
    )


def _journal_template() -> go.layout.Template:
    axis = dict(
        showgrid=True,
        gridcolor="#E8E8E8",
        gridwidth=1,
        zeroline=False,
        showline=True,
        linecolor="#000000",
        linewidth=1,
        mirror=True,
        ticks="inside",
        ticklen=4,
        tickcolor="#000000",
    )
    return go.layout.Template(
        layout=dict(
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            colorway=[
                "#0072B2",
                "#D55E00",
                "#009E73",
                "#CC79A7",
                "#E69F00",
                "#56B4E9",
                "#F0E442",
                "#000000",
            ],
            font=dict(family=FONT_SANS_COMPACT, size=11, color="#000000"),
            title=dict(font=dict(family=FONT_SANS_COMPACT, size=15, color="#000000"), x=0.0, xanchor="left"),
            xaxis=axis,
            yaxis=axis,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0.0,
                font=dict(size=11),
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="#000000",
                borderwidth=1,
            ),
            margin=dict(l=64, r=28, t=72, b=56),
            hoverlabel=dict(font=dict(family=FONT_SANS_COMPACT, size=11)),
        ),
    )


def _brand_template() -> go.layout.Template:
    axis = dict(
        showgrid=True,
        gridcolor="#E5E7EB",
        gridwidth=1,
        zeroline=True,
        zerolinecolor="#9CA3AF",
        zerolinewidth=1,
        showline=True,
        linecolor="#D1D5DB",
        linewidth=1,
        mirror=False,
        ticks="outside",
        ticklen=4,
        tickcolor="#D1D5DB",
    )
    return go.layout.Template(
        layout=dict(
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FAFAFA",
            colorway=list(FACTOR_COLORS),
            font=dict(family=FONT_SANS, size=13, color="#1F2937"),
            title=dict(font=dict(family=FONT_SANS, size=18, color="#1F2937"), x=0.0, xanchor="left"),
            xaxis=axis,
            yaxis=axis,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0.0,
                font=dict(size=12),
                bgcolor="rgba(255,255,255,0.6)",
                bordercolor="#E5E7EB",
                borderwidth=1,
            ),
            margin=dict(l=70, r=36, t=84, b=60),
            hoverlabel=dict(font=dict(family=FONT_SANS, size=12)),
        ),
    )


def _build_templates() -> dict[str, go.layout.Template]:
    """Build the four base templates keyed by their registered name."""
    return {
        THEME_TUFTE: _tufte_template(),
        THEME_ECONOMIST: _economist_template(),
        THEME_JOURNAL: _journal_template(),
        THEME_BRAND: _brand_template(),
    }


def register_themes() -> None:
    """Register all process-improve themes in ``plotly.io.templates``.

    Safe to call repeatedly; existing registrations are simply overwritten.
    """
    for name, template in _build_templates().items():
        pio.templates[name] = template


def set_theme(name: str = DEFAULT_THEME) -> None:
    """Set ``name`` as the global default Plotly template.

    Parameters
    ----------
    name : str, optional
        One of :data:`THEME_NAMES`, by default :data:`DEFAULT_THEME`.

    Raises
    ------
    ValueError
        If ``name`` is not a registered process-improve theme.
    """
    if name not in THEME_NAMES:
        raise ValueError(f"Unknown theme {name!r}. Choose one of {THEME_NAMES}.")
    if name not in pio.templates:
        register_themes()
    pio.templates.default = name
