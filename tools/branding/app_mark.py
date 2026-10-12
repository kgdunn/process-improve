"""Draw the mark of Factorial, the browser app, from a design process-improve generates.

The package's logo is a two-level factorial in two factors with its centre point. The app's mark
is the same in three factors: ``generate_design`` makes the 2^3 factorial with one centre run, and
each run is projected down the cube's diagonal. Six corners land on a hexagon. The corners
(+,+,+) and (-,-,-) land on the centre, behind the centre run, which is red as the README banner
draws centre runs. The lines are the hexagon and the three edges that meet at the corner nearest
the eye, so the cube is seen from above.

Usage, from the repository root::

    python tools/branding/app_mark.py web

It writes into ``web/``:

* ``favicon.svg``: the mark on a paper tile, for the browser tab;
* ``apple-touch-icon.png``, ``icon-192.png`` and ``icon-512.png``: the home-screen and
  installed-app icons that ``manifest.webmanifest`` names;
* ``og-card.png``: the 1200 x 630 card a chat or a blog post shows for a link to the app.

The page's header draws the same mark inline, in the page's own colours, so it follows the
reader's light or dark theme; ``tests/web/page/test_brand.py`` checks that the two agree.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Polygon

from process_improve.experiments import Factor, generate_design

# The README banner's and social card's colours (tools/branding/social_card.py).
NAVY, RED, SPINE, PAPER, INK, MUTED = "#2c5f7c", "#a93226", "#a6a49e", "#faf8f3", "#1d1d1f", "#5f6368"

#: The mark's geometry on a 64-unit square: the hexagon's radius, a run's radius, a line's width.
SIZE, RIM, DOT, LINE = 64.0, 20.0, 6.0, 2.4

#: Where each coded axis points once the cube is seen down its diagonal, z upwards. The three
#: unit vectors are 120 degrees apart and sum to zero, so (+,+,+) and (-,-,-) both land on 0.
AXES = np.array([[-np.sqrt(3) / 2, -0.5], [np.sqrt(3) / 2, -0.5], [0.0, 1.0]])


def runs() -> np.ndarray:
    """Return the coded runs of a 2^3 factorial with one centre run, as process-improve generates it."""
    factors = [Factor(name=name, low=-1, high=1) for name in "ABC"]
    design = generate_design(factors, design_type="full_factorial", n_center_points=1, random_state=1)
    return design.design[list("ABC")].to_numpy(float)


def mark() -> tuple[list[tuple[float, float]], tuple[float, float], list[tuple[float, float]]]:
    """Return the mark on the 64-unit square: the rim's six corners in turn, the centre, and the three edge ends.

    y points down, as SVG's does. A corner of the cube lands at twice an axis's length from the
    centre, so each projected run is scaled by RIM / 2.
    """
    coded = runs()
    projected = coded @ AXES * (RIM / 2)
    to_page = np.column_stack([SIZE / 2 + projected[:, 0], SIZE / 2 - projected[:, 1]])
    on_rim = np.hypot(*projected.T) > 1e-9
    rim = to_page[on_rim]
    rim = rim[np.argsort(np.arctan2(-(rim[:, 1] - SIZE / 2), rim[:, 0] - SIZE / 2))]
    # The corner nearest the eye, (+,+,+), meets the three corners with two plus signs.
    near = to_page[on_rim & (coded.sum(axis=1) == 1)]
    centre = (SIZE / 2, SIZE / 2)
    return [tuple(p) for p in rim.round(2)], centre, [tuple(p) for p in near.round(2)]


def svg(*, tile: bool) -> str:
    """Return the mark as SVG text; with ``tile``, on a rounded paper square, as the browser tab shows it."""
    rim, (cx, cy), near = mark()
    hexagon = " ".join(f"{x:g},{y:g}" for x, y in rim)
    edges = "".join(f'<line x1="{cx:g}" y1="{cy:g}" x2="{x:g}" y2="{y:g}"/>' for x, y in near)
    dots = "".join(f'<circle cx="{x:g}" cy="{y:g}" r="{DOT:g}"/>' for x, y in rim)
    back = f'<rect width="{SIZE:g}" height="{SIZE:g}" rx="14" fill="{PAPER}"/>' if tile else ""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SIZE:g} {SIZE:g}" role="img" aria-label="Factorial">'
        "<!-- A 2^3 factorial and its centre run, seen down the cube's diagonal; drawn by "
        "tools/branding/app_mark.py from generate_design. -->"
        f"{back}"
        f'<g fill="none" stroke="{SPINE}" stroke-width="{LINE:g}" stroke-linecap="round" stroke-linejoin="round">'
        f'<polygon points="{hexagon}"/>{edges}</g>'
        f'<g fill="{NAVY}" stroke="{PAPER}" stroke-width="1.6">{dots}</g>'
        f'<circle cx="{cx:g}" cy="{cy:g}" r="{DOT:g}" fill="{RED}" stroke="{PAPER}" stroke-width="1.6"/>'
        "</svg>\n"
    )


def draw(ax: plt.Axes, origin: tuple[float, float], scale: float) -> None:
    """Draw the mark on ``ax``, whose units are pixels with y downwards, its 64-unit square at ``origin``."""
    rim, centre, near = mark()

    def at(p: tuple[float, float]) -> tuple[float, float]:
        return origin[0] + scale * p[0], origin[1] + scale * p[1]

    line = {"color": SPINE, "lw": LINE * scale * 0.72, "solid_capstyle": "round", "zorder": 1}
    ax.add_patch(Polygon([at(p) for p in rim], closed=True, fill=False, ec=SPINE, lw=line["lw"], joinstyle="round"))
    for end in near:
        ax.plot(*zip(at(centre), at(end), strict=True), **line)
    edge = {"ec": PAPER, "lw": 1.6 * scale * 0.72, "zorder": 2}
    for p in rim:
        ax.add_patch(Circle(at(p), DOT * scale, fc=NAVY, **edge))
    ax.add_patch(Circle(at(centre), DOT * scale, fc=RED, **edge))


def canvas(width: int, height: int) -> tuple[plt.Figure, plt.Axes]:
    """Return a figure exactly ``width`` x ``height`` pixels, whose axes are those pixels, y downwards."""
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def icon(path: Path, pixels: int, *, safe: float) -> None:
    """Write a square icon: the mark on paper, filling ``safe`` of the width (the rest is margin)."""
    fig, ax = canvas(pixels, pixels)
    ax.add_patch(plt.Rectangle((0, 0), pixels, pixels, fc=PAPER, ec="none"))
    scale = pixels * safe / SIZE
    draw(ax, ((pixels - SIZE * scale) / 2,) * 2, scale)
    fig.savefig(path, facecolor=PAPER)
    plt.close(fig)


def card(path: Path) -> None:
    """Write the 1200 x 630 card for a shared link, in the social-preview card's layout and colours."""
    width, height = 1200, 630
    fig, ax = canvas(width, height)
    ax.add_patch(plt.Rectangle((0, 0), width, height, fc=PAPER, ec="none"))
    ax.add_patch(plt.Rectangle((56, 120), 7, 390, fc=RED, ec="none"))
    serif = {"family": "DejaVu Serif", "weight": "bold"}
    sans = {"family": "DejaVu Sans"}
    ax.text(96, 160, "PROCESS-IMPROVE  ·  IN YOUR BROWSER", color=RED, fontsize=15, **sans, weight="bold")
    ax.text(92, 268, "Factorial", color=INK, fontsize=70, **serif)
    ax.text(96, 352, "Design an experiment, run it,", color=INK, fontsize=24, **sans)
    ax.text(96, 392, "and analyse the results.", color=INK, fontsize=24, **sans)
    ax.text(96, 462, "Python runs inside the page: nothing to install,", color=MUTED, fontsize=17, **sans)
    ax.text(96, 492, "and your data stay on your machine.", color=MUTED, fontsize=17, **sans)
    ax.add_patch(
        FancyBboxPatch((760, 135), 360, 360, boxstyle="round,pad=0,rounding_size=36", fc="white", ec="#e7e3d9", lw=1.2)
    )
    scale = 300 / SIZE
    draw(ax, (940 - SIZE * scale / 2, 315 - SIZE * scale / 2), scale)
    fig.savefig(path, facecolor=PAPER)
    plt.close(fig)


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "favicon.svg").write_text(svg(tile=True), encoding="utf-8")
    icon(out / "apple-touch-icon.png", 180, safe=0.78)
    icon(out / "icon-192.png", 192, safe=0.78)
    icon(out / "icon-512.png", 512, safe=0.66)  # also the maskable icon: the mark sits in the central safe zone
    card(out / "og-card.png")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "web"))
