"""README banner and design gallery for process-improve, each design drawn from ``generate_design()``.

Usage::

    python design_gallery.py hero light docs/_static/readme-hero-light.png
    python design_gallery.py gallery dark docs/_static/design-gallery-dark.png

``hero`` is a wide four-panel banner sized to stay legible at phone width; ``gallery`` is
the full 4 x 4 grid. Three-factor designs are drawn in the coded cube, many-factor
screening designs as sign tables, mixtures on the simplex, and space-filling designs on
two factors with their one-dimensional projections as ticks.
"""

import itertools
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from process_improve.experiments import Constraint, Factor, generate_design

warnings.simplefilter("ignore")

#: The run/extra pair in each theme passes the dataviz palette validator (CVD separation,
#: lightness band, contrast) on that theme's surface, which matches GitHub's page colour.
THEMES = {
    "light": {
        "run": "#1f5f8f",
        "extra": "#b03a2e",
        "ink": "#1f2328",
        "muted": "#59636e",
        "edge": "#c8c6bf",
        "tint": "#dfe5ea",
        "surface": "#ffffff",
    },
    "dark": {
        "run": "#4f93cf",
        "extra": "#dc614f",
        "ink": "#e6edf3",
        "muted": "#9198a1",
        "edge": "#3d444d",
        "tint": "#1c2a38",
        "surface": "#0d1117",
    },
}


def box(n: int = 3) -> list[Factor]:
    """``n`` continuous factors coded -1 to +1."""
    return [Factor(name="ABCDEFGHIJKL"[i], low=-1, high=1) for i in range(n)]


DSD = {"factors": box(), "design_type": "dsd"}
CCD = {"factors": box(), "design_type": "ccd", "alpha": "rotatable"}
I_OPT = {
    "factors": box(),
    "design_type": "i_optimal",
    "budget": 12,
    "model_type": "quadratic",
    "constraints": [Constraint(expression="A + B + C <= 1.5")],
}
MAXPRO = {"factors": box(2), "design_type": "maxpro", "budget": 12}

#: (row label, panel title, drawing, generate_design keywords), in the order a project uses them.
GALLERY = [
    ("Screening", "Supersaturated", "signs", {"factors": box(10), "design_type": "supersaturated", "budget": 8}),
    (
        "Screening",
        "Plackett-Burman",
        "signs",
        {"factors": box(11), "design_type": "plackett_burman", "n_center_points": 0},
    ),
    (
        "Screening",
        "Fractional factorial",
        "cube",
        {"factors": box(), "design_type": "fractional_factorial", "resolution": 3},
    ),
    ("Screening", "Definitive screening", "cube", DSD),
    ("Response surface", "Full factorial", "cube", {"factors": box(), "design_type": "full_factorial"}),
    ("Response surface", "Central composite", "cube", CCD),
    ("Response surface", "Box-Behnken", "cube", {"factors": box(), "design_type": "box_behnken"}),
    ("Response surface", "OMARS", "cube", {"factors": box(), "design_type": "omars", "budget": 13}),
    (
        "Arrays, optimal, mixture",
        "Taguchi L9",
        "cube",
        {"factors": [Factor(name=n, low=-1, high=1, levels=[-1, 0, 1]) for n in "ABC"], "design_type": "taguchi"},
    ),
    (
        "Arrays, optimal, mixture",
        "D-optimal",
        "cube",
        {"factors": box(), "design_type": "d_optimal", "budget": 14, "model_type": "quadratic"},
    ),
    ("Arrays, optimal, mixture", "I-optimal, constrained", "cube", I_OPT),
    (
        "Arrays, optimal, mixture",
        "Constrained mixture",
        "simplex",
        {
            "factors": [Factor(name=f"m{i}", type="mixture", low=0.1, high=0.7) for i in (1, 2, 3)],
            "design_type": "mixture",
        },
    ),
    ("Space-filling", "Latin hypercube", "square", {"factors": box(2), "design_type": "latin_hypercube", "budget": 12}),
    ("Space-filling", "Uniform", "square", {"factors": box(2), "design_type": "uniform", "budget": 12}),
    ("Space-filling", "MaxPro", "square", MAXPRO),
    ("Space-filling", "Sobol", "square", {"factors": box(2), "design_type": "sobol", "budget": 12}),
]

#: (stage, drawing, generate_design keywords): one design per stage of a project, for the banner.
HERO = [
    ("Screening", "cube", DSD),
    ("Curvature", "cube", CCD),
    ("Constraints", "cube", I_OPT),
    ("Space-filling", "square", MAXPRO),
]


def draw_cube(ax: plt.Axes, x: np.ndarray, kwargs: dict, c: dict) -> None:
    """Coded cube with its 12 edges, the runs inside it, and any constraint plane."""
    for a, b in itertools.combinations(itertools.product((-1, 1), repeat=3), 2):
        if sum(i != j for i, j in zip(a, b, strict=True)) == 1:  # corners one step apart share an edge
            ax.plot(*zip(a, b, strict=True), color=c["edge"], lw=0.8)
    if kwargs.get("constraints"):  # A + B + C = 1.5 cuts the (1, 1, 1) corner off the cube
        cut = [(1, 1, -0.5), (1, -0.5, 1), (-0.5, 1, 1)]
        ax.add_collection3d(Poly3DCollection([cut], facecolor=c["tint"], edgecolor=c["muted"], lw=0.8, alpha=0.6))
    centre = np.all(np.isclose(x, 0), axis=1)
    axial = (np.count_nonzero(~np.isclose(x, 0), axis=1) == 1) & (kwargs["design_type"] == "ccd")
    ax.scatter(
        *x.T,
        s=34,
        c=np.where(centre | axial, c["extra"], c["run"]),
        edgecolors=c["surface"],
        linewidths=0.8,
        depthshade=False,
    )
    points, counts = np.unique(x.round(6), axis=0, return_counts=True)
    for point, n in zip(points[counts > 1], counts[counts > 1], strict=True):  # replicates share one dot
        label = f"×{n}"  # noqa: RUF001 - the multiplication sign, as in "replicated x3"
        ax.text(*(point + np.array((0.0, -0.2, 0.05))), label, fontsize=7, color=c["muted"], ha="right", va="center")
    reach = max(1.15, np.abs(x).max() * 1.05)
    ax.set(xlim=(-reach, reach), ylim=(-reach, reach), zlim=(-reach, reach))
    ax.set_box_aspect((1, 1, 1), zoom=1.15)
    # tan(azim) = 1/3 spreads the nine (A, B) columns of a 3-level grid evenly across the
    # screen, so no two runs of a {-1, 0, 1} design land on the same spot.
    ax.view_init(elev=28, azim=-np.degrees(np.arctan(1 / 3)))
    ax.set_axis_off()


def draw_signs(ax: plt.Axes, x: np.ndarray, c: dict) -> None:
    """Draw runs (rows) by factors (columns): a filled cell is the high level, a pale one the low level."""
    ax.pcolormesh(x > 0, cmap=ListedColormap([c["tint"], c["run"]]), edgecolors=c["surface"], lw=1.2)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_axis_off()


def draw_square(ax: plt.Axes, x: np.ndarray, kwargs: dict, c: dict) -> None:
    """Two factors with each run's projection onto either axis drawn as a tick."""
    if kwargs["design_type"] == "latin_hypercube":  # one run in every row slice and every column slice
        for edge in np.linspace(-1, 1, len(x) + 1)[1:-1]:
            ax.plot([edge, edge], [-1, 1], color=c["tint"], lw=0.7, zorder=0)
            ax.plot([-1, 1], [edge, edge], color=c["tint"], lw=0.7, zorder=0)
    ax.add_patch(plt.Rectangle((-1, -1), 2, 2, fill=False, edgecolor=c["edge"], lw=0.8))
    ax.plot(x[:, 0], np.full(len(x), -1.12), "|", color=c["muted"], ms=5, mew=0.9)
    ax.plot(np.full(len(x), -1.12), x[:, 1], "_", color=c["muted"], ms=5, mew=0.9)
    ax.scatter(*x.T, s=34, c=c["run"], edgecolors=c["surface"], linewidths=0.8, zorder=3)
    ax.set(xlim=(-1.22, 1.08), ylim=(-1.22, 1.08), aspect="equal")
    ax.set_axis_off()


def draw_simplex(ax: plt.Axes, p: np.ndarray, kwargs: dict, c: dict) -> None:
    """Three-component simplex; the shaded hexagon is the region the component bounds leave."""

    def xy(q: np.ndarray) -> np.ndarray:
        return np.column_stack([q[:, 1] + q[:, 2] / 2, q[:, 2] * np.sqrt(3) / 2])

    lo, hi = kwargs["factors"][0].low, kwargs["factors"][0].high
    hexagon = xy(np.array(sorted(set(itertools.permutations((lo, 1 - lo - hi, hi))))))
    hexagon = hexagon[np.argsort(np.arctan2(*(hexagon - hexagon.mean(axis=0)).T[::-1]))]
    ax.add_patch(plt.Polygon(xy(np.eye(3)), fill=False, edgecolor=c["edge"], lw=0.8))
    ax.add_patch(plt.Polygon(hexagon, facecolor=c["tint"], edgecolor=c["muted"], lw=0.8))
    blend_centre = np.all(np.isclose(p, 1 / 3, atol=1e-3), axis=1)
    ax.scatter(
        *xy(p).T,
        s=34,
        c=np.where(blend_centre, c["extra"], c["run"]),
        edgecolors=c["surface"],
        linewidths=0.8,
        zorder=3,
    )
    ax.set(xlim=(-0.05, 1.05), ylim=(-0.08, 0.95), aspect="equal")
    ax.set_axis_off()


def draw_panel(fig: plt.Figure, rect: tuple, kind: str, kwargs: dict, c: dict) -> object:
    """Generate one design and draw it in ``rect`` (figure coordinates); return the DesignResult."""
    result = generate_design(**kwargs)
    x = result.design[result.factor_names].to_numpy(float)
    ax = fig.add_axes(rect, projection="3d" if kind == "cube" else None)
    ax.set_facecolor(c["surface"])
    if kind == "cube":
        draw_cube(ax, x, kwargs, c)
    elif kind == "signs":
        draw_signs(ax, x, c)
    elif kind == "square":
        draw_square(ax, x, kwargs, c)
    else:
        draw_simplex(ax, x, kwargs, c)
    return result


def add_legend(fig: plt.Figure, c: dict, anchor: tuple, ncol: int, fontsize: float, *, full: bool) -> None:
    """Name every encoding the figure uses, so colour never carries meaning on its own."""
    dot = {"marker": "o", "ls": "", "ms": 7, "mec": c["surface"]}
    handles = [
        Line2D([], [], mfc=c["run"], label="run" + (" (filled cell: high level)" if full else ""), **dot),
        Line2D([], [], mfc=c["extra"], label="centre or axial run", **dot),
    ]
    if full:  # the banner names its one constraint in a panel title instead
        handles += [
            Patch(facecolor=c["tint"], edgecolor=c["muted"], lw=0.8, label="constraint"),
            Line2D([], [], marker="|", ls="", color=c["muted"], ms=7, label="projection onto one factor"),
        ]
    fig.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=anchor,
        ncol=ncol,
        frameon=False,
        fontsize=fontsize,
        labelcolor=c["muted"],
        handletextpad=0.4,
        columnspacing=1.4,
    )


def gallery(c: dict) -> plt.Figure:
    """All sixteen designs in a 4 x 4 grid, one row per kind of design."""
    fig = plt.figure(figsize=(9, 10.2), dpi=200, facecolor=c["surface"])
    fig.text(0.06, 0.965, "Sixteen designs, one function call", fontsize=17, weight="bold", color=c["ink"])
    fig.text(
        0.06, 0.94, 'generate_design(factors, design_type="...")', fontsize=10.5, color=c["muted"], family="monospace"
    )
    for i, (row, title, kind, kwargs) in enumerate(GALLERY):
        left, top = 0.1 + (i % 4) * 0.225, 0.905 - (i // 4) * 0.215
        result = draw_panel(fig, (left, top - 0.17, 0.2, 0.15), kind, kwargs, c)
        fig.text(left + 0.1, top, title, ha="center", fontsize=10, weight="bold", color=c["ink"])
        fig.text(
            left + 0.1,
            top - 0.017,
            f'"{kwargs["design_type"]}"  {result.n_runs} runs',
            ha="center",
            fontsize=8,
            color=c["muted"],
            family="monospace",
        )
        if kind == "signs":
            fig.text(
                left + 0.1,
                top - 0.185,
                f"{result.n_factors} factors as columns",
                ha="center",
                fontsize=8,
                color=c["muted"],
            )
        if i % 4 == 0:
            fig.text(0.045, top - 0.085, row, rotation=90, ha="center", va="center", fontsize=10.5, color=c["muted"])
    add_legend(fig, c, anchor=(0.085, 0.012), ncol=2, fontsize=8.5, full=True)
    fig.text(
        0.94,
        0.03,
        "pip install process-improve",
        ha="right",
        va="center",
        fontsize=8.5,
        color=c["muted"],
        family="monospace",
    )
    return fig


def hero(c: dict) -> plt.Figure:
    """Four stages of a project, one design each, with type large enough to read at phone width."""
    fig = plt.figure(figsize=(8, 4), dpi=200, facecolor=c["surface"])
    fig.text(0.035, 0.89, "Design an experiment in one call", fontsize=20, weight="bold", color=c["ink"])
    fig.text(
        0.035, 0.81, 'generate_design(factors, design_type="...")', fontsize=12.5, color=c["muted"], family="monospace"
    )
    for i, (stage, kind, kwargs) in enumerate(HERO):
        left = 0.02 + i * 0.245
        draw_panel(fig, (left, 0.12, 0.225, 0.52), kind, kwargs, c)
        fig.text(left + 0.1125, 0.695, stage, ha="center", fontsize=15, weight="bold", color=c["ink"])
        fig.text(
            left + 0.1125,
            0.635,
            f'"{kwargs["design_type"]}"',
            ha="center",
            fontsize=13,
            color=c["muted"],
            family="monospace",
        )
    fig.text(
        0.035,
        0.05,
        "+ 12 more designs, with evaluation, analysis and optimisation",
        fontsize=13,
        color=c["muted"],
        va="center",
    )
    add_legend(fig, c, anchor=(0.73, 0.79), ncol=1, fontsize=12, full=False)
    return fig


if __name__ == "__main__":
    layout, theme, output = sys.argv[1:4]
    colours = THEMES[theme]
    {"gallery": gallery, "hero": hero}[layout](colours).savefig(output, facecolor=colours["surface"])
