"""Redraw two panels of the GitHub social-preview card from process-improve itself.

Usage::

    python tools/branding/social_card.py tools/branding/social-card-base.png docs/_static/readme/social-preview.png

The other four panels are kept from the uploaded base card.

* **Designed experiments**: the sequential strategy of response surface methodology. A first
  factorial, the path of steepest ascent its fitted model implies, and a central composite design
  where the path stops improving. The yield surface is symmetric about the diagonal through the
  first design, so the path climbs at 45 degrees and its runs land on points the designs already
  use; rings mark the runs that serve more than one stage.
* **PLS with prediction intervals**: pectin yield from FTIR spectra (37 extractions bundled with
  the package). Every point is predicted by a model fitted without it, with that model's 95%
  interval from ``PLS.prediction_interval``.
"""

import io
import sys
import warnings
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from sklearn.model_selection import KFold

import process_improve
from process_improve.experiments import Factor, analyze_experiment, generate_design, optimize_responses
from process_improve.multivariate import PLS, MCUVScaler

warnings.simplefilter("ignore")

# Colours sampled from the base card, so the new panels match their neighbours.
NAVY, RED, SPINE, BAND, CONTOUR = "#2c5f7c", "#a93226", (143, 143, 142), "#e6edf2", "#b8c8d3"
INTERVAL = "#d4e0e8"  # one step darker than the card's band, so a single interval bar still reads
BACKGROUND = (250, 248, 243)
#: Panel faces as (left, top, right, bottom), right and bottom exclusive, in card pixels.
DOE_FACE, PLS_FACE = (1466, 791, 1912, 1202), (2055, 255, 2466, 667)
PX = 72 / 100  # points per pixel when a panel is rendered at dpi=100


def area(px: float) -> float:
    """Scatter marker size (points squared) for a circle ``px`` pixels across."""
    return (px * PX) ** 2


def hill(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Yield over two factors: a ridge along the diagonal, rising to a peak at (3.9, 3.9)."""
    u, v = x1 - 3.9, x2 - 3.9
    return 90 - u**2 - v**2 + 0.8 * u * v


def box(centre: tuple[float, float], half: float) -> list[Factor]:
    """Two factors spanning ``centre +/- half`` in actual units."""
    return [Factor(name=n, low=c - half, high=c + half) for n, c in zip(("T", "P"), centre, strict=True)]


def runs(design: object) -> np.ndarray:
    """Return the design's runs in actual units, as an (n, 2) array."""
    return design.design_actual[["T", "P"]].to_numpy(float)


def draw_design(ax: plt.Axes, design: object) -> None:
    """Outline through the factorial corners; navy factorial runs, red centre and axial runs."""
    xy, coded = runs(design), design.design[["T", "P"]].to_numpy(float)
    corners = xy[(np.abs(coded) == 1).all(axis=1)]
    order = np.argsort(np.arctan2(*(corners - corners.mean(axis=0)).T[::-1]))
    ax.fill(*corners[order].T, facecolor="none", edgecolor=NAVY, lw=1.3 * PX, alpha=0.55, zorder=3)
    extra = np.count_nonzero(~np.isclose(coded, 0), axis=1) <= 1
    ax.scatter(*xy.T, s=area(19.3), c=np.where(extra, RED, NAVY), edgecolors="white", linewidths=1.6 * PX, zorder=5)


def shared_runs(stages: list[np.ndarray]) -> np.ndarray:
    """Points that belong to more than one stage: runs that are reused rather than repeated."""
    points = np.unique(np.vstack(stages).round(6), axis=0)
    uses = np.array([sum(np.isclose(stage, p).all(axis=1).any() for stage in stages) for p in points])
    return points[uses > 1]


def journey(ax: plt.Axes) -> None:
    """Draw a factorial, steepest ascent from it, and a central composite design that shares its runs."""
    g1, g2 = np.meshgrid(np.linspace(-0.05, 5.65, 300), np.linspace(0.18, 5.43, 300))
    levels = np.linspace(72, 89.6, 7)
    ramp = LinearSegmentedColormap.from_list("ramp", ["#ffffff", BAND])
    ax.contourf(g1, g2, hill(g1, g2), levels=levels, cmap=ramp, extend="both", zorder=0)
    ax.contour(g1, g2, hill(g1, g2), levels=levels, colors=CONTOUR, linewidths=1.2 * PX, zorder=1)
    ax.set(xlim=(-0.05, 5.65), ylim=(0.18, 5.43))

    start, half = (1.6, 1.6), 1.0
    first = generate_design(box(start, half), design_type="full_factorial")
    fit = analyze_experiment(
        first.design,
        responses=pd.Series(hill(*runs(first).T), name="Yield"),
        model="main_effects",
        analysis_type="coefficients",
    )
    ranges = {f.name: {"low": f.low, "high": f.high} for f in box(start, half)}
    # On a 45-degree path a coded step of sqrt(2) is one corner-to-centre diagonal: step 1 lands on
    # the factorial's top-right corner, step 2 on the centre of the next design.
    steps = optimize_responses([fit], method="steepest_ascent", step_size=np.sqrt(2), n_steps=4, factor_ranges=ranges)[
        "steepest_path"
    ]["steps"]
    path = np.array([[s["actual"]["T"], s["actual"]["P"]] for s in steps])
    path = path[: int(np.argmax(hill(*path.T))) + 1]  # stop where the response stops improving
    second = generate_design(box(tuple(path[-1]), half), design_type="ccd", alpha="rotatable")

    heading = (path[-1] - path[-2]) / np.linalg.norm(path[-1] - path[-2])
    tip = path[-1] - 0.25 * heading  # arrowhead just short of the ring round the new centre run
    ax.plot(*np.vstack([path[:-1], tip]).T, color=NAVY, lw=1.8 * PX, ls=(0, (4, 3)), alpha=0.8, zorder=2)
    arrow = {"arrowstyle": "-|>", "color": NAVY, "lw": 1.8 * PX, "mutation_scale": 14, "alpha": 0.8}
    ax.annotate("", xy=tip, xytext=tip - 0.05 * heading, arrowprops=arrow, zorder=2)
    draw_design(ax, first)
    draw_design(ax, second)
    stages = [runs(first), path[1:], runs(second)]  # the path's runs exclude its starting point
    ax.scatter(*shared_runs(stages).T, s=area(36.6), facecolors="none", edgecolors=NAVY, linewidths=1.3 * PX, zorder=4)


def held_out_intervals(X: pd.DataFrame, y: pd.Series, n_components: int, folds: int = 10) -> pd.DataFrame:
    """Each sample predicted by a PLS model fitted without it, with that model's 95% prediction interval."""
    rows = []
    for train, test in KFold(folds, shuffle=True, random_state=1).split(X):
        sx, sy = MCUVScaler().fit(X.iloc[train]), MCUVScaler().fit(y.iloc[train].to_frame())
        model = PLS(n_components=n_components).fit(sx.transform(X.iloc[train]), sy.transform(y.iloc[train].to_frame()))
        interval = model.prediction_interval(sx.transform(X.iloc[test]))
        back = {key: sy.inverse_transform(interval[key]).to_numpy().ravel() for key in ("y_hat", "lower", "upper")}
        rows.append(pd.DataFrame({"observed": y.iloc[test].to_numpy(), **back}, index=X.index[test]))
    return pd.concat(rows).sort_index()


def prediction_intervals(ax: plt.Axes) -> None:
    """Measured against predicted pectin yield; every prediction is out of sample, with its 95% interval."""
    root = Path(process_improve.__file__).parent / "datasets" / "multivariate" / "dtu-pectin"
    ftir = pd.concat([pd.read_csv(root / f"ftir{i}.csv") for i in (1, 2, 3)], ignore_index=True)
    X, y = ftir.drop(columns="yield_g"), ftir["yield_g"]
    n_components = PLS.select_n_components(X, y.to_frame(), max_components=10).n_components  # 1-SE rule
    r = held_out_intervals(X, y, n_components)
    outside = (r.observed < r.lower) | (r.observed > r.upper)
    low, high = min(r.y_hat.min(), r.observed.min()), max(r.y_hat.max(), r.observed.max())
    limits = (low - 0.1 * (high - low), high + 0.1 * (high - low))  # the points fill the panel
    ax.plot(limits, limits, color=NAVY, lw=1.6 * PX, ls=(0, (4, 3)), alpha=0.6, zorder=1)
    ax.vlines(r.y_hat, r.lower, r.upper, color=INTERVAL, lw=6 * PX, capstyle="round", zorder=2)
    colours = np.where(outside, RED, NAVY)
    ax.scatter(r.y_hat, r.observed, s=area(10), c=colours, edgecolors="white", linewidths=1.2 * PX, zorder=3)
    ax.set(xlim=limits, ylim=limits)


def render(face: tuple[int, int, int, int], draw: Callable[[plt.Axes], None]) -> Image.Image:
    """Draw one panel at the card's own pixel size: one pixel per pixel at dpi=100."""
    left, top, right, bottom = face
    fig = plt.figure(figsize=((right - left) / 100, (bottom - top) / 100), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    draw(ax)
    ax.set_axis_off()
    buffer = io.BytesIO()
    fig.savefig(buffer, dpi=100, facecolor="white")
    plt.close(fig)
    return Image.open(buffer).convert("RGB")


def patch(base_path: str, out_path: str) -> None:
    """Replace the DoE and PLS panels of the base card and save the result."""
    card = Image.open(base_path).convert("RGB")
    for stray in ((1478, 1204, 1898, 1208), (1478, 786, 1486, 791)):  # ends of the old DoE panel's spines
        card.paste(BACKGROUND, stray)
    left, top, right, bottom = DOE_FACE
    card.paste(render(DOE_FACE, journey), (left, top))
    card.paste(SPINE, (left - 2, top - 1, left, bottom + 2))  # the DoE panel is wider than before: new spines
    card.paste(SPINE, (left - 2, bottom, right, bottom + 2))
    card.paste(render(PLS_FACE, prediction_intervals), PLS_FACE[:2])  # same face, so its spines stay
    card.save(out_path, optimize=True)


if __name__ == "__main__":
    patch(*sys.argv[1:3])
