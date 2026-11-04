"""``spe_plot`` and ``t2_plot`` honour ``settings["show_limit"]`` on every estimator that offers them (#1150)."""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from process_improve.batch._batch_pca import BatchPCA
from process_improve.batch._batch_pls import BatchPLS
from process_improve.multivariate.methods import OPLS, PCA, PLS, PLSDA, PRM, AdaptivePCA, AdaptivePLS, MCUVScaler

go = pytest.importorskip("plotly.graph_objects")

RNG = np.random.default_rng(1150)
LATENT = RNG.standard_normal((30, 2))
X = pd.DataFrame(LATENT @ RNG.standard_normal((2, 5)) + 0.1 * RNG.standard_normal((30, 5)), columns=list("abcde"))
Y = pd.DataFrame({"y": LATENT @ [1.0, -0.5] + 0.1 * RNG.standard_normal(30)})
CLASSES = pd.Series(np.where(LATENT[:, 0] > 0, "high", "low"))


def _batches() -> dict[int, pd.DataFrame]:
    """Twelve batches of ten samples on two tags, each tag a ramp scaled by the batch's own factor."""
    time = np.linspace(0.0, 1.0, 10)
    gains = 1 + 0.1 * RNG.standard_normal((12, 2))
    return {b: pd.DataFrame({"T": gains[b - 1, 0] * time, "P": gains[b - 1, 1] * time**2}) for b in range(1, 13)}


BATCHES = _batches()
BATCH_QUALITY = pd.DataFrame({"q": [float(b["T"].iloc[-1]) for b in BATCHES.values()]}, index=list(BATCHES))

FITTED: dict[str, Callable[[], object]] = {
    "PCA": lambda: PCA(n_components=2).fit(MCUVScaler().fit_transform(X)),
    "PLS": lambda: PLS(n_components=2).fit(X, Y),
    "PLSDA": lambda: PLSDA(n_components=2).fit(X, CLASSES),
    "PRM": lambda: PRM(n_components=2).fit(X, Y),
    "OPLS": lambda: OPLS(n_orthogonal_components=1).fit(X, Y),
    # An adaptive model plots the rows streamed through it after fit, so stream some.
    "AdaptivePCA": lambda: AdaptivePCA(n_components=2).fit(X).partial_fit(X.iloc[:10]),
    "AdaptivePLS": lambda: AdaptivePLS(n_components=2).fit(X, Y).partial_fit(X.iloc[:10], Y.iloc[:10]),
    "BatchPCA": lambda: BatchPCA(n_components=2).fit(BATCHES),
    "BatchPLS": lambda: BatchPLS(n_components=2).fit(BATCHES, BATCH_QUALITY),
}


def _limit_lines(fig: go.Figure) -> int:
    """How many confidence-limit lines the figure carries: horizontal lines with a "% limit" annotation."""
    return sum("% limit" in (annotation.text or "") for annotation in fig.layout.annotations)


#: OPLS's spe_plot and t2_plot raise TypeError on any settings: the component count they resolve
#: is a string for OPLS. Listed in #1149; strict, so a fix fails here until this mark goes.
OPLS_PLOTS_RAISE = pytest.mark.xfail(raises=TypeError, strict=True, reason="OPLS plots resolve a str count (#1149)")


@pytest.mark.parametrize("plot", ["spe_plot", "t2_plot"])
@pytest.mark.parametrize("name", [pytest.param(n, marks=OPLS_PLOTS_RAISE) if n == "OPLS" else n for n in FITTED])
def test_the_limit_line_is_drawn_only_when_show_limit_is_true(name: str, plot: str) -> None:
    model = FITTED[name]()
    shown = getattr(model, plot)(settings={"show_limit": True})
    hidden = getattr(model, plot)(settings={"show_limit": False})
    assert _limit_lines(shown) == 1
    assert _limit_lines(hidden) == 0
    # The y = 0 reference line stays either way.
    assert any(shape.y0 == 0 and shape.y1 == 0 for shape in hidden.layout.shapes)


def test_a_t2_plot_without_its_limit_line_does_not_warn_about_that_limit() -> None:
    """With the line off, the limit is not computed, so #978's FutureWarning about it is not issued."""
    model = FITTED["PCA"]()
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        model.t2_plot(settings={"show_limit": False})
