"""The pictures of the charts on the list of what Factorial does (``web/chart-pictures/``).

Each picture is a chart as the app draws it, in the light colours and on no ground of its own, written by this
file: run it with ``PI_WRITE_CHART_PICTURES=1`` to draw them all again, after a change to how a chart looks or
when a chart is added. No one study draws every chart, so five do, each named with the charts it is asked for.
Without the variable the file holds those studies to the app: each still draws its charts, and together they
draw every chart there is, at the size the page gives a picture.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

import pytest

from tests.web.e2e._page import first_paint_done, generate, practise, set_factors, settled

pytestmark = pytest.mark.e2e

PICTURES = Path(__file__).resolve().parents[3] / "web" / "chart-pictures"

#: Every chart of Learn is drawn, and Python has no more to say.
DRAWN = """() => { const cards = [...document.querySelectorAll('#charts figure[data-chart]')];
  return cards.length > 0 && window.__pi.idle() && cards.every((card) => card.querySelector('.js-plotly-plot')); }"""

#: The charts drawn, by their keys in ``CHARTS`` (web/logic.js).
KEYS = "() => [...document.querySelectorAll('#charts figure[data-chart]')].map((card) => card.dataset.chart)"

#: A chart as a picture of the size the page asks for: its address, as data.
PICTURE = """async (key) => {
  const { PICTURE_SIZE: [width, height] } = await import('./features.js');
  const plot = document.querySelector(`figure[data-chart=${key}] .plot`);
  return Plotly.toImage(plot, { format: 'webp', width, height, scale: 1 }); }"""


def opened(context: Any, site: str, query: str = "") -> Any:  # noqa: ANN401 - a Playwright BrowserContext and Page
    """Open the page at ``query``, in the light colours, and return it once Python has answered."""
    page = context.new_page()
    page.emulate_media(color_scheme="light")
    page.goto(f"{site}{query}")
    page.wait_for_function("window.__pi.count('api_catalogue') === 1 && window.__pi.idle()")
    return page


def worked(context: Any, site: str) -> Any:  # noqa: ANN401
    """Open the worked example, and its "More charts"."""
    page = opened(context, site, "?example=worked")
    first_paint_done(page)
    page.click("#more-charts > summary")
    return page


def saturated(context: Any, site: str) -> Any:  # noqa: ANN401
    """Open the book's saturated screening design: as many effects as runs, so they are read off a half-normal plot."""
    return opened(context, site, "?book=saturated-screening")


def one_factor(context: Any, site: str) -> Any:  # noqa: ANN401
    """Make a study of one factor at two levels, and simulate its results."""
    page = opened(context, site)
    set_factors(page, [{"name": "Supplier", "low": "A", "high": "B"}])
    page.wait_for_selector("#candidates .levels-plan tbody tr")
    settled(page)
    page.click("#generate")
    settled(page)
    practise(page)
    return page


def blend(context: Any, site: str, components: list[str], design: str) -> Any:  # noqa: ANN401
    """Make a mixture of ``components``, and simulate its results."""
    page = opened(context, site)
    set_factors(page, [{"name": name} for name in components])
    generate(page, design_type=design)
    practise(page)
    return page


def blend_of_three(context: Any, site: str) -> Any:  # noqa: ANN401
    """Make a mixture of three components: the response is drawn over their triangle."""
    return blend(context, site, ["Water", "Oil", "Wax"], "simplex_lattice")


def blend_of_four(context: Any, site: str) -> Any:  # noqa: ANN401
    """Make a mixture of four components: no triangle holds them, so each is traced along its Cox direction."""
    return blend(context, site, ["Water", "Oil", "Wax", "Salt"], "simplex_centroid")


#: Each study, with the charts whose pictures are taken from it.
STUDIES = {
    worked: [
        "coefficients",
        "contour",
        "true_contour",
        "residuals",
        "perturbation",
        "main_effects",
        "interaction",
        "surface_3d",
        "cube",
        "normal_probability",
        "residuals_order",
        "prediction_variance",
    ],
    saturated: ["half_normal"],
    one_factor: ["levels"],
    blend_of_three: ["ternary", "true_ternary"],
    blend_of_four: ["cox"],
}


@pytest.mark.parametrize("study", list(STUDIES), ids=lambda study: study.__name__)
def test_a_study_draws_the_charts_its_pictures_are_of(context, site, study):
    page = study(context, site)
    page.click("#go-learn")
    page.wait_for_function(DRAWN, timeout=60_000)
    assert set(STUDIES[study]) <= set(page.evaluate(KEYS))
    if not os.environ.get("PI_WRITE_CHART_PICTURES"):
        return
    PICTURES.mkdir(exist_ok=True)
    for key in STUDIES[study]:
        address = page.evaluate(PICTURE, key)
        assert address.startswith("data:image/webp;base64,")
        (PICTURES / f"{key}.webp").write_bytes(base64.b64decode(address.split(",", 1)[1]))


def test_the_studies_draw_every_chart_there_is_once(page):
    charts = page.evaluate("async () => Object.keys((await import('./logic.js')).CHARTS)")
    assert sorted(key for keys in STUDIES.values() for key in keys) == sorted(charts)
