"""The build: ``scripts/build_web_app.py`` assembles the page, a wheel of this checkout and the worked example.

The docs workflow publishes what it assembles. ``demo.json`` is the worked example computed at
build time through ``web/bootstrap.py``, which the page draws at first paint while Python loads.
"""

from __future__ import annotations

import json
import re

import pytest

from tests.web._support import no_bytecode, source


@pytest.fixture(scope="module")
def built_example(build_script) -> dict:
    """Run the worked example as the build does for ``demo.json``, and read it back as the page reads it."""
    pytest.importorskip("plotly", reason="the build draws the worked example's charts")
    with no_bytecode():
        return json.loads(json.dumps(build_script.worked_example()))


def test_every_module_the_page_imports_is_published(build_script):
    imported = set(re.findall(r'from "\./([\w.-]+)"', source("app.js")))
    assert imported, "app.js imports charts.js"
    assert imported <= set(build_script.APP_FILES)


@pytest.mark.slow
def test_the_built_worked_example_carries_what_the_page_draws(built_example):
    example = built_example
    assert example["version"]
    assert set(example["inputs"]) >= {"design", "scenario", "sigma", "seed", "model"}
    assert set(example["design"]) >= {"spec", "columns", "rows", "facts", "model", "n_runs"}
    assert "xlsx_base64" not in example["design"], "the page writes the workbook on demand"
    assert set(example["simulation"]) >= {"responses", "truth", "equation", "inactive", "label", "sigma", "seed"}
    assert all("true_value" in c for c in example["analysis"]["coefficients"])
    assert set(example["analysis"]) >= {"equation", "summary", "notes", "anova", "pending_runs", "n_completed"}
    assert set(example["plots"]["figures"]) == {"contour", "true_contour", "residuals"}
    assert example["plots"]["plotly_js"]


@pytest.mark.slow
def test_the_built_worked_example_includes_the_comparison(built_example):
    """The page draws step 4 at first paint too, from the comparison the build computes."""
    compared = built_example["comparison"]
    assert compared["model"] == "main_quadratic"
    assert [m["design_type"] for m in compared["made"]] == [
        "full_factorial",
        "fractional_factorial",
        "plackett_burman",
        "dsd",
        "box_behnken",
        "ccd",
        "d_optimal",
        "i_optimal",
    ]
    assert not any("xlsx_base64" in m["design"] for m in compared["made"]), "the page writes workbooks on demand"
    a_over_p = [m["key"] for m in compared["result"]["measures"]].index("a")
    assert compared["result"]["display"]["Box-Behnken [n=15]"][a_over_p] == "0.2173"
