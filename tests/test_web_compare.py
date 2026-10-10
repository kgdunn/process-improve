"""The compare step, the book links and "Show the Python" of the browser app.

``web/bootstrap.py`` is driven through its JSON entry points, as the page drives it. The
Python the page shows is produced by ``web/python.js`` under Node and run here, so the
code a reader copies is checked to reproduce what the page computed.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import compare_designs

pytest.importorskip("openpyxl")

WEB = Path(__file__).resolve().parent.parent / "web"
FACTORS = [
    {"name": "T", "low": 150, "high": 200, "units": "degC"},
    {"name": "P", "low": 1, "high": 3, "units": "bar"},
    {"name": "F", "low": 10, "high": 20, "units": "L/min"},
]


@pytest.fixture(scope="module")
def app():
    spec = importlib.util.spec_from_file_location("web_bootstrap_compare", WEB / "bootstrap.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reply(app, fn: str, payload: dict) -> dict:
    return json.loads(getattr(app, fn)(json.dumps(payload)))


def call(app, fn: str, payload: dict) -> dict:
    answer = reply(app, fn, payload)
    assert answer["ok"], answer.get("error")
    return answer["result"]


def generate_all(app, factors: list[dict], model: str) -> tuple[list[dict], list[dict]]:
    """Generate every design of a comparison, as the page does, one call each."""
    plan = call(app, "api_compare_plan", {"factors": factors, "model": model})
    made = [{**item, "design": call(app, "api_make_design", item["spec"])} for item in plan["designs"]]
    held = [
        {
            "label": m["label"],
            "spec": m["design"]["spec"],
            "columns": m["design"]["columns"],
            "rows": m["design"]["rows"],
        }
        for m in made
    ]
    return made, held


# --------------------------------------------------------------------------- the plan
@pytest.mark.parametrize(
    ("n_factors", "expected"),
    [
        (2, ["full_factorial", "plackett_burman", "ccd", "d_optimal", "i_optimal"]),
        (
            3,
            [
                "full_factorial",
                "fractional_factorial",
                "plackett_burman",
                "dsd",
                "box_behnken",
                "ccd",
                "d_optimal",
                "i_optimal",
            ],
        ),
        (8, ["fractional_factorial", "plackett_burman", "dsd", "d_optimal"]),
    ],
)
def test_the_comparison_includes_every_design_the_factor_count_allows(app, n_factors, expected):
    factors = [{"name": f"X{i}", "low": 0, "high": 1} for i in range(n_factors)]
    plan = call(app, "api_compare_plan", {"factors": factors})
    assert [d["design_type"] for d in plan["designs"]] == expected


def test_optimal_designs_are_built_for_the_smallest_model_containing_the_comparison_s(app):
    plan = call(app, "api_compare_plan", {"factors": FACTORS, "model": "main_quadratic"})
    optimal = [d for d in plan["designs"] if d["design_type"].endswith("_optimal")]
    assert {d["spec"]["model_type"] for d in optimal} == {"quadratic"}
    assert "built for the quadratic model" in plan["notes"][0]
    same = call(app, "api_compare_plan", {"factors": FACTORS, "model": "interactions"})
    assert same["notes"] == []


def test_a_model_the_comparison_does_not_define_is_refused(app):
    answer = reply(app, "api_compare_plan", {"factors": FACTORS, "model": "cubic"})
    assert not answer["ok"]
    assert "Model must be one of" in answer["error"]


# --------------------------------------------------------------------------- the comparison
def test_the_page_s_comparison_is_compare_designs_on_the_page_s_designs(app):
    made, held = generate_all(app, FACTORS, "main_quadratic")
    result = call(app, "api_compare", {"model": "main_quadratic", "designs": held})
    coded = {}
    for m in made:
        runs = pd.DataFrame(m["design"]["rows"], columns=m["design"]["columns"])
        coded[m["label"]] = pd.DataFrame(
            {f["name"]: (runs[f["name"]] - (f["high"] + f["low"]) / 2) / ((f["high"] - f["low"]) / 2) for f in FACTORS}
        )
    direct = compare_designs(coded, "main_quadratic")
    assert result["designs"] == list(direct.designs)
    for label in direct.designs:
        for key, value in direct.values[label].items():
            if isinstance(value, float):
                assert result["values"][label][key] == pytest.approx(value, rel=1e-9), (label, key)
        assert result["display"][label] == direct.table[label].tolist()
    assert result["values"]["Box-Behnken [n=15]"]["a"] == pytest.approx(0.2173, abs=5e-5), "the book's value"


def test_fds_curves_belong_to_the_designs_that_fit_the_model(app):
    _, held = generate_all(app, FACTORS, "main_quadratic")
    result = call(app, "api_compare", {"model": "main_quadratic", "designs": held})
    fits = {label for label in result["designs"] if result["values"][label]["fits"]}
    assert set(result["fds"]) == fits
    for curve in result["fds"].values():
        assert len(curve["fraction"]) == len(curve["variance"]) == 101
        assert np.all(np.diff(curve["variance"]) >= -1e-12), "sorted"


def test_designs_sent_back_are_checked_as_data(app):
    _, held = generate_all(app, FACTORS, "main_effects")
    held[0]["spec"] = {**held[0]["spec"], "response": "__import__('os')"}
    answer = reply(app, "api_compare", {"model": "main_effects", "designs": held})
    assert not answer["ok"]


# --------------------------------------------------------------------------- links to the book
def test_every_link_to_the_book_is_extensionless_and_one_the_page_checked():
    """The book is served without ``.html``; every link the page makes names a page and anchor that exist."""
    checked = {
        "omars-designs#doe-omars-metric-choice",
        "comparing-design-families#doe-omnibus-comparison",
        "full-factorial-designs/using-two-levels-for-two-or-more-factors#doe-two-level-factorials",
        "fractional-factorial-designs/design-resolution#doe-design-resolution",
        "fractional-factorial-designs/saturated-designs-for-screening#doe-saturated-screening-designs",
        "definitive-screening-designs#doe-definitive-screening-designs",
        "response-surface-methods#doe-box-behnken-designs",
        "response-surface-methods#doe-central-composite-designs",
        "optimal-designs#doe-optimal-designs",
        "full-factorial-designs/assessing-significance-of-main-effects-and-interactions#doe-lenth-method",
        "judging-and-comparing-designs#doe-fds-plot",
    }
    text = (WEB / "index.html").read_text(encoding="utf-8") + (WEB / "app.js").read_text(encoding="utf-8")
    links = set(re.findall(r"learnche\.org/pid/design-analysis-experiments/([\w/#.-]+)", text))
    links |= set(re.findall(r"\$\{BOOK\}/([\w/#.-]+)", text))
    assert links, "the page links to the book"
    assert not any(".html" in link for link in links)
    assert links <= checked


def test_every_design_links_to_the_book(app):
    source = (WEB / "app.js").read_text(encoding="utf-8")
    block = re.search(r"const BOOK_PAGES = \{(.*?)\};", source, re.DOTALL).group(1)
    assert set(app.DESIGNS) <= set(re.findall(r"^\s*(\w+):", block, re.MULTILINE))


# --------------------------------------------------------------------------- show the Python
NODE = shutil.which("node")


def page_code(function: str, *args: object) -> str:
    """Run one of web/python.js's functions under Node, as the page does, and return the code it writes."""
    driver = (
        "const [path, fn, args] = process.argv.slice(1);"
        "import(path).then((m) => process.stdout.write(m[fn](...JSON.parse(args))));"
    )
    module = Path(__file__).resolve().parent / "_python_panels.mjs"
    shutil.copyfile(WEB / "python.js", module)  # .mjs, so Node reads it as a module whatever its version
    try:
        done = subprocess.run(  # noqa: S603
            [NODE, "-e", driver, module.as_uri(), function, json.dumps(args)],
            capture_output=True,
            text=True,
            check=True,
        )
    finally:
        module.unlink()
    return done.stdout


def run(code: str, namespace: dict) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        exec(code, namespace)  # noqa: S102 - the page's own code, the subject of this test
    return namespace


@pytest.mark.slow
@pytest.mark.skipif(NODE is None, reason="Node.js runs the page's JavaScript")
def test_the_python_shown_reproduces_the_page(app):
    demo = call(app, "api_catalogue", {})["demo"]
    design = call(app, "api_make_design", demo["design"])
    held = {"spec": design["spec"], "columns": design["columns"], "rows": design["rows"]}
    simulation = call(app, "api_simulate", {**held, "scenario": "reactor", "sigma": 1.0, "seed": 1})
    analysis = call(app, "api_analyze_runs", {**held, "responses": simulation["responses"]})

    namespace = run(page_code("designCode", design["spec"]), {})
    names = [f["name"] for f in design["spec"]["factors"]]
    page_runs = pd.DataFrame(design["rows"], columns=design["columns"])[names]
    np.testing.assert_allclose(namespace["result"].design_actual[names].to_numpy(dtype=float), page_runs, atol=1e-9)

    run(page_code("simulateCode", simulation), namespace)
    np.testing.assert_allclose(namespace["y"], simulation["responses"], atol=0.005, err_msg="the page rounds")

    run(page_code("analyseCode", analysis["spec"], analysis["model"]), namespace)
    shown = {c["term"]: c["coefficient"] for c in namespace["analysis"]["coefficients"]}
    for c in analysis["coefficients"]:
        assert shown[c["term"]] == pytest.approx(c["coefficient"], abs=0.01), c["term"]

    entered = run(page_code("enteredCode", [*simulation["responses"][:-1], None]), {})
    assert entered["y"].isna().tolist() == [False] * 14 + [True]

    made, held_designs = generate_all(app, FACTORS, "main_quadratic")
    compared = call(app, "api_compare", {"model": "main_quadratic", "designs": held_designs})
    designs = [
        {"label": m["label"], "design_type": m["design_type"], "options": m["design"]["spec"]["options"]} for m in made
    ]
    namespace = run(page_code("compareCode", FACTORS, designs, "main_quadratic"), {})
    assert list(namespace["comparison"].designs) == compared["designs"]
    for label in compared["designs"]:
        assert namespace["comparison"].table[label].tolist() == compared["display"][label], label
